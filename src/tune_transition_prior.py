#!/usr/bin/env python3
"""
Tune a weak transition prior on top of saved validation logits.

Inputs expected from src/train_deberta.py:
  - val_predictions.csv with columns including:
      true_id, true_action, last_action, pred_action, confidence, correct ...
  - val_logits.npy with shape [N, num_classes]

Recommended usage inside project root:
  python src/tune_transition_prior.py \
    --val_csv submit/model/deberta/val_predictions.csv \
    --val_logits submit/model/deberta/val_logits.npy \
    --out_dir submit/model/deberta/prior_tuning

If you have data/train.csv and it contains action labels + last_action, use it to build the prior
without validation-label leakage:
  python src/tune_transition_prior.py \
    --val_csv submit/model/deberta/val_predictions.csv \
    --val_logits submit/model/deberta/val_logits.npy \
    --prior_source data/train.csv \
    --out_dir submit/model/deberta/prior_tuning
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


DEFAULT_ALPHAS = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
DEFAULT_SMOOTHINGS = [0.5, 1.0, 2.0, 5.0, 10.0]
NONE_TOKEN = "__NONE__"


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def infer_id2action(val_df: pd.DataFrame, num_classes: int) -> List[str]:
    """Infer class order used by logits from val_predictions true_id/pred_id columns."""
    pairs = []
    for id_col, action_col in [("true_id", "true_action"), ("pred_id", "pred_action")]:
        if id_col in val_df.columns and action_col in val_df.columns:
            pairs.append(val_df[[id_col, action_col]].rename(columns={id_col: "id", action_col: "action"}))
    if not pairs:
        raise ValueError("val_csv must contain true_id/true_action or pred_id/pred_action to infer class order.")

    all_pairs = pd.concat(pairs, ignore_index=True).dropna().drop_duplicates()
    id2actions: Dict[int, set] = {}
    for _, r in all_pairs.iterrows():
        idx = int(r["id"])
        id2actions.setdefault(idx, set()).add(str(r["action"]))

    id2action: List[str] = []
    for i in range(num_classes):
        acts = sorted(id2actions.get(i, []))
        if len(acts) != 1:
            raise ValueError(f"Could not infer a unique action for class id {i}: {acts}")
        id2action.append(acts[0])
    return id2action


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_prior_source(prior_source, val_df):
    """
    Load transition-prior source.

    Supported:
    - .csv
    - .jsonl / .json
    - None: fallback to validation labels, leaky probe only
    """
    if prior_source is None or str(prior_source).strip() == "":
        print("[WARNING] prior_source was not provided, so the prior was estimated from validation labels.")
        print("          This is useful only as a quick probe. For clean tuning, pass --prior_source data/train.csv or data/train.jsonl")
        return val_df.copy(), True

    p = Path(prior_source)
    if not p.exists():
        raise FileNotFoundError(f"prior_source not found: {prior_source}")

    suffix = p.suffix.lower()

    if suffix == ".csv":
        src = pd.read_csv(p)
    elif suffix in [".jsonl", ".json"]:
        # DACON train is often jsonl: one JSON object per line.
        src = pd.read_json(p, lines=True)
    else:
        raise ValueError(f"Unsupported prior_source format: {p.suffix}. Use .csv, .jsonl, or .json")

    return src, False


def standardize_prior_df(df, id2action):
    """
    Return dataframe with columns:
    - last_action
    - true_action

    Supports:
    1. val_predictions.csv-like files:
       last_action + true_action/action/label

    2. raw DACON train.jsonl-like files:
       history = list of turns, where assistant actions appear as:
       {"role": "assistant_action", "name": "..."}
       In this case, we estimate transition prior from adjacent assistant_action names
       inside each history:
       action_t -> action_{t+1}
    """
    df = df.copy()
    valid_actions = set(id2action)

    # ============================================================
    # Case 1: raw DACON jsonl with history
    # ============================================================
    if "history" in df.columns:
        rows = []

        import ast
        import json

        def parse_history(x):
            if isinstance(x, list):
                return x

            if x is None:
                return []

            s = str(x).strip()
            if not s or s.lower() in ["nan", "none", "null"]:
                return []

            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

            return []

        for hist in df["history"]:
            hist = parse_history(hist)

            actions = []
            for turn in hist:
                if not isinstance(turn, dict):
                    continue

                if turn.get("role") == "assistant_action":
                    name = turn.get("name")
                    if name in valid_actions:
                        actions.append(name)

            # adjacent transitions inside history
            for prev_a, next_a in zip(actions[:-1], actions[1:]):
                rows.append({
                    "last_action": prev_a,
                    "true_action": next_a,
                })

        if len(rows) > 0:
            out = pd.DataFrame(rows)

            action2id = {a: i for i, a in enumerate(id2action)}
            out["last_action_id"] = out["last_action"].map(action2id)
            out["action_id"] = out["true_action"].map(action2id)

            out = out.dropna(subset=["last_action_id", "action_id"]).copy()
            out["last_action_id"] = out["last_action_id"].astype(int)
            out["action_id"] = out["action_id"].astype(int)

            print(f"[INFO] Built transition prior source from history transitions: {len(out)} pairs")
            return out[["last_action", "true_action", "last_action_id", "action_id"]]

        # If history exists but no transitions found, continue to column-based fallback.

    # ============================================================
    # Case 2: val_predictions.csv-like / tabular labeled source
    # ============================================================

    # ---------- find true action ----------
    true_candidates = [
        "true_action",
        "action",
        "label",
        "target",
        "next_action",
        "answer",
        "output",
    ]

    true_col = None
    for c in true_candidates:
        if c in df.columns:
            true_col = c
            break

    if true_col is None:
        raise ValueError(
            "Prior source needs either a DACON-style history column "
            "or a true action column. "
            f"Available columns: {list(df.columns)}"
        )

    df["true_action"] = df[true_col].astype(str)

    def normalize_action(x):
        if x in valid_actions:
            return x
        try:
            ix = int(x)
            if 0 <= ix < len(id2action):
                return id2action[ix]
        except Exception:
            pass
        return x

    df["true_action"] = df["true_action"].map(normalize_action)

    # ---------- find last action ----------
    last_candidates = [
        "last_action",
        "previous_action",
        "prev_action",
        "last_tool",
        "previous_tool",
        "prev_tool",
    ]

    last_col = None
    for c in last_candidates:
        if c in df.columns:
            last_col = c
            break

    if last_col is not None:
        df["last_action"] = df[last_col].astype(str)

    elif "recent_actions" in df.columns:
        def get_last_from_recent_actions(x):
            if x is None:
                return "NONE"

            if isinstance(x, list):
                return str(x[-1]) if len(x) else "NONE"

            s = str(x).strip()
            if not s or s.lower() in ["nan", "none", "null"]:
                return "NONE"

            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and len(parsed):
                    return str(parsed[-1])
            except Exception:
                pass

            for sep in ["|", ",", " "]:
                parts = [p.strip().strip("'\"") for p in s.split(sep) if p.strip()]
                valid = [p for p in parts if p in valid_actions]
                if valid:
                    return valid[-1]

            return "NONE"

        df["last_action"] = df["recent_actions"].map(get_last_from_recent_actions)

    else:
        raise ValueError(
            "Prior source needs a last_action-like column, recent_actions, "
            "or DACON-style history. "
            f"Available columns: {list(df.columns)}"
        )

    df["last_action"] = df["last_action"].fillna("NONE").astype(str)
    df["true_action"] = df["true_action"].fillna("").astype(str)

    df = df[df["true_action"].isin(valid_actions)].copy()

    if len(df) == 0:
        raise ValueError(
            "After standardizing prior source, no valid true_action rows remained."
        )

    action2id = {a: i for i, a in enumerate(id2action)}
    df["last_action_id"] = df["last_action"].map(action2id)
    df["action_id"] = df["true_action"].map(action2id)

    df = df.dropna(subset=["action_id"]).copy()
    df["action_id"] = df["action_id"].astype(int)
    df["last_action_id"] = df["last_action_id"].fillna(-1).astype(int)

    return df[["last_action", "true_action", "last_action_id", "action_id"]]


def build_log_prior(prior_df: pd.DataFrame, num_classes: int, smoothing: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Return dict last_action -> log probability vector, and global log prior."""
    global_counts = np.bincount(prior_df["action_id"].values, minlength=num_classes).astype(float) + smoothing
    global_probs = global_counts / global_counts.sum()
    global_log_prior = np.log(global_probs)

    prior: Dict[str, np.ndarray] = {}
    for last_action, g in prior_df.groupby("last_action"):
        counts = np.bincount(g["action_id"].values, minlength=num_classes).astype(float) + smoothing
        probs = counts / counts.sum()
        prior[str(last_action)] = np.log(probs)
    return prior, global_log_prior


def apply_prior(
    logits: np.ndarray,
    last_actions: np.ndarray,
    log_prior: Dict[str, np.ndarray],
    global_log_prior: np.ndarray,
    alpha: float,
) -> np.ndarray:
    adjusted = logits.copy()
    for i, last_action in enumerate(last_actions):
        adjusted[i] += alpha * log_prior.get(str(last_action), global_log_prior)
    return adjusted


def per_class_f1_df(y_true: np.ndarray, y_pred: np.ndarray, id2action: List[str]) -> pd.DataFrame:
    rows = []
    for i, action in enumerate(id2action):
        mask_true = y_true == i
        mask_pred = y_pred == i
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append({
            "action": action,
            "support": int(mask_true.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows).sort_values("f1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", default="submit/model/deberta/val_predictions.csv")
    ap.add_argument("--val_logits", default="submit/model/deberta/val_logits.npy")
    ap.add_argument("--prior_source", default=None, help="Prefer data/train.csv here to avoid val-label leakage.")
    ap.add_argument("--out_dir", default="submit/model/deberta/prior_tuning")
    ap.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    ap.add_argument("--smoothings", default=",".join(map(str, DEFAULT_SMOOTHINGS)))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.val_csv)
    logits = np.load(args.val_logits)
    if len(val_df) != len(logits):
        raise ValueError(f"val_csv rows ({len(val_df)}) != val_logits rows ({len(logits)})")

    num_classes = logits.shape[1]
    id2action = infer_id2action(val_df, num_classes)
    y_true = val_df["true_id"].astype(int).values
    last_actions = val_df["last_action"].fillna(NONE_TOKEN).astype(str).values

    prior_src_raw, used_val_as_prior = load_prior_source(args.prior_source, val_df)
    prior_df = standardize_prior_df(prior_src_raw, id2action)

    alphas = parse_float_list(args.alphas)
    smoothings = parse_float_list(args.smoothings)

    base_pred = logits.argmax(axis=1)
    base_macro_f1 = f1_score(y_true, base_pred, average="macro")
    base_acc = accuracy_score(y_true, base_pred)

    rows = []
    best = None
    for smoothing in smoothings:
        log_prior, global_log_prior = build_log_prior(prior_df, num_classes, smoothing)
        for alpha in alphas:
            adjusted = apply_prior(logits, last_actions, log_prior, global_log_prior, alpha)
            pred = adjusted.argmax(axis=1)
            macro_f1 = f1_score(y_true, pred, average="macro")
            acc = accuracy_score(y_true, pred)
            row = {
                "smoothing": smoothing,
                "alpha": alpha,
                "macro_f1": macro_f1,
                "delta_macro_f1": macro_f1 - base_macro_f1,
                "accuracy": acc,
                "delta_accuracy": acc - base_acc,
                "changed_predictions": int((pred != base_pred).sum()),
            }
            rows.append(row)
            if best is None or row["macro_f1"] > best["macro_f1"]:
                best = row

    results = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    results.to_csv(out_dir / "transition_prior_grid.csv", index=False)

    # Save best adjusted validation predictions.
    assert best is not None
    best_log_prior, best_global = build_log_prior(prior_df, num_classes, best["smoothing"])
    best_logits = apply_prior(logits, last_actions, best_log_prior, best_global, best["alpha"])
    best_pred = best_logits.argmax(axis=1)
    best_conf = np.exp(best_logits - best_logits.max(axis=1, keepdims=True))
    best_conf = best_conf / best_conf.sum(axis=1, keepdims=True)

    adj_df = val_df.copy()
    adj_df["base_pred_id"] = base_pred
    adj_df["base_pred_action"] = [id2action[i] for i in base_pred]
    adj_df["prior_pred_id"] = best_pred
    adj_df["prior_pred_action"] = [id2action[i] for i in best_pred]
    adj_df["prior_confidence"] = best_conf.max(axis=1)
    adj_df["prior_correct"] = best_pred == y_true
    adj_df["prior_changed"] = best_pred != base_pred
    adj_df.to_csv(out_dir / "val_predictions_with_best_prior.csv", index=False)

    # Save transition prior table for inspection / future script_deberta.py implementation.
    prior_table_rows = []
    for last_action, logp in sorted(best_log_prior.items()):
        probs = np.exp(logp)
        for i, action in enumerate(id2action):
            prior_table_rows.append({
                "last_action": last_action,
                "action_id": i,
                "action": action,
                "prob": probs[i],
                "log_prob": logp[i],
            })
    pd.DataFrame(prior_table_rows).to_csv(out_dir / "best_transition_prior_table.csv", index=False)

    # Per-class comparison.
    base_pc = per_class_f1_df(y_true, base_pred, id2action).rename(columns={
        "precision": "base_precision", "recall": "base_recall", "f1": "base_f1"
    })
    prior_pc = per_class_f1_df(y_true, best_pred, id2action).rename(columns={
        "precision": "prior_precision", "recall": "prior_recall", "f1": "prior_f1"
    })
    pc = base_pc.merge(prior_pc[["action", "prior_precision", "prior_recall", "prior_f1"]], on="action")
    pc["delta_f1"] = pc["prior_f1"] - pc["base_f1"]
    pc.sort_values("delta_f1").to_csv(out_dir / "per_class_prior_delta.csv", index=False)

    summary = {
        "used_val_as_prior_source_warning": used_val_as_prior,
        "val_csv": args.val_csv,
        "val_logits": args.val_logits,
        "prior_source": args.prior_source or "VAL_PREDICTIONS_FALLBACK_LEAKY_FOR_PROBING_ONLY",
        "id2action": id2action,
        "base": {"macro_f1": base_macro_f1, "accuracy": base_acc},
        "best": best,
        "recommendation": "use prior only if best alpha > 0 and delta_macro_f1 is positive on a non-leaky prior source",
    }
    with open(out_dir / "transition_prior_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Transition prior tuning =====")
    if used_val_as_prior:
        print("[WARNING] prior_source was not provided, so the prior was estimated from validation labels.")
        print("          This is useful only as a quick probe. For clean tuning, pass --prior_source data/train.csv")
    print(f"Base Macro-F1: {base_macro_f1:.6f} | Acc: {base_acc:.6f}")
    print("\nTop grid rows:")
    print(results.head(15).to_string(index=False))
    print("\nBest:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
