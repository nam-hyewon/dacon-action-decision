#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def parse_float_list(value):
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_last_action(sample):
    actions = get_actions(sample)
    return actions[-1] if actions else "none"


def get_prev_action(sample):
    actions = get_actions(sample)
    return actions[-2] if len(actions) >= 2 else "none"


def get_step_from_id(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return m.group(1) if m else "none"


def bucket_int(value, cuts):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in cuts:
        if v <= upper:
            return name
    return f"gt_{cuts[-1][1]}"


def make_key(sample, mode):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    turn_bucket = bucket_int(
        meta.get("turn_index", 0),
        [("t1", 1), ("t2_4", 4), ("t5_8", 8), ("t9_12", 12)],
    )
    step_bucket = bucket_int(
        get_step_from_id(sample.get("id", "")),
        [("s1", 1), ("s2_4", 4), ("s5_8", 8), ("s9_12", 12)],
    )
    ci = str(workspace.get("last_ci_status", ""))
    dirty = str(workspace.get("git_dirty", ""))
    last = get_last_action(sample)
    prev = get_prev_action(sample)

    if mode == "last":
        return ("last", last)
    if mode == "prev_last":
        return ("prev_last", prev, last)
    if mode == "last_turn":
        return ("last_turn", last, turn_bucket)
    if mode == "last_step":
        return ("last_step", last, step_bucket)
    if mode == "last_ci":
        return ("last_ci", last, ci)
    if mode == "last_dirty":
        return ("last_dirty", last, dirty)
    if mode == "turn":
        return ("turn", turn_bucket)
    if mode == "step":
        return ("step", step_bucket)
    raise ValueError(f"unknown prior mode={mode}")


def load_rows(data_dir):
    samples = []
    with open(data_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    labels = {}
    with open(data_dir / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    rows = []
    for sample in samples:
        sid = str(sample["id"])
        label_name = labels[sid]
        rows.append({
            "id": sid,
            "sample": sample,
            "label": LABEL2ID[label_name],
            "label_name": label_name,
        })
    return rows


def split_like_train(rows, seed):
    labels = [r["label"] for r in rows]
    return train_test_split(
        rows,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )


def align_val_rows(split_val_rows, val_df):
    id_to_row = {r["id"]: r for r in split_val_rows}
    ordered = []
    missing = []
    for sid in val_df["id"].astype(str).tolist():
        row = id_to_row.get(sid)
        if row is None:
            missing.append(sid)
        else:
            ordered.append(row)
    if missing:
        raise ValueError(
            f"Validation split mismatch. Missing {len(missing)} ids. "
            f"First: {missing[:3]}"
        )
    return ordered


def build_prior(train_rows, mode, smoothing):
    n_classes = len(ALL_CLASSES)
    global_counts = np.zeros(n_classes, dtype=np.float64)
    key_counts = defaultdict(lambda: np.zeros(n_classes, dtype=np.float64))
    key_support = Counter()

    for row in train_rows:
        label = int(row["label"])
        key = make_key(row["sample"], mode)
        global_counts[label] += 1.0
        key_counts[key][label] += 1.0
        key_support[key] += 1

    global_prob = (global_counts + smoothing) / (
        global_counts.sum() + smoothing * n_classes
    )
    global_logprob = np.log(global_prob)

    key_logprob = {}
    for key, counts in key_counts.items():
        prob = (counts + smoothing) / (counts.sum() + smoothing * n_classes)
        key_logprob[key] = np.log(prob)

    return key_logprob, global_logprob, key_support


def apply_prior(logits, val_rows, mode, key_logprob, global_logprob, alpha, center_global):
    adjusted = logits.astype(np.float64).copy()
    for i, row in enumerate(val_rows):
        key = make_key(row["sample"], mode)
        prior = key_logprob.get(key, global_logprob)
        if center_global:
            prior = prior - global_logprob
        adjusted[i] += alpha * prior
    return adjusted


def infer_id2action(val_df, n_classes):
    if "true_action" in val_df.columns:
        pairs = val_df[["true_id", "true_action"]].drop_duplicates()
        mapping = {
            int(r["true_id"]): str(r["true_action"])
            for _, r in pairs.iterrows()
        }
        return [mapping.get(i, ID2LABEL[i]) for i in range(n_classes)]
    return ALL_CLASSES[:n_classes]


def per_class_metrics(y_true, pred, id2action):
    rows = []
    for i, action in enumerate(id2action):
        mt = y_true == i
        mp = pred == i
        tp = int((mt & mp).sum())
        fp = int((~mt & mp).sum())
        fn = int((mt & ~mp).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({
            "action": action,
            "support": int(mt.sum()),
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--val_logits", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        default="last,prev_last,last_turn,last_step,last_ci,last_dirty,turn,step",
    )
    parser.add_argument(
        "--alphas",
        default="0,0.02,0.04,0.06,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40",
    )
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--raw_logprob", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    val_df = pd.read_csv(args.val_csv)
    logits = np.load(args.val_logits).astype(np.float64)
    if len(val_df) != len(logits):
        raise ValueError(
            f"row mismatch: val_csv rows={len(val_df)} logits rows={len(logits)}"
        )

    rows = load_rows(data_dir)
    train_rows, val_rows_split = split_like_train(rows, args.seed)
    val_rows = align_val_rows(val_rows_split, val_df)

    n_classes = logits.shape[1]
    id2action = infer_id2action(val_df, n_classes)
    y_true = val_df["true_id"].astype(int).values
    base_pred = logits.argmax(axis=1)
    base_f1 = f1_score(
        y_true,
        base_pred,
        labels=list(range(n_classes)),
        average="macro",
        zero_division=0,
    )
    base_acc = accuracy_score(y_true, base_pred)

    print("===== Transition prior tuning =====")
    print("Base Macro-F1:", base_f1, "| Acc:", base_acc)
    print("center_global:", not args.raw_logprob)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    alphas = parse_float_list(args.alphas)
    center_global = not args.raw_logprob

    all_results = []
    best = None
    best_logits = None
    best_pred = None

    for mode in modes:
        key_logprob, global_logprob, key_support = build_prior(
            train_rows,
            mode,
            args.smoothing,
        )
        print(
            f"\n[INFO] mode={mode} keys={len(key_logprob)} "
            f"min_support={min(key_support.values()) if key_support else 0} "
            f"median_support={np.median(list(key_support.values())) if key_support else 0:.1f}"
        )
        for alpha in alphas:
            adjusted = apply_prior(
                logits,
                val_rows,
                mode,
                key_logprob,
                global_logprob,
                alpha,
                center_global=center_global,
            )
            pred = adjusted.argmax(axis=1)
            macro_f1 = f1_score(
                y_true,
                pred,
                labels=list(range(n_classes)),
                average="macro",
                zero_division=0,
            )
            acc = accuracy_score(y_true, pred)
            row = {
                "mode": mode,
                "alpha": alpha,
                "macro_f1": macro_f1,
                "delta_macro_f1": macro_f1 - base_f1,
                "accuracy": acc,
                "delta_accuracy": acc - base_acc,
                "changed_predictions": int((pred != base_pred).sum()),
                "num_keys": len(key_logprob),
            }
            all_results.append(row)
            if best is None or (macro_f1, acc) > (best["macro_f1"], best["accuracy"]):
                best = row
                best_logits = adjusted
                best_pred = pred

    res = pd.DataFrame(all_results).sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    )
    res.to_csv(out_dir / "transition_prior_grid.csv", index=False)

    print("\nTop transition prior grid:")
    print(res.head(30).to_string(index=False))
    print("\nBest:")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    out_pred = val_df.copy()
    out_pred["prior_pred_id"] = best_pred
    out_pred["prior_pred_action"] = [id2action[i] for i in best_pred]
    out_pred["prior_correct"] = best_pred == y_true
    out_pred["prior_changed"] = best_pred != base_pred
    out_pred.to_csv(out_dir / "val_predictions_with_transition_prior.csv", index=False)
    np.save(out_dir / "val_logits_with_transition_prior.npy", best_logits)

    base_pc = per_class_metrics(y_true, base_pred, id2action).rename(
        columns={
            "precision": "base_precision",
            "recall": "base_recall",
            "f1": "base_f1",
        }
    )
    prior_pc = per_class_metrics(y_true, best_pred, id2action).rename(
        columns={
            "precision": "prior_precision",
            "recall": "prior_recall",
            "f1": "prior_f1",
        }
    )
    pc = base_pc.merge(
        prior_pc[["action", "prior_precision", "prior_recall", "prior_f1"]],
        on="action",
    )
    pc["delta_f1"] = pc["prior_f1"] - pc["base_f1"]
    pc.sort_values("delta_f1").to_csv(out_dir / "per_class_transition_prior_delta.csv", index=False)

    summary = {
        "base": {"macro_f1": float(base_f1), "accuracy": float(base_acc)},
        "best": best,
        "modes": modes,
        "alphas": alphas,
        "smoothing": args.smoothing,
        "center_global": center_global,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "transition_prior_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(" -", out_dir / "transition_prior_grid.csv")
    print(" -", out_dir / "transition_prior_summary.json")
    print(" -", out_dir / "per_class_transition_prior_delta.csv")
    print(" -", out_dir / "val_predictions_with_transition_prior.csv")
    print(" -", out_dir / "val_logits_with_transition_prior.npy")


if __name__ == "__main__":
    main()
