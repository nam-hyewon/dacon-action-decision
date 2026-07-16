#!/usr/bin/env python3
"""
Specialist booster tuning for DACON AI Agent Action Decision Prediction.

Goal:
  Improve weak classes by training one-vs-rest binary TF-IDF classifiers
  on the same train split, then adding class-specific logit boosts on top
  of saved DeBERTa validation logits.

Recommended:
  python src/tune_specialist_boosters.py \
    --val_csv submit/model/deberta_small_3ep_len512/val_predictions.csv \
    --val_logits submit/model/deberta_small_3ep_len512/val_logits.npy \
    --out_dir submit/model/deberta_small_3ep_len512/specialist_boosters \
    --greedy
"""
from __future__ import annotations

import argparse
import csv
import json
import joblib
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {a: i for i, a in enumerate(ALL_CLASSES)}
ID2LABEL = {i: a for a, i in LABEL2ID.items()}

DEFAULT_TARGETS = [
    "web_search", "list_directory", "lint_or_typecheck", "ask_user", "plan_task",
    "read_file", "grep_search", "run_tests", "run_bash", "glob_pattern",
]
DEFAULT_GAMMAS = [0.0, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5]
EPS = 1e-5


def parse_list(s: str, typ=str):
    return [typ(x.strip()) for x in str(s).split(",") if x.strip()]


def get_last_action(sample):
    actions = [h.get("name", "") for h in sample.get("history", []) if h.get("role") == "assistant_action"]
    return actions[-1] if actions else "none"


def get_prev_action(sample):
    actions = [h.get("name", "") for h in sample.get("history", []) if h.get("role") == "assistant_action"]
    return actions[-2] if len(actions) >= 2 else "none"


def get_step_from_id(sid: str) -> str:
    m = re.search(r"step_(\d+)", str(sid))
    return m.group(1) if m else "none"


def compact_history(sample, max_turns=10):
    hist = sample.get("history", [])[-max_turns:]
    parts, actions, users = [], [], []
    last_result = ""
    for h in hist:
        role = h.get("role", "")
        if role == "user":
            txt = str(h.get("content", ""))
            users.append(txt)
            parts.append(f"U: {txt}")
        elif role == "assistant_action":
            name = str(h.get("name", ""))
            res = str(h.get("result_summary", ""))
            args = str(h.get("args", ""))
            actions.append(name)
            last_result = res
            parts.append(f"ACTION: {name} ARGS: {args} RESULT: {res}")
    return {
        "history_text": "\n".join(parts),
        "action_seq": " > ".join(actions),
        "recent_users": "\n".join(users[-3:]),
        "last_result": last_result,
    }


def build_text(sample, mode="router"):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    current_prompt = str(sample.get("current_prompt", ""))
    sid = str(sample.get("id", ""))
    ch = compact_history(sample)
    last_action = get_last_action(sample)
    prev_action = get_prev_action(sample)
    step = get_step_from_id(sid)
    lm = workspace.get("language_mix", {}) or {}
    main_lang = max(lm, key=lm.get) if lm else "none"
    open_files = " ".join(map(str, workspace.get("open_files", [])))
    lang_mix = " ".join(f"{k}:{v}" for k, v in lm.items())

    if mode == "prompt":
        return "\n".join([
            "[CURRENT_PROMPT]", current_prompt,
            "[CURRENT_PROMPT_AGAIN]", current_prompt,
            "[CURRENT_PROMPT_THIRD]", current_prompt,
        ])

    if mode == "prompt_meta":
        return "\n".join([
            "[CURRENT_PROMPT]", current_prompt,
            "[CURRENT_PROMPT_AGAIN]", current_prompt,
            "[ACTION_SEQUENCE]", ch["action_seq"],
            "[LAST_ACTION]", last_action,
            "[PREV_ACTION]", prev_action,
            "[LAST_RESULT]", ch["last_result"],
            "[STATE]",
            f"step: {step}",
            f"turn_index: {meta.get('turn_index', '')}",
            f"git_dirty: {workspace.get('git_dirty', '')}",
            f"last_ci_status: {workspace.get('last_ci_status', '')}",
            f"main_lang: {main_lang}",
            f"open_files: {open_files}",
        ])

    return "\n".join([
        "[ROUTER_TASK] Predict the next assistant action class.",
        "[CURRENT_USER_REQUEST]", current_prompt,
        "[CURRENT_USER_REQUEST_REPEAT]", current_prompt,
        "[RECENT_USER_REQUESTS]", ch["recent_users"],
        "[ACTION_SEQUENCE]", ch["action_seq"],
        "[PREVIOUS_ACTION]", prev_action,
        "[LAST_ACTION]", last_action,
        "[LAST_ACTION_RESULT]", ch["last_result"],
        "[WORKSPACE_STATE]",
        f"step: {step}",
        f"turn_index: {meta.get('turn_index', '')}",
        f"user_tier: {meta.get('user_tier', '')}",
        f"language_pref: {meta.get('language_pref', '')}",
        f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}",
        f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"main_lang: {main_lang}",
        f"language_mix: {lang_mix}",
        f"open_files: {open_files}",
        "[COMPACT_HISTORY]", ch["history_text"],
    ])


def load_rows(data_dir: Path, text_mode: str):
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
    for s in samples:
        sid = s["id"]
        label_name = labels[sid]
        rows.append({
            "id": sid,
            "label": LABEL2ID[label_name],
            "label_name": label_name,
            "text": build_text(s, mode=text_mode),
        })
    return rows


def split_like_train(rows, seed=42):
    labels = [r["label"] for r in rows]
    return train_test_split(rows, test_size=0.2, stratify=labels, random_state=seed)


def align_val_rows(split_val_rows, val_df):
    id_to_row = {r["id"]: r for r in split_val_rows}
    ordered = []
    missing = []
    for sid in val_df["id"].astype(str).tolist():
        if sid not in id_to_row:
            missing.append(sid)
        else:
            ordered.append(id_to_row[sid])
    if missing:
        raise ValueError(f"Validation split mismatch. Missing {len(missing)} ids. First: {missing[:3]}")
    return ordered


def infer_id2action(val_df, logits):
    n = logits.shape[1]
    pairs = []
    for id_col, act_col in [("true_id", "true_action"), ("pred_id", "pred_action")]:
        if id_col in val_df.columns and act_col in val_df.columns:
            pairs.append(val_df[[id_col, act_col]].rename(columns={id_col: "id", act_col: "action"}))
    if not pairs:
        return ALL_CLASSES[:n]
    all_pairs = pd.concat(pairs).drop_duplicates()
    id_to_acts = {}
    for _, r in all_pairs.iterrows():
        id_to_acts.setdefault(int(r["id"]), set()).add(str(r["action"]))
    out = []
    for i in range(n):
        acts = sorted(id_to_acts.get(i, []))
        out.append(acts[0] if len(acts) == 1 else ID2LABEL[i])
    return out


def safe_logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def make_vectorizer(mode: str, max_features: int, min_df: int):
    if mode == "prompt":
        return TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True,
            min_df=min_df, max_features=max_features, sublinear_tf=True, norm="l2"
        )
    return TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), lowercase=True,
        min_df=min_df, max_features=max_features, sublinear_tf=True, norm="l2"
    )


def train_specialists_for_mode(train_rows, val_rows, targets, id2action, mode, max_features=250000, min_df=2, C=2.0):
    train_texts = [r["text"] for r in train_rows]
    val_texts = [r["text"] for r in val_rows]
    y_train_multi = np.array([r["label"] for r in train_rows], dtype=np.int64)
    n = len(val_rows)
    cnum = len(id2action)

    vec = make_vectorizer(mode, max_features=max_features, min_df=min_df)
    Xtr = vec.fit_transform(train_texts)
    Xva = vec.transform(val_texts)

    scores = np.zeros((n, cnum), dtype=np.float32)
    diag_rows = []
    clfs = {}

    for target in targets:
        if target not in id2action:
            continue
        tid = id2action.index(target)
        y_bin = (y_train_multi == tid).astype(int)
        pos = int(y_bin.sum())
        neg = int((1 - y_bin).sum())
        if pos < 20:
            print(f"[WARN] skip {target}: too few positives {pos}")
            continue

        clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(Xtr, y_bin)
        clfs[target] = clf
        p = clf.predict_proba(Xva)[:, 1]
        s = safe_logit(p).astype(np.float32)
        s = np.clip(s, -8, 8)
        scores[:, tid] = s

        diag_rows.append({
            "mode": mode, "target": target, "train_pos": pos, "train_neg": neg,
            "score_mean": float(s.mean()), "score_std": float(s.std()),
            "score_min": float(s.min()), "score_max": float(s.max()),
        })
    artifact = {
        "mode": mode,
        "vectorizer": vec,
        "clfs": clfs,
    }
    return scores, pd.DataFrame(diag_rows), artifact


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
        rows.append({"action": action, "support": int(mt.sum()), "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--val_logits", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", default="router,prompt_meta,prompt")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--gammas", default=",".join(map(str, DEFAULT_GAMMAS)))
    ap.add_argument("--max_features", type=int, default=250000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    val_df = pd.read_csv(args.val_csv)
    logits = np.load(args.val_logits)
    id2action = infer_id2action(val_df, logits)
    y_true = val_df["true_id"].astype(int).values
    base_pred = logits.argmax(axis=1)
    base_f1 = f1_score(y_true, base_pred, labels=list(range(len(id2action))), average="macro", zero_division=0)
    base_acc = accuracy_score(y_true, base_pred)

    print("===== Specialist booster tuning =====")
    print("Base Macro-F1:", base_f1, "| Acc:", base_acc)

    modes = parse_list(args.modes, str)
    targets = parse_list(args.targets, str)
    gammas = parse_list(args.gammas, float)

    all_results, all_diag = [], []
    best, best_logits, best_pred, best_scores, best_artifact = None, None, None, None, None
    selected_for_artifact = None

    for mode in modes:
        print(f"\n[INFO] Training specialist mode={mode}")
        rows = load_rows(data_dir, text_mode=mode)
        train_rows, val_rows_split = split_like_train(rows, seed=args.seed)
        val_rows = align_val_rows(val_rows_split, val_df)
        scores, diag, artifact = train_specialists_for_mode(
            train_rows, val_rows, targets, id2action, mode,
            max_features=args.max_features, min_df=args.min_df, C=args.C
        )
        all_diag.append(diag)

        for gamma in gammas:
            adjusted = logits + gamma * scores
            pred = adjusted.argmax(axis=1)
            mf1 = f1_score(y_true, pred, labels=list(range(len(id2action))), average="macro", zero_division=0)
            acc = accuracy_score(y_true, pred)
            row = {
                "mode": mode, "gamma": gamma,
                "macro_f1": mf1, "delta_macro_f1": mf1 - base_f1,
                "accuracy": acc, "delta_accuracy": acc - base_acc,
                "changed_predictions": int((pred != base_pred).sum()),
            }
            all_results.append(row)
            if best is None or (mf1, acc) > (best["macro_f1"], best["accuracy"]):
                best, best_logits, best_pred, best_scores, best_artifact = row, adjusted, pred, scores, artifact

    res = pd.DataFrame(all_results).sort_values(["macro_f1", "accuracy"], ascending=False)
    res.to_csv(out_dir / "specialist_global_grid.csv", index=False)
    if all_diag:
        pd.concat(all_diag, ignore_index=True).to_csv(out_dir / "specialist_diagnostics.csv", index=False)

    print("\nTop global:")
    print(res.head(30).to_string(index=False))

    greedy_info = None
    if args.greedy and best_scores is not None:
        print("\n[INFO] Running greedy per-target gamma selection...")
        adjusted = logits.copy()
        selected = []
        cur_pred = adjusted.argmax(axis=1)
        cur_f1 = f1_score(y_true, cur_pred, labels=list(range(len(id2action))), average="macro", zero_division=0)
        action2id = {a: i for i, a in enumerate(id2action)}

        for target in targets:
            if target not in action2id:
                continue
            tid = action2id[target]
            local = {"target": target, "gamma": 0.0, "macro_f1": cur_f1, "accuracy": accuracy_score(y_true, cur_pred)}
            for gamma in gammas:
                cand = adjusted.copy()
                cand[:, tid] += gamma * best_scores[:, tid]
                pred = cand.argmax(axis=1)
                mf1 = f1_score(y_true, pred, labels=list(range(len(id2action))), average="macro", zero_division=0)
                acc = accuracy_score(y_true, pred)
                if (mf1, acc) > (local["macro_f1"], local["accuracy"]):
                    local = {"target": target, "gamma": gamma, "macro_f1": mf1, "accuracy": acc}
            if local["gamma"] != 0.0:
                adjusted[:, tid] += local["gamma"] * best_scores[:, tid]
                cur_pred = adjusted.argmax(axis=1)
                cur_f1 = local["macro_f1"]
            selected.append(local)

        greedy_pred = adjusted.argmax(axis=1)
        greedy_f1 = f1_score(y_true, greedy_pred, labels=list(range(len(id2action))), average="macro", zero_division=0)
        greedy_acc = accuracy_score(y_true, greedy_pred)
        pd.DataFrame(selected).to_csv(out_dir / "specialist_greedy_selected.csv", index=False)
        selected_for_artifact = selected
        greedy_info = {
            "macro_f1": float(greedy_f1),
            "delta_macro_f1": float(greedy_f1 - base_f1),
            "accuracy": float(greedy_acc),
            "delta_accuracy": float(greedy_acc - base_acc),
            "changed_predictions": int((greedy_pred != base_pred).sum()),
        }
        print("\nGreedy result:")
        print(json.dumps(greedy_info, indent=2))
        if (greedy_f1, greedy_acc) > (best["macro_f1"], best["accuracy"]):
            best = {"mode": best["mode"], "gamma": "greedy_per_target", **greedy_info}
            best_logits, best_pred = adjusted, greedy_pred

    print("\nBest:")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    # Save deployable specialist artifact for test-time inference.
    if best_artifact is not None:
        deploy_artifact = dict(best_artifact)
        deploy_artifact["all_classes"] = id2action
        deploy_artifact["targets"] = targets
        deploy_artifact["score_clip"] = [-8, 8]
        deploy_artifact["safe_logit_eps"] = EPS
        deploy_artifact["max_features"] = args.max_features
        deploy_artifact["min_df"] = args.min_df
        deploy_artifact["C"] = args.C

        if best.get("gamma") == "greedy_per_target" and selected_for_artifact is not None:
            gamma_map = {row["target"]: float(row["gamma"]) for row in selected_for_artifact}
        else:
            g = float(best.get("gamma", 0.0))
            gamma_map = {target: g for target in deploy_artifact["clfs"].keys()}

        deploy_artifact["gammas"] = gamma_map
        joblib.dump(deploy_artifact, out_dir / "specialist_booster.joblib")
        print(f"[INFO] Saved deployable specialist artifact to {out_dir / 'specialist_booster.joblib'}")

    probs = np.exp(best_logits - best_logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    out_pred = val_df.copy()
    out_pred["specialist_pred_id"] = best_pred
    out_pred["specialist_pred_action"] = [id2action[i] for i in best_pred]
    out_pred["specialist_confidence"] = probs.max(axis=1)
    out_pred["specialist_correct"] = best_pred == y_true
    out_pred["specialist_changed"] = best_pred != base_pred
    out_pred.to_csv(out_dir / "val_predictions_with_specialists.csv", index=False)

    base_pc = per_class_metrics(y_true, base_pred, id2action).rename(columns={"precision": "base_precision", "recall": "base_recall", "f1": "base_f1"})
    sp_pc = per_class_metrics(y_true, best_pred, id2action).rename(columns={"precision": "specialist_precision", "recall": "specialist_recall", "f1": "specialist_f1"})
    pc = base_pc.merge(sp_pc[["action", "specialist_precision", "specialist_recall", "specialist_f1"]], on="action")
    pc["delta_f1"] = pc["specialist_f1"] - pc["base_f1"]
    pc.sort_values("delta_f1").to_csv(out_dir / "per_class_specialist_delta.csv", index=False)

    summary = {
        "base": {"macro_f1": float(base_f1), "accuracy": float(base_acc)},
        "best": best,
        "greedy": greedy_info,
        "targets": targets,
        "modes": modes,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "specialist_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(" -", out_dir / "specialist_global_grid.csv")
    print(" -", out_dir / "specialist_summary.json")
    print(" -", out_dir / "per_class_specialist_delta.csv")
    print(" -", out_dir / "val_predictions_with_specialists.csv")


if __name__ == "__main__":
    main()
