import csv
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "eda_outputs/best_error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_DIR = ROOT / "submit/model/eda_mdeberta_deberta_w054_046"
VAL_CSV = BEST_DIR / "val_predictions.csv"
VAL_LOGITS = BEST_DIR / "val_logits.npy"

ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}

FILE_GROUP = {"read_file", "grep_search", "list_directory", "glob_pattern"}


def find_cols(df):
    id_col = "id"
    true_col = None
    pred_col = None

    for c in ["true_action", "true", "label", "true_label", "target"]:
        if c in df.columns:
            true_col = c
            break

    for c in ["pred_action", "pred", "prediction", "pred_label"]:
        if c in df.columns:
            pred_col = c
            break

    if true_col is None:
        raise ValueError(f"Cannot find true label col: {df.columns.tolist()}")
    if pred_col is None:
        raise ValueError(f"Cannot find pred label col: {df.columns.tolist()}")

    return id_col, true_col, pred_col


def clean(x, n=1200):
    x = re.sub(r"\s+", " ", str(x)).strip()
    return x[:n]


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_user_turns(sample):
    return [
        str(h.get("content", ""))
        for h in sample.get("history", [])
        if h.get("role") == "user"
    ]


def get_last_action_info(sample):
    for h in reversed(sample.get("history", [])):
        if h.get("role") == "assistant_action":
            return (
                str(h.get("name", "")),
                clean(json.dumps(h.get("args", {}), ensure_ascii=False), 600),
                clean(h.get("result_summary", ""), 1000),
            )
    return "none", "", ""


def get_step(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return int(m.group(1)) if m else -1


def summarize_sample(sample):
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    actions = get_actions(sample)
    users = get_user_turns(sample)
    last_action, last_args, last_result = get_last_action_info(sample)

    return {
        "step": get_step(sample.get("id", "")),
        "current_prompt": clean(sample.get("current_prompt", ""), 1600),
        "recent_users": clean(" || ".join(users[-3:]), 1200),
        "last_action": last_action,
        "prev_action": actions[-2] if len(actions) >= 2 else "none",
        "action_seq_last5": ">".join(actions[-5:]),
        "last_args": last_args,
        "last_result": last_result,
        "open_files": clean(workspace.get("open_files", []), 700),
        "last_ci_status": workspace.get("last_ci_status", ""),
        "git_dirty": workspace.get("git_dirty", ""),
        "language_mix": clean(workspace.get("language_mix", {}), 500),
    }


def macro_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        labels=list(range(len(ACTIONS))),
        average="macro",
        zero_division=0,
    )


def main():
    print("[LOAD]")
    val = pd.read_csv(VAL_CSV)
    logits = np.load(VAL_LOGITS)
    probs = softmax(logits, axis=1)

    id_col, true_col, pred_col = find_cols(val)

    ids = val[id_col].astype(str).tolist()
    true_labels = val[true_col].astype(str).tolist()
    y_true = np.array([ACTION_TO_ID[x] for x in true_labels], dtype=np.int64)

    pred_ids = probs.argmax(axis=1)
    pred_labels = [ID_TO_ACTION[i] for i in pred_ids]
    conf = probs.max(axis=1)

    top2_ids = np.argsort(-probs, axis=1)[:, :2]
    top2_labels = [[ID_TO_ACTION[i] for i in row] for row in top2_ids]
    margin = probs[top2_ids[:, 0], np.arange(len(probs))] if False else None
    top1_prob = probs[np.arange(len(probs)), top2_ids[:, 0]]
    top2_prob = probs[np.arange(len(probs)), top2_ids[:, 1]]
    prob_margin = top1_prob - top2_prob

    # load train samples
    samples = {}
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                samples[str(s["id"])] = s

    df = pd.DataFrame({
        "id": ids,
        "true_label": true_labels,
        "pred_label": pred_labels,
        "confidence": conf,
        "top1_prob": top1_prob,
        "top2_label": [x[1] for x in top2_labels],
        "top2_prob": top2_prob,
        "prob_margin": prob_margin,
        "correct": np.array(true_labels) == np.array(pred_labels),
    })

    # 1. overall score
    overall = {
        "macro_f1": macro_f1(y_true, pred_ids),
        "accuracy": accuracy_score(y_true, pred_ids),
        "n": len(df),
    }
    pd.DataFrame([overall]).to_csv(OUT_DIR / "00_overall_score.csv", index=False)
    print("[OVERALL]", overall)

    # 2. top confusion pairs
    wrong = df[df["correct"] == False].copy()
    pairs = (
        wrong.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    pairs.to_csv(OUT_DIR / "01_top_confusion_pairs.csv", index=False)
    print("\n[TOP CONFUSIONS]")
    print(pairs.head(30).to_string(index=False))

    # 3. confidence bucket analysis
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    labels = ["0-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    df["conf_bucket"] = pd.cut(df["confidence"], bins=bins, labels=labels, include_lowest=True)

    bucket_rows = []
    for b, sub in df.groupby("conf_bucket", observed=False):
        if len(sub) == 0:
            continue
        yt = np.array([ACTION_TO_ID[x] for x in sub["true_label"]])
        yp = np.array([ACTION_TO_ID[x] for x in sub["pred_label"]])
        bucket_rows.append({
            "conf_bucket": str(b),
            "n": len(sub),
            "accuracy": accuracy_score(yt, yp),
            "macro_f1": macro_f1(yt, yp),
            "wrong_count": int((sub["correct"] == False).sum()),
            "file_group_count": int(sub["true_label"].isin(FILE_GROUP).sum()),
            "file_group_wrong": int(((sub["true_label"].isin(FILE_GROUP)) & (sub["correct"] == False)).sum()),
        })

    bucket_df = pd.DataFrame(bucket_rows)
    bucket_df.to_csv(OUT_DIR / "02_confidence_bucket_scores.csv", index=False)
    print("\n[CONF BUCKETS]")
    print(bucket_df.to_string(index=False))

    # 4. class report
    report = classification_report(
        y_true,
        pred_ids,
        target_names=ACTIONS,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(OUT_DIR / "03_classification_report.csv")

    # 5. file-search group low-confidence wrong examples
    file_low_wrong = df[
        (df["true_label"].isin(FILE_GROUP))
        & (df["pred_label"].isin(FILE_GROUP))
        & (df["correct"] == False)
        & (df["confidence"] < 0.80)
    ].copy()

    file_low_wrong = file_low_wrong.sort_values(["confidence", "prob_margin"], ascending=[True, True])

    rows = []
    for _, r in file_low_wrong.head(100).iterrows():
        sid = str(r["id"])
        sample = samples.get(sid, {})
        info = summarize_sample(sample)
        row = r.to_dict()
        row.update(info)
        rows.append(row)

    review = pd.DataFrame(rows)
    review.to_csv(OUT_DIR / "04_file_group_low_conf_wrong_examples.csv", index=False)

    # 6. exactly 20 examples for quick viewing
    review.head(20).to_csv(OUT_DIR / "04_file_group_low_conf_wrong_examples_top20.csv", index=False)

    print("\n[FILE GROUP LOW CONF WRONG TOP20]")
    cols = [
        "id", "true_label", "pred_label", "confidence", "top2_label", "top2_prob",
        "prob_margin", "last_action", "prev_action", "action_seq_last5",
        "current_prompt", "last_result",
    ]
    print(review.head(20)[cols].to_string(index=False))

    # 7. top2 pair analysis
    df["top2_pair"] = [
        "|".join(sorted([a, b]))
        for a, b in zip(df["pred_label"], df["top2_label"])
    ]
    top2_pairs = (
        df[df["correct"] == False]
        .groupby(["top2_pair", "true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top2_pairs.to_csv(OUT_DIR / "05_wrong_top2_pair_counts.csv", index=False)

    # 8. file group confusion matrix
    fg = df[df["true_label"].isin(FILE_GROUP) | df["pred_label"].isin(FILE_GROUP)].copy()
    fg_ct = pd.crosstab(fg["true_label"], fg["pred_label"])
    fg_ct.to_csv(OUT_DIR / "06_file_group_confusion_matrix.csv")

    print("\n[SAVED]", OUT_DIR)
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(" -", p)


if __name__ == "__main__":
    main()
