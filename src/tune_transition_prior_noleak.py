import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


ALL_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]
LABEL2ID = {c: i for i, c in enumerate(ALL_CLASSES)}


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_key(sample, order):
    actions = get_actions(sample)
    if order == 1:
        return (actions[-1],) if len(actions) >= 1 else ("none",)
    if order == 2:
        return tuple(actions[-2:]) if len(actions) >= 2 else tuple(["none"] * (2 - len(actions)) + actions)
    if order == 3:
        return tuple(actions[-3:]) if len(actions) >= 3 else tuple(["none"] * (3 - len(actions)) + actions)
    raise ValueError(order)


def load_all_samples(data_dir):
    samples = {}
    with open(Path(data_dir) / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                samples[s["id"]] = s
    return samples


def load_labels(data_dir):
    labels = {}
    with open(Path(data_dir) / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]
    return labels


def make_prior(train_rows, order=1, smoothing=1.0):
    counts = defaultdict(Counter)
    global_counts = Counter()

    for s, label in train_rows:
        key = get_key(s, order)
        counts[key][label] += 1
        global_counts[label] += 1

    total_global = sum(global_counts.values()) + smoothing * len(ALL_CLASSES)
    global_prob = np.array([
        (global_counts[c] + smoothing) / total_global
        for c in ALL_CLASSES
    ], dtype=np.float64)

    table = {}
    for key, cnt in counts.items():
        total = sum(cnt.values()) + smoothing * len(ALL_CLASSES)
        prob = np.array([
            (cnt[c] + smoothing) / total
            for c in ALL_CLASSES
        ], dtype=np.float64)
        table[key] = np.log(prob)

    return table, np.log(global_prob)


def get_prior_matrix(samples, table, global_logp, order):
    rows = []
    for s in samples:
        key = get_key(s, order)
        rows.append(table.get(key, global_logp))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--val_logits", required=True)
    ap.add_argument("--smoothing", type=float, default=1.0)
    ap.add_argument("--out_logits", default=None)
    args = ap.parse_args()

    val_df = pd.read_csv(args.val_csv)
    logits = np.load(args.val_logits).astype(np.float64)

    y = val_df["true_id"].astype(int).values
    val_ids = set(val_df["id"].tolist())

    samples_by_id = load_all_samples(args.data_dir)
    labels_by_id = load_labels(args.data_dir)

    # 핵심: validation id 제외
    train_rows = [
        (samples_by_id[sid], label)
        for sid, label in labels_by_id.items()
        if sid not in val_ids and sid in samples_by_id
    ]

    val_samples = [samples_by_id[sid] for sid in val_df["id"].tolist()]

    print(f"transition train rows: {len(train_rows)}")
    print(f"validation rows excluded: {len(val_ids)}")

    base_pred = logits.argmax(axis=1)
    base_f1 = f1_score(y, base_pred, labels=list(range(14)), average="macro", zero_division=0)
    base_acc = accuracy_score(y, base_pred)
    print(f"Base Macro-F1: {base_f1:.6f} | Acc: {base_acc:.6f}")

    prior_mats = {}
    for order in [1, 2, 3]:
        table, global_logp = make_prior(train_rows, order=order, smoothing=args.smoothing)
        prior_mats[order] = get_prior_matrix(val_samples, table, global_logp, order)
        print(f"order={order} table size={len(table)}")

    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
    cands = []

    for a1 in grid:
        for a2 in grid:
            for a3 in grid:
                adjusted = logits + a1 * prior_mats[1] + a2 * prior_mats[2] + a3 * prior_mats[3]
                pred = adjusted.argmax(axis=1)
                f1 = f1_score(y, pred, labels=list(range(14)), average="macro", zero_division=0)
                acc = accuracy_score(y, pred)
                changed = int((pred != base_pred).sum())
                cands.append((f1, acc, changed, a1, a2, a3))

    cands = sorted(cands, reverse=True)

    print("\n===== TOP transition prior NO-LEAK =====")
    for f1, acc, changed, a1, a2, a3 in cands[:30]:
        print(
            f"macro_f1={f1:.6f} acc={acc:.6f} changed={changed} | "
            f"a_last={a1:.2f} a_bigram={a2:.2f} a_trigram={a3:.2f}"
        )

    best = cands[0]
    best_f1, best_acc, best_changed, a1, a2, a3 = best
    adjusted = logits + a1 * prior_mats[1] + a2 * prior_mats[2] + a3 * prior_mats[3]

    if args.out_logits:
        np.save(args.out_logits, adjusted)
        print(f"\nSaved adjusted logits: {args.out_logits}")

    print("\nBEST")
    print(
        f"macro_f1={best_f1:.6f} acc={best_acc:.6f} changed={best_changed} | "
        f"a_last={a1:.2f} a_bigram={a2:.2f} a_trigram={a3:.2f}"
    )


if __name__ == "__main__":
    main()
