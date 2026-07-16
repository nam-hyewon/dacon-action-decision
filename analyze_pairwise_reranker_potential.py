#!/usr/bin/env python3

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

root = Path("large + tab + diet + rob8 (0.76468)")

paths = {
    "large": root / "klue_roberta_large_eda_state_fold0/val_logits.npy",
    "tab": root / "tabular_histgb_fold0/val_logits.npy",
    "diet": root / "kluebert_diet_fold0/val_logits.npy",
    "rob8": root / "roberta_eda_state8_lr15_fold0/val_logits.npy",
}

pred_csv = root / "kluebert_diet_fold0/val_predictions.csv"

weights = {
    "large": 0.24114308710234722,
    "tab": 0.4438507372276068,
    "diet": 0.21685297669057865,
    "rob8": 0.09815319897946724,
}

temps = {
    "large": 1.0,
    "tab": 1.0,
    "diet": 1.0,
    "rob8": 0.85,
}

bias = np.array([
    0.0,
    -0.3000011444091797,
    0.39999818801879883,
    0.6499979496002197,
    -0.3000011444091797,
    0.0,
    0.0,
    -0.000001430511474609375,
    -0.2500011920928955,
    -0.35000109672546387,
    -0.2500011920928955,
    -0.10000133514404297,
    0.4999980926513672,
    1.3499972820281982,
], dtype=np.float64)

df = pd.read_csv(pred_csv)
y = df["true_id"].astype(int).values

raw = {
    k: np.load(p).astype(np.float64)
    for k, p in paths.items()
}

score = sum(
    weights[k] * (raw[k] / temps[k])
    for k in weights
)
score += bias

order = np.argsort(-score, axis=1)

top1 = order[:, 0]
top2 = order[:, 1]
top3 = order[:, 2]

base_f1 = f1_score(
    y,
    top1,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

wrong = top1 != y

print("BASE macro-F1:", base_f1)
print("errors:", wrong.sum())
print("error rate:", wrong.mean())

print()
print("===== TRUE LABEL RANK AMONG BASE LOGITS =====")

for k in [1, 2, 3, 4, 5]:
    hit = np.any(order[:, :k] == y[:, None], axis=1)
    print(
        f"top{k} overall:",
        int(hit.sum()),
        f"{hit.mean():.6f}",
    )

    hit_wrong = hit[wrong]
    print(
        f"top{k} among errors:",
        int(hit_wrong.sum()),
        f"{hit_wrong.mean():.6f}",
    )

print()
print("===== ORACLE TOP2 =====")

oracle2 = top1.copy()
mask2 = wrong & (top2 == y)
oracle2[mask2] = y[mask2]

oracle2_f1 = f1_score(
    y,
    oracle2,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print("top2-correctable errors:", int(mask2.sum()))
print("oracle top2 macro-F1:", oracle2_f1)
print("oracle top2 gain:", oracle2_f1 - base_f1)

print()
print("===== TOP CONFUSION PAIRS =====")

pair_counter = Counter()

for yi, p1, p2 in zip(y, top1, top2):
    if p1 == yi:
        continue

    pair = tuple(sorted((int(p1), int(yi))))
    pair_counter[pair] += 1

for rank, (pair, n) in enumerate(pair_counter.most_common(30), 1):
    a, b = pair

    idx = (
        ((top1 == a) & (y == b))
        | ((top1 == b) & (y == a))
    )

    pair_errors = int(idx.sum())

    top2_correctable = int(
        (
            idx
            & (top2 == y)
        ).sum()
    )

    print(
        f"{rank:2d}. "
        f"{CLASSES[a]:20s} <-> {CLASSES[b]:20s} "
        f"errors={pair_errors:4d} "
        f"top2_correctable={top2_correctable:4d} "
        f"rate={top2_correctable / max(pair_errors, 1):.4f}"
    )

print()
print("===== ORDERED TOP1->TRUE CONFUSIONS =====")

ordered = Counter(
    (int(p), int(t))
    for p, t in zip(top1, y)
    if p != t
)

for rank, ((p, t), n) in enumerate(ordered.most_common(30), 1):
    idx = (top1 == p) & (y == t)
    top2_hit = int((top2[idx] == t).sum())

    print(
        f"{rank:2d}. "
        f"{CLASSES[p]:20s} -> {CLASSES[t]:20s} "
        f"n={n:4d} "
        f"true_is_top2={top2_hit:4d} "
        f"rate={top2_hit / n:.4f}"
    )
