#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
C2I = {x: i for i, x in enumerate(CLASSES)}

REPRO_ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

PRED_CSV = (
    REPRO_ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def get_actions(sample):
    out = []

    for h in sample.get("history", []):
        if h.get("role") != "user":
            name = str(h.get("name", ""))

            if name in C2I:
                out.append(C2I[name])

    return out


def add_count(table, key, y):
    table[key][y] += 1


def make_logprob(counter, alpha=1.0):
    arr = np.full(
        len(CLASSES),
        alpha,
        dtype=np.float64,
    )

    for y, n in counter.items():
        arr[int(y)] += n

    arr /= arr.sum()

    return np.log(arr)


print("Loading...")

samples = load_jsonl("data/train.jsonl")

label_df = pd.read_csv(
    "data/train_labels.csv"
)

label_map = dict(zip(
    label_df["id"].astype(str),
    label_df["action"].astype(str),
))

ids = np.array([
    str(x["id"])
    for x in samples
])

labels = np.array([
    C2I[label_map[sid]]
    for sid in ids
], dtype=np.int64)


pred_df = pd.read_csv(PRED_CSV)

val_ids = pred_df["id"].astype(str).values
y_val = pred_df["true_id"].astype(int).values

val_set = set(val_ids)

id_to_idx = {
    sid: i
    for i, sid in enumerate(ids)
}

val_idx = np.array([
    id_to_idx[sid]
    for sid in val_ids
])

ref_idx = np.array([
    i
    for i, sid in enumerate(ids)
    if sid not in val_set
])


base_score = np.load(
    "exact_repro_base_score.npy"
).astype(np.float64)

base_pred = base_score.argmax(axis=1)

BASE_F1 = f1_score(
    y_val,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print("BASE F1:", BASE_F1)
print("reference:", len(ref_idx))
print("validation:", len(val_idx))


# ============================================================
# Learn trajectory-state conditional tables on non-fold0 only
# ============================================================

tables = {
    1: defaultdict(Counter),
    2: defaultdict(Counter),
    3: defaultdict(Counter),
    4: defaultdict(Counter),
    5: defaultdict(Counter),
}

global_count = Counter()


for i in ref_idx:
    actions = get_actions(samples[i])
    target = int(labels[i])

    global_count[target] += 1

    for k in range(1, 6):
        if len(actions) >= k:
            key = tuple(actions[-k:])
            add_count(
                tables[k],
                key,
                target,
            )


print()
print("unique states:")

for k in range(1, 6):
    print(
        "k=",
        k,
        "states=",
        len(tables[k]),
    )


# ============================================================
# Build backoff trajectory scores
# ============================================================

global_lp = make_logprob(
    global_count,
    alpha=1.0,
)

traj_score = np.zeros_like(
    base_score,
    dtype=np.float64,
)

support_used = np.zeros(
    len(y_val),
    dtype=np.int64,
)

k_used = np.zeros(
    len(y_val),
    dtype=np.int64,
)


# Try longest state first.
for pos, i in enumerate(val_idx):
    actions = get_actions(samples[i])

    found = False

    for k in range(5, 0, -1):
        if len(actions) < k:
            continue

        key = tuple(actions[-k:])

        if key not in tables[k]:
            continue

        cnt = tables[k][key]
        support = sum(cnt.values())

        traj_score[pos] = make_logprob(
            cnt,
            alpha=1.0,
        )

        support_used[pos] = support
        k_used[pos] = k

        found = True
        break

    if not found:
        traj_score[pos] = global_lp


print()
print("state coverage:")
print(
    pd.Series(k_used)
    .value_counts()
    .sort_index()
    .to_string()
)

print()

print("support:")
print(
    pd.Series(support_used)
    .describe(
        percentiles=[
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.95,
            0.99,
        ]
    )
    .to_string()
)


# ============================================================
# Standalone trajectory prior
# ============================================================

traj_pred = traj_score.argmax(axis=1)

print()
print("TRAJECTORY PRIOR STANDALONE")

print(
    "F1:",
    f1_score(
        y_val,
        traj_pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    ),
)

print(
    "ACC:",
    accuracy_score(
        y_val,
        traj_pred,
    ),
)


# ============================================================
# Blend sweep
# ============================================================

rows = []

for min_support in [
    0,
    2,
    3,
    5,
    10,
    20,
    30,
    50,
    100,
    200,
]:
    eligible = (
        support_used >= min_support
    )

    for lam in [
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
    ]:
        score = base_score.copy()

        score[eligible] += (
            lam * traj_score[eligible]
        )

        pred = score.argmax(axis=1)

        f1 = f1_score(
            y_val,
            pred,
            labels=range(len(CLASSES)),
            average="macro",
            zero_division=0,
        )

        rows.append({
            "min_support": min_support,
            "lambda": lam,
            "macro_f1": f1,
            "gain": f1 - BASE_F1,
            "accuracy": accuracy_score(
                y_val,
                pred,
            ),
            "eligible_n": int(
                eligible.sum()
            ),
            "changed_n": int(
                (pred != base_pred).sum()
            ),
        })


result = pd.DataFrame(rows).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

print()
print("=" * 110)
print("TRAJECTORY CONDITIONAL BLEND TOP 50")
print("=" * 110)

print(
    result.head(50).to_string(
        index=False,
    )
)

result.to_csv(
    "trajectory_conditional_blend_results.csv",
    index=False,
)


# ============================================================
# Oracle diagnostic:
# does the trajectory prior contain complementary information?
# ============================================================

base_correct = (
    base_pred == y_val
)

traj_correct = (
    traj_pred == y_val
)

recover = (
    ~base_correct
    & traj_correct
)

oracle = base_pred.copy()
oracle[recover] = traj_pred[recover]

oracle_f1 = f1_score(
    y_val,
    oracle,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print()
print("=" * 110)
print("TRAJECTORY ORACLE DIAGNOSTIC")
print("=" * 110)

print(
    "base wrong / trajectory correct:",
    int(recover.sum()),
)

print(
    "trajectory recovery among base errors:",
    float(
        traj_correct[~base_correct].mean()
    ),
)

print(
    "oracle F1:",
    oracle_f1,
)

print(
    "oracle gain:",
    oracle_f1 - BASE_F1,
)

print()
print("Saved trajectory_conditional_blend_results.csv")
