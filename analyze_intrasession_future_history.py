#!/usr/bin/env python3

import json
import re
from collections import defaultdict
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


def parse_id(sample_id):
    sample_id = str(sample_id)

    m = re.match(
        r"^(.*)-step_(\d+)$",
        sample_id,
    )

    if m is None:
        raise ValueError(
            f"Could not parse id: {sample_id}"
        )

    return m.group(1), int(m.group(2))


def assistant_actions(sample):
    actions = []

    for h in sample.get("history", []):
        if h.get("role") != "user":
            name = str(h.get("name", ""))

            if name in C2I:
                actions.append(name)

    return actions


print("Loading train rows...")

samples = []

with open(
    "data/train.jsonl",
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        line = line.strip()

        if line:
            samples.append(json.loads(line))


labels_df = pd.read_csv(
    "data/train_labels.csv"
)

label_map = dict(zip(
    labels_df["id"].astype(str),
    labels_df["action"].astype(str),
))


sample_by_id = {
    str(sample["id"]): sample
    for sample in samples
}


pred_df = pd.read_csv(PRED_CSV)

val_ids = pred_df["id"].astype(str).values
y = pred_df["true_id"].astype(int).values


base_pred = np.load(
    "exact_repro_base_pred.npy"
).astype(np.int64)

assert len(base_pred) == len(val_ids)
assert len(y) == len(val_ids)


# ============================================================
# Group validation rows by session
# ============================================================

session_rows = defaultdict(list)

for pos, sample_id in enumerate(val_ids):
    session_id, step = parse_id(sample_id)

    session_rows[session_id].append(
        (step, pos, sample_id)
    )

for session_id in session_rows:
    session_rows[session_id].sort()


sizes = np.array([
    len(rows)
    for rows in session_rows.values()
])

print()
print("=" * 100)
print("VALIDATION SESSION STRUCTURE")
print("=" * 100)

print("rows:", len(val_ids))
print("sessions:", len(session_rows))
print(
    pd.Series(sizes).describe().to_string()
)

print(
    "multi-row sessions:",
    int((sizes >= 2).sum()),
)

print(
    "rows in multi-row sessions:",
    int(
        sum(
            len(rows)
            for rows in session_rows.values()
            if len(rows) >= 2
        )
    ),
)


# ============================================================
# Check whether row t label is revealed by row t+1 history
# ============================================================

pairs = []

for session_id, rows in session_rows.items():

    for j in range(len(rows) - 1):
        step_t, pos_t, id_t = rows[j]
        step_next, pos_next, id_next = rows[j + 1]

        sample_t = sample_by_id[id_t]
        sample_next = sample_by_id[id_next]

        actions_t = assistant_actions(sample_t)
        actions_next = assistant_actions(sample_next)

        true_action = CLASSES[y[pos_t]]

        next_last_action = (
            actions_next[-1]
            if actions_next
            else None
        )

        # How many actions were appended from t to next row?
        prefix_match = (
            len(actions_next) >= len(actions_t)
            and actions_next[:len(actions_t)] == actions_t
        )

        appended = (
            actions_next[len(actions_t):]
            if prefix_match
            else []
        )

        first_appended = (
            appended[0]
            if appended
            else None
        )

        pairs.append({
            "session_id": session_id,
            "step_t": step_t,
            "step_next": step_next,
            "gap": step_next - step_t,
            "pos_t": pos_t,
            "pos_next": pos_next,
            "id_t": id_t,
            "id_next": id_next,
            "true_action": true_action,
            "base_action": CLASSES[base_pred[pos_t]],
            "history_len_t": len(actions_t),
            "history_len_next": len(actions_next),
            "prefix_match": prefix_match,
            "append_n": len(appended),
            "first_appended": first_appended,
            "next_last_action": next_last_action,
            "first_appended_matches_true": (
                first_appended == true_action
            ),
            "next_last_matches_true": (
                next_last_action == true_action
            ),
        })


pair_df = pd.DataFrame(pairs)

print()
print("=" * 100)
print("CONSECUTIVE VALIDATION ROW ANALYSIS")
print("=" * 100)

print("adjacent row pairs:", len(pair_df))

print()
print("step gap distribution:")
print(
    pair_df["gap"]
    .value_counts()
    .sort_index()
    .head(20)
    .to_string()
)

print()
print(
    "prefix-match rate:",
    float(pair_df["prefix_match"].mean()),
)

print(
    "exactly one appended action:",
    int((pair_df["append_n"] == 1).sum()),
    "/",
    len(pair_df),
)

print(
    "first appended matches current true action:",
    int(
        pair_df[
            "first_appended_matches_true"
        ].sum()
    ),
    "/",
    len(pair_df),
    "=",
    float(
        pair_df[
            "first_appended_matches_true"
        ].mean()
    ),
)

print(
    "next last action matches current true action:",
    int(
        pair_df[
            "next_last_matches_true"
        ].sum()
    ),
    "/",
    len(pair_df),
    "=",
    float(
        pair_df[
            "next_last_matches_true"
        ].mean()
    ),
)


# ============================================================
# Strict revelation rule
#
# Only use when histories are exact prefix and exactly one
# assistant action was appended.
# ============================================================

strict = (
    pair_df["prefix_match"].values
    & (pair_df["append_n"].values == 1)
)

strict_df = pair_df[strict].copy()

print()
print("=" * 100)
print("STRICT REVELATION")
print("=" * 100)

print("strict rows:", len(strict_df))

if len(strict_df):
    strict_acc = float(
        strict_df[
            "first_appended_matches_true"
        ].mean()
    )

    print(
        "revealed-action accuracy:",
        strict_acc,
    )

    strict_base_acc = float(
        (
            strict_df["base_action"]
            == strict_df["true_action"]
        ).mean()
    )

    print(
        "base accuracy on strict rows:",
        strict_base_acc,
    )


# ============================================================
# Override exact reproduced SOTA
# ============================================================

revealed_pred = base_pred.copy()
use = np.zeros(len(y), dtype=bool)

for row in strict_df.itertuples(index=False):
    action = row.first_appended

    if action not in C2I:
        continue

    pos = int(row.pos_t)

    revealed_pred[pos] = C2I[action]
    use[pos] = True


BASE_F1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

NEW_F1 = f1_score(
    y,
    revealed_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

BASE_ACC = accuracy_score(
    y,
    base_pred,
)

NEW_ACC = accuracy_score(
    y,
    revealed_pred,
)


print()
print("=" * 100)
print("STRICT REVELATION OVERRIDE")
print("=" * 100)

print("BASE F1:", BASE_F1)
print("NEW F1:", NEW_F1)
print("GAIN:", NEW_F1 - BASE_F1)

print()

print("BASE ACC:", BASE_ACC)
print("NEW ACC:", NEW_ACC)
print("ACC GAIN:", NEW_ACC - BASE_ACC)

print()

print("use_n:", int(use.sum()))

print(
    "changed_n:",
    int(
        (
            revealed_pred[use]
            != base_pred[use]
        ).sum()
    ),
)

print(
    "new acc on used rows:",
    accuracy_score(
        y[use],
        revealed_pred[use],
    )
    if use.sum()
    else np.nan,
)

print(
    "base acc on used rows:",
    accuracy_score(
        y[use],
        base_pred[use],
    )
    if use.sum()
    else np.nan,
)


# ============================================================
# Per-class diagnostics
# ============================================================

rows = []

for class_id, action in enumerate(CLASSES):
    mask = use & (y == class_id)

    if not mask.any():
        continue

    rows.append({
        "class": action,
        "n": int(mask.sum()),
        "base_acc": float(
            accuracy_score(
                y[mask],
                base_pred[mask],
            )
        ),
        "revealed_acc": float(
            accuracy_score(
                y[mask],
                revealed_pred[mask],
            )
        ),
    })

class_df = pd.DataFrame(rows)

print()
print("=" * 100)
print("PER-CLASS USED ROWS")
print("=" * 100)

print(
    class_df.to_string(index=False)
)


pair_df.to_csv(
    "intrasession_future_history_pairs.csv",
    index=False,
)

class_df.to_csv(
    "intrasession_future_history_per_class.csv",
    index=False,
)

np.save(
    "intrasession_revealed_pred.npy",
    revealed_pred,
)

print()
print("Saved:")
print(" - intrasession_future_history_pairs.csv")
print(" - intrasession_future_history_per_class.csv")
print(" - intrasession_revealed_pred.npy")
