#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

GROUPS = {
    "FILE": [
        "read_file",
        "grep_search",
        "list_directory",
        "glob_pattern",
    ],
    "WRITE": [
        "edit_file",
        "write_file",
        "apply_patch",
    ],
    "EXEC": [
        "run_bash",
        "run_tests",
        "lint_or_typecheck",
    ],
    "CONTROL": [
        "ask_user",
        "plan_task",
        "web_search",
        "respond_only",
    ],
}

C2I = {x: i for i, x in enumerate(CLASSES)}
GROUP_NAMES = list(GROUPS)

CLASS_TO_GROUP = {}

for gi, group in enumerate(GROUP_NAMES):
    for cls in GROUPS[group]:
        CLASS_TO_GROUP[C2I[cls]] = gi

class_to_group = np.array([
    CLASS_TO_GROUP[i]
    for i in range(len(CLASSES))
], dtype=np.int64)


REPRO_ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

PRED_CSV = (
    REPRO_ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)


pred_df = pd.read_csv(PRED_CSV)

y = pred_df["true_id"].astype(int).values

score = np.load(
    "exact_repro_base_score.npy"
).astype(np.float64)

base_pred = np.load(
    "exact_repro_base_pred.npy"
).astype(np.int64)


BASE_F1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

BASE_ACC = accuracy_score(y, base_pred)


true_group = class_to_group[y]
pred_group = class_to_group[base_pred]

base_correct = base_pred == y
group_correct = pred_group == true_group


print("=" * 110)
print("BASE ERROR DECOMPOSITION")
print("=" * 110)

print("BASE F1:", BASE_F1)
print("BASE ACC:", BASE_ACC)

print()
print("total rows:", len(y))
print("base correct:", int(base_correct.sum()))
print("base errors:", int((~base_correct).sum()))

print()
print(
    "group correct:",
    int(group_correct.sum()),
    float(group_correct.mean()),
)

print(
    "group wrong:",
    int((~group_correct).sum()),
    float((~group_correct).mean()),
)

within_error = (
    (~base_correct)
    & group_correct
)

cross_group_error = (
    ~group_correct
)

print()
print(
    "WITHIN-GROUP ERRORS:",
    int(within_error.sum()),
)

print(
    "CROSS-GROUP ERRORS:",
    int(cross_group_error.sum()),
)

print(
    "fraction of all base errors that are within-group:",
    float(
        within_error.sum()
        / (~base_correct).sum()
    ),
)

print(
    "fraction of all base errors that are cross-group:",
    float(
        cross_group_error.sum()
        / (~base_correct).sum()
    ),
)


# ============================================================
# Group-level confusion
# ============================================================

print()
print("=" * 110)
print("GROUP CONFUSION")
print("=" * 110)

cm = confusion_matrix(
    true_group,
    pred_group,
    labels=range(len(GROUP_NAMES)),
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"true_{x}"
        for x in GROUP_NAMES
    ],
    columns=[
        f"pred_{x}"
        for x in GROUP_NAMES
    ],
)

print(cm_df.to_string())


print()
print("=" * 110)
print("PER TRUE GROUP")
print("=" * 110)

rows = []

for gi, group in enumerate(GROUP_NAMES):
    mask = true_group == gi

    group_acc = accuracy_score(
        true_group[mask],
        pred_group[mask],
    )

    class_acc = accuracy_score(
        y[mask],
        base_pred[mask],
    )

    group_f1_mask = f1_score(
        y[mask],
        base_pred[mask],
        labels=[
            C2I[x]
            for x in GROUPS[group]
        ],
        average="macro",
        zero_division=0,
    )

    n = int(mask.sum())

    within_n = int(
        (
            mask
            & group_correct
            & (~base_correct)
        ).sum()
    )

    cross_n = int(
        (
            mask
            & (~group_correct)
        ).sum()
    )

    rows.append({
        "group": group,
        "n": n,
        "group_acc": group_acc,
        "class_acc": class_acc,
        "within_group_errors": within_n,
        "cross_group_errors": cross_n,
        "within_group_macro_f1": group_f1_mask,
    })


group_df = pd.DataFrame(rows)

print(group_df.to_string(index=False))


# ============================================================
# Oracle A:
# perfect within-group expert, keep base group
# ============================================================

oracle_within = base_pred.copy()

oracle_within[within_error] = y[within_error]

oracle_within_f1 = f1_score(
    y,
    oracle_within,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)


# ============================================================
# Oracle B:
# perfect group router, keep current within-group ordering
#
# For rows whose group is wrong, choose highest base-score
# class inside true group.
# ============================================================

oracle_group = base_pred.copy()

for i in np.where(~group_correct)[0]:
    gi = true_group[i]

    members = np.array([
        C2I[x]
        for x in GROUPS[GROUP_NAMES[gi]]
    ])

    oracle_group[i] = members[
        np.argmax(score[i, members])
    ]

oracle_group_f1 = f1_score(
    y,
    oracle_group,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)


# ============================================================
# Oracle C:
# perfect group + perfect within-group
# ============================================================

perfect = y.copy()

perfect_f1 = f1_score(
    y,
    perfect,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)


print()
print("=" * 110)
print("HIERARCHICAL ORACLES")
print("=" * 110)

print("BASE F1:", BASE_F1)

print()
print(
    "PERFECT WITHIN-GROUP EXPERT, BASE GROUP:",
    oracle_within_f1,
)
print(
    "GAIN:",
    oracle_within_f1 - BASE_F1,
)

print()
print(
    "PERFECT GROUP ROUTER, CURRENT WITHIN-GROUP SCORES:",
    oracle_group_f1,
)
print(
    "GAIN:",
    oracle_group_f1 - BASE_F1,
)

print()
print(
    "PERFECT BOTH:",
    perfect_f1,
)


# ============================================================
# Within-group error confusion pairs
# ============================================================

pair_rows = []

for true_cls in range(len(CLASSES)):
    for pred_cls in range(len(CLASSES)):
        if true_cls == pred_cls:
            continue

        if (
            class_to_group[true_cls]
            != class_to_group[pred_cls]
        ):
            continue

        mask = (
            (y == true_cls)
            & (base_pred == pred_cls)
        )

        n = int(mask.sum())

        if n:
            pair_rows.append({
                "group": GROUP_NAMES[
                    class_to_group[true_cls]
                ],
                "true": CLASSES[true_cls],
                "pred": CLASSES[pred_cls],
                "n": n,
            })


pair_df = pd.DataFrame(pair_rows).sort_values(
    "n",
    ascending=False,
)

print()
print("=" * 110)
print("TOP WITHIN-GROUP CONFUSIONS")
print("=" * 110)

print(
    pair_df.head(50).to_string(
        index=False,
    )
)

group_df.to_csv(
    "hierarchical_group_diagnostics.csv",
    index=False,
)

pair_df.to_csv(
    "hierarchical_within_group_confusions.csv",
    index=False,
)

print()
print("Saved:")
print(" - hierarchical_group_diagnostics.csv")
print(" - hierarchical_within_group_confusions.csv")
