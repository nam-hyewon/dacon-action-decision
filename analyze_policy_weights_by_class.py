#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = [
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

C2I = {x: i for i, x in enumerate(CLASSES)}

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

CLASS_TO_GROUP = {
    action: group
    for group, actions in GROUPS.items()
    for action in actions
}

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

WEIGHT_DIR = Path(
    "policy_cleaning_weights_fold0"
)


def main():
    with open(
        "data/train.jsonl",
        encoding="utf-8",
    ) as f:
        samples = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    with open(
        "data/train_labels.csv",
        encoding="utf-8",
    ) as f:
        label_map = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(f)
        }

    val_ids = set(
        pd.read_csv(VAL_REF)["id"]
        .astype(str)
        .tolist()
    )

    train_ids_expected = [
        str(s["id"])
        for s in samples
        if str(s["id"]) not in val_ids
    ]

    saved_ids = (
        np.load(
            WEIGHT_DIR / "train_ids.npy",
            allow_pickle=True,
        )
        .astype(str)
        .tolist()
    )

    if train_ids_expected != saved_ids:
        raise ValueError(
            "train ID order mismatch"
        )

    purity = np.load(
        WEIGHT_DIR / "purity.npy"
    )

    weights = np.load(
        WEIGHT_DIR / "sample_weights.npy"
    )

    rows = []

    for sid, p, w in zip(
        saved_ids,
        purity,
        weights,
    ):
        action = label_map[sid]

        rows.append({
            "id": sid,
            "action": action,
            "group": CLASS_TO_GROUP[action],
            "purity": float(p),
            "weight": float(w),
        })

    df = pd.DataFrame(rows)

    print("=" * 130)
    print("BY GROUP")
    print("=" * 130)

    group_rows = []

    for group, g in df.groupby("group"):
        group_rows.append({
            "group": group,
            "n": len(g),
            "purity_mean": g["purity"].mean(),
            "p_lt_060": (g["purity"] < 0.60).mean(),
            "p_lt_075": (g["purity"] < 0.75).mean(),
            "p_ge_090": (g["purity"] >= 0.90).mean(),
            "weight_mean": g["weight"].mean(),
            "effective_n": g["weight"].sum(),
            "effective_ratio": (
                g["weight"].sum() / len(g)
            ),
        })

    group_df = pd.DataFrame(
        group_rows
    ).sort_values("group")

    print(
        group_df.to_string(index=False)
    )

    print()
    print("=" * 160)
    print("BY CLASS")
    print("=" * 160)

    class_rows = []

    for action, g in df.groupby("action"):
        class_rows.append({
            "action": action,
            "group": CLASS_TO_GROUP[action],
            "n": len(g),
            "purity_mean": g["purity"].mean(),
            "p_lt_060": (g["purity"] < 0.60).mean(),
            "p_lt_075": (g["purity"] < 0.75).mean(),
            "p_ge_090": (g["purity"] >= 0.90).mean(),
            "weight_mean": g["weight"].mean(),
            "effective_n": g["weight"].sum(),
            "effective_ratio": (
                g["weight"].sum() / len(g)
            ),
        })

    class_df = pd.DataFrame(
        class_rows
    ).sort_values(
        ["group", "action"]
    )

    print(
        class_df.to_string(index=False)
    )

    group_df.to_csv(
        WEIGHT_DIR / "weight_stats_by_group.csv",
        index=False,
    )

    class_df.to_csv(
        WEIGHT_DIR / "weight_stats_by_class.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
