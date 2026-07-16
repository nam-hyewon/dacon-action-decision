#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np


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

SRC = Path("policy_cleaning_weights_fold0")
OUT = Path("policy_cleaning_weights_class_relative_fold0")


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_ids = (
        np.load(
            SRC / "train_ids.npy",
            allow_pickle=True,
        )
        .astype(str)
    )

    purity = np.load(
        SRC / "purity.npy"
    ).astype(np.float32)

    with open(
        "data/train_labels.csv",
        encoding="utf-8",
    ) as f:
        label_map = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(f)
        }

    y = np.asarray([
        C2I[label_map[sid]]
        for sid in train_ids
    ], dtype=np.int64)

    weights = np.zeros(
        len(train_ids),
        dtype=np.float32,
    )

    percentile = np.zeros(
        len(train_ids),
        dtype=np.float32,
    )

    print("=" * 130)
    print("CLASS-RELATIVE PURITY THRESHOLDS")
    print("=" * 130)

    for class_id, action in enumerate(CLASSES):

        idx = np.where(
            y == class_id
        )[0]

        p = purity[idx]

        q10, q25, q50 = np.percentile(
            p,
            [10, 25, 50],
        )

        # Percentile rank using deterministic stable ordering.
        order = np.argsort(
            p,
            kind="mergesort",
        )

        ranks = np.empty(
            len(idx),
            dtype=np.float32,
        )

        if len(idx) == 1:
            ranks[order] = 1.0
        else:
            ranks[order] = (
                np.arange(
                    len(idx),
                    dtype=np.float32,
                )
                / (len(idx) - 1)
            )

        percentile[idx] = ranks

        class_weights = np.select(
            [
                ranks >= 0.50,
                ranks >= 0.25,
                ranks >= 0.10,
            ],
            [
                1.00,
                0.80,
                0.60,
            ],
            default=0.35,
        ).astype(np.float32)

        weights[idx] = class_weights

        print(
            f"{action:20s}"
            f" n={len(idx):5d}"
            f" purity_mean={p.mean():.6f}"
            f" q10={q10:.4f}"
            f" q25={q25:.4f}"
            f" q50={q50:.4f}"
            f" weight_mean={class_weights.mean():.6f}"
        )

    print()
    print("=" * 100)
    print("GLOBAL WEIGHT DISTRIBUTION")
    print("=" * 100)

    unique, counts = np.unique(
        weights,
        return_counts=True,
    )

    for w, n in zip(unique, counts):
        print(
            f"weight={float(w):.2f}"
            f" n={int(n)}"
            f" ratio={float(n / len(weights)):.6f}"
        )

    print(
        "weight mean:",
        float(weights.mean()),
    )

    np.save(
        OUT / "train_ids.npy",
        train_ids,
    )

    np.save(
        OUT / "purity.npy",
        purity,
    )

    np.save(
        OUT / "class_relative_percentile.npy",
        percentile,
    )

    np.save(
        OUT / "sample_weights.npy",
        weights,
    )

    print()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
