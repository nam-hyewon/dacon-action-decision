#!/usr/bin/env python3

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_DIR = Path(
    "file_strict_raw_state_pairs"
)

PAIR_PATH = (
    PAIR_DIR / "all_raw_state_pairs.csv"
)

OUT_DIR = Path(
    "full_state_collision_audit"
)


def load_data():
    samples = {}

    with open(
        "data/train.jsonl",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            samples[str(obj["id"])] = obj

    with open(
        "data/train_labels.csv",
        encoding="utf-8",
    ) as f:
        labels = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(f)
        }

    return samples, labels


def normalize_scalar(x):
    if x is None:
        return "<NONE>"

    if isinstance(
        x,
        (
            bool,
            int,
            float,
        ),
    ):
        return x

    return str(x)


def recursive_diff(
    a,
    b,
    path="$",
    out=None,
):
    if out is None:
        out = []

    if type(a) is not type(b):
        out.append({
            "path": path,
            "kind": "type_diff",
            "a": type(a).__name__,
            "b": type(b).__name__,
        })

        return out

    if isinstance(a, dict):
        keys = sorted(
            set(a.keys())
            | set(b.keys())
        )

        for key in keys:
            child = f"{path}.{key}"

            if key not in a:
                out.append({
                    "path": child,
                    "kind": "missing_a",
                    "a": "<MISSING>",
                    "b": normalize_scalar(
                        b[key]
                    ),
                })

            elif key not in b:
                out.append({
                    "path": child,
                    "kind": "missing_b",
                    "a": normalize_scalar(
                        a[key]
                    ),
                    "b": "<MISSING>",
                })

            else:
                recursive_diff(
                    a[key],
                    b[key],
                    child,
                    out,
                )

        return out

    if isinstance(a, list):
        if len(a) != len(b):
            out.append({
                "path": f"{path}.__len__",
                "kind": "list_len",
                "a": len(a),
                "b": len(b),
            })

        n = min(
            len(a),
            len(b),
        )

        for i in range(n):
            recursive_diff(
                a[i],
                b[i],
                f"{path}[{i}]",
                out,
            )

        return out

    if a != b:
        out.append({
            "path": path,
            "kind": "value_diff",
            "a": normalize_scalar(a),
            "b": normalize_scalar(b),
        })

    return out


def canonicalize_path(path):
    # Make history indices comparable across samples.
    import re

    path = re.sub(
        r"\.history\[\d+\]",
        ".history[*]",
        path,
    )

    return path


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples, labels = load_data()

    pairs = pd.read_csv(
        PAIR_PATH
    )

    selected = pairs[
        pairs["prompt_jaccard"] >= 0.90
    ].copy()

    print(
        "selected jaccard090 pairs:",
        len(selected),
    )

    exact = selected[
        selected["same_prompt_norm"] == 1
    ].copy()

    print(
        "exact prompt pairs:",
        len(exact),
    )

    path_counter = Counter()
    kind_counter = Counter()
    pair_path_counter = defaultdict(Counter)

    rows = []

    for _, row in selected.iterrows():
        ida = str(row["id_a"])
        idb = str(row["id_b"])

        a = samples[ida]
        b = samples[idb]

        diffs = recursive_diff(
            a,
            b,
        )

        pair = str(row["pair"])

        for diff in diffs:
            path = canonicalize_path(
                diff["path"]
            )

            path_counter[path] += 1

            kind_counter[
                (
                    path,
                    diff["kind"],
                )
            ] += 1

            pair_path_counter[
                pair
            ][path] += 1

            rows.append({
                "pair": pair,
                "id_a": ida,
                "id_b": idb,
                "label_a": labels[ida],
                "label_b": labels[idb],
                "prompt_jaccard": float(
                    row["prompt_jaccard"]
                ),
                "same_prompt_norm": int(
                    row["same_prompt_norm"]
                ),
                "path": path,
                "kind": diff["kind"],
                "a": str(diff["a"])[:1000],
                "b": str(diff["b"])[:1000],
            })

    diff_df = pd.DataFrame(rows)

    diff_df.to_csv(
        OUT_DIR / "all_recursive_diffs.csv",
        index=False,
    )

    print()
    print("=" * 160)
    print("GLOBAL TOP DIFFERING RAW JSON PATHS")
    print("=" * 160)

    global_rows = []

    n_selected = len(selected)

    for path, count in (
        path_counter.most_common()
    ):
        global_rows.append({
            "path": path,
            "count": count,
            "pair_n": n_selected,
            "rate": count / max(
                1,
                n_selected,
            ),
        })

    global_df = pd.DataFrame(
        global_rows
    )

    print(
        global_df.head(100).to_string(
            index=False,
        )
    )

    global_df.to_csv(
        OUT_DIR
        / "global_diff_path_summary.csv",
        index=False,
    )

    print()
    print("=" * 160)
    print("PER LABEL-PAIR RAW PATH DIFFERENCES")
    print("=" * 160)

    pair_rows = []

    for pair in sorted(
        selected["pair"].unique()
    ):
        pair_n = int(
            (selected["pair"] == pair).sum()
        )

        print()
        print("-" * 140)
        print(
            pair,
            "| n=",
            pair_n,
        )
        print("-" * 140)

        cur = []

        for path, count in (
            pair_path_counter[pair]
            .most_common()
        ):
            item = {
                "pair": pair,
                "path": path,
                "count": count,
                "pair_n": pair_n,
                "rate": (
                    count
                    / max(1, pair_n)
                ),
            }

            pair_rows.append(item)
            cur.append(item)

        cur_df = pd.DataFrame(cur)

        print(
            cur_df.head(60).to_string(
                index=False,
            )
        )

    pair_df = pd.DataFrame(
        pair_rows
    )

    pair_df.to_csv(
        OUT_DIR
        / "pair_diff_path_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Fully identical raw JSON except id.
    # ---------------------------------------------------------

    exact_collision_rows = []

    for _, row in selected.iterrows():
        ida = str(row["id_a"])
        idb = str(row["id_b"])

        a = dict(samples[ida])
        b = dict(samples[idb])

        a.pop("id", None)
        b.pop("id", None)

        fully_identical = (
            a == b
        )

        if fully_identical:
            exact_collision_rows.append({
                "pair": row["pair"],
                "id_a": ida,
                "id_b": idb,
                "label_a": labels[ida],
                "label_b": labels[idb],
            })

    exact_collision_df = pd.DataFrame(
        exact_collision_rows
    )

    exact_collision_df.to_csv(
        OUT_DIR
        / "fully_identical_cross_label_states.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("FULLY IDENTICAL RAW STATE COLLISIONS")
    print("=" * 100)

    print(
        "n:",
        len(exact_collision_df),
    )

    if len(exact_collision_df):
        print(
            exact_collision_df[
                "pair"
            ].value_counts()
        )

        print()
        print(
            exact_collision_df.head(
                100
            ).to_string(
                index=False,
            )
        )

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print(
        "jaccard090 pairs:",
        len(selected),
    )

    print(
        "exact prompt pairs:",
        len(exact),
    )

    print(
        "fully identical raw-state "
        "cross-label pairs:",
        len(exact_collision_df),
    )

    print(
        "saved:",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
