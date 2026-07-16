#!/usr/bin/env python3

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score


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
I2C = {i: x for i, x in enumerate(CLASSES)}


REPRO_ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

VAL_REF = (
    REPRO_ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)


# -------------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------------

PATH_RE = re.compile(
    r"""
    (?:
        (?:[A-Za-z]:[\\/]) |
        (?:/[\w.\-]+)+ |
        (?:\.{0,2}/[\w.\-/]+) |
        (?:[\w.\-]+/){1,}[\w.\-]+
    )
    """,
    re.VERBOSE,
)

URL_RE = re.compile(
    r"https?://\S+",
    re.IGNORECASE,
)

HASH_RE = re.compile(
    r"\b[0-9a-f]{7,40}\b",
    re.IGNORECASE,
)

UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\b"
)

QUOTED_RE = re.compile(
    r"""
    (?:
        "(?:\\.|[^"\\]){1,100}"
        |
        '(?:\\.|[^'\\]){1,100}'
        |
        `[^`]{1,100}`
    )
    """,
    re.VERBOSE,
)

SPACE_RE = re.compile(r"\s+")


# Common coding identifiers which are likely instance-specific.
CAMEL_RE = re.compile(
    r"\b[A-Za-z]+(?:[A-Z][A-Za-z0-9]*)+\b"
)

SNAKE_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"
)


def normalize_text(
    text,
    *,
    aggressive=False,
    max_chars=500,
):
    text = str(text or "")

    text = text[:max_chars]

    text = URL_RE.sub(" <URL> ", text)
    text = UUID_RE.sub(" <UUID> ", text)
    text = HASH_RE.sub(" <HASH> ", text)
    text = PATH_RE.sub(" <PATH> ", text)
    text = QUOTED_RE.sub(" <STR> ", text)
    text = NUMBER_RE.sub(" <NUM> ", text)

    if aggressive:
        text = CAMEL_RE.sub(" <IDENT> ", text)
        text = SNAKE_RE.sub(" <IDENT> ", text)

    text = text.lower()

    text = SPACE_RE.sub(
        " ",
        text,
    ).strip()

    return text


def action_history(sample):
    return [
        h
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def user_history(sample):
    return [
        str(h.get("content", ""))
        for h in sample.get("history", [])
        if h.get("role") == "user"
    ]


def get_action(
    actions,
    k,
):
    if len(actions) < k:
        return "<NONE>"

    return str(
        actions[-k].get("name", "")
    )


def get_result(
    actions,
    k,
    aggressive=False,
):
    if len(actions) < k:
        return "<NONE>"

    return normalize_text(
        actions[-k].get(
            "result_summary",
            "",
        ),
        aggressive=aggressive,
        max_chars=300,
    )


def get_args(
    actions,
    k,
    aggressive=False,
):
    if len(actions) < k:
        return "<NONE>"

    return normalize_text(
        json.dumps(
            actions[-k].get("args", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        aggressive=aggressive,
        max_chars=200,
    )


def prompt_skeleton(
    sample,
    aggressive=False,
):
    return normalize_text(
        sample.get(
            "current_prompt",
            "",
        ),
        aggressive=aggressive,
        max_chars=500,
    )


def last_user_skeleton(
    sample,
    aggressive=False,
):
    users = user_history(sample)

    if not users:
        return "<NONE>"

    return normalize_text(
        users[-1],
        aggressive=aggressive,
        max_chars=300,
    )


# -------------------------------------------------------------------------
# Signature families
#
# Deliberately test multiple backoff levels.
# -------------------------------------------------------------------------

def make_signature(
    sample,
    sig_name,
):
    actions = action_history(sample)

    if sig_name == "prompt_basic":
        return (
            prompt_skeleton(sample, False),
        )

    if sig_name == "prompt_aggr":
        return (
            prompt_skeleton(sample, True),
        )

    if sig_name == "prompt_lastaction":
        return (
            prompt_skeleton(sample, True),
            get_action(actions, 1),
        )

    if sig_name == "prompt_last2actions":
        return (
            prompt_skeleton(sample, True),
            get_action(actions, 1),
            get_action(actions, 2),
        )

    if sig_name == "last_result":
        return (
            get_action(actions, 1),
            get_result(actions, 1, True),
        )

    if sig_name == "last_transition":
        return (
            get_action(actions, 2),
            get_action(actions, 1),
            get_result(actions, 1, True),
        )

    if sig_name == "prompt_transition":
        return (
            prompt_skeleton(sample, True),
            get_action(actions, 1),
            get_result(actions, 1, True),
        )

    if sig_name == "prompt_transition_args":
        return (
            prompt_skeleton(sample, True),
            get_action(actions, 1),
            get_args(actions, 1, True),
            get_result(actions, 1, True),
        )

    if sig_name == "workflow2":
        return (
            get_action(actions, 2),
            get_result(actions, 2, True),
            get_action(actions, 1),
            get_result(actions, 1, True),
        )

    if sig_name == "prompt_workflow2":
        return (
            prompt_skeleton(sample, True),
            get_action(actions, 2),
            get_result(actions, 2, True),
            get_action(actions, 1),
            get_result(actions, 1, True),
        )

    if sig_name == "user_transition":
        return (
            last_user_skeleton(sample, True),
            get_action(actions, 1),
            get_result(actions, 1, True),
        )

    raise ValueError(
        f"unknown signature: {sig_name}"
    )


SIGNATURES = [
    "prompt_basic",
    "prompt_aggr",
    "prompt_lastaction",
    "prompt_last2actions",
    "last_result",
    "last_transition",
    "prompt_transition",
    "prompt_transition_args",
    "workflow2",
    "prompt_workflow2",
    "user_transition",
]


def macro_f1(
    y,
    pred,
):
    return f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )


def load_data():
    samples = []

    with open(
        "data/train.jsonl",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                samples.append(
                    json.loads(line)
                )

    with open(
        "data/train_labels.csv",
        encoding="utf-8",
    ) as f:
        label_map = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(f)
        }

    return samples, label_map


def build_table(
    samples,
    labels,
    indices,
    sig_name,
):
    table = defaultdict(Counter)

    for i in indices:
        sample = samples[i]

        sig = make_signature(
            sample,
            sig_name,
        )

        y = C2I[
            labels[str(sample["id"])]
        ]

        table[sig][y] += 1

    return table


def table_predict(
    sample,
    table,
    sig_name,
):
    sig = make_signature(
        sample,
        sig_name,
    )

    counts = table.get(sig)

    if not counts:
        return None

    total = sum(counts.values())

    pred, count = counts.most_common(1)[0]

    purity = count / total

    return {
        "pred": int(pred),
        "support": int(total),
        "purity": float(purity),
        "counts": counts,
    }


def main():

    print("Loading data...")

    samples, labels = load_data()

    val_df = pd.read_csv(
        VAL_REF
    )

    val_ids = (
        val_df["id"]
        .astype(str)
        .tolist()
    )

    y_val = (
        val_df["true_id"]
        .astype(int)
        .values
    )

    base_pred = np.load(
        "exact_repro_base_pred.npy"
    ).astype(np.int64)


    if len(y_val) != len(base_pred):
        raise ValueError(
            f"length mismatch: {len(y_val)} vs {len(base_pred)}"
        )


    id_to_idx = {
        str(sample["id"]): i
        for i, sample in enumerate(samples)
    }

    val_indices = np.array([
        id_to_idx[sid]
        for sid in val_ids
    ], dtype=np.int64)

    val_set = set(val_ids)

    train_indices = np.array([
        i
        for i, sample in enumerate(samples)
        if str(sample["id"]) not in val_set
    ], dtype=np.int64)


    val_samples = [
        samples[i]
        for i in val_indices
    ]


    base_f1 = macro_f1(
        y_val,
        base_pred,
    )

    print("train:", len(train_indices))
    print("val:", len(val_indices))
    print("BASE F1:", base_f1)


    all_rows = []

    saved_preds = {}


    for sig_name in SIGNATURES:

        print()
        print("=" * 120)
        print("SIGNATURE:", sig_name)
        print("=" * 120)

        table = build_table(
            samples=samples,
            labels=labels,
            indices=train_indices,
            sig_name=sig_name,
        )

        print(
            "unique train signatures:",
            len(table),
        )


        retrieved = []

        for sample in val_samples:
            retrieved.append(
                table_predict(
                    sample,
                    table,
                    sig_name,
                )
            )


        coverage = np.array([
            x is not None
            for x in retrieved
        ])

        print(
            "raw coverage:",
            int(coverage.sum()),
            "/",
            len(coverage),
            "=",
            float(coverage.mean()),
        )


        if coverage.any():

            raw_pred = np.array([
                (
                    x["pred"]
                    if x is not None
                    else -1
                )
                for x in retrieved
            ])

            print(
                "retrieved standalone acc:",
                accuracy_score(
                    y_val[coverage],
                    raw_pred[coverage],
                ),
            )


        for min_support in [
            1,
            2,
            3,
            5,
            10,
            20,
            50,
        ]:

            for min_purity in [
                0.50,
                0.60,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
            ]:

                use = np.array([
                    (
                        x is not None
                        and x["support"] >= min_support
                        and x["purity"] >= min_purity
                    )
                    for x in retrieved
                ])


                pred = base_pred.copy()

                for row_idx in np.where(use)[0]:
                    pred[row_idx] = retrieved[
                        row_idx
                    ]["pred"]


                f1 = macro_f1(
                    y_val,
                    pred,
                )


                changed = (
                    use
                    & (pred != base_pred)
                )

                beneficial = (
                    changed
                    & (base_pred != y_val)
                    & (pred == y_val)
                )

                harmful = (
                    changed
                    & (base_pred == y_val)
                    & (pred != y_val)
                )


                all_rows.append({
                    "signature": sig_name,
                    "min_support": min_support,
                    "min_purity": min_purity,
                    "macro_f1": f1,
                    "gain": f1 - base_f1,
                    "accuracy": accuracy_score(
                        y_val,
                        pred,
                    ),
                    "use_n": int(use.sum()),
                    "changed_n": int(
                        changed.sum()
                    ),
                    "beneficial": int(
                        beneficial.sum()
                    ),
                    "harmful": int(
                        harmful.sum()
                    ),
                    "net_correct_gain": int(
                        beneficial.sum()
                        - harmful.sum()
                    ),
                })


        sig_df = pd.DataFrame([
            x
            for x in all_rows
            if x["signature"] == sig_name
        ]).sort_values(
            [
                "macro_f1",
                "accuracy",
            ],
            ascending=False,
        )


        print()
        print("TOP 10")

        print(
            sig_df.head(10).to_string(
                index=False
            )
        )


        best = sig_df.iloc[0]

        best_support = int(
            best["min_support"]
        )

        best_purity = float(
            best["min_purity"]
        )


        best_pred = base_pred.copy()

        for row_idx, x in enumerate(retrieved):

            if (
                x is not None
                and x["support"] >= best_support
                and x["purity"] >= best_purity
            ):
                best_pred[row_idx] = x["pred"]


        saved_preds[sig_name] = best_pred


    result_df = pd.DataFrame(
        all_rows
    ).sort_values(
        [
            "macro_f1",
            "accuracy",
        ],
        ascending=False,
    )


    print()
    print("=" * 140)
    print("GLOBAL TOP 50")
    print("=" * 140)

    print(
        result_df.head(50).to_string(
            index=False
        )
    )


    output_dir = Path(
        "generator_template_probe_fold0"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    result_df.to_csv(
        output_dir / "all_results.csv",
        index=False,
    )


    for sig_name, pred in saved_preds.items():

        np.save(
            output_dir
            / f"best_pred_{sig_name}.npy",
            pred,
        )


    # ------------------------------------------------------------------
    # Greedy composition of distinct signature rules.
    # Start from exact canonical base.
    # ------------------------------------------------------------------

    print()
    print("=" * 140)
    print("GREEDY RULE COMPOSITION")
    print("=" * 140)


    current_pred = base_pred.copy()
    current_f1 = base_f1

    remaining = list(
        result_df.itertuples(index=False)
    )

    selected = []


    for round_idx in range(1, 11):

        best_candidate = None
        best_candidate_pred = None
        best_candidate_f1 = current_f1


        for row in remaining:

            sig_name = row.signature

            table = build_table(
                samples=samples,
                labels=labels,
                indices=train_indices,
                sig_name=sig_name,
            )

            candidate_pred = (
                current_pred.copy()
            )


            for val_idx, sample in enumerate(
                val_samples
            ):

                x = table_predict(
                    sample,
                    table,
                    sig_name,
                )

                if (
                    x is not None
                    and x["support"]
                    >= row.min_support
                    and x["purity"]
                    >= row.min_purity
                ):
                    candidate_pred[
                        val_idx
                    ] = x["pred"]


            candidate_f1 = macro_f1(
                y_val,
                candidate_pred,
            )


            if (
                candidate_f1
                > best_candidate_f1
                + 1e-12
            ):

                best_candidate = row

                best_candidate_pred = (
                    candidate_pred
                )

                best_candidate_f1 = (
                    candidate_f1
                )


        if best_candidate is None:
            break


        gain = (
            best_candidate_f1
            - current_f1
        )


        selected.append({
            "round": round_idx,
            "signature": best_candidate.signature,
            "min_support": best_candidate.min_support,
            "min_purity": best_candidate.min_purity,
            "macro_f1": best_candidate_f1,
            "round_gain": gain,
        })


        current_pred = best_candidate_pred
        current_f1 = best_candidate_f1


        print(
            f"round={round_idx} "
            f"sig={best_candidate.signature} "
            f"support={best_candidate.min_support} "
            f"purity={best_candidate.min_purity} "
            f"F1={current_f1:.12f} "
            f"gain={gain:+.12f}"
        )


        remaining = [
            x
            for x in remaining
            if not (
                x.signature
                == best_candidate.signature
                and x.min_support
                == best_candidate.min_support
                and x.min_purity
                == best_candidate.min_purity
            )
        ]


    greedy_df = pd.DataFrame(
        selected
    )

    greedy_df.to_csv(
        output_dir
        / "greedy_selected.csv",
        index=False,
    )


    np.save(
        output_dir
        / "greedy_final_pred.npy",
        current_pred,
    )


    print()
    print("BASE F1:", base_f1)
    print("GREEDY FINAL F1:", current_f1)
    print(
        "GREEDY GAIN:",
        current_f1 - base_f1,
    )

    print()
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
