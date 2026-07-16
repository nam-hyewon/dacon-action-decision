#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


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

FILE_IDS = np.array([0, 1, 2, 3])

BASE_SCORE = Path(
    "exact_repro_base_v2_score.npy"
)

BASE_PRED = Path(
    "exact_repro_base_v2_pred.npy"
)

SPECIALIST = Path(
    "submit/model/"
    "file_policy_serializer_roberta_base_fold0_seed43"
)

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)


def metric(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(14),
        average="macro",
        zero_division=0,
    )


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ref = pd.read_csv(VAL_REF)

    y = ref["true_id"].astype(int).values
    ref_ids = ref["id"].astype(str).values

    base_score = np.load(BASE_SCORE).astype(np.float64)
    base_pred = np.load(BASE_PRED).astype(np.int64)

    spec_df = pd.read_csv(
        SPECIALIST / "val_predictions.csv"
    )

    spec_logits = np.load(
        SPECIALIST / "val_logits.npy"
    ).astype(np.float64)

    spec_ids = spec_df["id"].astype(str).values

    pos = {
        sid: i
        for i, sid in enumerate(ref_ids)
    }

    spec_global_rows = np.array(
        [pos[sid] for sid in spec_ids],
        dtype=np.int64,
    )

    assert np.array_equal(
        y[spec_global_rows],
        spec_df["true_id"].astype(int).values,
    )

    base_f1 = metric(y, base_pred)

    print("BASE F1:", base_f1)
    print("SPECIALIST ROWS:", len(spec_ids))

    base_file_pred = base_pred[spec_global_rows]

    spec_pred = spec_logits.argmax(axis=1)

    base_correct = (
        base_file_pred
        == y[spec_global_rows]
    )

    spec_correct = (
        spec_pred
        == y[spec_global_rows]
    )

    print(
        "base correct / spec correct:",
        int((base_correct & spec_correct).sum()),
    )
    print(
        "base correct / spec wrong:",
        int((base_correct & ~spec_correct).sum()),
    )
    print(
        "base wrong / spec correct:",
        int((~base_correct & spec_correct).sum()),
    )
    print(
        "both wrong:",
        int((~base_correct & ~spec_correct).sum()),
    )

    oracle_pred = base_pred.copy()

    use = (
        (~base_correct)
        & spec_correct
    )

    oracle_pred[
        spec_global_rows[use]
    ] = spec_pred[use]

    print(
        "ORACLE F1:",
        metric(y, oracle_pred),
    )

    best = None

    for temp in [
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
    ]:
        p = softmax(
            spec_logits / temp
        )

        conf = p.max(axis=1)

        for threshold in np.linspace(
            0.30,
            0.99,
            70,
        ):
            pred = base_pred.copy()

            gate = (
                np.isin(
                    base_file_pred,
                    FILE_IDS,
                )
                & (conf >= threshold)
                & (spec_pred != base_file_pred)
            )

            pred[
                spec_global_rows[gate]
            ] = spec_pred[gate]

            f1 = metric(y, pred)

            beneficial = int(
                (
                    (base_pred != y)
                    & (pred == y)
                ).sum()
            )

            harmful = int(
                (
                    (base_pred == y)
                    & (pred != y)
                ).sum()
            )

            row = {
                "temp": temp,
                "threshold": threshold,
                "f1": f1,
                "gain": f1 - base_f1,
                "use_n": int(gate.sum()),
                "beneficial": beneficial,
                "harmful": harmful,
            }

            if (
                best is None
                or f1 > best["f1"]
            ):
                best = row

    print()
    print("BEST")
    for k, v in best.items():
        print(k, ":", v)


if __name__ == "__main__":
    main()
