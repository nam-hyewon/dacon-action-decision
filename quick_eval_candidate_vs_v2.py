#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


BASE_SCORE = Path("exact_repro_base_v2_score.npy")
BASE_PRED = Path("exact_repro_base_v2_pred.npy")

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)


def macro_f1(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(14),
        average="macro",
        zero_division=0,
    )


def log_softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    return z - np.log(
        np.exp(z).sum(axis=1, keepdims=True)
    )


def row_zscore(x):
    return (
        x - x.mean(axis=1, keepdims=True)
    ) / np.maximum(
        x.std(axis=1, keepdims=True),
        1e-6,
    )


def transform(x, mode, temp):
    x = x / temp

    if mode == "raw":
        return x

    if mode == "logprob":
        return log_softmax(x)

    if mode == "row_zscore":
        return row_zscore(x)

    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_logits")
    parser.add_argument(
        "--candidate-predictions",
        default=None,
    )
    args = parser.parse_args()

    ref = pd.read_csv(VAL_REF)

    y = ref["true_id"].to_numpy(
        dtype=np.int64
    )

    base_score = np.load(
        BASE_SCORE
    ).astype(np.float64)

    base_pred = np.load(
        BASE_PRED
    ).astype(np.int64)

    cand_score = np.load(
        args.candidate_logits
    ).astype(np.float64)

    print("BASE:", base_score.shape)
    print("CAND:", cand_score.shape)
    print("Y:", y.shape)

    if cand_score.shape != base_score.shape:
        raise RuntimeError(
            f"shape mismatch: "
            f"{cand_score.shape} vs {base_score.shape}"
        )

    if args.candidate_predictions:
        cand_df = pd.read_csv(
            args.candidate_predictions
        )

        ref_ids = (
            ref["id"]
            .astype(str)
            .tolist()
        )

        cand_ids = (
            cand_df["id"]
            .astype(str)
            .tolist()
        )

        if ref_ids != cand_ids:
            raise RuntimeError(
                "candidate ID order mismatch"
            )

        print("ID ORDER: EXACT MATCH")

    cand_pred = cand_score.argmax(axis=1)

    base_f1 = macro_f1(y, base_pred)
    cand_f1 = macro_f1(y, cand_pred)

    base_correct = base_pred == y
    cand_correct = cand_pred == y

    beneficial = int(
        (~base_correct & cand_correct).sum()
    )

    harmful = int(
        (base_correct & ~cand_correct).sum()
    )

    oracle = base_pred.copy()

    use = (
        ~base_correct
        & cand_correct
    )

    oracle[use] = cand_pred[use]

    print()
    print("=" * 100)
    print("CANDIDATE DIAGNOSTIC")
    print("=" * 100)

    print("BASE F1:", base_f1)
    print(
        "BASE ACC:",
        accuracy_score(y, base_pred),
    )

    print("CAND F1:", cand_f1)
    print(
        "CAND ACC:",
        accuracy_score(y, cand_pred),
    )

    print(
        "base wrong / candidate correct:",
        beneficial,
    )

    print(
        "base correct / candidate wrong:",
        harmful,
    )

    print(
        "ORACLE UNION F1:",
        macro_f1(y, oracle),
    )

    rows = []

    modes = [
        "raw",
        "logprob",
        "row_zscore",
    ]

    temps = [
        0.50,
        0.60,
        0.75,
        0.85,
        1.00,
        1.15,
        1.30,
        1.50,
        2.00,
    ]

    alphas = np.concatenate([
        np.linspace(0.0, 0.10, 21),
        np.linspace(0.11, 0.30, 20),
        np.linspace(0.35, 1.00, 14),
    ])

    for mode in modes:
        bt = transform(
            base_score,
            mode,
            1.0,
        )

        for temp in temps:
            ct = transform(
                cand_score,
                mode,
                temp,
            )

            for alpha in alphas:
                score = (
                    (1.0 - alpha) * bt
                    + alpha * ct
                )

                pred = score.argmax(axis=1)

                rows.append({
                    "mode": mode,
                    "temp": temp,
                    "alpha": float(alpha),
                    "f1": macro_f1(y, pred),
                    "gain": (
                        macro_f1(y, pred)
                        - base_f1
                    ),
                    "acc": accuracy_score(
                        y,
                        pred,
                    ),
                    "changed": int(
                        (pred != base_pred).sum()
                    ),
                    "beneficial": int(
                        (
                            (base_pred != y)
                            & (pred == y)
                        ).sum()
                    ),
                    "harmful": int(
                        (
                            (base_pred == y)
                            & (pred != y)
                        ).sum()
                    ),
                })

    result = pd.DataFrame(rows).sort_values(
        "f1",
        ascending=False,
    )

    print()
    print("=" * 120)
    print("TOP BLENDS")
    print("=" * 120)

    print(
        result.head(30).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
