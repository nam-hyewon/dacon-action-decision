#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


N_CLASSES = 14

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

LGBM_DIR = Path("tabular_lightgbm_fold0")

BASE_SCORE_PATH = Path("exact_repro_base_score.npy")
BASE_PRED_PATH = Path("exact_repro_base_pred.npy")

OUT_DIR = Path("exact_sota_plus_lightgbm_probe")


def macro_f1(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(N_CLASSES),
        average="macro",
        zero_division=0,
    )


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = x - x.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def logprob(x, temperature=1.0):
    p = softmax(x / temperature)
    return np.log(
        np.clip(p, 1e-12, 1.0)
    )


def row_zscore(x):
    x = np.asarray(x, dtype=np.float64)

    mean = x.mean(
        axis=1,
        keepdims=True,
    )

    std = x.std(
        axis=1,
        keepdims=True,
    )

    return (
        x - mean
    ) / np.clip(
        std,
        1e-6,
        None,
    )


def evaluate(y, score):
    pred = score.argmax(axis=1)

    return (
        macro_f1(y, pred),
        accuracy_score(y, pred),
        pred,
    )


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_df = pd.read_csv(VAL_REF)

    y = (
        val_df["true_id"]
        .astype(int)
        .values
    )

    base_score = np.load(
        BASE_SCORE_PATH
    ).astype(np.float64)

    base_pred_saved = np.load(
        BASE_PRED_PATH
    ).astype(np.int64)

    lgbm_logits = np.load(
        LGBM_DIR / "val_logits.npy"
    ).astype(np.float64)

    if base_score.shape != lgbm_logits.shape:
        raise ValueError(
            f"shape mismatch: "
            f"{base_score.shape} vs "
            f"{lgbm_logits.shape}"
        )

    base_f1, base_acc, base_pred = evaluate(
        y,
        base_score,
    )

    print("BASE F1:", base_f1)
    print("BASE ACC:", base_acc)

    print(
        "BASE PRED EXACT MATCH:",
        np.array_equal(
            base_pred,
            base_pred_saved,
        ),
    )

    lgbm_f1, lgbm_acc, _ = evaluate(
        y,
        lgbm_logits,
    )

    print("LGBM F1:", lgbm_f1)
    print("LGBM ACC:", lgbm_acc)

    rows = []
    best_score = None

    # ----------------------------------------------------------
    # 1. Raw score blend.
    # ----------------------------------------------------------

    for temp in [
        0.50,
        0.65,
        0.75,
        0.85,
        1.00,
        1.20,
        1.50,
        2.00,
        3.00,
    ]:
        lgbm = lgbm_logits / temp

        for alpha in np.linspace(
            0.0,
            0.50,
            101,
        ):
            score = (
                (1.0 - alpha) * base_score
                + alpha * lgbm
            )

            f1, acc, pred = evaluate(
                y,
                score,
            )

            rows.append({
                "family": "raw",
                "temperature": temp,
                "alpha": alpha,
                "macro_f1": f1,
                "gain": f1 - base_f1,
                "accuracy": acc,
                "changed_n": int(
                    (pred != base_pred).sum()
                ),
            })

    # ----------------------------------------------------------
    # 2. Log-probability blend.
    # ----------------------------------------------------------

    base_lp = logprob(base_score)

    for temp in [
        0.50,
        0.65,
        0.75,
        0.85,
        1.00,
        1.20,
        1.50,
        2.00,
        3.00,
    ]:
        lgbm_lp = logprob(
            lgbm_logits,
            temperature=temp,
        )

        for alpha in np.linspace(
            0.0,
            0.50,
            101,
        ):
            score = (
                (1.0 - alpha) * base_lp
                + alpha * lgbm_lp
            )

            f1, acc, pred = evaluate(
                y,
                score,
            )

            rows.append({
                "family": "logprob",
                "temperature": temp,
                "alpha": alpha,
                "macro_f1": f1,
                "gain": f1 - base_f1,
                "accuracy": acc,
                "changed_n": int(
                    (pred != base_pred).sum()
                ),
            })

    # ----------------------------------------------------------
    # 3. Row-zscore blend.
    # ----------------------------------------------------------

    base_z = row_zscore(base_score)

    for temp in [
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
    ]:
        lgbm_z = row_zscore(
            lgbm_logits / temp
        )

        for alpha in np.linspace(
            0.0,
            0.50,
            101,
        ):
            score = (
                (1.0 - alpha) * base_z
                + alpha * lgbm_z
            )

            f1, acc, pred = evaluate(
                y,
                score,
            )

            rows.append({
                "family": "row_zscore",
                "temperature": temp,
                "alpha": alpha,
                "macro_f1": f1,
                "gain": f1 - base_f1,
                "accuracy": acc,
                "changed_n": int(
                    (pred != base_pred).sum()
                ),
            })

    result_df = pd.DataFrame(rows).sort_values(
        [
            "macro_f1",
            "accuracy",
        ],
        ascending=False,
    )

    print()
    print("=" * 140)
    print("TOP 50")
    print("=" * 140)

    print(
        result_df.head(50).to_string(
            index=False
        )
    )

    best = result_df.iloc[0]

    family = str(best["family"])
    temp = float(best["temperature"])
    alpha = float(best["alpha"])

    if family == "raw":
        best_score = (
            (1.0 - alpha) * base_score
            + alpha * (
                lgbm_logits / temp
            )
        )

    elif family == "logprob":
        best_score = (
            (1.0 - alpha) * base_lp
            + alpha * logprob(
                lgbm_logits,
                temperature=temp,
            )
        )

    elif family == "row_zscore":
        best_score = (
            (1.0 - alpha) * base_z
            + alpha * row_zscore(
                lgbm_logits / temp
            )
        )

    else:
        raise RuntimeError(family)

    best_pred = best_score.argmax(axis=1)

    beneficial = (
        (base_pred != y)
        & (best_pred == y)
    )

    harmful = (
        (base_pred == y)
        & (best_pred != y)
    )

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print("BASE F1:", base_f1)
    print(
        "BEST F1:",
        float(best["macro_f1"]),
    )
    print(
        "BEST GAIN:",
        float(best["gain"]),
    )
    print()
    print(best.to_string())

    print()
    print(
        "beneficial:",
        int(beneficial.sum()),
    )
    print(
        "harmful:",
        int(harmful.sum()),
    )
    print(
        "net correct gain:",
        int(
            beneficial.sum()
            - harmful.sum()
        ),
    )

    result_df.to_csv(
        OUT_DIR / "sweep_results.csv",
        index=False,
    )

    np.save(
        OUT_DIR / "best_score.npy",
        best_score.astype(np.float32),
    )

    np.save(
        OUT_DIR / "best_pred.npy",
        best_pred.astype(np.int64),
    )

    with open(
        OUT_DIR / "best_config.txt",
        "w",
    ) as f:
        f.write(best.to_string())
        f.write("\n")

    print()
    print("SAVED:", OUT_DIR)


if __name__ == "__main__":
    main()
