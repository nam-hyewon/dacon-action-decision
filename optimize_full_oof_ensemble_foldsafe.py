#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path("oof_bundle_20260714")
OOF_ROOT = ROOT / "oof"
INDEX_PATH = ROOT / "index.csv"

N_CLASSES = 14
TOP_K_MODELS = 10
MAX_ROUNDS = 12


def macro_f1(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(N_CLASSES),
        average="macro",
        zero_division=0,
    )


def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = x - x.max(axis=1, keepdims=True)

    return z - np.log(
        np.exp(z).sum(
            axis=1,
            keepdims=True,
        )
    )


def row_zscore(x):
    x = np.asarray(x, dtype=np.float64)

    return (
        x - x.mean(axis=1, keepdims=True)
    ) / np.maximum(
        x.std(axis=1, keepdims=True),
        1e-6,
    )


def transform(x, mode, temp):
    z = x / temp

    if mode == "raw":
        return z

    if mode == "logprob":
        return log_softmax(z)

    if mode == "row_zscore":
        return row_zscore(z)

    raise ValueError(mode)


def load_logits():
    result = {}

    for path in sorted(
        OOF_ROOT.glob("*/logits.npy")
    ):
        name = path.parent.name
        arr = np.load(path)

        if arr.shape != (70000, 14):
            continue

        if not np.isfinite(arr).all():
            continue

        result[name] = arr.astype(
            np.float64
        )

    return result


def greedy_fit(
    y,
    idx,
    transformed,
    ranked_names,
):
    # Start from best single model on train folds.
    best_name = max(
        ranked_names,
        key=lambda name: macro_f1(
            y[idx],
            transformed[name][idx].argmax(
                axis=1
            ),
        ),
    )

    current_score = transformed[
        best_name
    ][idx].copy()

    weights = {
        best_name: 1.0
    }

    current_f1 = macro_f1(
        y[idx],
        current_score.argmax(axis=1),
    )

    print(
        "START",
        best_name,
        current_f1,
        flush=True,
    )

    alpha_grid = np.concatenate([
        np.linspace(
            0.01,
            0.10,
            10,
        ),
        np.linspace(
            0.12,
            0.30,
            10,
        ),
        np.linspace(
            0.35,
            0.60,
            6,
        ),
    ])

    for round_idx in range(
        1,
        MAX_ROUNDS + 1,
    ):
        best = None

        for name in ranked_names:
            cand = transformed[name][idx]

            for alpha in alpha_grid:
                trial = (
                    (1.0 - alpha) * current_score
                    + alpha * cand
                )

                f1 = macro_f1(
                    y[idx],
                    trial.argmax(axis=1),
                )

                gain = f1 - current_f1

                if (
                    best is None
                    or gain > best["gain"]
                ):
                    best = {
                        "name": name,
                        "alpha": float(alpha),
                        "gain": gain,
                        "f1": f1,
                        "score": trial,
                    }

        if (
            best is None
            or best["gain"] <= 1e-7
        ):
            break

        alpha = best["alpha"]

        weights = {
            name: weight * (
                1.0 - alpha
            )
            for name, weight
            in weights.items()
        }

        weights[best["name"]] = (
            weights.get(
                best["name"],
                0.0,
            )
            + alpha
        )

        current_score = best["score"]
        current_f1 = best["f1"]

        print(
            f"ROUND {round_idx} "
            f"name={best['name']} "
            f"alpha={alpha:.4f} "
            f"gain={best['gain']:.9f} "
            f"f1={current_f1:.9f}",
            flush=True,
        )

    return weights, current_f1


def score_with_weights(
    transformed,
    idx,
    weights,
):
    score = None

    for name, weight in weights.items():
        cur = (
            weight
            * transformed[name][idx]
        )

        if score is None:
            score = cur.copy()
        else:
            score += cur

    return score


def main():
    index = pd.read_csv(
        INDEX_PATH
    )

    y = index["true_id"].to_numpy(
        dtype=np.int64
    )

    folds = index["fold"].to_numpy(
        dtype=np.int64
    )

    logits = load_logits()

    print("=" * 140)
    print("STANDALONE OOF")
    print("=" * 140)

    standalone = []

    for name, arr in logits.items():
        f1 = macro_f1(
            y,
            arr.argmax(axis=1),
        )

        standalone.append(
            (name, f1)
        )

    standalone.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    for name, f1 in standalone:
        print(
            f"{name:60s} {f1:.10f}"
        )

    # Do not ensemble the already-ensembled base as a component.
    ranked_names = [
        name
        for name, _
        in standalone
        if not name.startswith("ens_")
    ][:TOP_K_MODELS]

    print()
    print(
        "SELECTED COMPONENTS:"
    )

    for name in ranked_names:
        print(" -", name)

    modes = [
        "raw",
        "logprob",
        "row_zscore",
    ]

    temps = [
        0.75,
        0.85,
        1.00,
        1.15,
        1.30,
    ]

    all_results = []
    oof_predictions = {}

    for mode in modes:
        for temp in temps:
            print()
            print("=" * 140)
            print(
                "MODE",
                mode,
                "TEMP",
                temp,
            )
            print("=" * 140)

            transformed = {
                name: transform(
                    logits[name],
                    mode,
                    temp,
                )
                for name in ranked_names
            }

            oof_pred = np.zeros(
                len(y),
                dtype=np.int64,
            )

            fold_rows = []

            for fold in sorted(
                np.unique(folds)
            ):
                train_idx = np.where(
                    folds != fold
                )[0]

                val_idx = np.where(
                    folds == fold
                )[0]

                print()
                print(
                    f"--- HOLDOUT FOLD {fold} ---"
                )

                weights, train_f1 = (
                    greedy_fit(
                        y,
                        train_idx,
                        transformed,
                        ranked_names,
                    )
                )

                val_score = score_with_weights(
                    transformed,
                    val_idx,
                    weights,
                )

                val_pred = val_score.argmax(
                    axis=1
                )

                val_f1 = macro_f1(
                    y[val_idx],
                    val_pred,
                )

                oof_pred[
                    val_idx
                ] = val_pred

                print(
                    "WEIGHTS:",
                    weights,
                )

                print(
                    "TRAIN F1:",
                    train_f1,
                )

                print(
                    "VAL F1:",
                    val_f1,
                )

                fold_rows.append({
                    "fold": int(fold),
                    "train_f1": train_f1,
                    "val_f1": val_f1,
                    "weights": str(weights),
                })

            full_f1 = macro_f1(
                y,
                oof_pred,
            )

            print()
            print(
                "FULL FOLDSAFE OOF F1:",
                full_f1,
            )

            key = (
                f"{mode}_temp{temp}"
            )

            oof_predictions[key] = (
                oof_pred.copy()
            )

            all_results.append({
                "mode": mode,
                "temp": temp,
                "full_oof_f1": full_f1,
            })

    result = pd.DataFrame(
        all_results
    ).sort_values(
        "full_oof_f1",
        ascending=False,
    )

    print()
    print("=" * 140)
    print("FINAL")
    print("=" * 140)

    print(
        result.to_string(
            index=False,
        )
    )

    best = result.iloc[0]

    best_key = (
        f"{best['mode']}_temp"
        f"{best['temp']}"
    )

    np.save(
        "best_foldsafe_ensemble_oof_pred.npy",
        oof_predictions[best_key],
    )

    result.to_csv(
        "full_oof_ensemble_foldsafe_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
