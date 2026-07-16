#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier


BASE_SCORE_PATH = Path("exact_repro_base_score.npy")
BASE_PRED_PATH = Path("exact_repro_base_pred.npy")

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

PURITY_PATH = Path(
    "policy_neighborhood_probe_fold0/"
    "val_embeddings.npy"
)

NEIGHBOR_DIR = Path(
    "policy_neighborhood_probe_fold0"
)

TRAIN_LABELS_PATH = Path(
    "policy_cleaning_weights_fold0"
)


def get_neighbor_purity():
    neighbor_idx = np.load(
        NEIGHBOR_DIR / "neighbor_indices.npy"
    )[:, :20]

    train_ids = np.load(
        TRAIN_LABELS_PATH / "train_ids.npy",
        allow_pickle=True,
    ).astype(str)

    labels_df = pd.read_csv(
        "data/train_labels.csv"
    )

    label_map = dict(
        zip(
            labels_df["id"].astype(str),
            labels_df["action"].astype(str),
        )
    )

    classes = [
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

    c2i = {
        x: i
        for i, x in enumerate(classes)
    }

    train_y = np.asarray([
        c2i[label_map[x]]
        for x in train_ids
    ], dtype=np.int64)

    neighbor_labels = train_y[
        neighbor_idx
    ]

    purity = np.zeros(
        len(neighbor_idx),
        dtype=np.float64,
    )

    entropy = np.zeros(
        len(neighbor_idx),
        dtype=np.float64,
    )

    for i, labels in enumerate(
        neighbor_labels
    ):
        counts = np.bincount(
            labels,
            minlength=len(classes),
        ).astype(np.float64)

        probs = counts / counts.sum()

        purity[i] = probs.max()

        entropy[i] = -np.sum(
            probs
            * np.log(
                np.clip(
                    probs,
                    1e-12,
                    1.0,
                )
            )
        )

    return purity, entropy


def cv_auc(X, y):
    n = len(y)

    # deterministic 3-way index split
    folds = (
        np.arange(n) % 3
    )

    pred = np.zeros(
        n,
        dtype=np.float64,
    )

    for fold in range(3):
        tr = folds != fold
        va = folds == fold

        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42 + fold,
        )

        model.fit(
            X[tr],
            y[tr],
        )

        pred[va] = model.predict_proba(
            X[va]
        )[:, 1]

    return roc_auc_score(
        y,
        pred,
    )


def main():
    score = np.load(
        BASE_SCORE_PATH
    ).astype(np.float64)

    base_pred = np.load(
        BASE_PRED_PATH
    ).astype(np.int64)

    val_df = pd.read_csv(
        VAL_REF
    )

    y = (
        val_df["true_id"]
        .astype(int)
        .values
    )

    error = (
        base_pred != y
    ).astype(np.int64)

    order = np.sort(
        score,
        axis=1,
    )

    top1 = order[:, -1]
    top2 = order[:, -2]

    margin = top1 - top2

    # Stable softmax.
    z = (
        score
        - score.max(
            axis=1,
            keepdims=True,
        )
    )

    prob = np.exp(z)

    prob /= prob.sum(
        axis=1,
        keepdims=True,
    )

    prob_order = np.sort(
        prob,
        axis=1,
    )

    prob_top1 = prob_order[:, -1]

    prob_margin = (
        prob_order[:, -1]
        - prob_order[:, -2]
    )

    base_entropy = -np.sum(
        prob
        * np.log(
            np.clip(
                prob,
                1e-12,
                1.0,
            )
        ),
        axis=1,
    )

    purity, neighbor_entropy = (
        get_neighbor_purity()
    )

    confidence_X = np.column_stack([
        top1,
        margin,
        prob_top1,
        prob_margin,
        base_entropy,
    ])

    purity_X = np.column_stack([
        purity,
        neighbor_entropy,
    ])

    combined_X = np.column_stack([
        confidence_X,
        purity_X,
    ])

    print(
        "error rate:",
        float(error.mean()),
    )

    print()
    print(
        "AUC confidence only:",
        cv_auc(
            confidence_X,
            error,
        ),
    )

    print(
        "AUC purity only:",
        cv_auc(
            purity_X,
            error,
        ),
    )

    print(
        "AUC combined:",
        cv_auc(
            combined_X,
            error,
        ),
    )

    print()

    # Compare purity separation inside confidence quintiles.
    confidence = prob_margin

    bins = pd.qcut(
        confidence,
        q=5,
        labels=False,
        duplicates="drop",
    )

    rows = []

    for b in sorted(
        np.unique(bins)
    ):
        mask = bins == b

        correct = (
            mask
            & (error == 0)
        )

        wrong = (
            mask
            & (error == 1)
        )

        rows.append({
            "confidence_bin": int(b),
            "n": int(mask.sum()),
            "error_rate": float(
                error[mask].mean()
            ),
            "correct_purity": float(
                purity[correct].mean()
            ),
            "wrong_purity": float(
                purity[wrong].mean()
            ),
            "purity_gap": float(
                purity[correct].mean()
                - purity[wrong].mean()
            ),
        })

    out = pd.DataFrame(rows)

    print(
        out.to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()
