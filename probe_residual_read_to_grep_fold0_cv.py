#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


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

READ_ID = C2I["read_file"]
GREP_ID = C2I["grep_search"]

ROOT = Path(".")

REPRO = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

VAL_REF = (
    REPRO
    / "submit/model/kluebert_diet_fold0"
    / "val_predictions.csv"
)

BASE_SCORE_PATH = Path(
    "exact_repro_base_score.npy"
)

BASE_PRED_PATH = Path(
    "exact_repro_base_pred.npy"
)

OUTPUT_DIR = Path(
    "residual_read_to_grep_fold0_cv"
)


def macro_f1(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )


def load_samples():
    samples = []

    with open(
        "data/train.jsonl",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    return {
        str(sample["id"]): sample
        for sample in samples
    }


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def clip_text(x, n):
    x = str(x or "")
    x = x.replace("\n", " ")
    return x[:n]


def arg_path(args):
    if not isinstance(args, dict):
        return ""

    for key in [
        "path",
        "file_path",
        "directory",
        "dir",
        "pattern",
        "query",
        "command",
    ]:
        if key in args:
            return str(args[key])

    return ""


def build_residual_text(sample):
    """
    Intentionally compact.
    Residual detector should see decision cues,
    not re-learn the whole 14-way task.
    """

    prompt = clip_text(
        sample.get("current_prompt", ""),
        1600,
    )

    history = sample.get("history", [])

    actions = get_actions(sample)

    last_action = (
        actions[-1]
        if len(actions) >= 1
        else "NONE"
    )

    prev_action = (
        actions[-2]
        if len(actions) >= 2
        else "NONE"
    )

    recent = []

    for h in history[-12:]:
        role = str(h.get("role", ""))

        if role == "assistant_action":
            name = str(h.get("name", ""))
            path = clip_text(
                arg_path(h.get("args", {})),
                100,
            )
            result = clip_text(
                h.get("result_summary", ""),
                180,
            )

            recent.append(
                f"ACTION={name} "
                f"PATH={path} "
                f"RESULT={result}"
            )

        elif role == "user":
            content = clip_text(
                h.get("content", ""),
                300,
            )

            recent.append(
                f"USER={content}"
            )

    action_seq = " ".join(
        actions[-8:]
    )

    return "\n".join([
        "[CURRENT_PROMPT]",
        prompt,
        "[STATE]",
        f"LAST_ACTION={last_action}",
        f"PREV_ACTION={prev_action}",
        f"ACTION_SEQUENCE={action_seq}",
        f"HISTORY_LEN={len(history)}",
        "[RECENT_HISTORY]",
        "\n".join(recent),
    ])


def load_component_logits():
    """
    Locate canonical component val_logits by known repro structure.
    """

    wanted = {
        "large": "klue_roberta_large",
        "tab": "tabular",
        "diet": "kluebert_diet",
        "rob8": "roberta_eda_state8",
    }

    all_logits = list(
        REPRO.rglob("val_logits.npy")
    )

    print("Found val_logits files:")

    for p in all_logits:
        print(" ", p)

    result = {}

    for key, token in wanted.items():
        matches = [
            p
            for p in all_logits
            if token.lower() in str(p).lower()
        ]

        if len(matches) != 1:
            print(
                f"WARNING: {key} token={token} "
                f"matches={len(matches)}"
            )

            continue

        result[key] = np.load(
            matches[0]
        ).astype(np.float32)

    return result


def make_dense_features(
    base_score,
    component_logits,
):
    """
    Ensemble score geometry + individual model disagreement.
    """

    feats = []

    # Exact ensemble score.
    feats.append(base_score)

    order = np.sort(
        base_score,
        axis=1,
    )

    feats.append(
        order[:, -1:]
    )

    feats.append(
        order[:, -2:-1]
    )

    feats.append(
        (
            order[:, -1]
            - order[:, -2]
        )[:, None]
    )

    read_grep_margin = (
        base_score[:, READ_ID]
        - base_score[:, GREP_ID]
    )

    feats.append(
        read_grep_margin[:, None]
    )

    # Softmax geometry.
    z = (
        base_score
        - base_score.max(
            axis=1,
            keepdims=True,
        )
    )

    prob = np.exp(z)

    prob /= prob.sum(
        axis=1,
        keepdims=True,
    )

    feats.append(prob)

    feats.append(
        (
            prob[:, READ_ID]
            - prob[:, GREP_ID]
        )[:, None]
    )

    entropy = -np.sum(
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

    feats.append(
        entropy[:, None]
    )

    # Component-specific logits and margins.
    for name in sorted(
        component_logits
    ):
        logits = component_logits[name]

        feats.append(logits)

        feats.append(
            (
                logits[:, READ_ID]
                - logits[:, GREP_ID]
            )[:, None]
        )

        pred = logits.argmax(
            axis=1
        )

        feats.append(
            (pred == READ_ID)
            .astype(np.float32)[:, None]
        )

        feats.append(
            (pred == GREP_ID)
            .astype(np.float32)[:, None]
        )

    dense = np.concatenate(
        feats,
        axis=1,
    ).astype(np.float32)

    # Standardize globally only by fixed per-column arithmetic.
    # For OOF strictness, scaler stats are later fit per fold.
    return dense


def standardize_train_val(
    train_x,
    val_x,
):
    mean = train_x.mean(
        axis=0,
        keepdims=True,
    )

    std = train_x.std(
        axis=0,
        keepdims=True,
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std,
    )

    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading validation reference...")

    val_df = pd.read_csv(
        VAL_REF
    )

    val_ids = (
        val_df["id"]
        .astype(str)
        .tolist()
    )

    y_all = (
        val_df["true_id"]
        .astype(int)
        .values
    )

    base_score = np.load(
        BASE_SCORE_PATH
    ).astype(np.float32)

    base_pred = np.load(
        BASE_PRED_PATH
    ).astype(np.int64)

    if not (
        len(val_ids)
        == len(y_all)
        == len(base_score)
        == len(base_pred)
    ):
        raise ValueError(
            "Validation length mismatch"
        )

    print(
        "BASE F1:",
        macro_f1(
            y_all,
            base_pred,
        ),
    )

    print(
        "BASE ACC:",
        float(
            (base_pred == y_all).mean()
        ),
    )

    samples = load_samples()

    component_logits = (
        load_component_logits()
    )

    print(
        "Loaded components:",
        sorted(component_logits),
    )

    for name, logits in (
        component_logits.items()
    ):
        if len(logits) != len(y_all):
            raise ValueError(
                f"{name} length mismatch"
            )

    # ------------------------------------------------------------
    # Condition on exact SOTA predicting read_file.
    # ------------------------------------------------------------

    candidate_global_idx = np.where(
        base_pred == READ_ID
    )[0]

    y_true_candidate = y_all[
        candidate_global_idx
    ]

    # Positive = exact SOTA read_file prediction was actually grep_search.
    residual_y = (
        y_true_candidate == GREP_ID
    ).astype(np.int64)

    print()
    print("=" * 100)
    print("RESIDUAL DATASET")
    print("=" * 100)

    print(
        "base_pred=read_file n:",
        len(candidate_global_idx),
    )

    print(
        "positive true=grep_search:",
        int(residual_y.sum()),
    )

    print(
        "negative:",
        int(
            (residual_y == 0).sum()
        ),
    )

    print(
        "positive rate:",
        float(residual_y.mean()),
    )

    true_counts = pd.Series(
        y_true_candidate
    ).value_counts().sort_index()

    print()
    print(
        "true label distribution "
        "inside base_pred=read_file:"
    )

    for class_id, n in (
        true_counts.items()
    ):
        print(
            f"{CLASSES[int(class_id)]:20s}",
            int(n),
        )

    texts = [
        build_residual_text(
            samples[val_ids[i]]
        )
        for i in candidate_global_idx
    ]

    dense_all = make_dense_features(
        base_score[
            candidate_global_idx
        ],
        {
            name: logits[
                candidate_global_idx
            ]
            for name, logits
            in component_logits.items()
        },
    )

    print()
    print(
        "Dense features:",
        dense_all.shape,
    )

    # ------------------------------------------------------------
    # Strict OOF.
    # TF-IDF fit only on fold train.
    # Classifier learns only residual objective.
    # ------------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    oof_prob = np.zeros(
        len(residual_y),
        dtype=np.float64,
    )

    fold_rows = []

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        skf.split(
            np.zeros(len(residual_y)),
            residual_y,
        )
    ):
        print()
        print("=" * 100)
        print(
            f"FOLD {fold}"
        )
        print("=" * 100)

        train_text = [
            texts[i]
            for i in train_idx
        ]

        val_text = [
            texts[i]
            for i in val_idx
        ]

        word_vec = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.995,
            max_features=80000,
            sublinear_tf=True,
            strip_accents=None,
            dtype=np.float32,
        )

        char_vec = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=60000,
            sublinear_tf=True,
            dtype=np.float32,
        )

        word_train = word_vec.fit_transform(
            train_text
        )

        word_val = word_vec.transform(
            val_text
        )

        char_train = char_vec.fit_transform(
            train_text
        )

        char_val = char_vec.transform(
            val_text
        )

        dense_train, dense_val = (
            standardize_train_val(
                dense_all[train_idx],
                dense_all[val_idx],
            )
        )

        x_train = hstack(
            [
                word_train,
                char_train,
                csr_matrix(dense_train),
            ],
            format="csr",
        )

        x_val = hstack(
            [
                word_val,
                char_val,
                csr_matrix(dense_val),
            ],
            format="csr",
        )

        print(
            "X train:",
            x_train.shape,
            "| nnz:",
            x_train.nnz,
        )

        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42 + fold,
        )

        model.fit(
            x_train,
            residual_y[train_idx],
        )

        prob = model.predict_proba(
            x_val
        )[:, 1]

        oof_prob[val_idx] = prob

        fold_auc = roc_auc_score(
            residual_y[val_idx],
            prob,
        )

        fold_ap = average_precision_score(
            residual_y[val_idx],
            prob,
        )

        print(
            "AUC:",
            fold_auc,
        )

        print(
            "AP:",
            fold_ap,
        )

        fold_rows.append({
            "fold": fold,
            "auc": fold_auc,
            "ap": fold_ap,
            "n": len(val_idx),
            "positive_n": int(
                residual_y[val_idx].sum()
            ),
        })

    print()
    print("=" * 120)
    print("OOF RESIDUAL QUALITY")
    print("=" * 120)

    print(
        "OOF AUC:",
        roc_auc_score(
            residual_y,
            oof_prob,
        ),
    )

    print(
        "OOF AP:",
        average_precision_score(
            residual_y,
            oof_prob,
        ),
    )

    # ------------------------------------------------------------
    # Threshold sweep on OOF probabilities.
    # Evaluate the REAL final task metric.
    # ------------------------------------------------------------

    base_f1 = macro_f1(
        y_all,
        base_pred,
    )

    result_rows = []

    thresholds = np.unique(
        np.concatenate([
            np.linspace(
                0.05,
                0.95,
                91,
            ),
            np.quantile(
                oof_prob,
                np.linspace(
                    0.50,
                    0.995,
                    100,
                ),
            ),
        ])
    )

    for threshold in thresholds:
        flip_local = (
            oof_prob >= threshold
        )

        pred = base_pred.copy()

        flip_global_idx = (
            candidate_global_idx[
                flip_local
            ]
        )

        pred[
            flip_global_idx
        ] = GREP_ID

        changed = (
            pred != base_pred
        )

        beneficial = (
            changed
            & (base_pred != y_all)
            & (pred == y_all)
        )

        harmful = (
            changed
            & (base_pred == y_all)
            & (pred != y_all)
        )

        changed_n = int(
            changed.sum()
        )

        beneficial_n = int(
            beneficial.sum()
        )

        harmful_n = int(
            harmful.sum()
        )

        precision = (
            beneficial_n / changed_n
            if changed_n > 0
            else 0.0
        )

        pred_f1 = macro_f1(
            y_all,
            pred,
        )

        result_rows.append({
            "threshold": float(threshold),
            "macro_f1": pred_f1,
            "gain": pred_f1 - base_f1,
            "accuracy": float(
                (pred == y_all).mean()
            ),
            "changed_n": changed_n,
            "beneficial": beneficial_n,
            "harmful": harmful_n,
            "net_correct_gain": (
                beneficial_n
                - harmful_n
            ),
            "flip_precision": precision,
        })

    result_df = pd.DataFrame(
        result_rows
    ).sort_values(
        [
            "macro_f1",
            "net_correct_gain",
        ],
        ascending=False,
    )

    print()
    print("=" * 160)
    print("TOP THRESHOLDS BY MACRO-F1")
    print("=" * 160)

    print(
        result_df.head(50).to_string(
            index=False
        )
    )

    # High-precision operating points.
    hp = result_df[
        result_df["flip_precision"] >= 0.70
    ].sort_values(
        [
            "net_correct_gain",
            "macro_f1",
        ],
        ascending=False,
    )

    print()
    print("=" * 160)
    print("HIGH PRECISION >= 0.70")
    print("=" * 160)

    if len(hp):
        print(
            hp.head(30).to_string(
                index=False
            )
        )
    else:
        print(
            "NO threshold with precision >= 0.70"
        )

    # ------------------------------------------------------------
    # Score diagnostics.
    # ------------------------------------------------------------

    positive_prob = oof_prob[
        residual_y == 1
    ]

    negative_prob = oof_prob[
        residual_y == 0
    ]

    print()
    print("=" * 120)
    print("PROBABILITY DISTRIBUTION")
    print("=" * 120)

    for name, values in [
        (
            "POS true=grep",
            positive_prob,
        ),
        (
            "NEG not-grep",
            negative_prob,
        ),
    ]:
        print(
            name,
            "| mean=",
            float(values.mean()),
            "| p25=",
            float(
                np.percentile(values, 25)
            ),
            "| p50=",
            float(
                np.percentile(values, 50)
            ),
            "| p75=",
            float(
                np.percentile(values, 75)
            ),
            "| p90=",
            float(
                np.percentile(values, 90)
            ),
        )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    result_df.to_csv(
        OUTPUT_DIR / "threshold_results.csv",
        index=False,
    )

    np.save(
        OUTPUT_DIR / "candidate_global_idx.npy",
        candidate_global_idx,
    )

    np.save(
        OUTPUT_DIR / "residual_y.npy",
        residual_y,
    )

    np.save(
        OUTPUT_DIR / "oof_prob.npy",
        oof_prob,
    )

    best = result_df.iloc[0]

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print(
        "BASE F1:",
        base_f1,
    )

    print(
        "BEST F1:",
        float(best["macro_f1"]),
    )

    print(
        "BEST GAIN:",
        float(best["gain"]),
    )

    print()
    print("BEST CONFIG")
    print(
        best.to_string()
    )

    print()
    print(
        "SAVED:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
