#!/usr/bin/env python3

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


ROOT = Path(".")
OUT_DIR = Path("candidate_zoo_joint_stacker")

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

BASE_SCORE_PATH = Path("exact_repro_base_v2_score.npy")
BASE_PRED_PATH = Path("exact_repro_base_v2_pred.npy")

N_CLASSES = 14
SEED = 43
N_SPLITS = 5


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
    e = np.exp(z)
    return e / np.maximum(
        e.sum(axis=1, keepdims=True),
        1e-12,
    )


def normalize_to_probs(x):
    x = np.asarray(x, dtype=np.float64)

    # Already probability-like.
    row_sum = x.sum(axis=1)

    if (
        np.isfinite(x).all()
        and x.min() >= -1e-6
        and x.max() <= 1.0 + 1e-6
        and np.mean(np.abs(row_sum - 1.0)) < 1e-3
    ):
        return np.clip(x, 1e-9, 1.0)

    return softmax(x)


def row_zscore(x):
    x = np.asarray(x, dtype=np.float64)

    return (
        x - x.mean(axis=1, keepdims=True)
    ) / np.maximum(
        x.std(axis=1, keepdims=True),
        1e-6,
    )


def prediction_geometry(prob):
    order = np.argsort(prob, axis=1)
    top1 = prob[
        np.arange(len(prob)),
        order[:, -1],
    ]
    top2 = prob[
        np.arange(len(prob)),
        order[:, -2],
    ]

    margin = top1 - top2

    entropy = -np.sum(
        prob * np.log(
            np.maximum(prob, 1e-12)
        ),
        axis=1,
    )

    pred = prob.argmax(axis=1)

    onehot = np.eye(
        N_CLASSES,
        dtype=np.float32,
    )[pred]

    return np.column_stack([
        top1,
        top2,
        margin,
        entropy,
    ]).astype(np.float32), onehot


def exact_id_match(pred_path, ref_ids):
    try:
        df = pd.read_csv(pred_path)

        if "id" not in df.columns:
            return False

        ids = df["id"].astype(str).tolist()

        return ids == ref_ids

    except Exception:
        return False


def candidate_quality_name(path):
    return str(path).replace("/", "__")


def discover_candidates(ref_ids):
    found = []

    # Explicit canonical v2 first.
    base = np.load(BASE_SCORE_PATH)

    if base.shape != (
        len(ref_ids),
        N_CLASSES,
    ):
        raise RuntimeError(
            f"bad base shape: {base.shape}"
        )

    found.append({
        "name": "canonical_v2",
        "path": str(BASE_SCORE_PATH),
        "score": base.astype(np.float64),
        "source": "explicit",
    })

    # Strict automatic discovery:
    # 14-class val_logits + same-dir predictions + exact ID order.
    for path in ROOT.rglob("val_logits.npy"):
        if (
            OUT_DIR in path.parents
            or path.resolve() == BASE_SCORE_PATH.resolve()
        ):
            continue

        pred_path = (
            path.parent / "val_predictions.csv"
        )

        if not pred_path.exists():
            continue

        if not exact_id_match(
            pred_path,
            ref_ids,
        ):
            continue

        try:
            arr = np.load(
                path,
                mmap_mode="r",
            )
        except Exception as e:
            print(
                "SKIP LOAD ERROR:",
                path,
                repr(e),
                flush=True,
            )
            continue

        if arr.shape != (
            len(ref_ids),
            N_CLASSES,
        ):
            continue

        arr = np.asarray(
            arr,
            dtype=np.float64,
        )

        if not np.isfinite(arr).all():
            print(
                "SKIP NONFINITE:",
                path,
                flush=True,
            )
            continue

        found.append({
            "name": candidate_quality_name(
                path.parent
            ),
            "path": str(path),
            "score": arr,
            "source": "strict_auto",
        })

    # Known exact-order probability artifacts from the fusion probe.
    explicit_extra = [
        Path(
            "hidden_tabular_fusion_fold0/"
            "hidden_only_val_probs.npy"
        ),
        Path(
            "hidden_tabular_fusion_fold0/"
            "hidden_plus_tabular_val_probs.npy"
        ),
        Path(
            "hidden_tabular_fusion_fold0/"
            "tabular_only_val_probs.npy"
        ),
    ]

    for path in explicit_extra:
        if not path.exists():
            continue

        arr = np.load(path)

        if arr.shape != (
            len(ref_ids),
            N_CLASSES,
        ):
            print(
                "SKIP EXTRA SHAPE:",
                path,
                arr.shape,
            )
            continue

        found.append({
            "name": candidate_quality_name(
                path.with_suffix("")
            ),
            "path": str(path),
            "score": arr.astype(np.float64),
            "source": "explicit_exact_order",
        })

    # De-duplicate exact arrays.
    unique = []
    signatures = set()

    for item in found:
        arr = item["score"]

        # Cheap deterministic fingerprint.
        idx = np.linspace(
            0,
            len(arr) - 1,
            min(257, len(arr)),
            dtype=np.int64,
        )

        fingerprint = (
            arr.shape,
            np.round(
                arr[idx],
                6,
            ).tobytes(),
        )

        if fingerprint in signatures:
            print(
                "SKIP DUPLICATE:",
                item["path"],
                flush=True,
            )
            continue

        signatures.add(fingerprint)
        unique.append(item)

    return unique


def build_meta_features(candidates):
    blocks = []
    feature_names = []

    probs_by_name = {}

    for item in candidates:
        name = item["name"]
        score = item["score"]

        prob = normalize_to_probs(score)
        probs_by_name[name] = prob

        logp = np.log(
            np.maximum(prob, 1e-9)
        )

        z = row_zscore(score)

        geom, onehot = prediction_geometry(
            prob
        )

        # Model-opinion block.
        block = np.concatenate([
            prob.astype(np.float32),
            logp.astype(np.float32),
            z.astype(np.float32),
            geom,
            onehot,
        ], axis=1)

        blocks.append(block)

        feature_names.extend(
            [f"{name}__prob_{i}" for i in range(14)]
            + [f"{name}__logp_{i}" for i in range(14)]
            + [f"{name}__z_{i}" for i in range(14)]
            + [
                f"{name}__top1",
                f"{name}__top2",
                f"{name}__margin",
                f"{name}__entropy",
            ]
            + [f"{name}__pred_{i}" for i in range(14)]
        )

    # Cross-model disagreement / vote features.
    pred_matrix = np.column_stack([
        probs_by_name[item["name"]].argmax(axis=1)
        for item in candidates
    ])

    vote_counts = np.column_stack([
        (pred_matrix == cls).sum(axis=1)
        for cls in range(N_CLASSES)
    ]).astype(np.float32)

    vote_frac = vote_counts / max(
        1,
        len(candidates),
    )

    mean_prob = np.mean(
        np.stack(
            [
                probs_by_name[item["name"]]
                for item in candidates
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)

    std_prob = np.std(
        np.stack(
            [
                probs_by_name[item["name"]]
                for item in candidates
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)

    disagreement = (
        len(candidates)
        - vote_counts.max(axis=1)
    ).reshape(-1, 1).astype(np.float32)

    blocks.extend([
        vote_counts,
        vote_frac,
        mean_prob,
        std_prob,
        disagreement,
    ])

    feature_names.extend(
        [f"vote_count_{i}" for i in range(14)]
        + [f"vote_frac_{i}" for i in range(14)]
        + [f"mean_prob_{i}" for i in range(14)]
        + [f"std_prob_{i}" for i in range(14)]
        + ["disagreement_n"]
    )

    X = np.concatenate(
        blocks,
        axis=1,
    ).astype(np.float32)

    return X, feature_names


def fit_predict_logreg(
    X_train,
    y_train,
    X_val,
):
    scaler = StandardScaler()

    Xtr = scaler.fit_transform(
        X_train
    )
    Xva = scaler.transform(
        X_val
    )

    clf = LogisticRegression(
        C=0.10,
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=SEED,
    )

    clf.fit(
        Xtr,
        y_train,
    )

    return clf.predict_proba(
        Xva
    )


def fit_predict_histgb(
    X_train,
    y_train,
    X_val,
):
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEED,
    )

    clf.fit(
        X_train,
        y_train,
    )

    return clf.predict_proba(
        X_val
    )


def fit_predict_extratrees(
    X_train,
    y_train,
    X_val,
):
    clf = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )

    clf.fit(
        X_train,
        y_train,
    )

    return clf.predict_proba(
        X_val
    )


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ref = pd.read_csv(
        VAL_REF
    )

    ref_ids = (
        ref["id"]
        .astype(str)
        .tolist()
    )

    y = ref["true_id"].to_numpy(
        dtype=np.int64
    )

    base_pred = np.load(
        BASE_PRED_PATH
    ).astype(np.int64)

    base_f1 = macro_f1(
        y,
        base_pred,
    )

    candidates = discover_candidates(
        ref_ids
    )

    print()
    print("=" * 160)
    print("VALIDATED CANDIDATE INVENTORY")
    print("=" * 160)

    inventory_rows = []

    for idx, item in enumerate(candidates):
        pred = normalize_to_probs(
            item["score"]
        ).argmax(axis=1)

        f1 = macro_f1(
            y,
            pred,
        )

        acc = accuracy_score(
            y,
            pred,
        )

        row = {
            "idx": idx,
            "name": item["name"],
            "source": item["source"],
            "path": item["path"],
            "macro_f1": f1,
            "accuracy": acc,
        }

        inventory_rows.append(row)

        print(
            f"[{idx:02d}] "
            f"F1={f1:.9f} "
            f"ACC={acc:.9f} "
            f"{item['path']}",
            flush=True,
        )

    inventory = pd.DataFrame(
        inventory_rows
    )

    inventory.to_csv(
        OUT_DIR / "candidate_inventory.csv",
        index=False,
    )

    print()
    print(
        "N VALIDATED UNIQUE CANDIDATES:",
        len(candidates),
    )

    if len(candidates) < 3:
        raise RuntimeError(
            "Too few validated candidates for joint stacking"
        )

    X, feature_names = build_meta_features(
        candidates
    )

    print(
        "META FEATURE SHAPE:",
        X.shape,
    )

    (
        OUT_DIR / "feature_names.json"
    ).write_text(
        json.dumps(
            feature_names,
            indent=2,
        )
    )

    np.save(
        OUT_DIR / "meta_features.npy",
        X,
    )

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    model_fns = {
        "logreg": fit_predict_logreg,
        "histgb": fit_predict_histgb,
        "extratrees": fit_predict_extratrees,
    }

    oof_by_model = {
        name: np.zeros(
            (
                len(y),
                N_CLASSES,
            ),
            dtype=np.float64,
        )
        for name in model_fns
    }

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        print()
        print("=" * 160)
        print(
            f"META FOLD {fold}/{N_SPLITS} "
            f"train={len(train_idx)} "
            f"val={len(val_idx)}"
        )
        print("=" * 160)

        for name, fn in model_fns.items():
            print(
                "TRAINING:",
                name,
                flush=True,
            )

            prob = fn(
                X[train_idx],
                y[train_idx],
                X[val_idx],
            )

            oof_by_model[name][
                val_idx
            ] = prob

            fold_pred = prob.argmax(
                axis=1
            )

            fold_f1 = macro_f1(
                y[val_idx],
                fold_pred,
            )

            print(
                name,
                "FOLD F1:",
                fold_f1,
                flush=True,
            )

    print()
    print("=" * 160)
    print("OOF RESULTS")
    print("=" * 160)

    result_rows = []

    for name, prob in oof_by_model.items():
        pred = prob.argmax(axis=1)

        f1 = macro_f1(y, pred)
        acc = accuracy_score(y, pred)

        print(
            name,
            "F1:",
            f1,
            "ACC:",
            acc,
        )

        result_rows.append({
            "name": name,
            "macro_f1": f1,
            "accuracy": acc,
        })

        np.save(
            OUT_DIR / f"{name}_oof_probs.npy",
            prob.astype(np.float32),
        )

    # OOF model-average stacker.
    names = list(
        oof_by_model.keys()
    )

    avg_prob = np.mean(
        np.stack(
            [
                oof_by_model[name]
                for name in names
            ],
            axis=0,
        ),
        axis=0,
    )

    avg_pred = avg_prob.argmax(axis=1)

    avg_f1 = macro_f1(
        y,
        avg_pred,
    )

    avg_acc = accuracy_score(
        y,
        avg_pred,
    )

    print(
        "meta_average F1:",
        avg_f1,
        "ACC:",
        avg_acc,
    )

    result_rows.append({
        "name": "meta_average",
        "macro_f1": avg_f1,
        "accuracy": avg_acc,
    })

    np.save(
        OUT_DIR / "meta_average_oof_probs.npy",
        avg_prob.astype(np.float32),
    )

    # Blend each OOF stacker back with canonical v2.
    base_score = np.load(
        BASE_SCORE_PATH
    ).astype(np.float64)

    base_prob = normalize_to_probs(
        base_score
    )

    print()
    print("=" * 160)
    print("OOF STACKER × CANONICAL V2 BLEND")
    print("=" * 160)

    blend_rows = []

    all_meta = dict(
        oof_by_model
    )
    all_meta[
        "meta_average"
    ] = avg_prob

    alphas = np.concatenate([
        np.linspace(
            0.0,
            0.10,
            21,
        ),
        np.linspace(
            0.11,
            0.30,
            20,
        ),
        np.linspace(
            0.35,
            1.00,
            14,
        ),
    ])

    for name, meta_prob in all_meta.items():
        for alpha in alphas:
            prob = (
                (1.0 - alpha) * base_prob
                + alpha * meta_prob
            )

            pred = prob.argmax(axis=1)

            blend_rows.append({
                "name": name,
                "alpha": float(alpha),
                "macro_f1": macro_f1(
                    y,
                    pred,
                ),
                "gain": macro_f1(
                    y,
                    pred,
                ) - base_f1,
                "accuracy": accuracy_score(
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

    blend_df = pd.DataFrame(
        blend_rows
    ).sort_values(
        "macro_f1",
        ascending=False,
    )

    print(
        blend_df.head(50).to_string(
            index=False,
        )
    )

    blend_df.to_csv(
        OUT_DIR / "v2_blend_sweep.csv",
        index=False,
    )

    result_df = pd.DataFrame(
        result_rows
    ).sort_values(
        "macro_f1",
        ascending=False,
    )

    result_df.to_csv(
        OUT_DIR / "oof_results.csv",
        index=False,
    )

    print()
    print("=" * 160)
    print("FINAL SUMMARY")
    print("=" * 160)

    print(
        "CANONICAL V2 F1:",
        base_f1,
    )

    print()
    print(
        result_df.to_string(
            index=False,
        )
    )

    print()
    print("BEST BLEND:")
    print(
        blend_df.head(1).to_string(
            index=False,
        )
    )

    print()
    print(
        "saved:",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
