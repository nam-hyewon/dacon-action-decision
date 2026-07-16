import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

label2id = {c: i for i, c in enumerate(ALL_CLASSES)}
id2label = {i: c for i, c in enumerate(ALL_CLASSES)}

FILE4 = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]

FILE4_IDX = np.array(
    [label2id[c] for c in FILE4],
    dtype=int,
)

file4_label2id = {
    c: i for i, c in enumerate(FILE4)
}


ROOT = Path(".")

BASE_DIR = (
    ROOT
    / "team_4model_ensemble_07626"
    / "ensemble_07626"
    / "fold0_team"
)

OOF_PATHS = {
    "deberta": BASE_DIR / "deberta_base_eda_state4_seed43_oof",
    "kluebert": BASE_DIR / "kluebert_mid5_lr15_oof",
    "roberta": BASE_DIR / "roberta_eda_state4_oof",
    "tabular": BASE_DIR / "tabular_histgb_oof",
    "xlmr": BASE_DIR / "xlmr_base_eda_state_seed43_oof",
}

WEIGHTS = {
    "deberta": 0.17,
    "kluebert": 0.16,
    "roberta": 0.17,
    "tabular": 0.43,
    "xlmr": 0.07,
}

OUT_DIR = (
    BASE_DIR
    / "file4_logit_meta_specialist_all5_crossfit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_KINDS = [
    "hgb_small",
    "hgb",
]

THRESHOLDS = np.arange(
    0.25,
    0.96,
    0.025,
)

MARGINS = [
    None,
    0.00,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
]

MODES = [
    "base_pred_file4",
    "base_top2_has_file4",
]


def row_zscore(x, eps=1e-6):
    return (
        x - x.mean(axis=1, keepdims=True)
    ) / (
        x.std(axis=1, keepdims=True) + eps
    )


def softmax(x):
    x = x - x.max(
        axis=1,
        keepdims=True,
    )

    e = np.exp(x)

    return e / e.sum(
        axis=1,
        keepdims=True,
    )


def margin_from_logits(x):
    s = np.sort(
        x,
        axis=1,
    )

    return s[:, -1] - s[:, -2]


def rank_of_classes(
    logits,
    class_indices,
):
    order = np.argsort(
        -logits,
        axis=1,
    )

    ranks = np.zeros(
        (
            logits.shape[0],
            len(class_indices),
        ),
        dtype=np.float32,
    )

    for j, cls in enumerate(
        class_indices
    ):
        ranks[:, j] = np.where(
            order == cls
        )[1]

    return ranks


def find_label_column(df):
    for c in [
        "label",
        "true_action",
        "true_label",
        "action",
        "target",
    ]:
        if c in df.columns:
            return c

    return None


def load_one(
    name,
    path,
):
    logits = np.load(
        path / "val_logits.npy"
    )

    df = pd.read_csv(
        path / "val_predictions.csv"
    )

    df["id"] = df["id"].astype(str)

    label_col = find_label_column(df)

    if label_col is None:
        raise ValueError(
            f"{name}: cannot find label column: "
            f"{df.columns.tolist()}"
        )

    if label_col != "label":
        df["label"] = (
            df[label_col]
            .astype(str)
        )

    if "fold" not in df.columns:
        raise ValueError(
            f"{name}: OOF val_predictions.csv "
            f"must contain fold column"
        )

    if len(df) != logits.shape[0]:
        raise ValueError(
            f"{name}: df/logits mismatch "
            f"{len(df)} vs {logits.shape}"
        )

    return logits, df


def load_all():
    ref_ids = None
    ref_df = None

    zlogits = {}

    for name, path in OOF_PATHS.items():
        logits, df = load_one(
            name,
            path,
        )

        ids = (
            df["id"]
            .astype(str)
            .tolist()
        )

        if ref_ids is None:
            ref_ids = ids
            ref_df = df.copy()

        else:
            if ids != ref_ids:
                if set(ids) == set(ref_ids):
                    print(
                        f"[reorder] {name}"
                    )

                    id_to_pos = {
                        id_: i
                        for i, id_
                        in enumerate(ids)
                    }

                    order = [
                        id_to_pos[i]
                        for i in ref_ids
                    ]

                    logits = logits[order]

                    df = (
                        df.iloc[order]
                        .reset_index(drop=True)
                    )

                    assert (
                        df["id"]
                        .astype(str)
                        .tolist()
                        == ref_ids
                    )

                else:
                    raise ValueError(
                        f"{name}: id set mismatch"
                    )

        zlogits[name] = (
            row_zscore(logits)
            .astype(np.float32)
        )

        print(
            f"{name:9s} loaded "
            f"{logits.shape}"
        )

    y_true = np.array(
        [
            label2id[x]
            for x in (
                ref_df["label"]
                .astype(str)
                .tolist()
            )
        ],
        dtype=int,
    )

    folds = (
        ref_df["fold"]
        .astype(int)
        .to_numpy()
    )

    return (
        zlogits,
        ref_df,
        y_true,
        folds,
    )


def build_base_logits(zlogits):
    base = None

    for name, w in WEIGHTS.items():
        part = (
            w * zlogits[name]
        )

        base = (
            part
            if base is None
            else base + part
        )

    return base.astype(
        np.float32
    )


def build_features(
    zlogits,
    base_logits,
):
    feats = []
    names = []

    base_prob = softmax(
        base_logits
    )

    base_pred = base_logits.argmax(
        axis=1
    )

    base_conf = base_prob.max(
        axis=1
    )

    base_margin = margin_from_logits(
        base_logits
    )

    feats.append(base_logits)

    names += [
        f"base_logit_{c}"
        for c in ALL_CLASSES
    ]

    feats.append(base_prob)

    names += [
        f"base_prob_{c}"
        for c in ALL_CLASSES
    ]

    feats.append(
        base_logits[:, FILE4_IDX]
    )

    names += [
        f"base_file4_logit_{c}"
        for c in FILE4
    ]

    feats.append(
        base_prob[:, FILE4_IDX]
    )

    names += [
        f"base_file4_prob_{c}"
        for c in FILE4
    ]

    feats.append(
        rank_of_classes(
            base_logits,
            FILE4_IDX,
        )
    )

    names += [
        f"base_rank_{c}"
        for c in FILE4
    ]

    feats.append(
        base_conf.reshape(-1, 1)
    )

    names.append("base_conf")

    feats.append(
        base_margin.reshape(-1, 1)
    )

    names.append("base_margin")

    individual_preds = []

    for (
        model_name,
        logits,
    ) in zlogits.items():

        prob = softmax(logits)

        pred = logits.argmax(
            axis=1
        )

        individual_preds.append(
            pred
        )

        feats.append(logits)

        names += [
            f"{model_name}_logit_{c}"
            for c in ALL_CLASSES
        ]

        feats.append(prob)

        names += [
            f"{model_name}_prob_{c}"
            for c in ALL_CLASSES
        ]

        feats.append(
            logits[:, FILE4_IDX]
        )

        names += [
            f"{model_name}_file4_logit_{c}"
            for c in FILE4
        ]

        feats.append(
            prob[:, FILE4_IDX]
        )

        names += [
            f"{model_name}_file4_prob_{c}"
            for c in FILE4
        ]

        feats.append(
            rank_of_classes(
                logits,
                FILE4_IDX,
            )
        )

        names += [
            f"{model_name}_rank_{c}"
            for c in FILE4
        ]

        feats.append(
            prob.max(axis=1)
            .reshape(-1, 1)
        )

        names.append(
            f"{model_name}_conf"
        )

        feats.append(
            margin_from_logits(logits)
            .reshape(-1, 1)
        )

        names.append(
            f"{model_name}_margin"
        )

    pred_mat = np.stack(
        individual_preds,
        axis=1,
    )

    n_models = pred_mat.shape[1]

    disagree_count = np.zeros(
        pred_mat.shape[0],
        dtype=np.float32,
    )

    for i in range(n_models):
        for j in range(
            i + 1,
            n_models,
        ):
            disagree_count += (
                pred_mat[:, i]
                != pred_mat[:, j]
            ).astype(np.float32)

    feats.append(
        disagree_count.reshape(-1, 1)
    )

    names.append(
        "pairwise_disagree_count"
    )

    for (
        mi,
        model_name,
    ) in enumerate(zlogits.keys()):

        agree = (
            pred_mat[:, mi]
            == base_pred
        ).astype(np.float32)

        feats.append(
            agree.reshape(-1, 1)
        )

        names.append(
            f"{model_name}_agree_base"
        )

    X = np.concatenate(
        feats,
        axis=1,
    ).astype(np.float32)

    return X, names


def make_model(kind):
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=43,
        )

    if kind == "hgb_small":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.03,
            max_iter=160,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=43,
        )

    raise ValueError(kind)


def eval_idx(
    y_true,
    pred,
):
    return (
        f1_score(
            y_true,
            pred,
            average="macro",
        ),
        accuracy_score(
            y_true,
            pred,
        ),
    )


def make_route_prediction(
    base_pred_idx,
    meta_probs,
    route_base_mask,
    threshold,
    margin,
):
    meta_pred_file4 = (
        meta_probs.argmax(axis=1)
    )

    meta_conf = (
        meta_probs.max(axis=1)
    )

    sorted_probs = np.sort(
        meta_probs,
        axis=1,
    )

    meta_margin = (
        sorted_probs[:, -1]
        - sorted_probs[:, -2]
    )

    conf_mask = (
        meta_conf >= threshold
    )

    if margin is None:
        margin_mask = np.ones(
            len(base_pred_idx),
            dtype=bool,
        )

    else:
        margin_mask = (
            meta_margin >= margin
        )

    replace_mask = (
        route_base_mask
        & conf_mask
        & margin_mask
    )

    pred = base_pred_idx.copy()

    pred[replace_mask] = (
        FILE4_IDX[
            meta_pred_file4[
                replace_mask
            ]
        ]
    )

    return (
        pred,
        replace_mask,
    )


def main():
    print(
        "===== load ALL5 OOF logits ====="
    )

    (
        zlogits,
        ref_df,
        y_true,
        folds,
    ) = load_all()

    base_logits = build_base_logits(
        zlogits
    )

    base_pred_idx = (
        base_logits.argmax(axis=1)
    )

    base_f1, base_acc = eval_idx(
        y_true,
        base_pred_idx,
    )

    print()
    print(
        "===== base ALL5 ensemble ====="
    )

    print("weights:", WEIGHTS)

    print(
        f"macro_f1={base_f1:.10f}"
    )

    print(
        f"accuracy={base_acc:.10f}"
    )


    print()
    print(
        "===== build ALL5 features ====="
    )

    X, feature_names = build_features(
        zlogits,
        base_logits,
    )

    print("X shape:", X.shape)

    if X.shape[1] != 258:
        raise ValueError(
            f"Expected 258 features, "
            f"got {X.shape[1]}"
        )

    pd.Series(
        feature_names
    ).to_csv(
        OUT_DIR / "feature_names.csv",
        index=False,
        header=False,
    )


    true_file4_mask = np.isin(
        y_true,
        FILE4_IDX,
    )

    base_pred_file4 = np.isin(
        base_pred_idx,
        FILE4_IDX,
    )

    top2_idx = np.argsort(
        base_logits,
        axis=1,
    )[:, -2:]

    top2_has_file4 = np.isin(
        top2_idx,
        FILE4_IDX,
    ).any(axis=1)

    route_masks = {
        "base_pred_file4":
            base_pred_file4,

        "base_top2_has_file4":
            top2_has_file4,
    }


    y_file4 = np.array(
        [
            file4_label2id[
                id2label[int(i)]
            ]
            if i in FILE4_IDX
            else -1

            for i in y_true
        ],
        dtype=int,
    )


    print(
        "true_file4 n:",
        int(true_file4_mask.sum()),
    )

    print(
        "base_pred_file4 n:",
        int(base_pred_file4.sum()),
    )


    # ========================================================
    # STEP 1: meta model OOF probabilities
    # ========================================================

    fold_model_root = (
        OUT_DIR / "fold_models"
    )

    fold_model_root.mkdir(
        exist_ok=True
    )

    meta_probs_by_kind = {}


    for kind in MODEL_KINDS:
        print()
        print("=" * 80)

        print(
            f"META MODEL KIND: {kind}"
        )

        print("=" * 80)

        meta_probs = np.zeros(
            (
                len(y_true),
                len(FILE4),
            ),
            dtype=np.float32,
        )

        kind_dir = (
            fold_model_root / kind
        )

        kind_dir.mkdir(
            exist_ok=True
        )


        for fold in sorted(
            np.unique(folds)
        ):
            val_mask = (
                folds == fold
            )

            train_mask = (
                folds != fold
            )

            spec_train_mask = (
                train_mask
                & true_file4_mask
            )

            X_train = X[
                spec_train_mask
            ]

            y_train = y_file4[
                spec_train_mask
            ]

            X_val = X[
                val_mask
            ]

            val_idx = np.where(
                val_mask
            )[0]

            print(
                f"fold {fold}: "
                f"train_file4={len(y_train)} "
                f"val={len(val_idx)}"
            )

            model = make_model(kind)

            model.fit(
                X_train,
                y_train,
            )

            probs = model.predict_proba(
                X_val
            )

            classes = list(
                model.classes_
            )

            idx = [
                classes.index(i)
                for i in range(
                    len(FILE4)
                )
            ]

            probs = probs[:, idx]

            meta_probs[val_idx] = probs

            joblib.dump(
                model,
                kind_dir
                / f"{kind}_fold{fold}.joblib",
            )


        meta_probs_by_kind[kind] = (
            meta_probs
        )

        meta_pred = (
            meta_probs.argmax(axis=1)
        )

        spec_f1 = f1_score(
            y_file4[true_file4_mask],
            meta_pred[true_file4_mask],
            average="macro",
        )

        spec_acc = accuracy_score(
            y_file4[true_file4_mask],
            meta_pred[true_file4_mask],
        )

        print(
            f"{kind} standalone "
            f"true-file4 macro_f1="
            f"{spec_f1:.10f} "
            f"acc={spec_acc:.10f}"
        )

        np.save(
            OUT_DIR
            / f"{kind}_meta_probs.npy",
            meta_probs,
        )


    # ========================================================
    # STEP 2: cross-fitted routing hyperparameter selection
    # ========================================================

    crossfit_pred = np.full(
        len(y_true),
        -1,
        dtype=int,
    )

    crossfit_replace = np.zeros(
        len(y_true),
        dtype=bool,
    )

    selected_rows = []

    search_rows = []


    for outer_fold in sorted(
        np.unique(folds)
    ):
        print()
        print("=" * 80)

        print(
            f"ROUTING CROSSFIT FOLD "
            f"{outer_fold}"
        )

        print("=" * 80)

        tune_mask = (
            folds != outer_fold
        )

        val_mask = (
            folds == outer_fold
        )

        best = None


        for kind in MODEL_KINDS:
            meta_probs = (
                meta_probs_by_kind[kind]
            )

            for mode in MODES:
                route_base_mask = (
                    route_masks[mode]
                )

                for th in THRESHOLDS:
                    for m in MARGINS:
                        pred, replace = (
                            make_route_prediction(
                                base_pred_idx,
                                meta_probs,
                                route_base_mask,
                                float(th),
                                m,
                            )
                        )

                        tune_f1, tune_acc = (
                            eval_idx(
                                y_true[tune_mask],
                                pred[tune_mask],
                            )
                        )

                        row = {
                            "outer_fold":
                                int(outer_fold),

                            "kind":
                                kind,

                            "mode":
                                mode,

                            "threshold":
                                float(th),

                            "margin":
                                None
                                if m is None
                                else float(m),

                            "tune_macro_f1":
                                float(tune_f1),

                            "tune_accuracy":
                                float(tune_acc),

                            "tune_replace_n":
                                int(
                                    replace[
                                        tune_mask
                                    ].sum()
                                ),
                        }

                        search_rows.append(row)

                        if (
                            best is None
                            or tune_f1
                            > best["tune_macro_f1"]
                            or (
                                tune_f1
                                == best["tune_macro_f1"]
                                and tune_acc
                                > best["tune_accuracy"]
                            )
                        ):
                            best = row.copy()


        print(
            "selected routing params:"
        )

        print(
            json.dumps(
                best,
                indent=2,
                ensure_ascii=False,
            )
        )


        best_probs = (
            meta_probs_by_kind[
                best["kind"]
            ]
        )

        best_pred, best_replace = (
            make_route_prediction(
                base_pred_idx,
                best_probs,
                route_masks[
                    best["mode"]
                ],
                best["threshold"],
                best["margin"],
            )
        )


        crossfit_pred[val_mask] = (
            best_pred[val_mask]
        )

        crossfit_replace[val_mask] = (
            best_replace[val_mask]
        )


        val_f1, val_acc = eval_idx(
            y_true[val_mask],
            crossfit_pred[val_mask],
        )


        selected = {
            **best,

            "val_n":
                int(val_mask.sum()),

            "val_replace_n":
                int(
                    crossfit_replace[
                        val_mask
                    ].sum()
                ),

            "val_replace_rate":
                float(
                    crossfit_replace[
                        val_mask
                    ].mean()
                ),

            "val_macro_f1":
                float(val_f1),

            "val_accuracy":
                float(val_acc),
        }

        selected_rows.append(
            selected
        )


        print(
            f"heldout val_macro_f1="
            f"{val_f1:.10f}"
        )

        print(
            f"heldout val_accuracy="
            f"{val_acc:.10f}"
        )

        print(
            f"heldout replace_n="
            f"{selected['val_replace_n']}"
        )


    if np.any(
        crossfit_pred < 0
    ):
        raise RuntimeError(
            "crossfit predictions incomplete"
        )


    # ========================================================
    # Final cross-fit result
    # ========================================================

    crossfit_f1, crossfit_acc = eval_idx(
        y_true,
        crossfit_pred,
    )


    print()
    print("=" * 80)

    print(
        "FINAL ALL5 META-SPECIALIST "
        "CROSSFIT RESULT"
    )

    print("=" * 80)

    print(
        f"base_macro_f1     = "
        f"{base_f1:.10f}"
    )

    print(
        f"meta_crossfit_f1  = "
        f"{crossfit_f1:.10f}"
    )

    print(
        f"meta_crossfit_acc = "
        f"{crossfit_acc:.10f}"
    )

    print(
        f"gain_vs_base      = "
        f"{crossfit_f1 - base_f1:+.10f}"
    )

    print(
        f"replace_n         = "
        f"{crossfit_replace.sum()}"
    )

    print(
        f"replace_rate      = "
        f"{crossfit_replace.mean():.10f}"
    )


    print()
    print("CLASSIFICATION REPORT")
    print()

    print(
        classification_report(
            y_true,
            crossfit_pred,
            target_names=ALL_CLASSES,
            digits=4,
        )
    )


    # ========================================================
    # Correctness transitions
    # ========================================================

    before_correct = (
        base_pred_idx == y_true
    )

    after_correct = (
        crossfit_pred == y_true
    )


    print()
    print(
        "===== CORRECTNESS CHANGE ====="
    )

    print(
        pd.crosstab(
            before_correct,
            after_correct,
            rownames=["base_correct"],
            colnames=["final_correct"],
        )
    )


    # ========================================================
    # Save
    # ========================================================

    np.save(
        OUT_DIR
        / "crossfit_final_pred_idx.npy",
        crossfit_pred,
    )

    np.save(
        OUT_DIR
        / "crossfit_replace_mask.npy",
        crossfit_replace,
    )


    pd.DataFrame(
        selected_rows
    ).to_csv(
        OUT_DIR
        / "crossfit_selected_routing.csv",
        index=False,
    )


    pd.DataFrame(
        search_rows
    ).to_csv(
        OUT_DIR
        / "crossfit_routing_search.csv",
        index=False,
    )


    pred_df = ref_df.copy()

    pred_df["base_pred"] = [
        id2label[int(i)]
        for i in base_pred_idx
    ]

    pred_df["final_pred"] = [
        id2label[int(i)]
        for i in crossfit_pred
    ]

    pred_df["replaced"] = (
        crossfit_replace
    )


    pred_df.to_csv(
        OUT_DIR
        / "crossfit_meta_predictions.csv",
        index=False,
    )


    result = {
        "base_weights": WEIGHTS,

        "base_macro_f1":
            float(base_f1),

        "base_accuracy":
            float(base_acc),

        "meta_crossfit_macro_f1":
            float(crossfit_f1),

        "meta_crossfit_accuracy":
            float(crossfit_acc),

        "gain_vs_base":
            float(
                crossfit_f1
                - base_f1
            ),

        "replace_n":
            int(
                crossfit_replace.sum()
            ),

        "replace_rate":
            float(
                crossfit_replace.mean()
            ),

        "routing_selection":
            (
                "For each outer fold, "
                "kind/mode/threshold/margin "
                "are selected on the other "
                "four folds' OOF meta predictions "
                "and applied only to the held-out fold."
            ),

        "selected_by_fold":
            selected_rows,

        "file4":
            FILE4,
    }


    with open(
        OUT_DIR
        / "crossfit_meta_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )


    print()
    print("Saved:", OUT_DIR)


if __name__ == "__main__":
    main()
