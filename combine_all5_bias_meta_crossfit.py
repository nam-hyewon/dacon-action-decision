from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


ALL_CLASSES = [
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

label2id = {
    c: i
    for i, c in enumerate(ALL_CLASSES)
}

id2label = {
    i: c
    for i, c in enumerate(ALL_CLASSES)
}


FILE4 = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]

FILE4_IDX = np.asarray(
    [
        label2id[c]
        for c in FILE4
    ],
    dtype=np.int64,
)


ROOT = Path(".")

BASE_DIR = (
    ROOT
    / "team_4model_ensemble_07626"
    / "ensemble_07626"
    / "fold0_team"
)


OOF_PATHS = {
    "deberta":
        BASE_DIR
        / "deberta_base_eda_state4_seed43_oof",

    "kluebert":
        BASE_DIR
        / "kluebert_mid5_lr15_oof",

    "roberta":
        BASE_DIR
        / "roberta_eda_state4_oof",

    "tabular":
        BASE_DIR
        / "tabular_histgb_oof",

    "xlmr":
        BASE_DIR
        / "xlmr_base_eda_state_seed43_oof",
}


WEIGHTS = {
    "deberta": 0.17,
    "kluebert": 0.16,
    "roberta": 0.17,
    "tabular": 0.43,
    "xlmr": 0.07,
}


BIAS_DIR = (
    BASE_DIR
    / "oof_ensemble_bias_all5_crossfit"
)

META_DIR = (
    BASE_DIR
    / "file4_logit_meta_specialist_all5_crossfit"
)

OUT_DIR = (
    BASE_DIR
    / "all5_bias_meta_crossfit"
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


def row_zscore(
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    return (
        x
        - x.mean(
            axis=1,
            keepdims=True,
        )
    ) / (
        x.std(
            axis=1,
            keepdims=True,
        )
        + eps
    )


def find_label_column(
    df: pd.DataFrame,
) -> str:
    for col in [
        "label",
        "true_action",
        "true_label",
        "action",
        "target",
    ]:
        if col in df.columns:
            return col

    raise ValueError(
        "cannot find label column: "
        f"{df.columns.tolist()}"
    )


def load_one(
    name: str,
    path: Path,
    ref_ids: list[str] | None,
):
    logits = np.load(
        path / "val_logits.npy"
    ).astype(np.float32)

    df = pd.read_csv(
        path / "val_predictions.csv"
    )

    df["id"] = (
        df["id"]
        .astype(str)
    )

    label_col = find_label_column(df)

    if label_col != "label":
        df["label"] = (
            df[label_col]
            .astype(str)
        )

    if "fold" not in df.columns:
        raise ValueError(
            f"{name}: fold column missing"
        )

    ids = (
        df["id"]
        .astype(str)
        .tolist()
    )

    if logits.shape != (
        len(ids),
        len(ALL_CLASSES),
    ):
        raise ValueError(
            f"{name}: bad logits shape "
            f"{logits.shape}"
        )

    if ref_ids is None:
        print(
            f"{name:9s} loaded "
            f"{logits.shape}"
        )

        return (
            ids,
            logits,
            df,
        )


    if ids != ref_ids:
        if set(ids) != set(ref_ids):
            raise ValueError(
                f"{name}: ID set mismatch"
            )

        print(
            f"[reorder] {name}"
        )

        id_to_pos = {
            sample_id: i
            for i, sample_id
            in enumerate(ids)
        }

        order = np.asarray(
            [
                id_to_pos[x]
                for x in ref_ids
            ],
            dtype=np.int64,
        )

        logits = logits[order]

        df = (
            df.iloc[order]
            .reset_index(drop=True)
        )


    print(
        f"{name:9s} loaded "
        f"{logits.shape}"
    )

    return (
        ref_ids,
        logits,
        df,
    )


def load_all():
    ref_ids = None
    ref_df = None

    zlogits = {}

    for (
        name,
        path,
    ) in OOF_PATHS.items():

        (
            ids,
            logits,
            df,
        ) = load_one(
            name,
            path,
            ref_ids,
        )

        if ref_ids is None:
            ref_ids = ids
            ref_df = df.copy()

        zlogits[name] = (
            row_zscore(logits)
            .astype(np.float32)
        )


    y_true = np.asarray(
        [
            label2id[x]
            for x in (
                ref_df["label"]
                .astype(str)
                .tolist()
            )
        ],
        dtype=np.int64,
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


def build_base_logits(
    zlogits: dict[str, np.ndarray],
) -> np.ndarray:
    base_logits = np.zeros_like(
        next(
            iter(zlogits.values())
        ),
        dtype=np.float32,
    )

    for (
        name,
        weight,
    ) in WEIGHTS.items():

        base_logits += (
            weight
            * zlogits[name]
        )


    return base_logits


def eval_pred(
    y_true: np.ndarray,
    pred: np.ndarray,
):
    macro_f1 = f1_score(
        y_true,
        pred,
        average="macro",
        labels=np.arange(
            len(ALL_CLASSES)
        ),
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        pred,
    )

    return (
        float(macro_f1),
        float(accuracy),
    )


def make_route_masks(
    biased_logits: np.ndarray,
):
    biased_pred = (
        biased_logits.argmax(axis=1)
    )

    base_pred_file4 = np.isin(
        biased_pred,
        FILE4_IDX,
    )


    top2_idx = np.argsort(
        biased_logits,
        axis=1,
    )[:, -2:]


    top2_has_file4 = np.isin(
        top2_idx,
        FILE4_IDX,
    ).any(axis=1)


    return (
        biased_pred,
        {
            "base_pred_file4":
                base_pred_file4,

            "base_top2_has_file4":
                top2_has_file4,
        },
    )


def apply_meta_route(
    biased_pred: np.ndarray,
    meta_probs: np.ndarray,
    route_mask: np.ndarray,
    threshold: float,
    margin: float | None,
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
            len(biased_pred),
            dtype=bool,
        )

    else:
        margin_mask = (
            meta_margin >= margin
        )


    replace_mask = (
        route_mask
        & conf_mask
        & margin_mask
    )


    final_pred = (
        biased_pred.copy()
    )


    final_pred[replace_mask] = (
        FILE4_IDX[
            meta_pred_file4[
                replace_mask
            ]
        ]
    )


    return (
        final_pred,
        replace_mask,
    )


def main():
    print(
        "===== LOAD ALL5 OOF ====="
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


    base_pred = (
        base_logits.argmax(axis=1)
    )


    base_f1, base_acc = eval_pred(
        y_true,
        base_pred,
    )


    print()
    print(
        "===== BASE ALL5 ====="
    )

    print(
        f"macro_f1="
        f"{base_f1:.10f}"
    )

    print(
        f"accuracy="
        f"{base_acc:.10f}"
    )


    # ========================================================
    # Load cross-fit fold biases
    # ========================================================

    fold_biases_path = (
        BIAS_DIR
        / "fold_biases.npy"
    )


    if not fold_biases_path.exists():
        raise FileNotFoundError(
            fold_biases_path
        )


    fold_biases = np.load(
        fold_biases_path
    ).astype(np.float32)


    if fold_biases.shape != (
        5,
        len(ALL_CLASSES),
    ):
        raise ValueError(
            f"bad fold_biases shape="
            f"{fold_biases.shape}"
        )


    print()
    print(
        "fold_biases shape:",
        fold_biases.shape,
    )


    # ========================================================
    # Load leak-free OOF meta probabilities
    # ========================================================

    meta_probs_by_kind = {}


    for kind in MODEL_KINDS:
        path = (
            META_DIR
            / f"{kind}_meta_probs.npy"
        )


        if not path.exists():
            raise FileNotFoundError(
                path
            )


        probs = np.load(
            path
        ).astype(np.float32)


        if probs.shape != (
            len(y_true),
            len(FILE4),
        ):
            raise ValueError(
                f"{kind}: bad meta probs "
                f"shape={probs.shape}"
            )


        meta_probs_by_kind[kind] = (
            probs
        )


        print(
            f"{kind:10s} meta probs "
            f"{probs.shape}"
        )


    # ========================================================
    # Bias-only cross-fit reference
    # ========================================================

    bias_only_pred = np.full(
        len(y_true),
        -1,
        dtype=np.int64,
    )


    for fold in sorted(
        np.unique(folds)
    ):
        mask = (
            folds == fold
        )


        biased_logits = (
            base_logits[mask]
            + fold_biases[
                int(fold)
            ][None, :]
        )


        bias_only_pred[mask] = (
            biased_logits.argmax(axis=1)
        )


    bias_only_f1, bias_only_acc = (
        eval_pred(
            y_true,
            bias_only_pred,
        )
    )


    print()
    print(
        "===== BIAS-ONLY REFERENCE ====="
    )

    print(
        f"macro_f1="
        f"{bias_only_f1:.10f}"
    )

    print(
        f"accuracy="
        f"{bias_only_acc:.10f}"
    )


    expected_bias_f1 = 0.7573311042

    if abs(
        bias_only_f1
        - expected_bias_f1
    ) > 1e-7:
        raise ValueError(
            "Bias-only score does not reproduce "
            f"expected result. got={bias_only_f1}, "
            f"expected={expected_bias_f1}"
        )


    # ========================================================
    # Outer-fold cross-fit:
    # bias -> route search -> held-out apply
    # ========================================================

    crossfit_pred = np.full(
        len(y_true),
        -1,
        dtype=np.int64,
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
        outer_fold = int(
            outer_fold
        )


        print()
        print("=" * 80)

        print(
            f"BIAS + META CROSSFIT "
            f"FOLD {outer_fold}"
        )

        print("=" * 80)


        tune_mask = (
            folds != outer_fold
        )


        val_mask = (
            folds == outer_fold
        )


        # IMPORTANT:
        # bias[outer_fold] was tuned only using
        # the other four folds.
        fold_bias = (
            fold_biases[outer_fold]
        )


        biased_logits_all = (
            base_logits
            + fold_bias[None, :]
        )


        (
            biased_pred_all,
            route_masks,
        ) = make_route_masks(
            biased_logits_all
        )


        tune_bias_f1, tune_bias_acc = (
            eval_pred(
                y_true[tune_mask],
                biased_pred_all[tune_mask],
            )
        )


        val_bias_f1, val_bias_acc = (
            eval_pred(
                y_true[val_mask],
                biased_pred_all[val_mask],
            )
        )


        print(
            f"bias-only tune_f1="
            f"{tune_bias_f1:.10f}"
        )

        print(
            f"bias-only heldout_f1="
            f"{val_bias_f1:.10f}"
        )


        best = None


        for kind in MODEL_KINDS:
            meta_probs = (
                meta_probs_by_kind[kind]
            )


            for mode in MODES:
                route_mask = (
                    route_masks[mode]
                )


                for threshold in THRESHOLDS:
                    for margin in MARGINS:

                        (
                            candidate_pred,
                            replace_mask,
                        ) = apply_meta_route(
                            biased_pred=
                                biased_pred_all,

                            meta_probs=
                                meta_probs,

                            route_mask=
                                route_mask,

                            threshold=
                                float(threshold),

                            margin=
                                margin,
                        )


                        tune_f1, tune_acc = (
                            eval_pred(
                                y_true[tune_mask],
                                candidate_pred[
                                    tune_mask
                                ],
                            )
                        )


                        item = {
                            "outer_fold":
                                outer_fold,

                            "kind":
                                kind,

                            "mode":
                                mode,

                            "threshold":
                                float(threshold),

                            "margin":
                                (
                                    None
                                    if margin is None
                                    else float(margin)
                                ),

                            "tune_macro_f1":
                                float(tune_f1),

                            "tune_accuracy":
                                float(tune_acc),

                            "tune_gain_vs_bias":
                                float(
                                    tune_f1
                                    - tune_bias_f1
                                ),

                            "tune_replace_n":
                                int(
                                    replace_mask[
                                        tune_mask
                                    ].sum()
                                ),

                            "tune_replace_rate":
                                float(
                                    replace_mask[
                                        tune_mask
                                    ].mean()
                                ),
                        }


                        search_rows.append(
                            item
                        )


                        if (
                            best is None
                            or item[
                                "tune_macro_f1"
                            ]
                            > best[
                                "tune_macro_f1"
                            ]
                            or (
                                item[
                                    "tune_macro_f1"
                                ]
                                == best[
                                    "tune_macro_f1"
                                ]
                                and item[
                                    "tune_accuracy"
                                ]
                                > best[
                                    "tune_accuracy"
                                ]
                            )
                        ):
                            best = item.copy()


        print()
        print(
            "SELECTED ROUTING PARAMS"
        )

        print(
            json.dumps(
                best,
                indent=2,
                ensure_ascii=False,
            )
        )


        (
            final_pred_all,
            final_replace_all,
        ) = apply_meta_route(
            biased_pred=
                biased_pred_all,

            meta_probs=
                meta_probs_by_kind[
                    best["kind"]
                ],

            route_mask=
                route_masks[
                    best["mode"]
                ],

            threshold=
                best["threshold"],

            margin=
                best["margin"],
        )


        crossfit_pred[val_mask] = (
            final_pred_all[val_mask]
        )


        crossfit_replace[val_mask] = (
            final_replace_all[val_mask]
        )


        val_f1, val_acc = eval_pred(
            y_true[val_mask],
            crossfit_pred[val_mask],
        )


        selected = {
            **best,

            "bias_only_val_macro_f1":
                float(val_bias_f1),

            "bias_only_val_accuracy":
                float(val_bias_acc),

            "val_macro_f1":
                float(val_f1),

            "val_accuracy":
                float(val_acc),

            "val_gain_vs_bias":
                float(
                    val_f1
                    - val_bias_f1
                ),

            "val_n":
                int(
                    val_mask.sum()
                ),

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
        }


        selected_rows.append(
            selected
        )


        print()
        print(
            f"heldout bias-only f1="
            f"{val_bias_f1:.10f}"
        )

        print(
            f"heldout bias+meta f1="
            f"{val_f1:.10f}"
        )

        print(
            f"heldout gain vs bias="
            f"{val_f1 - val_bias_f1:+.10f}"
        )

        print(
            f"heldout replace_n="
            f"{selected['val_replace_n']}"
        )


    if np.any(
        crossfit_pred < 0
    ):
        raise RuntimeError(
            "crossfit prediction incomplete"
        )


    # ========================================================
    # Final result
    # ========================================================

    combo_f1, combo_acc = eval_pred(
        y_true,
        crossfit_pred,
    )


    print()
    print("=" * 80)

    print(
        "FINAL ALL5 BIAS + META "
        "CROSSFIT RESULT"
    )

    print("=" * 80)


    print(
        f"all5_base_f1       = "
        f"{base_f1:.10f}"
    )

    print(
        f"bias_only_f1       = "
        f"{bias_only_f1:.10f}"
    )

    print(
        f"bias_meta_f1       = "
        f"{combo_f1:.10f}"
    )

    print(
        f"bias_meta_accuracy = "
        f"{combo_acc:.10f}"
    )

    print(
        f"gain_vs_base       = "
        f"{combo_f1 - base_f1:+.10f}"
    )

    print(
        f"gain_vs_bias       = "
        f"{combo_f1 - bias_only_f1:+.10f}"
    )

    print(
        f"replace_n          = "
        f"{crossfit_replace.sum()}"
    )

    print(
        f"replace_rate       = "
        f"{crossfit_replace.mean():.10f}"
    )


    print()
    print(
        "CLASSIFICATION REPORT"
    )

    print()


    print(
        classification_report(
            y_true,
            crossfit_pred,
            labels=np.arange(
                len(ALL_CLASSES)
            ),
            target_names=ALL_CLASSES,
            digits=4,
            zero_division=0,
        )
    )


    # ========================================================
    # Correctness transitions vs bias
    # ========================================================

    before_correct = (
        bias_only_pred == y_true
    )


    after_correct = (
        crossfit_pred == y_true
    )


    print()
    print(
        "===== CORRECTNESS CHANGE "
        "VS BIAS-ONLY ====="
    )


    print(
        pd.crosstab(
            before_correct,
            after_correct,
            rownames=[
                "bias_correct"
            ],
            colnames=[
                "final_correct"
            ],
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


    np.save(
        OUT_DIR
        / "bias_only_pred_idx.npy",
        bias_only_pred,
    )


    pd.DataFrame(
        selected_rows
    ).to_csv(
        OUT_DIR
        / "selected_routing_by_fold.csv",
        index=False,
    )


    pd.DataFrame(
        search_rows
    ).to_csv(
        OUT_DIR
        / "routing_search_by_fold.csv",
        index=False,
    )


    pred_df = ref_df.copy()


    pred_df[
        "all5_base_pred"
    ] = [
        id2label[int(x)]
        for x in base_pred
    ]


    pred_df[
        "bias_only_pred"
    ] = [
        id2label[int(x)]
        for x in bias_only_pred
    ]


    pred_df[
        "bias_meta_pred"
    ] = [
        id2label[int(x)]
        for x in crossfit_pred
    ]


    pred_df[
        "meta_replaced"
    ] = crossfit_replace


    pred_df.to_csv(
        OUT_DIR
        / "crossfit_predictions.csv",
        index=False,
    )


    result = {
        "base_weights":
            WEIGHTS,

        "pipeline":
            (
                "all5 row-zscore ensemble "
                "-> outer-fold cross-fitted class bias "
                "-> routing params selected on other "
                "four folds using biased base "
                "-> held-out meta specialist routing"
            ),

        "all5_base_macro_f1":
            float(base_f1),

        "all5_base_accuracy":
            float(base_acc),

        "bias_only_macro_f1":
            float(bias_only_f1),

        "bias_only_accuracy":
            float(bias_only_acc),

        "bias_meta_macro_f1":
            float(combo_f1),

        "bias_meta_accuracy":
            float(combo_acc),

        "gain_vs_base":
            float(
                combo_f1 - base_f1
            ),

        "gain_vs_bias":
            float(
                combo_f1 - bias_only_f1
            ),

        "replace_n":
            int(
                crossfit_replace.sum()
            ),

        "replace_rate":
            float(
                crossfit_replace.mean()
            ),

        "selected_by_fold":
            selected_rows,
    }


    with open(
        OUT_DIR
        / "crossfit_bias_meta_result.json",
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
    print(
        "Saved:",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
