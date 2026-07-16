import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
)

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

FILE4 = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]

label2id = {c: i for i, c in enumerate(ALL_CLASSES)}
id2label = {i: c for i, c in enumerate(ALL_CLASSES)}

FILE4_IDX = np.array(
    [label2id[c] for c in FILE4],
    dtype=int,
)

ROOT = Path(".")

BASE_DIR = (
    ROOT
    / "team_4model_ensemble_07626"
    / "ensemble_07626"
    / "fold0_team"
)

PATHS = {
    "kluebert": (
        BASE_DIR
        / "kluebert_folds"
        / "kluebert_mid5_lr15_fold0"
    ),
    "tabular": (
        BASE_DIR
        / "tabular_folds"
        / "tabular_histgb_fold0"
    ),
}

SPEC_DIR = (
    ROOT
    / "submit"
    / "model"
    / "deberta_file4_eda_state_fold0_seed43_allval"
)

OUT_DIR = BASE_DIR / "deberta_file4_hybrid_v2_fold0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def row_zscore(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)

    return (
        x - x.mean(axis=1, keepdims=True)
    ) / (
        x.std(axis=1, keepdims=True) + eps
    )


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)

    return e / e.sum(axis=1, keepdims=True)


def get_margin(x):
    s = np.sort(x, axis=1)

    return s[:, -1] - s[:, -2]


def load_one(name, d):
    logits_path = d / "val_logits.npy"
    pred_path = d / "val_predictions.csv"

    if not logits_path.exists():
        raise FileNotFoundError(
            f"{name}: missing {logits_path}"
        )

    if not pred_path.exists():
        raise FileNotFoundError(
            f"{name}: missing {pred_path}"
        )

    logits = np.load(logits_path).astype(np.float32)

    df = pd.read_csv(pred_path)
    df["id"] = df["id"].astype(str)

    if "label" not in df.columns:
        if "true_action" in df.columns:
            df["label"] = df["true_action"].astype(str)
        elif "true_label" in df.columns:
            df["label"] = df["true_label"].astype(str)
        elif "action" in df.columns:
            df["label"] = df["action"].astype(str)
        elif "target" in df.columns:
            df["label"] = df["target"].astype(str)
        else:
            raise ValueError(
                f"{name}: cannot find label column "
                f"{df.columns.tolist()}"
            )

    if len(df) != logits.shape[0]:
        raise ValueError(
            f"{name}: df/logits mismatch "
            f"{len(df)} vs {logits.shape}"
        )

    return logits, df


def align_to_ref(logits, df, ref_ids, name):
    ids = df["id"].astype(str).tolist()

    if ids == ref_ids:
        return logits, df

    if set(ids) != set(ref_ids):
        raise ValueError(
            f"{name}: id set mismatch "
            f"missing={len(set(ref_ids) - set(ids))}, "
            f"extra={len(set(ids) - set(ref_ids))}"
        )

    print(
        f"[reorder] {name}: "
        "same id set but different order"
    )

    id_to_pos = {
        id_: i
        for i, id_ in enumerate(ids)
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
        df["id"].astype(str).tolist()
        == ref_ids
    )

    return logits, df


def eval_logits(logits, y_true):
    pred = logits.argmax(axis=1)

    f1 = f1_score(
        y_true,
        pred,
        average="macro",
    )

    acc = accuracy_score(
        y_true,
        pred,
    )

    return f1, acc, pred


def main():
    # =========================================================
    # 1. KLUEBERT + TABULAR load
    # =========================================================

    klue_raw, ref_df = load_one(
        "kluebert",
        PATHS["kluebert"],
    )

    ref_ids = (
        ref_df["id"]
        .astype(str)
        .tolist()
    )

    tab_raw, tab_df = load_one(
        "tabular",
        PATHS["tabular"],
    )

    tab_raw, tab_df = align_to_ref(
        tab_raw,
        tab_df,
        ref_ids,
        "tabular",
    )

    y_true = np.array(
        [
            label2id[x]
            for x in ref_df["label"]
            .astype(str)
            .tolist()
        ],
        dtype=int,
    )

    klue = row_zscore(klue_raw)
    tab = row_zscore(tab_raw)

    print("kluebert:", klue.shape)
    print("tabular  :", tab.shape)
    print("n val    :", len(y_true))

    # =========================================================
    # 2. Existing base reconstruction
    #    same logic as tune_oof_base_gated_deberta.py
    # =========================================================

    best_base = None

    for i in range(101):
        wk = i / 100
        wt = 1.0 - wk

        logits = (
            wk * klue
            + wt * tab
        )

        f1, acc, pred = eval_logits(
            logits,
            y_true,
        )

        if (
            best_base is None
            or f1 > best_base["macro_f1"]
        ):
            best_base = {
                "kluebert_weight": wk,
                "tabular_weight": wt,
                "macro_f1": float(f1),
                "accuracy": float(acc),
                "logits": logits.copy(),
                "pred": pred.copy(),
            }

    base = best_base["logits"]

    print("\n===== BASE =====")
    print(
        "kluebert weight:",
        best_base["kluebert_weight"],
    )
    print(
        "tabular weight :",
        best_base["tabular_weight"],
    )
    print(
        "macro_f1:",
        best_base["macro_f1"],
    )
    print(
        "accuracy:",
        best_base["accuracy"],
    )

    # =========================================================
    # 3. FILE4 specialist load + alignment
    # =========================================================

    spec_raw, spec_df = load_one(
        "file4_specialist",
        SPEC_DIR,
    )

    spec_raw, spec_df = align_to_ref(
        spec_raw,
        spec_df,
        ref_ids,
        "file4_specialist",
    )

    if spec_raw.shape != (
        len(y_true),
        len(FILE4),
    ):
        raise ValueError(
            "Unexpected specialist logits shape: "
            f"{spec_raw.shape}"
        )

    spec = row_zscore(spec_raw)

    print("\n===== SPECIALIST =====")
    print("specialist shape:", spec.shape)

    # True FILE4 subset diagnostic
    true_file4_mask = np.isin(
        y_true,
        FILE4_IDX,
    )

    spec_pred_local = spec.argmax(axis=1)

    spec_pred_global = FILE4_IDX[
        spec_pred_local
    ]

    spec_true_f1 = f1_score(
        y_true[true_file4_mask],
        spec_pred_global[true_file4_mask],
        labels=FILE4_IDX,
        average="macro",
    )

    spec_true_acc = accuracy_score(
        y_true[true_file4_mask],
        spec_pred_global[true_file4_mask],
    )

    print(
        "true FILE4 n:",
        int(true_file4_mask.sum()),
    )
    print(
        "specialist true-FILE4 macro_f1:",
        spec_true_f1,
    )
    print(
        "specialist true-FILE4 accuracy:",
        spec_true_acc,
    )

    # =========================================================
    # 4. Gate features
    # =========================================================

    base_prob = softmax(base)
    base_pred = base.argmax(axis=1)
    base_conf = base_prob.max(axis=1)
    base_margin = get_margin(base)

    top2_idx = np.argsort(
        base,
        axis=1,
    )[:, -2:]

    base_pred_file4 = np.isin(
        base_pred,
        FILE4_IDX,
    )

    top2_has_file4 = np.isin(
        top2_idx,
        FILE4_IDX,
    ).any(axis=1)

    spec_prob = softmax(spec)
    spec_conf = spec_prob.max(axis=1)
    spec_margin = get_margin(spec)

    print("\n===== GATE STATS =====")
    print(
        "base_pred_file4:",
        int(base_pred_file4.sum()),
    )
    print(
        "top2_has_file4:",
        int(top2_has_file4.sum()),
    )

    # =========================================================
    # 5. Search
    # =========================================================

    base_conf_qs = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ]

    spec_conf_thresholds = [
        0.00,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    spec_margin_thresholds = [
        -1.0,
        0.00,
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
    ]

    alphas = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.65,
        0.80,
        1.00,
    ]

    rows = []

    best = {
        "macro_f1": float(
            best_base["macro_f1"]
        ),
        "accuracy": float(
            best_base["accuracy"]
        ),
        "mode": "base_only",
        "_logits": base.copy(),
        "_pred": base_pred.copy(),
    }

    def register(
        logits,
        mode,
        gate,
        base_conf_q,
        spec_conf_thr,
        spec_margin_thr,
        alpha,
        use_mask,
    ):
        nonlocal best

        f1, acc, pred = eval_logits(
            logits,
            y_true,
        )

        item = {
            "macro_f1": float(f1),
            "accuracy": float(acc),
            "gain": float(
                f1 - best_base["macro_f1"]
            ),
            "mode": mode,
            "gate": gate,
            "base_conf_q": base_conf_q,
            "spec_conf_thr": spec_conf_thr,
            "spec_margin_thr": spec_margin_thr,
            "alpha": alpha,
            "replace_n": int(
                use_mask.sum()
            ),
            "replace_rate": float(
                use_mask.mean()
            ),
        }

        rows.append(item)

        if (
            f1 > best["macro_f1"]
            or (
                f1 == best["macro_f1"]
                and acc > best["accuracy"]
            )
        ):
            best = item.copy()
            best["_logits"] = logits.copy()
            best["_pred"] = pred.copy()

    gate_masks = {
        "base_pred_file4":
            base_pred_file4,

        "base_top2_has_file4":
            top2_has_file4,
    }

    for gate_name, initial_gate in gate_masks.items():

        for conf_q in base_conf_qs:

            conf_threshold = np.quantile(
                base_conf,
                conf_q,
            )

            # Low-confidence base only.
            base_uncertain = (
                base_conf <= conf_threshold
            )

            # q=1.0 effectively allows all gate rows.
            gate = (
                initial_gate
                & base_uncertain
            )

            for spec_conf_thr in spec_conf_thresholds:

                for spec_margin_thr in spec_margin_thresholds:

                    use_mask = (
                        gate
                        & (
                            spec_conf
                            >= spec_conf_thr
                        )
                        & (
                            spec_margin
                            >= spec_margin_thr
                        )
                    )

                    if use_mask.sum() == 0:
                        continue

                    # -----------------------------------------
                    # A. hard replace prediction
                    # -----------------------------------------

                    hard = base.copy()

                    # Force specialist-selected FILE4 class.
                    use_rows = np.where(use_mask)[0]

                    hard[
                        np.ix_(
                            use_rows,
                            FILE4_IDX,
                        )
                    ] = -10.0

                    hard[
                        np.where(use_mask)[0],
                        spec_pred_global[use_mask]
                    ] = 10.0

                    register(
                        hard,
                        mode="hard_replace",
                        gate=gate_name,
                        base_conf_q=float(conf_q),
                        spec_conf_thr=float(
                            spec_conf_thr
                        ),
                        spec_margin_thr=float(
                            spec_margin_thr
                        ),
                        alpha=1.0,
                        use_mask=use_mask,
                    )

                    # -----------------------------------------
                    # B. FILE4-logit blend
                    # -----------------------------------------

                    for alpha in alphas:

                        mixed = base.copy()

                        base_file4 = row_zscore(
                            base[:, FILE4_IDX]
                        )

                        mixed[
                            np.ix_(
                                use_mask,
                                FILE4_IDX,
                            )
                        ] = (
                            (1.0 - alpha)
                            * base_file4[use_mask]
                            + alpha
                            * spec[use_mask]
                        )

                        register(
                            mixed,
                            mode="file4_logit_blend",
                            gate=gate_name,
                            base_conf_q=float(conf_q),
                            spec_conf_thr=float(
                                spec_conf_thr
                            ),
                            spec_margin_thr=float(
                                spec_margin_thr
                            ),
                            alpha=float(alpha),
                            use_mask=use_mask,
                        )

    # =========================================================
    # 6. Save
    # =========================================================

    rows_df = pd.DataFrame(rows)

    rows_df = rows_df.sort_values(
        [
            "macro_f1",
            "accuracy",
        ],
        ascending=False,
    )

    rows_df.to_csv(
        OUT_DIR / "hybrid_v2_search_results.csv",
        index=False,
    )

    print("\n===== TOP 30 =====")

    print(
        rows_df.head(30).to_string(
            index=False
        )
    )

    print("\n===== BEST =====")

    printable_best = {
        k: v
        for k, v in best.items()
        if not k.startswith("_")
    }

    print(
        json.dumps(
            printable_best,
            indent=2,
            ensure_ascii=False,
        )
    )

    np.save(
        OUT_DIR / "best_hybrid_logits.npy",
        best["_logits"].astype(
            np.float32
        ),
    )

    pred_df = ref_df.copy()

    pred_df["hybrid_pred_id"] = (
        best["_pred"]
    )

    pred_df["hybrid_pred_action"] = [
        id2label[int(i)]
        for i in best["_pred"]
    ]

    pred_df["hybrid_correct"] = (
        best["_pred"] == y_true
    )

    pred_df.to_csv(
        OUT_DIR / "best_hybrid_predictions.csv",
        index=False,
    )

    with open(
        OUT_DIR / "best_hybrid_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                **printable_best,
                "base_macro_f1": float(
                    best_base["macro_f1"]
                ),
                "base_accuracy": float(
                    best_base["accuracy"]
                ),
                "base_weights": {
                    "kluebert": float(
                        best_base[
                            "kluebert_weight"
                        ]
                    ),
                    "tabular": float(
                        best_base[
                            "tabular_weight"
                        ]
                    ),
                },
                "specialist_true_file4_macro_f1":
                    float(spec_true_f1),
                "specialist_true_file4_accuracy":
                    float(spec_true_acc),
                "file4": FILE4,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_true,
            best["_pred"],
            target_names=ALL_CLASSES,
            digits=4,
        )
    )

    print("\n===== SAVED =====")
    print(
        OUT_DIR
        / "hybrid_v2_search_results.csv"
    )
    print(
        OUT_DIR
        / "best_hybrid_result.json"
    )
    print(
        OUT_DIR
        / "best_hybrid_logits.npy"
    )
    print(
        OUT_DIR
        / "best_hybrid_predictions.csv"
    )


if __name__ == "__main__":
    main()
