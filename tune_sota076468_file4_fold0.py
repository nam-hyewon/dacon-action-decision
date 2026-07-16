import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report


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

LABEL2ID = {c: i for i, c in enumerate(ALL_CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(ALL_CLASSES)}
FILE4_IDX = np.array([LABEL2ID[c] for c in FILE4], dtype=int)

ROOT = Path(".")
SOTA_DIR = ROOT / "large + tab + diet + rob8 (0.76468)"

MODEL_DIRS = {
    "large": SOTA_DIR / "klue_roberta_large_eda_state_fold0",
    "tab": SOTA_DIR / "tabular_histgb_fold0",
    "diet": SOTA_DIR / "kluebert_diet_fold0",
    "rob8": SOTA_DIR / "roberta_eda_state8_lr15_fold0",
}

REF_PRED_PATH = MODEL_DIRS["diet"] / "val_predictions.csv"

SPEC_DIR = (
    ROOT
    / "submit"
    / "model"
    / "deberta_file4_eda_state_fold0_seed43_allval"
)

OUT_DIR = SOTA_DIR / "sota_plus_file4_fold0"
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


def margin_from_logits(x):
    s = np.sort(x, axis=1)
    return s[:, -1] - s[:, -2]


def eval_logits(logits, y_true):
    pred = logits.argmax(axis=1)
    f1 = f1_score(y_true, pred, average="macro")
    acc = accuracy_score(y_true, pred)
    return float(f1), float(acc), pred


def get_class_order(model_dir):
    info_path = model_dir / "val_info.json"

    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        classes = info.get("all_classes")

        if isinstance(classes, list) and len(classes) == 14:
            return [str(x) for x in classes]

    labels_path = model_dir / "labels.json"

    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            obj = json.load(f)

        # {"label2id": {...}}
        if isinstance(obj, dict) and "label2id" in obj:
            m = obj["label2id"]
            return [
                k for k, v in sorted(
                    m.items(),
                    key=lambda kv: int(kv[1]),
                )
            ]

        # direct label -> id
        if (
            isinstance(obj, dict)
            and len(obj) == 14
            and all(str(k) in ALL_CLASSES for k in obj)
        ):
            return [
                k for k, v in sorted(
                    obj.items(),
                    key=lambda kv: int(kv[1]),
                )
            ]

    print(
        f"[WARN] Cannot infer class order for {model_dir}; "
        "assuming canonical ALL_CLASSES order"
    )
    return ALL_CLASSES


def load_14class_logits(name, model_dir):
    p = model_dir / "val_logits.npy"

    if not p.exists():
        raise FileNotFoundError(p)

    x = np.load(p).astype(np.float32)

    if x.shape != (13947, 14):
        raise ValueError(
            f"{name}: unexpected shape {x.shape}"
        )

    class_order = get_class_order(model_dir)

    print(f"{name:6s} class order:", class_order)

    if set(class_order) != set(ALL_CLASSES):
        raise ValueError(
            f"{name}: class set mismatch"
        )

    reorder = [
        class_order.index(c)
        for c in ALL_CLASSES
    ]

    x = x[:, reorder]

    print(
        f"{name:6s} loaded:",
        x.shape,
        "from",
        model_dir,
    )

    return x


def align_specialist(ref_ids):
    logits = np.load(
        SPEC_DIR / "val_logits.npy"
    ).astype(np.float32)

    df = pd.read_csv(
        SPEC_DIR / "val_predictions.csv"
    )

    df["id"] = df["id"].astype(str)
    ids = df["id"].tolist()

    if logits.shape != (13947, 4):
        raise ValueError(
            f"specialist shape={logits.shape}"
        )

    if ids != ref_ids:
        if set(ids) != set(ref_ids):
            raise ValueError(
                "specialist ID set mismatch: "
                f"missing={len(set(ref_ids) - set(ids))}, "
                f"extra={len(set(ids) - set(ref_ids))}"
            )

        print("[reorder] FILE4 specialist")

        pos = {
            sid: i
            for i, sid in enumerate(ids)
        }

        order = [
            pos[sid]
            for sid in ref_ids
        ]

        logits = logits[order]
        df = df.iloc[order].reset_index(drop=True)

    assert df["id"].tolist() == ref_ids

    return logits, df


def generate_coarse_weights(step=0.05):
    n = int(round(1.0 / step))

    weights = []

    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d = n - a - b - c

                weights.append(
                    np.array(
                        [a, b, c, d],
                        dtype=np.float32,
                    ) / n
                )

    return weights


def main():
    # ============================================================
    # Reference IDs / labels
    # ============================================================

    ref_df = pd.read_csv(REF_PRED_PATH)
    ref_df["id"] = ref_df["id"].astype(str)

    ref_ids = ref_df["id"].tolist()

    if "true_action" in ref_df.columns:
        true_action = ref_df["true_action"].astype(str)
    elif "label" in ref_df.columns:
        true_action = ref_df["label"].astype(str)
    else:
        raise ValueError(
            f"Cannot find label column: {ref_df.columns.tolist()}"
        )

    y_true = np.array(
        [LABEL2ID[x] for x in true_action],
        dtype=int,
    )

    print("reference n:", len(ref_ids))

    # ============================================================
    # Load SOTA component logits
    # ============================================================

    model_names = [
        "large",
        "tab",
        "diet",
        "rob8",
    ]

    raw = {
        name: load_14class_logits(
            name,
            MODEL_DIRS[name],
        )
        for name in model_names
    }

    z = {
        name: row_zscore(raw[name])
        for name in model_names
    }

    stack = np.stack(
        [z[name] for name in model_names],
        axis=0,
    )

    print("stack:", stack.shape)

    print("\n===== SINGLE MODEL SCORES =====")

    for name in model_names:
        f1, acc, _ = eval_logits(
            z[name],
            y_true,
        )

        print(
            f"{name:6s} "
            f"macro_f1={f1:.10f} "
            f"acc={acc:.10f}"
        )

    # ============================================================
    # Reconstruct SOTA weights
    # coarse 0.05 simplex search
    # ============================================================

    print("\n===== COARSE WEIGHT SEARCH =====")

    coarse_results = []

    for w in generate_coarse_weights(0.05):
        logits = np.tensordot(
            w,
            stack,
            axes=(0, 0),
        )

        f1, acc, _ = eval_logits(
            logits,
            y_true,
        )

        coarse_results.append(
            (
                f1,
                acc,
                w.copy(),
            )
        )

    coarse_results.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )

    print("top coarse:")

    for f1, acc, w in coarse_results[:10]:
        print(
            f"f1={f1:.10f} "
            f"acc={acc:.10f} "
            f"weights={dict(zip(model_names, w.tolist()))}"
        )

    # ============================================================
    # Local coordinate hill climbing
    # move 0.01 weight from one model to another
    # ============================================================

    best_f1, best_acc, best_w = coarse_results[0]

    print("\n===== FINE HILL CLIMB =====")

    step = 0.01
    improved = True
    iteration = 0

    while improved:
        improved = False
        iteration += 1

        current_w = best_w.copy()

        candidates = []

        for i in range(4):
            for j in range(4):
                if i == j:
                    continue

                for amount in [
                    0.01,
                    0.02,
                    0.03,
                    0.04,
                    0.05,
                ]:
                    if current_w[i] + 1e-8 < amount:
                        continue

                    w = current_w.copy()
                    w[i] -= amount
                    w[j] += amount

                    candidates.append(w)

        for w in candidates:
            logits = np.tensordot(
                w,
                stack,
                axes=(0, 0),
            )

            f1, acc, _ = eval_logits(
                logits,
                y_true,
            )

            if (
                f1 > best_f1
                or (
                    f1 == best_f1
                    and acc > best_acc
                )
            ):
                best_f1 = f1
                best_acc = acc
                best_w = w.copy()
                improved = True

        print(
            f"iter={iteration} "
            f"f1={best_f1:.10f} "
            f"weights={dict(zip(model_names, best_w.tolist()))}"
        )

        if iteration >= 30:
            break

    base = np.tensordot(
        best_w,
        stack,
        axes=(0, 0),
    )

    base_f1, base_acc, base_pred = eval_logits(
        base,
        y_true,
    )

    print("\n===== RECONSTRUCTED SOTA BASE =====")

    print(
        json.dumps(
            {
                "macro_f1": base_f1,
                "accuracy": base_acc,
                "weights": {
                    name: float(best_w[i])
                    for i, name in enumerate(model_names)
                },
            },
            indent=2,
        )
    )

    target_score = 0.76468

    print(
        "distance from reported 0.76468:",
        base_f1 - target_score,
    )

    if base_f1 < 0.760:
        raise RuntimeError(
            "Failed to reproduce the reported SOTA closely enough. "
            "Likely row-order or ensemble-calibration mismatch. "
            f"Recovered macro-F1={base_f1:.10f}"
        )

    # ============================================================
    # FILE4 specialist
    # ============================================================

    spec_raw, spec_df = align_specialist(
        ref_ids
    )

    spec = row_zscore(spec_raw)

    true_file4_mask = np.isin(
        y_true,
        FILE4_IDX,
    )

    spec_pred_global = FILE4_IDX[
        spec.argmax(axis=1)
    ]

    spec_f1 = f1_score(
        y_true[true_file4_mask],
        spec_pred_global[true_file4_mask],
        labels=FILE4_IDX,
        average="macro",
    )

    spec_acc = accuracy_score(
        y_true[true_file4_mask],
        spec_pred_global[true_file4_mask],
    )

    print("\n===== FILE4 SPECIALIST =====")
    print("true FILE4 n:", int(true_file4_mask.sum()))
    print("macro_f1:", spec_f1)
    print("accuracy:", spec_acc)

    # ============================================================
    # Gate features
    # ============================================================

    base_prob = softmax(base)
    base_conf = base_prob.max(axis=1)
    base_margin = margin_from_logits(base)
    base_pred = base.argmax(axis=1)

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
    spec_margin = margin_from_logits(spec)

    print("\n===== GATE STATS =====")
    print(
        "base_pred_file4:",
        int(base_pred_file4.sum()),
    )
    print(
        "top2_has_file4:",
        int(top2_has_file4.sum()),
    )

    # Precompute normalized base FILE4 logits
    base_file4_z = row_zscore(
        base[:, FILE4_IDX]
    )

    # ============================================================
    # Hybrid search
    # ============================================================

    base_conf_qs = [
        0.30, 0.40, 0.50, 0.60,
        0.70, 0.80, 0.90, 1.00,
    ]

    spec_conf_thresholds = [
        0.00, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90,
    ]

    spec_margin_thresholds = [
        -1.0, 0.0, 0.1, 0.2,
        0.3, 0.5, 0.75, 1.0,
    ]

    alphas = [
        0.10, 0.20, 0.30, 0.40,
        0.50, 0.65, 0.80, 1.00,
    ]

    gate_masks = {
        "base_pred_file4": base_pred_file4,
        "base_top2_has_file4": top2_has_file4,
    }

    rows = []

    best = {
        "macro_f1": base_f1,
        "accuracy": base_acc,
        "gain": 0.0,
        "mode": "base_only",
        "gate": None,
        "_logits": base.copy(),
        "_pred": base_pred.copy(),
    }

    def register(
        logits,
        mode,
        gate_name,
        conf_q,
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
            "macro_f1": f1,
            "accuracy": acc,
            "gain": f1 - base_f1,
            "mode": mode,
            "gate": gate_name,
            "base_conf_q": float(conf_q),
            "spec_conf_thr": float(spec_conf_thr),
            "spec_margin_thr": float(spec_margin_thr),
            "alpha": float(alpha),
            "replace_n": int(use_mask.sum()),
            "replace_rate": float(use_mask.mean()),
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

    for gate_name, initial_gate in gate_masks.items():

        for conf_q in base_conf_qs:

            conf_thr = np.quantile(
                base_conf,
                conf_q,
            )

            gate = (
                initial_gate
                & (base_conf <= conf_thr)
            )

            for spec_conf_thr in spec_conf_thresholds:

                for spec_margin_thr in spec_margin_thresholds:

                    use_mask = (
                        gate
                        & (spec_conf >= spec_conf_thr)
                        & (spec_margin >= spec_margin_thr)
                    )

                    if not use_mask.any():
                        continue

                    use_rows = np.where(
                        use_mask
                    )[0]

                    # Hard replacement
                    hard = base.copy()

                    hard[
                        np.ix_(
                            use_rows,
                            FILE4_IDX,
                        )
                    ] = -10.0

                    hard[
                        use_rows,
                        spec_pred_global[use_mask],
                    ] = 10.0

                    register(
                        hard,
                        "hard_replace",
                        gate_name,
                        conf_q,
                        spec_conf_thr,
                        spec_margin_thr,
                        1.0,
                        use_mask,
                    )

                    # FILE4-only logit blend
                    for alpha in alphas:

                        mixed = base.copy()

                        mixed[
                            np.ix_(
                                use_rows,
                                FILE4_IDX,
                            )
                        ] = (
                            (1.0 - alpha)
                            * base_file4_z[use_mask]
                            + alpha
                            * spec[use_mask]
                        )

                        register(
                            mixed,
                            "file4_logit_blend",
                            gate_name,
                            conf_q,
                            spec_conf_thr,
                            spec_margin_thr,
                            alpha,
                            use_mask,
                        )

    # ============================================================
    # Save results
    # ============================================================

    results_df = (
        pd.DataFrame(rows)
        .sort_values(
            ["macro_f1", "accuracy"],
            ascending=False,
        )
    )

    results_df.to_csv(
        OUT_DIR / "search_results.csv",
        index=False,
    )

    print("\n===== TOP 30 =====")
    print(
        results_df.head(30).to_string(
            index=False
        )
    )

    printable_best = {
        k: v
        for k, v in best.items()
        if not k.startswith("_")
    }

    print("\n===== BEST =====")
    print(
        json.dumps(
            printable_best,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n===== BASE VS BEST =====")
    print(f"base macro_f1 : {base_f1:.10f}")
    print(
        f"best macro_f1 : "
        f"{best['macro_f1']:.10f}"
    )
    print(
        f"gain          : "
        f"{best['macro_f1'] - base_f1:+.10f}"
    )

    np.save(
        OUT_DIR / "reconstructed_sota_logits.npy",
        base.astype(np.float32),
    )

    np.save(
        OUT_DIR / "best_sota_file4_logits.npy",
        best["_logits"].astype(np.float32),
    )

    out_df = ref_df.copy()

    out_df["base_pred_action"] = [
        ID2LABEL[int(i)]
        for i in base_pred
    ]

    out_df["hybrid_pred_action"] = [
        ID2LABEL[int(i)]
        for i in best["_pred"]
    ]

    out_df["base_correct"] = (
        base_pred == y_true
    )

    out_df["hybrid_correct"] = (
        best["_pred"] == y_true
    )

    out_df.to_csv(
        OUT_DIR / "best_predictions.csv",
        index=False,
    )

    result = {
        "reconstructed_base_macro_f1": base_f1,
        "reconstructed_base_accuracy": base_acc,
        "reported_local_score": 0.76468,
        "base_weights": {
            name: float(best_w[i])
            for i, name in enumerate(model_names)
        },
        "specialist_true_file4_macro_f1": float(spec_f1),
        "specialist_true_file4_accuracy": float(spec_acc),
        "best_hybrid": printable_best,
    }

    with open(
        OUT_DIR / "best_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
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
    print(OUT_DIR / "search_results.csv")
    print(OUT_DIR / "best_result.json")
    print(OUT_DIR / "reconstructed_sota_logits.npy")
    print(OUT_DIR / "best_sota_file4_logits.npy")
    print(OUT_DIR / "best_predictions.csv")


if __name__ == "__main__":
    main()
