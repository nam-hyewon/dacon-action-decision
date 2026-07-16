from __future__ import annotations

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


ROOT = Path(__file__).resolve().parent

BASE = (
    ROOT
    / "team_4model_ensemble_07626"
    / "ensemble_07626"
    / "fold0_team"
)

OUT_DIR = BASE / "oof_ensemble_bias_all5_crossfit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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

CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}

MODEL_DIRS = {
    "deberta": BASE / "deberta_base_eda_state4_seed43_oof",
    "kluebert": BASE / "kluebert_mid5_lr15_oof",
    "roberta": BASE / "roberta_eda_state4_oof",
    "tabular": BASE / "tabular_histgb_oof",
    "xlmr": BASE / "xlmr_base_eda_state_seed43_oof",
}

WEIGHTS = {
    "deberta": 0.17,
    "kluebert": 0.16,
    "roberta": 0.17,
    "tabular": 0.43,
    "xlmr": 0.07,
}

TABULAR_FOLDS_DIR = BASE / "tabular_folds"

PREV_BIAS_PATH = (
    BASE
    / "oof_ensemble_bias_3model"
    / "best_bias.npy"
)


def row_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)

    return (x - mean) / np.maximum(std, 1e-6)


def center_bias(bias: np.ndarray) -> np.ndarray:
    bias = np.asarray(bias, dtype=np.float32).copy()
    bias -= bias.mean()
    return bias


def detect_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "label",
        "target",
        "class",
        "action",
        "answer",
        "tool",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    class_set = set(ALL_CLASSES)
    valid = []

    for col in df.columns:
        if col == "id":
            continue

        vals = set(df[col].dropna().astype(str).unique())

        if vals and vals.issubset(class_set):
            valid.append(col)

    if len(valid) == 1:
        return valid[0]

    raise ValueError(
        f"Could not detect label column. "
        f"columns={df.columns.tolist()} valid={valid}"
    )


def load_model_oof(
    name: str,
    model_dir: Path,
    ref_ids: list[str] | None,
) -> tuple[list[str], np.ndarray]:
    pred_path = model_dir / "val_predictions.csv"
    logit_path = model_dir / "val_logits.npy"

    pred_df = pd.read_csv(pred_path)

    ids = pred_df["id"].astype(str).tolist()
    logits = np.load(logit_path).astype(np.float32)

    if logits.shape != (len(ids), len(ALL_CLASSES)):
        raise ValueError(
            f"{name}: bad logits shape {logits.shape}"
        )

    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate IDs")

    if ref_ids is None:
        print(f"{name:9s} loaded {logits.shape}")
        return ids, logits

    if ids == ref_ids:
        print(f"{name:9s} loaded {logits.shape}")
        return ids, logits

    if set(ids) != set(ref_ids):
        raise ValueError(f"{name}: ID set mismatch")

    print(f"[reorder] {name}")

    id_to_pos = {
        sample_id: i
        for i, sample_id in enumerate(ids)
    }

    order = np.asarray(
        [id_to_pos[x] for x in ref_ids],
        dtype=np.int64,
    )

    logits = logits[order]

    print(f"{name:9s} loaded {logits.shape}")

    return ref_ids, logits


def load_targets(ref_ids: list[str]) -> np.ndarray:
    path = ROOT / "data" / "train_labels.csv"

    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)

    label_col = detect_label_column(df)

    print("label column:", label_col)

    label_map = dict(
        zip(
            df["id"],
            df[label_col].astype(str),
        )
    )

    missing = [
        sample_id
        for sample_id in ref_ids
        if sample_id not in label_map
    ]

    if missing:
        raise ValueError(
            f"missing target IDs: {len(missing)}"
        )

    labels = [label_map[x] for x in ref_ids]

    return np.asarray(
        [CLASS_TO_IDX[x] for x in labels],
        dtype=np.int64,
    )


def load_fold_ids() -> dict[int, set[str]]:
    fold_ids = {}

    for fold in range(5):
        path = (
            TABULAR_FOLDS_DIR
            / f"tabular_histgb_fold{fold}"
            / "val_predictions.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        ids = set(df["id"].astype(str))

        fold_ids[fold] = ids

        print(
            f"fold{fold}: n={len(ids)}"
        )

    all_fold_ids = set().union(
        *fold_ids.values()
    )

    total_count = sum(
        len(x)
        for x in fold_ids.values()
    )

    if len(all_fold_ids) != total_count:
        raise ValueError(
            "fold validation ID overlap detected"
        )

    return fold_ids


def build_fold_assignment(
    ref_ids: list[str],
    fold_ids: dict[int, set[str]],
) -> np.ndarray:
    id_to_fold = {}

    for fold, ids in fold_ids.items():
        for sample_id in ids:
            if sample_id in id_to_fold:
                raise ValueError(
                    f"duplicate fold assignment: {sample_id}"
                )

            id_to_fold[sample_id] = fold

    missing = [
        sample_id
        for sample_id in ref_ids
        if sample_id not in id_to_fold
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} OOF IDs have no fold assignment. "
            f"examples={missing[:5]}"
        )

    assignment = np.asarray(
        [id_to_fold[x] for x in ref_ids],
        dtype=np.int64,
    )

    return assignment


def score_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    bias: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    pred = np.argmax(
        logits + bias[None, :],
        axis=1,
    )

    macro_f1 = f1_score(
        y_true,
        pred,
        average="macro",
        labels=np.arange(len(ALL_CLASSES)),
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        pred,
    )

    return (
        float(macro_f1),
        float(accuracy),
        pred,
    )


def tune_bias(
    logits: np.ndarray,
    y_true: np.ndarray,
    initial_bias: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, float, float, list[dict]]:
    best_bias = center_bias(initial_bias)

    best_f1, best_acc, _ = score_logits(
        logits,
        y_true,
        best_bias,
    )

    history = []

    search_stages = [
        {
            "name": "coarse",
            "step": 0.04,
            "radius": 0.12,
            "max_passes": 5,
        },
        {
            "name": "medium",
            "step": 0.02,
            "radius": 0.06,
            "max_passes": 6,
        },
        {
            "name": "fine",
            "step": 0.01,
            "radius": 0.03,
            "max_passes": 8,
        },
        {
            "name": "ultrafine",
            "step": 0.005,
            "radius": 0.015,
            "max_passes": 8,
        },
    ]

    for stage_cfg in search_stages:
        stage = stage_cfg["name"]
        step = float(stage_cfg["step"])
        radius = float(stage_cfg["radius"])
        max_passes = int(stage_cfg["max_passes"])

        deltas = np.arange(
            -radius,
            radius + step * 0.5,
            step,
            dtype=np.float32,
        )

        print()
        print(
            f"[fold{fold}] stage={stage} "
            f"step={step} radius={radius}"
        )

        for pass_idx in range(max_passes):
            pass_start_f1 = best_f1

            for class_idx, class_name in enumerate(
                ALL_CLASSES
            ):
                local_best_bias = best_bias.copy()
                local_best_f1 = best_f1
                local_best_acc = best_acc
                local_best_delta = 0.0

                for delta in deltas:
                    candidate = best_bias.copy()

                    candidate[class_idx] += float(delta)
                    candidate = center_bias(candidate)

                    macro_f1, accuracy, _ = score_logits(
                        logits,
                        y_true,
                        candidate,
                    )

                    history.append(
                        {
                            "fold": fold,
                            "stage": stage,
                            "pass_idx": pass_idx,
                            "class_idx": class_idx,
                            "class_name": class_name,
                            "delta": float(delta),
                            "macro_f1": macro_f1,
                            "accuracy": accuracy,
                        }
                    )

                    if (
                        macro_f1 > local_best_f1 + 1e-12
                        or (
                            abs(
                                macro_f1 - local_best_f1
                            ) <= 1e-12
                            and accuracy > local_best_acc
                        )
                    ):
                        local_best_bias = candidate
                        local_best_f1 = macro_f1
                        local_best_acc = accuracy
                        local_best_delta = float(delta)

                if local_best_f1 > best_f1 + 1e-12:
                    old_f1 = best_f1

                    best_bias = local_best_bias
                    best_f1 = local_best_f1
                    best_acc = local_best_acc

                    print(
                        f"[fold{fold} improve] "
                        f"{class_name:20s} "
                        f"delta={local_best_delta:+.4f} "
                        f"{old_f1:.8f} -> {best_f1:.8f}"
                    )

            pass_gain = best_f1 - pass_start_f1

            print(
                f"[fold{fold}] "
                f"{stage} pass={pass_idx} "
                f"f1={best_f1:.10f} "
                f"gain={pass_gain:+.10f}"
            )

            if best_f1 <= pass_start_f1 + 1e-12:
                print(
                    f"[fold{fold}] {stage} converged"
                )
                break

    return (
        best_bias,
        best_f1,
        best_acc,
        history,
    )


# ============================================================
# Load all5 OOF
# ============================================================

model_logits = {}
ref_ids = None

for name, model_dir in MODEL_DIRS.items():
    ids, logits = load_model_oof(
        name,
        model_dir,
        ref_ids,
    )

    if ref_ids is None:
        ref_ids = ids

    model_logits[name] = logits


assert ref_ids is not None

y_true = load_targets(ref_ids)

fold_ids = load_fold_ids()

fold_assignment = build_fold_assignment(
    ref_ids,
    fold_ids,
)


print()
print("fold assignment counts:")

for fold in range(5):
    print(
        f"fold{fold}: "
        f"{int((fold_assignment == fold).sum())}"
    )


# ============================================================
# Build fixed all5 ensemble logits
# ============================================================

ensemble_logits = np.zeros(
    (len(ref_ids), len(ALL_CLASSES)),
    dtype=np.float32,
)

for name, weight in WEIGHTS.items():
    ensemble_logits += (
        weight
        * row_zscore(model_logits[name])
    )


# ============================================================
# Baseline
# ============================================================

zero_bias = np.zeros(
    len(ALL_CLASSES),
    dtype=np.float32,
)

baseline_f1, baseline_acc, baseline_pred = score_logits(
    ensemble_logits,
    y_true,
    zero_bias,
)

print()
print("=" * 80)
print("ALL5 NO-BIAS BASELINE")
print("=" * 80)

print(f"macro_f1={baseline_f1:.10f}")
print(f"accuracy={baseline_acc:.10f}")


# ============================================================
# Initial bias
# ============================================================

if PREV_BIAS_PATH.exists():
    initial_bias = np.load(
        PREV_BIAS_PATH
    ).astype(np.float32)

    initial_bias = center_bias(initial_bias)

    print(
        "initial bias: previous 3model bias"
    )
else:
    initial_bias = zero_bias.copy()

    print("initial bias: zeros")


# ============================================================
# 5-fold cross-fit bias
# ============================================================

crossfit_pred = np.full(
    len(ref_ids),
    -1,
    dtype=np.int64,
)

crossfit_logits = np.zeros_like(
    ensemble_logits,
    dtype=np.float32,
)

fold_biases = np.zeros(
    (5, len(ALL_CLASSES)),
    dtype=np.float32,
)

fold_results = []
all_history = []


for fold in range(5):
    print()
    print("=" * 80)
    print(f"CROSSFIT FOLD {fold}")
    print("=" * 80)

    train_mask = fold_assignment != fold
    val_mask = fold_assignment == fold

    train_logits = ensemble_logits[train_mask]
    train_y = y_true[train_mask]

    val_logits = ensemble_logits[val_mask]
    val_y = y_true[val_mask]

    print(
        f"train_n={train_mask.sum()} "
        f"val_n={val_mask.sum()}"
    )

    best_bias, train_f1, train_acc, history = tune_bias(
        train_logits,
        train_y,
        initial_bias,
        fold,
    )

    fold_biases[fold] = best_bias
    all_history.extend(history)

    val_f1, val_acc, val_pred = score_logits(
        val_logits,
        val_y,
        best_bias,
    )

    crossfit_pred[val_mask] = val_pred

    crossfit_logits[val_mask] = (
        val_logits + best_bias[None, :]
    )

    fold_result = {
        "fold": fold,
        "train_n": int(train_mask.sum()),
        "val_n": int(val_mask.sum()),
        "train_macro_f1": train_f1,
        "train_accuracy": train_acc,
        "val_macro_f1": val_f1,
        "val_accuracy": val_acc,
    }

    fold_results.append(fold_result)

    print()
    print(f"FOLD {fold} RESULT")
    print(
        f"train_macro_f1={train_f1:.10f}"
    )
    print(
        f"val_macro_f1  ={val_f1:.10f}"
    )
    print(
        f"val_accuracy  ={val_acc:.10f}"
    )

    print("bias:")

    for class_name, bias in zip(
        ALL_CLASSES,
        best_bias,
    ):
        print(
            f"  {class_name:20s} {bias:+.6f}"
        )


if np.any(crossfit_pred < 0):
    raise RuntimeError(
        "some crossfit predictions were not filled"
    )


# ============================================================
# Overall cross-fit score
# ============================================================

crossfit_f1 = f1_score(
    y_true,
    crossfit_pred,
    average="macro",
    labels=np.arange(len(ALL_CLASSES)),
    zero_division=0,
)

crossfit_acc = accuracy_score(
    y_true,
    crossfit_pred,
)


print()
print("=" * 80)
print("FINAL 5-FOLD CROSSFIT BIAS RESULT")
print("=" * 80)

print(
    f"no_bias_macro_f1       = "
    f"{baseline_f1:.10f}"
)

print(
    f"crossfit_bias_macro_f1  = "
    f"{crossfit_f1:.10f}"
)

print(
    f"crossfit_bias_accuracy  = "
    f"{crossfit_acc:.10f}"
)

print(
    f"gain_vs_no_bias         = "
    f"{crossfit_f1 - baseline_f1:+.10f}"
)


print()
print("CLASSIFICATION REPORT")
print()

print(
    classification_report(
        y_true,
        crossfit_pred,
        labels=np.arange(len(ALL_CLASSES)),
        target_names=ALL_CLASSES,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# Bias stability
# ============================================================

bias_mean = fold_biases.mean(axis=0)
bias_std = fold_biases.std(axis=0)

print()
print("=" * 80)
print("BIAS STABILITY ACROSS FOLDS")
print("=" * 80)

for idx, class_name in enumerate(ALL_CLASSES):
    values = fold_biases[:, idx]

    values_str = " ".join(
        f"{x:+.4f}"
        for x in values
    )

    print(
        f"{class_name:20s} "
        f"mean={bias_mean[idx]:+.5f} "
        f"std={bias_std[idx]:.5f} "
        f"| {values_str}"
    )


# ============================================================
# Save
# ============================================================

np.save(
    OUT_DIR / "fold_biases.npy",
    fold_biases,
)

np.save(
    OUT_DIR / "mean_fold_bias.npy",
    bias_mean,
)

np.save(
    OUT_DIR / "std_fold_bias.npy",
    bias_std,
)

np.save(
    OUT_DIR / "crossfit_biased_oof_logits.npy",
    crossfit_logits,
)

np.save(
    OUT_DIR / "crossfit_pred_idx.npy",
    crossfit_pred,
)


pd.DataFrame(fold_results).to_csv(
    OUT_DIR / "fold_results.csv",
    index=False,
)


pd.DataFrame(all_history).to_csv(
    OUT_DIR / "bias_search_history.csv",
    index=False,
)


pred_df = pd.DataFrame(
    {
        "id": ref_ids,
        "label": [
            ALL_CLASSES[i]
            for i in crossfit_pred
        ],
        "fold": fold_assignment,
    }
)

pred_df.to_csv(
    OUT_DIR / "crossfit_oof_predictions.csv",
    index=False,
)


bias_df = pd.DataFrame(
    {
        "class": ALL_CLASSES,
        **{
            f"fold{fold}": fold_biases[fold]
            for fold in range(5)
        },
        "mean": bias_mean,
        "std": bias_std,
    }
)

bias_df.to_csv(
    OUT_DIR / "fold_bias_stability.csv",
    index=False,
)


result = {
    "models": list(WEIGHTS.keys()),
    "weights": WEIGHTS,
    "normalization": "row_zscore",
    "bias_cv": "5-fold cross-fitted using tabular validation fold IDs",
    "initial_bias": "previous_3model_bias_centered",
    "no_bias_macro_f1": float(baseline_f1),
    "no_bias_accuracy": float(baseline_acc),
    "crossfit_bias_macro_f1": float(crossfit_f1),
    "crossfit_bias_accuracy": float(crossfit_acc),
    "gain_vs_no_bias": float(
        crossfit_f1 - baseline_f1
    ),
    "fold_results": fold_results,
    "bias_mean": {
        class_name: float(value)
        for class_name, value in zip(
            ALL_CLASSES,
            bias_mean,
        )
    },
    "bias_std": {
        class_name: float(value)
        for class_name, value in zip(
            ALL_CLASSES,
            bias_std,
        )
    },
}


with open(
    OUT_DIR / "crossfit_bias_result.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("Saved:", OUT_DIR)

print()
print("Saved:", OUT_DIR)
