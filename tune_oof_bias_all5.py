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

OUT_DIR = BASE / "oof_ensemble_bias_all5"
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


PREV_BIAS_PATH = BASE / "oof_ensemble_bias_3model" / "best_bias.npy"


def row_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)

    return (x - mean) / np.maximum(std, 1e-6)


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

    non_id = [c for c in df.columns if c != "id"]

    valid = []
    class_set = set(ALL_CLASSES)

    for col in non_id:
        vals = set(df[col].dropna().astype(str).unique())
        if vals and vals.issubset(class_set):
            valid.append(col)

    if len(valid) == 1:
        return valid[0]

    raise ValueError(
        f"Could not detect label column. columns={df.columns.tolist()}, "
        f"possible label cols={valid}"
    )


def load_model_oof(
    name: str,
    model_dir: Path,
    ref_ids: list[str] | None,
) -> tuple[list[str], np.ndarray]:
    pred_path = model_dir / "val_predictions.csv"
    logit_path = model_dir / "val_logits.npy"

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    if not logit_path.exists():
        raise FileNotFoundError(logit_path)

    pred_df = pd.read_csv(pred_path)
    ids = pred_df["id"].astype(str).tolist()

    logits = np.load(logit_path).astype(np.float32)

    if logits.shape != (len(ids), len(ALL_CLASSES)):
        raise ValueError(
            f"{name}: bad logits shape={logits.shape}, "
            f"expected=({len(ids)}, {len(ALL_CLASSES)})"
        )

    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate IDs detected")

    if ref_ids is None:
        print(f"{name:9s} loaded {logits.shape}")
        return ids, logits

    if ids == ref_ids:
        print(f"{name:9s} loaded {logits.shape}")
        return ids, logits

    if set(ids) != set(ref_ids):
        missing = len(set(ref_ids) - set(ids))
        extra = len(set(ids) - set(ref_ids))

        raise ValueError(
            f"{name}: ID set mismatch, missing={missing}, extra={extra}"
        )

    print(f"[reorder] {name}")

    id_to_pos = {sample_id: i for i, sample_id in enumerate(ids)}
    order = np.asarray([id_to_pos[x] for x in ref_ids], dtype=np.int64)

    logits = logits[order]

    print(f"{name:9s} loaded {logits.shape}")

    return ref_ids, logits


def load_targets(ref_ids: list[str]) -> np.ndarray:
    labels_path = ROOT / "data" / "train_labels.csv"

    df = pd.read_csv(labels_path)
    df["id"] = df["id"].astype(str)

    label_col = detect_label_column(df)

    print("label column:", label_col)

    label_map = dict(
        zip(
            df["id"].astype(str),
            df[label_col].astype(str),
        )
    )

    missing = [x for x in ref_ids if x not in label_map]

    if missing:
        raise ValueError(
            f"Missing labels for {len(missing)} IDs. "
            f"examples={missing[:5]}"
        )

    labels = [label_map[x] for x in ref_ids]

    unknown = sorted(set(labels) - set(ALL_CLASSES))

    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    return np.asarray([CLASS_TO_IDX[x] for x in labels], dtype=np.int64)


def score_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    bias: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    pred = np.argmax(logits + bias[None, :], axis=1)

    macro_f1 = f1_score(
        y_true,
        pred,
        average="macro",
        labels=np.arange(len(ALL_CLASSES)),
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, pred)

    return float(macro_f1), float(accuracy), pred


def center_bias(bias: np.ndarray) -> np.ndarray:
    """
    Adding the same constant to every class does not change argmax.
    Remove that redundant degree of freedom.
    """
    bias = np.asarray(bias, dtype=np.float32).copy()
    bias -= bias.mean()
    return bias


# ============================================================
# Load OOF logits
# ============================================================

all_logits: dict[str, np.ndarray] = {}

ref_ids = None

for name, model_dir in MODEL_DIRS.items():
    ids, logits = load_model_oof(name, model_dir, ref_ids)

    if ref_ids is None:
        ref_ids = ids

    all_logits[name] = logits


assert ref_ids is not None

y_true = load_targets(ref_ids)

print()
print("n samples:", len(ref_ids))
print("target shape:", y_true.shape)


# ============================================================
# Build fixed-weight all5 ensemble logits
# ============================================================

ensemble_logits = np.zeros(
    (len(ref_ids), len(ALL_CLASSES)),
    dtype=np.float32,
)

for name, weight in WEIGHTS.items():
    ensemble_logits += weight * row_zscore(all_logits[name])

print()
print("all5 ensemble logits:", ensemble_logits.shape)
print("weights:", WEIGHTS)


# ============================================================
# Baselines
# ============================================================

zero_bias = np.zeros(len(ALL_CLASSES), dtype=np.float32)

zero_f1, zero_acc, _ = score_logits(
    ensemble_logits,
    y_true,
    zero_bias,
)

print()
print("=" * 80)
print("BASELINE: ALL5 NO BIAS")
print("=" * 80)
print(f"macro_f1={zero_f1:.10f}")
print(f"accuracy={zero_acc:.10f}")


if PREV_BIAS_PATH.exists():
    prev_bias_raw = np.load(PREV_BIAS_PATH).astype(np.float32)
else:
    raise FileNotFoundError(PREV_BIAS_PATH)

if prev_bias_raw.shape != (len(ALL_CLASSES),):
    raise ValueError(f"bad previous bias shape={prev_bias_raw.shape}")

prev_bias = center_bias(prev_bias_raw)

prev_f1, prev_acc, _ = score_logits(
    ensemble_logits,
    y_true,
    prev_bias,
)

print()
print("=" * 80)
print("BASELINE: ALL5 + PREVIOUS 3MODEL BIAS")
print("=" * 80)
print(f"macro_f1={prev_f1:.10f}")
print(f"accuracy={prev_acc:.10f}")

print()
print("previous bias, centered:")

for cls, b in zip(ALL_CLASSES, prev_bias):
    print(f"{cls:20s} {b:+.6f}")


# ============================================================
# Coordinate bias search
# ============================================================

best_bias = prev_bias.copy()
best_f1 = prev_f1
best_acc = prev_acc

history: list[dict] = []

SEARCH_STAGES = [
    # Re-adapt old bias to the new ensemble calibration.
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


print()
print("=" * 80)
print("START ALL5 BIAS SEARCH")
print("=" * 80)


for stage_cfg in SEARCH_STAGES:
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
    print("-" * 80)
    print(
        f"STAGE={stage} step={step} radius={radius} "
        f"max_passes={max_passes}"
    )
    print("-" * 80)

    for pass_idx in range(max_passes):
        pass_start_f1 = best_f1
        improved_classes = []

        for class_idx, class_name in enumerate(ALL_CLASSES):
            local_best_bias = best_bias.copy()
            local_best_f1 = best_f1
            local_best_acc = best_acc
            local_best_delta = 0.0

            for delta in deltas:
                candidate = best_bias.copy()
                candidate[class_idx] += float(delta)
                candidate = center_bias(candidate)

                macro_f1, accuracy, _ = score_logits(
                    ensemble_logits,
                    y_true,
                    candidate,
                )

                history.append(
                    {
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
                        abs(macro_f1 - local_best_f1) <= 1e-12
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

                improved_classes.append(
                    (
                        class_name,
                        local_best_delta,
                        old_f1,
                        best_f1,
                    )
                )

                print(
                    f"[improve] {class_name:20s} "
                    f"delta={local_best_delta:+.4f} "
                    f"{old_f1:.10f} -> {best_f1:.10f}"
                )

        gain_this_pass = best_f1 - pass_start_f1

        print(
            f"[{stage}] pass={pass_idx} "
            f"macro_f1={best_f1:.10f} "
            f"acc={best_acc:.10f} "
            f"pass_gain={gain_this_pass:+.10f} "
            f"improved_classes={len(improved_classes)}"
        )

        if best_f1 <= pass_start_f1 + 1e-12:
            print(f"[{stage}] converged")
            break


# ============================================================
# Final evaluation
# ============================================================

best_f1, best_acc, best_pred = score_logits(
    ensemble_logits,
    y_true,
    best_bias,
)

print()
print("=" * 80)
print("FINAL ALL5 BIAS RESULT")
print("=" * 80)

print(f"no_bias_macro_f1  = {zero_f1:.10f}")
print(f"old_bias_macro_f1 = {prev_f1:.10f}")
print(f"best_macro_f1     = {best_f1:.10f}")
print(f"accuracy          = {best_acc:.10f}")

print(f"gain_vs_no_bias   = {best_f1 - zero_f1:+.10f}")
print(f"gain_vs_old_bias  = {best_f1 - prev_f1:+.10f}")

print()
print("BEST BIAS:")

for cls, bias in zip(ALL_CLASSES, best_bias):
    print(f"{cls:20s} {bias:+.6f}")

print()
print(
    classification_report(
        y_true,
        best_pred,
        labels=np.arange(len(ALL_CLASSES)),
        target_names=ALL_CLASSES,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# Save outputs
# ============================================================

np.save(
    OUT_DIR / "all5_oof_logits.npy",
    ensemble_logits,
)

np.save(
    OUT_DIR / "best_bias.npy",
    best_bias,
)

np.save(
    OUT_DIR / "biased_oof_logits.npy",
    ensemble_logits + best_bias[None, :],
)

np.save(
    OUT_DIR / "best_pred_idx.npy",
    best_pred,
)


pred_df = pd.DataFrame(
    {
        "id": ref_ids,
        "label": [ALL_CLASSES[i] for i in best_pred],
    }
)

pred_df.to_csv(
    OUT_DIR / "biased_oof_predictions.csv",
    index=False,
)


history_df = pd.DataFrame(history)

history_df.to_csv(
    OUT_DIR / "bias_search_history.csv",
    index=False,
)


result = {
    "models": list(WEIGHTS.keys()),
    "weights": WEIGHTS,
    "normalization": "row_zscore",
    "initial_bias": "prev_3model_bias_centered",
    "no_bias_macro_f1": zero_f1,
    "no_bias_accuracy": zero_acc,
    "old_bias_macro_f1": prev_f1,
    "old_bias_accuracy": prev_acc,
    "best_macro_f1": best_f1,
    "best_accuracy": best_acc,
    "gain_vs_no_bias": best_f1 - zero_f1,
    "gain_vs_old_bias": best_f1 - prev_f1,
    "best_bias": {
        cls: float(bias)
        for cls, bias in zip(ALL_CLASSES, best_bias)
    },
}


with open(
    OUT_DIR / "bias_result.json",
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