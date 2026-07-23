#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "submit" / "model"

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

RECIPE = {
    "large": {
        "path": "klue_roberta_large_eda_state_fold0",
        "weight": 0.24114308710234722,
        "temperature": 1.0,
    },
    "tab": {
        "path": "tabular_histgb_fold0",
        "weight": 0.4438507372276068,
        "temperature": 1.0,
    },
    "diet": {
        "path": "kluebert_diet_fold0",
        "weight": 0.21685297669057865,
        "temperature": 1.0,
    },
    "rob8": {
        "path": "roberta_eda_state8_lr15_fold0",
        "weight": 0.09815319897946724,
        "temperature": 0.85,
    },
}

CLASS_BIAS = {
    "read_file": 0.0,
    "grep_search": -0.3000011444091797,
    "list_directory": 0.39999818801879883,
    "glob_pattern": 0.6499979496002197,
    "edit_file": -0.3000011444091797,
    "write_file": 0.0,
    "apply_patch": 0.0,
    "run_bash": -0.000001430511474609375,
    "run_tests": -0.2500011920928955,
    "lint_or_typecheck": -0.35000109672546387,
    "ask_user": -0.2500011920928955,
    "plan_task": -0.10000133514404297,
    "web_search": 0.4999980926513672,
    "respond_only": 1.3499972820281982,
}


def load_component(name: str, spec: dict):
    path = MODEL_DIR / spec["path"]
    logits_path = path / "val_logits.npy"
    pred_path = path / "val_predictions.csv"
    if not logits_path.exists():
        raise FileNotFoundError(logits_path)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    logits = np.load(logits_path).astype("float32")
    pred = pd.read_csv(pred_path)
    if logits.shape[1] != len(CLASSES):
        raise ValueError(f"{name}: expected 14 logits, got {logits.shape}")
    return logits, pred


def main():
    ref_logits, ref_pred = load_component("large", RECIPE["large"])
    ids = ref_pred["id"].astype(str).to_numpy()
    y = ref_pred["true_id"].to_numpy().astype(int)
    scores = np.zeros_like(ref_logits, dtype="float32")

    component_scores = {}
    for name, spec in RECIPE.items():
        logits, pred = load_component(name, spec)
        cur_ids = pred["id"].astype(str).to_numpy()
        if not np.array_equal(ids, cur_ids):
            raise ValueError(f"{name}: val_predictions id order differs from large")
        pred_id = logits.argmax(axis=1)
        component_scores[name] = {
            "macro_f1": float(f1_score(y, pred_id, average="macro")),
            "accuracy": float(accuracy_score(y, pred_id)),
        }
        scores += float(spec["weight"]) * (logits / float(spec["temperature"]))

    pred_before_bias = scores.argmax(axis=1)
    before = {
        "macro_f1": float(f1_score(y, pred_before_bias, average="macro")),
        "accuracy": float(accuracy_score(y, pred_before_bias)),
    }

    bias_vec = np.array([CLASS_BIAS[c] for c in CLASSES], dtype="float32")
    scores_with_bias = scores + bias_vec
    pred_after_bias = scores_with_bias.argmax(axis=1)
    after = {
        "macro_f1": float(f1_score(y, pred_after_bias, average="macro")),
        "accuracy": float(accuracy_score(y, pred_after_bias)),
    }

    out = {
        "recipe": "large_tab_diet_rob8_gated40_fold0_eval",
        "n": int(len(y)),
        "classes": CLASSES,
        "components": component_scores,
        "before_class_bias": before,
        "after_class_bias": after,
        "weights": {k: v["weight"] for k, v in RECIPE.items()},
        "temperatures": {k: v["temperature"] for k, v in RECIPE.items()},
        "class_bias": CLASS_BIAS,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
