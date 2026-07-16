#!/usr/bin/env python3

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
C2I = {x: i for i, x in enumerate(CLASSES)}

ROOT = Path("large + tab + diet + rob8 (0.76468)")
PRED_CSV = ROOT / "kluebert_diet_fold0/val_predictions.csv"

MODEL_PATHS = {
    "large": ROOT / "klue_roberta_large_eda_state_fold0/val_logits.npy",
    "tab": ROOT / "tabular_histgb_fold0/val_logits.npy",
    "diet": ROOT / "kluebert_diet_fold0/val_logits.npy",
    "rob8": ROOT / "roberta_eda_state8_lr15_fold0/val_logits.npy",
}

WEIGHTS = {
    "large": 0.24114308710234722,
    "tab": 0.4438507372276068,
    "diet": 0.21685297669057865,
    "rob8": 0.09815319897946724,
}

TEMPS = {
    "large": 1.0,
    "tab": 1.0,
    "diet": 1.0,
    "rob8": 0.85,
}

BIAS = np.array([
    0.0, -0.3000011444091797, 0.39999818801879883,
    0.6499979496002197, -0.3000011444091797, 0.0, 0.0,
    -0.000001430511474609375, -0.2500011920928955,
    -0.35000109672546387, -0.2500011920928955,
    -0.10000133514404297, 0.4999980926513672,
    1.3499972820281982,
], dtype=np.float64)

PAIR_CONFIGS = [
    ("run_tests", "lint_or_typecheck",
     [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]),

    ("ask_user", "web_search",
     [0.50, 0.55, 0.60, 0.65, 0.70]),

    ("edit_file", "apply_patch",
     [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]),

    ("run_bash", "run_tests",
     [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]),
]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_team_module():
    p = Path("submit/script_large_tab_diet_rob8_gated40.py")
    spec = importlib.util.spec_from_file_location("team_submit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("Loading data...")

samples = load_jsonl("data/train.jsonl")
label_df = pd.read_csv("data/train_labels.csv")

label_map = dict(zip(
    label_df["id"].astype(str),
    label_df["action"].astype(str),
))

ids = np.array([str(x["id"]) for x in samples])

labels = np.array(
    [C2I[label_map[sid]] for sid in ids],
    dtype=np.int64,
)

team = load_team_module()

print("Building features...")

feat = pd.DataFrame([
    team.make_features(x)
    for x in samples
])

pred_df = pd.read_csv(PRED_CSV)

val_ids = pred_df["id"].astype(str).values
val_set = set(val_ids)

id_to_idx = {
    sid: i
    for i, sid in enumerate(ids)
}

val_idx = np.array([
    id_to_idx[sid]
    for sid in val_ids
])

train_mask = np.array([
    sid not in val_set
    for sid in ids
])

y_val = labels[val_idx]
y_train = labels[train_mask]

assert np.array_equal(
    y_val,
    pred_df["true_id"].astype(int).values,
)


# -----------------------------
# base score
# -----------------------------

raw = {
    k: np.load(p).astype(np.float64)
    for k, p in MODEL_PATHS.items()
}

score = sum(
    WEIGHTS[k] * (raw[k] / TEMPS[k])
    for k in WEIGHTS
)
score += BIAS

order = np.argsort(-score, axis=1)
base_pred = order[:, 0]
second_pred = order[:, 1]

BASE_F1 = f1_score(
    y_val,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print("BASE F1:", BASE_F1)


# -----------------------------
# features
# -----------------------------

X_all = feat.drop(
    columns=["id", "session_id"],
    errors="ignore",
).copy()

cat_cols = [
    c for c in X_all.columns
    if not pd.api.types.is_numeric_dtype(X_all[c])
]

num_cols = [
    c for c in X_all.columns
    if pd.api.types.is_numeric_dtype(X_all[c])
]

for c in cat_cols:
    X_all[c] = X_all[c].fillna("<NA>").astype(str)

for c in num_cols:
    X_all[c] = pd.to_numeric(
        X_all[c],
        errors="coerce",
    ).fillna(0.0)

pre = ColumnTransformer([
    (
        "cat",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
        cat_cols,
    ),
    ("num", "passthrough", num_cols),
])

print("Encoding...")

X_train = pre.fit_transform(
    X_all.loc[train_mask]
)

X_val = pre.transform(
    X_all.iloc[val_idx]
)


# -----------------------------
# train pair models once
# -----------------------------

pair_outputs = {}

for name_a, name_b, thresholds in PAIR_CONFIGS:
    a = C2I[name_a]
    b = C2I[name_b]

    tr = (y_train == a) | (y_train == b)

    gate = (
        ((base_pred == a) & (second_pred == b))
        | ((base_pred == b) & (second_pred == a))
    )

    idx = np.where(gate)[0]

    y_bin = (y_train[tr] == b).astype(np.int64)

    print(
        f"Training {name_a}<->{name_b}: "
        f"train={tr.sum()} gate={len(idx)}"
    )

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    clf.fit(
        X_train[tr],
        y_bin,
    )

    prob_b = clf.predict_proba(
        X_val[idx]
    )[:, 1]

    pair_pred = np.where(
        prob_b >= 0.5,
        b,
        a,
    )

    key = f"{name_a}<->{name_b}"

    pair_outputs[key] = {
        "idx": idx,
        "prob_b": prob_b,
        "pair_pred": pair_pred,
        "thresholds": thresholds,
    }


# -----------------------------
# exhaustive threshold combination
# -----------------------------

keys = list(pair_outputs)

threshold_grids = [
    pair_outputs[k]["thresholds"]
    for k in keys
]

rows = []

print("Searching threshold combinations...")

for thresholds in itertools.product(*threshold_grids):

    pred = base_pred.copy()

    total_use = 0
    total_changed = 0

    # sequential application
    for key, thr in zip(keys, thresholds):
        info = pair_outputs[key]

        idx = info["idx"]
        prob_b = info["prob_b"]
        pair_pred = info["pair_pred"]

        confident = (
            (prob_b >= thr)
            | (prob_b <= 1.0 - thr)
        )

        use_idx = idx[confident]
        new_pred = pair_pred[confident]

        total_use += len(use_idx)
        total_changed += int(
            (pred[use_idx] != new_pred).sum()
        )

        pred[use_idx] = new_pred

    f1 = f1_score(
        y_val,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    acc = accuracy_score(y_val, pred)

    row = {
        "macro_f1": f1,
        "gain": f1 - BASE_F1,
        "accuracy": acc,
        "total_use": total_use,
        "total_changed": total_changed,
    }

    for key, thr in zip(keys, thresholds):
        row[f"thr_{key}"] = thr

    rows.append(row)


result = pd.DataFrame(rows).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

result.to_csv(
    "pairwise_combined_results.csv",
    index=False,
)

print()
print("#" * 120)
print("GLOBAL TOP 50")
print("#" * 120)

print(result.head(50).to_string(index=False))

print()
print("Saved pairwise_combined_results.csv")
