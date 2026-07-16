#!/usr/bin/env python3

import importlib.util
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

PAIRS = [
    ("read_file", "grep_search"),
    ("read_file", "list_directory"),
    ("ask_user", "plan_task"),
    ("run_bash", "run_tests"),
    ("edit_file", "apply_patch"),
    ("run_tests", "lint_or_typecheck"),
    ("run_bash", "lint_or_typecheck"),
    ("ask_user", "web_search"),
]

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


print("Loading train...")
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
    team.make_features(sample)
    for sample in samples
]).reset_index(drop=True)

pred_df = pd.read_csv(PRED_CSV)
val_ids = pred_df["id"].astype(str).values
val_set = set(val_ids)

id_to_idx = {sid: i for i, sid in enumerate(ids)}

val_idx = np.array(
    [id_to_idx[sid] for sid in val_ids],
    dtype=np.int64,
)

train_mask = np.array(
    [sid not in val_set for sid in ids],
    dtype=bool,
)

y_val = labels[val_idx]

assert np.array_equal(
    y_val,
    pred_df["true_id"].astype(int).values,
)

print("reference:", int(train_mask.sum()))
print("validation:", len(val_idx))


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

print("categorical cols:", cat_cols)
print("numeric cols:", num_cols)

# Make dtypes explicit before ColumnTransformer.
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

print("Encoding features...")
X_train = pre.fit_transform(X_all.loc[train_mask])
X_val = pre.transform(X_all.iloc[val_idx])
y_train = labels[train_mask]

results = []

for name_a, name_b in PAIRS:
    a = C2I[name_a]
    b = C2I[name_b]

    print("\n" + "=" * 90)
    print(name_a, "<->", name_b)
    print("=" * 90)

    tr_mask = (y_train == a) | (y_train == b)

    gate = (
        ((base_pred == a) & (second_pred == b))
        | ((base_pred == b) & (second_pred == a))
    )

    idx = np.where(gate)[0]

    print("train pair samples:", int(tr_mask.sum()))
    print("validation gate:", len(idx))

    if tr_mask.sum() < 50 or len(idx) == 0:
        continue

    y_bin = (y_train[tr_mask] == b).astype(np.int64)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    clf.fit(X_train[tr_mask], y_bin)

    prob_b = clf.predict_proba(X_val[idx])[:, 1]
    pair_pred = np.where(prob_b >= 0.5, b, a)

    true_pair = (
        (y_val[idx] == a)
        | (y_val[idx] == b)
    )

    if true_pair.any():
        pair_acc = accuracy_score(
            y_val[idx][true_pair],
            pair_pred[true_pair],
        )
        base_pair_acc = accuracy_score(
            y_val[idx][true_pair],
            base_pred[idx][true_pair],
        )
    else:
        pair_acc = np.nan
        base_pair_acc = np.nan

    print("pair reranker acc true-pair:", pair_acc)
    print("base acc true-pair:", base_pair_acc)

    for thr in [
        0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
    ]:
        confident = (
            (prob_b >= thr)
            | (prob_b <= 1.0 - thr)
        )

        use_idx = idx[confident]

        pred = base_pred.copy()
        pred[use_idx] = pair_pred[confident]

        f1 = f1_score(
            y_val,
            pred,
            labels=range(len(CLASSES)),
            average="macro",
            zero_division=0,
        )

        acc = accuracy_score(y_val, pred)

        changed = int(
            (pred[use_idx] != base_pred[use_idx]).sum()
        )

        results.append({
            "pair": f"{name_a}<->{name_b}",
            "conf_thr": thr,
            "macro_f1": f1,
            "gain": f1 - BASE_F1,
            "accuracy": acc,
            "gate_n": len(idx),
            "use_n": len(use_idx),
            "changed_n": changed,
            "pair_acc_truepair": pair_acc,
            "base_acc_truepair": base_pair_acc,
        })

        print(
            f"thr={thr:.2f} "
            f"F1={f1:.9f} "
            f"gain={f1 - BASE_F1:+.9f} "
            f"use={len(use_idx)} "
            f"changed={changed}"
        )


result_df = pd.DataFrame(results).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

result_df.to_csv(
    "pairwise_histgb_results.csv",
    index=False,
)

print("\n" + "#" * 110)
print("GLOBAL TOP 50")
print("#" * 110)
print(result_df.head(50).to_string(index=False))

print("\nSaved pairwise_histgb_results.csv")
