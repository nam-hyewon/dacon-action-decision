#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
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
    0.0,
    -0.3000011444091797,
    0.39999818801879883,
    0.6499979496002197,
    -0.3000011444091797,
    0.0,
    0.0,
    -0.000001430511474609375,
    -0.2500011920928955,
    -0.35000109672546387,
    -0.2500011920928955,
    -0.10000133514404297,
    0.4999980926513672,
    1.3499972820281982,
], dtype=np.float64)

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

    spec = importlib.util.spec_from_file_location(
        "team_submit",
        p,
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def add_meta_features(X, raw, score):
    X = X.copy()

    n = len(score)

    order = np.argsort(-score, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]

    sorted_score = np.sort(score, axis=1)

    X["ens_top1"] = top1
    X["ens_top2"] = top2
    X["ens_margin"] = (
        sorted_score[:, -1] - sorted_score[:, -2]
    )
    X["ens_margin13"] = (
        sorted_score[:, -1] - sorted_score[:, -3]
    )

    for model_name, logits in raw.items():
        cur = logits / TEMPS[model_name]

        pred = cur.argmax(axis=1)
        sorted_cur = np.sort(cur, axis=1)

        X[f"{model_name}_pred"] = pred
        X[f"{model_name}_margin"] = (
            sorted_cur[:, -1] - sorted_cur[:, -2]
        )

        for j, cls in enumerate(CLASSES):
            X[f"{model_name}_logit_{cls}"] = cur[:, j]

    model_preds = np.column_stack([
        (raw[name] / TEMPS[name]).argmax(axis=1)
        for name in MODEL_PATHS
    ])

    X["num_unique_model_preds"] = np.array([
        len(set(row.tolist()))
        for row in model_preds
    ])

    for j, cls in enumerate(CLASSES):
        X[f"vote_{cls}"] = (
            model_preds == j
        ).sum(axis=1)

        X[f"ens_logit_{cls}"] = score[:, j]

    for name_a, name_b in PAIRS:
        a = C2I[name_a]
        b = C2I[name_b]

        X[f"ens_gap_{name_a}_minus_{name_b}"] = (
            score[:, a] - score[:, b]
        )

        for model_name, logits in raw.items():
            cur = logits / TEMPS[model_name]

            X[
                f"{model_name}_gap_{name_a}_minus_{name_b}"
            ] = cur[:, a] - cur[:, b]

    return X


print("Loading data...")

samples = load_jsonl("data/train.jsonl")

label_df = pd.read_csv("data/train_labels.csv")

label_map = dict(zip(
    label_df["id"].astype(str),
    label_df["action"].astype(str),
))

ids = np.array([
    str(sample["id"])
    for sample in samples
])

team = load_team_module()

print("Building structural features...")

feat_all = pd.DataFrame([
    team.make_features(sample)
    for sample in samples
])

pred_df = pd.read_csv(PRED_CSV)

val_ids = pred_df["id"].astype(str).values
y = pred_df["true_id"].astype(int).values

id_to_idx = {
    sid: i
    for i, sid in enumerate(ids)
}

val_idx = np.array([
    id_to_idx[sid]
    for sid in val_ids
])

X = feat_all.iloc[val_idx].reset_index(drop=True)

raw = {
    name: np.load(path).astype(np.float64)
    for name, path in MODEL_PATHS.items()
}

score = sum(
    WEIGHTS[name] * (
        raw[name] / TEMPS[name]
    )
    for name in WEIGHTS
)

score += BIAS

order = np.argsort(-score, axis=1)
base_pred = order[:, 0]
second_pred = order[:, 1]

BASE_F1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print("BASE F1:", BASE_F1)

X = X.drop(
    columns=["id", "session_id"],
    errors="ignore",
)

X = add_meta_features(
    X=X,
    raw=raw,
    score=score,
)

cat_cols = [
    c
    for c in X.columns
    if not pd.api.types.is_numeric_dtype(X[c])
]

num_cols = [
    c
    for c in X.columns
    if pd.api.types.is_numeric_dtype(X[c])
]

for c in cat_cols:
    X[c] = X[c].fillna("<NA>").astype(str)

for c in num_cols:
    X[c] = pd.to_numeric(
        X[c],
        errors="coerce",
    ).fillna(0.0)

print("features:", X.shape)
print("categorical:", len(cat_cols))
print("numeric:", len(num_cols))

pair_outputs = {}

for name_a, name_b in PAIRS:
    a = C2I[name_a]
    b = C2I[name_b]

    gate = (
        ((base_pred == a) & (second_pred == b))
        | ((base_pred == b) & (second_pred == a))
    )

    idx = np.where(gate)[0]

    pair_truth_mask = (
        (y[idx] == a)
        | (y[idx] == b)
    )

    trainable_idx = idx[pair_truth_mask]

    print()
    print("=" * 100)
    print(name_a, "<->", name_b)
    print("gate:", len(idx))
    print("true-pair trainable:", len(trainable_idx))

    if len(trainable_idx) < 100:
        print("SKIP: too few pair samples")
        continue

    pair_y = (
        y[trainable_idx] == b
    ).astype(np.int64)

    pair_prob_b = np.full(
        len(idx),
        np.nan,
        dtype=np.float64,
    )

    idx_to_gatepos = {
        global_idx: pos
        for pos, global_idx in enumerate(idx)
    }

    skf = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    for fold, (tr_local, va_local) in enumerate(
        skf.split(trainable_idx, pair_y)
    ):
        tr_idx = trainable_idx[tr_local]
        va_idx = trainable_idx[va_local]

        pre = ColumnTransformer([
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                cat_cols,
            ),
            (
                "num",
                "passthrough",
                num_cols,
            ),
        ])

        X_tr = pre.fit_transform(X.iloc[tr_idx])
        X_va = pre.transform(X.iloc[va_idx])

        y_tr = (
            y[tr_idx] == b
        ).astype(np.int64)

        clf = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=400,
            max_leaf_nodes=31,
            l2_regularization=2.0,
            random_state=42 + fold,
        )

        clf.fit(X_tr, y_tr)

        prob = clf.predict_proba(X_va)[:, 1]

        for global_idx, p in zip(va_idx, prob):
            pair_prob_b[
                idx_to_gatepos[global_idx]
            ] = p

    oof_known = ~np.isnan(pair_prob_b)

    pair_pred = np.where(
        pair_prob_b >= 0.5,
        b,
        a,
    )

    true_pair_gate = (
        (y[idx] == a)
        | (y[idx] == b)
    )

    eval_mask = oof_known & true_pair_gate

    pair_acc = accuracy_score(
        y[idx][eval_mask],
        pair_pred[eval_mask],
    )

    base_pair_acc = accuracy_score(
        y[idx][eval_mask],
        base_pred[idx][eval_mask],
    )

    print("OOF pair acc:", pair_acc)
    print("base pair acc:", base_pair_acc)

    pair_outputs[f"{name_a}<->{name_b}"] = {
        "a": a,
        "b": b,
        "idx": idx,
        "prob_b": pair_prob_b,
        "pair_pred": pair_pred,
        "oof_known": oof_known,
        "pair_acc": pair_acc,
        "base_pair_acc": base_pair_acc,
    }


print()
print("#" * 110)
print("INDIVIDUAL PAIR SWEEP")
print("#" * 110)

individual_rows = []

thresholds = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95,
]

for key, info in pair_outputs.items():
    idx = info["idx"]
    prob_b = info["prob_b"]
    pair_pred = info["pair_pred"]
    known = info["oof_known"]

    for thr in thresholds:
        confident = known & (
            (prob_b >= thr)
            | (prob_b <= 1.0 - thr)
        )

        use_idx = idx[confident]

        pred = base_pred.copy()
        pred[use_idx] = pair_pred[confident]

        f1 = f1_score(
            y,
            pred,
            labels=range(len(CLASSES)),
            average="macro",
            zero_division=0,
        )

        individual_rows.append({
            "pair": key,
            "thr": thr,
            "macro_f1": f1,
            "gain": f1 - BASE_F1,
            "use_n": len(use_idx),
            "changed_n": int(
                (
                    pred[use_idx]
                    != base_pred[use_idx]
                ).sum()
            ),
            "pair_acc": info["pair_acc"],
            "base_pair_acc": info["base_pair_acc"],
        })


individual_df = pd.DataFrame(
    individual_rows
).sort_values(
    ["macro_f1"],
    ascending=False,
)

print(
    individual_df.head(50).to_string(
        index=False
    )
)

individual_df.to_csv(
    "pairwise_meta_probe_individual.csv",
    index=False,
)


print()
print("#" * 110)
print("GREEDY COMBINATION SEARCH")
print("#" * 110)

best_by_pair = {}

for key in pair_outputs:
    rows = individual_df[
        individual_df["pair"] == key
    ]

    if len(rows):
        best_by_pair[key] = rows.iloc[0]

selected = []
current_pred = base_pred.copy()
current_f1 = BASE_F1

remaining = set(best_by_pair)

while remaining:
    best_candidate = None

    for key in remaining:
        row = best_by_pair[key]
        thr = float(row["thr"])
        info = pair_outputs[key]

        idx = info["idx"]
        prob_b = info["prob_b"]
        pair_pred = info["pair_pred"]
        known = info["oof_known"]

        confident = known & (
            (prob_b >= thr)
            | (prob_b <= 1.0 - thr)
        )

        use_idx = idx[confident]

        pred = current_pred.copy()
        pred[use_idx] = pair_pred[confident]

        f1 = f1_score(
            y,
            pred,
            labels=range(len(CLASSES)),
            average="macro",
            zero_division=0,
        )

        if (
            best_candidate is None
            or f1 > best_candidate["f1"]
        ):
            best_candidate = {
                "key": key,
                "thr": thr,
                "pred": pred,
                "f1": f1,
                "use_n": len(use_idx),
            }

    if best_candidate["f1"] <= current_f1:
        break

    selected.append({
        "pair": best_candidate["key"],
        "thr": best_candidate["thr"],
        "macro_f1": best_candidate["f1"],
        "gain_vs_base": (
            best_candidate["f1"] - BASE_F1
        ),
        "use_n": best_candidate["use_n"],
    })

    current_pred = best_candidate["pred"]
    current_f1 = best_candidate["f1"]

    remaining.remove(best_candidate["key"])

    print(
        "ADD",
        best_candidate["key"],
        "thr=",
        best_candidate["thr"],
        "F1=",
        best_candidate["f1"],
        "gain=",
        best_candidate["f1"] - BASE_F1,
    )


print()
print("FINAL GREEDY F1:", current_f1)
print("FINAL GAIN:", current_f1 - BASE_F1)

print()
print("SELECTED:")
for row in selected:
    print(row)

pd.DataFrame(selected).to_csv(
    "pairwise_meta_probe_greedy.csv",
    index=False,
)

print()
print("Saved:")
print(" - pairwise_meta_probe_individual.csv")
print(" - pairwise_meta_probe_greedy.csv")
