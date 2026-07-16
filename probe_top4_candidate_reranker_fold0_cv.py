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

team = load_team_module()

print("Building structural features...")

feat_all = pd.DataFrame([
    team.make_features(sample)
    for sample in samples
])

ids = np.array([
    str(sample["id"])
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

X_struct = feat_all.iloc[val_idx].reset_index(drop=True)

raw = {
    name: np.load(path).astype(np.float64)
    for name, path in MODEL_PATHS.items()
}

scaled = {
    name: raw[name] / TEMPS[name]
    for name in raw
}

score = sum(
    WEIGHTS[name] * scaled[name]
    for name in WEIGHTS
)
score += BIAS

order = np.argsort(-score, axis=1)

top4 = order[:, :4]
base_pred = top4[:, 0]

BASE_F1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

BASE_ACC = accuracy_score(y, base_pred)

true_rank = np.full(len(y), -1, dtype=np.int64)

for rank in range(4):
    true_rank[top4[:, rank] == y] = rank

in_top4 = true_rank >= 0

print("BASE F1:", BASE_F1)
print("BASE ACC:", BASE_ACC)
print("top4 covered:", int(in_top4.sum()), "/", len(y))
print("top4 coverage:", float(in_top4.mean()))

for rank in range(4):
    print(
        f"true rank {rank}:",
        int((true_rank == rank).sum())
    )

print("outside top4:", int((true_rank < 0).sum()))


# ============================================================
# Build candidate-relative meta features
# ============================================================

X = X_struct.drop(
    columns=["id", "session_id"],
    errors="ignore",
).copy()

meta = {}

sorted_score = np.take_along_axis(
    score,
    top4,
    axis=1,
)

for rank in range(4):
    cand = top4[:, rank]

    meta[f"cand{rank}_class"] = cand
    meta[f"cand{rank}_ens_score"] = sorted_score[:, rank]

    for model_name, logits in scaled.items():
        meta[f"cand{rank}_{model_name}_score"] = (
            logits[np.arange(len(y)), cand]
        )

        meta[f"cand{rank}_{model_name}_rank"] = np.array([
            int(np.where(
                np.argsort(-logits[i]) == cand[i]
            )[0][0])
            for i in range(len(y))
        ])

    meta[f"cand{rank}_vote_count"] = np.zeros(
        len(y),
        dtype=np.int64,
    )

model_preds = np.column_stack([
    scaled[name].argmax(axis=1)
    for name in MODEL_PATHS
])

for rank in range(4):
    cand = top4[:, rank]

    meta[f"cand{rank}_vote_count"] = (
        model_preds == cand[:, None]
    ).sum(axis=1)


# pairwise candidate score gaps
for r1 in range(4):
    for r2 in range(r1 + 1, 4):
        meta[f"ens_gap_r{r1}_r{r2}"] = (
            sorted_score[:, r1] - sorted_score[:, r2]
        )

        for model_name, logits in scaled.items():
            c1 = top4[:, r1]
            c2 = top4[:, r2]

            meta[
                f"{model_name}_gap_r{r1}_r{r2}"
            ] = (
                logits[np.arange(len(y)), c1]
                - logits[np.arange(len(y)), c2]
            )


# general disagreement features
meta["num_unique_model_preds"] = np.array([
    len(set(row.tolist()))
    for row in model_preds
])

meta["top1_vote_count"] = (
    model_preds == top4[:, 0][:, None]
).sum(axis=1)

meta["top2_vote_count"] = (
    model_preds == top4[:, 1][:, None]
).sum(axis=1)

meta["top3_vote_count"] = (
    model_preds == top4[:, 2][:, None]
).sum(axis=1)

meta["top4_vote_count"] = (
    model_preds == top4[:, 3][:, None]
).sum(axis=1)

meta_df = pd.DataFrame(meta)

X = pd.concat(
    [
        X.reset_index(drop=True),
        meta_df.reset_index(drop=True),
    ],
    axis=1,
)

# defragment
X = X.copy()

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

print("feature shape:", X.shape)
print("categorical:", len(cat_cols))
print("numeric:", len(num_cols))


# ============================================================
# 3-fold OOF candidate-rank classifier
# ============================================================

trainable_idx = np.where(in_top4)[0]
rank_y = true_rank[trainable_idx]

print()
print("trainable candidate rows:", len(trainable_idx))

skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

oof_prob = np.full(
    (len(y), 4),
    np.nan,
    dtype=np.float64,
)

for fold, (tr_local, va_local) in enumerate(
    skf.split(trainable_idx, rank_y)
):
    tr_idx = trainable_idx[tr_local]
    va_idx = trainable_idx[va_local]

    print()
    print(
        f"Fold {fold}: "
        f"train={len(tr_idx)} val={len(va_idx)}"
    )

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

    y_tr = true_rank[tr_idx]

    clf = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=500,
        max_leaf_nodes=31,
        l2_regularization=2.0,
        random_state=42 + fold,
    )

    clf.fit(X_tr, y_tr)

    prob = clf.predict_proba(X_va)

    # sklearn class order safety
    full_prob = np.zeros(
        (len(va_idx), 4),
        dtype=np.float64,
    )

    for col, cls in enumerate(clf.classes_):
        full_prob[:, int(cls)] = prob[:, col]

    oof_prob[va_idx] = full_prob


known = ~np.isnan(oof_prob).any(axis=1)

assert known.sum() == in_top4.sum()

rank_pred = oof_prob.argmax(axis=1)

candidate_pred = base_pred.copy()

known_idx = np.where(known)[0]

candidate_pred[known_idx] = top4[
    known_idx,
    rank_pred[known_idx],
]

RAW_RERANK_F1 = f1_score(
    y,
    candidate_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

RAW_RERANK_ACC = accuracy_score(
    y,
    candidate_pred,
)

print()
print("#" * 110)
print("RAW TOP4 RERANK")
print("#" * 110)

print("base F1:", BASE_F1)
print("rerank F1:", RAW_RERANK_F1)
print("gain:", RAW_RERANK_F1 - BASE_F1)

print("base ACC:", BASE_ACC)
print("rerank ACC:", RAW_RERANK_ACC)

print(
    "rank prediction acc:",
    accuracy_score(
        true_rank[known],
        rank_pred[known],
    ),
)


# ============================================================
# Confidence gating
# Apply reranker only when reranker confidence is high
# ============================================================

print()
print("#" * 110)
print("CONFIDENCE GATE SWEEP")
print("#" * 110)

rows = []

max_prob = np.nanmax(oof_prob, axis=1)

for conf_thr in [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]:
    use = known & (max_prob >= conf_thr)

    pred = base_pred.copy()

    use_idx = np.where(use)[0]

    pred[use_idx] = top4[
        use_idx,
        rank_pred[use_idx],
    ]

    f1 = f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    acc = accuracy_score(y, pred)

    changed = int(
        (
            pred[use_idx]
            != base_pred[use_idx]
        ).sum()
    )

    rerank_acc_use = (
        accuracy_score(
            y[use_idx],
            pred[use_idx],
        )
        if len(use_idx)
        else np.nan
    )

    base_acc_use = (
        accuracy_score(
            y[use_idx],
            base_pred[use_idx],
        )
        if len(use_idx)
        else np.nan
    )

    rows.append({
        "conf_thr": conf_thr,
        "macro_f1": f1,
        "gain": f1 - BASE_F1,
        "accuracy": acc,
        "use_n": len(use_idx),
        "changed_n": changed,
        "rerank_acc_use": rerank_acc_use,
        "base_acc_use": base_acc_use,
    })


result = pd.DataFrame(rows).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

print(result.to_string(index=False))

result.to_csv(
    "top4_candidate_reranker_probe_results.csv",
    index=False,
)


# ============================================================
# Extra: only override when predicted rank != 0
# ============================================================

print()
print("#" * 110)
print("NON-TOP1 OVERRIDE SWEEP")
print("#" * 110)

rows2 = []

for conf_thr in [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]:
    use = (
        known
        & (max_prob >= conf_thr)
        & (rank_pred != 0)
    )

    pred = base_pred.copy()

    use_idx = np.where(use)[0]

    pred[use_idx] = top4[
        use_idx,
        rank_pred[use_idx],
    ]

    f1 = f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    acc = accuracy_score(y, pred)

    changed = int(
        (
            pred[use_idx]
            != base_pred[use_idx]
        ).sum()
    )

    rerank_acc_use = (
        accuracy_score(
            y[use_idx],
            pred[use_idx],
        )
        if len(use_idx)
        else np.nan
    )

    base_acc_use = (
        accuracy_score(
            y[use_idx],
            base_pred[use_idx],
        )
        if len(use_idx)
        else np.nan
    )

    rows2.append({
        "conf_thr": conf_thr,
        "macro_f1": f1,
        "gain": f1 - BASE_F1,
        "accuracy": acc,
        "use_n": len(use_idx),
        "changed_n": changed,
        "rerank_acc_use": rerank_acc_use,
        "base_acc_use": base_acc_use,
    })


result2 = pd.DataFrame(rows2).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

print(result2.to_string(index=False))

result2.to_csv(
    "top4_candidate_reranker_non_top1_results.csv",
    index=False,
)

print()
print("Saved:")
print(" - top4_candidate_reranker_probe_results.csv")
print(" - top4_candidate_reranker_non_top1_results.csv")
