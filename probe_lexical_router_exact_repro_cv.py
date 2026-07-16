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

TEMPS = {
    "large": 1.0,
    "tab": 1.0,
    "diet": 1.0,
    "rob8": 0.85,
}

REPRO_ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

PRED_CSV = (
    REPRO_ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)

MODEL_PATHS = {
    "large": (
        REPRO_ROOT
        / "submit/model/klue_roberta_large_eda_state_fold0/val_logits.npy"
    ),
    "tab": (
        REPRO_ROOT
        / "submit/model/tabular_histgb_fold0/val_logits.npy"
    ),
    "diet": (
        REPRO_ROOT
        / "submit/model/kluebert_diet_fold0/val_logits.npy"
    ),
    "rob8": (
        REPRO_ROOT
        / "submit/model/roberta_eda_state8_lr15_fold0/val_logits.npy"
    ),
}


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_team_module():
    path = Path(
        "submit/script_large_tab_diet_rob8_gated40.py"
    )

    spec = importlib.util.spec_from_file_location(
        "team_submit",
        path,
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


print("Loading data...")

samples = load_jsonl("data/train.jsonl")
ids = np.array([str(x["id"]) for x in samples])

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


# ============================================================
# Structural features
# ============================================================

team = load_team_module()

print("Building structural features...")

feat_all = pd.DataFrame([
    team.make_features(sample)
    for sample in samples
])

X = (
    feat_all
    .iloc[val_idx]
    .reset_index(drop=True)
)

X = X.drop(
    columns=["id", "session_id"],
    errors="ignore",
)


# ============================================================
# Exact 0.764678 base
# ============================================================

base_score = np.load(
    "exact_repro_base_score.npy"
).astype(np.float64)

base_pred = np.load(
    "exact_repro_base_pred.npy"
).astype(np.int64)

assert len(base_pred) == len(y)

BASE_F1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print("BASE F1:", BASE_F1)


# exact repro component logits
raw = {
    name: np.load(path).astype(np.float64)
    for name, path in MODEL_PATHS.items()
}

scaled = {
    name: raw[name] / TEMPS[name]
    for name in raw
}


# ============================================================
# Lexical expert
# ============================================================

lex_score = np.load(
    "lexical_expert_fold0_decision.npy"
).astype(np.float64)

lex_df = pd.read_csv(
    "lexical_expert_fold0_predictions.csv"
)

assert np.array_equal(
    lex_df["id"].astype(str).values,
    val_ids,
)

lex_pred = (
    lex_df["lex_pred_id"]
    .astype(int)
    .values
)


# ============================================================
# Router target
# ============================================================

disagree = base_pred != lex_pred

benefit = (
    (base_pred != y)
    & (lex_pred == y)
)

harm = (
    (base_pred == y)
    & (lex_pred != y)
)

neutral = (
    disagree
    & ~benefit
    & ~harm
)

router_idx = np.where(disagree)[0]
router_y = benefit[router_idx].astype(np.int64)

print("disagreement:", int(disagree.sum()))
print("benefit:", int(benefit.sum()))
print("harm:", int(harm.sum()))
print("neutral:", int(neutral.sum()))
print("positive rate:", float(router_y.mean()))


# ============================================================
# Meta features
# ============================================================

base_sorted = np.sort(base_score, axis=1)
lex_sorted = np.sort(lex_score, axis=1)

base_margin = (
    base_sorted[:, -1]
    - base_sorted[:, -2]
)

lex_margin = (
    lex_sorted[:, -1]
    - lex_sorted[:, -2]
)

row_idx = np.arange(len(y))

meta = {
    "base_pred": base_pred,
    "lex_pred": lex_pred,
    "base_margin": base_margin,
    "lex_margin": lex_margin,

    "base_pred_score": base_score[
        row_idx,
        base_pred,
    ],

    "base_score_on_lex": base_score[
        row_idx,
        lex_pred,
    ],

    "lex_pred_score": lex_score[
        row_idx,
        lex_pred,
    ],

    "lex_score_on_base": lex_score[
        row_idx,
        base_pred,
    ],
}

meta["base_gap_base_minus_lex"] = (
    meta["base_pred_score"]
    - meta["base_score_on_lex"]
)

meta["lex_gap_lex_minus_base"] = (
    meta["lex_pred_score"]
    - meta["lex_score_on_base"]
)


for j, cls in enumerate(CLASSES):
    meta[f"ens_{cls}"] = base_score[:, j]
    meta[f"lex_{cls}"] = lex_score[:, j]


model_preds = []

for model_name, scores in scaled.items():
    pred = scores.argmax(axis=1)

    model_preds.append(pred)

    meta[f"{model_name}_pred"] = pred

    cur_sorted = np.sort(scores, axis=1)

    meta[f"{model_name}_margin"] = (
        cur_sorted[:, -1]
        - cur_sorted[:, -2]
    )

    meta[f"{model_name}_score_base"] = scores[
        row_idx,
        base_pred,
    ]

    meta[f"{model_name}_score_lex"] = scores[
        row_idx,
        lex_pred,
    ]

    meta[f"{model_name}_gap_base_minus_lex"] = (
        meta[f"{model_name}_score_base"]
        - meta[f"{model_name}_score_lex"]
    )

    for j, cls in enumerate(CLASSES):
        meta[
            f"{model_name}_{cls}"
        ] = scores[:, j]


model_preds = np.column_stack(model_preds)

meta["num_unique_model_preds"] = np.array([
    len(set(row.tolist()))
    for row in model_preds
])

meta["base_vote_count"] = (
    model_preds
    == base_pred[:, None]
).sum(axis=1)

meta["lex_vote_count"] = (
    model_preds
    == lex_pred[:, None]
).sum(axis=1)

meta["any_model_votes_lex"] = (
    meta["lex_vote_count"] > 0
).astype(np.int64)


X = pd.concat(
    [
        X.reset_index(drop=True),
        pd.DataFrame(meta),
    ],
    axis=1,
).copy()


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
    X[c] = (
        X[c]
        .fillna("<NA>")
        .astype(str)
    )

for c in num_cols:
    X[c] = pd.to_numeric(
        X[c],
        errors="coerce",
    ).fillna(0.0)


print("feature shape:", X.shape)
print("router samples:", len(router_idx))


# ============================================================
# 3-fold OOF router
# ============================================================

skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

oof_prob = np.full(
    len(y),
    np.nan,
    dtype=np.float64,
)


for fold, (tr_local, va_local) in enumerate(
    skf.split(router_idx, router_y)
):
    tr_idx = router_idx[tr_local]
    va_idx = router_idx[va_local]

    print(
        f"fold {fold}: "
        f"train={len(tr_idx)} "
        f"val={len(va_idx)} "
        f"train_pos={int(benefit[tr_idx].sum())}"
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

    X_tr = pre.fit_transform(
        X.iloc[tr_idx]
    )

    X_va = pre.transform(
        X.iloc[va_idx]
    )

    y_tr = benefit[
        tr_idx
    ].astype(np.int64)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=3.0,
        random_state=42 + fold,
    )

    clf.fit(
        X_tr,
        y_tr,
    )

    oof_prob[va_idx] = (
        clf.predict_proba(X_va)[:, 1]
    )


known = ~np.isnan(oof_prob)

assert known.sum() == disagree.sum()


# ============================================================
# Threshold sweep
# ============================================================

print()
print("=" * 110)
print("ROUTER THRESHOLD SWEEP")
print("=" * 110)

thresholds = [
    0.025,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
    0.225,
    0.25,
    0.275,
    0.30,
    0.325,
    0.35,
    0.375,
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
]

rows = []

for thr in thresholds:
    use = (
        known
        & (oof_prob >= thr)
    )

    pred = base_pred.copy()
    pred[use] = lex_pred[use]

    f1 = f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    benefit_n = int(
        benefit[use].sum()
    )

    harm_n = int(
        harm[use].sum()
    )

    neutral_n = int(
        neutral[use].sum()
    )

    rows.append({
        "thr": thr,
        "macro_f1": f1,
        "gain": f1 - BASE_F1,
        "accuracy": accuracy_score(
            y,
            pred,
        ),
        "use_n": int(use.sum()),
        "benefit_n": benefit_n,
        "harm_n": harm_n,
        "neutral_n": neutral_n,
        "benefit_precision": (
            benefit_n / use.sum()
            if use.sum()
            else np.nan
        ),
        "benefit_recall": (
            benefit_n / benefit.sum()
            if benefit.sum()
            else np.nan
        ),
        "net_correct_gain": (
            benefit_n - harm_n
        ),
    })


result = pd.DataFrame(rows).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

print(result.to_string(index=False))

result.to_csv(
    "lexical_router_exact_repro_cv_results.csv",
    index=False,
)


# ============================================================
# Probability diagnostics
# ============================================================

print()
print("=" * 110)
print("ROUTER PROBABILITY DIAGNOSTICS")
print("=" * 110)


for name, mask in [
    ("BENEFIT", benefit),
    ("HARM", harm),
    ("NEUTRAL", neutral),
]:
    vals = oof_prob[
        mask & known
    ]

    print()
    print(name, "n=", len(vals))

    if len(vals):
        print(
            pd.Series(vals)
            .describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
            .to_string()
        )


np.save(
    "lexical_router_exact_repro_oof_prob.npy",
    oof_prob,
)

print()
print("Saved:")
print(
    " - lexical_router_exact_repro_cv_results.csv"
)
print(
    " - lexical_router_exact_repro_oof_prob.npy"
)
