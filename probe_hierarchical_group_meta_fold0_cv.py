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

GROUPS = {
    "FILE": [
        "read_file",
        "grep_search",
        "list_directory",
        "glob_pattern",
    ],
    "WRITE": [
        "edit_file",
        "write_file",
        "apply_patch",
    ],
    "EXEC": [
        "run_bash",
        "run_tests",
        "lint_or_typecheck",
    ],
    "CONTROL": [
        "ask_user",
        "plan_task",
        "web_search",
        "respond_only",
    ],
}

GROUP_NAMES = list(GROUPS)

GROUP_IDS = {
    name: np.array(
        [C2I[x] for x in members],
        dtype=np.int64,
    )
    for name, members in GROUPS.items()
}

CLASS_TO_GROUP = {}

for gi, group in enumerate(GROUP_NAMES):
    for cls in GROUPS[group]:
        CLASS_TO_GROUP[C2I[cls]] = gi

CLASS_TO_GROUP_ARR = np.array([
    CLASS_TO_GROUP[i]
    for i in range(len(CLASSES))
], dtype=np.int64)


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

TEMPS = {
    "large": 1.0,
    "tab": 1.0,
    "diet": 1.0,
    "rob8": 0.85,
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

ids = np.array([
    str(x["id"])
    for x in samples
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


# ============================================================
# Exact base
# ============================================================

base_score = np.load(
    "exact_repro_base_score.npy"
).astype(np.float64)

base_pred = np.load(
    "exact_repro_base_pred.npy"
).astype(np.int64)

BASE_F1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

BASE_ACC = accuracy_score(y, base_pred)

pred_group = CLASS_TO_GROUP_ARR[base_pred]
true_group = CLASS_TO_GROUP_ARR[y]

print("BASE F1:", BASE_F1)
print("BASE ACC:", BASE_ACC)
print(
    "BASE GROUP ACC:",
    accuracy_score(true_group, pred_group),
)


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
# Exact component meta features
# ============================================================

raw = {
    name: np.load(path).astype(np.float64)
    for name, path in MODEL_PATHS.items()
}

scaled = {
    name: raw[name] / TEMPS[name]
    for name in raw
}

meta = {}

row_idx = np.arange(len(y))

for j, cls in enumerate(CLASSES):
    meta[f"ensemble_{cls}"] = base_score[:, j]

for model_name, score in scaled.items():

    pred = score.argmax(axis=1)
    sorted_score = np.sort(score, axis=1)

    meta[f"{model_name}_pred"] = pred

    meta[f"{model_name}_margin"] = (
        sorted_score[:, -1]
        - sorted_score[:, -2]
    )

    for j, cls in enumerate(CLASSES):
        meta[f"{model_name}_{cls}"] = score[:, j]


model_preds = np.column_stack([
    scaled[name].argmax(axis=1)
    for name in MODEL_PATHS
])

meta["num_unique_model_preds"] = np.array([
    len(set(row.tolist()))
    for row in model_preds
])


# group-relative features
for group_name, member_ids in GROUP_IDS.items():

    group_score = base_score[:, member_ids]

    group_order_local = np.argsort(
        -group_score,
        axis=1,
    )

    group_sorted = np.sort(
        group_score,
        axis=1,
    )

    meta[f"{group_name}_ens_margin12"] = (
        group_sorted[:, -1]
        - group_sorted[:, -2]
    )

    if len(member_ids) >= 3:
        meta[f"{group_name}_ens_margin13"] = (
            group_sorted[:, -1]
            - group_sorted[:, -3]
        )

    for rank in range(len(member_ids)):
        candidate = member_ids[
            group_order_local[:, rank]
        ]

        meta[
            f"{group_name}_rank{rank}_class"
        ] = candidate

    for model_name, score in scaled.items():

        cur = score[:, member_ids]
        cur_sorted = np.sort(cur, axis=1)

        meta[
            f"{group_name}_{model_name}_margin12"
        ] = (
            cur_sorted[:, -1]
            - cur_sorted[:, -2]
        )


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
print("categorical:", len(cat_cols))
print("numeric:", len(num_cols))


# ============================================================
# Group-specific 3-fold OOF experts
#
# IMPORTANT:
# Train only on rows whose TRUE class belongs to the group.
# Apply only where EXISTING BASE predicts that group.
# ============================================================

group_outputs = {}

for gi, group_name in enumerate(GROUP_NAMES):

    member_ids = GROUP_IDS[group_name]

    train_idx = np.where(
        true_group == gi
    )[0]

    apply_idx = np.where(
        pred_group == gi
    )[0]

    local_map = {
        int(class_id): local_id
        for local_id, class_id in enumerate(member_ids)
    }

    y_local = np.array([
        local_map[int(y[i])]
        for i in train_idx
    ], dtype=np.int64)


    print()
    print("=" * 110)
    print(group_name)
    print("=" * 110)

    print("members:", GROUPS[group_name])
    print("train rows:", len(train_idx))
    print("apply rows:", len(apply_idx))

    print(
        "class counts:",
        np.bincount(
            y_local,
            minlength=len(member_ids),
        ).tolist(),
    )


    oof_prob = np.full(
        (len(y), len(member_ids)),
        np.nan,
        dtype=np.float64,
    )


    skf = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )


    for fold, (tr_local, va_local) in enumerate(
        skf.split(train_idx, y_local)
    ):

        tr_idx = train_idx[tr_local]
        va_idx = train_idx[va_local]

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

        y_tr = np.array([
            local_map[int(y[i])]
            for i in tr_idx
        ], dtype=np.int64)


        clf = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=3.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42 + fold,
        )

        clf.fit(
            X_tr,
            y_tr,
        )

        prob = clf.predict_proba(
            X_va
        )

        full_prob = np.zeros(
            (len(va_idx), len(member_ids)),
            dtype=np.float64,
        )

        for col, cls_local in enumerate(clf.classes_):
            full_prob[:, int(cls_local)] = prob[:, col]

        oof_prob[va_idx] = full_prob


    known = ~np.isnan(oof_prob).any(axis=1)

    assert known.sum() == len(train_idx)


    local_pred = oof_prob[known].argmax(axis=1)

    specialist_pred_known = member_ids[
        local_pred
    ]

    known_idx = np.where(known)[0]


    specialist_acc = accuracy_score(
        y[known_idx],
        specialist_pred_known,
    )

    base_acc_same_rows = accuracy_score(
        y[known_idx],
        base_pred[known_idx],
    )

    specialist_f1 = f1_score(
        y[known_idx],
        specialist_pred_known,
        labels=member_ids,
        average="macro",
        zero_division=0,
    )

    base_f1_same_rows = f1_score(
        y[known_idx],
        base_pred[known_idx],
        labels=member_ids,
        average="macro",
        zero_division=0,
    )


    print()
    print("OOF SPECIALIST")

    print(
        "base acc on true-group rows:",
        base_acc_same_rows,
    )

    print(
        "specialist acc:",
        specialist_acc,
    )

    print(
        "base within-group macro-F1:",
        base_f1_same_rows,
    )

    print(
        "specialist within-group macro-F1:",
        specialist_f1,
    )


    group_outputs[group_name] = {
        "gi": gi,
        "member_ids": member_ids,
        "train_idx": train_idx,
        "apply_idx": apply_idx,
        "oof_prob": oof_prob,
        "known": known,
    }


# ============================================================
# Honest OOF application
#
# Only rows with OOF specialist prediction are eligible.
# This excludes the 63 rows whose BASE predicts a group different
# from true group, because that group's specialist wasn't trained/
# OOF-predicted on those rows.
#
# That's okay: this is a conservative hierarchy feasibility probe.
# ============================================================

print()
print("=" * 120)
print("INDIVIDUAL GROUP SWEEP")
print("=" * 120)


thresholds = [
    0.00,
    0.30,
    0.40,
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
]


individual_rows = []


for group_name, info in group_outputs.items():

    gi = info["gi"]
    member_ids = info["member_ids"]
    oof_prob = info["oof_prob"]
    known = info["known"]

    use_pool = (
        known
        & (pred_group == gi)
    )

    pool_idx = np.where(use_pool)[0]

    prob = oof_prob[pool_idx]

    local_pred = prob.argmax(axis=1)

    specialist_pred = member_ids[
        local_pred
    ]

    confidence = prob.max(axis=1)


    for conf_thr in thresholds:

        use_local = (
            confidence >= conf_thr
        )

        use_idx = pool_idx[
            use_local
        ]

        new_pred = specialist_pred[
            use_local
        ]

        pred = base_pred.copy()

        pred[use_idx] = new_pred


        f1 = f1_score(
            y,
            pred,
            labels=range(len(CLASSES)),
            average="macro",
            zero_division=0,
        )


        beneficial = int(
            (
                (base_pred[use_idx] != y[use_idx])
                & (pred[use_idx] == y[use_idx])
            ).sum()
        )

        harmful = int(
            (
                (base_pred[use_idx] == y[use_idx])
                & (pred[use_idx] != y[use_idx])
            ).sum()
        )


        individual_rows.append({
            "group": group_name,
            "conf_thr": conf_thr,
            "macro_f1": f1,
            "gain": f1 - BASE_F1,
            "accuracy": accuracy_score(y, pred),
            "pool_n": len(pool_idx),
            "use_n": len(use_idx),
            "changed_n": int(
                (
                    pred[use_idx]
                    != base_pred[use_idx]
                ).sum()
            ),
            "beneficial_changes": beneficial,
            "harmful_changes": harmful,
            "net_correct_gain": beneficial - harmful,
        })


individual_df = pd.DataFrame(
    individual_rows
).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)


print(
    individual_df.head(60).to_string(
        index=False,
    )
)

individual_df.to_csv(
    "hierarchical_group_meta_individual.csv",
    index=False,
)


# ============================================================
# Greedy combination using each group's best threshold
# ============================================================

print()
print("=" * 120)
print("GREEDY GROUP COMBINATION")
print("=" * 120)


best_config = {}

for group_name in GROUP_NAMES:

    sub = individual_df[
        individual_df["group"] == group_name
    ]

    best_config[group_name] = sub.iloc[0]


current_pred = base_pred.copy()
current_f1 = BASE_F1

remaining = set(GROUP_NAMES)

selected = []


while remaining:

    best_candidate = None

    for group_name in remaining:

        info = group_outputs[group_name]

        gi = info["gi"]
        member_ids = info["member_ids"]
        oof_prob = info["oof_prob"]
        known = info["known"]

        thr = float(
            best_config[group_name]["conf_thr"]
        )

        use_pool = (
            known
            & (pred_group == gi)
        )

        pool_idx = np.where(use_pool)[0]

        prob = oof_prob[pool_idx]

        confidence = prob.max(axis=1)

        specialist_pred = member_ids[
            prob.argmax(axis=1)
        ]

        use_local = (
            confidence >= thr
        )

        use_idx = pool_idx[
            use_local
        ]

        pred = current_pred.copy()

        pred[use_idx] = specialist_pred[
            use_local
        ]

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
                "group": group_name,
                "thr": thr,
                "pred": pred,
                "f1": f1,
                "use_n": len(use_idx),
            }


    if best_candidate["f1"] <= current_f1:
        break


    current_pred = best_candidate["pred"]
    current_f1 = best_candidate["f1"]

    selected.append({
        "group": best_candidate["group"],
        "thr": best_candidate["thr"],
        "macro_f1": current_f1,
        "gain_vs_base": current_f1 - BASE_F1,
        "use_n": best_candidate["use_n"],
    })


    print(
        "ADD",
        best_candidate["group"],
        "thr=",
        best_candidate["thr"],
        "F1=",
        current_f1,
        "gain=",
        current_f1 - BASE_F1,
    )


    remaining.remove(
        best_candidate["group"]
    )


print()
print("FINAL GREEDY F1:", current_f1)
print("FINAL GAIN:", current_f1 - BASE_F1)

print()
print("SELECTED:")

for row in selected:
    print(row)


pd.DataFrame(selected).to_csv(
    "hierarchical_group_meta_greedy.csv",
    index=False,
)

print()
print("Saved:")
print(" - hierarchical_group_meta_individual.csv")
print(" - hierarchical_group_meta_greedy.csv")
