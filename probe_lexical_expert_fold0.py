#!/usr/bin/env python3

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


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


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "old_train",
        "src/train.py",
    )

    # src/train.py has top-level training code, so do NOT import it.
    # Instead copy only the builders locally below.
    raise RuntimeError("unused")


def build_text(sample):
    parts = []

    parts.append("[CURRENT_PROMPT]")
    parts.append(str(sample.get("current_prompt", "")))

    meta = sample.get("session_meta", {})
    parts.append("[SESSION_META]")
    parts.append(f"user_tier={meta.get('user_tier', '')}")
    parts.append(f"language_pref={meta.get('language_pref', '')}")
    parts.append(
        f"budget_tokens_remaining={meta.get('budget_tokens_remaining', '')}"
    )
    parts.append(f"turn_index={meta.get('turn_index', '')}")
    parts.append(
        f"elapsed_session_sec={meta.get('elapsed_session_sec', '')}"
    )

    workspace = meta.get("workspace", {})

    parts.append("[WORKSPACE]")
    parts.append(f"loc={workspace.get('loc', '')}")
    parts.append(f"git_dirty={workspace.get('git_dirty', '')}")
    parts.append(
        f"last_ci_status={workspace.get('last_ci_status', '')}"
    )

    open_files = workspace.get("open_files", [])
    parts.append(
        "open_files=" + " ".join(map(str, open_files))
    )

    language_mix = workspace.get("language_mix", {})

    parts.append(
        "language_mix="
        + " ".join(
            f"{k}:{v}"
            for k, v in language_mix.items()
        )
    )

    history = sample.get("history", [])

    parts.append("[HISTORY]")

    for h in history:
        role = h.get("role", "")

        if role == "user":
            parts.append("[USER]")
            parts.append(str(h.get("content", "")))
        else:
            parts.append("[ASSISTANT_ACTION]")
            parts.append(str(h.get("name", "")))
            parts.append(str(h.get("args", "")))
            parts.append(str(h.get("result_summary", "")))

    recent_actions = []

    for h in history[-6:]:
        if h.get("role") != "user":
            recent_actions.append(
                str(h.get("name", ""))
            )

    parts.append("[RECENT_ACTIONS]")
    parts.append(" ".join(recent_actions))

    if recent_actions:
        last_action = recent_actions[-1]

        parts.append("[LAST_ACTION]")
        parts.append(
            " ".join([last_action] * 5)
        )

    recent_users = []

    for h in history[-6:]:
        if h.get("role") == "user":
            recent_users.append(
                str(h.get("content", ""))
            )

    if recent_users:
        parts.append("[RECENT_USER_PROMPTS]")
        parts.append(
            "\n".join(recent_users[-2:])
        )

    return "\n".join(parts)


def build_record(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])

    actions = [
        h.get("name", "")
        for h in history
        if h.get("role") != "user"
    ]

    user_turns = [
        h.get("content", "")
        for h in history
        if h.get("role") == "user"
    ]

    language_mix = workspace.get(
        "language_mix",
        {},
    )

    open_files = workspace.get(
        "open_files",
        [],
    )

    return {
        "text": build_text(sample),

        "user_tier": str(meta.get("user_tier", "")),
        "language_pref": str(meta.get("language_pref", "")),
        "git_dirty": str(workspace.get("git_dirty", "")),
        "last_ci_status": str(
            workspace.get("last_ci_status", "")
        ),
        "last_action": (
            str(actions[-1])
            if actions
            else "none"
        ),
        "prev_action": (
            str(actions[-2])
            if len(actions) >= 2
            else "none"
        ),

        "budget_tokens_remaining": (
            meta.get("budget_tokens_remaining", 0)
            or 0
        ),
        "turn_index": (
            meta.get("turn_index", 0)
            or 0
        ),
        "elapsed_session_sec": (
            meta.get("elapsed_session_sec", 0)
            or 0
        ),
        "loc": (
            workspace.get("loc", 0)
            or 0
        ),
        "num_open_files": len(open_files),
        "num_history": len(history),
        "num_actions": len(actions),
        "num_user_turns": len(user_turns),

        "ratio_py": language_mix.get("py", 0.0),
        "ratio_js": language_mix.get("js", 0.0),
        "ratio_ts": language_mix.get("ts", 0.0),
        "ratio_sql": language_mix.get("sql", 0.0),
        "ratio_java": language_mix.get("java", 0.0),
        "ratio_cpp": language_mix.get("cpp", 0.0),
    }


print("Loading data...")

samples = [
    json.loads(line)
    for line in open(
        "data/train.jsonl",
        encoding="utf-8",
    )
    if line.strip()
]

labels = {
    row["id"]: row["action"]
    for row in csv.DictReader(
        open(
            "data/train_labels.csv",
            encoding="utf-8",
        )
    )
}

ids = np.array([
    str(sample["id"])
    for sample in samples
])

y_all = np.array([
    C2I[labels[sid]]
    for sid in ids
])

print("Building lexical records...")

X_all = pd.DataFrame([
    build_record(sample)
    for sample in samples
])

pred_df = pd.read_csv(PRED_CSV)

val_ids = pred_df["id"].astype(str).values
val_id_set = set(val_ids)

id_to_idx = {
    sid: i
    for i, sid in enumerate(ids)
}

val_idx = np.array([
    id_to_idx[sid]
    for sid in val_ids
])

train_mask = np.array([
    sid not in val_id_set
    for sid in ids
])

train_idx = np.where(train_mask)[0]

X_train = X_all.iloc[train_idx]
X_val = X_all.iloc[val_idx]

y_train = y_all[train_idx]
y_val = y_all[val_idx]

assert np.array_equal(
    y_val,
    pred_df["true_id"].astype(int).values,
)

print("train:", len(train_idx))
print("val:", len(val_idx))


categorical_cols = [
    "user_tier",
    "language_pref",
    "git_dirty",
    "last_ci_status",
    "last_action",
    "prev_action",
]

numeric_cols = [
    "budget_tokens_remaining",
    "turn_index",
    "elapsed_session_sec",
    "loc",
    "num_open_files",
    "num_history",
    "num_actions",
    "num_user_turns",
    "ratio_py",
    "ratio_js",
    "ratio_ts",
    "ratio_sql",
    "ratio_java",
    "ratio_cpp",
]


preprocess = ColumnTransformer([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
            lowercase=True,
        ),
        "text",
    ),

    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
            lowercase=True,
        ),
        "text",
    ),

    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore",
        ),
        categorical_cols,
    ),

    (
        "num",
        StandardScaler(),
        numeric_cols,
    ),
])


pipe = Pipeline([
    ("features", preprocess),
    (
        "clf",
        LinearSVC(
            C=2.0,
            class_weight="balanced",
            max_iter=5000,
        ),
    ),
])


print("Training lexical expert...")

pipe.fit(
    X_train,
    y_train,
)

print("Predicting...")

lex_pred = pipe.predict(X_val)
lex_score = pipe.decision_function(X_val)

lex_f1 = f1_score(
    y_val,
    lex_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

lex_acc = accuracy_score(
    y_val,
    lex_pred,
)

print()
print("LEXICAL")
print("macro_f1:", lex_f1)
print("accuracy:", lex_acc)


# ============================================================
# Current SOTA base
# ============================================================

raw = {
    name: np.load(path).astype(np.float64)
    for name, path in MODEL_PATHS.items()
}

base_score = sum(
    WEIGHTS[name] * (
        raw[name] / TEMPS[name]
    )
    for name in WEIGHTS
)

base_score += BIAS

base_pred = base_score.argmax(axis=1)

base_f1 = f1_score(
    y_val,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

base_acc = accuracy_score(
    y_val,
    base_pred,
)

print()
print("BASE")
print("macro_f1:", base_f1)
print("accuracy:", base_acc)


# ============================================================
# Complementarity
# ============================================================

base_correct = base_pred == y_val
lex_correct = lex_pred == y_val

print()
print("COMPLEMENTARITY")

print(
    "base correct / lexical correct:",
    int((base_correct & lex_correct).sum()),
)

print(
    "base correct / lexical wrong:",
    int((base_correct & ~lex_correct).sum()),
)

print(
    "base wrong / lexical correct:",
    int((~base_correct & lex_correct).sum()),
)

print(
    "both wrong:",
    int((~base_correct & ~lex_correct).sum()),
)

print(
    "lexical recovery among base errors:",
    float(
        lex_correct[~base_correct].mean()
    ),
)


# ============================================================
# Oracle union
# ============================================================

oracle_pred = base_pred.copy()

recover = (
    (~base_correct)
    & lex_correct
)

oracle_pred[recover] = lex_pred[recover]

oracle_f1 = f1_score(
    y_val,
    oracle_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print()
print("ORACLE UNION")
print("recoverable:", int(recover.sum()))
print("macro_f1:", oracle_f1)
print("gain:", oracle_f1 - base_f1)


# ============================================================
# Agreement / disagreement analysis
# ============================================================

agree = lex_pred == base_pred
disagree = ~agree

print()
print("AGREEMENT")

print(
    "agree n:",
    int(agree.sum()),
)

print(
    "agree accuracy:",
    accuracy_score(
        y_val[agree],
        base_pred[agree],
    ),
)

print(
    "disagree n:",
    int(disagree.sum()),
)

print(
    "base acc on disagreement:",
    accuracy_score(
        y_val[disagree],
        base_pred[disagree],
    ),
)

print(
    "lex acc on disagreement:",
    accuracy_score(
        y_val[disagree],
        lex_pred[disagree],
    ),
)


# ============================================================
# Lexical confidence / base margin diagnostics
# ============================================================

lex_order = np.argsort(
    -lex_score,
    axis=1,
)

lex_sorted = np.sort(
    lex_score,
    axis=1,
)

lex_margin = (
    lex_sorted[:, -1]
    - lex_sorted[:, -2]
)

base_sorted = np.sort(
    base_score,
    axis=1,
)

base_margin = (
    base_sorted[:, -1]
    - base_sorted[:, -2]
)

rows = []

for lex_thr in [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
]:
    for base_thr in [
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        999.0,
    ]:
        use = (
            disagree
            & (lex_margin >= lex_thr)
            & (base_margin <= base_thr)
        )

        pred = base_pred.copy()
        pred[use] = lex_pred[use]

        f1 = f1_score(
            y_val,
            pred,
            labels=range(len(CLASSES)),
            average="macro",
            zero_division=0,
        )

        rows.append({
            "lex_margin_thr": lex_thr,
            "base_margin_thr": base_thr,
            "macro_f1": f1,
            "gain": f1 - base_f1,
            "accuracy": accuracy_score(
                y_val,
                pred,
            ),
            "use_n": int(use.sum()),
            "changed_n": int(use.sum()),
            "lex_acc_use": (
                accuracy_score(
                    y_val[use],
                    lex_pred[use],
                )
                if use.sum()
                else np.nan
            ),
            "base_acc_use": (
                accuracy_score(
                    y_val[use],
                    base_pred[use],
                )
                if use.sum()
                else np.nan
            ),
        })


result = pd.DataFrame(rows).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

print()
print("LEXICAL OVERRIDE TOP 50")
print(
    result.head(50).to_string(
        index=False,
    )
)

result.to_csv(
    "lexical_expert_fold0_override_results.csv",
    index=False,
)

np.save(
    "lexical_expert_fold0_decision.npy",
    lex_score,
)

out_pred = pred_df[[
    "id",
    "true_id",
]].copy()

out_pred["lex_pred_id"] = lex_pred
out_pred["lex_pred_action"] = [
    CLASSES[x]
    for x in lex_pred
]
out_pred["lex_margin"] = lex_margin
out_pred["base_pred_id"] = base_pred
out_pred["base_margin"] = base_margin

out_pred.to_csv(
    "lexical_expert_fold0_predictions.csv",
    index=False,
)

print()
print("Saved:")
print(" - lexical_expert_fold0_override_results.csv")
print(" - lexical_expert_fold0_decision.npy")
print(" - lexical_expert_fold0_predictions.csv")
