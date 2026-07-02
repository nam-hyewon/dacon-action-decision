import csv
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]


def build_text(sample):
    parts = []

    parts.append("[CURRENT_PROMPT]")
    parts.append(str(sample.get("current_prompt", "")))

    meta = sample.get("session_meta", {})
    parts.append("[SESSION_META]")
    parts.append(f"user_tier={meta.get('user_tier', '')}")
    parts.append(f"language_pref={meta.get('language_pref', '')}")
    parts.append(f"budget_tokens_remaining={meta.get('budget_tokens_remaining', '')}")
    parts.append(f"turn_index={meta.get('turn_index', '')}")
    parts.append(f"elapsed_session_sec={meta.get('elapsed_session_sec', '')}")

    workspace = meta.get("workspace", {})
    parts.append("[WORKSPACE]")
    parts.append(f"loc={workspace.get('loc', '')}")
    parts.append(f"git_dirty={workspace.get('git_dirty', '')}")
    parts.append(f"last_ci_status={workspace.get('last_ci_status', '')}")

    open_files = workspace.get("open_files", [])
    parts.append("open_files=" + " ".join(map(str, open_files)))

    language_mix = workspace.get("language_mix", {})
    parts.append("language_mix=" + " ".join(
        f"{k}:{v}" for k, v in language_mix.items()
    ))

    history = sample.get("history", [])

    parts.append("[HISTORY]")
    for h in history:
        role = h.get("role", "")

        if role == "user":
            parts.append("[USER]")
            parts.append(str(h.get("content", "")))
        else:
            action_name = str(h.get("name", ""))
            parts.append("[ASSISTANT_ACTION]")
            parts.append(action_name)
            parts.append(str(h.get("args", "")))
            parts.append(str(h.get("result_summary", "")))

    recent_actions = []
    for h in history[-6:]:
        if h.get("role") != "user":
            recent_actions.append(str(h.get("name", "")))

    parts.append("[RECENT_ACTIONS]")
    parts.append(" ".join(recent_actions))

    if recent_actions:
        last_action = recent_actions[-1]
        parts.append("[LAST_ACTION]")
        parts.append(" ".join([last_action] * 5))

    recent_users = []
    for h in history[-6:]:
        if h.get("role") == "user":
            recent_users.append(str(h.get("content", "")))

    if recent_users:
        parts.append("[RECENT_USER_PROMPTS]")
        parts.append("\n".join(recent_users[-2:]))

    return "\n".join(parts)


def build_record(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])

    actions = [h.get("name", "") for h in history if h.get("role") != "user"]
    user_turns = [h.get("content", "") for h in history if h.get("role") == "user"]

    language_mix = workspace.get("language_mix", {})
    open_files = workspace.get("open_files", [])

    return {
        "text": build_text(sample),

        # categorical features
        "user_tier": str(meta.get("user_tier", "")),
        "language_pref": str(meta.get("language_pref", "")),
        "git_dirty": str(workspace.get("git_dirty", "")),
        "last_ci_status": str(workspace.get("last_ci_status", "")),
        "last_action": str(actions[-1]) if actions else "none",
        "prev_action": str(actions[-2]) if len(actions) >= 2 else "none",

        # numeric features
        "budget_tokens_remaining": meta.get("budget_tokens_remaining", 0) or 0,
        "turn_index": meta.get("turn_index", 0) or 0,
        "elapsed_session_sec": meta.get("elapsed_session_sec", 0) or 0,
        "loc": workspace.get("loc", 0) or 0,
        "num_open_files": len(open_files),
        "num_history": len(history),
        "num_actions": len(actions),
        "num_user_turns": len(user_turns),

        # language ratio features
        "ratio_py": language_mix.get("py", 0.0),
        "ratio_js": language_mix.get("js", 0.0),
        "ratio_ts": language_mix.get("ts", 0.0),
        "ratio_sql": language_mix.get("sql", 0.0),
        "ratio_java": language_mix.get("java", 0.0),
        "ratio_cpp": language_mix.get("cpp", 0.0),
    }


DATA_DIR = "../data"

samples = [
    json.loads(line)
    for line in open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8")
    if line.strip()
]

labels = {
    row["id"]: row["action"]
    for row in csv.DictReader(
        open(os.path.join(DATA_DIR, "train_labels.csv"), encoding="utf-8")
    )
}

X = pd.DataFrame([build_record(s) for s in samples])
y = [labels[s["id"]] for s in samples]

print("samples:", len(X), "| classes:", len(set(y)))

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("train:", len(X_train), "| val:", len(X_val))

text_col = "text"

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
    ("word_tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
        lowercase=True,
    ), text_col),

    ("char_tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
        lowercase=True,
    ), text_col),

    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),

    ("num", StandardScaler(), numeric_cols),
])

pipe = Pipeline([
    ("features", preprocess),
    ("clf", LinearSVC(
        C=2.0,
        class_weight="balanced",
        max_iter=5000,
    )),
])

pipe.fit(X_train, y_train)
print("학습 완료")

val_pred = pipe.predict(X_val)
macro_f1 = f1_score(
    y_val,
    val_pred,
    labels=ALL_CLASSES,
    average="macro",
    zero_division=0,
)

print(f"Validation Macro-F1: {macro_f1:.4f}")

pipe.fit(X, y)

os.makedirs("../submit/model", exist_ok=True)
joblib.dump(pipe, "../submit/model/tfidf_logreg.pkl", compress=3)

print("저장 완료: ../submit/model/tfidf_logreg.pkl")