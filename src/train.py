import csv
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

DATA_DIR = "../data"

# 데이터 로드 (한 번만)
samples = [json.loads(line)
           for line in open(f"{DATA_DIR}/train.jsonl", encoding="utf-8")
           if line.strip()]
labels = {row["id"]: row["action"]
          for row in csv.DictReader(open(f"{DATA_DIR}/train_labels.csv", encoding="utf-8"))}

y = [labels[s["id"]] for s in samples]


# ============================================
# 버전 1: full (압축 없이 다 넣기)
# ============================================
def build_text_full(sample):
    parts = []
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    parts.append(current_prompt)
    parts.append(f"user_tier: {meta.get('user_tier', '')}")
    parts.append(f"language_pref: {meta.get('language_pref', '')}")
    parts.append(f"turn_index: {meta.get('turn_index', '')}")
    parts.append(f"loc: {workspace.get('loc', '')}")
    parts.append(f"git_dirty: {workspace.get('git_dirty', '')}")
    parts.append(f"last_ci_status: {workspace.get('last_ci_status', '')}")

    open_files = workspace.get("open_files", [])
    parts.append("open_files: " + " ".join(map(str, open_files)))

    recent_history = history[-10:]
    recent_actions = []
    for h in recent_history:
        if h.get("role") == "user":
            parts.append(str(h.get("content", "")))
        else:
            action_name = str(h.get("name", ""))
            recent_actions.append(action_name)
            parts.append(f"args: {h.get('args', '')}")
            parts.append(f"result: {h.get('result_summary', '')}")

    parts.append(" ".join(recent_actions))
    if recent_actions:
        parts.append(f"last_action: {recent_actions[-1]}")

    return " ".join(parts)


# ============================================
# 버전 2: eda_state (압축 + 본인 EDA 피처)
# ============================================
def build_text_eda_state(sample):
    parts = []
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    # 1. 현재 프롬프트 (최우선)
    parts.append(current_prompt)

    # 2. 본인 EDA 피처 — prompt 길이
    length = len(current_prompt)
    if length < 45:
        len_tag = "short"
    elif length < 75:
        len_tag = "medium"
    else:
        len_tag = "long"
    parts.append(f"prompt_len:{len_tag}")

    # 3. 본인 EDA 피처 — 첫 턴 여부
    is_first = len(history) == 0
    parts.append(f"is_first_turn:{is_first}")

    # 4. 직전/전전 action (핵심 피처)
    assistant_actions = [h['name'] for h in history if h.get('role') == 'assistant_action']
    last_action = assistant_actions[-1] if assistant_actions else 'none'
    prev_action = assistant_actions[-2] if len(assistant_actions) >= 2 else 'none'
    parts.append(f"last_action:{last_action}")
    parts.append(f"prev_action:{prev_action}")

    # 5. turn_index
    parts.append(f"turn_index:{meta.get('turn_index', '')}")

    # 6. CI / git 상태
    parts.append(f"ci:{workspace.get('last_ci_status', '')}")
    parts.append(f"git_dirty:{workspace.get('git_dirty', '')}")

    # 7. 압축된 action sequence (이름만, 최근 5개)
    parts.append("actions:" + ",".join(assistant_actions[-5:]))

    # 8. 최근 user 발화 (최근 2개만, 축약)
    user_utterances = [str(h.get('content', ''))[:50]
                        for h in history if h.get('role') == 'user']
    parts.append("recent_user:" + " | ".join(user_utterances[-2:]))

    return " ".join(parts)


def get_assistant_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_user_utterances(sample):
    return [
        str(h.get("content", ""))
        for h in sample.get("history", [])
        if h.get("role") == "user"
    ]


def bucket_int(value, cuts):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in cuts:
        if v <= upper:
            return name
    return f"gt_{cuts[-1][1]}"


def build_text_ablation(sample, features):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))
    actions = get_assistant_actions(sample)
    users = get_user_utterances(sample)

    parts = []

    if "prompt" in features:
        parts.extend(["[PROMPT]", current_prompt])

    if "prompt_len" in features:
        prompt_len = len(current_prompt)
        parts.extend([
            f"prompt_len:{prompt_len}",
            f"prompt_bucket:{bucket_int(prompt_len, [('short', 45), ('medium', 80), ('long', 120)])}",
        ])

    if "turn" in features:
        turn_index = meta.get("turn_index", 0)
        parts.extend([
            f"turn_index:{turn_index}",
            f"turn_bucket:{bucket_int(turn_index, [('t1', 1), ('t2_4', 4), ('t5_8', 8), ('t9_12', 12)])}",
        ])

    if "history_len" in features:
        history_len = len(history)
        parts.extend([
            f"is_first_turn:{history_len == 0}",
            f"history_len:{history_len}",
            f"history_bucket:{bucket_int(history_len, [('h0', 0), ('h1_3', 3), ('h4_7', 7), ('h8_12', 12)])}",
        ])

    if "last_action" in features:
        last_action = actions[-1] if actions else "none"
        parts.append(f"last_action:{last_action}")

    if "prev_action" in features:
        prev_action = actions[-2] if len(actions) >= 2 else "none"
        parts.append(f"prev_action:{prev_action}")

    if "action_seq" in features:
        parts.append("action_seq:" + " > ".join(actions[-8:]))

    if "ci_git" in features:
        ci = workspace.get("last_ci_status", "")
        parts.extend([
            f"last_ci_status:{ci}",
            f"ci_failed:{ci == 'failed'}",
            f"git_dirty:{workspace.get('git_dirty', '')}",
        ])

    if "workspace" in features:
        language_mix = workspace.get("language_mix", {}) or {}
        main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
        parts.extend([
            f"loc:{workspace.get('loc', '')}",
            f"num_open_files:{len(workspace.get('open_files', []))}",
            f"open_files:{' '.join(map(str, workspace.get('open_files', [])))}",
            f"main_lang:{main_lang}",
        ])

    if "recent_user" in features:
        parts.extend(["[RECENT_USER]", " | ".join(users[-3:])])

    if "last_result" in features:
        last_result = ""
        for h in reversed(history):
            if h.get("role") == "assistant_action":
                last_result = str(h.get("result_summary", ""))
                break
        parts.extend(["[LAST_RESULT]", last_result])

    if "compact_history" in features:
        compact = []
        for h in history[-8:]:
            if h.get("role") == "user":
                compact.append("U:" + str(h.get("content", "")))
            elif h.get("role") == "assistant_action":
                compact.append(
                    "A:"
                    + str(h.get("name", ""))
                    + " "
                    + str(h.get("args", ""))
                    + " "
                    + str(h.get("result_summary", ""))
                )
        parts.extend(["[COMPACT_HISTORY]", " ".join(compact)])

    return " ".join(parts)


# ============================================
# 실험 함수
# ============================================
def run_experiment(name, build_fn, show_report=False):
    X = [build_fn(s) for s in samples]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=80_000,
            sublinear_tf=True, lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=500, class_weight="balanced", C=2.0,
        )),
    ])

    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    macro_f1 = f1_score(y_val, val_pred, labels=ALL_CLASSES, average="macro", zero_division=0)

    print(f"[{name}] Macro-F1: {macro_f1:.4f}")
    if show_report:
        print(classification_report(
            y_val,
            val_pred,
            labels=ALL_CLASSES,
            target_names=ALL_CLASSES,
            digits=4,
            zero_division=0,
        ))
    return macro_f1


def run_ablation_suite():
    experiments = [
        ("ab00_prompt", ["prompt"]),
        ("ab01_prompt_last", ["prompt", "last_action"]),
        ("ab02_plus_prev", ["prompt", "last_action", "prev_action"]),
        ("ab03_plus_action_seq", ["prompt", "last_action", "prev_action", "action_seq"]),
        ("ab04_plus_prompt_len", ["prompt", "last_action", "prev_action", "action_seq", "prompt_len"]),
        ("ab05_plus_history_len", ["prompt", "last_action", "prev_action", "action_seq", "prompt_len", "history_len"]),
        ("ab06_plus_turn", ["prompt", "last_action", "prev_action", "action_seq", "prompt_len", "history_len", "turn"]),
        ("ab07_plus_ci_git", ["prompt", "last_action", "prev_action", "action_seq", "prompt_len", "history_len", "turn", "ci_git"]),
        ("ab08_plus_recent_user", ["prompt", "last_action", "prev_action", "action_seq", "prompt_len", "history_len", "turn", "ci_git", "recent_user"]),
        ("ab09_plus_last_result", ["prompt", "last_action", "prev_action", "action_seq", "prompt_len", "history_len", "turn", "ci_git", "recent_user", "last_result"]),
        ("ab10_plus_workspace", ["prompt", "last_action", "prev_action", "action_seq", "prompt_len", "history_len", "turn", "ci_git", "recent_user", "last_result", "workspace"]),
        ("ab11_plus_compact_history", ["prompt", "last_action", "prev_action", "action_seq", "prompt_len", "history_len", "turn", "ci_git", "recent_user", "last_result", "workspace", "compact_history"]),
    ]

    results = []
    best_name = None
    best_f1 = -1.0
    for name, features in experiments:
        f1 = run_experiment(
            name,
            lambda s, features=features: build_text_ablation(s, features),
        )
        results.append((name, f1, features))
        if f1 > best_f1:
            best_name = name
            best_f1 = f1

    print("\n=== Ablation Summary ===")
    prev = None
    for name, f1, features in results:
        delta = 0.0 if prev is None else f1 - prev
        print(f"{name:28s} {f1:.4f} delta={delta:+.4f} features={','.join(features)}")
        prev = f1

    print(f"\nBest ablation: {best_name} Macro-F1={best_f1:.4f}")
    return results


# ============================================
# 실행
# ============================================
print("=== 실험 시작 ===\n")
f1_full = run_experiment("full", build_text_full)
f1_eda = run_experiment("eda_state", build_text_eda_state)

print("\n=== 결과 비교 ===")
print(f"full:      {f1_full:.4f}")
print(f"eda_state: {f1_eda:.4f}")
print(f"차이:      {f1_eda - f1_full:+.4f}")

print("\n=== Feature Ablation ===")
run_ablation_suite()
