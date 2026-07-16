import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "stacking_data"
OUT_DIR.mkdir(exist_ok=True)

REF_VAL_CSV = ROOT / "submit/model/deberta_base_eda_state4/val_predictions.csv"

ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}

def get_session_id(sid):
    return re.sub(r"-step_\d+$", "", str(sid))

def get_step(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return int(m.group(1)) if m else -1

def contains(pat, text):
    return bool(re.search(pat, str(text).lower()))

def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]

def get_last_result(sample):
    for h in reversed(sample.get("history", [])):
        if h.get("role") == "assistant_action":
            return str(h.get("result_summary", ""))
    return ""

def get_last_args(sample):
    for h in reversed(sample.get("history", [])):
        if h.get("role") == "assistant_action":
            return json.dumps(h.get("args", {}), ensure_ascii=False)
    return ""

def extract_paths(text):
    text = str(text)
    paths = re.findall(
        r"[\w./-]+\.(?:py|js|ts|tsx|jsx|json|yaml|yml|toml|md|txt|css|html|go|rs|java|cpp|c|h|sh|sql|env)",
        text,
    )
    globs = re.findall(r"(?:\*[\w./-]*\.\w+|[\w./-]*\*\w*|_\w+\.\w+)", text)
    return paths, globs

def has_symbol_like(text):
    t = str(text)
    return bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\s*(?:\(|\.|::)?", t))

def bucket_step(step):
    if step <= 2:
        return "early"
    if step <= 7:
        return "mid"
    return "late"

def bucket_len_words(n_words):
    if n_words <= 40:
        return "short"
    if n_words <= 120:
        return "mid"
    return "long"

def find_true_col(df):
    for c in ["true_action", "true", "label", "true_label", "target"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find true label column: {df.columns.tolist()}")

def feature_row(sample):
    sid = str(sample["id"])
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    prompt = str(sample.get("current_prompt", ""))
    p = prompt.lower()

    actions = get_actions(sample)
    prev1 = actions[-1] if len(actions) >= 1 else "none"
    prev2 = actions[-2] if len(actions) >= 2 else "none"
    prev3 = actions[-3] if len(actions) >= 3 else "none"

    step = get_step(sid)
    last_result = get_last_result(sample)
    last_args = get_last_args(sample)

    open_files = workspace.get("open_files", []) or []

    prompt_paths, prompt_globs = extract_paths(prompt)
    args_paths, args_globs = extract_paths(last_args)
    result_paths, result_globs = extract_paths(last_result)

    all_path_text = " ".join(map(str, prompt_paths + args_paths + result_paths + open_files))

    has_explicit_file = len(prompt_paths) > 0
    has_glob_pattern = (
        len(prompt_globs) > 0
        or contains(r"\b(every|all)\s+[\w.-]+\.\w+\b", p)
        or contains(r"\*\.", p)
        or contains(r"\b전부\b.*\.\w+", p)
    )
    has_dir_phrase = contains(r"\b(directory|folder|dir|what'?s in|list|ls|tree)\b|뭐뭐|뭐 있|목록|디렉토리|폴더", p)
    has_search_phrase = contains(r"\b(where|search|grep|find|occurrence|occurrences|reference|references|used|uses|call|calls)\b|어디|찾아|쓰이는|호출", p)
    has_open_phrase = contains(r"\b(open|read|show|view|display|inspect|look at|cat)\b|열어|보자|봐줘", p)
    has_test_phrase = contains(r"\b(test|tests|pytest|unit test|spec|verify)\b|검증|테스트", p)
    has_lint_phrase = contains(r"\b(lint|typecheck|type check|mypy|ruff|eslint|tsc|format|formatter|prettier)\b", p)
    has_web_phrase = contains(r"\b(web|internet|online|latest|current|look up|search online|browse|google)\b|라이브러리|버전|최신", p)
    has_ask_phrase = contains(r"\b(ask|clarify|confirm|which one|do you want|should i|need more info)\b|확인해|물어|어느", p)
    has_plan_phrase = contains(r"\b(plan|steps|strategy|approach|roadmap|todo)\b|계획|순서|전략", p)

    symbol_like = has_symbol_like(prompt)

    # file-search 4형제 intent
    intent_read_file = has_open_phrase and has_explicit_file and not has_search_phrase
    intent_grep_search = has_search_phrase and (symbol_like or not has_explicit_file)
    intent_list_directory = has_dir_phrase and not has_glob_pattern
    intent_glob_pattern = has_glob_pattern or contains(r"\b(files?|전부|모든)\b.*\.\w+", p)

    after_edit_or_patch = prev1 in {"edit_file", "apply_patch", "write_file"}
    after_search = prev1 in {"grep_search", "glob_pattern", "list_directory"}
    after_read = prev1 == "read_file"
    after_run = prev1 in {"run_bash", "run_tests", "lint_or_typecheck"}

    lr = last_result.lower()

    return {
        "id": sid,
        "session_id": get_session_id(sid),
        "step": step,
        "step_bucket": bucket_step(step),
        "prompt_len": len(prompt),
        "prompt_word_count": len(prompt.split()),
        "prompt_len_bucket": bucket_len_words(len(prompt.split())),
        "history_len": len(sample.get("history", [])),
        "num_actions": len(actions),

        "prev1": prev1,
        "prev2": prev2,
        "prev3": prev3,
        "action_seq_last3": ">".join(actions[-3:]) if actions else "none",

        "after_edit_or_patch": after_edit_or_patch,
        "after_search": after_search,
        "after_read": after_read,
        "after_run": after_run,

        "has_explicit_file": has_explicit_file,
        "has_glob_pattern": has_glob_pattern,
        "has_dir_phrase": has_dir_phrase,
        "has_search_phrase": has_search_phrase,
        "has_open_phrase": has_open_phrase,
        "has_symbol_like": symbol_like,
        "has_test_phrase": has_test_phrase,
        "has_lint_phrase": has_lint_phrase,
        "has_web_phrase": has_web_phrase,
        "has_ask_phrase": has_ask_phrase,
        "has_plan_phrase": has_plan_phrase,

        "intent_read_file": intent_read_file,
        "intent_grep_search": intent_grep_search,
        "intent_list_directory": intent_list_directory,
        "intent_glob_pattern": intent_glob_pattern,

        "last_result_has_error": contains(r"\b(error|failed|fail|traceback|exception|not found|missing)\b", lr),
        "last_result_has_pass": contains(r"\b(pass|passed|success|ok)\b", lr),
        "last_result_has_match": contains(r"\b(match|matches|occurrence|found)\b", lr),
        "last_result_no_match": contains(r"\b(no matches|0 matches|not found)\b", lr),
        "last_result_has_patch": contains(r"\b(patched|updated|modified|changed|applied)\b", lr),
        "last_result_has_test": contains(r"\b(test|tests|pytest|unit)\b", lr),
        "last_result_has_lint": contains(r"\b(lint|typecheck|mypy|ruff|eslint|tsc)\b", lr),

        "num_open_files": len(open_files),
        "git_dirty": str(workspace.get("git_dirty", "")),
        "last_ci_status": str(workspace.get("last_ci_status", "")),
    }

def main():
    samples = {}
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                samples[str(s["id"])] = s

    val = pd.read_csv(REF_VAL_CSV)
    true_col = find_true_col(val)

    rows = []
    labels = []

    for _, r in val.iterrows():
        sid = str(r["id"])
        if sid not in samples:
            raise ValueError(f"id not found in train.jsonl: {sid}")
        rows.append(feature_row(samples[sid]))
        labels.append(str(r[true_col]))

    feat = pd.DataFrame(rows)
    feat["label"] = labels
    feat["label_id"] = [ACTION_TO_ID[x] for x in labels]

    feat.to_csv(OUT_DIR / "val_meta_features.csv", index=False)
    np.save(OUT_DIR / "y_val.npy", feat["label_id"].values.astype(np.int64))
    np.save(OUT_DIR / "groups_val.npy", feat["session_id"].astype(str).values)
    np.save(OUT_DIR / "ids_val.npy", feat["id"].astype(str).values)

    print("[SAVED]", OUT_DIR / "val_meta_features.csv")
    print("shape:", feat.shape)
    print(feat.head().to_string(index=False))

if __name__ == "__main__":
    main()
