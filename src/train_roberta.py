### mid4 포함한 결과

import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.metrics import f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = int(os.environ.get("SEED", 42))
# 한국어 특화 모델
MODEL_NAME = os.environ.get("MODEL_NAME", "klue/roberta-base")
MAX_LEN = int(os.environ.get("MAX_LEN", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
EPOCHS = int(os.environ.get("EPOCHS", 4))
LR = float(os.environ.get("LR", 2e-5))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01)) 
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.06)) 
TEXT_MODE = os.environ.get("TEXT_MODE", "eda_state").strip().lower()
SPLIT_MODE = os.environ.get("SPLIT_MODE", "row").strip().lower()
FOLD_FILE = os.environ.get("FOLD_FILE", "").strip()
FOLD_ID = int(os.environ.get("FOLD_ID", 0))
USE_CLASS_WEIGHT = os.environ.get("USE_CLASS_WEIGHT", "0") == "1"
WEIGHT_MODE = os.environ.get("WEIGHT_MODE", "sqrt_inv")
LOSS_TYPE = os.environ.get("LOSS_TYPE", "ce").strip().lower()   # ce | focal
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))
SAVE_EPOCH_CHECKPOINTS = {
    int(x.strip())
    for x in os.environ.get("SAVE_EPOCH_CHECKPOINTS", "").split(",")
    if x.strip()
}

_classes_env = os.environ.get("ACTION_CLASSES", "").strip()
ALL_CLASSES = [c.strip() for c in _classes_env.split(",") if c.strip()] or [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def compute_class_weights(labels, num_classes, mode="sqrt_inv"):
    """클래스 빈도 기반 가중치 (평균 1로 정규화)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1)
    if mode == "sqrt_inv":
        weights = 1.0 / np.sqrt(counts)
    elif mode == "inv":
        weights = 1.0 / counts
    else:
        weights = np.ones(num_classes)
    weights = weights / weights.mean()  # 평균 1 → loss 스케일/LR 유지
    return torch.tensor(weights, dtype=torch.float)


class FocalLoss(torch.nn.Module):
    """Multiclass Focal Loss. weight = per-class alpha (없으면 균등)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # tensor[num_classes] or None

    def forward(self, logits, target):
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)  # log p_t
        pt = logpt.exp()
        focal = (1.0 - pt) ** self.gamma * (-logpt)
        if self.weight is not None:
            focal = focal * self.weight.gather(0, target)  # alpha_t
        return focal.mean()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(ROOT_DIR / "data")))
SAVE_NAME = os.environ.get("SAVE_NAME", f"roberta_{TEXT_MODE}_{EPOCHS}ep")
SAVE_DIR = ROOT_DIR / "submit" / "model" / SAVE_NAME
EXCLUDE_AUG_FROM_VAL = os.environ.get("EXCLUDE_AUG_FROM_VAL", "0") == "1"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_last_action(sample):
    actions = get_actions(sample)
    return actions[-1] if actions else "none"


def get_prev_action(sample):
    actions = get_actions(sample)
    return actions[-2] if len(actions) >= 2 else "none"


def get_step_from_id(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return m.group(1) if m else "none"


def get_session_id(sid):
    return re.sub(r"-step_\d+$", "", str(sid))


def split_signature(rows):
    ids = "\n".join(sorted(str(r["id"]) for r in rows))
    return hashlib.sha1(ids.encode("utf-8")).hexdigest()[:12]


def bucket_int(value, cuts):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in cuts:
        if v <= upper:
            return name
    return f"gt_{cuts[-1][1]}"


def clip_text(value, max_chars):
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def arg_path(args):
    """B1 diet: reduce an assistant_action's args dict to just its file-path signal."""
    if not isinstance(args, dict):
        return ""
    for k in ("path", "file", "scope", "target", "pattern"):
        v = args.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def compact_history(sample, max_turns=10):
    hist = sample.get("history", [])[-max_turns:]
    parts = []
    actions = []
    user_utts = []
    last_result = ""

    for h in hist:
        role = h.get("role", "")
        if role == "user":
            txt = str(h.get("content", ""))
            user_utts.append(txt)
            parts.append(f"U: {txt}")
        elif role == "assistant_action":
            name = str(h.get("name", ""))
            args = str(h.get("args", ""))
            result = str(h.get("result_summary", ""))
            actions.append(name)
            last_result = result
            parts.append(f"ACTION: {name} ARGS: {args} RESULT: {result}")

    return {
        "history_text": "\n".join(parts),
        "action_seq": " > ".join(actions),
        "recent_users": "\n".join(user_utts[-3:]),
        "last_result": last_result,
    }


def build_text_full(sample):
    parts = []

    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    parts.append("[CURRENT_PROMPT]")
    parts.append(current_prompt)

    parts.append("[SESSION_META]")
    parts.append(f"user_tier: {meta.get('user_tier', '')}")
    parts.append(f"language_pref: {meta.get('language_pref', '')}")
    parts.append(f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}")
    parts.append(f"turn_index: {meta.get('turn_index', '')}")
    parts.append(f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}")
    parts.append(f"id_step: {get_step_from_id(sample.get('id', ''))}")

    parts.append("[WORKSPACE]")
    parts.append(f"loc: {workspace.get('loc', '')}")
    parts.append(f"git_dirty: {workspace.get('git_dirty', '')}")
    parts.append(f"last_ci_status: {workspace.get('last_ci_status', '')}")

    open_files = workspace.get("open_files", [])
    parts.append("open_files: " + " ".join(map(str, open_files)))

    language_mix = workspace.get("language_mix", {})
    parts.append("language_mix: " + " ".join(
        f"{k}:{v}" for k, v in language_mix.items()
    ))

    recent_history = history[-10:]
    parts.append("[RECENT_HISTORY]")

    recent_actions = []
    for h in recent_history:
        role = h.get("role", "")

        if role == "user":
            parts.append("[USER]")
            parts.append(str(h.get("content", "")))
        elif role == "assistant_action":
            action_name = str(h.get("name", ""))
            recent_actions.append(action_name)

            parts.append("[ASSISTANT_ACTION]")
            parts.append(f"name: {action_name}")
            parts.append(f"args: {h.get('args', '')}")
            parts.append(f"result: {h.get('result_summary', '')}")

    parts.append("[RECENT_ACTIONS]")
    parts.append(" ".join(recent_actions))

    parts.append("[PREV_ACTION]")
    parts.append(get_prev_action(sample))

    parts.append("[LAST_ACTION]")
    parts.append(get_last_action(sample))

    parts.append("[CURRENT_PROMPT_AGAIN]")
    parts.append(current_prompt)

    return "\n".join(parts)


def build_text_router(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    current_prompt = str(sample.get("current_prompt", ""))

    ch = compact_history(sample)
    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = " ".join(map(str, workspace.get("open_files", [])))
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())

    return "\n".join([
        "[ROUTER_TASK]",
        "Predict the next assistant action class.",
        "[CURRENT_USER_REQUEST]",
        current_prompt,
        "[CURRENT_USER_REQUEST_REPEAT]",
        current_prompt,
        "[RECENT_USER_REQUESTS]",
        ch["recent_users"],
        "[ACTION_SEQUENCE]",
        ch["action_seq"],
        "[PREVIOUS_ACTION]",
        get_prev_action(sample),
        "[LAST_ACTION]",
        get_last_action(sample),
        "[LAST_ACTION_RESULT]",
        ch["last_result"],
        "[WORKSPACE_STATE]",
        f"step: {get_step_from_id(sample.get('id', ''))}",
        f"turn_index: {meta.get('turn_index', '')}",
        f"user_tier: {meta.get('user_tier', '')}",
        f"language_pref: {meta.get('language_pref', '')}",
        f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}",
        f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"main_lang: {main_lang}",
        f"language_mix: {lang_mix}",
        f"open_files: {open_files}",
        "[COMPACT_HISTORY]",
        ch["history_text"],
    ])


def build_text_eda_state(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    ch = compact_history(sample)
    user_turns = [
        str(h.get("content", ""))
        for h in history
        if h.get("role") == "user"
    ]

    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = workspace.get("open_files", [])
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())

    prompt_len = len(current_prompt)
    history_len = len(history)
    turn_index = meta.get("turn_index", 0)

    return "\n".join([
        "[EDA_STATE_V2_PRIORITY_FRONT]",
        "[CURRENT_PROMPT]",
        current_prompt,
        "[CURRENT_PROMPT_REPEAT]",
        current_prompt,

        "[HIGH_PRIORITY_EDA_FEATURES]",
        f"prompt_len: {prompt_len}",
        f"prompt_len_bucket: {bucket_int(prompt_len, [('short', 45), ('medium', 80), ('long', 120)])}",
        f"is_first_turn: {str(history_len == 0)}",
        f"history_len: {history_len}",
        f"history_bucket: {bucket_int(history_len, [('h0', 0), ('h1_3', 3), ('h4_7', 7), ('h8_12', 12)])}",
        f"last_action: {get_last_action(sample)}",
        f"prev_action: {get_prev_action(sample)}",
        f"action_sequence: {ch['action_seq']}",
        f"turn_index: {turn_index}",
        f"turn_bucket: {bucket_int(turn_index, [('t1', 1), ('t2_4', 4), ('t5_8', 8), ('t9_12', 12)])}",
        f"step: {get_step_from_id(sample.get('id', ''))}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"ci_failed: {str(workspace.get('last_ci_status', '') == 'failed')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",

        "[MEDIUM_PRIORITY_CONTEXT]",
        "[RECENT_USER_REQUESTS]",
        "\n".join(user_turns[-3:]),
        "[LAST_ACTION_RESULT]",
        ch["last_result"],

        "[LOW_PRIORITY_WORKSPACE]",
        f"num_open_files: {len(open_files)}",
        f"main_lang: {main_lang}",
        f"user_tier: {meta.get('user_tier', '')}",
        f"language_pref: {meta.get('language_pref', '')}",
        f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}",
        f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}",
        "[OPEN_FILES]",
        " ".join(map(str, open_files)),
        f"language_mix: {lang_mix}",
        f"loc: {workspace.get('loc', '')}",
        "[COMPACT_HISTORY]",
        ch["history_text"],
    ])


def build_text_eda_state_light(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    actions = get_actions(sample)
    action_seq = " > ".join(actions[-8:])

    prompt_len = len(current_prompt)
    history_len = len(history)
    turn_index = meta.get("turn_index", 0)

    return "\n".join([
        "[EDA_STATE_LIGHT]",
        "[CURRENT_PROMPT]",
        current_prompt,
        "[CURRENT_PROMPT_REPEAT]",
        current_prompt,
        "[ACTION_STATE]",
        f"last_action: {get_last_action(sample)}",
        f"prev_action: {get_prev_action(sample)}",
        f"action_sequence: {action_seq}",
        "[LIGHT_EDA_FEATURES]",
        f"prompt_len: {prompt_len}",
        f"prompt_len_bucket: {bucket_int(prompt_len, [('short', 45), ('medium', 80), ('long', 120)])}",
        f"is_first_turn: {str(history_len == 0)}",
        f"history_len: {history_len}",
        f"history_bucket: {bucket_int(history_len, [('h0', 0), ('h1_3', 3), ('h4_7', 7), ('h8_12', 12)])}",
        f"turn_index: {turn_index}",
        f"turn_bucket: {bucket_int(turn_index, [('t1', 1), ('t2_4', 4), ('t5_8', 8), ('t9_12', 12)])}",
        f"step: {get_step_from_id(sample.get('id', ''))}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"ci_failed: {str(workspace.get('last_ci_status', '') == 'failed')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
    ])


def build_text_eda_state_mid(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    ch = compact_history(sample)
    actions = get_actions(sample)
    action_seq = " > ".join(actions[-8:])
    user_turns = [
        str(h.get("content", ""))
        for h in history
        if h.get("role") == "user"
    ]

    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = workspace.get("open_files", [])

    prompt_len = len(current_prompt)
    history_len = len(history)
    turn_index = meta.get("turn_index", 0)
    recent_user = clip_text(user_turns[-1], 120) if user_turns else ""

    return "\n".join([
        "[EDA_STATE_MID_PRIORITY_FRONT]",
        "[CURRENT_PROMPT]",
        current_prompt,
        "[CURRENT_PROMPT_REPEAT]",
        current_prompt,
        "[ACTION_STATE]",
        f"last_action: {get_last_action(sample)}",
        f"prev_action: {get_prev_action(sample)}",
        f"action_sequence: {action_seq}",
        "[LIGHT_EDA_FEATURES]",
        f"prompt_len: {prompt_len}",
        f"prompt_len_bucket: {bucket_int(prompt_len, [('short', 45), ('medium', 80), ('long', 120)])}",
        f"is_first_turn: {str(history_len == 0)}",
        f"history_len: {history_len}",
        f"history_bucket: {bucket_int(history_len, [('h0', 0), ('h1_3', 3), ('h4_7', 7), ('h8_12', 12)])}",
        f"turn_index: {turn_index}",
        f"turn_bucket: {bucket_int(turn_index, [('t1', 1), ('t2_4', 4), ('t5_8', 8), ('t9_12', 12)])}",
        f"step: {get_step_from_id(sample.get('id', ''))}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"ci_failed: {str(workspace.get('last_ci_status', '') == 'failed')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",

        "[MEDIUM_PRIORITY_CONTEXT]",
        f"recent_user_count: {len(user_turns)}",
        "[LAST_USER_REQUEST_CLIPPED]",
        recent_user,
        "[LAST_ACTION_RESULT_CLIPPED]",
        clip_text(ch["last_result"], 80),

        "[LOW_PRIORITY_WORKSPACE_COMPACT]",
        f"num_open_files: {len(open_files)}",
        f"main_lang: {main_lang}",
    ])


def build_text_eda_state_diet(sample):
    """B1 (handoff 5-1): diet serialization to fit the full signal in 512 tokens.
    - no CURRENT_PROMPT repeat
    - history args -> file path only
    - result_summary -> 40 chars
    - last 3 user utterances
    - up to 12 history action turns
    target char dist: mean ~366 / p90 ~562.
    """
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    user_turns = [
        str(h.get("content", ""))
        for h in history
        if h.get("role") == "user"
    ]

    hist_lines = []
    for h in history:
        if h.get("role") == "assistant_action":
            name = str(h.get("name", ""))
            path = clip_text(arg_path(h.get("args", {})), 40)
            res = clip_text(h.get("result_summary", ""), 40)
            hist_lines.append(f"{name} | {path} | {res}")
    hist_lines = hist_lines[-12:]

    prompt_len = len(current_prompt)
    history_len = len(history)
    turn_index = meta.get("turn_index", 0)

    return "\n".join([
        "[EDA_STATE_DIET]",
        "[CURRENT_PROMPT]",
        current_prompt,
        "[ACTION_STATE]",
        f"last_action: {get_last_action(sample)}",
        f"prev_action: {get_prev_action(sample)}",
        "[EDA_FEATURES]",
        f"prompt_len: {prompt_len}",
        f"prompt_len_bucket: {bucket_int(prompt_len, [('short', 45), ('medium', 80), ('long', 120)])}",
        f"is_first_turn: {str(history_len == 0)}",
        f"history_len: {history_len}",
        f"turn_index: {turn_index}",
        f"step: {get_step_from_id(sample.get('id', ''))}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
        "[RECENT_USER_REQUESTS]",
        "\n".join(user_turns[-3:]),
        "[COMPACT_HISTORY]",
        "\n".join(hist_lines),
    ])


def build_text_explore_front(sample):
    """Front-load signals that separate read/grep/list/glob.

    This view is a gate experiment for the exploration cluster. It keeps the
    current prompt first, then puts path/pattern/result/open-file signals before
    generic metadata so truncation preserves the fields most relevant to
    read_file vs grep_search vs list_directory vs glob_pattern.
    """
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    history = sample.get("history", []) or []
    current_prompt = str(sample.get("current_prompt", ""))

    actions = get_actions(sample)
    assistant = [
        h for h in history
        if h.get("role") == "assistant_action"
    ]
    last = assistant[-1] if assistant else {}
    prev = assistant[-2] if len(assistant) >= 2 else {}

    def action_line(h, max_result=30):
        if not h:
            return "none"
        name = str(h.get("name", ""))
        args = h.get("args", {}) or {}
        path = clip_text(arg_path(args), 50)
        result = clip_text(h.get("result_summary", ""), max_result)
        return f"{name} | target: {path} | result: {result}"

    recent_action_lines = [action_line(h, 20) for h in assistant[-3:]]
    open_files = " ".join(map(str, workspace.get("open_files", []) or []))
    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"

    lower = current_prompt.lower()
    cue_flags = []
    for name, words in {
        "cue_read": ["open", "read", "cat ", "show", "열어", "읽", "보여", "확인"],
        "cue_grep": ["grep", "search", "find", "where", "usage", "찾", "검색", "어디"],
        "cue_list": ["list", "ls ", "tree", "what's in", "whats in", "뭐 있", "목록", "나열"],
        "cue_glob": ["glob", "pattern", "matching", "all files", "every", "전부", "모든", "패턴"],
    }.items():
        if any(w in lower for w in words):
            cue_flags.append(name)

    return "\n".join([
        "[EXPLORE_FRONT]",
        "[CURRENT_PROMPT]",
        current_prompt,
        "[PROMPT_CUES]",
        " ".join(cue_flags),
        "[LAST_ACTION_DETAIL]",
        action_line(last, 45),
        "[PREV_ACTION_DETAIL]",
        action_line(prev, 25),
        "[OPEN_FILES]",
        clip_text(open_files, 60),
        "[RECENT_ACTION_TARGETS]",
        "\n".join(recent_action_lines),
        "[STATE]",
        f"last_action: {get_last_action(sample)}",
        f"prev_action: {get_prev_action(sample)}",
        f"action_sequence: {' > '.join(actions[-8:])}",
        f"turn_index: {meta.get('turn_index', '')}",
        f"step: {get_step_from_id(sample.get('id', ''))}",
        f"history_len: {len(history)}",
        f"main_lang: {main_lang}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
    ])


def build_text_raw_serialized(sample):
    """Raw-ish serialization baseline.

    This intentionally avoids clever summarization: keep current_prompt up front,
    repeat it, then dump session_meta and recent history with full args and
    result_summary. With MAX_LEN=512, truncation is allowed to decide what fits.
    """
    current_prompt = str(sample.get("current_prompt", ""))
    meta = sample.get("session_meta", {}) or {}
    history = sample.get("history", []) or []
    raw_history = []
    for h in history:
        if h.get("role") == "assistant_action":
            raw_history.append({
                "role": h.get("role"),
                "name": h.get("name"),
                "args": h.get("args", {}),
                "result_summary": h.get("result_summary", ""),
            })
        else:
            raw_history.append({
                "role": h.get("role"),
                "content": h.get("content", ""),
            })
    payload = {
        "id": sample.get("id", ""),
        "session_meta": meta,
        "history": raw_history,
        "current_prompt": current_prompt,
    }
    return "\n".join([
        "[RAW_SERIALIZED]",
        "[CURRENT_PROMPT]",
        current_prompt,
        "[CURRENT_PROMPT_REPEAT]",
        current_prompt,
        "[RAW_JSON]",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    ])


def build_text(sample):
    if TEXT_MODE == "full":
        return build_text_full(sample)
    if TEXT_MODE == "router":
        return build_text_router(sample)
    if TEXT_MODE == "eda_state":
        return build_text_eda_state(sample)
    if TEXT_MODE == "eda_state_light":
        return build_text_eda_state_light(sample)
    if TEXT_MODE == "eda_state_mid":
        return build_text_eda_state_mid(sample)
    if TEXT_MODE == "eda_state_diet":
        return build_text_eda_state_diet(sample)
    if TEXT_MODE == "explore_front":
        return build_text_explore_front(sample)
    if TEXT_MODE == "raw_serialized":
        return build_text_raw_serialized(sample)
    raise ValueError(
        "Unknown TEXT_MODE="
        f"{TEXT_MODE}. Use 'full', 'router', 'eda_state', 'eda_state_light', "
        "'eda_state_mid', 'eda_state_diet', 'explore_front', or 'raw_serialized'."
    )


def load_data():
    samples = []
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    labels = {}
    with open(DATA_DIR / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    texts = []
    y = []
    rows = []
    for s in samples:
        sid = s["id"]
        label_id = LABEL2ID[labels[sid]]
        meta = s.get("session_meta", {})
        workspace = meta.get("workspace", {})
        actions = get_actions(s)
        language_mix = workspace.get("language_mix", {}) or {"none": 1}
        main_lang = max(language_mix, key=language_mix.get)

        texts.append(build_text(s))
        y.append(label_id)
        rows.append({
            "id": sid,
            "session_id": str(s.get("source_session_id") or get_session_id(sid)),
            "label": label_id,
            "sample_weight": float(s.get("sample_weight", 1.0)),
            "is_aug": bool(s.get("is_aug", False) or "__histaug_" in str(sid)),
            "current_prompt": str(s.get("current_prompt", "")),
            "last_action": actions[-1] if actions else "none",
            "prev_action": actions[-2] if len(actions) >= 2 else "none",
            "turn_index": meta.get("turn_index", 0),
            "git_dirty": workspace.get("git_dirty", ""),
            "last_ci_status": workspace.get("last_ci_status", ""),
            "main_lang": main_lang,
            "step_id": get_step_from_id(sid),
        })

    return texts, y, rows


def take_indices(values, indices):
    return [values[i] for i in indices]


def load_fold_ids(fold_file, fold_id):
    if not fold_file:
        raise ValueError("SPLIT_MODE=fold requires FOLD_FILE=/path/to/folds_seed42.csv")

    fold_path = Path(fold_file)
    id_to_fold = {}
    with open(fold_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_fold[row["id"]] = int(row["fold"])
    return id_to_fold, fold_path


def split_train_val(texts, labels, rows, split_mode=SPLIT_MODE, seed=SEED):
    if split_mode == "row":
        return train_test_split(
            texts,
            labels,
            rows,
            test_size=0.2,
            stratify=labels,
            random_state=seed,
        )

    if split_mode == "group":
        groups = [r["session_id"] for r in rows]
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(texts, labels, groups))

        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        return train_texts, val_texts, train_labels, val_labels, train_rows, val_rows

    if split_mode == "fold":
        id_to_fold, _ = load_fold_ids(FOLD_FILE, FOLD_ID)
        missing = [r["id"] for r in rows if r["id"] not in id_to_fold]
        if missing:
            raise ValueError(f"FOLD_FILE is missing {len(missing)} ids. First: {missing[:3]}")

        train_idx = [i for i, r in enumerate(rows) if id_to_fold[r["id"]] != FOLD_ID]
        val_idx = [i for i, r in enumerate(rows) if id_to_fold[r["id"]] == FOLD_ID]
        if EXCLUDE_AUG_FROM_VAL:
            val_idx = [i for i in val_idx if not rows[i].get("is_aug", False)]
        if not val_idx:
            raise ValueError(f"FOLD_ID={FOLD_ID} has no validation rows in {FOLD_FILE}")

        return (
            take_indices(texts, train_idx),
            take_indices(texts, val_idx),
            take_indices(labels, train_idx),
            take_indices(labels, val_idx),
            take_indices(rows, train_idx),
            take_indices(rows, val_idx),
        )

    raise ValueError("Unknown SPLIT_MODE=" f"{split_mode}. Use 'row', 'group', or 'fold'.")


def report_token_length_stats(tokenizer, texts, sample_size=5000):
    sample_texts = texts[:sample_size]
    lengths = [
        len(tokenizer.encode(t, add_special_tokens=True, truncation=False))
        for t in sample_texts
    ]
    if not lengths:
        return

    arr = np.array(lengths)
    over = float((arr > MAX_LEN).mean())
    print("Token length stats:")
    print(
        "  n:",
        len(arr),
        "| mean:",
        f"{arr.mean():.1f}",
        "| p50:",
        int(np.percentile(arr, 50)),
        "| p90:",
        int(np.percentile(arr, 90)),
        "| p95:",
        int(np.percentile(arr, 95)),
        "| p99:",
        int(np.percentile(arr, 99)),
        "| max:",
        int(arr.max()),
    )
    print(f"  over_MAX_LEN({MAX_LEN}): {over:.2%}")


class ActionDataset(Dataset):
    def __init__(self, texts, labels=None, ids=None, weights=None):
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.weights = weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.ids is not None:
            item["id"] = self.ids[idx]
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        if self.weights is not None:
            item["weight"] = float(self.weights[idx])
        return item


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        ids = [x.get("id", "") for x in batch]
        texts = [x["text"] for x in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        if "label" in batch[0]:
            enc["labels"] = torch.tensor(
                [x["label"] for x in batch],
                dtype=torch.long,
            )
        if "weight" in batch[0]:
            enc["sample_weights"] = torch.tensor(
                [x["weight"] for x in batch],
                dtype=torch.float,
            )

        enc["_ids"] = ids
        return enc

    return collate_fn


@torch.no_grad()
def evaluate(model, loader, device, return_logits=False):
    model.eval()

    all_ids = []
    all_preds = []
    all_labels = []
    all_logits = []

    for batch in loader:
        ids = batch.pop("_ids", [])
        labels = batch.pop("labels").to(device)
        # sample_weights 는 loss 가중용이라 평가에선 쓰이지 않는다.
        # 넘기면 HF forward() 가 unexpected kwarg 로 죽는다.
        batch.pop("sample_weights", None)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits.float()
        preds = torch.argmax(logits, dim=-1)

        all_ids.extend(ids)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        if return_logits:
            all_logits.append(logits.cpu().numpy())

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=list(range(len(ALL_CLASSES))),
        average="macro",
        zero_division=0,
    )

    if return_logits:
        return macro_f1, all_ids, all_labels, all_preds, np.concatenate(all_logits, axis=0)

    return macro_f1, all_ids, all_labels, all_preds, None


def save_val_outputs(
    save_dir,
    val_rows,
    val_ids,
    y_true,
    y_pred,
    logits,
    best_f1,
    best_epoch,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    id_to_row = {r["id"]: r for r in val_rows}
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    conf = probs.max(axis=1)

    out_path = save_dir / "val_predictions.csv"
    fieldnames = [
        "id", "true_id", "pred_id", "true_action", "pred_action",
        "confidence", "correct", "last_action", "prev_action",
        "turn_index", "git_dirty", "last_ci_status", "main_lang",
        "step_id", "current_prompt",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid, yt, yp, c in zip(val_ids, y_true, y_pred, conf):
            r = id_to_row[sid]
            writer.writerow({
                "id": sid,
                "true_id": yt,
                "pred_id": yp,
                "true_action": ID2LABEL[int(yt)],
                "pred_action": ID2LABEL[int(yp)],
                "confidence": float(c),
                "correct": int(yt == yp),
                "last_action": r["last_action"],
                "prev_action": r["prev_action"],
                "turn_index": r["turn_index"],
                "git_dirty": r["git_dirty"],
                "last_ci_status": r["last_ci_status"],
                "main_lang": r["main_lang"],
                "step_id": r["step_id"],
                "current_prompt": r["current_prompt"],
            })

    np.save(save_dir / "val_logits.npy", logits)
    with open(save_dir / "val_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_macro_f1": float(best_f1),
                "best_epoch": int(best_epoch),
                "num_val": len(val_ids),
                "all_classes": ALL_CLASSES,
                "model_name": MODEL_NAME,
                "text_mode": TEXT_MODE,
                "max_len": MAX_LEN,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "seed": SEED,
                "split_mode": SPLIT_MODE,
                "fold_file": FOLD_FILE,
                "fold_id": FOLD_ID if SPLIT_MODE == "fold" else None,
                "val_split_signature": split_signature(val_rows),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved validation outputs:")
    print(" -", out_path)
    print(" -", save_dir / "val_logits.npy")
    print(" -", save_dir / "val_info.json")


def save_model_bundle(save_dir, model, tokenizer, epoch, val_f1):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(save_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "all_classes": ALL_CLASSES,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "val_macro_f1": float(val_f1),
                "checkpoint_epoch": int(epoch),
                "model_name": MODEL_NAME,
                "text_mode": TEXT_MODE,
                "max_len": MAX_LEN,
                "epochs": EPOCHS,
                "lr": LR,
                "seed": SEED,
                "split_mode": SPLIT_MODE,
                "fold_file": FOLD_FILE,
                "fold_id": FOLD_ID if SPLIT_MODE == "fold" else None,
                "use_class_weight": USE_CLASS_WEIGHT,
                "weight_mode": WEIGHT_MODE,
                "loss_type": LOSS_TYPE,
                "focal_gamma": FOCAL_GAMMA,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("MAX_LEN:", MAX_LEN, "| BATCH_SIZE:", BATCH_SIZE, "| EPOCHS:", EPOCHS)
    print("TEXT_MODE:", TEXT_MODE)
    print("SPLIT_MODE:", SPLIT_MODE)
    if SPLIT_MODE == "fold":
        print("FOLD_FILE:", FOLD_FILE)
        print("FOLD_ID:", FOLD_ID)
    print("SAVE_DIR:", SAVE_DIR)
    if SAVE_EPOCH_CHECKPOINTS:
        print("SAVE_EPOCH_CHECKPOINTS:", sorted(SAVE_EPOCH_CHECKPOINTS))

    texts, labels, rows = load_data()
    print("samples:", len(texts), "| classes:", len(set(labels)))
    print("label counts:", Counter(labels))

    train_texts, val_texts, train_labels, val_labels, train_rows, val_rows = split_train_val(
        texts,
        labels,
        rows,
    )

    print("train:", len(train_texts), "| val:", len(val_texts))
    train_sessions = {r["session_id"] for r in train_rows}
    val_sessions = {r["session_id"] for r in val_rows}
    print(
        "sessions train:",
        len(train_sessions),
        "| val:",
        len(val_sessions),
        "| overlap:",
        len(train_sessions & val_sessions),
    )
    print("val_split_signature:", split_signature(val_rows))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    report_token_length_stats(tokenizer, texts)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ALL_CLASSES),
        id2label={str(i): label for i, label in ID2LABEL.items()},
        label2id=LABEL2ID,
    )

    model.float()
    model.to(device)

    class_weights = None
    if USE_CLASS_WEIGHT:
        class_weights = compute_class_weights(
            train_labels, num_classes=len(ALL_CLASSES), mode=WEIGHT_MODE
        ).to(device)
        print(f"class weights ({WEIGHT_MODE}):")
        for name, w in zip(ALL_CLASSES, class_weights.cpu().numpy()):
            print(f"  {name:18s} {w:.3f}")

    criterion = None
    if LOSS_TYPE == "focal":
        criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
        print(f"loss: focal (gamma={FOCAL_GAMMA}, "
              f"alpha={'class_weights' if class_weights is not None else 'none'})")
    elif class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("loss: weighted CE")
    else:
        print("loss: CE (model built-in)")

    train_ids = [r["id"] for r in train_rows]
    val_ids = [r["id"] for r in val_rows]
    train_weights = [float(r.get("sample_weight", 1.0)) for r in train_rows]
    val_weights = [float(r.get("sample_weight", 1.0)) for r in val_rows]
    if any(abs(w - 1.0) > 1e-9 for w in train_weights):
        print(
            "sample weights:",
            "train min/max/mean",
            f"{min(train_weights):.3f}/{max(train_weights):.3f}/{np.mean(train_weights):.3f}",
            "| val min/max/mean",
            f"{min(val_weights):.3f}/{max(val_weights):.3f}/{np.mean(val_weights):.3f}",
        )
    train_ds = ActionDataset(train_texts, train_labels, train_ids, train_weights)
    val_ds = ActionDataset(val_texts, val_labels, val_ids, val_weights)

    collate_fn = make_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_epoch = 0
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch.pop("_ids", None)
            labels_batch = batch.pop("labels").to(device)
            sample_weights = batch.pop("sample_weights", None)
            if sample_weights is not None:
                sample_weights = sample_weights.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            if sample_weights is not None:
                outputs = model(**batch)
                if criterion is not None and isinstance(criterion, FocalLoss):
                    logp = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                    logpt = logp.gather(1, labels_batch.unsqueeze(1)).squeeze(1)
                    pt = logpt.exp()
                    per_sample_loss = (1.0 - pt) ** criterion.gamma * (-logpt)
                    if criterion.weight is not None:
                        per_sample_loss = per_sample_loss * criterion.weight.gather(0, labels_batch)
                else:
                    per_sample_loss = torch.nn.functional.cross_entropy(
                        outputs.logits,
                        labels_batch,
                        weight=class_weights,
                        reduction="none",
                    )
                loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
            elif criterion is not None:
                outputs = model(**batch)
                loss = criterion(outputs.logits, labels_batch)
            else:
                outputs = model(**batch, labels=labels_batch)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 200 == 0:
                avg_loss = running_loss / 200
                print(f"epoch {epoch} | step {step}/{len(train_loader)} | loss {avg_loss:.4f}")
                running_loss = 0.0

        val_f1, val_ids_eval, y_true, y_pred, val_logits = evaluate(
            model,
            val_loader,
            device,
            return_logits=True,
        )
        print(f"\nEpoch {epoch} Validation Macro-F1: {val_f1:.4f}")
        print(classification_report(
            y_true,
            y_pred,
            target_names=ALL_CLASSES,
            digits=4,
            zero_division=0,
        ))

        if epoch in SAVE_EPOCH_CHECKPOINTS:
            ckpt_dir = SAVE_DIR / f"checkpoint-epoch{epoch}"
            print(f"Saving epoch checkpoint to {ckpt_dir}")
            save_model_bundle(ckpt_dir, model, tokenizer, epoch, val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            print(
                f"New best F1: {best_f1:.4f} at epoch {best_epoch}. "
                f"Saving to {SAVE_DIR}"
            )

            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)

            with open(SAVE_DIR / "labels.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "all_classes": ALL_CLASSES,
                        "label2id": LABEL2ID,
                        "id2label": ID2LABEL,
                        "best_val_macro_f1": best_f1,
                        "best_epoch": best_epoch,
                        "model_name": MODEL_NAME,
                        "text_mode": TEXT_MODE,
                        "max_len": MAX_LEN,
                        "epochs": EPOCHS,
                        "lr": LR,
                        "seed": SEED,
                        "split_mode": SPLIT_MODE,
                        "fold_file": FOLD_FILE,
                        "fold_id": FOLD_ID if SPLIT_MODE == "fold" else None,
                        "val_split_signature": split_signature(val_rows),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            save_val_outputs(
                save_dir=SAVE_DIR,
                val_rows=val_rows,
                val_ids=val_ids_eval,
                y_true=y_true,
                y_pred=y_pred,
                logits=val_logits,
                best_f1=best_f1,
                best_epoch=best_epoch,
            )

    print("Training done.")
    print("Best Validation Macro-F1:", best_f1)
    print("Best epoch:", best_epoch)
    print("Saved model dir:", SAVE_DIR)


if __name__ == "__main__":
    main()
