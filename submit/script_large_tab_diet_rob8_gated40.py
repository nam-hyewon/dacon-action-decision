# =============================================================================
# DACON — Inference (gated large + diet + rob8 + tabular)
#
# 1) fast 앙상블 = diet + rob8 + tabular 를 전체 샘플에 실행
# 2) fast 의 margin(1등-2등)이 낮은 하위 GATE_FRACTION(40%) = '애매한' 샘플 선별
# 3) 그 40% 에만 klue/roberta-large 추가 실행 -> full 앙상블로 교체
#      full = 0.241*large + 0.217*diet + 0.098*rob8 + 0.444*tabular
# 4) + class bias -> argmax
#
# Fold0 full-large local Macro-F1: 0.764678. 제출은 시간 안전을 위해 large 40% gated.
# =============================================================================
import csv
import json
import os
import re
from collections import Counter

import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
DATA_DIR = "./data"
OUT_DIR = "./output"
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

MAX_LEN = 512
BATCH_SIZE = 32
ENABLE_LOOKUP_OVERRIDE = os.environ.get("ENABLE_LOOKUP_OVERRIDE", "0") == "1"

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

# fast 멤버 (전체 샘플 실행): (폴더, text_mode, fast가중치, full가중치, temperature)
KLUEBERT = ("./model/kluebert_diet_fold0_fp16", "eda_state_diet", 0.21685297669057865 / 0.7588569128976528, 0.21685297669057865, 1.00)
ROBERTA  = ("./model/roberta_eda_state8_lr15_fold0_fp16", "eda_state", 0.09815319897946724 / 0.7588569128976528, 0.09815319897946724, 0.85)
TABULAR_PATH = "./model/tabular_histgb_fold0/tabular_model.joblib"
TAB_FAST_W, TAB_FULL_W, TAB_TEMP = 0.4438507372276068 / 0.7588569128976528, 0.4438507372276068, 1.00
# gated 멤버 (선별 샘플만 실행)
LARGE = ("./model/klue_roberta_large_eda_state_fold0_fp16", "eda_state", 0.24114308710234722, 1.00)
GATE_FRACTION = float(os.environ.get("GATE_FRACTION", "0.40"))   # margin 하위 40% 에만 large 실행

# class-bias (large+tab+diet+rob8 fold0 튜닝, ALL_CLASSES 순서)
CLASS_BIAS = np.array(
    [0.0, -0.3000011444091797, 0.39999818801879883, 0.6499979496002197,
     -0.3000011444091797, 0.0, 0.0, -0.000001430511474609375,
     -0.2500011920928955, -0.35000109672546387, -0.2500011920928955,
     -0.10000133514404297, 0.4999980926513672, 1.3499972820281982],
    dtype=np.float64,
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #
def get_actions(sample):
    return [str(h.get("name", "")) for h in sample.get("history", [])
            if h.get("role") == "assistant_action"]


def _lookup_parse_id(sid):
    m = re.match(r"(.+)-step_(\d+)$", str(sid))
    if not m:
        return str(sid), None
    return m.group(1), int(m.group(2))


def build_lookup_override(samples):
    lookup = {}
    conflicts = 0
    valid = set(ALL_CLASSES)
    for sample in samples:
        sess, step = _lookup_parse_id(sample.get("id", ""))
        if step is None:
            continue
        acts = [
            str(h.get("name", ""))
            for h in sample.get("history", []) or []
            if h.get("role") == "assistant_action" and h.get("name") in valid
        ]
        start = step - len(acts)
        for i, action in enumerate(acts):
            past_id = f"{sess}-step_{start + i:02d}"
            if past_id in lookup and lookup[past_id] != action:
                conflicts += 1
                continue
            lookup[past_id] = action
    sample_ids = {str(s.get("id", "")) for s in samples}
    hits = {sid: action for sid, action in lookup.items() if sid in sample_ids}
    return hits, conflicts


def get_last_action(sample):
    a = get_actions(sample)
    return a[-1] if a else "none"


def get_prev_action(sample):
    a = get_actions(sample)
    return a[-2] if len(a) >= 2 else "none"


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
    return text if len(text) <= max_chars else text[:max_chars]


def arg_path(args):
    if not isinstance(args, dict):
        return ""
    for key in ("path", "file", "filepath", "scope", "pattern", "cmd"):
        if key in args:
            return str(args.get(key, ""))
    return ""


def compact_history(sample, max_turns=10):
    hist = sample.get("history", [])[-max_turns:]
    parts, actions, user_utts = [], [], []
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


def _step_str(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return m.group(1) if m else "none"


# --------------------------------------------------------------------------- #
# Text builders                                                                #
# --------------------------------------------------------------------------- #
def build_text_eda_state(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))
    ch = compact_history(sample)
    user_turns = [str(h.get("content", "")) for h in history if h.get("role") == "user"]
    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = workspace.get("open_files", [])
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())
    prompt_len = len(current_prompt)
    history_len = len(history)
    turn_index = meta.get("turn_index", 0)
    return "\n".join([
        "[EDA_STATE_V2_PRIORITY_FRONT]",
        "[CURRENT_PROMPT]", current_prompt,
        "[CURRENT_PROMPT_REPEAT]", current_prompt,
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
        f"step: {_step_str(sample.get('id', ''))}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"ci_failed: {str(workspace.get('last_ci_status', '') == 'failed')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
        "[MEDIUM_PRIORITY_CONTEXT]",
        "[RECENT_USER_REQUESTS]", "\n".join(user_turns[-3:]),
        "[LAST_ACTION_RESULT]", ch["last_result"],
        "[LOW_PRIORITY_WORKSPACE]",
        f"num_open_files: {len(open_files)}",
        f"main_lang: {main_lang}",
        f"user_tier: {meta.get('user_tier', '')}",
        f"language_pref: {meta.get('language_pref', '')}",
        f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}",
        f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}",
        "[OPEN_FILES]", " ".join(map(str, open_files)),
        f"language_mix: {lang_mix}",
        f"loc: {workspace.get('loc', '')}",
        "[COMPACT_HISTORY]", ch["history_text"],
    ])


def build_text_eda_state_mid(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))
    ch = compact_history(sample)
    actions = get_actions(sample)
    action_seq = " > ".join(actions[-8:])
    user_turns = [str(h.get("content", "")) for h in history if h.get("role") == "user"]
    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = workspace.get("open_files", [])
    prompt_len = len(current_prompt)
    history_len = len(history)
    turn_index = meta.get("turn_index", 0)
    recent_user = clip_text(user_turns[-1], 120) if user_turns else ""
    return "\n".join([
        "[EDA_STATE_MID_PRIORITY_FRONT]",
        "[CURRENT_PROMPT]", current_prompt,
        "[CURRENT_PROMPT_REPEAT]", current_prompt,
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
        f"step: {_step_str(sample.get('id', ''))}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"ci_failed: {str(workspace.get('last_ci_status', '') == 'failed')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
        "[MEDIUM_PRIORITY_CONTEXT]",
        f"recent_user_count: {len(user_turns)}",
        "[LAST_USER_REQUEST_CLIPPED]", recent_user,
        "[LAST_ACTION_RESULT_CLIPPED]", clip_text(ch["last_result"], 80),
        "[LOW_PRIORITY_WORKSPACE_COMPACT]",
        f"num_open_files: {len(open_files)}",
        f"main_lang: {main_lang}",
    ])


def build_text_eda_state_diet(sample):
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
        f"step: {_step_str(sample.get('id', ''))}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
        "[RECENT_USER_REQUESTS]",
        "\n".join(user_turns[-3:]),
        "[COMPACT_HISTORY]",
        "\n".join(hist_lines),
    ])


TEXT_BUILDERS = {
    "eda_state": build_text_eda_state,
    "eda_state_mid": build_text_eda_state_mid,
    "eda_state_diet": build_text_eda_state_diet,
}


# --------------------------------------------------------------------------- #
# Tabular feature builder (matches train_tabular_router.make_features)          #
# --------------------------------------------------------------------------- #
def _tab_session_id(sid):
    return re.sub(r"-step_\d+$", "", str(sid))


def _tab_step(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return int(m.group(1)) if m else -1


def _get_assistant_actions(sample):
    return [h for h in sample.get("history", []) if h.get("role") == "assistant_action"]


def _result_status(result_summary):
    r = str(result_summary)
    rl = r.lower()
    if (rl.startswith("error") or r.startswith("FAIL") or "command failed" in rl
            or "target string not found" in rl):
        return "fail"
    if re.match(r"^\d+ errors", r):
        return "fail"
    if (r.startswith("0 matches") or r == "no relevant results"
            or r.startswith("0 files") or r.startswith("0 results")):
        return "zero"
    if rl.startswith("ok") or r.startswith("PASS") or r.startswith("exit=0"):
        return "ok"
    return "other"


def _prompt_flags(text):
    lower = str(text).lower()
    return {
        "kw_test": int(any(x in lower for x in ["test", "pytest", "테스트"])),
        "kw_lint": int(any(x in lower for x in ["lint", "typecheck", "type check", "mypy", "ruff"])),
        "kw_search": int(any(x in lower for x in ["grep", "search", "find", "찾", "검색"])),
        "kw_web": int(any(x in lower for x in ["web", "internet", "latest", "recent", "검색해", "찾아봐"])),
        "kw_edit": int(any(x in lower for x in ["edit", "modify", "fix", "수정", "고쳐"])),
        "kw_plan": int(any(x in lower for x in ["plan", "계획", "단계", "쪼개"])),
        "kw_ask": int(any(x in lower for x in ["?", "어떻게", "뭐", "which", "what"])),
    }


def make_features(sample):
    sid = str(sample["id"])
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    current_prompt = str(sample.get("current_prompt", ""))
    history = sample.get("history", []) or []
    actions = get_actions(sample)
    action_counts = Counter(actions)
    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = workspace.get("open_files", []) or []
    last_action = actions[-1] if actions else "none"
    prev_action = actions[-2] if len(actions) >= 2 else "none"
    prev3_action = actions[-3] if len(actions) >= 3 else "none"
    prompt_len = len(current_prompt)
    turn_index = meta.get("turn_index", 0)
    step = _tab_step(sid)
    row = {
        "id": sid,
        "session_id": _tab_session_id(sid),
        "last_action": last_action,
        "prev_action": prev_action,
        "prev3_action": prev3_action,
        "last_prev": f"{prev_action}>{last_action}",
        "action_tail3": ">".join(actions[-3:]) if actions else "none",
        "history_len": len(history),
        "num_actions": len(actions),
        "num_user_turns": sum(1 for h in history if h.get("role") == "user"),
        "prompt_len": prompt_len,
        "prompt_len_bucket": bucket_int(prompt_len, [("short", 45), ("medium", 80), ("long", 120)]),
        "turn_index": turn_index,
        "turn_bucket": bucket_int(turn_index, [("t1", 1), ("t2_4", 4), ("t5_8", 8), ("t9_12", 12)]),
        "step": step,
        "step_bucket": bucket_int(step, [("s1", 1), ("s2_4", 4), ("s5_8", 8), ("s9_12", 12)]),
        "user_tier": str(meta.get("user_tier", "")),
        "language_pref": str(meta.get("language_pref", "")),
        "budget_tokens_remaining": meta.get("budget_tokens_remaining", -1),
        "elapsed_session_sec": meta.get("elapsed_session_sec", -1),
        "git_dirty": str(workspace.get("git_dirty", "")),
        "last_ci_status": str(workspace.get("last_ci_status", "")),
        "main_lang": main_lang,
        "loc": workspace.get("loc", -1),
        "num_open_files": len(open_files),
        "num_langs": len(language_mix),
        "max_lang_ratio": max(language_mix.values()) if language_mix else 0.0,
    }
    for action in ALL_CLASSES:
        row[f"cnt_{action}"] = action_counts.get(action, 0)
    row.update(_prompt_flags(current_prompt))
    aacts = _get_assistant_actions(sample)
    last_result = _result_status(aacts[-1].get("result_summary", "")) if aacts else "none"
    prev_result = _result_status(aacts[-2].get("result_summary", "")) if len(aacts) >= 2 else "none"
    row["last_result"] = last_result
    row["prev_result"] = prev_result
    row["last_act_result"] = f"{last_action}:{last_result}"
    row["last_result_fail"] = int(last_result == "fail")
    row["last_result_zero"] = int(last_result == "zero")
    row["last_result_ok"] = int(last_result == "ok")
    row["last_ci_fail"] = int(str(workspace.get("last_ci_status", "")) == "failing")
    recent5 = Counter(actions[-5:])
    for action in ALL_CLASSES:
        row[f"r5_{action}"] = recent5.get(action, 0)
    run = 0
    for name in reversed(actions):
        if name == last_action:
            run += 1
        else:
            break
    row["same_action_run"] = run
    cp = current_prompt
    row["pf_code"] = int("```" in cp or bool(re.search(r"`[^`]+`", cp)))
    row["pf_path"] = int(bool(re.search(r"[\w/]+\.\w{1,5}\b|[\w-]+/[\w/-]+", cp)))
    row["pf_error"] = int(bool(re.search(r"error|exception|traceback|fail|에러|오류", cp, re.I)))
    row["pf_qmark"] = int("?" in cp)
    return row


# --------------------------------------------------------------------------- #
# IO / math                                                                    #
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON parsing failed: {e}")
    return samples


def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(f"sample_submission columns are not ['id', 'action']: {fieldnames}")
    return fieldnames, rows


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def zscore_rows(logits):
    m = logits.mean(axis=1, keepdims=True)
    s = np.maximum(logits.std(axis=1, keepdims=True), 1e-6)
    return (logits - m) / s


# --------------------------------------------------------------------------- #
# transformer inference on a given sample list -> z-scored logits (aligned)     #
# --------------------------------------------------------------------------- #
class _TextDataset(Dataset):
    def __init__(self, samples, builder):
        self.ids = [s.get("id", "") for s in samples]
        self.texts = [builder(s) for s in samples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {"id": self.ids[i], "text": self.texts[i]}


def _collate(tokenizer):
    def fn(batch):
        ids = [x["id"] for x in batch]
        enc = tokenizer([x["text"] for x in batch], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        return ids, enc
    return fn


def run_transformer(model_dir, text_mode, samples, device):
    """samples 순서대로 z-scored logits (len(samples), 14) 반환."""
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except OSError:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.half().to(device).eval() if device.type == "cuda" else model.float().to(device).eval()
    loader = DataLoader(_TextDataset(samples, TEXT_BUILDERS[text_mode]),
                        batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=_collate(tok), num_workers=2)
    pos = {s["id"]: i for i, s in enumerate(samples)}
    out = np.zeros((len(samples), len(ALL_CLASSES)), dtype=np.float64)
    with torch.no_grad():
        for ids, batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lg = model(**batch).logits.float().cpu().numpy()
            for sid, row in zip(ids, lg):
                out[pos[sid]] = row
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def run_tabular_z(samples):
    import pandas as pd
    fdf = pd.DataFrame([make_features(s) for s in samples])
    drop = {"id", "session_id", "label", "label_name"}
    feature_cols = [c for c in fdf.columns if c not in drop]
    model = joblib.load(TABULAR_PATH)
    probs = model.predict_proba(fdf[feature_cols])
    classes = model.named_steps["clf"].classes_
    full = np.full((len(fdf), len(ALL_CLASSES)), 1e-12, dtype=np.float64)
    full[:, classes] = probs
    full = full / full.sum(axis=1, keepdims=True)
    return np.log(np.maximum(full, 1e-12))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    print("Load test data...")
    samples = load_jsonl(TEST_PATH)
    n = len(samples)
    print(f"samples={n}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type != "cuda":
        print("WARNING: CPU 환경 — 느립니다(10분 제한 위험).")

    # ---- 1) fast: diet + roberta_ep6 + tabular (전체) ---------------------
    print("fast: diet ...");  klu_z = run_transformer(KLUEBERT[0], KLUEBERT[1], samples, device) / KLUEBERT[4]
    print("fast: roberta_ep6 ...");   rob_z = run_transformer(ROBERTA[0],  ROBERTA[1],  samples, device) / ROBERTA[4]
    print("fast: tabular ...");   tab_z = run_tabular_z(samples) / TAB_TEMP

    fast = KLUEBERT[2] * klu_z + ROBERTA[2] * rob_z + TAB_FAST_W * tab_z

    # ---- 2) gate: fast margin 하위 GATE_FRACTION 선별 ----------------------
    sf = np.sort(fast, axis=1)
    margin = sf[:, -1] - sf[:, -2]
    k = max(1, int(round(n * GATE_FRACTION)))
    sel_idx = np.argsort(margin)[:k]          # margin 작은 순 = 애매한 순
    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[sel_idx] = True
    print(f"gate: {k}/{n} ({k/n*100:.1f}%) 샘플에 large 실행")

    # ---- 3) 선별 샘플에만 large 실행 -> full 앙상블 ----------------------
    ens = fast.copy()
    if k > 0:
        sel_samples = [samples[i] for i in sel_idx]
        large_z_sel = run_transformer(LARGE[0], LARGE[1], sel_samples, device) / LARGE[3]  # (k,14)
        full_sel = (LARGE[2] * large_z_sel
                    + KLUEBERT[3] * klu_z[sel_idx]
                    + ROBERTA[3] * rob_z[sel_idx]
                    + TAB_FULL_W * tab_z[sel_idx])
        ens[sel_idx] = full_sel

    # ---- 4) class bias -> argmax ------------------------------------------
    ens = ens + CLASS_BIAS
    pred_ids = ens.argmax(axis=1)
    pred_map = {samples[i]["id"]: ALL_CLASSES[int(pred_ids[i])] for i in range(n)}
    if ENABLE_LOOKUP_OVERRIDE:
        lookup_map, conflicts = build_lookup_override(samples)
        pred_map.update(lookup_map)
        print(
            f"lookup override: hits={len(lookup_map)}/{n} "
            f"({len(lookup_map)/max(n,1)*100:.2f}%) conflicts={conflicts}"
        )
    else:
        print("lookup override: disabled")

    print("Build submission...")
    fieldnames, rows = load_sample_submission(SAMPLE_SUB_PATH)
    missing = 0
    for row in rows:
        if row["id"] in pred_map:
            row["action"] = pred_map[row["id"]]
        else:
            missing += 1
    if missing:
        print(f"Warning: missing predictions for {missing} rows")
    save_submission(OUT_PATH, fieldnames, rows)
    print(f"Saved: {OUT_PATH} rows={len(rows)}")
    print("Done.")


if __name__ == "__main__":
    main()
