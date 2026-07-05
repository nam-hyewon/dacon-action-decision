# =============================================================================
# DACON AI Agent Action Decision Prediction — Inference (submission entrypoint)
#
# Pipeline:
#   1. klue/roberta-base classifier (model/roberta_eda_state4, text_mode=eda_state)
#      -> raw fp32 logits over 14 action classes
#   2. Specialist boosters (model/roberta_eda_state4/specialist_boosters/
#      specialists.joblib): per-target one-vs-rest TF-IDF LogisticRegression on
#      router-mode text, frozen after tuning. Adds gamma * safe_logit(p) onto
#      the matching class logit.
#   3. argmax -> action -> submission.csv
#
# IMPORTANT (correctness):
#   - The klue model MUST run in fp32. The specialist gammas were tuned against
#     fp32 validation logits (export_val_roberta.py uses model.float()); running
#     the transformer in fp16 shifts the logit scale and erodes the booster gain.
#   - The klue model uses the `eda_state` text builder; the specialists use the
#     `router` text builder. These are two independent feature pipelines combined
#     at the logit level, so both builders are reproduced verbatim below.
#
# Local validation Macro-F1: base 0.7194 -> +booster 0.7318.
# =============================================================================
import csv
import json
import os
import re

import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
MODEL_DIR = "./model/roberta_eda_state4"
SPECIALISTS_PATH = "./model/roberta_eda_state4/specialist_boosters/specialists.joblib"
DATA_DIR = "./data"
OUT_DIR = "./output"

TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

MAX_LEN = 512
BATCH_SIZE = 32
EPS = 1e-5


# --------------------------------------------------------------------------- #
# Shared history / feature helpers (identical in train_roberta & tune scripts) #
# --------------------------------------------------------------------------- #
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


def bucket_int(value, cuts):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in cuts:
        if v <= upper:
            return name
    return f"gt_{cuts[-1][1]}"


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


# --------------------------------------------------------------------------- #
# klue text builder — must match train_roberta.build_text_eda_state exactly    #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Specialist text builder — must match tune_specialist_boosters.build_text     #
# (mode="router") exactly, since the TF-IDF vectorizer was fit on this text.   #
# --------------------------------------------------------------------------- #
def build_text_router(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    current_prompt = str(sample.get("current_prompt", ""))
    sid = str(sample.get("id", ""))
    ch = compact_history(sample)
    last_action = get_last_action(sample)
    prev_action = get_prev_action(sample)
    step = get_step_from_id(sid)
    lm = workspace.get("language_mix", {}) or {}
    main_lang = max(lm, key=lm.get) if lm else "none"
    open_files = " ".join(map(str, workspace.get("open_files", [])))
    lang_mix = " ".join(f"{k}:{v}" for k, v in lm.items())

    return "\n".join([
        "[ROUTER_TASK] Predict the next assistant action class.",
        "[CURRENT_USER_REQUEST]", current_prompt,
        "[CURRENT_USER_REQUEST_REPEAT]", current_prompt,
        "[RECENT_USER_REQUESTS]", ch["recent_users"],
        "[ACTION_SEQUENCE]", ch["action_seq"],
        "[PREVIOUS_ACTION]", prev_action,
        "[LAST_ACTION]", last_action,
        "[LAST_ACTION_RESULT]", ch["last_result"],
        "[WORKSPACE_STATE]",
        f"step: {step}",
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
        "[COMPACT_HISTORY]", ch["history_text"],
    ])


# --------------------------------------------------------------------------- #
# IO helpers                                                                   #
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


def load_labels(model_dir):
    with open(os.path.join(model_dir, "labels.json"), encoding="utf-8") as f:
        return json.load(f)["all_classes"]


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


def safe_logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
# klue dataset / collate                                                       #
# --------------------------------------------------------------------------- #
class TestDataset(Dataset):
    def __init__(self, samples):
        self.ids = [s.get("id", "") for s in samples]
        self.texts = [build_text_eda_state(s) for s in samples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"id": self.ids[idx], "text": self.texts[idx]}


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        ids = [x["id"] for x in batch]
        texts = [x["text"] for x in batch]
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        )
        return ids, enc
    return collate_fn


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    print("Load test data...")
    samples = load_jsonl(TEST_PATH)
    print(f"samples={len(samples)}")

    all_classes = load_labels(MODEL_DIR)
    print(f"classes={len(all_classes)}")

    # ---- 1) klue transformer logits (fp32) ---------------------------------
    print("Load klue tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model.float()  # fp32 to match the scale the booster gammas were tuned on
    model.to(device)
    model.eval()

    dataset = TestDataset(samples)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=make_collate_fn(tokenizer), num_workers=2,
    )

    ordered_ids = []
    logits_chunks = []
    print("klue inference...")
    with torch.no_grad():
        for step, (ids, batch) in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits_chunks.append(outputs.logits.float().cpu().numpy())
            ordered_ids.extend(ids)
            if step % 100 == 0:
                print(f"step {step}/{len(loader)}")

    logits = np.concatenate(logits_chunks, axis=0).astype(np.float64)
    print("logits:", logits.shape)

    # ---- 2) specialist boosters -------------------------------------------
    print("Load specialists and apply boosters...")
    payload = joblib.load(SPECIALISTS_PATH)
    spec_classes = payload["all_classes"]
    if spec_classes != all_classes:
        raise ValueError(
            "Class order mismatch between klue labels.json and specialists.joblib:\n"
            f"  klue={all_classes}\n  spec={spec_classes}"
        )
    vec = payload["vectorizer"]
    clfs = payload["clfs"]
    gammas = payload["gammas"]

    router_texts = [build_text_router(s) for s in samples]
    # Align specialist scores (built in `samples` order) to the transformer's
    # output order (`ordered_ids`). With shuffle=False these match, but scatter
    # by id keeps it correct regardless.
    id_to_pos = {sid: i for i, sid in enumerate(ordered_ids)}
    sample_ids = [s.get("id", "") for s in samples]
    dst = np.array([id_to_pos[sid] for sid in sample_ids], dtype=np.int64)
    X = vec.transform(router_texts)

    for target, clf in clfs.items():
        tid = all_classes.index(target)
        p = clf.predict_proba(X)[:, 1]
        s = np.clip(safe_logit(p), -8, 8)
        contrib = np.zeros(logits.shape[0], dtype=np.float64)
        np.add.at(contrib, dst, gammas[target] * s)
        logits[:, tid] += contrib

    pred_ids = logits.argmax(axis=1)
    pred_map = {sid: all_classes[int(pid)] for sid, pid in zip(ordered_ids, pred_ids)}

    # ---- 3) write submission ----------------------------------------------
    print("Build submission...")
    fieldnames, rows = load_sample_submission(SAMPLE_SUB_PATH)
    missing = 0
    for row in rows:
        sid = row["id"]
        if sid in pred_map:
            row["action"] = pred_map[sid]
        else:
            missing += 1
    if missing:
        print(f"Warning: missing predictions for {missing} rows")

    save_submission(OUT_PATH, fieldnames, rows)
    print(f"Saved: {OUT_PATH} rows={len(rows)}")
    print("Done.")


if __name__ == "__main__":
    main()
