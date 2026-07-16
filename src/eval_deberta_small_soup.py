import csv
import json
import os
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MODEL_DIR = ROOT / "submit/model/deberta_small_eda_kd_soup_a01_a02_a03"
REFERENCE_VAL_CSV = ROOT / "submit/model/deberta_small_eda_ce_full_lr2e5/val_predictions.csv"

MAX_LEN = int(os.environ.get("MAX_LEN", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {x: i for i, x in enumerate(ALL_CLASSES)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}


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


def clip_text(value, max_chars):
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


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


class ActionDataset(Dataset):
    def __init__(self, rows, labels):
        self.rows = rows
        self.labels = labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]
        return {
            "id": s["id"],
            "text": build_text_eda_state(s),
            "label": int(self.labels[idx]),
        }


def collate_fn(tokenizer):
    def fn(batch):
        ids = [x["id"] for x in batch]
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc["labels"] = labels
        enc["_ids"] = ids
        return enc
    return fn


def load_samples_and_labels():
    samples = []
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    labels = {}
    with open(DATA_DIR / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    sample_by_id = {s["id"]: s for s in samples}

    ref_df = pd.read_csv(REFERENCE_VAL_CSV)
    val_ids = ref_df["id"].astype(str).tolist()

    val_samples = [sample_by_id[sid] for sid in val_ids]
    y = np.array([LABEL2ID[labels[sid]] for sid in val_ids], dtype=np.int64)
    return val_samples, y


@torch.no_grad()
def main():
    print("MODEL_DIR:", MODEL_DIR)
    print("REFERENCE_VAL_CSV:", REFERENCE_VAL_CSV)
    print("MAX_LEN:", MAX_LEN, "BATCH_SIZE:", BATCH_SIZE)

    val_samples, y_true = load_samples_and_labels()
    print("val samples:", len(val_samples), "label counts:", Counter(y_true))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    if device.type == "cuda":
        model = model.half()
    model.to(device)
    model.eval()

    ds = ActionDataset(val_samples, y_true)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn(tokenizer),
        num_workers=2,
    )

    all_ids, all_logits = [], []
    for batch in loader:
        ids = batch.pop("_ids")
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits.float().cpu().numpy()
        all_ids.extend(ids)
        all_logits.append(logits)

    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits.argmax(axis=1)

    mf1 = f1_score(y_true, y_pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print(f"Soup Validation Macro-F1: {mf1:.6f} | Acc: {acc:.6f}")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(ALL_CLASSES))),
        target_names=ALL_CLASSES,
        digits=4,
        zero_division=0,
    ))

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    conf = probs.max(axis=1)

    out_df = pd.DataFrame({
        "id": all_ids,
        "true_id": y_true,
        "pred_id": y_pred,
        "true_action": [ID2LABEL[int(i)] for i in y_true],
        "pred_action": [ID2LABEL[int(i)] for i in y_pred],
        "confidence": conf,
        "correct": (y_true == y_pred).astype(int),
    })

    out_df.to_csv(MODEL_DIR / "val_predictions.csv", index=False)
    np.save(MODEL_DIR / "val_logits.npy", logits.astype(np.float32))

    with open(MODEL_DIR / "val_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_val_macro_f1": float(mf1),
            "accuracy": float(acc),
            "source": "soup a01+a02+a03",
            "all_classes": ALL_CLASSES,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
        }, f, ensure_ascii=False, indent=2)

    print("saved:")
    print(" -", MODEL_DIR / "val_predictions.csv")
    print(" -", MODEL_DIR / "val_logits.npy")
    print(" -", MODEL_DIR / "val_info.json")


if __name__ == "__main__":
    main()
