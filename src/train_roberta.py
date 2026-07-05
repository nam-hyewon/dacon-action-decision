### mid4 포함한 결과

import csv
import json
import os
import random
import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = 42
# 한국어 특화 모델
MODEL_NAME = os.environ.get("MODEL_NAME", "klue/roberta-base")
MAX_LEN = int(os.environ.get("MAX_LEN", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
EPOCHS = int(os.environ.get("EPOCHS", 4))
LR = float(os.environ.get("LR", 2e-5))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01)) 
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.06)) 
TEXT_MODE = os.environ.get("TEXT_MODE", "eda_state").strip().lower()

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAVE_NAME = os.environ.get("SAVE_NAME", f"roberta_{TEXT_MODE}_{EPOCHS}ep")
SAVE_DIR = ROOT_DIR / "submit" / "model" / SAVE_NAME


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
    raise ValueError(
        "Unknown TEXT_MODE="
        f"{TEXT_MODE}. Use 'full', 'router', 'eda_state', 'eda_state_light', "
        "or 'eda_state_mid'."
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
            "label": label_id,
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
    def __init__(self, texts, labels=None, ids=None):
        self.texts = texts
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.ids is not None:
            item["id"] = self.ids[idx]
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
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


def save_val_outputs(save_dir, val_rows, val_ids, y_true, y_pred, logits, best_f1):
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
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved validation outputs:")
    print(" -", out_path)
    print(" -", save_dir / "val_logits.npy")
    print(" -", save_dir / "val_info.json")


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("MAX_LEN:", MAX_LEN, "| BATCH_SIZE:", BATCH_SIZE, "| EPOCHS:", EPOCHS)
    print("TEXT_MODE:", TEXT_MODE)
    print("SAVE_DIR:", SAVE_DIR)

    texts, labels, rows = load_data()
    print("samples:", len(texts), "| classes:", len(set(labels)))
    print("label counts:", Counter(labels))

    train_texts, val_texts, train_labels, val_labels, train_rows, val_rows = train_test_split(
        texts,
        labels,
        rows,
        test_size=0.2,
        stratify=labels,
        random_state=SEED,
    )

    print("train:", len(train_texts), "| val:", len(val_texts))

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

    train_ids = [r["id"] for r in train_rows]
    val_ids = [r["id"] for r in val_rows]
    train_ds = ActionDataset(train_texts, train_labels, train_ids)
    val_ds = ActionDataset(val_texts, val_labels, val_ids)

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
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch.pop("_ids", None)
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

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

        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"New best F1: {best_f1:.4f}. Saving to {SAVE_DIR}")

            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)

            with open(SAVE_DIR / "labels.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "all_classes": ALL_CLASSES,
                        "label2id": LABEL2ID,
                        "id2label": ID2LABEL,
                        "best_val_macro_f1": best_f1,
                        "model_name": MODEL_NAME,
                        "text_mode": TEXT_MODE,
                        "max_len": MAX_LEN,
                        "epochs": EPOCHS,
                        "lr": LR,
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
            )

    print("Training done.")
    print("Best Validation Macro-F1:", best_f1)
    print("Saved model dir:", SAVE_DIR)


if __name__ == "__main__":
    main()
