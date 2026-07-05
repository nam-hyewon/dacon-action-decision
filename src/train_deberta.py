# import csv
# import json
# import os
# import random
# from pathlib import Path
# from collections import Counter

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, classification_report

# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     get_linear_schedule_with_warmup,
# )

# SEED = int(os.environ.get("SEED", 42))
# MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-small")

# MAX_LEN = int(os.environ.get("MAX_LEN", 512))
# BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
# EPOCHS = int(os.environ.get("EPOCHS", 2))
# LR = float(os.environ.get("LR", 2e-5))
# WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
# WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.06))

# # FULL_TRAIN=1이면 validation 없이 전체 70k로 학습하고 저장
# FULL_TRAIN = int(os.environ.get("FULL_TRAIN", "0"))
# TEXT_MODE = os.environ.get("TEXT_MODE", "full")

# ALL_CLASSES = [
#     "read_file", "grep_search", "list_directory", "glob_pattern",
#     "edit_file", "write_file", "apply_patch",
#     "run_bash", "run_tests", "lint_or_typecheck",
#     "ask_user", "plan_task", "web_search", "respond_only",
# ]

# LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}
# ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# ROOT_DIR = Path(__file__).resolve().parents[1]
# DATA_DIR = ROOT_DIR / "data"
# SAVE_NAME = os.environ.get("SAVE_NAME", "deberta")
# SAVE_DIR = ROOT_DIR / "submit" / "model" / SAVE_NAME


# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def get_last_action(sample):
#     history = sample.get("history", [])
#     actions = [
#         h.get("name", "")
#         for h in history
#         if h.get("role") == "assistant_action"
#     ]
#     return actions[-1] if actions else "none"


# def get_prev_action(sample):
#     history = sample.get("history", [])
#     actions = [
#         h.get("name", "")
#         for h in history
#         if h.get("role") == "assistant_action"
#     ]
#     return actions[-2] if len(actions) >= 2 else "none"


# def build_text(sample):
#     parts = []

#     meta = sample.get("session_meta", {})
#     workspace = meta.get("workspace", {})
#     history = sample.get("history", [])
#     current_prompt = str(sample.get("current_prompt", ""))

#     parts.append("[CURRENT_PROMPT]")
#     parts.append(current_prompt)

#     parts.append("[SESSION_META]")
#     parts.append(f"user_tier: {meta.get('user_tier', '')}")
#     parts.append(f"language_pref: {meta.get('language_pref', '')}")
#     parts.append(f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}")
#     parts.append(f"turn_index: {meta.get('turn_index', '')}")
#     parts.append(f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}")

#     parts.append("[WORKSPACE]")
#     parts.append(f"loc: {workspace.get('loc', '')}")
#     parts.append(f"git_dirty: {workspace.get('git_dirty', '')}")
#     parts.append(f"last_ci_status: {workspace.get('last_ci_status', '')}")

#     open_files = workspace.get("open_files", [])
#     parts.append("open_files: " + " ".join(map(str, open_files)))

#     language_mix = workspace.get("language_mix", {})
#     parts.append("language_mix: " + " ".join(
#         f"{k}:{v}" for k, v in language_mix.items()
#     ))

#     # 기존 8개보다 조금 더 길게 본다. MAX_LEN에서 잘리더라도 최근 맥락이 더 들어감.
#     recent_history = history[-10:]
#     parts.append("[RECENT_HISTORY]")

#     recent_actions = []
#     for h in recent_history:
#         role = h.get("role", "")

#         if role == "user":
#             parts.append("[USER]")
#             parts.append(str(h.get("content", "")))
#         elif role == "assistant_action":
#             action_name = str(h.get("name", ""))
#             recent_actions.append(action_name)

#             parts.append("[ASSISTANT_ACTION]")
#             parts.append(f"name: {action_name}")
#             parts.append(f"args: {h.get('args', '')}")
#             parts.append(f"result: {h.get('result_summary', '')}")

#     parts.append("[RECENT_ACTIONS]")
#     parts.append(" ".join(recent_actions))

#     if recent_actions:
#         parts.append("[LAST_ACTION]")
#         parts.append(recent_actions[-1])

#     # prompt를 한 번 더 넣는 건 유지. current_prompt가 가장 강한 신호일 가능성이 큼.
#     parts.append("[CURRENT_PROMPT_AGAIN]")
#     parts.append(current_prompt)

#     return "\n".join(parts)


# def load_data():
#     samples = []
#     with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 samples.append(json.loads(line))

#     labels = {}
#     with open(DATA_DIR / "train_labels.csv", encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             labels[row["id"]] = row["action"]

#     rows = []
#     for s in samples:
#         sid = s["id"]
#         label = labels[sid]
#         meta = s.get("session_meta", {})
#         workspace = meta.get("workspace", {})
#         current_prompt = str(s.get("current_prompt", ""))

#         rows.append({
#             "id": sid,
#             "text": build_text(s),
#             "label": LABEL2ID[label],
#             "label_name": label,
#             "current_prompt": current_prompt,
#             "last_action": get_last_action(s),
#             "prev_action": get_prev_action(s),
#             "turn_index": meta.get("turn_index", 0),
#             "git_dirty": workspace.get("git_dirty", ""),
#             "last_ci_status": workspace.get("last_ci_status", ""),
#             "main_lang": max(workspace.get("language_mix", {}) or {"none": 1}, key=(workspace.get("language_mix", {}) or {"none": 1}).get),
#         })

#     return rows


# class ActionDataset(Dataset):
#     def __init__(self, rows, use_labels=True):
#         self.rows = rows
#         self.use_labels = use_labels

#     def __len__(self):
#         return len(self.rows)

#     def __getitem__(self, idx):
#         r = self.rows[idx]
#         item = {
#             "id": r["id"],
#             "text": r["text"],
#         }
#         if self.use_labels:
#             item["label"] = int(r["label"])
#         return item


# def make_collate_fn(tokenizer):
#     def collate_fn(batch):
#         ids = [x["id"] for x in batch]
#         texts = [x["text"] for x in batch]

#         enc = tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=MAX_LEN,
#             return_tensors="pt",
#         )

#         if "label" in batch[0]:
#             enc["labels"] = torch.tensor(
#                 [x["label"] for x in batch],
#                 dtype=torch.long,
#             )

#         return ids, enc

#     return collate_fn


# @torch.no_grad()
# def evaluate(model, loader, device, return_logits=False):
#     model.eval()

#     all_ids = []
#     all_preds = []
#     all_labels = []
#     all_logits = []

#     for ids, batch in loader:
#         labels = batch.pop("labels").to(device)
#         batch = {k: v.to(device) for k, v in batch.items()}

#         outputs = model(**batch)
#         logits = outputs.logits.float()
#         preds = torch.argmax(logits, dim=-1)

#         all_ids.extend(ids)
#         all_preds.extend(preds.cpu().numpy().tolist())
#         all_labels.extend(labels.cpu().numpy().tolist())

#         if return_logits:
#             all_logits.append(logits.cpu().numpy())

#     macro_f1 = f1_score(
#         all_labels,
#         all_preds,
#         labels=list(range(len(ALL_CLASSES))),
#         average="macro",
#         zero_division=0,
#     )

#     if return_logits:
#         all_logits = np.concatenate(all_logits, axis=0)
#         return macro_f1, all_ids, all_labels, all_preds, all_logits

#     return macro_f1, all_ids, all_labels, all_preds, None


# def save_val_outputs(save_dir, val_rows, val_ids, y_true, y_pred, logits, best_f1):
#     save_dir.mkdir(parents=True, exist_ok=True)

#     id_to_row = {r["id"]: r for r in val_rows}

#     probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
#     conf = probs.max(axis=1)

#     out_rows = []
#     for sid, yt, yp, c in zip(val_ids, y_true, y_pred, conf):
#         r = id_to_row[sid]
#         out_rows.append({
#             "id": sid,
#             "true_id": yt,
#             "pred_id": yp,
#             "true_action": ID2LABEL[int(yt)],
#             "pred_action": ID2LABEL[int(yp)],
#             "confidence": float(c),
#             "correct": int(yt == yp),
#             "last_action": r["last_action"],
#             "prev_action": r["prev_action"],
#             "turn_index": r["turn_index"],
#             "git_dirty": r["git_dirty"],
#             "last_ci_status": r["last_ci_status"],
#             "main_lang": r["main_lang"],
#             "current_prompt": r["current_prompt"],
#         })

#     pred_df = pd.DataFrame(out_rows)
#     pred_df.to_csv(save_dir / "val_predictions.csv", index=False)

#     np.save(save_dir / "val_logits.npy", logits)

#     with open(save_dir / "val_info.json", "w", encoding="utf-8") as f:
#         json.dump(
#             {
#                 "best_val_macro_f1": float(best_f1),
#                 "num_val": len(val_ids),
#                 "all_classes": ALL_CLASSES,
#             },
#             f,
#             ensure_ascii=False,
#             indent=2,
#         )

#     print("Saved validation outputs:")
#     print(" -", save_dir / "val_predictions.csv")
#     print(" -", save_dir / "val_logits.npy")
#     print(" -", save_dir / "val_info.json")


# def main():
#     set_seed(SEED)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)
#     print("Model:", MODEL_NAME)
#     print("MAX_LEN:", MAX_LEN, "| BATCH_SIZE:", BATCH_SIZE, "| EPOCHS:", EPOCHS)
#     print("LR:", LR, "| WEIGHT_DECAY:", WEIGHT_DECAY, "| WARMUP_RATIO:", WARMUP_RATIO)
#     print("FULL_TRAIN:", FULL_TRAIN)

#     rows = load_data()
#     labels = [r["label"] for r in rows]

#     print("samples:", len(rows), "| classes:", len(set(labels)))
#     print("label counts:", Counter(labels))

#     if FULL_TRAIN:
#         train_rows = rows
#         val_rows = []
#         print("FULL_TRAIN mode: using all rows for training, no validation.")
#     else:
#         train_rows, val_rows = train_test_split(
#             rows,
#             test_size=0.2,
#             stratify=labels,
#             random_state=SEED,
#         )
#         print("train:", len(train_rows), "| val:", len(val_rows))

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=len(ALL_CLASSES),
#         id2label={str(i): label for i, label in ID2LABEL.items()},
#         label2id=LABEL2ID,
#     )

#     model.float()
#     model.to(device)

#     train_ds = ActionDataset(train_rows, use_labels=True)
#     collate_fn = make_collate_fn(tokenizer)

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=2,
#     )

#     if not FULL_TRAIN:
#         val_ds = ActionDataset(val_rows, use_labels=True)
#         val_loader = DataLoader(
#             val_ds,
#             batch_size=BATCH_SIZE * 2,
#             shuffle=False,
#             collate_fn=collate_fn,
#             num_workers=2,
#         )
#     else:
#         val_loader = None

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=LR,
#         weight_decay=WEIGHT_DECAY,
#     )

#     total_steps = len(train_loader) * EPOCHS
#     warmup_steps = int(total_steps * WARMUP_RATIO)

#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=total_steps,
#     )

#     best_f1 = -1.0
#     SAVE_DIR.mkdir(parents=True, exist_ok=True)

#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         running_loss = 0.0

#         for step, (ids, batch) in enumerate(train_loader, start=1):
#             labels_batch = batch.pop("labels").to(device)
#             batch = {k: v.to(device) for k, v in batch.items()}

#             optimizer.zero_grad(set_to_none=True)

#             outputs = model(**batch, labels=labels_batch)
#             loss = outputs.loss

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#             optimizer.step()
#             scheduler.step()

#             running_loss += loss.item()

#             if step % 200 == 0:
#                 avg_loss = running_loss / 200
#                 print(f"epoch {epoch} | step {step}/{len(train_loader)} | loss {avg_loss:.4f}")
#                 running_loss = 0.0

#         if FULL_TRAIN:
#             print(f"Epoch {epoch} done. FULL_TRAIN mode, saving current model.")
#             model.save_pretrained(SAVE_DIR)
#             tokenizer.save_pretrained(SAVE_DIR)

#             with open(SAVE_DIR / "labels.json", "w", encoding="utf-8") as f:
#                 json.dump(
#                     {
#                         "all_classes": ALL_CLASSES,
#                         "label2id": LABEL2ID,
#                         "id2label": ID2LABEL,
#                         "best_val_macro_f1": None,
#                         "full_train": True,
#                         "epoch_saved": epoch,
#                     },
#                     f,
#                     ensure_ascii=False,
#                     indent=2,
#                 )
#             continue

#         val_f1, val_ids, y_true, y_pred, val_logits = evaluate(
#             model,
#             val_loader,
#             device,
#             return_logits=True,
#         )

#         print(f"\nEpoch {epoch} Validation Macro-F1: {val_f1:.4f}")
#         print(classification_report(
#             y_true,
#             y_pred,
#             target_names=ALL_CLASSES,
#             digits=4,
#             zero_division=0,
#         ))

#         if val_f1 > best_f1:
#             best_f1 = val_f1
#             print(f"New best F1: {best_f1:.4f}. Saving to {SAVE_DIR}")

#             model.save_pretrained(SAVE_DIR)
#             tokenizer.save_pretrained(SAVE_DIR)

#             with open(SAVE_DIR / "labels.json", "w", encoding="utf-8") as f:
#                 json.dump(
#                     {
#                         "all_classes": ALL_CLASSES,
#                         "label2id": LABEL2ID,
#                         "id2label": ID2LABEL,
#                         "best_val_macro_f1": best_f1,
#                         "full_train": False,
#                     },
#                     f,
#                     ensure_ascii=False,
#                     indent=2,
#                 )

#             save_val_outputs(
#                 save_dir=SAVE_DIR,
#                 val_rows=val_rows,
#                 val_ids=val_ids,
#                 y_true=y_true,
#                 y_pred=y_pred,
#                 logits=val_logits,
#                 best_f1=best_f1,
#             )

#     print("Training done.")
#     print("Best Validation Macro-F1:", best_f1)
#     print("Saved model dir:", SAVE_DIR)
#     print("SEED:", SEED)
#     print("TEXT_MODE:", TEXT_MODE)


# if __name__ == "__main__":
#     main()


import csv
import json
import os
import random
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# =========================
# Config from env
# =========================
SEED = int(os.environ.get("SEED", 42))
MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-small")

MAX_LEN = int(os.environ.get("MAX_LEN", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
EPOCHS = int(os.environ.get("EPOCHS", 2))
LR = float(os.environ.get("LR", 2e-5))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.06))

# TEXT_MODE:
# - full: previous full-history format
# - router: compressed action-router format
TEXT_MODE = os.environ.get("TEXT_MODE", "full").strip().lower()

# USE_CLASS_WEIGHT=1 enables sqrt inverse-frequency class-weighted CE.
USE_CLASS_WEIGHT = int(os.environ.get("USE_CLASS_WEIGHT", "0"))

# FULL_TRAIN=1이면 validation 없이 전체 70k로 학습하고 저장
FULL_TRAIN = int(os.environ.get("FULL_TRAIN", "0"))

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
SAVE_NAME = os.environ.get("SAVE_NAME", "deberta")
SAVE_DIR = ROOT_DIR / "submit" / "model" / SAVE_NAME


# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_last_action(sample):
    history = sample.get("history", [])
    actions = [
        h.get("name", "")
        for h in history
        if h.get("role") == "assistant_action"
    ]
    return actions[-1] if actions else "none"


def get_prev_action(sample):
    history = sample.get("history", [])
    actions = [
        h.get("name", "")
        for h in history
        if h.get("role") == "assistant_action"
    ]
    return actions[-2] if len(actions) >= 2 else "none"


def get_step_from_id(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return m.group(1) if m else "none"


def compact_history(sample, max_turns=8):
    """
    Compact recent history for router-style input.
    Keeps:
    - recent user utterances
    - assistant action sequence
    - last action result
    - compact action/result trace
    """
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
            actions.append(name)

            args = str(h.get("args", ""))
            result = str(h.get("result_summary", ""))
            last_result = result

            parts.append(f"ACTION: {name} ARGS: {args} RESULT: {result}")

    return {
        "history_text": "\n".join(parts),
        "action_seq": " > ".join(actions),
        "recent_users": "\n".join(user_utts[-3:]),
        "last_result": last_result,
    }


def build_text_full(sample):
    """
    Original full-history style input, with step/prev_action added.
    """
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

    # prompt를 한 번 더 넣는 건 유지. current_prompt가 가장 강한 신호일 가능성이 큼.
    parts.append("[CURRENT_PROMPT_AGAIN]")
    parts.append(current_prompt)

    return "\n".join(parts)


def build_text_router(sample):
    """
    Compressed action-router style input.

    This is designed for action prediction, not free-form conversation reading.
    It emphasizes:
    - current user request
    - recent user requests
    - action sequence
    - last action/result
    - workspace state
    """
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    current_prompt = str(sample.get("current_prompt", ""))

    ch = compact_history(sample)

    last_action = get_last_action(sample)
    prev_action = get_prev_action(sample)
    step = get_step_from_id(sample.get("id", ""))

    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"

    open_files = " ".join(map(str, workspace.get("open_files", [])))
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())

    router_parts = [
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
        prev_action,

        "[LAST_ACTION]",
        last_action,

        "[LAST_ACTION_RESULT]",
        ch["last_result"],

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

        "[COMPACT_HISTORY]",
        ch["history_text"],
    ]

    return "\n".join(router_parts)


def build_text(sample):
    if TEXT_MODE == "router":
        return build_text_router(sample)
    if TEXT_MODE == "full":
        return build_text_full(sample)
    raise ValueError(f"Unknown TEXT_MODE={TEXT_MODE}. Use 'full' or 'router'.")


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

    rows = []
    for s in samples:
        sid = s["id"]
        label = labels[sid]
        meta = s.get("session_meta", {})
        workspace = meta.get("workspace", {})
        current_prompt = str(s.get("current_prompt", ""))

        lm = workspace.get("language_mix", {}) or {"none": 1}
        main_lang = max(lm, key=lm.get)

        rows.append({
            "id": sid,
            "text": build_text(s),
            "label": LABEL2ID[label],
            "label_name": label,
            "current_prompt": current_prompt,
            "last_action": get_last_action(s),
            "prev_action": get_prev_action(s),
            "turn_index": meta.get("turn_index", 0),
            "git_dirty": workspace.get("git_dirty", ""),
            "last_ci_status": workspace.get("last_ci_status", ""),
            "main_lang": main_lang,
            "step_id": get_step_from_id(sid),
        })

    return rows


# =========================
# Dataset / DataLoader
# =========================
class ActionDataset(Dataset):
    def __init__(self, rows, use_labels=True):
        self.rows = rows
        self.use_labels = use_labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        item = {
            "id": r["id"],
            "text": r["text"],
        }
        if self.use_labels:
            item["label"] = int(r["label"])
        return item


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        ids = [x["id"] for x in batch]
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

        return ids, enc

    return collate_fn


# =========================
# Eval / save
# =========================
@torch.no_grad()
def evaluate(model, loader, device, return_logits=False):
    model.eval()

    all_ids = []
    all_preds = []
    all_labels = []
    all_logits = []

    for ids, batch in loader:
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
        all_logits = np.concatenate(all_logits, axis=0)
        return macro_f1, all_ids, all_labels, all_preds, all_logits

    return macro_f1, all_ids, all_labels, all_preds, None


def save_val_outputs(save_dir, val_rows, val_ids, y_true, y_pred, logits, best_f1):
    save_dir.mkdir(parents=True, exist_ok=True)

    id_to_row = {r["id"]: r for r in val_rows}

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    conf = probs.max(axis=1)

    out_rows = []
    for sid, yt, yp, c in zip(val_ids, y_true, y_pred, conf):
        r = id_to_row[sid]
        out_rows.append({
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
            "step_id": r.get("step_id", ""),
            "current_prompt": r["current_prompt"],
        })

    pred_df = pd.DataFrame(out_rows)
    pred_df.to_csv(save_dir / "val_predictions.csv", index=False)

    np.save(save_dir / "val_logits.npy", logits)

    with open(save_dir / "val_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_macro_f1": float(best_f1),
                "num_val": len(val_ids),
                "all_classes": ALL_CLASSES,
                "model_name": MODEL_NAME,
                "max_len": MAX_LEN,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "seed": SEED,
                "text_mode": TEXT_MODE,
                "use_class_weight": bool(USE_CLASS_WEIGHT),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved validation outputs:")
    print(" -", save_dir / "val_predictions.csv")
    print(" -", save_dir / "val_logits.npy")
    print(" -", save_dir / "val_info.json")


def save_labels_json(save_dir, best_f1, full_train, epoch_saved=None):
    with open(save_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "all_classes": ALL_CLASSES,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "best_val_macro_f1": None if best_f1 is None else float(best_f1),
                "full_train": bool(full_train),
                "epoch_saved": epoch_saved,
                "model_name": MODEL_NAME,
                "max_len": MAX_LEN,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "seed": SEED,
                "text_mode": TEXT_MODE,
                "use_class_weight": bool(USE_CLASS_WEIGHT),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("MAX_LEN:", MAX_LEN, "| BATCH_SIZE:", BATCH_SIZE, "| EPOCHS:", EPOCHS)
    print("LR:", LR, "| WEIGHT_DECAY:", WEIGHT_DECAY, "| WARMUP_RATIO:", WARMUP_RATIO)
    print("FULL_TRAIN:", FULL_TRAIN)
    print("SEED:", SEED)
    print("TEXT_MODE:", TEXT_MODE)
    print("USE_CLASS_WEIGHT:", USE_CLASS_WEIGHT)
    print("SAVE_DIR:", SAVE_DIR)

    rows = load_data()
    labels = [r["label"] for r in rows]

    print("samples:", len(rows), "| classes:", len(set(labels)))
    print("label counts:", Counter(labels))

    if FULL_TRAIN:
        train_rows = rows
        val_rows = []
        print("FULL_TRAIN mode: using all rows for training, no validation.")
    else:
        train_rows, val_rows = train_test_split(
            rows,
            test_size=0.2,
            stratify=labels,
            random_state=SEED,
        )
        print("train:", len(train_rows), "| val:", len(val_rows))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ALL_CLASSES),
        id2label={str(i): label for i, label in ID2LABEL.items()},
        label2id=LABEL2ID,
    )

    model.float()
    model.to(device)

    train_ds = ActionDataset(train_rows, use_labels=True)
    collate_fn = make_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    if not FULL_TRAIN:
        val_ds = ActionDataset(val_rows, use_labels=True)
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )
    else:
        val_loader = None

    if USE_CLASS_WEIGHT:
        counts = np.bincount(
            [r["label"] for r in train_rows],
            minlength=len(ALL_CLASSES),
        )
        weights = np.sqrt(len(train_rows) / (len(ALL_CLASSES) * counts))
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)

        print("Class weights:")
        for i, w in enumerate(weights):
            print(ID2LABEL[i], float(w))
    else:
        class_weights = None

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

        for step, (ids, batch) in enumerate(train_loader, start=1):
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)
            logits = outputs.logits

            if class_weights is not None:
                loss = F.cross_entropy(
                    logits,
                    labels_batch,
                    weight=class_weights,
                )
            else:
                loss = F.cross_entropy(
                    logits,
                    labels_batch,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 200 == 0:
                avg_loss = running_loss / 200
                print(f"epoch {epoch} | step {step}/{len(train_loader)} | loss {avg_loss:.4f}")
                running_loss = 0.0

        if FULL_TRAIN:
            print(f"Epoch {epoch} done. FULL_TRAIN mode, saving current model.")
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            save_labels_json(SAVE_DIR, best_f1=None, full_train=True, epoch_saved=epoch)
            continue

        val_f1, val_ids, y_true, y_pred, val_logits = evaluate(
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
            save_labels_json(SAVE_DIR, best_f1=best_f1, full_train=False, epoch_saved=epoch)

            save_val_outputs(
                save_dir=SAVE_DIR,
                val_rows=val_rows,
                val_ids=val_ids,
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
