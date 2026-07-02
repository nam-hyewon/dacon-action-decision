import csv
import json
import os
import random
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


# =========================
# Config
# =========================

SEED = 42

# 첫 실행은 small 추천. 파이프라인 확인 후 base로 변경.
MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-small")
# MODEL_NAME = "microsoft/deberta-v3-base"

MAX_LEN = int(os.environ.get("MAX_LEN", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
EPOCHS = int(os.environ.get("EPOCHS", 2))
LR = float(os.environ.get("LR", 2e-5))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.06))

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
SAVE_DIR = ROOT_DIR / "submit" / "model" / "deberta"


# =========================
# Utils
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_text(sample):
    """
    Transformer용 입력 포맷.
    current_prompt와 최근 history를 특히 강조한다.
    """
    parts = []

    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    # 1. 현재 유저 요청: 가장 중요하므로 맨 앞에 배치
    parts.append("[CURRENT_PROMPT]")
    parts.append(current_prompt)

    # 2. 세션 메타
    parts.append("[SESSION_META]")
    parts.append(f"user_tier: {meta.get('user_tier', '')}")
    parts.append(f"language_pref: {meta.get('language_pref', '')}")
    parts.append(f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}")
    parts.append(f"turn_index: {meta.get('turn_index', '')}")
    parts.append(f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}")

    # 3. 워크스페이스 정보
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

    # 4. 최근 history 중심
    recent_history = history[-8:]
    parts.append("[RECENT_HISTORY]")

    recent_actions = []
    for h in recent_history:
        role = h.get("role", "")

        if role == "user":
            parts.append("[USER]")
            parts.append(str(h.get("content", "")))
        else:
            action_name = str(h.get("name", ""))
            recent_actions.append(action_name)

            parts.append("[ASSISTANT_ACTION]")
            parts.append(f"name: {action_name}")
            parts.append(f"args: {h.get('args', '')}")
            parts.append(f"result: {h.get('result_summary', '')}")

    # 5. 최근 action sequence 강조
    parts.append("[RECENT_ACTIONS]")
    parts.append(" ".join(recent_actions))

    if recent_actions:
        parts.append("[LAST_ACTION]")
        parts.append(recent_actions[-1])

    # 6. current prompt를 마지막에도 한 번 더 넣음
    # 긴 입력에서 앞/뒤 모두에 중요한 정보가 남게 하기 위함
    parts.append("[CURRENT_PROMPT_AGAIN]")
    parts.append(current_prompt)

    return "\n".join(parts)


def load_data():
    train_path = DATA_DIR / "train.jsonl"
    label_path = DATA_DIR / "train_labels.csv"

    samples = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    labels = {}
    with open(label_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    texts = [build_text(s) for s in samples]
    y = [LABEL2ID[labels[s["id"]]] for s in samples]

    return texts, y


class ActionDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        texts = [x["text"] for x in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        if "label" in batch[0]:
            labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
            enc["labels"] = labels

        return enc

    return collate_fn


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=list(range(len(ALL_CLASSES))),
        average="macro",
        zero_division=0,
    )

    return macro_f1, all_labels, all_preds


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("MAX_LEN:", MAX_LEN, "| BATCH_SIZE:", BATCH_SIZE, "| EPOCHS:", EPOCHS)

    texts, labels = load_data()
    print("samples:", len(texts), "| classes:", len(set(labels)))
    print("label counts:", Counter(labels))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED,
    )

    print("train:", len(train_texts), "| val:", len(val_texts))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ALL_CLASSES),
        id2label={str(i): label for i, label in ID2LABEL.items()},
        label2id=LABEL2ID,
    )
    model.to(device)

    train_ds = ActionDataset(train_texts, train_labels)
    val_ds = ActionDataset(val_texts, val_labels)

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

    # class imbalance 대응: weighted CE
    counts = Counter(train_labels)
    total = len(train_labels)
    weights = []
    for i in range(len(ALL_CLASSES)):
        c = counts.get(i, 1)
        weights.append(total / (len(ALL_CLASSES) * c))

    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

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

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(**batch)
                logits = outputs.logits
                loss = criterion(logits, labels_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            if step % 200 == 0:
                avg_loss = running_loss / 200
                print(f"epoch {epoch} | step {step}/{len(train_loader)} | loss {avg_loss:.4f}")
                running_loss = 0.0

        val_f1, y_true, y_pred = evaluate(model, val_loader, device)
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
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    print("Training done.")
    print("Best Validation Macro-F1:", best_f1)
    print("Saved model dir:", SAVE_DIR)


if __name__ == "__main__":
    main()