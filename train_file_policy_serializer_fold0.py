#!/usr/bin/env python3

import csv
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


SEED = 43

MODEL_NAME = "klue/roberta-base"

MAX_LEN = 512
BATCH_SIZE = 8
GRAD_ACCUM = 2
EPOCHS = 4
LR = 1.5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

FILE_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]

FILE_SET = set(FILE_CLASSES)

FILE_TO_ID = {
    x: i
    for i, x in enumerate(FILE_CLASSES)
}

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

SAVE_DIR = Path(
    "submit/model/"
    "file_policy_serializer_roberta_base_fold0_seed43"
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_text(x, limit=4000):
    x = str(x or "")
    x = re.sub(r"\s+", " ", x)
    return x.strip()[:limit]


def action_records(sample):
    return [
        h
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def user_records(sample):
    return [
        h
        for h in sample.get("history", [])
        if h.get("role") == "user"
    ]


def result_text(record):
    if not record:
        return ""

    for key in [
        "result_summary",
        "result",
        "content",
        "text",
    ]:
        value = record.get(key)

        if value not in [None, ""]:
            return norm_text(value)

    return ""


def arg_first(args, keys):
    if not isinstance(args, dict):
        return ""

    for key in keys:
        value = args.get(key)

        if value not in [None, ""]:
            return norm_text(value, 1000)

    return ""


def flag(text, pattern):
    return int(
        re.search(
            pattern,
            str(text or "").lower(),
        )
        is not None
    )


def build_policy_text(sample):
    history = sample.get("history", [])

    acts = action_records(sample)
    users = user_records(sample)

    current_prompt = norm_text(
        sample.get("current_prompt", ""),
        5000,
    )

    last = acts[-1] if acts else {}
    prev = acts[-2] if len(acts) >= 2 else {}

    last_name = str(
        last.get("name", "NONE")
    )

    prev_name = str(
        prev.get("name", "NONE")
    )

    last_args = last.get("args", {})

    last_path = arg_first(
        last_args,
        [
            "path",
            "file_path",
            "directory",
            "dir",
        ],
    )

    last_query = arg_first(
        last_args,
        [
            "query",
            "pattern",
            "regex",
            "glob",
        ],
    )

    last_result = result_text(last)
    prev_result = result_text(prev)

    recent_actions = [
        str(x.get("name", ""))
        for x in acts[-8:]
    ]

    recent_file_actions = [
        x
        for x in recent_actions
        if x in FILE_TO_ID
    ]

    last_user = ""

    if users:
        u = users[-1]

        last_user = norm_text(
            u.get(
                "content",
                u.get("text", ""),
            ),
            3000,
        )

    prompt_l = current_prompt.lower()
    result_l = last_result.lower()

    known_path = int(
        bool(
            re.search(
                r"(?:/[\w.\-]+){1,}|"
                r"\b[\w.\-]+\.(?:py|js|ts|tsx|jsx|"
                r"java|cpp|c|h|go|rs|json|yaml|yml|"
                r"md|txt|sh|sql)\b",
                current_prompt,
                flags=re.I,
            )
        )
        or bool(last_path)
    )

    search_query_present = int(
        bool(last_query)
        or flag(
            current_prompt,
            r"\b(find|search|locate|grep|reference|"
            r"usage|definition|symbol|pattern)\b",
        )
    )

    target_symbol = flag(
        current_prompt,
        r"\b(function|class|method|variable|symbol|"
        r"definition|reference|usage|implementation)\b",
    )

    directory_intent = flag(
        current_prompt,
        r"\b(directory|directories|folder|folders|"
        r"tree|structure|list files)\b",
    )

    glob_intent = flag(
        current_prompt,
        r"\b(glob|pattern|extension|all files|"
        r"matching files|files matching)\b",
    )

    read_intent = flag(
        current_prompt,
        r"\b(read|open|inspect|show|view|contents?|"
        r"look at)\b",
    )

    search_intent = flag(
        current_prompt,
        r"\b(find|search|locate|grep|references?|"
        r"usages?|definition|where is)\b",
    )

    result_found = flag(
        last_result,
        r"\b(found|match|matches|located|exists?)\b",
    )

    result_empty = flag(
        last_result,
        r"\b(no matches?|not found|empty|zero results?|"
        r"0 matches?)\b",
    )

    result_error = flag(
        last_result,
        r"\b(error|failed|failure|exception|invalid)\b",
    )

    same_action_run = 0

    if recent_actions:
        same_action_run = 1

        for name in reversed(
            recent_actions[:-1]
        ):
            if name != recent_actions[-1]:
                break

            same_action_run += 1

    fields = [
        "[FILE_POLICY_STATE]",
        f"LAST_ACTION={last_name}",
        f"PREV_ACTION={prev_name}",
        "RECENT_ACTIONS="
        + " > ".join(recent_actions),
        "RECENT_FILE_ACTIONS="
        + " > ".join(recent_file_actions),
        f"SAME_ACTION_RUN={same_action_run}",
        f"HISTORY_LEN={len(history)}",
        f"ACTION_COUNT={len(acts)}",
        f"KNOWN_PATH={known_path}",
        f"SEARCH_QUERY_PRESENT={search_query_present}",
        f"TARGET_SYMBOL={target_symbol}",
        f"DIRECTORY_INTENT={directory_intent}",
        f"GLOB_INTENT={glob_intent}",
        f"READ_INTENT={read_intent}",
        f"SEARCH_INTENT={search_intent}",
        f"RESULT_FOUND={result_found}",
        f"RESULT_EMPTY={result_empty}",
        f"RESULT_ERROR={result_error}",
        f"LAST_PATH={last_path or 'NONE'}",
        f"LAST_QUERY={last_query or 'NONE'}",
        f"LAST_RESULT={last_result or 'NONE'}",
        f"PREV_RESULT={prev_result or 'NONE'}",
        "[/FILE_POLICY_STATE]",
        "",
        "[CURRENT_REQUEST]",
        current_prompt,
        "[/CURRENT_REQUEST]",
        "",
        "[LAST_USER_REQUEST]",
        last_user,
        "[/LAST_USER_REQUEST]",
    ]

    return "\n".join(fields)


class TextDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
        )

        encoded["labels"] = int(
            self.labels[idx]
        )

        return encoded


def collate(batch, tokenizer):
    labels = torch.tensor(
        [
            x.pop("labels")
            for x in batch
        ],
        dtype=torch.long,
    )

    encoded = tokenizer.pad(
        batch,
        padding=True,
        return_tensors="pt",
    )

    encoded["labels"] = labels

    return encoded


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    logits_all = []
    labels_all = []

    for batch in loader:
        labels = batch["labels"].numpy()

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        out = model(**batch)

        logits_all.append(
            out.logits.float().cpu().numpy()
        )

        labels_all.append(labels)

    logits = np.concatenate(
        logits_all,
        axis=0,
    )

    labels = np.concatenate(
        labels_all,
        axis=0,
    )

    pred = logits.argmax(axis=1)

    f1 = f1_score(
        labels,
        pred,
        labels=range(4),
        average="macro",
        zero_division=0,
    )

    return f1, labels, pred, logits


def main():
    seed_everything(SEED)

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_ids = set(
        pd.read_csv(VAL_REF)["id"]
        .astype(str)
        .tolist()
    )

    samples = []

    with open(
        "data/train.jsonl",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    with open(
        "data/train_labels.csv",
        encoding="utf-8",
    ) as f:
        label_map = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(f)
        }

    train_samples = []
    train_y = []

    val_samples = []
    val_y = []
    val_sample_ids = []

    for sample in samples:
        sid = str(sample["id"])
        action = label_map[sid]

        if action not in FILE_SET:
            continue

        y = FILE_TO_ID[action]

        if sid in val_ids:
            val_samples.append(sample)
            val_y.append(y)
            val_sample_ids.append(sid)
        else:
            train_samples.append(sample)
            train_y.append(y)

    print("TRAIN:", len(train_y))
    print(
        "TRAIN COUNTS:",
        np.bincount(
            train_y,
            minlength=4,
        ),
    )

    print("VAL:", len(val_y))
    print(
        "VAL COUNTS:",
        np.bincount(
            val_y,
            minlength=4,
        ),
    )

    print("Building policy serializer texts...")

    train_texts = [
        build_policy_text(x)
        for x in train_samples
    ]

    val_texts = [
        build_policy_text(x)
        for x in val_samples
    ]

    lengths = [
        len(x.split())
        for x in train_texts[:5000]
    ]

    print(
        "WORD LEN SAMPLE mean/p50/p90/p95/max:",
        float(np.mean(lengths)),
        float(np.percentile(lengths, 50)),
        float(np.percentile(lengths, 90)),
        float(np.percentile(lengths, 95)),
        int(np.max(lengths)),
    )

    print()
    print("TEXT EXAMPLE")
    print("=" * 100)
    print(train_texts[0][:5000])
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
    )

    train_ds = TextDataset(
        train_texts,
        train_y,
        tokenizer,
    )

    val_ds = TextDataset(
        val_texts,
        val_y,
        tokenizer,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: collate(
            x,
            tokenizer,
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: collate(
            x,
            tokenizer,
        ),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = (
        len(train_loader)
        + GRAD_ACCUM
        - 1
    ) // GRAD_ACCUM

    total_updates = (
        updates_per_epoch
        * EPOCHS
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_updates
            * WARMUP_RATIO
        ),
        num_training_steps=total_updates,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    best_f1 = -1.0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )

        running = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ):
                out = model(**batch)

                loss = (
                    out.loss / GRAD_ACCUM
                )

            scaler.scale(loss).backward()

            if (
                step % GRAD_ACCUM == 0
                or step == len(train_loader)
            ):
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                scheduler.step()

            running += (
                loss.item()
                * GRAD_ACCUM
            )

            if step % 500 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={running / step:.6f}",
                    flush=True,
                )

        f1, y_true, pred, logits = evaluate(
            model,
            val_loader,
            device,
        )

        print()
        print(
            f"EPOCH {epoch} "
            f"VAL FILE MACRO-F1 = {f1:.10f}"
        )

        print(
            classification_report(
                y_true,
                pred,
                target_names=FILE_CLASSES,
                digits=4,
                zero_division=0,
            )
        )

        if f1 > best_f1:
            best_f1 = f1

            model.save_pretrained(
                SAVE_DIR
            )

            tokenizer.save_pretrained(
                SAVE_DIR
            )

            np.save(
                SAVE_DIR / "val_logits.npy",
                logits.astype(np.float32),
            )

            pd.DataFrame({
                "id": val_sample_ids,
                "true_id": y_true,
                "pred_id": pred,
                "true_action": [
                    FILE_CLASSES[x]
                    for x in y_true
                ],
                "pred_action": [
                    FILE_CLASSES[x]
                    for x in pred
                ],
            }).to_csv(
                SAVE_DIR
                / "val_predictions.csv",
                index=False,
            )

            (
                SAVE_DIR / "val_info.json"
            ).write_text(
                json.dumps(
                    {
                        "best_val_macro_f1": (
                            float(best_f1)
                        ),
                        "model": MODEL_NAME,
                        "serializer": (
                            "file_policy_state_v1"
                        ),
                        "max_len": MAX_LEN,
                        "batch_size": BATCH_SIZE,
                        "grad_accum": GRAD_ACCUM,
                        "epochs": EPOCHS,
                        "lr": LR,
                        "seed": SEED,
                        "file_classes": FILE_CLASSES,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

            print(
                "SAVED NEW BEST:",
                SAVE_DIR,
            )

    print()
    print("BEST FILE F1:", best_f1)


if __name__ == "__main__":
    main()
