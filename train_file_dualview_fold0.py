#!/usr/bin/env python3

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


SEED = 43
MODEL_NAME = "klue/roberta-base"

MAX_REQ_LEN = 256
MAX_STATE_LEN = 256

BATCH_SIZE = 8
GRAD_ACCUM = 2
EPOCHS = 4

ENCODER_LR = 1.5e-5
HEAD_LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

FILE_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]
FILE2ID = {x: i for i, x in enumerate(FILE_CLASSES)}

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

SAVE_DIR = Path(
    "submit/model/file_dualview_roberta_base_fold0_seed43"
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_action_name(h):
    if h.get("role") != "assistant":
        return None

    action = h.get("action")

    if isinstance(action, dict):
        return action.get("name")

    if isinstance(action, str):
        return action

    return h.get("action_name")


def get_action_args(h):
    action = h.get("action")

    if isinstance(action, dict):
        return action.get("args", {}) or {}

    return h.get("args", {}) or {}


def request_view(sample):
    current = str(sample.get("current_prompt", ""))

    users = [
        str(h.get("content", ""))
        for h in sample.get("history", []) or []
        if h.get("role") == "user"
    ]

    parts = [
        "[CURRENT_REQUEST]",
        current,
        "[RECENT_USER_REQUESTS]",
    ]

    for text in users[-3:]:
        parts.append(text)

    return "\n".join(parts)


def state_view(sample):
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    history = sample.get("history", []) or []

    assistant_steps = [
        h for h in history
        if h.get("role") == "assistant"
        and get_action_name(h)
    ]

    actions = [
        get_action_name(h)
        for h in assistant_steps
    ]

    last_action = actions[-1] if actions else "none"
    prev_action = actions[-2] if len(actions) >= 2 else "none"

    parts = [
        "[SESSION]",
        f"turn_index={meta.get('turn_index', '')}",
        f"budget_tokens_remaining={meta.get('budget_tokens_remaining', '')}",
        f"elapsed_session_sec={meta.get('elapsed_session_sec', '')}",
        f"user_tier={meta.get('user_tier', '')}",
        f"language_pref={meta.get('language_pref', '')}",
        "[WORKSPACE]",
        f"git_dirty={workspace.get('git_dirty', '')}",
        f"last_ci_status={workspace.get('last_ci_status', '')}",
        f"loc={workspace.get('loc', '')}",
        f"open_files={workspace.get('open_files', [])}",
        f"language_mix={workspace.get('language_mix', {})}",
        "[ACTION_STATE]",
        f"last_action={last_action}",
        f"prev_action={prev_action}",
        f"action_sequence={'>'.join(actions[-8:])}",
        "[RECENT_TRAJECTORY]",
    ]

    for h in assistant_steps[-8:]:
        name = get_action_name(h)
        args = get_action_args(h)
        result = str(h.get("result_summary", ""))

        parts.append(
            f"ACTION={name} | ARGS={json.dumps(args, ensure_ascii=False)} "
            f"| RESULT={result}"
        )

    return "\n".join(parts)


class FileDataset(Dataset):
    def __init__(self, samples, labels, tokenizer):
        self.samples = samples
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        req = self.tokenizer(
            request_view(sample),
            truncation=True,
            max_length=MAX_REQ_LEN,
        )

        state = self.tokenizer(
            state_view(sample),
            truncation=True,
            max_length=MAX_STATE_LEN,
        )

        return {
            "req": req,
            "state": state,
            "label": int(self.labels[idx]),
        }


def make_collate(tokenizer):
    def collate(batch):
        reqs = [x["req"] for x in batch]
        states = [x["state"] for x in batch]

        labels = torch.tensor(
            [x["label"] for x in batch],
            dtype=torch.long,
        )

        req_enc = tokenizer.pad(
            reqs,
            padding=True,
            return_tensors="pt",
        )

        state_enc = tokenizer.pad(
            states,
            padding=True,
            return_tensors="pt",
        )

        return req_enc, state_enc, labels

    return collate


class DualViewModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        fusion_dim = hidden * 4

        self.norm = nn.LayerNorm(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, len(FILE_CLASSES)),
        )

    def encode(self, enc):
        out = self.encoder(
            **enc,
            return_dict=True,
        )

        return out.last_hidden_state[:, 0, :]

    def forward(self, req_enc, state_enc):
        h_req = self.encode(req_enc)
        h_state = self.encode(state_enc)

        fused = torch.cat(
            [
                h_req,
                h_state,
                torch.abs(h_req - h_state),
                h_req * h_state,
            ],
            dim=1,
        )

        fused = self.norm(fused)

        return self.classifier(fused)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    ys = []
    preds = []
    logits_all = []

    for req_enc, state_enc, labels in loader:
        req_enc = {
            k: v.to(device)
            for k, v in req_enc.items()
        }

        state_enc = {
            k: v.to(device)
            for k, v in state_enc.items()
        }

        logits = model(req_enc, state_enc)
        pred = logits.argmax(dim=1)

        ys.append(labels.numpy())
        preds.append(pred.cpu().numpy())
        logits_all.append(
            logits.float().cpu().numpy()
        )

    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    logits = np.concatenate(logits_all)

    f1 = f1_score(
        y,
        pred,
        labels=range(4),
        average="macro",
        zero_division=0,
    )

    acc = accuracy_score(y, pred)

    return f1, acc, y, pred, logits


def main():
    seed_everything(SEED)

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_df = pd.read_csv(VAL_REF)

    val_ids_order = (
        val_df["id"].astype(str).tolist()
    )

    val_ids = set(val_ids_order)

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

    by_id = {
        str(x["id"]): x
        for x in samples
    }

    train_ids = [
        str(x["id"])
        for x in samples
        if str(x["id"]) not in val_ids
        and label_map[str(x["id"])] in FILE2ID
    ]

    file_val_ids = [
        x
        for x in val_ids_order
        if label_map[x] in FILE2ID
    ]

    train_samples = [
        by_id[x]
        for x in train_ids
    ]

    val_samples = [
        by_id[x]
        for x in file_val_ids
    ]

    train_y = np.asarray(
        [
            FILE2ID[label_map[x]]
            for x in train_ids
        ],
        dtype=np.int64,
    )

    val_y = np.asarray(
        [
            FILE2ID[label_map[x]]
            for x in file_val_ids
        ],
        dtype=np.int64,
    )

    print("TRAIN FILE:", len(train_y))
    print("VAL FILE:", len(val_y))
    print("TRAIN COUNTS:", np.bincount(train_y, minlength=4))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_ds = FileDataset(
        train_samples,
        train_y,
        tokenizer,
    )

    val_ds = FileDataset(
        val_samples,
        val_y,
        tokenizer,
    )

    collate = make_collate(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("DEVICE:", device)

    model = DualViewModel(MODEL_NAME).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LR,
            },
            {
                "params": model.classifier.parameters(),
                "lr": HEAD_LR,
            },
            {
                "params": model.norm.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = (
        len(train_loader) + GRAD_ACCUM - 1
    ) // GRAD_ACCUM

    total_updates = (
        updates_per_epoch * EPOCHS
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_updates * WARMUP_RATIO
        ),
        num_training_steps=total_updates,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0

        for step, (
            req_enc,
            state_enc,
            labels,
        ) in enumerate(train_loader, start=1):

            req_enc = {
                k: v.to(device)
                for k, v in req_enc.items()
            }

            state_enc = {
                k: v.to(device)
                for k, v in state_enc.items()
            }

            labels = labels.to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    req_enc,
                    state_enc,
                )

                loss = nn.functional.cross_entropy(
                    logits,
                    labels,
                )

                loss = loss / GRAD_ACCUM

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
                loss.item() * GRAD_ACCUM
            )

            if step % 500 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={running / step:.6f}",
                    flush=True,
                )

        f1, acc, y, pred, logits = evaluate(
            model,
            val_loader,
            device,
        )

        print()
        print(
            f"EPOCH {epoch} VAL FILE MACRO-F1 = {f1:.10f}"
        )
        print(
            f"VAL FILE ACC = {acc:.10f}"
        )

        print(
            classification_report(
                y,
                pred,
                target_names=FILE_CLASSES,
                digits=4,
                zero_division=0,
            )
        )

        if f1 > best_f1:
            best_f1 = f1

            torch.save(
                model.state_dict(),
                SAVE_DIR / "dualview_state.pt",
            )

            tokenizer.save_pretrained(
                SAVE_DIR
            )

            np.save(
                SAVE_DIR / "val_logits.npy",
                logits.astype(np.float32),
            )

            pd.DataFrame({
                "id": file_val_ids,
                "true_id": y,
                "pred_id": pred,
                "true_action": [
                    FILE_CLASSES[x]
                    for x in y
                ],
                "pred_action": [
                    FILE_CLASSES[x]
                    for x in pred
                ],
            }).to_csv(
                SAVE_DIR / "val_predictions.csv",
                index=False,
            )

            (
                SAVE_DIR / "val_info.json"
            ).write_text(
                json.dumps(
                    {
                        "best_val_macro_f1": float(best_f1),
                        "accuracy": float(acc),
                        "model": MODEL_NAME,
                        "method": "dual_view_request_state_late_fusion",
                        "request_max_len": MAX_REQ_LEN,
                        "state_max_len": MAX_STATE_LEN,
                        "batch_size": BATCH_SIZE,
                        "grad_accum": GRAD_ACCUM,
                        "epochs": EPOCHS,
                        "encoder_lr": ENCODER_LR,
                        "head_lr": HEAD_LR,
                        "seed": SEED,
                    },
                    indent=2,
                )
            )

            print("SAVED NEW BEST:", SAVE_DIR)

    print()
    print("BEST FILE MACRO-F1:", best_f1)


if __name__ == "__main__":
    main()
