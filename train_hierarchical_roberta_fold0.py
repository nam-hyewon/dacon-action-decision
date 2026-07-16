#!/usr/bin/env python3

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import submit.script_large_tab_diet_rob8_gated40 as sota


SEED = 43

MODEL_NAME = "klue/roberta-base"

MAX_LEN = 512
BATCH_SIZE = 8
GRAD_ACCUM = 2

EPOCHS = 5
LR = 1.5e-5
HEAD_LR = 5e-5

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

GROUP_LOSS_WEIGHT = 0.35
LOCAL_LOSS_WEIGHT = 1.0

CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

CLASS2ID = {
    x: i
    for i, x in enumerate(CLASSES)
}

GROUPS = {
    "FILE": [
        "read_file",
        "grep_search",
        "list_directory",
        "glob_pattern",
    ],
    "WRITE": [
        "edit_file",
        "write_file",
        "apply_patch",
    ],
    "EXEC": [
        "run_bash",
        "run_tests",
        "lint_or_typecheck",
    ],
    "CONTROL": [
        "ask_user",
        "plan_task",
        "web_search",
        "respond_only",
    ],
}

GROUP_NAMES = list(GROUPS.keys())

GROUP2ID = {
    x: i
    for i, x in enumerate(GROUP_NAMES)
}

CLASS_TO_GROUP = {}
CLASS_TO_LOCAL = {}

for group_name, actions in GROUPS.items():
    gid = GROUP2ID[group_name]

    for local_id, action in enumerate(actions):
        CLASS_TO_GROUP[
            CLASS2ID[action]
        ] = gid

        CLASS_TO_LOCAL[
            CLASS2ID[action]
        ] = local_id


VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

SAVE_DIR = Path(
    "submit/model/"
    "hierarchical_roberta_base_eda_state_fold0_seed43"
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ActionDataset(Dataset):
    def __init__(
        self,
        samples,
        labels,
        tokenizer,
    ):
        self.samples = samples
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = sota.build_text_eda_state(
            self.samples[idx]
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
        )

        y = int(self.labels[idx])

        enc["global_label"] = y
        enc["group_label"] = CLASS_TO_GROUP[y]
        enc["local_label"] = CLASS_TO_LOCAL[y]

        return enc


def make_collate(tokenizer):
    def collate(batch):
        global_labels = torch.tensor(
            [
                x.pop("global_label")
                for x in batch
            ],
            dtype=torch.long,
        )

        group_labels = torch.tensor(
            [
                x.pop("group_label")
                for x in batch
            ],
            dtype=torch.long,
        )

        local_labels = torch.tensor(
            [
                x.pop("local_label")
                for x in batch
            ],
            dtype=torch.long,
        )

        enc = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )

        return (
            enc,
            global_labels,
            group_labels,
            local_labels,
        )

    return collate


class HierarchicalModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.group_head = nn.Linear(
            hidden,
            len(GROUP_NAMES),
        )

        self.local_heads = nn.ModuleDict({
            group_name: nn.Linear(
                hidden,
                len(actions),
            )
            for group_name, actions in GROUPS.items()
        })

    def encode(self, enc):
        out = self.encoder(
            **enc,
            return_dict=True,
        )

        h = out.last_hidden_state[:, 0, :]

        return self.dropout(h)

    def forward(self, enc):
        h = self.encode(enc)

        group_logits = self.group_head(h)

        local_logits = {
            group_name: head(h)
            for group_name, head
            in self.local_heads.items()
        }

        return (
            group_logits,
            local_logits,
        )


def hierarchical_loss(
    group_logits,
    local_logits,
    group_labels,
    local_labels,
):
    group_loss = nn.functional.cross_entropy(
        group_logits,
        group_labels,
    )

    local_loss_sum = torch.zeros(
        (),
        device=group_logits.device,
    )

    local_n = 0

    for group_name in GROUP_NAMES:
        gid = GROUP2ID[group_name]

        mask = group_labels == gid

        if not mask.any():
            continue

        cur_logits = local_logits[group_name][mask]
        cur_labels = local_labels[mask]

        cur_loss = nn.functional.cross_entropy(
            cur_logits,
            cur_labels,
            reduction="sum",
        )

        local_loss_sum = (
            local_loss_sum + cur_loss
        )

        local_n += int(mask.sum().item())

    local_loss = (
        local_loss_sum
        / max(1, local_n)
    )

    loss = (
        GROUP_LOSS_WEIGHT * group_loss
        + LOCAL_LOSS_WEIGHT * local_loss
    )

    return (
        loss,
        group_loss.detach(),
        local_loss.detach(),
    )


def decode(
    group_logits,
    local_logits,
):
    group_pred = group_logits.argmax(
        dim=1
    )

    batch_size = len(group_pred)

    global_pred = torch.empty(
        batch_size,
        dtype=torch.long,
        device=group_pred.device,
    )

    final_scores = torch.full(
        (
            batch_size,
            len(CLASSES),
        ),
        -1e9,
        dtype=group_logits.dtype,
        device=group_logits.device,
    )

    group_logp = torch.log_softmax(
        group_logits,
        dim=1,
    )

    for group_name in GROUP_NAMES:
        gid = GROUP2ID[group_name]

        class_names = GROUPS[group_name]

        class_ids = torch.tensor(
            [
                CLASS2ID[x]
                for x in class_names
            ],
            dtype=torch.long,
            device=group_pred.device,
        )

        local_logp = torch.log_softmax(
            local_logits[group_name],
            dim=1,
        )

        combined = (
            group_logp[:, gid:gid + 1]
            + local_logp
        )

        final_scores[
            :,
            class_ids,
        ] = combined

        mask = group_pred == gid

        if mask.any():
            local_pred = local_logits[
                group_name
            ][mask].argmax(dim=1)

            global_pred[mask] = class_ids[
                local_pred
            ]

    return (
        global_pred,
        group_pred,
        final_scores,
    )


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    y_all = []
    pred_all = []
    group_true_all = []
    group_pred_all = []
    score_all = []

    for (
        enc,
        global_labels,
        group_labels,
        local_labels,
    ) in loader:
        enc = {
            k: v.to(device)
            for k, v in enc.items()
        }

        group_logits, local_logits = model(
            enc
        )

        (
            pred,
            group_pred,
            final_scores,
        ) = decode(
            group_logits,
            local_logits,
        )

        y_all.append(
            global_labels.numpy()
        )

        pred_all.append(
            pred.cpu().numpy()
        )

        group_true_all.append(
            group_labels.numpy()
        )

        group_pred_all.append(
            group_pred.cpu().numpy()
        )

        score_all.append(
            final_scores
            .float()
            .cpu()
            .numpy()
        )

    y = np.concatenate(
        y_all
    )

    pred = np.concatenate(
        pred_all
    )

    group_true = np.concatenate(
        group_true_all
    )

    group_pred = np.concatenate(
        group_pred_all
    )

    scores = np.concatenate(
        score_all
    )

    f1 = f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    acc = accuracy_score(
        y,
        pred,
    )

    group_acc = accuracy_score(
        group_true,
        group_pred,
    )

    return (
        f1,
        acc,
        group_acc,
        y,
        pred,
        scores,
    )


def main():
    seed_everything(SEED)

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_df = pd.read_csv(
        VAL_REF
    )

    val_ids_order = (
        val_df["id"]
        .astype(str)
        .tolist()
    )

    val_ids = set(
        val_ids_order
    )

    samples = []

    with open(
        "data/train.jsonl",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                samples.append(
                    json.loads(line)
                )

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
    ]

    train_samples = [
        by_id[x]
        for x in train_ids
    ]

    val_samples = [
        by_id[x]
        for x in val_ids_order
    ]

    train_y = np.asarray(
        [
            CLASS2ID[
                label_map[x]
            ]
            for x in train_ids
        ],
        dtype=np.int64,
    )

    val_y = np.asarray(
        [
            CLASS2ID[
                label_map[x]
            ]
            for x in val_ids_order
        ],
        dtype=np.int64,
    )

    print(
        "TRAIN:",
        len(train_y)
    )

    print(
        "VAL:",
        len(val_y)
    )

    print(
        "TRAIN COUNTS:",
        np.bincount(
            train_y,
            minlength=len(CLASSES),
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_ds = ActionDataset(
        train_samples,
        train_y,
        tokenizer,
    )

    val_ds = ActionDataset(
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
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "DEVICE:",
        device
    )

    model = HierarchicalModel(
        MODEL_NAME
    )

    model.to(device)

    encoder_params = list(
        model.encoder.parameters()
    )

    head_params = (
        list(
            model.group_head.parameters()
        )
        + list(
            model.local_heads.parameters()
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": LR,
            },
            {
                "params": head_params,
                "lr": HEAD_LR,
            },
        ],
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
            total_updates * WARMUP_RATIO
        ),
        num_training_steps=total_updates,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    best_f1 = -1.0

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        running_group = 0.0
        running_local = 0.0

        for step, (
            enc,
            global_labels,
            group_labels,
            local_labels,
        ) in enumerate(
            train_loader,
            start=1,
        ):
            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            group_labels = group_labels.to(
                device
            )

            local_labels = local_labels.to(
                device
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    device.type == "cuda"
                ),
            ):
                (
                    group_logits,
                    local_logits,
                ) = model(enc)

                (
                    loss,
                    group_loss,
                    local_loss,
                ) = hierarchical_loss(
                    group_logits,
                    local_logits,
                    group_labels,
                    local_labels,
                )

                scaled_loss = (
                    loss / GRAD_ACCUM
                )

            scaler.scale(
                scaled_loss
            ).backward()

            if (
                step % GRAD_ACCUM == 0
                or step == len(train_loader)
            ):
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                scheduler.step()

            running_loss += loss.item()
            running_group += group_loss.item()
            running_local += local_loss.item()

            if step % 500 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={running_loss / step:.6f} "
                    f"group={running_group / step:.6f} "
                    f"local={running_local / step:.6f}",
                    flush=True,
                )

        (
            f1,
            acc,
            group_acc,
            y_true,
            pred,
            scores,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        print()
        print(
            f"EPOCH {epoch} "
            f"VAL MACRO-F1 = {f1:.10f}"
        )

        print(
            f"VAL ACC = {acc:.10f}"
        )

        print(
            f"VAL GROUP ACC = {group_acc:.10f}"
        )

        print(
            classification_report(
                y_true,
                pred,
                target_names=CLASSES,
                digits=4,
                zero_division=0,
            )
        )

        if f1 > best_f1:
            best_f1 = f1

            torch.save(
                model.state_dict(),
                SAVE_DIR
                / "hierarchical_state.pt",
            )

            tokenizer.save_pretrained(
                SAVE_DIR
            )

            np.save(
                SAVE_DIR
                / "val_logits.npy",
                scores.astype(
                    np.float32
                ),
            )

            pd.DataFrame({
                "id": val_ids_order,
                "true_id": y_true,
                "pred_id": pred,
                "true_action": [
                    CLASSES[x]
                    for x in y_true
                ],
                "pred_action": [
                    CLASSES[x]
                    for x in pred
                ],
            }).to_csv(
                SAVE_DIR
                / "val_predictions.csv",
                index=False,
            )

            (
                SAVE_DIR
                / "val_info.json"
            ).write_text(
                json.dumps(
                    {
                        "best_val_macro_f1":
                        float(best_f1),
                        "accuracy":
                        float(acc),
                        "group_accuracy":
                        float(group_acc),
                        "model":
                        MODEL_NAME,
                        "method":
                        "end_to_end_conditional_hierarchy",
                        "text_mode":
                        "eda_state",
                        "max_len":
                        MAX_LEN,
                        "batch_size":
                        BATCH_SIZE,
                        "grad_accum":
                        GRAD_ACCUM,
                        "epochs":
                        EPOCHS,
                        "lr":
                        LR,
                        "head_lr":
                        HEAD_LR,
                        "group_loss_weight":
                        GROUP_LOSS_WEIGHT,
                        "local_loss_weight":
                        LOCAL_LOSS_WEIGHT,
                        "seed":
                        SEED,
                        "groups":
                        GROUPS,
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
    print(
        "BEST VAL MACRO-F1:",
        best_f1,
    )


if __name__ == "__main__":
    main()
