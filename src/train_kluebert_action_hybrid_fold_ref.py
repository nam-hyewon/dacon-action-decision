from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Reuse the exact text representation that produced the KLUEBERT baseline.
from train_roberta_fold_ref import (
    build_text_eda_state_mid,
    get_actions,
    get_last_action,
    get_prev_action,
    get_step_from_id,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


SEED = int(os.environ.get("SEED", "42"))
MODEL_NAME = os.environ.get("MODEL_NAME", "klue/bert-base")

MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
MAX_ACTION_LEN = int(os.environ.get("MAX_ACTION_LEN", "24"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "8"))

LR = float(os.environ.get("LR", "1.5e-5"))
HEAD_LR = float(os.environ.get("HEAD_LR", "1e-4"))

WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.06"))

ACTION_EMB_DIM = int(os.environ.get("ACTION_EMB_DIM", "64"))
ACTION_HIDDEN = int(os.environ.get("ACTION_HIDDEN", "128"))
FUSION_HIDDEN = int(os.environ.get("FUSION_HIDDEN", "384"))
DROPOUT = float(os.environ.get("DROPOUT", "0.15"))


ALL_CLASSES = [
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

LABEL2ID = {
    label: i
    for i, label in enumerate(ALL_CLASSES)
}

ID2LABEL = {
    i: label
    for label, i in LABEL2ID.items()
}


# Action vocabulary.
# 0 is reserved for PAD.
ACTION2ID = {
    action: i + 1
    for i, action in enumerate(ALL_CLASSES)
}

PAD_ACTION_ID = 0

NUM_ACTION_IDS = len(ACTION2ID) + 1


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

SAVE_NAME = os.environ.get(
    "SAVE_NAME",
    "kluebert_action_hybrid_fold0",
)

SAVE_DIR = (
    ROOT_DIR
    / "submit"
    / "model"
    / SAVE_NAME
)


# Numeric features used by the structured branch.
NUMERIC_FEATURE_NAMES = [
    "prompt_len",
    "history_len",
    "action_count",
    "turn_index",
    "budget_tokens_log1p",
    "elapsed_session_log1p",
    "step_id",
    "num_open_files",
    "git_dirty",
    "ci_failed",
    "ci_success",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_step_num(sample_id: str) -> float:
    match = re.search(
        r"step_(\d+)",
        str(sample_id),
    )

    if match is None:
        return 0.0

    return float(match.group(1))


def build_numeric_features(sample: dict) -> np.ndarray:
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}

    history = sample.get("history", []) or []
    actions = get_actions(sample)

    current_prompt = str(
        sample.get("current_prompt", "")
    )

    budget_tokens = max(
        0.0,
        safe_float(
            meta.get("budget_tokens_remaining", 0)
        ),
    )

    elapsed = max(
        0.0,
        safe_float(
            meta.get("elapsed_session_sec", 0)
        ),
    )

    git_dirty_raw = str(
        workspace.get("git_dirty", "")
    ).strip().lower()

    git_dirty = float(
        git_dirty_raw
        in {
            "1",
            "true",
            "yes",
            "dirty",
        }
    )

    ci_status = str(
        workspace.get("last_ci_status", "")
    ).strip().lower()

    values = [
        float(len(current_prompt)),
        float(len(history)),
        float(len(actions)),
        safe_float(
            meta.get("turn_index", 0)
        ),
        float(np.log1p(budget_tokens)),
        float(np.log1p(elapsed)),
        parse_step_num(
            sample.get("id", "")
        ),
        float(
            len(
                workspace.get(
                    "open_files",
                    [],
                )
                or []
            )
        ),
        git_dirty,
        float(ci_status == "failed"),
        float(
            ci_status
            in {
                "success",
                "passed",
                "pass",
            }
        ),
    ]

    return np.asarray(
        values,
        dtype=np.float32,
    )


def encode_action_sequence(sample: dict) -> list[int]:
    actions = get_actions(sample)[-MAX_ACTION_LEN:]

    return [
        ACTION2ID.get(action, PAD_ACTION_ID)
        for action in actions
    ]


def load_data():
    samples = []

    with open(
        DATA_DIR / "train.jsonl",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                samples.append(
                    json.loads(line)
                )


    labels = {}

    with open(
        DATA_DIR / "train_labels.csv",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):
            labels[str(row["id"])] = row["action"]


    items = []

    for sample in samples:
        sid = str(sample["id"])

        action = labels[sid]
        label_id = LABEL2ID[action]

        actions = get_actions(sample)

        meta = sample.get(
            "session_meta",
            {},
        ) or {}

        workspace = meta.get(
            "workspace",
            {},
        ) or {}

        language_mix = (
            workspace.get(
                "language_mix",
                {},
            )
            or {"none": 1}
        )

        main_lang = max(
            language_mix,
            key=language_mix.get,
        )


        items.append(
            {
                "id": sid,
                "text": build_text_eda_state_mid(
                    sample
                ),
                "label": label_id,
                "action_ids": encode_action_sequence(
                    sample
                ),
                "numeric": build_numeric_features(
                    sample
                ),
                "last_action_id": ACTION2ID.get(
                    get_last_action(sample),
                    PAD_ACTION_ID,
                ),
                "prev_action_id": ACTION2ID.get(
                    get_prev_action(sample),
                    PAD_ACTION_ID,
                ),
                "current_prompt": str(
                    sample.get(
                        "current_prompt",
                        "",
                    )
                ),
                "last_action": (
                    actions[-1]
                    if actions
                    else "none"
                ),
                "prev_action": (
                    actions[-2]
                    if len(actions) >= 2
                    else "none"
                ),
                "turn_index": meta.get(
                    "turn_index",
                    0,
                ),
                "git_dirty": workspace.get(
                    "git_dirty",
                    "",
                ),
                "last_ci_status": workspace.get(
                    "last_ci_status",
                    "",
                ),
                "main_lang": main_lang,
                "step_id": get_step_from_id(sid),
            }
        )


    return items


class HybridDataset(Dataset):
    def __init__(
        self,
        items: list[dict],
        numeric_mean: np.ndarray,
        numeric_std: np.ndarray,
    ):
        self.items = items

        self.numeric_mean = numeric_mean.astype(
            np.float32
        )

        self.numeric_std = numeric_std.astype(
            np.float32
        )


    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        item = self.items[idx]

        numeric = (
            item["numeric"]
            - self.numeric_mean
        ) / self.numeric_std

        return {
            "id": item["id"],
            "text": item["text"],
            "label": int(item["label"]),
            "action_ids": item["action_ids"],
            "numeric": numeric.astype(
                np.float32
            ),
            "last_action_id": int(
                item["last_action_id"]
            ),
            "prev_action_id": int(
                item["prev_action_id"]
            ),
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        ids = [
            x["id"]
            for x in batch
        ]

        texts = [
            x["text"]
            for x in batch
        ]


        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )


        batch_size = len(batch)

        max_action_len = max(
            1,
            min(
                MAX_ACTION_LEN,
                max(
                    len(x["action_ids"])
                    for x in batch
                ),
            ),
        )


        action_ids = torch.full(
            (
                batch_size,
                max_action_len,
            ),
            PAD_ACTION_ID,
            dtype=torch.long,
        )

        action_lengths = torch.zeros(
            batch_size,
            dtype=torch.long,
        )


        for i, item in enumerate(batch):
            seq = item["action_ids"][-max_action_len:]

            length = len(seq)

            if length > 0:
                action_ids[
                    i,
                    :length,
                ] = torch.tensor(
                    seq,
                    dtype=torch.long,
                )

            action_lengths[i] = length


        enc["action_ids"] = action_ids

        enc["action_lengths"] = action_lengths

        enc["numeric_features"] = torch.tensor(
            np.stack(
                [
                    x["numeric"]
                    for x in batch
                ]
            ),
            dtype=torch.float32,
        )

        enc["last_action_ids"] = torch.tensor(
            [
                x["last_action_id"]
                for x in batch
            ],
            dtype=torch.long,
        )

        enc["prev_action_ids"] = torch.tensor(
            [
                x["prev_action_id"]
                for x in batch
            ],
            dtype=torch.long,
        )

        enc["labels"] = torch.tensor(
            [
                x["label"]
                for x in batch
            ],
            dtype=torch.long,
        )

        enc["_ids"] = ids

        return enc


    return collate_fn


class ActionHistoryHybridModel(nn.Module):
    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()


        self.text_encoder = AutoModel.from_pretrained(
            model_name
        )

        text_hidden = (
            self.text_encoder.config.hidden_size
        )


        self.action_embedding = nn.Embedding(
            NUM_ACTION_IDS,
            ACTION_EMB_DIM,
            padding_idx=PAD_ACTION_ID,
        )


        self.action_gru = nn.GRU(
            input_size=ACTION_EMB_DIM,
            hidden_size=ACTION_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )


        self.numeric_encoder = nn.Sequential(
            nn.Linear(
                len(NUMERIC_FEATURE_NAMES),
                64,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                64,
                64,
            ),
            nn.GELU(),
        )


        fusion_dim = (
            text_hidden
            + ACTION_HIDDEN
            + ACTION_EMB_DIM
            + ACTION_EMB_DIM
            + 64
        )


        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(
                fusion_dim,
                FUSION_HIDDEN,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                FUSION_HIDDEN,
                len(ALL_CLASSES),
            ),
        )


    def forward(
        self,
        input_ids,
        attention_mask,
        action_ids,
        action_lengths,
        numeric_features,
        last_action_ids,
        prev_action_ids,
        token_type_ids=None,
        labels=None,
    ):
        text_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            text_kwargs[
                "token_type_ids"
            ] = token_type_ids


        text_outputs = self.text_encoder(
            **text_kwargs
        )


        text_vec = (
            text_outputs.last_hidden_state[
                :,
                0,
                :,
            ]
        )


        action_emb = self.action_embedding(
            action_ids
        )


        # GRU output for every action position.
        action_output, _ = self.action_gru(
            action_emb
        )


        safe_lengths = torch.clamp(
            action_lengths,
            min=1,
        )


        last_position = (
            safe_lengths - 1
        ).view(-1, 1, 1)


        last_position = last_position.expand(
            -1,
            1,
            ACTION_HIDDEN,
        )


        action_vec = torch.gather(
            action_output,
            dim=1,
            index=last_position,
        ).squeeze(1)


        # Explicit exact-state embeddings.
        last_action_vec = self.action_embedding(
            last_action_ids
        )

        prev_action_vec = self.action_embedding(
            prev_action_ids
        )


        numeric_vec = self.numeric_encoder(
            numeric_features
        )


        fused = torch.cat(
            [
                text_vec,
                action_vec,
                last_action_vec,
                prev_action_vec,
                numeric_vec,
            ],
            dim=1,
        )


        logits = self.classifier(
            fused
        )


        loss = None

        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits,
                labels,
            )


        return {
            "loss": loss,
            "logits": logits,
        }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    all_ids = []
    all_labels = []
    all_preds = []
    all_logits = []


    for batch in loader:
        ids = batch.pop("_ids")

        labels = batch["labels"]

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }


        outputs = model(
            **batch
        )

        logits = outputs["logits"]

        preds = logits.argmax(
            dim=1
        )


        all_ids.extend(ids)

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_preds.extend(
            preds.cpu().numpy().tolist()
        )

        all_logits.append(
            logits
            .float()
            .cpu()
            .numpy()
        )


    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=list(
            range(len(ALL_CLASSES))
        ),
        average="macro",
        zero_division=0,
    )


    return (
        float(macro_f1),
        all_ids,
        all_labels,
        all_preds,
        np.concatenate(
            all_logits,
            axis=0,
        ),
    )


def save_checkpoint(
    model,
    tokenizer,
    numeric_mean,
    numeric_std,
    best_f1,
    best_epoch,
):
    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.save(
        model.state_dict(),
        SAVE_DIR / "hybrid_model.pt",
    )


    tokenizer.save_pretrained(
        SAVE_DIR
    )


    model.text_encoder.config.save_pretrained(
        SAVE_DIR
    )


    np.save(
        SAVE_DIR / "numeric_mean.npy",
        numeric_mean,
    )

    np.save(
        SAVE_DIR / "numeric_std.npy",
        numeric_std,
    )


    with open(
        SAVE_DIR / "hybrid_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "all_classes": ALL_CLASSES,
                "action2id": ACTION2ID,
                "numeric_feature_names":
                    NUMERIC_FEATURE_NAMES,
                "max_len": MAX_LEN,
                "max_action_len":
                    MAX_ACTION_LEN,
                "action_emb_dim":
                    ACTION_EMB_DIM,
                "action_hidden":
                    ACTION_HIDDEN,
                "fusion_hidden":
                    FUSION_HIDDEN,
                "dropout": DROPOUT,
                "best_val_macro_f1":
                    float(best_f1),
                "best_epoch":
                    int(best_epoch),
                "seed": SEED,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_val_outputs(
    val_items,
    val_ids,
    y_true,
    y_pred,
    logits,
    best_f1,
    best_epoch,
):
    id_to_item = {
        item["id"]: item
        for item in val_items
    }


    probs = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()


    df = pd.DataFrame(
        {
            "id": val_ids,
            "true_id": y_true,
            "pred_id": y_pred,
            "true_action": [
                ID2LABEL[int(x)]
                for x in y_true
            ],
            "pred_action": [
                ID2LABEL[int(x)]
                for x in y_pred
            ],
            "confidence": probs.max(
                axis=1
            ),
            "correct": (
                np.asarray(y_true)
                == np.asarray(y_pred)
            ).astype(int),
            "current_prompt": [
                id_to_item[sid][
                    "current_prompt"
                ]
                for sid in val_ids
            ],
        }
    )


    df.to_csv(
        SAVE_DIR / "val_predictions.csv",
        index=False,
    )


    np.save(
        SAVE_DIR / "val_logits.npy",
        logits,
    )


    with open(
        SAVE_DIR / "val_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "best_val_macro_f1":
                    float(best_f1),
                "best_epoch":
                    int(best_epoch),
                "num_val":
                    len(val_ids),
                "all_classes":
                    ALL_CLASSES,
                "model_name":
                    MODEL_NAME,
                "architecture":
                    "text_cls_action_gru_numeric_mlp",
                "epochs":
                    EPOCHS,
                "lr":
                    LR,
                "head_lr":
                    HEAD_LR,
                "seed":
                    SEED,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    set_seed(SEED)


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("Device:", device)

    print("Model:", MODEL_NAME)

    print("Architecture:")
    print(
        "  text CLS + action GRU "
        "+ last/prev action embeddings "
        "+ numeric MLP"
    )

    print(
        "EPOCHS:",
        EPOCHS,
        "| LR:",
        LR,
        "| HEAD_LR:",
        HEAD_LR,
    )


    items = load_data()


    print(
        "samples:",
        len(items),
    )

    print(
        "label counts:",
        Counter(
            item["label"]
            for item in items
        ),
    )


    ref_val_path = Path(
        os.environ["REF_VAL_PATH"]
    )

    if not ref_val_path.is_absolute():
        ref_val_path = (
            ROOT_DIR / ref_val_path
        )


    ref_val_df = pd.read_csv(
        ref_val_path
    )


    val_ids_order = (
        ref_val_df["id"]
        .astype(str)
        .tolist()
    )


    val_ids_set = set(
        val_ids_order
    )


    id_to_item = {
        item["id"]: item
        for item in items
    }


    missing = [
        sid
        for sid in val_ids_order
        if sid not in id_to_item
    ]


    if missing:
        raise ValueError(
            f"missing validation IDs: "
            f"{len(missing)} "
            f"examples={missing[:5]}"
        )


    train_items = [
        item
        for item in items
        if item["id"]
        not in val_ids_set
    ]


    val_items = [
        id_to_item[sid]
        for sid in val_ids_order
    ]


    print(
        "train:",
        len(train_items),
        "| val:",
        len(val_items),
    )


    train_numeric = np.stack(
        [
            item["numeric"]
            for item in train_items
        ]
    ).astype(np.float32)


    numeric_mean = train_numeric.mean(
        axis=0
    )


    numeric_std = train_numeric.std(
        axis=0
    )


    numeric_std = np.maximum(
        numeric_std,
        1e-6,
    )


    print()
    print("NUMERIC FEATURE STATS")

    for (
        name,
        mean,
        std,
    ) in zip(
        NUMERIC_FEATURE_NAMES,
        numeric_mean,
        numeric_std,
    ):
        print(
            f"{name:25s} "
            f"mean={mean:.4f} "
            f"std={std:.4f}"
        )


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )


    train_ds = HybridDataset(
        train_items,
        numeric_mean,
        numeric_std,
    )


    val_ds = HybridDataset(
        val_items,
        numeric_mean,
        numeric_std,
    )


    collate_fn = make_collate_fn(
        tokenizer
    )


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


    model = ActionHistoryHybridModel(
        MODEL_NAME
    )


    model.to(device)


    encoder_params = list(
        model.text_encoder.parameters()
    )


    head_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith(
            "text_encoder."
        )
    ]


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


    total_steps = (
        len(train_loader) * EPOCHS
    )


    warmup_steps = int(
        total_steps * WARMUP_RATIO
    )


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


    best_f1 = -1.0
    best_epoch = -1


    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        running_loss = 0.0


        for step, batch in enumerate(
            train_loader,
            start=1,
        ):
            batch.pop(
                "_ids",
                None,
            )


            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }


            optimizer.zero_grad(
                set_to_none=True
            )


            outputs = model(
                **batch
            )


            loss = outputs["loss"]


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )


            optimizer.step()

            scheduler.step()


            running_loss += float(
                loss.item()
            )


            if step % 500 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(train_loader)} "
                    f"loss="
                    f"{running_loss / step:.6f}"
                )


        (
            val_f1,
            val_ids,
            y_true,
            y_pred,
            val_logits,
        ) = evaluate(
            model,
            val_loader,
            device,
        )


        print()
        print(
            f"EPOCH {epoch} "
            f"VAL MACRO-F1 = "
            f"{val_f1:.10f}"
        )


        print(
            classification_report(
                y_true,
                y_pred,
                labels=list(
                    range(len(ALL_CLASSES))
                ),
                target_names=ALL_CLASSES,
                digits=4,
                zero_division=0,
            )
        )


        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch


            print(
                f"[BEST] "
                f"{best_f1:.10f} "
                f"at epoch {best_epoch}"
            )


            save_checkpoint(
                model,
                tokenizer,
                numeric_mean,
                numeric_std,
                best_f1,
                best_epoch,
            )


            save_val_outputs(
                val_items,
                val_ids,
                y_true,
                y_pred,
                val_logits,
                best_f1,
                best_epoch,
            )


    print()
    print(
        "Training done."
    )

    print(
        "Best Validation Macro-F1:",
        best_f1,
    )

    print(
        "Best Epoch:",
        best_epoch,
    )

    print(
        "Saved model dir:",
        SAVE_DIR,
    )


if __name__ == "__main__":
    main()
