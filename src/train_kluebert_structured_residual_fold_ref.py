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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from train_roberta_fold_ref import (
    build_text_eda_state_mid,
    get_actions,
    get_last_action,
    get_prev_action,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


SEED = int(os.environ.get("SEED", "42"))

BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "team_4model_ensemble_07626/"
    "ensemble_07626/fold0_team/"
    "kluebert_folds/"
    "kluebert_mid5_lr15_fold0",
)

MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "6"))

BASE_LR = float(os.environ.get("BASE_LR", "3e-6"))
STRUCT_LR = float(os.environ.get("STRUCT_LR", "2e-4"))

WEIGHT_DECAY = float(
    os.environ.get("WEIGHT_DECAY", "0.01")
)

WARMUP_RATIO = float(
    os.environ.get("WARMUP_RATIO", "0.06")
)

ACTION_EMB_DIM = int(
    os.environ.get("ACTION_EMB_DIM", "32")
)

STRUCT_HIDDEN = int(
    os.environ.get("STRUCT_HIDDEN", "128")
)

DROPOUT = float(
    os.environ.get("DROPOUT", "0.10")
)

PATIENCE = int(
    os.environ.get("PATIENCE", "2")
)


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


PAD_ACTION_ID = 0

ACTION2ID = {
    action: i + 1
    for i, action in enumerate(ALL_CLASSES)
}

NUM_ACTION_IDS = len(ACTION2ID) + 1


NUMERIC_FEATURE_NAMES = [
    "prompt_len_log1p",
    "history_len_log1p",
    "action_count_log1p",
    "turn_index_log1p",
    "budget_tokens_log1p",
    "elapsed_session_log1p",
    "step_id_log1p",
    "num_open_files_log1p",
    "git_dirty",
    "ci_failed",
    "ci_success",
]


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

SAVE_NAME = os.environ.get(
    "SAVE_NAME",
    "kluebert_structured_residual_fold0",
)

SAVE_DIR = (
    ROOT_DIR
    / "submit"
    / "model"
    / SAVE_NAME
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_step_num(sample_id):
    match = re.search(
        r"step_(\d+)",
        str(sample_id),
    )

    if match is None:
        return 0.0

    return float(match.group(1))


def log1p_nonnegative(value):
    return float(
        np.log1p(
            max(
                0.0,
                safe_float(value),
            )
        )
    )


def build_numeric_features(sample):
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}

    history = sample.get("history", []) or []
    actions = get_actions(sample)

    prompt = str(
        sample.get("current_prompt", "")
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
        np.log1p(len(prompt)),
        np.log1p(len(history)),
        np.log1p(len(actions)),
        log1p_nonnegative(
            meta.get("turn_index", 0)
        ),
        log1p_nonnegative(
            meta.get(
                "budget_tokens_remaining",
                0,
            )
        ),
        log1p_nonnegative(
            meta.get(
                "elapsed_session_sec",
                0,
            )
        ),
        np.log1p(
            get_step_num(sample["id"])
        ),
        np.log1p(
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


def action_id(action):
    return ACTION2ID.get(
        action,
        PAD_ACTION_ID,
    )


def build_recent_action_ids(sample):
    actions = get_actions(sample)

    recent = actions[-3:]

    padded = (
        [None] * (3 - len(recent))
        + recent
    )

    return [
        action_id(x)
        if x is not None
        else PAD_ACTION_ID
        for x in padded
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

        label_id = LABEL2ID[
            labels[sid]
        ]


        items.append(
            {
                "id": sid,

                "text":
                    build_text_eda_state_mid(
                        sample
                    ),

                "label": label_id,

                "last_action_id":
                    action_id(
                        get_last_action(sample)
                    ),

                "prev_action_id":
                    action_id(
                        get_prev_action(sample)
                    ),

                "recent_action_ids":
                    build_recent_action_ids(
                        sample
                    ),

                "numeric":
                    build_numeric_features(
                        sample
                    ),

                "current_prompt": str(
                    sample.get(
                        "current_prompt",
                        "",
                    )
                ),
            }
        )


    return items


class ResidualDataset(Dataset):
    def __init__(
        self,
        items,
        numeric_mean,
        numeric_std,
    ):
        self.items = items

        self.numeric_mean = (
            numeric_mean.astype(np.float32)
        )

        self.numeric_std = (
            numeric_std.astype(np.float32)
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
            "label": item["label"],

            "last_action_id":
                item["last_action_id"],

            "prev_action_id":
                item["prev_action_id"],

            "recent_action_ids":
                item["recent_action_ids"],

            "numeric":
                numeric.astype(np.float32),
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        enc = tokenizer(
            [
                x["text"]
                for x in batch
            ],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
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


        enc["recent_action_ids"] = torch.tensor(
            [
                x["recent_action_ids"]
                for x in batch
            ],
            dtype=torch.long,
        )


        enc["numeric_features"] = torch.tensor(
            np.stack(
                [
                    x["numeric"]
                    for x in batch
                ]
            ),
            dtype=torch.float32,
        )


        enc["labels"] = torch.tensor(
            [
                x["label"]
                for x in batch
            ],
            dtype=torch.long,
        )


        enc["_ids"] = [
            x["id"]
            for x in batch
        ]


        return enc


    return collate_fn


class StructuredResidualModel(nn.Module):
    def __init__(
        self,
        base_model_path,
    ):
        super().__init__()


        self.base_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                base_model_path,
                num_labels=len(ALL_CLASSES),
            )
        )


        self.action_embedding = nn.Embedding(
            NUM_ACTION_IDS,
            ACTION_EMB_DIM,
            padding_idx=PAD_ACTION_ID,
        )


        struct_input_dim = (
            ACTION_EMB_DIM
            + ACTION_EMB_DIM
            + ACTION_EMB_DIM
            + len(NUMERIC_FEATURE_NAMES)
        )


        self.structured_mlp = nn.Sequential(
            nn.Linear(
                struct_input_dim,
                STRUCT_HIDDEN,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),

            nn.Linear(
                STRUCT_HIDDEN,
                STRUCT_HIDDEN,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),

            nn.Linear(
                STRUCT_HIDDEN,
                len(ALL_CLASSES),
            ),
        )


        # sigmoid(-2.2) ~= 0.10
        self.alpha_logit = nn.Parameter(
            torch.tensor(-2.2)
        )


    def forward(
        self,
        input_ids,
        attention_mask,
        last_action_ids,
        prev_action_ids,
        recent_action_ids,
        numeric_features,
        token_type_ids=None,
        labels=None,
    ):

        base_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


        if token_type_ids is not None:
            base_kwargs[
                "token_type_ids"
            ] = token_type_ids


        base_outputs = self.base_model(
            **base_kwargs
        )


        base_logits = base_outputs.logits


        last_vec = self.action_embedding(
            last_action_ids
        )


        prev_vec = self.action_embedding(
            prev_action_ids
        )


        recent_emb = self.action_embedding(
            recent_action_ids
        )


        recent_mask = (
            recent_action_ids
            != PAD_ACTION_ID
        ).float().unsqueeze(-1)


        recent_sum = (
            recent_emb * recent_mask
        ).sum(dim=1)


        recent_count = (
            recent_mask
            .sum(dim=1)
            .clamp(min=1.0)
        )


        recent_vec = (
            recent_sum / recent_count
        )


        structured_features = torch.cat(
            [
                last_vec,
                prev_vec,
                recent_vec,
                numeric_features,
            ],
            dim=1,
        )


        delta_logits = self.structured_mlp(
            structured_features
        )


        alpha = torch.sigmoid(
            self.alpha_logit
        )


        logits = (
            base_logits
            + alpha * delta_logits
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
            "base_logits": base_logits,
            "delta_logits": delta_logits,
            "alpha": alpha,
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

    last_alpha = None


    for batch in loader:
        ids = batch.pop("_ids")

        labels_cpu = batch[
            "labels"
        ].numpy()


        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }


        outputs = model(
            **batch
        )


        logits = outputs["logits"]


        preds = logits.argmax(dim=1)


        all_ids.extend(ids)

        all_labels.extend(
            labels_cpu.tolist()
        )

        all_preds.extend(
            preds.cpu().numpy().tolist()
        )

        all_logits.append(
            logits.float().cpu().numpy()
        )


        last_alpha = float(
            outputs["alpha"]
            .detach()
            .cpu()
            .item()
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
        last_alpha,
    )


def save_best(
    model,
    tokenizer,
    numeric_mean,
    numeric_std,
    val_ids,
    y_true,
    y_pred,
    logits,
    val_items,
    best_f1,
    best_epoch,
    alpha,
):
    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.save(
        model.state_dict(),
        SAVE_DIR / "residual_model.pt",
    )


    tokenizer.save_pretrained(
        SAVE_DIR
    )


    model.base_model.config.save_pretrained(
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


    probs = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()


    id_to_item = {
        item["id"]: item
        for item in val_items
    }


    pd.DataFrame(
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

            "confidence":
                probs.max(axis=1),

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
    ).to_csv(
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
                "architecture":
                    "trained_kluebert_plus_structured_residual",

                "base_model_path":
                    BASE_MODEL_PATH,

                "best_val_macro_f1":
                    float(best_f1),

                "best_epoch":
                    int(best_epoch),

                "alpha":
                    float(alpha),

                "base_lr":
                    BASE_LR,

                "struct_lr":
                    STRUCT_LR,

                "epochs":
                    EPOCHS,

                "seed":
                    SEED,

                "all_classes":
                    ALL_CLASSES,

                "action2id":
                    ACTION2ID,

                "numeric_feature_names":
                    NUMERIC_FEATURE_NAMES,
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
    print("BASE_MODEL_PATH:", BASE_MODEL_PATH)

    print(
        "BASE_LR:",
        BASE_LR,
        "| STRUCT_LR:",
        STRUCT_LR,
    )


    items = load_data()


    print("samples:", len(items))

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


    ref_df = pd.read_csv(
        ref_val_path
    )


    val_ids_order = (
        ref_df["id"]
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
            x["numeric"]
            for x in train_items
        ]
    ).astype(np.float32)


    numeric_mean = train_numeric.mean(
        axis=0
    )


    numeric_std = np.maximum(
        train_numeric.std(axis=0),
        1e-6,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=True,
    )


    collate_fn = make_collate_fn(
        tokenizer
    )


    train_loader = DataLoader(
        ResidualDataset(
            train_items,
            numeric_mean,
            numeric_std,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )


    val_loader = DataLoader(
        ResidualDataset(
            val_items,
            numeric_mean,
            numeric_std,
        ),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )


    model = StructuredResidualModel(
        BASE_MODEL_PATH
    ).to(device)


    base_params = list(
        model.base_model.parameters()
    )


    structured_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith(
            "base_model."
        )
    ]


    optimizer = torch.optim.AdamW(
        [
            {
                "params": base_params,
                "lr": BASE_LR,
            },
            {
                "params": structured_params,
                "lr": STRUCT_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


    total_steps = (
        len(train_loader) * EPOCHS
    )


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_steps * WARMUP_RATIO
        ),
        num_training_steps=total_steps,
    )


    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0


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
            batch.pop("_ids", None)


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
                alpha = float(
                    torch.sigmoid(
                        model.alpha_logit
                    )
                    .detach()
                    .cpu()
                    .item()
                )


                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={running_loss / step:.6f} "
                    f"alpha={alpha:.6f}"
                )


        (
            val_f1,
            val_ids,
            y_true,
            y_pred,
            logits,
            alpha,
        ) = evaluate(
            model,
            val_loader,
            device,
        )


        print()
        print(
            f"EPOCH {epoch} "
            f"VAL MACRO-F1 = "
            f"{val_f1:.10f} "
            f"| alpha={alpha:.6f}"
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
            no_improve = 0


            print(
                f"[BEST] "
                f"{best_f1:.10f} "
                f"at epoch {best_epoch}"
            )


            save_best(
                model,
                tokenizer,
                numeric_mean,
                numeric_std,
                val_ids,
                y_true,
                y_pred,
                logits,
                val_items,
                best_f1,
                best_epoch,
                alpha,
            )


        else:
            no_improve += 1


            print(
                "No improvement:",
                no_improve,
                "/",
                PATIENCE,
            )


            if no_improve >= PATIENCE:
                print(
                    "[EARLY STOP]"
                )
                break


    print()
    print("Training done.")

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
