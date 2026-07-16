import os
import re
import json
import random
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
label2id = {c: i for i, c in enumerate(ALL_CLASSES)}
id2label = {i: c for i, c in enumerate(ALL_CLASSES)}

PROJECT_ROOT = Path("/home/skaodjs/projects/dacon-action-decision")
DATA_DIR = PROJECT_ROOT / "data"

MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-base")
OUT_DIR = Path(os.environ.get("OUT_DIR", "deberta_base_eda_state4_fold0_seed43"))

FOLD_FILE = Path(os.environ.get("FOLD_FILE", str(DATA_DIR / "folds_seed42.csv")))
FOLD_ID = int(os.environ.get("FOLD_ID", "0"))
REF_VAL_PRED = Path(os.environ.get("REF_VAL_PRED", "fold0_team/deberta_fold0/val_predictions.csv"))

SEED = int(os.environ.get("SEED", "43"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
LR = float(os.environ.get("LR", "1e-5"))
EPOCHS = float(os.environ.get("EPOCHS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.06"))
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "0.0"))

OUT_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_str(x):
    if x is None:
        return ""
    return str(x)


def compact_text(x, max_chars=700):
    s = safe_str(x)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars]
    return s


def compact_history(history, max_items=4):
    if not isinstance(history, list):
        return ""

    chunks = []
    for item in history[-max_items:]:
        if isinstance(item, dict):
            role = safe_str(item.get("role", item.get("type", "")))
            action = safe_str(item.get("action", item.get("tool", item.get("name", ""))))
            content = item.get("content", item.get("text", item.get("observation", "")))
            c = compact_text(content, max_chars=700)

            if action:
                chunks.append(f"role={role} action={action} content={c}")
            else:
                chunks.append(f"role={role} content={c}")
        else:
            chunks.append(compact_text(item, max_chars=700))

    return "\n".join(chunks)


def compact_meta(meta):
    if not isinstance(meta, dict):
        return ""

    keys = [
        "turn_index", "step",
        "last_action", "prev_action", "prev3_action",
        "git_dirty", "last_ci_status", "main_lang",
        "num_open_files", "loc",
        "user_tier", "language_pref",
        "budget_tokens_remaining",
    ]

    parts = []
    for k in keys:
        if k in meta:
            parts.append(f"{k}={meta.get(k)}")
    return " ".join(parts)


def make_text(sample):
    cur = compact_text(sample.get("current_prompt", ""), max_chars=3000)
    meta = compact_meta(sample.get("session_meta", {}) or {})
    hist = compact_history(sample.get("history", []), max_items=4)

    # current_prompt를 맨 앞에 둬서 max_len truncation에 덜 잘리게 함
    return (
        f"[CURRENT_PROMPT]\n{cur}\n\n"
        f"[SESSION_META]\n{meta}\n\n"
        f"[RECENT_HISTORY]\n{hist}"
    )


def load_fold_ids(fold_file, fold_id):
    folds = pd.read_csv(fold_file)
    cols = folds.columns.tolist()

    id_col = "id" if "id" in cols else cols[0]

    fold_col = None
    for cand in ["fold", "fold_id", "kfold", "cv", "split"]:
        if cand in cols:
            fold_col = cand
            break

    if fold_col is None:
        raise ValueError(f"Cannot find fold column in {fold_file}. columns={cols}")

    return folds.loc[folds[fold_col].astype(int) == fold_id, id_col].astype(str).tolist()


class ActionDataset(Dataset):
    def __init__(self, samples, labels, tokenizer, max_len):
        self.samples = samples
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = make_text(self.samples[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if FOCAL_GAMMA > 0:
            ce = nn.functional.cross_entropy(logits, labels, reduction="none")
            pt = torch.exp(-ce)
            loss = ((1.0 - pt) ** FOCAL_GAMMA * ce).mean()
        else:
            loss = nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = logits.argmax(axis=1)
    return {
        "macro_f1": f1_score(labels, pred, average="macro"),
        "accuracy": accuracy_score(labels, pred),
    }


def build_training_args():
    kwargs = dict(
        output_dir=str(OUT_DIR),
        seed=SEED,
        data_seed=SEED,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        max_grad_norm=0.5,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=False,
        bf16=False,
        report_to=[],
        dataloader_num_workers=2,
    )

    # transformers 버전별 호환
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def main():
    seed_everything(SEED)

    print("=" * 80, flush=True)
    print("MODEL_NAME:", MODEL_NAME, flush=True)
    print("OUT_DIR:", OUT_DIR, flush=True)
    print("FOLD_FILE:", FOLD_FILE, flush=True)
    print("FOLD_ID:", FOLD_ID, flush=True)
    print("REF_VAL_PRED:", REF_VAL_PRED, flush=True)
    print("SEED:", SEED, flush=True)
    print("MAX_LEN:", MAX_LEN, flush=True)
    print("LR:", LR, flush=True)
    print("EPOCHS:", EPOCHS, flush=True)
    print("BATCH_SIZE:", BATCH_SIZE, flush=True)
    print("GRAD_ACCUM:", GRAD_ACCUM, flush=True)
    print("WEIGHT_DECAY:", WEIGHT_DECAY, flush=True)
    print("WARMUP_RATIO:", WARMUP_RATIO, flush=True)
    print("FOCAL_GAMMA:", FOCAL_GAMMA, flush=True)
    print("CUDA:", torch.cuda.is_available(), flush=True)
    print("=" * 80, flush=True)

    samples = read_jsonl(DATA_DIR / "train.jsonl")
    labels_df = pd.read_csv(DATA_DIR / "train_labels.csv")

    label_col = "action" if "action" in labels_df.columns else "label"
    label_map = dict(zip(labels_df["id"].astype(str), labels_df[label_col].astype(str)))
    id_to_sample = {str(s.get("id")): s for s in samples}

    # Prefer REF_VAL_PRED id order. This lets us train even if data/folds_seed42.csv is missing.
    if REF_VAL_PRED.exists():
        ref_df = pd.read_csv(REF_VAL_PRED)
        valid_ids_order = ref_df["id"].astype(str).tolist()
        print("Using REF_VAL_PRED id order.", flush=True)
        print("ref valid n:", len(valid_ids_order), flush=True)
    else:
        valid_ids_order = load_fold_ids(FOLD_FILE, FOLD_ID)
        print("Using FOLD_FILE id order.", flush=True)

    valid_ids = set(valid_ids_order)
    train_ids = [str(s.get("id")) for s in samples if str(s.get("id")) not in valid_ids]

    train_samples = [id_to_sample[i] for i in train_ids]
    valid_samples = [id_to_sample[i] for i in valid_ids_order]

    y_train = np.array([label2id[label_map[i]] for i in train_ids], dtype=np.int64)
    y_valid = np.array([label2id[label_map[i]] for i in valid_ids_order], dtype=np.int64)

    print("train n:", len(train_ids), flush=True)
    print("valid n:", len(valid_ids_order), flush=True)
    print("valid first:", valid_ids_order[:3], flush=True)
    print("valid last:", valid_ids_order[-3:], flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ALL_CLASSES),
        id2label=id2label,
        label2id=label2id,
    )
    model.config.problem_type = "single_label_classification"
    model = model.float()
    print("model dtype:", next(model.parameters()).dtype, flush=True)

    train_ds = ActionDataset(train_samples, y_train, tokenizer, MAX_LEN)
    valid_ds = ActionDataset(valid_samples, y_valid, tokenizer, MAX_LEN)

    args = build_training_args()

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("Predicting best model...", flush=True)
    pred_out = trainer.predict(valid_ds)
    logits = pred_out.predictions.astype(np.float32)
    pred = logits.argmax(axis=1)

    macro_f1 = f1_score(y_valid, pred, average="macro")
    acc = accuracy_score(y_valid, pred)

    print("FINAL macro_f1:", macro_f1, flush=True)
    print("FINAL accuracy:", acc, flush=True)

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    np.save(OUT_DIR / "val_logits.npy", logits)

    out_df = pd.DataFrame({
        "id": valid_ids_order,
        "true_id": y_valid,
        "true_action": [ALL_CLASSES[i] for i in y_valid],
        "pred_id": pred,
        "pred_action": [ALL_CLASSES[i] for i in pred],
    })
    out_df.to_csv(OUT_DIR / "val_predictions.csv", index=False)

    info = {
        "model": "deberta_base_eda_state4",
        "model_name": MODEL_NAME,
        "text_mode": "eda_state",
        "seed": SEED,
        "split_mode": "fold",
        "fold_file": str(FOLD_FILE.relative_to(PROJECT_ROOT)) if str(FOLD_FILE).startswith(str(PROJECT_ROOT)) else str(FOLD_FILE),
        "fold_id": FOLD_ID,
        "ref_val_pred": str(REF_VAL_PRED),
        "max_len": MAX_LEN,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "focal_gamma": FOCAL_GAMMA,
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "train_n": len(train_ids),
        "valid_n": len(valid_ids_order),
        "all_classes": ALL_CLASSES,
    }

    json.dump(info, open(OUT_DIR / "val_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(ALL_CLASSES, open(OUT_DIR / "labels.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("saved:", flush=True)
    print(" -", OUT_DIR / "val_logits.npy", flush=True)
    print(" -", OUT_DIR / "val_predictions.csv", flush=True)
    print(" -", OUT_DIR / "val_info.json", flush=True)
    print(" -", OUT_DIR / "labels.json", flush=True)
    print(" - model/tokenizer files in", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
