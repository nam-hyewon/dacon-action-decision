#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

C2I = {x: i for i, x in enumerate(CLASSES)}

REPRO_ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

PRED_CSV = (
    REPRO_ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text(sample):
    parts = []

    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])

    parts.append("[CURRENT_PROMPT]")
    parts.append(str(sample.get("current_prompt", "")))

    parts.append("[STATE]")
    parts.append(
        f"user_tier={meta.get('user_tier', '')} "
        f"language={meta.get('language_pref', '')} "
        f"turn={meta.get('turn_index', '')} "
        f"budget={meta.get('budget_tokens_remaining', '')} "
        f"elapsed={meta.get('elapsed_session_sec', '')}"
    )

    parts.append(
        f"git_dirty={workspace.get('git_dirty', '')} "
        f"ci={workspace.get('last_ci_status', '')} "
        f"loc={workspace.get('loc', '')}"
    )

    open_files = workspace.get("open_files", [])

    if open_files:
        parts.append("[OPEN_FILES]")
        parts.append(" ".join(map(str, open_files)))

    parts.append("[HISTORY]")

    for h in history:
        role = str(h.get("role", ""))

        if role == "user":
            parts.append("[USER]")
            parts.append(str(h.get("content", "")))

        else:
            parts.append("[ACTION]")
            parts.append(str(h.get("name", "")))

            args = h.get("args", "")

            if args:
                parts.append("[ARGS]")
                parts.append(str(args))

            result = h.get("result_summary", "")

            if result:
                parts.append("[RESULT]")
                parts.append(str(result))

    # Explicit recent trajectory emphasis.
    actions = [
        str(h.get("name", ""))
        for h in history
        if h.get("role") != "user"
    ]

    if actions:
        parts.append("[ACTION_TAIL]")
        parts.append(" ".join(actions[-8:]))

        parts.append("[LAST_ACTION]")
        parts.append(actions[-1])

    recent_users = [
        str(h.get("content", ""))
        for h in history
        if h.get("role") == "user"
    ]

    if recent_users:
        parts.append("[RECENT_USERS]")
        parts.append("\n".join(recent_users[-3:]))

    return "\n".join(parts)


class PairDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_len,
    ):
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        self.labels = np.asarray(
            labels,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            k: torch.tensor(v[idx])
            for k, v in self.enc.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )

        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    pred = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, pred),
        "macro_f1": f1_score(
            labels,
            pred,
            average="macro",
            zero_division=0,
        ),
    }


parser = argparse.ArgumentParser()

parser.add_argument("--class-a", required=True)
parser.add_argument("--class-b", required=True)

parser.add_argument(
    "--model-id",
    default="microsoft/deberta-v3-base",
)

parser.add_argument("--max-len", type=int, default=512)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--grad-accum", type=int, default=4)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=43)

args = parser.parse_args()

assert args.class_a in C2I
assert args.class_b in C2I
assert args.class_a != args.class_b

seed_everything(args.seed)

a = C2I[args.class_a]
b = C2I[args.class_b]

pair_name = (
    args.class_a
    + "__"
    + args.class_b
)

out_dir = Path(
    "submit/model"
) / f"pair_{pair_name}_fold0"

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)


print("PAIR:", args.class_a, "<->", args.class_b)
print("MODEL:", args.model_id)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Load train
# ============================================================

print("Loading train data...")

samples = []

with open(
    "data/train.jsonl",
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        line = line.strip()

        if line:
            samples.append(json.loads(line))


labels = {
    str(row["id"]): str(row["action"])
    for row in csv.DictReader(
        open(
            "data/train_labels.csv",
            encoding="utf-8",
        )
    )
}

ids = np.array([
    str(x["id"])
    for x in samples
])

all_y = np.array([
    C2I[labels[sid]]
    for sid in ids
], dtype=np.int64)


pred_df = pd.read_csv(PRED_CSV)

val_ids = (
    pred_df["id"]
    .astype(str)
    .values
)

val_set = set(val_ids)

id_to_idx = {
    sid: i
    for i, sid in enumerate(ids)
}

val_idx = np.array([
    id_to_idx[sid]
    for sid in val_ids
])

ref_idx = np.array([
    i
    for i, sid in enumerate(ids)
    if sid not in val_set
])


# ============================================================
# Pair-specific training rows
# ============================================================

train_pair_idx = ref_idx[
    (all_y[ref_idx] == a)
    | (all_y[ref_idx] == b)
]

train_text = [
    build_text(samples[i])
    for i in train_pair_idx
]

train_y = (
    all_y[train_pair_idx] == b
).astype(np.int64)


print("train pair samples:", len(train_pair_idx))
print(
    "class counts:",
    np.bincount(
        train_y,
        minlength=2,
    ).tolist(),
)


# ============================================================
# Exact base gate
# ============================================================

base_score = np.load(
    "exact_repro_base_score.npy"
).astype(np.float64)

base_pred = np.load(
    "exact_repro_base_pred.npy"
).astype(np.int64)

y_val = (
    pred_df["true_id"]
    .astype(int)
    .values
)

order = np.argsort(
    -base_score,
    axis=1,
)

top1 = order[:, 0]
top2 = order[:, 1]

gate = (
    ((top1 == a) & (top2 == b))
    | ((top1 == b) & (top2 == a))
)

gate_idx = np.where(gate)[0]


print("validation gate:", len(gate_idx))

gate_text = [
    build_text(
        samples[val_idx[i]]
    )
    for i in gate_idx
]

gate_y = (
    y_val[gate_idx] == b
).astype(np.int64)


# ============================================================
# Model
# ============================================================

print("Loading tokenizer/model...")

tokenizer = AutoTokenizer.from_pretrained(
    args.model_id,
    use_fast=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_id,
    num_labels=2,
    torch_dtype=torch.float32,
)

# Keep trainable parameters in FP32.
# Trainer AMP will handle FP16 autocast/scaling safely.
model = model.float()

param_dtypes = sorted({
    str(p.dtype)
    for p in model.parameters()
})

print("MODEL PARAM DTYPES:", param_dtypes)


train_ds = PairDataset(
    train_text,
    train_y,
    tokenizer,
    args.max_len,
)

gate_ds = PairDataset(
    gate_text,
    gate_y,
    tokenizer,
    args.max_len,
)


training_args = TrainingArguments(
    output_dir=str(out_dir / "trainer"),
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch * 2,
    gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    warmup_ratio=0.06,
    logging_steps=100,
    save_strategy="no",
    eval_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=args.seed,
    data_seed=args.seed,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
)


print("Training...")

trainer.train()


# ============================================================
# Predict gate
# ============================================================

print("Predicting gate...")

pred_out = trainer.predict(gate_ds)

pair_logits = pred_out.predictions.astype(
    np.float64
)

pair_prob = torch.softmax(
    torch.tensor(pair_logits),
    dim=1,
).numpy()

pair_pred_bin = pair_prob.argmax(axis=1)

pair_pred = np.where(
    pair_pred_bin == 0,
    a,
    b,
)


true_pair = (
    (y_val[gate_idx] == a)
    | (y_val[gate_idx] == b)
)


print()
print("=" * 100)
print("PAIR ACCURACY")
print("=" * 100)

print(
    "true-pair rows:",
    int(true_pair.sum()),
)

print(
    "base pair acc:",
    accuracy_score(
        y_val[gate_idx][true_pair],
        base_pred[gate_idx][true_pair],
    ),
)

print(
    "specialist pair acc:",
    accuracy_score(
        y_val[gate_idx][true_pair],
        pair_pred[true_pair],
    ),
)


# ============================================================
# Confidence sweep
# ============================================================

BASE_F1 = f1_score(
    y_val,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)


confidence = pair_prob.max(axis=1)

rows = []


for conf_thr in [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
]:
    use_local = (
        confidence >= conf_thr
    )

    use_idx = gate_idx[use_local]

    pred = base_pred.copy()

    pred[use_idx] = pair_pred[
        use_local
    ]

    f1 = f1_score(
        y_val,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    rows.append({
        "class_a": args.class_a,
        "class_b": args.class_b,
        "conf_thr": conf_thr,
        "macro_f1": f1,
        "gain": f1 - BASE_F1,
        "accuracy": accuracy_score(
            y_val,
            pred,
        ),
        "gate_n": len(gate_idx),
        "use_n": len(use_idx),
        "changed_n": int(
            (
                pred[use_idx]
                != base_pred[use_idx]
            ).sum()
        ),
        "beneficial_changes": int(
            (
                (base_pred[use_idx] != y_val[use_idx])
                & (pred[use_idx] == y_val[use_idx])
            ).sum()
        ),
        "harmful_changes": int(
            (
                (base_pred[use_idx] == y_val[use_idx])
                & (pred[use_idx] != y_val[use_idx])
            ).sum()
        ),
    })


result = pd.DataFrame(rows).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

print()
print("=" * 120)
print("GATED HYBRID TOP")
print("=" * 120)

print(
    result.to_string(
        index=False,
    )
)


result.to_csv(
    out_dir / "gated_results.csv",
    index=False,
)

np.save(
    out_dir / "gate_logits.npy",
    pair_logits,
)

gate_out = pred_df.iloc[
    gate_idx
][["id", "true_id"]].copy()

gate_out["base_pred"] = base_pred[
    gate_idx
]

gate_out["pair_pred"] = pair_pred
gate_out["pair_confidence"] = confidence

gate_out.to_csv(
    out_dir / "gate_predictions.csv",
    index=False,
)


model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)


print()
print("Saved:", out_dir)
