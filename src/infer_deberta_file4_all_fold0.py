import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from train_deberta_file4_fold_ref import build_text, get_actions, get_step_from_id


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "submit/model/deberta_file4_eda_state_fold0_seed43"

REF_VAL_PATH = (
    ROOT_DIR
    / "team_4model_ensemble_07626"
    / "ensemble_07626"
    / "fold0_team"
    / "tabular_folds"
    / "tabular_histgb_fold0"
    / "val_predictions.csv"
)

OUT_DIR = ROOT_DIR / "submit/model/deberta_file4_eda_state_fold0_seed43_allval"

MAX_LEN = 512
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

FILE4 = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]


class InferDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        return enc


def collate_fn(batch):
    keys = batch[0].keys()

    max_len = max(len(x["input_ids"]) for x in batch)

    out = {}

    for key in keys:
        pad_value = 0

        values = []

        for x in batch:
            v = x[key]
            values.append(
                v + [pad_value] * (max_len - len(v))
            )

        out[key] = torch.tensor(values, dtype=torch.long)

    return out


def load_all_samples():
    samples = []

    jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))

    print("jsonl files:")
    for p in jsonl_files:
        print(" -", p)

        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    labels = {}

    with open(DATA_DIR / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    sample_map = {
        str(s["id"]): s
        for s in samples
        if str(s["id"]) in labels
    }

    ref_df = pd.read_csv(REF_VAL_PATH)

    ids = ref_df["id"].astype(str).tolist()

    texts = []
    rows = []

    missing = []

    for sid in ids:
        if sid not in sample_map:
            missing.append(sid)
            continue

        s = sample_map[sid]

        meta = s.get("session_meta", {})
        workspace = meta.get("workspace", {})
        actions = get_actions(s)

        language_mix = workspace.get("language_mix", {}) or {"none": 1}
        main_lang = max(language_mix, key=language_mix.get)

        texts.append(build_text(s))

        rows.append({
            "id": sid,
            "true_action": labels[sid],
            "last_action": actions[-1] if actions else "none",
            "prev_action": actions[-2] if len(actions) >= 2 else "none",
            "turn_index": meta.get("turn_index", 0),
            "git_dirty": workspace.get("git_dirty", ""),
            "last_ci_status": workspace.get("last_ci_status", ""),
            "main_lang": main_lang,
            "step_id": get_step_from_id(sid),
            "current_prompt": str(s.get("current_prompt", "")),
        })

    if missing:
        print("missing ids:", len(missing))
        print("missing first:", missing[:10])
        raise ValueError("Some reference validation IDs were not found")

    return ids, texts, rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("model:", MODEL_DIR)
    print("reference val:", REF_VAL_PATH)

    ids, texts, rows = load_all_samples()

    print("all fold0 samples:", len(texts))
    print("first ids:", ids[:3])
    print("last ids:", ids[-3:])

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)
    model.eval()

    ds = InferDataset(texts, tokenizer)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    all_logits = []

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
            }

            outputs = model(**batch)

            all_logits.append(
                outputs.logits.detach().cpu().numpy()
            )

            if step % 100 == 0:
                print(
                    f"inference step={step}/{len(loader)}"
                )

    logits = np.concatenate(all_logits, axis=0)

    print("logits shape:", logits.shape)

    assert logits.shape == (len(ids), 4)
    assert len(ids) == len(rows)

    pred_id = logits.argmax(axis=1)

    prob = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()

    confidence = prob.max(axis=1)

    out_df = pd.DataFrame(rows)

    out_df["pred_id"] = pred_id
    out_df["pred_action"] = [
        FILE4[i]
        for i in pred_id
    ]
    out_df["confidence"] = confidence

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        OUT_DIR / "val_logits.npy",
        logits.astype(np.float32),
    )

    out_df.to_csv(
        OUT_DIR / "val_predictions.csv",
        index=False,
    )

    with open(
        OUT_DIR / "val_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "num_val": len(ids),
                "all_classes": FILE4,
                "source_model": str(MODEL_DIR),
                "reference_val": str(REF_VAL_PATH),
                "note": (
                    "FILE4 specialist inference on ALL fold0 "
                    "validation samples."
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== SAVED =====")
    print("logits:", OUT_DIR / "val_logits.npy")
    print("predictions:", OUT_DIR / "val_predictions.csv")
    print("info:", OUT_DIR / "val_info.json")


if __name__ == "__main__":
    main()
