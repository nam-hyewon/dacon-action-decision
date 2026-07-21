import csv
import json
import os
import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "./model/roberta_eda_state_light4"
DATA_DIR = "./data"
OUT_DIR = "./output"

TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

MAX_LEN = 512
BATCH_SIZE = 64


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_last_action(sample):
    actions = get_actions(sample)
    return actions[-1] if actions else "none"


def get_prev_action(sample):
    actions = get_actions(sample)
    return actions[-2] if len(actions) >= 2 else "none"


def get_step_from_id(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return m.group(1) if m else "none"


def bucket_int(value, cuts):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in cuts:
        if v <= upper:
            return name
    return f"gt_{cuts[-1][1]}"


def build_text_eda_state_light(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    actions = get_actions(sample)
    action_seq = " > ".join(actions[-8:])

    prompt_len = len(current_prompt)
    history_len = len(history)
    turn_index = meta.get("turn_index", 0)

    return "\n".join([
        "[EDA_STATE_LIGHT]",
        "[CURRENT_PROMPT]",
        current_prompt,
        "[CURRENT_PROMPT_REPEAT]",
        current_prompt,
        "[ACTION_STATE]",
        f"last_action: {get_last_action(sample)}",
        f"prev_action: {get_prev_action(sample)}",
        f"action_sequence: {action_seq}",
        "[LIGHT_EDA_FEATURES]",
        f"prompt_len: {prompt_len}",
        f"prompt_len_bucket: {bucket_int(prompt_len, [('short', 45), ('medium', 80), ('long', 120)])}",
        f"is_first_turn: {str(history_len == 0)}",
        f"history_len: {history_len}",
        f"history_bucket: {bucket_int(history_len, [('h0', 0), ('h1_3', 3), ('h4_7', 7), ('h8_12', 12)])}",
        f"turn_index: {turn_index}",
        f"turn_bucket: {bucket_int(turn_index, [('t1', 1), ('t2_4', 4), ('t5_8', 8), ('t9_12', 12)])}",
        f"step: {get_step_from_id(sample.get('id', ''))}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"ci_failed: {str(workspace.get('last_ci_status', '') == 'failed')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
    ])


def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON parsing failed: {e}")
    return samples


def load_labels(model_dir):
    with open(os.path.join(model_dir, "labels.json"), encoding="utf-8") as f:
        return json.load(f)["all_classes"]


def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(f"sample_submission columns are not ['id', 'action']: {fieldnames}")
    return fieldnames, rows


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestDataset(Dataset):
    def __init__(self, samples):
        self.ids = [s.get("id", "") for s in samples]
        self.texts = [build_text_eda_state_light(s) for s in samples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"id": self.ids[idx], "text": self.texts[idx]}


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
        return ids, enc
    return collate_fn


def main():
    print("Load test data...")
    samples = load_jsonl(TEST_PATH)
    print(f"samples={len(samples)}")

    all_classes = load_labels(MODEL_DIR)
    print(f"classes={len(all_classes)}")

    print("Load tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model.to(device)
    model.eval()

    loader = DataLoader(
        TestDataset(samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=0,
    )

    ordered_ids = []
    pred_ids = []
    print("inference...")
    with torch.no_grad():
        for step, (ids, batch) in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pred_ids.extend(outputs.logits.argmax(dim=-1).cpu().numpy().tolist())
            ordered_ids.extend(ids)
            if step % 100 == 0:
                print(f"step {step}/{len(loader)}")

    pred_map = {
        sid: all_classes[int(pid)]
        for sid, pid in zip(ordered_ids, pred_ids)
    }

    print("Build submission...")
    fieldnames, rows = load_sample_submission(SAMPLE_SUB_PATH)
    missing = 0
    for row in rows:
        sid = row["id"]
        if sid in pred_map:
            row["action"] = pred_map[sid]
        else:
            missing += 1
    if missing:
        print(f"Warning: missing predictions for {missing} rows")

    save_submission(OUT_PATH, fieldnames, rows)
    print(f"Saved: {OUT_PATH} rows={len(rows)}")
    print("Done.")


if __name__ == "__main__":
    main()
