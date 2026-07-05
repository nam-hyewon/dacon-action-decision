import csv
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "./model/roberta"
DATA_DIR = "./data"
OUT_DIR = "./output"

TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

MAX_LEN = 512
BATCH_SIZE = 32


def build_text(sample):
    parts = []

    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    parts.append("[CURRENT_PROMPT]")
    parts.append(current_prompt)

    parts.append("[SESSION_META]")
    parts.append(f"user_tier: {meta.get('user_tier', '')}")
    parts.append(f"language_pref: {meta.get('language_pref', '')}")
    parts.append(f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}")
    parts.append(f"turn_index: {meta.get('turn_index', '')}")
    parts.append(f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}")

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

    parts.append("[RECENT_ACTIONS]")
    parts.append(" ".join(recent_actions))

    if recent_actions:
        parts.append("[LAST_ACTION]")
        parts.append(recent_actions[-1])

    parts.append("[CURRENT_PROMPT_AGAIN]")
    parts.append(current_prompt)

    return "\n".join(parts)


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
    labels_path = os.path.join(model_dir, "labels.json")
    with open(labels_path, encoding="utf-8") as f:
        obj = json.load(f)
    return obj["all_classes"]


class TestDataset(Dataset):
    def __init__(self, samples):
        self.ids = [s.get("id", "") for s in samples]
        self.texts = [build_text(s) for s in samples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "text": self.texts[idx],
        }


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


def main():
    print("Load test data...")
    samples = load_jsonl(TEST_PATH)
    print(f"samples={len(samples)}")

    print("Load labels...")
    all_classes = load_labels(MODEL_DIR)
    print(f"classes={len(all_classes)}")

    print("Load tokenizer/model from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model.to(device)
    model.eval()

    # 평가 서버는 T4 GPU이므로 half precision으로 추론 속도 개선
    if device.type == "cuda":
        model.half()

    dataset = TestDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=2,
    )

    pred_map = {}

    print("Inference...")
    with torch.no_grad():
        for step, (ids, batch) in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits.float()
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()

            for sid, pid in zip(ids, pred_ids):
                pred_map[sid] = all_classes[int(pid)]

            if step % 100 == 0:
                print(f"step {step}/{len(loader)}")

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
