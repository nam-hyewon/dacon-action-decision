import csv
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {a: i for i, a in enumerate(ALL_CLASSES)}
ID2LABEL = {i: a for a, i in LABEL2ID.items()}

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
FULL_MODEL_DIR = MODEL_DIR / "full5"
ROUTER_MODEL_DIR = MODEL_DIR / "router3"
SPECIALIST_PATH = MODEL_DIR / "specialist_booster.pkl"

DATA_DIR = ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = ROOT.parent / "data"
OUTPUT_DIR = ROOT / "output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = ROOT.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = int(os.environ.get("MAX_LEN", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
FULL_W = 0.75
ROUTER_W = 0.25


def get_last_action(sample):
    actions = [h.get("name", "") for h in sample.get("history", []) if h.get("role") == "assistant_action"]
    return actions[-1] if actions else "none"


def get_prev_action(sample):
    actions = [h.get("name", "") for h in sample.get("history", []) if h.get("role") == "assistant_action"]
    return actions[-2] if len(actions) >= 2 else "none"


def get_step_from_id(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return m.group(1) if m else "none"


def compact_history(sample, max_turns=8):
    hist = sample.get("history", [])[-max_turns:]
    parts, actions, user_utts = [], [], []
    last_result = ""
    for h in hist:
        role = h.get("role", "")
        if role == "user":
            txt = str(h.get("content", ""))
            user_utts.append(txt)
            parts.append(f"U: {txt}")
        elif role == "assistant_action":
            name = str(h.get("name", ""))
            actions.append(name)
            args = str(h.get("args", ""))
            result = str(h.get("result_summary", ""))
            last_result = result
            parts.append(f"ACTION: {name} ARGS: {args} RESULT: {result}")
    return {"history_text": "\n".join(parts), "action_seq": " > ".join(actions), "recent_users": "\n".join(user_utts[-3:]), "last_result": last_result}


def build_text_full(sample):
    parts = []
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))
    parts += ["[CURRENT_PROMPT]", current_prompt]
    parts += ["[SESSION_META]", f"user_tier: {meta.get('user_tier', '')}", f"language_pref: {meta.get('language_pref', '')}", f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}", f"turn_index: {meta.get('turn_index', '')}", f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}", f"id_step: {get_step_from_id(sample.get('id', ''))}"]
    parts += ["[WORKSPACE]", f"loc: {workspace.get('loc', '')}", f"git_dirty: {workspace.get('git_dirty', '')}", f"last_ci_status: {workspace.get('last_ci_status', '')}", "open_files: " + " ".join(map(str, workspace.get("open_files", [])))]
    language_mix = workspace.get("language_mix", {})
    parts.append("language_mix: " + " ".join(f"{k}:{v}" for k, v in language_mix.items()))
    parts.append("[RECENT_HISTORY]")
    recent_actions = []
    for h in history[-10:]:
        role = h.get("role", "")
        if role == "user":
            parts += ["[USER]", str(h.get("content", ""))]
        elif role == "assistant_action":
            action_name = str(h.get("name", ""))
            recent_actions.append(action_name)
            parts += ["[ASSISTANT_ACTION]", f"name: {action_name}", f"args: {h.get('args', '')}", f"result: {h.get('result_summary', '')}"]
    parts += ["[RECENT_ACTIONS]", " ".join(recent_actions), "[PREV_ACTION]", get_prev_action(sample), "[LAST_ACTION]", get_last_action(sample), "[CURRENT_PROMPT_AGAIN]", current_prompt]
    return "\n".join(parts)


def build_text_router(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    current_prompt = str(sample.get("current_prompt", ""))
    ch = compact_history(sample)
    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = " ".join(map(str, workspace.get("open_files", [])))
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())
    return "\n".join([
        "[ROUTER_TASK]", "Predict the next assistant action class.",
        "[CURRENT_USER_REQUEST]", current_prompt,
        "[CURRENT_USER_REQUEST_REPEAT]", current_prompt,
        "[RECENT_USER_REQUESTS]", ch["recent_users"],
        "[ACTION_SEQUENCE]", ch["action_seq"],
        "[PREVIOUS_ACTION]", get_prev_action(sample),
        "[LAST_ACTION]", get_last_action(sample),
        "[LAST_ACTION_RESULT]", ch["last_result"],
        "[WORKSPACE_STATE]", f"step: {get_step_from_id(sample.get('id', ''))}", f"turn_index: {meta.get('turn_index', '')}", f"user_tier: {meta.get('user_tier', '')}", f"language_pref: {meta.get('language_pref', '')}", f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}", f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}", f"git_dirty: {workspace.get('git_dirty', '')}", f"last_ci_status: {workspace.get('last_ci_status', '')}", f"main_lang: {main_lang}", f"language_mix: {lang_mix}", f"open_files: {open_files}", "[COMPACT_HISTORY]", ch["history_text"],
    ])


def load_test_samples():
    candidates = [DATA_DIR / "test.jsonl", DATA_DIR / "test.json", DATA_DIR / "test.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No test file found in {DATA_DIR}. Tried: {candidates}")
    if path.suffix == ".jsonl":
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj:
            return obj["data"]
        raise ValueError(f"Unsupported json test format: {path}")
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return [row.to_dict() for _, row in df.iterrows()]
    raise ValueError(f"Unsupported test path: {path}")


class TextDataset(Dataset):
    def __init__(self, ids, texts):
        self.ids = ids
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.ids[idx], self.texts[idx]


def make_collate(tokenizer):
    def collate(batch):
        ids = [x[0] for x in batch]
        texts = [x[1] for x in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return ids, enc
    return collate


@torch.no_grad()
def predict_logits(model_dir: Path, texts, ids, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    loader = DataLoader(TextDataset(ids, texts), batch_size=BATCH_SIZE, shuffle=False, collate_fn=make_collate(tokenizer), num_workers=0)
    all_ids, all_logits = [], []
    for batch_ids, enc in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        all_ids.extend(batch_ids)
        all_logits.append(out.logits.detach().float().cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_ids, np.concatenate(all_logits, axis=0)


def apply_specialist_booster(logits, router_texts):
    if not SPECIALIST_PATH.exists():
        print("[WARN] specialist_booster.pkl not found. Skipping specialist booster.")
        return logits
    artifact = joblib.load(SPECIALIST_PATH)
    X = artifact["vectorizer"].transform(router_texts)
    classifiers = artifact["classifiers"]
    target_gammas = artifact["target_gammas"]
    label2id = artifact.get("label2id", LABEL2ID)
    eps = float(artifact.get("eps", 1e-5))
    clip = float(artifact.get("clip", 8.0))
    boosted = logits.copy()
    for target, gamma in target_gammas.items():
        gamma = float(gamma)
        if gamma == 0.0 or target not in classifiers or target not in label2id:
            continue
        clf = classifiers[target]
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X)[:, 1]
            p = np.clip(p, eps, 1.0 - eps)
            score = np.log(p / (1.0 - p))
        else:
            score = clf.decision_function(X)
        score = np.asarray(score, dtype=np.float32)
        score = np.clip(score, -clip, clip)
        boosted[:, int(label2id[target])] += gamma * score
    return boosted


def main():
    print("[INFO] DATA_DIR:", DATA_DIR)
    print("[INFO] OUTPUT_DIR:", OUTPUT_DIR)
    print("[INFO] FULL_MODEL_DIR:", FULL_MODEL_DIR)
    print("[INFO] ROUTER_MODEL_DIR:", ROUTER_MODEL_DIR)
    print("[INFO] SPECIALIST_PATH:", SPECIALIST_PATH)
    samples = load_test_samples()
    ids = [str(s.get("id", i)) for i, s in enumerate(samples)]
    full_texts = [build_text_full(s) for s in samples]
    router_texts = [build_text_router(s) for s in samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print("[INFO] num samples:", len(samples))
    full_ids, full_logits = predict_logits(FULL_MODEL_DIR, full_texts, ids, device)
    router_ids, router_logits = predict_logits(ROUTER_MODEL_DIR, router_texts, ids, device)
    assert full_ids == router_ids == ids, "ID order mismatch between full/router predictions"
    logits = FULL_W * full_logits + ROUTER_W * router_logits
    logits = apply_specialist_booster(logits, router_texts)
    pred_actions = [ID2LABEL[int(i)] for i in logits.argmax(axis=1)]
    out_path = OUTPUT_DIR / "submission.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action"])
        for sid, action in zip(ids, pred_actions):
            writer.writerow([sid, action])
    print("[DONE] saved:", out_path)
    print(pd.Series(pred_actions).value_counts().to_string())


if __name__ == "__main__":
    main()
