import os
import re
import csv
import json
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {x: i for i, x in enumerate(ALL_CLASSES)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}

SEED = int(os.environ.get("SEED", 42))
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "submit/model/deberta_base_eda_state4")
SAVE_NAME = os.environ.get("SAVE_NAME", "deberta_base_eda_v3_from_state4_lr5e6")
OUT_DIR = ROOT / "submit/model" / SAVE_NAME

MAX_LEN = int(os.environ.get("MAX_LEN", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
EPOCHS = int(os.environ.get("EPOCHS", 2))
LR = float(os.environ.get("LR", 5e-6))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.03))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 1))
USE_CLASS_WEIGHT = int(os.environ.get("USE_CLASS_WEIGHT", 0))

REFERENCE_VAL_CSV = Path(os.environ.get(
    "REFERENCE_VAL_CSV",
    "submit/model/deberta_base_eda_state4/val_predictions.csv"
))

MAX_TRAIN = int(os.environ.get("MAX_TRAIN", 0))
MAX_VAL = int(os.environ.get("MAX_VAL", 0))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(x, max_chars=None):
    x = re.sub(r"\s+", " ", str(x)).strip()
    if max_chars and len(x) > max_chars:
        return x[:max_chars]
    return x


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_last_action(sample):
    a = get_actions(sample)
    return a[-1] if a else "none"


def get_prev_action(sample):
    a = get_actions(sample)
    return a[-2] if len(a) >= 2 else "none"


def get_step_from_id(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return int(m.group(1)) if m else -1


def bucket_int(v, cuts):
    try:
        v = int(v)
    except Exception:
        return "unknown"
    for name, upper in cuts:
        if v <= upper:
            return name
    return f"gt_{cuts[-1][1]}"


def extract_paths(text):
    text = str(text)
    # src/foo.py, package.json, tests/test_x.py, *.py 등
    pats = re.findall(r"[\w./-]+\.(?:py|js|ts|tsx|jsx|json|yaml|yml|toml|md|txt|css|html|go|rs|java|cpp|c|h|sh|sql|env)", text)
    globs = re.findall(r"\*\.([A-Za-z0-9_]+)", text)
    return pats[:20], globs[:20]


def extract_extensions(paths):
    exts = []
    for p in paths:
        m = re.search(r"\.([A-Za-z0-9_]+)$", str(p))
        if m:
            exts.append(m.group(1).lower())
    return sorted(set(exts))


def parse_result_summary(result):
    r = str(result).lower()

    d = {
        "result_empty": len(r.strip()) == 0,
        "result_ok": "ok" in r,
        "result_pass": bool(re.search(r"\bpass\b|passed|success", r)),
        "result_fail": bool(re.search(r"\bfail\b|failed|error|traceback|exception", r)),
        "result_matches": "match" in r,
        "result_no_matches": "no match" in r or "0 match" in r,
        "result_patched": "patched" in r,
        "result_read": "read " in r,
        "result_listed": "listed" in r or "files" in r or "entries" in r,
        "result_ci": "ci" in r,
    }

    m = re.search(r"(\d+)\s+matches?", r)
    d["match_count"] = int(m.group(1)) if m else -1

    m = re.search(r"in\s+(\d+)\s+files?", r)
    d["match_file_count"] = int(m.group(1)) if m else -1

    m = re.search(r"(\d+)l", r)
    d["line_count"] = int(m.group(1)) if m else -1

    m = re.search(r"patched\s+(\d+)\s+files?", r)
    d["patched_file_count"] = int(m.group(1)) if m else -1

    m = re.search(r"(\d+)\s+tests?\s+passed", r)
    d["passed_test_count"] = int(m.group(1)) if m else -1

    return d


def current_intent_cues(prompt):
    p = str(prompt).lower()

    cues = {
        "wants_read": bool(re.search(r"\b(open|read|show|view|display|inspect|look at|current impl|current implementation|cat)\b", p)),
        "wants_search": bool(re.search(r"\b(search|grep|find|look for|where|which file|contains|occurrences?|matches?)\b", p)),
        "wants_list": bool(re.search(r"\b(list|ls|directory|folder|tree|what'?s in|files under|show files)\b", p)),
        "wants_glob": bool(re.search(r"\b(glob|pattern|wildcard|files matching|all .* files|\*\.)\b", p)),
        "wants_edit": bool(re.search(r"\b(fix|change|update|modify|edit|patch|add|remove|replace|refactor|touch|drop in|wire up)\b", p)),
        "wants_write": bool(re.search(r"\b(create|write|new file|generate file|save as)\b", p)),
        "wants_run": bool(re.search(r"\b(run|execute|launch|start|fire up|rebuild|install|command|bash|shell)\b", p)),
        "wants_test": bool(re.search(r"\b(test|tests|pytest|unit test|integration test|happy path|spec)\b", p)),
        "wants_lint": bool(re.search(r"\b(lint|typecheck|type check|mypy|ruff|eslint|tsc|format|formatter|prettier)\b", p)),
        "wants_ask": bool(re.search(r"\b(ask|clarify|confirm|which one|do you want|should i|need more info)\b", p)),
        "wants_plan": bool(re.search(r"\b(plan|steps|strategy|approach|roadmap|todo|before coding|think through)\b", p)),
        "wants_web": bool(re.search(r"\b(web|internet|online|latest|current|look up|search online|browse|google)\b", p)),
        "mentions_error": bool(re.search(r"\b(error|traceback|exception|failed|failing|bug|break|broken)\b", p)),
        "mentions_perf": bool(re.search(r"\b(perf|performance|profile|latency|speed|slow|memory)\b", p)),
        "mentions_config": bool(re.search(r"\b(config|settings|env|yaml|toml|json|package\.json|pyproject)\b", p)),
        "mentions_test_file": bool(re.search(r"\b(test_|_test|spec|tests?/)\b", p)),
    }
    return cues


def compact_history(sample, max_turns=12):
    hist = sample.get("history", [])[-max_turns:]
    parts = []
    user_utts = []
    action_parts = []
    last_result = ""
    last_args = ""

    for h in hist:
        role = h.get("role", "")
        if role == "user":
            txt = clean_text(h.get("content", ""), 500)
            user_utts.append(txt)
            parts.append(f"U: {txt}")
        elif role == "assistant_action":
            name = str(h.get("name", ""))
            args = h.get("args", {})
            result = clean_text(h.get("result_summary", ""), 300)
            last_result = result
            last_args = json.dumps(args, ensure_ascii=False)
            action_parts.append(name)
            parts.append(f"ACTION: {name} ARGS: {last_args} RESULT: {result}")

    return {
        "history_text": "\n".join(parts),
        "recent_users": "\n".join(user_utts[-3:]),
        "action_seq": " > ".join(action_parts),
        "last_result": last_result,
        "last_args": last_args,
    }


def build_text_eda_state_v3(sample):
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    history = sample.get("history", []) or []
    current_prompt = str(sample.get("current_prompt", ""))

    ch = compact_history(sample)
    actions = get_actions(sample)
    last_action = get_last_action(sample)
    prev_action = get_prev_action(sample)

    step = get_step_from_id(sample.get("id", ""))
    turn_index = meta.get("turn_index", step)

    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())

    open_files = workspace.get("open_files", []) or []
    open_exts = extract_extensions(open_files)

    prompt_paths, prompt_globs = extract_paths(current_prompt)
    hist_paths, hist_globs = extract_paths(ch["history_text"])
    all_paths = list(dict.fromkeys(prompt_paths + hist_paths + open_files))[:30]
    all_exts = extract_extensions(all_paths)

    cues = current_intent_cues(current_prompt)
    cue_text = " ".join(f"{k}:{v}" for k, v in cues.items())

    last_result_info = parse_result_summary(ch["last_result"])
    result_text = " ".join(f"{k}:{v}" for k, v in last_result_info.items())

    action_counter = Counter(actions)
    action_counts = " ".join(f"count_{a}:{action_counter.get(a,0)}" for a in ALL_CLASSES)

    workflow_flags = {
        "after_read": last_action == "read_file",
        "after_grep": last_action == "grep_search",
        "after_list": last_action == "list_directory",
        "after_glob": last_action == "glob_pattern",
        "after_edit": last_action == "edit_file",
        "after_patch": last_action == "apply_patch",
        "after_tests": last_action == "run_tests",
        "after_lint": last_action == "lint_or_typecheck",
        "after_bash": last_action == "run_bash",
        "has_ever_read": "read_file" in action_counter,
        "has_ever_grep": "grep_search" in action_counter,
        "has_ever_edit": "edit_file" in action_counter or "apply_patch" in action_counter,
        "has_ever_test": "run_tests" in action_counter,
        "has_ever_lint": "lint_or_typecheck" in action_counter,
    }
    workflow_text = " ".join(f"{k}:{v}" for k, v in workflow_flags.items())

    prompt_len = len(current_prompt)
    hist_len = len(history)

    return "\n".join([
        "[EDA_STATE_V3_WORKFLOW_FIRST]",
        "[CURRENT_PROMPT]",
        clean_text(current_prompt, 1200),
        "[CURRENT_PROMPT_REPEAT]",
        clean_text(current_prompt, 1200),

        "[INTENT_CUES]",
        cue_text,

        "[WORKFLOW_STATE]",
        f"step: {step}",
        f"turn_index: {turn_index}",
        f"step_bucket: {bucket_int(step, [('s1',1),('s2_3',3),('s4_6',6),('s7_10',10),('s11_14',14)])}",
        f"history_len: {hist_len}",
        f"history_bucket: {bucket_int(hist_len, [('h0',0),('h1_3',3),('h4_7',7),('h8_12',12),('h13_20',20)])}",
        f"prompt_len: {prompt_len}",
        f"prompt_len_bucket: {bucket_int(prompt_len, [('short',45),('medium',80),('long',130),('xlong',220)])}",
        f"last_action: {last_action}",
        f"prev_action: {prev_action}",
        f"action_sequence: {ch['action_seq']}",
        workflow_text,
        action_counts,

        "[LAST_RESULT_PARSED]",
        result_text,
        "[LAST_ACTION_ARGS]",
        ch["last_args"],
        "[LAST_ACTION_RESULT]",
        ch["last_result"],

        "[PATH_AND_FILE_CUES]",
        f"prompt_paths: {' '.join(prompt_paths[:15])}",
        f"prompt_globs: {' '.join(prompt_globs[:15])}",
        f"open_files: {' '.join(map(str, open_files[:20]))}",
        f"open_exts: {' '.join(open_exts)}",
        f"all_paths: {' '.join(map(str, all_paths[:25]))}",
        f"all_exts: {' '.join(all_exts)}",
        f"has_py: {'py' in all_exts}",
        f"has_ts: {'ts' in all_exts or 'tsx' in all_exts}",
        f"has_js: {'js' in all_exts or 'jsx' in all_exts}",
        f"has_config_ext: {bool(set(all_exts) & {'json','yaml','yml','toml','env'})}",
        f"has_markdown: {'md' in all_exts}",
        f"has_test_path: {bool(re.search(r'(test|spec)', ' '.join(map(str, all_paths)).lower()))}",

        "[WORKSPACE]",
        f"user_tier: {meta.get('user_tier','')}",
        f"language_pref: {meta.get('language_pref','')}",
        f"main_lang: {main_lang}",
        f"language_mix: {lang_mix}",
        f"loc: {workspace.get('loc','')}",
        f"git_dirty: {workspace.get('git_dirty','')}",
        f"last_ci_status: {workspace.get('last_ci_status','')}",
        f"budget_tokens_remaining: {meta.get('budget_tokens_remaining','')}",
        f"elapsed_session_sec: {meta.get('elapsed_session_sec','')}",

        "[RECENT_USER_REQUESTS]",
        ch["recent_users"],

        "[COMPACT_HISTORY]",
        ch["history_text"],
    ])


class ActionDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "id": s["id"],
            "text": build_text_eda_state_v3(s),
            "label": int(self.labels[idx]),
        }


def collate_fn(tokenizer):
    def fn(batch):
        ids = [x["id"] for x in batch]
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc["labels"] = labels
        enc["_ids"] = ids
        return enc
    return fn


def load_samples_and_labels():
    samples = []
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    labels = {}
    with open(DATA_DIR / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    sample_by_id = {s["id"]: s for s in samples}

    if REFERENCE_VAL_CSV.exists():
        ref = pd.read_csv(REFERENCE_VAL_CSV)
        val_ids = ref["id"].astype(str).tolist()
        val_id_set = set(val_ids)
        train_ids = [s["id"] for s in samples if s["id"] not in val_id_set]
        print("[INFO] using reference val split:", REFERENCE_VAL_CSV)
    else:
        print("[WARN] reference val not found. using deterministic split.")
        ids = [s["id"] for s in samples]
        rng = np.random.default_rng(SEED)
        rng.shuffle(ids)
        n_val = int(len(ids) * 0.2)
        val_ids = ids[:n_val]
        train_ids = ids[n_val:]

    train_samples = [sample_by_id[i] for i in train_ids]
    val_samples = [sample_by_id[i] for i in val_ids]

    y_train = np.array([LABEL2ID[labels[i]] for i in train_ids], dtype=np.int64)
    y_val = np.array([LABEL2ID[labels[i]] for i in val_ids], dtype=np.int64)

    if MAX_TRAIN > 0:
        train_samples = train_samples[:MAX_TRAIN]
        y_train = y_train[:MAX_TRAIN]
    if MAX_VAL > 0:
        val_samples = val_samples[:MAX_VAL]
        y_val = y_val[:MAX_VAL]

    return train_samples, y_train, val_samples, y_val


@torch.no_grad()
def evaluate(model, tokenizer, val_samples, y_val, device):
    model.eval()
    ds = ActionDataset(val_samples, y_val)
    loader = DataLoader(
        ds,
        batch_size=max(1, BATCH_SIZE * 2),
        shuffle=False,
        collate_fn=collate_fn(tokenizer),
        num_workers=2,
    )

    all_ids, all_logits = [], []
    for batch in loader:
        ids = batch.pop("_ids")
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits.float().cpu().numpy()
        all_ids.extend(ids)
        all_logits.append(logits)

    logits = np.concatenate(all_logits, axis=0)
    pred = logits.argmax(axis=1)
    mf1 = f1_score(y_val, pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0)
    acc = accuracy_score(y_val, pred)
    return mf1, acc, logits, pred, all_ids


def save_best(model, tokenizer, mf1, acc, logits, pred, ids, y_val):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in OUT_DIR.glob("model*.safetensors"):
        p.unlink(missing_ok=True)
    for p in OUT_DIR.glob("pytorch_model*.bin"):
        p.unlink(missing_ok=True)

    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUT_DIR)

    (OUT_DIR / "labels.json").write_text(json.dumps({
        "all_classes": ALL_CLASSES,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }, ensure_ascii=False, indent=2))

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    conf = probs.max(axis=1)

    df = pd.DataFrame({
        "id": ids,
        "true_id": y_val,
        "pred_id": pred,
        "true_action": [ID2LABEL[int(i)] for i in y_val],
        "pred_action": [ID2LABEL[int(i)] for i in pred],
        "confidence": conf,
        "correct": (y_val == pred).astype(int),
    })
    df.to_csv(OUT_DIR / "val_predictions.csv", index=False)
    np.save(OUT_DIR / "val_logits.npy", logits.astype(np.float32))

    info = {
        "best_val_macro_f1": float(mf1),
        "accuracy": float(acc),
        "save_name": SAVE_NAME,
        "student_model": STUDENT_MODEL,
        "text_mode": "eda_state_v3",
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
    }
    (OUT_DIR / "val_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print("[SAVED BEST]", OUT_DIR, info)


def main():
    seed_everything(SEED)

    print("STUDENT_MODEL:", STUDENT_MODEL)
    print("OUT_DIR:", OUT_DIR)
    print("MAX_LEN:", MAX_LEN, "BATCH_SIZE:", BATCH_SIZE, "EPOCHS:", EPOCHS, "LR:", LR)

    train_samples, y_train, val_samples, y_val = load_samples_and_labels()
    print("train:", len(train_samples), Counter(y_train))
    print("val:", len(val_samples), Counter(y_val))

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL,
        num_labels=len(ALL_CLASSES),
        ignore_mismatched_sizes=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = ActionDataset(train_samples, y_train)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn(tokenizer),
        num_workers=2,
        drop_last=False,
    )

    if USE_CLASS_WEIGHT:
        counts = np.bincount(y_train, minlength=len(ALL_CLASSES))
        weights = 1.0 / np.sqrt(np.maximum(counts, 1))
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print("[INFO] using class weights:", weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = math.ceil(len(loader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_mf1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, 1):
            batch.pop("_ids")
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(**inputs)
                loss = loss_fn(out.logits, labels)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * GRAD_ACCUM

            if step % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 500 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {total_loss/step:.4f}")

        mf1, acc, logits, pred, ids = evaluate(model, tokenizer, val_samples, y_val, device)
        print(f"[EPOCH {epoch}] val_macro_f1={mf1:.6f} acc={acc:.6f}")

        print(classification_report(
            y_val,
            pred,
            labels=list(range(len(ALL_CLASSES))),
            target_names=ALL_CLASSES,
            digits=4,
            zero_division=0,
        ))

        if mf1 > best_mf1:
            best_mf1 = mf1
            save_best(model, tokenizer, mf1, acc, logits, pred, ids, y_val)

    print("Training done. Best Macro-F1:", best_mf1)


if __name__ == "__main__":
    main()
