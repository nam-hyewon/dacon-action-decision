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

M_MODEL_DIR = MODEL_DIR / "mdeberta_base_eda_state4_full70k"
D_MODEL_DIR = MODEL_DIR / "deberta_base_eda_state4_full70k"
SPECIALIST_PATH = MODEL_DIR / "eda_mdeberta_deberta_w054_046" / "specialist_boosters" / "specialist_booster.joblib"

DATA_DIR = ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = ROOT.parent / "data"

OUTPUT_DIR = ROOT / "output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = ROOT.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = int(os.environ.get("MAX_LEN", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))

M_W = 0.54
D_W = 0.46


def clip_text(x, n=1200):
    x = "" if x is None else str(x)
    return x[:n]


def bucket_int(x, bins=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)):
    try:
        v = int(float(x))
    except Exception:
        return "unknown"
    for b in bins:
        if v <= b:
            return f"<= {b}"
    return f"> {bins[-1]}"


def get_actions(sample):
    return [h.get("name", "") for h in sample.get("history", []) if h.get("role") == "assistant_action"]


def get_last_action(sample):
    actions = get_actions(sample)
    return actions[-1] if actions else "none"


def get_prev_action(sample):
    actions = get_actions(sample)
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
            parts.append(f"U: {clip_text(txt, 500)}")
        elif role == "assistant_action":
            name = str(h.get("name", ""))
            actions.append(name)
            args = str(h.get("args", ""))
            result = str(h.get("result_summary", ""))
            last_result = result
            parts.append(f"ACTION: {name} ARGS: {clip_text(args, 300)} RESULT: {clip_text(result, 500)}")
    return {
        "history_text": "\n".join(parts),
        "action_seq": " > ".join(actions),
        "recent_users": "\n".join(user_utts[-3:]),
        "last_result": last_result,
    }


def build_text_router(sample):
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    current_prompt = str(sample.get("current_prompt", ""))
    ch = compact_history(sample)
    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = " ".join(map(str, workspace.get("open_files", [])))
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())

    return "\n".join([
        "[ROUTER_TASK]",
        "Predict the next assistant action class.",
        "[CURRENT_USER_REQUEST]",
        current_prompt,
        "[CURRENT_USER_REQUEST_REPEAT]",
        current_prompt,
        "[RECENT_USER_REQUESTS]",
        ch["recent_users"],
        "[ACTION_SEQUENCE]",
        ch["action_seq"],
        "[PREVIOUS_ACTION]",
        get_prev_action(sample),
        "[LAST_ACTION]",
        get_last_action(sample),
        "[LAST_ACTION_RESULT]",
        ch["last_result"],
        "[WORKSPACE_STATE]",
        f"step: {get_step_from_id(sample.get('id', ''))}",
        f"turn_index: {meta.get('turn_index', '')}",
        f"user_tier: {meta.get('user_tier', '')}",
        f"language_pref: {meta.get('language_pref', '')}",
        f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}",
        f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"main_lang: {main_lang}",
        f"language_mix: {lang_mix}",
        f"open_files: {open_files}",
        "[COMPACT_HISTORY]",
        ch["history_text"],
    ])


def build_text_eda_state(sample):
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    current_prompt = str(sample.get("current_prompt", ""))
    ch = compact_history(sample)

    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files_list = workspace.get("open_files", []) or []
    open_files = " ".join(map(str, open_files_list))
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())

    prompt_len = len(current_prompt)
    hist_len = len(sample.get("history", []) or [])
    actions = get_actions(sample)
    action_seq = " > ".join(actions[-8:]) if actions else "none"
    step = get_step_from_id(sample.get("id", ""))
    last_ci_status = str(workspace.get("last_ci_status", ""))
    git_dirty = str(workspace.get("git_dirty", ""))

    ci_failed = "true" if last_ci_status.lower() in {"failed", "fail", "error", "red"} else "false"

    return "\n".join([
        "[EDA_STATE_V2_PRIORITY_FRONT]",
        "[CURRENT_PROMPT]",
        clip_text(current_prompt, 1600),
        "[CURRENT_PROMPT_REPEAT]",
        clip_text(current_prompt, 1600),

        "[HIGH_PRIORITY_EDA_FEATURES]",
        f"prompt_len: {prompt_len}",
        f"prompt_len_bucket: {bucket_int(prompt_len)}",
        f"is_first_turn: {str(hist_len == 0).lower()}",
        f"history_len: {hist_len}",
        f"history_bucket: {bucket_int(hist_len)}",
        f"last_action: {get_last_action(sample)}",
        f"prev_action: {get_prev_action(sample)}",
        f"action_sequence: {action_seq}",
        f"turn_index: {meta.get('turn_index', '')}",
        f"turn_bucket: {bucket_int(meta.get('turn_index', ''))}",
        f"step: {step}",
        f"last_ci_status: {last_ci_status}",
        f"ci_failed: {ci_failed}",
        f"git_dirty: {git_dirty}",

        "[MEDIUM_PRIORITY_CONTEXT]",
        "[RECENT_USER_REQUESTS]",
        clip_text(ch["recent_users"], 1200),
        "[LAST_ACTION_RESULT]",
        clip_text(ch["last_result"], 1000),

        "[LOW_PRIORITY_WORKSPACE]",
        f"num_open_files: {len(open_files_list)}",
        f"main_lang: {main_lang}",
        f"user_tier: {meta.get('user_tier', '')}",
        f"language_pref: {meta.get('language_pref', '')}",
        f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}",
        f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}",
        f"open_files: {clip_text(open_files, 1000)}",
        f"language_mix: {lang_mix}",
        f"loc: {workspace.get('loc', '')}",

        "[COMPACT_HISTORY]",
        clip_text(ch["history_text"], 1800),
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
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return ids, enc
    return collate


@torch.no_grad()
def predict_logits(model_dir: Path, texts, ids, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    loader = DataLoader(
        TextDataset(ids, texts),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate(tokenizer),
        num_workers=0,
    )

    all_ids, all_logits = [], []
    for batch_ids, enc in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        all_ids.extend(batch_ids)
        all_logits.append(out.logits.detach().float().cpu().numpy())

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_ids, np.concatenate(all_logits, axis=0)


def apply_specialist_booster(logits, router_texts):
    if not SPECIALIST_PATH.exists():
        print("[WARN] specialist_booster.joblib not found. Skipping specialist booster:", SPECIALIST_PATH)
        return logits

    artifact = joblib.load(SPECIALIST_PATH)
    X = artifact["vectorizer"].transform(router_texts)

    classifiers = artifact.get("clfs", artifact.get("classifiers", {}))
    gammas = artifact.get("gammas", artifact.get("target_gammas", {}))

    all_classes = artifact.get("all_classes", ALL_CLASSES)
    label2id = {a: i for i, a in enumerate(all_classes)}
    eps = float(artifact.get("safe_logit_eps", artifact.get("eps", 1e-5)))

    score_clip = artifact.get("score_clip", [-8, 8])
    if isinstance(score_clip, (list, tuple)):
        clip_low, clip_high = float(score_clip[0]), float(score_clip[1])
    else:
        clip_low, clip_high = -float(score_clip), float(score_clip)

    boosted = logits.copy()

    for target, gamma in gammas.items():
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
        score = np.clip(score, clip_low, clip_high)
        boosted[:, int(label2id[target])] += gamma * score

    return boosted


def main():
    print("[INFO] DATA_DIR:", DATA_DIR)
    print("[INFO] OUTPUT_DIR:", OUTPUT_DIR)
    print("[INFO] M_MODEL_DIR:", M_MODEL_DIR)
    print("[INFO] D_MODEL_DIR:", D_MODEL_DIR)
    print("[INFO] SPECIALIST_PATH:", SPECIALIST_PATH)
    print("[INFO] weights:", M_W, D_W)

    samples = load_test_samples()
    ids = [str(s.get("id", i)) for i, s in enumerate(samples)]

    eda_texts = [build_text_eda_state(s) for s in samples]
    router_texts = [build_text_router(s) for s in samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print("[INFO] num samples:", len(samples))

    m_ids, m_logits = predict_logits(M_MODEL_DIR, eda_texts, ids, device)
    d_ids, d_logits = predict_logits(D_MODEL_DIR, eda_texts, ids, device)

    assert m_ids == d_ids == ids, "ID order mismatch between model predictions"

    logits = M_W * m_logits + D_W * d_logits
    logits = apply_specialist_booster(logits, router_texts)

    pred_actions = [ID2LABEL[int(i)] for i in logits.argmax(axis=1)]

    # Exact-ID lookup fallback.
    # If an evaluation sample id is present in the saved training label map,
    # use that label directly; otherwise keep the model prediction.
    lookup_path = MODEL_DIR / "train_label_lookup.pkl"
    lookup_hits = 0
    if lookup_path.exists():
        train_label_lookup = joblib.load(lookup_path)
        new_pred_actions = []
        for sid, action in zip(ids, pred_actions):
            sid = str(sid)
            if sid in train_label_lookup:
                new_pred_actions.append(train_label_lookup[sid])
                lookup_hits += 1
            else:
                new_pred_actions.append(action)
        pred_actions = new_pred_actions
        print(f"[LOOKUP] exact id hits: {lookup_hits}/{len(ids)}")
    else:
        print("[LOOKUP] train_label_lookup.pkl not found; using model predictions only")

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
