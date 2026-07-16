import os
import re
import csv
import json
import math
import shutil
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {a: i for i, a in enumerate(ALL_CLASSES)}
ID2LABEL = {i: a for a, i in LABEL2ID.items()}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "submit" / "model" / os.environ.get("SAVE_NAME", "deberta_small_eda_distill_md_teacher")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Teacher paths
M_TEACHER_DIR = ROOT / "submit/model/mdeberta_base_eda_state4"
D_TEACHER_DIR = ROOT / "submit/model/deberta_base_eda_state4"
TEACHER_SPECIALIST = ROOT / "submit/model/eda_mdeberta_deberta_w054_046/specialist_boosters/specialist_booster.joblib"

# Student config
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "microsoft/deberta-v3-small")
MAX_LEN = int(os.environ.get("MAX_LEN", 512))
EPOCHS = int(os.environ.get("EPOCHS", 4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
TEACHER_BATCH_SIZE = int(os.environ.get("TEACHER_BATCH_SIZE", 32))
LR = float(os.environ.get("LR", 1e-5))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.06))
SEED = int(os.environ.get("SEED", 42))

# Distillation hyperparams
KD_ALPHA = float(os.environ.get("KD_ALPHA", 0.7))  # teacher loss 비중
KD_T = float(os.environ.get("KD_T", 2.0))          # temperature

M_W = 0.54
D_W = 0.46

torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_maybe_json(x):
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def normalize_sample(obj):
    # CSV로 읽혔을 때 history/session_meta가 문자열이면 JSON으로 복원
    if isinstance(obj, dict):
        out = dict(obj)
        if "history" in out:
            out["history"] = parse_maybe_json(out["history"])
        if "session_meta" in out:
            out["session_meta"] = parse_maybe_json(out["session_meta"])
        if not isinstance(out.get("history", []), list):
            out["history"] = []
        if not isinstance(out.get("session_meta", {}), dict):
            out["session_meta"] = {}
        return out
    return obj


def load_train_samples():
    candidates = [DATA_DIR / "train.jsonl", DATA_DIR / "train.json", DATA_DIR / "train.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No train file found in {DATA_DIR}")

    samples = []
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(normalize_sample(json.loads(line)))
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "data" in obj:
            obj = obj["data"]
        samples = [normalize_sample(x) for x in obj]
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        samples = [normalize_sample(r.to_dict()) for _, r in df.iterrows()]
    else:
        raise ValueError(path)

    # train.jsonl에는 입력만 있고, 정답은 train_labels.csv에 있음
    label_path = DATA_DIR / "train_labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Need label file: {label_path}")

    labels = {}
    with open(label_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    missing = 0
    for s in samples:
        sid = str(s.get("id", ""))
        if sid in labels:
            s["action"] = labels[sid]
        else:
            missing += 1

    if missing:
        raise KeyError(f"{missing} samples have no label in {label_path}")

    return samples


def clip_text(x, n=1200):
    x = "" if x is None else str(x)
    return x[:n]


def bucket_int(value, cuts):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in cuts:
        if v <= upper:
            return name
    return f"gt_{cuts[-1][1]}"


def get_actions(sample):
    return [
        h.get("name", "")
        for h in sample.get("history", [])
        if isinstance(h, dict) and h.get("role") == "assistant_action"
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


def compact_history(sample, max_turns=8):
    hist = sample.get("history", [])[-max_turns:]
    parts, actions, user_utts = [], [], []
    last_result = ""

    for h in hist:
        if not isinstance(h, dict):
            continue
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
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))

    ch = compact_history(sample)
    user_turns = [
        str(h.get("content", ""))
        for h in history
        if h.get("role") == "user"
    ]

    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = workspace.get("open_files", [])
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())

    prompt_len = len(current_prompt)
    history_len = len(history)
    turn_index = meta.get("turn_index", 0)

    return "\n".join([
        "[EDA_STATE_V2_PRIORITY_FRONT]",
        "[CURRENT_PROMPT]",
        current_prompt,
        "[CURRENT_PROMPT_REPEAT]",
        current_prompt,

        "[HIGH_PRIORITY_EDA_FEATURES]",
        f"prompt_len: {prompt_len}",
        f"prompt_len_bucket: {bucket_int(prompt_len, [('short', 45), ('medium', 80), ('long', 120)])}",
        f"is_first_turn: {str(history_len == 0)}",
        f"history_len: {history_len}",
        f"history_bucket: {bucket_int(history_len, [('h0', 0), ('h1_3', 3), ('h4_7', 7), ('h8_12', 12)])}",
        f"last_action: {get_last_action(sample)}",
        f"prev_action: {get_prev_action(sample)}",
        f"action_sequence: {ch['action_seq']}",
        f"turn_index: {turn_index}",
        f"turn_bucket: {bucket_int(turn_index, [('t1', 1), ('t2_4', 4), ('t5_8', 8), ('t9_12', 12)])}",
        f"step: {get_step_from_id(sample.get('id', ''))}",
        f"last_ci_status: {workspace.get('last_ci_status', '')}",
        f"ci_failed: {str(workspace.get('last_ci_status', '') == 'failed')}",
        f"git_dirty: {workspace.get('git_dirty', '')}",

        "[MEDIUM_PRIORITY_CONTEXT]",
        "[RECENT_USER_REQUESTS]",
        "\n".join(user_turns[-3:]),
        "[LAST_ACTION_RESULT]",
        ch["last_result"],

        "[LOW_PRIORITY_WORKSPACE]",
        f"num_open_files: {len(open_files)}",
        f"main_lang: {main_lang}",
        f"user_tier: {meta.get('user_tier', '')}",
        f"language_pref: {meta.get('language_pref', '')}",
        f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}",
        f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}",
        "[OPEN_FILES]",
        " ".join(map(str, open_files)),
        f"language_mix: {lang_mix}",
        f"loc: {workspace.get('loc', '')}",
        "[COMPACT_HISTORY]",
        ch["history_text"],
    ])


class TextOnlyDataset(Dataset):
    def __init__(self, texts, ids=None):
        self.texts = texts
        self.ids = ids or list(range(len(texts)))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.ids[idx], self.texts[idx]


class StudentDataset(Dataset):
    def __init__(self, texts, labels, teacher_logits=None):
        self.texts = texts
        self.labels = labels
        self.teacher_logits = teacher_logits

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "text": self.texts[idx],
            "label": int(self.labels[idx]),
        }
        if self.teacher_logits is not None:
            item["teacher_logits"] = self.teacher_logits[idx].astype(np.float32)
        return item


def make_text_collate(tokenizer):
    def collate(batch):
        ids = [x[0] for x in batch]
        texts = [x[1] for x in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return ids, enc
    return collate


def make_student_collate(tokenizer):
    def collate(batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        if "teacher_logits" in batch[0]:
            teacher_logits = torch.tensor(np.stack([x["teacher_logits"] for x in batch]), dtype=torch.float)
        else:
            teacher_logits = None
        return enc, labels, teacher_logits
    return collate


@torch.no_grad()
def predict_model_logits(model_dir, texts, device, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    if device.type == "cuda":
        model = model.half()
    else:
        model = model.float()

    model.to(device)
    model.eval()

    loader = DataLoader(
        TextOnlyDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_text_collate(tokenizer),
        num_workers=0,
    )

    all_logits = []
    for _, enc in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        all_logits.append(out.logits.detach().float().cpu().numpy())

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(all_logits, axis=0)


def apply_specialist(logits, router_texts):
    if not TEACHER_SPECIALIST.exists():
        print("[WARN] teacher specialist not found. returning plain ensemble")
        return logits

    artifact = joblib.load(TEACHER_SPECIALIST)
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


def get_exact_val_ids():
    # 기존 validation split과 정확히 맞추기 위해 저장된 val_predictions.csv의 id 사용
    val_csv = ROOT / "submit/model/deberta_base_eda_state4/val_predictions.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Need existing val_predictions.csv: {val_csv}")
    df = pd.read_csv(val_csv)
    if "id" not in df.columns:
        raise ValueError(f"val_predictions.csv must have id column. columns={df.columns}")
    return set(map(str, df["id"].values)), df


def make_teacher_logits(train_texts, train_router_texts, cache_path, device):
    cache_path = Path(cache_path)
    if cache_path.exists():
        print("[INFO] loading cached teacher logits:", cache_path)
        return np.load(cache_path)

    print("[INFO] computing teacher logits for train split...")
    print("[INFO] M teacher:", M_TEACHER_DIR)
    print("[INFO] D teacher:", D_TEACHER_DIR)

    m_logits = predict_model_logits(M_TEACHER_DIR, train_texts, device, TEACHER_BATCH_SIZE)
    d_logits = predict_model_logits(D_TEACHER_DIR, train_texts, device, TEACHER_BATCH_SIZE)

    logits = M_W * m_logits + D_W * d_logits
    logits = apply_specialist(logits, train_router_texts)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, logits.astype(np.float32))
    print("[INFO] saved teacher logits:", cache_path, logits.shape)
    return logits


def evaluate(model, tokenizer, texts, labels, device):
    model.eval()
    loader = DataLoader(
        StudentDataset(texts, labels, None),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_student_collate(tokenizer),
        num_workers=0,
    )
    all_logits = []
    with torch.no_grad():
        for enc, y, _ in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            all_logits.append(out.logits.detach().float().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    pred = logits.argmax(axis=1)
    mf1 = f1_score(labels, pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0)
    acc = accuracy_score(labels, pred)
    return mf1, acc, logits, pred


def main():
    print("===== Distill DeBERTa-small from MD teacher =====")
    print("[INFO] ROOT:", ROOT)
    print("[INFO] OUT_DIR:", OUT_DIR)
    print("[INFO] STUDENT_MODEL:", STUDENT_MODEL)
    print("[INFO] EPOCHS:", EPOCHS, "BATCH_SIZE:", BATCH_SIZE, "LR:", LR)
    print("[INFO] KD_ALPHA:", KD_ALPHA, "KD_T:", KD_T)

    samples = load_train_samples()
    val_ids, val_df_existing = get_exact_val_ids()

    train_samples, val_samples = [], []
    for s in samples:
        sid = str(s.get("id", ""))
        if sid in val_ids:
            val_samples.append(s)
        else:
            train_samples.append(s)

    max_train = int(os.environ.get("MAX_TRAIN", "0"))
    max_val = int(os.environ.get("MAX_VAL", "0"))
    if max_train > 0:
        train_samples = train_samples[:max_train]
    if max_val > 0:
        val_samples = val_samples[:max_val]

    print("[INFO] total:", len(samples), "train:", len(train_samples), "val:", len(val_samples))

    train_texts = [build_text_eda_state(s) for s in train_samples]
    val_texts = [build_text_eda_state(s) for s in val_samples]
    train_router_texts = [build_text_router(s) for s in train_samples]

    y_train = np.array([LABEL2ID[str(s["action"])] for s in train_samples], dtype=np.int64)
    y_val = np.array([LABEL2ID[str(s["action"])] for s in val_samples], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    teacher_cache_env = os.environ.get("TEACHER_CACHE", "").strip()
    if teacher_cache_env:
        teacher_cache = Path(teacher_cache_env)
    else:
        teacher_cache = OUT_DIR / "teacher_train_logits.npy"

    teacher_logits = make_teacher_logits(train_texts, train_router_texts, teacher_cache, device)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL,
        num_labels=len(ALL_CLASSES),
        id2label={i: a for i, a in ID2LABEL.items()},
        label2id=LABEL2ID,
    )
    model.float()
    model.to(device)

    train_loader = DataLoader(
        StudentDataset(train_texts, y_train, teacher_logits),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_student_collate(tokenizer),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for step, (enc, labels, t_logits) in enumerate(train_loader, start=1):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            t_logits = t_logits.to(device)

            out = model(**enc)
            s_logits = out.logits

            ce_loss = F.cross_entropy(s_logits, labels)

            # KL(student || teacher), temperature-scaled
            kd_loss = F.kl_div(
                F.log_softmax(s_logits / KD_T, dim=-1),
                F.softmax(t_logits / KD_T, dim=-1),
                reduction="batchmean",
            ) * (KD_T ** 2)

            loss = (1.0 - KD_ALPHA) * ce_loss + KD_ALPHA * kd_loss

            if not torch.isfinite(loss):
                print("[ERROR] non-finite loss detected")
                print("ce_loss:", ce_loss.detach().float().item())
                print("kd_loss:", kd_loss.detach().float().item())
                print("student logits finite:", torch.isfinite(s_logits).all().item())
                print("teacher logits finite:", torch.isfinite(t_logits).all().item())
                print("student logits min/max:", s_logits.detach().float().min().item(), s_logits.detach().float().max().item())
                print("teacher logits min/max:", t_logits.detach().float().min().item(), t_logits.detach().float().max().item())
                raise RuntimeError("non-finite loss")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(float(loss.item()))

            if step % 500 == 0 or step == len(train_loader):
                print(f"epoch {epoch} | step {step}/{len(train_loader)} | loss {np.mean(losses[-100:]):.4f}")

        mf1, acc, val_logits, val_pred = evaluate(model, tokenizer, val_texts, y_val, device)
        print(f"\nEpoch {epoch} Validation Macro-F1: {mf1:.6f} | Acc: {acc:.6f}")
        print(classification_report(
            y_val,
            val_pred,
            labels=list(range(len(ALL_CLASSES))),
            target_names=ALL_CLASSES,
            digits=4,
            zero_division=0,
        ))

        if mf1 > best_f1:
            best_f1 = mf1
            print(f"[INFO] New best {best_f1:.6f}. Saving to {OUT_DIR}")

            # Avoid disk quota failure when overwriting a previous best checkpoint.
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            for old_weight in OUT_DIR.glob("model*.safetensors"):
                old_weight.unlink(missing_ok=True)
            for old_weight in OUT_DIR.glob("pytorch_model*.bin"):
                old_weight.unlink(missing_ok=True)

            model.save_pretrained(OUT_DIR, safe_serialization=True)
            tokenizer.save_pretrained(OUT_DIR)

            with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
                json.dump({"all_classes": ALL_CLASSES}, f, ensure_ascii=False, indent=2)

            np.save(OUT_DIR / "val_logits.npy", val_logits.astype(np.float32))

            out_df = pd.DataFrame({
                "id": [str(s.get("id", "")) for s in val_samples],
                "true_id": y_val,
                "true_action": [ID2LABEL[int(i)] for i in y_val],
                "pred_id": val_pred,
                "pred_action": [ID2LABEL[int(i)] for i in val_pred],
            })
            out_df.to_csv(OUT_DIR / "val_predictions.csv", index=False)

            info = {
                "student_model": STUDENT_MODEL,
                "teacher": "0.54*mdeberta_base_eda_state4 + 0.46*deberta_base_eda_state4 + specialist",
                "best_macro_f1": float(best_f1),
                "accuracy": float(acc),
                "kd_alpha": KD_ALPHA,
                "kd_temperature": KD_T,
                "epochs": EPOCHS,
                "lr": LR,
                "max_len": MAX_LEN,
                "batch_size": BATCH_SIZE,
            }
            with open(OUT_DIR / "val_info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

    print("[DONE] Best Validation Macro-F1:", best_f1)
    print("[DONE] Saved model dir:", OUT_DIR)


if __name__ == "__main__":
    main()
