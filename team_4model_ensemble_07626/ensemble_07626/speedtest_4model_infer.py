import os
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

CLASS_BIAS = np.array([
    0.00,   # read_file
    0.00,   # grep_search
    0.15,   # list_directory
    0.15,   # glob_pattern
    0.00,   # edit_file
    0.00,   # write_file
    0.00,   # apply_patch
   -0.05,   # run_bash
    0.00,   # run_tests
    0.05,   # lint_or_typecheck
    0.00,   # ask_user
    0.00,   # plan_task
   -0.10,   # web_search
    0.00,   # respond_only
], dtype=np.float32)

MODEL_ROOT = Path("model") if Path("model").exists() else Path(".")
DATA_PATH = Path("data/test.jsonl")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))


def now():
    return time.perf_counter()


def elapsed(t0):
    return time.perf_counter() - t0


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def history_to_text(history, max_chars=4500):
    if not isinstance(history, list):
        return ""
    parts = []
    for h in history[-8:]:
        if isinstance(h, dict):
            role = h.get("role", "")
            content = h.get("content", "")
            action = h.get("action", "")
            result = h.get("result", "")
            txt = " ".join(str(x) for x in [role, content, action, result] if x is not None)
        else:
            txt = str(h)
        txt = txt.replace("\n", " ")
        if txt:
            parts.append(txt)
    out = " [H] ".join(parts)
    return out[-max_chars:]


def meta_to_text(meta):
    if not isinstance(meta, dict):
        return ""
    keys = [
        "last_action", "prev_action", "prev3_action", "last_prev",
        "turn_index", "step", "git_dirty", "last_ci_status",
        "main_lang", "user_tier", "language_pref",
        "budget_tokens_remaining", "elapsed_session_sec",
        "num_open_files", "num_langs",
    ]
    parts = []
    for k in keys:
        if k in meta:
            parts.append(f"{k}={meta.get(k)}")
    return " ".join(parts)


def build_eda_state(sample):
    cur = str(sample.get("current_prompt", "")).replace("\n", " ")
    meta = meta_to_text(sample.get("session_meta", {}))
    hist = history_to_text(sample.get("history", {}), max_chars=5500)
    return f"[CURRENT] {cur}\n[META] {meta}\n[HISTORY] {hist}"


def build_eda_state_mid(sample):
    cur = str(sample.get("current_prompt", "")).replace("\n", " ")
    meta = meta_to_text(sample.get("session_meta", {}))
    hist = history_to_text(sample.get("history", {}), max_chars=3000)
    return f"[CUR] {cur}\n[STATE] {meta}\n[RECENT_HISTORY] {hist}"


def row_zscore(x, eps=1e-6):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def infer_transformer(model_dir, texts, name):
    print(f"\n===== {name} =====")
    t0 = now()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    load_sec = elapsed(t0)
    print(f"[{name}] load_sec={load_sec:.2f}")

    # Tokenize without padding, then length-sort and dynamically pad batches.
    t_tok = now()
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        add_special_tokens=True,
    )
    lengths = [len(x) for x in encoded["input_ids"]]
    order = np.argsort(lengths)

    tok_sec = elapsed(t_tok)
    print(f"[{name}] tokenize_sec={tok_sec:.2f}")
    print(f"[{name}] len mean={np.mean(lengths):.1f} p50={np.percentile(lengths,50):.0f} p90={np.percentile(lengths,90):.0f} max={np.max(lengths)}")

    logits_all = np.zeros((len(texts), len(ALL_CLASSES)), dtype=np.float32)

    t_inf = now()
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            idx = order[start:start+BATCH_SIZE]
            batch = {
                k: [encoded[k][i] for i in idx]
                for k in encoded.keys()
            }
            padded = tokenizer.pad(
                batch,
                padding=True,
                return_tensors="pt",
            )
            padded = {k: v.to(device) for k, v in padded.items()}

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**padded)
                logits = out.logits.detach().float().cpu().numpy()

            logits_all[idx] = logits

            if (start // BATCH_SIZE) % 20 == 0:
                print(f"[{name}] {min(start+BATCH_SIZE, len(texts))}/{len(texts)}")

    inf_sec = elapsed(t_inf)
    total_sec = elapsed(t0)
    print(f"[{name}] infer_sec={inf_sec:.2f}")
    print(f"[{name}] total_sec={total_sec:.2f}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return logits_all


def safe_get(meta, key, default=""):
    if isinstance(meta, dict):
        return meta.get(key, default)
    return default


def make_tabular_frame(samples):
    # 속도 측정용 fallback feature builder.
    # 정확한 제출 성능용 feature builder는 팀원/기존 tabular 코드와 맞춰야 함.
    rows = []
    for s in samples:
        meta = s.get("session_meta", {}) or {}
        hist = s.get("history", []) or []
        cur = str(s.get("current_prompt", ""))

        hist_str = json.dumps(hist, ensure_ascii=False) if not isinstance(hist, str) else hist

        row = {
            "last_action": safe_get(meta, "last_action", "NONE"),
            "prev_action": safe_get(meta, "prev_action", "NONE"),
            "prev3_action": safe_get(meta, "prev3_action", "NONE"),
            "last_prev": str(safe_get(meta, "last_action", "NONE")) + "_" + str(safe_get(meta, "prev_action", "NONE")),
            "action_tail3": safe_get(meta, "action_tail3", "NONE"),
            "num_actions": str(safe_get(meta, "num_actions", 0)),
            "prompt_len_bucket": str(min(len(cur) // 100, 20)),
            "turn_bucket": str(min(int(safe_get(meta, "turn_index", 0) or 0) // 3, 20)),
            "step_bucket": str(min(int(safe_get(meta, "step", safe_get(meta, "step_id", 0)) or 0) // 3, 20)),
            "user_tier": safe_get(meta, "user_tier", "UNK"),
            "language_pref": safe_get(meta, "language_pref", "UNK"),
            "git_dirty": str(safe_get(meta, "git_dirty", "UNK")),
            "last_ci_status": safe_get(meta, "last_ci_status", "UNK"),
            "main_lang": safe_get(meta, "main_lang", "UNK"),

            "history_len": len(hist_str),
            "num_user_turns": hist_str.lower().count("user"),
            "prompt_len": len(cur),
            "turn_index": int(safe_get(meta, "turn_index", 0) or 0),
            "step": int(safe_get(meta, "step", safe_get(meta, "step_id", 0)) or 0),
            "budget_tokens_remaining": float(safe_get(meta, "budget_tokens_remaining", 0) or 0),
            "elapsed_session_sec": float(safe_get(meta, "elapsed_session_sec", 0) or 0),
            "loc": float(safe_get(meta, "loc", 0) or 0),
            "num_open_files": int(safe_get(meta, "num_open_files", 0) or 0),
            "num_langs": int(safe_get(meta, "num_langs", 0) or 0),
            "max_lang_ratio": float(safe_get(meta, "max_lang_ratio", 0) or 0),

            "kw_test": int(any(k in cur.lower() for k in ["test", "pytest", "테스트", "깨졌", "failed", "failure"])),
            "kw_lint": int(any(k in cur.lower() for k in ["lint", "typecheck", "mypy", "eslint", "타입"])),
            "kw_search": int(any(k in cur.lower() for k in ["search", "grep", "find", "찾", "검색"])),
            "kw_web": int(any(k in cur.lower() for k in ["web", "browser", "online", "인터넷", "검색해"])),
            "kw_edit": int(any(k in cur.lower() for k in ["edit", "modify", "fix", "수정", "고쳐"])),
            "kw_plan": int(any(k in cur.lower() for k in ["plan", "todo", "계획", "먼저"])),
            "kw_ask": int(any(k in cur.lower() for k in ["ask", "clarify", "물어", "확인해"])),
        }

        for a in ALL_CLASSES:
            row[f"cnt_{a}"] = hist_str.count(a)

        rows.append(row)

    return pd.DataFrame(rows)


def infer_tabular(samples):
    print("\n===== tabular =====")
    t0 = now()

    model_path = MODEL_ROOT / "tabular_histgb_row" / "tabular_model.joblib"
    model = joblib.load(model_path)
    df = make_tabular_frame(samples)

    print("[tabular] feature shape:", df.shape)

    # sklearn Pipeline이면 DataFrame 그대로 가능할 확률이 높음.
    try:
        proba = model.predict_proba(df)
    except Exception as e:
        print("[tabular] predict_proba failed with DataFrame:", repr(e))
        raise

    proba = np.asarray(proba, dtype=np.float32)
    logits = np.log(np.clip(proba, 1e-7, 1.0))
    print(f"[tabular] total_sec={elapsed(t0):.2f}")
    return logits


def main():
    total_t0 = now()

    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    print("MAX_LEN:", MAX_LEN, "BATCH_SIZE:", BATCH_SIZE)

    t_read = now()
    samples = read_jsonl(DATA_PATH)
    ids = [str(s.get("id", i)) for i, s in enumerate(samples)]
    print("num samples:", len(samples))
    print("read_sec:", elapsed(t_read))

    t_text = now()
    texts_eda = [build_eda_state(s) for s in samples]
    texts_mid = [build_eda_state_mid(s) for s in samples]
    print("text_build_sec:", elapsed(t_text))

    deberta_logits = infer_transformer(MODEL_ROOT / "deberta_base_eda_state4", texts_eda, "deberta")
    tabular_logits = infer_tabular(samples)
    kluebert_logits = infer_transformer(MODEL_ROOT / "kluebert_mid5_lr15_focal1", texts_mid, "kluebert")
    roberta_logits = infer_transformer(MODEL_ROOT / "roberta_eda_state4", texts_eda, "roberta")

    t_ens = now()
    final_logits = (
        0.30 * row_zscore(deberta_logits)
        + 0.30 * row_zscore(tabular_logits)
        + 0.15 * row_zscore(kluebert_logits)
        + 0.25 * row_zscore(roberta_logits)
    )
    final_logits = final_logits + CLASS_BIAS[None, :]

    pred_ids = final_logits.argmax(axis=1)
    pred_actions = [ALL_CLASSES[i] for i in pred_ids]
    print("ensemble_sec:", elapsed(t_ens))

    sub = pd.DataFrame({"id": ids, "action": pred_actions})
    out_path = OUT_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)
    print("saved:", out_path)
    print(sub.head())

    print("\n===== TOTAL =====")
    print(f"total_sec={elapsed(total_t0):.2f}")
    print(f"total_min={elapsed(total_t0)/60:.2f}")


if __name__ == "__main__":
    main()
