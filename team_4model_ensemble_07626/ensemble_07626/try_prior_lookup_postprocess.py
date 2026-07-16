import json
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
label2id = {c: i for i, c in enumerate(ALL_CLASSES)}


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def normalize_prompt(s):
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[0-9]+", "<num>", s)
    s = s.strip()
    return s[:300]


def prompt_signature(s):
    s = str(s).lower()
    flags = []
    kws = {
        "test": ["test", "pytest", "테스트", "다시 돌", "깨졌", "failed", "fail"],
        "lint": ["lint", "typecheck", "mypy", "eslint", "타입"],
        "search": ["search", "grep", "find", "찾", "검색", "어디"],
        "web": ["web", "browser", "online", "인터넷", "구글"],
        "edit": ["edit", "modify", "fix", "수정", "고쳐", "바꿔"],
        "read": ["open", "read", "show", "열", "보여", "확인"],
        "list": ["list", "directory", "ls", "폴더", "목록"],
        "plan": ["plan", "todo", "먼저", "계획"],
        "ask": ["ask", "clarify", "물어", "질문"],
    }
    for name, words in kws.items():
        if any(w in s for w in words):
            flags.append(name)
    return "|".join(flags) if flags else "none"


def get_meta(sample):
    return sample.get("session_meta", {}) or {}


def get_key(sample, mode):
    meta = get_meta(sample)
    cur = sample.get("current_prompt", "")

    last_action = str(meta.get("last_action", "NONE"))
    prev_action = str(meta.get("prev_action", "NONE"))
    prev3_action = str(meta.get("prev3_action", meta.get("action_tail3", "NONE")))
    main_lang = str(meta.get("main_lang", "UNK"))
    ci = str(meta.get("last_ci_status", "UNK"))
    dirty = str(meta.get("git_dirty", "UNK"))

    turn = meta.get("turn_index", 0) or 0
    try:
        turn_bucket = int(turn) // 3
    except Exception:
        turn_bucket = 0

    if mode == "prompt_exact":
        return ("prompt_exact", normalize_prompt(cur))

    if mode == "prompt_sig_state":
        return ("prompt_sig_state", prompt_signature(cur), last_action, prev_action, main_lang, ci)

    if mode == "transition2":
        return ("transition2", last_action, prev_action)

    if mode == "transition3":
        return ("transition3", last_action, prev_action, prev3_action)

    if mode == "state":
        return ("state", last_action, prev_action, main_lang, ci, dirty, turn_bucket)

    raise ValueError(mode)


def build_prior_maps(train_samples, train_labels, val_ids):
    maps = {m: defaultdict(Counter) for m in [
        "prompt_exact",
        "prompt_sig_state",
        "transition3",
        "transition2",
        "state",
    ]}

    for s in train_samples:
        sid = str(s.get("id", ""))
        if sid in val_ids:
            continue
        y = train_labels.get(sid)
        if y is None:
            continue

        for mode in maps:
            k = get_key(s, mode)
            maps[mode][k][y] += 1

    return maps


def prior_for_sample(sample, maps, min_support=8, min_conf=0.45):
    # 우선순위: 더 구체적인 key부터
    modes = ["prompt_exact", "prompt_sig_state", "transition3", "state", "transition2"]

    best = None
    for mode in modes:
        k = get_key(sample, mode)
        cnt = maps[mode].get(k)
        if not cnt:
            continue

        total = sum(cnt.values())
        if total < min_support:
            continue

        top_label, top_count = cnt.most_common(1)[0]
        conf = top_count / total
        if conf < min_conf:
            continue

        if best is None or (conf, total) > (best["conf"], best["support"]):
            best = {
                "mode": mode,
                "counter": cnt,
                "support": total,
                "conf": conf,
                "top_label": top_label,
            }

    if best is None:
        return None

    # smoothed log prior
    alpha = 0.5
    arr = np.full(len(ALL_CLASSES), alpha, dtype=np.float32)
    for lab, c in best["counter"].items():
        if lab in label2id:
            arr[label2id[lab]] += c

    prob = arr / arr.sum()
    logp = np.log(prob + 1e-8).astype(np.float32)
    return logp, best


def main():
    # validation ids / labels
    val_df = pd.read_csv("roberta_eda_state4/val_predictions.csv")
    val_ids = set(val_df["id"].astype(str))
    y = val_df["true_id"].values.astype(int)

    # 현재 best 0.76257 logits
    base_logits = np.load("ensemble_greedy4_weightgrid_row_zscore/class_bias/val_logits_with_bias.npy")
    base_pred = base_logits.argmax(1)
    base_f1 = f1_score(y, base_pred, average="macro")
    print("base f1:", base_f1, "acc", (base_pred == y).mean())

    # train labels
    labels_df = pd.read_csv("/home/skaodjs/projects/dacon-action-decision/data/train_labels.csv")
    # 컬럼명 자동 대응
    id_col = "id"
    label_col = "action" if "action" in labels_df.columns else "label"
    train_labels = dict(zip(labels_df[id_col].astype(str), labels_df[label_col].astype(str)))

    # train samples
    train_samples = []
    with open("/home/skaodjs/projects/dacon-action-decision/data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_samples.append(json.loads(line))

    # val samples in same order
    id_to_sample = {str(s.get("id", "")): s for s in train_samples}
    val_samples = [id_to_sample[i] for i in val_df["id"].astype(str)]

    maps = build_prior_maps(train_samples, train_labels, val_ids)

    print("built maps:")
    for mode, mp in maps.items():
        print(mode, len(mp))

    # prior logits 만들기
    prior_logits = np.zeros_like(base_logits, dtype=np.float32)
    has_prior = np.zeros(len(val_samples), dtype=bool)
    prior_info = []

    for i, s in enumerate(val_samples):
        out = prior_for_sample(s, maps, min_support=8, min_conf=0.45)
        if out is None:
            prior_info.append(None)
            continue
        logp, info = out
        prior_logits[i] = logp
        has_prior[i] = True
        prior_info.append(info)

    print("has_prior:", has_prior.sum(), "/", len(has_prior))

    base_prob = softmax(base_logits)
    conf = base_prob.max(axis=1)

    # 1) add prior sweep
    print("\n=== add prior logits sweep ===")
    best = (base_f1, None)
    for alpha in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]:
        logits = base_logits.copy()
        logits[has_prior] += alpha * prior_logits[has_prior]
        pred = logits.argmax(1)
        f1 = f1_score(y, pred, average="macro")
        acc = (pred == y).mean()
        changed = (pred != base_pred).sum()
        print(f"alpha={alpha:.2f} f1={f1:.6f} acc={acc:.6f} changed={changed}")
        if f1 > best[0]:
            best = (f1, ("add", alpha))
    print("best add:", best)

    # 2) low confidence only
    print("\n=== add prior only low-confidence samples ===")
    best2 = (base_f1, None)
    for q in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        thr = np.quantile(conf, q)
        mask = has_prior & (conf <= thr)
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]:
            logits = base_logits.copy()
            logits[mask] += alpha * prior_logits[mask]
            pred = logits.argmax(1)
            f1 = f1_score(y, pred, average="macro")
            acc = (pred == y).mean()
            changed = (pred != base_pred).sum()
            print(f"q={q:.2f} alpha={alpha:.2f} f1={f1:.6f} acc={acc:.6f} mask={mask.sum()} changed={changed}")
            if f1 > best2[0]:
                best2 = (f1, ("lowconf", q, alpha))
    print("best lowconf:", best2)

    # 3) hard override only when prior very confident and model low confidence
    print("\n=== hard override sweep ===")
    best3 = (base_f1, None)
    for min_prior_conf in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for q in [0.10, 0.20, 0.30, 0.40, 0.50]:
            thr = np.quantile(conf, q)
            pred = base_pred.copy()
            changed = 0

            for i, info in enumerate(prior_info):
                if info is None:
                    continue
                if conf[i] > thr:
                    continue
                if info["conf"] < min_prior_conf:
                    continue
                pred[i] = label2id[info["top_label"]]
                changed += 1

            f1 = f1_score(y, pred, average="macro")
            acc = (pred == y).mean()
            print(f"prior_conf>={min_prior_conf:.2f} q={q:.2f} f1={f1:.6f} acc={acc:.6f} changed={changed}")
            if f1 > best3[0]:
                best3 = (f1, ("override", min_prior_conf, q))
    print("best override:", best3)


if __name__ == "__main__":
    main()
