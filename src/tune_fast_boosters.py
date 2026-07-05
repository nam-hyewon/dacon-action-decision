#!/usr/bin/env python3
"""
Fast booster tuning for DACON AI Agent Action Decision Prediction.

Tests validation-time add-ons on top of saved DeBERTa logits:
1) TF-IDF kNN retrieval logits
2) Keyword rule logits
3) Transition prior logits

Recommended:
  python src/tune_fast_boosters.py \
    --val_csv submit/model/deberta/val_predictions.csv \
    --val_logits submit/model/deberta/val_logits.npy \
    --out_dir submit/model/deberta/fast_boosters
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {a: i for i, a in enumerate(ALL_CLASSES)}
ID2LABEL = {i: a for a, i in LABEL2ID.items()}

DEFAULT_KS = [1, 3, 5, 10, 20, 50]
DEFAULT_BETAS = [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
DEFAULT_GAMMAS = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
DEFAULT_ALPHAS = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15]


def parse_list(s: str, typ=float):
    return [typ(x.strip()) for x in str(s).split(",") if x.strip()]


def get_last_action(sample):
    actions = [h.get("name", "") for h in sample.get("history", []) if h.get("role") == "assistant_action"]
    return actions[-1] if actions else "none"


def get_prev_action(sample):
    actions = [h.get("name", "") for h in sample.get("history", []) if h.get("role") == "assistant_action"]
    return actions[-2] if len(actions) >= 2 else "none"


def get_step_from_id(sid: str) -> str:
    m = re.search(r"step_(\d+)", str(sid))
    return m.group(1) if m else "none"


def build_text(sample, mode="full"):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    history = sample.get("history", [])
    current_prompt = str(sample.get("current_prompt", ""))
    last_action = get_last_action(sample)
    prev_action = get_prev_action(sample)
    step_id = get_step_from_id(sample.get("id", ""))
    lm = workspace.get("language_mix", {}) or {"none": 1}
    main_lang = max(lm, key=lm.get)

    if mode == "prompt":
        return "\n".join(["[CURRENT_PROMPT]", current_prompt, "[CURRENT_PROMPT_AGAIN]", current_prompt])

    if mode == "prompt_meta":
        return "\n".join([
            "[CURRENT_PROMPT]", current_prompt, "[CURRENT_PROMPT_AGAIN]", current_prompt,
            "[LAST_ACTION]", str(last_action), "[PREV_ACTION]", str(prev_action), "[STEP]", str(step_id),
            "[META]",
            f"turn_index: {meta.get('turn_index', '')}",
            f"git_dirty: {workspace.get('git_dirty', '')}",
            f"last_ci_status: {workspace.get('last_ci_status', '')}",
            f"main_lang: {main_lang}",
            "open_files: " + " ".join(map(str, workspace.get("open_files", []))),
        ])

    parts = []
    parts += ["[CURRENT_PROMPT]", current_prompt]
    parts += ["[SESSION_META]", f"user_tier: {meta.get('user_tier', '')}", f"language_pref: {meta.get('language_pref', '')}"]
    parts += [f"budget_tokens_remaining: {meta.get('budget_tokens_remaining', '')}", f"turn_index: {meta.get('turn_index', '')}"]
    parts += [f"elapsed_session_sec: {meta.get('elapsed_session_sec', '')}", f"id_step: {step_id}"]
    parts += ["[WORKSPACE]", f"loc: {workspace.get('loc', '')}", f"git_dirty: {workspace.get('git_dirty', '')}"]
    parts += [f"last_ci_status: {workspace.get('last_ci_status', '')}"]
    parts.append("open_files: " + " ".join(map(str, workspace.get("open_files", []))))
    language_mix = workspace.get("language_mix", {}) or {}
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
    parts += ["[RECENT_ACTIONS]", " ".join(recent_actions), "[PREV_ACTION]", str(prev_action), "[LAST_ACTION]", str(last_action)]
    parts += ["[CURRENT_PROMPT_AGAIN]", current_prompt]
    return "\n".join(parts)


def load_rows(data_dir: Path, text_mode: str):
    samples = []
    with open(data_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    labels = {}
    with open(data_dir / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]
    rows = []
    for s in samples:
        sid = s["id"]
        label_name = labels[sid]
        meta = s.get("session_meta", {})
        workspace = meta.get("workspace", {})
        lm = workspace.get("language_mix", {}) or {"none": 1}
        rows.append({
            "id": sid,
            "label": LABEL2ID[label_name],
            "label_name": label_name,
            "text": build_text(s, mode=text_mode),
            "current_prompt": str(s.get("current_prompt", "")),
            "last_action": get_last_action(s),
            "prev_action": get_prev_action(s),
            "turn_index": meta.get("turn_index", 0),
            "git_dirty": workspace.get("git_dirty", ""),
            "last_ci_status": workspace.get("last_ci_status", ""),
            "main_lang": max(lm, key=lm.get),
            "step_id": get_step_from_id(sid),
        })
    return rows, samples


def split_like_train_deberta(rows, seed=42):
    labels = [r["label"] for r in rows]
    return train_test_split(rows, test_size=0.2, stratify=labels, random_state=seed)


def infer_id2action(val_df, logits):
    n = logits.shape[1]
    pairs = []
    for id_col, act_col in [("true_id", "true_action"), ("pred_id", "pred_action")]:
        if id_col in val_df.columns and act_col in val_df.columns:
            pairs.append(val_df[[id_col, act_col]].rename(columns={id_col: "id", act_col: "action"}))
    if not pairs:
        return ALL_CLASSES
    all_pairs = pd.concat(pairs).drop_duplicates()
    id2 = {}
    for _, r in all_pairs.iterrows():
        id2.setdefault(int(r["id"]), set()).add(str(r["action"]))
    out = []
    for i in range(n):
        acts = sorted(id2.get(i, []))
        out.append(acts[0] if len(acts) == 1 else ID2LABEL[i])
    return out


def build_rule_logits(val_df: pd.DataFrame, id2action: List[str]) -> np.ndarray:
    action2id = {a: i for i, a in enumerate(id2action)}
    out = np.zeros((len(val_df), len(id2action)), dtype=np.float32)
    rules = [
        ("lint_or_typecheck", [r"\btype\s*check\b", r"\btypecheck\b", r"\blint\b", r"\beslint\b", r"\bmypy\b", r"\bpyright\b", r"\btsc\b", r"\bruff\b"], 1.0),
        ("run_tests", [r"\btest\b", r"\btests\b", r"\bpytest\b", r"\bunit test\b", r"\bintegration test\b", r"\bregression\b", r"\bci\b", r"\bhappy path\b", r"\bmake sure\b.*\bbreak\b"], 1.0),
        ("web_search", [r"\bweb\b", r"\bsearch online\b", r"\blook up\b", r"\bgoogle\b", r"\blatest\b", r"\bdocs?\b.*\bonline\b", r"\b검색\b", r"\b최신\b", r"\b찾아봐\b"], 1.0),
        ("ask_user", [r"\bask\b.*\buser\b", r"\bclarify\b", r"\bdo you want\b", r"\bshould i\b", r"\bconfirm\b", r"\b물어\b", r"\b확인\b.*\?"], 0.9),
        ("plan_task", [r"\bplan\b", r"\boutline\b", r"\bsteps?\b", r"\bbreak.*down\b", r"\bstrategy\b", r"\btodo\b", r"\b단계\b", r"\b계획\b", r"\b쪼개\b"], 0.8),
        ("list_directory", [r"\bls\b", r"\blist\b.*\b(files?|dir|directory|folder)\b", r"\bwhat'?s in\b", r"\bshow\b.*\b(files?|folders?)\b", r"\b목록\b", r"\b폴더\b.*\b봐\b"], 1.0),
        ("grep_search", [r"\bgrep\b", r"\bripgrep\b", r"\brg\b", r"\bsearch\b.*\b(files?|code|repo|src)\b", r"\bfind\b.*\boccurrences?\b", r"\bwhere\b.*\bdefined\b", r"\b찾아\b.*\b파일\b"], 1.0),
        ("read_file", [r"\bread\b", r"\bopen\b", r"\bshow me\b", r"\bcat\b", r"\bcurrent impl\b", r"\bimplementation\b", r"\blook at\b", r"\b열어\b", r"\b보여\b"], 0.8),
        ("edit_file", [r"\bedit\b", r"\bchange\b", r"\bfix\b", r"\bupdate\b", r"\bmodify\b", r"\btweak\b", r"\badjust\b", r"\b수정\b", r"\b고쳐\b"], 0.8),
        ("apply_patch", [r"\bpatch\b", r"\bapply\b.*\bchange\b", r"\btouch both\b", r"\bdrop in\b", r"\badd\b.*\bconfig\b"], 0.8),
        ("write_file", [r"\bwrite\b.*\bfile\b", r"\bcreate\b.*\bfile\b", r"\bnew file\b", r"\b파일\b.*\b만들\b"], 0.8),
        ("run_bash", [r"\brun\b.*\b(command|script|server|app|build|rebuild)\b", r"\bbash\b", r"\bshell\b", r"\bnpm\b", r"\byarn\b", r"\bmake\b", r"\bdocker\b", r"\b실행\b"], 0.7),
    ]
    prompts = val_df["current_prompt"].fillna("").astype(str).str.lower().tolist()
    for i, p in enumerate(prompts):
        for action, pats, weight in rules:
            if action not in action2id:
                continue
            hits = sum(1 for pat in pats if re.search(pat, p))
            if hits:
                out[i, action2id[action]] += weight * math.sqrt(hits)
    return out


def build_transition_prior_from_samples(samples, id2action, smoothing=1.0):
    valid = set(id2action)
    n = len(id2action)
    idx = {a: i for i, a in enumerate(id2action)}
    counts_by_last = defaultdict(lambda: np.zeros(n, dtype=np.float64))
    global_counts = np.zeros(n, dtype=np.float64)
    for s in samples:
        actions = [h.get("name") for h in s.get("history", []) if h.get("role") == "assistant_action" and h.get("name") in valid]
        for a, b in zip(actions[:-1], actions[1:]):
            j = idx[b]
            counts_by_last[a][j] += 1
            global_counts[j] += 1
    global_probs = (global_counts + smoothing) / (global_counts.sum() + smoothing * n)
    global_logp = np.log(global_probs)
    log_prior = {}
    for a, cnt in counts_by_last.items():
        probs = (cnt + smoothing) / (cnt.sum() + smoothing * n)
        log_prior[a] = np.log(probs)
    return log_prior, global_logp


def make_prior_logits(val_df, id2action, train_samples):
    log_prior, global_logp = build_transition_prior_from_samples(train_samples, id2action, smoothing=1.0)
    out = np.zeros((len(val_df), len(id2action)), dtype=np.float32)
    for i, last in enumerate(val_df["last_action"].fillna("none").astype(str).tolist()):
        out[i] = log_prior.get(last, global_logp)
    return out


def build_knn_logits(train_rows, val_rows, id2action, text_mode, ks, max_features=250000, min_df=2):
    train_texts = [r["text"] for r in train_rows]
    val_texts = [r["text"] for r in val_rows]
    y_train = np.array([r["label"] for r in train_rows], dtype=np.int64)
    n_classes = len(id2action)
    if text_mode == "prompt":
        analyzer = "char_wb"; ngram_range = (3, 5)
    else:
        analyzer = "word"; ngram_range = (1, 2)
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=True, min_df=min_df, max_features=max_features, sublinear_tf=True, norm="l2")
    Xtr = vec.fit_transform(train_texts)
    Xva = vec.transform(val_texts)
    max_k = min(max(ks), Xtr.shape[0])
    nn = NearestNeighbors(n_neighbors=max_k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(Xtr)
    dists, inds = nn.kneighbors(Xva, return_distance=True)
    sims = np.maximum(0.0, 1.0 - dists)
    vote_by_k = {}
    for k in ks:
        if k > max_k:
            continue
        votes = np.zeros((len(val_rows), n_classes), dtype=np.float32)
        for i in range(len(val_rows)):
            neigh = inds[i, :k]
            weights = sims[i, :k].copy()
            if weights.sum() <= 1e-12:
                weights = np.ones_like(weights)
            for ni, w in zip(neigh, weights):
                votes[i, y_train[ni]] += float(w)
            votes[i] += 1e-3
            votes[i] /= votes[i].sum()
        vote_by_k[k] = np.log(votes)
    return vote_by_k


def align_val_rows_by_id(split_val_rows, val_df):
    id_to_row = {r["id"]: r for r in split_val_rows}
    missing = [sid for sid in val_df["id"].astype(str).tolist() if sid not in id_to_row]
    if missing:
        raise ValueError(f"Validation split mismatch: {len(missing)} missing. First: {missing[:3]}")
    return [id_to_row[sid] for sid in val_df["id"].astype(str).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--val_csv", default="submit/model/deberta/val_predictions.csv")
    ap.add_argument("--val_logits", default="submit/model/deberta/val_logits.npy")
    ap.add_argument("--out_dir", default="submit/model/deberta/fast_boosters")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text_modes", default="full,prompt_meta,prompt")
    ap.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    ap.add_argument("--betas", default=",".join(map(str, DEFAULT_BETAS)))
    ap.add_argument("--gammas", default=",".join(map(str, DEFAULT_GAMMAS)))
    ap.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    ap.add_argument("--max_features", type=int, default=250000)
    ap.add_argument("--min_df", type=int, default=2)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_df = pd.read_csv(args.val_csv)
    logits = np.load(args.val_logits)
    id2action = infer_id2action(val_df, logits)
    y_true = val_df["true_id"].astype(int).values
    base_pred = logits.argmax(axis=1)
    base_f1 = f1_score(y_true, base_pred, labels=list(range(len(id2action))), average="macro", zero_division=0)
    base_acc = accuracy_score(y_true, base_pred)
    print("===== Fast booster tuning =====")
    print("Base Macro-F1:", base_f1, "| Acc:", base_acc)

    text_modes = [x.strip() for x in args.text_modes.split(",") if x.strip()]
    ks = parse_list(args.ks, int)
    betas = parse_list(args.betas, float)
    gammas = parse_list(args.gammas, float)
    alphas = parse_list(args.alphas, float)

    rule_logits = build_rule_logits(val_df, id2action)
    rows_full, samples = load_rows(data_dir, text_mode="full")
    _, val_rows_full = split_like_train_deberta(rows_full, seed=args.seed)
    align_val_rows_by_id(val_rows_full, val_df)
    prior_logits = make_prior_logits(val_df, id2action, samples)

    all_results = []
    best = None
    best_pred = None
    best_logits = None

    for mode in text_modes:
        print(f"\n[INFO] Building kNN logits for text_mode={mode}")
        rows, _ = load_rows(data_dir, text_mode=mode)
        train_rows, val_rows = split_like_train_deberta(rows, seed=args.seed)
        val_rows_aligned = align_val_rows_by_id(val_rows, val_df)
        knn_by_k = build_knn_logits(train_rows, val_rows_aligned, id2action, mode, ks, args.max_features, args.min_df)
        for k, knn_logits in knn_by_k.items():
            knn_pred = knn_logits.argmax(axis=1)
            knn_f1 = f1_score(y_true, knn_pred, labels=list(range(len(id2action))), average="macro", zero_division=0)
            knn_acc = accuracy_score(y_true, knn_pred)
            print(f"[DIAG] mode={mode} k={k} kNN-only Macro-F1={knn_f1:.6f} Acc={knn_acc:.6f}")
            for beta in betas:
                for gamma in gammas:
                    for alpha in alphas:
                        adjusted = logits.copy()
                        if beta:
                            adjusted = adjusted + beta * knn_logits
                        if gamma:
                            adjusted = adjusted + gamma * rule_logits
                        if alpha:
                            adjusted = adjusted + alpha * prior_logits
                        pred = adjusted.argmax(axis=1)
                        macro_f1 = f1_score(y_true, pred, labels=list(range(len(id2action))), average="macro", zero_division=0)
                        acc = accuracy_score(y_true, pred)
                        row = {
                            "text_mode": mode, "k": k, "beta_knn": beta, "gamma_rule": gamma, "alpha_prior": alpha,
                            "macro_f1": macro_f1, "delta_macro_f1": macro_f1 - base_f1,
                            "accuracy": acc, "delta_accuracy": acc - base_acc,
                            "changed_predictions": int((pred != base_pred).sum()),
                            "knn_only_macro_f1": knn_f1, "knn_only_accuracy": knn_acc,
                        }
                        all_results.append(row)
                        if best is None or (macro_f1, acc) > (best["macro_f1"], best["accuracy"]):
                            best = row
                            best_pred = pred
                            best_logits = adjusted

    res = pd.DataFrame(all_results).sort_values(["macro_f1", "accuracy"], ascending=False)
    res.to_csv(out_dir / "fast_booster_grid.csv", index=False)
    print("\nTop 30:")
    print(res.head(30).to_string(index=False))
    print("\nBest:")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    probs = np.exp(best_logits - best_logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    out_pred = val_df.copy()
    out_pred["booster_pred_id"] = best_pred
    out_pred["booster_pred_action"] = [id2action[i] for i in best_pred]
    out_pred["booster_confidence"] = probs.max(axis=1)
    out_pred["booster_correct"] = best_pred == y_true
    out_pred["booster_changed"] = best_pred != base_pred
    out_pred.to_csv(out_dir / "val_predictions_with_fast_boosters.csv", index=False)

    rows_pc = []
    for i, action in enumerate(id2action):
        mask_true = y_true == i
        for name, pred in [("base", base_pred), ("booster", best_pred)]:
            mask_pred = pred == i
            tp = int((mask_true & mask_pred).sum())
            fp = int((~mask_true & mask_pred).sum())
            fn = int((mask_true & ~mask_pred).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows_pc.append({"action": action, "system": name, "support": int(mask_true.sum()), "precision": prec, "recall": rec, "f1": f1})
    pc = pd.DataFrame(rows_pc)
    base_pc = pc[pc.system == "base"].drop(columns=["system"]).rename(columns={"precision":"base_precision", "recall":"base_recall", "f1":"base_f1"})
    boo_pc = pc[pc.system == "booster"].drop(columns=["system"]).rename(columns={"precision":"booster_precision", "recall":"booster_recall", "f1":"booster_f1"})
    merged = base_pc.merge(boo_pc[["action", "booster_precision", "booster_recall", "booster_f1"]], on="action")
    merged["delta_f1"] = merged["booster_f1"] - merged["base_f1"]
    merged.sort_values("delta_f1").to_csv(out_dir / "per_class_fast_booster_delta.csv", index=False)

    summary = {"base": {"macro_f1": float(base_f1), "accuracy": float(base_acc)}, "best": best, "out_dir": str(out_dir)}
    with open(out_dir / "fast_booster_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nSaved:")
    print(" -", out_dir / "fast_booster_grid.csv")
    print(" -", out_dir / "fast_booster_summary.json")
    print(" -", out_dir / "val_predictions_with_fast_boosters.csv")
    print(" -", out_dir / "per_class_fast_booster_delta.csv")


if __name__ == "__main__":
    main()
