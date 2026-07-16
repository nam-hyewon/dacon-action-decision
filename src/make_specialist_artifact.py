#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {a: i for i, a in enumerate(ALL_CLASSES)}


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
        "[COMPACT_HISTORY]", ch["history_text"],
    ])


def load_train_rows(data_dir: Path):
    samples = []
    with open(data_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    labels = {}
    with open(data_dir / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]
    texts, y = [], []
    for s in samples:
        sid = s["id"]
        texts.append(build_text_router(s))
        y.append(LABEL2ID[labels[sid]])
    return texts, np.array(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--greedy_csv", required=True)
    ap.add_argument("--out", default="submit/model/specialist_booster.pkl")
    ap.add_argument("--max_features", type=int, default=250000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--C", type=float, default=2.0)
    args = ap.parse_args()
    greedy = pd.read_csv(args.greedy_csv)
    target_gammas = {str(r["target"]): float(r["gamma"]) for _, r in greedy.iterrows() if str(r["target"]) in LABEL2ID}
    print("[INFO] target gammas:")
    for k, v in target_gammas.items():
        print(" ", k, v)
    texts, y = load_train_rows(Path(args.data_dir))
    print("[INFO] train samples:", len(texts))
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=args.min_df, max_features=args.max_features, sublinear_tf=True, norm="l2")
    X = vectorizer.fit_transform(texts)
    print("[INFO] X shape:", X.shape)
    classifiers = {}
    for target, gamma in target_gammas.items():
        tid = LABEL2ID[target]
        y_bin = (y == tid).astype(int)
        pos = int(y_bin.sum())
        neg = int((1 - y_bin).sum())
        if pos < 20:
            print("[WARN] skip", target, "too few positives", pos)
            continue
        clf = LogisticRegression(C=args.C, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42)
        print(f"[INFO] train specialist {target}: pos={pos} neg={neg} gamma={gamma}")
        clf.fit(X, y_bin)
        classifiers[target] = clf
    artifact = {"all_classes": ALL_CLASSES, "label2id": LABEL2ID, "target_gammas": target_gammas, "vectorizer": vectorizer, "classifiers": classifiers, "mode": "router", "clip": 8.0, "eps": 1e-5, "ensemble": {"full5_weight": 0.75, "router3_weight": 0.25}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    print("[DONE] saved:", out_path)


if __name__ == "__main__":
    main()
