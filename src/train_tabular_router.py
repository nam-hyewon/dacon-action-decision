#!/usr/bin/env python3
"""Train a structured-feature router model and export validation logits."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def get_session_id(sid):
    return re.sub(r"-step_\d+$", "", str(sid))


def get_step_from_id(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return int(m.group(1)) if m else -1


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
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def prompt_flags(text):
    lower = str(text).lower()
    return {
        "kw_test": int(any(x in lower for x in ["test", "pytest", "테스트"])),
        "kw_lint": int(any(x in lower for x in ["lint", "typecheck", "type check", "mypy", "ruff"])),
        "kw_search": int(any(x in lower for x in ["grep", "search", "find", "찾", "검색"])),
        "kw_web": int(any(x in lower for x in ["web", "internet", "latest", "recent", "검색해", "찾아봐"])),
        "kw_edit": int(any(x in lower for x in ["edit", "modify", "fix", "수정", "고쳐"])),
        "kw_plan": int(any(x in lower for x in ["plan", "계획", "단계", "쪼개"])),
        "kw_ask": int(any(x in lower for x in ["?", "어떻게", "뭐", "which", "what"])),
    }


def make_features(sample):
    sid = str(sample["id"])
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    current_prompt = str(sample.get("current_prompt", ""))
    history = sample.get("history", []) or []
    actions = get_actions(sample)
    action_counts = Counter(actions)
    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = workspace.get("open_files", []) or []
    last_action = actions[-1] if actions else "none"
    prev_action = actions[-2] if len(actions) >= 2 else "none"
    prev3_action = actions[-3] if len(actions) >= 3 else "none"
    prompt_len = len(current_prompt)
    turn_index = meta.get("turn_index", 0)
    step = get_step_from_id(sid)

    row = {
        "id": sid,
        "session_id": get_session_id(sid),
        "last_action": last_action,
        "prev_action": prev_action,
        "prev3_action": prev3_action,
        "last_prev": f"{prev_action}>{last_action}",
        "action_tail3": ">".join(actions[-3:]) if actions else "none",
        "history_len": len(history),
        "num_actions": len(actions),
        "num_user_turns": sum(1 for h in history if h.get("role") == "user"),
        "prompt_len": prompt_len,
        "prompt_len_bucket": bucket_int(prompt_len, [("short", 45), ("medium", 80), ("long", 120)]),
        "turn_index": turn_index,
        "turn_bucket": bucket_int(turn_index, [("t1", 1), ("t2_4", 4), ("t5_8", 8), ("t9_12", 12)]),
        "step": step,
        "step_bucket": bucket_int(step, [("s1", 1), ("s2_4", 4), ("s5_8", 8), ("s9_12", 12)]),
        "user_tier": str(meta.get("user_tier", "")),
        "language_pref": str(meta.get("language_pref", "")),
        "budget_tokens_remaining": meta.get("budget_tokens_remaining", -1),
        "elapsed_session_sec": meta.get("elapsed_session_sec", -1),
        "git_dirty": str(workspace.get("git_dirty", "")),
        "last_ci_status": str(workspace.get("last_ci_status", "")),
        "main_lang": main_lang,
        "loc": workspace.get("loc", -1),
        "num_open_files": len(open_files),
        "num_langs": len(language_mix),
        "max_lang_ratio": max(language_mix.values()) if language_mix else 0.0,
    }
    for action in ALL_CLASSES:
        row[f"cnt_{action}"] = action_counts.get(action, 0)
    row.update(prompt_flags(current_prompt))
    return row


def load_frame(data_dir):
    labels = {}
    with open(data_dir / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    rows = []
    with open(data_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            row = make_features(sample)
            row["label"] = LABEL2ID[labels[row["id"]]]
            row["label_name"] = labels[row["id"]]
            rows.append(row)
    return pd.DataFrame(rows)


def split_indices(df, split_mode, seed):
    idx = np.arange(len(df))
    y = df["label"].values
    if split_mode == "row":
        return train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    if split_mode == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        return next(splitter.split(idx, y, df["session_id"].values))
    raise ValueError(f"unknown split_mode={split_mode}")


def build_model(model_type, categorical_cols, numeric_cols, seed):
    sparse_output = model_type != "histgb"
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=sparse_output,
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
        ],
        sparse_threshold=1.0 if sparse_output else 0.0,
    )

    if model_type == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif model_type == "histgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=500,
            l2_regularization=0.02,
            max_leaf_nodes=31,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model_type={model_type}")

    return Pipeline([("prep", preprocess), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_mode", choices=["row", "group"], default="group")
    ap.add_argument("--model_type", choices=["extratrees", "histgb"], default="extratrees")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_frame(Path(args.data_dir))

    train_idx, val_idx = split_indices(df, args.split_mode, args.seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_sessions = set(train_df["session_id"])
    val_sessions = set(val_df["session_id"])
    print("rows train:", len(train_df), "val:", len(val_df))
    print("sessions overlap:", len(train_sessions & val_sessions))

    drop_cols = {"id", "session_id", "label", "label_name"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    categorical_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" or c.endswith("_bucket") or "action" in c
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    print("categorical:", len(categorical_cols), "numeric:", len(numeric_cols))

    model = build_model(args.model_type, categorical_cols, numeric_cols, args.seed)
    model.fit(train_df[feature_cols], train_df["label"].values)
    probs = model.predict_proba(val_df[feature_cols])
    classes = model.named_steps["clf"].classes_

    full_probs = np.full((len(val_df), len(ALL_CLASSES)), 1e-12, dtype=np.float64)
    full_probs[:, classes] = probs
    full_probs = full_probs / full_probs.sum(axis=1, keepdims=True)
    logits = np.log(np.maximum(full_probs, 1e-12)).astype(np.float32)
    pred = full_probs.argmax(axis=1)
    y_true = val_df["label"].values.astype(int)

    macro_f1 = f1_score(y_true, pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0)
    acc = accuracy_score(y_true, pred)
    print("Validation Macro-F1:", macro_f1)
    print("Accuracy:", acc)
    print(classification_report(y_true, pred, target_names=ALL_CLASSES, digits=4, zero_division=0))

    out_pred = pd.DataFrame({
        "id": val_df["id"].values,
        "true_id": y_true,
        "pred_id": pred,
        "true_action": [ID2LABEL[i] for i in y_true],
        "pred_action": [ID2LABEL[i] for i in pred],
        "confidence": full_probs.max(axis=1),
        "correct": pred == y_true,
        "current_prompt": "",
    })
    out_pred.to_csv(out_dir / "val_predictions.csv", index=False)
    np.save(out_dir / "val_logits.npy", logits)
    joblib.dump(model, out_dir / "tabular_model.joblib")

    with open(out_dir / "val_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_val_macro_f1": float(macro_f1),
            "accuracy": float(acc),
            "model_type": args.model_type,
            "split_mode": args.split_mode,
            "seed": args.seed,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "all_classes": ALL_CLASSES,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
