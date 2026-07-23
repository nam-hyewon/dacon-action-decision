#!/usr/bin/env python3
"""Train a structured-feature router model and export validation logits."""

from __future__ import annotations

import argparse
import csv
import json
import os
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


def get_assistant_actions(sample):
    return [
        h
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def result_status(result_summary):
    """Normalize a result_summary string into a coarse status.

    Rule-based (no label leakage). Categories: ok / fail / zero / other.
    """
    r = str(result_summary)
    rl = r.lower()
    if (
        rl.startswith("error")
        or r.startswith("FAIL")
        or "command failed" in rl
        or "target string not found" in rl
    ):
        return "fail"
    if re.match(r"^\d+ errors", r):  # lint: "2 errors, 1 files affected"
        return "fail"
    if (
        r.startswith("0 matches")
        or r == "no relevant results"
        or r.startswith("0 files")
        or r.startswith("0 results")
    ):
        return "zero"
    if rl.startswith("ok") or r.startswith("PASS") or r.startswith("exit=0"):
        return "ok"
    return "other"


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

    # --- result_summary group (validated +last_result; concentrated on the
    #     result-varying actions: run_tests / lint / apply_patch / grep) ---
    aacts = get_assistant_actions(sample)
    last_result = result_status(aacts[-1].get("result_summary", "")) if aacts else "none"
    prev_result = result_status(aacts[-2].get("result_summary", "")) if len(aacts) >= 2 else "none"
    row["last_result"] = last_result
    row["prev_result"] = prev_result
    row["last_act_result"] = f"{last_action}:{last_result}"
    row["last_result_fail"] = int(last_result == "fail")
    row["last_result_zero"] = int(last_result == "zero")
    row["last_result_ok"] = int(last_result == "ok")
    row["last_ci_fail"] = int(str(workspace.get("last_ci_status", "")) == "failing")

    # --- recent window / run-length group (fixes the whole-session cnt_ defect) ---
    recent5 = Counter(actions[-5:])
    for action in ALL_CLASSES:
        row[f"r5_{action}"] = recent5.get(action, 0)
    run = 0
    for name in reversed(actions):
        if name == last_action:
            run += 1
        else:
            break
    row["same_action_run"] = run

    # --- prompt surface flags (separates "?" from kw_ask; biggest standalone lift,
    #     drives ask_user / plan_task / respond_only) ---
    cp = current_prompt
    row["pf_code"] = int("```" in cp or bool(re.search(r"`[^`]+`", cp)))
    row["pf_path"] = int(bool(re.search(r"[\w/]+\.\w{1,5}\b|[\w-]+/[\w/-]+", cp)))
    row["pf_error"] = int(bool(re.search(r"error|exception|traceback|fail|에러|오류", cp, re.I)))
    row["pf_qmark"] = int("?" in cp)
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


def split_indices(df, split_mode, seed, fold_file=None, fold_id=0):
    idx = np.arange(len(df))
    y = df["label"].values
    if split_mode == "row":
        return train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    if split_mode == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        return next(splitter.split(idx, y, df["session_id"].values))
    if split_mode == "fold":
        if not fold_file:
            raise ValueError("split_mode=fold requires --fold_file")
        id_to_fold = {}
        with open(fold_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                id_to_fold[row["id"]] = int(row["fold"])
        ids = df["id"].astype(str).values
        missing = [i for i in ids if i not in id_to_fold]
        if missing:
            raise ValueError(f"fold_file missing {len(missing)} ids. First: {missing[:3]}")
        train_idx = np.array([k for k, i in enumerate(ids) if id_to_fold[i] != fold_id])
        val_idx = np.array([k for k, i in enumerate(ids) if id_to_fold[i] == fold_id])
        if len(val_idx) == 0:
            raise ValueError(f"fold_id={fold_id} has no rows in {fold_file}")
        return train_idx, val_idx
    raise ValueError(f"unknown split_mode={split_mode}")


def build_model(model_type, categorical_cols, numeric_cols, seed):
    sparse_output = model_type not in {"histgb"}
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
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm") from exc
        clf = LGBMClassifier(
            objective="multiclass",
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=0.2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    elif model_type == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("catboost is not installed. Run: pip install catboost") from exc
        clf = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=900,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=seed,
            auto_class_weights="Balanced",
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        raise ValueError(f"unknown model_type={model_type}")

    return Pipeline([("prep", preprocess), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_mode", choices=["row", "group", "fold"], default="group")
    ap.add_argument("--model_type", choices=["extratrees", "histgb", "lightgbm", "catboost"], default="extratrees")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold_file", default="data/folds_seed42.csv")
    ap.add_argument("--fold_id", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_frame(Path(args.data_dir))

    train_idx, val_idx = split_indices(df, args.split_mode, args.seed, args.fold_file, args.fold_id)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_sessions = set(train_df["session_id"])
    val_sessions = set(val_df["session_id"])
    print("rows train:", len(train_df), "val:", len(val_df))
    print("sessions overlap:", len(train_sessions & val_sessions))

    drop_cols = {"id", "session_id", "label", "label_name"}
    drop_prefixes = tuple(
        p for p in os.environ.get("DROP_FEATURE_PREFIXES", "").split(",") if p
    )
    keep_prefixes = tuple(
        p for p in os.environ.get("KEEP_FEATURE_PREFIXES", "").split(",") if p
    )
    drop_features = {
        p for p in os.environ.get("DROP_FEATURES", "").split(",") if p
    }
    keep_features = {
        p for p in os.environ.get("KEEP_FEATURES", "").split(",") if p
    }
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols
        and c not in drop_features
        and not (drop_prefixes and c.startswith(drop_prefixes))
        and (not keep_prefixes or c.startswith(keep_prefixes) or c in keep_features)
    ]
    if drop_prefixes:
        print("dropping feature prefixes:", drop_prefixes)
    if drop_features:
        print("dropping features:", sorted(drop_features))
    if keep_prefixes or keep_features:
        print("keeping feature prefixes:", keep_prefixes)
        print("keeping features:", sorted(keep_features))
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
