import json
import os
import re
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
label2id = {c: i for i, c in enumerate(ALL_CLASSES)}


PROJECT_ROOT = Path("/home/skaodjs/projects/dacon-action-decision")
DATA_DIR = PROJECT_ROOT / "data"

TEAM_VAL_DIR = Path(os.environ.get("TEAM_VAL_DIR", "fold0_team/deberta_fold0"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "tabular_fold0_seed_ensemble"))
N_SEEDS = int(os.environ.get("N_SEEDS", "8"))
BASE_SEED = int(os.environ.get("BASE_SEED", "42"))

OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_str(x, default="none"):
    if x is None:
        return default
    s = str(x)
    return s if s else default


def get_meta(sample):
    return sample.get("session_meta", {}) or {}


def extract_actions(sample):
    meta = get_meta(sample)
    actions = []

    hist = sample.get("history", [])
    if isinstance(hist, list):
        for item in hist:
            if isinstance(item, dict):
                a = item.get("action") or item.get("tool") or item.get("name")
                if a in ALL_CLASSES:
                    actions.append(a)

    for key in ["prev3_action", "prev_action", "last_action"]:
        v = meta.get(key)
        if isinstance(v, str):
            for p in re.split(r"[,\|>/\s]+", v):
                if p in ALL_CLASSES:
                    actions.append(p)

    return [a for a in actions if a in ALL_CLASSES]


def bucket_num(x, bins):
    try:
        x = float(x)
    except Exception:
        return "unk"
    for i, b in enumerate(bins):
        if x <= b:
            return f"b{i}"
    return f"b{len(bins)}"


def has_any(text, words):
    text = str(text).lower()
    return int(any(w.lower() in text for w in words))


def keyword_features(text):
    return {
        "kw_test": has_any(text, ["test", "tests", "pytest", "unittest", "테스트", "깨졌", "fail", "failed", "failing", "다시 돌", "log"]),
        "kw_lint": has_any(text, ["lint", "typecheck", "type-check", "mypy", "pyright", "eslint", "tsc", "ruff", "flake8", "타입", "린트"]),
        "kw_search": has_any(text, ["search", "grep", "find", "찾", "검색", "어디", "where", "reference", "references", "contains", "사용처", "쓰는"]),
        "kw_web": has_any(text, ["web", "browser", "online", "internet", "google", "구글", "인터넷", "검색해봐"]),
        "kw_edit": has_any(text, ["edit", "modify", "fix", "change", "수정", "고쳐", "바꿔"]),
        "kw_plan": has_any(text, ["plan", "todo", "strategy", "먼저", "계획", "단계"]),
        "kw_ask": has_any(text, ["ask", "clarify", "confirm", "물어", "질문", "확인해줘"]),
        "kw_read": has_any(text, ["read", "open", "show", "cat", "열어", "보여", "읽어", "확인"]),
        "kw_list": has_any(text, ["list", "directory", "folder", "ls", "목록", "폴더", "디렉토리"]),
        "kw_glob": has_any(text, ["glob", "pattern", "*.py", "*.js", "*.ts", "*.tsx", "*.json", "*.yaml", "all files", "파일들", "전부"]),
    }


def file_features(text):
    t = str(text).lower()
    exts = [".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml", ".md", ".java", ".rs", ".go", ".sql", ".tf"]
    out = {}
    for e in exts:
        out["has_ext_" + e.replace(".", "")] = int(e in t)
    out["has_path_like"] = int(bool(re.search(r"\b[\w\-.]+/[\w\-./]+", t)))
    out["has_star_glob"] = int("*." in t or "**/" in t)
    return out


def build_features(sample):
    meta = get_meta(sample)
    cur = safe_str(sample.get("current_prompt", ""))

    actions = extract_actions(sample)
    cnt = Counter(actions)

    last_action = safe_str(meta.get("last_action", actions[-1] if len(actions) >= 1 else "none"))
    prev_action = safe_str(meta.get("prev_action", actions[-2] if len(actions) >= 2 else "none"))
    tail3 = "_".join(actions[-3:]) if actions else "none"

    hist = sample.get("history", [])
    try:
        history_len = len(json.dumps(hist, ensure_ascii=False))
    except Exception:
        history_len = len(str(hist))

    turn_index = meta.get("turn_index", meta.get("turn", 0)) or 0
    step = meta.get("step", 0) or 0
    prompt_len = len(cur)

    row = {
        "last_action": last_action,
        "prev_action": prev_action,
        "prev3_action": safe_str(meta.get("prev3_action", tail3)),
        "last_prev": f"{last_action}|{prev_action}",
        "action_tail3": tail3,
        "num_actions": bucket_num(len(actions), [0, 1, 2, 4, 7, 12]),
        "prompt_len_bucket": bucket_num(prompt_len, [20, 50, 100, 200, 400]),
        "turn_bucket": bucket_num(turn_index, [0, 1, 2, 4, 7, 12]),
        "step_bucket": bucket_num(step, [0, 1, 2, 4, 7, 12]),
        "user_tier": safe_str(meta.get("user_tier", "none")),
        "language_pref": safe_str(meta.get("language_pref", "none")),
        "git_dirty": safe_str(meta.get("git_dirty", "none")),
        "last_ci_status": safe_str(meta.get("last_ci_status", "none")),
        "main_lang": safe_str(meta.get("main_lang", "none")),

        "history_len": history_len,
        "num_user_turns": float(meta.get("num_user_turns", 0) or 0),
        "prompt_len": prompt_len,
        "turn_index": float(turn_index or 0),
        "step": float(step or 0),
        "budget_tokens_remaining": float(meta.get("budget_tokens_remaining", 0) or 0),
        "elapsed_session_sec": float(meta.get("elapsed_session_sec", 0) or 0),
        "loc": float(meta.get("loc", 0) or 0),
        "num_open_files": float(meta.get("num_open_files", 0) or 0),
        "num_langs": float(meta.get("num_langs", 0) or 0),
        "max_lang_ratio": float(meta.get("max_lang_ratio", 0) or 0),
    }

    for a in ALL_CLASSES:
        row[f"cnt_{a}"] = cnt[a]

    row.update(keyword_features(cur))
    row.update(file_features(cur))

    return row


def main():
    print("TEAM_VAL_DIR:", TEAM_VAL_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("N_SEEDS:", N_SEEDS)
    print("BASE_SEED:", BASE_SEED)

    team_val_df = pd.read_csv(TEAM_VAL_DIR / "val_predictions.csv")
    valid_ids_order = team_val_df["id"].astype(str).tolist()
    valid_ids = set(valid_ids_order)

    print("team valid n:", len(valid_ids_order))

    samples = read_jsonl(DATA_DIR / "train.jsonl")
    labels_df = pd.read_csv(DATA_DIR / "train_labels.csv")

    label_col = "action" if "action" in labels_df.columns else "label"
    label_map = dict(zip(labels_df["id"].astype(str), labels_df[label_col].astype(str)))
    id_to_sample = {str(s.get("id")): s for s in samples}

    train_ids = [str(s.get("id")) for s in samples if str(s.get("id")) not in valid_ids]

    print("train n:", len(train_ids), "valid n:", len(valid_ids_order))

    train_samples = [id_to_sample[i] for i in train_ids]
    valid_samples = [id_to_sample[i] for i in valid_ids_order]

    y_train = np.array([label2id[label_map[i]] for i in train_ids], dtype=np.int64)
    y_valid = np.array([label2id[label_map[i]] for i in valid_ids_order], dtype=np.int64)

    print("building features...")
    X_train = pd.DataFrame([build_features(s) for s in train_samples])
    X_valid = pd.DataFrame([build_features(s) for s in valid_samples])

    categorical_cols = [
        "last_action", "prev_action", "prev3_action", "last_prev",
        "action_tail3", "num_actions", "prompt_len_bucket",
        "turn_bucket", "step_bucket", "user_tier", "language_pref",
        "git_dirty", "last_ci_status", "main_lang",
    ]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    print("X_train:", X_train.shape, "X_valid:", X_valid.shape)
    print("numeric n:", len(numeric_cols))

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    all_probs = []
    models = []

    for i in range(N_SEEDS):
        seed = BASE_SEED + i
        print("\n" + "=" * 80)
        print("training seed:", seed)

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]), categorical_cols),
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]), numeric_cols),
            ],
            remainder="drop",
        )

        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_iter=450,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=30,
            verbose=0,
        )

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("clf", clf),
        ])

        pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)

        prob_raw = pipe.predict_proba(X_valid)

        prob = np.zeros((len(X_valid), len(ALL_CLASSES)), dtype=np.float32)
        for j, cls in enumerate(pipe.named_steps["clf"].classes_):
            prob[:, int(cls)] = prob_raw[:, j]

        pred = prob.argmax(axis=1)
        f1 = f1_score(y_valid, pred, average="macro")
        acc = accuracy_score(y_valid, pred)

        print(f"seed={seed} val macro_f1={f1:.6f} acc={acc:.6f}")

        model_path = OUT_DIR / f"tabular_seed{seed}.joblib"
        joblib.dump(pipe, model_path)
        models.append(str(model_path))
        all_probs.append(prob)

    print("\n" + "=" * 80)
    print("SEED SELECTION")

    seed_scores = []
    for i, prob in enumerate(all_probs):
        seed = BASE_SEED + i
        pred = prob.argmax(axis=1)
        f1 = f1_score(y_valid, pred, average="macro")
        acc = accuracy_score(y_valid, pred)
        seed_scores.append((f1, acc, i, seed))
        print(f"single seed={seed} macro_f1={f1:.6f} acc={acc:.6f}")

    seed_scores.sort(reverse=True, key=lambda x: x[0])

    all_mean_prob = np.mean(all_probs, axis=0)
    all_pred = all_mean_prob.argmax(axis=1)
    all_f1 = f1_score(y_valid, all_pred, average="macro")
    all_acc = accuracy_score(y_valid, all_pred)
    print(f"all-seed average macro_f1={all_f1:.6f} acc={all_acc:.6f}")

    selected = [seed_scores[0][2]]
    best_prob = all_probs[selected[0]].copy()
    best_f1 = seed_scores[0][0]
    best_acc = seed_scores[0][1]

    print(f"greedy start seed={BASE_SEED + selected[0]} f1={best_f1:.6f}")

    while True:
        best_candidate = None

        for _, _, i, seed in seed_scores:
            if i in selected:
                continue

            cand_indices = selected + [i]
            cand_prob = np.mean([all_probs[j] for j in cand_indices], axis=0)
            cand_pred = cand_prob.argmax(axis=1)
            cand_f1 = f1_score(y_valid, cand_pred, average="macro")
            cand_acc = accuracy_score(y_valid, cand_pred)

            if cand_f1 > best_f1 + 1e-12:
                best_candidate = (cand_f1, cand_acc, i, seed, cand_prob)

        if best_candidate is None:
            break

        best_f1, best_acc, i, seed, best_prob = best_candidate
        selected.append(i)
        print(f"greedy add seed={seed} -> f1={best_f1:.6f} acc={best_acc:.6f} selected={[BASE_SEED+j for j in selected]}")

    greedy_seeds = [BASE_SEED + i for i in selected]
    print(f"greedy selected seeds={greedy_seeds}")
    print(f"greedy ensemble macro_f1={best_f1:.6f} acc={best_acc:.6f}")

    if all_f1 > best_f1:
        selected_mode = "all_average"
        selected_seeds = [BASE_SEED + i for i in range(N_SEEDS)]
        mean_prob = all_mean_prob
        mean_f1 = all_f1
        mean_acc = all_acc
    else:
        selected_mode = "greedy_subset"
        selected_seeds = greedy_seeds
        mean_prob = best_prob
        mean_f1 = best_f1
        mean_acc = best_acc

    print("\n" + "=" * 80)
    print(f"SELECTED mode={selected_mode}")
    print(f"SELECTED seeds={selected_seeds}")
    print(f"SELECTED macro_f1={mean_f1:.6f} acc={mean_acc:.6f}")

    np.save(OUT_DIR / "val_probs_all_seeds.npy", np.stack(all_probs, axis=0).astype(np.float32))

    val_logits = np.log(np.clip(mean_prob, 1e-8, 1.0)).astype(np.float32)
    np.save(OUT_DIR / "val_logits.npy", val_logits)

    pred = mean_prob.argmax(axis=1)
    out_df = pd.DataFrame({
        "id": valid_ids_order,
        "true_id": y_valid,
        "true_action": [ALL_CLASSES[i] for i in y_valid],
        "pred_id": pred,
        "pred_action": [ALL_CLASSES[i] for i in pred],
        "confidence": mean_prob.max(axis=1),
    })
    out_df.to_csv(OUT_DIR / "val_predictions.csv", index=False)

    info = {
        "model": "tabular_histgb_seed_ensemble",
        "team_val_dir": str(TEAM_VAL_DIR),
        "n_seeds": N_SEEDS,
        "base_seed": BASE_SEED,
        "seeds": [BASE_SEED + i for i in range(N_SEEDS)],
        "selected_mode": selected_mode,
        "selected_seeds": selected_seeds,
        "macro_f1": float(mean_f1),
        "accuracy": float(mean_acc),
        "train_n": len(train_ids),
        "valid_n": len(valid_ids_order),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "all_classes": ALL_CLASSES,
        "models": models,
    }

    json.dump(info, open(OUT_DIR / "val_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(ALL_CLASSES, open(OUT_DIR / "labels.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\nsaved:")
    print(" -", OUT_DIR / "val_logits.npy")
    print(" -", OUT_DIR / "val_predictions.csv")
    print(" -", OUT_DIR / "val_info.json")
    print(" -", OUT_DIR / "labels.json")
    print(" -", OUT_DIR / "val_probs_all_seeds.npy")


if __name__ == "__main__":
    main()
