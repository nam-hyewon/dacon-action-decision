import csv
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MODEL_DIR = Path("submit/model/deberta_base_eda_state4")
VAL_CSV = MODEL_DIR / "val_predictions.csv"
VAL_LOGITS = MODEL_DIR / "val_logits.npy"
SPECIALIST_PATH = MODEL_DIR / "specialist_boosters" / "specialist_booster.joblib"
OUT_PATH = MODEL_DIR / "group_reranker.joblib"

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {x: i for i, x in enumerate(ALL_CLASSES)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}

GROUPS = {
    "file_group": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "run_group": ["run_bash", "run_tests", "lint_or_typecheck"],
    "ask_group": ["ask_user", "plan_task", "web_search"],
    "edit_group": ["edit_file", "apply_patch"],
}


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
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


def compact_history(sample, max_turns=10):
    hist = sample.get("history", [])[-max_turns:]
    parts = []
    actions = []
    user_utts = []
    last_result = ""

    for h in hist:
        role = h.get("role", "")
        if role == "user":
            txt = str(h.get("content", ""))
            user_utts.append(txt)
            parts.append(f"U: {txt}")
        elif role == "assistant_action":
            name = str(h.get("name", ""))
            args = str(h.get("args", ""))
            result = str(h.get("result_summary", ""))
            actions.append(name)
            last_result = result
            parts.append(f"ACTION: {name} ARGS: {args} RESULT: {result}")

    return {
        "history_text": "\n".join(parts),
        "action_seq": " > ".join(actions),
        "recent_users": "\n".join(user_utts[-3:]),
        "last_result": last_result,
    }


def build_text_router(sample):
    meta = sample.get("session_meta", {})
    workspace = meta.get("workspace", {})
    current_prompt = str(sample.get("current_prompt", ""))
    lower = current_prompt.lower()

    ch = compact_history(sample)
    language_mix = workspace.get("language_mix", {}) or {}
    main_lang = max(language_mix, key=language_mix.get) if language_mix else "none"
    open_files = " ".join(map(str, workspace.get("open_files", [])))
    lang_mix = " ".join(f"{k}:{v}" for k, v in language_mix.items())

    cues = {
        "cue_read_show": bool(re.search(r"\b(show|open|read|cat|display|view|inspect|current impl|current implementation)\b", lower)),
        "cue_search": bool(re.search(r"\b(search|grep|find|look for|where|which file|matches|occurrences|contains)\b", lower)),
        "cue_list": bool(re.search(r"\b(list|ls|directory|folder|what'?s in|files under|tree)\b", lower)),
        "cue_glob": bool(re.search(r"\b(glob|pattern|wildcard|all .*\.|files matching|\*\.|recursive)\b", lower)),
        "cue_test": bool(re.search(r"\b(test|tests|pytest|unit test|integration test|ci|failing test)\b", lower)),
        "cue_lint": bool(re.search(r"\b(lint|typecheck|type check|mypy|ruff|eslint|tsc|format|formatter)\b", lower)),
        "cue_run": bool(re.search(r"\b(run|execute|start|launch|rebuild|install|command|bash|shell)\b", lower)),
        "cue_ask": bool(re.search(r"\b(ask|clarify|confirm|should i|do you want|which one|need more info)\b", lower)),
        "cue_plan": bool(re.search(r"\b(plan|steps|strategy|approach|todo|roadmap|before coding)\b", lower)),
        "cue_web": bool(re.search(r"\b(web|internet|latest|current|search online|look up|google|browse)\b", lower)),
        "cue_edit": bool(re.search(r"\b(fix|change|update|modify|patch|edit|add|remove|replace|refactor)\b", lower)),
    }
    cue_text = " ".join(f"{k}:{v}" for k, v in cues.items())

    return "\n".join([
        "[GROUP_RERANKER_CUES]",
        cue_text,
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


def load_data_by_id():
    samples = {}
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                samples[s["id"]] = s

    labels = {}
    with open(DATA_DIR / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    return samples, labels


def apply_specialist(logits, texts):
    if not SPECIALIST_PATH.exists():
        print("[WARN] specialist not found:", SPECIALIST_PATH)
        return logits.copy()

    artifact = joblib.load(SPECIALIST_PATH)

    vec = artifact["vectorizer"]
    X = vec.transform(texts)

    clfs = artifact.get("clfs", artifact.get("classifiers", {}))
    gammas = artifact.get("gammas", artifact.get("target_gammas", {}))
    all_classes = artifact.get("all_classes", ALL_CLASSES)
    label2id = {a: i for i, a in enumerate(all_classes)}

    eps = float(artifact.get("safe_logit_eps", artifact.get("eps", 1e-5)))
    score_clip = artifact.get("score_clip", [-8, 8])
    clip_low, clip_high = float(score_clip[0]), float(score_clip[1])

    boosted = logits.copy()

    for target, gamma in gammas.items():
        if target not in clfs or target not in label2id:
            continue

        clf = clfs[target]

        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X)[:, 1]
            p = np.clip(p, eps, 1.0 - eps)
            score = np.log(p / (1.0 - p))
        else:
            score = clf.decision_function(X)

        score = np.clip(score, clip_low, clip_high)
        boosted[:, label2id[target]] += float(gamma) * score

    return boosted


def macro(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        labels=list(range(len(ALL_CLASSES))),
        average="macro",
        zero_division=0,
    )


def train_group_models(df, texts):
    models = {}

    for gname, classes in GROUPS.items():
        idx = df["true_action"].isin(classes).values
        sub = df[idx].copy()
        sub_texts = [texts[i] for i, ok in enumerate(idx) if ok]
        y = sub["true_action"].values

        print(f"[TRAIN] {gname}: n={len(sub)} counts={dict(pd.Series(y).value_counts())}")

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        )
        X = vec.fit_transform(sub_texts)

        clf = LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X, y)

        models[gname] = {
            "classes": classes,
            "vectorizer": vec,
            "classifier": clf,
        }

    return models


def apply_group_rerank(logits_after_specialist, texts, models, topk=3):
    pred = logits_after_specialist.argmax(axis=1).copy()
    new_pred = pred.copy()
    top = np.argsort(-logits_after_specialist, axis=1)[:, :topk]

    changed = 0

    for i in range(len(texts)):
        top_classes = [ID2LABEL[int(j)] for j in top[i]]

        for gname, obj in models.items():
            group = obj["classes"]
            in_group = [c for c in top_classes if c in group]

            if len(set(in_group)) < 2:
                continue

            X = obj["vectorizer"].transform([texts[i]])
            clf = obj["classifier"]
            proba = clf.predict_proba(X)[0]
            cls = list(clf.classes_)
            best_local = cls[int(np.argmax(proba))]

            old = ID2LABEL[int(new_pred[i])]

            # 보수적으로: reranker가 고른 클래스가 base top-k 안에 있을 때만 변경
            if best_local in top_classes and best_local != old:
                new_pred[i] = LABEL2ID[best_local]
                changed += 1

            break

    return new_pred, changed


def main():
    print("MODEL_DIR:", MODEL_DIR)
    print("VAL_CSV:", VAL_CSV)
    print("VAL_LOGITS:", VAL_LOGITS)
    print("SPECIALIST_PATH:", SPECIALIST_PATH)

    df = pd.read_csv(VAL_CSV)
    logits = np.load(VAL_LOGITS)

    samples, labels = load_data_by_id()
    ids = df["id"].astype(str).tolist()
    texts = [build_text_router(samples[sid]) for sid in ids]
    y_true = np.array([LABEL2ID[labels[sid]] for sid in ids], dtype=np.int64)

    base_pred = logits.argmax(axis=1)
    print("[BASE MODEL ONLY]")
    print("macro:", macro(y_true, base_pred), "acc:", accuracy_score(y_true, base_pred))

    spec_logits = apply_specialist(logits, texts)
    spec_pred = spec_logits.argmax(axis=1)
    print("[AFTER SPECIALIST]")
    print("macro:", macro(y_true, spec_pred), "acc:", accuracy_score(y_true, spec_pred))

    from sklearn.model_selection import StratifiedShuffleSplit

    df_all = df.copy()
    df_all["true_action"] = [labels[sid] for sid in ids]

    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.30, random_state=42)

    holdout_results = []

    for split_idx, (tr_idx, ho_idx) in enumerate(splitter.split(df_all, y_true), 1):
        print(f"\\n===== HOLDOUT SPLIT {split_idx} =====")

        df_train = df_all.iloc[tr_idx].reset_index(drop=True)
        train_texts = [texts[i] for i in tr_idx]

        ho_texts = [texts[i] for i in ho_idx]
        ho_y = y_true[ho_idx]
        ho_logits = spec_logits[ho_idx]

        ho_spec_pred = ho_logits.argmax(axis=1)
        print("[HOLDOUT SPECIALIST ONLY]")
        print("macro:", macro(ho_y, ho_spec_pred), "acc:", accuracy_score(ho_y, ho_spec_pred))

        models = train_group_models(df_train, train_texts)

        best_split = None
        for topk in [2, 3, 4]:
            pred2, changed = apply_group_rerank(ho_logits, ho_texts, models, topk=topk)
            mf1 = macro(ho_y, pred2)
            acc = accuracy_score(ho_y, pred2)

            print(f"[HOLDOUT GROUP RERANK] topk={topk} macro={mf1:.6f} acc={acc:.6f} changed={changed}")

            if best_split is None or mf1 > best_split["macro_f1"]:
                best_split = {
                    "split": split_idx,
                    "topk": topk,
                    "macro_f1": float(mf1),
                    "accuracy": float(acc),
                    "changed": int(changed),
                }

        print("[BEST SPLIT]", best_split)
        holdout_results.append(best_split)

    print("\\n===== HOLDOUT SUMMARY =====")
    for r in holdout_results:
        print(r)

    print("mean macro:", float(np.mean([r["macro_f1"] for r in holdout_results])))
    print("mean acc:", float(np.mean([r["accuracy"] for r in holdout_results])))

    # 마지막에는 전체 val로 artifact 학습/저장만 함. 점수는 holdout summary를 믿어야 함.
    models = train_group_models(df_all, texts)
    best_topk = max(set([r["topk"] for r in holdout_results]), key=[r["topk"] for r in holdout_results].count)

    artifact = {
        "groups": GROUPS,
        "models": models,
        "topk": best_topk,
        "all_classes": ALL_CLASSES,
        "note": "trained on full validation after holdout evaluation; holdout score is the relevant estimate",
    }
    joblib.dump(artifact, OUT_PATH)
    print("[INFO] saved full-val-trained group reranker:", OUT_PATH)
    print("[INFO] selected topk from holdout majority:", best_topk)


if __name__ == "__main__":
    main()
