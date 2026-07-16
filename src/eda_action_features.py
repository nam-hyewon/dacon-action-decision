import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "eda_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

LABEL2ID = {x: i for i, x in enumerate(ALL_CLASSES)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}


def load_data():
    samples = []
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    labels = {}
    with open(DATA_DIR / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    rows = []
    for s in samples:
        sid = s["id"]
        if sid not in labels:
            continue
        row = extract_features(s)
        row["id"] = sid
        row["label"] = labels[sid]
        row["label_id"] = LABEL2ID[labels[sid]]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, samples, labels


def get_session_id(sid):
    # sess_sim_20260522_024730-step_08 -> sess_sim_20260522_024730
    return re.sub(r"-step_\d+$", "", str(sid))


def get_step(sid):
    m = re.search(r"step_(\d+)", str(sid))
    return int(m.group(1)) if m else -1


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_user_turns(sample):
    return [
        str(h.get("content", ""))
        for h in sample.get("history", [])
        if h.get("role") == "user"
    ]


def compact_history_text(sample, max_turns=8):
    hist = sample.get("history", [])[-max_turns:]
    parts = []
    for h in hist:
        role = h.get("role", "")
        if role == "user":
            txt = clean(h.get("content", ""), 500)
            parts.append(f"U: {txt}")
        elif role == "assistant_action":
            name = str(h.get("name", ""))
            args = clean(json.dumps(h.get("args", {}), ensure_ascii=False), 300)
            result = clean(h.get("result_summary", ""), 300)
            parts.append(f"ACTION:{name} ARGS:{args} RESULT:{result}")
    return "\n".join(parts)


def clean(x, max_chars=None):
    x = re.sub(r"\s+", " ", str(x)).strip()
    if max_chars is not None and len(x) > max_chars:
        return x[:max_chars]
    return x


def contains(pattern, text):
    return bool(re.search(pattern, str(text).lower()))


def extract_paths(text):
    text = str(text)
    paths = re.findall(
        r"[\w./-]+\.(?:py|js|ts|tsx|jsx|json|yaml|yml|toml|md|txt|css|html|go|rs|java|cpp|c|h|sh|sql|env)",
        text,
    )
    globs = re.findall(r"\*\.[A-Za-z0-9_]+", text)
    return paths[:20], globs[:20]


def exts_from_paths(paths):
    exts = []
    for p in paths:
        m = re.search(r"\.([A-Za-z0-9_]+)$", str(p))
        if m:
            exts.append(m.group(1).lower())
    return sorted(set(exts))


def parse_last_result(text):
    t = str(text).lower()
    return {
        "last_result_empty": len(t.strip()) == 0,
        "last_result_has_error": contains(r"\b(error|failed|fail|traceback|exception|not found|missing)\b", t),
        "last_result_has_pass": contains(r"\b(pass|passed|success|ok)\b", t),
        "last_result_has_match": contains(r"\b(match|matches|occurrence|found)\b", t),
        "last_result_no_match": contains(r"\b(no matches|0 matches|not found)\b", t),
        "last_result_has_patch": contains(r"\b(patched|updated|modified|changed|applied)\b", t),
        "last_result_has_list": contains(r"\b(files|entries|directory|listed)\b", t),
        "last_result_has_test": contains(r"\b(test|tests|pytest|unit)\b", t),
        "last_result_has_lint": contains(r"\b(lint|typecheck|mypy|ruff|eslint|tsc)\b", t),
    }


def extract_features(sample):
    sid = str(sample.get("id", ""))
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    history = sample.get("history", []) or []
    prompt = str(sample.get("current_prompt", ""))
    prompt_l = prompt.lower()

    actions = get_actions(sample)
    user_turns = get_user_turns(sample)
    last_action = actions[-1] if actions else "none"
    prev_action = actions[-2] if len(actions) >= 2 else "none"

    last_result = ""
    last_args = ""
    for h in reversed(history):
        if h.get("role") == "assistant_action":
            last_result = str(h.get("result_summary", ""))
            last_args = json.dumps(h.get("args", {}), ensure_ascii=False)
            break

    open_files = workspace.get("open_files", []) or []
    prompt_paths, prompt_globs = extract_paths(prompt)
    args_paths, args_globs = extract_paths(last_args)
    result_paths, result_globs = extract_paths(last_result)
    open_exts = exts_from_paths(open_files)
    prompt_exts = exts_from_paths(prompt_paths)
    all_exts = sorted(set(open_exts + prompt_exts + exts_from_paths(args_paths + result_paths)))

    result_info = parse_last_result(last_result)

    row = {
        "session_id": get_session_id(sid),
        "step": get_step(sid),
        "turn_index": meta.get("turn_index", -1),
        "history_len": len(history),
        "num_actions": len(actions),
        "num_user_turns": len(user_turns),
        "prompt_len": len(prompt),
        "prompt_word_count": len(prompt.split()),
        "last_action": last_action,
        "prev_action": prev_action,
        "action_seq_last3": ">".join(actions[-3:]) if actions else "none",
        "action_seq_last5": ">".join(actions[-5:]) if actions else "none",
        "git_dirty": str(workspace.get("git_dirty", "")),
        "last_ci_status": str(workspace.get("last_ci_status", "")),
        "num_open_files": len(open_files),
        "main_lang": main_lang(workspace.get("language_mix", {}) or {}),
        "has_open_file": len(open_files) > 0,
        "prompt_has_path": len(prompt_paths) > 0,
        "prompt_has_glob": len(prompt_globs) > 0,
        "args_has_path": len(args_paths) > 0,
        "result_has_path": len(result_paths) > 0,
        "has_py": "py" in all_exts,
        "has_js_ts": bool(set(all_exts) & {"js", "jsx", "ts", "tsx"}),
        "has_config_ext": bool(set(all_exts) & {"json", "yaml", "yml", "toml", "env"}),
        "has_md": "md" in all_exts,
        "has_test_path": contains(r"(test|spec)", " ".join(map(str, prompt_paths + open_files + args_paths))),
        "current_prompt": prompt,
        "recent_users": "\n".join(user_turns[-3:]),
        "last_result": last_result,
        "last_args": last_args,
        "compact_history": compact_history_text(sample),
    }

    # prompt intent cues
    cue_patterns = {
        "kw_read": r"\b(read|open|show|view|display|inspect|look at|cat)\b",
        "kw_search": r"\b(search|grep|find|look for|where|which file|contains|occurrences?|matches?)\b",
        "kw_list": r"\b(list|ls|directory|folder|tree|what'?s in|files under)\b",
        "kw_glob": r"\b(glob|pattern|wildcard|files matching|all .* files|\*\.)\b",
        "kw_edit": r"\b(fix|change|update|modify|edit|patch|add|remove|replace|refactor)\b",
        "kw_write": r"\b(create|write|new file|generate file|save)\b",
        "kw_run": r"\b(run|execute|launch|start|install|command|bash|shell)\b",
        "kw_test": r"\b(test|tests|pytest|unit test|integration test|spec)\b",
        "kw_lint": r"\b(lint|typecheck|type check|mypy|ruff|eslint|tsc|format|formatter|prettier)\b",
        "kw_ask": r"\b(ask|clarify|confirm|which one|do you want|should i|need more info)\b",
        "kw_plan": r"\b(plan|steps|strategy|approach|roadmap|todo|before coding)\b",
        "kw_web": r"\b(web|internet|online|latest|current|look up|search online|browse|google)\b",
        "kw_error": r"\b(error|traceback|exception|failed|failing|bug|broken)\b",
    }
    for k, pat in cue_patterns.items():
        row[k] = contains(pat, prompt_l)

    for k, v in result_info.items():
        row[k] = v

    # action history booleans
    for a in ALL_CLASSES:
        row[f"ever_{a}"] = a in actions
        row[f"count_{a}"] = actions.count(a)

    return row


def main_lang(language_mix):
    if not language_mix:
        return "none"
    try:
        return max(language_mix, key=language_mix.get)
    except Exception:
        return "none"


def save_value_counts(df):
    with open(OUT_DIR / "00_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"n={len(df)}\n")
        f.write(f"sessions={df['session_id'].nunique()}\n")
        f.write(f"avg_steps_per_session={len(df)/df['session_id'].nunique():.3f}\n")
        f.write("\n[label counts]\n")
        f.write(df["label"].value_counts().to_string())
        f.write("\n\n[step describe]\n")
        f.write(df["step"].describe().to_string())

    df["label"].value_counts().rename_axis("label").reset_index(name="count").to_csv(
        OUT_DIR / "01_label_counts.csv", index=False
    )


def conditional_tables(df):
    # feature value -> label distribution tables
    features = [
        "last_action", "prev_action", "action_seq_last3",
        "step", "history_len", "num_actions",
        "last_ci_status", "git_dirty", "main_lang",
        "has_open_file", "prompt_has_path", "prompt_has_glob",
        "has_py", "has_js_ts", "has_config_ext", "has_test_path",
        "last_result_has_error", "last_result_has_pass",
        "last_result_has_match", "last_result_no_match",
        "last_result_has_patch", "last_result_has_list",
        "kw_read", "kw_search", "kw_list", "kw_glob",
        "kw_edit", "kw_run", "kw_test", "kw_lint",
        "kw_ask", "kw_plan", "kw_web", "kw_error",
    ]

    records = []
    for feat in features:
        if feat not in df.columns:
            continue
        tmp = pd.crosstab(df[feat], df["label"], normalize="index")
        cnt = df[feat].value_counts()
        for val in tmp.index:
            dist = tmp.loc[val].sort_values(ascending=False)
            top = dist.index[0]
            records.append({
                "feature": feat,
                "value": str(val),
                "count": int(cnt.get(val, 0)),
                "top_label": top,
                "top_prob": float(dist.iloc[0]),
                "second_label": dist.index[1] if len(dist) > 1 else "",
                "second_prob": float(dist.iloc[1]) if len(dist) > 1 else 0.0,
                "entropy": float(-(dist * np.log2(dist + 1e-12)).sum()),
            })

    out = pd.DataFrame(records).sort_values(
        ["feature", "count"], ascending=[True, False]
    )
    out.to_csv(OUT_DIR / "02_feature_value_label_distribution.csv", index=False)


def transition_matrix(df):
    mat = pd.crosstab(df["last_action"], df["label"], normalize="index")
    mat.to_csv(OUT_DIR / "03_last_action_to_label_prob.csv")

    cnt = pd.crosstab(df["last_action"], df["label"])
    cnt.to_csv(OUT_DIR / "03_last_action_to_label_count.csv")

    # top transitions
    records = []
    for last_action, row in mat.iterrows():
        counts = cnt.loc[last_action]
        for label in ALL_CLASSES:
            records.append({
                "last_action": last_action,
                "label": label,
                "prob": float(row.get(label, 0.0)),
                "count": int(counts.get(label, 0)),
                "support_last_action": int(counts.sum()),
            })
    pd.DataFrame(records).sort_values(
        ["last_action", "prob"], ascending=[True, False]
    ).to_csv(OUT_DIR / "03_last_action_top_transitions_long.csv", index=False)


def simple_feature_models(df):
    y = df["label_id"].values
    groups = df["session_id"].values

    # random split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(sss.split(df, y))

    # group split by session
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    gtr_idx, gva_idx = next(gss.split(df, y, groups=groups))

    configs = [
        ("random", tr_idx, va_idx),
        ("session_group", gtr_idx, gva_idx),
    ]

    cat_features = [
        "last_action", "prev_action", "action_seq_last3",
        "step", "history_len", "num_actions",
        "last_ci_status", "git_dirty", "main_lang",
        "has_open_file", "prompt_has_path", "prompt_has_glob",
        "has_py", "has_js_ts", "has_config_ext", "has_test_path",
        "last_result_has_error", "last_result_has_pass",
        "last_result_has_match", "last_result_no_match",
        "last_result_has_patch", "last_result_has_list",
        "kw_read", "kw_search", "kw_list", "kw_glob",
        "kw_edit", "kw_run", "kw_test", "kw_lint",
        "kw_ask", "kw_plan", "kw_web", "kw_error",
    ]

    text_fields = {
        "prompt": "current_prompt",
        "prompt_history": "current_prompt_plus_history",
    }

    df = df.copy()
    df["current_prompt_plus_history"] = (
        df["current_prompt"].fillna("") + "\n" +
        df["recent_users"].fillna("") + "\n" +
        df["last_result"].fillna("")
    )

    results = []

    for split_name, train_idx, val_idx in configs:
        y_train = y[train_idx]
        y_val = y[val_idx]

        # transition-only baseline: predict most common label for last_action
        pred = transition_baseline(df.iloc[train_idx], df.iloc[val_idx], key="last_action")
        results.append(score_row(split_name, "transition_last_action", y_val, pred))

        pred = transition_baseline(df.iloc[train_idx], df.iloc[val_idx], key="action_seq_last3")
        results.append(score_row(split_name, "transition_action_seq_last3", y_val, pred))

        # tabular categorical LR
        enc = OneHotEncoder(handle_unknown="ignore")
        Xtr_cat = enc.fit_transform(df.iloc[train_idx][cat_features].astype(str))
        Xva_cat = enc.transform(df.iloc[val_idx][cat_features].astype(str))
        clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr_cat, y_train)
        pred = clf.predict(Xva_cat)
        results.append(score_row(split_name, "tabular_cues_lr", y_val, pred))

        # text-only and text+tabular
        for model_name, col in text_fields.items():
            vec = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=120000,
                sublinear_tf=True,
            )
            Xtr_txt = vec.fit_transform(df.iloc[train_idx][col].fillna(""))
            Xva_txt = vec.transform(df.iloc[val_idx][col].fillna(""))

            clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", solver="lbfgs")
            clf.fit(Xtr_txt, y_train)
            pred = clf.predict(Xva_txt)
            results.append(score_row(split_name, f"tfidf_{model_name}", y_val, pred))

            Xtr = hstack([Xtr_txt, Xtr_cat]).tocsr()
            Xva = hstack([Xva_txt, Xva_cat]).tocsr()
            clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", solver="lbfgs")
            clf.fit(Xtr, y_train)
            pred = clf.predict(Xva)
            results.append(score_row(split_name, f"tfidf_{model_name}_plus_tabular", y_val, pred))

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "04_simple_feature_model_scores.csv", index=False)
    print(res.sort_values(["split", "macro_f1"], ascending=[True, False]).to_string(index=False))


def transition_baseline(train_df, val_df, key):
    global_major = train_df["label"].value_counts().idxmax()
    mp = {}
    for val, sub in train_df.groupby(key):
        mp[str(val)] = sub["label"].value_counts().idxmax()
    pred_labels = [mp.get(str(v), global_major) for v in val_df[key]]
    return np.array([LABEL2ID[x] for x in pred_labels], dtype=np.int64)


def score_row(split, model, y_true, y_pred):
    return {
        "split": split,
        "model": model,
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def top_keywords_by_label(df):
    # quick interpretable TF-IDF coefficients for current_prompt
    y = df["label_id"].values
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=80000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(df["current_prompt"].fillna(""))
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", solver="lbfgs")
    clf.fit(X, y)

    names = np.array(vec.get_feature_names_out())
    rows = []
    for class_id, label in ID2LABEL.items():
        coef = clf.coef_[class_id]
        top_idx = np.argsort(-coef)[:50]
        for rank, idx in enumerate(top_idx, 1):
            rows.append({
                "label": label,
                "rank": rank,
                "term": names[idx],
                "coef": float(coef[idx]),
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "05_prompt_top_terms_by_label.csv", index=False)


def confusion_groups(df):
    groups = {
        "file_group": ["read_file", "grep_search", "list_directory", "glob_pattern"],
        "run_group": ["run_bash", "run_tests", "lint_or_typecheck"],
        "ask_group": ["ask_user", "plan_task", "web_search"],
        "edit_group": ["edit_file", "apply_patch"],
    }
    rows = []
    for gname, labels in groups.items():
        sub = df[df["label"].isin(labels)].copy()
        for feat in [
            "last_action", "prev_action",
            "kw_read", "kw_search", "kw_list", "kw_glob",
            "kw_run", "kw_test", "kw_lint",
            "kw_ask", "kw_plan", "kw_web",
            "last_result_has_match", "last_result_has_error",
            "prompt_has_path", "prompt_has_glob", "has_test_path",
        ]:
            if feat not in sub.columns:
                continue
            ct = pd.crosstab(sub[feat], sub["label"], normalize="index")
            cnt = sub[feat].value_counts()
            for val in ct.index:
                dist = ct.loc[val].sort_values(ascending=False)
                rows.append({
                    "group": gname,
                    "feature": feat,
                    "value": str(val),
                    "count": int(cnt.get(val, 0)),
                    "top_label": dist.index[0],
                    "top_prob": float(dist.iloc[0]),
                    "second_label": dist.index[1] if len(dist) > 1 else "",
                    "second_prob": float(dist.iloc[1]) if len(dist) > 1 else 0.0,
                })
    pd.DataFrame(rows).sort_values(
        ["group", "feature", "count"], ascending=[True, True, False]
    ).to_csv(OUT_DIR / "06_confusion_group_feature_tables.csv", index=False)


def main():
    print("[LOAD]")
    df, samples, labels = load_data()
    print("df:", df.shape)
    print("labels:", df["label"].value_counts().to_dict())
    print("sessions:", df["session_id"].nunique())

    print("[SAVE SUMMARY]")
    save_value_counts(df)

    print("[CONDITIONAL TABLES]")
    conditional_tables(df)

    print("[TRANSITIONS]")
    transition_matrix(df)

    print("[SIMPLE FEATURE MODELS]")
    simple_feature_models(df)

    print("[TOP KEYWORDS]")
    top_keywords_by_label(df)

    print("[CONFUSION GROUP TABLES]")
    confusion_groups(df)

    print("[DONE] outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
