import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BEST_DIR = ROOT / "submit/model/eda_mdeberta_deberta_w054_046"
OUT_DIR = ROOT / "eda_outputs/file_group_lr_reranker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_CSV = BEST_DIR / "val_predictions.csv"
VAL_LOGITS = BEST_DIR / "val_logits.npy"

ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}

FILE_ACTIONS = ["read_file", "grep_search", "list_directory", "glob_pattern"]
FILE_SET = set(FILE_ACTIONS)
FILE_TO_ID = {a: i for i, a in enumerate(FILE_ACTIONS)}
ID_TO_FILE = {i: a for a, i in FILE_TO_ID.items()}


def find_cols(df):
    id_col = "id"
    true_col = None
    for c in ["true_action", "true", "label", "true_label", "target"]:
        if c in df.columns:
            true_col = c
            break
    if true_col is None:
        raise ValueError(df.columns.tolist())
    return id_col, true_col


def clean(x):
    return re.sub(r"\s+", " ", str(x)).strip()


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_last_user_text(sample):
    users = [
        str(h.get("content", ""))
        for h in sample.get("history", [])
        if h.get("role") == "user"
    ]
    return users[-1] if users else ""


def get_last_result(sample):
    for h in reversed(sample.get("history", [])):
        if h.get("role") == "assistant_action":
            return str(h.get("result_summary", ""))
    return ""


def get_step(sample):
    sid = str(sample.get("id", ""))
    m = re.search(r"step_(\d+)", sid)
    return int(m.group(1)) if m else -1


def path_counts(text):
    text = str(text)
    file_paths = re.findall(
        r"[\w./-]+\.(?:py|js|ts|tsx|jsx|json|yaml|yml|toml|md|txt|css|html|go|rs|java|cpp|c|h|sh|sql|env)",
        text,
    )
    glob_patterns = re.findall(r"(?:\*[\w./-]*\.\w+|[\w./-]*\*\w*)", text)
    return len(file_paths), len(glob_patterns)


def make_text(sample):
    prompt = clean(sample.get("current_prompt", ""))
    last_user = clean(get_last_user_text(sample))
    last_result = clean(get_last_result(sample))
    actions = get_actions(sample)
    prev = " ".join(actions[-5:])
    return f"[PROMPT] {prompt} [LAST_USER] {last_user} [PREV_ACTIONS] {prev} [LAST_RESULT] {last_result[:500]}"


def make_num_cat(sample):
    prompt = str(sample.get("current_prompt", ""))
    p = prompt.lower()
    actions = get_actions(sample)
    prev1 = actions[-1] if len(actions) >= 1 else "none"
    prev2 = actions[-2] if len(actions) >= 2 else "none"
    step = get_step(sample)

    n_paths, n_globs = path_counts(prompt)

    bools = [
        int(n_paths > 0),
        int(n_globs > 0),
        int(bool(re.search(r"\*\.", p))),
        int(bool(re.search(r"\b(where|search|grep|find|reference|references|used|uses|call|calls|todo)\b|어디|찾아|참조|호출|비슷한", p))),
        int(bool(re.search(r"\b(open|read|show|view|display|inspect|look at|cat)\b|열어|보자|봐줘|펼쳐줘", p))),
        int(bool(re.search(r"\b(directory|folder|dir|what'?s in|list|ls|tree)\b|디렉토리|폴더|뭐뭐|뭐 있|목록|펼쳐", p))),
        int(bool(re.search(r"\b(all|every)\b|전부|모든", p))),
        int(bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\s*(?:\(|\.|::)?", prompt))),
        len(prompt),
        len(prompt.split()),
        n_paths,
        n_globs,
        step,
        len(actions),
    ]

    cats = [prev1, prev2]
    return bools, cats


def macro_f1(y, pred):
    return f1_score(y, pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)


def main():
    val_df = pd.read_csv(VAL_CSV)
    logits = np.load(VAL_LOGITS)
    probs = softmax(logits, axis=1)

    id_col, true_col = find_cols(val_df)
    val_ids = set(val_df[id_col].astype(str).tolist())

    all_samples = []
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                all_samples.append(s)

    # Load labels from separate label file
    label_df = pd.read_csv(DATA_DIR / "train_labels.csv")
    label_by_id = dict(zip(label_df["id"].astype(str), label_df["action"].astype(str)))

    rows = []
    missing = 0
    for s in all_samples:
        sid = str(s["id"])
        label = label_by_id.get(sid)
        if label is None:
            missing += 1
            continue
        rows.append((sid, label, s))

    if missing:
        print("[WARN] missing labels:", missing)

    train_rows = [(sid, y, s) for sid, y, s in rows if sid not in val_ids and y in FILE_SET]
    val_rows = [(sid, y, s) for sid, y, s in rows if sid in val_ids]

    print("[labeled rows]", len(rows))
    print("[train file rows]", len(train_rows))
    print("[val rows]", len(val_rows))

    X_text_train = [make_text(s) for _, _, s in train_rows]
    y_train_file = np.array([FILE_TO_ID[y] for _, y, _ in train_rows])

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=200000,
        sublinear_tf=True,
        lowercase=True,
    )
    Xtr_text = vec.fit_transform(X_text_train)

    train_num = []
    train_cat = []
    for _, _, s in train_rows:
        nums, cats = make_num_cat(s)
        train_num.append(nums)
        train_cat.append(cats)

    enc = OneHotEncoder(handle_unknown="ignore")
    Xtr_cat = enc.fit_transform(train_cat)
    Xtr_num = csr_matrix(np.asarray(train_num, dtype=np.float32))
    Xtr = hstack([Xtr_text, Xtr_num, Xtr_cat]).tocsr()

    clf = LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        solver="saga",
    )
    clf.fit(Xtr, y_train_file)

    # Build val features in val_df order
    sample_by_id = {sid: s for sid, _, s in rows}
    ids = val_df[id_col].astype(str).tolist()
    true_labels = val_df[true_col].astype(str).tolist()
    y_true = np.array([ACTION_TO_ID[y] for y in true_labels])

    X_text_val = [make_text(sample_by_id[sid]) for sid in ids]
    Xv_text = vec.transform(X_text_val)

    val_num = []
    val_cat = []
    for sid in ids:
        nums, cats = make_num_cat(sample_by_id[sid])
        val_num.append(nums)
        val_cat.append(cats)

    Xv_num = csr_matrix(np.asarray(val_num, dtype=np.float32))
    Xv_cat = enc.transform(val_cat)
    Xv = hstack([Xv_text, Xv_num, Xv_cat]).tocsr()

    file_probs = clf.predict_proba(Xv)
    file_pred = file_probs.argmax(1)
    file_conf = file_probs.max(1)
    file_label_pred = np.array([ID_TO_FILE[i] for i in file_pred])

    base_pred = probs.argmax(1)
    base_labels = np.array([ID_TO_ACTION[i] for i in base_pred])
    base_conf = probs.max(1)
    top2 = np.argsort(-probs, axis=1)[:, :2]
    top2_labels = np.array([[ID_TO_ACTION[j] for j in row] for row in top2])
    margin = probs[np.arange(len(probs)), top2[:, 0]] - probs[np.arange(len(probs)), top2[:, 1]]

    base_f1 = macro_f1(y_true, base_pred)
    base_acc = accuracy_score(y_true, base_pred)
    print("[BASE]", base_f1, base_acc)

    rows_out = []

    for mode in ["top2_only", "top4_allowed", "any_file_pred"]:
        for conf_th in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
            for margin_th in [0.02, 0.05, 0.10, 0.15, 0.20, 1.01]:
                for lr_conf_th in [0.35, 0.45, 0.55, 0.65, 0.75]:
                    new_pred = base_pred.copy()
                    changed = 0
                    good = 0
                    bad = 0

                    for i in range(len(ids)):
                        top1 = top2_labels[i, 0]
                        top2l = top2_labels[i, 1]
                        lr_label = file_label_pred[i]

                        if base_conf[i] > conf_th:
                            continue
                        if margin[i] > margin_th:
                            continue
                        if file_conf[i] < lr_conf_th:
                            continue

                        if top1 not in FILE_SET:
                            continue

                        if mode == "top2_only":
                            if top2l not in FILE_SET:
                                continue
                            if lr_label not in {top1, top2l}:
                                continue
                        elif mode == "top4_allowed":
                            top4 = {ID_TO_ACTION[j] for j in np.argsort(-probs[i])[:4]}
                            if not (top4 & FILE_SET):
                                continue
                            if lr_label not in top4:
                                continue
                        elif mode == "any_file_pred":
                            # only require base top1 is file group; allow LR to choose any file action
                            pass

                        old = new_pred[i]
                        new = ACTION_TO_ID[lr_label]
                        if new == old:
                            continue

                        changed += 1
                        old_ok = old == y_true[i]
                        new_ok = new == y_true[i]
                        if (not old_ok) and new_ok:
                            good += 1
                        elif old_ok and (not new_ok):
                            bad += 1
                        new_pred[i] = new

                    f1 = macro_f1(y_true, new_pred)
                    acc = accuracy_score(y_true, new_pred)
                    rows_out.append({
                        "mode": mode,
                        "conf_th": conf_th,
                        "margin_th": margin_th,
                        "lr_conf_th": lr_conf_th,
                        "macro_f1": f1,
                        "accuracy": acc,
                        "delta_f1": f1 - base_f1,
                        "delta_acc": acc - base_acc,
                        "changed": changed,
                        "good": good,
                        "bad": bad,
                        "net": good - bad,
                    })

    out = pd.DataFrame(rows_out).sort_values("macro_f1", ascending=False)
    out.to_csv(OUT_DIR / "file_group_lr_reranker_sweep.csv", index=False)
    print("\n[SWEEP TOP30]")
    print(out.head(30).to_string(index=False))

    best = out.iloc[0]
    print("\n[BEST]")
    print(best.to_string())

    # save changed examples for best
    mode = best["mode"]
    conf_th = float(best["conf_th"])
    margin_th = float(best["margin_th"])
    lr_conf_th = float(best["lr_conf_th"])

    details = []
    for i, sid in enumerate(ids):
        top1 = top2_labels[i, 0]
        top2l = top2_labels[i, 1]
        lr_label = file_label_pred[i]

        if base_conf[i] > conf_th or margin[i] > margin_th or file_conf[i] < lr_conf_th:
            continue
        if top1 not in FILE_SET:
            continue

        allowed = False
        if mode == "top2_only":
            allowed = top2l in FILE_SET and lr_label in {top1, top2l}
        elif mode == "top4_allowed":
            top4 = {ID_TO_ACTION[j] for j in np.argsort(-probs[i])[:4]}
            allowed = lr_label in top4
        else:
            allowed = True

        if not allowed:
            continue

        old = base_labels[i]
        if lr_label == old:
            continue

        s = sample_by_id[sid]
        details.append({
            "id": sid,
            "true": true_labels[i],
            "old": old,
            "new": lr_label,
            "top2": top2l,
            "base_conf": base_conf[i],
            "margin": margin[i],
            "lr_conf": file_conf[i],
            "old_correct": old == true_labels[i],
            "new_correct": lr_label == true_labels[i],
            "prompt": clean(s.get("current_prompt", ""))[:500],
        })

    det = pd.DataFrame(details)
    det.to_csv(OUT_DIR / "best_lr_reranker_changes.csv", index=False)
    print("[SAVED]", OUT_DIR)


if __name__ == "__main__":
    main()
