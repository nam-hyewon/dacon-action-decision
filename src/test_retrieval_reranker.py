import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BEST_DIR = ROOT / "submit/model/eda_mdeberta_deberta_w054_046"
OUT_DIR = ROOT / "eda_outputs/retrieval_reranker"
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


def clean(x):
    return re.sub(r"\s+", " ", str(x)).strip()


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


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_users(sample):
    return [
        str(h.get("content", ""))
        for h in sample.get("history", [])
        if h.get("role") == "user"
    ]


def get_last_result(sample):
    for h in reversed(sample.get("history", [])):
        if h.get("role") == "assistant_action":
            return str(h.get("result_summary", ""))
    return ""


def make_text(sample, mode):
    prompt = clean(sample.get("current_prompt", ""))
    users = get_users(sample)
    actions = get_actions(sample)
    last_result = clean(get_last_result(sample))
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}

    if mode == "prompt":
        return prompt

    if mode == "prompt_prev":
        return f"{prompt} [ACTIONS] {' '.join(actions[-5:])}"

    if mode == "prompt_hist":
        return (
            f"[PROMPT] {prompt} "
            f"[RECENT_USERS] {' || '.join(users[-3:])} "
            f"[ACTIONS] {' '.join(actions[-5:])}"
        )

    if mode == "full_state":
        return (
            f"[PROMPT] {prompt} "
            f"[RECENT_USERS] {' || '.join(users[-3:])} "
            f"[ACTIONS] {' '.join(actions[-8:])} "
            f"[LAST_RESULT] {last_result[:700]} "
            f"[OPEN_FILES] {ws.get('open_files', [])} "
            f"[CI] {ws.get('last_ci_status', '')} "
            f"[LANG] {ws.get('language_mix', {})}"
        )

    raise ValueError(mode)


def macro_f1(y, p):
    return f1_score(y, p, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)


def vote_labels(nei_labels, nei_sims, k, sim_power=2.0):
    scores = Counter()
    for lab, sim in zip(nei_labels[:k], nei_sims[:k]):
        scores[lab] += float(max(sim, 0.0) ** sim_power)
    return scores.most_common(1)[0][0], scores


def main():
    val_df = pd.read_csv(VAL_CSV)
    logits = np.load(VAL_LOGITS)
    probs = softmax(logits, axis=1)

    id_col, true_col = find_cols(val_df)
    val_ids = val_df[id_col].astype(str).tolist()
    val_id_set = set(val_ids)
    true_labels = val_df[true_col].astype(str).tolist()
    y_true = np.array([ACTION_TO_ID[x] for x in true_labels])

    base_pred = probs.argmax(axis=1)
    base_labels = np.array([ID_TO_ACTION[i] for i in base_pred])
    base_conf = probs.max(axis=1)
    top2 = np.argsort(-probs, axis=1)[:, :2]
    top2_labels = np.array([[ID_TO_ACTION[j] for j in row] for row in top2])
    margin = probs[np.arange(len(probs)), top2[:, 0]] - probs[np.arange(len(probs)), top2[:, 1]]

    base_f1 = macro_f1(y_true, base_pred)
    base_acc = accuracy_score(y_true, base_pred)
    print("[BASE]", base_f1, base_acc)

    label_df = pd.read_csv(DATA_DIR / "train_labels.csv")
    label_by_id = dict(zip(label_df["id"].astype(str), label_df["action"].astype(str)))

    samples = []
    sample_by_id = {}
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                sid = str(s["id"])
                sample_by_id[sid] = s
                samples.append(s)

    train_samples = []
    train_labels = []
    val_samples = []

    for s in samples:
        sid = str(s["id"])
        if sid in val_id_set:
            continue
        if sid not in label_by_id:
            continue
        train_samples.append(s)
        train_labels.append(label_by_id[sid])

    for sid in val_ids:
        val_samples.append(sample_by_id[sid])

    print("[retrieval train]", len(train_samples), "val", len(val_samples))

    rows = []

    for mode in ["prompt", "prompt_prev", "prompt_hist", "full_state"]:
        print("\n[MODE]", mode)
        tr_texts = [make_text(s, mode) for s in train_samples]
        va_texts = [make_text(s, mode) for s in val_samples]

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=300000,
            sublinear_tf=True,
            lowercase=True,
            analyzer="word",
        )
        Xtr = vec.fit_transform(tr_texts)
        Xva = vec.transform(va_texts)

        nn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(Xtr)
        dist, ind = nn.kneighbors(Xva, return_distance=True)
        sims = 1.0 - dist

        # pure retrieval scores
        for k in [1, 3, 5, 7]:
            retr_labels = []
            for i in range(len(val_ids)):
                labs = [train_labels[j] for j in ind[i]]
                pred_lab, _ = vote_labels(labs, sims[i], k=k)
                retr_labels.append(pred_lab)
            retr_pred = np.array([ACTION_TO_ID[x] for x in retr_labels])
            rows.append({
                "mode": mode,
                "strategy": f"pure_k{k}",
                "sim_th": -1,
                "conf_th": -1,
                "margin_th": -1,
                "top_constraint": "none",
                "macro_f1": macro_f1(y_true, retr_pred),
                "accuracy": accuracy_score(y_true, retr_pred),
                "delta_f1": macro_f1(y_true, retr_pred) - base_f1,
                "delta_acc": accuracy_score(y_true, retr_pred) - base_acc,
                "changed": int(np.sum(retr_pred != base_pred)),
                "good": -1,
                "bad": -1,
                "net": -1,
            })

        for k in [1, 3, 5]:
            retr_labels = []
            retr_conf = []
            for i in range(len(val_ids)):
                labs = [train_labels[j] for j in ind[i]]
                pred_lab, scores = vote_labels(labs, sims[i], k=k)
                total = sum(scores.values())
                conf = scores[pred_lab] / total if total > 0 else 0.0
                retr_labels.append(pred_lab)
                retr_conf.append(conf)

            retr_labels = np.array(retr_labels)
            retr_pred = np.array([ACTION_TO_ID[x] for x in retr_labels])
            retr_conf = np.array(retr_conf)
            top_sim = sims[:, 0]

            for sim_th in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
                for conf_th in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
                    for margin_th in [0.02, 0.05, 0.10, 0.15, 0.20, 1.01]:
                        for top_constraint in ["none", "top2", "top4"]:
                            new_pred = base_pred.copy()
                            changed = good = bad = 0

                            for i in range(len(val_ids)):
                                if top_sim[i] < sim_th:
                                    continue
                                if base_conf[i] > conf_th:
                                    continue
                                if margin[i] > margin_th:
                                    continue

                                lab = retr_labels[i]
                                if top_constraint == "top2":
                                    if lab not in set(top2_labels[i]):
                                        continue
                                elif top_constraint == "top4":
                                    top4 = {ID_TO_ACTION[j] for j in np.argsort(-probs[i])[:4]}
                                    if lab not in top4:
                                        continue

                                new = ACTION_TO_ID[lab]
                                old = new_pred[i]
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
                            rows.append({
                                "mode": mode,
                                "strategy": f"override_k{k}",
                                "sim_th": sim_th,
                                "conf_th": conf_th,
                                "margin_th": margin_th,
                                "top_constraint": top_constraint,
                                "macro_f1": f1,
                                "accuracy": acc,
                                "delta_f1": f1 - base_f1,
                                "delta_acc": acc - base_acc,
                                "changed": changed,
                                "good": good,
                                "bad": bad,
                                "net": good - bad,
                            })

    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out.to_csv(OUT_DIR / "retrieval_reranker_sweep.csv", index=False)

    print("\n[TOP 40]")
    print(out.head(40).to_string(index=False))

    best = out.iloc[0]
    print("\n[BEST]")
    print(best.to_string())

    print("[SAVED]", OUT_DIR)


if __name__ == "__main__":
    main()
