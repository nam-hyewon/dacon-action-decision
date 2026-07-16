import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

BEST_DIR = ROOT / "submit/model/eda_mdeberta_deberta_w054_046"
VAL_CSV = BEST_DIR / "val_predictions.csv"
VAL_LOGITS = BEST_DIR / "val_logits.npy"

OUT_DIR = ROOT / "eda_outputs/file_group_reranker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}

FILE_GROUP = {"read_file", "grep_search", "list_directory", "glob_pattern"}


def find_cols(df):
    for c in ["true_action", "true", "label", "true_label", "target"]:
        if c in df.columns:
            return "id", c
    raise ValueError(df.columns.tolist())


def macro_f1(y, pred):
    return f1_score(y, pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)


def contains(pat, text):
    return bool(re.search(pat, str(text).lower()))


def extract_paths(text):
    text = str(text)
    paths = re.findall(
        r"[\w./-]+\.(?:py|js|ts|tsx|jsx|json|yaml|yml|toml|md|txt|css|html|go|rs|java|cpp|c|h|sh|sql|env)",
        text,
    )
    globs = re.findall(r"(?:\*[\w./-]*\.\w+|[\w./-]*\*\w*|_\w+\.\w+)", text)
    return paths, globs


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_last_result(sample):
    for h in reversed(sample.get("history", [])):
        if h.get("role") == "assistant_action":
            return str(h.get("result_summary", ""))
    return ""


def rule_file_intent(sample):
    prompt = str(sample.get("current_prompt", ""))
    p = prompt.lower()
    last_result = get_last_result(sample).lower()
    actions = get_actions(sample)
    last_action = actions[-1] if actions else "none"

    paths, globs = extract_paths(prompt)

    has_explicit_file = len(paths) > 0
    has_glob = (
        len(globs) > 0
        or contains(r"\*\.", p)
        or contains(r"\b(every|all)\s+[\w./-]+\.\w+\b", p)
        or contains(r"\b(all|every)\s+\w+\s+files?\b", p)
        or contains(r"\b전부\b.*\.\w+", p)
        or contains(r"\b모든\b.*파일", p)
    )

    has_dir = contains(
        r"\b(directory|folder|dir|what'?s in|list files?|ls|tree)\b|디렉토리|폴더|뭐뭐|뭐 있|목록|펼쳐",
        p,
    )

    has_search = contains(
        r"\b(where|search|grep|find|occurrence|occurrences|reference|references|used|uses|call|calls|todo|scattered|left)\b|어디|찾아|쓰이는|호출|참조|비슷한|남아",
        p,
    )

    has_open = contains(
        r"\b(open|read|show|view|display|inspect|look at|cat|see)\b|열어|보자|봐줘|펼쳐줘",
        p,
    )

    symbol_like = bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\s*(?:\(|\.|::)?", prompt))

    # Very high precision rules first
    if has_glob:
        return "glob_pattern", "rule_glob_pattern"

    if has_dir:
        # If it asks what's inside a directory/folder, prefer list_directory.
        return "list_directory", "rule_list_directory"

    if has_search and (symbol_like or not has_explicit_file):
        return "grep_search", "rule_grep_search"

    if has_open and has_explicit_file:
        return "read_file", "rule_read_file"

    # Last action/result based weak hints
    if last_action == "glob_pattern" and contains(r"\bfiles? matched\b|matched '\*\.", last_result):
        # after glob result, many followups ask to inspect one/list.
        # do not force a label unless prompt has a clear cue.
        return None, "no_clear_after_glob"

    return None, "no_rule"


def main():
    val = pd.read_csv(VAL_CSV)
    logits = np.load(VAL_LOGITS)
    probs = softmax(logits, axis=1)

    id_col, true_col = find_cols(val)

    ids = val[id_col].astype(str).tolist()
    y_true_labels = val[true_col].astype(str).tolist()
    y_true = np.array([ACTION_TO_ID[x] for x in y_true_labels])

    pred = probs.argmax(1)
    conf = probs.max(1)
    top2 = np.argsort(-probs, axis=1)[:, :2]

    samples = {}
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                samples[str(s["id"])] = s

    base_f1 = macro_f1(y_true, pred)
    base_acc = accuracy_score(y_true, pred)
    print("[BASE]", base_f1, base_acc)

    rows = []

    conf_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    margin_thresholds = [0.02, 0.05, 0.10, 0.15, 0.20]

    for conf_th in conf_thresholds:
        for margin_th in margin_thresholds:
            new_pred = pred.copy()
            changed = 0
            good_change = 0
            bad_change = 0
            rule_counts = {}

            for i, sid in enumerate(ids):
                top1_label = ID_TO_ACTION[top2[i, 0]]
                top2_label = ID_TO_ACTION[top2[i, 1]]
                margin = probs[i, top2[i, 0]] - probs[i, top2[i, 1]]

                # only low confidence, low margin, file-group ambiguity
                if conf[i] > conf_th:
                    continue
                if margin > margin_th:
                    continue
                if top1_label not in FILE_GROUP or top2_label not in FILE_GROUP:
                    continue

                sample = samples[sid]
                rule_label, reason = rule_file_intent(sample)
                rule_counts[reason] = rule_counts.get(reason, 0) + 1

                if rule_label is None:
                    continue

                # conservative: only choose rule label if it is in model top2
                if rule_label not in {top1_label, top2_label}:
                    continue

                old = new_pred[i]
                new = ACTION_TO_ID[rule_label]
                if new == old:
                    continue

                changed += 1
                old_correct = old == y_true[i]
                new_correct = new == y_true[i]
                if (not old_correct) and new_correct:
                    good_change += 1
                elif old_correct and (not new_correct):
                    bad_change += 1

                new_pred[i] = new

            f1 = macro_f1(y_true, new_pred)
            acc = accuracy_score(y_true, new_pred)

            rows.append({
                "conf_th": conf_th,
                "margin_th": margin_th,
                "macro_f1": f1,
                "accuracy": acc,
                "delta_f1": f1 - base_f1,
                "delta_acc": acc - base_acc,
                "changed": changed,
                "good_change": good_change,
                "bad_change": bad_change,
                "net_good": good_change - bad_change,
                "rule_counts": str(rule_counts),
            })

    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(out.head(30).to_string(index=False))
    out.to_csv(OUT_DIR / "file_group_rule_reranker_sweep.csv", index=False)

    best = out.iloc[0].to_dict()
    print("\n[BEST]", best)

    # Save detailed changes for best setting
    conf_th = float(best["conf_th"])
    margin_th = float(best["margin_th"])
    details = []
    new_pred = pred.copy()

    for i, sid in enumerate(ids):
        top1_label = ID_TO_ACTION[top2[i, 0]]
        top2_label = ID_TO_ACTION[top2[i, 1]]
        margin = probs[i, top2[i, 0]] - probs[i, top2[i, 1]]

        if conf[i] > conf_th or margin > margin_th:
            continue
        if top1_label not in FILE_GROUP or top2_label not in FILE_GROUP:
            continue

        sample = samples[sid]
        rule_label, reason = rule_file_intent(sample)
        if rule_label is None:
            continue
        if rule_label not in {top1_label, top2_label}:
            continue

        old_label = ID_TO_ACTION[pred[i]]
        if rule_label == old_label:
            continue

        new_pred[i] = ACTION_TO_ID[rule_label]

        prompt = str(sample.get("current_prompt", ""))
        details.append({
            "id": sid,
            "true_label": y_true_labels[i],
            "old_pred": old_label,
            "new_pred": rule_label,
            "top2_label": top2_label,
            "confidence": conf[i],
            "margin": margin,
            "reason": reason,
            "old_correct": old_label == y_true_labels[i],
            "new_correct": rule_label == y_true_labels[i],
            "current_prompt": re.sub(r"\s+", " ", prompt).strip()[:500],
        })

    detail_df = pd.DataFrame(details)
    detail_df.to_csv(OUT_DIR / "best_file_group_rule_changes.csv", index=False)
    print("[SAVED]", OUT_DIR)


if __name__ == "__main__":
    main()
