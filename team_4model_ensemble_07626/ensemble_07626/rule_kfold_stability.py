import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
label2id = {a: i for i, a in enumerate(ALL_CLASSES)}

df = pd.read_csv("ensemble_greedy4_weightgrid_row_zscore/class_bias/val_predictions_with_bias.csv")
base_logits = np.load("ensemble_greedy4_weightgrid_row_zscore/class_bias/val_logits_with_bias.npy")

y = df["true_id"].values.astype(int)
base_pred = base_logits.argmax(1)

prompt = df["current_prompt"].fillna("").astype(str).str.lower()
pred_action_base = df["bias_pred_action"].fillna("").astype(str).values
conf = df["bias_confidence"].values

def col(name, default=""):
    if name in df.columns:
        return df[name].fillna(default).astype(str)
    return pd.Series([default] * len(df))

last_action = col("last_action").values
prev_action = col("prev_action").values
ci = col("last_ci_status").values
lang = col("main_lang").values
git_dirty = col("git_dirty").values

def contains_any(words):
    words = [w.lower() for w in words]
    return prompt.apply(lambda x: any(w in x for w in words)).values

KW = {
    "kw_open_read_show": ["open", "read", "show", "cat", "열어", "보여", "읽어", "확인"],
    "kw_find_search": ["find", "search", "grep", "scan", "찾", "검색", "어디", "where", "reference", "references", "사용", "쓰는", "흩어", "spots", "contains"],
    "kw_list_dir": ["list", "directory", "folder", "ls", "목록", "폴더", "디렉토리"],
    "kw_glob_pattern": ["glob", "pattern", "*.py", "*.js", "*.ts", "*.tsx", "*.json", "*.yaml", "all files", "파일들", "전부"],

    "kw_test": ["test", "tests", "pytest", "unittest", "테스트", "깨졌", "fail", "failed", "failing", "다시 돌", "log"],
    "kw_lint": ["lint", "typecheck", "type-check", "mypy", "pyright", "eslint", "tsc", "ruff", "flake8", "타입", "린트"],
    "kw_bash": ["run", "bash", "command", "shell", "terminal", "실행", "명령"],

    "kw_ask": ["ask", "clarify", "confirm", "물어", "질문", "확인해줘"],
    "kw_plan": ["plan", "todo", "strategy", "먼저", "계획", "단계"],
    "kw_web": ["web", "browser", "online", "internet", "google", "구글", "인터넷", "검색해봐"],

    "kw_edit": ["edit", "modify", "fix", "change", "수정", "고쳐", "바꿔"],
    "kw_patch": ["patch", "diff", "apply"],
}
kw_masks = {name: contains_any(words) for name, words in KW.items()}

def scope_mask(scope, idx=None):
    if idx is None:
        idx = np.arange(len(df))

    if scope == "all":
        m = np.ones(len(df), dtype=bool)
    elif scope == "low30":
        thr = np.quantile(conf[idx], 0.30)
        m = conf <= thr
    elif scope == "low50":
        thr = np.quantile(conf[idx], 0.50)
        m = conf <= thr
    elif scope == "low70":
        thr = np.quantile(conf[idx], 0.70)
        m = conf <= thr
    else:
        raise ValueError(scope)
    return m

def make_mask(kind, source_pred, condition, train_idx_for_quantile=None):
    mask = pred_action_base == source_pred

    if "+" in condition:
        main_cond, scope = condition.rsplit("+", 1)
    else:
        main_cond, scope = condition, condition

    if kind == "pred_scope":
        mask = mask & scope_mask(condition, train_idx_for_quantile)

    elif kind == "keyword":
        mask = mask & kw_masks[main_cond] & scope_mask(scope, train_idx_for_quantile)

    elif kind == "prev_action":
        val = main_cond.split("=", 1)[1]
        mask = mask & (prev_action == val) & scope_mask(scope, train_idx_for_quantile)

    elif kind == "last_action":
        val = main_cond.split("=", 1)[1]
        mask = mask & (last_action == val) & scope_mask(scope, train_idx_for_quantile)

    elif kind == "ci":
        val = main_cond.split("=", 1)[1]
        mask = mask & (ci == val) & scope_mask(scope, train_idx_for_quantile)

    elif kind == "lang":
        val = main_cond.split("=", 1)[1]
        mask = mask & (lang == val) & scope_mask(scope, train_idx_for_quantile)

    elif kind == "git_dirty":
        val = main_cond.split("=", 1)[1]
        mask = mask & (git_dirty == val) & scope_mask(scope, train_idx_for_quantile)

    else:
        raise ValueError((kind, condition))

    return mask

def generate_candidates(train_idx):
    train_idx = np.asarray(train_idx)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True

    pred = base_logits.argmax(1)
    true_action = np.array([ALL_CLASSES[i] for i in y])
    pred_action = np.array([ALL_CLASSES[i] for i in pred])

    bad_train = train_mask & (true_action != pred_action)

    pairs = (
        pd.DataFrame({
            "true_action": true_action[bad_train],
            "pred_action": pred_action[bad_train],
        })
        .groupby(["true_action", "pred_action"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    candidates = []
    boosts = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]
    scopes = ["all", "low30", "low50", "low70"]

    for _, row in pairs.head(30).iterrows():
        target = row["true_action"]
        source_pred = row["pred_action"]
        pair_count = int(row["count"])
        if pair_count < 12:
            continue

        pair_err = bad_train & (true_action == target) & (pred_action == source_pred)

        # pred_scope
        for sc in scopes:
            mask = make_mask("pred_scope", source_pred, sc, train_idx)
            if (mask & train_mask).sum() >= 3:
                for b in boosts:
                    candidates.append({
                        "target": target,
                        "source_pred": source_pred,
                        "kind": "pred_scope",
                        "condition": sc,
                        "boost": b,
                        "mask": mask,
                        "pair_count": pair_count,
                    })

        # keyword
        for kw_name in KW:
            for sc in scopes:
                cond = f"{kw_name}+{sc}"
                mask = make_mask("keyword", source_pred, cond, train_idx)
                if (mask & train_mask).sum() >= 3:
                    for b in boosts:
                        candidates.append({
                            "target": target,
                            "source_pred": source_pred,
                            "kind": "keyword",
                            "condition": cond,
                            "boost": b,
                            "mask": mask,
                            "pair_count": pair_count,
                        })

        # metadata candidates from common values in pair error
        meta_sources = [
            ("last_action", last_action),
            ("prev_action", prev_action),
            ("ci", ci),
            ("lang", lang),
            ("git_dirty", git_dirty),
        ]
        for meta_name, arr in meta_sources:
            vals, counts = np.unique(arr[pair_err], return_counts=True)
            order = np.argsort(-counts)
            for j in order[:4]:
                val = vals[j]
                freq = counts[j]
                if freq < 4:
                    continue
                for sc in scopes:
                    cond = f"{meta_name}={val}+{sc}"
                    mask = make_mask(meta_name, source_pred, cond, train_idx)
                    if (mask & train_mask).sum() >= 3:
                        for b in boosts:
                            candidates.append({
                                "target": target,
                                "source_pred": source_pred,
                                "kind": meta_name,
                                "condition": cond,
                                "boost": b,
                                "mask": mask,
                                "pair_count": pair_count,
                            })

    return candidates, pairs

def eval_logits(logits, idx):
    pred = logits[idx].argmax(1)
    return f1_score(y[idx], pred, average="macro"), (pred == y[idx]).mean()

def greedy_select(train_idx, max_steps=16, top_single=200):
    train_idx = np.asarray(train_idx)
    candidates, pairs = generate_candidates(train_idx)

    current_logits = base_logits.copy()
    base_train_f1, _ = eval_logits(current_logits, train_idx)
    current_f1 = base_train_f1

    # single score on train
    single = []
    for cand in candidates:
        logits = base_logits.copy()
        logits[cand["mask"], label2id[cand["target"]]] += cand["boost"]
        f1, acc = eval_logits(logits, train_idx)
        single.append((f1, acc, cand))

    single.sort(reverse=True, key=lambda x: x[0])
    remaining = [x[2] for x in single[:top_single]]

    chosen = []
    for step in range(max_steps):
        best = (current_f1, None, None)

        for cand in remaining:
            logits = current_logits.copy()
            logits[cand["mask"], label2id[cand["target"]]] += cand["boost"]
            f1, _ = eval_logits(logits, train_idx)

            if f1 > best[0]:
                best = (f1, cand, logits)

        if best[1] is None:
            break

        best_f1, cand, current_logits = best
        current_f1 = best_f1

        chosen.append({
            "step": step,
            "target": cand["target"],
            "source_pred": cand["source_pred"],
            "kind": cand["kind"],
            "condition": cand["condition"],
            "boost": cand["boost"],
            "mask_count_all": int(cand["mask"].sum()),
            "mask_count_train": int(cand["mask"][train_idx].sum()),
            "train_macro_f1": current_f1,
        })

        remaining = [
            c for c in remaining
            if not (
                c["target"] == cand["target"] and
                c["source_pred"] == cand["source_pred"] and
                c["kind"] == cand["kind"] and
                c["condition"] == cand["condition"]
            )
        ]

    return chosen

def apply_rules(rule_rows, idx_for_quantile=None):
    logits = base_logits.copy()
    for row in rule_rows:
        mask = make_mask(
            row["kind"],
            row["source_pred"],
            row["condition"],
            idx_for_quantile,
        )
        logits[mask, label2id[row["target"]]] += float(row["boost"])
    return logits

def main():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_fold_rows = []
    all_rules = []

    print("overall base:", f1_score(y, base_pred, average="macro"), (base_pred == y).mean())

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print("\n" + "=" * 100)
        print("FOLD", fold)

        base_tr_f1, base_tr_acc = eval_logits(base_logits, tr_idx)
        base_va_f1, base_va_acc = eval_logits(base_logits, va_idx)
        print(f"base train f1={base_tr_f1:.6f} acc={base_tr_acc:.6f}")
        print(f"base valid f1={base_va_f1:.6f} acc={base_va_acc:.6f}")

        chosen = greedy_select(tr_idx, max_steps=16, top_single=200)
        fold_rules = pd.DataFrame(chosen)
        fold_rules["fold"] = fold
        all_rules.append(fold_rules)

        print("chosen rules:")
        if len(fold_rules):
            print(fold_rules[["step","target","source_pred","kind","condition","boost","train_macro_f1"]].to_string(index=False))
        else:
            print("(none)")

        # prefix evaluation on train/valid
        for prefix in [0, 3, 6, 9, 12, 16]:
            selected = chosen[:prefix]
            logits = apply_rules(selected, idx_for_quantile=tr_idx)

            tr_f1, tr_acc = eval_logits(logits, tr_idx)
            va_f1, va_acc = eval_logits(logits, va_idx)

            row = {
                "fold": fold,
                "prefix": prefix,
                "train_f1": tr_f1,
                "train_acc": tr_acc,
                "valid_f1": va_f1,
                "valid_acc": va_acc,
                "train_gain": tr_f1 - base_tr_f1,
                "valid_gain": va_f1 - base_va_f1,
                "num_rules": len(selected),
            }
            all_fold_rows.append(row)
            print(
                f"prefix={prefix:2d} "
                f"train_f1={tr_f1:.6f} gain={tr_f1-base_tr_f1:+.6f} "
                f"valid_f1={va_f1:.6f} gain={va_f1-base_va_f1:+.6f}"
            )

    res = pd.DataFrame(all_fold_rows)
    res.to_csv("rule_kfold_prefix_scores.csv", index=False)

    if all_rules:
        rules_df = pd.concat(all_rules, ignore_index=True)
        rules_df.to_csv("rule_kfold_chosen_rules.csv", index=False)
    else:
        rules_df = pd.DataFrame()

    print("\n" + "=" * 100)
    print("SUMMARY BY PREFIX")
    summary = (
        res.groupby("prefix")
        .agg(
            train_f1_mean=("train_f1", "mean"),
            train_gain_mean=("train_gain", "mean"),
            valid_f1_mean=("valid_f1", "mean"),
            valid_gain_mean=("valid_gain", "mean"),
            valid_gain_std=("valid_gain", "std"),
            valid_acc_mean=("valid_acc", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    summary.to_csv("rule_kfold_summary.csv", index=False)

    print("\nSaved:")
    print(" - rule_kfold_prefix_scores.csv")
    print(" - rule_kfold_summary.csv")
    print(" - rule_kfold_chosen_rules.csv")

if __name__ == "__main__":
    main()
