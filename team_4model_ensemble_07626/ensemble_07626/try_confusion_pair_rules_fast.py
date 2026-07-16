import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

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
base_f1 = f1_score(y, base_pred, average="macro")

print("base f1:", base_f1, "acc:", (base_pred == y).mean())

prompt = df["current_prompt"].fillna("").astype(str).str.lower()
pred_action = df["bias_pred_action"].fillna("").astype(str)
conf = df["bias_confidence"].values

def col(name, default=""):
    if name in df.columns:
        return df[name].fillna(default).astype(str)
    return pd.Series([default] * len(df))

last_action = col("last_action")
prev_action = col("prev_action")
ci = col("last_ci_status")
lang = col("main_lang")
git_dirty = col("git_dirty")

def contains_any(words):
    words = [w.lower() for w in words]
    return prompt.apply(lambda x: any(w in x for w in words)).values

# 사람이 보기 쉬운 keyword 그룹
KW = {
    # file discovery/read family
    "kw_open_read_show": ["open", "read", "show", "cat", "열어", "보여", "읽어", "확인"],
    "kw_find_search": ["find", "search", "grep", "scan", "찾", "검색", "어디", "where", "reference", "references", "사용", "쓰는", "흩어", "spots", "contains"],
    "kw_list_dir": ["list", "directory", "folder", "ls", "목록", "폴더", "디렉토리"],
    "kw_glob_pattern": ["glob", "pattern", "*.py", "*.js", "*.ts", "*.tsx", "*.json", "*.yaml", "all files", "파일들", "전부"],

    # test/lint/bash
    "kw_test": ["test", "tests", "pytest", "unittest", "테스트", "깨졌", "fail", "failed", "failing", "다시 돌", "log"],
    "kw_lint": ["lint", "typecheck", "type-check", "mypy", "pyright", "eslint", "tsc", "ruff", "flake8", "타입", "린트"],
    "kw_bash": ["run", "bash", "command", "shell", "terminal", "실행", "명령"],

    # ask/plan/web
    "kw_ask": ["ask", "clarify", "confirm", "물어", "질문", "확인해줘"],
    "kw_plan": ["plan", "todo", "strategy", "먼저", "계획", "단계"],
    "kw_web": ["web", "browser", "online", "internet", "google", "구글", "인터넷", "검색해봐"],

    # edit family
    "kw_edit": ["edit", "modify", "fix", "change", "수정", "고쳐", "바꿔"],
    "kw_patch": ["patch", "diff", "apply"],
}

kw_masks = {name: contains_any(words) for name, words in KW.items()}

# top confusion pairs
bad = df[df["true_action"] != df["bias_pred_action"]].copy()
pairs = (
    bad.groupby(["true_action", "bias_pred_action"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\n=== top pairs ===")
print(pairs.head(30).to_string(index=False))

# 후보 rule 생성
candidates = []

# confidence scope
scopes = {
    "all": np.ones(len(df), dtype=bool),
    "low30": conf <= np.quantile(conf, 0.30),
    "low50": conf <= np.quantile(conf, 0.50),
    "low70": conf <= np.quantile(conf, 0.70),
}

# boost 후보
boosts = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]

# top 30 confusion pair에 대해 rule 후보 생성
for _, row in pairs.head(30).iterrows():
    target = row["true_action"]
    source_pred = row["bias_pred_action"]
    count = int(row["count"])

    # 현재 source_pred로 예측한 샘플에만 적용
    pred_mask = (pred_action.values == source_pred)

    # pair-specific error samples
    pair_err = (
        (df["true_action"].values == target) &
        (df["bias_pred_action"].values == source_pred)
    )

    # 너무 작은 pair는 skip
    if count < 20:
        continue

    # 1) pred class + low confidence
    for scope_name, scope_mask in scopes.items():
        mask = pred_mask & scope_mask
        if mask.sum() > 0:
            for b in boosts:
                candidates.append({
                    "kind": "pred_scope",
                    "target": target,
                    "source_pred": source_pred,
                    "condition": scope_name,
                    "boost": b,
                    "mask": mask,
                    "pair_count": count,
                })

    # 2) pred class + keyword
    for kw_name, kw_mask in kw_masks.items():
        for scope_name, scope_mask in scopes.items():
            mask = pred_mask & kw_mask & scope_mask
            if mask.sum() >= 3:
                for b in boosts:
                    candidates.append({
                        "kind": "keyword",
                        "target": target,
                        "source_pred": source_pred,
                        "condition": f"{kw_name}+{scope_name}",
                        "boost": b,
                        "mask": mask,
                        "pair_count": count,
                    })

    # 3) pred class + metadata values that are common in this pair error
    for meta_name, series in [
        ("last_action", last_action),
        ("prev_action", prev_action),
        ("ci", ci),
        ("lang", lang),
        ("git_dirty", git_dirty),
    ]:
        vc = series[pair_err].value_counts().head(4)
        for val, freq in vc.items():
            if freq < 5:
                continue
            meta_mask = (series.values == val)
            for scope_name, scope_mask in scopes.items():
                mask = pred_mask & meta_mask & scope_mask
                if mask.sum() >= 3:
                    for b in boosts:
                        candidates.append({
                            "kind": meta_name,
                            "target": target,
                            "source_pred": source_pred,
                            "condition": f"{meta_name}={val}+{scope_name}",
                            "boost": b,
                            "mask": mask,
                            "pair_count": count,
                        })

print("\nnum candidates:", len(candidates))

# single candidate sweep
single_results = []
for cand in candidates:
    logits = base_logits.copy()
    target_id = label2id[cand["target"]]
    logits[cand["mask"], target_id] += cand["boost"]
    pred = logits.argmax(1)
    f1 = f1_score(y, pred, average="macro")
    acc = (pred == y).mean()
    changed = int((pred != base_pred).sum())
    single_results.append((f1, acc, changed, cand))

single_results.sort(reverse=True, key=lambda x: x[0])

print("\n=== top single rules ===")
for f1, acc, changed, cand in single_results[:50]:
    print(
        f"f1={f1:.6f} acc={acc:.6f} changed={changed:4d} "
        f"target={cand['target']:18s} from_pred={cand['source_pred']:18s} "
        f"kind={cand['kind']:10s} cond={cand['condition']:35s} "
        f"boost={cand['boost']:.2f} mask={int(cand['mask'].sum())} pair_count={cand['pair_count']}"
    )

# greedy combine
print("\n=== greedy combine ===")
current_logits = base_logits.copy()
current_pred = current_logits.argmax(1)
current_f1 = f1_score(y, current_pred, average="macro")
chosen = []


# Use only top single candidates for faster greedy search
top_single_cands = [r[3] for r in single_results[:200]]
remaining = top_single_cands[:]
print("greedy candidates:", len(remaining))

for step in range(20):
    best = (current_f1, None, None, None)

    for cand in remaining:
        logits = current_logits.copy()
        target_id = label2id[cand["target"]]
        logits[cand["mask"], target_id] += cand["boost"]

        pred = logits.argmax(1)
        f1 = f1_score(y, pred, average="macro")
        if f1 > best[0]:
            best = (f1, cand, logits, pred)

    if best[1] is None:
        print("no more improvement")
        break

    best_f1, cand, current_logits, current_pred = best
    current_f1 = best_f1

    record = {
        "step": step,
        "macro_f1": current_f1,
        "accuracy": float((current_pred == y).mean()),
        "changed_vs_base": int((current_pred != base_pred).sum()),
        "target": cand["target"],
        "source_pred": cand["source_pred"],
        "kind": cand["kind"],
        "condition": cand["condition"],
        "boost": cand["boost"],
        "mask_count": int(cand["mask"].sum()),
        "pair_count": cand["pair_count"],
    }
    chosen.append(record)

    print(
        f"step={step:02d} f1={record['macro_f1']:.6f} acc={record['accuracy']:.6f} "
        f"changed={record['changed_vs_base']:4d} "
        f"target={record['target']:18s} from={record['source_pred']:18s} "
        f"{record['kind']} {record['condition']} boost={record['boost']:.2f} mask={record['mask_count']}"
    )

    # 동일 rule 무한반복 방지: 같은 target/source/kind/condition은 제거
    remaining = [
        c for c in remaining
        if not (
            c["target"] == cand["target"] and
            c["source_pred"] == cand["source_pred"] and
            c["kind"] == cand["kind"] and
            c["condition"] == cand["condition"]
        )
    ]

print("\nfinal f1:", current_f1, "acc:", (current_pred == y).mean())
pd.DataFrame(chosen).to_csv("chosen_confusion_pair_rules.csv", index=False)
np.save("val_logits_with_confusion_pair_rules.npy", current_logits.astype(np.float32))

print("saved:")
print(" - chosen_confusion_pair_rules.csv")
print(" - val_logits_with_confusion_pair_rules.npy")
