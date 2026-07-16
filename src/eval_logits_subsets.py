from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
STACK_DIR = ROOT / "stacking_data"

ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
N_CLASS = len(ACTIONS)

MODEL_DIRS = [
    ROOT / "submit/model/deberta_base_eda_state4",
    ROOT / "submit/model/mdeberta_base_eda_state4",
    ROOT / "submit/model/mdeberta_base_4ep_full",
    ROOT / "submit/model/xlmr_base_router4",
    ROOT / "submit/model/deberta_group_full_router_base",
    ROOT / "submit/model/deberta_group_v3_full_router_mdeberta",
    ROOT / "submit/model/deberta_small_eda_ce_full_lr2e5",
    ROOT / "submit/model/deberta_small_eda_kd_soup_a01_a02_a03",

    # 이미 만들어진 ensemble logits도 2차 후보로 포함
    ROOT / "submit/model/eda_mdeberta_deberta_w054_046",
    ROOT / "submit/model/ensemble_full3_router3_w050",
    ROOT / "submit/model/ensemble_full4_router3_w045_055",
    ROOT / "submit/model/ensemble_full5_router3_w075_025",
]

def macro(y, pred):
    return f1_score(y, pred, labels=list(range(N_CLASS)), average="macro", zero_division=0)

def score_probs(y, probs):
    pred = probs.argmax(1)
    return macro(y, pred), accuracy_score(y, pred)

def find_cols(df):
    for c in ["true_action", "true", "label", "true_label", "target"]:
        if c in df.columns:
            true_col = c
            break
    else:
        raise ValueError(df.columns.tolist())
    return "id", true_col

def load_probs():
    feat = pd.read_csv(STACK_DIR / "val_meta_features.csv")
    y = np.load(STACK_DIR / "y_val.npy")
    ids = feat["id"].astype(str).values

    names, probs = [], []

    for d in MODEL_DIRS:
        csv = d / "val_predictions.csv"
        npy = d / "val_logits.npy"
        if not csv.exists() or not npy.exists():
            print("[SKIP missing]", d.name)
            continue

        df = pd.read_csv(csv)
        id_col, _ = find_cols(df)
        logits = np.load(npy)

        tmp = pd.DataFrame({"id": df[id_col].astype(str), "_idx": np.arange(len(df))})
        order = pd.DataFrame({"id": ids}).merge(tmp, on="id", how="left")["_idx"]

        if order.isna().any():
            print("[SKIP id mismatch]", d.name)
            continue

        order = order.values.astype(int)
        p = softmax(logits[order], axis=1)

        names.append(d.name)
        probs.append(p)

    return y, names, probs

def tune_bias(probs, y, rounds=3):
    grid = np.arange(-0.15, 0.151, 0.01)
    bias = np.zeros(N_CLASS, dtype=np.float32)
    best = macro(y, (probs + bias).argmax(1))

    for _ in range(rounds):
        improved = False
        for c in range(N_CLASS):
            best_c = bias[c]
            best_s = best
            for b in grid:
                trial = bias.copy()
                trial[c] = b
                s = macro(y, (probs + trial).argmax(1))
                if s > best_s:
                    best_s = s
                    best_c = b
            if best_c != bias[c]:
                bias[c] = best_c
                best = best_s
                improved = True
        if not improved:
            break

    return bias, best

def avg_subset(probs, idxs):
    return np.mean([probs[i] for i in idxs], axis=0)

def main():
    y, names, probs = load_probs()

    print("===== individual models =====")
    rows = []
    for i, name in enumerate(names):
        f1, acc = score_probs(y, probs[i])
        b, bf1 = tune_bias(probs[i].copy(), y)
        rows.append({
            "kind": "single",
            "models": name,
            "macro_f1": f1,
            "acc": acc,
            "bias_macro_f1": bf1,
        })

    ind = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(ind.to_string(index=False))

    print("\n===== greedy add by equal average + bias =====")
    selected = []
    remaining = list(range(len(names)))
    best_f1 = -1
    greedy_rows = []

    for step in range(min(8, len(names))):
        best_candidate = None
        best_candidate_score = -1
        best_candidate_raw = None

        for i in remaining:
            idxs = selected + [i]
            p = avg_subset(probs, idxs)
            raw_f1, raw_acc = score_probs(y, p)
            _, bias_f1 = tune_bias(p.copy(), y, rounds=2)

            if bias_f1 > best_candidate_score:
                best_candidate_score = bias_f1
                best_candidate = i
                best_candidate_raw = raw_f1

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_f1 = best_candidate_score

        greedy_rows.append({
            "step": step + 1,
            "added": names[best_candidate],
            "selected": " + ".join(names[i] for i in selected),
            "raw_macro_f1": best_candidate_raw,
            "bias_macro_f1": best_candidate_score,
        })

        print(greedy_rows[-1])

    greedy = pd.DataFrame(greedy_rows)
    print("\n===== greedy summary =====")
    print(greedy.to_string(index=False))

    print("\n===== fast exhaustive subsets up to size 4, top 8 models only =====")
    # exhaustive + bias is slow. First restrict to top 8 individual models
    # and only tune bias for the top raw subsets.
    indiv_scores = []
    for i in range(len(names)):
        f1, _ = score_probs(y, probs[i])
        indiv_scores.append((f1, i))
    top_idxs = [i for _, i in sorted(indiv_scores, reverse=True)[:8]]

    raw_rows = []
    for k in range(1, min(4, len(top_idxs)) + 1):
        for idxs_local in itertools.combinations(top_idxs, k):
            pavg = avg_subset(probs, idxs_local)
            raw_f1, raw_acc = score_probs(y, pavg)
            raw_rows.append({
                "k": k,
                "idxs": idxs_local,
                "models": " + ".join(names[i] for i in idxs_local),
                "raw_macro_f1": raw_f1,
                "acc": raw_acc,
            })

    raw_df = pd.DataFrame(raw_rows).sort_values("raw_macro_f1", ascending=False)

    subset_rows = []
    for _, r in raw_df.head(40).iterrows():
        idxs = tuple(r["idxs"])
        pavg = avg_subset(probs, idxs)
        _, bias_f1 = tune_bias(pavg.copy(), y, rounds=2)
        subset_rows.append({
            "k": int(r["k"]),
            "models": r["models"],
            "raw_macro_f1": float(r["raw_macro_f1"]),
            "bias_macro_f1": float(bias_f1),
            "acc": float(r["acc"]),
        })

    sub = pd.DataFrame(subset_rows).sort_values("bias_macro_f1", ascending=False)
    print(sub.head(30).to_string(index=False))

    OUT = STACK_DIR / "subset_ensemble_scores.csv"
    sub.to_csv(OUT, index=False)
    ind.to_csv(STACK_DIR / "individual_model_scores.csv", index=False)
    greedy.to_csv(STACK_DIR / "greedy_subset_scores.csv", index=False)
    print("[SAVED]", OUT)

if __name__ == "__main__":
    main()
