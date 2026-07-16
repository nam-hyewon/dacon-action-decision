import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.sparse import hstack
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception as e:
    HAS_LGB = False
    print("[WARN] lightgbm unavailable:", repr(e))

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

    # strong pre-ensembled logits
    ROOT / "submit/model/eda_mdeberta_deberta_w054_046",
    ROOT / "submit/model/ensemble_full3_router3_w050",
    ROOT / "submit/model/ensemble_full4_router3_w045_055",
    ROOT / "submit/model/ensemble_full5_router3_w075_025",
]

CAT_COLS = [
    "step_bucket", "prompt_len_bucket",
    "prev1", "prev2", "prev3", "action_seq_last3",
    "git_dirty", "last_ci_status",
]

BOOL_COLS = [
    "after_edit_or_patch", "after_search", "after_read", "after_run",
    "has_explicit_file", "has_glob_pattern", "has_dir_phrase",
    "has_search_phrase", "has_open_phrase", "has_symbol_like",
    "has_test_phrase", "has_lint_phrase", "has_web_phrase",
    "has_ask_phrase", "has_plan_phrase",
    "intent_read_file", "intent_grep_search",
    "intent_list_directory", "intent_glob_pattern",
    "last_result_has_error", "last_result_has_pass",
    "last_result_has_match", "last_result_no_match",
    "last_result_has_patch", "last_result_has_test", "last_result_has_lint",
]

NUM_COLS = [
    "step", "prompt_len", "prompt_word_count", "history_len", "num_actions", "num_open_files",
]

def find_cols(df):
    id_col = "id"
    true_col = None
    for c in ["true_action", "true", "label", "true_label", "target"]:
        if c in df.columns:
            true_col = c
            break
    if true_col is None:
        raise ValueError(f"Cannot find true label column: {df.columns.tolist()}")
    return id_col, true_col

def macro_f1(y, pred):
    return f1_score(
        y, pred,
        labels=list(range(N_CLASS)),
        average="macro",
        zero_division=0,
    )

def score_row(name, y, pred):
    return {
        "model": name,
        "macro_f1": float(macro_f1(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
    }

def load_model_probs(ref_ids):
    probs = []
    names = []

    for d in MODEL_DIRS:
        val_csv = d / "val_predictions.csv"
        val_logits = d / "val_logits.npy"

        if not val_csv.exists() or not val_logits.exists():
            print("[SKIP missing]", d)
            continue

        df = pd.read_csv(val_csv)
        id_col, _ = find_cols(df)
        logits = np.load(val_logits)

        tmp = pd.DataFrame({
            "id": df[id_col].astype(str),
            "_idx": np.arange(len(df)),
        })

        order_df = pd.DataFrame({"id": ref_ids}).merge(tmp, on="id", how="left")
        if order_df["_idx"].isna().any():
            print("[SKIP id mismatch]", d)
            continue

        order = order_df["_idx"].values.astype(int)
        logits = logits[order]

        if logits.shape[1] != N_CLASS:
            print("[SKIP bad shape]", d, logits.shape)
            continue

        p = softmax(logits, axis=1)
        probs.append(p)
        names.append(d.name)
        print("[USE]", d.name, p.shape)

    if not probs:
        raise RuntimeError("No usable model logits")

    return names, np.hstack(probs)

def tune_bias(probs, y, rounds=3):
    bias = np.zeros(N_CLASS, dtype=np.float32)
    grid = np.arange(-0.15, 0.151, 0.01)

    best = macro_f1(y, (probs + bias).argmax(1))
    for _ in range(rounds):
        improved = False
        for c in range(N_CLASS):
            best_c = bias[c]
            best_score = best
            for b in grid:
                trial = bias.copy()
                trial[c] = b
                s = macro_f1(y, (probs + trial).argmax(1))
                if s > best_score:
                    best_score = s
                    best_c = b
            if best_c != bias[c]:
                bias[c] = best_c
                best = best_score
                improved = True
        if not improved:
            break

    return bias, best

def build_X(feat, X_probs):
    cat = feat[CAT_COLS].astype(str)
    enc = OneHotEncoder(handle_unknown="ignore")
    X_cat = enc.fit_transform(cat)

    X_bool = feat[BOOL_COLS].astype(int).values.astype(np.float32)

    X_num = feat[NUM_COLS].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    X_dense = np.hstack([X_probs, X_bool, X_num])
    X_all = hstack([X_dense, X_cat]).tocsr()
    return X_all

def main():
    feat = pd.read_csv(STACK_DIR / "val_meta_features.csv")
    y = np.load(STACK_DIR / "y_val.npy")
    groups = np.load(STACK_DIR / "groups_val.npy", allow_pickle=True)
    ids = feat["id"].astype(str).values

    model_names, X_probs = load_model_probs(ids)

    print("[MODELS]", model_names)
    print("[X_probs]", X_probs.shape)

    rows = []

    n_models = len(model_names)
    avg_probs = X_probs.reshape(len(y), n_models, N_CLASS).mean(axis=1)

    rows.append(score_row("equal_avg", y, avg_probs.argmax(1)))

    avg_bias, _ = tune_bias(avg_probs.copy(), y)
    rows.append(score_row("equal_avg_plus_bias", y, (avg_probs + avg_bias).argmax(1)))
    print("[equal_avg_bias]", avg_bias)

    X_all = build_X(feat, X_probs)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic stacking
    logit_oof = np.zeros((len(y), N_CLASS), dtype=np.float32)

    for fold, (tr, va) in enumerate(sgkf.split(X_all, y, groups)):
        clf = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
        )
        clf.fit(X_all[tr], y[tr])
        logit_oof[va] = clf.predict_proba(X_all[va])
        print("[LOGIT fold]", fold, "macro", macro_f1(y[va], logit_oof[va].argmax(1)))

    rows.append(score_row("logistic_stacking_oof", y, logit_oof.argmax(1)))

    logit_bias, _ = tune_bias(logit_oof.copy(), y)
    rows.append(score_row("logistic_stacking_oof_plus_bias", y, (logit_oof + logit_bias).argmax(1)))
    print("[logistic_bias]", logit_bias)

    np.save(STACK_DIR / "logistic_stack_oof_probs.npy", logit_oof)
    np.save(STACK_DIR / "logistic_stack_bias.npy", logit_bias)

    # LightGBM stacking
    if HAS_LGB:
        lgb_oof = np.zeros((len(y), N_CLASS), dtype=np.float32)

        for fold, (tr, va) in enumerate(sgkf.split(X_all, y, groups)):
            train_set = lgb.Dataset(X_all[tr], label=y[tr])
            valid_set = lgb.Dataset(X_all[va], label=y[va])

            params = {
                "objective": "multiclass",
                "num_class": N_CLASS,
                "metric": "multi_logloss",
                "learning_rate": 0.04,
                "num_leaves": 31,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "min_data_in_leaf": 40,
                "lambda_l2": 1.0,
                "verbose": -1,
                "seed": 42 + fold,
            }

            model = lgb.train(
                params,
                train_set,
                valid_sets=[valid_set],
                num_boost_round=2000,
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(100),
                ],
            )

            lgb_oof[va] = model.predict(X_all[va], num_iteration=model.best_iteration)
            print("[LGB fold]", fold, "macro", macro_f1(y[va], lgb_oof[va].argmax(1)))

        rows.append(score_row("lgb_stacking_oof", y, lgb_oof.argmax(1)))

        lgb_bias, _ = tune_bias(lgb_oof.copy(), y)
        rows.append(score_row("lgb_stacking_oof_plus_bias", y, (lgb_oof + lgb_bias).argmax(1)))
        print("[lgb_bias]", lgb_bias)

        np.save(STACK_DIR / "lgb_stack_oof_probs.npy", lgb_oof)
        np.save(STACK_DIR / "lgb_stack_bias.npy", lgb_bias)
    else:
        print("[WARN] Skipping LightGBM. Logistic result only.")

    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n===== META STACKING SCORES =====")
    print(res.to_string(index=False))

    res.to_csv(STACK_DIR / "meta_stacking_scores.csv", index=False)
    np.save(STACK_DIR / "equal_avg_probs.npy", avg_probs)
    np.save(STACK_DIR / "equal_avg_bias.npy", avg_bias)

    with open(STACK_DIR / "meta_model_names.json", "w", encoding="utf-8") as f:
        json.dump(model_names, f, ensure_ascii=False, indent=2)

    print("[SAVED]", STACK_DIR / "meta_stacking_scores.csv")

if __name__ == "__main__":
    main()
