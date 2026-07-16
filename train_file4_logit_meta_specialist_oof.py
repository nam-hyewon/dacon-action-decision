import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
label2id = {c: i for i, c in enumerate(ALL_CLASSES)}
id2label = {i: c for i, c in enumerate(ALL_CLASSES)}

FILE4 = ["read_file", "grep_search", "list_directory", "glob_pattern"]
FILE4_IDX = np.array([label2id[c] for c in FILE4], dtype=int)
file4_label2id = {c: i for i, c in enumerate(FILE4)}
file4_id2label = {i: c for i, c in enumerate(FILE4)}

ROOT = Path(".")
BASE_DIR = ROOT / "team_4model_ensemble_07626" / "ensemble_07626" / "fold0_team"

OOF_PATHS = {
    "deberta": BASE_DIR / "deberta_base_eda_state4_seed43_oof",
    "kluebert": BASE_DIR / "kluebert_mid5_lr15_oof",
    "roberta": BASE_DIR / "roberta_eda_state4_oof",
    "tabular": BASE_DIR / "tabular_histgb_oof",
}

WEIGHTS = {
    "deberta": 0.25,
    "kluebert": 0.20,
    "roberta": 0.15,
    "tabular": 0.40,
}

OUT_DIR = BASE_DIR / "file4_logit_meta_specialist_oof"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def row_zscore(x, eps=1e-6):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def margin_from_logits(x):
    s = np.sort(x, axis=1)
    return s[:, -1] - s[:, -2]


def rank_of_classes(logits, class_indices):
    # rank 0 = top1, rank 13 = worst
    order = np.argsort(-logits, axis=1)
    ranks = np.zeros((logits.shape[0], len(class_indices)), dtype=np.float32)
    for j, cls in enumerate(class_indices):
        ranks[:, j] = np.where(order == cls)[1]
    return ranks


def find_label_column(df):
    for c in ["label", "true_action", "true_label", "action", "target"]:
        if c in df.columns:
            return c
    return None


def load_one(name, path):
    logits = np.load(path / "val_logits.npy")
    df = pd.read_csv(path / "val_predictions.csv")
    df["id"] = df["id"].astype(str)

    label_col = find_label_column(df)
    if label_col is None:
        raise ValueError(f"{name}: cannot find label column: {df.columns.tolist()}")
    if label_col != "label":
        df["label"] = df[label_col].astype(str)

    if "fold" not in df.columns:
        raise ValueError(f"{name}: OOF val_predictions.csv must contain fold column")

    if len(df) != logits.shape[0]:
        raise ValueError(f"{name}: df/logits mismatch {len(df)} vs {logits.shape}")

    return logits, df


def load_all():
    ref_ids = None
    ref_df = None
    zlogits = {}

    for name, path in OOF_PATHS.items():
        logits, df = load_one(name, path)
        ids = df["id"].astype(str).tolist()

        if ref_ids is None:
            ref_ids = ids
            ref_df = df.copy()
        else:
            if ids != ref_ids:
                if set(ids) == set(ref_ids):
                    print(f"[reorder] {name}")
                    id_to_pos = {id_: i for i, id_ in enumerate(ids)}
                    order = [id_to_pos[i] for i in ref_ids]
                    logits = logits[order]
                    df = df.iloc[order].reset_index(drop=True)
                    assert df["id"].astype(str).tolist() == ref_ids
                else:
                    raise ValueError(f"{name}: id set mismatch")

        zlogits[name] = row_zscore(logits).astype(np.float32)
        print(f"{name:8s} loaded {logits.shape}")

    y_true = np.array([label2id[x] for x in ref_df["label"].astype(str).tolist()], dtype=int)
    folds = ref_df["fold"].astype(int).to_numpy()
    return zlogits, ref_df, y_true, folds


def build_base_logits(zlogits):
    base = None
    for name, w in WEIGHTS.items():
        part = w * zlogits[name]
        base = part if base is None else base + part
    return base.astype(np.float32)


def build_features(zlogits, base_logits):
    feats = []
    names = []

    base_prob = softmax(base_logits)
    base_pred = base_logits.argmax(axis=1)
    base_conf = base_prob.max(axis=1)
    base_margin = margin_from_logits(base_logits)

    # base features
    feats.append(base_logits)
    names += [f"base_logit_{c}" for c in ALL_CLASSES]

    feats.append(base_prob)
    names += [f"base_prob_{c}" for c in ALL_CLASSES]

    feats.append(base_logits[:, FILE4_IDX])
    names += [f"base_file4_logit_{c}" for c in FILE4]

    feats.append(base_prob[:, FILE4_IDX])
    names += [f"base_file4_prob_{c}" for c in FILE4]

    feats.append(rank_of_classes(base_logits, FILE4_IDX))
    names += [f"base_rank_{c}" for c in FILE4]

    feats.append(base_conf.reshape(-1, 1))
    names.append("base_conf")

    feats.append(base_margin.reshape(-1, 1))
    names.append("base_margin")

    # individual model features
    individual_preds = []
    for model_name, logits in zlogits.items():
        prob = softmax(logits)
        pred = logits.argmax(axis=1)
        individual_preds.append(pred)

        feats.append(logits)
        names += [f"{model_name}_logit_{c}" for c in ALL_CLASSES]

        feats.append(prob)
        names += [f"{model_name}_prob_{c}" for c in ALL_CLASSES]

        feats.append(logits[:, FILE4_IDX])
        names += [f"{model_name}_file4_logit_{c}" for c in FILE4]

        feats.append(prob[:, FILE4_IDX])
        names += [f"{model_name}_file4_prob_{c}" for c in FILE4]

        feats.append(rank_of_classes(logits, FILE4_IDX))
        names += [f"{model_name}_rank_{c}" for c in FILE4]

        feats.append(prob.max(axis=1).reshape(-1, 1))
        names.append(f"{model_name}_conf")

        feats.append(margin_from_logits(logits).reshape(-1, 1))
        names.append(f"{model_name}_margin")

    # disagreement features
    pred_mat = np.stack(individual_preds, axis=1)
    n_models = pred_mat.shape[1]
    disagree_count = np.zeros(pred_mat.shape[0], dtype=np.float32)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            disagree_count += (pred_mat[:, i] != pred_mat[:, j]).astype(np.float32)

    feats.append(disagree_count.reshape(-1, 1))
    names.append("pairwise_disagree_count")

    # each model agrees with base?
    for mi, model_name in enumerate(zlogits.keys()):
        agree = (pred_mat[:, mi] == base_pred).astype(np.float32)
        feats.append(agree.reshape(-1, 1))
        names.append(f"{model_name}_agree_base")

    X = np.concatenate(feats, axis=1).astype(np.float32)
    return X, names


def make_model(kind):
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=43,
        )

    if kind == "hgb_small":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.03,
            max_iter=160,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=43,
        )

    if kind == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=43,
        )

    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=43,
        )

    if kind == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.4,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=43,
            )),
        ])

    raise ValueError(kind)


def eval_idx(y_true, pred):
    return f1_score(y_true, pred, average="macro"), accuracy_score(y_true, pred)


def main():
    print("===== load OOF logits =====")
    zlogits, ref_df, y_true, folds = load_all()

    base_logits = build_base_logits(zlogits)
    base_pred_idx = base_logits.argmax(axis=1)
    base_pred_name = np.array([id2label[int(i)] for i in base_pred_idx], dtype=object)

    base_f1, base_acc = eval_idx(y_true, base_pred_idx)

    print("\n===== base 4-model ensemble =====")
    print("weights:", WEIGHTS)
    print("macro_f1:", base_f1)
    print("accuracy:", base_acc)

    print("\n===== build logit features =====")
    X, feature_names = build_features(zlogits, base_logits)
    print("X shape:", X.shape)

    pd.Series(feature_names).to_csv(OUT_DIR / "feature_names.csv", index=False, header=False)

    true_file4_mask = np.isin(y_true, FILE4_IDX)
    base_pred_file4 = np.isin(base_pred_idx, FILE4_IDX)

    print("true_file4 n:", int(true_file4_mask.sum()))
    print("base_pred_file4 n:", int(base_pred_file4.sum()))

    # meta-specialist는 true label이 file4인 train rows만 학습
    y_file4 = np.array([file4_label2id[id2label[int(i)]] if i in FILE4_IDX else -1 for i in y_true], dtype=int)

    model_kinds = ["hgb_small", "hgb"]
    thresholds = np.arange(0.25, 0.96, 0.025)
    margins = [None, 0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    modes = ["base_pred_file4", "base_top2_has_file4"]

    top2_idx = np.argsort(base_logits, axis=1)[:, -2:]
    top2_has_file4 = np.isin(top2_idx, FILE4_IDX).any(axis=1)

    all_rows = []
    global_best = None

    fold_model_root = OUT_DIR / "fold_models"
    fold_model_root.mkdir(exist_ok=True)

    for kind in model_kinds:
        print(f"\n==============================")
        print(f"MODEL KIND: {kind}")
        print(f"==============================")

        meta_probs = np.zeros((len(y_true), len(FILE4)), dtype=np.float32)
        meta_pred_file4 = np.zeros(len(y_true), dtype=int)

        kind_dir = fold_model_root / kind
        kind_dir.mkdir(exist_ok=True)

        for fold in sorted(np.unique(folds)):
            val_mask = folds == fold
            train_mask = folds != fold

            spec_train_mask = train_mask & true_file4_mask

            X_train = X[spec_train_mask]
            y_train = y_file4[spec_train_mask]

            X_val = X[val_mask]
            val_idx = np.where(val_mask)[0]

            print(f"fold {fold}: train_file4={len(y_train)}, val={len(val_idx)}")

            model = make_model(kind)
            model.fit(X_train, y_train)

            probs = model.predict_proba(X_val)

            # 혹시 class 순서가 다르면 FILE4 0..3 순서로 정렬
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
            else:
                classes = list(model.named_steps["clf"].classes_)

            idx = [classes.index(i) for i in range(len(FILE4))]
            probs = probs[:, idx]

            meta_probs[val_idx] = probs
            meta_pred_file4[val_idx] = probs.argmax(axis=1)

            joblib.dump(model, kind_dir / f"{kind}_fold{fold}.joblib")

        meta_conf = meta_probs.max(axis=1)
        sorted_probs = np.sort(meta_probs, axis=1)
        meta_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        # standalone specialist on true-file4
        spec_true = y_file4[true_file4_mask]
        spec_pred = meta_pred_file4[true_file4_mask]
        spec_f1 = f1_score(spec_true, spec_pred, average="macro")
        spec_acc = accuracy_score(spec_true, spec_pred)

        print(f"{kind} standalone true-file4 macro_f1={spec_f1:.6f} acc={spec_acc:.6f}")

        route_masks = {
            "base_pred_file4": base_pred_file4,
            "base_top2_has_file4": top2_has_file4,
        }

        kind_best = None

        for mode in modes:
            route_base_mask = route_masks[mode]

            for th in thresholds:
                for m in margins:
                    conf_mask = meta_conf >= th
                    if m is None:
                        margin_mask = np.ones(len(y_true), dtype=bool)
                    else:
                        margin_mask = meta_margin >= m

                    replace_mask = route_base_mask & conf_mask & margin_mask

                    final_pred_idx = base_pred_idx.copy()
                    final_pred_idx[replace_mask] = FILE4_IDX[meta_pred_file4[replace_mask]]

                    f1, acc = eval_idx(y_true, final_pred_idx)

                    item = {
                        "kind": kind,
                        "mode": mode,
                        "threshold": float(th),
                        "margin": None if m is None else float(m),
                        "replace_n": int(replace_mask.sum()),
                        "replace_rate": float(replace_mask.mean()),
                        "macro_f1": float(f1),
                        "accuracy": float(acc),
                        "gain": float(f1 - base_f1),
                        "standalone_file4_macro_f1": float(spec_f1),
                        "standalone_file4_acc": float(spec_acc),
                    }
                    all_rows.append(item)

                    if kind_best is None or item["macro_f1"] > kind_best["macro_f1"]:
                        kind_best = item.copy()
                        kind_best["_pred"] = final_pred_idx.copy()
                        kind_best["_replace_mask"] = replace_mask.copy()
                        kind_best["_meta_probs"] = meta_probs.copy()
                        kind_best["_meta_pred_file4"] = meta_pred_file4.copy()

                    if global_best is None or item["macro_f1"] > global_best["macro_f1"]:
                        global_best = item.copy()
                        global_best["_pred"] = final_pred_idx.copy()
                        global_best["_replace_mask"] = replace_mask.copy()
                        global_best["_meta_probs"] = meta_probs.copy()
                        global_best["_meta_pred_file4"] = meta_pred_file4.copy()

        print(f"\nBEST for {kind}:")
        print(json.dumps({k: v for k, v in kind_best.items() if not k.startswith('_')}, indent=2, ensure_ascii=False))

    rows_df = pd.DataFrame(all_rows).sort_values("macro_f1", ascending=False)
    rows_df.to_csv(OUT_DIR / "routing_search_results.csv", index=False)

    print("\n===== TOP 50 META-SPECIALIST ROUTING RESULTS =====")
    cols = [
        "macro_f1", "accuracy", "gain", "kind", "mode",
        "threshold", "margin", "replace_n", "replace_rate",
        "standalone_file4_macro_f1", "standalone_file4_acc",
    ]
    print(rows_df[cols].head(50).to_string(index=False))

    best_print = {k: v for k, v in global_best.items() if not k.startswith("_")}

    print("\n===== GLOBAL BEST META-SPECIALIST RESULT =====")
    print(json.dumps(best_print, indent=2, ensure_ascii=False))

    print("\n===== classification report after meta-specialist =====")
    print(classification_report(y_true, global_best["_pred"], target_names=ALL_CLASSES, digits=4))

    final_pred_name = np.array([id2label[int(i)] for i in global_best["_pred"]], dtype=object)
    meta_pred_name = np.array([FILE4[i] for i in global_best["_meta_pred_file4"]], dtype=object)
    replace_mask = global_best["_replace_mask"]

    pred_df = ref_df.copy()
    pred_df["base_pred"] = base_pred_name
    pred_df["meta_pred"] = meta_pred_name
    pred_df["final_pred"] = final_pred_name
    pred_df["replaced"] = replace_mask
    pred_df["meta_conf"] = global_best["_meta_probs"].max(axis=1)
    pred_df["meta_margin"] = np.sort(global_best["_meta_probs"], axis=1)[:, -1] - np.sort(global_best["_meta_probs"], axis=1)[:, -2]
    pred_df.to_csv(OUT_DIR / "meta_specialist_oof_predictions.csv", index=False)

    np.save(OUT_DIR / "best_meta_probs.npy", global_best["_meta_probs"])
    np.save(OUT_DIR / "best_final_pred_idx.npy", global_best["_pred"])

    with open(OUT_DIR / "best_meta_specialist_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_weights": WEIGHTS,
            "base_macro_f1": float(base_f1),
            "base_accuracy": float(base_acc),
            "best": best_print,
            "file4": FILE4,
            "note": "OOF logit-feature meta-specialist. Trained on other folds' true file4 samples only.",
        }, f, indent=2, ensure_ascii=False)

    print("\n===== replacement transitions =====")
    rep = pred_df[pred_df["replaced"] == True]
    if len(rep):
        print(rep.groupby(["base_pred", "meta_pred"]).size().sort_values(ascending=False).head(50).to_string())
        print("\nCorrectness change:")
        before = pred_df["base_pred"].astype(str).to_numpy() == ref_df["label"].astype(str).to_numpy()
        after = pred_df["final_pred"].astype(str).to_numpy() == ref_df["label"].astype(str).to_numpy()
        print(pd.crosstab(before, after, rownames=["base_correct"], colnames=["final_correct"]))
    else:
        print("no replacements")

    print("\nSaved:")
    print(" -", OUT_DIR / "best_meta_specialist_result.json")
    print(" -", OUT_DIR / "routing_search_results.csv")
    print(" -", OUT_DIR / "meta_specialist_oof_predictions.csv")
    print(" -", OUT_DIR / "best_meta_probs.npy")
    print(" -", OUT_DIR / "best_final_pred_idx.npy")
    print(" -", fold_model_root)


if __name__ == "__main__":
    main()
