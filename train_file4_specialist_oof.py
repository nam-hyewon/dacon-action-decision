import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
FILE4_SET = set(FILE4)

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

OUT_DIR = BASE_DIR / "file4_specialist_oof"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def row_zscore(x, eps=1e-6):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def flatten_obj(x):
    parts = []

    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).lower() in {"label", "target", "answer"}:
                continue
            parts.append(str(k))
            parts.append(flatten_obj(v))
    elif isinstance(x, list):
        for item in x:
            parts.append(flatten_obj(item))
    elif x is None:
        pass
    else:
        parts.append(str(x))

    return " ".join(p for p in parts if p)


def normalize_text(s):
    s = str(s).replace("\\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_train_texts():
    rows = []
    with open(ROOT / "data/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rid = str(obj["id"])
            text = normalize_text(flatten_obj(obj))
            rows.append({"id": rid, "text": text})
    return pd.DataFrame(rows)


def find_label_col(df):
    for c in ["label", "true_label", "action", "target"]:
        if c in df.columns:
            return c
    return None


def load_train_labels():
    df = pd.read_csv(ROOT / "data/train_labels.csv")
    df["id"] = df["id"].astype(str)
    label_col = find_label_col(df)
    if label_col is None:
        raise ValueError(f"Cannot find label column in train_labels.csv: {df.columns.tolist()}")

    def to_label_name(x):
        if pd.isna(x):
            raise ValueError("NaN label found")
        xs = str(x).strip()
        if xs.isdigit():
            return id2label[int(xs)]
        return xs

    df["label"] = df[label_col].map(to_label_name)
    return df[["id", "label"]]


def load_oof_one(name, path):
    logits = np.load(path / "val_logits.npy")
    df = pd.read_csv(path / "val_predictions.csv")
    df["id"] = df["id"].astype(str)

    if "label" not in df.columns:
        if "true_action" in df.columns:
            df["label"] = df["true_action"].astype(str)
        elif "true_label" in df.columns:
            df["label"] = df["true_label"].astype(str)
        elif "action" in df.columns:
            df["label"] = df["action"].astype(str)
        elif "target" in df.columns:
            df["label"] = df["target"].astype(str)
        else:
            raise ValueError(f"{name}: cannot find label column: {df.columns.tolist()}")

    if "fold" not in df.columns:
        raise ValueError(f"{name}: OOF prediction must have fold column. columns={df.columns.tolist()}")

    if len(df) != logits.shape[0]:
        raise ValueError(f"{name}: df/logits mismatch {len(df)} vs {logits.shape}")

    return logits, df


def load_base_ensemble():
    ref_ids = None
    ref_df = None
    combined = None
    model_logits = {}

    for name, path in OOF_PATHS.items():
        logits, df = load_oof_one(name, path)
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

        z = row_zscore(logits)
        model_logits[name] = z
        part = WEIGHTS[name] * z
        combined = part if combined is None else combined + part
        print(f"{name:8s} loaded {logits.shape}, weight={WEIGHTS[name]}")

    y_true = np.array([label2id[x] for x in ref_df["label"].astype(str).tolist()], dtype=int)
    return combined, ref_df, y_true, model_logits


def make_specialist_model(C=2.0):
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 3),
            min_df=2,
            max_features=120000,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=120000,
            sublinear_tf=True,
        )),
    ])

    clf = LogisticRegression(
        C=C,
        max_iter=1500,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        random_state=43,
        verbose=0,
    )

    return Pipeline([
        ("features", features),
        ("clf", clf),
    ])


def eval_preds(y_true, pred):
    return f1_score(y_true, pred, average="macro"), accuracy_score(y_true, pred)


def main():
    print("===== load base ensemble =====")
    base_logits, ref_df, y_true, model_logits = load_base_ensemble()

    base_pred_idx = base_logits.argmax(axis=1)
    base_pred = np.array([id2label[int(i)] for i in base_pred_idx])
    base_prob = softmax(base_logits)
    base_conf = base_prob.max(axis=1)

    base_f1, base_acc = eval_preds(y_true, base_pred_idx)

    print("\n===== base 4-model ensemble =====")
    print("weights:", WEIGHTS)
    print("macro_f1:", base_f1)
    print("accuracy:", base_acc)

    print("\n===== load train texts/labels =====")
    text_df = load_train_texts()
    label_df = load_train_labels()
    full_df = text_df.merge(label_df, on="id", how="inner")
    print("full train rows:", len(full_df))

    # ref_df order에 text 붙이기
    id_to_text = dict(zip(full_df["id"], full_df["text"]))
    id_to_label = dict(zip(full_df["id"], full_df["label"]))

    ids = ref_df["id"].astype(str).tolist()
    folds = ref_df["fold"].astype(int).to_numpy()

    oof_text = np.array([id_to_text[i] for i in ids], dtype=object)
    oof_label = np.array([id_to_label[i] for i in ids], dtype=object)

    assert np.all(oof_label == ref_df["label"].astype(str).to_numpy())

    # Specialist OOF predictions
    spec_probs = np.zeros((len(ids), len(FILE4)), dtype=np.float32)
    spec_pred = np.empty(len(ids), dtype=object)

    # fold별 specialist 모델 저장
    fold_model_dir = OUT_DIR / "fold_models"
    fold_model_dir.mkdir(exist_ok=True)

    print("\n===== train 5 fold file4 specialists =====")
    for fold in sorted(np.unique(folds)):
        val_mask = folds == fold
        train_mask = folds != fold

        # specialist는 train fold 중 true label이 file4인 샘플만 학습
        spec_train_mask = train_mask & np.isin(oof_label, FILE4)

        X_train = oof_text[spec_train_mask].tolist()
        y_train = oof_label[spec_train_mask].tolist()

        X_val = oof_text[val_mask].tolist()
        val_idx = np.where(val_mask)[0]

        print(f"\n--- fold {fold} ---")
        print("specialist train n:", len(X_train))
        print(pd.Series(y_train).value_counts().to_string())
        print("val n:", len(X_val))

        model = make_specialist_model(C=2.0)
        model.fit(X_train, y_train)

        classes = list(model.classes_)
        probs_raw = model.predict_proba(X_val)

        # FILE4 순서로 정렬
        idx = [classes.index(c) for c in FILE4]
        probs = probs_raw[:, idx]

        pred_idx = probs.argmax(axis=1)
        preds = np.array([FILE4[i] for i in pred_idx], dtype=object)

        spec_probs[val_idx] = probs
        spec_pred[val_idx] = preds

        joblib.dump(model, fold_model_dir / f"file4_tfidf_logreg_fold{fold}.joblib")

        # true file4 subset에서 specialist 자체 성능
        true_file4_val = np.isin(oof_label[val_mask], FILE4)
        if true_file4_val.sum() > 0:
            spec_true = np.array([label2id[x] for x in oof_label[val_mask][true_file4_val]])
            spec_p = np.array([label2id[x] for x in preds[true_file4_val]])
            sf1 = f1_score(spec_true, spec_p, labels=[label2id[c] for c in FILE4], average="macro")
            sacc = accuracy_score(spec_true, spec_p)
            print("specialist true-file4 macro_f1:", sf1)
            print("specialist true-file4 acc:", sacc)

    spec_conf = spec_probs.max(axis=1)
    sorted_probs = np.sort(spec_probs, axis=1)
    spec_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    # specialist 자체 성능: true label이 file4인 샘플에서만
    true_file4_mask = np.isin(oof_label, FILE4)
    spec_true = np.array([label2id[x] for x in oof_label[true_file4_mask]])
    spec_p = np.array([label2id[x] for x in spec_pred[true_file4_mask]])
    spec_file4_f1 = f1_score(spec_true, spec_p, labels=[label2id[c] for c in FILE4], average="macro")
    spec_file4_acc = accuracy_score(spec_true, spec_p)

    print("\n===== specialist standalone on true-file4 OOF subset =====")
    print("true-file4 n:", int(true_file4_mask.sum()))
    print("specialist file4 macro_f1:", spec_file4_f1)
    print("specialist file4 accuracy:", spec_file4_acc)

    # routing search
    print("\n===== route specialist on top of base ensemble =====")

    base_pred_name = base_pred
    base_pred_file4 = np.isin(base_pred_name, FILE4)

    # top2 안에 file4가 있는지
    top2_idx = np.argsort(base_logits, axis=1)[:, -2:]
    top2_file4 = np.isin(top2_idx, [label2id[c] for c in FILE4]).any(axis=1)

    results = []
    best = None

    def update_best(item, final_pred_idx, replace_mask):
        nonlocal best
        if best is None or item["macro_f1"] > best["macro_f1"]:
            best = item.copy()
            best["_pred_idx"] = final_pred_idx.copy()
            best["_replace_mask"] = replace_mask.copy()

    thresholds = np.arange(0.25, 0.96, 0.025)
    margins = [None, 0.00, 0.03, 0.05, 0.08, 0.10, 0.15]
    modes = {
        # 가장 안전한 모드
        "base_pred_file4": base_pred_file4,

        # 좀 더 공격적: base top2에 file4가 있으면 specialist 허용
        "base_top2_has_file4": top2_file4,

        # base가 file4를 예측했고 confidence 낮을 때만
        "base_pred_file4_lowconf": base_pred_file4 & (base_conf < np.quantile(base_conf, 0.70)),
    }

    for mode_name, route_base_mask in modes.items():
        for th in thresholds:
            for m in margins:
                conf_mask = spec_conf >= th
                if m is None:
                    margin_mask = np.ones(len(ids), dtype=bool)
                else:
                    margin_mask = spec_margin >= m

                replace_mask = route_base_mask & conf_mask & margin_mask

                final_pred_name = base_pred_name.copy()
                final_pred_name[replace_mask] = spec_pred[replace_mask]

                final_pred_idx = np.array([label2id[x] for x in final_pred_name], dtype=int)
                f1, acc = eval_preds(y_true, final_pred_idx)

                item = {
                    "mode": mode_name,
                    "threshold": float(th),
                    "margin": None if m is None else float(m),
                    "replace_n": int(replace_mask.sum()),
                    "replace_rate": float(replace_mask.mean()),
                    "macro_f1": float(f1),
                    "accuracy": float(acc),
                    "gain": float(f1 - base_f1),
                    "spec_file4_macro_f1": float(spec_file4_f1),
                    "spec_file4_acc": float(spec_file4_acc),
                }
                results.append(item)
                update_best(item, final_pred_idx, replace_mask)

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(OUT_DIR / "routing_search_results.csv", index=False)

    print("\n===== TOP 30 routing results =====")
    cols = ["macro_f1", "accuracy", "gain", "mode", "threshold", "margin", "replace_n", "replace_rate"]
    print(results_df[cols].head(30).to_string(index=False))

    print("\n===== BEST FILE4 SPECIALIST ROUTING =====")
    best_print = {k: v for k, v in best.items() if not k.startswith("_")}
    print(json.dumps(best_print, indent=2, ensure_ascii=False))

    print("\n===== classification report after file4 specialist =====")
    print(classification_report(y_true, best["_pred_idx"], target_names=ALL_CLASSES, digits=4))

    # replacement 분석
    best_replace = best["_replace_mask"]
    best_pred_name = np.array([id2label[int(i)] for i in best["_pred_idx"]], dtype=object)

    rep_df = pd.DataFrame({
        "id": ids,
        "fold": folds,
        "label": oof_label,
        "base_pred": base_pred_name,
        "spec_pred": spec_pred,
        "spec_conf": spec_conf,
        "spec_margin": spec_margin,
        "final_pred": best_pred_name,
        "replaced": best_replace,
    })

    rep_df.to_csv(OUT_DIR / "file4_specialist_oof_predictions.csv", index=False)

    print("\n===== replacement transitions top 30 =====")
    if best_replace.sum() > 0:
        print(
            rep_df[rep_df["replaced"]]
            .groupby(["base_pred", "spec_pred"])
            .size()
            .sort_values(ascending=False)
            .head(30)
            .to_string()
        )
    else:
        print("no replacements")

    np.save(OUT_DIR / "specialist_probs.npy", spec_probs)
    np.save(OUT_DIR / "final_pred_idx.npy", best["_pred_idx"])

    with open(OUT_DIR / "best_file4_specialist_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_weights": WEIGHTS,
            "base_macro_f1": float(base_f1),
            "base_accuracy": float(base_acc),
            "specialist_true_file4_macro_f1": float(spec_file4_f1),
            "specialist_true_file4_accuracy": float(spec_file4_acc),
            "best": best_print,
            "classes": ALL_CLASSES,
            "file4_classes": FILE4,
            "note": "OOF specialist. For each fold, train TF-IDF LogisticRegression on other folds' true file4 samples only.",
        }, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(" -", OUT_DIR / "best_file4_specialist_result.json")
    print(" -", OUT_DIR / "routing_search_results.csv")
    print(" -", OUT_DIR / "file4_specialist_oof_predictions.csv")
    print(" -", OUT_DIR / "specialist_probs.npy")
    print(" -", fold_model_dir)


if __name__ == "__main__":
    main()
