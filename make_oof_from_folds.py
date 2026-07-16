import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
label2id = {c: i for i, c in enumerate(ALL_CLASSES)}
id2label = {i: c for i, c in enumerate(ALL_CLASSES)}

ROOT = Path(".")
FOLDS_DIR = Path(os.environ["FOLDS_DIR"])
OUT_DIR = Path(os.environ["OUT_DIR"])
MODEL_NAME = os.environ.get("MODEL_NAME", OUT_DIR.name)

if not FOLDS_DIR.is_absolute():
    FOLDS_DIR = ROOT / FOLDS_DIR
if not OUT_DIR.is_absolute():
    OUT_DIR = ROOT / OUT_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_label_col(df):
    for c in ["label", "true_label", "action", "target"]:
        if c in df.columns:
            return c
    return None


def load_train_labels():
    p = ROOT / "data/train_labels.csv"
    df = pd.read_csv(p)
    df["id"] = df["id"].astype(str)

    label_col = find_label_col(df)
    if label_col is None:
        raise ValueError(f"Cannot find label column in {p}: {df.columns.tolist()}")

    # Robustly handle pandas StringDtype / object / integer label columns.
    vals = df[label_col]

    def to_label_name(x):
        if pd.isna(x):
            raise ValueError("NaN label found in train_labels.csv")
        xs = str(x).strip()
        # numeric labels like "0", "1", ... or int labels
        if xs.isdigit():
            return id2label[int(xs)]
        return xs

    df["label_name"] = vals.map(to_label_name)

    return dict(zip(df["id"], df["label_name"]))


def find_fold_dir(fold_id):
    candidates = [
        FOLDS_DIR / f"deberta_base_eda_state4_fold{fold_id}_seed43",
        FOLDS_DIR / f"tabular_histgb_fold{fold_id}",
        FOLDS_DIR / f"kluebert_fold{fold_id}",
        FOLDS_DIR / f"roberta_fold{fold_id}",
        FOLDS_DIR / f"fold{fold_id}",
    ]

    for c in candidates:
        if (c / "val_logits.npy").exists() and (c / "val_predictions.csv").exists():
            return c

    # fallback: any subdir containing fold{fold_id}
    matches = []
    for d in FOLDS_DIR.iterdir():
        if d.is_dir() and f"fold{fold_id}" in d.name:
            if (d / "val_logits.npy").exists() and (d / "val_predictions.csv").exists():
                matches.append(d)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple candidate dirs for fold{fold_id}: {matches}")

    raise FileNotFoundError(f"Cannot find fold{fold_id} dir under {FOLDS_DIR}")


def main():
    print("FOLDS_DIR:", FOLDS_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("MODEL_NAME:", MODEL_NAME)

    id_to_label = load_train_labels()

    all_logits = []
    all_dfs = []
    fold_infos = []

    seen_ids = set()

    for fold_id in range(5):
        d = find_fold_dir(fold_id)
        logits = np.load(d / "val_logits.npy")
        df = pd.read_csv(d / "val_predictions.csv")
        df["id"] = df["id"].astype(str)

        if len(df) != logits.shape[0]:
            raise ValueError(f"fold{fold_id}: df/logits length mismatch {len(df)} vs {logits.shape}")

        ids = df["id"].tolist()
        dup = set(ids) & seen_ids
        if dup:
            raise ValueError(f"fold{fold_id}: duplicated ids across folds, example={list(dup)[:5]}")
        seen_ids.update(ids)

        # true label 붙이기
        df["label"] = [id_to_label[i] for i in ids]

        # prediction column 없거나 이상하면 logits 기준으로 다시 생성
        pred_idx = logits.argmax(axis=1)
        df["prediction"] = [id2label[int(i)] for i in pred_idx]

        df["fold"] = fold_id

        y_true = np.array([label2id[x] for x in df["label"]])
        y_pred = pred_idx

        f1 = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)

        print(f"fold{fold_id}: dir={d}")
        print(f"  n={len(df)} logits={logits.shape} macro_f1={f1:.6f} acc={acc:.6f}")

        all_logits.append(logits)
        all_dfs.append(df)
        fold_infos.append({
            "fold": fold_id,
            "dir": str(d),
            "n": int(len(df)),
            "macro_f1": float(f1),
            "accuracy": float(acc),
        })

    oof_logits = np.concatenate(all_logits, axis=0)
    oof_df = pd.concat(all_dfs, ignore_index=True)

    if len(oof_df) != len(set(oof_df["id"])):
        raise ValueError("OOF ids contain duplicates")

    y_true = np.array([label2id[x] for x in oof_df["label"]])
    y_pred = oof_logits.argmax(axis=1)

    oof_f1 = f1_score(y_true, y_pred, average="macro")
    oof_acc = accuracy_score(y_true, y_pred)

    print("\n===== OOF RESULT =====")
    print("n:", len(oof_df))
    print("logits:", oof_logits.shape)
    print("macro_f1:", oof_f1)
    print("accuracy:", oof_acc)

    print("\n===== classification report =====")
    print(classification_report(y_true, y_pred, labels=list(range(len(ALL_CLASSES))), target_names=ALL_CLASSES, digits=4))

    np.save(OUT_DIR / "oof_logits.npy", oof_logits)
    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    info = {
        "model_name": MODEL_NAME,
        "folds_dir": str(FOLDS_DIR),
        "num_folds": 5,
        "n": int(len(oof_df)),
        "logits_shape": list(oof_logits.shape),
        "macro_f1": float(oof_f1),
        "accuracy": float(oof_acc),
        "folds": fold_infos,
        "classes": ALL_CLASSES,
    }

    with open(OUT_DIR / "oof_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(" -", OUT_DIR / "oof_logits.npy")
    print(" -", OUT_DIR / "oof_predictions.csv")
    print(" -", OUT_DIR / "oof_info.json")


if __name__ == "__main__":
    main()
