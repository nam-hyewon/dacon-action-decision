#!/usr/bin/env python3

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

C2I = {x: i for i, x in enumerate(CLASSES)}

FILE_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]

FILE_IDS = np.asarray(
    [C2I[x] for x in FILE_CLASSES],
    dtype=np.int64,
)

GLOBAL_TO_FILE = {
    global_id: file_id
    for file_id, global_id in enumerate(FILE_IDS)
}

MODEL_DIR = Path(
    "submit/model/kluebert_mid12_lr15_fold0"
)

SOTA_SCRIPT = Path(
    "submit/script_large_tab_diet_rob8_gated40.py"
)

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
    "/submit/model/klue_roberta_large_eda_state_fold0"
    "/val_predictions.csv"
)

OUT_DIR = Path(
    "large_file_latent_geometry_fold0"
)

MAX_LEN = 512
BATCH_SIZE = 16


def macro_f1_file(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(4),
        average="macro",
        zero_division=0,
    )


def load_sota_module():
    spec = importlib.util.spec_from_file_location(
        "sota",
        SOTA_SCRIPT,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_data():
    samples = []

    with open(
        "data/train.jsonl",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    with open(
        "data/train_labels.csv",
        encoding="utf-8",
    ) as f:
        label_map = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(f)
        }

    return samples, label_map


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def make_collate(tokenizer):
    def collate(texts):
        return tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    return collate


@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    texts,
    device,
    name,
):
    loader = DataLoader(
        TextDataset(texts),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_collate(tokenizer),
    )

    embeddings = []
    head_logits = []

    seen = 0

    model.eval()

    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )

        # KLUE RoBERTa classification head consumes first-token state.
        h = out.hidden_states[-1][:, 0, :]

        embeddings.append(
            h.float().cpu().numpy().astype(np.float32)
        )

        head_logits.append(
            out.logits.float().cpu().numpy().astype(np.float32)
        )

        seen += len(h)

        if seen % 2000 < BATCH_SIZE:
            print(
                f"{name}:",
                seen,
                "/",
                len(texts),
            )

    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(head_logits, axis=0),
    )


def class_geometry(x, y):
    rows = []

    centroids = []

    for class_id, action in enumerate(FILE_CLASSES):
        xc = x[y == class_id]

        centroid = xc.mean(axis=0)

        centroid /= (
            np.linalg.norm(centroid) + 1e-12
        )

        centroids.append(centroid)

        normalized = xc / np.clip(
            np.linalg.norm(
                xc,
                axis=1,
                keepdims=True,
            ),
            1e-12,
            None,
        )

        cosine_to_centroid = (
            normalized @ centroid
        )

        rows.append({
            "class_id": class_id,
            "action": action,
            "n": len(xc),
            "within_cos_mean": float(
                cosine_to_centroid.mean()
            ),
            "within_cos_std": float(
                cosine_to_centroid.std()
            ),
            "within_cos_p10": float(
                np.percentile(
                    cosine_to_centroid,
                    10,
                )
            ),
        })

    centroids = np.stack(
        centroids,
        axis=0,
    )

    centroid_cos = (
        centroids @ centroids.T
    )

    return (
        pd.DataFrame(rows),
        centroids,
        centroid_cos,
    )


def centroid_predict(
    train_x,
    train_y,
    val_x,
):
    train_norm = train_x / np.clip(
        np.linalg.norm(
            train_x,
            axis=1,
            keepdims=True,
        ),
        1e-12,
        None,
    )

    val_norm = val_x / np.clip(
        np.linalg.norm(
            val_x,
            axis=1,
            keepdims=True,
        ),
        1e-12,
        None,
    )

    centroids = []

    for class_id in range(4):
        c = train_norm[
            train_y == class_id
        ].mean(axis=0)

        c /= (
            np.linalg.norm(c) + 1e-12
        )

        centroids.append(c)

    centroids = np.stack(
        centroids,
        axis=0,
    )

    score = val_norm @ centroids.T

    return score.argmax(axis=1)


def print_result(name, y, pred):
    print()
    print(name)
    print(
        "ACC:",
        accuracy_score(y, pred),
    )
    print(
        "MACRO-F1:",
        macro_f1_file(y, pred),
    )
    print("CONFUSION:")
    print(
        confusion_matrix(
            y,
            pred,
            labels=range(4),
        )
    )


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("DEVICE:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("MODEL DIR:", MODEL_DIR)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(MODEL_DIR)

    # Explicitly test the weight before wasting a GPU allocation.
    weight_files = list(
        MODEL_DIR.glob("*.safetensors")
    )

    print(
        "safetensors:",
        weight_files,
    )

    samples, label_map = load_data()

    sample_by_id = {
        str(s["id"]): s
        for s in samples
    }

    val_df = pd.read_csv(
        VAL_REF
    )

    val_ids = (
        val_df["id"]
        .astype(str)
        .tolist()
    )

    val_id_set = set(val_ids)

    train_file_ids = []

    train_y = []

    for sample in samples:
        sid = str(sample["id"])

        if sid in val_id_set:
            continue

        action = label_map[sid]

        if action not in FILE_CLASSES:
            continue

        train_file_ids.append(sid)

        train_y.append(
            FILE_CLASSES.index(action)
        )

    val_file_ids = []

    val_y = []

    for sid in val_ids:
        action = label_map[sid]

        if action not in FILE_CLASSES:
            continue

        val_file_ids.append(sid)

        val_y.append(
            FILE_CLASSES.index(action)
        )

    train_y = np.asarray(
        train_y,
        dtype=np.int64,
    )

    val_y = np.asarray(
        val_y,
        dtype=np.int64,
    )

    print(
        "TRAIN FILE:",
        len(train_file_ids),
    )

    print(
        "VAL FILE:",
        len(val_file_ids),
    )

    print(
        "TRAIN COUNTS:",
        np.bincount(
            train_y,
            minlength=4,
        ),
    )

    print(
        "VAL COUNTS:",
        np.bincount(
            val_y,
            minlength=4,
        ),
    )

    sota = load_sota_module()

    builder = sota.build_text_eda_state_mid

    print(
        "Building exact eda_state texts..."
    )

    train_texts = [
        builder(sample_by_id[sid])
        for sid in train_file_ids
    ]

    val_texts = [
        builder(sample_by_id[sid])
        for sid in val_file_ids
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=True,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )

    model.float()
    model.to(device)

    train_emb_path = (
        OUT_DIR / "train_file_embeddings.npy"
    )

    val_emb_path = (
        OUT_DIR / "val_file_embeddings.npy"
    )

    train_logits_path = (
        OUT_DIR / "train_file_head_logits.npy"
    )

    val_logits_path = (
        OUT_DIR / "val_file_head_logits.npy"
    )

    if (
        train_emb_path.exists()
        and train_logits_path.exists()
    ):
        print(
            "Loading cached train outputs..."
        )

        train_x = np.load(
            train_emb_path
        )

        train_head_logits = np.load(
            train_logits_path
        )

    else:
        (
            train_x,
            train_head_logits,
        ) = extract_embeddings(
            model,
            tokenizer,
            train_texts,
            device,
            "train",
        )

        np.save(
            train_emb_path,
            train_x,
        )

        np.save(
            train_logits_path,
            train_head_logits,
        )

    if (
        val_emb_path.exists()
        and val_logits_path.exists()
    ):
        print(
            "Loading cached val outputs..."
        )

        val_x = np.load(
            val_emb_path
        )

        val_head_logits = np.load(
            val_logits_path
        )

    else:
        (
            val_x,
            val_head_logits,
        ) = extract_embeddings(
            model,
            tokenizer,
            val_texts,
            device,
            "val",
        )

        np.save(
            val_emb_path,
            val_x,
        )

        np.save(
            val_logits_path,
            val_head_logits,
        )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print()
    print(
        "TRAIN EMB:",
        train_x.shape,
    )

    print(
        "VAL EMB:",
        val_x.shape,
    )

    # ------------------------------------------------------------
    # Original classifier head.
    # ------------------------------------------------------------

    head_pred_global = val_head_logits.argmax(
        axis=1
    )

    # Restrict original head scores to FILE candidates.
    head_file_score = val_head_logits[
        :,
        FILE_IDS,
    ]

    head_file_pred = head_file_score.argmax(
        axis=1
    )

    print_result(
        "ORIGINAL HEAD, FILE-RESTRICTED",
        val_y,
        head_file_pred,
    )

    print(
        "\nOriginal global head group escape rate:",
        float(
            (
                ~np.isin(
                    head_pred_global,
                    FILE_IDS,
                )
            ).mean()
        ),
    )

    # ------------------------------------------------------------
    # Geometry.
    # ------------------------------------------------------------

    (
        geometry_df,
        train_centroids,
        centroid_cos,
    ) = class_geometry(
        val_x,
        val_y,
    )

    print()
    print("=" * 120)
    print("VAL WITHIN-CLASS GEOMETRY")
    print("=" * 120)

    print(
        geometry_df.to_string(
            index=False,
        )
    )

    print()
    print("CENTROID COSINE MATRIX")
    print(
        pd.DataFrame(
            centroid_cos,
            index=FILE_CLASSES,
            columns=FILE_CLASSES,
        ).to_string()
    )

    offdiag = centroid_cos[
        ~np.eye(
            4,
            dtype=bool,
        )
    ]

    print()
    print(
        "centroid offdiag cosine mean:",
        float(offdiag.mean()),
    )

    print(
        "centroid offdiag cosine max:",
        float(offdiag.max()),
    )

    # ------------------------------------------------------------
    # Nearest centroid.
    # ------------------------------------------------------------

    centroid_pred = centroid_predict(
        train_x,
        train_y,
        val_x,
    )

    print_result(
        "TRAIN-CENTROID COSINE HEAD",
        val_y,
        centroid_pred,
    )

    # ------------------------------------------------------------
    # Fresh linear head.
    # ------------------------------------------------------------

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_x
    )

    val_scaled = scaler.transform(
        val_x
    )

    linear = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    linear.fit(
        train_scaled,
        train_y,
    )

    linear_pred = linear.predict(
        val_scaled
    )

    print_result(
        "FRESH LINEAR HEAD",
        val_y,
        linear_pred,
    )

    # ------------------------------------------------------------
    # LDA: shared-covariance metric head.
    # ------------------------------------------------------------

    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
    )

    lda.fit(
        train_x,
        train_y,
    )

    lda_pred = lda.predict(
        val_x
    )

    print_result(
        "SHRINKAGE LDA HEAD",
        val_y,
        lda_pred,
    )

    # ------------------------------------------------------------
    # Silhouette on val, after PCA for stable/faster geometry.
    # ------------------------------------------------------------

    rng = np.random.default_rng(42)

    sample_n = min(
        5000,
        len(val_x),
    )

    sample_idx = rng.choice(
        len(val_x),
        size=sample_n,
        replace=False,
    )

    pca = PCA(
        n_components=64,
        random_state=42,
    )

    # Fit PCA on train representation only.
    pca.fit(
        train_x
    )

    val_pca_sample = pca.transform(
        val_x[sample_idx]
    )

    sil_cos = silhouette_score(
        val_pca_sample,
        val_y[sample_idx],
        metric="cosine",
    )

    sil_euc = silhouette_score(
        val_pca_sample,
        val_y[sample_idx],
        metric="euclidean",
    )

    print()
    print("=" * 120)
    print("GLOBAL GEOMETRY")
    print("=" * 120)

    print(
        "PCA64 explained variance:",
        float(
            pca.explained_variance_ratio_.sum()
        ),
    )

    print(
        "silhouette cosine:",
        float(sil_cos),
    )

    print(
        "silhouette euclidean:",
        float(sil_euc),
    )

    # ------------------------------------------------------------
    # Save summary.
    # ------------------------------------------------------------

    results = {
        "original_head_file_acc": float(
            accuracy_score(
                val_y,
                head_file_pred,
            )
        ),
        "original_head_file_f1": float(
            macro_f1_file(
                val_y,
                head_file_pred,
            )
        ),
        "centroid_acc": float(
            accuracy_score(
                val_y,
                centroid_pred,
            )
        ),
        "centroid_f1": float(
            macro_f1_file(
                val_y,
                centroid_pred,
            )
        ),
        "fresh_linear_acc": float(
            accuracy_score(
                val_y,
                linear_pred,
            )
        ),
        "fresh_linear_f1": float(
            macro_f1_file(
                val_y,
                linear_pred,
            )
        ),
        "lda_acc": float(
            accuracy_score(
                val_y,
                lda_pred,
            )
        ),
        "lda_f1": float(
            macro_f1_file(
                val_y,
                lda_pred,
            )
        ),
        "centroid_offdiag_cos_mean": float(
            offdiag.mean()
        ),
        "centroid_offdiag_cos_max": float(
            offdiag.max()
        ),
        "silhouette_cosine": float(
            sil_cos
        ),
        "silhouette_euclidean": float(
            sil_euc
        ),
    }

    geometry_df.to_csv(
        OUT_DIR / "class_geometry.csv",
        index=False,
    )

    pd.DataFrame(
        centroid_cos,
        index=FILE_CLASSES,
        columns=FILE_CLASSES,
    ).to_csv(
        OUT_DIR / "centroid_cosine.csv"
    )

    with open(
        OUT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 120)
    print("FINAL SUMMARY")
    print("=" * 120)

    for k, v in results.items():
        print(
            f"{k}: {v}"
        )

    print()
    print(
        "SAVED:",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
