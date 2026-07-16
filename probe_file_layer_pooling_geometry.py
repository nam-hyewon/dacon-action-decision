#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import submit.script_large_tab_diet_rob8_gated40 as sota


MODEL_DIR = Path(
    "submit/model/kluebert_mid12_lr15_fold0"
)

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

OUT_DIR = Path(
    "file_layer_pooling_geometry"
)

FILE_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]

FILE_SET = set(FILE_CLASSES)

LAYERS = [3, 6, 9, 12]

MAX_LEN = 512
BATCH_SIZE = 16


def metric(y, pred):
    return (
        f1_score(
            y,
            pred,
            labels=range(4),
            average="macro",
            zero_division=0,
        ),
        accuracy_score(y, pred),
    )


def load_rows():
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
            str(r["id"]): str(r["action"])
            for r in csv.DictReader(f)
        }

    val_ids = set(
        pd.read_csv(VAL_REF)["id"]
        .astype(str)
        .tolist()
    )

    train_samples = []
    train_y = []

    val_samples = []
    val_y = []

    for sample in samples:
        sid = str(sample["id"])
        label = label_map[sid]

        if label not in FILE_SET:
            continue

        y = FILE_CLASSES.index(label)

        if sid in val_ids:
            val_samples.append(sample)
            val_y.append(y)
        else:
            train_samples.append(sample)
            train_y.append(y)

    return (
        train_samples,
        np.asarray(train_y, dtype=np.int64),
        val_samples,
        np.asarray(val_y, dtype=np.int64),
    )


def build_text(sample):
    return sota.build_text_eda_state_mid(sample)


@torch.inference_mode()
def extract(
    name,
    samples,
    tokenizer,
    model,
    device,
):
    cache = {
        (layer, pool): []
        for layer in LAYERS
        for pool in ["cls", "mean"]
    }

    for start in range(
        0,
        len(samples),
        BATCH_SIZE,
    ):
        batch_samples = samples[
            start:start + BATCH_SIZE
        ]

        texts = [
            build_text(x)
            for x in batch_samples
        ]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        enc = {
            k: v.to(device)
            for k, v in enc.items()
        }

        out = model(
            **enc,
            output_hidden_states=True,
            return_dict=True,
        )

        mask = enc["attention_mask"].unsqueeze(-1)

        for layer in LAYERS:
            h = out.hidden_states[layer]

            cls = h[:, 0, :]

            mean = (
                (h * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1)
            )

            cache[(layer, "cls")].append(
                cls.float().cpu().numpy()
            )

            cache[(layer, "mean")].append(
                mean.float().cpu().numpy()
            )

        done = min(
            start + BATCH_SIZE,
            len(samples),
        )

        if done % 1000 < BATCH_SIZE:
            print(
                name,
                done,
                "/",
                len(samples),
                flush=True,
            )

    result = {}

    for key, parts in cache.items():
        result[key] = np.concatenate(
            parts,
            axis=0,
        )

    return result


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_samples,
        train_y,
        val_samples,
        val_y,
    ) = load_rows()

    print("TRAIN:", len(train_y))
    print("VAL:", len(val_y))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print("DEVICE:", device)

    train_cache = OUT_DIR / "train_poolings.npz"
    val_cache = OUT_DIR / "val_poolings.npz"

    if train_cache.exists() and val_cache.exists():
        print("LOADING CACHE")

        tr = np.load(train_cache)
        va = np.load(val_cache)

        train_emb = {
            (
                int(key.split("_")[1]),
                key.split("_")[2],
            ): tr[key]
            for key in tr.files
        }

        val_emb = {
            (
                int(key.split("_")[1]),
                key.split("_")[2],
            ): va[key]
            for key in va.files
        }

    else:
        train_emb = extract(
            "train",
            train_samples,
            tokenizer,
            model,
            device,
        )

        val_emb = extract(
            "val",
            val_samples,
            tokenizer,
            model,
            device,
        )

        np.savez_compressed(
            train_cache,
            **{
                f"layer_{layer}_{pool}": arr
                for (layer, pool), arr
                in train_emb.items()
            },
        )

        np.savez_compressed(
            val_cache,
            **{
                f"layer_{layer}_{pool}": arr
                for (layer, pool), arr
                in val_emb.items()
            },
        )

    del model
    torch.cuda.empty_cache()

    rows = []

    for layer in LAYERS:
        for pool in ["cls", "mean"]:
            print()
            print("=" * 100)
            print("LAYER", layer, "|", pool)
            print("=" * 100)

            x_train = train_emb[
                (layer, pool)
            ].astype(np.float32)

            x_val = val_emb[
                (layer, pool)
            ].astype(np.float32)

            scaler = StandardScaler()

            x_train_s = scaler.fit_transform(
                x_train
            )

            x_val_s = scaler.transform(
                x_val
            )

            linear = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=500,
                solver="lbfgs",
                n_jobs=-1,
            )

            linear.fit(
                x_train_s,
                train_y,
            )

            pred = linear.predict(
                x_val_s
            )

            f1, acc = metric(
                val_y,
                pred,
            )

            print(
                "LINEAR F1:",
                f1,
                "| ACC:",
                acc,
            )

            rows.append({
                "layer": layer,
                "pool": pool,
                "head": "linear",
                "macro_f1": f1,
                "accuracy": acc,
            })

            lda = LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            )

            lda.fit(
                x_train_s,
                train_y,
            )

            pred = lda.predict(
                x_val_s
            )

            f1, acc = metric(
                val_y,
                pred,
            )

            print(
                "LDA F1:",
                f1,
                "| ACC:",
                acc,
            )

            rows.append({
                "layer": layer,
                "pool": pool,
                "head": "lda",
                "macro_f1": f1,
                "accuracy": acc,
            })

    df = pd.DataFrame(rows).sort_values(
        "macro_f1",
        ascending=False,
    )

    print()
    print("=" * 120)
    print("FINAL RANKING")
    print("=" * 120)

    print(
        df.to_string(index=False)
    )

    df.to_csv(
        OUT_DIR / "results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
