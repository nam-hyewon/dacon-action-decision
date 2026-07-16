#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.neural_network import MLPClassifier

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import submit.script_large_tab_diet_rob8_gated40 as sota


MODEL_DIR = Path(
    "submit/model/kluebert_mid12_lr15_fold0"
)

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

BASE_SCORE_PATH = Path(
    "exact_repro_base_v2_score.npy"
)

BASE_PRED_PATH = Path(
    "exact_repro_base_v2_pred.npy"
)

LGB_INFO_PATH = Path(
    "tabular_lightgbm_fold0/val_info.json"
)

OUT_DIR = Path(
    "hidden_tabular_fusion_fold0"
)

MAX_LEN = 512
BATCH_SIZE = 16

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

LABEL2ID = {
    x: i
    for i, x in enumerate(CLASSES)
}


def macro_f1(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(14),
        average="macro",
        zero_division=0,
    )


def load_data():
    samples = []

    with open(
        "data/train.jsonl",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                samples.append(
                    json.loads(line)
                )

    with open(
        "data/train_labels.csv",
        encoding="utf-8",
    ) as f:
        labels = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(f)
        }

    val_df = pd.read_csv(
        VAL_REF
    )

    val_ids_order = (
        val_df["id"]
        .astype(str)
        .tolist()
    )

    val_ids = set(
        val_ids_order
    )

    by_id = {
        str(x["id"]): x
        for x in samples
    }

    train_ids = [
        str(x["id"])
        for x in samples
        if str(x["id"]) not in val_ids
    ]

    train_samples = [
        by_id[x]
        for x in train_ids
    ]

    val_samples = [
        by_id[x]
        for x in val_ids_order
    ]

    y_train = np.asarray(
        [
            LABEL2ID[labels[x]]
            for x in train_ids
        ],
        dtype=np.int64,
    )

    y_val = np.asarray(
        [
            LABEL2ID[labels[x]]
            for x in val_ids_order
        ],
        dtype=np.int64,
    )

    return (
        train_samples,
        y_train,
        val_samples,
        y_val,
    )


@torch.inference_mode()
def extract_embeddings(
    name,
    samples,
    tokenizer,
    model,
    device,
):
    result = []

    for start in range(
        0,
        len(samples),
        BATCH_SIZE,
    ):
        cur = samples[
            start:start + BATCH_SIZE
        ]

        texts = [
            sota.build_text_eda_state_mid(x)
            for x in cur
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

        h = out.hidden_states[-1][
            :,
            0,
            :,
        ]

        result.append(
            h.float()
            .cpu()
            .numpy()
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

    return np.concatenate(
        result,
        axis=0,
    ).astype(np.float32)


def build_tabular(
    samples,
    categorical_cols,
    numeric_cols,
):
    df = pd.DataFrame(
        [
            sota.make_features(x)
            for x in samples
        ]
    )

    expected = (
        categorical_cols
        + numeric_cols
    )

    missing = [
        x
        for x in expected
        if x not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"missing exact LGB features: {missing}"
        )

    return df[expected].copy()


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_samples,
        y_train,
        val_samples,
        y_val,
    ) = load_data()

    print(
        "TRAIN:",
        len(y_train)
    )

    print(
        "VAL:",
        len(y_val)
    )

    info = json.loads(
        LGB_INFO_PATH.read_text()
    )

    categorical_cols = info[
        "categorical_cols"
    ]

    numeric_cols = info[
        "numeric_cols"
    ]

    print(
        "CATEGORICAL:",
        len(categorical_cols),
    )

    print(
        "NUMERIC:",
        len(numeric_cols),
    )

    train_emb_path = (
        OUT_DIR
        / "train_embeddings.npy"
    )

    val_emb_path = (
        OUT_DIR
        / "val_embeddings.npy"
    )

    if (
        train_emb_path.exists()
        and val_emb_path.exists()
    ):
        print(
            "LOADING EMBEDDING CACHE"
        )

        train_emb = np.load(
            train_emb_path
        )

        val_emb = np.load(
            val_emb_path
        )

    else:
        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                MODEL_DIR,
                local_files_only=True,
            )
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                MODEL_DIR,
                local_files_only=True,
            )
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "DEVICE:",
            device
        )

        model.to(device)
        model.eval()

        train_emb = extract_embeddings(
            "train",
            train_samples,
            tokenizer,
            model,
            device,
        )

        val_emb = extract_embeddings(
            "val",
            val_samples,
            tokenizer,
            model,
            device,
        )

        np.save(
            train_emb_path,
            train_emb,
        )

        np.save(
            val_emb_path,
            val_emb,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "TRAIN EMB:",
        train_emb.shape
    )

    print(
        "VAL EMB:",
        val_emb.shape
    )

    print(
        "Building exact tabular features..."
    )

    train_tab = build_tabular(
        train_samples,
        categorical_cols,
        numeric_cols,
    )

    val_tab = build_tabular(
        val_samples,
        categorical_cols,
        numeric_cols,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="missing",
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            min_frequency=5,
                        ),
                    ),
                ]),
                categorical_cols,
            ),
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                        ),
                    ),
                    (
                        "scale",
                        StandardScaler(),
                    ),
                ]),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )

    train_tab_x = (
        preprocessor.fit_transform(
            train_tab
        )
    ).astype(np.float32)

    val_tab_x = (
        preprocessor.transform(
            val_tab
        )
    ).astype(np.float32)

    print(
        "TAB TRAIN:",
        train_tab_x.shape
    )

    print(
        "TAB VAL:",
        val_tab_x.shape
    )

    emb_scaler = StandardScaler()

    train_emb_x = emb_scaler.fit_transform(
        train_emb
    ).astype(np.float32)

    val_emb_x = emb_scaler.transform(
        val_emb
    ).astype(np.float32)

    # ----------------------------------------------------------
    # Three controlled conditions.
    # ----------------------------------------------------------

    experiments = {
        "hidden_only": (
            train_emb_x,
            val_emb_x,
        ),
        "tabular_only": (
            train_tab_x,
            val_tab_x,
        ),
        "hidden_plus_tabular": (
            np.concatenate(
                [
                    train_emb_x,
                    train_tab_x,
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    val_emb_x,
                    val_tab_x,
                ],
                axis=1,
            ),
        ),
    }

    base_pred = np.load(
        BASE_PRED_PATH
    ).astype(np.int64)

    base_f1 = macro_f1(
        y_val,
        base_pred,
    )

    print()
    print(
        "CANONICAL V2 BASE F1:",
        base_f1,
    )

    rows = []

    for name, (
        x_train,
        x_val,
    ) in experiments.items():
        print()
        print("=" * 120)
        print(name)
        print("=" * 120)

        print(
            "shape:",
            x_train.shape,
            x_val.shape,
        )

        clf = MLPClassifier(
            hidden_layer_sizes=(
                512,
                128,
            ),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=5e-4,
            max_iter=60,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=6,
            random_state=43,
            verbose=True,
        )

        clf.fit(
            x_train,
            y_train,
        )

        probs = clf.predict_proba(
            x_val
        )

        pred = probs.argmax(
            axis=1
        )

        f1 = macro_f1(
            y_val,
            pred,
        )

        acc = accuracy_score(
            y_val,
            pred,
        )

        print(
            "F1:",
            f1
        )

        print(
            "ACC:",
            acc
        )

        base_correct = (
            base_pred == y_val
        )

        cur_correct = (
            pred == y_val
        )

        beneficial = int(
            (
                ~base_correct
                & cur_correct
            ).sum()
        )

        harmful = int(
            (
                base_correct
                & ~cur_correct
            ).sum()
        )

        oracle_pred = base_pred.copy()

        use = (
            ~base_correct
            & cur_correct
        )

        oracle_pred[
            use
        ] = pred[use]

        oracle_f1 = macro_f1(
            y_val,
            oracle_pred,
        )

        print(
            "base wrong / model correct:",
            beneficial,
        )

        print(
            "base correct / model wrong:",
            harmful,
        )

        print(
            "ORACLE UNION F1:",
            oracle_f1,
        )

        np.save(
            OUT_DIR
            / f"{name}_val_probs.npy",
            probs.astype(np.float32),
        )

        joblib.dump(
            clf,
            OUT_DIR
            / f"{name}_mlp.joblib",
        )

        rows.append({
            "name": name,
            "macro_f1": f1,
            "accuracy": acc,
            "beneficial": beneficial,
            "harmful": harmful,
            "oracle_union_f1": oracle_f1,
        })

    result = pd.DataFrame(
        rows
    ).sort_values(
        "macro_f1",
        ascending=False,
    )

    print()
    print("=" * 140)
    print("FINAL")
    print("=" * 140)

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        OUT_DIR / "results.csv",
        index=False,
    )

    joblib.dump(
        preprocessor,
        OUT_DIR
        / "tabular_preprocessor.joblib",
    )

    joblib.dump(
        emb_scaler,
        OUT_DIR
        / "embedding_scaler.joblib",
    )


if __name__ == "__main__":
    main()
