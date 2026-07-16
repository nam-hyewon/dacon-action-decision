#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
I2C = {i: x for i, x in enumerate(CLASSES)}

GROUPS = {
    "FILE": [
        "read_file",
        "grep_search",
        "list_directory",
        "glob_pattern",
    ],
    "WRITE": [
        "edit_file",
        "write_file",
        "apply_patch",
    ],
    "EXEC": [
        "run_bash",
        "run_tests",
        "lint_or_typecheck",
    ],
    "CONTROL": [
        "ask_user",
        "plan_task",
        "web_search",
        "respond_only",
    ],
}

GROUP_NAMES = list(GROUPS)

CLASS_TO_GROUP = {}

for gi, group_name in enumerate(GROUP_NAMES):
    for action in GROUPS[group_name]:
        CLASS_TO_GROUP[C2I[action]] = gi

CLASS_TO_GROUP_ARR = np.asarray([
    CLASS_TO_GROUP[i]
    for i in range(len(CLASSES))
], dtype=np.int64)


MODEL_DIR = Path(
    "submit/model/kluebert_mid12_lr15_fold0"
)

SOTA_SCRIPT = Path(
    "submit/script_large_tab_diet_rob8_gated40.py"
)

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
    "/submit/model/kluebert_diet_fold0"
    "/val_predictions.csv"
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--batch",
        type=int,
        default=64,
    )

    p.add_argument(
        "--max-len",
        type=int,
        default=512,
    )

    p.add_argument(
        "--query-chunk",
        type=int,
        default=256,
    )

    p.add_argument(
        "--train-chunk",
        type=int,
        default=8192,
    )

    p.add_argument(
        "--max-k",
        type=int,
        default=100,
    )

    p.add_argument(
        "--output-dir",
        default="policy_neighborhood_probe_fold0",
    )

    return p.parse_args()


def macro_f1(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
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


def make_collate(
    tokenizer,
    max_len,
):

    def collate(texts):
        return tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    return collate


@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    texts,
    device,
    batch_size,
    max_len,
    name,
):
    ds = TextDataset(texts)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_collate(
            tokenizer,
            max_len,
        ),
    )

    all_emb = []

    model.eval()

    processed = 0

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
            outputs = model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )

        # Last-layer first token.
        emb = outputs.hidden_states[-1][:, 0, :]

        emb = torch.nn.functional.normalize(
            emb.float(),
            dim=-1,
        )

        all_emb.append(
            emb.cpu().numpy().astype(np.float32)
        )

        processed += len(emb)

        if processed % 5000 < batch_size:
            print(
                f"{name} embeddings:",
                processed,
                "/",
                len(texts),
            )

    return np.concatenate(
        all_emb,
        axis=0,
    )


@torch.no_grad()
def chunked_topk_cosine(
    query_emb,
    train_emb,
    k,
    device,
    query_chunk,
    train_chunk,
):
    """
    Exact cosine top-k using normalized embeddings.
    Chunked to avoid allocating Q x N full similarity matrix.
    """

    train_emb = np.asarray(
        train_emb,
        dtype=np.float32,
    )

    query_emb = np.asarray(
        query_emb,
        dtype=np.float32,
    )

    all_scores = []
    all_indices = []

    n_query = len(query_emb)

    train_cpu = torch.from_numpy(train_emb)

    for qs in range(
        0,
        n_query,
        query_chunk,
    ):
        qe = min(
            qs + query_chunk,
            n_query,
        )

        q = torch.from_numpy(
            query_emb[qs:qe]
        ).to(device)

        cur_n = qe - qs

        best_scores = torch.full(
            (cur_n, k),
            -float("inf"),
            device=device,
            dtype=torch.float32,
        )

        best_indices = torch.full(
            (cur_n, k),
            -1,
            device=device,
            dtype=torch.long,
        )

        for ts in range(
            0,
            len(train_emb),
            train_chunk,
        ):
            te = min(
                ts + train_chunk,
                len(train_emb),
            )

            t = train_cpu[ts:te].to(
                device,
                non_blocking=True,
            )

            sim = q @ t.T

            local_k = min(
                k,
                sim.shape[1],
            )

            local_scores, local_idx = torch.topk(
                sim,
                k=local_k,
                dim=1,
            )

            local_idx = local_idx + ts

            merged_scores = torch.cat(
                [
                    best_scores,
                    local_scores,
                ],
                dim=1,
            )

            merged_indices = torch.cat(
                [
                    best_indices,
                    local_idx,
                ],
                dim=1,
            )

            top_scores, positions = torch.topk(
                merged_scores,
                k=k,
                dim=1,
            )

            best_indices = torch.gather(
                merged_indices,
                dim=1,
                index=positions,
            )

            best_scores = top_scores

            del t
            del sim
            del local_scores
            del local_idx
            del merged_scores
            del merged_indices
            del positions

        all_scores.append(
            best_scores.cpu().numpy()
        )

        all_indices.append(
            best_indices.cpu().numpy()
        )

        print(
            "neighbor search:",
            qe,
            "/",
            n_query,
        )

    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_indices, axis=0),
    )


def neighbor_stats(
    neighbor_idx,
    neighbor_scores,
    train_y,
    k,
    weighted=False,
):
    idx = neighbor_idx[:, :k]
    sim = neighbor_scores[:, :k]

    labels = train_y[idx]

    n = len(idx)

    probs = np.zeros(
        (n, len(CLASSES)),
        dtype=np.float64,
    )

    if weighted:
        # Positive exponential similarity weighting.
        # Center within row for numerical stability.
        weights = np.exp(
            15.0 * (
                sim
                - sim.max(axis=1, keepdims=True)
            )
        )
    else:
        weights = np.ones_like(
            sim,
            dtype=np.float64,
        )

    for class_id in range(len(CLASSES)):
        probs[:, class_id] = (
            weights
            * (labels == class_id)
        ).sum(axis=1)

    probs /= probs.sum(
        axis=1,
        keepdims=True,
    ).clip(min=1e-12)

    pred = probs.argmax(axis=1)

    purity = probs.max(axis=1)

    entropy = -(
        probs
        * np.log(
            probs.clip(min=1e-12)
        )
    ).sum(axis=1)

    entropy /= math.log(
        len(CLASSES)
    )

    margin = np.sort(
        probs,
        axis=1,
    )[:, -1] - np.sort(
        probs,
        axis=1,
    )[:, -2]

    return {
        "probs": probs,
        "pred": pred,
        "purity": purity,
        "entropy": entropy,
        "margin": margin,
    }


def print_distribution_stats(
    name,
    values,
):
    values = np.asarray(values)

    print(
        name,
        "| n=",
        len(values),
        "| mean=",
        float(values.mean()),
        "| p25=",
        float(np.percentile(values, 25)),
        "| p50=",
        float(np.percentile(values, 50)),
        "| p75=",
        float(np.percentile(values, 75)),
    )


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
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
        raise FileNotFoundError(
            f"MODEL_DIR does not exist: {MODEL_DIR}"
        )

    print("Loading data...")

    samples, label_map = load_data()

    val_df = pd.read_csv(
        VAL_REF
    )

    val_ids = (
        val_df["id"]
        .astype(str)
        .tolist()
    )

    y_val = (
        val_df["true_id"]
        .astype(int)
        .values
    )

    base_pred = np.load(
        "exact_repro_base_pred.npy"
    ).astype(np.int64)

    if len(y_val) != len(base_pred):
        raise ValueError(
            (
                "validation/base length mismatch: "
                f"{len(y_val)} vs {len(base_pred)}"
            )
        )

    id_to_idx = {
        str(sample["id"]): i
        for i, sample in enumerate(samples)
    }

    val_indices = np.asarray([
        id_to_idx[sid]
        for sid in val_ids
    ], dtype=np.int64)

    val_set = set(val_ids)

    train_indices = np.asarray([
        i
        for i, sample in enumerate(samples)
        if str(sample["id"]) not in val_set
    ], dtype=np.int64)

    train_y = np.asarray([
        C2I[
            label_map[
                str(samples[i]["id"])
            ]
        ]
        for i in train_indices
    ], dtype=np.int64)

    print(
        "train:",
        len(train_indices),
        "| val:",
        len(val_indices),
    )

    print(
        "BASE F1:",
        macro_f1(
            y_val,
            base_pred,
        ),
    )

    sota = load_sota_module()

    state_builder = (
        sota.build_text_eda_state_mid
    )

    print(
        "Building EDA_STATE_DIET texts..."
    )

    train_texts = [
        state_builder(samples[i])
        for i in train_indices
    ]

    val_texts = [
        state_builder(samples[i])
        for i in val_indices
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
        output_dir
        / "train_embeddings.npy"
    )

    val_emb_path = (
        output_dir
        / "val_embeddings.npy"
    )

    if train_emb_path.exists():
        print(
            "Loading cached train embeddings..."
        )

        train_emb = np.load(
            train_emb_path
        )

    else:
        train_emb = extract_embeddings(
            model=model,
            tokenizer=tokenizer,
            texts=train_texts,
            device=device,
            batch_size=args.batch,
            max_len=args.max_len,
            name="train",
        )

        np.save(
            train_emb_path,
            train_emb,
        )

    if val_emb_path.exists():
        print(
            "Loading cached val embeddings..."
        )

        val_emb = np.load(
            val_emb_path
        )

    else:
        val_emb = extract_embeddings(
            model=model,
            tokenizer=tokenizer,
            texts=val_texts,
            device=device,
            batch_size=args.batch,
            max_len=args.max_len,
            name="val",
        )

        np.save(
            val_emb_path,
            val_emb,
        )

    print(
        "TRAIN EMB:",
        train_emb.shape,
    )

    print(
        "VAL EMB:",
        val_emb.shape,
    )

    # Free model before exact all-train neighbor search.
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    neighbor_score_path = (
        output_dir
        / "neighbor_scores.npy"
    )

    neighbor_idx_path = (
        output_dir
        / "neighbor_indices.npy"
    )

    if (
        neighbor_score_path.exists()
        and neighbor_idx_path.exists()
    ):
        print(
            "Loading cached neighbor search..."
        )

        neighbor_scores = np.load(
            neighbor_score_path
        )

        neighbor_idx = np.load(
            neighbor_idx_path
        )

    else:
        print(
            "Running exact chunked cosine top-k..."
        )

        neighbor_scores, neighbor_idx = (
            chunked_topk_cosine(
                query_emb=val_emb,
                train_emb=train_emb,
                k=args.max_k,
                device=device,
                query_chunk=args.query_chunk,
                train_chunk=args.train_chunk,
            )
        )

        np.save(
            neighbor_score_path,
            neighbor_scores,
        )

        np.save(
            neighbor_idx_path,
            neighbor_idx,
        )

    base_correct = (
        base_pred == y_val
    )

    true_group = (
        CLASS_TO_GROUP_ARR[
            y_val
        ]
    )

    pred_group = (
        CLASS_TO_GROUP_ARR[
            base_pred
        ]
    )

    group_correct = (
        true_group == pred_group
    )

    within_error = (
        (~base_correct)
        & group_correct
    )

    cross_group_error = (
        ~group_correct
    )

    base_f1 = macro_f1(
        y_val,
        base_pred,
    )

    result_rows = []

    print()
    print("=" * 140)
    print("NEIGHBORHOOD PURITY ANALYSIS")
    print("=" * 140)

    for k in [
        3,
        5,
        10,
        20,
        30,
        50,
        100,
    ]:
        if k > args.max_k:
            continue

        for weighted in [
            False,
            True,
        ]:

            stats = neighbor_stats(
                neighbor_idx=neighbor_idx,
                neighbor_scores=neighbor_scores,
                train_y=train_y,
                k=k,
                weighted=weighted,
            )

            tag = (
                f"k={k} weighted={weighted}"
            )

            print()
            print("-" * 120)
            print(tag)
            print("-" * 120)

            print_distribution_stats(
                "purity base correct",
                stats["purity"][base_correct],
            )

            print_distribution_stats(
                "purity base wrong",
                stats["purity"][~base_correct],
            )

            print_distribution_stats(
                "purity within error",
                stats["purity"][within_error],
            )

            print_distribution_stats(
                "purity cross-group error",
                stats["purity"][cross_group_error],
            )

            print_distribution_stats(
                "entropy base correct",
                stats["entropy"][base_correct],
            )

            print_distribution_stats(
                "entropy base wrong",
                stats["entropy"][~base_correct],
            )

            print(
                "neighbor standalone acc:",
                accuracy_score(
                    y_val,
                    stats["pred"],
                ),
            )

            print(
                "neighbor standalone F1:",
                macro_f1(
                    y_val,
                    stats["pred"],
                ),
            )

            beneficial_all = (
                (base_pred != y_val)
                & (stats["pred"] == y_val)
            )

            harmful_all = (
                (base_pred == y_val)
                & (stats["pred"] != y_val)
            )

            print(
                "oracle complementarity beneficial:",
                int(
                    beneficial_all.sum()
                ),
            )

            print(
                "harmful if full replace:",
                int(
                    harmful_all.sum()
                ),
            )

            for purity_thr in [
                0.40,
                0.50,
                0.60,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
            ]:

                for margin_thr in [
                    0.00,
                    0.10,
                    0.20,
                    0.30,
                    0.40,
                    0.50,
                ]:

                    use = (
                        (stats["purity"] >= purity_thr)
                        & (stats["margin"] >= margin_thr)
                    )

                    pred = base_pred.copy()

                    pred[use] = stats["pred"][
                        use
                    ]

                    f1 = macro_f1(
                        y_val,
                        pred,
                    )

                    changed = (
                        use
                        & (pred != base_pred)
                    )

                    beneficial = (
                        changed
                        & (base_pred != y_val)
                        & (pred == y_val)
                    )

                    harmful = (
                        changed
                        & (base_pred == y_val)
                        & (pred != y_val)
                    )

                    result_rows.append({
                        "k": k,
                        "weighted": weighted,
                        "purity_thr": purity_thr,
                        "margin_thr": margin_thr,
                        "macro_f1": f1,
                        "gain": f1 - base_f1,
                        "accuracy": accuracy_score(
                            y_val,
                            pred,
                        ),
                        "use_n": int(
                            use.sum()
                        ),
                        "changed_n": int(
                            changed.sum()
                        ),
                        "beneficial": int(
                            beneficial.sum()
                        ),
                        "harmful": int(
                            harmful.sum()
                        ),
                        "net_correct_gain": int(
                            beneficial.sum()
                            - harmful.sum()
                        ),
                    })

    result_df = pd.DataFrame(
        result_rows
    ).sort_values(
        [
            "macro_f1",
            "accuracy",
        ],
        ascending=False,
    )

    print()
    print("=" * 160)
    print("GLOBAL GATED TOP 50")
    print("=" * 160)

    print(
        result_df.head(50).to_string(
            index=False,
        )
    )

    result_df.to_csv(
        output_dir
        / "gated_results.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Group-specific diagnostic.
    # ---------------------------------------------------------------------

    print()
    print("=" * 160)
    print("GROUP-SPECIFIC PURITY: K=20 WEIGHTED")
    print("=" * 160)

    diag = neighbor_stats(
        neighbor_idx=neighbor_idx,
        neighbor_scores=neighbor_scores,
        train_y=train_y,
        k=min(20, args.max_k),
        weighted=True,
    )

    group_rows = []

    for gi, group_name in enumerate(
        GROUP_NAMES
    ):
        mask = (
            true_group == gi
        )

        correct_mask = (
            mask
            & base_correct
        )

        error_mask = (
            mask
            & within_error
        )

        row = {
            "group": group_name,
            "n": int(mask.sum()),
            "correct_n": int(
                correct_mask.sum()
            ),
            "within_error_n": int(
                error_mask.sum()
            ),
            "correct_purity_mean": (
                float(
                    diag["purity"][
                        correct_mask
                    ].mean()
                )
                if correct_mask.any()
                else np.nan
            ),
            "error_purity_mean": (
                float(
                    diag["purity"][
                        error_mask
                    ].mean()
                )
                if error_mask.any()
                else np.nan
            ),
            "correct_entropy_mean": (
                float(
                    diag["entropy"][
                        correct_mask
                    ].mean()
                )
                if correct_mask.any()
                else np.nan
            ),
            "error_entropy_mean": (
                float(
                    diag["entropy"][
                        error_mask
                    ].mean()
                )
                if error_mask.any()
                else np.nan
            ),
            "neighbor_acc": float(
                accuracy_score(
                    y_val[mask],
                    diag["pred"][mask],
                )
            ),
        }

        group_rows.append(row)

    group_df = pd.DataFrame(
        group_rows
    )

    print(
        group_df.to_string(
            index=False,
        )
    )

    group_df.to_csv(
        output_dir
        / "group_purity.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Top confusion diagnostics.
    # ---------------------------------------------------------------------

    confusion_rows = []

    for true_id in range(len(CLASSES)):
        for pred_id in range(len(CLASSES)):

            mask = (
                (y_val == true_id)
                & (base_pred == pred_id)
                & (true_id != pred_id)
            )

            if mask.sum() < 20:
                continue

            confusion_rows.append({
                "true": I2C[true_id],
                "base_pred": I2C[pred_id],
                "n": int(mask.sum()),
                "purity_mean": float(
                    diag["purity"][mask].mean()
                ),
                "entropy_mean": float(
                    diag["entropy"][mask].mean()
                ),
                "neighbor_correct_rate": float(
                    (
                        diag["pred"][mask]
                        == y_val[mask]
                    ).mean()
                ),
                "neighbor_base_rate": float(
                    (
                        diag["pred"][mask]
                        == base_pred[mask]
                    ).mean()
                ),
            })

    confusion_df = pd.DataFrame(
        confusion_rows
    ).sort_values(
        "n",
        ascending=False,
    )

    print()
    print("=" * 160)
    print("TOP BASE CONFUSIONS + NEIGHBORHOOD")
    print("=" * 160)

    print(
        confusion_df.head(40).to_string(
            index=False,
        )
    )

    confusion_df.to_csv(
        output_dir
        / "confusion_purity.csv",
        index=False,
    )

    best = result_df.iloc[0]

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print(
        "BASE F1:",
        base_f1,
    )

    print(
        "BEST GATED F1:",
        float(best["macro_f1"]),
    )

    print(
        "BEST GAIN:",
        float(best["gain"]),
    )

    print()
    print("BEST CONFIG")
    print(
        best.to_string()
    )

    print()
    print(
        "SAVED:",
        output_dir,
    )


if __name__ == "__main__":
    main()
