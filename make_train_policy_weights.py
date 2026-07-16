#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import torch


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

EMB_PATH = Path(
    "policy_neighborhood_probe_fold0/train_embeddings.npy"
)

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/val_predictions.csv"
)

OUT_DIR = Path("policy_cleaning_weights_fold0")


def load_data():
    samples = []

    with open("data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    with open("data/train_labels.csv", encoding="utf-8") as f:
        label_map = {
            str(r["id"]): str(r["action"])
            for r in csv.DictReader(f)
        }

    return samples, label_map


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples, label_map = load_data()

    import pandas as pd

    val_ids = set(
        pd.read_csv(VAL_REF)["id"]
        .astype(str)
        .tolist()
    )

    train_samples = [
        s
        for s in samples
        if str(s["id"]) not in val_ids
    ]

    train_ids = np.asarray([
        str(s["id"])
        for s in train_samples
    ])

    y = np.asarray([
        C2I[label_map[sid]]
        for sid in train_ids
    ], dtype=np.int64)

    emb = np.load(EMB_PATH).astype(np.float32)

    if len(emb) != len(y):
        raise ValueError(
            f"embedding/label mismatch: {len(emb)} vs {len(y)}"
        )

    print("emb:", emb.shape)
    print("rows:", len(y))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    train_cpu = torch.from_numpy(emb)

    K = 20
    query_chunk = 256
    train_chunk = 8192

    purity = np.zeros(
        len(emb),
        dtype=np.float32,
    )

    entropy = np.zeros(
        len(emb),
        dtype=np.float32,
    )

    neighbor_pred = np.zeros(
        len(emb),
        dtype=np.int64,
    )

    for qs in range(
        0,
        len(emb),
        query_chunk,
    ):
        qe = min(
            qs + query_chunk,
            len(emb),
        )

        q = torch.from_numpy(
            emb[qs:qe]
        ).to(device)

        cur_n = qe - qs

        # k+1 because self should be removed.
        best_scores = torch.full(
            (cur_n, K + 1),
            -float("inf"),
            device=device,
        )

        best_idx = torch.full(
            (cur_n, K + 1),
            -1,
            dtype=torch.long,
            device=device,
        )

        for ts in range(
            0,
            len(emb),
            train_chunk,
        ):
            te = min(
                ts + train_chunk,
                len(emb),
            )

            t = train_cpu[ts:te].to(device)

            sim = q @ t.T

            local_k = min(
                K + 1,
                sim.shape[1],
            )

            score, idx = torch.topk(
                sim,
                k=local_k,
                dim=1,
            )

            idx += ts

            merged_score = torch.cat(
                [best_scores, score],
                dim=1,
            )

            merged_idx = torch.cat(
                [best_idx, idx],
                dim=1,
            )

            best_scores, pos = torch.topk(
                merged_score,
                k=K + 1,
                dim=1,
            )

            best_idx = torch.gather(
                merged_idx,
                1,
                pos,
            )

        idx_np = best_idx.cpu().numpy()

        for local_i in range(cur_n):
            global_i = qs + local_i

            neighbors = [
                int(x)
                for x in idx_np[local_i]
                if int(x) != global_i
            ][:K]

            labels = y[neighbors]

            counts = np.bincount(
                labels,
                minlength=len(CLASSES),
            ).astype(np.float64)

            probs = counts / counts.sum()

            purity[global_i] = probs.max()

            entropy[global_i] = float(
                -np.sum(
                    probs
                    * np.log(
                        np.clip(probs, 1e-12, 1.0)
                    )
                )
            )

            neighbor_pred[global_i] = int(
                probs.argmax()
            )

        print(
            "processed:",
            qe,
            "/",
            len(emb),
        )

    # Smooth weighting: don't annihilate hard groups.
    weights = np.select(
        [
            purity >= 0.90,
            purity >= 0.75,
            purity >= 0.60,
        ],
        [
            1.00,
            0.70,
            0.40,
        ],
        default=0.15,
    ).astype(np.float32)

    print()
    print("PURITY")
    print(
        "mean:",
        float(purity.mean()),
    )

    for q in [10, 25, 50, 75, 90]:
        print(
            f"p{q}:",
            float(np.percentile(purity, q)),
        )

    print()
    print("WEIGHT COUNTS")

    unique, counts = np.unique(
        weights,
        return_counts=True,
    )

    for w, n in zip(unique, counts):
        print(
            float(w),
            int(n),
            float(n / len(weights)),
        )

    print()
    print(
        "neighbor train agreement:",
        float((neighbor_pred == y).mean()),
    )

    np.save(
        OUT_DIR / "train_ids.npy",
        train_ids,
    )

    np.save(
        OUT_DIR / "purity.npy",
        purity,
    )

    np.save(
        OUT_DIR / "entropy.npy",
        entropy,
    )

    np.save(
        OUT_DIR / "sample_weights.npy",
        weights,
    )

    np.save(
        OUT_DIR / "neighbor_pred.npy",
        neighbor_pred,
    )

    print()
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    main()
