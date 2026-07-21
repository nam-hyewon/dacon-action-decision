#!/usr/bin/env python3
"""Tune weighted ensembles for two or more validation logit files."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_model_spec(value):
    parts = str(value).split(":")
    if len(parts) != 3:
        raise ValueError(
            "--model must be formatted as name:val_predictions.csv:val_logits.npy"
        )
    name, csv_path, logits_path = parts
    if not name:
        raise ValueError("model name cannot be empty")
    return name, Path(csv_path), Path(logits_path)


def load_model(spec):
    name, csv_path, logits_path = parse_model_spec(spec)
    df = pd.read_csv(csv_path)
    logits = np.load(logits_path).astype(np.float32)
    if len(df) != len(logits):
        raise ValueError(
            f"row mismatch for {name}: {csv_path} has {len(df)} rows, "
            f"logits has {len(logits)}"
        )
    if "id" not in df.columns or "true_id" not in df.columns:
        raise ValueError(f"{csv_path} must contain id and true_id columns")
    return {
        "name": name,
        "csv": str(csv_path),
        "logits_path": str(logits_path),
        "df": df,
        "logits": logits,
    }


def align_to_base(base_df, other_df, other_logits):
    id_to_idx = {
        str(sid): i for i, sid in enumerate(other_df["id"].astype(str).tolist())
    }
    indices = []
    missing = []
    for sid in base_df["id"].astype(str).tolist():
        idx = id_to_idx.get(sid)
        if idx is None:
            missing.append(sid)
        else:
            indices.append(idx)
    if missing:
        raise ValueError(
            f"missing {len(missing)} ids in other csv. first={missing[:3]}"
        )
    return other_logits[np.array(indices)]


def normalize_logits(logits, mode):
    if mode == "none":
        return logits
    if mode == "zscore":
        mean = logits.mean(axis=1, keepdims=True)
        std = logits.std(axis=1, keepdims=True)
        return (logits - mean) / np.maximum(std, 1e-6)
    if mode == "softmax_log":
        probs = softmax(logits)
        return np.log(np.maximum(probs, 1e-12))
    raise ValueError(f"unknown normalize={mode}")


def infer_id2action(df, n_classes):
    pairs = []
    for id_col, action_col in [("true_id", "true_action"), ("pred_id", "pred_action")]:
        if id_col in df.columns and action_col in df.columns:
            pairs.append(
                df[[id_col, action_col]].rename(
                    columns={id_col: "id", action_col: "action"}
                )
            )
    if not pairs:
        return [str(i) for i in range(n_classes)]

    all_pairs = pd.concat(pairs, ignore_index=True).drop_duplicates()
    out = []
    for i in range(n_classes):
        acts = sorted(
            set(all_pairs.loc[all_pairs["id"].astype(int) == i, "action"].astype(str))
        )
        out.append(acts[0] if len(acts) == 1 else str(i))
    return out


def softmax(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def parse_float_list(value):
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def simplex_weights(n_models, step):
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--step must divide 1.0, e.g. 0.1, 0.05, 0.025")

    def rec(remaining, slots):
        if slots == 1:
            yield [remaining]
            return
        for value in range(remaining + 1):
            for tail in rec(remaining - value, slots - 1):
                yield [value] + tail

    for combo in rec(units, n_models):
        yield np.array(combo, dtype=np.float32) / units


def explicit_weights(weight_text, n_models):
    weights = []
    for chunk in str(weight_text).split(";"):
        vals = parse_float_list(chunk)
        if not vals:
            continue
        if len(vals) != n_models:
            raise ValueError(
                f"weight combo {vals} has {len(vals)} values, expected {n_models}"
            )
        arr = np.array(vals, dtype=np.float32)
        total = float(arr.sum())
        if total <= 0:
            raise ValueError(f"weight combo must sum positive: {vals}")
        weights.append(arr / total)
    return weights


def make_ensemble(logits_list, weights):
    out = np.zeros_like(logits_list[0], dtype=np.float32)
    for weight, logits in zip(weights, logits_list):
        out += float(weight) * logits
    return out


def main():
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Tune weighted ensemble of multiple validation logits files."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeatable. Format: name:val_predictions.csv:val_logits.npy",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "zscore", "softmax_log"],
        default="zscore",
    )
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--weights",
        default="",
        help="Optional explicit combos. Example: '0.5,0.3,0.2;0.4,0.4,0.2'",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--benchmark_repeats",
        type=int,
        default=20,
        help="Repeat best-weight ensemble prediction to estimate postprocess latency.",
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    load_start = time.perf_counter()
    models = [load_model(spec) for spec in args.model]
    if len(models) < 2:
        raise ValueError("provide at least two --model entries")
    load_sec = time.perf_counter() - load_start

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = models[0]["df"]
    y_true = base_df["true_id"].astype(int).values
    n_classes = models[0]["logits"].shape[1]
    id2action = infer_id2action(base_df, n_classes)

    prep_start = time.perf_counter()
    logits_list = []
    base_preds = {}
    for i, model in enumerate(models):
        logits = model["logits"]
        if i > 0:
            logits = align_to_base(base_df, model["df"], logits)
        if logits.shape != models[0]["logits"].shape:
            raise ValueError(
                f"logit shape mismatch for {model['name']}: "
                f"{logits.shape} vs {models[0]['logits'].shape}"
            )
        logits = normalize_logits(logits, args.normalize)
        logits_list.append(logits)
        base_preds[model["name"]] = logits.argmax(axis=1)
    prep_sec = time.perf_counter() - prep_start

    grid_start = time.perf_counter()
    if args.weights:
        weight_grid = explicit_weights(args.weights, len(models))
    else:
        weight_grid = list(simplex_weights(len(models), args.step))
    grid_build_sec = time.perf_counter() - grid_start

    rows = []
    best = None
    best_logits = None
    best_pred = None
    labels = list(range(n_classes))

    search_start = time.perf_counter()
    for weights in weight_grid:
        ens = make_ensemble(logits_list, weights)
        pred = ens.argmax(axis=1)
        macro_f1 = f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)
        acc = accuracy_score(y_true, pred)

        row = {
            "macro_f1": macro_f1,
            "accuracy": acc,
            "changed_vs_first": int((pred != base_preds[models[0]["name"]]).sum()),
        }
        for model, weight in zip(models, weights):
            name = model["name"]
            row[f"weight_{name}"] = float(weight)
            row[f"changed_vs_{name}"] = int((pred != base_preds[name]).sum())

        rows.append(row)
        if best is None or (macro_f1, acc) > (best["macro_f1"], best["accuracy"]):
            best = row
            best_logits = ens
            best_pred = pred
    search_sec = time.perf_counter() - search_start

    benchmark = None
    repeats = max(0, int(args.benchmark_repeats))
    if repeats:
        best_weights = np.array(
            [best[f"weight_{model['name']}"] for model in models],
            dtype=np.float32,
        )
        bench_start = time.perf_counter()
        for _ in range(repeats):
            bench_logits = make_ensemble(logits_list, best_weights)
            bench_pred = bench_logits.argmax(axis=1)
        if len(bench_pred) != len(y_true):
            raise RuntimeError("benchmark prediction length mismatch")
        bench_sec = time.perf_counter() - bench_start
        benchmark = {
            "repeats": repeats,
            "total_sec": bench_sec,
            "sec_per_repeat": bench_sec / repeats,
            "ms_per_repeat": 1000.0 * bench_sec / repeats,
            "ms_per_sample": 1000.0 * bench_sec / (repeats * len(y_true)),
            "samples_per_repeat": int(len(y_true)),
        }

    write_start = time.perf_counter()
    grid = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    grid.to_csv(out_dir / "ensemble_grid.csv", index=False)

    probs = softmax(best_logits)
    out_pred = base_df.copy()
    out_pred["pred_id"] = best_pred
    out_pred["pred_action"] = [id2action[i] for i in best_pred]
    out_pred["confidence"] = probs.max(axis=1)
    out_pred["correct"] = best_pred == y_true
    out_pred["ensemble_pred_id"] = best_pred
    out_pred["ensemble_pred_action"] = out_pred["pred_action"]
    out_pred["ensemble_confidence"] = out_pred["confidence"]
    out_pred["ensemble_correct"] = out_pred["correct"]
    out_pred.to_csv(out_dir / "val_predictions_ensemble.csv", index=False)
    np.save(out_dir / "val_logits_ensemble.npy", best_logits)
    write_sec = time.perf_counter() - write_start

    total_sec = time.perf_counter() - total_start
    timing = {
        "note": (
            "This measures validation logit loading, weight search, and ensemble "
            "postprocessing only. Transformer forward inference time for each base "
            "model is not included because this script consumes precomputed logits."
        ),
        "num_samples": int(len(y_true)),
        "num_models": int(len(models)),
        "num_classes": int(n_classes),
        "num_weight_combos": int(len(weight_grid)),
        "load_logits_sec": load_sec,
        "align_normalize_sec": prep_sec,
        "weight_grid_build_sec": grid_build_sec,
        "weight_search_sec": search_sec,
        "write_outputs_sec": write_sec,
        "total_sec": total_sec,
        "benchmark_best_weights": benchmark,
    }

    summary = {
        "models": [
            {
                "name": model["name"],
                "csv": model["csv"],
                "logits": model["logits_path"],
            }
            for model in models
        ],
        "normalize": args.normalize,
        "step": args.step,
        "num_weight_combos": len(weight_grid),
        "best": best,
        "timing": timing,
    }
    with open(out_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)

    print("Top ensemble grid:")
    print(grid.head(args.top_k).to_string(index=False))
    print("\nBest:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("\nTiming:")
    print(json.dumps(timing, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(" -", out_dir / "ensemble_grid.csv")
    print(" -", out_dir / "ensemble_summary.json")
    print(" -", out_dir / "timing.json")
    print(" -", out_dir / "val_predictions_ensemble.csv")
    print(" -", out_dir / "val_logits_ensemble.npy")


if __name__ == "__main__":
    main()
