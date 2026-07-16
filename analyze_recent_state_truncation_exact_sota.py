#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


MAX_LEN = 512

REPRO_ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

PRED_CSV = (
    REPRO_ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)

SCRIPT_PATH = Path(
    "submit/script_large_tab_diet_rob8_gated40.py"
)

MODEL_CONFIGS = {
    "large": {
        "model_dir": "klue/roberta-large",
        "builder_name": "build_text_eda_state",
    },
    "rob8": {
        "model_dir": "klue/roberta-base",
        "builder_name": "build_text_eda_state",
    },
    "diet": {
        "model_dir": "klue/bert-base",
        "builder_name": "build_text_eda_state_diet",
    },
}



def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_submission_module():
    spec = importlib.util.spec_from_file_location(
        "sota_script",
        SCRIPT_PATH,
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def assistant_history(sample):
    return [
        h
        for h in sample.get("history", [])
        if h.get("role") != "user"
    ]


def normalized_probe(value, max_chars=60):
    text = str(value).strip()

    if len(text) < 8:
        return None

    # Builder may flatten newlines.
    text = " ".join(text.split())

    return text[:max_chars]


def visible_in_decoded(value, decoded):
    probe = normalized_probe(value)

    if probe is None:
        return np.nan

    decoded_norm = " ".join(decoded.split())

    return probe in decoded_norm


print("Loading data...")

samples = load_jsonl("data/train.jsonl")

sample_map = {
    str(s["id"]): s
    for s in samples
}

pred_df = pd.read_csv(PRED_CSV)

val_ids = pred_df["id"].astype(str).values

sota = load_submission_module()


all_results = []


for model_name, cfg in MODEL_CONFIGS.items():

    print()
    print("=" * 110)
    print("MODEL:", model_name)
    print("BUILDER:", cfg["builder_name"])
    print("MODEL DIR:", cfg["model_dir"])
    print("=" * 110)

    builder = getattr(
        sota,
        cfg["builder_name"],
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_dir"],
            use_fast=True,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_dir"],
            use_fast=False,
        )

    rows = []

    for pos, sid in enumerate(val_ids):

        sample = sample_map[sid]

        text = builder(sample)

        full_enc = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        trunc_enc = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN,
        )

        full_ids = full_enc["input_ids"]
        trunc_ids = trunc_enc["input_ids"]

        decoded = tokenizer.decode(
            trunc_ids,
            skip_special_tokens=True,
        )

        hist = assistant_history(sample)

        row = {
            "model": model_name,
            "id": sid,
            "full_tokens": len(full_ids),
            "truncated": len(full_ids) > MAX_LEN,
            "action_history_n": len(hist),
        }

        for k in [1, 2, 3, 4]:

            if len(hist) < k:
                row[f"last{k}_name_present"] = np.nan
                row[f"last{k}_args_present"] = np.nan
                row[f"last{k}_result_present"] = np.nan
                continue

            h = hist[-k]

            row[f"last{k}_name_present"] = (
                str(h.get("name", "")).strip()
                in decoded
            )

            row[f"last{k}_args_present"] = visible_in_decoded(
                h.get("args", ""),
                decoded,
            )

            row[f"last{k}_result_present"] = visible_in_decoded(
                h.get("result_summary", ""),
                decoded,
            )

        rows.append(row)

        if (pos + 1) % 3000 == 0:
            print("processed", pos + 1)

    df = pd.DataFrame(rows)

    all_results.append(df)

    print()
    print("TOKEN LENGTH")
    print(
        df["full_tokens"]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print()
    print(
        "TRUNCATED:",
        int(df["truncated"].sum()),
        "/",
        len(df),
        "=",
        float(df["truncated"].mean()),
    )

    print()
    print("ALL ROWS VISIBILITY")

    for k in [1, 2, 3, 4]:
        print()
        print("LAST", k)

        for field in [
            "name_present",
            "args_present",
            "result_present",
        ]:
            col = f"last{k}_{field}"

            valid = df[col].notna()

            print(
                field,
                "n=",
                int(valid.sum()),
                "rate=",
                float(
                    df.loc[valid, col]
                    .astype(float)
                    .mean()
                )
                if valid.any()
                else np.nan,
            )

    print()
    print("TRUNCATED ROWS ONLY")

    trunc_df = df[
        df["truncated"]
    ]

    for k in [1, 2, 3, 4]:
        print()
        print("LAST", k)

        for field in [
            "name_present",
            "args_present",
            "result_present",
        ]:
            col = f"last{k}_{field}"

            valid = trunc_df[col].notna()

            print(
                field,
                "n=",
                int(valid.sum()),
                "rate=",
                float(
                    trunc_df.loc[valid, col]
                    .astype(float)
                    .mean()
                )
                if valid.any()
                else np.nan,
            )


out = pd.concat(
    all_results,
    ignore_index=True,
)

out.to_csv(
    "recent_state_truncation_exact_sota.csv",
    index=False,
)


print()
print("=" * 110)
print("SUMMARY")
print("=" * 110)

summary_rows = []

for model_name in MODEL_CONFIGS:

    df = out[
        out["model"] == model_name
    ]

    trunc = df[
        df["truncated"]
    ]

    row = {
        "model": model_name,
        "mean_tokens": df["full_tokens"].mean(),
        "median_tokens": df["full_tokens"].median(),
        "p90_tokens": df["full_tokens"].quantile(0.90),
        "p95_tokens": df["full_tokens"].quantile(0.95),
        "trunc_rate": df["truncated"].mean(),
    }

    for k in [1, 2, 3, 4]:
        for field in [
            "args_present",
            "result_present",
        ]:
            col = f"last{k}_{field}"

            valid = trunc[col].notna()

            row[
                f"trunc_last{k}_{field}"
            ] = (
                trunc.loc[valid, col]
                .astype(float)
                .mean()
                if valid.any()
                else np.nan
            )

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)

print(
    summary.to_string(
        index=False,
    )
)

summary.to_csv(
    "recent_state_truncation_exact_sota_summary.csv",
    index=False,
)

print()
print("Saved:")
print(" - recent_state_truncation_exact_sota.csv")
print(" - recent_state_truncation_exact_sota_summary.csv")
