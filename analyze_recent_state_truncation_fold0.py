#!/usr/bin/env python3

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


MODEL_ID = "microsoft/deberta-v3-base"
MAX_LEN = 512

REPRO_ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

PRED_CSV = (
    REPRO_ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_team_module():
    path = Path(
        "submit/script_large_tab_diet_rob8_gated40.py"
    )

    spec = importlib.util.spec_from_file_location(
        "team_submit",
        path,
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def action_history(sample):
    return [
        h
        for h in sample.get("history", [])
        if h.get("role") != "user"
    ]


def field_text(h):
    return "\n".join([
        str(h.get("name", "")),
        str(h.get("args", "")),
        str(h.get("result_summary", "")),
    ])


print("Loading...")

samples = load_jsonl("data/train.jsonl")

id_to_sample = {
    str(x["id"]): x
    for x in samples
}

pred_df = pd.read_csv(PRED_CSV)
val_ids = pred_df["id"].astype(str).values

team = load_team_module()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)


rows = []

for n, sid in enumerate(val_ids):
    sample = id_to_sample[sid]

    # This is the text representation used by the current team script.
    text = team.build_text(sample)

    full_ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    trunc_ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LEN,
    )["input_ids"]

    decoded = tokenizer.decode(
        trunc_ids,
        skip_special_tokens=True,
    )

    hist = action_history(sample)

    row = {
        "id": sid,
        "full_tokens": len(full_ids),
        "truncated": len(full_ids) > MAX_LEN,
        "history_actions": len(hist),
    }

    for k in [1, 2, 3, 4]:
        if len(hist) >= k:
            h = hist[-k]

            name = str(h.get("name", ""))
            result = str(h.get("result_summary", ""))
            args = str(h.get("args", ""))

            row[f"last{k}_name_present"] = (
                bool(name)
                and name in decoded
            )

            # Result presence: use a reasonably distinctive prefix.
            result_probe = result.strip()[:80]

            row[f"last{k}_result_present"] = (
                len(result_probe) >= 10
                and result_probe in decoded
            )

            args_probe = args.strip()[:80]

            row[f"last{k}_args_present"] = (
                len(args_probe) >= 10
                and args_probe in decoded
            )

        else:
            row[f"last{k}_name_present"] = np.nan
            row[f"last{k}_result_present"] = np.nan
            row[f"last{k}_args_present"] = np.nan

    rows.append(row)

    if (n + 1) % 2000 == 0:
        print("processed", n + 1)


df = pd.DataFrame(rows)

print()
print("=" * 100)
print("TOKEN LENGTH")
print("=" * 100)

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
    "truncated:",
    int(df["truncated"].sum()),
    "/",
    len(df),
    "=",
    float(df["truncated"].mean()),
)


print()
print("=" * 100)
print("RECENT ACTION VISIBILITY AFTER 512 TRUNCATION")
print("=" * 100)

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
            float(df.loc[valid, col].mean()),
        )


print()
print("=" * 100)
print("TRUNCATED ROWS ONLY")
print("=" * 100)

sub = df[df["truncated"]]

for k in [1, 2, 3, 4]:
    print()
    print("LAST", k)

    for field in [
        "name_present",
        "args_present",
        "result_present",
    ]:
        col = f"last{k}_{field}"

        valid = sub[col].notna()

        print(
            field,
            "n=",
            int(valid.sum()),
            "rate=",
            float(sub.loc[valid, col].mean()),
        )


df.to_csv(
    "recent_state_truncation_diagnostics.csv",
    index=False,
)

print()
print("Saved recent_state_truncation_diagnostics.csv")
