#!/usr/bin/env python3

import importlib.util
import json

import numpy as np

from transformers import AutoTokenizer


PATH = "src/train_roberta_decision_window_fold_ref.py"


spec = importlib.util.spec_from_file_location(
    "dw",
    PATH,
)

dw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dw)


samples = []

with open(
    "data/train.jsonl",
    encoding="utf-8",
) as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))


tok = AutoTokenizer.from_pretrained(
    "klue/roberta-base",
    use_fast=True,
)


lengths = []

visibility = {
    k: {
        "name": [],
        "args": [],
        "result": [],
    }
    for k in range(1, 5)
}


for sample in samples:

    text = dw.build_text_decision_window(sample)

    full_ids = tok(
        text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    trunc_ids = tok(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )["input_ids"]

    decoded = " ".join(
        tok.decode(
            trunc_ids,
            skip_special_tokens=True,
        ).split()
    )

    lengths.append(len(full_ids))

    hist = [
        h
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]

    for k in range(1, 5):

        if len(hist) < k:
            continue

        h = hist[-k]

        name = str(h.get("name", "")).strip()
        args = " ".join(
            str(h.get("args", "")).split()
        )[:50]

        result = " ".join(
            str(
                h.get(
                    "result_summary",
                    "",
                )
            ).split()
        )[:50]

        visibility[k]["name"].append(
            bool(name)
            and name in decoded
        )

        if len(args) >= 8:
            visibility[k]["args"].append(
                args in decoded
            )

        if len(result) >= 8:
            visibility[k]["result"].append(
                result in decoded
            )


arr = np.asarray(lengths)

print("=" * 100)
print("DECISION WINDOW TOKEN LENGTH")
print("=" * 100)

print("n:", len(arr))
print("mean:", arr.mean())
print("p50:", np.percentile(arr, 50))
print("p90:", np.percentile(arr, 90))
print("p95:", np.percentile(arr, 95))
print("p99:", np.percentile(arr, 99))
print("max:", arr.max())
print("over512:", (arr > 512).mean())


print()
print("=" * 100)
print("RECENT STATE VISIBILITY")
print("=" * 100)

for k in range(1, 5):
    print()
    print("T-", k)

    for field in [
        "name",
        "args",
        "result",
    ]:
        vals = visibility[k][field]

        print(
            field,
            "n=",
            len(vals),
            "rate=",
            np.mean(vals)
            if vals
            else np.nan,
        )
