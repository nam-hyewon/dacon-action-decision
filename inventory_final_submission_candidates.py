#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(".")

KNOWN = [
    # Current exact SOTA
    ("v2_score", "exact_repro_base_v2_score.npy"),
    ("v2_pred", "exact_repro_base_v2_pred.npy"),

    # Candidate models / outputs
    (
        "hierarchy",
        "submit/model/hierarchical_roberta_base_eda_state_fold0_seed43",
    ),
    (
        "weighted_large",
        "submit/model/klue_roberta_large_eda_state_policy_weighted_fold0",
    ),
    (
        "file4_deberta",
        "submit/model/deberta_file4_eda_state_fold0_seed43",
    ),
    (
        "file_serializer",
        "submit/model/file_policy_serializer_roberta_base_fold0_seed43",
    ),
    (
        "ask_plan_specialist",
        "submit/model/ask_plan_specialist_fold0",
    ),

    # Probe results
    ("candidate_zoo", "candidate_zoo_joint_stacker"),
    ("hidden_zoo", "hidden_zoo_joint_stacker"),
    ("hidden_tab_fusion", "hidden_tabular_fusion_fold0"),

    # Repro bundle / recipes
    (
        "team_oof_bundle",
        "oof_bundle_20260714",
    ),
]

print("=" * 120)
print("KNOWN FINAL CANDIDATES")
print("=" * 120)

for name, raw in KNOWN:
    p = Path(raw)

    print()
    print(f"[{name}]")
    print("path:", p)
    print("exists:", p.exists())

    if not p.exists():
        continue

    if p.is_file():
        print("size:", p.stat().st_size)

        if p.suffix == ".npy":
            try:
                a = np.load(p, mmap_mode="r")
                print(
                    "npy:",
                    a.shape,
                    a.dtype,
                )
            except Exception as e:
                print("load error:", repr(e))

        continue

    for filename in [
        "val_logits.npy",
        "val_predictions.csv",
        "test_logits.npy",
        "test_predictions.csv",
        "meta.json",
        "val_info.json",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ]:
        q = p / filename

        if not q.exists():
            continue

        info = f"{filename}: YES size={q.stat().st_size}"

        if q.suffix == ".npy":
            try:
                a = np.load(q, mmap_mode="r")
                info += f" shape={a.shape} dtype={a.dtype}"
            except Exception as e:
                info += f" LOAD_ERROR={repr(e)}"

        if q.suffix == ".csv":
            try:
                df = pd.read_csv(q)
                info += (
                    f" shape={df.shape}"
                    f" cols={df.columns.tolist()}"
                )
            except Exception as e:
                info += f" LOAD_ERROR={repr(e)}"

        print(info)

print()
print("=" * 120)
print("SCRIPTS WITH LIKELY FINAL METHODS")
print("=" * 120)

keywords = [
    "transition",
    "prior",
    "pair",
    "specialist",
    "gated",
    "generator",
    "template",
    "hierarchical",
    "policy_weighted",
]

for p in sorted(ROOT.glob("*.py")):
    low = p.name.lower()

    if any(k in low for k in keywords):
        print(p)

print()
print("=" * 120)
print("SUBMIT SCRIPTS")
print("=" * 120)

for p in sorted(Path("submit").glob("script*.py")):
    print(
        p,
        p.stat().st_size,
    )
