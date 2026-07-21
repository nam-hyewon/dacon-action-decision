#!/usr/bin/env python3
"""
Build & persist specialist boosters for submission.

The specialist booster gain (val Macro-F1 0.7194 -> 0.7318) is produced by
`src/tune_specialist_boosters.py`, but that script only tunes on saved
validation logits and never persists the fitted TF-IDF / LogisticRegression
models. This script re-trains the *winning* configuration (mode=router,
greedy per-target gammas) on the exact same 80% train split (seed 42) and
saves everything needed for inference into a single joblib file.

Output:
  submit/model/roberta_eda_state4/specialist_boosters/specialists.joblib
    {
      "mode": "router",
      "all_classes": [...14...],
      "vectorizer": <fitted TfidfVectorizer>,
      "clfs": {target_class: <fitted LogisticRegression>},
      "gammas": {target_class: float},
    }

It then verifies the saved models reproduce the tuned validation Macro-F1
(~0.7318) so we know the deployed booster matches what was tuned.

Run:
  cd dacon-action-decision
  python src/build_specialists.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Reuse the EXACT feature/text/split logic used during tuning so the deployed
# specialists are calibrated identically to the greedy gammas.
import tune_specialist_boosters as tsb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "submit" / "model" / "roberta_eda_state4"
BOOST_DIR = MODEL_DIR / "specialist_boosters"
OUT_PATH = BOOST_DIR / "specialists.joblib"

MODE = "router"          # summary.json best.mode
SEED = 42
MAX_FEATURES = 250000    # tune script defaults
MIN_DF = 2
C = 2.0


def load_gammas():
    """Per-target greedy gammas selected during tuning."""
    df = pd.read_csv(BOOST_DIR / "specialist_greedy_selected.csv")
    return {str(r["target"]): float(r["gamma"]) for _, r in df.iterrows()}


def main():
    all_classes = tsb.ALL_CLASSES
    gammas = load_gammas()
    active = {t: g for t, g in gammas.items() if g != 0.0}
    print("Gammas:", gammas)
    print("Active targets (gamma>0):", active)

    # ---- Same rows / split / vectorizer as tuning (mode=router) --------------
    print(f"\nLoading train rows (mode={MODE})...")
    rows = tsb.load_rows(DATA_DIR, text_mode=MODE)
    train_rows, _val_rows_split = tsb.split_like_train(rows, seed=SEED)
    print(f"train rows: {len(train_rows)}")

    train_texts = [r["text"] for r in train_rows]
    y_train = np.array([r["label"] for r in train_rows], dtype=np.int64)

    print("Fitting TF-IDF vectorizer...")
    vec = tsb.make_vectorizer(MODE, max_features=MAX_FEATURES, min_df=MIN_DF)
    Xtr = vec.fit_transform(train_texts)
    print("Xtr:", Xtr.shape)

    # ---- One-vs-rest specialist per active target ---------------------------
    clfs = {}
    for target, gamma in active.items():
        tid = all_classes.index(target)
        y_bin = (y_train == tid).astype(int)
        pos = int(y_bin.sum())
        if pos < 20:
            print(f"[WARN] skip {target}: too few positives {pos}")
            continue
        clf = LogisticRegression(
            C=C, class_weight="balanced", solver="liblinear",
            max_iter=1000, random_state=SEED,
        )
        clf.fit(Xtr, y_bin)
        clfs[target] = clf
        print(f"  trained specialist {target} (gamma={gamma}, pos={pos})")

    payload = {
        "mode": MODE,
        "all_classes": all_classes,
        "vectorizer": vec,
        "clfs": clfs,
        "gammas": {t: gammas[t] for t in clfs},
    }
    BOOST_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, OUT_PATH, compress=3)
    print(f"\nSaved: {OUT_PATH}")

    # ---- Verify: reproduce tuned val Macro-F1 --------------------------------
    verify(payload, all_classes)


def apply_boost(logits, vec, clfs, gammas, val_texts, all_classes):
    """Same math as tune_specialist_boosters: adjusted = logits + gamma*safe_logit(p)."""
    adjusted = logits.copy().astype(np.float64)
    Xva = vec.transform(val_texts)
    for target, clf in clfs.items():
        tid = all_classes.index(target)
        p = clf.predict_proba(Xva)[:, 1]
        s = tsb.safe_logit(p).astype(np.float32)
        s = np.clip(s, -8, 8)
        adjusted[:, tid] += gammas[target] * s
    return adjusted


def verify(payload, all_classes):
    print("\n===== Verification against tuned validation =====")
    val_df = pd.read_csv(MODEL_DIR / "val_predictions.csv")
    logits = np.load(MODEL_DIR / "val_logits.npy")
    y_true = val_df["true_id"].astype(int).values

    # Rebuild the router-mode val rows aligned to val_df order.
    rows = tsb.load_rows(DATA_DIR, text_mode=MODE)
    _train, val_rows_split = tsb.split_like_train(rows, seed=SEED)
    val_rows = tsb.align_val_rows(val_rows_split, val_df)
    val_texts = [r["text"] for r in val_rows]

    base_pred = logits.argmax(axis=1)
    base_f1 = f1_score(y_true, base_pred, labels=list(range(len(all_classes))),
                       average="macro", zero_division=0)

    adjusted = apply_boost(logits, payload["vectorizer"], payload["clfs"],
                           payload["gammas"], val_texts, all_classes)
    boost_pred = adjusted.argmax(axis=1)
    boost_f1 = f1_score(y_true, boost_pred, labels=list(range(len(all_classes))),
                        average="macro", zero_division=0)

    print(f"base  Macro-F1: {base_f1:.6f} | Acc: {accuracy_score(y_true, base_pred):.6f}")
    print(f"boost Macro-F1: {boost_f1:.6f} | Acc: {accuracy_score(y_true, boost_pred):.6f}")
    print(f"delta Macro-F1: {boost_f1 - base_f1:+.6f}")
    print(f"changed predictions: {int((boost_pred != base_pred).sum())}")
    expected = 0.7318280289889249
    if abs(boost_f1 - expected) < 1e-3:
        print(f"OK: reproduces tuned Macro-F1 (~{expected:.4f})")
    else:
        print(f"WARNING: expected ~{expected:.4f}, got {boost_f1:.4f}")


if __name__ == "__main__":
    main()
