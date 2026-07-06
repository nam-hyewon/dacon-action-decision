#!/usr/bin/env python3
"""
Export the specialist boosters into a *version-independent* format so the
submission never has to unpickle sklearn objects on the eval server (which
runs scikit-learn 1.8.0, while these were fit with 1.5.1 -> cross-version
unpickling is unsupported and may break).

We extract only raw learned parameters:
  - TF-IDF vocabulary (term -> column) and idf weights
  - per-target LogisticRegression coef_ / intercept_
  - per-target gamma
and save them as .npz + .json. Inference (src/../submit/script.py) then
recomputes the TF-IDF transform + logistic decision in pure numpy, so it
depends on nothing but numpy/scipy (both preinstalled on the server).

This script also VERIFIES a numpy reimplementation reproduces:
  (a) sklearn's decision_function to ~1e-6, and
  (b) the tuned validation Macro-F1 (~0.7318).

Run:
  cd dacon-action-decision
  python src/export_specialists_npz.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import joblib
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score

import tune_specialist_boosters as tsb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "submit" / "model" / "roberta_eda_state4"
BOOST_DIR = MODEL_DIR / "specialist_boosters"
JOBLIB_PATH = BOOST_DIR / "specialists.joblib"
NPZ_PATH = BOOST_DIR / "specialists.npz"
VOCAB_PATH = BOOST_DIR / "specialists_vocab.json"

MODE = "router"
SEED = 42
EPS = 1e-5
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # sklearn default token_pattern


# --------------------------------------------------------------------------- #
# Pure-numpy TF-IDF transform (must match sklearn TfidfVectorizer settings:    #
# analyzer=word, ngram_range=(1,2), lowercase, sublinear_tf=True, norm="l2")   #
# --------------------------------------------------------------------------- #
def word_ngrams(tokens):
    grams = list(tokens)
    for i in range(len(tokens) - 1):
        grams.append(tokens[i] + " " + tokens[i + 1])
    return grams


def transform_tfidf(texts, vocab, idf):
    n = len(texts)
    n_feat = idf.shape[0]
    indptr = [0]
    indices = []
    data = []
    for text in texts:
        tokens = TOKEN_RE.findall(text.lower())
        counts = Counter(g for g in word_ngrams(tokens) if g in vocab)
        for term, cnt in counts.items():
            col = vocab[term]
            tf = 1.0 + np.log(cnt)          # sublinear_tf
            data.append(tf * idf[col])      # * idf
        indices.extend(vocab[t] for t in counts)
        indptr.append(len(indices))
    X = sparse.csr_matrix((data, indices, indptr), shape=(n, n_feat), dtype=np.float64)
    # l2 row normalization
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    X = X.multiply(1.0 / norms[:, None]).tocsr()
    return X


def safe_logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
def main():
    payload = joblib.load(JOBLIB_PATH)
    mode = payload["mode"]
    assert mode == MODE, mode
    all_classes = payload["all_classes"]
    vec = payload["vectorizer"]
    clfs = payload["clfs"]
    gammas = payload["gammas"]

    vocab = {t: int(i) for t, i in vec.vocabulary_.items()}
    idf = vec.idf_.astype(np.float64)
    n_feat = idf.shape[0]
    print(f"vocab={len(vocab)} n_feat={n_feat} targets={list(clfs)}")

    targets = list(clfs.keys())
    coefs = np.zeros((len(targets), n_feat), dtype=np.float64)
    intercepts = np.zeros(len(targets), dtype=np.float64)
    gvec = np.zeros(len(targets), dtype=np.float64)
    for k, t in enumerate(targets):
        clf = clfs[t]
        assert list(clf.classes_) == [0, 1], clf.classes_
        coefs[k] = clf.coef_.ravel()
        intercepts[k] = float(clf.intercept_[0])
        gvec[k] = float(gammas[t])

    # ---- Save version-independent artifacts --------------------------------
    np.savez_compressed(
        NPZ_PATH,
        idf=idf.astype(np.float32),
        coefs=coefs.astype(np.float32),
        intercepts=intercepts.astype(np.float32),
        gammas=gvec.astype(np.float32),
        targets=np.array(targets),
        all_classes=np.array(all_classes),
    )
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Saved: {NPZ_PATH}")
    print(f"Saved: {VOCAB_PATH}")

    # ---- Verify against sklearn on the validation split --------------------
    verify(vec, clfs, targets, coefs, intercepts, gvec, vocab, idf, all_classes)


def verify(vec, clfs, targets, coefs, intercepts, gvec, vocab, idf, all_classes):
    import pandas as pd
    print("\n===== Verify numpy reimpl vs sklearn + reproduce val F1 =====")
    val_df = pd.read_csv(MODEL_DIR / "val_predictions.csv")
    logits = np.load(MODEL_DIR / "val_logits.npy").astype(np.float64)
    y_true = val_df["true_id"].astype(int).values

    rows = tsb.load_rows(DATA_DIR, text_mode=MODE)
    _tr, val_rows_split = tsb.split_like_train(rows, seed=SEED)
    val_rows = tsb.align_val_rows(val_rows_split, val_df)
    val_texts = [r["text"] for r in val_rows]

    # sklearn reference decisions
    Xsk = vec.transform(val_texts)
    # numpy reimplementation
    Xnp = transform_tfidf(val_texts, vocab, idf)
    print("max |X_sklearn - X_numpy| =", abs(Xsk - Xnp).max())

    # apply booster both ways and compare F1
    def boost_with(decision_fn):
        adj = logits.copy()
        for k, t in enumerate(targets):
            tid = all_classes.index(t)
            d = decision_fn(k)
            p = expit(d)
            s = np.clip(safe_logit(p), -8, 8)
            adj[:, tid] += gvec[k] * s
        return adj

    adj_sk = boost_with(lambda k: clfs[targets[k]].decision_function(Xsk))
    adj_np = boost_with(lambda k: Xnp.dot(coefs[k]) + intercepts[k])

    for name, adj in [("sklearn", adj_sk), ("numpy", adj_np)]:
        pred = adj.argmax(axis=1)
        mf1 = f1_score(y_true, pred, labels=list(range(len(all_classes))),
                       average="macro", zero_division=0)
        acc = accuracy_score(y_true, pred)
        print(f"{name:8s} Macro-F1={mf1:.6f} Acc={acc:.6f}")

    expected = 0.7318280289889249
    pred_np = adj_np.argmax(axis=1)
    f1_np = f1_score(y_true, pred_np, labels=list(range(len(all_classes))),
                     average="macro", zero_division=0)
    print("OK" if abs(f1_np - expected) < 1e-3 else f"WARNING expected {expected}")


if __name__ == "__main__":
    main()
