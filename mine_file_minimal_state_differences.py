#!/usr/bin/env python3

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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

FILE_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]

FILE_SET = set(FILE_CLASSES)

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/klue_roberta_large_eda_state_fold0/"
    "val_predictions.csv"
)

EMB_DIR = Path(
    "large_file_latent_geometry_fold0"
)

OUT_DIR = Path(
    "file_minimal_state_difference_probe"
)

TOPK = 30
QUERY_CHUNK = 256
TRAIN_CHUNK = 8192


def clip(x, n=500):
    return str(x or "").replace("\n", " ")[:n]


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


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_users(sample):
    out = []

    for h in sample.get("history", []):
        if h.get("role") == "user":
            out.append(
                str(
                    h.get(
                        "content",
                        h.get("text", ""),
                    )
                )
            )

    return out


def arg_path(args):
    if not isinstance(args, dict):
        return ""

    for key in [
        "path",
        "file_path",
        "directory",
        "dir",
        "pattern",
        "query",
        "command",
    ]:
        if key in args:
            return str(args[key])

    return ""


def last_action_record(sample):
    actions = [
        h
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]

    if not actions:
        return {}

    return actions[-1]


def normalize_text(text):
    text = str(text or "").lower()

    text = re.sub(
        r"https?://\S+",
        "<URL>",
        text,
    )

    text = re.sub(
        r"\b[0-9a-f]{8,}\b",
        "<HASH>",
        text,
    )

    text = re.sub(
        r"(?<!\w)(?:/[\w.\-]+){2,}",
        "<PATH>",
        text,
    )

    text = re.sub(
        r"\b\d+\b",
        "<NUM>",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def token_set(text):
    return set(
        re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*|[가-힣]+",
            normalize_text(text),
        )
    )


def jaccard(a, b):
    a = token_set(a)
    b = token_set(b)

    if not a and not b:
        return 1.0

    return len(a & b) / max(
        1,
        len(a | b),
    )


def bool_cues(text):
    t = str(text or "").lower()

    patterns = {
        "cue_find": r"\bfind\b|찾",
        "cue_search": r"\bsearch\b|검색",
        "cue_locate": r"\blocat\w*\b|위치",
        "cue_where": r"\bwhere\b|어디",
        "cue_read": r"\bread\b|열어|읽",
        "cue_open": r"\bopen\b",
        "cue_list": r"\blist\b|목록",
        "cue_directory": r"\bdirector(?:y|ies)\b|폴더|디렉토리",
        "cue_file": r"\bfiles?\b|파일",
        "cue_pattern": r"\bpattern\b|패턴",
        "cue_glob": r"\bglob\b",
        "cue_definition": r"\bdefin\w*\b|정의",
        "cue_usage": r"\busages?\b|\breferences?\b|사용처|참조",
        "cue_exact": r"\bexact\b|정확",
        "cue_recursive": r"\brecursive\w*\b|재귀",
        "cue_exists": r"\bexist\w*\b|존재",
        "cue_error": r"\berror\b|\bfailed\b|\bfailure\b|오류|실패",
        "cue_found": r"\bfound\b|\bmatch(?:es)?\b|발견|찾았",
        "cue_empty": r"\bempty\b|\bno matches?\b|\bnot found\b|없",
    }

    return {
        name: int(
            re.search(pattern, t) is not None
        )
        for name, pattern in patterns.items()
    }


def state_features(sample):
    history = sample.get(
        "history",
        [],
    )

    actions = get_actions(sample)
    users = get_users(sample)

    last = last_action_record(sample)

    current_prompt = str(
        sample.get(
            "current_prompt",
            "",
        )
    )

    last_result = str(
        last.get(
            "result_summary",
            "",
        )
    )

    last_path = arg_path(
        last.get(
            "args",
            {},
        )
    )

    last_user = (
        users[-1]
        if users
        else ""
    )

    feat = {
        "last_action": (
            actions[-1]
            if len(actions) >= 1
            else "NONE"
        ),
        "prev_action": (
            actions[-2]
            if len(actions) >= 2
            else "NONE"
        ),
        "prev2_action": (
            actions[-3]
            if len(actions) >= 3
            else "NONE"
        ),
        "history_len": len(history),
        "action_count": len(actions),
        "prompt_len": len(current_prompt),
        "last_result_len": len(last_result),
        "last_path_ext": (
            Path(last_path).suffix.lower()
            if last_path
            else ""
        ),
        "last_path_depth": (
            last_path.count("/")
            if last_path
            else 0
        ),
        "prompt_norm": normalize_text(
            current_prompt
        ),
        "last_result_norm": normalize_text(
            last_result
        ),
        "last_user_norm": normalize_text(
            last_user
        ),
        "action_seq4": ">".join(
            actions[-4:]
        ),
        "action_seq8": ">".join(
            actions[-8:]
        ),
    }

    for prefix, text in [
        (
            "prompt",
            current_prompt,
        ),
        (
            "result",
            last_result,
        ),
        (
            "user",
            last_user,
        ),
    ]:
        for name, value in bool_cues(
            text
        ).items():
            feat[
                f"{prefix}_{name}"
            ] = value

    return feat


@torch.no_grad()
def cross_label_topk(
    emb,
    y,
    device,
):
    emb = np.asarray(
        emb,
        dtype=np.float32,
    )

    emb = emb / np.clip(
        np.linalg.norm(
            emb,
            axis=1,
            keepdims=True,
        ),
        1e-12,
        None,
    )

    emb_cpu = torch.from_numpy(
        emb
    )

    all_scores = []
    all_indices = []

    n = len(emb)

    for qs in range(
        0,
        n,
        QUERY_CHUNK,
    ):
        qe = min(
            qs + QUERY_CHUNK,
            n,
        )

        q = torch.from_numpy(
            emb[qs:qe]
        ).to(device)

        q_y = torch.from_numpy(
            y[qs:qe]
        ).to(device)

        cur_n = qe - qs

        best_score = torch.full(
            (
                cur_n,
                TOPK,
            ),
            -float("inf"),
            device=device,
        )

        best_idx = torch.full(
            (
                cur_n,
                TOPK,
            ),
            -1,
            dtype=torch.long,
            device=device,
        )

        for ts in range(
            0,
            n,
            TRAIN_CHUNK,
        ):
            te = min(
                ts + TRAIN_CHUNK,
                n,
            )

            t = emb_cpu[
                ts:te
            ].to(device)

            t_y = torch.from_numpy(
                y[ts:te]
            ).to(device)

            sim = q @ t.T

            # Only different true FILE labels.
            same_label = (
                q_y[:, None]
                == t_y[None, :]
            )

            sim = sim.masked_fill(
                same_label,
                -float("inf"),
            )

            local_k = min(
                TOPK,
                sim.shape[1],
            )

            score, idx = torch.topk(
                sim,
                k=local_k,
                dim=1,
            )

            idx += ts

            merged_score = torch.cat(
                [
                    best_score,
                    score,
                ],
                dim=1,
            )

            merged_idx = torch.cat(
                [
                    best_idx,
                    idx,
                ],
                dim=1,
            )

            best_score, pos = torch.topk(
                merged_score,
                k=TOPK,
                dim=1,
            )

            best_idx = torch.gather(
                merged_idx,
                1,
                pos,
            )

        all_scores.append(
            best_score.cpu().numpy()
        )

        all_indices.append(
            best_idx.cpu().numpy()
        )

        print(
            "search:",
            qe,
            "/",
            n,
        )

    return (
        np.concatenate(
            all_scores,
            axis=0,
        ),
        np.concatenate(
            all_indices,
            axis=0,
        ),
    )


def pair_diff(
    feat_a,
    feat_b,
):
    out = {}

    categorical = [
        "last_action",
        "prev_action",
        "prev2_action",
        "last_path_ext",
        "action_seq4",
        "action_seq8",
    ]

    numeric = [
        "history_len",
        "action_count",
        "prompt_len",
        "last_result_len",
        "last_path_depth",
    ]

    text_fields = [
        "prompt_norm",
        "last_result_norm",
        "last_user_norm",
    ]

    for key in categorical:
        out[
            f"same_{key}"
        ] = int(
            feat_a[key]
            == feat_b[key]
        )

    for key in numeric:
        out[
            f"absdiff_{key}"
        ] = abs(
            feat_a[key]
            - feat_b[key]
        )

    for key in text_fields:
        out[
            f"jaccard_{key}"
        ] = jaccard(
            feat_a[key],
            feat_b[key],
        )

    cue_keys = [
        key
        for key in feat_a
        if "_cue_" in key
    ]

    for key in cue_keys:
        a = feat_a[key]
        b = feat_b[key]

        out[
            f"diff_{key}"
        ] = int(
            a != b
        )

        out[
            f"a_only_{key}"
        ] = int(
            a == 1
            and b == 0
        )

        out[
            f"b_only_{key}"
        ] = int(
            b == 1
            and a == 0
        )

    return out


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples, label_map = load_data()

    val_ids = set(
        pd.read_csv(
            VAL_REF
        )["id"]
        .astype(str)
        .tolist()
    )

    file_samples = []

    file_ids = []

    y = []

    for sample in samples:
        sid = str(
            sample["id"]
        )

        if sid in val_ids:
            continue

        action = label_map[sid]

        if action not in FILE_SET:
            continue

        file_samples.append(
            sample
        )

        file_ids.append(
            sid
        )

        y.append(
            FILE_CLASSES.index(
                action
            )
        )

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    emb = np.load(
        EMB_DIR
        / "train_file_embeddings.npy"
    ).astype(np.float32)

    if len(emb) != len(y):
        raise ValueError(
            (
                "embedding/order mismatch "
                f"{len(emb)} vs {len(y)}"
            )
        )

    print(
        "FILE rows:",
        len(y),
    )

    print(
        "counts:",
        np.bincount(
            y,
            minlength=4,
        ),
    )

    print(
        "Extracting interpretable state features..."
    )

    features = [
        state_features(sample)
        for sample in file_samples
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
    )

    score_path = (
        OUT_DIR
        / "cross_label_scores.npy"
    )

    idx_path = (
        OUT_DIR
        / "cross_label_indices.npy"
    )

    if (
        score_path.exists()
        and idx_path.exists()
    ):
        print(
            "Loading cached cross-label neighbors..."
        )

        scores = np.load(
            score_path
        )

        indices = np.load(
            idx_path
        )

    else:
        scores, indices = (
            cross_label_topk(
                emb,
                y,
                device,
            )
        )

        np.save(
            score_path,
            scores,
        )

        np.save(
            idx_path,
            indices,
        )

    # ----------------------------------------------------------
    # Create unique nearest cross-label pair table.
    # ----------------------------------------------------------

    rows = []

    seen = set()

    for i in range(
        len(y)
    ):
        for rank in range(
            TOPK
        ):
            j = int(
                indices[i, rank]
            )

            if j < 0:
                continue

            key = tuple(
                sorted(
                    [
                        i,
                        j,
                    ]
                )
            )

            if key in seen:
                continue

            seen.add(key)

            label_a = FILE_CLASSES[
                y[i]
            ]

            label_b = FILE_CLASSES[
                y[j]
            ]

            diff = pair_diff(
                features[i],
                features[j],
            )

            row = {
                "idx_a": i,
                "idx_b": j,
                "id_a": file_ids[i],
                "id_b": file_ids[j],
                "label_a": label_a,
                "label_b": label_b,
                "pair": "->".join(
                    sorted(
                        [
                            label_a,
                            label_b,
                        ]
                    )
                ),
                "cosine": float(
                    scores[i, rank]
                ),
                "rank_from_a": rank,
                **diff,
            }

            rows.append(row)

    pair_df = pd.DataFrame(
        rows
    )

    pair_df.to_csv(
        OUT_DIR
        / "all_cross_label_pairs.csv",
        index=False,
    )

    print()
    print("=" * 150)
    print("PAIR SIMILARITY DISTRIBUTION")
    print("=" * 150)

    sim_rows = []

    for pair, g in pair_df.groupby(
        "pair"
    ):
        row = {
            "pair": pair,
            "n": len(g),
            "cos_mean": g["cosine"].mean(),
            "cos_p50": g["cosine"].median(),
            "cos_p90": np.percentile(
                g["cosine"],
                90,
            ),
            "cos_p95": np.percentile(
                g["cosine"],
                95,
            ),
            "cos_p99": np.percentile(
                g["cosine"],
                99,
            ),
        }

        sim_rows.append(row)

    sim_df = pd.DataFrame(
        sim_rows
    ).sort_values(
        "cos_p95",
        ascending=False,
    )

    print(
        sim_df.to_string(
            index=False,
        )
    )

    sim_df.to_csv(
        OUT_DIR
        / "pair_similarity_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Ultra-near pairs: inspect which features still differ.
    # Pair-relative threshold avoids glob dominating.
    # ----------------------------------------------------------

    ultra_parts = []

    for pair, g in pair_df.groupby(
        "pair"
    ):
        threshold = np.percentile(
            g["cosine"],
            99,
        )

        ultra_parts.append(
            g[
                g["cosine"]
                >= threshold
            ]
        )

    ultra = pd.concat(
        ultra_parts,
        ignore_index=True,
    )

    print()
    print(
        "Ultra-near pairs:",
        len(ultra),
    )

    diff_cols = [
        c
        for c in ultra.columns
        if (
            c.startswith(
                "same_"
            )
            or c.startswith(
                "absdiff_"
            )
            or c.startswith(
                "jaccard_"
            )
            or c.startswith(
                "diff_"
            )
        )
    ]

    feature_rows = []

    for pair, g in ultra.groupby(
        "pair"
    ):
        for col in diff_cols:
            values = g[col]

            if col.startswith(
                "same_"
            ):
                stat = 1.0 - values.mean()
                meaning = "different_rate"

            elif col.startswith(
                "jaccard_"
            ):
                stat = 1.0 - values.mean()
                meaning = "1_minus_jaccard"

            else:
                stat = values.mean()
                meaning = "mean"

            feature_rows.append({
                "pair": pair,
                "feature": col,
                "stat": float(stat),
                "meaning": meaning,
                "n": len(g),
            })

    feature_df = pd.DataFrame(
        feature_rows
    )

    feature_df.to_csv(
        OUT_DIR
        / "ultra_near_feature_differences.csv",
        index=False,
    )

    print()
    print("=" * 170)
    print("TOP DIFFERENCES INSIDE ULTRA-NEAR PAIRS")
    print("=" * 170)

    for pair in (
        sim_df["pair"].tolist()
    ):
        g = feature_df[
            feature_df["pair"]
            == pair
        ].sort_values(
            "stat",
            ascending=False,
        )

        print()
        print("-" * 130)
        print(pair)
        print("-" * 130)

        print(
            g.head(20).to_string(
                index=False,
            )
        )

    # ----------------------------------------------------------
    # Directional cue analysis.
    # For each label pair, orient low lexical label name order A/B
    # consistently and count cues present on only one side.
    # ----------------------------------------------------------

    directional_rows = []

    for pair, g in ultra.groupby(
        "pair"
    ):
        label_left, label_right = (
            pair.split("->")
        )

        cue_counter = Counter()

        pair_n = 0

        for _, row in g.iterrows():
            i = int(row["idx_a"])
            j = int(row["idx_b"])

            if (
                FILE_CLASSES[y[i]]
                == label_left
            ):
                left_i = i
                right_i = j
            else:
                left_i = j
                right_i = i

            left_feat = features[
                left_i
            ]

            right_feat = features[
                right_i
            ]

            cue_keys = [
                key
                for key in left_feat
                if "_cue_" in key
            ]

            for key in cue_keys:
                lv = left_feat[key]
                rv = right_feat[key]

                if (
                    lv == 1
                    and rv == 0
                ):
                    cue_counter[
                        f"{label_left}_ONLY::{key}"
                    ] += 1

                elif (
                    rv == 1
                    and lv == 0
                ):
                    cue_counter[
                        f"{label_right}_ONLY::{key}"
                    ] += 1

            pair_n += 1

        for cue, count in (
            cue_counter.most_common()
        ):
            directional_rows.append({
                "pair": pair,
                "cue": cue,
                "count": count,
                "pair_n": pair_n,
                "rate": count / max(
                    1,
                    pair_n,
                ),
            })

    directional_df = pd.DataFrame(
        directional_rows
    )

    directional_df.to_csv(
        OUT_DIR
        / "directional_cue_differences.csv",
        index=False,
    )

    print()
    print("=" * 170)
    print("DIRECTIONAL CUES")
    print("=" * 170)

    for pair in (
        sim_df["pair"].tolist()
    ):
        g = directional_df[
            directional_df["pair"]
            == pair
        ].sort_values(
            [
                "rate",
                "count",
            ],
            ascending=False,
        )

        print()
        print("-" * 130)
        print(pair)
        print("-" * 130)

        print(
            g.head(25).to_string(
                index=False,
            )
        )

    # ----------------------------------------------------------
    # Export actual closest pairs to JSONL for manual reading.
    # ----------------------------------------------------------

    closest = pair_df.sort_values(
        "cosine",
        ascending=False,
    )

    with open(
        OUT_DIR
        / "closest_pairs.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        pair_counts = Counter()

        for _, row in (
            closest.iterrows()
        ):
            pair = row["pair"]

            if pair_counts[pair] >= 50:
                continue

            i = int(
                row["idx_a"]
            )

            j = int(
                row["idx_b"]
            )

            obj = {
                "pair": pair,
                "cosine": float(
                    row["cosine"]
                ),
                "id_a": file_ids[i],
                "label_a": FILE_CLASSES[
                    y[i]
                ],
                "state_a": features[i],
                "current_prompt_a": clip(
                    file_samples[i].get(
                        "current_prompt",
                        "",
                    ),
                    3000,
                ),
                "last_history_a": (
                    file_samples[i]
                    .get(
                        "history",
                        [],
                    )[-8:]
                ),
                "id_b": file_ids[j],
                "label_b": FILE_CLASSES[
                    y[j]
                ],
                "state_b": features[j],
                "current_prompt_b": clip(
                    file_samples[j].get(
                        "current_prompt",
                        "",
                    ),
                    3000,
                ),
                "last_history_b": (
                    file_samples[j]
                    .get(
                        "history",
                        [],
                    )[-8:]
                ),
            }

            f.write(
                json.dumps(
                    obj,
                    ensure_ascii=False,
                )
                + "\n"
            )

            pair_counts[pair] += 1

            if all(
                pair_counts[p]
                >= 50
                for p in sim_df["pair"]
            ):
                break

    ultra.to_csv(
        OUT_DIR
        / "ultra_near_pairs.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print(
        "all pairs:",
        len(pair_df),
    )

    print(
        "ultra-near pairs:",
        len(ultra),
    )

    print(
        "saved:",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
