#!/usr/bin/env python3

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


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

OUT_DIR = Path("file_strict_raw_state_pairs")

TOPK = 50


def load_data():
    samples = []

    with open("data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    with open("data/train_labels.csv", encoding="utf-8") as f:
        label_map = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(f)
        }

    return samples, label_map


def normalize_text(text):
    text = str(text or "").lower()

    text = re.sub(r"https?://\S+", "<URL>", text)
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<HASH>", text)
    text = re.sub(
        r"(?<!\w)(?:/[\w.\-]+){2,}",
        "<PATH>",
        text,
    )
    text = re.sub(r"\b\d+\b", "<NUM>", text)
    text = re.sub(
        r"""(["'`])[^"'`]{1,120}\1""",
        "<QUOTED>",
        text,
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_actions(sample):
    return [
        str(h.get("name", ""))
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def get_action_records(sample):
    return [
        h
        for h in sample.get("history", [])
        if h.get("role") == "assistant_action"
    ]


def arg_value(args, keys):
    if not isinstance(args, dict):
        return ""

    for key in keys:
        if key in args:
            return str(args[key])

    return ""


def get_state(sample):
    history = sample.get("history", [])
    actions = get_actions(sample)
    records = get_action_records(sample)

    last = records[-1] if records else {}

    args = last.get("args", {})

    current_prompt = str(
        sample.get("current_prompt", "")
    )

    result = str(
        last.get("result_summary", "")
    )

    path = arg_value(
        args,
        [
            "path",
            "file_path",
            "directory",
            "dir",
        ],
    )

    query = arg_value(
        args,
        [
            "query",
            "pattern",
            "regex",
            "glob",
        ],
    )

    return {
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
        "action_seq4": ">".join(actions[-4:]),
        "action_seq8": ">".join(actions[-8:]),
        "history_len": len(history),
        "action_count": len(actions),
        "prompt": current_prompt,
        "prompt_norm": normalize_text(current_prompt),
        "result": result,
        "result_norm": normalize_text(result),
        "path": path,
        "path_norm": normalize_text(path),
        "query": query,
        "query_norm": normalize_text(query),
        "path_ext": (
            Path(path).suffix.lower()
            if path
            else ""
        ),
        "path_depth": (
            path.count("/")
            if path
            else 0
        ),
    }


def token_set(text):
    return set(
        re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*|[가-힣]+",
            text,
        )
    )


def jaccard(a, b):
    a = token_set(a)
    b = token_set(b)

    if not a and not b:
        return 1.0

    return len(a & b) / max(1, len(a | b))


def bool_cues(text):
    text = str(text or "").lower()

    patterns = {
        "find": r"\bfind\b|찾",
        "search": r"\bsearch\b|검색",
        "locate": r"\blocat\w*\b|위치",
        "where": r"\bwhere\b|어디",
        "read": r"\bread\b|읽|열어",
        "open": r"\bopen\b",
        "list": r"\blist\b|목록",
        "directory": r"\bdirector(?:y|ies)\b|폴더|디렉토리",
        "file": r"\bfiles?\b|파일",
        "pattern": r"\bpattern\b|패턴",
        "glob": r"\bglob\b",
        "definition": r"\bdefin\w*\b|정의",
        "usage": r"\busages?\b|\breferences?\b|사용처|참조",
        "exact": r"\bexact\b|정확",
        "recursive": r"\brecursive\w*\b|재귀",
        "exists": r"\bexist\w*\b|존재",
        "error": r"\berror\b|\bfailed\b|\bfailure\b|오류|실패",
        "found": r"\bfound\b|\bmatches?\b|발견|찾았",
        "empty": r"\bempty\b|\bno matches?\b|\bnot found\b|없",
        "symbol": r"\bsymbol\b|함수|클래스|method|variable",
        "content": r"\bcontent\b|내용",
        "tree": r"\btree\b|구조",
    }

    return {
        key: int(re.search(pattern, text) is not None)
        for key, pattern in patterns.items()
    }


def make_bucket_key(state):
    return (
        state["last_action"],
        state["prev_action"],
        state["action_seq4"],
    )


def directional_stats(pair_df, states):
    rows = []

    for pair, g in pair_df.groupby("pair"):
        left_label, right_label = pair.split("->")

        counter = Counter()

        for _, row in g.iterrows():
            ia = int(row["idx_a"])
            ib = int(row["idx_b"])

            if row["label_a"] == left_label:
                il, ir = ia, ib
            else:
                il, ir = ib, ia

            left = states[il]
            right = states[ir]

            for prefix, field in [
                ("prompt", "prompt"),
                ("result", "result"),
            ]:
                lc = bool_cues(left[field])
                rc = bool_cues(right[field])

                for cue in lc:
                    if lc[cue] and not rc[cue]:
                        counter[
                            f"{left_label}_ONLY::{prefix}_{cue}"
                        ] += 1

                    if rc[cue] and not lc[cue]:
                        counter[
                            f"{right_label}_ONLY::{prefix}_{cue}"
                        ] += 1

            for field in [
                "path_ext",
                "last_action",
                "prev_action",
                "prev2_action",
            ]:
                if left[field] != right[field]:
                    counter[
                        f"DIFF::{field}"
                    ] += 1

            if left["result_norm"] != right["result_norm"]:
                counter["DIFF::result_norm"] += 1

            if left["path_norm"] != right["path_norm"]:
                counter["DIFF::path_norm"] += 1

            if left["query_norm"] != right["query_norm"]:
                counter["DIFF::query_norm"] += 1

        n = len(g)

        for name, count in counter.most_common():
            rows.append({
                "pair": pair,
                "feature": name,
                "count": count,
                "n": n,
                "rate": count / max(1, n),
            })

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples, label_map = load_data()

    val_ids = set(
        pd.read_csv(VAL_REF)["id"]
        .astype(str)
        .tolist()
    )

    file_samples = []
    file_ids = []
    labels = []

    for sample in samples:
        sid = str(sample["id"])

        if sid in val_ids:
            continue

        label = label_map[sid]

        if label not in FILE_SET:
            continue

        file_samples.append(sample)
        file_ids.append(sid)
        labels.append(label)

    states = [
        get_state(sample)
        for sample in file_samples
    ]

    print("FILE rows:", len(states))
    print("label counts:", Counter(labels))

    # ------------------------------------------------------------
    # Bucket by strict trajectory/action state.
    # ------------------------------------------------------------

    buckets = {}

    for i, state in enumerate(states):
        key = make_bucket_key(state)
        buckets.setdefault(key, []).append(i)

    print("strict state buckets:", len(buckets))

    candidate_indices = set()

    bucket_sizes = []

    for idxs in buckets.values():
        label_set = {
            labels[i]
            for i in idxs
        }

        if len(label_set) < 2:
            continue

        if len(idxs) < 2:
            continue

        candidate_indices.update(idxs)
        bucket_sizes.append(len(idxs))

    candidate_indices = sorted(candidate_indices)

    print(
        "cross-label strict-bucket rows:",
        len(candidate_indices),
    )

    print(
        "cross-label strict buckets:",
        len(bucket_sizes),
    )

    if bucket_sizes:
        print(
            "bucket size mean/p50/p90/max:",
            float(np.mean(bucket_sizes)),
            float(np.percentile(bucket_sizes, 50)),
            float(np.percentile(bucket_sizes, 90)),
            int(np.max(bucket_sizes)),
        )

    # ------------------------------------------------------------
    # TF-IDF only for prompt similarity inside same strict bucket.
    # ------------------------------------------------------------

    prompts = [
        states[i]["prompt_norm"]
        for i in candidate_indices
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    x = vectorizer.fit_transform(prompts)

    global_to_local = {
        global_i: local_i
        for local_i, global_i in enumerate(candidate_indices)
    }

    pair_rows = []

    seen = set()

    for bucket_num, idxs in enumerate(buckets.values()):
        bucket_labels = {
            labels[i]
            for i in idxs
        }

        if len(bucket_labels) < 2:
            continue

        usable = [
            i
            for i in idxs
            if i in global_to_local
        ]

        if len(usable) < 2:
            continue

        local = [
            global_to_local[i]
            for i in usable
        ]

        xb = x[local]

        nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(
                TOPK + 1,
                len(usable),
            ),
            n_jobs=-1,
        )

        nn.fit(xb)

        distances, neighbors = nn.kneighbors(xb)

        for source_pos, i in enumerate(usable):
            for rank in range(1, neighbors.shape[1]):
                target_pos = int(
                    neighbors[source_pos, rank]
                )

                j = usable[target_pos]

                if labels[i] == labels[j]:
                    continue

                key = tuple(sorted((i, j)))

                if key in seen:
                    continue

                seen.add(key)

                prompt_cos = 1.0 - float(
                    distances[source_pos, rank]
                )

                prompt_jac = jaccard(
                    states[i]["prompt_norm"],
                    states[j]["prompt_norm"],
                )

                row = {
                    "idx_a": i,
                    "idx_b": j,
                    "id_a": file_ids[i],
                    "id_b": file_ids[j],
                    "label_a": labels[i],
                    "label_b": labels[j],
                    "pair": "->".join(
                        sorted(
                            [
                                labels[i],
                                labels[j],
                            ]
                        )
                    ),
                    "prompt_cos": prompt_cos,
                    "prompt_jaccard": prompt_jac,
                    "same_prompt_norm": int(
                        states[i]["prompt_norm"]
                        == states[j]["prompt_norm"]
                    ),
                    "history_len_diff": abs(
                        states[i]["history_len"]
                        - states[j]["history_len"]
                    ),
                    "action_count_diff": abs(
                        states[i]["action_count"]
                        - states[j]["action_count"]
                    ),
                    "same_prev2_action": int(
                        states[i]["prev2_action"]
                        == states[j]["prev2_action"]
                    ),
                    "same_result_norm": int(
                        states[i]["result_norm"]
                        == states[j]["result_norm"]
                    ),
                    "result_jaccard": jaccard(
                        states[i]["result_norm"],
                        states[j]["result_norm"],
                    ),
                    "same_path_norm": int(
                        states[i]["path_norm"]
                        == states[j]["path_norm"]
                    ),
                    "same_query_norm": int(
                        states[i]["query_norm"]
                        == states[j]["query_norm"]
                    ),
                    "same_path_ext": int(
                        states[i]["path_ext"]
                        == states[j]["path_ext"]
                    ),
                    "path_depth_diff": abs(
                        states[i]["path_depth"]
                        - states[j]["path_depth"]
                    ),
                }

                pair_rows.append(row)

        if bucket_num % 1000 == 0:
            print(
                "bucket:",
                bucket_num,
                "| pairs:",
                len(pair_rows),
            )

    pair_df = pd.DataFrame(pair_rows)

    if len(pair_df) == 0:
        print("NO CROSS-LABEL RAW-STATE PAIRS")
        return

    pair_df.to_csv(
        OUT_DIR / "all_raw_state_pairs.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Threshold diagnostics.
    # ------------------------------------------------------------

    thresholds = [
        ("cos070", pair_df["prompt_cos"] >= 0.70),
        ("cos080", pair_df["prompt_cos"] >= 0.80),
        ("cos090", pair_df["prompt_cos"] >= 0.90),
        ("cos095", pair_df["prompt_cos"] >= 0.95),
        (
            "jaccard080",
            pair_df["prompt_jaccard"] >= 0.80,
        ),
        (
            "jaccard090",
            pair_df["prompt_jaccard"] >= 0.90,
        ),
        (
            "exact_prompt",
            pair_df["same_prompt_norm"] == 1,
        ),
    ]

    summary_rows = []

    print()
    print("=" * 150)
    print("STRICT RAW-STATE PAIR COUNTS")
    print("=" * 150)

    for threshold_name, mask in thresholds:
        cur = pair_df[mask]

        print()
        print("###", threshold_name)
        print("n:", len(cur))

        if len(cur) == 0:
            continue

        counts = (
            cur["pair"]
            .value_counts()
            .rename_axis("pair")
            .reset_index(name="n")
        )

        print(
            counts.to_string(index=False)
        )

        for pair, g in cur.groupby("pair"):
            summary_rows.append({
                "threshold": threshold_name,
                "pair": pair,
                "n": len(g),
                "prompt_cos_mean": g[
                    "prompt_cos"
                ].mean(),
                "prompt_jaccard_mean": g[
                    "prompt_jaccard"
                ].mean(),
                "same_result_rate": g[
                    "same_result_norm"
                ].mean(),
                "result_jaccard_mean": g[
                    "result_jaccard"
                ].mean(),
                "same_path_rate": g[
                    "same_path_norm"
                ].mean(),
                "same_query_rate": g[
                    "same_query_norm"
                ].mean(),
                "same_prev2_rate": g[
                    "same_prev2_action"
                ].mean(),
                "same_path_ext_rate": g[
                    "same_path_ext"
                ].mean(),
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUT_DIR / "threshold_pair_summary.csv",
        index=False,
    )

    print()
    print("=" * 180)
    print("THRESHOLD PAIR SUMMARY")
    print("=" * 180)

    print(
        summary_df.sort_values(
            ["threshold", "n"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    # ------------------------------------------------------------
    # Use the strictest threshold with enough pairs.
    # ------------------------------------------------------------

    selected_name = None
    selected = None

    for threshold_name, mask in reversed(thresholds):
        cur = pair_df[mask]

        if len(cur) >= 100:
            selected_name = threshold_name
            selected = cur.copy()
            break

    if selected is None:
        selected_name = "all"
        selected = pair_df.copy()

    print()
    print("=" * 120)
    print("SELECTED FOR MINIMAL-DIFF ANALYSIS")
    print("=" * 120)

    print("threshold:", selected_name)
    print("n:", len(selected))

    directional_df = directional_stats(
        selected,
        states,
    )

    directional_df.to_csv(
        OUT_DIR / "directional_minimal_differences.csv",
        index=False,
    )

    print()
    print("=" * 180)
    print("DIRECTIONAL MINIMAL DIFFERENCES")
    print("=" * 180)

    for pair in sorted(selected["pair"].unique()):
        g = directional_df[
            directional_df["pair"] == pair
        ].sort_values(
            ["rate", "count"],
            ascending=False,
        )

        print()
        print("-" * 140)
        print(pair)
        print("-" * 140)

        print(
            g.head(30).to_string(index=False)
        )

    # ------------------------------------------------------------
    # Export actual pairs.
    # ------------------------------------------------------------

    selected = selected.sort_values(
        [
            "prompt_cos",
            "prompt_jaccard",
            "result_jaccard",
        ],
        ascending=False,
    )

    pair_counter = Counter()

    with open(
        OUT_DIR / "closest_raw_pairs.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for _, row in selected.iterrows():
            pair = row["pair"]

            if pair_counter[pair] >= 100:
                continue

            ia = int(row["idx_a"])
            ib = int(row["idx_b"])

            obj = {
                "pair": pair,
                "prompt_cos": float(
                    row["prompt_cos"]
                ),
                "prompt_jaccard": float(
                    row["prompt_jaccard"]
                ),
                "id_a": file_ids[ia],
                "label_a": labels[ia],
                "state_a": states[ia],
                "history_a": file_samples[ia]
                .get("history", [])[-10:],
                "id_b": file_ids[ib],
                "label_b": labels[ib],
                "state_b": states[ib],
                "history_b": file_samples[ib]
                .get("history", [])[-10:],
            }

            f.write(
                json.dumps(
                    obj,
                    ensure_ascii=False,
                )
                + "\n"
            )

            pair_counter[pair] += 1

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print("all raw pairs:", len(pair_df))
    print("selected threshold:", selected_name)
    print("selected pairs:", len(selected))
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    main()
