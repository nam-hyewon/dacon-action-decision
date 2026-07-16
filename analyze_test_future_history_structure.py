#!/usr/bin/env python3

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
CLASS_SET = set(CLASSES)


def parse_id(sample_id):
    m = re.match(
        r"^(.*)-step_(\d+)$",
        str(sample_id),
    )

    if m is None:
        raise ValueError(f"bad id: {sample_id}")

    return m.group(1), int(m.group(2))


def assistant_actions(sample):
    out = []

    for h in sample.get("history", []):
        if h.get("role") != "user":
            name = str(h.get("name", ""))

            if name in CLASS_SET:
                out.append(name)

    return out


print("Loading test...")

samples = []

with open("data/test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line:
            samples.append(json.loads(line))


sample_by_id = {
    str(x["id"]): x
    for x in samples
}

session_rows = defaultdict(list)

for pos, sample in enumerate(samples):
    sid = str(sample["id"])
    session, step = parse_id(sid)

    session_rows[session].append(
        (step, pos, sid)
    )


for session in session_rows:
    session_rows[session].sort()


sizes = np.array([
    len(rows)
    for rows in session_rows.values()
])


print()
print("=" * 100)
print("TEST SESSION STRUCTURE")
print("=" * 100)

print("rows:", len(samples))
print("sessions:", len(session_rows))
print(pd.Series(sizes).describe().to_string())

print(
    "multi-row sessions:",
    int((sizes >= 2).sum()),
)

print(
    "rows in multi-row sessions:",
    sum(
        len(rows)
        for rows in session_rows.values()
        if len(rows) >= 2
    ),
)


rows_out = []

for session, rows in session_rows.items():

    for j in range(len(rows) - 1):
        step_t, pos_t, id_t = rows[j]
        step_n, pos_n, id_n = rows[j + 1]

        sample_t = sample_by_id[id_t]
        sample_n = sample_by_id[id_n]

        actions_t = assistant_actions(sample_t)
        actions_n = assistant_actions(sample_n)

        prefix_match = (
            len(actions_n) >= len(actions_t)
            and actions_n[:len(actions_t)] == actions_t
        )

        appended = (
            actions_n[len(actions_t):]
            if prefix_match
            else []
        )

        rows_out.append({
            "session_id": session,
            "step_t": step_t,
            "step_next": step_n,
            "gap": step_n - step_t,
            "pos_t": pos_t,
            "id_t": id_t,
            "id_next": id_n,
            "history_len_t": len(actions_t),
            "history_len_next": len(actions_n),
            "prefix_match": prefix_match,
            "append_n": len(appended),
            "first_appended": (
                appended[0]
                if appended
                else None
            ),
            "next_last_action": (
                actions_n[-1]
                if actions_n
                else None
            ),
        })


df = pd.DataFrame(rows_out)

strict = (
    df["prefix_match"]
    & (df["append_n"] == 1)
    & df["first_appended"].isin(CLASSES)
)


print()
print("=" * 100)
print("TEST FUTURE-HISTORY STRUCTURE")
print("=" * 100)

print("adjacent row pairs:", len(df))

print()
print("step gap:")
print(
    df["gap"]
    .value_counts()
    .sort_index()
    .head(20)
    .to_string()
)

print()

print(
    "prefix-match:",
    int(df["prefix_match"].sum()),
    "/",
    len(df),
    "=",
    float(df["prefix_match"].mean()),
)

print(
    "append_n == 1:",
    int((df["append_n"] == 1).sum()),
    "/",
    len(df),
    "=",
    float((df["append_n"] == 1).mean()),
)

print(
    "STRICT REVELATION ROWS:",
    int(strict.sum()),
    "/",
    len(samples),
    "=",
    float(strict.sum() / len(samples)),
)

print()

print("strict first_appended distribution:")
print(
    df.loc[strict, "first_appended"]
    .value_counts()
    .reindex(CLASSES, fill_value=0)
    .to_string()
)


# uniqueness safety
strict_df = df[strict].copy()

dup = (
    strict_df["pos_t"]
    .value_counts()
)

print()
print(
    "rows with multiple strict assignments:",
    int((dup > 1).sum()),
)

strict_df.to_csv(
    "test_future_history_strict_assignments.csv",
    index=False,
)

print()
print("Saved test_future_history_strict_assignments.csv")
