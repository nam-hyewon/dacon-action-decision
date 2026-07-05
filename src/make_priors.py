import csv
import json
import math
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path("data")
OUT_DIR = Path("submit/model/deberta")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_labels(path):
    labels = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]
    return labels

def get_last_action(sample):
    history = sample.get("history", [])
    actions = [
        h.get("name", "")
        for h in history
        if h.get("role") == "assistant_action"
    ]
    return actions[-1] if actions else "none"

samples = load_jsonl(DATA_DIR / "train.jsonl")
labels = load_labels(DATA_DIR / "train_labels.csv")

# last_action -> next_action counts
counts = defaultdict(Counter)
global_counts = Counter()

for s in samples:
    y = labels[s["id"]]
    last_action = get_last_action(s)
    counts[last_action][y] += 1
    global_counts[y] += 1

# Laplace smoothing
alpha = 1.0
transition_logprob = {}

for last_action, counter in counts.items():
    total = sum(counter.values()) + alpha * len(ALL_CLASSES)
    probs = {}
    for cls in ALL_CLASSES:
        p = (counter[cls] + alpha) / total
        probs[cls] = math.log(p)
    transition_logprob[last_action] = probs

global_total = sum(global_counts.values()) + alpha * len(ALL_CLASSES)
global_logprob = {
    cls: math.log((global_counts[cls] + alpha) / global_total)
    for cls in ALL_CLASSES
}

prior = {
    "all_classes": ALL_CLASSES,
    "transition_logprob": transition_logprob,
    "global_logprob": global_logprob,
}

out_path = OUT_DIR / "transition_prior.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(prior, f, ensure_ascii=False, indent=2)

print("saved:", out_path)
print("last_action keys:", sorted(transition_logprob.keys()))