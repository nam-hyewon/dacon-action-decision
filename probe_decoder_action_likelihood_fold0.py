#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


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

C2I = {
    name: i
    for i, name in enumerate(CLASSES)
}


GROUPS = {
    "FILE": [
        "read_file",
        "grep_search",
        "list_directory",
        "glob_pattern",
    ],
    "WRITE": [
        "edit_file",
        "write_file",
        "apply_patch",
    ],
    "EXEC": [
        "run_bash",
        "run_tests",
        "lint_or_typecheck",
    ],
    "CONTROL": [
        "ask_user",
        "plan_task",
        "web_search",
        "respond_only",
    ],
}

GROUP_NAMES = list(GROUPS)


CLASS_TO_GROUP = {}

for group_idx, group_name in enumerate(GROUP_NAMES):
    for action in GROUPS[group_name]:
        CLASS_TO_GROUP[C2I[action]] = group_idx


CLASS_TO_GROUP_ARR = np.asarray([
    CLASS_TO_GROUP[i]
    for i in range(len(CLASSES))
])


REPRO_ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

VAL_PRED_PATH = (
    REPRO_ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)

SOTA_SCRIPT = Path(
    "submit/script_large_tab_diet_rob8_gated40.py"
)


ACTION_GUIDE = """
Allowed next actions:

read_file:
Inspect the contents of a specific known file.

grep_search:
Search text or code contents for a symbol, keyword, string, or pattern.

list_directory:
Inspect the entries and structure of a known directory.

glob_pattern:
Find file paths matching a filename or path pattern.

edit_file:
Make a targeted modification to an existing file.

write_file:
Create or write complete file contents.

apply_patch:
Apply structured patch or diff changes.

run_bash:
Execute a general shell command or program.

run_tests:
Run behavioral tests or a test suite.

lint_or_typecheck:
Run static linting, diagnostics, or type checking.

ask_user:
Ask the user for missing information or a required decision.

plan_task:
Create or update a multi-step execution plan.

web_search:
Search external public web information.

respond_only:
Respond directly without another tool action.
""".strip()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        required=True,
    )

    parser.add_argument(
        "--subset-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-state-tokens",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=43,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


def macro_f1(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )


def load_samples():
    samples = []

    with open(
        "data/train.jsonl",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                samples.append(
                    json.loads(line)
                )

    return samples


def load_sota_module():
    spec = importlib.util.spec_from_file_location(
        "sota",
        SOTA_SCRIPT,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def make_subset(
    y,
    base_pred,
    subset_size,
    seed,
):
    """
    Stratify by true class and base correctness.

    This prevents a random subset from accidentally containing
    almost no difficult within-group errors.
    """
    rng = np.random.default_rng(seed)

    strata = {}

    for idx in range(len(y)):
        key = (
            int(y[idx]),
            int(base_pred[idx] == y[idx]),
        )

        strata.setdefault(
            key,
            [],
        ).append(idx)


    selected = []

    total = len(y)

    for key, indices in strata.items():
        indices = np.asarray(indices)

        n = max(
            1,
            round(
                subset_size
                * len(indices)
                / total
            ),
        )

        n = min(
            n,
            len(indices),
        )

        picked = rng.choice(
            indices,
            size=n,
            replace=False,
        )

        selected.extend(
            picked.tolist()
        )


    selected = list(dict.fromkeys(selected))


    if len(selected) > subset_size:
        selected = rng.choice(
            np.asarray(selected),
            size=subset_size,
            replace=False,
        ).tolist()

    elif len(selected) < subset_size:
        remaining = np.setdiff1d(
            np.arange(total),
            np.asarray(selected),
        )

        extra = rng.choice(
            remaining,
            size=subset_size - len(selected),
            replace=False,
        )

        selected.extend(
            extra.tolist()
        )


    selected = np.asarray(
        sorted(selected),
        dtype=np.int64,
    )

    return selected


def build_user_prompt(
    state_text,
    group_name,
):
    candidates = GROUPS[group_name]

    candidate_text = "\n".join(
        f"- {x}"
        for x in candidates
    )

    return f"""
You are predicting the NEXT tool action of an AI coding agent.

Use the current user request, workspace state, and previous action history.

{ACTION_GUIDE}

The action group has already been determined correctly.

Only choose one of these candidate actions:
{candidate_text}

[CURRENT AGENT STATE]
{state_text}

Predict the single most appropriate NEXT action.

Answer with exactly one action name and nothing else.
""".strip()


def make_prefix(
    tokenizer,
    user_prompt,
):
    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    if hasattr(
        tokenizer,
        "apply_chat_template",
    ) and tokenizer.chat_template:

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        user_prompt
        + "\n\nAssistant:\n"
    )


def truncate_prefix(
    tokenizer,
    prefix,
    max_state_tokens,
):
    ids = tokenizer(
        prefix,
        add_special_tokens=False,
    )["input_ids"]

    if len(ids) <= max_state_tokens:
        return ids

    # Keep the front because diet places prompt/state first and
    # compact history is already compressed.
    return ids[:max_state_tokens]


@torch.no_grad()
def score_batch(
    model,
    tokenizer,
    batch_pairs,
    device,
    max_state_tokens,
):
    sequences = []
    candidate_masks = []


    for prefix, candidate in batch_pairs:

        prefix_ids = truncate_prefix(
            tokenizer,
            prefix,
            max_state_tokens,
        )

        candidate_ids = tokenizer(
            candidate,
            add_special_tokens=False,
        )["input_ids"]

        if not candidate_ids:
            raise ValueError(
                f"empty candidate tokenization: {candidate}"
            )

        seq = (
            prefix_ids
            + candidate_ids
        )

        mask = (
            [0] * len(prefix_ids)
            + [1] * len(candidate_ids)
        )

        sequences.append(seq)
        candidate_masks.append(mask)


    max_len = max(
        len(x)
        for x in sequences
    )

    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        pad_id = tokenizer.eos_token_id


    input_ids = []
    attention_mask = []
    score_mask = []


    for seq, mask in zip(
        sequences,
        candidate_masks,
    ):
        pad_n = max_len - len(seq)

        input_ids.append(
            seq
            + [pad_id] * pad_n
        )

        attention_mask.append(
            [1] * len(seq)
            + [0] * pad_n
        )

        score_mask.append(
            mask
            + [0] * pad_n
        )


    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        attention_mask,
        dtype=torch.long,
        device=device,
    )

    score_mask = torch.tensor(
        score_mask,
        dtype=torch.bool,
        device=device,
    )


    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


    # logits[:, t] predicts input_ids[:, t + 1].
    shift_logits = logits[:, :-1, :].float()

    shift_labels = input_ids[:, 1:]

    shift_mask = score_mask[:, 1:]


    token_logp = torch.log_softmax(
        shift_logits,
        dim=-1,
    ).gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)


    scores = (
        (
            token_logp
            * shift_mask
        ).sum(dim=1)
        / shift_mask.sum(dim=1).clamp_min(1)
    )


    return scores.cpu().numpy()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("DEVICE:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )


    print("MODEL:", args.model_id)


    print("Loading validation reference...")

    val_df = pd.read_csv(
        VAL_PRED_PATH
    )

    val_ids = (
        val_df["id"]
        .astype(str)
        .values
    )

    y = (
        val_df["true_id"]
        .astype(int)
        .values
    )


    base_pred = np.load(
        "exact_repro_base_pred.npy"
    ).astype(np.int64)


    if len(base_pred) != len(y):
        raise ValueError(
            (
                "base prediction length mismatch: "
                f"{len(base_pred)} vs {len(y)}"
            )
        )


    samples = load_samples()

    id_to_sample = {
        str(s["id"]): s
        for s in samples
    }

    val_samples = [
        id_to_sample[sid]
        for sid in val_ids
    ]


    subset_idx = make_subset(
        y=y,
        base_pred=base_pred,
        subset_size=args.subset_size,
        seed=args.seed,
    )


    y_sub = y[subset_idx]

    base_sub = base_pred[subset_idx]

    samples_sub = [
        val_samples[i]
        for i in subset_idx
    ]


    print("SUBSET N:", len(subset_idx))
    print(
        "SUBSET BASE F1:",
        macro_f1(
            y_sub,
            base_sub,
        ),
    )
    print(
        "SUBSET BASE ACC:",
        accuracy_score(
            y_sub,
            base_sub,
        ),
    )


    sota = load_sota_module()

    state_builder = (
        sota.build_text_eda_state_diet
    )


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()


    all_pairs = []
    pair_meta = []


    for row_idx, (
        sample,
        base_class,
    ) in enumerate(
        zip(
            samples_sub,
            base_sub,
        )
    ):

        group_idx = int(
            CLASS_TO_GROUP_ARR[
                base_class
            ]
        )

        group_name = GROUP_NAMES[
            group_idx
        ]

        state_text = state_builder(
            sample
        )

        user_prompt = build_user_prompt(
            state_text,
            group_name,
        )

        prefix = make_prefix(
            tokenizer,
            user_prompt,
        )


        for candidate in GROUPS[group_name]:

            all_pairs.append(
                (
                    prefix,
                    candidate,
                )
            )

            pair_meta.append(
                (
                    row_idx,
                    C2I[candidate],
                )
            )


    print("CANDIDATE PAIRS:", len(all_pairs))


    score_matrix = np.full(
        (
            len(samples_sub),
            len(CLASSES),
        ),
        -np.inf,
        dtype=np.float32,
    )


    processed = 0


    for start in range(
        0,
        len(all_pairs),
        args.batch,
    ):

        batch_pairs = all_pairs[
            start:start + args.batch
        ]

        scores = score_batch(
            model=model,
            tokenizer=tokenizer,
            batch_pairs=batch_pairs,
            device=device,
            max_state_tokens=args.max_state_tokens,
        )


        meta_batch = pair_meta[
            start:start + len(batch_pairs)
        ]


        for (
            row_idx,
            class_id,
        ), score in zip(
            meta_batch,
            scores,
        ):

            score_matrix[
                row_idx,
                class_id,
            ] = float(score)


        processed += len(batch_pairs)

        if processed % 500 < args.batch:
            print(
                "scored",
                processed,
                "/",
                len(all_pairs),
            )


    decoder_pred = (
        score_matrix.argmax(axis=1)
    )


    print()
    print("=" * 100)
    print("RESULT")
    print("=" * 100)


    print(
        "BASE F1:",
        macro_f1(
            y_sub,
            base_sub,
        ),
    )

    print(
        "DECODER F1:",
        macro_f1(
            y_sub,
            decoder_pred,
        ),
    )


    print(
        "BASE ACC:",
        accuracy_score(
            y_sub,
            base_sub,
        ),
    )

    print(
        "DECODER ACC:",
        accuracy_score(
            y_sub,
            decoder_pred,
        ),
    )


    true_group = (
        CLASS_TO_GROUP_ARR[
            y_sub
        ]
    )

    base_group = (
        CLASS_TO_GROUP_ARR[
            base_sub
        ]
    )

    group_correct = (
        true_group == base_group
    )


    print()
    print("GROUP-CORRECT ONLY")

    print(
        "N:",
        int(group_correct.sum()),
    )

    print(
        "BASE ACC:",
        accuracy_score(
            y_sub[group_correct],
            base_sub[group_correct],
        ),
    )

    print(
        "DECODER ACC:",
        accuracy_score(
            y_sub[group_correct],
            decoder_pred[group_correct],
        ),
    )


    beneficial = (
        (base_sub != y_sub)
        & (decoder_pred == y_sub)
    )

    harmful = (
        (base_sub == y_sub)
        & (decoder_pred != y_sub)
    )


    print()
    print(
        "BASE WRONG / DECODER CORRECT:",
        int(beneficial.sum()),
    )

    print(
        "BASE CORRECT / DECODER WRONG:",
        int(harmful.sum()),
    )

    print(
        "NET CORRECT GAIN:",
        int(
            beneficial.sum()
            - harmful.sum()
        ),
    )


    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    np.save(
        output_dir
        / "subset_idx.npy",
        subset_idx,
    )

    np.save(
        output_dir
        / "candidate_score.npy",
        score_matrix,
    )


    out_df = pd.DataFrame({
        "id": val_ids[subset_idx],
        "true_id": y_sub,
        "base_pred": base_sub,
        "decoder_pred": decoder_pred,
        "true_action": [
            CLASSES[x]
            for x in y_sub
        ],
        "base_action": [
            CLASSES[x]
            for x in base_sub
        ],
        "decoder_action": [
            CLASSES[x]
            for x in decoder_pred
        ],
    })


    out_df.to_csv(
        output_dir
        / "predictions.csv",
        index=False,
    )


    with open(
        output_dir
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "model_id": args.model_id,
                "subset_n": len(subset_idx),
                "base_f1": float(
                    macro_f1(
                        y_sub,
                        base_sub,
                    )
                ),
                "decoder_f1": float(
                    macro_f1(
                        y_sub,
                        decoder_pred,
                    )
                ),
                "base_acc": float(
                    accuracy_score(
                        y_sub,
                        base_sub,
                    )
                ),
                "decoder_acc": float(
                    accuracy_score(
                        y_sub,
                        decoder_pred,
                    )
                ),
                "group_correct_base_acc": float(
                    accuracy_score(
                        y_sub[group_correct],
                        base_sub[group_correct],
                    )
                ),
                "group_correct_decoder_acc": float(
                    accuracy_score(
                        y_sub[group_correct],
                        decoder_pred[group_correct],
                    )
                ),
                "beneficial": int(
                    beneficial.sum()
                ),
                "harmful": int(
                    harmful.sum()
                ),
            },
            f,
            indent=2,
        )


    print()
    print(
        "SAVED:",
        output_dir,
    )


if __name__ == "__main__":
    main()
