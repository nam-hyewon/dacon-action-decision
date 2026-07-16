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
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


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
    x: i
    for i, x in enumerate(CLASSES)
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

for gi, group_name in enumerate(GROUP_NAMES):
    for action in GROUPS[group_name]:
        CLASS_TO_GROUP[C2I[action]] = gi

CLASS_TO_GROUP_ARR = np.array([
    CLASS_TO_GROUP[i]
    for i in range(len(CLASSES))
])


# ---------------------------------------------------------------------------
# Action semantics.
#
# Descriptions deliberately distinguish OPERATIONAL PRECONDITIONS,
# not just paraphrase class names.
# ---------------------------------------------------------------------------

ACTION_DESCRIPTIONS = {
    "read_file": """
Action: read_file.
Purpose: Open and inspect the contents of a specific known file.
Use when a concrete file path or file of interest has already been identified
and the next step is to examine its implementation, text, configuration,
or a relevant region of that file.
Do not use primarily to search across many files or discover filenames.
""",

    "grep_search": """
Action: grep_search.
Purpose: Search file contents for a symbol, identifier, keyword, string,
error text, import, definition, or other textual pattern.
Use when occurrences or relevant code locations must be discovered across
one or more files before choosing a specific file or region to inspect.
""",

    "list_directory": """
Action: list_directory.
Purpose: Inspect the entries and immediate structure of a known directory.
Use when the repository or folder layout must be explored, when the contents
of a directory are unknown, or when deciding which subdirectory or file to
investigate next.
""",

    "glob_pattern": """
Action: glob_pattern.
Purpose: Find file paths whose filenames or paths match a pattern.
Use when the exact file location is unknown but a filename, extension,
directory pattern, or wildcard-style path constraint is known.
This searches for paths rather than searching text inside files.
""",

    "edit_file": """
Action: edit_file.
Purpose: Modify an existing file using a direct targeted edit.
Use when the file and intended local change are known and existing content
should be replaced, inserted, or updated.
""",

    "write_file": """
Action: write_file.
Purpose: Write complete file contents, typically to create a new file or
replace a file with a full known body.
Use when whole-file content is being produced rather than applying a small
localized modification to existing text.
""",

    "apply_patch": """
Action: apply_patch.
Purpose: Apply a structured patch or diff containing one or more precise
changes to existing files.
Use when modifications are naturally expressed as patch hunks, potentially
across multiple regions or files.
""",

    "run_bash": """
Action: run_bash.
Purpose: Execute a general shell command or program.
Use for builds, scripts, package commands, filesystem operations, git commands,
or arbitrary command-line execution that is not specifically a test run or
static lint/type-check operation.
""",

    "run_tests": """
Action: run_tests.
Purpose: Execute tests to verify program behavior or confirm whether a change
passes the test suite.
Use when the immediate intent is testing, reproducing a test failure,
or validating a fix through tests.
""",

    "lint_or_typecheck": """
Action: lint_or_typecheck.
Purpose: Run static linting, formatting checks, compiler-style diagnostics,
or type checking.
Use when the goal is static code-quality or type analysis rather than
executing behavioral tests.
""",

    "ask_user": """
Action: ask_user.
Purpose: Ask the user for missing information, clarification, permission,
a choice, credentials, or a decision required before proceeding.
Use when the assistant cannot responsibly choose or continue without
additional user input.
""",

    "plan_task": """
Action: plan_task.
Purpose: Formulate or update a multi-step execution plan for the task.
Use when the assistant has enough information to reason about the work but
should organize substantial upcoming steps before executing them.
Do not use merely because clarification from the user is missing.
""",

    "web_search": """
Action: web_search.
Purpose: Search external public web information.
Use when current, external, or publicly sourced information is required and
cannot be obtained from the local workspace or conversation state.
""",

    "respond_only": """
Action: respond_only.
Purpose: Give the user a direct final or conversational response without
performing another tool action.
Use when enough information is already available and no file, shell, test,
planning, clarification, or web operation is needed next.
""",
}


REPRO_ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

PRED_CSV = (
    REPRO_ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)

SOTA_SCRIPT = Path(
    "submit/script_large_tab_diet_rob8_gated40.py"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        default="klue/roberta-base",
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=43,
    )

    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=2,
    )

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_sota_module():
    spec = importlib.util.spec_from_file_location(
        "sota_script",
        SOTA_SCRIPT,
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def build_candidate_text(
    state_text,
    candidate_action,
):
    description = ACTION_DESCRIPTIONS[
        candidate_action
    ].strip()

    return "\n".join([
        "[ACTION_COMPATIBILITY_TASK]",
        (
            "Decide whether the candidate assistant action is "
            "the appropriate NEXT action for the current state."
        ),

        "[CURRENT_STATE]",
        state_text,

        "[CANDIDATE_ACTION]",
        description,

        "[JUDGMENT]",
        (
            "Is this candidate the appropriate next assistant "
            "action given the current state?"
        ),
    ])


class PairDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
    ):
        self.texts = texts
        self.labels = np.asarray(
            labels,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": int(self.labels[idx]),
        }


def make_collate(
    tokenizer,
    max_len,
):

    def collate(batch):

        texts = [
            x["text"]
            for x in batch
        ]

        labels = torch.tensor(
            [
                x["label"]
                for x in batch
            ],
            dtype=torch.long,
        )

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        enc["labels"] = labels

        return enc

    return collate


@torch.no_grad()
def score_candidates(
    model,
    tokenizer,
    samples,
    state_builder,
    base_pred,
    device,
    max_len,
    batch_size,
):
    model.eval()

    pair_texts = []
    pair_rows = []

    pred_groups = CLASS_TO_GROUP_ARR[
        base_pred
    ]

    for row_idx, sample in enumerate(samples):

        gi = int(pred_groups[row_idx])
        group_name = GROUP_NAMES[gi]

        state_text = state_builder(sample)

        for candidate in GROUPS[group_name]:

            pair_texts.append(
                build_candidate_text(
                    state_text,
                    candidate,
                )
            )

            pair_rows.append(
                (
                    row_idx,
                    C2I[candidate],
                )
            )


    scores = np.full(
        (
            len(samples),
            len(CLASSES),
        ),
        -np.inf,
        dtype=np.float64,
    )


    for start in range(
        0,
        len(pair_texts),
        batch_size,
    ):
        texts = pair_texts[
            start:start + batch_size
        ]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        enc = {
            k: v.to(device)
            for k, v in enc.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(**enc).logits

        # Binary relevance logit difference.
        relevance = (
            logits[:, 1]
            - logits[:, 0]
        ).float().cpu().numpy()


        cur_rows = pair_rows[
            start:start + len(texts)
        ]

        for (
            row_idx,
            class_id,
        ), score in zip(
            cur_rows,
            relevance,
        ):
            scores[
                row_idx,
                class_id,
            ] = float(score)


    return scores


def macro_f1(y, pred):

    return f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )


def main():

    args = parse_args()

    seed_everything(args.seed)


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


    print("Loading data...")

    samples = load_jsonl(
        "data/train.jsonl"
    )

    sample_ids = np.array([
        str(s["id"])
        for s in samples
    ])


    with open(
        "data/train_labels.csv",
        encoding="utf-8",
    ) as f:
        label_map = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(f)
        }


    all_y = np.array([
        C2I[label_map[sid]]
        for sid in sample_ids
    ], dtype=np.int64)


    pred_df = pd.read_csv(
        PRED_CSV
    )

    val_ids = (
        pred_df["id"]
        .astype(str)
        .values
    )

    y_val = (
        pred_df["true_id"]
        .astype(int)
        .values
    )


    id_to_idx = {
        sid: i
        for i, sid in enumerate(sample_ids)
    }

    val_indices = np.array([
        id_to_idx[sid]
        for sid in val_ids
    ])

    val_set = set(val_ids)

    train_indices = np.array([
        i
        for i, sid in enumerate(sample_ids)
        if sid not in val_set
    ])


    print("train rows:", len(train_indices))
    print("val rows:", len(val_indices))


    sota = load_sota_module()

    # Reuse the already successful compact state serializer.
    state_builder = (
        sota.build_text_eda_state_diet
    )


    print("Building compact states...")

    train_states = [
        state_builder(samples[i])
        for i in train_indices
    ]

    val_samples = [
        samples[i]
        for i in val_indices
    ]


    # ----------------------------------------------------------------------
    # Build binary compatibility training pairs.
    #
    # One positive plus N same-group hard negatives.
    # ----------------------------------------------------------------------

    print("Building candidate training pairs...")

    rng = random.Random(args.seed)

    pair_texts = []
    pair_labels = []


    for local_pos, global_idx in enumerate(
        train_indices
    ):

        true_id = int(
            all_y[global_idx]
        )

        true_action = CLASSES[
            true_id
        ]

        gi = int(
            CLASS_TO_GROUP_ARR[
                true_id
            ]
        )

        group_name = GROUP_NAMES[gi]

        state_text = train_states[
            local_pos
        ]


        # Positive.
        pair_texts.append(
            build_candidate_text(
                state_text,
                true_action,
            )
        )

        pair_labels.append(1)


        negatives = [
            x
            for x in GROUPS[group_name]
            if x != true_action
        ]

        rng.shuffle(negatives)

        negatives = negatives[
            :args.negatives_per_positive
        ]


        for candidate in negatives:

            pair_texts.append(
                build_candidate_text(
                    state_text,
                    candidate,
                )
            )

            pair_labels.append(0)


    print(
        "training pairs:",
        len(pair_texts),
    )

    print(
        "pair labels:",
        np.bincount(
            np.asarray(pair_labels),
            minlength=2,
        ).tolist(),
    )


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.model_id,
            num_labels=2,
        )
    )

    # Critical: keep parameters FP32.
    model = model.float()
    model.to(device)

    print(
        "MODEL PARAM DTYPES:",
        sorted({
            str(p.dtype)
            for p in model.parameters()
        }),
    )


    dataset = PairDataset(
        pair_texts,
        pair_labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=make_collate(
            tokenizer,
            args.max_len,
        ),
        num_workers=2,
        pin_memory=(
            device.type == "cuda"
        ),
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )


    optimizer_steps_per_epoch = (
        len(loader)
        + args.grad_accum
        - 1
    ) // args.grad_accum

    total_steps = (
        optimizer_steps_per_epoch
        * args.epochs
    )

    warmup_steps = int(
        total_steps * 0.06
    )


    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )


    base_score = np.load(
        "exact_repro_base_score.npy"
    ).astype(np.float64)

    base_pred = np.load(
        "exact_repro_base_pred.npy"
    ).astype(np.int64)


    BASE_F1 = macro_f1(
        y_val,
        base_pred,
    )

    print("BASE F1:", BASE_F1)


    save_dir = Path(
        "submit/model/"
        "candidate_crossencoder_fold0"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    best_f1 = -1.0
    best_epoch = None


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print("=" * 100)
        print("EPOCH", epoch)
        print("=" * 100)


        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0


        for step, batch in enumerate(
            loader,
            start=1,
        ):

            labels = batch.pop(
                "labels"
            ).to(device)

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }


            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    device.type == "cuda"
                ),
            ):
                outputs = model(
                    **batch,
                    labels=labels,
                )

                loss = (
                    outputs.loss
                    / args.grad_accum
                )


            scaler.scale(loss).backward()

            running_loss += (
                float(loss.item())
                * args.grad_accum
            )


            should_step = (
                step % args.grad_accum == 0
                or step == len(loader)
            )


            if should_step:

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )


            if step % 500 == 0:

                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(loader)} "
                    f"loss={running_loss / step:.6f}"
                )


        print(
            "epoch train loss:",
            running_loss / len(loader),
        )


        print(
            "Scoring validation candidates..."
        )


        candidate_score = score_candidates(
            model=model,
            tokenizer=tokenizer,
            samples=val_samples,
            state_builder=state_builder,
            base_pred=base_pred,
            device=device,
            max_len=args.max_len,
            batch_size=args.batch * 2,
        )


        candidate_pred = (
            candidate_score.argmax(axis=1)
        )


        true_group = (
            CLASS_TO_GROUP_ARR[y_val]
        )

        pred_group = (
            CLASS_TO_GROUP_ARR[
                base_pred
            ]
        )

        group_correct = (
            true_group == pred_group
        )


        print()
        print("GROUP-CORRECT ROWS")

        print(
            "base acc:",
            accuracy_score(
                y_val[group_correct],
                base_pred[group_correct],
            ),
        )

        print(
            "candidate acc:",
            accuracy_score(
                y_val[group_correct],
                candidate_pred[group_correct],
            ),
        )


        # --------------------------------------------------------------
        # Confidence/margin gated replacement.
        # --------------------------------------------------------------

        finite = np.isfinite(
            candidate_score
        )

        masked_score = np.where(
            finite,
            candidate_score,
            -1e30,
        )

        sorted_score = np.sort(
            masked_score,
            axis=1,
        )

        candidate_margin = (
            sorted_score[:, -1]
            - sorted_score[:, -2]
        )


        rows = []


        for margin_thr in [
            -1e9,
            0.0,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.75,
            1.00,
            1.25,
            1.50,
            2.00,
            2.50,
            3.00,
            4.00,
        ]:

            use = (
                candidate_margin
                >= margin_thr
            )

            pred = base_pred.copy()

            pred[use] = candidate_pred[
                use
            ]


            f1 = macro_f1(
                y_val,
                pred,
            )


            beneficial = int(
                (
                    (base_pred[use] != y_val[use])
                    & (pred[use] == y_val[use])
                ).sum()
            )

            harmful = int(
                (
                    (base_pred[use] == y_val[use])
                    & (pred[use] != y_val[use])
                ).sum()
            )


            rows.append({
                "epoch": epoch,
                "margin_thr": margin_thr,
                "macro_f1": f1,
                "gain": f1 - BASE_F1,
                "accuracy": accuracy_score(
                    y_val,
                    pred,
                ),
                "use_n": int(
                    use.sum()
                ),
                "changed_n": int(
                    (
                        pred[use]
                        != base_pred[use]
                    ).sum()
                ),
                "beneficial_changes": beneficial,
                "harmful_changes": harmful,
                "net_correct_gain": (
                    beneficial - harmful
                ),
            })


        result = (
            pd.DataFrame(rows)
            .sort_values(
                [
                    "macro_f1",
                    "accuracy",
                ],
                ascending=False,
            )
        )


        print()
        print("=" * 120)
        print("CANDIDATE CROSS-ENCODER GATED SWEEP")
        print("=" * 120)

        print(
            result.to_string(
                index=False,
            )
        )


        epoch_f1 = float(
            result.iloc[0]["macro_f1"]
        )


        np.save(
            save_dir
            / f"candidate_score_epoch{epoch}.npy",
            candidate_score,
        )

        result.to_csv(
            save_dir
            / f"gated_results_epoch{epoch}.csv",
            index=False,
        )


        if epoch_f1 > best_f1:

            best_f1 = epoch_f1
            best_epoch = epoch

            model.save_pretrained(
                save_dir
            )

            tokenizer.save_pretrained(
                save_dir
            )

            np.save(
                save_dir
                / "best_candidate_score.npy",
                candidate_score,
            )

            result.to_csv(
                save_dir
                / "best_gated_results.csv",
                index=False,
            )

            print(
                "NEW BEST:",
                best_f1,
            )


    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print(
        "BASE F1:",
        BASE_F1,
    )

    print(
        "BEST F1:",
        best_f1,
    )

    print(
        "BEST GAIN:",
        best_f1 - BASE_F1,
    )

    print(
        "BEST EPOCH:",
        best_epoch,
    )

    print(
        "SAVE DIR:",
        save_dir,
    )


if __name__ == "__main__":
    main()
