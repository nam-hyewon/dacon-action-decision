import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export validation logits/predictions from a saved RoBERTa model."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text_mode", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", choices=["row", "group"], default="row")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["TEXT_MODE"] = args.text_mode
    os.environ["MAX_LEN"] = str(args.max_len)
    os.environ["SPLIT_MODE"] = args.split_mode
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import train_roberta as tr

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("MODEL_DIR:", model_dir)
    print("TEXT_MODE:", args.text_mode)
    print("SPLIT_MODE:", args.split_mode)
    print("OUTPUT_DIR:", output_dir)

    texts, labels, rows = tr.load_data()

    _, val_texts, _, val_labels, _, val_rows = tr.split_train_val(
        texts,
        labels,
        rows,
        split_mode=args.split_mode,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.float()
    model.to(device)

    val_ids = [r["id"] for r in val_rows]
    val_ds = tr.ActionDataset(val_texts, val_labels, val_ids)
    collate_fn = tr.make_collate_fn(tokenizer)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    val_f1, val_ids_eval, y_true, y_pred, val_logits = tr.evaluate(
        model,
        val_loader,
        device,
        return_logits=True,
    )
    print("Validation Macro-F1:", val_f1)

    tr.save_val_outputs(
        save_dir=output_dir,
        val_rows=val_rows,
        val_ids=val_ids_eval,
        y_true=y_true,
        y_pred=y_pred,
        logits=val_logits,
        best_f1=val_f1,
        best_epoch=0,
    )


if __name__ == "__main__":
    main()
