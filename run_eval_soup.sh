#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

BATCH_SIZE=32 \
MAX_LEN=512 \
python src/eval_deberta_small_soup.py
