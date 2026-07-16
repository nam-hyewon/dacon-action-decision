#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

MODEL_NAME=microsoft/deberta-v3-base \
TEXT_MODE=eda_state_short \
SAVE_NAME=deberta_base_eda_state_short_4ep_len384 \
EPOCHS=4 \
BATCH_SIZE=16 \
MAX_LEN=384 \
LR=1e-5 \
python src/train_roberta_short.py
