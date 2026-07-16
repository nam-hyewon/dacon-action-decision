#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

MODEL_NAME=microsoft/deberta-v3-base \
TEXT_MODE=eda_state_v2plus \
SAVE_NAME=deberta_base_eda_state_v2plus \
EPOCHS=4 \
BATCH_SIZE=8 \
MAX_LEN=512 \
LR=2e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.03 \
python src/train_roberta_v2plus.py
