#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

MAX_TRAIN=3000 \
MAX_VAL=1000 \
MODEL_NAME=microsoft/deberta-v3-base \
TEXT_MODE=eda_state_v2plus \
SAVE_NAME=debug_deberta_base_eda_state_v2plus \
EPOCHS=1 \
BATCH_SIZE=4 \
MAX_LEN=512 \
LR=2e-5 \
python src/train_roberta_v2plus.py
