#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

MODEL_NAME=microsoft/mdeberta-v3-base \
TEXT_MODE=eda_state \
SAVE_NAME=mdeberta_base_eda_state4_full70k \
FULL_TRAIN=1 \
EPOCHS=4 \
BATCH_SIZE=4 \
MAX_LEN=512 \
LR=1e-5 \
python src/train_roberta_full.py
