#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

MODEL_NAME=klue/roberta-base \
TEXT_MODE=eda_state_mid_header \
SAVE_NAME=debug_roberta_mid_header \
EPOCHS=1 \
BATCH_SIZE=8 \
MAX_LEN=512 \
LR=2e-5 \
python src/train_roberta_header.py
