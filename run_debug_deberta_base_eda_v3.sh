#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

MAX_TRAIN=3000 \
MAX_VAL=1000 \
STUDENT_MODEL=submit/model/deberta_base_eda_state4 \
SAVE_NAME=debug_deberta_base_eda_v3 \
EPOCHS=1 \
BATCH_SIZE=4 \
MAX_LEN=512 \
LR=5e-6 \
python src/train_deberta_base_eda_v3.py
