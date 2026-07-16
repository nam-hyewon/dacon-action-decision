#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

MODEL_NAME=submit/model/debug_deberta_base_compact_cue \
TEXT_MODE=eda_state_compact_cue \
SAVE_NAME=deberta_base_compact_cue_continue3 \
EPOCHS=3 \
BATCH_SIZE=8 \
MAX_LEN=512 \
LR=1e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.03 \
python src/train_roberta_v2plus.py
