#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
STUDENT_MODEL=microsoft/deberta-v3-small \
SAVE_NAME=debug_deberta_small_ce_only \
EPOCHS=1 \
BATCH_SIZE=8 \
TEACHER_BATCH_SIZE=32 \
MAX_LEN=512 \
MAX_TRAIN=3000 \
MAX_VAL=1000 \
LR=5e-6 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
KD_ALPHA=0.0 \
KD_T=2.0 \
TEACHER_CACHE=submit/cache/md_teacher_train_logits.npy \
python src/train_distill_deberta_small_eda.py
