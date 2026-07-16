#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
STUDENT_MODEL=submit/model/deberta_base_eda_state4 \
SAVE_NAME=deberta_base_eda_kd_fixed_builder_a005_t4_lr2e6 \
EPOCHS=1 \
BATCH_SIZE=8 \
TEACHER_BATCH_SIZE=32 \
MAX_LEN=512 \
LR=2e-6 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.03 \
KD_ALPHA=0.05 \
KD_T=4.0 \
TEACHER_CACHE=submit/cache/md_teacher_train_logits.npy \
python src/train_distill_deberta_small_eda.py
