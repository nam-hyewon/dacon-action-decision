#!/bin/bash
#SBATCH --job-name=large-policy-w
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=large_policy_weighted_f0_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

export MODEL_NAME="klue/roberta-large"
export TEXT_MODE="eda_state"
export MAX_LEN=512
export BATCH_SIZE=8
export EPOCHS=6
export LR=8e-6
export WEIGHT_DECAY=0.01
export WARMUP_RATIO=0.06

export SAVE_NAME="klue_roberta_large_eda_state_policy_weighted_fold0"

export REF_VAL_PATH="repro_bundle/large_tab_diet_rob8_repro/submit/model/kluebert_diet_fold0/val_predictions.csv"

python -u src/train_roberta_policy_weighted_fold_ref.py
