#!/bin/bash
#SBATCH --job-name=klue_hybrid_f0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-klue-hybrid-f0-%j.out
#SBATCH --error=slurm-klue-hybrid-f0-%j.err

set -e

echo "===== KLUEBERT ACTION HYBRID FOLD0 START ====="

date
hostname
pwd
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
MODEL_NAME=klue/bert-base \
SAVE_NAME=kluebert_action_hybrid_fold0 \
EPOCHS=8 \
BATCH_SIZE=16 \
MAX_LEN=512 \
MAX_ACTION_LEN=24 \
LR=1.5e-5 \
HEAD_LR=1e-4 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
ACTION_EMB_DIM=64 \
ACTION_HIDDEN=128 \
FUSION_HIDDEN=384 \
DROPOUT=0.15 \
REF_VAL_PATH=team_4model_ensemble_07626/ensemble_07626/fold0_team/tabular_folds/tabular_histgb_fold0/val_predictions.csv \
python -u src/train_kluebert_action_hybrid_fold_ref.py \
2>&1 | tee kluebert_action_hybrid_fold0.log

echo "===== KLUEBERT ACTION HYBRID FOLD0 END ====="

date

ls -lh \
submit/model/kluebert_action_hybrid_fold0
