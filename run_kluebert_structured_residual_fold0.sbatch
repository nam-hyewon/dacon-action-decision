#!/bin/bash
#SBATCH --job-name=klue_resid_f0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-klue-resid-f0-%j.out
#SBATCH --error=slurm-klue-resid-f0-%j.err

set -e

echo "===== KLUEBERT STRUCTURED RESIDUAL FOLD0 START ====="

date
hostname
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
BASE_MODEL_PATH=team_4model_ensemble_07626/ensemble_07626/fold0_team/kluebert_folds/kluebert_mid5_lr15_fold0 \
SAVE_NAME=kluebert_structured_residual_fold0 \
EPOCHS=6 \
PATIENCE=2 \
BATCH_SIZE=16 \
MAX_LEN=512 \
BASE_LR=3e-6 \
STRUCT_LR=2e-4 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
ACTION_EMB_DIM=32 \
STRUCT_HIDDEN=128 \
DROPOUT=0.10 \
REF_VAL_PATH=team_4model_ensemble_07626/ensemble_07626/fold0_team/tabular_folds/tabular_histgb_fold0/val_predictions.csv \
python -u src/train_kluebert_structured_residual_fold_ref.py \
2>&1 | tee kluebert_structured_residual_fold0.log

echo "===== KLUEBERT STRUCTURED RESIDUAL FOLD0 END ====="

date

ls -lh \
submit/model/kluebert_structured_residual_fold0
