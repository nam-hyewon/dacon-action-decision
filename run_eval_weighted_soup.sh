#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

MODEL_DIR=team_4model_ensemble_07626/ensemble_07626/fold0_team/deberta_base_eda_state4_fold0_weighted_soup_42_43_44 \
BATCH_SIZE=16 \
MAX_LEN=512 \
python eval_fold0_hf_model.py
