#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision/team_4model_ensemble_07626/ensemble_07626

rm -rf output
mkdir -p output

echo "🚀 START: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "----- RUN 4MODEL SPEEDTEST -----"
BATCH_SIZE=32 MAX_LEN=512 python speedtest_4model_infer.py

echo "----- OUTPUT HEAD -----"
head output/submission.csv

echo "🏁 END: $(date)"
