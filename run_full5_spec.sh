#!/bin/bash
set -e

echo "===== DEBERTA FULL5 + SPECIALIST TEST ====="
date
hostname
pwd
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision/submit_deberta_full5_spec

echo "===== CHECK FILES ====="
ls -lh
ls -lh model
ls -lh model/full5 | head
ls -lh model/specialist_booster.pkl

echo "===== ENV CHECK ====="
python - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
PY

echo "===== RUN SCRIPT WITH TIME ====="
time python script.py

echo "===== CHECK OUTPUT ====="
if [ -f ../output/submission.csv ]; then
    head ../output/submission.csv
elif [ -f output/submission.csv ]; then
    head output/submission.csv
else
    echo "ERROR: submission.csv not found"
    exit 1
fi

echo "===== DONE ====="
date
