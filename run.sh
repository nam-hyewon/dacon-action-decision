#!/bin/bash
set -e

echo "===== DEBERTA SOTA SUBMISSION SMOKE TEST START ====="
date
hostname
pwd

nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision/submit

echo "===== CURRENT DIR ====="
pwd

echo "===== CHECK FILES ====="
ls -lh script.py requirements.txt
ls -lh model
ls -lh model/full5 | head
ls -lh model/router3 | head
ls -lh model/specialist_booster.pkl

echo "===== PYTHON ENV CHECK ====="
python - <<'PY'
import torch, transformers, sklearn, joblib
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("sklearn:", sklearn.__version__)
PY

echo "===== RUN SCRIPT ====="
python script.py

echo "===== CHECK OUTPUT ====="
ls -lh ../output || true
ls -lh output || true

if [ -f ../output/submission.csv ]; then
    echo "Found ../output/submission.csv"
    head ../output/submission.csv
elif [ -f output/submission.csv ]; then
    echo "Found output/submission.csv"
    head output/submission.csv
else
    echo "ERROR: submission.csv not found"
    exit 1
fi

echo "===== DONE ====="
date