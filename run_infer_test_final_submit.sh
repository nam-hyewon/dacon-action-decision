#!/bin/bash
set -e

# 콘다 환경 활성화 (본인 환경 이름에 맞게 수정)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision/final_submit

rm -rf output
mkdir -p output

echo "🚀 START: $(date)"
time python script.py
echo "🏁 END: $(date)"

head output/submission.csv
