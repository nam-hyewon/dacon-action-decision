#!/bin/bash
#SBATCH --job-name=cand-xenc-f0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=candidate_crossencoder_f0_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

python -u train_candidate_crossencoder_fold0.py \
  --model-id klue/roberta-base \
  --max-len 512 \
  --batch 8 \
  --grad-accum 2 \
  --epochs 2 \
  --lr 1.5e-5 \
  --seed 43 \
  --negatives-per-positive 2
