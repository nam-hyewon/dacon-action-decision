#!/bin/bash
#SBATCH --job-name=pair-ap-f0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=pair_ask_plan_f0_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

python train_binary_pair_specialist_fold0.py \
  --class-a ask_user \
  --class-b plan_task \
  --model-id microsoft/deberta-v3-base \
  --max-len 512 \
  --batch 4 \
  --grad-accum 4 \
  --epochs 3 \
  --lr 1e-5 \
  --seed 43
