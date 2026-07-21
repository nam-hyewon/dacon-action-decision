#!/usr/bin/env bash
set -euo pipefail

run_one () {
  local save_name="$1"
  local text_mode="$2"
  local epochs="$3"
  local lr="$4"
  local max_len="$5"

  echo "===== START ${save_name} ====="

  TEXT_MODE="${text_mode}" \
  SAVE_NAME="${save_name}" \
  EPOCHS="${epochs}" \
  MODEL_NAME="klue/roberta-base" \
  MAX_LEN="${max_len}" \
  BATCH_SIZE=16 \
  LR="${lr}" \
  python src/train_roberta.py

  python src/tune_specialist_boosters.py \
    --data_dir data \
    --val_csv "submit/model/${save_name}/val_predictions.csv" \
    --val_logits "submit/model/${save_name}/val_logits.npy" \
    --out_dir "submit/model/${save_name}/specialist_boosters" \
    --greedy

  echo "===== DONE ${save_name} ====="
}

run_one "roberta_mid5_lr2" "eda_state_mid" 5 "2e-5" 512
run_one "roberta_state5_lr2" "eda_state" 5 "2e-5" 512
