#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision/final_submit

rm -f output/submission.csv

python - <<'PY'
import time
import subprocess
import sys
from pathlib import Path

t0 = time.perf_counter()
ret = subprocess.run([sys.executable, "script.py"])
t1 = time.perf_counter()

print("\n===== TIME RESULT =====")
print(f"elapsed_sec: {t1 - t0:.2f}")
print(f"elapsed_min: {(t1 - t0) / 60:.2f}")
print("submission_exists:", Path("output/submission.csv").exists())

raise SystemExit(ret.returncode)
PY
