#!/usr/bin/env bash
# 학습 발사 전 디스크 가드.
#
# 배경(2026-07-11 사고): /root 는 /dev/loop0 (xfs, prjquota 100G) 이고,
# `df -h /` 가 보여주는 overlay 878G 는 무관하다. 쿼터가 차자 mdeberta 학습이
# HF 모델 *다운로드 도중* `No space left on device (os error 28)` 로 죽었고,
# 동시에 VSCode server / jupyter / git / Claude 하네스의 쓰기가 전부 실패했다.
#
# 사용: MODEL_NAME=klue/bert-base bash preflight.sh && <학습 커맨드>
#       MIN_GB=20 bash preflight.sh   (임계값 조정)
set -euo pipefail

MIN_GB="${MIN_GB:-15}"

MIN_INODES="${MIN_INODES:-200000}"

avail_gb="$(df -BG /root | tail -1 | awk '{print $4}' | tr -dc '0-9')"
free_inodes="$(df -i /root | tail -1 | awk '{print $4}' | tr -dc '0-9')"
echo "[preflight] /root 용량 여유 : ${avail_gb}G          (최소 ${MIN_GB}G)"
echo "[preflight] /root inode 여유: ${free_inodes}개  (최소 ${MIN_INODES}개)"

# 2026-07-13 사고: 용량은 34G 남았는데 inode 가 99% 라 학습이 저장 중에 죽었다.
# (touch 조차 실패. xlm-roberta-large 가 epoch1 저장 직후 사망, GPU 3시간 낭비)
# -> 용량과 inode 를 둘 다 봐야 한다.
if [ "${avail_gb:-0}" -lt "${MIN_GB}" ] || [ "${free_inodes:-0}" -lt "${MIN_INODES}" ]; then
  cat >&2 <<EOF
[preflight] ABORT: 디스크 부족
    용량  여유 ${avail_gb}G     (최소 ${MIN_GB}G)
    inode 여유 ${free_inodes}개 (최소 ${MIN_INODES}개)

  ※ 둘 중 하나만 부족해도 학습은 '저장 시점'에 조용히 죽는다 (Traceback 없이).

  회수 후보 (파일 '개수'가 많은 곳부터 = inode 회수):
    rm -rf /root/.venv, dacon-action-decision/.venv   # venv (수천 개 파일, 학습엔 시스템 python 사용)
    rm -rf /root/.vscode-server                       # IDE (재설치 가능, 12k+ 파일)
  용량 회수:
    submit/model/*/checkpoint-epoch*                  # best 모델은 부모에 별도 저장됨
    submit/*.zip                                      # 구버전 제출물

  주의: 'df -h /' 는 보지 말 것 (overlay 878G, 실제 상한과 무관). /root 를 볼 것.
EOF
  exit 1
fi

# HF 모델을 미리 받아둔다: 학습 도중 다운로드하다 ENOSPC 로 죽는 사고 방지(= mdeberta 사망 원인).
if [ -n "${MODEL_NAME:-}" ]; then
  echo "[preflight] HF 모델 사전 확보: ${MODEL_NAME}"
  python - <<'PY' || { echo "[preflight] ABORT: 모델 확보 실패 (네트워크/용량/이름 확인)" >&2; exit 1; }
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
m = os.environ["MODEL_NAME"]
AutoConfig.from_pretrained(m)
AutoTokenizer.from_pretrained(m)
AutoModelForSequenceClassification.from_pretrained(m, num_labels=14)  # 가중치까지 캐시에 확보
print(f"[preflight] 모델 캐시 확보 완료: {m}")
PY
fi

echo "[preflight] PASS — 발사 가능"
