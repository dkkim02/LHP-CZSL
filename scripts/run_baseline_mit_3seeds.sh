#!/bin/bash
# Sequential 3-seed mit-states cluspro_baseline re-run (post AMP/NaN fix).
# Replaces pre-fix checkpoint/cluspro_baseline_l14_mit as the trustworthy reference.
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
TS=$(date +%Y%m%d_%H%M%S)
RUNLOG=logs/run_baseline_mit_3seeds_${TS}.log

echo "=== START $(date) on GPU 1, sequential 3 seeds" | tee -a ${RUNLOG}

for SEED in 0 1 2; do
  YML=config/cluspro_baseline_mit_l14_v2_seed${SEED}.yml
  LOG=logs/train_baseline_mit_v2_seed${SEED}_${TS}.log
  echo "=== [seed=${SEED}] start $(date) -> ${LOG}" | tee -a ${RUNLOG}
  CUDA_VISIBLE_DEVICES=1 ${PY} train.py --yml_path ${YML} > ${LOG} 2>&1
  echo "=== [seed=${SEED}] end   $(date)" | tee -a ${RUNLOG}
done

echo "=== ALL SEEDS DONE $(date)" | tee -a ${RUNLOG}
