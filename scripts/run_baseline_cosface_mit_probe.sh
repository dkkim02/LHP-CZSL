#!/bin/bash
# Stage 0 probe (§16-5): cosface_margin m=0.0 (sanity) then m=0.1, 1 epoch each, mit-states seed 0.
# m=0 path must produce loss curve indistinguishable from plain baseline (no-op verification).
# m=0.1 path must show: no NaN/inf, monotone-ish loss decrease, gradient norms bounded.
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TS=$(date +%Y%m%d_%H%M%S)
RUNLOG=logs/run_baseline_cosface_mit_probe_${TS}.log

echo "=== Stage 0 probe START $(date) on GPU 1" | tee -a ${RUNLOG}

for M_TAG in m00 m01; do
  YML=config/lhp_czsl_baseline_cosface_${M_TAG}_mit_l14_probe.yml
  LOG=logs/train_baseline_cosface_${M_TAG}_mit_probe_${TS}.log
  echo "=== [${M_TAG}] start $(date) -> ${LOG}" | tee -a ${RUNLOG}
  ${PY} train.py --yml_path ${YML} > ${LOG} 2>&1
  echo "=== [${M_TAG}] end   $(date)" | tee -a ${RUNLOG}
done

echo "=== Stage 0 probe DONE $(date)" | tee -a ${RUNLOG}
