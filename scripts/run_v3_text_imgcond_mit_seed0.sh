#!/bin/bash
# Stage 1: image-conditional description selection, mit-states, seed 0, 15 epoch.
# Cutoff: HM >= 0.3937 (baseline+0.005) -> Stage 2 3-seed.
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
TS=$(date +%Y%m%d_%H%M%S)
YML=config/lhp_czsl_v3_text_imgcond_mit_l14_seed0.yml
LOG=logs/train_v3_text_imgcond_mit_seed0_${TS}.log

echo "=== Stage 1 (image-cond, mit, seed0, 15ep) start $(date) on GPU 1 -> ${LOG} ==="
CUDA_VISIBLE_DEVICES=1 ${PY} train.py --yml_path ${YML} > ${LOG} 2>&1
echo "=== Stage 1 end $(date) ==="
