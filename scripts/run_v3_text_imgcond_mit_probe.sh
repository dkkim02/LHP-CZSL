#!/bin/bash
# Stage 0 probe: image-conditional description selection, mit-states, seed 0, 1 epoch.
# Goal: NaN check + loss monotone decrease + memory fit.
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
TS=$(date +%Y%m%d_%H%M%S)
YML=config/lhp_czsl_v3_text_imgcond_mit_l14_probe.yml
LOG=logs/train_v3_text_imgcond_mit_probe_${TS}.log

echo "=== Stage 0 probe (image-cond, mit, seed0, 1 ep) start $(date) on GPU 1 -> ${LOG} ==="
CUDA_VISIBLE_DEVICES=1 ${PY} train.py --yml_path ${YML} > ${LOG} 2>&1
echo "=== Stage 0 probe end $(date) ==="
