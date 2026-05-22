#!/bin/bash
# Stage 1 (§16-5): cosface_margin m=0.1, mit-states seed 0, 15 epoch, val_metric=best_hm.
# Pre-registered cutoff vs baseline 0.3887:
#   HM >= 0.3937 -> Stage 2 launch
#   HM <= 0.3837 -> axis seal (8연패)
#   tie -> m in {0.05, 0.2} ablation
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TS=$(date +%Y%m%d_%H%M%S)
YML=config/lhp_czsl_baseline_cosface_m01_mit_l14_seed0.yml
LOG=logs/train_baseline_cosface_m01_mit_seed0_${TS}.log

echo "=== Stage 1 start $(date) on GPU 1 -> ${LOG}" | tee -a ${LOG}
${PY} train.py --yml_path ${YML} >> ${LOG} 2>&1
echo "=== Stage 1 end $(date)" | tee -a ${LOG}
