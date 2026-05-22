#!/bin/bash
# H3 (§17-10) — Sinkhorn-balanced batch-level coupling for prototype update.
# Pre-registered judgment vs single-seed baseline 0.3879 (±0.0008 std):
#   HM Δ >= +0.005 (≥ 0.3929) -> 3-seed 본런 + paper 0.407 gap 측정
#   HM Δ ∈ [-0.005, +0.005]   -> ε ∈ {0.03, 0.1} × iters ∈ {3, 10} 4-cell ablation
#   HM Δ <  -0.005            -> 10th sealed axis; framework 전환 검토
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TS=$(date +%Y%m%d_%H%M%S)
YML=config/cluspro_baseline_mit_l14_v2_ot_seed0.yml
LOG=logs/train_ot_mit_seed0_${TS}.log

echo "=== H3 OT seed 0 start $(date) on GPU 1 -> ${LOG}" | tee -a ${LOG}
${PY} train.py --yml_path ${YML} >> ${LOG} 2>&1
echo "=== H3 OT seed 0 end $(date)" | tee -a ${LOG}
