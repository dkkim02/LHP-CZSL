#!/bin/bash
set -e
cd /data1/workspaces/jgshin22/LHP-CZSL
LOG=logs/llm_gen_cgqa_*.log
PY=/data1/workspaces/jgshin22/miniconda3/envs/CZSL/bin/python
until grep -q "^\[gen\] done in" $LOG 2>/dev/null; do sleep 120; done
echo "[$(date)] cgqa gen DONE — launching trainings"
TS=$(date +%Y%m%d_%H%M%S)
# Use GPU 5, 6 (most likely still free; MIT runs on 3, 4)
tmux new-session -d -s flow-cgqa "CUDA_VISIBLE_DEVICES=5 $PY train.py --yml_path config/flow_composer_cgqa_l14_seed0.yml > logs/train_flow_composer_cgqa_seed0_${TS}.log 2>&1"
tmux new-session -d -s flow-cgqa-llm "CUDA_VISIBLE_DEVICES=6 $PY train.py --yml_path config/flow_composer_cgqa_l14_llm_seed0.yml > logs/train_flow_composer_cgqa_llm_seed0_${TS}.log 2>&1"
echo "[$(date)] launched flow-cgqa (GPU 5) and flow-cgqa-llm (GPU 6)"
