#!/bin/bash
set -e
cd /data1/workspaces/jgshin22/LHP-CZSL
LOG=logs/llm_gen_mit_*.log
PY=/data1/workspaces/jgshin22/miniconda3/envs/CZSL/bin/python
# Wait for description gen done
until grep -q "^\[gen\] done in" $LOG 2>/dev/null; do sleep 60; done
echo "[$(date)] mit-states gen DONE — launching trainings"
TS=$(date +%Y%m%d_%H%M%S)
# Fallback on GPU 1
tmux new-session -d -s flow-mit "CUDA_VISIBLE_DEVICES=1 $PY train.py --yml_path config/flow_composer_mit_l14_seed0.yml > logs/train_flow_composer_mit_seed0_${TS}.log 2>&1"
# LLM on GPU 2
tmux new-session -d -s flow-mit-llm "CUDA_VISIBLE_DEVICES=2 $PY train.py --yml_path config/flow_composer_mit_l14_llm_seed0.yml > logs/train_flow_composer_mit_llm_seed0_${TS}.log 2>&1"
echo "[$(date)] launched flow-mit and flow-mit-llm"
