#!/bin/bash
# Check status / extract numbers from paperhp probe run
set -e
LOG=$(ls -t /home/student/dongki/LHP-CZSL/logs/train_cluspro_baseline_mit_paperhp_probe_*.log 2>/dev/null | head -1)
CKPT_DIR=/home/student/dongki/LHP-CZSL/checkpoint/cluspro_baseline_l14_mit_paperhp_probe

if [[ -z "$LOG" ]]; then
  echo "no probe log found"
  exit 1
fi

echo "=== log: $LOG"
echo "=== running PIDs:"
ps -ef | grep "paperhp_probe" | grep -v grep | head
echo
echo "=== last 5 epoch best lines (val_pairs running max):"
grep -aE "best_seen" "$LOG" | tail -10
echo
echo "=== epoch counts:"
grep -aE "^epoch [0-9]+ train loss" "$LOG" | tail -5

if [[ -f "$CKPT_DIR/val_best.closed.json" ]]; then
  echo
  echo "=== test_pairs of val_best (existing JSON):"
  python3 -c "import json; d = json.load(open('$CKPT_DIR/val_best.closed.json'))['test']; print(f'  HM={d[\"best_hm\"]:.4f} AUC={d[\"AUC\"]:.4f} seen={d[\"best_seen\"]:.4f} unseen={d[\"best_unseen\"]:.4f}')"
elif [[ -f "$CKPT_DIR/val_best.pt" ]]; then
  echo
  echo "=== val_best.pt exists but no closed.json. Re-eval:"
  echo "cd /home/student/dongki/LHP-CZSL && CUDA_VISIBLE_DEVICES=1 /home/student/anaconda3/envs/lhp_czsl/bin/python test.py --yml_path config/cluspro_baseline_mit_l14_paperhp_probe.yml --load_model $CKPT_DIR/val_best.pt"
fi

echo
echo "=== reference (cluspro_baseline_l14_mit, 4-27 baseline, test_pairs):"
echo "  HM=0.3893 AUC=0.2169 seen=0.4899 unseen=0.5203"
echo "=== reference (paper ClusPro Table 1, MIT-States CW):"
echo "  HM=0.407  AUC=0.238  seen=0.521  unseen=0.540"
