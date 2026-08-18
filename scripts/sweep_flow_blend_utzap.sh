#!/bin/bash
cd /data1/workspaces/jgshin22/LHP-CZSL
PY=/data1/workspaces/jgshin22/miniconda3/envs/CZSL/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

YML=config/flow_composer_utzap_l14_llm_seed0.yml
CKPT_DIR=checkpoint/flow_composer_l14_utzap_llm_seed0
BLENDS=(0.5 0.7 0.8 0.9 1.0)
CKPTS=(val_best epoch_12)

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
OUT=logs/sweep_flow_blend_utzap_${TS}.log
exec > >(tee -a "$OUT") 2>&1

echo "=== flow_blend sweep on UT-Zap LLM ckpt ==="
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

for ckpt_stem in "${CKPTS[@]}"; do
  ckpt="$CKPT_DIR/${ckpt_stem}.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] missing $ckpt"
    continue
  fi
  for fb in "${BLENDS[@]}"; do
    tag="${ckpt_stem}_fb${fb}"
    fb_pad=$(printf "%.2f" "$fb")
    json="${CKPT_DIR}/${ckpt_stem}.closed.fb${fb_pad}.json"
    echo "=== $tag (json=$json) ==="
    if [[ -f "$json" ]]; then
      echo "  (cached, skipping run)"
    else
      raw="logs/sweep_${TS}_${tag}.raw.log"
      echo "  raw -> $raw"
      $PY test.py --yml_path "$YML" --load_model "$ckpt" --flow_blend "$fb" --open_world False > "$raw" 2>&1
      rc=$?
      echo "  exit=$rc"
      tail -2 "$raw"
    fi
    if [[ -f "$json" ]]; then
      $PY -c "import json; d=json.load(open('$json'))['test']; print(f'  >>> $tag test: HM={d[\"best_hm\"]:.4f} AUC={d[\"AUC\"]:.4f} seen={d[\"best_seen\"]:.4f} unseen={d[\"best_unseen\"]:.4f}')"
    else
      echo "  >>> $tag: JSON missing!"
    fi
  done
done
echo "DONE"
echo "Log: $OUT"
