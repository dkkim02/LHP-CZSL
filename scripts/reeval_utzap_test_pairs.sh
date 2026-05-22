#!/bin/bash
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
export CUDA_VISIBLE_DEVICES=1

runs=(
  "cluspro_baseline_l14_utzap_v2_seed0:cluspro_baseline_utzap_l14_v2_seed0"
  "cluspro_baseline_l14_utzap_v2_seed1:cluspro_baseline_utzap_l14_v2_seed1"
  "cluspro_baseline_l14_utzap_v2_seed2:cluspro_baseline_utzap_l14_v2_seed2"
  "lhp_czsl_v1_init_only_l14_utzap_seed0:lhp_czsl_v1_init_only_utzap_l14_seed0"
  "lhp_czsl_v1_init_only_l14_utzap_seed1:lhp_czsl_v1_init_only_utzap_l14_seed1"
  "lhp_czsl_v1_init_only_l14_utzap_seed2:lhp_czsl_v1_init_only_utzap_l14_seed2"
  "lhp_czsl_v3_text_l14_utzap_seed0:lhp_czsl_v3_text_utzap_l14_seed0"
  "lhp_czsl_v3_text_l14_utzap_seed1:lhp_czsl_v3_text_utzap_l14_seed1"
  "lhp_czsl_v3_text_l14_utzap_seed2:lhp_czsl_v3_text_utzap_l14_seed2"
)

for entry in "${runs[@]}"; do
  ckpt_dir="${entry%%:*}"
  yml_stem="${entry##*:}"
  ckpt="checkpoint/${ckpt_dir}/val_best.pt"
  yml="config/${yml_stem}.yml"
  json="checkpoint/${ckpt_dir}/val_best.closed.json"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] missing $ckpt"
    continue
  fi
  echo "=== $ckpt_dir ==="
  $PY test.py --yml_path "$yml" --load_model "$ckpt" --open_world False 2>&1 | tail -1
  if [[ -f "$json" ]]; then
    $PY -c "import json; d=json.load(open('$json'))['test']; print(f'  test: HM={d[\"best_hm\"]:.4f} AUC={d[\"AUC\"]:.4f} seen={d[\"best_seen\"]:.4f} unseen={d[\"best_unseen\"]:.4f}')"
  fi
done
echo "DONE"
