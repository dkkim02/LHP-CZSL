"""
probe_infer_calib.py — Inference-only calibration probe (NO training).

PART A spec: 3 scoring methods on 3 seeds of ClusProBaseline (MIT-States, ViT-L/14).
  (i)  BASELINE   — exact logit_infer reproduction (sanity gate ±0.001 vs stored JSON)
  (ii) RANK-1     — comp-debias with/without within-block centering control
  (iii) IDEA-5    — disagreement gate

Usage:
  cd /data1/workspaces/jgshin22/LHP-CZSL
  CUDA_VISIBLE_DEVICES=0 python tools/harness/probe_infer_calib.py --seed 0
  CUDA_VISIBLE_DEVICES=0 python tools/harness/probe_infer_calib.py --seed 1
  CUDA_VISIBLE_DEVICES=0 python tools/harness/probe_infer_calib.py --seed 2
  CUDA_VISIBLE_DEVICES=0 python tools/harness/probe_infer_calib.py --all
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import hmean
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------- repo root on sys.path ----------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args
from test import Evaluator, test   # reuse existing evaluator + test()

# ---------------------------------------------------------------------------
# CHECKPOINT REGISTRY
# ---------------------------------------------------------------------------
SEED_CONFIGS = {
    0: {
        "yml":  "config/cluspro_baseline_mit_l14_v2_ot_seed0.yml",
        "ckpt": "checkpoint/cluspro_baseline_l14_mit_v2_ot_seed0/val_best.pt",
        "stored_json": "checkpoint/cluspro_baseline_l14_mit_v2_ot_seed0/val_best.closed.json",
    },
    1: {
        "yml":  "config/cluspro_baseline_mit_l14_v2_seed1.yml",
        "ckpt": "checkpoint/cluspro_baseline_l14_mit_v2_seed1/val_best.pt",
        "stored_json": None,
    },
    2: {
        "yml":  "config/cluspro_baseline_mit_l14_v2_seed2.yml",
        "ckpt": "checkpoint/cluspro_baseline_l14_mit_v2_seed2/val_best.pt",
        "stored_json": None,
    },
}

OUTPUT_DIR = os.path.join(REPO_ROOT, "tools/harness")


# ---------------------------------------------------------------------------
# FORWARD PASS: collect raw (COMP, ATTR, OBJ) tensors — ONE pass per split
# ---------------------------------------------------------------------------
def collect_raw_logits(model, dataset, config):
    """
    Run val_forward in eval mode.  Returns:
        COMP  [N, P]  — raw composition logits
        ATTR  [N, |A|] — raw attribute logits
        OBJ   [N, |O|] — raw object logits
        attr_gt, obj_gt, pair_gt — [N] ground-truth indices
        pairs  [P, 2] — LongTensor of pair indices (attr_idx, obj_idx)
    """
    model.eval()
    attr2idx = dataset.attr2idx
    obj2idx  = dataset.obj2idx

    pairs = torch.tensor(
        [(attr2idx[a], obj2idx[o]) for a, o in dataset.pairs]
    ).cuda()  # [P, 2]

    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=getattr(config, "num_workers", 4),
    )

    all_comp, all_attr, all_obj = [], [], []
    all_attr_gt, all_obj_gt, all_pair_gt = [], [], []

    with torch.no_grad():
        for _, data in tqdm(enumerate(loader), total=len(loader), desc="Forward"):
            comp, attr, obj = model.val_forward(data, pairs)
            all_comp.append(comp.cpu())
            all_attr.append(attr.cpu())
            all_obj.append(obj.cpu())
            all_attr_gt.append(data[1])
            all_obj_gt.append(data[2])
            all_pair_gt.append(data[3])

    COMP   = torch.cat(all_comp,  dim=0)   # [N, P]
    ATTR   = torch.cat(all_attr,  dim=0)   # [N, |A|]
    OBJ    = torch.cat(all_obj,   dim=0)   # [N, |O|]
    attr_gt = torch.cat(all_attr_gt).cpu()
    obj_gt  = torch.cat(all_obj_gt).cpu()
    pair_gt = torch.cat(all_pair_gt).cpu()

    return COMP, ATTR, OBJ, attr_gt, obj_gt, pair_gt, pairs.cpu()


# ---------------------------------------------------------------------------
# SCORER: BASELINE (exact logit_infer reproduction)
# ---------------------------------------------------------------------------
def score_baseline(COMP, ATTR, OBJ, pairs, pair_inf_w, attr_inf_w, obj_inf_w):
    """
    Reproduce logit_infer exactly without mutating in place.
    pairs: [P, 2]  — int64 on CPU

    logit_infer:
        attr_pred = softmax(ATTR,-1)
        obj_pred  = softmax(OBJ,-1)
        for i in range(P):
            w_attr = 1 if attr_inf_w==0 else attr_pred[:, pairs[i][0]] * attr_inf_w
            w_obj  = 1 if obj_inf_w==0  else obj_pred[:,  pairs[i][1]] * obj_inf_w
            comp_logits[:, i] = comp_logits[:, i] * pair_inf_w + w_attr * w_obj
    """
    pa = pairs[:, 0]   # [P]
    po = pairs[:, 1]   # [P]

    attr_pred = F.softmax(ATTR, dim=-1)   # [N, |A|]
    obj_pred  = F.softmax(OBJ,  dim=-1)   # [N, |O|]

    w_attr = attr_pred[:, pa]   # [N, P]
    w_obj  = obj_pred[:, po]    # [N, P]

    if attr_inf_w == 0:
        w_attr_scaled = torch.ones_like(w_attr)
    else:
        w_attr_scaled = w_attr * attr_inf_w

    if obj_inf_w == 0:
        w_obj_scaled = torch.ones_like(w_obj)
    else:
        w_obj_scaled = w_obj * obj_inf_w

    S = COMP * pair_inf_w + w_attr_scaled * w_obj_scaled
    return S   # [N, P]


# ---------------------------------------------------------------------------
# SCORER: RANK-1 comp-debias (raw + centered)
# ---------------------------------------------------------------------------
def compute_train_prior(model, train_dataset, config):
    """
    Compute c(p) = softmax(COMP_train,-1).mean(0) from the TRAIN split.
    Returns:
        c_train          [P_train]   — mean softmax probability per train pair
        train_pair_to_idx  dict       — (attr,obj) -> index in train pairs list
        train_pairs_list   list       — ordered train pairs
    """
    model.eval()
    attr2idx = train_dataset.attr2idx
    obj2idx  = train_dataset.obj2idx
    pairs = torch.tensor(
        [(attr2idx[a], obj2idx[o]) for a, o in train_dataset.pairs]
    ).cuda()

    loader = DataLoader(
        train_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=getattr(config, "num_workers", 4),
    )

    all_comp_softmax = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader), total=len(loader), desc="Train fwd (prior)"):
            comp, _, _ = model.val_forward(data, pairs)
            all_comp_softmax.append(F.softmax(comp, dim=-1).cpu())

    COMP_train_soft = torch.cat(all_comp_softmax, dim=0)  # [N_tr, P_train]
    c_train = COMP_train_soft.mean(0)                      # [P_train]

    train_pair_to_idx = {p: i for i, p in enumerate(train_dataset.pairs)}

    return c_train, train_pair_to_idx, train_dataset.pairs


def build_rank1_scores(COMP, ATTR, OBJ, pairs, pair_inf_w,
                       c_train, train_pair_to_idx, train_pairs_list,
                       eval_pairs_list,
                       seen_mask, closed_mask, tau):
    """
    Build S_r1_raw and S_r1_ctr for a given tau.
      off_raw = -tau * log(c(p) + 1e-12)
      off_ctr = off_raw with within-block (seen / unseen_closed) mean removed

    pairs [P,2] aligned to eval_pairs_list.
    c_train [P_train] = per-train-pair prior (softmax mean).
    """
    pa = pairs[:, 0]
    po = pairs[:, 1]
    P  = pairs.shape[0]

    attr_pred = F.softmax(ATTR, dim=-1)
    obj_pred  = F.softmax(OBJ,  dim=-1)
    w_attr = attr_pred[:, pa]
    w_obj  = obj_pred[:, po]
    # baseline additive ao component (no extra pair_inf_w scaling — matches baseline)
    ao_term = w_attr * w_obj  # [N,P]

    # Map eval pairs -> train prior c(p)
    # For pairs absent from train: use uniform 1/P_train
    P_train = len(train_pairs_list)
    uniform_c = 1.0 / P_train if P_train > 0 else 1e-12

    c_eval = torch.zeros(P, dtype=torch.float32)
    for i, p in enumerate(eval_pairs_list):
        if p in train_pair_to_idx:
            c_eval[i] = c_train[train_pair_to_idx[p]]
        else:
            c_eval[i] = uniform_c

    logc = torch.log(c_eval + 1e-12)  # [P]

    # off_raw = -tau * logc
    off_raw = -tau * logc  # [P]

    # off_ctr: seen block -= mean(seen), unseen_closed block -= mean(unseen_closed)
    unseen_closed = closed_mask & ~seen_mask
    off_ctr = off_raw.clone()
    if seen_mask.any():
        off_ctr[seen_mask] = off_raw[seen_mask] - off_raw[seen_mask].mean()
    if unseen_closed.any():
        off_ctr[unseen_closed] = off_raw[unseen_closed] - off_raw[unseen_closed].mean()

    S_r1_raw = (COMP + off_raw[None]) * pair_inf_w + ao_term
    S_r1_ctr = (COMP + off_ctr[None]) * pair_inf_w + ao_term

    return S_r1_raw, S_r1_ctr


# ---------------------------------------------------------------------------
# SCORER: IDEA-5 disagreement gate
# ---------------------------------------------------------------------------
def score_idea5(COMP, ATTR, OBJ, pairs, pair_inf_w, kappa):
    """
    disagree[n,p] = (max_attr_conf[n] - w_attr[n,p]) + (max_obj_conf[n] - w_obj[n,p])
    gate[n,p]     = sigmoid(-kappa * disagree[n,p])
    S[n,p]        = COMP[n,p] * pair_inf_w * gate[n,p] + w_attr[n,p]*w_obj[n,p]
    """
    pa = pairs[:, 0]
    po = pairs[:, 1]

    attr_pred = F.softmax(ATTR, dim=-1)
    obj_pred  = F.softmax(OBJ,  dim=-1)

    # max confidence for each sample
    max_attr = attr_pred.max(-1).values[:, None]   # [N,1]
    max_obj  = obj_pred.max(-1).values[:, None]    # [N,1]

    w_attr = attr_pred[:, pa]  # [N,P]
    w_obj  = obj_pred[:, po]   # [N,P]

    disagree = (max_attr - w_attr) + (max_obj - w_obj)  # [N,P], ≥0
    gate = torch.sigmoid(-kappa * disagree)               # [N,P]

    S = COMP * pair_inf_w * gate + w_attr * w_obj
    return S


# ---------------------------------------------------------------------------
# EVAL HELPER: run test() on a score matrix S [N,P]
# ---------------------------------------------------------------------------
def eval_scores(S, dataset, evaluator, attr_gt, obj_gt, pair_gt):
    """
    Feed S [N,P] into test() and return stats dict.
    test() expects all_logits [N,P] where column i corresponds to dataset.pairs[i].
    """
    stats = test(dataset, evaluator, S.clone(), attr_gt, obj_gt, pair_gt, config=None)
    return stats


# ---------------------------------------------------------------------------
# MAIN per-seed function
# ---------------------------------------------------------------------------
def run_seed(seed_id):
    cfg_info = SEED_CONFIGS[seed_id]
    yml_path  = os.path.join(REPO_ROOT, cfg_info["yml"])
    ckpt_path = os.path.join(REPO_ROOT, cfg_info["ckpt"])

    print(f"\n{'='*70}")
    print(f"  SEED {seed_id}  |  {ckpt_path}")
    print(f"{'='*70}")

    # ---------- config ----------
    config = parser.parse_args([])
    load_args(yml_path, config)
    # override dataset_path to the repo-local copy (seed1/seed2 use remote path)
    config.dataset_path = os.path.join(REPO_ROOT, "data/mit-states")
    config.open_world   = False
    if not hasattr(config, "num_workers"):
        config.num_workers = 4

    # ---------- datasets ----------
    print("[datasets] loading val / test / train ...")
    val_dataset = CompositionDataset(
        config.dataset_path, phase="val",
        split="compositional-split-natural", open_world=False)
    test_dataset = CompositionDataset(
        config.dataset_path, phase="test",
        split="compositional-split-natural", open_world=False)
    train_dataset = CompositionDataset(
        config.dataset_path, phase="train",
        split="compositional-split-natural", open_world=False)

    allattrs   = val_dataset.attrs
    allobj     = val_dataset.objs
    classes    = [c.replace(".", " ").lower() for c in allobj]
    attributes = [a.replace(".", " ").lower() for a in allattrs]
    offset     = len(attributes)

    # ---------- model ----------
    print("[model] building and loading checkpoint ...")
    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    state = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state)
    model.eval()

    # ---------- inference weights ----------
    pair_inf_w = float(getattr(config, "pair_inference_weight", 1.0))
    attr_inf_w = float(getattr(config, "attr_inference_weight", 1.0))
    obj_inf_w  = float(getattr(config, "obj_inference_weight",  1.0))
    print(f"[config] pair_inf_w={pair_inf_w}  attr_inf_w={attr_inf_w}  obj_inf_w={obj_inf_w}")

    # ---------- evaluators ----------
    val_evaluator  = Evaluator(val_dataset,  model=None)
    test_evaluator = Evaluator(test_dataset, model=None)

    # ---------- forward pass: val + test ----------
    t0 = time.time()
    print("[forward] val split ...")
    VAL_COMP, VAL_ATTR, VAL_OBJ, val_attr_gt, val_obj_gt, val_pair_gt, val_pairs = \
        collect_raw_logits(model, val_dataset, config)
    print(f"  val done  ({time.time()-t0:.1f}s)  COMP{tuple(VAL_COMP.shape)}")

    t1 = time.time()
    print("[forward] test split ...")
    TST_COMP, TST_ATTR, TST_OBJ, tst_attr_gt, tst_obj_gt, tst_pair_gt, tst_pairs = \
        collect_raw_logits(model, test_dataset, config)
    print(f"  test done ({time.time()-t1:.1f}s)  COMP{tuple(TST_COMP.shape)}")

    # ================================================================
    # (i) BASELINE — reproduce logit_infer exactly
    # ================================================================
    print("\n[scorer] BASELINE ...")
    S_val_base = score_baseline(VAL_COMP, VAL_ATTR, VAL_OBJ, val_pairs,
                                pair_inf_w, attr_inf_w, obj_inf_w)
    S_tst_base = score_baseline(TST_COMP, TST_ATTR, TST_OBJ, tst_pairs,
                                pair_inf_w, attr_inf_w, obj_inf_w)

    val_stats_base = eval_scores(S_val_base, val_dataset,  val_evaluator,
                                 val_attr_gt, val_obj_gt, val_pair_gt)
    tst_stats_base = eval_scores(S_tst_base, test_dataset, test_evaluator,
                                 tst_attr_gt, tst_obj_gt, tst_pair_gt)

    val_hm_base  = val_stats_base["best_hm"]
    val_auc_base = val_stats_base["AUC"]
    tst_hm_base  = tst_stats_base["best_hm"]
    tst_auc_base = tst_stats_base["AUC"]

    print(f"  BASE  val HM={val_hm_base:.4f}  AUC={val_auc_base:.4f}")
    print(f"  BASE  tst HM={tst_hm_base:.4f}  AUC={tst_auc_base:.4f}")

    # --- SANITY GATE ---
    assert_base_matches = None
    if cfg_info["stored_json"]:
        stored_path = os.path.join(REPO_ROOT, cfg_info["stored_json"])
        with open(stored_path) as f:
            stored = json.load(f)
        stored_val_hm  = stored["val"]["best_hm"]
        stored_tst_hm  = stored["test"]["best_hm"]
        stored_val_auc = stored["val"]["AUC"]
        stored_tst_auc = stored["test"]["AUC"]

        val_hm_diff  = abs(val_hm_base  - stored_val_hm)
        tst_hm_diff  = abs(tst_hm_base  - stored_tst_hm)
        val_auc_diff = abs(val_auc_base - stored_val_auc)
        tst_auc_diff = abs(tst_auc_base - stored_tst_auc)

        print(f"\n  [SANITY GATE]")
        print(f"    stored val HM  = {stored_val_hm:.4f}  |  computed = {val_hm_base:.4f}  |  diff = {val_hm_diff:.4f}")
        print(f"    stored tst HM  = {stored_tst_hm:.4f}  |  computed = {tst_hm_base:.4f}  |  diff = {tst_hm_diff:.4f}")
        print(f"    stored val AUC = {stored_val_auc:.4f}  |  computed = {val_auc_base:.4f}  |  diff = {val_auc_diff:.4f}")
        print(f"    stored tst AUC = {stored_tst_auc:.4f}  |  computed = {tst_auc_base:.4f}  |  diff = {tst_auc_diff:.4f}")

        TOLERANCE = 0.001
        gate_pass = (val_hm_diff  <= TOLERANCE and
                     tst_hm_diff  <= TOLERANCE and
                     val_auc_diff <= TOLERANCE and
                     tst_auc_diff <= TOLERANCE)

        if gate_pass:
            assert_base_matches = True
            print(f"  *** SANITY GATE: PASS (all diffs <= {TOLERANCE}) ***")
        else:
            assert_base_matches = False
            print(f"  *** SANITY GATE: FAIL -- differences exceed +/-{TOLERANCE} ***")
            print("  STOPPING -- downstream methods are untrustworthy without a valid baseline.")
            result = {
                "seed": seed_id,
                "sanity_gate_passed": False,
                "assert_base_matches_stored": False,
                "base": {
                    "val_best_hm":     val_hm_base,
                    "val_AUC":         val_auc_base,
                    "test_best_hm":    tst_hm_base,
                    "test_AUC":        tst_auc_base,
                    "stored_val_hm":   stored_val_hm,
                    "stored_tst_hm":   stored_tst_hm,
                    "val_hm_diff":     val_hm_diff,
                    "tst_hm_diff":     tst_hm_diff,
                },
            }
            out_path = os.path.join(OUTPUT_DIR, f"probe_seed{seed_id}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n[output] saved {out_path}")
            return result
    else:
        print("  [SANITY GATE] no stored JSON for this seed -- skipping gate")

    # ================================================================
    # seen / unseen masks (aligned to each evaluator's dset.pairs)
    # ================================================================
    val_seen_mask   = val_evaluator.seen_mask    # [P_val]
    val_closed_mask = val_evaluator.closed_mask  # [P_val]
    tst_seen_mask   = test_evaluator.seen_mask
    tst_closed_mask = test_evaluator.closed_mask

    # ================================================================
    # Compute train prior for Rank-1
    # ================================================================
    print("\n[rank-1] computing train-split prior c(p) ...")
    c_train, train_pair_to_idx, train_pairs_list = compute_train_prior(
        model, train_dataset, config
    )

    # ================================================================
    # (ii) RANK-1: sweep tau on VAL using centered variant, freeze for test
    # ================================================================
    print("\n[scorer] RANK-1 tau sweep on val ...")
    tau_grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    best_tau_ctr    = None
    best_val_hm_ctr = -1.0

    tau_results = {}
    for tau in tau_grid:
        S_val_r1_raw, S_val_r1_ctr = build_rank1_scores(
            VAL_COMP, VAL_ATTR, VAL_OBJ, val_pairs, pair_inf_w,
            c_train, train_pair_to_idx, train_pairs_list,
            val_dataset.pairs,
            val_seen_mask, val_closed_mask, tau
        )
        st_raw = eval_scores(S_val_r1_raw, val_dataset, val_evaluator,
                             val_attr_gt, val_obj_gt, val_pair_gt)
        st_ctr = eval_scores(S_val_r1_ctr, val_dataset, val_evaluator,
                             val_attr_gt, val_obj_gt, val_pair_gt)
        tau_results[str(tau)] = {
            "raw_val_hm": float(st_raw["best_hm"]),
            "ctr_val_hm": float(st_ctr["best_hm"]),
        }
        print(f"  tau={tau:4.1f}  raw_val_hm={st_raw['best_hm']:.4f}  ctr_val_hm={st_ctr['best_hm']:.4f}")
        if st_ctr["best_hm"] > best_val_hm_ctr:
            best_val_hm_ctr = st_ctr["best_hm"]
            best_tau_ctr = tau

    print(f"  -> best tau (ctr, val) = {best_tau_ctr}")

    # Apply best tau to val + test
    S_val_r1_raw_best, S_val_r1_ctr_best = build_rank1_scores(
        VAL_COMP, VAL_ATTR, VAL_OBJ, val_pairs, pair_inf_w,
        c_train, train_pair_to_idx, train_pairs_list,
        val_dataset.pairs,
        val_seen_mask, val_closed_mask, best_tau_ctr
    )
    S_tst_r1_raw, S_tst_r1_ctr = build_rank1_scores(
        TST_COMP, TST_ATTR, TST_OBJ, tst_pairs, pair_inf_w,
        c_train, train_pair_to_idx, train_pairs_list,
        test_dataset.pairs,
        tst_seen_mask, tst_closed_mask, best_tau_ctr
    )

    val_r1_raw_stats = eval_scores(S_val_r1_raw_best, val_dataset, val_evaluator,
                                   val_attr_gt, val_obj_gt, val_pair_gt)
    val_r1_ctr_stats = eval_scores(S_val_r1_ctr_best, val_dataset, val_evaluator,
                                   val_attr_gt, val_obj_gt, val_pair_gt)
    tst_r1_raw_stats = eval_scores(S_tst_r1_raw, test_dataset, test_evaluator,
                                   tst_attr_gt, tst_obj_gt, tst_pair_gt)
    tst_r1_ctr_stats = eval_scores(S_tst_r1_ctr, test_dataset, test_evaluator,
                                   tst_attr_gt, tst_obj_gt, tst_pair_gt)

    print(f"  R1-raw val HM={val_r1_raw_stats['best_hm']:.4f}  tst HM={tst_r1_raw_stats['best_hm']:.4f}")
    print(f"  R1-ctr val HM={val_r1_ctr_stats['best_hm']:.4f}  tst HM={tst_r1_ctr_stats['best_hm']:.4f}")

    # ================================================================
    # (iii) IDEA-5: disagreement gate, sweep kappa on val
    # ================================================================
    print("\n[scorer] IDEA-5 kappa sweep on val ...")
    kappa_grid = [1.0, 2.0, 4.0, 8.0, 16.0]
    best_kappa      = None
    best_val_hm_id5 = -1.0

    kappa_results = {}
    for kappa in kappa_grid:
        S_val_id5 = score_idea5(VAL_COMP, VAL_ATTR, VAL_OBJ, val_pairs, pair_inf_w, kappa)
        st_id5 = eval_scores(S_val_id5, val_dataset, val_evaluator,
                             val_attr_gt, val_obj_gt, val_pair_gt)
        kappa_results[str(kappa)] = float(st_id5["best_hm"])
        print(f"  kappa={kappa:5.1f}  val_hm={st_id5['best_hm']:.4f}")
        if st_id5["best_hm"] > best_val_hm_id5:
            best_val_hm_id5 = st_id5["best_hm"]
            best_kappa = kappa

    print(f"  -> best kappa (val) = {best_kappa}")

    S_val_id5_best = score_idea5(VAL_COMP, VAL_ATTR, VAL_OBJ, val_pairs, pair_inf_w, best_kappa)
    S_tst_id5_best = score_idea5(TST_COMP, TST_ATTR, TST_OBJ, tst_pairs, pair_inf_w, best_kappa)

    val_id5_stats = eval_scores(S_val_id5_best, val_dataset, val_evaluator,
                                val_attr_gt, val_obj_gt, val_pair_gt)
    tst_id5_stats = eval_scores(S_tst_id5_best, test_dataset, test_evaluator,
                                tst_attr_gt, tst_obj_gt, tst_pair_gt)

    print(f"  ID5   val HM={val_id5_stats['best_hm']:.4f}  tst HM={tst_id5_stats['best_hm']:.4f}")

    # ================================================================
    # Collect results
    # ================================================================
    def make_row(val_st, tst_st, **extra):
        return {
            "val_best_hm":      float(val_st["best_hm"]),
            "val_AUC":          float(val_st["AUC"]),
            "test_best_hm":     float(tst_st["best_hm"]),
            "test_AUC":         float(tst_st["AUC"]),
            "test_best_seen":   float(tst_st["best_seen"]),
            "test_best_unseen": float(tst_st["best_unseen"]),
            **extra,
        }

    result = {
        "seed":                       seed_id,
        "assert_base_matches_stored": assert_base_matches,
        "base":   make_row(val_stats_base,   tst_stats_base),
        "r1_raw": make_row(val_r1_raw_stats, tst_r1_raw_stats, chosen_tau=best_tau_ctr),
        "r1_ctr": make_row(val_r1_ctr_stats, tst_r1_ctr_stats, chosen_tau=best_tau_ctr),
        "id5":    make_row(val_id5_stats,     tst_id5_stats,    chosen_kappa=best_kappa),
        "tau_sweep":   tau_results,
        "kappa_sweep": kappa_results,
    }

    out_path = os.path.join(OUTPUT_DIR, f"probe_seed{seed_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[output] saved {out_path}")

    # ================================================================
    # Print clean table for this seed
    # ================================================================
    print(f"\n{'─'*78}")
    print(f"  SEED {seed_id} RESULTS")
    print(f"{'─'*78}")
    hdr = f"  {'Method':<12}  {'valHM':>7}  {'tstHM':>7}  {'Dtst':>7}  {'valAUC':>8}  {'tstAUC':>8}"
    print(hdr)
    print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")
    for name, v_st, t_st in [
        ("base",   val_stats_base,   tst_stats_base),
        ("r1_raw", val_r1_raw_stats, tst_r1_raw_stats),
        ("r1_ctr", val_r1_ctr_stats, tst_r1_ctr_stats),
        ("id5",    val_id5_stats,    tst_id5_stats),
    ]:
        delta = t_st["best_hm"] - tst_hm_base
        print(f"  {name:<12}  {v_st['best_hm']:>7.4f}  {t_st['best_hm']:>7.4f}  {delta:>+7.4f}  "
              f"{v_st['AUC']:>8.4f}  {t_st['AUC']:>8.4f}")
    print(f"{'─'*78}")

    total_time = time.time() - t0
    print(f"  Wall time: {total_time/60:.1f} min")

    return result


# ---------------------------------------------------------------------------
# MULTI-SEED TABLE
# ---------------------------------------------------------------------------
def print_combined_table(all_results):
    methods = ["base", "r1_raw", "r1_ctr", "id5"]
    seeds   = sorted(all_results.keys())

    print(f"\n{'='*90}")
    print("  COMBINED 3-SEED TABLE")
    print(f"{'='*90}")
    header = f"  {'Method':<12}"
    for s in seeds:
        header += f"  S{s}_tstHM  S{s}_Delta"
    print(header)
    sep = f"  {'-'*12}"
    for _ in seeds:
        sep += f"  {'-'*9}  {'-'*8}"
    print(sep)

    for m in methods:
        row_str = f"  {m:<12}"
        for s in seeds:
            r = all_results[s]
            if m not in r:
                row_str += f"  {'N/A':>9}  {'N/A':>8}"
                continue
            base_hm = r["base"]["test_best_hm"]
            tst_hm  = r[m]["test_best_hm"]
            delta   = tst_hm - base_hm
            row_str += f"  {tst_hm:>9.4f}  {delta:>+8.4f}"
        print(row_str)

    print(f"{'='*90}")

    # PRE-REGISTERED DECISION
    print("\n  PRE-REGISTERED DECISION (REPORT ONLY -- YOU DECIDE):")
    print("  PROMOTE iff: seed0 Delta_test >= +0.005 AND seed1,seed2 Delta_test >= +0.003")
    print("               AND (Rank-1) centered variant carries the gain")
    print("  KILL    iff: seed0 Delta_test <= +0.002, or any seed regresses,")
    print("               or (Rank-1) only raw gains")
    print()

    for m in ["r1_raw", "r1_ctr", "id5"]:
        deltas = {}
        for s in seeds:
            r = all_results[s]
            if m not in r:
                continue
            deltas[s] = r[m]["test_best_hm"] - r["base"]["test_best_hm"]

        if not deltas:
            continue

        if 0 in deltas and deltas[0] <= 0.002:
            verdict = f"KILL (seed0 Delta={deltas.get(0,0):+.4f} <= +0.002)"
        elif any(d < 0 for d in deltas.values()):
            regress_seeds = [s for s, d in deltas.items() if d < 0]
            verdict = f"KILL (regression on seed(s) {regress_seeds})"
        elif (0 in deltas and deltas[0] >= 0.005 and
              all(deltas.get(s, 0) >= 0.003 for s in seeds if s != 0)):
            if m == "r1_raw":
                ctr_deltas = {s: all_results[s]["r1_ctr"]["test_best_hm"] - all_results[s]["base"]["test_best_hm"]
                              for s in seeds if "r1_ctr" in all_results[s]}
                if all(d >= 0.003 for d in ctr_deltas.values()):
                    verdict = "PROMOTE (raw; r1_ctr also gains -- see r1_ctr row)"
                else:
                    verdict = "KILL (r1_raw gains but r1_ctr does NOT -- bias-sweep artefact)"
            elif m == "r1_ctr":
                verdict = "PROMOTE (centered variant carries gain)"
            else:
                verdict = "PROMOTE"
        else:
            delta_strs = ", ".join(f"s{s}={d:+.4f}" for s, d in deltas.items())
            verdict = f"INCONCLUSIVE ({delta_strs})"

        print(f"  {m:<10}:  {verdict}")

    print()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=[0, 1, 2], help="run a single seed")
    ap.add_argument("--all",  action="store_true", help="run all 3 seeds sequentially")
    args = ap.parse_args()

    if args.all:
        seeds_to_run = [0, 1, 2]
    elif args.seed is not None:
        seeds_to_run = [args.seed]
    else:
        seeds_to_run = [0]

    all_results = {}

    # Load previously saved results for seeds we won't re-run
    for s in [0, 1, 2]:
        p = os.path.join(OUTPUT_DIR, f"probe_seed{s}.json")
        if os.path.exists(p) and s not in seeds_to_run:
            with open(p) as f:
                all_results[s] = json.load(f)
            print(f"[loaded] probe_seed{s}.json from disk")

    for s in seeds_to_run:
        result = run_seed(s)
        all_results[s] = result

    if len(all_results) >= 2:
        print_combined_table(all_results)
