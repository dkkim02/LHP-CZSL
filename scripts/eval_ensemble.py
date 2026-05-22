"""H2 + H4: Multi-seed ensemble + inference-weight/temperature calibration.

Loads val_best.pt from cluspro_baseline_l14_mit_v2_seed{0,1,2}, caches raw
(comp, attr, obj) logits on val and test, then:
  - H2: averages raw logits across seeds, applies default logit_infer, evaluates.
  - H4: sweeps (pair_w, attr_w, obj_w, T_a, T_o) on val, applies best to test.
  - Stacked: applies best H4 combo to averaged H2 logits.

Run: CUDA_VISIBLE_DEVICES=1 python scripts/eval_ensemble.py
"""
import argparse
import copy
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dataset import CompositionDataset
from model.model_factory import get_model
from parameters import parser as base_parser
from utils import load_args, set_seed
import test as test_module

cudnn.benchmark = True

SEEDS = [0, 1, 2]
CKPT_ROOT = os.path.join(ROOT, "checkpoint")
CKPT_TMPL = os.path.join(CKPT_ROOT, "cluspro_baseline_l14_mit_v2_seed{seed}", "val_best.pt")
YML_TMPL = os.path.join(ROOT, "config", "cluspro_baseline_mit_l14_v2_seed{seed}.yml")
CACHE_DIR = os.path.join(CKPT_ROOT, "ensemble_cache_mit_l14_v2")


def load_config_from_yml(yml_path):
    """Use parameters parser then overlay yml — replicates train.py flow."""
    args = base_parser.parse_args([])
    args.yml_path = yml_path
    load_args(yml_path, args)
    return args


def collect_raw_logits(model, dataset, pairs, eval_batch_size, num_workers, desc=""):
    model.eval()
    loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    comp_all, attr_all, obj_all = [], [], []
    attr_gt, obj_gt, pair_gt = [], [], []
    with torch.no_grad():
        for data in tqdm(loader, desc=desc, total=len(loader)):
            comp, a, o = model(data, pairs)
            comp_all.append(comp.detach().cpu().float())
            attr_all.append(a.detach().cpu().float())
            obj_all.append(o.detach().cpu().float())
            attr_gt.append(data[1])
            obj_gt.append(data[2])
            pair_gt.append(data[3])
    return {
        "comp": torch.cat(comp_all, dim=0),
        "attr": torch.cat(attr_all, dim=0),
        "obj": torch.cat(obj_all, dim=0),
        "attr_gt": torch.cat(attr_gt),
        "obj_gt": torch.cat(obj_gt),
        "pair_gt": torch.cat(pair_gt),
    }


def apply_logit_infer(comp, attr, obj, pairs_cpu, pair_w=1.0, attr_w=1.0, obj_w=1.0, T_a=1.0, T_o=1.0):
    """Vectorized version of model.logit_infer with sweepable weights/temps.

    Inputs are raw logits on CPU.
    pairs_cpu: LongTensor (P, 2) of (attr_idx, obj_idx) for ALL dataset.pairs.
    """
    attr_pred = torch.softmax(attr / T_a, dim=-1)  # (B, n_attr)
    obj_pred = torch.softmax(obj / T_o, dim=-1)    # (B, n_obj)
    pairs_attr = pairs_cpu[:, 0]
    pairs_obj = pairs_cpu[:, 1]
    w_attr = attr_pred[:, pairs_attr] * attr_w  # (B, P)
    w_obj = obj_pred[:, pairs_obj] * obj_w
    return comp * pair_w + w_attr * w_obj


def eval_combined(dataset, all_logits, attr_gt, obj_gt, pair_gt, config):
    evaluator = test_module.Evaluator(dataset, model=None)
    return test_module.test(dataset, evaluator, all_logits, attr_gt, obj_gt, pair_gt, config)


def fmt_stats(stats):
    keys = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    return " | ".join(f"{k} {stats[k]:.4f}" for k in keys)


def main():
    t0 = time.time()
    print(f"[i] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"[i] cache dir: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Use seed 0 config as base — only `seed` differs across runs.
    config = load_config_from_yml(YML_TMPL.format(seed=0))
    print(f"[i] dataset: {config.dataset} | clip: {config.clip_arch}")

    val_dataset = CompositionDataset(
        config.dataset_path, phase="val",
        split="compositional-split-natural", open_world=False,
    )
    test_dataset = CompositionDataset(
        config.dataset_path, phase="test",
        split="compositional-split-natural", open_world=False,
    )
    allattrs = val_dataset.attrs
    allobj = val_dataset.objs
    classes = [c.replace(".", " ").lower() for c in allobj]
    attributes = [a.replace(".", " ").lower() for a in allattrs]
    offset = len(attributes)

    attr2idx = val_dataset.attr2idx
    obj2idx = val_dataset.obj2idx
    pairs_list = [(attr2idx[a], obj2idx[o]) for a, o in val_dataset.pairs]
    pairs_gpu = torch.tensor(pairs_list).cuda()
    pairs_cpu = torch.tensor(pairs_list)
    print(f"[i] # pairs (all dataset): {pairs_cpu.shape[0]} | # val items: {len(val_dataset)} | # test items: {len(test_dataset)}")

    # ----- Step 1: collect (or load) raw logits per seed --------------------
    per_seed = {}
    for seed in SEEDS:
        cache_val = os.path.join(CACHE_DIR, f"seed{seed}_val.pt")
        cache_test = os.path.join(CACHE_DIR, f"seed{seed}_test.pt")
        if os.path.exists(cache_val) and os.path.exists(cache_test):
            print(f"[seed {seed}] cache hit, loading...")
            v = torch.load(cache_val, map_location="cpu")
            t = torch.load(cache_test, map_location="cpu")
        else:
            ckpt_path = CKPT_TMPL.format(seed=seed)
            print(f"[seed {seed}] loading model from {ckpt_path}")
            cfg_s = copy.deepcopy(config)
            cfg_s.seed = seed
            set_seed(seed)
            model = get_model(cfg_s, attributes=attributes, classes=classes, offset=offset).cuda()
            state = torch.load(ckpt_path, map_location="cuda")
            model.load_state_dict(state)
            v = collect_raw_logits(model, val_dataset, pairs_gpu, config.eval_batch_size, config.num_workers, desc=f"seed{seed} val")
            t = collect_raw_logits(model, test_dataset, pairs_gpu, config.eval_batch_size, config.num_workers, desc=f"seed{seed} test")
            torch.save(v, cache_val)
            torch.save(t, cache_test)
            del model
            torch.cuda.empty_cache()
        per_seed[seed] = {"val": v, "test": t}
        print(f"[seed {seed}] raw logits | comp {v['comp'].shape} | attr {v['attr'].shape} | obj {v['obj'].shape}")

    # ----- Step 2: single-seed sanity (apply default logit_infer) ------------
    print("\n=== H0: per-seed default (pair_w=attr_w=obj_w=1, T=1) — sanity vs §12-1 ===")
    per_seed_stats = {}
    for seed in SEEDS:
        d_val = per_seed[seed]["val"]
        d_test = per_seed[seed]["test"]
        logits_val = apply_logit_infer(d_val["comp"], d_val["attr"], d_val["obj"], pairs_cpu)
        logits_test = apply_logit_infer(d_test["comp"], d_test["attr"], d_test["obj"], pairs_cpu)
        s_val = eval_combined(val_dataset, logits_val, d_val["attr_gt"], d_val["obj_gt"], d_val["pair_gt"], config)
        s_test = eval_combined(test_dataset, logits_test, d_test["attr_gt"], d_test["obj_gt"], d_test["pair_gt"], config)
        per_seed_stats[seed] = {"val": s_val, "test": s_test}
        print(f"[seed {seed}] VAL  {fmt_stats(s_val)}")
        print(f"[seed {seed}] TEST {fmt_stats(s_test)}")

    # ----- Step 3: H2 ensemble (avg raw logits across seeds) -----------------
    print("\n=== H2: 3-seed ensemble (avg raw comp/attr/obj logits, default weights) ===")
    def avg_raw(split):
        avg = {}
        for k in ["comp", "attr", "obj"]:
            avg[k] = torch.stack([per_seed[s][split][k] for s in SEEDS], dim=0).mean(dim=0)
        # GT identical across seeds (deterministic dataloader)
        avg["attr_gt"] = per_seed[SEEDS[0]][split]["attr_gt"]
        avg["obj_gt"] = per_seed[SEEDS[0]][split]["obj_gt"]
        avg["pair_gt"] = per_seed[SEEDS[0]][split]["pair_gt"]
        return avg

    ens_val = avg_raw("val")
    ens_test = avg_raw("test")
    logits_ens_val = apply_logit_infer(ens_val["comp"], ens_val["attr"], ens_val["obj"], pairs_cpu)
    logits_ens_test = apply_logit_infer(ens_test["comp"], ens_test["attr"], ens_test["obj"], pairs_cpu)
    s_ens_val = eval_combined(val_dataset, logits_ens_val, ens_val["attr_gt"], ens_val["obj_gt"], ens_val["pair_gt"], config)
    s_ens_test = eval_combined(test_dataset, logits_ens_test, ens_test["attr_gt"], ens_test["obj_gt"], ens_test["pair_gt"], config)
    print(f"[H2 ensemble] VAL  {fmt_stats(s_ens_val)}")
    print(f"[H2 ensemble] TEST {fmt_stats(s_ens_test)}")

    # ----- Step 4: H4 sweep on per-seed-0 logits (cheap proxy for sweep target) ---
    print("\n=== H4: sweep (pair_w, attr_w, obj_w, T_a, T_o) on val — seed 0 raw logits ===")
    # Coarse first sweep — refine later if any combo wins.
    PAIR_GRID = [1.0, 2.0]
    ATTR_GRID = [1.0, 3.0, 10.0]
    OBJ_GRID  = [1.0, 3.0, 10.0]
    T_GRID    = [1.0]

    def sweep_on(d_val, d_test, label):
        best = {"val_hm": -1, "cfg": None, "val": None, "test": None}
        rows = []
        total = len(PAIR_GRID) * len(ATTR_GRID) * len(OBJ_GRID) * len(T_GRID) * len(T_GRID)
        pbar = tqdm(total=total, desc=f"sweep[{label}]")
        for pw in PAIR_GRID:
            for aw in ATTR_GRID:
                for ow in OBJ_GRID:
                    for ta in T_GRID:
                        for to_ in T_GRID:
                            lv = apply_logit_infer(d_val["comp"], d_val["attr"], d_val["obj"], pairs_cpu,
                                                   pair_w=pw, attr_w=aw, obj_w=ow, T_a=ta, T_o=to_)
                            sv = eval_combined(val_dataset, lv, d_val["attr_gt"], d_val["obj_gt"], d_val["pair_gt"], config)
                            rows.append((pw, aw, ow, ta, to_, sv["best_hm"], sv["AUC"]))
                            if sv["best_hm"] > best["val_hm"]:
                                # Also eval test at this point for reporting
                                lt = apply_logit_infer(d_test["comp"], d_test["attr"], d_test["obj"], pairs_cpu,
                                                       pair_w=pw, attr_w=aw, obj_w=ow, T_a=ta, T_o=to_)
                                st = eval_combined(test_dataset, lt, d_test["attr_gt"], d_test["obj_gt"], d_test["pair_gt"], config)
                                best = {"val_hm": sv["best_hm"], "cfg": (pw, aw, ow, ta, to_), "val": sv, "test": st}
                            pbar.update(1)
        pbar.close()
        return best, rows

    # H4 on seed 0 single
    best_seed0, rows_seed0 = sweep_on(per_seed[0]["val"], per_seed[0]["test"], "seed0")
    print(f"[H4 seed0] best val cfg (pair_w, attr_w, obj_w, T_a, T_o) = {best_seed0['cfg']}")
    print(f"[H4 seed0] VAL  {fmt_stats(best_seed0['val'])}")
    print(f"[H4 seed0] TEST {fmt_stats(best_seed0['test'])}")

    # ----- Step 5: stack H2 + H4 (apply best H4 cfg from val on ensemble val → test) ---
    print("\n=== H2+H4: sweep on ensemble val, apply to ensemble test ===")
    best_ens, rows_ens = sweep_on(ens_val, ens_test, "ensemble")
    print(f"[H2+H4] best val cfg (pair_w, attr_w, obj_w, T_a, T_o) = {best_ens['cfg']}")
    print(f"[H2+H4] VAL  {fmt_stats(best_ens['val'])}")
    print(f"[H2+H4] TEST {fmt_stats(best_ens['test'])}")

    # ----- Step 6: dump results JSON ----------------------------------------
    out = {
        "per_seed": {str(s): {"val": per_seed_stats[s]["val"], "test": per_seed_stats[s]["test"]} for s in SEEDS},
        "h2_ensemble": {"val": s_ens_val, "test": s_ens_test},
        "h4_seed0": {"cfg": best_seed0["cfg"], "val": best_seed0["val"], "test": best_seed0["test"]},
        "h2_plus_h4": {"cfg": best_ens["cfg"], "val": best_ens["val"], "test": best_ens["test"]},
        "sweep_grid": {
            "PAIR_GRID": PAIR_GRID, "ATTR_GRID": ATTR_GRID, "OBJ_GRID": OBJ_GRID,
            "T_GRID": T_GRID,
        },
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(CACHE_DIR, "results.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n[i] wrote {out_path}")
    print(f"[i] total elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
