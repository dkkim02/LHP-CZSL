"""H4 fine sweep + multi-checkpoint ensemble using cached logits.

Reads cache from checkpoint/ensemble_cache_mit_l14_v2/seed{0,1,2}_{val,test}.pt
- Expanded H4 grid (pair_w < 1, temperature < 1)
- Optional multi-checkpoint ensemble (last K epochs of each seed)
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dataset import CompositionDataset
from parameters import parser as base_parser
from utils import load_args
import test as test_module

SEEDS = [0, 1, 2]
CACHE_DIR = os.path.join(ROOT, "checkpoint", "ensemble_cache_mit_l14_v2")
YML = os.path.join(ROOT, "config", "cluspro_baseline_mit_l14_v2_seed0.yml")


def load_config():
    args = base_parser.parse_args([])
    args.yml_path = YML
    load_args(YML, args)
    return args


def apply_logit_infer(comp, attr, obj, pairs_cpu, pair_w=1.0, attr_w=1.0, obj_w=1.0, T_a=1.0, T_o=1.0):
    attr_pred = torch.softmax(attr / T_a, dim=-1)
    obj_pred = torch.softmax(obj / T_o, dim=-1)
    pairs_attr = pairs_cpu[:, 0]
    pairs_obj = pairs_cpu[:, 1]
    w_attr = attr_pred[:, pairs_attr] * attr_w
    w_obj = obj_pred[:, pairs_obj] * obj_w
    return comp * pair_w + w_attr * w_obj


def eval_combined(dataset, all_logits, attr_gt, obj_gt, pair_gt, config):
    evaluator = test_module.Evaluator(dataset, model=None)
    return test_module.test(dataset, evaluator, all_logits, attr_gt, obj_gt, pair_gt, config)


def fmt(stats):
    return " | ".join(f"{k} {stats[k]:.4f}" for k in ["best_seen", "best_unseen", "best_hm", "AUC"])


def main():
    config = load_config()

    val_dataset = CompositionDataset(config.dataset_path, phase="val",
                                     split="compositional-split-natural", open_world=False)
    test_dataset = CompositionDataset(config.dataset_path, phase="test",
                                      split="compositional-split-natural", open_world=False)

    attr2idx = val_dataset.attr2idx
    obj2idx = val_dataset.obj2idx
    pairs_cpu = torch.tensor([(attr2idx[a], obj2idx[o]) for a, o in val_dataset.pairs])

    # Load cached raw logits.
    per = {}
    for s in SEEDS:
        per[s] = {
            "val": torch.load(os.path.join(CACHE_DIR, f"seed{s}_val.pt"), map_location="cpu"),
            "test": torch.load(os.path.join(CACHE_DIR, f"seed{s}_test.pt"), map_location="cpu"),
        }

    def avg_raw(split, seed_subset=SEEDS):
        out = {}
        for k in ["comp", "attr", "obj"]:
            out[k] = torch.stack([per[s][split][k] for s in seed_subset], dim=0).mean(dim=0)
        out["attr_gt"] = per[seed_subset[0]][split]["attr_gt"]
        out["obj_gt"] = per[seed_subset[0]][split]["obj_gt"]
        out["pair_gt"] = per[seed_subset[0]][split]["pair_gt"]
        return out

    ens_val = avg_raw("val")
    ens_test = avg_raw("test")

    # Stage A: focused grid around 1.0 with small deviations + temperature.
    # If signal appears, expand. Total = 4 * 3 * 3 * 3 * 3 = 324 configs.
    PAIR_GRID = [0.1, 0.5, 1.0, 2.0]
    ATTR_GRID = [0.5, 1.0, 5.0]
    OBJ_GRID  = [0.5, 1.0, 5.0]
    TA_GRID   = [0.5, 1.0, 2.0]
    TO_GRID   = [0.5, 1.0, 2.0]

    total = len(PAIR_GRID) * len(ATTR_GRID) * len(OBJ_GRID) * len(TA_GRID) * len(TO_GRID)
    print(f"[i] fine grid total: {total} configs")

    def sweep(d_val, d_test, tag):
        best = {"val_hm": -1, "cfg": None, "val": None, "test": None}
        all_rows = []
        pbar = tqdm(total=total, desc=f"sweep[{tag}]")
        for pw in PAIR_GRID:
            for aw in ATTR_GRID:
                for ow in OBJ_GRID:
                    # Skip degenerate: pair_w=0 AND (attr_w=0 OR obj_w=0)
                    if pw == 0 and (aw == 0 or ow == 0):
                        for _ in TA_GRID:
                            for _ in TO_GRID:
                                pbar.update(1)
                        continue
                    for ta in TA_GRID:
                        for to_ in TO_GRID:
                            lv = apply_logit_infer(d_val["comp"], d_val["attr"], d_val["obj"], pairs_cpu,
                                                   pw, aw, ow, ta, to_)
                            sv = eval_combined(val_dataset, lv, d_val["attr_gt"], d_val["obj_gt"], d_val["pair_gt"], config)
                            row = {"cfg": (pw, aw, ow, ta, to_), "val_hm": sv["best_hm"], "val_AUC": sv["AUC"]}
                            all_rows.append(row)
                            if sv["best_hm"] > best["val_hm"]:
                                lt = apply_logit_infer(d_test["comp"], d_test["attr"], d_test["obj"], pairs_cpu,
                                                       pw, aw, ow, ta, to_)
                                st = eval_combined(test_dataset, lt, d_test["attr_gt"], d_test["obj_gt"], d_test["pair_gt"], config)
                                best = {"val_hm": sv["best_hm"], "cfg": (pw, aw, ow, ta, to_), "val": sv, "test": st}
                            pbar.update(1)
        pbar.close()
        return best, all_rows

    print("\n=== H4-fine on ensemble (avg raw logits) ===")
    best_ens, rows_ens = sweep(ens_val, ens_test, "ens")
    print(f"[H4-fine ens] best val cfg: {best_ens['cfg']}")
    print(f"[H4-fine ens] VAL  {fmt(best_ens['val'])}")
    print(f"[H4-fine ens] TEST {fmt(best_ens['test'])}")

    # Top-10 val configs from sweep — see if there's structure
    rows_sorted = sorted(rows_ens, key=lambda r: -r["val_hm"])[:10]
    print("\n[H4-fine ens] top-10 val configs:")
    for i, r in enumerate(rows_sorted):
        print(f"  #{i+1} cfg={r['cfg']} val_hm={r['val_hm']:.4f} val_AUC={r['val_AUC']:.4f}")

    out = {
        "h4_fine_ensemble": {"cfg": best_ens["cfg"], "val": best_ens["val"], "test": best_ens["test"]},
        "top10_val": rows_sorted,
        "grid": {"PAIR_GRID": PAIR_GRID, "ATTR_GRID": ATTR_GRID, "OBJ_GRID": OBJ_GRID,
                 "TA_GRID": TA_GRID, "TO_GRID": TO_GRID},
    }
    with open(os.path.join(CACHE_DIR, "results_fine.json"), "w") as fp:
        json.dump(out, fp, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


if __name__ == "__main__":
    main()
