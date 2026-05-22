"""Multi-checkpoint x multi-seed ensemble (extension of §17 H2).

For each seed in {0,1,2}, forward the top-K val-HM epochs (selected from the
training log), cache raw (comp, attr, obj) logits per (seed, epoch), then:
  - report per-ckpt individual stats (sanity).
  - report within-seed K-ckpt average (intra-seed diagnostic).
  - report 3K-ckpt all-pairs average (the main candidate).
  - keep default logit_infer weights (H4 axis dead per §17-5).

Compare against H2 (3-seed val_best.pt only) which lives at TEST HM 0.3924.
Pre-registered cutoff for axis pass: +0.005 over H2 -> TEST HM >= 0.3974.

Run: CUDA_VISIBLE_DEVICES=1 python scripts/eval_multi_ckpt.py
"""
import copy
import json
import os
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

# Per-seed top-3 val-HM epochs (from training logs 5-15).
# seed 0: ep11 0.4267, ep12 0.4264, ep13 0.4252
# seed 1: ep13 0.4275, ep11 0.4272, ep12 0.4265
# seed 2: ep14 0.4243, ep13 0.4225, ep11 0.4210  (epoch_15.pt not saved; epochs=15 ran 0..14)
SEED_EPOCHS = {
    0: [11, 12, 13],
    1: [13, 11, 12],
    2: [14, 13, 11],
}

CKPT_ROOT = os.path.join(ROOT, "checkpoint")
CKPT_TMPL = os.path.join(CKPT_ROOT, "cluspro_baseline_l14_mit_v2_seed{seed}", "epoch_{epoch}.pt")
YML_TMPL = os.path.join(ROOT, "config", "cluspro_baseline_mit_l14_v2_seed{seed}.yml")
CACHE_DIR = os.path.join(CKPT_ROOT, "ensemble_cache_mit_l14_v2")


def load_config_from_yml(yml_path):
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


def fmt_stats(stats):
    keys = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    return " | ".join(f"{k} {stats[k]:.4f}" for k in keys)


def avg_raw(cache_map, keys):
    """Average raw logits across (seed, epoch) keys."""
    avg = {}
    for k in ["comp", "attr", "obj"]:
        avg[k] = torch.stack([cache_map[key][k] for key in keys], dim=0).mean(dim=0)
    first = cache_map[keys[0]]
    avg["attr_gt"] = first["attr_gt"]
    avg["obj_gt"] = first["obj_gt"]
    avg["pair_gt"] = first["pair_gt"]
    return avg


def main():
    t0 = time.time()
    print(f"[i] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"[i] cache dir: {CACHE_DIR}")
    print(f"[i] SEED_EPOCHS = {SEED_EPOCHS}")
    os.makedirs(CACHE_DIR, exist_ok=True)

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
    print(f"[i] # pairs: {pairs_cpu.shape[0]} | # val: {len(val_dataset)} | # test: {len(test_dataset)}")

    # ----- Step 1: forward (or load cache) per (seed, epoch) ----------------
    cache_val = {}   # (seed, epoch) -> dict of tensors on val
    cache_test = {}  # (seed, epoch) -> dict of tensors on test
    for seed, epochs in SEED_EPOCHS.items():
        for ep in epochs:
            key = (seed, ep)
            cv_path = os.path.join(CACHE_DIR, f"seed{seed}_ep{ep}_val.pt")
            ct_path = os.path.join(CACHE_DIR, f"seed{seed}_ep{ep}_test.pt")
            if os.path.exists(cv_path) and os.path.exists(ct_path):
                print(f"[seed {seed} ep {ep}] cache hit, loading...")
                v = torch.load(cv_path, map_location="cpu")
                t = torch.load(ct_path, map_location="cpu")
            else:
                ckpt_path = CKPT_TMPL.format(seed=seed, epoch=ep)
                print(f"[seed {seed} ep {ep}] loading {ckpt_path}")
                cfg_s = copy.deepcopy(config)
                cfg_s.seed = seed
                set_seed(seed)
                model = get_model(cfg_s, attributes=attributes, classes=classes, offset=offset).cuda()
                state = torch.load(ckpt_path, map_location="cuda")
                model.load_state_dict(state)
                v = collect_raw_logits(model, val_dataset, pairs_gpu, config.eval_batch_size, config.num_workers, desc=f"s{seed}e{ep} val")
                t = collect_raw_logits(model, test_dataset, pairs_gpu, config.eval_batch_size, config.num_workers, desc=f"s{seed}e{ep} test")
                torch.save(v, cv_path)
                torch.save(t, ct_path)
                del model
                torch.cuda.empty_cache()
            cache_val[key] = v
            cache_test[key] = t

    # ----- Step 2: per-ckpt individual stats (sanity) -----------------------
    print("\n=== H0: per-ckpt individual stats ===")
    indiv = {}
    for key in cache_val:
        seed, ep = key
        lv = apply_logit_infer(cache_val[key]["comp"], cache_val[key]["attr"], cache_val[key]["obj"], pairs_cpu)
        lt = apply_logit_infer(cache_test[key]["comp"], cache_test[key]["attr"], cache_test[key]["obj"], pairs_cpu)
        sv = eval_combined(val_dataset, lv, cache_val[key]["attr_gt"], cache_val[key]["obj_gt"], cache_val[key]["pair_gt"], config)
        st = eval_combined(test_dataset, lt, cache_test[key]["attr_gt"], cache_test[key]["obj_gt"], cache_test[key]["pair_gt"], config)
        indiv[f"seed{seed}_ep{ep}"] = {"val": sv, "test": st}
        print(f"[s{seed} ep{ep}] VAL  {fmt_stats(sv)}")
        print(f"[s{seed} ep{ep}] TEST {fmt_stats(st)}")

    # ----- Step 3: within-seed K-ckpt average -------------------------------
    print("\n=== M1: within-seed K-ckpt average (intra-seed ensemble) ===")
    within = {}
    for seed, epochs in SEED_EPOCHS.items():
        keys = [(seed, ep) for ep in epochs]
        ens_v = avg_raw(cache_val, keys)
        ens_t = avg_raw(cache_test, keys)
        lv = apply_logit_infer(ens_v["comp"], ens_v["attr"], ens_v["obj"], pairs_cpu)
        lt = apply_logit_infer(ens_t["comp"], ens_t["attr"], ens_t["obj"], pairs_cpu)
        sv = eval_combined(val_dataset, lv, ens_v["attr_gt"], ens_v["obj_gt"], ens_v["pair_gt"], config)
        st = eval_combined(test_dataset, lt, ens_t["attr_gt"], ens_t["obj_gt"], ens_t["pair_gt"], config)
        within[f"seed{seed}_top{len(epochs)}"] = {"val": sv, "test": st}
        print(f"[within s{seed} top{len(epochs)}] VAL  {fmt_stats(sv)}")
        print(f"[within s{seed} top{len(epochs)}] TEST {fmt_stats(st)}")

    # ----- Step 4: full 3K-ckpt all-pairs ensemble --------------------------
    print("\n=== M2: full multi-ckpt x multi-seed ensemble (main) ===")
    all_keys = [(s, e) for s, eps in SEED_EPOCHS.items() for e in eps]
    print(f"[M2] averaging {len(all_keys)} ckpts: {all_keys}")
    full_v = avg_raw(cache_val, all_keys)
    full_t = avg_raw(cache_test, all_keys)
    lv = apply_logit_infer(full_v["comp"], full_v["attr"], full_v["obj"], pairs_cpu)
    lt = apply_logit_infer(full_t["comp"], full_t["attr"], full_t["obj"], pairs_cpu)
    sv_full = eval_combined(val_dataset, lv, full_v["attr_gt"], full_v["obj_gt"], full_v["pair_gt"], config)
    st_full = eval_combined(test_dataset, lt, full_t["attr_gt"], full_t["obj_gt"], full_t["pair_gt"], config)
    print(f"[M2 full {len(all_keys)}ckpt] VAL  {fmt_stats(sv_full)}")
    print(f"[M2 full {len(all_keys)}ckpt] TEST {fmt_stats(st_full)}")

    # ----- Step 5: top-1-per-seed (== H2 replica, 3 ckpts) ------------------
    print("\n=== M0: top-1-per-seed (H2 replica, 3 ckpts) ===")
    top1_keys = [(s, eps[0]) for s, eps in SEED_EPOCHS.items()]
    print(f"[M0] keys: {top1_keys}")
    top1_v = avg_raw(cache_val, top1_keys)
    top1_t = avg_raw(cache_test, top1_keys)
    lv = apply_logit_infer(top1_v["comp"], top1_v["attr"], top1_v["obj"], pairs_cpu)
    lt = apply_logit_infer(top1_t["comp"], top1_t["attr"], top1_t["obj"], pairs_cpu)
    sv_top1 = eval_combined(val_dataset, lv, top1_v["attr_gt"], top1_v["obj_gt"], top1_v["pair_gt"], config)
    st_top1 = eval_combined(test_dataset, lt, top1_t["attr_gt"], top1_t["obj_gt"], top1_t["pair_gt"], config)
    print(f"[M0 top1] VAL  {fmt_stats(sv_top1)}")
    print(f"[M0 top1] TEST {fmt_stats(st_top1)}")

    # ----- Step 6: top-2-per-seed (6 ckpts) ---------------------------------
    print("\n=== M1.5: top-2-per-seed (6 ckpts) ===")
    top2_keys = [(s, ep) for s, eps in SEED_EPOCHS.items() for ep in eps[:2]]
    print(f"[M1.5] keys: {top2_keys}")
    top2_v = avg_raw(cache_val, top2_keys)
    top2_t = avg_raw(cache_test, top2_keys)
    lv = apply_logit_infer(top2_v["comp"], top2_v["attr"], top2_v["obj"], pairs_cpu)
    lt = apply_logit_infer(top2_t["comp"], top2_t["attr"], top2_t["obj"], pairs_cpu)
    sv_top2 = eval_combined(val_dataset, lv, top2_v["attr_gt"], top2_v["obj_gt"], top2_v["pair_gt"], config)
    st_top2 = eval_combined(test_dataset, lt, top2_t["attr_gt"], top2_t["obj_gt"], top2_t["pair_gt"], config)
    print(f"[M1.5 top2] VAL  {fmt_stats(sv_top2)}")
    print(f"[M1.5 top2] TEST {fmt_stats(st_top2)}")

    # ----- Step 7: dump results JSON ----------------------------------------
    out = {
        "SEED_EPOCHS": {str(k): v for k, v in SEED_EPOCHS.items()},
        "individual": indiv,
        "within_seed": within,
        "m0_top1_per_seed": {"keys": top1_keys, "val": sv_top1, "test": st_top1},
        "m1_5_top2_per_seed": {"keys": top2_keys, "val": sv_top2, "test": st_top2},
        "m2_top3_per_seed": {"keys": all_keys, "val": sv_full, "test": st_full},
        "h2_baseline_reference": {"test_hm": 0.3924, "cutoff_pass": 0.3974},
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(CACHE_DIR, "results_multickpt.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (
            [list(t) if isinstance(t, tuple) else t for t in o] if isinstance(o, list) else str(o)))
    print(f"\n[i] wrote {out_path}")
    print(f"[i] total elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
