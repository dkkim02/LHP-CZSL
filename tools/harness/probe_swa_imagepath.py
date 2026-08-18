"""
SCREEN 1 — IDEA F: BN-corrected image-path SWA offline screen (inference-only).

Pipeline:
  1. BASIN CHECK: load epoch_10..14.pt; identify image-path params by name
     (visual_adapters.*, attr_disentangler.*, obj_disentangler.*, attr_proj.*, obj_proj.*);
     report pairwise L2 distance of the concatenated image-path param vector vs param norm.
  2. BUILD SWA: base = val_best.pt (all params); OVERWRITE only image-path params with the
     mean over epoch_10..14. (Affine BN weight/bias ARE averaged; BN running stats are NOT
     averaged — they are recomputed in step 3.)
  3. BN RECOMPUTE (load-bearing): reset the 4 Disentangler BN running stats, set model.train(),
     run a few hundred TRAIN batches through ONLY the image encoder + disentangler chain under
     torch.no_grad() so BN running_mean/var re-estimate for the averaged weights. Prototype EMA
     queues are NEVER touched (we do not call train_forward / _update_prototypes).
  4. EVAL: reuse test.predict_logits + test.test() closed-world eval on val & test.
  5. FULL-SWA variant: average ALL trainable (non-clip, non-queue) params over epoch_10..14 +
     same BN recompute, for comparison.

Usage: CUDA_VISIBLE_DEVICES=3 python tools/harness/probe_swa_imagepath.py
"""

import os
import sys
import copy
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args
import test as testmod

CKPT_DIR = os.path.join(REPO_ROOT, "checkpoint/cluspro_baseline_l14_mit_v2_ot_seed0")
CFG = os.path.join(REPO_ROOT, "config/cluspro_baseline_mit_l14_v2_seed0.yml")
EPOCHS = [10, 11, 12, 13, 14]
SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"
BN_ITERS = 300  # train batches (batch=4) for BN re-estimation

IMG_PREFIXES = ("visual_adapters.", "attr_disentangler.", "obj_disentangler.",
                "attr_proj.", "obj_proj.")
# BN running statistics: reset+recompute, never average.
BN_STAT_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def is_img_path(k):
    return any(k.startswith(p) for p in IMG_PREFIXES)


def is_bn_stat(k):
    return any(k.endswith(s) for s in BN_STAT_SUFFIXES)


def is_queue(k):
    return ("attr_queue" in k) or ("obj_queue" in k)


def load_sd(ep):
    return torch.load(os.path.join(CKPT_DIR, f"epoch_{ep}.pt"), map_location="cpu")


def basin_check(sds):
    """Concatenate image-path FLOAT params (exclude BN stats / int counters) per epoch;
    report pairwise L2 distance and relative distance vs mean param-vector norm."""
    keys = [k for k in sds[EPOCHS[0]].keys()
            if is_img_path(k) and not is_bn_stat(k)
            and sds[EPOCHS[0]][k].dtype.is_floating_point]
    vecs = {}
    for ep in EPOCHS:
        vecs[ep] = torch.cat([sds[ep][k].float().flatten() for k in keys])
    norms = {ep: float(vecs[ep].norm()) for ep in EPOCHS}
    mean_norm = float(np.mean(list(norms.values())))
    print(f"\n[basin] image-path param vector: dim={vecs[EPOCHS[0]].numel()}  "
          f"mean L2 norm={mean_norm:.4f}")
    print("[basin] pairwise L2 distances (and as % of mean norm):")
    dists = {}
    for i, a in enumerate(EPOCHS):
        for b in EPOCHS[i + 1:]:
            d = float((vecs[a] - vecs[b]).norm())
            rel = 100.0 * d / mean_norm
            dists[f"{a}-{b}"] = dict(dist=d, rel_pct=rel)
            print(f"    e{a}-e{b}: {d:.4f}  ({rel:.2f}% of norm)")
    max_rel = max(v["rel_pct"] for v in dists.values())
    same_basin = max_rel < 25.0
    print(f"[basin] max pairwise dist = {max_rel:.2f}% of param norm -> "
          f"{'LIKELY SAME BASIN' if same_basin else 'POSSIBLY DIFFERENT BASINS (caution)'}")
    return dict(per_epoch_norm=norms, mean_norm=mean_norm, dists=dists,
                max_rel_pct=max_rel, same_basin=same_basin)


def build_averaged_sd(base_sd, sds, which):
    """which='imagepath' -> overwrite only image-path float params (excl BN stats) with mean;
       which='full'      -> overwrite ALL trainable float params (excl clip*, queues, BN stats)
                            with mean. BN stats handled separately (reset+recompute)."""
    out = copy.deepcopy(base_sd)
    if which == "imagepath":
        sel = [k for k in base_sd.keys()
               if is_img_path(k) and not is_bn_stat(k)
               and base_sd[k].dtype.is_floating_point]
    else:  # full
        sel = [k for k in base_sd.keys()
               if (not k.startswith("clip.")) and (not is_queue(k))
               and (not is_bn_stat(k)) and base_sd[k].dtype.is_floating_point]
    for k in sel:
        acc = torch.zeros_like(base_sd[k].float())
        for ep in EPOCHS:
            acc += sds[ep][k].float()
        acc /= len(EPOCHS)
        out[k] = acc.to(base_sd[k].dtype)
    return out, sel


def reset_disentangler_bn(model):
    """Reset BN running stats on the 4 Disentanglers so they re-estimate from scratch."""
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.reset_running_stats()  # running_mean=0, running_var=1, num_batches=0
            m.momentum = None         # cumulative moving average over the BN pass
            n += 1
    return n


@torch.no_grad()
def recompute_bn(model, train_loader, n_iters):
    """Refresh BN running stats by forwarding ONLY image encoder + disentangler chain.
    Mirrors train_forward's disentangler usage but NEVER touches prototype queues."""
    model.train()  # BN updates running stats in train mode
    it = 0
    for batch in train_loader:
        img = batch[0].cuda().type(model.clip.dtype)
        f_global, _ = model.encode_image(img)
        f_attr = model.attr_disentangler(f_global)   # BN1
        f_obj = model.obj_disentangler(f_global)     # BN2
        _ = model.attr_proj(f_attr)                  # BN3 (consumes disentangler output)
        _ = model.obj_proj(f_obj)                    # BN4
        it += 1
        if it >= n_iters:
            break
    model.eval()
    return it


def evaluate_closed(model, config, val_ds, test_ds):
    """Reuse test.predict_logits + test.test() exactly (closed-world, bias sweep)."""
    out = {}
    for split, ds in [("val", val_ds), ("test", test_ds)]:
        evaluator = testmod.Evaluator(ds, model=None)
        with torch.no_grad():
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt, _ = testmod.predict_logits(
                model, ds, config)
            stats = testmod.test(ds, evaluator, all_logits, all_attr_gt,
                                 all_obj_gt, all_pair_gt, config)
        out[split] = dict(best_hm=float(stats["best_hm"]), AUC=float(stats["AUC"]),
                          best_seen=float(stats["best_seen"]),
                          best_unseen=float(stats["best_unseen"]))
    out["gap"] = out["val"]["best_hm"] - out["test"]["best_hm"]
    return out


def make_model(config, attributes, classes, offset):
    return get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()


def main():
    print(f"[swa] device={torch.cuda.get_device_name(0)}")
    config = parser.parse_args([])
    load_args(CFG, config)
    config.dataset_path = os.path.join(REPO_ROOT, "data/mit-states")
    config.open_world = False

    val_ds = CompositionDataset(config.dataset_path, phase="val",
                                split="compositional-split-natural", open_world=False)
    test_ds = CompositionDataset(config.dataset_path, phase="test",
                                 split="compositional-split-natural", open_world=False)
    train_ds = CompositionDataset(config.dataset_path, phase="train",
                                  split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in val_ds.attrs]
    classes = [c.replace(".", " ").lower() for c in val_ds.objs]
    offset = len(attributes)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, drop_last=True)

    # ---- load checkpoints ----
    print("[swa] loading epoch_10..14 + val_best ...")
    sds = {ep: load_sd(ep) for ep in EPOCHS}
    val_best_sd = torch.load(os.path.join(CKPT_DIR, "val_best.pt"), map_location="cpu")

    # ---- (1) BASIN CHECK ----
    basin = basin_check(sds)

    results = {}

    # ---- BASELINE (val_best, as stored) ----
    print("\n[swa] eval BASELINE (val_best) ...")
    m = make_model(config, attributes, classes, offset)
    m.load_state_dict(val_best_sd)
    m.eval()
    results["baseline"] = evaluate_closed(m, config, val_ds, test_ds)
    del m
    torch.cuda.empty_cache()

    # ---- (2,3) IMAGE-PATH SWA + BN recompute ----
    print("\n[swa] building image-path SWA + BN recompute ...")
    sd_img, sel_img = build_averaged_sd(val_best_sd, sds, "imagepath")
    print(f"       averaged {len(sel_img)} image-path float params")
    m = make_model(config, attributes, classes, offset)
    m.load_state_dict(sd_img)
    n_bn = reset_disentangler_bn(m)
    n_it = recompute_bn(m, train_loader, BN_ITERS)
    print(f"       reset {n_bn} BN layers, recomputed over {n_it} train batches")
    results["imagepath_SWA"] = evaluate_closed(m, config, val_ds, test_ds)
    del m
    torch.cuda.empty_cache()

    # ---- (5) FULL SWA + BN recompute ----
    print("\n[swa] building FULL SWA + BN recompute ...")
    sd_full, sel_full = build_averaged_sd(val_best_sd, sds, "full")
    print(f"       averaged {len(sel_full)} trainable float params")
    m = make_model(config, attributes, classes, offset)
    m.load_state_dict(sd_full)
    n_bn = reset_disentangler_bn(m)
    n_it = recompute_bn(m, train_loader, BN_ITERS)
    print(f"       reset {n_bn} BN layers, recomputed over {n_it} train batches")
    results["full_SWA"] = evaluate_closed(m, config, val_ds, test_ds)
    del m
    torch.cuda.empty_cache()

    # ---- TABLE ----
    base = results["baseline"]
    print("\n" + "=" * 96)
    print("  SCREEN 1 — IDEA F: BN-corrected image-path SWA  (seed0, epochs 10-14)")
    print("=" * 96)
    hdr = f"  {'variant':<16} {'valHM':>8} {'tstHM':>8} {'dTst':>8} {'valAUC':>8} {'tstAUC':>8} {'gap':>8}"
    print(hdr)
    print("  " + "-" * 70)
    for name in ["baseline", "imagepath_SWA", "full_SWA"]:
        r = results[name]
        dtst = r["test"]["best_hm"] - base["test"]["best_hm"]
        print(f"  {name:<16} {r['val']['best_hm']:>8.4f} {r['test']['best_hm']:>8.4f} "
              f"{dtst:>+8.4f} {r['val']['AUC']:>8.4f} {r['test']['AUC']:>8.4f} {r['gap']:>8.4f}")
    print("=" * 96)

    # ---- VERDICT ----
    print("\n  VERDICT (BN recompute DONE — required for validity):")
    base_gap = base["gap"]
    any_promote = False
    for name in ["imagepath_SWA", "full_SWA"]:
        r = results[name]
        dtst = r["test"]["best_hm"] - base["test"]["best_hm"]
        widened = r["gap"] > base_gap + 1e-4
        promote = (dtst >= 0.003) and (not widened)
        any_promote = any_promote or promote
        print(f"    {name}: dTestHM={dtst:+.4f}  gap {base_gap:.4f}->{r['gap']:.4f} "
              f"({'WIDENED' if widened else 'ok'})  -> {'PROMOTE' if promote else 'no'}")
    print(f"\n  OVERALL: {'PROMOTE (some SWA variant raises test HM >= +0.003 w/o widening gap)' if any_promote else 'KILL (no SWA variant clears +0.003 test HM without widening gap)'}")

    os.makedirs(SCRATCH, exist_ok=True)
    with open(os.path.join(SCRATCH, "swa_imagepath_seed0.json"), "w") as f:
        json.dump(dict(basin=basin, results=results, bn_recompute_iters=BN_ITERS), f, indent=2)
    print(f"\n[swa] saved {os.path.join(SCRATCH, 'swa_imagepath_seed0.json')}")


if __name__ == "__main__":
    main()
