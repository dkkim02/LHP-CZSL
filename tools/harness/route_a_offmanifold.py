"""
ROUTE A — confirm Rank-3 died from OFF-MANIFOLD synthesis (not pool size).

Recompute the OLD Rank-3 counterfactual synthesis on real train images via the no-norm model:
  f_cf(i,j) = f_global_i + (f_obj_j - f_obj_i)     [the exact Rank-3 'global_objswap' recomb]
for swaps where the synthesized pair (a_i, o_j) is a REAL pair in some split (so a real centroid
exists). Measure, in the QUEUE/feature space the model actually uses (we test BOTH f_global space
and the disentangler-output f_obj space the recomb lives in; report f_global-space, the comp branch
consumes normalize(f_global)):
  cos(f_cf, centroid of REAL features of pair (a_i,o_j))  vs  cos(f_cf, nearest WRONG-pair centroid)
MARGIN = (cos to true-pair centroid) - (cos to nearest wrong-pair centroid), averaged.
If margin <= ~0.02 (f_cf NOT closer to its true-pair real centroid) -> off-manifold confirmed -> Route A dead.

Usage: CUDA_VISIBLE_DEVICES=1 python tools/harness/route_a_offmanifold.py
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args

CFG = "config/cluspro_baseline_mit_l14_v2_nonorm_full_seed0.yml"
CKPT = "checkpoint/cluspro_baseline_l14_mit_v2_nonorm_full_seed0/val_best.pt"
N_IMGS = 1500            # train images to sample (enough centroids + swaps)
N_SWAPS = 4000           # number of valid (a_i,o_j)-real swaps to evaluate


@torch.no_grad()
def main():
    config = parser.parse_args([])
    load_args(os.path.join(REPO, CFG), config)
    config.dataset_path = os.path.join(REPO, "data/mit-states")
    config.open_world = False

    ds = CompositionDataset(config.dataset_path, phase="train",
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    model = get_model(config, attributes=attributes, classes=classes, offset=len(attributes)).cuda()
    model.load_state_dict(torch.load(os.path.join(REPO, CKPT), map_location="cuda"))
    model.eval()

    # all REAL pairs across splits as INTEGER-index tuples (ds.*_pairs are string tuples)
    a2i, o2i = ds.attr2idx, ds.obj2idx
    def to_idx(pairs):
        return {(a2i[a], o2i[o]) for (a, o) in pairs}
    real_pairs = to_idx(ds.train_pairs) | to_idx(ds.val_pairs) | to_idx(ds.test_pairs)

    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    fg_all, fo_all, a_all, o_all = [], [], [], []
    seen = 0
    for batch in loader:
        img = batch[0].cuda().type(model.clip.dtype)
        fg, _ = model.encode_image(img)                 # f_global (CLS)
        fo = model.obj_disentangler(fg)                 # f_obj (disentangler output, recomb space)
        fg_all.append(fg.float().cpu()); fo_all.append(fo.float().cpu())
        a_all.append(batch[1]); o_all.append(batch[2])
        seen += img.shape[0]
        if seen >= N_IMGS:
            break
    f_global = torch.cat(fg_all)[:N_IMGS]
    f_obj = torch.cat(fo_all)[:N_IMGS]
    attr_i = torch.cat(a_all)[:N_IMGS]
    obj_i = torch.cat(o_all)[:N_IMGS]
    N = f_global.shape[0]
    print(f"[routeA] captured {N} train imgs")

    # build per-pair centroids in f_global space (the comp branch consumes normalize(f_global))
    fg_n = F.normalize(f_global, dim=-1)
    pair_key = [(int(attr_i[k]), int(obj_i[k])) for k in range(N)]
    from collections import defaultdict
    cent_accum = defaultdict(list)
    for k in range(N):
        cent_accum[pair_key[k]].append(k)
    centroids = {}                                       # pair -> normalized centroid [D]
    for p, idxs in cent_accum.items():
        c = fg_n[idxs].mean(0)
        centroids[p] = F.normalize(c, dim=-1)
    cent_pairs = list(centroids.keys())
    cent_mat = torch.stack([centroids[p] for p in cent_pairs])   # [P_c, D]
    print(f"[routeA] {len(cent_pairs)} real-pair centroids from sampled imgs")

    # Build index: attr -> image rows, obj -> image rows (to construct valid swaps directly).
    from collections import defaultdict
    by_attr = defaultdict(list); by_obj = defaultdict(list)
    for k in range(N):
        by_attr[int(attr_i[k])].append(k)
        by_obj[int(obj_i[k])].append(k)
    cent_idx = {p: t for t, p in enumerate(cent_pairs)}
    # valid targets: real pairs that HAVE a centroid AND have donor images for both a and o
    valid_targets = [(a, o) for (a, o) in cent_pairs
                     if (a, o) in real_pairs and by_attr.get(a) and by_obj.get(o)]
    print(f"[routeA] {len(valid_targets)} valid swap-target pairs")

    rng = np.random.default_rng(0)
    margins, true_cos, wrong_cos = [], [], []
    n_eval = 0
    while n_eval < N_SWAPS:
        a, o = valid_targets[int(rng.integers(0, len(valid_targets)))]
        i = by_attr[a][int(rng.integers(0, len(by_attr[a])))]   # donor for attr a (gives f_global & f_obj_i)
        j = by_obj[o][int(rng.integers(0, len(by_obj[o])))]     # donor for obj o (gives f_obj_j)
        if i == j:
            continue
        # OLD Rank-3 synthesis in f_global space: keep i's attr/context, swap j's object
        f_cf = f_global[i] + (f_obj[j] - f_obj[i])
        f_cf_n = F.normalize(f_cf, dim=-1)
        sims = (cent_mat @ f_cf_n)
        t = cent_idx[(a, o)]
        c_true = float(sims[t])
        sims_wrong = sims.clone(); sims_wrong[t] = -2.0
        c_wrong = float(sims_wrong.max())
        true_cos.append(c_true); wrong_cos.append(c_wrong); margins.append(c_true - c_wrong)
        n_eval += 1
        if n_eval >= N_SWAPS:
            break

    margins = np.array(margins)
    print("\n" + "=" * 78)
    print("  ROUTE A — Rank-3 off-manifold check (no-norm, f_global space)")
    print("=" * 78)
    print(f"  evaluated {n_eval} real-target swaps")
    print(f"  mean cos(f_cf, TRUE-pair centroid)        = {np.mean(true_cos):.4f}")
    print(f"  mean cos(f_cf, nearest WRONG-pair centroid)= {np.mean(wrong_cos):.4f}")
    print(f"  MEAN MARGIN (true - nearest-wrong)        = {np.mean(margins):+.4f}")
    print(f"  median margin                             = {np.median(margins):+.4f}")
    print(f"  fraction of swaps with margin>0 (f_cf nearest to its true pair) = {np.mean(margins>0)*100:.1f}%")
    print("=" * 78)
    if np.mean(margins) <= 0.02:
        print(f"  VERDICT: OFF-MANIFOLD CONFIRMED (margin {np.mean(margins):+.4f} <= 0.02). "
              f"f_cf is NOT reliably closest to its true-pair real features -> Rank-3 synthesis is "
              f"off-manifold -> Route A DEAD (do not build memory bank).")
    else:
        print(f"  VERDICT: ON-MANIFOLD-ish (margin {np.mean(margins):+.4f} > 0.02) -> synthesis lands "
              f"near real features -> pool-size, not off-manifold, was the issue -> Route A worth a look.")
    print("=" * 78)


if __name__ == "__main__":
    main()
