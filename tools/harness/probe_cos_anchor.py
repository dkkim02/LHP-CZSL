"""
probe_cos_anchor.py — Idea-A headroom PRE-CHECK (inference-only, no images, no training).

Devil's gating question: is there ANY headroom for text-composition anchoring?

For the seed0 baseline checkpoint, build:
  - t_comp(a,o) : comp text feature for ALL dataset pairs (comp template i=0), L2-norm.
  - t_attr(a)   : attr text feature per attribute (attr template i=1), L2-norm.
  - t_obj(o)    : obj  text feature per object   (obj template i=2), L2-norm.

For each pair p=(a,o):
  additive_target = normalize(t_attr[a] + t_obj[o])
  cos_p = cosine(t_comp(a,o), additive_target)

Report mean/std/min/max of cos_p separately over SEEN pairs (in train_pairs) vs
UNSEEN test pairs (in test_pairs, not in train_pairs).

Verdict:
  - unseen cos ~0.85+ (near-additive)            -> tiny headroom, Idea A ~no-op
  - unseen cos notably < seen cos (e.g. 0.6 vs 0.8) -> real binding gap -> headroom

Usage:
  CUDA_VISIBLE_DEVICES=7 python tools/harness/probe_cos_anchor.py
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args

CKPT = "checkpoint/cluspro_baseline_l14_mit_v2_ot_seed0/val_best.pt"
CFG = "config/cluspro_baseline_mit_l14_v2_seed0.yml"
SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"


@torch.no_grad()
def encode_comp_text(model, pairs_idx, batch_size=64):
    """Encode comp text (template i=0) for the given [P,2] pair-index tensor.
    Returns L2-normalized [P, D]. Batched over pairs to bound memory."""
    feats = []
    P = pairs_idx.shape[0]
    for s in range(0, P, batch_size):
        sub = pairs_idx[s:s + batch_size]
        tok = model._construct_token_tensors(sub)[0]  # comp template
        f, _ = model.text_encoder(model.token_ids[0], tok, enable_pos_emb=model.enable_pos_emb)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_part_text(model, any_pair_idx):
    """Encode attr (i=1) and obj (i=2) part text. _construct_token_tensors fills
    template 1 with ALL attributes and template 2 with ALL objects regardless of the
    pair_idx passed, so one call covers every primitive. Returns (t_attr[|A|,D], t_obj[|O|,D])."""
    tt = model._construct_token_tensors(any_pair_idx)
    fa, _ = model.text_encoder(model.token_ids[1], tt[1], enable_pos_emb=model.enable_pos_emb)
    fo, _ = model.text_encoder(model.token_ids[2], tt[2], enable_pos_emb=model.enable_pos_emb)
    fa = (fa / fa.norm(dim=-1, keepdim=True)).float().cpu()
    fo = (fo / fo.norm(dim=-1, keepdim=True)).float().cpu()
    return fa, fo


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), max=float(x.max()),
                p25=float(np.percentile(x, 25)), p50=float(np.percentile(x, 50)),
                p75=float(np.percentile(x, 75)))


def main():
    print(f"[cos-anchor] device={torch.cuda.get_device_name(0)}")
    config = parser.parse_args([])
    load_args(os.path.join(REPO_ROOT, CFG), config)
    config.dataset_path = os.path.join(REPO_ROOT, "data/mit-states")
    config.open_world = False

    # Use the val dataset just to get the full attr/obj/pairs vocab (all share the same
    # closed-world pair universe). dset.pairs = union(train,val,test) in fixed order.
    dset = CompositionDataset(config.dataset_path, phase="val",
                              split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in dset.attrs]
    classes = [c.replace(".", " ").lower() for c in dset.objs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    state = torch.load(os.path.join(REPO_ROOT, CKPT), map_location="cuda")
    model.load_state_dict(state)
    model.eval()

    a2i, o2i = dset.attr2idx, dset.obj2idx
    all_pairs = dset.pairs                       # full closed-world pair list
    pairs_idx = torch.tensor([(a2i[a], o2i[o]) for a, o in all_pairs]).cuda()

    print(f"[cos-anchor] |pairs|={len(all_pairs)}  |A|={offset}  |O|={len(classes)}")
    print("[cos-anchor] encoding comp text (all pairs) ...")
    t_comp = encode_comp_text(model, pairs_idx, batch_size=64)        # [P,D]
    print("[cos-anchor] encoding part text (attr/obj) ...")
    t_attr, t_obj = encode_part_text(model, pairs_idx)               # [|A|,D],[|O|,D]

    # Seen / unseen membership (aligned to all_pairs order)
    train_set = set(dset.train_pairs)
    test_set = set(dset.test_pairs)
    seen_mask = np.array([p in train_set for p in all_pairs])
    unseen_mask = np.array([(p in test_set) and (p not in train_set) for p in all_pairs])

    pa = pairs_idx[:, 0].cpu()
    po = pairs_idx[:, 1].cpu()

    # --- Variant 1 (KEY): additive_target = normalize(t_attr[a] + t_obj[o]) ---
    add_raw = t_attr[pa] + t_obj[po]                                 # [P,D]
    add_n = F.normalize(add_raw, dim=-1)
    cos_add = (t_comp * add_n).sum(-1).numpy()                      # [P]

    # --- Variant 2: g = normalize(t_attr[a]) only (attr-anchor) ---
    cos_attr_only = (t_comp * F.normalize(t_attr[pa], dim=-1)).sum(-1).numpy()

    # --- Variant 3: g = mean(t_attr[a], t_obj[o]) then normalize (== same dir as sum) ---
    mean_raw = 0.5 * (t_attr[pa] + t_obj[po])
    cos_mean = (t_comp * F.normalize(mean_raw, dim=-1)).sum(-1).numpy()

    def report(name, cos):
        seen = stats(cos[seen_mask])
        unseen = stats(cos[unseen_mask])
        return name, seen, unseen

    results = {}
    print("\n" + "=" * 92)
    print(f"  COS(t_comp, anchor)  —  seed0 baseline ({CKPT})")
    print("=" * 92)
    for name, cos in [("additive(t_attr+t_obj)", cos_add),
                      ("attr_only(t_attr)", cos_attr_only),
                      ("mean(0.5*(t_attr+t_obj))", cos_mean)]:
        _, seen, unseen = report(name, cos)
        results[name] = dict(seen=seen, unseen=unseen)
        print(f"\n  [{name}]")
        print(f"    SEEN   (n={seen['n']:>4}): mean={seen['mean']:.4f} std={seen['std']:.4f} "
              f"min={seen['min']:.4f} p25={seen['p25']:.4f} p50={seen['p50']:.4f} "
              f"p75={seen['p75']:.4f} max={seen['max']:.4f}")
        print(f"    UNSEEN (n={unseen['n']:>4}): mean={unseen['mean']:.4f} std={unseen['std']:.4f} "
              f"min={unseen['min']:.4f} p25={unseen['p25']:.4f} p50={unseen['p50']:.4f} "
              f"p75={unseen['p75']:.4f} max={unseen['max']:.4f}")
        print(f"    gap (seen-unseen mean) = {seen['mean'] - unseen['mean']:+.4f}")

    # Verdict from the KEY number: additive unseen cos + seen-vs-unseen gap.
    key = results["additive(t_attr+t_obj)"]
    u_mean = key["unseen"]["mean"]
    s_mean = key["seen"]["mean"]
    gap = s_mean - u_mean
    print("\n" + "=" * 92)
    print("  HEADROOM VERDICT (additive anchor, the loss Idea A would impose):")
    if u_mean >= 0.85:
        verdict = (f"LOW HEADROOM — unseen cos mean={u_mean:.4f} >= 0.85 (already near-additive). "
                   f"Idea A anchoring likely a no-op.")
    elif gap >= 0.08:
        verdict = (f"REAL HEADROOM — unseen cos mean={u_mean:.4f} is notably BELOW seen "
                   f"({s_mean:.4f}), gap={gap:+.4f}. This is the binding gap Idea A targets.")
    else:
        verdict = (f"MARGINAL — unseen cos mean={u_mean:.4f} (not near-additive) but seen-unseen "
                   f"gap small ({gap:+.4f}). Anchoring may help globally but not specifically on unseen.")
    print(f"  {verdict}")
    print("=" * 92)

    os.makedirs(SCRATCH, exist_ok=True)
    out = dict(ckpt=CKPT, results=results,
               key=dict(unseen_mean=u_mean, seen_mean=s_mean, gap=gap, verdict=verdict))
    with open(os.path.join(SCRATCH, "cos_anchor_seed0.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[cos-anchor] saved {os.path.join(SCRATCH, 'cos_anchor_seed0.json')}")


if __name__ == "__main__":
    main()
