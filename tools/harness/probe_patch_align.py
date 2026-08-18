"""
SCREEN 2 — IDEA E: patch->text attention-entropy pre-check (inference-only).

Load seed0 baseline; encode ONE batch of ~32 train images. encode_image returns
(cls, all_tokens) where all_tokens = [B, L+1, D] already mapped to the CLIP joint
space (ln_post + visual.proj) — so patch tokens (index 1:) are text-aligned. The
baseline discards them; we probe whether they carry attr/obj-specific signal.

Per image:
  - attr_text = normalized text feat of the sample's TRUE attribute (attr template i=1)
  - obj_text  = normalized text feat of the sample's TRUE object    (obj  template i=2)
  - attn_attr = softmax_L( patch . attr_text ),  attn_obj = softmax_L( patch . obj_text )
  - normalized entropy H/log(L) of each (mean over batch)
  - top-5 attended patch indices for attr vs obj -> mean overlap fraction

VERDICT:
  entropy ~1.0 (uniform -> patches not text-aligned) AND attr/obj top patches overlap
    heavily  -> E is stealth-CMT, dead, abort.
  entropy < 0.8 AND attr/obj attend to DISTINCT patches -> E has signal, worth a probe.

Usage: CUDA_VISIBLE_DEVICES=7 python tools/harness/probe_patch_align.py
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args

CKPT = os.path.join(REPO_ROOT, "checkpoint/cluspro_baseline_l14_mit_v2_ot_seed0/val_best.pt")
CFG = os.path.join(REPO_ROOT, "config/cluspro_baseline_mit_l14_v2_seed0.yml")
SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"
TOPK = 5


@torch.no_grad()
def all_part_text(model, any_pair_idx):
    """attr text (i=1) per attribute, obj text (i=2) per object, L2-normalized.
    _construct_token_tensors fills template 1 with ALL attrs, template 2 with ALL objs."""
    tt = model._construct_token_tensors(any_pair_idx)
    fa, _ = model.text_encoder(model.token_ids[1], tt[1], enable_pos_emb=model.enable_pos_emb)
    fo, _ = model.text_encoder(model.token_ids[2], tt[2], enable_pos_emb=model.enable_pos_emb)
    fa = F.normalize(fa.float(), dim=-1)
    fo = F.normalize(fo.float(), dim=-1)
    return fa, fo


def main():
    print(f"[patch-align] device={torch.cuda.get_device_name(0)}")
    config = parser.parse_args([])
    load_args(CFG, config)
    config.dataset_path = os.path.join(REPO_ROOT, "data/mit-states")
    config.open_world = False

    train_ds = CompositionDataset(config.dataset_path, phase="train",
                                  split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in train_ds.attrs]
    classes = [c.replace(".", " ").lower() for c in train_ds.objs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(CKPT, map_location="cuda"))
    model.eval()

    a2i, o2i = train_ds.attr2idx, train_ds.obj2idx
    pairs_idx = torch.tensor([(a2i[a], o2i[o]) for a, o in train_ds.pairs]).cuda()
    t_attr, t_obj = all_part_text(model, pairs_idx)   # [|A|,D], [|O|,D]

    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    batch = next(iter(loader))
    imgs = batch[0].cuda()
    attr_gt = batch[1]    # true attr idx per sample
    obj_gt = batch[2]     # true obj idx per sample

    with torch.no_grad():
        cls, all_tokens = model.encode_image(imgs.type(model.clip.dtype))
        # all_tokens: [B, L+1, D] in CLIP joint space (ln_post + proj already applied).
        patches = all_tokens[:, 1:, :].float()        # drop CLS -> [B, L, D]
        patches_n = F.normalize(patches, dim=-1)

    B, L, D = patches_n.shape
    print(f"[patch-align] batch B={B}  patches L={L}  D={D}  (L=16x16 grid for ViT-L/14@224)")

    attr_vec = t_attr[attr_gt.cuda()]   # [B,D] true-attr text
    obj_vec = t_obj[obj_gt.cuda()]      # [B,D] true-obj text

    # patch . text  -> [B, L]
    sim_attr = torch.einsum("bld,bd->bl", patches_n, attr_vec)
    sim_obj = torch.einsum("bld,bd->bl", patches_n, obj_vec)
    attn_attr = F.softmax(sim_attr, dim=-1)
    attn_obj = F.softmax(sim_obj, dim=-1)

    def norm_entropy(attn):
        ent = -(attn * (attn.clamp_min(1e-12)).log()).sum(-1)   # [B], nats
        return (ent / np.log(L)).cpu().numpy()                   # normalized to [0,1]

    H_attr = norm_entropy(attn_attr)
    H_obj = norm_entropy(attn_obj)

    # top-K patch overlap between attr and obj attention, per image
    topk_attr = attn_attr.topk(TOPK, dim=-1).indices.cpu().numpy()   # [B,K]
    topk_obj = attn_obj.topk(TOPK, dim=-1).indices.cpu().numpy()
    overlaps = []
    for i in range(B):
        ov = len(set(topk_attr[i]) & set(topk_obj[i])) / TOPK
        overlaps.append(ov)
    overlaps = np.array(overlaps)

    # Also: peak attention mass (max single-patch prob) — high peak => peaky, not uniform.
    peak_attr = attn_attr.max(-1).values.cpu().numpy()
    peak_obj = attn_obj.max(-1).values.cpu().numpy()
    uniform_prob = 1.0 / L

    res = dict(
        B=B, L=L,
        H_attr_mean=float(H_attr.mean()), H_attr_std=float(H_attr.std()),
        H_obj_mean=float(H_obj.mean()), H_obj_std=float(H_obj.std()),
        top5_overlap_mean=float(overlaps.mean()), top5_overlap_std=float(overlaps.std()),
        peak_attr_mean=float(peak_attr.mean()), peak_obj_mean=float(peak_obj.mean()),
        uniform_prob=float(uniform_prob),
    )

    print("\n" + "=" * 80)
    print("  SCREEN 2 — IDEA E: patch->text attention pre-check (seed0 baseline)")
    print("=" * 80)
    print(f"  normalized entropy H/log(L)  (1.0 = uniform/no-alignment, low = peaky):")
    print(f"    attr-text attn : mean={res['H_attr_mean']:.4f} std={res['H_attr_std']:.4f}")
    print(f"    obj-text  attn : mean={res['H_obj_mean']:.4f} std={res['H_obj_std']:.4f}")
    print(f"  peak attention mass (max single-patch prob; uniform={uniform_prob:.4f}):")
    print(f"    attr={res['peak_attr_mean']:.4f}  obj={res['peak_obj_mean']:.4f}")
    print(f"  top-{TOPK} patch overlap attr-vs-obj (1.0=identical regions, 0=disjoint):")
    print(f"    mean={res['top5_overlap_mean']:.4f} std={res['top5_overlap_std']:.4f}")
    print("=" * 80)

    # VERDICT
    H_mean = 0.5 * (res["H_attr_mean"] + res["H_obj_mean"])
    ov = res["top5_overlap_mean"]
    print("\n  VERDICT:")
    if H_mean >= 0.95 and ov >= 0.5:
        verdict = (f"DEAD — entropy~uniform (H={H_mean:.3f}>=0.95: patches NOT text-aligned) "
                   f"AND attr/obj top patches overlap heavily (ov={ov:.2f}). "
                   f"E is stealth-CMT. ABORT.")
    elif H_mean < 0.8 and ov < 0.5:
        verdict = (f"SIGNAL — entropy low (H={H_mean:.3f}<0.8: peaky, patches text-aligned) "
                   f"AND attr/obj attend to DISTINCT patches (ov={ov:.2f}<0.5). "
                   f"E has signal -> worth a training probe.")
    else:
        verdict = (f"MIXED — H={H_mean:.3f}, top5-overlap={ov:.2f}. "
                   f"Neither clean-dead nor clean-signal; see reasoning below.")
    print(f"  {verdict}")
    print("=" * 80)

    res["verdict"] = verdict
    os.makedirs(SCRATCH, exist_ok=True)
    with open(os.path.join(SCRATCH, "patch_align_seed0.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[patch-align] saved {os.path.join(SCRATCH, 'patch_align_seed0.json')}")


if __name__ == "__main__":
    main()
