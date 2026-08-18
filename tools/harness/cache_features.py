"""
cache_features.py — run the ClusPro model once over val + test, cache image and text
features for fast repeated evaluation.

Run from the repo root:
  CUDA_VISIBLE_DEVICES=0 python tools/harness/cache_features.py \
      --checkpoint checkpoint/cluspro_baseline_l14_mit_v2_ot_seed0/val_best.pt \
      --config    config/cluspro_baseline_mit_l14_v2_ot_seed0.yml \
      --out_dir   tools/harness/cache

Outputs
-------
tools/harness/cache/text.pt
    feat_pair  (N_pairs, 768)  float32, L2-normalized
    feat_attr  (N_attr,  768)  float32, L2-normalized
    feat_obj   (N_obj,   768)  float32, L2-normalized
    pairs      (N_pairs, 2)    long [attr_idx, obj_idx]
    logit_scale scalar float32
    attrs      list[str]
    objs       list[str]

tools/harness/cache/val.pt   (N=10420 for MIT-States)
tools/harness/cache/test.pt  (N=12995 for MIT-States)
    img_global  (N, 768) float32   — L2-normalized global CLS token
    img_attr    (N, 768) float32   — L2-normalized attr-projected feature
    img_obj     (N, 768) float32   — L2-normalized obj-projected feature
    patch_tokens (N, N_patch, 768) float16  — raw (post-proj) patch tokens, NOT normalized
    attr_softmax (N, N_attr)  float32
    obj_softmax  (N, N_obj)   float32
    attr_gt  (N,) long
    obj_gt   (N,) long
    pair_gt  (N,) long
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure we import from the *main* repo, not the worktree.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from utils import load_args
from parameters import parser as base_parser
from dataset import CompositionDataset
from model.model_factory import get_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to val_best.pt")
    p.add_argument("--config", required=True, help="path to .yml config")
    p.add_argument("--out_dir", default="tools/harness/cache")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_config(yml_path: str):
    """Parse the yml into the base argparse Namespace (no argv)."""
    config = base_parser.parse_args([])
    load_args(yml_path, config)
    return config


def make_pairs_tensor(dataset, attr2idx, obj2idx):
    return torch.tensor(
        [(attr2idx[a], obj2idx[o]) for a, o in dataset.pairs],
        dtype=torch.long,
    )


# ---------------------------------------------------------------------------
# Text encoding — runs once since val/test share the same pair universe
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_text(model, pairs_tensor, device):
    """
    Returns feat_pair, feat_attr, feat_obj — all float32, L2-normalized.
    pairs_tensor: (N_pairs, 2) on CUDA.
    """
    model.eval()
    token_tensors = model._construct_token_tensors(pairs_tensor)
    feats = []
    for i in range(model.token_ids.shape[0]):
        feat, _ = model.text_encoder(
            model.token_ids[i],
            token_tensors[i],
            enable_pos_emb=model.enable_pos_emb,
        )
        feat = feat.float()
        feat = F.normalize(feat, p=2, dim=-1)
        feats.append(feat.cpu())
    return feats[0], feats[1], feats[2]   # pair, attr, obj


# ---------------------------------------------------------------------------
# Image encoding over one split
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_images(model, dataset, batch_size, num_workers, device):
    """
    Returns dict with:
        img_global, img_attr, img_obj : (N, D) float32 L2-normalized
        patch_tokens                  : (N, N_patch, D) float16
        attr_softmax, obj_softmax     : (N, n_attr/n_obj) float32
        attr_gt, obj_gt, pair_gt      : (N,) long
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_global, all_attr_f, all_obj_f = [], [], []
    all_patch = []
    all_attr_s, all_obj_s = [], []
    all_attr_gt, all_obj_gt, all_pair_gt = [], [], []

    for batch in tqdm(loader, desc=f"  Encoding {dataset.phase}"):
        imgs = batch[0].to(device)
        attr_gt = batch[1]
        obj_gt  = batch[2]
        pair_gt = batch[3]

        # Encode image
        f_global_fp16, x_all_fp16 = model.encode_image(imgs.type(model.clip.dtype))

        # Work in float32 for the projections
        f_global = f_global_fp16.float()
        x_all    = x_all_fp16.float()   # (B, 1+N_patch, D)

        f_attr = model.attr_disentangler(f_global)
        f_obj  = model.obj_disentangler(f_global)
        f_attr_proj = model.attr_proj(f_attr)
        f_obj_proj  = model.obj_proj(f_obj)

        # L2-normalize
        img_global = F.normalize(f_global, p=2, dim=-1)
        img_attr   = F.normalize(f_attr_proj, p=2, dim=-1)
        img_obj    = F.normalize(f_obj_proj, p=2, dim=-1)

        # patch tokens: exclude CLS token (index 0), save as float16
        patch = x_all[:, 1:, :].half()  # (B, N_patch, D)

        # softmax scores — we can compute these later from img_attr/obj but
        # having them pre-computed avoids re-loading text features every time
        # They require text features, defer to fast_eval.py's assemble step.
        # NOTE: We'll compute these in a second pass after text encoding.
        # Store raw (non-softmaxed) img_attr and img_obj for now.

        all_global.append(img_global.cpu())
        all_attr_f.append(img_attr.cpu())
        all_obj_f.append(img_obj.cpu())
        all_patch.append(patch.cpu())

        all_attr_gt.append(attr_gt)
        all_obj_gt.append(obj_gt)
        all_pair_gt.append(pair_gt)

    result = {
        "img_global":  torch.cat(all_global,  dim=0),
        "img_attr":    torch.cat(all_attr_f,  dim=0),
        "img_obj":     torch.cat(all_obj_f,   dim=0),
        "patch_tokens": torch.cat(all_patch,  dim=0),
        "attr_gt":  torch.cat(all_attr_gt),
        "obj_gt":   torch.cat(all_obj_gt),
        "pair_gt":  torch.cat(all_pair_gt),
    }
    return result


def add_softmax(cache, feat_attr, feat_obj, logit_scale):
    """
    Compute attr_softmax and obj_softmax from cached img_attr/img_obj.
    logit_scale * img_attr @ feat_attr.T  -> softmax -> attr_softmax
    Both feat_attr/feat_obj are moved to CPU for this operation.
    """
    fa = feat_attr.cpu().float()
    fo = feat_obj.cpu().float()

    # Process in chunks to avoid OOM
    N = cache["img_attr"].shape[0]
    chunk = 1024
    attr_sm, obj_sm = [], []
    for i in range(0, N, chunk):
        a = cache["img_attr"][i:i+chunk].float()  # (B, D) already on CPU
        o = cache["img_obj"][i:i+chunk].float()

        a_logit = logit_scale * a @ fa.T   # (B, n_attr)
        o_logit = logit_scale * o @ fo.T   # (B, n_obj)

        attr_sm.append(F.softmax(a_logit, dim=-1))
        obj_sm.append(F.softmax(o_logit, dim=-1))

    cache["attr_softmax"] = torch.cat(attr_sm, dim=0)
    cache["obj_softmax"]  = torch.cat(obj_sm,  dim=0)
    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")

    # --- Config ---
    config = build_config(args.config)

    # --- Datasets (metadata only — lazy image load) ---
    print("Loading val dataset metadata...")
    val_dataset = CompositionDataset(
        config.dataset_path,
        phase="val",
        split="compositional-split-natural",
        open_world=False,
    )
    print("Loading test dataset metadata...")
    test_dataset = CompositionDataset(
        config.dataset_path,
        phase="test",
        split="compositional-split-natural",
        open_world=False,
    )

    allattrs  = val_dataset.attrs
    allobj    = val_dataset.objs
    attributes = [a.replace(".", " ").lower() for a in allattrs]
    classes    = [c.replace(".", " ").lower() for c in allobj]
    offset     = len(attributes)

    # --- Model ---
    print("Loading model...")
    model = get_model(config, attributes=attributes, classes=classes, offset=offset)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()
    print("Model loaded. Checking for CMT (should be absent in clean path)...")
    has_cmt = hasattr(model, "cmt")
    has_lamda = hasattr(model, "lamda")
    print(f"  has_cmt: {has_cmt},  has_lamda: {has_lamda}")
    if has_cmt:
        print("  WARNING: CMT detected! Zeroing lamda to bypass CMT.")
        with torch.no_grad():
            model.lamda.zero_()

    logit_scale = model.clip.logit_scale.exp().item()
    print(f"logit_scale = {logit_scale:.4f}")

    # --- Pairs tensor (val and test share the same universe) ---
    pairs_tensor = make_pairs_tensor(val_dataset, val_dataset.attr2idx, val_dataset.obj2idx)
    pairs_tensor = pairs_tensor.to(device)

    # --- Text features (shared) ---
    print("Encoding text features...")
    feat_pair, feat_attr, feat_obj = encode_text(model, pairs_tensor, device)
    print(f"  feat_pair {feat_pair.shape}, feat_attr {feat_attr.shape}, feat_obj {feat_obj.shape}")

    text_cache = {
        "feat_pair":   feat_pair,
        "feat_attr":   feat_attr,
        "feat_obj":    feat_obj,
        "pairs":       pairs_tensor.cpu(),
        "logit_scale": torch.tensor(logit_scale, dtype=torch.float32),
        "attrs":       attributes,
        "objs":        classes,
    }
    text_path = os.path.join(args.out_dir, "text.pt")
    torch.save(text_cache, text_path)
    size_text = os.path.getsize(text_path) / 1e6
    print(f"  Saved text.pt  ({size_text:.1f} MB)")

    # --- Image features: val ---
    print("Encoding val images...")
    val_cache = encode_images(model, val_dataset, args.batch_size, args.num_workers, device)
    val_cache = add_softmax(val_cache, feat_attr.to(device), feat_obj.to(device), logit_scale)
    val_path = os.path.join(args.out_dir, "val.pt")
    torch.save(val_cache, val_path)
    size_val = os.path.getsize(val_path) / 1e6
    print(f"  Saved val.pt   ({size_val:.1f} MB)")
    print(f"  val shapes: img_global {val_cache['img_global'].shape}, patch {val_cache['patch_tokens'].shape}")
    print(f"  val dtypes: img_global {val_cache['img_global'].dtype}, patch {val_cache['patch_tokens'].dtype}")

    # --- Image features: test ---
    print("Encoding test images...")
    test_cache = encode_images(model, test_dataset, args.batch_size, args.num_workers, device)
    test_cache = add_softmax(test_cache, feat_attr.to(device), feat_obj.to(device), logit_scale)
    test_path = os.path.join(args.out_dir, "test.pt")
    torch.save(test_cache, test_path)
    size_test = os.path.getsize(test_path) / 1e6
    print(f"  Saved test.pt  ({size_test:.1f} MB)")
    print(f"  test shapes: img_global {test_cache['img_global'].shape}, patch {test_cache['patch_tokens'].shape}")

    total_gb = (size_text + size_val + size_test) / 1e3
    print(f"\nTotal cache size: {total_gb:.2f} GB")
    if total_gb > 40:
        print("WARNING: Total cache > 40 GB. Consider saving patch_tokens as float16 (already done) or splitting.")

    print("\nDone. Run `python tools/harness/fast_eval.py --baseline` to verify metrics.")


if __name__ == "__main__":
    main()
