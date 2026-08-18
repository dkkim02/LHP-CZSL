"""
fast_eval.py — cached-feature evaluation harness for ClusPro CZSL.

The Evaluator and test() logic is imported directly from test.py to guarantee
metric-identical results.

Public API
----------
evaluate(
    comp_logits,     # (N, N_pairs) float tensor — the final per-pair scores
    phase,           # 'val' or 'test'
    cache_dir,       # path to tools/harness/cache/
) -> dict with best_hm, AUC, best_seen, best_unseen

--baseline mode
  Reconstructs clean ClusPro logits from cache and prints val + test metrics.

Tier-0 usage example (post-process embeddings, then evaluate)
-------------------------------------------------------------
    from tools.harness.fast_eval import evaluate, load_cache, build_comp_logits

    text  = load_cache('tools/harness/cache', 'text')
    val   = load_cache('tools/harness/cache', 'val')

    # (a) Tier-0: inject post-processed img embeddings then score
    # Replace val['img_global'] with your enhanced embeddings (N, 768, L2-normed)
    new_global = my_module(val['img_global'])           # still (N, 768) L2-normed
    comp = build_comp_logits(new_global, val['img_attr'], val['img_obj'],
                             text['feat_pair'], text['feat_attr'], text['feat_obj'],
                             text['pairs'], text['logit_scale'])
    metrics = evaluate(comp, phase='val', cache_dir='tools/harness/cache')
    print(metrics)

    # (b) patch_tokens location
    print(val['patch_tokens'].shape)  # (N_val, N_patch, 768) float16
"""

import argparse
import os
import sys
import copy

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make sure we import from the *main* repo, not the worktree.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

# Import exactly the Evaluator and test() from test.py
from test import Evaluator, test as _test_fn
from dataset import CompositionDataset


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_cache(cache_dir: str, name: str) -> dict:
    """Load a cache file (text.pt, val.pt, or test.pt)."""
    path = os.path.join(cache_dir, f"{name}.pt")
    return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Logit reconstruction
# ---------------------------------------------------------------------------

def build_comp_logits(
    img_global,   # (N, D) float32, L2-normalized
    img_attr,     # (N, D) float32, L2-normalized
    img_obj,      # (N, D) float32, L2-normalized
    feat_pair,    # (N_pairs, D) float32, L2-normalized
    feat_attr,    # (N_attr,  D) float32, L2-normalized
    feat_obj,     # (N_obj,   D) float32, L2-normalized
    pairs,        # (N_pairs, 2) long [attr_idx, obj_idx]
    logit_scale,  # scalar float or 0-d tensor
) -> torch.Tensor:
    """
    Reconstruct the final logit_infer scores (N, N_pairs) from cached features.

    Follows model.val_forward + model.logit_infer exactly:
      comp[:,i] = scale * img_global @ feat_pair[i]      (pair head)
    Then logit_infer (all inf_w=1):
      comp[:,i] += softmax(scale * img_attr @ feat_attr.T)[:,a_i]
                 * softmax(scale * img_obj  @ feat_obj.T )[:,o_i]
    """
    if isinstance(logit_scale, torch.Tensor):
        logit_scale = logit_scale.item()

    ig = img_global.float()
    ia = img_attr.float()
    io = img_obj.float()
    fp = feat_pair.float()
    fa = feat_attr.float()
    fo = feat_obj.float()

    # Pair logits
    comp = logit_scale * (ig @ fp.T)   # (N, N_pairs)

    # Attr / obj softmax predictions
    attr_logits = logit_scale * (ia @ fa.T)   # (N, N_attr)
    obj_logits  = logit_scale * (io @ fo.T)   # (N, N_obj)
    attr_pred = F.softmax(attr_logits, dim=-1)
    obj_pred  = F.softmax(obj_logits,  dim=-1)

    # logit_infer: for each pair add attr_pred[:,a_i] * obj_pred[:,o_i]
    attr_idx = pairs[:, 0]   # (N_pairs,)
    obj_idx  = pairs[:, 1]   # (N_pairs,)
    comp = comp + attr_pred[:, attr_idx] * obj_pred[:, obj_idx]

    return comp


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    comp_logits: torch.Tensor,
    phase: str,
    cache_dir: str,
    dataset_path: str = None,
) -> dict:
    """
    Run the bias-sweep + HM/AUC evaluation exactly as in test.py.

    Parameters
    ----------
    comp_logits : (N, N_pairs) float tensor — final pair scores
    phase       : 'val' or 'test'
    cache_dir   : path to tools/harness/cache/
    dataset_path: override dataset root (default: inferred from cache meta or
                  'data/mit-states' relative to repo root)

    Returns
    -------
    dict with best_hm, AUC, best_seen, best_unseen and other metrics
    """
    img_cache = load_cache(cache_dir, phase)
    attr_gt  = img_cache["attr_gt"]
    obj_gt   = img_cache["obj_gt"]
    pair_gt  = img_cache["pair_gt"]

    # Build dataset metadata object (image loading is lazy, only metadata needed)
    if dataset_path is None:
        dataset_path = os.path.join(_REPO_ROOT, "data", "mit-states")

    dataset = CompositionDataset(
        dataset_path,
        phase=phase,
        split="compositional-split-natural",
        open_world=False,
    )

    evaluator = Evaluator(dataset, model=None)

    # Reuse test.py's test() function directly
    stats = _test_fn(
        dataset,
        evaluator,
        comp_logits.float(),
        attr_gt,
        obj_gt,
        pair_gt,
        config=_DummyConfig(),
    )
    return stats


class _DummyConfig:
    """Minimal config object satisfying test.py's test() function."""
    pass


# ---------------------------------------------------------------------------
# Baseline CLI
# ---------------------------------------------------------------------------

def run_baseline(cache_dir: str, dataset_path: str = None):
    print(f"Loading cache from: {cache_dir}")
    text  = load_cache(cache_dir, "text")
    logit_scale = text["logit_scale"]
    feat_pair   = text["feat_pair"]
    feat_attr   = text["feat_attr"]
    feat_obj    = text["feat_obj"]
    pairs       = text["pairs"]

    print(f"logit_scale = {logit_scale.item():.4f}")
    print(f"feat_pair {feat_pair.shape}, feat_attr {feat_attr.shape}, feat_obj {feat_obj.shape}")

    for phase in ["val", "test"]:
        print(f"\n{'='*60}")
        print(f"Phase: {phase}")
        img = load_cache(cache_dir, phase)
        print(f"  img_global {img['img_global'].shape}  {img['img_global'].dtype}")
        print(f"  img_attr   {img['img_attr'].shape}  {img['img_attr'].dtype}")
        print(f"  img_obj    {img['img_obj'].shape}  {img['img_obj'].dtype}")
        print(f"  patch_tokens {img['patch_tokens'].shape}  {img['patch_tokens'].dtype}")

        comp = build_comp_logits(
            img["img_global"],
            img["img_attr"],
            img["img_obj"],
            feat_pair,
            feat_attr,
            feat_obj,
            pairs,
            logit_scale,
        )
        print(f"  comp_logits {comp.shape}  (min {comp.min():.3f}, max {comp.max():.3f})")

        stats = evaluate(comp, phase=phase, cache_dir=cache_dir, dataset_path=dataset_path)

        best_hm   = stats["best_hm"]
        auc       = stats["AUC"]
        best_seen = stats["best_seen"]
        best_unseen = stats["best_unseen"]

        print(f"\n  RESULTS ({phase}):")
        print(f"    best_hm   = {best_hm:.4f}  (target: 0.4221 / 0.3829)")
        print(f"    AUC       = {auc:.4f}  (target: 0.2489 / 0.2114)")
        print(f"    best_seen = {best_seen:.4f}  (target: 0.5038 / 0.4773)")
        print(f"    best_unseen = {best_unseen:.4f}  (target: 0.5721 / 0.5194)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true",
                   help="Reconstruct baseline logits and print metrics")
    p.add_argument("--cache_dir", default="tools/harness/cache",
                   help="Path to cache directory")
    p.add_argument("--dataset_path", default=None,
                   help="Override dataset root path")
    args = p.parse_args()

    if args.baseline:
        run_baseline(args.cache_dir, args.dataset_path)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
