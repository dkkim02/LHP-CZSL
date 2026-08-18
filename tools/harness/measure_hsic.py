"""
Measure mean HSIC(f_attr_proj, f_obj_proj) on VAL for a trained checkpoint.
The Round-9 tripwire: did MCM (esp. imgmask) raise HSIC vs the no-norm baseline (0.8840)?

Usage:
  python tools/harness/measure_hsic.py --tag <tag> --config <cfg> --ckpt <ckpt>
Appends to tmp/hsic_results.json.
"""
import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args
from model.hsic import hsic_normalized

SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"
OUT = os.path.join(SCRATCH, "hsic_results.json")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    config = parser.parse_args([])
    load_args(os.path.join(REPO, args.config), config)
    dp = getattr(config, "dataset_path", "data/mit-states")
    config.dataset_path = dp if os.path.isabs(dp) else os.path.join(REPO, dp)
    config.open_world = False

    ds = CompositionDataset(config.dataset_path, phase="val",
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    model = get_model(config, attributes=attributes, classes=classes, offset=len(attributes)).cuda()
    model.load_state_dict(torch.load(os.path.join(REPO, args.ckpt), map_location="cuda"))
    model.eval()

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    vals = []
    for batch in loader:
        f_global, _ = model.encode_image(batch[0].cuda().type(model.clip.dtype))
        fa = model.attr_proj(model.attr_disentangler(f_global))
        fo = model.obj_proj(model.obj_disentangler(f_global))
        vals.append(float(hsic_normalized(fa, fo).item()))
    mean_h = float(np.mean(vals))
    print(f"[hsic {args.tag}] mean HSIC(f_attr_proj,f_obj_proj) val = {mean_h:.4f} ({len(vals)} batches)")

    os.makedirs(SCRATCH, exist_ok=True)
    allr = {}
    if os.path.exists(OUT):
        allr = json.load(open(OUT))
    allr[args.tag] = dict(mean_hsic=mean_h, n_batches=len(vals), ckpt=args.ckpt)
    json.dump(allr, open(OUT, "w"), indent=2)
    print(f"[hsic {args.tag}] saved to {OUT}")


if __name__ == "__main__":
    main()
