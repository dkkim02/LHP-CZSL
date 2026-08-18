"""
Round-7 LogitNorm fusion-balance diagnostic (inference, devil's load-bearing check).

logit_infer fuses: comp (raw logit_scale*cosine, ~±100) + attr*obj (softmax product, [0,1]),
ADDITIVELY (per pair: comp[:,i]*pair_w + w_attr*w_obj). LogitNorm changes comp's TRAINING
scale but inference still uses logit_scale*cosine, so the comp-vs-attr*obj balance the model
learned can silently shift. We measure, on a few val batches:
  - mean |comp term| (= |comp_logits * pair_inf_w|) over all pairs
  - mean |attr*obj term| (= w_attr*w_obj) over all pairs
  - their ratio
plus attr_acc / obj_acc (unbiased argmax) for the eval, to compare vs the no-norm baseline.

Usage:
  python tools/harness/fusion_balance.py --tag <tag> --config <cfg> --ckpt <ckpt> [--n_batches 6]
Appends to tmp/fusion_balance.json.
"""
import os, sys, json, argparse
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

SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"
OUT = os.path.join(SCRATCH, "fusion_balance.json")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=6)
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
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(os.path.join(REPO, args.ckpt), map_location="cuda"))
    model.eval()

    a2i, o2i = ds.attr2idx, ds.obj2idx
    pairs = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.pairs]).cuda()
    pa, po = pairs[:, 0], pairs[:, 1]
    pair_w = float(getattr(config, "pair_inference_weight", 1.0))
    attr_w = float(getattr(config, "attr_inference_weight", 1.0))
    obj_w = float(getattr(config, "obj_inference_weight", 1.0))

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    comp_mag, ao_mag = [], []
    nb = 0
    for batch in loader:
        comp, attr, obj = model.val_forward(batch, pairs)   # comp = logit_scale*cosine
        ap_ = F.softmax(attr, -1)[:, pa]
        op_ = F.softmax(obj, -1)[:, po]
        wa = ap_ * attr_w
        wo = op_ * obj_w
        comp_term = (comp * pair_w).abs()        # [N,P]
        ao_term = (wa * wo).abs()                # [N,P]
        comp_mag.append(comp_term.mean().item())
        ao_mag.append(ao_term.mean().item())
        nb += 1
        if nb >= args.n_batches:
            break

    cm = sum(comp_mag) / len(comp_mag)
    am = sum(ao_mag) / len(ao_mag)
    rec = dict(tag=args.tag, ckpt=args.ckpt,
               mean_comp_term=cm, mean_attrobj_term=am,
               comp_over_attrobj_ratio=(cm / am if am > 0 else float("inf")),
               pair_w=pair_w, attr_w=attr_w, obj_w=obj_w, n_batches=nb)

    print(f"[fusion {args.tag}] mean|comp term|={cm:.4f}  mean|attr*obj term|={am:.6f}  "
          f"ratio={rec['comp_over_attrobj_ratio']:.1f}")

    os.makedirs(SCRATCH, exist_ok=True)
    allr = {}
    if os.path.exists(OUT):
        allr = json.load(open(OUT))
    allr[args.tag] = rec
    json.dump(allr, open(OUT, "w"), indent=2)
    print(f"[fusion {args.tag}] saved to {OUT}")


if __name__ == "__main__":
    main()
