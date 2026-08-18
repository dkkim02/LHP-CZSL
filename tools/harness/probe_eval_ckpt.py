"""
probe_eval_ckpt.py — closed-world test-eval of arbitrary (config, checkpoint) pairs.

Reuses the PART A probe machinery (collect_raw_logits -> score_baseline -> test.test()),
which reproduces test.py's closed-world HM+AUC exactly (validated to +-0.0000 vs stored JSON).

Each checkpoint is loaded into a model BUILT WITH ITS OWN config's disent_norm, so the
Disentangler's renamed `norm` submodule matches the saved state_dict.

Usage:
  CUDA_VISIBLE_DEVICES=1 python tools/harness/probe_eval_ckpt.py \
      --tag nonorm_s0 --config config/..._nonorm_full_seed0.yml \
      --ckpt checkpoint/..._nonorm_full_seed0/epoch_3.pt
  (repeat per checkpoint; results appended to tmp/eval_ckpt_results.json)
"""
import os, sys, json, argparse, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parameters import parser as base_parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args
from test import Evaluator, test as test_eval

SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"
RESULTS = os.path.join(SCRATCH, "eval_ckpt_results.json")


@torch.no_grad()
def collect_raw_logits(model, dataset, config):
    model.eval()
    a2i, o2i = dataset.attr2idx, dataset.obj2idx
    pairs = torch.tensor([(a2i[a], o2i[o]) for a, o in dataset.pairs]).cuda()
    loader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False,
                        num_workers=getattr(config, "num_workers", 4))
    comps, attrs, objs, agt, ogt, pgt = [], [], [], [], [], []
    for data in loader:
        c, a, o = model.val_forward(data, pairs)
        comps.append(c.cpu()); attrs.append(a.cpu()); objs.append(o.cpu())
        agt.append(data[1]); ogt.append(data[2]); pgt.append(data[3])
    return (torch.cat(comps), torch.cat(attrs), torch.cat(objs),
            torch.cat(agt).cpu(), torch.cat(ogt).cpu(), torch.cat(pgt).cpu(),
            pairs.cpu())


def score_baseline(COMP, ATTR, OBJ, pairs, pair_w, attr_w, obj_w):
    pa, po = pairs[:, 0], pairs[:, 1]
    ap = F.softmax(ATTR, -1)[:, pa]
    op = F.softmax(OBJ, -1)[:, po]
    wa = torch.ones_like(ap) if attr_w == 0 else ap * attr_w
    wo = torch.ones_like(op) if obj_w == 0 else op * obj_w
    return COMP * pair_w + wa * wo


def eval_split(model, ds, config, attr_gt_unused=None):
    evaluator = Evaluator(ds, model=None)
    COMP, ATTR, OBJ, agt, ogt, pgt, pairs = collect_raw_logits(model, ds, config)
    pw = float(getattr(config, "pair_inference_weight", 1.0))
    aw = float(getattr(config, "attr_inference_weight", 1.0))
    ow = float(getattr(config, "obj_inference_weight", 1.0))
    S = score_baseline(COMP, ATTR, OBJ, pairs, pw, aw, ow)
    stats = test_eval(ds, evaluator, S.clone(), agt, ogt, pgt, config)
    return dict(best_hm=float(stats["best_hm"]), AUC=float(stats["AUC"]),
                best_seen=float(stats["best_seen"]), best_unseen=float(stats["best_unseen"]),
                attr_acc=float(stats.get("attr_acc", float("nan"))),
                obj_acc=float(stats.get("obj_acc", float("nan"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--remap_bn1fc", action="store_true",
                    help="Remap pre-rename '.bn1_fc.' keys -> '.norm.' for old BN checkpoints "
                         "saved before the Disentangler bn1_fc->norm rename.")
    ap.add_argument("--dataset_path", default=None,
                    help="Override dataset_path (else use the config's, resolved under repo root).")
    args = ap.parse_args()

    config = base_parser.parse_args([])
    load_args(os.path.join(REPO_ROOT, args.config), config)
    # Resolve dataset_path: CLI override > config value (made repo-relative if not absolute).
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    else:
        dp = getattr(config, "dataset_path", "data/mit-states")
        config.dataset_path = dp if os.path.isabs(dp) else os.path.join(REPO_ROOT, dp)
    assert os.path.isdir(config.dataset_path), f"dataset_path not found: {config.dataset_path}"
    config.open_world = False
    dn = getattr(config, "disent_norm", "bn")
    print(f"[{args.tag}] disent_norm={dn}  ckpt={args.ckpt}  device={torch.cuda.get_device_name(0)}")

    val_ds = CompositionDataset(config.dataset_path, phase="val",
                                split="compositional-split-natural", open_world=False)
    test_ds = CompositionDataset(config.dataset_path, phase="test",
                                 split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in val_ds.attrs]
    classes = [c.replace(".", " ").lower() for c in val_ds.objs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    sd = torch.load(os.path.join(REPO_ROOT, args.ckpt), map_location="cuda")
    if args.remap_bn1fc:
        # Old BN checkpoints predate the Disentangler bn1_fc->norm rename. With disent_norm
        # default 'bn', the module builds nn.BatchNorm1d at submodule name `norm`, so we just
        # rename the parameter/buffer keys '.bn1_fc.' -> '.norm.' (values unchanged).
        remapped = {}
        n = 0
        for k, v in sd.items():
            if ".bn1_fc." in k:
                remapped[k.replace(".bn1_fc.", ".norm.")] = v
                n += 1
            else:
                remapped[k] = v
        sd = remapped
        print(f"[{args.tag}] remapped {n} '.bn1_fc.' -> '.norm.' keys")
    # strict load: verifies the renamed `norm` module matches (right disent_norm).
    model.load_state_dict(sd)
    model.eval()
    # sanity: confirm the Disentangler norm module type matches the config
    print(f"[{args.tag}] attr_disentangler.norm = {type(model.attr_disentangler.norm).__name__}")

    t0 = time.time()
    val = eval_split(model, val_ds, config)
    tst = eval_split(model, test_ds, config)
    gap = val["best_hm"] - tst["best_hm"]
    print(f"[{args.tag}] valHM={val['best_hm']:.4f} tstHM={tst['best_hm']:.4f} "
          f"valAUC={val['AUC']:.4f} tstAUC={tst['AUC']:.4f} gap={gap:.4f} "
          f"({time.time()-t0:.0f}s)")

    rec = dict(tag=args.tag, disent_norm=dn, ckpt=args.ckpt,
               val=val, test=tst, gap=gap)
    os.makedirs(SCRATCH, exist_ok=True)
    allres = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            allres = json.load(f)
    allres[args.tag] = rec
    with open(RESULTS, "w") as f:
        json.dump(allres, f, indent=2)
    print(f"[{args.tag}] saved to {RESULTS}")


if __name__ == "__main__":
    main()
