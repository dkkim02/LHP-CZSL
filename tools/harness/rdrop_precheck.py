"""
R-Drop PRE-CHECK (cheap, no training).

On the no-norm checkpoint, run the VAL set through the image path TWICE in model.train()
mode (dropout ACTIVE: adapter dropout, disentangler dropout, attr_dropout on the soft prompt)
under torch.no_grad(). val_forward does NOT touch prototype/EMA queues (only train_forward
does), and no-norm uses Identity (no BN), so this is a pure dropout-perturbed forward. We
additionally force any BatchNorm to eval() as a belt-and-suspenders guard.

Per image: symKL(comp_view1, comp_view2) over the CLOSED-WORLD pair softmax.
Split mean symKL by:
  (a) SEEN-pair val images
  (b) UNSEEN-pair val images the model gets CORRECT (closed-world top-1, deterministic eval view)
  (c) UNSEEN-pair val images the model gets WRONG

DEVIL'S RULE: R-Drop helps only if WRONG-unseen symKL >= ~1.3x CORRECT-unseen symKL.
If ~uniform -> suppressing it is trivial smoothness -> no-op -> STOP.

Usage:
  CUDA_VISIBLE_DEVICES=5 python tools/harness/rdrop_precheck.py
"""
import os, sys, json
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
from test import Evaluator

CFG = "config/cluspro_baseline_mit_l14_v2_nonorm_full_seed0.yml"
CKPT = "checkpoint/cluspro_baseline_l14_mit_v2_nonorm_full_seed0/val_best.pt"
SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"


def symkl(p, q, eps=1e-8):
    """Symmetric KL between two prob rows [N,P]: KL(p||q)+KL(q||p)."""
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    return ((p * (p / q).log()).sum(-1) + (q * (q / p).log()).sum(-1))


@torch.no_grad()
def main():
    config = parser.parse_args([])
    load_args(os.path.join(REPO, CFG), config)
    config.dataset_path = os.path.join(REPO, "data/mit-states")
    config.open_world = False

    ds = CompositionDataset(config.dataset_path, phase="val",
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(os.path.join(REPO, CKPT), map_location="cuda"))

    a2i, o2i = ds.attr2idx, ds.obj2idx
    pairs = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.pairs]).cuda()   # [P,2]
    P = pairs.shape[0]

    # closed-world mask (val): pairs in train+val (exactly Evaluator.closed_mask)
    ev = Evaluator(ds, model=None)
    closed_mask = ev.closed_mask.cuda()        # [P] bool
    seen_mask = ev.seen_mask.cuda()            # [P] bool (in train_pairs)
    closed_idx = torch.nonzero(closed_mask, as_tuple=False).squeeze(1)

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    # --- guard: belt-and-suspenders, no BN in no-norm but force any to eval ---
    def set_dropout_train_bn_eval(m):
        m.train()                                  # activate dropouts
        for mod in m.modules():
            if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                mod.eval()
        # also ensure prototype EMA cannot run: val_forward never calls it, but no-op guard:
        m._update_prototypes = lambda *a, **k: None

    set_dropout_train_bn_eval(model)

    all_symkl = []
    all_seen_flag = []     # True if the image's GT pair is a seen (train) pair
    all_correct = []       # deterministic eval-view closed-world top-1 correct?
    # deterministic eval view: run once in eval() for the "is it correct" judgement
    for batch in loader:
        attr_gt = batch[1].cuda(); obj_gt = batch[2].cuda(); pair_gt = batch[3].cuda()
        # --- two dropout views (train mode) ---
        comp1, _, _ = model.val_forward(batch, pairs)   # [N,P]
        comp2, _, _ = model.val_forward(batch, pairs)
        # softmax over CLOSED-WORLD pairs only (mask out non-closed by -inf)
        def closed_softmax(comp):
            c = comp.clone()
            c[:, ~closed_mask] = -1e4
            return F.softmax(c.float(), dim=-1)[:, closed_mask]   # [N, |closed|]
        p1 = closed_softmax(comp1); p2 = closed_softmax(comp2)
        sk = symkl(p1, p2).cpu().numpy()                 # [N]
        all_symkl.append(sk)

        # --- deterministic correctness via eval-mode view ---
        model.eval()
        compe, _, _ = model.val_forward(batch, pairs)    # [N,P] eval (no dropout)
        ce = compe.clone(); ce[:, ~closed_mask] = -1e10
        top1 = ce.argmax(dim=-1)                          # [N] pair index
        # correct if predicted pair == GT pair (closed-world)
        gt_attr = attr_gt; gt_obj = obj_gt
        pred_attr = pairs[top1, 0]; pred_obj = pairs[top1, 1]
        correct = ((pred_attr == gt_attr) & (pred_obj == gt_obj)).cpu().numpy()
        all_correct.append(correct)
        # seen flag: is the GT (attr,obj) a train pair?
        # build per-image: find the pair index of GT, check seen_mask
        # GT pair index: match attr&obj against pairs table
        gt_pair_idx = []
        pa = pairs[:, 0]; po = pairs[:, 1]
        for j in range(attr_gt.shape[0]):
            m = (pa == gt_attr[j]) & (po == gt_obj[j])
            idx = torch.nonzero(m, as_tuple=False)
            gt_pair_idx.append(int(idx[0]) if idx.numel() else -1)
        gt_pair_idx = torch.tensor(gt_pair_idx, device=pairs.device)
        seen_flag = seen_mask[gt_pair_idx].cpu().numpy()
        all_seen_flag.append(seen_flag)
        set_dropout_train_bn_eval(model)                 # back to dropout mode for next batch

    symkl_all = np.concatenate(all_symkl)
    seen_all = np.concatenate(all_seen_flag).astype(bool)
    correct_all = np.concatenate(all_correct).astype(bool)
    unseen = ~seen_all

    def stat(mask):
        v = symkl_all[mask]
        return dict(n=int(mask.sum()), mean=float(v.mean()) if v.size else float("nan"),
                    median=float(np.median(v)) if v.size else float("nan"),
                    std=float(v.std()) if v.size else float("nan"))

    seen_s = stat(seen_all)
    unseen_correct = stat(unseen & correct_all)
    unseen_wrong = stat(unseen & ~correct_all)

    print("\n" + "=" * 78)
    print("  R-DROP PRE-CHECK — comp-logit dropout instability (symKL)  [no-norm seed0]")
    print("=" * 78)
    print(f"  (a) SEEN val images          : n={seen_s['n']:>4}  meanSymKL={seen_s['mean']:.4f}  median={seen_s['median']:.4f}")
    print(f"  (b) UNSEEN correct (top-1)   : n={unseen_correct['n']:>4}  meanSymKL={unseen_correct['mean']:.4f}  median={unseen_correct['median']:.4f}")
    print(f"  (c) UNSEEN wrong             : n={unseen_wrong['n']:>4}  meanSymKL={unseen_wrong['mean']:.4f}  median={unseen_wrong['median']:.4f}")
    ratio = unseen_wrong['mean'] / unseen_correct['mean'] if unseen_correct['mean'] > 0 else float("nan")
    ratio_med = unseen_wrong['median'] / unseen_correct['median'] if unseen_correct['median'] > 0 else float("nan")
    print(f"\n  wrong-unseen / correct-unseen  symKL ratio = {ratio:.3f} (mean), {ratio_med:.3f} (median)")
    print("=" * 78)
    if ratio >= 1.3:
        verdict = (f"PROCEED — wrong-unseen symKL is {ratio:.2f}x correct-unseen (>=1.3): instability "
                   f"concentrates on errors, so suppressing it (R-Drop) has real headroom.")
    else:
        verdict = (f"STOP — wrong-unseen / correct-unseen symKL ratio = {ratio:.2f} < 1.3 "
                   f"(instability ~uniform). R-Drop would enforce trivial smoothness -> no-op.")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)

    out = dict(seen=seen_s, unseen_correct=unseen_correct, unseen_wrong=unseen_wrong,
               ratio_mean=ratio, ratio_median=ratio_med, verdict=verdict)
    os.makedirs(SCRATCH, exist_ok=True)
    json.dump(out, open(os.path.join(SCRATCH, "rdrop_precheck.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(SCRATCH, 'rdrop_precheck.json')}")


if __name__ == "__main__":
    main()
