"""
Round-8 R-Drop GATE (NOT training).

GATE: with rdrop_weight=0, one training-step loss MUST equal the no-norm baseline within
1e-5 (the R-Drop path must be fully inert when off). Also runs rdrop_weight>0 once to confirm
finite + that it changes the loss (wired) and that loss_rdrop (the returned symKL) is > 0.

In-process, fixed batch, RNG restored before each loss. Prototype EMA neutralized so the OFF
path is bit-reproducible. (no-norm has no BN.)

Usage: CUDA_VISIBLE_DEVICES=5 python tools/harness/rdrop_gate.py
"""
import os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args, set_seed
from torch.utils.data import DataLoader

CFG = "config/cluspro_baseline_mit_l14_v2_nonorm_full_seed0.yml"
CKPT = "checkpoint/cluspro_baseline_l14_mit_v2_nonorm_full_seed0/val_best.pt"


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = parser.parse_args([])
    load_args(os.path.join(REPO, CFG), config)
    config.dataset_path = os.path.join(REPO, "data/mit-states")
    config.open_world = False

    ds = CompositionDataset(config.dataset_path, phase="train",
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    a2i, o2i = ds.attr2idx, ds.obj2idx
    train_pairs = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.train_pairs]).cuda()
    loader = DataLoader(ds, batch_size=config.train_batch_size, shuffle=False, num_workers=2)
    batch = next(iter(loader))

    set_seed(0)
    model = get_model(config, attributes=attributes, classes=classes, offset=len(attributes)).cuda()
    model.load_state_dict(torch.load(os.path.join(REPO, CKPT), map_location="cuda"))
    model.train()
    model.set_epoch(0)
    model._update_prototypes = lambda *a, **k: None   # neutralize stateful EMA

    set_seed(123)
    rng = torch.get_rng_state(); crng = torch.cuda.get_rng_state()

    def one(rdw):
        # Gate only needs loss VALUES (not grads) -> no_grad keeps memory bounded even with
        # the R-Drop second forward. (Backward path is exercised separately at train time.)
        model.config.rdrop_weight = rdw
        torch.set_rng_state(rng); torch.cuda.set_rng_state(crng)
        with torch.no_grad(), torch.cuda.amp.autocast():
            predict = model(batch, train_pairs)
            rdrop_term = float(predict[-1].item())     # loss_rdrop (last tuple element)
            loss = model.loss_calu(predict, batch)
        return float(loss.item()), rdrop_term

    base, rd_off = one(0.0)
    base_rep, _ = one(0.0)
    on1, rd_on1 = one(1.0)
    on4, rd_on4 = one(4.0)

    print("\n" + "=" * 70)
    print("  ROUND-8 R-DROP GATE")
    print("=" * 70)
    print(f"  rdrop_weight=0   loss={base:.8f}  (rdrop_term={rd_off:.6f})")
    print(f"  rdrop_weight=0   loss={base_rep:.8f}  (replay)")
    print(f"  rdrop_weight=1   loss={on1:.8f}  (Δ {on1-base:+.5f}, rdrop_term={rd_on1:.4f})")
    print(f"  rdrop_weight=4   loss={on4:.8f}  (Δ {on4-base:+.5f}, rdrop_term={rd_on4:.4f})")
    inert = abs(base - base_rep)
    print(f"\n  OFF-path replay diff = {inert:.2e}  (must be < 1e-5)")
    print(f"  rdrop_term==0 when off: {rd_off == 0.0}")
    finite = all(np.isfinite(x) for x in [on1, on4, rd_on1, rd_on4])
    wired = rd_on1 > 0 and abs(on1 - base) > 1e-4
    g1 = inert < 1e-5 and rd_off == 0.0
    ok = g1 and finite and wired
    print(f"  GATE-1 off==baseline (<1e-5): {'PASS' if g1 else 'FAIL'}")
    print(f"  GATE-2 on finite + symKL>0 + changes loss: {'PASS' if (finite and wired) else 'FAIL'}")
    print("=" * 70)
    print("  GATE RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
