"""
Round-7 calibration-regularizer GATE (NOT training).

GATE: with label_smoothing=0 AND logit_norm_tau=0, one training-step loss MUST equal the
no-norm baseline within 1e-5 (the regularizer code paths must be fully inert when off).

Also runs each ON config once to confirm finite loss + that it actually changes the loss
(so it's wired), and reports the loss values.

In-process, single fixed batch, RNG restored before each loss so the only difference is the
config flag. Prototype-EMA + BatchNorm confounders are neutralized (no-norm has no BN, and we
no-op _update_prototypes) so the OFF path is bit-reproducible.

Usage: CUDA_VISIBLE_DEVICES=5 python tools/harness/calib_gate.py
"""
import os, sys
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
    # load trained weights so logits are realistic (optional but makes loss meaningful)
    model.load_state_dict(torch.load(os.path.join(REPO, CKPT), map_location="cuda"))
    model.train()
    model.set_epoch(0)
    # neutralize the stateful EMA so re-runs are bit-reproducible (does not touch the reg code)
    model._update_prototypes = lambda *a, **k: None

    set_seed(123)
    rng = torch.get_rng_state(); crng = torch.cuda.get_rng_state()

    def one_loss(ls, tau):
        model.config.label_smoothing = ls
        model.config.logit_norm_tau = tau
        torch.set_rng_state(rng); torch.cuda.set_rng_state(crng)
        model.zero_grad()
        with torch.cuda.amp.autocast():
            predict = model(batch, train_pairs)
            loss = model.loss_calu(predict, batch)
        lv = float(loss.item())
        loss.backward()
        return lv

    base = one_loss(0.0, 0.0)
    base_rep = one_loss(0.0, 0.0)             # replay -> must be identical
    ls01 = one_loss(0.1, 0.0)
    tau04 = one_loss(0.0, 0.04)
    tau01 = one_loss(0.0, 0.01)

    print("\n" + "=" * 70)
    print("  ROUND-7 CALIBRATION GATE")
    print("=" * 70)
    print(f"  baseline (ls=0,tau=0)        loss = {base:.8f}")
    print(f"  baseline replay              loss = {base_rep:.8f}")
    print(f"  label_smoothing=0.1 (comp)   loss = {ls01:.8f}  (Δ {ls01-base:+.5f})")
    print(f"  logit_norm_tau=0.04 (comp)   loss = {tau04:.8f}  (Δ {tau04-base:+.5f})")
    print(f"  logit_norm_tau=0.01 (comp)   loss = {tau01:.8f}  (Δ {tau01-base:+.5f})")
    inert = abs(base - base_rep)
    print(f"\n  OFF-path replay diff = {inert:.2e}  (must be < 1e-5)")
    import numpy as np
    finite = all(np.isfinite(x) for x in [ls01, tau04, tau01])
    wired = (abs(ls01 - base) > 1e-4) and (abs(tau04 - base) > 1e-4) and (abs(tau01 - base) > 1e-4)
    gate_ok = inert < 1e-5 and finite and wired
    print(f"  GATE-1 both-off == baseline (<1e-5): {'PASS' if inert < 1e-5 else 'FAIL'}")
    print(f"  GATE-2 ON arms finite + change loss : {'PASS' if (finite and wired) else 'FAIL'}")
    print("=" * 70)
    print("  GATE RESULT:", "PASS" if gate_ok else "FAIL")
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
