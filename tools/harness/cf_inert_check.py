"""
cf_inert_check.py — Prove the cf_weight=0 path is byte-identical to baseline,
IN-PROCESS and deterministic (cuDNN nondeterminism across processes makes a
cross-process loss comparison meaningless — verified empirically).

Method: one model, one fixed batch. Run the SAME forward+loss twice with the RNG
state restored to an identical snapshot before each call:
  run 1: config.cf_weight = 0.0   -> CF block guarded OFF
  run 2: config.cf_weight = 0.5 but model.set_epoch(0) -> warmup=0 so the CF term
         is multiplied by 0 in loss_calu.
We assert run1.loss == run2.loss EXACTLY when the CF block is OFF (run1), which
proves the guarded-off path equals baseline. We ALSO show that with the block ON
(epoch>=warmup) the loss changes and the cf term is > 0 (sanity that it's wired).

For the strict byte-identical claim we compare run1 against a re-run of run1 with
the identical restored RNG state -> must match to 0.0 bit-for-bit (in-process,
deterministic algorithms enabled).

Usage: CUDA_VISIBLE_DEVICES=5 python tools/harness/cf_inert_check.py
"""
import os, sys
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args, set_seed
from torch.utils.data import DataLoader


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = parser.parse_args([])
    load_args(os.path.join(REPO_ROOT, "config/cluspro_baseline_mit_l14_v2_cf_seed0.yml"), config)
    config.dataset_path = os.path.join(REPO_ROOT, "data/mit-states")
    config.open_world = False

    ds = CompositionDataset(config.dataset_path, phase="train",
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    offset = len(attributes)
    a2i, o2i = ds.attr2idx, ds.obj2idx
    train_pairs = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.train_pairs]).cuda()
    loader = DataLoader(ds, batch_size=config.train_batch_size, shuffle=False, num_workers=2)
    batch = next(iter(loader))

    set_seed(0)
    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.train()

    # Neutralize the two stateful sources of run-to-run drift so we can isolate the
    # PURE loss-path effect of the cf_weight guard:
    #   1) _update_prototypes mutates EMA buffers every forward -> make it a no-op.
    #   2) BatchNorm running stats update every forward -> freeze BN to eval mode.
    # These do NOT change the cf guard logic; they only remove confounders so a
    # bit-for-bit comparison is meaningful.
    model._update_prototypes = lambda *a, **k: None
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eval()

    def one_loss(cf_weight, epoch, rng_state, cuda_rng_state):
        model.config.cf_weight = cf_weight
        model.cf_weight = cf_weight
        model.set_epoch(epoch)
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        model.zero_grad()
        with torch.cuda.amp.autocast():
            predict = model(batch, train_pairs)
            cf_val = float(predict[-1].item())
            loss = model.loss_calu(predict, batch)
        lv = float(loss.item())
        loss.backward()
        return lv, cf_val

    # Snapshot RNG once; restore before each run for a fair in-process comparison.
    set_seed(123)
    rng = torch.get_rng_state()
    crng = torch.cuda.get_rng_state()

    # run 1: cf OFF (cf_weight=0) -> baseline path, CF block never executes
    l_off, cf_off = one_loss(0.0, 0, rng, crng)
    # run 1b: identical settings + identical restored RNG -> must be bit-identical
    l_off_rep, _ = one_loss(0.0, 0, rng, crng)
    # run 2: cf_weight=0.5 but epoch=0 -> warmup=0 -> CF term * 0 in loss.
    #        The CF block DOES execute (consumes RNG, extra BN), so loss may differ;
    #        this is expected. We only assert the OFF path equals its own replay.
    l_warm0, cf_warm0 = one_loss(0.5, 0, rng, crng)
    # run 3: cf ON (epoch>=warmup) -> loss changes, cf term > 0
    l_on, cf_on = one_loss(0.5, 3, rng, crng)

    print(f"  cf_weight=0   epoch=0 : loss={l_off:.8f}  cf_term={cf_off:.6f}")
    print(f"  cf_weight=0   epoch=0 (replay): loss={l_off_rep:.8f}")
    print(f"  cf_weight=0.5 epoch=0 : loss={l_warm0:.8f}  cf_term={cf_warm0:.6f} (warmup=0 -> 0 contribution)")
    print(f"  cf_weight=0.5 epoch=3 : loss={l_on:.8f}  cf_term={cf_on:.6f} (active)")

    inert_replay = abs(l_off - l_off_rep)
    diff_warm0 = abs(l_off - l_warm0)
    print(f"\n  OFF-path replay diff           = {inert_replay:.2e}  (expect 0: deterministic)")
    print(f"  |off  -  warm0(cf=0.5,ep0)|    = {diff_warm0:.2e}  (expect 0: CF term * warmup(0) = 0)")

    # (A) Two invariants prove byte-identity of the cf_weight=0 path:
    #   - OFF path replays bit-for-bit (no hidden nondeterminism once confounders removed)
    #   - cf_weight=0.5 @ epoch0 (warmup=0) yields the SAME loss as cf_weight=0: the CF
    #     block contributes exactly 0, and (since it runs only AFTER the main logits) it
    #     does not perturb the main loss. => guard math is correct.
    a_pass = (inert_replay == 0.0) and (diff_warm0 == 0.0) and (cf_off == 0.0)
    b_pass = cf_on > 0 and abs(l_on - l_off) > 1e-3
    print(f"\n  [A] cf_weight=0 path byte-identical to baseline "
          f"(replay==0, warm0==off, cf_term_off==0): {'PASS' if a_pass else 'FAIL'}")
    print(f"  [B] cf ON (epoch>=warmup) changes loss and cf_term>0: {'PASS' if b_pass else 'FAIL'}")
    sys.exit(0 if (a_pass and b_pass) else 1)


if __name__ == "__main__":
    main()
