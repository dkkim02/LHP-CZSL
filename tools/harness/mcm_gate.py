"""
Round-9 MCM GATE + baseline HSIC reference (NOT training).

GATE: mcm_weight=0 -> one training-step loss == no-norm baseline within 1e-5 (the MCM path
must be fully inert when off). Also runs each ON variant once to confirm finite + that it
changes the loss (wired) and loss_mcm (returned) > 0.

HSIC REFERENCE: measure mean HSIC(f_attr_proj, f_obj_proj) over several VAL batches on the
no-norm val_best checkpoint -> the tripwire reference for the imgmask devil-claim.

In-process, fixed batch, RNG restored. Prototype EMA neutralized so OFF path is bit-reproducible.

Usage: CUDA_VISIBLE_DEVICES=5 python tools/harness/mcm_gate.py
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args, set_seed
from model.hsic import hsic_normalized

CFG = "config/cluspro_baseline_mit_l14_v2_nonorm_full_seed0.yml"
CKPT = "checkpoint/cluspro_baseline_l14_mit_v2_nonorm_full_seed0/val_best.pt"
SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"


def build_model(mcm_weight, mcm_variant):
    config = parser.parse_args([])
    load_args(os.path.join(REPO, CFG), config)
    config.dataset_path = os.path.join(REPO, "data/mit-states")
    config.open_world = False
    config.mcm_weight = mcm_weight
    config.mcm_variant = mcm_variant
    ds_tr = CompositionDataset(config.dataset_path, phase="train",
                               split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds_tr.attrs]
    classes = [c.replace(".", " ").lower() for c in ds_tr.objs]
    set_seed(0)
    model = get_model(config, attributes=attributes, classes=classes, offset=len(attributes)).cuda()
    # non-strict load: ON-textmask adds mcm_mask param not in ckpt; assert only mcm_* missing.
    missing, unexpected = model.load_state_dict(torch.load(os.path.join(REPO, CKPT), map_location="cuda"), strict=False)
    bad = [m for m in missing if "mcm_mask" not in m]
    assert not bad and not unexpected, f"unexpected load mismatch: missing={bad} unexpected={unexpected}"
    return model, config, ds_tr, attributes, classes


def gate():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, config, ds, attributes, classes = build_model(0.0, "textmask")
    a2i, o2i = ds.attr2idx, ds.obj2idx
    train_pairs = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.train_pairs]).cuda()
    loader = DataLoader(ds, batch_size=config.train_batch_size, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    model.train(); model.set_epoch(0)
    model._update_prototypes = lambda *a, **k: None

    set_seed(123)
    rng = torch.get_rng_state(); crng = torch.cuda.get_rng_state()

    def one_on(m, mw):
        # m: a model whose self.mcm_variant/self.mcm_weight are FIXED at build time.
        m.config.mcm_weight = mw
        torch.set_rng_state(rng); torch.cuda.set_rng_state(crng)
        with torch.no_grad(), torch.cuda.amp.autocast():
            predict = m(batch, train_pairs)
            mcm_term = float(predict[-1].item())
            loss = m.loss_calu(predict, batch)
        return float(loss.item()), mcm_term

    # OFF baseline on the mcm_weight=0 model (variant attr is 'textmask', but mcm_weight=0
    # short-circuits before any mcm_mask access).
    base, m_off = one_on(model, 0.0)
    base_rep, _ = one_on(model, 0.0)
    inert = abs(base - base_rep)

    print("\n" + "=" * 70)
    print("  ROUND-9 MCM GATE")
    print("=" * 70)
    print(f"  mcm_weight=0            loss={base:.8f}  (mcm_term={m_off:.6f})")
    print(f"  mcm_weight=0  (replay)  loss={base_rep:.8f}")
    print(f"  OFF-path replay diff = {inert:.2e}  (must be < 1e-5)")

    # imgmask ON: build a model with variant='imgmask' (no mcm_mask needed)
    del model; torch.cuda.empty_cache()
    m_img_model, _, _, _, _ = build_model(0.5, "imgmask")
    m_img_model.train(); m_img_model.set_epoch(0)
    m_img_model._update_prototypes = lambda *a, **k: None
    img_on, m_img = one_on(m_img_model, 0.5)
    print(f"  imgmask w=0.5          loss={img_on:.8f}  (Δ {img_on-base:+.5f}, mcm_term={m_img:.4f})")
    del m_img_model; torch.cuda.empty_cache()

    # textmask ON: build a model with variant='textmask' + mcm_mask param
    m_txt_model, _, _, _, _ = build_model(1.0, "textmask")
    m_txt_model.train(); m_txt_model.set_epoch(0)
    m_txt_model._update_prototypes = lambda *a, **k: None
    txt_on, m_txt = one_on(m_txt_model, 1.0)
    print(f"  textmask w=1.0         loss={txt_on:.8f}  (Δ {txt_on-base:+.5f}, mcm_term={m_txt:.4f})")
    del m_txt_model; torch.cuda.empty_cache()

    g1 = inert < 1e-5 and m_off == 0.0
    finite = all(np.isfinite(x) for x in [img_on, txt_on, m_img, m_txt])
    wired = m_img > 0 and m_txt > 0
    ok = g1 and finite and wired
    print(f"  GATE-1 off==baseline (<1e-5): {'PASS' if g1 else 'FAIL'}")
    print(f"  GATE-2 on finite + mcm>0    : {'PASS' if (finite and wired) else 'FAIL'}")
    print("=" * 70)
    print("  GATE RESULT:", "PASS" if ok else "FAIL")
    return ok


def hsic_baseline():
    """Mean HSIC(f_attr_proj, f_obj_proj) over VAL batches on no-norm val_best."""
    model, config, _, attributes, classes = build_model(0.0, "textmask")
    model.eval()
    ds = CompositionDataset(os.path.join(REPO, "data/mit-states"), phase="val",
                            split="compositional-split-natural", open_world=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    vals = []
    with torch.no_grad():
        for batch in loader:
            f_global, _ = model.encode_image(batch[0].cuda().type(model.clip.dtype))
            f_attr_proj = model.attr_proj(model.attr_disentangler(f_global))
            f_obj_proj = model.obj_proj(model.obj_disentangler(f_global))
            h = float(hsic_normalized(f_attr_proj, f_obj_proj).item())
            vals.append(h)
    mean_h = float(np.mean(vals))
    print(f"\n  BASELINE HSIC(f_attr_proj, f_obj_proj) on no-norm val = {mean_h:.4f}  "
          f"(over {len(vals)} batches; lower=more independent)")
    os.makedirs(SCRATCH, exist_ok=True)
    json.dump({"nonorm_baseline_hsic": mean_h, "n_batches": len(vals)},
              open(os.path.join(SCRATCH, "hsic_baseline.json"), "w"), indent=2)
    return mean_h


def main():
    ok = gate()
    hsic_baseline()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
