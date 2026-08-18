"""Round-10 KD GATE: kd_weight=0 one-step loss == no-norm baseline within 1e-5 (inert when off).
Also runs kd_weight>0 once to confirm finite + loss_kd>0 + changes loss (wired).
In-process, fixed batch, RNG restored, EMA neutralized."""
import os, sys
import numpy as np
import torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args, set_seed
from torch.utils.data import DataLoader

CFG = "config/cluspro_baseline_mit_l14_v2_nonorm_full_seed0.yml"
CKPT = "checkpoint/cluspro_baseline_l14_mit_v2_nonorm_full_seed0/val_best.pt"
KD_TEACHER = "cache/kd_teacher_dcL14_mit_train.pt"


def build(kd_weight):
    c = parser.parse_args([]); load_args(os.path.join(REPO, CFG), c)
    c.dataset_path = os.path.join(REPO, "data/mit-states"); c.open_world = False
    c.kd_weight = kd_weight; c.kd_tau = 4.0; c.kd_warmup_epochs = 3
    c.kd_teacher_path = os.path.join(REPO, KD_TEACHER)
    return c


def main():
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    c0 = build(0.0)
    ds = CompositionDataset(c0.dataset_path, "train", "compositional-split-natural", open_world=False)
    ds.expose_index = True   # so batch has index for the kd>0 run
    attrs = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [o.replace(".", " ").lower() for o in ds.objs]
    a2i, o2i = ds.attr2idx, ds.obj2idx
    tp = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.train_pairs]).cuda()
    loader = DataLoader(ds, batch_size=c0.train_batch_size, shuffle=False, num_workers=2)
    batch = next(iter(loader))   # has index at batch[4]

    def run(kd_w):
        set_seed(0)
        m = get_model(build(kd_w), attributes=attrs, classes=classes, offset=len(attrs)).cuda()
        # load no-norm weights non-strict (kd buffers are extra)
        miss, unexp = m.load_state_dict(torch.load(os.path.join(REPO, CKPT), map_location="cuda"), strict=False)
        bad = [x for x in miss if "kd_teacher" not in x]
        assert not bad and not unexp, f"load mismatch miss={bad} unexp={unexp}"
        m.train(); m.set_epoch(3); m._update_prototypes = lambda *a, **k: None
        set_seed(123); rng = torch.get_rng_state(); crng = torch.cuda.get_rng_state()
        torch.set_rng_state(rng); torch.cuda.set_rng_state(crng)
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred = m(batch, tp); kd_term = float(pred[-1].item()); loss = m.loss_calu(pred, batch)
        return float(loss.item()), kd_term, m

    off, kd_off, _ = run(0.0)
    off2, _, _ = run(0.0)
    on, kd_on, _ = run(1.0)
    inert = abs(off - off2)
    print("=" * 64)
    print("  ROUND-10 KD GATE")
    print("=" * 64)
    print(f"  kd_weight=0 loss={off:.8f} (kd_term={kd_off:.6f})")
    print(f"  kd_weight=0 loss={off2:.8f} (replay)")
    print(f"  kd_weight=1 loss={on:.8f} (d {on-off:+.5f}, kd_term={kd_on:.4f})")
    print(f"  OFF replay diff = {inert:.2e} (<1e-5)")
    g1 = inert < 1e-5 and kd_off == 0.0
    wired = kd_on > 0 and abs(on - off) > 1e-4 and np.isfinite(on)
    print(f"  GATE-1 off==baseline: {'PASS' if g1 else 'FAIL'}")
    print(f"  GATE-2 on finite+kd>0+changes loss: {'PASS' if wired else 'FAIL'}")
    print("  RESULT:", "PASS" if (g1 and wired) else "FAIL")
    sys.exit(0 if (g1 and wired) else 1)


if __name__ == "__main__":
    main()
