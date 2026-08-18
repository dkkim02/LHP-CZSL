"""
Round-4 Disentangler-norm smoke test (NOT training).

Each norm type runs in its OWN process (10GB card can't hold multiple ViT-L/14 builds),
2 real train iters, writing a JSON shard. Confirms finite losses / no NaN / no shape
errors, and that 'bn' first-iter loss ~matches the existing baseline (~16.x scale, same
seed/data as the CF inert check used).

Usage:
  CUDA_VISIBLE_DEVICES=7 python tools/harness/disent_norm_smoke.py          # driver: all
  CUDA_VISIBLE_DEVICES=7 python tools/harness/disent_norm_smoke.py bn        # one scenario
"""
import os, sys, json, subprocess
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SHARD_DIR = "/home/jgshin22/.claude/jobs/664681e9/tmp"
N_ITERS = 2
DATA_SEED = 123
INIT_SEED = 0
NORMS = ["bn", "ln", "none"]


def run_scenario(norm):
    from parameters import parser
    from dataset import CompositionDataset
    from model.model_factory import get_model
    from utils import load_args, set_seed
    from torch.utils.data import DataLoader

    config = parser.parse_args([])
    load_args(os.path.join(REPO_ROOT, "config/cluspro_baseline_mit_l14_v2_cf0_seed0.yml"), config)
    config.dataset_path = os.path.join(REPO_ROOT, "data/mit-states")
    config.disent_norm = norm
    config.open_world = False
    print(f"[{norm}] disent_norm={config.disent_norm} device={torch.cuda.get_device_name(0)}")

    ds = CompositionDataset(config.dataset_path, phase="train",
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    offset = len(attributes)
    a2i, o2i = ds.attr2idx, ds.obj2idx
    train_pairs = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.train_pairs]).cuda()
    loader = DataLoader(ds, batch_size=config.train_batch_size, shuffle=False, num_workers=2)
    batches = []
    for b in loader:
        batches.append(b)
        if len(batches) >= N_ITERS:
            break

    set_seed(INIT_SEED)
    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.train()
    model.set_epoch(0)

    # Confirm the norm module wired correctly.
    nt = model.attr_disentangler.norm_type
    nm = type(model.attr_disentangler.norm).__name__
    print(f"[{norm}] attr_disentangler.norm_type={nt}  module={nm}")

    set_seed(DATA_SEED)
    losses, finite = [], True
    for it in range(N_ITERS):
        model.zero_grad()
        batch = batches[it]
        with torch.cuda.amp.autocast():
            predict = model(batch, train_pairs)
            loss = model.loss_calu(predict, batch)
        lv = float(loss.item())
        losses.append(lv)
        if not np.isfinite(lv):
            finite = False
        print(f"  iter {it}: loss={lv:.6f}")
        loss.backward()

    shard = dict(norm=norm, norm_type=nt, module=nm, losses=losses, finite=finite)
    os.makedirs(SHARD_DIR, exist_ok=True)
    with open(os.path.join(SHARD_DIR, f"disent_smoke_{norm}.json"), "w") as f:
        json.dump(shard, f, indent=2)
    print(f"[{norm}] wrote shard.")


def compare():
    shards = {}
    for n in NORMS:
        with open(os.path.join(SHARD_DIR, f"disent_smoke_{n}.json")) as f:
            shards[n] = json.load(f)
    print("\n" + "=" * 64)
    print("  DISENTANGLER-NORM SMOKE SUMMARY")
    print("=" * 64)
    for n in NORMS:
        s = shards[n]
        print(f"  {n:<5} module={s['module']:<14} losses={[f'{x:.4f}' for x in s['losses']]} finite={s['finite']}")
    bn0 = shards["bn"]["losses"][0]
    bn_ok = shards["bn"]["finite"] and (10.0 <= bn0 <= 25.0)  # baseline first-iter ~16.x scale
    all_finite = all(shards[n]["finite"] for n in NORMS)
    print(f"\n  [bn≈baseline] bn first-iter loss={bn0:.4f} in ~16.x range: {'PASS' if bn_ok else 'FAIL'}")
    print(f"  [all finite]  bn/ln/none no NaN/shape errors: {'PASS' if all_finite else 'FAIL'}")
    ok = bn_ok and all_finite
    print(f"\n  OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)
    return ok


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in NORMS:
        run_scenario(sys.argv[1]); return
    if len(sys.argv) >= 2 and sys.argv[1] == "--compare":
        sys.exit(0 if compare() else 1)
    env = dict(os.environ)
    for n in NORMS:
        print(f"\n>>> scenario {n}")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), n], env=env)
        if r.returncode != 0:
            print(f"[driver] {n} FAILED rc={r.returncode}"); sys.exit(2)
    sys.exit(0 if compare() else 1)


if __name__ == "__main__":
    main()
