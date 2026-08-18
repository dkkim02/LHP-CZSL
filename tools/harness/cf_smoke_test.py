"""
cf_smoke_test.py — PART B smoke test (NOT training).

Each scenario runs in its OWN process (10GB card can't hold two ViT-L/14 builds at
once), writing loss values to a JSON shard. A final --compare pass aggregates them.

Scenarios:
  off_a   : cf_weight=0                      -> baseline path
  off_b   : cf_weight=0, cf_recomb=mlp       -> must equal off_a (block guarded off)
  on_swap : cf_weight=0.5, global_objswap    -> must run clean, cf term > 0
  on_mlp  : cf_weight=0.5, mlp               -> must run clean, finite

Verifies:
  (A) off_a == off_b byte-identical  -> cf_weight=0 path is fully inert
  (B) on_swap finite + cf term active
  (C) on_mlp finite

Usage:
  CUDA_VISIBLE_DEVICES=5 python tools/harness/cf_smoke_test.py off_a
  ... (driver below runs all four + compare automatically when given no arg)
"""

import os
import sys
import json
import subprocess

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SHARD_DIR = "/home/jgshin22/.claude/jobs/664681e9/tmp"
N_ITERS = 4
DATA_SEED = 123
INIT_SEED = 0

SCENARIOS = {
    "off_a":   dict(cf_weight=0.0, cf_recomb="global_objswap", epoch=0),
    "off_b":   dict(cf_weight=0.0, cf_recomb="mlp",            epoch=0),
    "on_swap": dict(cf_weight=0.5, cf_recomb="global_objswap", epoch=3),
    "on_mlp":  dict(cf_weight=0.5, cf_recomb="mlp",            epoch=3),
}


def run_scenario(name):
    from parameters import parser
    from dataset import CompositionDataset
    from model.model_factory import get_model
    from utils import load_args, set_seed
    from torch.utils.data import DataLoader

    spec = SCENARIOS[name]
    config = parser.parse_args([])
    load_args(os.path.join(REPO_ROOT, "config/cluspro_baseline_mit_l14_v2_cf_seed0.yml"), config)
    config.dataset_path = os.path.join(REPO_ROOT, "data/mit-states")
    config.cf_weight = spec["cf_weight"]
    config.cf_recomb = spec["cf_recomb"]
    config.open_world = False

    print(f"[{name}] cf_weight={config.cf_weight} cf_recomb={config.cf_recomb} "
          f"device={torch.cuda.get_device_name(0)}")

    train_dataset = CompositionDataset(config.dataset_path, phase="train",
                                       split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in train_dataset.attrs]
    classes = [c.replace(".", " ").lower() for c in train_dataset.objs]
    offset = len(attributes)
    attr2idx, obj2idx = train_dataset.attr2idx, train_dataset.obj2idx
    train_pairs = torch.tensor([(attr2idx[a], obj2idx[o])
                                for a, o in train_dataset.train_pairs]).cuda()

    loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                        shuffle=False, num_workers=2)
    batches = []
    for b in loader:
        batches.append(b)
        if len(batches) >= N_ITERS:
            break

    set_seed(INIT_SEED)
    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.train()
    model.set_epoch(spec["epoch"])

    set_seed(DATA_SEED)
    losses, cf_terms = [], []
    finite = True
    for it in range(N_ITERS):
        model.zero_grad()
        batch = batches[it]
        with torch.cuda.amp.autocast():
            predict = model(batch, train_pairs)
            cf_val = float(predict[-1].item())  # loss_cf is last element of the train tuple
            loss = model.loss_calu(predict, batch)
        lv = float(loss.item())
        losses.append(lv)
        cf_terms.append(cf_val)
        if not (np.isfinite(lv) and np.isfinite(cf_val)):
            finite = False
        print(f"  iter {it}: loss={lv:.6f}  loss_cf={cf_val:.6f}")
        loss.backward()  # free graph

    shard = dict(name=name, losses=losses, cf_terms=cf_terms, finite=finite)
    os.makedirs(SHARD_DIR, exist_ok=True)
    with open(os.path.join(SHARD_DIR, f"cf_smoke_{name}.json"), "w") as f:
        json.dump(shard, f, indent=2)
    print(f"[{name}] wrote shard.")


def compare():
    shards = {}
    for name in SCENARIOS:
        p = os.path.join(SHARD_DIR, f"cf_smoke_{name}.json")
        with open(p) as f:
            shards[name] = json.load(f)

    print("\n" + "=" * 64)
    print("  CF SMOKE TEST SUMMARY")
    print("=" * 64)
    for name, s in shards.items():
        print(f"  {name:<8} losses={[f'{x:.5f}' for x in s['losses']]}  "
              f"cf={[f'{x:.5f}' for x in s['cf_terms']]}  finite={s['finite']}")

    # (A) off_a == off_b byte-identical
    max_diff = max(abs(a - b) for a, b in zip(shards["off_a"]["losses"], shards["off_b"]["losses"]))
    a_pass = max_diff < 1e-6
    cf_off_zero = all(x == 0.0 for x in shards["off_a"]["cf_terms"] + shards["off_b"]["cf_terms"])
    print(f"\n  [A] cf_weight=0 inert: max|off_a-off_b|={max_diff:.2e} (<1e-6), "
          f"cf_term==0 when off: {cf_off_zero}  -> {'PASS' if (a_pass and cf_off_zero) else 'FAIL'}")

    # (B) on_swap finite + cf active
    b_finite = shards["on_swap"]["finite"]
    b_active = any(x > 0 for x in shards["on_swap"]["cf_terms"])
    b_pass = b_finite and b_active
    print(f"  [B] on_swap finite={b_finite}, cf_term_active={b_active}  -> {'PASS' if b_pass else 'FAIL'}")

    # (C) on_mlp finite
    c_pass = shards["on_mlp"]["finite"]
    print(f"  [C] on_mlp finite={c_pass}  -> {'PASS' if c_pass else 'FAIL'}")

    overall = (a_pass and cf_off_zero) and b_pass and c_pass
    print("\n  OVERALL:", "PASS" if overall else "FAIL")
    print("=" * 64)
    return overall


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in SCENARIOS:
        run_scenario(sys.argv[1])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "--compare":
        ok = compare()
        sys.exit(0 if ok else 1)

    # Driver: run each scenario in its own subprocess (full mem reclaim between).
    env = dict(os.environ)
    py = sys.executable
    for name in SCENARIOS:
        print(f"\n>>> launching scenario {name} ...")
        r = subprocess.run([py, os.path.abspath(__file__), name], env=env)
        if r.returncode != 0:
            print(f"[driver] scenario {name} FAILED (rc={r.returncode})")
            sys.exit(2)
    ok = compare()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
