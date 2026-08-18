"""
E1 step 4 — prototype-spread PERSISTENCE measurement.

Given a trained checkpoint, measure the intra-cluster mean pairwise cosine of the trained
attr_queue/obj_queue buffers (raw queue space, the space they live/update in). Compares:
  - the SEED bank's spread (k-means, from cache)
  - this checkpoint's trained-queue spread
  - reference collapsed values (~0.83 attr / 0.77 obj)
Tells us whether k-means seeding kept clusters spread or they re-collapsed.

Usage:
  python tools/harness/measure_proto_persistence.py --ckpt <ckpt.pt> [--seed_bank cache/kmeans_seed_mit.pt] [--tag kmseed]
"""
import os, sys, argparse, json
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"


def intra_cos_bank(get_queue, n):
    vals = []
    for i in range(n):
        q = F.normalize(get_queue(i).float(), dim=-1)
        S = q @ q.t()
        off = S[~torch.eye(q.shape[0], dtype=torch.bool)]
        vals.append(off.mean().item())
    return sum(vals) / len(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed_bank", default="cache/kmeans_seed_mit.pt")
    ap.add_argument("--tag", default="ckpt")
    args = ap.parse_args()

    sd = torch.load(os.path.join(REPO_ROOT, args.ckpt), map_location="cpu")
    nA = len([k for k in sd if k.startswith("attr_queue")])
    nO = len([k for k in sd if k.startswith("obj_queue")])

    a_mean, _ = intra_cos_bank(lambda i: sd[f"attr_queue{i}"], nA)
    o_mean, _ = intra_cos_bank(lambda i: sd[f"obj_queue{i}"], nO)

    out = dict(tag=args.tag, ckpt=args.ckpt,
               trained_attr_intra_cos=a_mean, trained_obj_intra_cos=o_mean,
               nA=nA, nO=nO)

    sb_path = os.path.join(REPO_ROOT, args.seed_bank)
    if os.path.exists(sb_path):
        sb = torch.load(sb_path, map_location="cpu")
        out["seed_attr_intra_cos"] = float(sb.get("attr_intra_cos", float("nan")))
        out["seed_obj_intra_cos"] = float(sb.get("obj_intra_cos", float("nan")))

    print("\n" + "=" * 78)
    print(f"  PROTOTYPE PERSISTENCE — {args.tag}  ({args.ckpt})")
    print("=" * 78)
    print(f"  intra-cluster mean cos (raw queue space; LOWER=more spread):")
    if "seed_attr_intra_cos" in out:
        print(f"    SEED (k-means)   : attr={out['seed_attr_intra_cos']:.4f}  obj={out['seed_obj_intra_cos']:.4f}")
    print(f"    TRAINED (this ckpt): attr={a_mean:.4f}  obj={o_mean:.4f}")
    print(f"    collapsed reference: attr~0.83  obj~0.77")
    if "seed_attr_intra_cos" in out:
        da = a_mean - out["seed_attr_intra_cos"]; do = o_mean - out["seed_obj_intra_cos"]
        print(f"    drift seed->trained: attr {da:+.4f}  obj {do:+.4f} "
              f"({'RE-COLLAPSED' if a_mean > 0.78 else 'STAYED SPREAD'})")
    print("=" * 78)

    os.makedirs(SCRATCH, exist_ok=True)
    p = os.path.join(SCRATCH, f"proto_persistence_{args.tag}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
