"""
EXPERIMENT 1 step 1 — build visual k-means prototype seed bank.

Loads the nonorm seed0 val_best checkpoint, runs a forward over the MIT TRAIN set
capturing the QUEUE-SPACE disentangled features:
    f_attr = attr_disentangler(f_global)   (this is exactly what _update_prototypes writes)
    f_obj  = obj_disentangler(f_global)
(NOTE: the queues live in the disentangler-OUTPUT space, NOT post-attr_proj. Verified in
 train_forward: self._update_prototypes(f_attr, f_obj, ...) and _get_cluster_labels(f_attr).)

Per attribute: spherical k-means (K=cluster_num) on the L2-normalized attr-features of
images having that attribute -> K centroids. Same per object. Pad by duplicating if a
primitive has < K images. Saves bank {'attr':(|A|,K,D), 'obj':(|O|,K,D)} to the cache path.

Usage:
  CUDA_VISIBLE_DEVICES=6 python tools/harness/build_kmeans_seed.py \
      --ckpt checkpoint/cluspro_baseline_l14_mit_v2_nonorm_full_seed0/val_best.pt \
      --config config/cluspro_baseline_mit_l14_v2_nonorm_full_seed0.yml \
      --out cache/kmeans_seed_mit.pt
"""
import os, sys, argparse, time
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


@torch.no_grad()
def spherical_kmeans(X, K, iters=50, seed=0):
    """X: [N, D] (will be L2-normalized). Returns centroids [K, D] (L2-normalized).
    Cosine/spherical k-means: assign by max cosine, update = mean then renormalize.
    Handles N < K by duplicating rows. Re-seeds empty clusters from farthest points."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    Xn = F.normalize(X.float(), dim=-1)
    N = Xn.shape[0]
    if N == 0:
        return torch.zeros(K, Xn.shape[1], device=Xn.device)
    if N < K:
        reps = (K + N - 1) // N
        Xn = Xn.repeat(reps, 1)[:K]
        N = K
    # k-means++-ish init on the sphere: random first, then farthest by min-cos
    idx0 = torch.randint(0, N, (1,), generator=g, device=X.device)
    centers = Xn[idx0]
    while centers.shape[0] < K:
        sim = Xn @ centers.t()              # [N, c]
        mincos = sim.max(dim=1).values      # closeness to nearest center
        nxt = torch.argmin(mincos)          # farthest point
        centers = torch.cat([centers, Xn[nxt:nxt + 1]], dim=0)
    for _ in range(iters):
        sim = Xn @ centers.t()              # [N, K]
        assign = sim.argmax(dim=1)
        new = torch.zeros_like(centers)
        for k in range(K):
            m = assign == k
            if m.any():
                new[k] = Xn[m].mean(0)
            else:
                # re-seed empty cluster at the worst-covered point
                worst = (Xn @ centers.t()).max(dim=1).values.argmin()
                new[k] = Xn[worst]
        new = F.normalize(new, dim=-1)
        if torch.allclose(new, centers, atol=1e-6):
            centers = new
            break
        centers = new
    return F.normalize(centers, dim=-1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--km_iters", type=int, default=50)
    args = ap.parse_args()

    config = base_parser.parse_args([])
    load_args(os.path.join(REPO_ROOT, args.config), config)
    dp = getattr(config, "dataset_path", "data/mit-states")
    config.dataset_path = dp if os.path.isabs(dp) else os.path.join(REPO_ROOT, dp)
    config.open_world = False

    K = int(getattr(config, "cluster_num", 5))
    print(f"[kmseed] ckpt={args.ckpt} K={K} device={torch.cuda.get_device_name(0)}")

    ds = CompositionDataset(config.dataset_path, phase="train",
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    offset = len(attributes)
    nA, nO = len(attributes), len(classes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(os.path.join(REPO_ROOT, args.ckpt), map_location="cuda"))
    model.eval()
    D = model.clip.visual.output_dim

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=6)
    # accumulate queue-space features per primitive on CPU
    attr_feats = [[] for _ in range(nA)]
    obj_feats = [[] for _ in range(nO)]
    t0 = time.time()
    seen = 0
    for batch in loader:
        img = batch[0].cuda().type(model.clip.dtype)
        aidx = batch[1]; oidx = batch[2]
        f_global, _ = model.encode_image(img)
        f_attr = model.attr_disentangler(f_global).float().cpu()   # QUEUE SPACE
        f_obj = model.obj_disentangler(f_global).float().cpu()
        for j in range(img.shape[0]):
            attr_feats[int(aidx[j])].append(f_attr[j])
            obj_feats[int(oidx[j])].append(f_obj[j])
        seen += img.shape[0]
    print(f"[kmseed] captured {seen} imgs in {time.time()-t0:.0f}s; running k-means ...")

    attr_bank = torch.zeros(nA, K, D)
    obj_bank = torch.zeros(nO, K, D)
    attr_counts, obj_counts = [], []
    for i in range(nA):
        X = torch.stack(attr_feats[i], 0).cuda() if attr_feats[i] else torch.zeros(0, D).cuda()
        attr_counts.append(X.shape[0])
        attr_bank[i] = spherical_kmeans(X, K, iters=args.km_iters, seed=i).cpu()
    for i in range(nO):
        X = torch.stack(obj_feats[i], 0).cuda() if obj_feats[i] else torch.zeros(0, D).cuda()
        obj_counts.append(X.shape[0])
        obj_bank[i] = spherical_kmeans(X, K, iters=args.km_iters, seed=1000 + i).cpu()

    # diagnostic: mean intra-set pairwise cos of the k-means centroids (spread of the SEED)
    def intra_cos(bank):
        vals = []
        for i in range(bank.shape[0]):
            c = F.normalize(bank[i], dim=-1)
            S = c @ c.t()
            off = S[~torch.eye(c.shape[0], dtype=torch.bool)]
            vals.append(off.mean().item())
        return sum(vals) / len(vals)

    a_spread = intra_cos(attr_bank)
    o_spread = intra_cos(obj_bank)
    print(f"[kmseed] attr prims with <K imgs: {sum(c < K for c in attr_counts)}/{nA}; "
          f"obj: {sum(c < K for c in obj_counts)}/{nO}")
    print(f"[kmseed] SEED intra-cluster mean cos: attr={a_spread:.4f}  obj={o_spread:.4f} "
          f"(lower = more spread; trained-collapsed ref ~0.83/0.77)")

    out = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"attr": attr_bank, "obj": obj_bank,
                "attr_intra_cos": a_spread, "obj_intra_cos": o_spread,
                "source_ckpt": args.ckpt, "space": "disentangler_output"}, out)
    print(f"[kmseed] saved bank attr{tuple(attr_bank.shape)} obj{tuple(obj_bank.shape)} -> {out}")


if __name__ == "__main__":
    main()
