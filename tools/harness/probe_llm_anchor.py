"""
EXPERIMENT 2 — LLM-anchor pre-check (no training).

Tests whether LLM "visual variety" sub-type anchors are useful in the space the
prototype queues are actually CONSUMED.

Space note (verified in train_forward): queues store attr_disentangler(f_global) vectors,
but they are CONSUMED by being pushed through attr_proj (-> all_protos_flat) and matched
in that projected space, which is the same joint space the soft-prompt text path emits.
So we compare in the projected/joint space:
  - trained prototype k in primitive i:  attr_proj(queue_attr[i][k])      (its consumption path)
  - LLM anchor for sub-phrase:           text_encoder(soft-prompt(sub-phrase))   (joint space)
Both L2-normalized. (We also report the raw-queue-space variant for completeness.)

We reuse the EXISTING cached LLM descriptions (cache/llm_descriptions/mit-states/), taking
the first K=cluster_num of the K=8 sub-phrases per primitive — they are already distinct
visual sub-types (e.g. "old" -> worn wood / cracked leather / frosted glass / tarnished brass).

PC-1: greedy-matched mean cos between LLM anchors and the trained K prototypes per primitive,
      vs (a) randn control, (b) shuffled-primitive control (anchors from a different primitive).
PC-2: LLM-anchor intra-set spread (mean pairwise cos) vs k-means-seed spread vs trained 0.83/0.77.

Usage:
  CUDA_VISIBLE_DEVICES=7 python tools/harness/probe_llm_anchor.py \
    --ckpt checkpoint/cluspro_baseline_l14_mit_v2_nonorm_full_seed0/val_best.pt \
    --config config/cluspro_baseline_mit_l14_v2_nonorm_full_seed0.yml \
    --llm_root cache/llm_descriptions/mit-states \
    --kmeans_bank cache/kmeans_seed_mit.pt
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parameters import parser as base_parser
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args

SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"


@torch.no_grad()
def encode_phrases(model, phrases, batch=64):
    """Embed free-text phrases through the model's CLIP text path (joint space), L2-norm.
    Uses the raw CLIP token embeddings + the model's text_encoder transformer (the same
    transformer the soft-prompt path uses), so anchors land in the joint space."""
    tok = model.tokenizer
    feats = []
    for s in range(0, len(phrases), batch):
        sub = phrases[s:s + batch]
        ids = torch.cat([tok(p, context_length=model.config.context_length) for p in sub]).cuda()
        emb = model.clip.token_embedding(ids).type(model.clip.dtype)  # [n, L, d]
        f, _ = model.text_encoder(ids, emb, enable_pos_emb=model.enable_pos_emb)
        f = F.normalize(f.float(), dim=-1)
        feats.append(f.cpu())
    return torch.cat(feats, 0)


def greedy_match_cos(A, P):
    """A:[K,D] anchors, P:[K,D] prototypes (both L2-norm). Greedy 1-1 max-cos matching;
    return mean matched cosine."""
    K = A.shape[0]
    S = (A @ P.t()).clone()           # [K,K]
    used_p = set(); total = 0.0
    for _ in range(K):
        idx = torch.argmax(S).item()
        i, j = idx // K, idx % K
        total += S[i, j].item()
        S[i, :] = -2; S[:, j] = -2
        used_p.add(j)
    return total / K


def intra_cos(M):
    """M:[K,D] -> mean off-diagonal pairwise cos."""
    c = F.normalize(M, dim=-1)
    S = c @ c.t()
    off = S[~torch.eye(c.shape[0], dtype=torch.bool)]
    return off.mean().item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--llm_root", required=True)
    ap.add_argument("--kmeans_bank", required=True)
    args = ap.parse_args()

    config = base_parser.parse_args([])
    load_args(os.path.join(REPO_ROOT, args.config), config)
    dp = getattr(config, "dataset_path", "data/mit-states")
    config.dataset_path = dp if os.path.isabs(dp) else os.path.join(REPO_ROOT, dp)
    config.open_world = False
    K = int(getattr(config, "cluster_num", 5))

    ds = CompositionDataset(config.dataset_path, phase="train",
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    offset = len(attributes); nA, nO = len(attributes), len(classes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(os.path.join(REPO_ROOT, args.ckpt), map_location="cuda"))
    model.eval()

    # ---- load cached LLM sub-phrases (first K of 8) ----
    def load_phrases(kind, key):
        p = os.path.join(REPO_ROOT, args.llm_root, kind, f"{key.replace(' ', '_')}.json")
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        return d["prompts"][:K]

    # ---- trained prototypes from the checkpoint (queue space) ----
    sd = torch.load(os.path.join(REPO_ROOT, args.ckpt), map_location="cpu")

    km = torch.load(os.path.join(REPO_ROOT, args.kmeans_bank), map_location="cpu")

    # Build per-primitive: LLM anchors (joint), trained protos pushed through proj (joint),
    # trained protos raw (queue space), kmeans seeds raw (queue space).
    def proj_protos(prim_type, i, proj_module):
        q = sd[f"{prim_type}_queue{i}"].float().cuda()       # [K,D] queue space
        pj = proj_module(q)                                   # consumption path -> joint-ish
        return F.normalize(pj.float(), dim=-1).cpu(), F.normalize(q, dim=-1).cpu()

    results = {"K": K}
    rng = np.random.default_rng(0)

    for prim_type, names, n, proj_module in [
        ("attr", attributes, nA, model.attr_proj),
        ("obj", classes, nO, model.obj_proj),
    ]:
        pc1_llm, pc1_randn, pc1_shuf = [], [], []
        anchor_spread, proto_spread_joint, proto_spread_raw, km_spread = [], [], [], []
        missing = 0
        # Pre-embed all primitives' anchors (joint space)
        anchors = {}
        for i, nm in enumerate(names):
            ph = load_phrases("attribute" if prim_type == "attr" else "object", nm)
            if ph is None or len(ph) < K:
                missing += 1
                anchors[i] = None
            else:
                anchors[i] = encode_phrases(model, ph)          # [K,D] joint
        for i, nm in enumerate(names):
            if anchors[i] is None:
                continue
            A = anchors[i]
            P_joint, P_raw = proj_protos(prim_type, i, proj_module)
            # PC-1
            pc1_llm.append(greedy_match_cos(A, P_joint))
            randn = F.normalize(torch.randn(K, A.shape[1]), dim=-1)
            pc1_randn.append(greedy_match_cos(randn, P_joint))
            # shuffled-primitive control: anchors from a different random primitive
            others = [j for j in range(n) if anchors.get(j) is not None and j != i]
            jb = int(rng.choice(others))
            pc1_shuf.append(greedy_match_cos(anchors[jb], P_joint))
            # PC-2 spreads
            anchor_spread.append(intra_cos(A))
            proto_spread_joint.append(intra_cos(P_joint))
            proto_spread_raw.append(intra_cos(P_raw))
            km_spread.append(intra_cos(km[prim_type][i]))
        results[prim_type] = dict(
            n_used=len(pc1_llm), missing=missing,
            pc1_llm=float(np.mean(pc1_llm)),
            pc1_randn=float(np.mean(pc1_randn)),
            pc1_shuffled=float(np.mean(pc1_shuf)),
            anchor_intra_cos=float(np.mean(anchor_spread)),
            proto_intra_cos_joint=float(np.mean(proto_spread_joint)),
            proto_intra_cos_raw=float(np.mean(proto_spread_raw)),
            kmeans_intra_cos=float(np.mean(km_spread)),
        )

    # ---- report ----
    print("\n" + "=" * 92)
    print("  EXPERIMENT 2 — LLM-anchor pre-check (nonorm seed0 val_best, joint/consumption space)")
    print("=" * 92)
    for pt, lab in [("attr", "ATTRIBUTE"), ("obj", "OBJECT")]:
        r = results[pt]
        print(f"\n  [{lab}]  (n_used={r['n_used']}, missing={r['missing']})")
        print(f"    PC-1 greedy-match cos to TRAINED protos (joint/consumption space):")
        print(f"        LLM anchors      = {r['pc1_llm']:.4f}")
        print(f"        randn control    = {r['pc1_randn']:.4f}")
        print(f"        shuffled-prim    = {r['pc1_shuffled']:.4f}")
        print(f"        LLM lift vs shuffled = {r['pc1_llm']-r['pc1_shuffled']:+.4f}, vs randn = {r['pc1_llm']-r['pc1_randn']:+.4f}")
        print(f"    PC-2 intra-set spread (mean pairwise cos; LOWER=more diverse):")
        print(f"        LLM anchors      = {r['anchor_intra_cos']:.4f}")
        print(f"        kmeans seeds     = {r['kmeans_intra_cos']:.4f}")
        print(f"        trained protos (joint) = {r['proto_intra_cos_joint']:.4f}   (raw queue = {r['proto_intra_cos_raw']:.4f})")
    print("=" * 92)

    os.makedirs(SCRATCH, exist_ok=True)
    with open(os.path.join(SCRATCH, "llm_anchor_precheck.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[llm-anchor] saved {os.path.join(SCRATCH, 'llm_anchor_precheck.json')}")


if __name__ == "__main__":
    main()
