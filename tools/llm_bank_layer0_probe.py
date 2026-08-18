"""Layer 0 sanity probe for LLM-pair embedding banks.

Measures whether the precomputed CLIP-text bank
(`data/llm_descriptions/<dataset>/text_bank_K8_vitl14*.pt`) carries
enough pair-discriminative information to support PG-LPR-style additive
priors. Two reductions are probed:

  attr+obj decomposition:   LLM_attr[a] + LLM_obj[o]  (K-mean over attr/obj banks)
  pair-level direct:        bank['comp'][p]           (K-mean over comp bank)

For each, reports:
  - same-obj diff-attr cos
  - same-attr diff-obj cos
  - random pair cos
  - asymmetry = |same-obj - same-attr|
  - pair-spread (off-diag std)

Plus the alignment between pair-level direct and attr+obj reconstruction.

Used to set the entry gate for PG-LPR (asymmetry > 0.05, spread > 0.05).
Result on MIT-States / UT-Zap, both Claude and Qwen3b banks, both
reductions: asymmetry 0.0002 ~ 0.0119 -> gate fails by 7~250x.
PG-LPR sealed (RESEARCH_LOG entry: 2026-05-26).
"""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F


DATASETS = ["mit-states", "ut-zappos"]
BANK_VARIANTS = ["", "_qwen3b"]


def load_pair_list(dataset: str) -> list[tuple[str, str]]:
    base = f"data/{dataset}/compositional-split-natural"
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for fn in ("train_pairs.txt", "val_pairs.txt", "test_pairs.txt"):
        path = os.path.join(base, fn)
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                p = (parts[0], parts[1])
                if p not in seen:
                    seen.add(p)
                    ordered.append(p)
    return ordered


def probe_attr_obj(bank: dict, label: str) -> None:
    A = bank["attr"].mean(dim=1)
    O = bank["obj"].mean(dim=1)
    nA, nO = A.shape[0], O.shape[0]
    torch.manual_seed(0)
    N = 50_000

    o = torch.randint(0, nO, (N,))
    a1 = torch.randint(0, nA, (N,))
    a2 = torch.randint(0, nA, (N,))
    mask = a1 != a2
    o, a1, a2 = o[mask], a1[mask], a2[mask]
    so = (F.normalize(A[a1] + O[o], dim=-1) * F.normalize(A[a2] + O[o], dim=-1)).sum(-1)

    ax = torch.randint(0, nA, (N,))
    o1 = torch.randint(0, nO, (N,))
    o2 = torch.randint(0, nO, (N,))
    mask = o1 != o2
    ax, o1, o2 = ax[mask], o1[mask], o2[mask]
    sa = (F.normalize(A[ax] + O[o1], dim=-1) * F.normalize(A[ax] + O[o2], dim=-1)).sum(-1)

    ar1 = torch.randint(0, nA, (N,))
    ar2 = torch.randint(0, nA, (N,))
    or1 = torch.randint(0, nO, (N,))
    or2 = torch.randint(0, nO, (N,))
    mask = (ar1 != ar2) & (or1 != or2)
    ar1, ar2, or1, or2 = ar1[mask], ar2[mask], or1[mask], or2[mask]
    rd = (F.normalize(A[ar1] + O[or1], dim=-1) * F.normalize(A[ar2] + O[or2], dim=-1)).sum(-1)

    An = F.normalize(A, dim=-1)
    attr_std = (An @ An.t())[~torch.eye(nA, dtype=torch.bool)].std().item()

    print(f"=== {label}  attr+obj decomposition  (|A|={nA}, |O|={nO}) ===")
    print(f"  same-obj  diff-attr cos  mean={so.mean():.4f}  std={so.std():.4f}")
    print(f"  same-attr diff-obj  cos  mean={sa.mean():.4f}  std={sa.std():.4f}")
    print(f"  random              cos  mean={rd.mean():.4f}  std={rd.std():.4f}")
    asym = abs(so.mean() - sa.mean()).item()
    print(f"  asymmetry          = {asym:.4f}  (gate >0.02)")
    print(f"  LLM_attr inter-attr cos std = {attr_std:.4f}  (gate >0.05)")
    print()


def probe_pair_level(bank: dict, label: str, dataset: str) -> None:
    comp = bank["comp"].mean(dim=1)
    Np, D = comp.shape
    pairs = load_pair_list(dataset)
    if len(pairs) != Np:
        print(f"  [{label}] WARN: bank Np={Np} != dataset pairs={len(pairs)} — skipping classified stats")
        return
    attrs = sorted({a for a, _ in pairs})
    objs = sorted({o for _, o in pairs})
    a2i = {a: i for i, a in enumerate(attrs)}
    o2i = {o: i for i, o in enumerate(objs)}
    pa = torch.tensor([a2i[a] for a, _ in pairs])
    po = torch.tensor([o2i[o] for _, o in pairs])

    LLM_pair_n = F.normalize(comp, dim=-1)
    cos = LLM_pair_n @ LLM_pair_n.t()
    eye = ~torch.eye(Np, dtype=torch.bool)
    so_m = (po.unsqueeze(0) == po.unsqueeze(1)) & (pa.unsqueeze(0) != pa.unsqueeze(1))
    sa_m = (pa.unsqueeze(0) == pa.unsqueeze(1)) & (po.unsqueeze(0) != po.unsqueeze(1))
    rd_m = (pa.unsqueeze(0) != pa.unsqueeze(1)) & (po.unsqueeze(0) != po.unsqueeze(1))
    so, sa, rd = cos[so_m], cos[sa_m], cos[rd_m]
    spread = cos[eye].std().item()

    A = bank["attr"].mean(dim=1)
    O = bank["obj"].mean(dim=1)
    rec = torch.stack([A[a2i[a]] + O[o2i[o]] for a, o in pairs], dim=0)
    rec_n = F.normalize(rec, dim=-1)
    align = (LLM_pair_n * rec_n).sum(-1)

    print(f"=== {label}  pair-level direct  (Np={Np}) ===")
    print(f"  same-obj  diff-attr cos  mean={so.mean():.4f}  std={so.std():.4f}  n={len(so)}")
    print(f"  same-attr diff-obj  cos  mean={sa.mean():.4f}  std={sa.std():.4f}  n={len(sa)}")
    print(f"  random              cos  mean={rd.mean():.4f}  std={rd.std():.4f}  n={len(rd)}")
    asym = abs(so.mean() - sa.mean()).item()
    print(f"  asymmetry          = {asym:.4f}  (gate >0.05)")
    print(f"  pair spread (off-diag std) = {spread:.4f}  (gate >0.05)")
    print(f"  alignment(pair-level vs attr+obj) cos mean={align.mean():.4f}  std={align.std():.4f}")
    print()


def main() -> None:
    for ds in DATASETS:
        for variant in BANK_VARIANTS:
            path = f"data/llm_descriptions/{ds}/text_bank_K8_vitl14{variant}.pt"
            if not os.path.exists(path):
                continue
            bank = torch.load(path, map_location="cpu")
            label = f"{ds}{variant}"
            probe_attr_obj(bank, label)
            probe_pair_level(bank, label, ds)


if __name__ == "__main__":
    main()
