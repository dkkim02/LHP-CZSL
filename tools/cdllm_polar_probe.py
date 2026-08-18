"""CD-LLM v2 Layer 0 — polar pair list + direction-vector probe.

For each manual antonym candidate (a+, a-) we compute:
  d_clip = e_clip("a photo of {a+} object") - e_clip("a photo of {a-} object")
  d_llm  = mean_k e_clip(desc_{a+}^k) - mean_k e_clip(desc_{a-}^k)
plus a per-k version of d_llm (no mean) for self-consistency.

Reported metrics:
  cos(d_clip, d_llm)                 -- alignment between CLIP and LLM direction
  ||d_llm|| / ||d_clip||             -- energy ratio (low => direction degenerate)
  cos among K individual d_llm^k     -- LLM self-agreement on the axis
plus a control: random non-antonym attr pairs run through the same pipeline.

Calibration note: the originally proposed "CLIP cos < 0.4" filter is
unreachable in CLIP-text space (hub-cos floor ~0.7-0.85). We drop it and
report the actual Layer-0 metric over all candidates, with the random
control as the discriminating baseline.

Bank source: data/llm_descriptions/mit-states/text_bank_K8_vitl14.pt
"""
from __future__ import annotations
import os
from typing import Iterable
import torch
import torch.nn.functional as F


MITSTATES_ATTRS_PATH = "data/mit-states/compositional-split-natural/train_pairs.txt"
BANK_PATH = "data/llm_descriptions/mit-states/text_bank_K8_vitl14.pt"

# Manual polar candidate list. Each item: ("positive", "negative")
# Curated from the 115 MIT-States attrs; chosen for visual/perceptual polarity
# (filterable by CLIP cos later). Direction convention: "+" -> "-" reads
# naturally (e.g. modern -> ancient, full -> empty).
POLAR_CANDIDATES: list[tuple[str, str]] = [
    ("modern", "ancient"),
    ("new", "old"),
    ("young", "old"),
    ("fresh", "wilted"),
    ("ripe", "unripe"),
    ("cooked", "raw"),
    ("full", "empty"),
    ("inflated", "deflated"),
    ("dry", "wet"),
    ("dry", "damp"),
    ("frozen", "melted"),
    ("frozen", "thawed"),
    ("clean", "dirty"),
    ("clean", "grimy"),
    ("clean", "muddy"),
    ("bright", "dark"),
    ("clear", "cloudy"),
    ("clear", "foggy"),
    ("clear", "murky"),
    ("sharp", "blunt"),
    ("sharp", "dull"),
    ("smooth", "rough"),
    ("smooth", "wrinkled"),
    ("smooth", "creased"),
    ("straight", "bent"),
    ("straight", "curved"),
    ("straight", "coiled"),
    ("straight", "winding"),
    ("heavy", "lightweight"),
    ("large", "small"),
    ("large", "tiny"),
    ("huge", "tiny"),
    ("tall", "short"),
    ("thick", "thin"),
    ("wide", "narrow"),
    ("loose", "tight"),
    ("open", "closed"),
    ("painted", "unpainted"),
    ("whole_proxy_via_complement", None),  # placeholder; whole missing
    ("upright", "fallen"),
    ("upright", "toppled"),
    ("standing", "fallen"),
    ("shiny", "rusty"),
    ("shiny", "dull"),
    ("shiny", "weathered"),
    ("verdant", "barren"),
    ("verdant", "eroded"),
    ("filled", "empty"),
]


def load_attrs() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for split in ("train_pairs.txt", "val_pairs.txt", "test_pairs.txt"):
        with open(f"data/mit-states/compositional-split-natural/{split}") as fh:
            for line in fh:
                parts = line.strip().split()
                if parts and parts[0] not in seen:
                    seen.add(parts[0])
                    out.append(parts[0])
    return sorted(out)


def load_clip_l14():
    """Return (model, tokenize, device) for CLIP ViT-L/14 text encoding,
    matching the cluspro_baseline preprocessing convention (project's own
    clip_modules wrapper)."""
    from clip_modules.clip_model import load_clip
    from clip_modules.tokenization_clip import SimpleTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_clip("ViT-L/14", device=device)
    model.eval()
    tokenizer = SimpleTokenizer()

    def tokenize(prompts):
        return torch.cat([tokenizer(p, context_length=77) for p in prompts]).to(device)

    return model, tokenize, device


@torch.no_grad()
def clip_text_embed(words: Iterable[str], model, tokenize, device) -> torch.Tensor:
    """Encode each word as a sentence 'a photo of {word} object' to match
    CLIP usage in CZSL. Returns (N, D) float32 on cpu, unnormalized."""
    prompts = [f"a photo of {w} object" for w in words]
    tokens = tokenize(prompts)
    feats = model.encode_text(tokens).float().cpu()
    return feats


def direction_metrics(d_clip, d_llm_k):
    """Return cos(d_clip, d_llm_mean), ||d_clip||, ||d_llm_mean||, mean
    pairwise cos among the K individual d_llm^k directions."""
    d_llm_mean = d_llm_k.mean(dim=0)
    d_clip_n = F.normalize(d_clip, dim=-1)
    d_llm_n = F.normalize(d_llm_mean, dim=-1)
    cos_align = (d_clip_n * d_llm_n).sum().item()
    nc = d_clip.norm().item()
    nl = d_llm_mean.norm().item()
    # K-individual self-consistency
    d_llm_k_n = F.normalize(d_llm_k, dim=-1)
    K = d_llm_k_n.shape[0]
    self_cos = (d_llm_k_n @ d_llm_k_n.t())
    off = ~torch.eye(K, dtype=torch.bool)
    k_self = self_cos[off].mean().item()
    return cos_align, nc, nl, k_self


def main() -> None:
    attrs = set(load_attrs())
    print(f"MIT-States attrs: {len(attrs)}")

    pairs = [(p, n) for (p, n) in POLAR_CANDIDATES if n is not None]
    pairs = [(p, n) for (p, n) in pairs if p in attrs and n in attrs]
    print(f"Polar candidates (in attr list): {len(pairs)}")
    print("(CLIP cos<0.4 filter dropped — unreachable in CLIP-text space; "
          "we use a random-pair control instead)\n")

    model, tokenize, device = load_clip_l14()
    unique_attrs = sorted({a for pair in pairs for a in pair})
    name2idx = {a: i for i, a in enumerate(unique_attrs)}
    e_clip = clip_text_embed(unique_attrs, model, tokenize, device)

    # Per-K bank: bank['attr'] shape (115, K=8, D)
    bank = torch.load(BANK_PATH, map_location="cpu")
    bank_attr_order = sorted(attrs)
    a2bi = {a: i for i, a in enumerate(bank_attr_order)}
    A_llm_k = bank["attr"]  # (115, 8, 768)
    K = A_llm_k.shape[1]
    print(f"LLM bank: {A_llm_k.shape}  K={K}\n")

    # ----- polar pair metrics -----
    print(f"{'polar pair':<32} {'cos(d_clip,d_llm)':>18} {'||d_clip||':>11} {'||d_llm||':>11} "
          f"{'nl/nc':>7} {'K self-cos':>11}")
    print("-" * 96)
    rows: list[tuple] = []
    for (p, n) in pairs:
        d_clip = e_clip[name2idx[p]] - e_clip[name2idx[n]]
        d_llm_k = A_llm_k[a2bi[p]] - A_llm_k[a2bi[n]]  # (K, D)
        cos, nc, nl, k_self = direction_metrics(d_clip, d_llm_k)
        rows.append((p, n, cos, nc, nl, k_self))
        print(f"{p+' ↔ '+n:<32} {cos:>18.4f} {nc:>11.3f} {nl:>11.3f} "
              f"{nl/nc:>7.3f} {k_self:>11.4f}")

    cos_arr = torch.tensor([r[2] for r in rows])
    nc_arr = torch.tensor([r[3] for r in rows])
    nl_arr = torch.tensor([r[4] for r in rows])
    ratio_arr = nl_arr / nc_arr
    ks_arr = torch.tensor([r[5] for r in rows])
    print("\n--- POLAR summary ---")
    print(f"  N pairs                          = {len(rows)}")
    print(f"  cos(d_clip, d_llm)   mean={cos_arr.mean():.4f}  std={cos_arr.std():.4f}  "
          f"min={cos_arr.min():.4f}  max={cos_arr.max():.4f}")
    print(f"  ||d_clip||           mean={nc_arr.mean():.3f}  std={nc_arr.std():.3f}")
    print(f"  ||d_llm||            mean={nl_arr.mean():.3f}  std={nl_arr.std():.3f}")
    print(f"  ||d_llm||/||d_clip|| mean={ratio_arr.mean():.3f}  std={ratio_arr.std():.3f}")
    print(f"  K self-cos (within polar dir)    mean={ks_arr.mean():.4f}  std={ks_arr.std():.4f}")

    # ----- random control: non-antonym random attr pairs -----
    print("\n--- RANDOM CONTROL (non-antonym attr pairs from full 115 attr set) ---")
    all_attrs_sorted = bank_attr_order
    e_all = clip_text_embed(all_attrs_sorted, model, tokenize, device)
    polar_set = {(p, n) for p, n in pairs} | {(n, p) for p, n in pairs}
    torch.manual_seed(0)
    R = 200
    sampled = []
    while len(sampled) < R:
        i = torch.randint(0, len(all_attrs_sorted), (1,)).item()
        j = torch.randint(0, len(all_attrs_sorted), (1,)).item()
        if i == j:
            continue
        a, b = all_attrs_sorted[i], all_attrs_sorted[j]
        if (a, b) in polar_set or (b, a) in polar_set:
            continue
        sampled.append((a, b, i, j))

    cos_r, ratio_r, ks_r = [], [], []
    for (a, b, i, j) in sampled:
        d_clip = e_all[i] - e_all[j]
        d_llm_k = A_llm_k[i] - A_llm_k[j]
        cos, nc, nl, k_self = direction_metrics(d_clip, d_llm_k)
        cos_r.append(cos); ratio_r.append(nl / max(nc, 1e-8)); ks_r.append(k_self)
    cos_r = torch.tensor(cos_r); ratio_r = torch.tensor(ratio_r); ks_r = torch.tensor(ks_r)
    print(f"  N random pairs                   = {R}")
    print(f"  cos(d_clip, d_llm)   mean={cos_r.mean():.4f}  std={cos_r.std():.4f}")
    print(f"  ||d_llm||/||d_clip|| mean={ratio_r.mean():.3f}  std={ratio_r.std():.3f}")
    print(f"  K self-cos                       mean={ks_r.mean():.4f}  std={ks_r.std():.4f}")

    print("\n--- POLAR vs RANDOM deltas ---")
    print(f"  Δ cos(d_clip,d_llm)  = {(cos_arr.mean() - cos_r.mean()):+.4f}  "
          f"(polar - random)")
    print(f"  Δ K self-cos         = {(ks_arr.mean() - ks_r.mean()):+.4f}")
    print(f"  Δ nl/nc ratio        = {(ratio_arr.mean() - ratio_r.mean()):+.4f}")

    out_path = "data/llm_descriptions/mit-states/polar_pairs_v2.txt"
    with open(out_path, "w") as fh:
        fh.write("#attr_pos\tattr_neg\tcos_clip_llm\t||d_clip||\t||d_llm||\tK_self_cos\n")
        for p, n, cos, nc, nl, ks in rows:
            fh.write(f"{p}\t{n}\t{cos:.4f}\t{nc:.3f}\t{nl:.3f}\t{ks:.4f}\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
