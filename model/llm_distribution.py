"""LLM-augmented text embedding distribution bank.

Each attribute/object/composition is represented by K diverse natural-language
descriptions, encoded once via the (frozen) CLIP text encoder and cached to disk.

Spec: coding_agent_instructions.md §2.

When an LLM API is unavailable, falls back to a deterministic template set
(see §4 of the spec). Pipeline-correct; quality scales with description source.
"""
from __future__ import annotations
import json, os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


# Deterministic K=8 template families. Vary phrasing & framing.
_COMP_TEMPLATES = [
    "a photo of {a} {o}.",
    "a close-up photograph of a {a} {o}.",
    "an image of a {a} {o} on a plain background.",
    "a picture showing a {a} {o}.",
    "a clear photo of a {a} {o}.",
    "a {a} {o}, photographed in natural light.",
    "a high-quality image of a {a} {o}.",
    "a {a} {o} in a typical setting.",
]
_ATTR_TEMPLATES = [
    "a photo of a {a} object.",
    "an image showing the property {a}.",
    "a close-up of something {a}.",
    "a clear photo of a {a} thing.",
    "a picture emphasizing how {a} an object looks.",
    "a {a} surface in natural light.",
    "a photograph of a {a} item.",
    "a {a} appearance, in detail.",
]
_OBJ_TEMPLATES = [
    "a photo of a {o}.",
    "a close-up photograph of a {o}.",
    "an image of a {o} on a plain background.",
    "a picture of a {o}.",
    "a clear photo of a {o}.",
    "a {o}, photographed in natural light.",
    "a high-quality image of a {o}.",
    "a {o} in a typical setting.",
]


def _clean(s: str) -> str:
    return s.replace(".", " ").replace("_", " ").lower().strip()


class LLMDistributionBank:
    """Holds [V, K, D] text embedding banks for attrs, objs, and pairs.

    Encoded once at construction (cached). All tensors live on CUDA in fp32.
    """

    def __init__(
        self,
        attributes: List[str],
        objects: List[str],
        pairs: List[Tuple[str, str]],
        text_encoder,
        tokenizer,
        clip_dtype,
        cache_path: str,
        K: int = 8,
        device: str = "cuda",
        context_length: int = 8,
        descriptions_root: str = None,
    ):
        self.K = K
        self.device = device
        self.attributes = [_clean(a) for a in attributes]
        self.objects = [_clean(o) for o in objects]
        self.pairs = [(_clean(a), _clean(o)) for (a, o) in pairs]
        self.attr2idx = {a: i for i, a in enumerate(self.attributes)}
        self.obj2idx = {o: i for i, o in enumerate(self.objects)}
        self.pair2idx = {p: i for i, p in enumerate(self.pairs)}
        self._pair_lookup = {(self.attr2idx[a], self.obj2idx[o]): i
                             for i, (a, o) in enumerate(self.pairs)}

        if cache_path and os.path.exists(cache_path):
            blob = torch.load(cache_path, map_location=device)
            self.attr = blob["attr"].to(device).float()
            self.obj = blob["obj"].to(device).float()
            self.comp = blob["comp"].to(device).float()
            assert self.attr.shape[1] == K and self.obj.shape[1] == K and self.comp.shape[1] == K, \
                f"cache K mismatch (got {self.attr.shape[1]}, want {K}); delete {cache_path}"
        else:
            attr_prompts = self._build_prompts(
                "attribute", self.attributes, descriptions_root, K,
                [[t.format(a=a) for t in _ATTR_TEMPLATES[:K]] for a in self.attributes],
            )
            obj_prompts = self._build_prompts(
                "object", self.objects, descriptions_root, K,
                [[t.format(o=o) for t in _OBJ_TEMPLATES[:K]] for o in self.objects],
            )
            comp_keys = [f"{a}__{o}" for (a, o) in self.pairs]
            comp_prompts = self._build_prompts(
                "composition", comp_keys, descriptions_root, K,
                [[t.format(a=a, o=o) for t in _COMP_TEMPLATES[:K]] for (a, o) in self.pairs],
            )
            self.attr = self._encode_bank(attr_prompts, text_encoder, tokenizer, clip_dtype, context_length)
            self.obj = self._encode_bank(obj_prompts, text_encoder, tokenizer, clip_dtype, context_length)
            self.comp = self._encode_bank(comp_prompts, text_encoder, tokenizer, clip_dtype, context_length)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save({"attr": self.attr.cpu(), "obj": self.obj.cpu(),
                            "comp": self.comp.cpu()}, cache_path)

        # Precompute L2-normed copies (for cosine inference)
        self.attr_n = F.normalize(self.attr, dim=-1)
        self.obj_n = F.normalize(self.obj, dim=-1)
        self.comp_n = F.normalize(self.comp, dim=-1)
        self.attr_mean_n = F.normalize(self.attr.mean(dim=1), dim=-1)
        self.obj_mean_n = F.normalize(self.obj.mean(dim=1), dim=-1)
        self.comp_mean_n = F.normalize(self.comp.mean(dim=1), dim=-1)

    @staticmethod
    def _build_prompts(kind: str, keys, descriptions_root, K: int, fallback):
        """For each key, load K LLM-generated lines from
        `{descriptions_root}/{kind}/{key}.json` (written by llm/generate.py);
        fall back to deterministic templates per-item if missing.
        """
        if not descriptions_root:
            return fallback
        out, n_real, n_fb = [], 0, 0
        for i, k in enumerate(keys):
            safe = k.replace("/", "_").replace(" ", "_")
            p = os.path.join(descriptions_root, kind, f"{safe}.json")
            if os.path.exists(p):
                try:
                    blob = json.load(open(p))
                    prompts = blob["prompts"][:K]
                    if len(prompts) < K:
                        prompts = prompts + fallback[i][:K - len(prompts)]
                    out.append(prompts)
                    n_real += 1
                    continue
                except Exception:
                    pass
            out.append(fallback[i])
            n_fb += 1
        print(f"[LLMDistributionBank] {kind}: {n_real}/{len(keys)} from LLM, {n_fb} fallback",
              flush=True)
        return out

    @torch.no_grad()
    def _encode_bank(self, batched_prompts, text_encoder, tokenizer, clip_dtype, context_length):
        """batched_prompts: list[V] of list[K] of str → tensor [V, K, D]."""
        V = len(batched_prompts)
        K = len(batched_prompts[0])
        # Flatten to V*K, tokenize, encode in chunks.
        flat = [s for row in batched_prompts for s in row]
        token_ids = torch.cat([tokenizer(s, context_length=context_length) for s in flat]).to(self.device)
        feats = []
        bs = 256
        for i in range(0, token_ids.size(0), bs):
            ti = token_ids[i:i+bs]
            f, _ = text_encoder(ti, None, enable_pos_emb=True)
            feats.append(f.float().cpu())
        out = torch.cat(feats, dim=0).reshape(V, K, -1).to(self.device)
        return out

    def sample_attr(self, attr_idx: torch.Tensor) -> torch.Tensor:
        """attr_idx: [B] longs → [B, D] randomly sampled embedding."""
        B = attr_idx.size(0)
        k = torch.randint(0, self.K, (B,), device=self.attr.device)
        return self.attr[attr_idx, k]

    def sample_obj(self, obj_idx: torch.Tensor) -> torch.Tensor:
        B = obj_idx.size(0)
        k = torch.randint(0, self.K, (B,), device=self.obj.device)
        return self.obj[obj_idx, k]

    def sample_comp(self, pair_idx: torch.Tensor) -> torch.Tensor:
        B = pair_idx.size(0)
        k = torch.randint(0, self.K, (B,), device=self.comp.device)
        return self.comp[pair_idx, k]

    def pair_indices(self, attrs: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        """Map (attr_idx, obj_idx) batched longs → pair_idx in self.pairs.

        Pairs not in bank get -1; caller must filter/handle. For closed-world training,
        all train pairs are in `pairs` (the full pair set passed at construction).
        """
        out = torch.full((attrs.size(0),), -1, dtype=torch.long, device=attrs.device)
        # Build a fast lookup
        if not hasattr(self, "_pair_lookup"):
            self._pair_lookup = {(self.attr2idx[a], self.obj2idx[o]): i
                                 for i, (a, o) in enumerate(self.pairs)}
        a_list = attrs.cpu().tolist()
        o_list = objs.cpu().tolist()
        for i, (a, o) in enumerate(zip(a_list, o_list)):
            out[i] = self._pair_lookup.get((a, o), -1)
        return out
