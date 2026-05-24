"""Local-LLM K visual descriptions per attribute / object / composition.

Spec §2.2 prompts. Uses HuggingFace transformers + chat template.
One generation per item; the model is asked to return K distinct lines.
Outputs JSON files under cache/llm_descriptions/{dataset}/{type}/{key}.json.

Run example:
  python -m llm.generate --dataset ut-zappos --K 8 \
      --model Qwen/Qwen2.5-3B-Instruct --device cuda:2
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import CompositionDataset


def _clean(s: str) -> str:
    return s.replace(".", " ").replace("_", " ").lower().strip()


PROMPT_COMP = (
    'Generate exactly {K} diverse, photographic descriptions of "{attr} {obj}". '
    "Each description must be a single short sentence focusing on visual appearance. "
    "Vary phrasing, viewpoint, and adjectives but stay faithful to the literal meaning. "
    "Output one description per line. Do not number them. Do not add commentary."
)
PROMPT_ATTR = (
    'Generate exactly {K} diverse, photographic descriptions emphasizing the visual property "{attr}". '
    "Each description should describe a generic object exhibiting this property, "
    "focusing on what the property looks like visually. "
    "Output one description per line. Do not number them. Do not add commentary."
)
PROMPT_OBJ = (
    'Generate exactly {K} diverse photographic descriptions of "{obj}". '
    "Each description must be a single short sentence focusing on visual appearance. "
    "Vary viewpoint, lighting, and surroundings but keep the object as the main subject. "
    "Output one description per line. Do not number them. Do not add commentary."
)


_TEMPLATE_COMP = [
    "a photo of a {a} {o}.", "a close-up photograph of a {a} {o}.",
    "an image of a {a} {o} on a plain background.", "a picture showing a {a} {o}.",
    "a clear photo of a {a} {o}.", "a {a} {o}, photographed in natural light.",
    "a high-quality image of a {a} {o}.", "a {a} {o} in a typical setting.",
]
_TEMPLATE_ATTR = [
    "a photo of a {a} object.", "an image showing the property {a}.",
    "a close-up of something {a}.", "a clear photo of a {a} thing.",
    "a picture emphasizing how {a} an object looks.", "a {a} surface in natural light.",
    "a photograph of a {a} item.", "a {a} appearance, in detail.",
]
_TEMPLATE_OBJ = [
    "a photo of a {o}.", "a close-up photograph of a {o}.",
    "an image of a {o} on a plain background.", "a picture of a {o}.",
    "a clear photo of a {o}.", "a {o}, photographed in natural light.",
    "a high-quality image of a {o}.", "a {o} in a typical setting.",
]


def _parse_lines(text: str, K: int) -> List[str]:
    raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in raw:
        ln = re.sub(r"^\s*[\d]+[\.\):\-]\s*", "", ln)
        ln = re.sub(r"^\s*[-*•]\s*", "", ln)
        ln = ln.strip().strip('"').strip("'")
        if len(ln) > 3:
            cleaned.append(ln)
    return cleaned[:K]


@torch.no_grad()
def _generate_chat(model, tok, system_prompt: str, user_prompt: str, max_new=512) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(
        inputs, max_new_tokens=max_new, do_sample=True, temperature=0.85, top_p=0.95,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def _gen_one(model, tok, prompt_user: str, K: int, fallback: List[str]) -> List[str]:
    sys_p = "You write concise, varied, photographic image captions. Always output exactly the requested number of lines."
    for attempt in range(5):
        text = _generate_chat(model, tok, sys_p, prompt_user)
        lines = _parse_lines(text, K)
        if len(lines) >= K:
            return lines[:K]
    # fallback: pad with templates
    n_missing = K - len(lines)
    out = list(lines) + fallback[:n_missing] if lines else fallback[:K]
    return out[:K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ut-zappos")
    ap.add_argument("--dataset_path", default=None,
                    help="path to dataset root; default data/{dataset}")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache_root", default="cache/llm_descriptions")
    ap.add_argument("--torch_dtype", default="float16", choices=["float16","bfloat16","float32"])
    args = ap.parse_args()

    dataset_path = args.dataset_path or f"data/{args.dataset}"
    out_dir = Path(args.cache_root) / args.dataset
    (out_dir / "attribute").mkdir(parents=True, exist_ok=True)
    (out_dir / "object").mkdir(parents=True, exist_ok=True)
    (out_dir / "composition").mkdir(parents=True, exist_ok=True)

    print(f"[gen] loading {args.model} on {args.device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.torch_dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    model.eval()

    ds_tr = CompositionDataset(dataset_path, "train", "compositional-split-natural")
    ds_va = CompositionDataset(dataset_path, "val",   "compositional-split-natural")
    ds_te = CompositionDataset(dataset_path, "test",  "compositional-split-natural")
    attrs = [_clean(a) for a in ds_tr.attrs]
    objs  = [_clean(o) for o in ds_tr.objs]
    all_pairs = sorted({(_clean(a), _clean(o))
                        for ds in (ds_tr, ds_va, ds_te) for (a, o) in ds.pairs})
    print(f"[gen] attrs={len(attrs)}  objs={len(objs)}  pairs={len(all_pairs)}", flush=True)

    def _path(kind, key):
        safe = key.replace("/", "_").replace(" ", "_")
        return out_dir / kind / f"{safe}.json"

    t0 = time.time()
    total = 0; done = 0
    items = (
        [("attribute", a, a, [t.format(a=a) for t in _TEMPLATE_ATTR],
          PROMPT_ATTR.format(K=args.K, attr=a)) for a in attrs]
        + [("object", o, o, [t.format(o=o) for t in _TEMPLATE_OBJ],
            PROMPT_OBJ.format(K=args.K, obj=o)) for o in objs]
        + [("composition", f"{a}__{o}", f"{a} {o}",
            [t.format(a=a, o=o) for t in _TEMPLATE_COMP],
            PROMPT_COMP.format(K=args.K, attr=a, obj=o)) for (a, o) in all_pairs]
    )
    total = len(items)
    for kind, key, label, fb, user_prompt in items:
        p = _path(kind, key)
        if p.exists():
            done += 1
            continue
        lines = _gen_one(model, tok, user_prompt, args.K, fb)
        p.write_text(json.dumps({
            "key": key, "type": kind, "label": label, "K": args.K, "prompts": lines,
        }, ensure_ascii=False, indent=2))
        done += 1
        if done % 10 == 0 or done == total:
            elapsed = time.time() - t0
            eta = elapsed * (total - done) / max(done, 1)
            print(f"[gen] {done}/{total}  elapsed {elapsed:.0f}s  eta {eta:.0f}s  last={kind}:{key}",
                  flush=True)

    print(f"[gen] done in {time.time()-t0:.0f}s. cache at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
