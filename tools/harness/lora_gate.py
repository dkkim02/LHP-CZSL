"""
Round-6 Vision-LoRA MANDATORY step-0 gate + param accounting (NOT training).

Each scenario builds a fresh model (B=0 LoRA init) and compares the encode_image CLS
logits-space output (the projected CLS feature) against a reference, on one fixed image
batch, with identical seed. A silent attention bug shows up here as a >1e-4 mismatch.

GATE-1: lora_rank>0, in_out, drop_adapters=False  -> MUST equal the adapters-only baseline.
GATE-2: lora_rank>0, in_out, drop_adapters=True   -> MUST equal the no-adapter frozen-CLIP path.
GATE-3 (out-only): lora_rank>0, out, drop_adapters=False -> MUST equal adapters-only baseline.
Also prints trainable param counts for the two training arms.

Runs scenarios in separate processes (10GB card). Driver aggregates.

Usage: CUDA_VISIBLE_DEVICES=5 python tools/harness/lora_gate.py
"""
import os, sys, json, subprocess
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SH = "/home/jgshin22/.claude/jobs/664681e9/tmp"
CKPT = "checkpoint/cluspro_baseline_l14_mit_v2_nonorm_full_seed0/val_best.pt"
CFG = "config/cluspro_baseline_mit_l14_v2_nonorm_full_seed0.yml"


def build(overrides):
    from parameters import parser
    from utils import load_args
    c = parser.parse_args([])
    load_args(os.path.join(REPO, CFG), c)
    c.dataset_path = os.path.join(REPO, "data/mit-states")
    c.open_world = False
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def named_trainable_groups(model):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = n.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    return groups


def run(name):
    import numpy as np
    from dataset import CompositionDataset
    from model.model_factory import get_model
    from utils import load_args, set_seed
    from torch.utils.data import DataLoader

    specs = {
        # reference: plain adapters baseline (no lora) — loads the trained nonorm ckpt
        "ref_adapters": dict(lora_rank=0, drop_adapters=False, load_ckpt=True),
        # reference: no-adapter frozen path (drop adapters, no lora) — loads ckpt then bypasses adapters
        "ref_noadapter": dict(lora_rank=0, drop_adapters=True, load_ckpt=True),
        # GATE-1: in_out lora, adapters kept, B=0
        "gate1_inout_keep": dict(lora_rank=32, lora_targets="in_out", drop_adapters=False, load_ckpt=True),
        # GATE-2: in_out lora, adapters dropped, B=0
        "gate2_inout_drop": dict(lora_rank=32, lora_targets="in_out", drop_adapters=True, load_ckpt=True),
        # GATE-3: out-only lora, adapters kept, B=0
        "gate3_out_keep": dict(lora_rank=64, lora_targets="out", drop_adapters=False, load_ckpt=True),
        # param-accounting builds (fresh, no ckpt needed)
        "params_matched": dict(lora_rank=32, lora_targets="in_out", drop_adapters=True, load_ckpt=False),
        "params_r8": dict(lora_rank=8, lora_targets="in_out", drop_adapters=True, load_ckpt=False),
        "params_out64": dict(lora_rank=64, lora_targets="out", drop_adapters=True, load_ckpt=False),
        "params_adapters": dict(lora_rank=0, drop_adapters=False, load_ckpt=False),
    }
    spec = dict(specs[name])
    load_ckpt = spec.pop("load_ckpt")
    cfg = build(spec)

    ds = CompositionDataset(cfg.dataset_path, phase="val",
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    set_seed(0)
    model = get_model(cfg, attributes=attributes, classes=classes, offset=len(attributes)).cuda()
    if load_ckpt:
        sd = torch.load(os.path.join(REPO, CKPT), map_location="cuda")
        # ckpt has no lora_* keys; load non-strict, then assert only lora_* are missing.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [m for m in missing if not (m.startswith("lora_"))]
        assert not bad, f"unexpected missing (non-lora) keys: {bad[:5]}"
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model.eval()

    tp = trainable_params(model)
    groups = named_trainable_groups(model)

    out = dict(name=name, trainable=tp, groups=groups,
               lora_rank=getattr(cfg, "lora_rank", 0),
               lora_targets=getattr(cfg, "lora_targets", "out"),
               drop_adapters=getattr(cfg, "drop_adapters", False))

    if load_ckpt:
        # deterministic single batch CLS feature (projected) — the logits-space signal.
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
        batch = next(iter(loader))
        with torch.no_grad():
            cls, _ = model.encode_image(batch[0].cuda().type(model.clip.dtype))
        out["cls_sig"] = cls.float().cpu()  # [16, 768]
        torch.save(out, os.path.join(SH, f"lora_gate_{name}.pt"))
        print(f"[{name}] trainable={tp:,}  cls_sig saved {tuple(cls.shape)}")
    else:
        torch.save(out, os.path.join(SH, f"lora_gate_{name}.pt"))
        print(f"[{name}] trainable={tp:,}  groups={ {k:v for k,v in groups.items()} }")


def compare():
    def load(n):
        return torch.load(os.path.join(SH, f"lora_gate_{n}.pt"), map_location="cpu")
    R = {n: load(n) for n in ["ref_adapters", "ref_noadapter", "gate1_inout_keep",
                              "gate2_inout_drop", "gate3_out_keep",
                              "params_matched", "params_r8", "params_out64", "params_adapters"]}

    def maxdiff(a, b):
        return float((a["cls_sig"] - b["cls_sig"]).abs().max())

    print("\n" + "=" * 78)
    print("  ROUND-6 VISION-LoRA STEP-0 GATE")
    print("=" * 78)
    d1 = maxdiff(R["gate1_inout_keep"], R["ref_adapters"])
    d2 = maxdiff(R["gate2_inout_drop"], R["ref_noadapter"])
    d3 = maxdiff(R["gate3_out_keep"], R["ref_adapters"])
    print(f"  GATE-1 in_out+keep  vs adapters-baseline : max|Δ|={d1:.2e}  {'PASS' if d1 < 1e-4 else 'FAIL'}")
    print(f"  GATE-2 in_out+drop  vs no-adapter path    : max|Δ|={d2:.2e}  {'PASS' if d2 < 1e-4 else 'FAIL'}")
    print(f"  GATE-3 out-only+keep vs adapters-baseline : max|Δ|={d3:.2e}  {'PASS' if d3 < 1e-4 else 'FAIL'}")
    # sanity: ref_adapters and ref_noadapter MUST differ (adapters do something)
    dref = maxdiff(R["ref_adapters"], R["ref_noadapter"])
    print(f"  (sanity) adapters vs no-adapter differ     : max|Δ|={dref:.2e}  {'ok' if dref > 1e-3 else 'SUSPICIOUS'}")

    print("\n  PARAM ACCOUNTING (trainable):")
    print(f"    adapters baseline      : {R['params_adapters']['trainable']:,}")
    print(f"    PRIMARY in_out r=32 (drop): {R['params_matched']['trainable']:,}  groups={R['params_matched']['groups']}")
    print(f"    SECONDARY in_out r=8 (drop): {R['params_r8']['trainable']:,}")
    print(f"    alt out-only r=64 (drop)  : {R['params_out64']['trainable']:,}")
    print("=" * 78)
    ok = d1 < 1e-4 and d2 < 1e-4 and d3 < 1e-4 and dref > 1e-3
    print("  GATE RESULT:", "PASS" if ok else "FAIL")
    return ok


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--compare":
        sys.exit(0 if compare() else 1)
    if len(sys.argv) >= 2:
        run(sys.argv[1]); return
    names = ["ref_adapters", "ref_noadapter", "gate1_inout_keep", "gate2_inout_drop",
             "gate3_out_keep", "params_matched", "params_r8", "params_out64", "params_adapters"]
    env = dict(os.environ)
    for n in names:
        print(f">>> {n}")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), n], env=env)
        if r.returncode != 0:
            print(f"[driver] {n} FAILED rc={r.returncode}"); sys.exit(2)
    sys.exit(0 if compare() else 1)


if __name__ == "__main__":
    main()
