"""
Generalized 2nd-encoder ensemble probe (val-tuned alpha, per-image z-score blend).

Supports an arbitrary 2nd image/text encoder as the ensemble partner for a ClusPro-like
checkpoint, to test cross-pretraining complementarity (Tier 1/2) and model-agnosticism (Tier 3).

Encoder specs (--encoder):
  openclip:hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP   (Recap, our winner)
  openclip:ViT-L-14:laion2b_s32b_b82k                  (open_clip arch:pretrained)
  openai:ViT-L/14                                       (OpenAI CLIP via `clip` pkg, cached)
  openai:ViT-B/16

STAGE 1 (--stage encode, run in env WITH open_clip + clip, e.g. mosaic3d):
  Encode a (dataset,split)'s images with the encoder's OWN preprocessing in CompositionDataset.data
  order; "a photo of {a} {o}" text over dset.pairs; save cosine logits[N,P] + GT to a per-(enc,dataset,split) dump.
STAGE 2 (--stage score, run in CZSL env):
  ClusPro (config,ckpt) closed-world scores on val+test; z-score each per image; tune alpha on VAL,
  freeze, apply to TEST. Report ClusPro-alone, 2nd-alone, val-tuned ensemble, oracle gap, decorrelation%.

Usage:
  CUDA_VISIBLE_DEVICES=1 $MOS  tools/harness/probe_ensemble_2nd.py --stage encode --encoder openai:ViT-L/14 \
      --enc_tag openaiL14 --dataset mit-states --split val
  CUDA_VISIBLE_DEVICES=1 $CZSL tools/harness/probe_ensemble_2nd.py --stage score --enc_tag openaiL14 \
      --dataset mit-states --config <cfg> --ckpt <ckpt> --tag mit_openaiL14_s0
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"
DATA_DIR = {"mit-states": "data/mit-states", "ut-zappos": "data/ut-zappos"}


def dump_path(enc_tag, dataset, split):
    return os.path.join(SCRATCH, f"enc_{enc_tag}_{dataset}_{split}.pt")


# ----------------------------------------------------- encoder loaders (stage 1 env)
def load_encoder(spec):
    """Returns (encode_image_fn(PIL->feat normalized cpu batch), tokenizer_encode_text_fn, logit_scale)."""
    if spec.startswith("openclip:"):
        import open_clip
        rest = spec[len("openclip:"):]
        if rest.startswith("hf-hub:"):
            model, _, preprocess = open_clip.create_model_and_transforms(rest)
            tokenizer = open_clip.get_tokenizer(rest)
        else:
            arch, pretrained = rest.split(":")
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(arch)
        model = model.cuda().eval()
        ls = model.logit_scale.exp().item()

        @torch.no_grad()
        def enc_img(pil_batch):
            x = torch.stack([preprocess(im) for im in pil_batch], 0).cuda()
            return F.normalize(model.encode_image(x).float(), dim=-1).cpu()

        @torch.no_grad()
        def enc_txt(prompts):
            out = []
            for s in range(0, len(prompts), 256):
                toks = tokenizer(prompts[s:s + 256]).cuda()
                out.append(F.normalize(model.encode_text(toks).float(), dim=-1).cpu())
            return torch.cat(out, 0)
        return enc_img, enc_txt, ls

    elif spec.startswith("openai:"):
        import clip
        name = spec[len("openai:"):]
        model, preprocess = clip.load(name, device="cuda")
        model = model.eval()
        ls = model.logit_scale.exp().item()

        @torch.no_grad()
        def enc_img(pil_batch):
            x = torch.stack([preprocess(im) for im in pil_batch], 0).cuda()
            return F.normalize(model.encode_image(x).float(), dim=-1).cpu()

        @torch.no_grad()
        def enc_txt(prompts):
            out = []
            for s in range(0, len(prompts), 256):
                toks = clip.tokenize(prompts[s:s + 256]).cuda()
                out.append(F.normalize(model.encode_text(toks).float(), dim=-1).cpu())
            return torch.cat(out, 0)
        return enc_img, enc_txt, ls
    else:
        raise ValueError(f"unknown encoder spec: {spec}")


# ----------------------------------------------------- STAGE 1: encode
@torch.no_grad()
def stage_encode(encoder, enc_tag, dataset, split):
    from dataset import CompositionDataset
    from PIL import Image

    dpath = os.path.join(REPO, DATA_DIR[dataset])
    ds = CompositionDataset(dpath, phase=split, split="compositional-split-natural", open_world=False)
    a2i, o2i = ds.attr2idx, ds.obj2idx
    print(f"[enc {enc_tag} {dataset}/{split}] loading {encoder} ...", flush=True)
    enc_img, enc_txt, ls = load_encoder(encoder)

    prompts = [f"a photo of {a.replace('.',' ').lower()} {o.replace('.',' ').lower()}" for (a, o) in ds.pairs]
    text_feat = enc_txt(prompts)
    P = len(prompts)
    print(f"[enc {enc_tag} {dataset}/{split}] text {tuple(text_feat.shape)} ls={ls:.2f}", flush=True)

    img_root = ds.root + "/images/"
    N = len(ds.data)
    agt, ogt, pgt, feats, buf = [], [], [], [], []
    t0 = time.time()

    def flush():
        if buf:
            feats.append(enc_img(buf))

    for i, (image, attr, obj) in enumerate(ds.data):
        fp = img_root + image
        if not os.path.exists(fp):
            parts = image.split("/")
            if len(parts) >= 2:
                parts[0] = parts[0].replace("_", " ")
                fp = img_root + "/".join(parts)
        buf.append(Image.open(fp).convert("RGB"))
        agt.append(a2i[attr]); ogt.append(o2i[obj]); pgt.append(ds.pair2idx[(attr, obj)])
        if len(buf) >= 64:
            flush(); buf = []
        if (i + 1) % 3000 == 0:
            print(f"[enc {enc_tag} {dataset}/{split}] img {i+1}/{N} {time.time()-t0:.0f}s", flush=True)
    flush()
    image_feat = torch.cat(feats, 0)
    logits = ls * (image_feat @ text_feat.t())
    out = dict(logits=logits, attr_gt=torch.tensor(agt), obj_gt=torch.tensor(ogt),
               pair_gt=torch.tensor(pgt), encoder=encoder, logit_scale=ls)
    os.makedirs(SCRATCH, exist_ok=True)
    torch.save(out, dump_path(enc_tag, dataset, split))
    print(f"[enc {enc_tag} {dataset}/{split}] saved {dump_path(enc_tag,dataset,split)} imgs={image_feat.shape[0]} P={P}", flush=True)


# ----------------------------------------------------- STAGE 2: score (CZSL env)
@torch.no_grad()
def clus_scores(config_path, ckpt, dataset, split):
    from parameters import parser
    from dataset import CompositionDataset
    from model.model_factory import get_model
    from utils import load_args
    from torch.utils.data import DataLoader

    config = parser.parse_args([])
    load_args(os.path.join(REPO, config_path), config)
    config.dataset_path = os.path.join(REPO, DATA_DIR[dataset])
    config.open_world = False
    ds = CompositionDataset(config.dataset_path, phase=split,
                            split="compositional-split-natural", open_world=False)
    attributes = [a.replace(".", " ").lower() for a in ds.attrs]
    classes = [c.replace(".", " ").lower() for c in ds.objs]
    # flow_composer needs attach_pairs over the closed-world universe (matches test.py)
    model = get_model(config, attributes=attributes, classes=classes, offset=len(attributes)).cuda()
    if getattr(config, "model_name", "") == "flow_composer":
        tr = CompositionDataset(config.dataset_path, "train", "compositional-split-natural", open_world=False)
        te = CompositionDataset(config.dataset_path, "test", "compositional-split-natural", open_world=False)
        all_pairs = list({p for d in (tr, ds, te) for p in d.pairs})
        model.attach_pairs(all_pairs)
    sd = torch.load(os.path.join(REPO, ckpt), map_location="cuda")
    # Bidirectional Disentangler norm-key remap to match THIS model's naming.
    model_keys = set(model.state_dict().keys())
    m_norm = any(".norm." in k for k in model_keys); m_bn = any(".bn1_fc." in k for k in model_keys)
    s_norm = any(".norm." in k for k in sd); s_bn = any(".bn1_fc." in k for k in sd)
    if m_norm and s_bn and not s_norm:
        sd = {k.replace(".bn1_fc.", ".norm."): v for k, v in sd.items()}
    elif m_bn and s_norm and not s_bn:
        sd = {k.replace(".norm.", ".bn1_fc."): v for k, v in sd.items()}
    # lhp_czsl (text_ensemble off) has a few unused sub-meaning buffers absent in ckpt -> non-strict.
    model.load_state_dict(sd, strict=(getattr(config, "model_name", "") != "lhp_czsl"))
    model.eval()
    a2i, o2i = ds.attr2idx, ds.obj2idx
    pairs = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.pairs]).cuda()
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    comps, attrs, objs, agt, ogt, pgt = [], [], [], [], [], []
    for batch in loader:
        out = model.val_forward(batch, pairs)
        c, a, o = out[0], out[1], out[2]
        comps.append(c.cpu()); attrs.append(a.cpu()); objs.append(o.cpu())
        agt.append(batch[1]); ogt.append(batch[2]); pgt.append(batch[3])
    COMP = torch.cat(comps); ATTR = torch.cat(attrs); OBJ = torch.cat(objs)
    agt = torch.cat(agt); ogt = torch.cat(ogt); pgt = torch.cat(pgt)
    pa, po = pairs[:, 0].cpu(), pairs[:, 1].cpu()
    S = COMP * 1.0 + F.softmax(ATTR, -1)[:, pa] * F.softmax(OBJ, -1)[:, po]
    return S, agt, ogt, pgt, ds


def zscore_closed(S, closed):
    Sz = torch.full_like(S, -1e10)
    cs = S[:, closed]
    cs = (cs - cs.mean(1, keepdim=True)) / (cs.std(1, keepdim=True) + 1e-6)
    Sz[:, closed] = cs
    return Sz


def _load_dataset(dataset, split):
    from parameters import parser
    from dataset import CompositionDataset
    from utils import load_args
    c = parser.parse_args([])
    c.dataset_path = os.path.join(REPO, DATA_DIR[dataset]); c.open_world = False
    return CompositionDataset(c.dataset_path, phase=split, split="compositional-split-natural", open_world=False)


def stage_score(enc_tag, dataset, config_path, ckpt, tag, clus2_config=None, clus2_ckpt=None,
                base_enc_tag=None):
    """2nd expert = encoder dump (enc_tag) OR 2nd ClusPro (clus2_*).
    BASE = ClusPro (config,ckpt) OR an encoder dump (base_enc_tag, e.g. basic-CLIP-as-method)."""
    from test import Evaluator, test as test_eval

    use_clus2 = clus2_ckpt is not None
    base_from_dump = base_enc_tag is not None
    splits = {}
    for split in ["val", "test"]:
        if base_from_dump:
            ds = _load_dataset(dataset, split)
            bd = torch.load(dump_path(base_enc_tag, dataset, split), map_location="cpu")
            S_clus = bd["logits"].float()
            agt, ogt, pgt = bd["attr_gt"], bd["obj_gt"], bd["pair_gt"]
        else:
            S_clus, agt, ogt, pgt, ds = clus_scores(config_path, ckpt, dataset, split)
        if use_clus2:
            S_2nd, a2, o2, p2, _ = clus_scores(clus2_config, clus2_ckpt, dataset, split)
            assert torch.equal(agt, a2) and torch.equal(ogt, o2) and torch.equal(pgt, p2), \
                f"GT mismatch clus2 {dataset}/{split}"
        else:
            dump = torch.load(dump_path(enc_tag, dataset, split), map_location="cpu")
            S_2nd = dump["logits"].float()
            assert torch.equal(agt, dump["attr_gt"]) and torch.equal(ogt, dump["obj_gt"]) \
                and torch.equal(pgt, dump["pair_gt"]), f"GT mismatch {enc_tag}/{dataset}/{split}"
        ev = Evaluator(ds, model=None)
        closed = ev.closed_mask.bool()
        splits[split] = dict(S_clus=S_clus, S_2nd=S_2nd, agt=agt, ogt=ogt, pgt=pgt, ds=ds, ev=ev,
                             closed=closed, Zc=zscore_closed(S_clus, closed), Zr=zscore_closed(S_2nd, closed))

    def hm_auc(sp, S):
        st = test_eval(sp["ds"], sp["ev"], S.clone(), sp["agt"], sp["ogt"], sp["pgt"], config=None)
        return dict(hm=float(st["best_hm"]), auc=float(st["AUC"]),
                    seen=float(st["best_seen"]), unseen=float(st["best_unseen"]))

    ALPHAS = [round(0.05 * k, 2) for k in range(21)]
    v, t = splits["val"], splits["test"]
    best_alpha, best_val_hm, val_curve = None, -1, {}
    for a in ALPHAS:
        r = hm_auc(v, a * v["Zc"] + (1 - a) * v["Zr"])
        val_curve[f"{a:.2f}"] = r["hm"]
        if r["hm"] > best_val_hm:
            best_val_hm, best_alpha = r["hm"], a

    clus_test = hm_auc(t, t["S_clus"]); second_test = hm_auc(t, t["S_2nd"])
    ens_test = hm_auc(t, best_alpha * t["Zc"] + (1 - best_alpha) * t["Zr"])
    best_oracle, best_oa = -1, None
    for a in ALPHAS:
        r = hm_auc(t, a * t["Zc"] + (1 - a) * t["Zr"])
        if r["hm"] > best_oracle:
            best_oracle, best_oa = r["hm"], a

    # decorrelation on unseen test imgs
    pairs_t = t["ev"].pairs; seen = t["ev"].seen_mask.bool(); closed = t["closed"]
    def top1(S):
        Sm = S.clone(); Sm[:, ~closed] = -1e10; return Sm.argmax(1)
    tc, tr = top1(t["S_clus"]), top1(t["S_2nd"])
    pa_t, po_t = pairs_t[:, 0], pairs_t[:, 1]
    cc = (pa_t[tc] == t["agt"]) & (po_t[tc] == t["ogt"])
    rc = (pa_t[tr] == t["agt"]) & (po_t[tr] == t["ogt"])
    P = t["S_clus"].shape[1]
    pmap = {(int(pairs_t[i, 0]), int(pairs_t[i, 1])): i for i in range(P)}
    gt_idx = torch.tensor([pmap[(int(t["agt"][i]), int(t["ogt"][i]))] for i in range(len(t["agt"]))])
    seen_pair = torch.zeros(P, dtype=torch.bool); seen_pair[seen] = True
    img_unseen = ~seen_pair[gt_idx]
    cw = img_unseen & ~cc
    decorr = float(rc[cw].float().mean()) if cw.any() else 0.0

    res = dict(tag=tag, encoder_tag=enc_tag, dataset=dataset,
               clus_test=clus_test, second_test=second_test, ensemble_test=ens_test,
               val_tuned_alpha=best_alpha, val_best_hm=best_val_hm,
               oracle_test_hm=best_oracle, oracle_alpha=best_oa,
               delta_hm=ens_test["hm"] - clus_test["hm"], delta_auc=ens_test["auc"] - clus_test["auc"],
               decorr=decorr, val_curve=val_curve)
    print(f"[{tag}] ClusPro {clus_test['hm']:.4f}/{clus_test['auc']:.4f}  2nd {second_test['hm']:.4f}/{second_test['auc']:.4f}  "
          f"a*={best_alpha}  ENS {ens_test['hm']:.4f}/{ens_test['auc']:.4f}  dHM {res['delta_hm']:+.4f} dAUC {res['delta_auc']:+.4f}  "
          f"decorr {decorr*100:.1f}%  (oracle {best_oracle:.4f}@{best_oa})")
    out = os.path.join(SCRATCH, "ensemble_2nd_results.json")
    allr = json.load(open(out)) if os.path.exists(out) else {}
    allr[tag] = res
    json.dump(allr, open(out, "w"), indent=2)
    print(f"[{tag}] saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["encode", "score"])
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--enc_tag", default="clus2")
    ap.add_argument("--dataset", default="mit-states", choices=list(DATA_DIR))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--clus2_config", default=None)
    ap.add_argument("--clus2_ckpt", default=None)
    ap.add_argument("--base_enc_tag", default=None)
    args = ap.parse_args()
    if args.stage == "encode":
        stage_encode(args.encoder, args.enc_tag, args.dataset, args.split)
    else:
        stage_score(args.enc_tag, args.dataset, args.config, args.ckpt, args.tag,
                    args.clus2_config, args.clus2_ckpt, args.base_enc_tag)


if __name__ == "__main__":
    main()
