"""
Recap-CLIP ensemble — VAL-TUNED-alpha protocol (honest: alpha picked on VAL, frozen, applied to TEST).

Two stages (different conda envs):

STAGE 1 (--stage recap, run in an env WITH open_clip, e.g. mosaic3d):
  Load UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP (offline cache). Encode a dataset SPLIT's images with
  Recap's OWN preprocessing in CompositionDataset.data (shuffle=False) order; build text embeddings
  "a photo of {attr} {obj}" over dset.pairs; save cosine logits [N,P] + GT to a per-(dataset,split) dump.
    --dataset {mit-states|ut-zappos}  --split {val|test}

STAGE 2 (--stage score, run in the CZSL env):
  For a given ClusPro (config,ckpt) on a dataset:
    - compute ClusPro closed-world score S_clus on val AND test
    - load Recap dumps for val AND test
    - z-score each model's closed scores per image; sweep alpha on VAL closed-world HM (Evaluator);
      pick alpha* = argmax_val; FREEZE; report TEST HM/AUC at alpha*.
    - also report ClusPro-alone test, Recap-alone test, error-decorrelation.
    --dataset {mit-states|ut-zappos} --config <cfg> --ckpt <ckpt> --tag <tag>

Usage:
  MOS=/data1/workspaces/jgshin22/miniconda3/envs/mosaic3d/bin/python
  CZSL=/data1/workspaces/jgshin22/miniconda3/envs/CZSL/bin/python
  CUDA_VISIBLE_DEVICES=6 $MOS  tools/harness/probe_recap_clip.py --stage recap --dataset mit-states --split val
  CUDA_VISIBLE_DEVICES=6 $CZSL tools/harness/probe_recap_clip.py --stage score --dataset mit-states \
       --config config/..._nonorm_full_seed0.yml --ckpt checkpoint/..._nonorm_full_seed0/val_best.pt --tag mit_s0
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"
RECAP_ID = "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP"
DATA_DIR = {"mit-states": "data/mit-states", "ut-zappos": "data/ut-zappos"}


def dump_path(dataset, split):
    return os.path.join(SCRATCH, f"recap_{dataset}_{split}.pt")


# ---------------------------------------------------------------- STAGE 1: Recap encode
@torch.no_grad()
def stage_recap(dataset, split):
    import open_clip
    from dataset import CompositionDataset
    from PIL import Image

    dpath = os.path.join(REPO, DATA_DIR[dataset])
    ds = CompositionDataset(dpath, phase=split,
                            split="compositional-split-natural", open_world=False)
    a2i, o2i = ds.attr2idx, ds.obj2idx

    print(f"[recap {dataset}/{split}] loading {RECAP_ID} ...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(RECAP_ID)
    tokenizer = open_clip.get_tokenizer(RECAP_ID)
    model = model.cuda().eval()
    logit_scale = model.logit_scale.exp().item()

    prompts = [f"a photo of {a.replace('.',' ').lower()} {o.replace('.',' ').lower()}"
               for (a, o) in ds.pairs]
    P = len(prompts)
    txt = []
    for s in range(0, P, 256):
        toks = tokenizer(prompts[s:s + 256]).cuda()
        txt.append(F.normalize(model.encode_text(toks).float(), dim=-1).cpu())
    text_feat = torch.cat(txt, 0)
    print(f"[recap {dataset}/{split}] text {tuple(text_feat.shape)}", flush=True)

    img_root = ds.root + "/images/"
    N = len(ds.data)
    all_attr_gt, all_obj_gt, all_pair_gt, img_feats = [], [], [], []
    buf = []
    t0 = time.time()

    def flush():
        if buf:
            x = torch.stack(buf, 0).cuda()
            img_feats.append(F.normalize(model.encode_image(x).float(), dim=-1).cpu())

    for i, (image, attr, obj) in enumerate(ds.data):
        fp = img_root + image
        if not os.path.exists(fp):
            parts = image.split("/")
            if len(parts) >= 2:
                parts[0] = parts[0].replace("_", " ")
                fp = img_root + "/".join(parts)
        buf.append(preprocess(Image.open(fp).convert("RGB")))
        all_attr_gt.append(a2i[attr]); all_obj_gt.append(o2i[obj])
        all_pair_gt.append(ds.pair2idx[(attr, obj)])
        if len(buf) >= 64:
            flush(); buf = []
        if (i + 1) % 3000 == 0:
            print(f"[recap {dataset}/{split}] img {i+1}/{N} {time.time()-t0:.0f}s", flush=True)
    flush()
    image_feat = torch.cat(img_feats, 0)
    logits = logit_scale * (image_feat @ text_feat.t())
    out = dict(logits=logits,
               attr_gt=torch.tensor(all_attr_gt),
               obj_gt=torch.tensor(all_obj_gt),
               pair_gt=torch.tensor(all_pair_gt),
               pairs=[(a2i[a], o2i[o]) for (a, o) in ds.pairs],
               recap_id=RECAP_ID, logit_scale=logit_scale)
    os.makedirs(SCRATCH, exist_ok=True)
    torch.save(out, dump_path(dataset, split))
    print(f"[recap {dataset}/{split}] saved {dump_path(dataset, split)}  imgs={image_feat.shape[0]} P={P}", flush=True)


# ---------------------------------------------------------------- STAGE 2: score
@torch.no_grad()
def clus_scores(config_path, ckpt, dataset, split):
    """ClusPro closed-world baseline score S[N,P] in CompositionDataset(split,shuffle=False) order."""
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
    model = get_model(config, attributes=attributes, classes=classes, offset=len(attributes)).cuda()
    model.load_state_dict(torch.load(os.path.join(REPO, ckpt), map_location="cuda"))
    model.eval()
    a2i, o2i = ds.attr2idx, ds.obj2idx
    pairs = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.pairs]).cuda()
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    comps, attrs, objs, agt, ogt, pgt = [], [], [], [], [], []
    for batch in loader:
        c, a, o = model.val_forward(batch, pairs)
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


def stage_score(dataset, config_path, ckpt, tag):
    from test import Evaluator, test as test_eval

    results = {"tag": tag, "dataset": dataset}
    blends = {}
    splits = {}
    for split in ["val", "test"]:
        S_clus, agt, ogt, pgt, ds = clus_scores(config_path, ckpt, dataset, split)
        dump = torch.load(dump_path(dataset, split), map_location="cpu")
        S_recap = dump["logits"].float()
        # align by GT order (both are CompositionDataset(split,shuffle=False).data order)
        assert torch.equal(agt, dump["attr_gt"]) and torch.equal(ogt, dump["obj_gt"]) \
            and torch.equal(pgt, dump["pair_gt"]), f"GT order mismatch {dataset}/{split}"
        ev = Evaluator(ds, model=None)
        closed = ev.closed_mask.bool()
        splits[split] = dict(S_clus=S_clus, S_recap=S_recap, agt=agt, ogt=ogt, pgt=pgt,
                             ds=ds, ev=ev, closed=closed,
                             Zc=zscore_closed(S_clus, closed), Zr=zscore_closed(S_recap, closed))

    def hm_auc(ev, ds, S, agt, ogt, pgt):
        st = test_eval(ds, ev, S.clone(), agt, ogt, pgt, config=None)
        return dict(hm=float(st["best_hm"]), auc=float(st["AUC"]),
                    seen=float(st["best_seen"]), unseen=float(st["best_unseen"]))

    ALPHAS = [round(0.05 * k, 2) for k in range(21)]   # 0.00..1.00 step 0.05
    # --- tune alpha on VAL ---
    v = splits["val"]
    val_curve = {}
    best_alpha, best_val_hm = None, -1
    for a in ALPHAS:
        Sb = a * v["Zc"] + (1 - a) * v["Zr"]
        r = hm_auc(v["ev"], v["ds"], Sb, v["agt"], v["ogt"], v["pgt"])
        val_curve[f"{a:.2f}"] = r["hm"]
        if r["hm"] > best_val_hm:
            best_val_hm, best_alpha = r["hm"], a

    # --- standalone references ---
    t = splits["test"]
    clus_test = hm_auc(t["ev"], t["ds"], t["S_clus"], t["agt"], t["ogt"], t["pgt"])
    recap_test = hm_auc(t["ev"], t["ds"], t["S_recap"], t["agt"], t["ogt"], t["pgt"])
    clus_val = hm_auc(v["ev"], v["ds"], v["S_clus"], v["agt"], v["ogt"], v["pgt"])

    # --- FROZEN alpha* applied to TEST ---
    Sb_test = best_alpha * t["Zc"] + (1 - best_alpha) * t["Zr"]
    ens_test = hm_auc(t["ev"], t["ds"], Sb_test, t["agt"], t["ogt"], t["pgt"])

    # oracle (alpha tuned on TEST) for honesty gap reporting only
    best_oracle, best_oa = -1, None
    for a in ALPHAS:
        r = hm_auc(t["ev"], t["ds"], a * t["Zc"] + (1 - a) * t["Zr"], t["agt"], t["ogt"], t["pgt"])
        if r["hm"] > best_oracle:
            best_oracle, best_oa = r["hm"], a

    # --- error decorrelation on UNSEEN test imgs ---
    pairs_t = t["ev"].pairs; seen = t["ev"].seen_mask.bool(); closed = t["closed"]
    def top1(S):
        Sm = S.clone(); Sm[:, ~closed] = -1e10
        return Sm.argmax(1)
    tc, tr = top1(t["S_clus"]), top1(t["S_recap"])
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

    results.update(dict(
        clus_val=clus_val, clus_test=clus_test, recap_test=recap_test,
        val_tuned_alpha=best_alpha, val_best_hm=best_val_hm,
        ensemble_test=ens_test,
        oracle_test_hm=best_oracle, oracle_alpha=best_oa,
        delta_hm=ens_test["hm"] - clus_test["hm"], delta_auc=ens_test["auc"] - clus_test["auc"],
        decorr_recap_saves=decorr, val_curve=val_curve))

    print(f"\n[{tag}] ClusPro-alone test HM={clus_test['hm']:.4f} AUC={clus_test['auc']:.4f}")
    print(f"[{tag}] Recap-alone   test HM={recap_test['hm']:.4f} AUC={recap_test['auc']:.4f}")
    print(f"[{tag}] val-tuned alpha*={best_alpha} (val HM {best_val_hm:.4f})")
    print(f"[{tag}] VAL-TUNED ENSEMBLE test HM={ens_test['hm']:.4f} AUC={ens_test['auc']:.4f}  "
          f"ΔHM={results['delta_hm']:+.4f} ΔAUC={results['delta_auc']:+.4f}")
    print(f"[{tag}] (oracle test-tuned alpha={best_oa} -> HM={best_oracle:.4f}; honesty gap "
          f"{best_oracle-ens_test['hm']:+.4f})")
    print(f"[{tag}] decorrelation (Recap saves ClusPro-wrong-unseen) = {decorr*100:.1f}%")

    out = os.path.join(SCRATCH, "recap_valtuned_results.json")
    allr = json.load(open(out)) if os.path.exists(out) else {}
    allr[tag] = results
    json.dump(allr, open(out, "w"), indent=2)
    print(f"[{tag}] saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["recap", "score"])
    ap.add_argument("--dataset", default="mit-states", choices=list(DATA_DIR))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.stage == "recap":
        stage_recap(args.dataset, args.split)
    else:
        stage_score(args.dataset, args.config, args.ckpt, args.tag)


if __name__ == "__main__":
    main()
