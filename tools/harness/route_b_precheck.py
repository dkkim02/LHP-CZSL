"""
ROUTE B pre-check (DECISIVE) — does the teacher's off-target softmax mass concentrate on
UNSEEN primitive-neighbors (real compositional structure) or spread ~uniformly (sealed LS)?

Teacher = frozen DataComp ViT-L/14 (and OpenAI ViT-L/14 control). We have its TRAIN-split
closed-world logits [N_train, P] over dset.pairs (P=1962). For each TRAIN image (a SEEN pair):
  softmax teacher logits over the CLOSED-WORLD pairs.
  OFF-TARGET mass = 1 - softmax[true_pair].
  Devil's TARGET metric: of the off-target mass, the FRACTION landing on UNSEEN pairs (not in
    train_pairs) that share the true image's attr OR obj, divided by the UNIFORM FLOOR
    = (#unseen-pairs-sharing-true-attr-or-obj) / (P - 1).
  Ratio >> 1 => off-target mass is structured toward unseen compositional neighbors.

Report ratio for DataComp AND OpenAI (control). DataComp ratio >= 3x floor and clearly above
OpenAI => corpus-structured-on-unseen real => BUILD KD. < 2x => sealed LS => STOP.

Usage: CUDA_VISIBLE_DEVICES="" python tools/harness/route_b_precheck.py   (CPU ok)
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from parameters import parser
from dataset import CompositionDataset
from utils import load_args

SCRATCH = "/home/jgshin22/.claude/jobs/664681e9/tmp"


def analyze(enc_tag, ds, pairs_idx, seen_pair_set, train_pair_set):
    """pairs_idx: [P,2] int. Returns dict of stats."""
    dump = torch.load(os.path.join(SCRATCH, f"enc_{enc_tag}_mit-states_train.pt"), map_location="cpu")
    logits = dump["logits"].float()                # [N,P]
    agt, ogt, pgt = dump["attr_gt"], dump["obj_gt"], dump["pair_gt"]
    N, P = logits.shape
    pa = pairs_idx[:, 0]; po = pairs_idx[:, 1]

    # closed-world softmax (train phase closed set = train_pairs only, but for KD the teacher
    # ranks over the full pair universe; the devil's metric uses the full closed-world P pairs).
    probs = F.softmax(logits, dim=-1)              # [N,P]

    # map each image's GT (attr,obj) -> its pair index in pairs_idx
    pmap = {(int(pa[i]), int(po[i])): i for i in range(P)}
    # precompute pair properties
    is_train = torch.tensor([(int(pa[i]), int(po[i])) in train_pair_set for i in range(P)])
    is_unseen = ~is_train                          # pairs NOT in train_pairs

    off_target_total = []
    frac_on_unseen_nbr = []
    floor_list = []
    ratio_list = []
    # also: seen primitive-neighbors (control) and raw off-target uniform-ratio
    frac_on_seen_nbr = []

    # cap to a sample for speed
    rng = np.random.default_rng(0)
    idxs = rng.permutation(N)[:min(N, 6000)]
    for n in idxs:
        a, o = int(agt[n]), int(ogt[n])
        true_pi = pmap.get((a, o))
        if true_pi is None:
            continue
        p = probs[n]                               # [P]
        off = 1.0 - float(p[true_pi])
        if off <= 1e-6:
            continue
        # pairs sharing the true attr OR obj (excluding the true pair)
        share = ((pa == a) | (po == o))
        share = share.clone(); share[true_pi] = False
        # unseen neighbors
        unseen_nbr = share & is_unseen
        seen_nbr = share & is_train
        mass_unseen_nbr = float(p[unseen_nbr].sum())
        mass_seen_nbr = float(p[seen_nbr].sum())
        n_unseen_nbr = int(unseen_nbr.sum())
        # devil's metric: fraction of OFF-TARGET mass on unseen-neighbors / uniform floor
        frac = mass_unseen_nbr / off
        floor = n_unseen_nbr / (P - 1)             # uniform expectation
        off_target_total.append(off)
        frac_on_unseen_nbr.append(frac)
        floor_list.append(floor)
        if floor > 0:
            ratio_list.append(frac / floor)
        frac_on_seen_nbr.append(mass_seen_nbr / off)

    return dict(
        n=len(off_target_total),
        mean_off_target_total=float(np.mean(off_target_total)),
        mean_frac_on_unseen_nbr=float(np.mean(frac_on_unseen_nbr)),
        mean_floor=float(np.mean(floor_list)),
        mean_ratio=float(np.mean(ratio_list)),
        median_ratio=float(np.median(ratio_list)),
        mean_frac_on_seen_nbr=float(np.mean(frac_on_seen_nbr)),
    )


def main():
    config = parser.parse_args([])
    load_args(os.path.join(REPO, "config/cluspro_baseline_mit_l14_v2_nonorm_full_seed0.yml"), config)
    config.dataset_path = os.path.join(REPO, "data/mit-states"); config.open_world = False
    ds = CompositionDataset(config.dataset_path, phase="train",
                            split="compositional-split-natural", open_world=False)
    a2i, o2i = ds.attr2idx, ds.obj2idx
    pairs_idx = torch.tensor([(a2i[a], o2i[o]) for a, o in ds.pairs])
    train_pair_set = {(a2i[a], o2i[o]) for (a, o) in ds.train_pairs}
    seen_pair_set = train_pair_set

    res = {}
    for enc_tag in ["dcL14", "openaiL14"]:
        p = os.path.join(SCRATCH, f"enc_{enc_tag}_mit-states_train.pt")
        if not os.path.exists(p):
            print(f"[routeB] MISSING dump {p}"); continue
        res[enc_tag] = analyze(enc_tag, ds, pairs_idx, seen_pair_set, train_pair_set)

    print("\n" + "=" * 90)
    print("  ROUTE B PRE-CHECK — teacher off-target mass on UNSEEN primitive-neighbors (MIT train)")
    print("=" * 90)
    for tag, name in [("dcL14", "DataComp ViT-L/14"), ("openaiL14", "OpenAI ViT-L/14 (control)")]:
        if tag not in res:
            continue
        r = res[tag]
        print(f"\n  [{name}]  (n={r['n']} seen-train imgs)")
        print(f"    mean total off-target mass (1 - p[true])     = {r['mean_off_target_total']:.4f}")
        print(f"    mean frac of off-target on UNSEEN-prim-nbrs   = {r['mean_frac_on_unseen_nbr']:.4f}")
        print(f"    uniform floor (mean)                          = {r['mean_floor']:.4f}")
        print(f"    >>> RATIO (frac / floor) mean={r['mean_ratio']:.2f}x  median={r['median_ratio']:.2f}x <<<")
        print(f"    (frac of off-target on SEEN-prim-nbrs control = {r['mean_frac_on_seen_nbr']:.4f})")
    print("\n" + "=" * 90)
    if "dcL14" in res:
        dc = res["dcL14"]["mean_ratio"]
        oa = res.get("openaiL14", {}).get("mean_ratio", float("nan"))
        print(f"  DataComp ratio = {dc:.2f}x   |   OpenAI control ratio = {oa:.2f}x")
        if dc >= 3.0:
            verdict = (f"GO — DataComp off-target mass on unseen primitive-neighbors is {dc:.1f}x the "
                       f"uniform floor (>=3x): real compositional structure on UNSEEN pairs -> BUILD KD + probe.")
        elif dc < 2.0:
            verdict = (f"STOP — DataComp ratio {dc:.2f}x < 2x: off-target mass ~uniform -> KD degenerates "
                       f"to sealed label-smoothing -> do NOT build.")
        else:
            verdict = (f"MARGINAL — DataComp ratio {dc:.2f}x in [2,3): weak structure; compare to OpenAI "
                       f"control ({oa:.2f}x) and decide.")
        print(f"  VERDICT: {verdict}")
        if not np.isnan(oa):
            print(f"  corpus-distinction: DataComp/OpenAI ratio-of-ratios = {dc/oa:.2f}x "
                  f"({'DataComp structures on unseen MORE than OpenAI' if dc>oa else 'no corpus distinction'})")
    print("=" * 90)
    os.makedirs(SCRATCH, exist_ok=True)
    json.dump(res, open(os.path.join(SCRATCH, "route_b_precheck.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
