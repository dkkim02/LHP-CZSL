"""Decisive Tier-0 probe for CFD (#1) and TT-Consist (#2).

Dumps test-set disentangled features ONCE, then runs the falsification tests:

CFD kill criterion (advisor-sharpened): the model's logit_infer ALREADY fuses
text-based attr_pred*obj_pred into comp_logits. CFD adds a SECOND, *visual*
primitive head (f_attr_proj/f_obj_proj scored vs momentum queues). CFD only
survives if this visual head is NOT strictly dominated by the text head it
would compete with -- i.e. higher primitive accuracy OR decorrelated errors.

TT-Consist probe: k-NN smoothing of the visual primitive evidence in the
disentangled subspaces -- does the smoothed primitive prediction beat the raw
one (necessary condition for the transductive denoising claim)?
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from dataset import CompositionDataset
from model.model_factory import get_model
from utils import load_args
from torch.utils.data import DataLoader

CKPT = "checkpoint/cluspro_baseline_l14_mit_v2_ot_seed0/val_best.pt"
YML  = "config/cluspro_baseline_mit_l14_v2_ot_seed0.yml"
DATA = "/home/jgshin22/work/LHP-CZSL/data/mit-states"
DUMP = "/home/jgshin22/work/LHP-CZSL/cache/cfd_probe_dump.pt"


def build_dump():
    config = SimpleNamespace()
    load_args(YML, config)
    config.dataset_path = DATA
    config.load_model = CKPT

    val_ds = CompositionDataset(DATA, phase='val',
                                split='compositional-split-natural', open_world=False)
    allattrs = val_ds.attrs; allobj = val_ds.objs
    classes = [c.replace(".", " ").lower() for c in allobj]
    attributes = [a.replace(".", " ").lower() for a in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(CKPT, map_location='cuda'))
    model.eval()

    test_ds = CompositionDataset(DATA, phase='test',
                                 split='compositional-split-natural', open_world=False)
    pairs = torch.tensor([(test_ds.attr2idx[a], test_ds.obj2idx[o])
                          for a, o in test_ds.pairs]).cuda()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8)

    fa, fo, cl, al, ol, agt, ogt, pgt = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for data in loader:
            img = data[0].cuda()
            f_global, _ = model.encode_image(img.type(model.clip.dtype))
            f_attr = model.attr_disentangler(f_global)
            f_obj = model.obj_disentangler(f_global)
            f_attr_proj = model.attr_proj(f_attr)
            f_obj_proj = model.obj_proj(f_obj)
            comp, attr_lg, obj_lg = model.val_forward(data, pairs)
            fa.append(f_attr_proj.float().cpu()); fo.append(f_obj_proj.float().cpu())
            cl.append(comp.float().cpu()); al.append(attr_lg.float().cpu()); ol.append(obj_lg.float().cpu())
            agt.append(data[1]); ogt.append(data[2]); pgt.append(data[3])

    dump = dict(
        f_attr_proj=torch.cat(fa), f_obj_proj=torch.cat(fo),
        comp_logits=torch.cat(cl), attr_logits=torch.cat(al), obj_logits=torch.cat(ol),
        attr_gt=torch.cat(agt), obj_gt=torch.cat(ogt), pair_gt=torch.cat(pgt),
        pairs=pairs.cpu(),
        attrs=attributes, objs=classes,
        train_pairs=[(test_ds.attr2idx[a], test_ds.obj2idx[o]) for a, o in test_ds.train_pairs],
    )
    # attr_proj-projected momentum queues (CFD scores against these)
    sd = torch.load(CKPT, map_location='cpu')
    n_attr = len(attributes); n_obj = len(classes); K = sd['attr_queue0'].shape[0]
    aq = torch.stack([sd[f'attr_queue{k}'].float() for k in range(n_attr)])  # (A,K,D)
    oq = torch.stack([sd[f'obj_queue{k}'].float() for k in range(n_obj)])    # (O,K,D)
    with torch.no_grad():
        aq_proj = model.attr_proj(aq.reshape(-1, aq.shape[-1]).cuda()).reshape(n_attr, K, -1).float().cpu()
        oq_proj = model.obj_proj(oq.reshape(-1, oq.shape[-1]).cuda()).reshape(n_obj, K, -1).float().cpu()
    dump['attr_queue_proj'] = aq_proj
    dump['obj_queue_proj'] = oq_proj
    dump['attr_queue_raw'] = aq
    dump['obj_queue_raw'] = oq
    torch.save(dump, DUMP)
    print(f"[dump] saved {DUMP}  N_test={dump['f_attr_proj'].shape[0]}")
    return dump


def lse_density(feat, queue):
    """feat (N,D), queue (C,K,D) -> (N,C) log-sum-exp soft cosine density."""
    fn = F.normalize(feat, dim=-1)
    qn = F.normalize(queue, dim=-1)
    sim = torch.einsum('nd,ckd->nck', fn, qn)  # (N,C,K)
    return torch.logsumexp(sim / 0.05, dim=-1)  # (N,C)


def run_probes(dump):
    fa, fo = dump['f_attr_proj'], dump['f_obj_proj']
    attr_gt, obj_gt = dump['attr_gt'], dump['obj_gt']
    al, ol = dump['attr_logits'], dump['obj_logits']

    # ---- TEXT head accuracy (what logit_infer ALREADY uses) ----
    text_attr_pred = al.argmax(1)
    text_obj_pred = ol.argmax(1)
    text_attr_acc = (text_attr_pred == attr_gt).float().mean().item()
    text_obj_acc = (text_obj_pred == obj_gt).float().mean().item()

    # ---- CFD VISUAL primitive head (LSE density vs projected queues) ----
    rho_a = lse_density(fa, dump['attr_queue_proj'])  # (N, A)
    rho_o = lse_density(fo, dump['obj_queue_proj'])   # (N, O)
    vis_attr_pred = rho_a.argmax(1)
    vis_obj_pred = rho_o.argmax(1)
    vis_attr_acc = (vis_attr_pred == attr_gt).float().mean().item()
    vis_obj_acc = (vis_obj_pred == obj_gt).float().mean().item()

    # ---- error decorrelation: fraction of TEXT errors that VISUAL gets right ----
    text_attr_err = (text_attr_pred != attr_gt)
    vis_rescues_attr = ((text_attr_err) & (vis_attr_pred == attr_gt)).sum().item() / max(1, text_attr_err.sum().item())
    text_obj_err = (text_obj_pred != obj_gt)
    vis_rescues_obj = ((text_obj_err) & (vis_obj_pred == obj_gt)).sum().item() / max(1, text_obj_err.sum().item())

    print("\n==== CFD head-to-head (CFD survives only if visual head not dominated) ====")
    print(f"ATTR  text-head acc={text_attr_acc:.4f}  visual-head acc={vis_attr_acc:.4f}  "
          f"visual rescues {vis_rescues_attr:.1%} of text errors")
    print(f"OBJ   text-head acc={text_obj_acc:.4f}  visual-head acc={vis_obj_acc:.4f}  "
          f"visual rescues {vis_rescues_obj:.1%} of text errors")

    # ---- TT-Consist necessary condition: kNN smoothing in disentangled space ----
    # smooth the VISUAL primitive evidence (rho) toward kNN-mean in feature space.
    def knn_smooth_acc(feat, rho, gt, k=10, alpha=0.5):
        fn = F.normalize(feat, dim=-1)
        S = fn @ fn.t()
        S.fill_diagonal_(-1e9)
        nn_idx = S.topk(k, dim=1).indices  # (N,k)
        rho_nn = rho[nn_idx].mean(1)       # (N,C)
        rho_sm = (1 - alpha) * rho + alpha * rho_nn
        return (rho_sm.argmax(1) == gt).float().mean().item()

    tt_attr = knn_smooth_acc(fa, rho_a, attr_gt)
    tt_obj = knn_smooth_acc(fo, rho_o, obj_gt)
    print("\n==== TT-Consist necessary condition (kNN smoothing helps primitive acc?) ====")
    print(f"ATTR  raw-visual={vis_attr_acc:.4f}  knn-smoothed={tt_attr:.4f}  delta={tt_attr-vis_attr_acc:+.4f}")
    print(f"OBJ   raw-visual={vis_obj_acc:.4f}  knn-smoothed={tt_obj:.4f}  delta={tt_obj-vis_obj_acc:+.4f}")

    return dict(text_attr_acc=text_attr_acc, text_obj_acc=text_obj_acc,
                vis_attr_acc=vis_attr_acc, vis_obj_acc=vis_obj_acc,
                vis_rescues_attr=vis_rescues_attr, vis_rescues_obj=vis_rescues_obj,
                tt_attr=tt_attr, tt_obj=tt_obj)


if __name__ == "__main__":
    if os.path.exists(DUMP) and "--rebuild" not in sys.argv:
        print(f"[dump] loading cached {DUMP}")
        dump = torch.load(DUMP, map_location='cpu')
    else:
        dump = build_dump()
    run_probes(dump)
