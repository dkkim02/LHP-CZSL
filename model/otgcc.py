"""Optimal Transport based cluster assignment for prototype updates."""

import torch
import torch.nn.functional as F


def local_assign(features, similarity_scores, top_percent=1.0):
    """Softmax-based soft assignment (legacy, pre-§17-10).

    Args:
        features:          [B, D] — batch features (detached); kept for parity.
        similarity_scores: [B, K] — similarity to K cluster centers.
        top_percent:       float — fraction of samples to retain (1.0 = all).

    Returns:
        couplings:     [B, K] — temperature-softmaxed soft assignment.
        selected_mask: [B] boolean — high-confidence samples.
    """
    B, K = similarity_scores.shape

    max_sim = similarity_scores.max(dim=-1)[0]
    num_select = max(1, int(B * top_percent))
    _, top_indices = torch.topk(max_sim, min(num_select, B))
    selected_mask = torch.zeros(B, dtype=torch.bool, device=features.device)
    selected_mask[top_indices] = True

    tau = 0.5
    couplings = F.softmax(similarity_scores / tau, dim=-1)
    return couplings, selected_mask


def local_assign_ot(features, similarity_scores, sinkhorn_iters=3, epsilon=0.05, top_percent=1.0):
    """Sinkhorn-balanced batch-level coupling (§17-10 H3).

    Mirrors official ClusPro `distributed_sinkhorn` (root cluspro_baseline.py:163-187):
    enforces approximate double-stochasticity so the B samples are spread across the K
    clusters, preventing prototype collapse onto one cluster while keeping the soft
    coupling for downstream gumbel-softmax hard assignment.

    Caller is expected to apply gumbel_softmax on the returned `couplings`
    (mirrors official train_forward at root cluspro_baseline.py:709) — this function
    intentionally does NOT gumbel itself so the legacy and OT paths share the same
    interface contract.

    Args:
        features:          [B, D] — batch features (detached); kept for parity with local_assign.
        similarity_scores: [B, K] — cosine/dot sim of every batch sample to K cluster centers.
        sinkhorn_iters:    int — row/col normalization iterations.
        epsilon:           float — entropic regularization; smaller = sharper, risk of collapse.
        top_percent:       float — fraction of high-confidence samples retained (1.0 = all).
    """
    B, K = similarity_scores.shape

    max_sim = similarity_scores.max(dim=-1)[0]
    num_select = max(1, int(B * top_percent))
    _, top_indices = torch.topk(max_sim, min(num_select, B))
    selected_mask = torch.zeros(B, dtype=torch.bool, device=features.device)
    selected_mask[top_indices] = True

    L = torch.exp(similarity_scores.float() / epsilon).t()  # K x B
    L = L / (L.sum() + 1e-12)
    for _ in range(sinkhorn_iters):
        L = L / (L.sum(dim=1, keepdim=True) + 1e-12) / K
        L = L / (L.sum(dim=0, keepdim=True) + 1e-12) / B
    L = L * B
    couplings = L.t()
    return couplings, selected_mask
