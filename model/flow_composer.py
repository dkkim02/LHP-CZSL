"""FlowComposer: Flow Matching + LLM-augmented distribution module on top of ClusProBaseline.

Implements modules A (attr/obj flow nets v_θa, v_θo), B (Composer MLP), C (leakage augmentation),
and D (LLM text-embedding distribution bank). Plugs into the repo's existing train/test contract:

    model(batch, idx) → predict
    model.loss_calu(predict, batch) → scalar
    model.logit_infer(predict, pairs) → [B, n_pairs] scores

Spec: coding_agent_instructions.md.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cluspro_baseline import ClusProBaseline
from model.llm_distribution import LLMDistributionBank


def _sinusoidal_t_emb(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """t: [B] in [0,1] → [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, t_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mod = nn.Linear(t_dim, 3 * dim)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, t_emb):
        shift, scale, gate = self.mod(t_emb).chunk(3, dim=-1)
        h = self.ln(x) * (1 + scale) + shift
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + gate * h


class FlowNet(nn.Module):
    """v_θ(x_t, t) — lightweight residual MLP."""

    def __init__(self, dim: int, hidden_dim: int = 1024, num_blocks: int = 6, t_emb_dim: int = 128):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.t_emb_dim = t_emb_dim
        self.in_proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([ResBlock(dim, hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = _sinusoidal_t_emb(t, self.t_emb_dim).to(x.dtype)
        te = self.t_mlp(te)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, te)
        return self.out_proj(h)


class Composer(nn.Module):
    """(v_a_norm, v_o_norm) → (a, b) coefficients."""

    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, va: torch.Tensor, vo: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([va, vo], dim=-1))


def solve_least_squares(va: torch.Tensor, vo: torch.Tensor, vc: torch.Tensor):
    """Batched 2x2 closed-form for argmin_{a,b} || a*va + b*vo - vc ||."""
    aa = (va * va).sum(-1)
    ao = (va * vo).sum(-1)
    oo = (vo * vo).sum(-1)
    ac = (va * vc).sum(-1)
    oc = (vo * vc).sum(-1)
    det = aa * oo - ao * ao + 1e-6
    a_star = (oo * ac - ao * oc) / det
    b_star = (aa * oc - ao * ac) / det
    return a_star, b_star


class FlowComposer(ClusProBaseline):
    """ClusProBaseline + Flow Matching + LLM distribution module.

    Inherits all CLIP/adapter/disentangler/prototype machinery from ClusProBaseline.
    Adds:
      - LLMDistributionBank (attr / obj / composition text embedding banks, K=8)
      - FlowNet_a, FlowNet_o
      - Composer
    The baseline's `train_forward` outputs (comp_logits/attr_logits/obj_logits + aux losses)
    are preserved and re-used for the original CE losses (so `L_baseline` is exactly the
    ClusPro loss). Flow-matching, composer, and leakage losses are added on top.

    At inference, `logit_infer` blends:
      blend * baseline_logit  +  (1 - blend) * (p_c + p_a * p_o)
    where p_* are cosine similarities between Euler-integrated flow endpoints and the LLM bank.
    """

    def __init__(self, config, attributes, classes, offset):
        super().__init__(config, attributes, classes, offset)

        D = self.clip.visual.output_dim
        self.D = D

        # ---- Flow networks ----
        h = int(getattr(config, "flow_hidden_dim", 1024))
        nb = int(getattr(config, "flow_num_blocks", 6))
        self.flow_a = FlowNet(D, h, nb)
        self.flow_o = FlowNet(D, h, nb)
        self.composer = Composer(D, int(getattr(config, "composer_hidden_dim", 512)))

        # ---- LLM bank (lazily built; needs pair list) ----
        self.K = int(getattr(config, "llm_bank_K", 8))
        self._llm_cache_path = getattr(config, "llm_cache_path", None)
        if self._llm_cache_path is None:
            ds = getattr(config, "dataset", "dataset").replace("/", "_")
            self._llm_cache_path = f"data/llm_descriptions/{ds}/text_bank_K{self.K}_vitl14.pt"
        self._descriptions_root = getattr(config, "llm_descriptions_root", None)
        self._all_pairs = list(getattr(config, "_all_pairs", []))
        # Will be populated by `attach_pairs` (called from get_model) before training.
        self.bank: LLMDistributionBank | None = None

        # ---- Loss / inference weights ----
        self.lambda_flow = float(getattr(config, "lambda_flow", 1.0))
        self.lambda_comp = float(getattr(config, "lambda_comp", 1.0))
        self.lambda_leak = float(getattr(config, "lambda_leak", 0.5))
        self.tau_endpoint = float(getattr(config, "endpoint_tau", 0.07))
        self.flow_step_h = float(getattr(config, "flow_step_h", 1.0))
        self.flow_blend = float(getattr(config, "flow_blend", 0.5))  # baseline vs flow score
        self.inference_mode = str(getattr(config, "inference_mode", "mean"))  # 'mean' or 'max'
        self.use_leakage = bool(getattr(config, "use_leakage", True))

    # ------------------------------------------------------------------
    # Setup helper — must be called once after model construction.
    # ------------------------------------------------------------------
    def attach_pairs(self, all_pairs):
        """all_pairs: list[(attr_str, obj_str)] covering train+val+test (closed-world full pair set)."""
        self.bank = LLMDistributionBank(
            attributes=self.attributes,
            objects=self.classes,
            pairs=all_pairs,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            clip_dtype=self.clip.dtype,
            cache_path=self._llm_cache_path,
            K=self.K,
            device="cuda",
            context_length=self.config.context_length,
            descriptions_root=self._descriptions_root,
        )

    # ------------------------------------------------------------------
    # Feature extraction shared by train & val.
    # ------------------------------------------------------------------
    def _extract_features(self, batch_img):
        f_global, _ = self.encode_image(batch_img.type(self.clip.dtype))
        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)
        f_attr_proj = self.attr_proj(f_attr)
        f_obj_proj = self.obj_proj(f_obj)
        return f_global.float(), f_attr_proj.float(), f_obj_proj.float()

    def _fm_loss_branch(self, x0, x1, flow_net, label_idx, vocab_mean_n):
        """Single branch FM loss = MSE on velocity + CE on predicted endpoint vs vocab."""
        B = x0.size(0)
        t = torch.rand(B, device=x0.device)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
        v_hat = flow_net(x_t, t)
        v_star = x1 - x0
        l_mse = F.mse_loss(v_hat, v_star)
        # endpoint prediction
        x1_hat = x_t + (1 - t[:, None]) * v_hat
        x1_n = F.normalize(x1_hat, dim=-1)
        logits = (x1_n @ vocab_mean_n.t()) / self.tau_endpoint
        l_ce = F.cross_entropy(logits, label_idx)
        return l_mse, l_ce, v_hat

    def _leak_loss_branch(self, x_other, x1, flow_net, label_idx, vocab_mean_n):
        B = x_other.size(0)
        t = torch.rand(B, device=x_other.device)
        x_t = (1 - t[:, None]) * x_other + t[:, None] * x1
        v_hat = flow_net(x_t, t)
        v_star = x1 - x_other
        l_mse = F.mse_loss(v_hat, v_star)
        x1_hat = x_t + (1 - t[:, None]) * v_hat
        x1_n = F.normalize(x1_hat, dim=-1)
        logits = (x1_n @ vocab_mean_n.t()) / self.tau_endpoint
        l_ce = F.cross_entropy(logits, label_idx)
        return l_mse + l_ce

    def train_forward(self, batch, idx):
        # Single forward — replicate the baseline's compute path here so flow path
        # reuses the same image features (avoids double encode_image OOM).
        batch_img = batch[0].cuda()
        attr_idx = batch[1].cuda()
        obj_idx = batch[2].cuda()

        f_global, _ = self.encode_image(batch_img.type(self.clip.dtype))
        B = f_global.shape[0]

        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)
        self._update_prototypes(f_attr, f_obj, attr_idx.cpu(), obj_idx.cpu())

        attr_labels = self._get_cluster_labels(f_attr, attr_idx.cpu(), self.attributes, "attr")
        obj_labels = self._get_cluster_labels(f_obj, obj_idx.cpu(), self.classes, "obj")
        attr_protos = self._gather_all_prototypes("attr", self.attributes)
        obj_protos = self._gather_all_prototypes("obj", self.classes)

        f_attr_proj = self.attr_proj(f_attr)
        f_obj_proj = self.obj_proj(f_obj)

        all_protos_flat = torch.cat([
            self.attr_proj(attr_protos.reshape(-1, attr_protos.shape[-1])).reshape(-1, attr_protos.shape[-1]),
            self.obj_proj(obj_protos.reshape(-1, obj_protos.shape[-1])).reshape(-1, obj_protos.shape[-1]),
        ], dim=0)
        attr_all_protos = all_protos_flat.unsqueeze(0).expand(B, -1, -1)
        obj_all_protos = all_protos_flat.unsqueeze(0).expand(B, -1, -1)
        loss_contrastive = (
            self.nceloss(f_attr_proj, attr_all_protos, attr_labels) +
            self.nceloss(f_obj_proj, obj_all_protos, obj_labels)
        )

        attr_proto_selected = torch.stack([
            getattr(self, f"attr_queue{int(attr_idx[i])}")[attr_labels[i] % self.cluster_num]
            for i in range(B)
        ])
        obj_proto_selected = torch.stack([
            getattr(self, f"obj_queue{int(obj_idx[i])}")[obj_labels[i] % self.cluster_num]
            for i in range(B)
        ])
        from model.hsic import hsic_normalized
        loss_hsic = (
            hsic_normalized(f_attr_proj, obj_proto_selected.detach()) +
            0.5 * hsic_normalized(f_obj_proj, attr_proto_selected.detach())
        )

        token_tensors = self._construct_token_tensors(idx)
        img_feats = [f_global, f_attr_proj, f_obj_proj]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in img_feats]
        logit_scale = self.clip.logit_scale.exp()
        logits = []
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(self.token_ids[i], token_tensors[i], enable_pos_emb=self.enable_pos_emb)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits.append(logit_scale * norm_img[i] @ feat.t())
        comp_logits, attr_logits, obj_logits = logits
        loss_hard = f_global.new_zeros(())
        base_pred = (comp_logits, attr_logits, obj_logits, loss_contrastive, loss_hsic, loss_hard)

        # ---- Flow path: reuse the SAME features ----
        x_c_0 = f_global.float()
        x_a_0 = f_attr_proj.float()
        x_o_0 = f_obj_proj.float()

        # Sample LLM bank endpoints
        bank = self.bank
        pair_idx = bank.pair_indices(attr_idx, obj_idx)
        # All train pairs should be in the bank; if not, mask them out.
        valid = pair_idx >= 0
        if valid.sum() < pair_idx.size(0):
            x_a_0 = x_a_0[valid]; x_o_0 = x_o_0[valid]; x_c_0 = x_c_0[valid]
            attr_idx = attr_idx[valid]; obj_idx = obj_idx[valid]
            pair_idx = pair_idx[valid]

        if x_a_0.size(0) == 0:
            zero = base_pred[0].new_zeros(())
            # Append zero aux losses for loss_calu
            return base_pred + (zero, zero, zero)

        x_a_1 = bank.sample_attr(attr_idx)
        x_o_1 = bank.sample_obj(obj_idx)
        x_c_1 = bank.sample_comp(pair_idx)

        # Per-branch FM losses
        l_a_mse, l_a_ce, v_a_hat = self._fm_loss_branch(x_a_0, x_a_1, self.flow_a, attr_idx, bank.attr_mean_n)
        l_o_mse, l_o_ce, v_o_hat = self._fm_loss_branch(x_o_0, x_o_1, self.flow_o, obj_idx, bank.obj_mean_n)
        l_a_fm = l_a_mse + l_a_ce
        l_o_fm = l_o_mse + l_o_ce

        # Composer loss: least-squares target vs predicted coeffs
        v_a_n = F.normalize(v_a_hat, dim=-1)
        v_o_n = F.normalize(v_o_hat, dim=-1)
        v_c_star = x_c_1 - x_c_0
        a_star, b_star = solve_least_squares(v_a_n, v_o_n, v_c_star)
        coeffs = self.composer(v_a_n.detach(), v_o_n.detach())
        l_comp = F.mse_loss(coeffs, torch.stack([a_star, b_star], dim=-1))

        # Leakage augmentation (use composition feature as the "other" branch source)
        if self.use_leakage:
            l_a_leak = self._leak_loss_branch(x_c_0, x_a_1, self.flow_a, attr_idx, bank.attr_mean_n)
            l_o_leak = self._leak_loss_branch(x_c_0, x_o_1, self.flow_o, obj_idx, bank.obj_mean_n)
        else:
            l_a_leak = x_a_0.new_zeros(())
            l_o_leak = x_a_0.new_zeros(())

        l_fm_total = self.lambda_flow * (l_a_fm + l_o_fm)
        l_comp_total = self.lambda_comp * l_comp
        l_leak_total = self.lambda_leak * (l_a_leak + l_o_leak)

        return base_pred + (l_fm_total, l_comp_total, l_leak_total)

    def val_forward(self, batch, idx):
        # Single forward: compute baseline logits AND flow endpoints from one image encode.
        batch_img = batch[0].cuda()
        f_global, _ = self.encode_image(batch_img.type(self.clip.dtype))
        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)
        f_attr_proj = self.attr_proj(f_attr)
        f_obj_proj = self.obj_proj(f_obj)

        token_tensors = self._construct_token_tensors(idx)
        img_feats = [f_global, f_attr_proj, f_obj_proj]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in img_feats]
        logit_scale = self.clip.logit_scale.exp()
        logits = []
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(self.token_ids[i], token_tensors[i], enable_pos_emb=self.enable_pos_emb)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits.append(logit_scale * norm_img[i] @ feat.t())
        comp_logits, attr_logits, obj_logits = logits

        x_c_0 = f_global.float()
        x_a_0 = f_attr_proj.float()
        x_o_0 = f_obj_proj.float()
        # Euler one-step from t=0 (h * v(x_0, 0))
        t0 = torch.zeros(x_a_0.size(0), device=x_a_0.device)
        with torch.no_grad():
            v_a = self.flow_a(x_a_0, t0)
            v_o = self.flow_o(x_o_0, t0)
            v_a_n = F.normalize(v_a, dim=-1)
            v_o_n = F.normalize(v_o, dim=-1)
            coeffs = self.composer(v_a_n, v_o_n)
            a_hat, b_hat = coeffs[:, 0:1], coeffs[:, 1:2]
            v_c = a_hat * v_a_n + b_hat * v_o_n
            x_a_T = x_a_0 + self.flow_step_h * v_a
            x_o_T = x_o_0 + self.flow_step_h * v_o
            x_c_T = x_c_0 + self.flow_step_h * v_c

        return (comp_logits, attr_logits, obj_logits, x_a_T, x_o_T, x_c_T)

    def loss_calu(self, predict, target):
        if self.training:
            # Last 3 elements are flow/comp/leak; strip and delegate the rest to baseline
            base_pred = predict[:-3]
            l_fm, l_comp, l_leak = predict[-3], predict[-2], predict[-1]
            base_loss = super().loss_calu(base_pred, target)
            return base_loss + l_fm + l_comp + l_leak
        else:
            base_pred = predict[:3]
            return super().loss_calu(base_pred, target)

    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits, x_a_T, x_o_T, x_c_T = predict
        # Baseline blended score (existing pipeline)
        base_score = super().logit_infer((comp_logits, attr_logits, obj_logits), pairs)

        bank = self.bank
        x_a_n = F.normalize(x_a_T, dim=-1)
        x_o_n = F.normalize(x_o_T, dim=-1)
        x_c_n = F.normalize(x_c_T, dim=-1)
        # pairs: [N_pairs, 2] of (attr_idx, obj_idx)
        attr_ids = pairs[:, 0]
        obj_ids = pairs[:, 1]

        if self.inference_mode == "mean":
            T_a = bank.attr_mean_n[attr_ids]                  # [Np, D]
            T_o = bank.obj_mean_n[obj_ids]
            p_a = x_a_n @ T_a.t()                              # [B, Np]
            p_o = x_o_n @ T_o.t()
            # comp embedding per pair: look up by (a,o)
            pair_keys = [(int(a), int(o)) for a, o in pairs.cpu().tolist()]
            comp_ids = torch.tensor(
                [bank._pair_lookup.get(k, -1) for k in pair_keys],
                device=x_c_n.device,
            )
            valid = comp_ids >= 0
            T_c = bank.comp_mean_n[comp_ids.clamp(min=0)]
            p_c = x_c_n @ T_c.t()
            # Mask out invalid pairs (shouldn't happen in closed-world)
            if (~valid).any():
                p_c[:, ~valid] = 0.0
        else:  # 'max'
            # [B, Np, K]
            T_a = bank.attr_n[attr_ids]                        # [Np, K, D]
            T_o = bank.obj_n[obj_ids]
            p_a = torch.einsum("bd,nkd->bnk", x_a_n, T_a).max(dim=-1).values
            p_o = torch.einsum("bd,nkd->bnk", x_o_n, T_o).max(dim=-1).values
            pair_keys = [(int(a), int(o)) for a, o in pairs.cpu().tolist()]
            comp_ids = torch.tensor(
                [bank._pair_lookup.get(k, -1) for k in pair_keys],
                device=x_c_n.device,
            )
            valid = comp_ids >= 0
            T_c = bank.comp_n[comp_ids.clamp(min=0)]
            p_c = torch.einsum("bd,nkd->bnk", x_c_n, T_c).max(dim=-1).values
            if (~valid).any():
                p_c[:, ~valid] = 0.0

        flow_score = p_c + p_a * p_o
        # Bring to comparable scale w/ baseline logits (which are logit_scale-multiplied)
        # baseline scores can be in the ~30 range; flow_score in ~[-2, 2]. Scale.
        flow_score = flow_score * float(self.clip.logit_scale.exp().detach())
        return self.flow_blend * base_score + (1 - self.flow_blend) * flow_score
