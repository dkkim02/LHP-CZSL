"""
ClusPro Baseline: ClusDPC에서 DPC/MSCI/VAPS/DHNO를 제거한 순수 ClusPro 재현.

ClusPro 핵심만 남김:
- Visual Adapter (ViT 매 블록)
- Attr/Obj Disentangler
- Prototype Clustering (momentum update)
- Prototype Contrastive Loss + HSIC Decorrelation
- 단일 Soft Prompt (Dual X, Shifting X, Gating X)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import CustomTextEncoder
from model.nce_loss import ContrastiveLoss
from model.hsic import hsic_normalized


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck=64, dropout=0.0, adapter_scalar="0.1"):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.dropout = dropout
        self.scale = float(adapter_scalar)
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        down = self.non_linear_func(self.down_proj(x))
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale
        return (up + residual) if add_residual else up


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        return F.dropout(F.relu(self.bn1_fc(self.fc1(x))), training=self.training)


class ClusProBaseline(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()

        clip_arch = config.clip_arch if hasattr(config, 'clip_arch') and config.clip_arch else config.clip_model
        self.clip = load_clip(name=clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.offset = offset
        self.enable_pos_emb = True

        dtype = self.clip.dtype or torch.float16
        self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, dtype)

        # Freeze CLIP
        for p in self.parameters():
            p.requires_grad = False

        output_dim = self.clip.visual.output_dim

        # ---- Visual Adapters ----
        num_blocks = self.clip.visual.transformer.layers
        vision_width = self.clip.visual.transformer.width
        adapter_dim = getattr(config, 'adapter_dim', 64)
        adapter_dropout = getattr(config, 'adapter_dropout', 0.1)
        self.visual_adapters = nn.ModuleList([
            Adapter(vision_width, adapter_dim, adapter_dropout)
            for _ in range(2 * num_blocks)
        ])

        # ---- Disentanglers ----
        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)
        self.attr_proj = Disentangler(output_dim)
        self.obj_proj = Disentangler(output_dim)

        # ---- Prototype Memory ----
        self.cluster_num = getattr(config, 'cluster_num', 5)
        self.momentum = getattr(config, 'proto_momentum', 0.99)

        for i in range(len(self.attributes)):
            self.register_buffer(f"attr_queue{i}", torch.randn(self.cluster_num, output_dim))
        for i in range(len(self.classes)):
            self.register_buffer(f"obj_queue{i}", torch.randn(self.cluster_num, output_dim))

        # ---- Contrastive Loss ----
        self.nceloss = ContrastiveLoss()
        self.attr_dropout = nn.Dropout(getattr(config, 'attr_dropout', 0.3))

        # ---- Single Soft Prompt (NO dual prompt) ----
        self.token_ids, base_soft_emb, self.comp_ctx, self.attr_ctx, self.obj_ctx = \
            self._construct_soft_prompt()
        self.soft_att_obj = nn.Parameter(base_soft_emb)
        self.comp_ctx = nn.Parameter(self.comp_ctx)
        self.attr_ctx = nn.Parameter(self.attr_ctx)
        self.obj_ctx = nn.Parameter(self.obj_ctx)

        # ---- Loss weights ----
        self.pair_loss_weight = getattr(config, 'pair_loss_weight', 1.0)
        self.attr_loss_weight = getattr(config, 'attr_loss_weight', 1.0)
        self.obj_loss_weight = getattr(config, 'obj_loss_weight', 1.0)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.1)
        self.hsic_weight = getattr(config, 'hsic_weight', 0.1)

        # ---- Inference weights ----
        self.pair_inf_w = getattr(config, 'pair_inference_weight', 1.0)
        self.attr_inf_w = getattr(config, 'attr_inference_weight', 1.0)
        self.obj_inf_w = getattr(config, 'obj_inference_weight', 1.0)

    # ==================================================================
    # Image Encoding
    # ==================================================================

    def encode_image(self, x):
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.clip.visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        num_blocks = self.clip.visual.transformer.layers
        for i in range(num_blocks):
            block = self.clip.visual.transformer.resblocks[i]
            adapt_x = self.visual_adapters[i](x, add_residual=False)
            residual = x
            x = block.attention(block.ln_1(x))
            x = x + adapt_x + residual
            adapt_x = self.visual_adapters[i + num_blocks](x, add_residual=False)
            residual = x
            x = block.mlp(block.ln_2(x))
            x = x + adapt_x + residual

        x = x.permute(1, 0, 2)
        x = self.clip.visual.ln_post(x)
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x[:, 0, :], x

    # ==================================================================
    # Soft Prompt
    # ==================================================================

    def _construct_soft_prompt(self):
        prompt_template = self.config.prompt_template
        ctx_init = self.config.ctx_init

        token_ids = self.tokenizer(
            prompt_template, context_length=self.config.context_length
        ).cuda()

        tokenized = torch.cat([
            self.tokenizer(tok, context_length=self.config.context_length)
            for tok in self.attributes + self.classes
        ])
        orig_emb = self.clip.token_embedding(tokenized.cuda())
        soft_emb = torch.zeros(
            len(self.attributes) + len(self.classes), orig_emb.size(-1)
        )
        for idx, rep in enumerate(orig_emb):
            eos_idx = tokenized[idx].argmax()
            soft_emb[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx = embedding[0, 1:1+n_ctx[0], :].to(self.clip.dtype)
        attr_ctx = embedding[1, 1:1+n_ctx[1], :].to(self.clip.dtype)
        obj_ctx = embedding[2, 1:1+n_ctx[2], :].to(self.clip.dtype)

        return token_ids, soft_emb, comp_ctx, attr_ctx, obj_ctx

    def _construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        num_elements = [len(pair_idx), self.offset, len(self.classes)]
        token_tensor = []

        for i in range(self.token_ids.shape[0]):
            ids = self.token_ids[i].repeat(num_elements[i], 1)
            token_tensor.append(
                self.clip.token_embedding(ids.cuda()).type(self.clip.dtype)
            )

        eos_idx = [int(self.token_ids[i].argmax()) for i in range(self.token_ids.shape[0])]
        embs = self.attr_dropout(self.soft_att_obj)

        token_tensor[0][:, eos_idx[0]-2, :] = embs[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0]-1, :] = embs[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1:len(self.comp_ctx)+1, :] = self.comp_ctx.type(self.clip.dtype)

        token_tensor[1][:, eos_idx[1]-1, :] = embs[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1:len(self.attr_ctx)+1, :] = self.attr_ctx.type(self.clip.dtype)

        token_tensor[2][:, eos_idx[2]-1, :] = embs[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1:len(self.obj_ctx)+1, :] = self.obj_ctx.type(self.clip.dtype)

        return token_tensor

    # ==================================================================
    # Prototype Update
    # ==================================================================

    @torch.no_grad()
    @torch.autocast(device_type='cuda', enabled=False)
    def _update_prototypes(self, batch_attr, batch_obj, attr_idx, obj_idx):
        batch_attr_f = batch_attr.float()
        batch_obj_f = batch_obj.float()

        if not (torch.isfinite(batch_attr_f).all() and torch.isfinite(batch_obj_f).all()):
            return

        if bool(getattr(self.config, 'use_ot_assignment', False)):
            eps = float(getattr(self.config, 'sinkhorn_epsilon', 0.05))
            iters = int(getattr(self.config, 'sinkhorn_iters', 3))
            self._update_prototypes_ot(batch_attr_f, attr_idx, "attr", self.attributes, eps, iters)
            self._update_prototypes_ot(batch_obj_f, obj_idx, "obj", self.classes, eps, iters)
            return

        # Hard one-hot cluster assignment (matches official ClusPro train_forward, ICLR 2025)
        # Soft softmax assignment (previous version) collapses prototypes since every sample
        # contributes to every cluster — see RESEARCH_LOG §5.3.
        for k in range(len(self.attributes)):
            mask = (attr_idx == k)
            if mask.sum() == 0:
                continue
            feats_k = batch_attr_f[mask]
            queue_k = getattr(self, f"attr_queue{k}")
            queue_f = queue_k.float()
            sim = l2_normalize(feats_k) @ l2_normalize(queue_f).t()
            couplings = F.softmax(sim / 0.5, dim=-1)
            assign = F.gumbel_softmax(couplings, tau=0.5, hard=True, dim=-1)
            new_proto = assign.t() @ feats_k
            counts = assign.sum(dim=0)
            valid = counts > 0
            if valid.any():
                new_proto[valid] = l2_normalize(new_proto[valid])
                queue_f[valid] = queue_f[valid] * self.momentum + new_proto[valid] * (1 - self.momentum)
            queue_k.copy_(l2_normalize(queue_f).to(queue_k.dtype))

        for k in range(len(self.classes)):
            mask = (obj_idx == k)
            if mask.sum() == 0:
                continue
            feats_k = batch_obj_f[mask]
            queue_k = getattr(self, f"obj_queue{k}")
            queue_f = queue_k.float()
            sim = l2_normalize(feats_k) @ l2_normalize(queue_f).t()
            couplings = F.softmax(sim / 0.5, dim=-1)
            assign = F.gumbel_softmax(couplings, tau=0.5, hard=True, dim=-1)
            new_proto = assign.t() @ feats_k
            counts = assign.sum(dim=0)
            valid = counts > 0
            if valid.any():
                new_proto[valid] = l2_normalize(new_proto[valid])
                queue_f[valid] = queue_f[valid] * self.momentum + new_proto[valid] * (1 - self.momentum)
            queue_k.copy_(l2_normalize(queue_f).to(queue_k.dtype))

    @torch.no_grad()
    @torch.autocast(device_type='cuda', enabled=False)
    def _update_prototypes_ot(self, batch_feat_f, idx, prim_type, primitives, eps, iters):
        # §17-10 H3: full-batch Sinkhorn coupling, then per-class subset for EMA update.
        # Matches official ClusPro train_forward (root cluspro_baseline.py:692-725):
        # init_q computed over ALL B samples (not per-class subset) so Sinkhorn balance
        # is meaningful, then gumbel hard assign, then restrict to samples with attr/obj == k.
        from model.otgcc import local_assign_ot

        all_protos = torch.stack(
            [getattr(self, f"{prim_type}_queue{k}").float() for k in range(len(primitives))],
            dim=0,
        )  # (num_prim, cluster_num, D)
        all_protos_n = l2_normalize(all_protos)
        batch_n = l2_normalize(batch_feat_f)
        # masks[b, m, k] = sim(sample b, prototype m of class k)
        masks = torch.einsum('bd,kmd->bmk', batch_n, all_protos_n)

        for k in range(len(primitives)):
            init_q = masks[..., k]  # (B, cluster_num)
            couplings, _ = local_assign_ot(
                batch_feat_f.detach(), init_q.detach(),
                sinkhorn_iters=iters, epsilon=eps, top_percent=1.0,
            )
            couplings = couplings.float()
            q = F.gumbel_softmax(couplings, tau=0.5, hard=True, dim=-1)  # (B, cluster_num)
            sel = (idx == k)
            if sel.sum() == 0:
                continue
            q_k = q[sel]
            feats_k = batch_feat_f[sel]
            new_proto = q_k.t() @ feats_k  # (cluster_num, D)
            counts = q_k.sum(dim=0)
            valid = counts > 0
            queue_k = getattr(self, f"{prim_type}_queue{k}")
            queue_f = queue_k.float()
            if valid.any():
                new_proto[valid] = l2_normalize(new_proto[valid])
                queue_f[valid] = queue_f[valid] * self.momentum + new_proto[valid] * (1 - self.momentum)
            queue_k.copy_(l2_normalize(queue_f).to(queue_k.dtype))

    def _get_cluster_labels(self, batch_feat, idx_labels, primitives, prim_type):
        B = batch_feat.shape[0]
        labels = torch.zeros(B, dtype=torch.long, device=batch_feat.device)
        for k in range(len(primitives)):
            mask = (idx_labels == k)
            if mask.sum() == 0:
                continue
            queue_k = getattr(self, f"{prim_type}_queue{k}")
            sim = l2_normalize(batch_feat[mask]) @ l2_normalize(queue_k).t()
            cluster_idx = sim.argmax(dim=-1)
            labels[mask] = cluster_idx + k * self.cluster_num
        return labels

    def _gather_all_prototypes(self, prim_type, primitives):
        protos = []
        for k in range(len(primitives)):
            protos.append(getattr(self, f"{prim_type}_queue{k}"))
        return torch.stack(protos, dim=0)

    # ==================================================================
    # Forward
    # ==================================================================

    def train_forward(self, batch, idx):
        batch_img = batch[0].cuda()
        attr_idx, obj_idx = batch[1], batch[2]

        # H1 (§17-8): when same_prim_sample=True + hard_pair_weight>0, batch carries
        # (same_attr/diff_obj img, same_obj/diff_attr img) + masks at indices 4..13.
        hpw = float(getattr(self.config, 'hard_pair_weight', 0.0))
        has_hard_pair = (len(batch) >= 14) and (hpw > 0) and self.training
        if has_hard_pair:
            sa_img = batch[4].cuda()
            sa_mask = batch[8]
            so_img = batch[9].cuda()
            so_mask = batch[13]
            all_img = torch.cat([batch_img, sa_img, so_img], dim=0)
            f_global_all, _ = self.encode_image(all_img.type(self.clip.dtype))
            B = batch_img.shape[0]
            f_global = f_global_all[:B]
            f_sa = f_global_all[B:2*B]
            f_so = f_global_all[2*B:]
        else:
            f_global, _ = self.encode_image(batch_img.type(self.clip.dtype))
            B = f_global.shape[0]

        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)

        self._update_prototypes(f_attr, f_obj, attr_idx, obj_idx)

        attr_labels = self._get_cluster_labels(f_attr, attr_idx, self.attributes, "attr")
        obj_labels = self._get_cluster_labels(f_obj, obj_idx, self.classes, "obj")

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
            getattr(self, f"attr_queue{int(attr_idx[i])}")[
                attr_labels[i] % self.cluster_num
            ] for i in range(B)
        ])
        obj_proto_selected = torch.stack([
            getattr(self, f"obj_queue{int(obj_idx[i])}")[
                obj_labels[i] % self.cluster_num
            ] for i in range(B)
        ])

        loss_hsic = (
            hsic_normalized(f_attr_proj, obj_proto_selected.detach()) +
            0.5 * hsic_normalized(f_obj_proj, attr_proto_selected.detach())
        )

        # ---- Single text encoding (NO dual prompt) ----
        token_tensors = self._construct_token_tensors(idx)

        img_feats = [f_global, f_attr_proj, f_obj_proj]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in img_feats]
        logit_scale = self.clip.logit_scale.exp()

        logits = []
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(
                self.token_ids[i], token_tensors[i], enable_pos_emb=self.enable_pos_emb
            )
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits.append(logit_scale * norm_img[i] @ feat.t())

        comp_logits, attr_logits, obj_logits = logits

        if has_hard_pair:
            sa_attr_proj = self.attr_proj(self.attr_disentangler(f_sa))
            sa_obj_proj  = self.obj_proj(self.obj_disentangler(f_sa))
            so_attr_proj = self.attr_proj(self.attr_disentangler(f_so))
            so_obj_proj  = self.obj_proj(self.obj_disentangler(f_so))
            tau = float(getattr(self.config, 'hard_pair_temperature', 0.1))
            loss_hard = self._hard_pair_loss(
                f_attr_proj, f_obj_proj,
                sa_attr_proj, sa_obj_proj,
                so_attr_proj, so_obj_proj,
                sa_mask, so_mask, tau,
            )
        else:
            loss_hard = f_global.new_zeros(())

        return comp_logits, attr_logits, obj_logits, loss_contrastive, loss_hsic, loss_hard

    def val_forward(self, batch, idx):
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
            feat, _ = self.text_encoder(
                self.token_ids[i], token_tensors[i], enable_pos_emb=self.enable_pos_emb
            )
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits.append(logit_scale * norm_img[i] @ feat.t())

        return logits[0], logits[1], logits[2]

    # ==================================================================
    # Loss & Inference
    # ==================================================================

    def _cosface_margin(self, logits, target, m, scale):
        # CosFace additive cosine margin (§16-3 D1/D5).
        # logits = scale * cos(theta). Subtract scale*m from target class only.
        one_hot = F.one_hot(target, num_classes=logits.size(-1)).to(logits.dtype)
        return logits - (m * scale) * one_hot

    def _hard_pair_loss(self, anchor_attr, anchor_obj,
                        sa_attr, sa_obj, so_attr, so_obj,
                        sa_mask, so_mask, tau):
        # H1 (§17-8): InfoNCE on disentangled projection features.
        # L_attr: anchor_attr ↔ sa_attr (same attribute) positive,
        #         within-batch so_attr (different attribute, same object) negatives.
        # L_obj : anchor_obj ↔ so_obj (same object) positive,
        #         within-batch sa_obj (different object, same attribute) negatives.
        aa = F.normalize(anchor_attr.float(), dim=-1)
        ao = F.normalize(anchor_obj.float(),  dim=-1)
        pa = F.normalize(sa_attr.float(), dim=-1)
        po = F.normalize(so_obj.float(),  dim=-1)
        na = F.normalize(so_attr.float(), dim=-1)
        no_ = F.normalize(sa_obj.float(),  dim=-1)

        B = aa.size(0)
        device = aa.device
        labels = torch.zeros(B, dtype=torch.long, device=device)

        pos_a = (aa * pa).sum(dim=-1, keepdim=True) / tau
        neg_a = aa @ na.t() / tau
        loss_a_per = F.cross_entropy(torch.cat([pos_a, neg_a], dim=1), labels, reduction='none')

        pos_o = (ao * po).sum(dim=-1, keepdim=True) / tau
        neg_o = ao @ no_.t() / tau
        loss_o_per = F.cross_entropy(torch.cat([pos_o, neg_o], dim=1), labels, reduction='none')

        sa_m = sa_mask.float().to(device)
        so_m = so_mask.float().to(device)
        eps = 1e-8
        loss_a = (loss_a_per * sa_m).sum() / (sa_m.sum() + eps)
        loss_o = (loss_o_per * so_m).sum() / (so_m.sum() + eps)
        return loss_a + loss_o

    def loss_calu(self, predict, target):
        loss_fn = nn.CrossEntropyLoss()
        batch_attr, batch_obj, batch_target = target[1], target[2], target[3]
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()

        if self.training:
            comp_logits, attr_logits, obj_logits, loss_contras, loss_hsic, loss_hard = predict
        else:
            comp_logits, attr_logits, obj_logits = predict

        m = float(getattr(self.config, 'cosface_margin', 0.0))
        if self.training and m > 0:
            scale = self.clip.logit_scale.exp().detach()
            comp_logits = self._cosface_margin(comp_logits, batch_target, m, scale)
            attr_logits = self._cosface_margin(attr_logits, batch_attr, m, scale)
            obj_logits = self._cosface_margin(obj_logits, batch_obj, m, scale)

        loss = (
            self.pair_loss_weight * loss_fn(comp_logits, batch_target) +
            self.attr_loss_weight * loss_fn(attr_logits, batch_attr) +
            self.obj_loss_weight * loss_fn(obj_logits, batch_obj)
        )

        if self.training:
            loss = loss + self.contrastive_weight * loss_contras + self.hsic_weight * loss_hsic
            hpw = float(getattr(self.config, 'hard_pair_weight', 0.0))
            if hpw > 0:
                loss = loss + hpw * loss_hard

        return loss

    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)

        for i in range(comp_logits.shape[-1]):
            w_attr = 1 if self.attr_inf_w == 0 else attr_pred[:, pairs[i][0]] * self.attr_inf_w
            w_obj = 1 if self.obj_inf_w == 0 else obj_pred[:, pairs[i][1]] * self.obj_inf_w
            comp_logits[:, i] = comp_logits[:, i] * self.pair_inf_w + w_attr * w_obj

        return comp_logits

    def forward(self, batch, idx):
        if self.training:
            return self.train_forward(batch, idx)
        else:
            with torch.no_grad():
                return self.val_forward(batch, idx)
