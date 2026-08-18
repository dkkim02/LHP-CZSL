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


class LoRADelta(nn.Module):
    """Low-rank weight delta  ΔW = (alpha/r) * B @ A,  A in (r,in), B in (out,r).
    A: kaiming_uniform, B: ZERO -> ΔW == 0 at init (step-0 identity).
    Stored in fp32; callers cast as needed. `delta_weight()` returns ΔW [out,in];
    `forward(x)` returns x @ ΔW.T  (a low-rank Linear residual)."""
    def __init__(self, in_dim, out_dim, r, alpha):
        super().__init__()
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.A = nn.Parameter(torch.empty(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B stays zero -> delta starts at exactly 0.

    def delta_weight(self):
        return self.scale * (self.B @ self.A)          # [out, in], fp32

    def forward(self, x):
        # x: [..., in] -> [..., out];  low-rank, grad flows to A,B.
        return self.scale * (x @ self.A.t()) @ self.B.t()


class _FP32LayerNorm(nn.LayerNorm):
    """LayerNorm that computes in fp32 then casts back (fp16/AMP-safe).
    Mirrors model/common.py:113-119."""
    def forward(self, x: torch.Tensor):
        orig = x.dtype
        return super().forward(x.type(torch.float32)).type(orig)


class _GroupNorm1d(nn.Module):
    """GroupNorm over a (B, D) tensor: treat D as channels (unsqueeze a spatial dim).
    fp32-safe like _FP32LayerNorm."""
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x: torch.Tensor):
        orig = x.dtype
        y = self.gn(x.type(torch.float32).unsqueeze(-1)).squeeze(-1)
        return y.type(orig)


class Disentangler(nn.Module):
    def __init__(self, emb_dim, norm_type='bn'):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.norm_type = norm_type
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(emb_dim)
        elif norm_type == 'ln':
            self.norm = _FP32LayerNorm(emb_dim)
        elif norm_type == 'gn':
            self.norm = _GroupNorm1d(32, emb_dim)
        elif norm_type == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown disent_norm: {norm_type}")

    def forward(self, x):
        return F.dropout(F.relu(self.norm(self.fc1(x))), training=self.training)


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

        # ---- Vision LoRA (Round-6) ----
        # ΔW low-rank deltas on the ViT attention projections, per block. B=0 init so
        # step-0 forward is byte-identical to the (adapters-only) baseline. drop_adapters
        # makes LoRA REPLACE the bottleneck adapters. lora_rank=0 -> entirely off.
        self.lora_rank = int(getattr(config, 'lora_rank', 0))
        _alpha = float(getattr(config, 'lora_alpha', 0.0))
        self.lora_alpha = _alpha if _alpha > 0 else float(self.lora_rank if self.lora_rank else 1.0)
        self.lora_targets = str(getattr(config, 'lora_targets', 'out'))   # 'out' | 'in_out'
        self.drop_adapters = bool(getattr(config, 'drop_adapters', False))
        self.use_lora = self.lora_rank > 0
        if self.use_lora:
            r, a = self.lora_rank, self.lora_alpha
            # out_proj: [width, width]; in_proj (packed QKV): [3*width, width].
            self.lora_out = nn.ModuleList([
                LoRADelta(vision_width, vision_width, r, a) for _ in range(num_blocks)
            ])
            if self.lora_targets == 'in_out':
                self.lora_in = nn.ModuleList([
                    LoRADelta(vision_width, 3 * vision_width, r, a) for _ in range(num_blocks)
                ])
        # When adapters are dropped, freeze them so they don't count as trainable / get updated.
        if self.drop_adapters:
            for p in self.visual_adapters.parameters():
                p.requires_grad = False

        # ---- Disentanglers ----
        disent_norm = getattr(config, 'disent_norm', 'bn')
        self.attr_disentangler = Disentangler(output_dim, norm_type=disent_norm)
        self.obj_disentangler = Disentangler(output_dim, norm_type=disent_norm)
        self.attr_proj = Disentangler(output_dim, norm_type=disent_norm)
        self.obj_proj = Disentangler(output_dim, norm_type=disent_norm)

        # ---- Prototype Memory ----
        self.cluster_num = getattr(config, 'cluster_num', 5)
        self.momentum = getattr(config, 'proto_momentum', 0.99)

        # Optional prototype seeding bank: dict with 'attr' (|A|,K,D) and 'obj' (|O|,K,D).
        # Absent -> randn init (baseline-identical). Seeds live in the queue space
        # (attr_disentangler/obj_disentangler output), L2-normalized.
        proto_init_path = getattr(config, 'proto_init_path', None)
        _proto_bank = None
        if proto_init_path:
            _proto_bank = torch.load(proto_init_path, map_location='cpu')
            assert _proto_bank['attr'].shape[0] == len(self.attributes), \
                f"proto bank attr {_proto_bank['attr'].shape} != {len(self.attributes)}"
            assert _proto_bank['obj'].shape[0] == len(self.classes), \
                f"proto bank obj {_proto_bank['obj'].shape} != {len(self.classes)}"
            assert _proto_bank['attr'].shape[1] == self.cluster_num, \
                f"proto bank K {_proto_bank['attr'].shape[1]} != cluster_num {self.cluster_num}"

        for i in range(len(self.attributes)):
            if _proto_bank is not None:
                init = F.normalize(_proto_bank['attr'][i].float(), dim=-1)
            else:
                init = torch.randn(self.cluster_num, output_dim)
            self.register_buffer(f"attr_queue{i}", init)
        for i in range(len(self.classes)):
            if _proto_bank is not None:
                init = F.normalize(_proto_bank['obj'][i].float(), dim=-1)
            else:
                init = torch.randn(self.cluster_num, output_dim)
            self.register_buffer(f"obj_queue{i}", init)

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

        # ---- Round-9: Masked Composition Modeling (gated by mcm_weight>0) ----
        self.mcm_weight = float(getattr(config, 'mcm_weight', 0.0))
        self.mcm_variant = str(getattr(config, 'mcm_variant', 'textmask'))
        if self.mcm_weight > 0 and self.mcm_variant == 'textmask':
            # learnable [MASK] primitive token, initialized at the mean soft primitive embedding
            self.mcm_mask = nn.Parameter(base_soft_emb.mean(0).detach().clone())  # [width]

        # ---- Round-10: comp-only Knowledge Distillation from a frozen teacher ----
        # Teacher train logits over the 1962 closed-world pairs, indexed by dataset image index.
        # We distill into the comp branch (over train_pairs cols). Gated by kd_weight>0.
        self.kd_weight = float(getattr(config, 'kd_weight', 0.0))
        self.kd_tau = float(getattr(config, 'kd_tau', 4.0))
        self.kd_warmup_epochs = int(getattr(config, 'kd_warmup_epochs', 3))
        if self.kd_weight > 0:
            kd_path = getattr(config, 'kd_teacher_path', None)
            assert kd_path, "kd_weight>0 requires kd_teacher_path"
            tdict = torch.load(kd_path, map_location='cpu')
            tl = tdict['teacher_logits'].float()                      # [N_train, 1962]
            self.register_buffer('kd_teacher_logits', tl)
            self.register_buffer('kd_teacher_pairs', tdict['teacher_pairs'].long())  # [1962,2]
            self._kd_col_map = None
            # the buffer columns are over the dataset's 1962 closed-world pairs; we slice to
            # the train_pairs columns inside train_forward (idx == train_pairs there).

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

        # ---- PG-LPR: Per-pair Gated LLM Prior Residual ----
        self.lambda_gate = float(getattr(config, 'lambda_gate', 0.0))
        self.pglpr_warmup_epochs = int(getattr(config, 'pglpr_warmup_epochs', 10))
        self.current_epoch = 0
        bank_path = getattr(config, 'llm_pglpr_bank_path', None)
        self.use_pglpr = self.lambda_gate > 0 and bank_path is not None
        if self.use_pglpr:
            bank = torch.load(bank_path, map_location='cpu')
            # bank: dict with 'attr' (|A|, K, D), 'obj' (|O|, K, D), optionally 'comp'
            attr_mean = bank['attr'].float().mean(dim=1)  # (|A|, D)
            obj_mean = bank['obj'].float().mean(dim=1)   # (|O|, D)
            assert attr_mean.shape[0] == len(self.attributes), \
                f"LLM bank attr count {attr_mean.shape[0]} != model {len(self.attributes)}"
            assert obj_mean.shape[0] == len(self.classes), \
                f"LLM bank obj count {obj_mean.shape[0]} != model {len(self.classes)}"
            # L2-normalize the mean embeddings (Mode B uses these as text-prior directions).
            self.register_buffer('llm_attr_emb', F.normalize(attr_mean, dim=-1))
            self.register_buffer('llm_obj_emb', F.normalize(obj_mean, dim=-1))
            # sigmoid(-2.94) ≈ 0.05 — small initial gate
            self.g_attr = nn.Parameter(torch.full((len(self.attributes),), -2.94))
            self.g_obj = nn.Parameter(torch.full((len(self.classes),), -2.94))
            self.llm_beta = nn.Parameter(torch.tensor(0.1))

        # ---- Rank-3: Counterfactual-consistency (PART B) ----
        # Guarded so nothing new is constructed when cf_weight<=0 (baseline byte-identical).
        self.cf_weight = float(getattr(config, 'cf_weight', 0.0))
        self.cf_warmup_epochs = int(getattr(config, 'cf_warmup_epochs', 3))
        self.cf_recomb = str(getattr(config, 'cf_recomb', 'global_objswap'))
        if self.cf_weight > 0 and self.cf_recomb == 'mlp':
            # 2-layer MLP over cat([f_attr, f_obj[perm]]) -> D; last layer near-zero init
            # so the synthetic feature starts ≈ f_attr (identity-ish) and learns the swap.
            self.cf_mlp = nn.Sequential(
                nn.Linear(2 * output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )
            with torch.no_grad():
                nn.init.zeros_(self.cf_mlp[-1].weight)
                nn.init.zeros_(self.cf_mlp[-1].bias)

    # ==================================================================
    # Image Encoding
    # ==================================================================

    def _block_attention(self, block, x_ln, blk_i):
        """Attention forward for ViT block `blk_i` on the LN'd input x_ln (shape [L,B,W]).
        If LoRA is active, add ΔW to the PACKED in_proj weight (CLIP splits Q/K/V itself —
        we never reshape QKV) and add the out-proj low-rank residual. Otherwise this is
        exactly block.attention(x_ln)."""
        if not self.use_lora:
            return block.attention(x_ln)

        mha = block.attn
        # in_proj: optional packed-weight delta (let F.multi_head_attention_forward split).
        in_w = mha.in_proj_weight
        if self.lora_targets == 'in_out':
            in_w = in_w + self.lora_in[blk_i].delta_weight().to(in_w.dtype)
        attn_mask = block.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x_ln.dtype, device=x_ln.device)
        attn_out, _ = F.multi_head_attention_forward(
            x_ln, x_ln, x_ln, mha.embed_dim, mha.num_heads,
            in_w, mha.in_proj_bias,
            mha.bias_k, mha.bias_v, mha.add_zero_attn,
            mha.dropout if self.training else 0.0,
            mha.out_proj.weight, mha.out_proj.bias,
            training=self.training, key_padding_mask=None,
            need_weights=False, attn_mask=attn_mask,
        )
        # out_proj LoRA: low-rank residual driven by the block input (a global ΔW_out @ x_ln).
        attn_out = attn_out + self.lora_out[blk_i](x_ln).to(attn_out.dtype)
        return attn_out

    def _vit_block(self, x, i):
        """One ViT block (attention + adapters + mlp) on x [L,B,W]."""
        block = self.clip.visual.transformer.resblocks[i]
        use_adapters = not self.drop_adapters
        residual = x
        attn = self._block_attention(block, block.ln_1(x), i)
        if use_adapters:
            x = attn + self.visual_adapters[i](x, add_residual=False) + residual
        else:
            x = attn + residual
        residual = x
        mlp = block.mlp(block.ln_2(x))
        if use_adapters:
            x = mlp + self.visual_adapters[i + self.clip.visual.transformer.layers](x, add_residual=False) + residual
        else:
            x = mlp + residual
        return x

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
        # Gradient-checkpoint the blocks ONLY during R-Drop training (two-view 2x forward in
        # 10GB). Gated by self._rdrop_ckpt -> baseline (rdrop off) path is unchanged/full-speed.
        ckpt = getattr(self, "_rdrop_ckpt", False) and self.training and torch.is_grad_enabled()
        if ckpt:
            from torch.utils.checkpoint import checkpoint
            for i in range(num_blocks):
                x = checkpoint(self._vit_block, x, i, use_reentrant=False)
        else:
            for i in range(num_blocks):
                x = self._vit_block(x, i)

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

    def _llm_bias(self, f_img_norm, pair_idx):
        """PG-LPR Mode B: per-pair gated LLM logit bias.
        f_img_norm: (B, D) L2-normalized image features.
        pair_idx: (|pairs|, 2) [attr_id, obj_id].
        Returns: (B, |pairs|) bias to add to comp_logits.
        """
        attr_idx = pair_idx[:, 0]
        obj_idx = pair_idx[:, 1]
        dtype = f_img_norm.dtype
        llm_pair = self.llm_attr_emb[attr_idx] + self.llm_obj_emb[obj_idx]
        llm_pair_n = F.normalize(llm_pair, dim=-1)
        prior = f_img_norm.float() @ llm_pair_n.t().float()
        g = torch.sigmoid(self.g_attr[attr_idx]) * torch.sigmoid(self.g_obj[obj_idx])
        bias = self.llm_beta * g.unsqueeze(0) * prior
        return bias.to(dtype)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _counterfactual_loss(self, f_global, f_attr, f_obj,
                             f_attr_proj, f_obj_proj,
                             attr_idx, obj_idx, logit_scale,
                             attr_text, obj_text):
        """Rank-3 counterfactual-consistency (PART B).

        Within a batch, recombine sample i's attribute/context with sample j's
        object-content (j = within-batch derangement) to synthesize a feature for
        the (generally UNSEEN) composition (a_i, o_{perm[i]}). Supervise it through
        the COMP text-matching branch (the leak) plus a light primitive aux. This
        injects unseen-composition supervision the comp branch never otherwise sees.

        All inputs live in the CLS / output_dim space (f_global, f_attr, f_obj are
        all output_dim — Disentangler preserves dim). attr_text/obj_text are the
        already-normalized primitive text features from the main forward (no image
        re-encode).
        """
        B = f_global.shape[0]
        device = f_global.device

        # Within-batch derangement: perm[i] != i for all i.
        perm = torch.randperm(B, device=device)
        fixed = (perm == torch.arange(B, device=device))
        if fixed.any():
            perm = (perm + 1) % B

        # --- Recombination: keep i's attr/context, swap j=perm[i]'s object ---
        if self.cf_recomb == 'mlp':
            cf_in = torch.cat([f_attr, f_obj[perm]], dim=-1)
            f_cf = f_global + self.cf_mlp(cf_in.float()).to(f_global.dtype)
        else:  # 'global_objswap' (param-free default)
            f_cf = f_global + (f_obj[perm] - f_obj)

        # --- COMP-branch routing (load-bearing leak) ---
        # Synthetic target pairs = (a_i, o_{perm[i]}). Encode their comp text on the fly.
        attr_idx_d = attr_idx.to(device)
        obj_idx_d = obj_idx.to(device)
        cf_pair_idx = torch.stack([attr_idx_d, obj_idx_d[perm]], dim=1)  # [B,2]
        cf_token = self._construct_token_tensors(cf_pair_idx)[0]          # comp template (idx 0)
        cf_text, _ = self.text_encoder(
            self.token_ids[0], cf_token, enable_pos_emb=self.enable_pos_emb
        )
        cf_text = cf_text / cf_text.norm(dim=-1, keepdim=True)            # [B,D]

        f_cf_n = f_cf / f_cf.norm(dim=-1, keepdim=True)
        cf_logits = logit_scale * f_cf_n @ cf_text.t()                   # [B,B]; diag = correct
        labels = torch.arange(B, device=device)
        loss_cf_comp = F.cross_entropy(cf_logits, labels)

        # --- Light primitive aux (reuse main-forward primitive text feats) ---
        cf_attr_proj = self.attr_proj(self.attr_disentangler(f_cf))
        cf_obj_proj  = self.obj_proj(self.obj_disentangler(f_cf))
        cf_attr_n = cf_attr_proj / cf_attr_proj.norm(dim=-1, keepdim=True)
        cf_obj_n  = cf_obj_proj / cf_obj_proj.norm(dim=-1, keepdim=True)
        cf_attr_logits = logit_scale * cf_attr_n @ attr_text.t()         # [B,|A|]
        cf_obj_logits  = logit_scale * cf_obj_n @ obj_text.t()           # [B,|O|]
        cf_attr_ce = F.cross_entropy(cf_attr_logits, attr_idx_d)
        cf_obj_ce  = F.cross_entropy(cf_obj_logits, obj_idx_d[perm])

        return loss_cf_comp + 0.5 * (cf_attr_ce + cf_obj_ce)

    def _build_kd_col_map(self, idx):
        """Map each comp column (idx = train_pairs [P_tr,2]) to its column in the teacher's
        1962-pair ordering (kd_teacher_pairs). Returns a LongTensor [P_tr]."""
        tp = self.kd_teacher_pairs.cpu()
        key2col = {(int(tp[c, 0]), int(tp[c, 1])): c for c in range(tp.shape[0])}
        idx_cpu = idx.cpu()
        cols = [key2col[(int(idx_cpu[r, 0]), int(idx_cpu[r, 1]))] for r in range(idx_cpu.shape[0])]
        return torch.tensor(cols, dtype=torch.long, device=self.kd_teacher_logits.device)

    def _mcm_textmask_loss(self, f_global, attr_idx, obj_idx, logit_scale):
        """Masked Composition Modeling (text-mask variant, Round-9 primary).

        For each sample's TRUE pair (a_i, o_i), build the comp prompt but replace ONE
        primitive slot (attr or obj, 50/50) with the learnable [MASK] token. Encode these B
        masked-comp texts (comp template), L2-normalize. logits = logit_scale * norm(f_global)
        @ masked_text.t() -> [B,B]; CE against arange(B): each image must recover/identify its
        OWN full composition from the partially-masked prompt better than other samples'.
        Supervision is image->ground-truth-pair (no text-text cosine; not Idea-A)."""
        device = f_global.device
        B = f_global.shape[0]
        attr_idx = attr_idx.to(device)
        obj_idx = obj_idx.to(device)

        # comp token tensors for the B true pairs (mirror _construct_token_tensors comp slot).
        ids = self.token_ids[0].repeat(B, 1)
        tok = self.clip.token_embedding(ids.cuda()).type(self.clip.dtype)   # [B, L, d]
        eos0 = int(self.token_ids[0].argmax())
        embs = self.attr_dropout(self.soft_att_obj)
        attr_slot = embs[attr_idx].type(self.clip.dtype)                    # [B,d]
        obj_slot = embs[obj_idx + self.offset].type(self.clip.dtype)        # [B,d]
        mask_tok = self.mcm_mask.type(self.clip.dtype)                      # [d]

        # 50/50 per-sample: mask the attr slot (eos-2) or the obj slot (eos-1).
        mask_attr = (torch.rand(B, device=device) < 0.5)                    # True -> mask attr
        a_slot = torch.where(mask_attr.unsqueeze(-1), mask_tok.expand(B, -1), attr_slot)
        o_slot = torch.where(mask_attr.unsqueeze(-1), obj_slot, mask_tok.expand(B, -1))
        tok[:, eos0 - 2, :] = a_slot
        tok[:, eos0 - 1, :] = o_slot
        tok[:, 1:len(self.comp_ctx) + 1, :] = self.comp_ctx.type(self.clip.dtype)

        feat, _ = self.text_encoder(self.token_ids[0], tok, enable_pos_emb=self.enable_pos_emb)
        feat = feat / feat.norm(dim=-1, keepdim=True)                       # [B,d]
        fg_n = f_global / f_global.norm(dim=-1, keepdim=True)
        logits_mcm = logit_scale * fg_n @ feat.t()                         # [B,B]
        labels = torch.arange(B, device=device)
        return F.cross_entropy(logits_mcm, labels)

    def _mcm_imgmask_loss(self, f_global, attr_idx, obj_idx, attr_text, obj_text, logit_scale):
        """MCM image-mask variant (Round-9, the devil's predicted-disease version).
        Zero f_obj -> obj head -> obj-text logits -> CE vs true obj; symmetric: zero f_attr ->
        attr head -> CE vs true attr. (Recover the masked primitive from the OTHER stream.)"""
        device = f_global.device
        attr_idx = attr_idx.to(device); obj_idx = obj_idx.to(device)

        # zero the obj stream, recover obj from the attr-derived path's obj head
        f_obj_zero = self.obj_disentangler(f_global) * 0.0
        f_obj_proj_z = self.obj_proj(f_obj_zero)
        fo_n = f_obj_proj_z / f_obj_proj_z.norm(dim=-1, keepdim=True)
        obj_logits_z = logit_scale * fo_n @ obj_text.t()
        obj_ce = F.cross_entropy(obj_logits_z, obj_idx)

        f_attr_zero = self.attr_disentangler(f_global) * 0.0
        f_attr_proj_z = self.attr_proj(f_attr_zero)
        fa_n = f_attr_proj_z / f_attr_proj_z.norm(dim=-1, keepdim=True)
        attr_logits_z = logit_scale * fa_n @ attr_text.t()
        attr_ce = F.cross_entropy(attr_logits_z, attr_idx)
        return obj_ce + attr_ce

    def train_forward(self, batch, idx):
        batch_img = batch[0].cuda()
        attr_idx, obj_idx = batch[1], batch[2]

        # R-Drop (Round-8): when active, gradient-checkpoint BOTH image forwards so the
        # two-view 2x ViT-L/14 fits in 10GB. Flag is read inside encode_image; off otherwise.
        self._rdrop_ckpt = float(getattr(self.config, 'rdrop_weight', 0.0)) > 0

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
        text_feats = []  # capture normalized text feats (i=1 attr, i=2 obj) for CF aux
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(
                self.token_ids[i], token_tensors[i], enable_pos_emb=self.enable_pos_emb
            )
            feat = feat / feat.norm(dim=-1, keepdim=True)
            text_feats.append(feat)
            logit_i = logit_scale * norm_img[i] @ feat.t()
            if i == 0 and self.use_pglpr:
                logit_i = logit_i + self._llm_bias(norm_img[0], idx)
            logits.append(logit_i)

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

        # ---- Rank-3: Counterfactual-consistency (PART B) ----
        # Guarded by cf_weight>0 so baseline is byte-identical when disabled.
        cf_weight = float(getattr(self.config, 'cf_weight', 0.0))
        if cf_weight > 0 and self.training and B > 1:
            loss_cf = self._counterfactual_loss(
                f_global, f_attr, f_obj, f_attr_proj, f_obj_proj,
                attr_idx, obj_idx, logit_scale,
                attr_text=text_feats[1], obj_text=text_feats[2],
            )
        else:
            loss_cf = f_global.new_zeros(())

        # ---- Round-8: R-Drop comp-consistency (gated by rdrop_weight>0) ----
        # Second image forward with INDEPENDENT dropout (encode_image adapters + disentangler
        # dropout re-sampled). TEXT is encoded once and SHARED (reuse text_feats[0]) -> <2x cost.
        # symKL on the COMP softmax only (attr/obj left untouched). CE stays on view-1 only
        # (computed in loss_calu from the returned comp_logits). When off -> 0, returned.
        rdrop_weight = float(getattr(self.config, 'rdrop_weight', 0.0))
        if rdrop_weight > 0 and self.training:
            # Second image forward with INDEPENDENT dropout. encode_image gradient-checkpoints
            # its blocks (self._rdrop_ckpt set above) so both views fit in 10GB. TEXT shared.
            f_global_2, _ = self.encode_image(batch_img.type(self.clip.dtype))
            f_global_2_n = f_global_2 / f_global_2.norm(dim=-1, keepdim=True)
            comp_logits_v2 = logit_scale * f_global_2_n @ text_feats[0].t()       # shared comp text
            if self.use_pglpr:
                comp_logits_v2 = comp_logits_v2 + self._llm_bias(f_global_2_n, idx)
            p1 = F.softmax(comp_logits.float(), dim=-1)
            p2 = F.softmax(comp_logits_v2.float(), dim=-1)
            eps = 1e-8
            p1c = p1.clamp_min(eps); p2c = p2.clamp_min(eps)
            loss_rdrop = ((p1c * (p1c / p2c).log()).sum(-1)
                          + (p2c * (p2c / p1c).log()).sum(-1)).mean()             # symKL
        else:
            loss_rdrop = f_global.new_zeros(())

        # ---- Round-9: Masked Composition Modeling (gated by mcm_weight>0) ----
        mcm_weight = float(getattr(self.config, 'mcm_weight', 0.0))
        if mcm_weight > 0 and self.training and B > 1:
            if self.mcm_variant == 'imgmask':
                loss_mcm = self._mcm_imgmask_loss(
                    f_global, attr_idx, obj_idx, text_feats[1], text_feats[2], logit_scale)
            else:  # 'textmask' (primary)
                loss_mcm = self._mcm_textmask_loss(f_global, attr_idx, obj_idx, logit_scale)
        else:
            loss_mcm = f_global.new_zeros(())

        # ---- Round-10: comp-only KD from frozen teacher (gated by kd_weight>0) ----
        kd_weight = float(getattr(self.config, 'kd_weight', 0.0))
        if kd_weight > 0 and self.training and len(batch) >= 5:
            # build train_pairs -> teacher-column map once (idx here == train_pairs [P_tr,2])
            if getattr(self, '_kd_col_map', None) is None:
                # teacher cols are over the dataset's 1962 pairs in a FIXED order; we need the
                # column for each train_pair. We reconstruct that order from the comp idx's
                # universe via the teacher buffer width and a pair->col dict passed at build.
                self._kd_col_map = self._build_kd_col_map(idx)
            ds_idx = batch[4].long().to(comp_logits.device)           # [B] dataset image index
            teacher_full = self.kd_teacher_logits[ds_idx.cpu()].to(comp_logits.device).float()  # [B,1962]
            teacher_comp = teacher_full[:, self._kd_col_map]           # [B, P_tr] aligned to comp_logits
            tau = self.kd_tau
            log_p_student = F.log_softmax(comp_logits.float() / tau, dim=-1)
            q_teacher = F.softmax(teacher_comp / tau, dim=-1)
            loss_kd = (tau * tau) * F.kl_div(log_p_student, q_teacher, reduction='batchmean')
        else:
            loss_kd = f_global.new_zeros(())

        return (comp_logits, attr_logits, obj_logits,
                loss_contrastive, loss_hsic, loss_hard, loss_cf, loss_rdrop, loss_mcm, loss_kd)

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
            logit_i = logit_scale * norm_img[i] @ feat.t()
            if i == 0 and self.use_pglpr:
                logit_i = logit_i + self._llm_bias(norm_img[0], idx)
            logits.append(logit_i)

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
        # Round-7 calibration regularizers (both gated; baseline-identical when off).
        # Label smoothing: COMP CE only (attr/obj feed the fusion as softmax probs; smoothing
        # them would perturb the attr*obj fusion calibration). LogitNorm: COMP CE only.
        label_smoothing = float(getattr(self.config, 'label_smoothing', 0.0))
        logit_norm_tau = float(getattr(self.config, 'logit_norm_tau', 0.0))
        loss_fn = nn.CrossEntropyLoss()                                   # attr/obj (plain)
        comp_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)  # comp (smoothed if >0)
        batch_attr, batch_obj, batch_target = target[1], target[2], target[3]
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()

        if self.training:
            (comp_logits, attr_logits, obj_logits,
             loss_contras, loss_hsic, loss_hard, loss_cf, loss_rdrop, loss_mcm, loss_kd) = predict
        else:
            comp_logits, attr_logits, obj_logits = predict

        m = float(getattr(self.config, 'cosface_margin', 0.0))
        if self.training and m > 0:
            scale = self.clip.logit_scale.exp().detach()
            comp_logits = self._cosface_margin(comp_logits, batch_target, m, scale)
            attr_logits = self._cosface_margin(attr_logits, batch_attr, m, scale)
            obj_logits = self._cosface_margin(obj_logits, batch_obj, m, scale)

        # COMP logit fed to the CE. LogitNorm (training only) replaces the scaled cosine with
        # an L2-normalized cosine / tau. Inference (logit_infer/val_forward) is UNTOUCHED and
        # still uses logit_scale*cosine, so inference ranking is unchanged.
        comp_logit_for_ce = comp_logits
        if logit_norm_tau > 0 and self.training:
            assert not self.use_pglpr, "LogitNorm assumes comp_logits = logit_scale*cosine (no pglpr bias)."
            logit_scale = self.clip.logit_scale.exp()
            s = comp_logits / logit_scale                                 # recover raw cosine
            comp_logit_for_ce = s / (s.norm(dim=-1, keepdim=True) + 1e-7) / logit_norm_tau

        loss = (
            self.pair_loss_weight * comp_loss_fn(comp_logit_for_ce, batch_target) +
            self.attr_loss_weight * loss_fn(attr_logits, batch_attr) +
            self.obj_loss_weight * loss_fn(obj_logits, batch_obj)
        )

        if self.training:
            loss = loss + self.contrastive_weight * loss_contras + self.hsic_weight * loss_hsic
            hpw = float(getattr(self.config, 'hard_pair_weight', 0.0))
            if hpw > 0:
                loss = loss + hpw * loss_hard
            if self.use_pglpr:
                warmup = min(1.0, max(0.0, self.current_epoch / max(1, self.pglpr_warmup_epochs)))
                gate_l1 = torch.sigmoid(self.g_attr).mean() + torch.sigmoid(self.g_obj).mean()
                loss = loss + warmup * self.lambda_gate * gate_l1
            # Rank-3 counterfactual-consistency (PART B): guarded by cf_weight>0.
            cfw = float(getattr(self.config, 'cf_weight', 0.0))
            if cfw > 0:
                cf_warm = min(1.0, self.current_epoch / max(1, int(getattr(self.config, 'cf_warmup_epochs', 3))))
                loss = loss + cf_warm * cfw * loss_cf
            # Round-8 R-Drop comp-consistency: guarded by rdrop_weight>0.
            rdw = float(getattr(self.config, 'rdrop_weight', 0.0))
            if rdw > 0:
                loss = loss + rdw * loss_rdrop
            # Round-9 Masked Composition Modeling: guarded by mcm_weight>0.
            mcw = float(getattr(self.config, 'mcm_weight', 0.0))
            if mcw > 0:
                loss = loss + mcw * loss_mcm
            # Round-10 comp-only KD: guarded by kd_weight>0, linear warmup.
            kdw = float(getattr(self.config, 'kd_weight', 0.0))
            if kdw > 0:
                kd_warm = min(1.0, self.current_epoch / max(1, int(getattr(self.config, 'kd_warmup_epochs', 3))))
                loss = loss + kd_warm * kdw * loss_kd

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
