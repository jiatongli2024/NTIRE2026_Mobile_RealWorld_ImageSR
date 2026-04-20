"""
IDaS-ESR — model architecture
=======================================
Contains:
  • Sin/cos 2-D positional embedding utilities 
  • DiT building blocks                          (from facebookresearch/DiT)
  • DiT, DiT_REPA, DiT_Unified_NoisePredictor
  • AdaIN colour-correction utility 

No diffusion weights are defined here; all weights are loaded in io.py.
External deps: torch, numpy, timm
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


# ===========================================================================
# Sin/Cos 2-D Positional Embedding
# ===========================================================================

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int,
                             cls_token: bool = False,
                             extra_tokens: int = 0) -> np.ndarray:
    """Return (grid_size²[+extra_tokens], embed_dim) sin/cos position embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ===========================================================================
# DiT building blocks
# ===========================================================================

def build_mlp(hidden_size: int, projector_dim: int, z_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if hasattr(self.mlp[0], 'weight'):
            t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Class-label embedder kept for DiT base-class compatibility."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + int(dropout_prob > 0), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels: torch.Tensor, train: bool) -> torch.Tensor:
        if train and self.dropout_prob > 0:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop, self.num_classes, labels)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden,
                       act_layer=lambda: nn.GELU(approximate='tanh'), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final adaLN + linear projection layer of DiT."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int,
                 cls_token_dim: int = None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.linear_cls = (nn.Linear(hidden_size, cls_token_dim, bias=True)
                           if cls_token_dim is not None else None)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, cls: bool = False):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        if cls:
            return self.linear(x[:, 1:]), self.linear_cls(x[:, 0])
        return self.linear(x)


# ===========================================================================
# DiT backbone
# ===========================================================================

class DiT(nn.Module):
    """Diffusion Transformer backbone (Peebles & Xie, 2023)."""

    def __init__(self, input_size=32, patch_size=2, in_channels=4,
                 hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0,
                 class_dropout_prob=0.1, num_classes=1000, learn_sigma=True):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels,
                                     hidden_size, bias=True, strict_img_size=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size),
                                      requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h: int = None, w: int = None) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if h is None or w is None:
            h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def interpolate_pos_embed(self, num_patches: int) -> torch.Tensor:
        embed_dim = self.pos_embed.shape[-1]
        orig_n = self.pos_embed.shape[1]
        orig_size = int(orig_n ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size ** 2 == orig_n and new_size ** 2 == num_patches:
            pos = self.pos_embed.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(new_size, new_size), mode='bicubic', align_corners=False)
            return pos.permute(0, 2, 3, 1).reshape(1, num_patches, embed_dim)
        return F.interpolate(
            self.pos_embed.transpose(1, 2), size=num_patches, mode='linear', align_corners=False
        ).transpose(1, 2)


# ===========================================================================
# DiT_REPA — adds prompt conditioning + optional REPA projectors
# ===========================================================================

class DiT_REPA(DiT):
    """DiT with prompt conditioning and REPA projection hooks."""

    def __init__(self, z_dims=None, projector_dim=2048, encoder_depth=4,
                 prompt_hidden_size=None, use_prompt_condition=True, **kwargs):
        super().__init__(**kwargs)
        self.z_dims = z_dims or []
        self.encoder_depth = encoder_depth
        self.use_prompt_condition = use_prompt_condition

        hidden_size = self.x_embedder.proj.out_channels
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z) for z in self.z_dims
        ])
        if use_prompt_condition and prompt_hidden_size is not None:
            self.prompt_projector = nn.Linear(prompt_hidden_size, hidden_size, bias=True)
            nn.init.normal_(self.prompt_projector.weight, std=0.01)
            nn.init.zeros_(self.prompt_projector.bias)
            for p in self.y_embedder.parameters():
                p.requires_grad = False
        else:
            self.prompt_projector = None

    def _condition(self, t: torch.Tensor,
                   prompt_embeds: torch.Tensor = None,
                   y: torch.Tensor = None) -> torch.Tensor:
        t_emb = self.t_embedder(t)
        if self.use_prompt_condition and prompt_embeds is not None \
                and self.prompt_projector is not None:
            return t_emb + self.prompt_projector(torch.mean(prompt_embeds, dim=1))
        return t_emb + self.y_embedder(y, self.training)

    def forward(self, x, t, y=None, prompt_embeds=None):
        x = self.x_embedder(x)
        pos = (self.interpolate_pos_embed(x.shape[1])
               if x.shape[1] != self.pos_embed.shape[1] else self.pos_embed)
        x = x + pos
        N, T, D = x.shape
        c = self._condition(t, prompt_embeds, y)
        zs = []
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) == self.encoder_depth and self.projectors:
                zs = [p(x.reshape(-1, D)).reshape(N, T, -1) for p in self.projectors]
        return self.unpatchify(self.final_layer(x, c)), zs


# ===========================================================================
# DiT_Unified_NoisePredictor  (our primary model)
# Jointly predicts: noise ε, continuous timestep τ, UNet condition embedding
# ===========================================================================

class DiT_Unified_NoisePredictor(DiT_REPA):
    """
    Prepends K learnable tokens to the image patch sequence.
    After self-attention, they produce a predicted timestep and condition
    embedding; image tokens produce the noise prediction.

    forward() returns:
        noise_pred       (N, 4, H, W)
        zs_sem           list of semantic REPA projections  (train only)
        zs_deg           list of degradation REPA projections (train only)
        predicted_t      (N,) continuous timestep in [time_min, time_max]
        condition_emb    (N, condition_output_dim)
    """

    def __init__(self, input_size=64, in_channels=4, hidden_size=384, depth=12,
                 num_heads=6, patch_size=2, mlp_ratio=4.0, class_dropout_prob=0.0,
                 num_classes=1, learn_sigma=False, prompt_hidden_size=1024,
                 z_dims=None, projector_dim=2048, encoder_depth=4,
                 num_learnable_tokens=4, condition_output_dim=1024,
                 time_range=(50, 450), **kwargs):
        super().__init__(
            input_size=input_size, in_channels=in_channels, hidden_size=hidden_size,
            depth=depth, num_heads=num_heads, patch_size=patch_size,
            mlp_ratio=mlp_ratio, class_dropout_prob=class_dropout_prob,
            num_classes=num_classes, learn_sigma=learn_sigma,
            prompt_hidden_size=prompt_hidden_size, z_dims=z_dims or [],
            projector_dim=projector_dim, encoder_depth=encoder_depth,
            use_prompt_condition=True, **kwargs
        )
        self.num_learnable_tokens = num_learnable_tokens
        self.condition_output_dim = condition_output_dim
        self.time_min, self.time_max = time_range
        self.register_buffer('time_min_tensor', torch.tensor(float(self.time_min)))
        self.register_buffer('time_max_tensor', torch.tensor(float(self.time_max)))

        D = self.x_embedder.proj.out_channels
        self.learnable_tokens    = nn.Parameter(torch.randn(1, num_learnable_tokens, D) * 0.02)
        self.learnable_pos_embed = nn.Parameter(torch.randn(1, num_learnable_tokens, D) * 0.02)

        self.timestep_head = nn.Sequential(
            nn.LayerNorm(D, elementwise_affine=True, eps=1e-6),
            nn.Linear(D, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 1),
        )
        self.condition_head = nn.Sequential(
            nn.LayerNorm(D, elementwise_affine=True, eps=1e-6),
            nn.Linear(D, condition_output_dim),
        )
        # Dual REPA projectors: semantic (image tokens) + degradation (learnable tokens)
        self.projectors_sem = nn.ModuleList([build_mlp(D, projector_dim, z) for z in (z_dims or [])])
        self.projectors_deg = nn.ModuleList([build_mlp(D, projector_dim, z) for z in (z_dims or [])])
        self._init_prediction_heads()

    def _init_prediction_heads(self):
        for module in [self.timestep_head, self.condition_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y=None, prompt_embeds=None):
        _H, _W = x.shape[2], x.shape[3]
        x = self.x_embedder(x)
        N, T, D = x.shape

        pos = (self.interpolate_pos_embed(T)
               if T != self.pos_embed.shape[1] else self.pos_embed)

        K = self.num_learnable_tokens
        lrn     = self.learnable_tokens.expand(N, -1, -1)
        lrn_pos = self.learnable_pos_embed.expand(N, -1, -1)
        x = torch.cat([lrn, x], dim=1) \
            + torch.cat([lrn_pos, pos.expand(N, -1, -1)], dim=1)

        c = self._condition(t, prompt_embeds, y)

        zs_sem, zs_deg = [], []
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) == self.encoder_depth and self.projectors_sem:
                img_tok = x[:, K:, :]
                deg_tok = x[:, :K, :]
                zs_sem = [p(img_tok.reshape(-1, D)).reshape(N, T, -1) for p in self.projectors_sem]
                zs_deg = [p(deg_tok.reshape(-1, D)).reshape(N, K, -1) for p in self.projectors_deg]

        lrn_out = x[:, :K, :]
        img_out = x[:, K:, :]

        # Noise prediction (image tokens)
        shift, scale = self.final_layer.adaLN_modulation(c).chunk(2, dim=1)
        img_out = modulate(self.final_layer.norm_final(img_out), shift, scale)
        img_out = self.final_layer.linear(img_out)
        p = self.x_embedder.patch_size[0]
        noise_pred = self.unpatchify(img_out, _H // p, _W // p)

        # Timestep + condition embedding (learnable tokens)
        lrn_out = modulate(self.final_layer.norm_final(lrn_out), shift, scale)
        pooled  = lrn_out.mean(dim=1)
        pred_t  = (torch.sigmoid(self.timestep_head(pooled).squeeze(-1))
                   * (self.time_max_tensor - self.time_min_tensor) + self.time_min_tensor)
        cond_emb = self.condition_head(pooled)

        return noise_pred, zs_sem, zs_deg, pred_t, cond_emb


# ===========================================================================
# AdaIN colour correction
# (ported from sd-webui-stablesr / Li Yi; original licence retained)
# ===========================================================================

def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    b, c = feat.shape[:2]
    var  = feat.reshape(b, c, -1).var(dim=2) + eps
    std  = var.sqrt().reshape(b, c, 1, 1)
    mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return mean, std


def adain_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """Transfer colour statistics from *source* to *target* via AdaIN."""
    to_t = ToTensor()
    t_t, s_t = to_t(target).unsqueeze(0), to_t(source).unsqueeze(0)
    s_mean, s_std = _calc_mean_std(s_t)
    c_mean, c_std = _calc_mean_std(t_t)
    out = (t_t - c_mean.expand_as(t_t)) / c_std.expand_as(t_t) \
          * s_std.expand_as(t_t) + s_mean.expand_as(t_t)
    return ToPILImage()(out.squeeze(0).clamp_(0.0, 1.0))
