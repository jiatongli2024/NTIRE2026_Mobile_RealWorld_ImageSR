"""
IDaS-ESR — inference engine & challenge entry point
=============================================================
Implements:
  • IDaSESR   — one-step SR model (loads checkpoint, runs inference)
  • main()   — NTIRE 2026 challenge harness entry point

Architecture lives in model.py.
Diffusion weights (VAE / UNet / scheduler) are loaded from SD 2.1-base.
Our weights live in a single .pkl checkpoint (see README.md).
"""

import os
import glob

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig

from .model import DiT_Unified_NoisePredictor, adain_color_fix


# ===========================================================================
# Fixed inference hyper-parameters
# ===========================================================================
_NUM_LEARNABLE_TOKENS = 4
_RESTORATION_STRENGTH = 0.6   # 0 = strict fidelity · 1 = max texture
_MAX_NOISE_RATIO      = 1.0
_PROCESS_SIZE         = 512   # minimum output long-side (pixels)
_UPSCALE              = 4

# Weight dtype: fp32 to match --mixed_precision fp32 in test_dit_unified_mc.sh
_WEIGHT_DTYPE = torch.float32

# MultiDiffusion hyper-parameters
_MD_TILE_SIZE    = 64   # latent pixels (= 512 px image)
_MD_TILE_OVERLAP = 16   # latent pixels (25 % overlap — per paper stride 32)
_MD_BATCH_SIZE   = 4


# ===========================================================================
# MultiDiffusion tile utilities
# (ported verbatim from osediff_inv.py :: _compute_multidiff_tiles /
#  _make_multidiff_weights)
# ===========================================================================

def _compute_multidiff_tiles(h, w, tile_size, tile_stride):
    """Compute overlapping tile coordinates covering an (h, w) latent region.

    Returns:
        List of (y0, y1, x0, x1) tuples.  Edge tiles are snapped to borders.
    """
    coords = []
    y = 0
    while True:
        y0 = y
        y1 = y0 + tile_size
        if y1 >= h:
            y0 = max(0, h - tile_size)
            y1 = h
        x = 0
        while True:
            x0 = x
            x1 = x0 + tile_size
            if x1 >= w:
                x0 = max(0, w - tile_size)
                x1 = w
            if (y0, y1, x0, x1) not in coords:
                coords.append((y0, y1, x0, x1))
            if x1 >= w:
                break
            x += tile_stride
        if y1 >= h:
            break
        y += tile_stride
    return coords


def _make_multidiff_weights(tile_h, tile_w, num_channels, device):
    """Generate a Gaussian blending mask for MultiDiffusion tile fusion.

    Returns:
        Tensor of shape (1, num_channels, tile_h, tile_w).
    """
    from numpy import pi, exp, sqrt
    import numpy as np

    var = 0.01
    midpoint_x = (tile_w - 1) / 2
    x_probs = [exp(-(x - midpoint_x) ** 2 / (tile_w ** 2) / (2 * var)) / sqrt(2 * pi * var)
               for x in range(tile_w)]
    midpoint_y = (tile_h - 1) / 2
    y_probs = [exp(-(y - midpoint_y) ** 2 / (tile_h ** 2) / (2 * var)) / sqrt(2 * pi * var)
               for y in range(tile_h)]
    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=device, dtype=torch.float32),
                      (1, num_channels, 1, 1))


# ===========================================================================
# IDaSESR — inference engine
# ===========================================================================

class IDaSESR(torch.nn.Module):
    """
    One-step SR with Unified DiT + Manifold Coverage.

    Pipeline:
      1.  VAE encode x_lq                              → z_lq
      2.  DiT(z_lq, t_max, prompt_embeds)              → ε_pred, τ, cond_emb
      3.  aug_embeds = prompt_embeds + cond_emb
      4.  Manifold Coverage noise mix                  → ε_use, t_start
      5.  x_t = √ᾱ_t · z_lq + √(1-ᾱ_t) · ε_use
      6.  UNet(x_t, t_start, aug_embeds)               → model_pred
      7.  x_0 = (x_t − √(1-ᾱ)·model_pred) / √ᾱ
      8.  VAE decode x_0                               → SR image
    """

    def __init__(self, model_path: str, sd21_base: str, cached_embeds_path: str,
                 restoration_strength: float = 0.6,
                 max_noise_ratio: float = 1.0,
                 num_learnable_tokens: int = 4,
                 weight_dtype: torch.dtype = torch.float16,
                 device: torch.device = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_dtype = weight_dtype
        self.restoration_strength = restoration_strength
        self.max_noise_ratio = max_noise_ratio

        # ── Noise scheduler ──────────────────────────────────────────────
        self.scheduler = DDPMScheduler.from_pretrained(
            sd21_base, subfolder='scheduler', local_files_only=True)
        self.scheduler.set_timesteps(1000, device=self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # ── VAE ──────────────────────────────────────────────────────────
        self.vae = AutoencoderKL.from_pretrained(
            sd21_base, subfolder='vae', local_files_only=True)

        # ── UNet ─────────────────────────────────────────────────────────
        self.unet = UNet2DConditionModel.from_pretrained(
            sd21_base, subfolder='unet', local_files_only=True)

        # ── Cached fixed prompt embeddings (no CLIP at inference) ─────────
        data = torch.load(cached_embeds_path, map_location='cpu', weights_only=False)
        raw_embeds = data['prompt_embeds'] if isinstance(data, dict) else data
        if isinstance(data, dict):
            print(f"[IDaSSR] Prompt cache: '{data.get('prompt', '?')}' "
                  f"shape={list(raw_embeds.shape)}")
        self._prompt_embeds: torch.Tensor = raw_embeds   # (1, seq_len, 1024)

        # ── DiT-S/2-Unified ──────────────────────────────────────────────
        self.dit = DiT_Unified_NoisePredictor(
            input_size=64, in_channels=4, hidden_size=384, depth=12, num_heads=6,
            patch_size=2, learn_sigma=False, prompt_hidden_size=1024,
            z_dims=[], projector_dim=2048, encoder_depth=4,
            num_learnable_tokens=num_learnable_tokens,
            condition_output_dim=1024, time_range=(50, 450),
        )

        # ── Load checkpoint ───────────────────────────────────────────────
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        self._load_ckpt(ckpt)

        # ── Move to device / dtype ────────────────────────────────────────
        self.vae.to(self.device, dtype=weight_dtype)
        self.unet.to(self.device, dtype=weight_dtype)
        self.dit.to(self.device, dtype=weight_dtype)
        self.scheduler.alphas_cumprod = \
            self.scheduler.alphas_cumprod.to(self.device, dtype=torch.float32)

        # ── Tiled VAE — matches OSEDiff_DiT_test._init_tiled_vae() ───────
        # encoder_tile_size=1024 px: does not fire for ≤512 px inputs
        # decoder_tile_size= 224 px: tiles the decoder for memory safety
        self.vae.enable_tiling()

        self.eval()

    # ------------------------------------------------------------------
    def _load_ckpt(self, ckpt: dict):
        """Apply LoRA adapters and load DiT weights from .pkl checkpoint."""

        # UNet LoRA (encoder / decoder / other modules)
        for name, modules in [
            ('default_encoder', ckpt['unet_lora_encoder_modules']),
            ('default_decoder', ckpt['unet_lora_decoder_modules']),
            ('default_others',  ckpt['unet_lora_others_modules']),
        ]:
            self.unet.add_adapter(
                LoraConfig(r=ckpt['rank_unet'], init_lora_weights='gaussian',
                           target_modules=modules),
                adapter_name=name,
            )
        for n, p in self.unet.named_parameters():
            if 'lora' in n or 'conv_in' in n:
                p.data.copy_(ckpt['state_dict_unet'][n])
        self.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

        # VAE encoder LoRA
        self.vae.add_adapter(
            LoraConfig(r=ckpt['rank_vae'], init_lora_weights='gaussian',
                       target_modules=ckpt['vae_lora_encoder_modules']),
            adapter_name='default_encoder',
        )
        for n, p in self.vae.named_parameters():
            if 'lora' in n:
                p.data.copy_(ckpt['state_dict_vae'][n])
        self.vae.set_adapter(['default_encoder'])

        # DiT weights
        if 'state_dict_dit' in ckpt:
            try:
                self.dit.load_state_dict(ckpt['state_dict_dit'], strict=True)
            except RuntimeError as e:
                print(f'[IDaSESR] Non-strict DiT load: {e}')
                self.dit.load_state_dict(ckpt['state_dict_dit'], strict=False)
            print('[IDaSESR] Loaded Unified DiT checkpoint.')
        else:
            print('[IDaSESR] Warning: no DiT weights found in checkpoint.')

    # ------------------------------------------------------------------
    def _get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate α̅_t from the DDPM schedule table."""
        alphas = self.scheduler.alphas_cumprod.to(device=t.device, dtype=torch.float32)
        steps  = torch.arange(1000, device=t.device, dtype=torch.float32)
        t      = torch.clamp(t, steps.min(), steps.max())
        idx    = torch.searchsorted(steps, t).clamp(1, 999)
        w      = (t - steps[idx - 1]) / (steps[idx] - steps[idx - 1] + 1e-8)
        alpha  = alphas[idx - 1] + w * (alphas[idx] - alphas[idx - 1])
        while alpha.dim() < 4:
            alpha = alpha.unsqueeze(-1)
        return alpha

    # ------------------------------------------------------------------
    def _manifold_coverage(self, eps_pred, pred_t):
        """Apply Manifold Coverage noise mixing.

        Returns (eps_use, t_start) — the mixed noise and extended timestep.
        Matches osediff_inv.py :: forward() unified + use_manifold_coverage path.
        """
        s     = self.restoration_strength
        gamma = s * self.max_noise_ratio
        if gamma <= 0.0:
            return eps_pred, pred_t

        eps_rand   = torch.randn_like(eps_pred)
        pred_std   = eps_pred.std(dim=(1, 2, 3), keepdim=True)
        target_std = (1.0 - gamma) * pred_std + gamma * 1.0
        eps_raw    = (1.0 - gamma) * eps_pred + gamma * eps_rand
        eps_use    = eps_raw * (target_std / (eps_raw.std(dim=(1, 2, 3), keepdim=True) + 1e-8))
        t_start    = torch.clamp(pred_t + s * (float(self.dit.time_max) - pred_t), 0, 999)
        return eps_use, t_start

    # ------------------------------------------------------------------
    def _forward_diffusion(self, z, eps, t):
        """x_t = √ᾱ_t · z + √(1−ᾱ_t) · ε  (continuous-time)."""
        alpha = self._get_alpha(t).to(torch.float32)
        return (torch.sqrt(alpha) * z.float()
                + torch.sqrt(1.0 - alpha) * eps.float())

    # ------------------------------------------------------------------
    def _denoise_step(self, model_pred, t, x_t):
        """x_0 = (x_t − √(1−ᾱ)·ε) / √ᾱ  (one-step continuous-time DDPM)."""
        alpha = self._get_alpha(t).to(model_pred.dtype)
        return (x_t.to(model_pred.dtype) - torch.sqrt(1.0 - alpha) * model_pred) \
               / (torch.sqrt(alpha) + 1e-8)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x_lq: torch.Tensor):
        """
        Args:
            x_lq : (B, 3, H, W) in [-1, 1]  — pre-upscaled LR image.
        Returns:
            output : (B, 3, H, W) in [-1, 1]
            pred_t : scalar (mean predicted timestep, for logging)
        """
        B, dtype = x_lq.shape[0], self.weight_dtype

        # 1. VAE encode
        z_lq = (self.vae.encode(x_lq.to(dtype)).latent_dist.sample()
                * self.vae.config.scaling_factor)

        # 2. Fixed DiT input timestep (= time_max for adaLN conditioning)
        t_fixed = torch.full((B,), int(self.dit.time_max),
                             device=self.device, dtype=torch.long)

        # 3. Cached prompt embeddings → broadcast to batch
        prompt_embeds = (self._prompt_embeds
                         .to(device=self.device, dtype=dtype)
                         .expand(B, -1, -1))

        # 4. DiT forward: noise, timestep, condition
        eps_pred, _, _, pred_t, cond_emb = self.dit(
            z_lq, t_fixed, prompt_embeds=prompt_embeds
        )

        # 5. Augment UNet cross-attention conditioning
        aug_embeds = prompt_embeds + cond_emb.unsqueeze(1)   # (B, seq, 1024)

        # 6. Manifold Coverage — mix eps and extend effective timestep
        eps_use, t_start = self._manifold_coverage(eps_pred, pred_t)

        # 7. Forward diffusion
        x_t = self._forward_diffusion(z_lq, eps_use, t_start).to(dtype)

        # 8. UNet denoising step
        model_pred = self.unet(x_t, t_start,
                               encoder_hidden_states=aug_embeds.to(dtype),
                               encoder_attention_mask=None).sample

        # 9. One-step DDPM decode
        x_denoised = self._denoise_step(model_pred, t_start, x_t)

        # 10. VAE decode
        output = self.vae.decode(
            x_denoised.to(dtype) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        return output, pred_t.mean().item()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_multidiffusion(self, x_lq: torch.Tensor,
                               md_tile_size: int = 64,
                               md_tile_overlap: int = 32,
                               md_batch_size: int = 4):
        """MultiDiffusion inference: overlapping per-tile DiT+UNet, fused with
        Gaussian blending weights.  Falls back to ``forward()`` when the latent
        fits in a single tile.

        Matches ``OSEDiff_DiT_test.forward_multidiffusion()`` (osediff_inv.py:2596)
        for the ``timestep_pred_mode='unified'`` + ``use_manifold_coverage=True``
        path, simplified to only that branch.

        Args:
            x_lq            : (1, 3, H, W) in [-1, 1] — pre-upscaled LR image.
            md_tile_size    : Tile side in latent pixels (must be even). 64 = 512 px.
            md_tile_overlap : Overlap in latent pixels. 32 = 256 px (50 %).
            md_batch_size   : Max tiles per GPU forward pass.

        Returns:
            (output_image, predicted_timestep)  — same signature as ``forward()``.
        """
        assert md_tile_size % 2 == 0, (
            f"md_tile_size must be even (DiT patch_size=2), got {md_tile_size}")
        assert x_lq.shape[0] == 1, "forward_multidiffusion only supports batch_size=1"

        dtype = self.weight_dtype

        # ── 1. Prompt embeddings (global, once) ──────────────────────────
        prompt_embeds = (self._prompt_embeds
                         .to(device=self.device, dtype=dtype)
                         .expand(1, -1, -1))   # (1, seq, 1024)

        # ── 2. VAE encode (global) ────────────────────────────────────────
        lq_latent = (self.vae.encode(x_lq.to(dtype)).latent_dist.sample()
                     * self.vae.config.scaling_factor)
        _, C, h, w = lq_latent.shape
        tile_stride = md_tile_size - md_tile_overlap

        # ── 3. Fallback: latent fits in one tile ──────────────────────────
        if h <= md_tile_size and w <= md_tile_size:
            return self.forward(x_lq)

        print(f"[MultiDiffusion]: latent {h}x{w}, tile {md_tile_size}, "
              f"overlap {md_tile_overlap}, stride {tile_stride}")

        # ── 4. Tile coordinates & Gaussian blending weights ───────────────
        tile_coords  = _compute_multidiff_tiles(h, w, md_tile_size, tile_stride)
        tile_weights = _make_multidiff_weights(md_tile_size, md_tile_size, C, lq_latent.device)

        # ── 5. Accumulators ───────────────────────────────────────────────
        denoised_accum = torch.zeros_like(lq_latent)
        weight_accum   = torch.zeros(1, 1, h, w, device=lq_latent.device, dtype=torch.float32)
        all_pred_ts    = []

        t_fixed_val = int(self.dit.time_max)

        # ── 6. Process tiles in batches ───────────────────────────────────
        for batch_start in range(0, len(tile_coords), md_batch_size):
            batch_coords = tile_coords[batch_start:batch_start + md_batch_size]
            B_tile = len(batch_coords)

            # 6a. Extract lq_latent tiles → (B_tile, C, ts, ts)
            lq_tiles = torch.stack([
                lq_latent[0, :, y0:y1, x0:x1] for y0, y1, x0, x1 in batch_coords
            ], dim=0)

            # 6b. Expand prompt_embeds for tile batch
            tile_prompt_embeds = prompt_embeds.expand(B_tile, -1, -1)

            # 6c. DiT forward — per-tile local conditioning
            t_fixed = torch.tensor([t_fixed_val] * B_tile, device=self.device).long()
            eps_pred, _, _, pred_t, cond_emb = self.dit(
                lq_tiles, t_fixed, prompt_embeds=tile_prompt_embeds
            )

            # 6d. Augmented UNet conditioning (prompt + per-tile cond_emb)
            aug_embeds = tile_prompt_embeds + cond_emb.unsqueeze(1)   # (B_tile, seq, 1024)
            all_pred_ts.append(pred_t.detach().cpu())

            # 6e. Manifold Coverage — per-tile noise mixing + timestep extension
            eps_use, t_start = self._manifold_coverage(eps_pred, pred_t)

            # 6f. Forward diffusion per-tile
            x_t = self._forward_diffusion(lq_tiles, eps_use, t_start).to(dtype)

            # 6g. UNet denoising per-tile
            model_pred = self.unet(
                x_t, t_start,
                encoder_hidden_states=aug_embeds.to(dtype),
                encoder_attention_mask=None,
            ).sample

            # 6h. One-step denoise per-tile
            x_denoised_tiles = self._denoise_step(model_pred, t_start, x_t)

            # 6i. Accumulate with Gaussian blending weights
            for i, (y0, y1, x0, x1) in enumerate(batch_coords):
                th, tw = y1 - y0, x1 - x0
                tw_mask = tile_weights[:, :, :th, :tw]
                denoised_accum[:, :, y0:y1, x0:x1] += x_denoised_tiles[i:i+1] * tw_mask
                weight_accum[:, :, y0:y1, x0:x1]   += tw_mask[:, :1, :, :]

        # ── 7. Fuse tiles ─────────────────────────────────────────────────
        x_denoised = denoised_accum / weight_accum.clamp(min=1e-8)

        # ── 8. VAE decode ─────────────────────────────────────────────────
        output = self.vae.decode(
            x_denoised.to(dtype) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        predicted_t = torch.cat(all_pred_ts).mean().item() if all_pred_ts else 0.0
        return output, predicted_t


# ===========================================================================
# Challenge entry point  (called by NTIRE2026 test.py harness)
# ===========================================================================

def main(model_dir: str, input_path: str, output_path: str,
         device: torch.device = None):
    """
    Args:
        model_dir   : path to the IDaS-SR-efficient .pkl checkpoint
        input_path  : directory (or single file) of LR input images
        output_path : directory where SR outputs will be written
        device      : torch.device provided by the challenge harness
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(output_path, exist_ok=True)

    model_zoo_dir = os.path.dirname(model_dir)
    sd21_base = (os.environ.get("IDaSESR_SD21_BASE_PATH")
                 or os.path.join(model_zoo_dir, 'stable-diffusion-2-1-base'))
    cached_embeds_path = os.path.join(model_zoo_dir, 'fixed_prompt_embeds_more-fp32.pt')

    model = IDaSESR(
        model_path=model_dir,
        sd21_base=sd21_base,
        cached_embeds_path=cached_embeds_path,
        restoration_strength=_RESTORATION_STRENGTH,
        max_noise_ratio=_MAX_NOISE_RATIO,
        num_learnable_tokens=_NUM_LEARNABLE_TOKENS,
        weight_dtype=_WEIGHT_DTYPE,
        device=device,
    )

    to_tensor = transforms.ToTensor()
    min_side  = _PROCESS_SIZE // _UPSCALE

    # Collect images
    if os.path.isdir(input_path):
        image_paths = sorted(
            glob.glob(os.path.join(input_path, '*.png'))  +
            glob.glob(os.path.join(input_path, '*.jpg'))  +
            glob.glob(os.path.join(input_path, '*.jpeg')) +
            glob.glob(os.path.join(input_path, '*.bmp'))
        )
    else:
        image_paths = [input_path]

    print(f'[IDaS-ESR] {len(image_paths)} image(s) to process.')

    for img_path in image_paths:
        pil_img = Image.open(img_path).convert('RGB')
        ori_w, ori_h = pil_img.size

        # Ensure minimum LR side length before upscaling
        resize_flag = False
        if ori_w < min_side or ori_h < min_side:
            scale = min_side / min(ori_w, ori_h)
            pil_img = pil_img.resize(
                (int(scale * ori_w), int(scale * ori_h)), Image.LANCZOS
            )
            resize_flag = True

        # Bicubic pre-upscale LR → HR resolution
        pil_img = pil_img.resize(
            (pil_img.width * _UPSCALE, pil_img.height * _UPSCALE), Image.LANCZOS
        )
        # Pad to multiple of 16 (required by VAE)
        pil_img = pil_img.resize(
            (pil_img.width - pil_img.width % 16,
             pil_img.height - pil_img.height % 16),
            Image.LANCZOS,
        )

        lq = to_tensor(pil_img).unsqueeze(0).to(device)
        output_image, t_pred = model.forward_multidiffusion(
            lq * 2.0 - 1.0,
            md_tile_size=_MD_TILE_SIZE,
            md_tile_overlap=_MD_TILE_OVERLAP,
            md_batch_size=_MD_BATCH_SIZE,
        )

        out_pil = ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        out_pil = adain_color_fix(target=out_pil, source=pil_img)

        if resize_flag:
            out_pil = out_pil.resize((_UPSCALE * ori_w, _UPSCALE * ori_h), Image.LANCZOS)

        bname = os.path.basename(img_path)
        out_pil.save(os.path.join(output_path, bname))
        print(f'  τ={t_pred:.1f}  →  {bname}')

    print(f'[IDaS-ESR] Done. Results in: {os.path.abspath(output_path)}')
