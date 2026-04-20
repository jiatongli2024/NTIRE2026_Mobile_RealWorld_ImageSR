import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import PeftModel


class IMAG2006(torch.nn.Module):
    def __init__(self, sd_path, lora_path, mid_timestep, device, weight_dtype):
        super().__init__()
        self.device = device
        self.mid_timestep = int(mid_timestep)
        self.weight_dtype = weight_dtype

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.alpha_t = self.scheduler.alphas_cumprod[self.mid_timestep]

        vae.encoder = PeftModel.from_pretrained(
            vae.encoder,
            os.path.join(lora_path, "vae_encoder_lora_adapter"),
        ).merge_and_unload()

        unet = PeftModel.from_pretrained(
            unet,
            os.path.join(lora_path, "unet_lora_adapter"),
        ).merge_and_unload()

        vae = vae.to(device=self.device, dtype=weight_dtype)
        unet = unet.to(device=self.device, dtype=weight_dtype)

        vae.eval()
        unet.eval()

        self.vae = vae
        self.unet = unet

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height
        var = 0.01

        midpoint_x = (latent_width - 1) / 2
        x_probs = [
            exp(-(x - midpoint_x) * (x - midpoint_x) / (latent_width * latent_width) / (2 * var))
            / sqrt(2 * pi * var)
            for x in range(latent_width)
        ]

        midpoint_y = latent_height / 2
        y_probs = [
            exp(-(y - midpoint_y) * (y - midpoint_y) / (latent_height * latent_height) / (2 * var))
            / sqrt(2 * pi * var)
            for y in range(latent_height)
        ]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(
            torch.tensor(weights, device=self.device),
            (nbatches, self.unet.config.in_channels, 1, 1),
        )

    def _forward_no_tile(self, lq_latent, prompt_embeds):
        model_pred = self.unet(
            lq_latent.to(self.unet.dtype),
            self.mid_timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample

        denoised_latent = (
            lq_latent - (1 - self.alpha_t).sqrt() * model_pred.to(self.vae.dtype)
        ) / self.alpha_t.sqrt()

        pred_img = self.vae.decode(
            denoised_latent / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        return pred_img

    def _forward_tile(self, lq_latent, prompt_embeds, tile_size, tile_overlap):
        _, _, h, w = lq_latent.shape
        tile_size = min(tile_size, min(h, w))
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        prompt_embeds = prompt_embeds.to(self.unet.dtype)

        grid_rows = 0
        cur_x = 0
        while cur_x < w:
            cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < h:
            cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
            grid_cols += 1

        noise_preds = []
        coords = []

        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

                model_out = self.unet(
                    input_tile.to(self.unet.dtype),
                    self.mid_timestep,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                noise_preds.append(model_out)
                coords.append((input_start_y, input_end_y, input_start_x, input_end_x))

        noise_pred = torch.zeros_like(lq_latent)
        contributors = torch.zeros_like(lq_latent)

        for model_out, (ys, ye, xs, xe) in zip(noise_preds, coords):
            noise_pred[:, :, ys:ye, xs:xe] += model_out * tile_weights
            contributors[:, :, ys:ye, xs:xe] += tile_weights

        noise_pred = noise_pred / contributors.clamp_min(1e-8)

        denoised_latent = (
            lq_latent - (1 - self.alpha_t).sqrt() * noise_pred.to(self.vae.dtype)
        ) / self.alpha_t.sqrt()

        pred_img = self.vae.decode(
            denoised_latent / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        return pred_img

    def forward(self, lq_img, prompt_embeds, tile_size, tile_overlap, verbose=True):
        lq_latent = self.vae.encode(
            lq_img.to(self.vae.dtype)
        ).latent_dist.sample() * self.vae.config.scaling_factor

        _, _, h, w = lq_latent.size()

        if h * w <= tile_size * tile_size:
            pred_img = self._forward_no_tile(lq_latent, prompt_embeds)
        else:
            pred_img = self._forward_tile(lq_latent, prompt_embeds, tile_size, tile_overlap)

        return pred_img
