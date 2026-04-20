import glob
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from .model import mobilehgsr
except ImportError:
    from model import mobilehgsr


class TiledInference:
    def __init__(self, model, tile_size=128, overlap=24, device="cuda"):
        self.model = model
        self.tile = tile_size
        self.overlap = overlap
        self.device = device
        self.scale = 4
        self._window_cache = {}

    def _get_window(self, hr_tile):
        if hr_tile not in self._window_cache:
            w1d = torch.hann_window(hr_tile, periodic=False, device=self.device) + 1e-8
            self._window_cache[hr_tile] = w1d[None, :] * w1d[:, None]
        return self._window_cache[hr_tile]

    @torch.no_grad()
    def infer(self, lr_img):
        _, _, h, w = lr_img.shape
        pad = self.overlap
        s = self.scale

        lr_padded = F.pad(lr_img, (pad, pad, pad, pad), mode="reflect")
        _, _, ph, pw = lr_padded.shape

        tile = min(self.tile, ph, pw)
        hr_tile = tile * s
        window = self._get_window(hr_tile)

        out_h, out_w = ph * s, pw * s
        output = torch.zeros(1, 3, out_h, out_w, device=self.device)
        weight = torch.zeros(1, 1, out_h, out_w, device=self.device)

        step = max(1, tile - self.overlap)
        rows = list(range(0, ph - tile + 1, step))
        cols = list(range(0, pw - tile + 1, step))
        if rows[-1] != ph - tile:
            rows.append(ph - tile)
        if cols[-1] != pw - tile:
            cols.append(pw - tile)

        for y in rows:
            for x in cols:
                lr_tile = lr_padded[:, :, y:y + tile, x:x + tile]
                sr_tile = self.model(lr_tile)

                oy, ox = y * s, x * s
                th, tw = sr_tile.shape[2], sr_tile.shape[3]
                output[:, :, oy:oy + th, ox:ox + tw] += sr_tile * window
                weight[:, :, oy:oy + th, ox:ox + tw] += window

        sr_full = output / weight.clamp(min=1e-8)
        crop = pad * s
        sr_full = sr_full[:, :, crop:crop + h * s, crop:crop + w * s]
        return sr_full.clamp(0, 1)


def load_model(ckpt_path, device="cuda"):
    model = mobilehgsr().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "params_ema" in ckpt:
        model.load_state_dict(ckpt["params_ema"], strict=True)
    elif "params" in ckpt:
        model.load_state_dict(ckpt["params"], strict=True)
    elif "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    return model


def run(model, input_path, output_path, device, tile_size=128, overlap=24):
    tiler = TiledInference(model, tile_size=tile_size, overlap=overlap, device=device)
    os.makedirs(output_path, exist_ok=True)

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    input_files = []
    for ext in extensions:
        input_files.extend(glob.glob(os.path.join(input_path, ext)))
    input_files = sorted(input_files)

    if not input_files:
        print("No images found in %s" % input_path)
        return

    print("Processing %d images from %s" % (len(input_files), input_path))
    for i, path in enumerate(input_files):
        name = os.path.basename(path)
        out_name = os.path.splitext(name)[0] + ".png"
        out_path = os.path.join(output_path, out_name)

        lr_pil = Image.open(path).convert("RGB")
        lr_np = np.array(lr_pil).astype(np.float32) / 255.0
        lr_t = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0).to(device)

        sr_t = tiler.infer(lr_t)
        sr_np = (
            (sr_t[0].cpu().permute(1, 2, 0).numpy() * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        Image.fromarray(sr_np).save(out_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(input_files):
            print("  [%d/%d] %s" % (i + 1, len(input_files), out_name))


def main(model_dir, input_path, output_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_dir, device)
    run(model, input_path, output_path, device)
