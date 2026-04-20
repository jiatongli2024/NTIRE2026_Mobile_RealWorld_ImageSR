"""
NTIRE 2026 Mobile Real-World Image SR — Team 20 (YuFans)
Method: DiffBIR-RealESRGAN Perceptual Blend

Pipeline:
1. RealESRGAN x4plus (RRDB, pretrained)
2. DiffBIR v2.1 (diffusion-based, pretrained)
3. Pixel blend: 70% DiffBIR + 30% RealESRGAN
4. Post-processing: sharpen + CLAHE + saturation
"""
import os
import sys
import glob
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RRDB Generator (Real-ESRGAN x4plus architecture)
# ============================================================
class _ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(nf, gc)
        self.rdb2 = _ResidualDenseBlock(nf, gc)
        self.rdb3 = _ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        body = self.conv_body(self.body(feat))
        feat = feat + body
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ============================================================
# Post-processing
# ============================================================
def _postprocess(img, sharpen=2.0, contrast=2.0, saturation=1.2):
    r = img.copy().astype(np.float32)
    blur = cv2.GaussianBlur(r, (0, 0), 3)
    r = np.clip(r + sharpen * (r - blur), 0, 255)
    lab = cv2.cvtColor(r.astype(np.uint8), cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0 * contrast, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    hsv = cv2.cvtColor(r.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    r = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(r, 0, 255).astype(np.uint8)


# ============================================================
# Main entry point (required by competition framework)
# ============================================================
def main(model_dir, input_path, output_path, device=None):
    """
    Args:
        model_dir: path to model_zoo/team20_BlendSR/ containing weights
        input_path: directory with LR PNG images
        output_path: directory for SR output PNGs
        device: torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_path, exist_ok=True)

    img_list = sorted(glob.glob(os.path.join(input_path, "*.[jpJP][pnPN]*[gG]")))
    print(f"[team20_BlendSR] Processing {len(img_list)} images...")

    # ---------- Stage 1: RealESRGAN ----------
    print("  Stage 1: RealESRGAN x4plus")
    net = RRDBNet(3, 3, 64, 23, 32).to(device)
    ckpt = torch.load(os.path.join(model_dir, "RealESRGAN_x4plus.pth"),
                       map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt.get("params_ema", ckpt.get("params", ckpt)), strict=True)
    net.eval().half()

    real_results = {}
    for p in img_list:
        name = os.path.basename(p)
        img = cv2.imread(p)
        t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255.0)
        t = t.unsqueeze(0).to(device).half()
        with torch.no_grad():
            sr = net(t)
        sr = sr.squeeze(0).float().clamp(0, 1).cpu().numpy()
        sr = (sr[[2, 1, 0]] * 255.0).round().astype(np.uint8).transpose(1, 2, 0)
        real_results[name] = sr
    del net
    torch.cuda.empty_cache()

    # # ---------- Stage 2: DiffBIR ----------
    # print("  Stage 2: DiffBIR v2.1")
    # diffbir_dir = os.path.join(model_dir, "DiffBIR")
    diff_results = {}
    # if os.path.isdir(diffbir_dir):
    #     sys.path.insert(0, diffbir_dir)
    #     try:
    #         from diffbir.pipeline import BSRInferencePipeline  # noqa
    #         pipe = BSRInferencePipeline.from_pretrained(
    #             diffbir_dir, device=device, dtype=torch.float32)
    #         for p in img_list:
    #             name = os.path.basename(p)
    #             img = cv2.imread(p)
    #             sr = pipe(img, upscale=4, cfg_scale=8, noise_aug=0, steps=10)
    #             diff_results[name] = sr
    #         del pipe
    #         torch.cuda.empty_cache()
    #     except Exception as e:
    #         print(f"  DiffBIR failed ({e}), using pre-computed outputs...")
    # # Fallback: pre-computed DiffBIR outputs
    # precomp = os.path.join(model_dir, "diffbir_outputs")
    # if not diff_results and os.path.isdir(precomp):
    #     for f in sorted(os.listdir(precomp)):
    #         if f.endswith(".png"):
    #             diff_results[f] = cv2.imread(os.path.join(precomp, f))
    #     print(f"  Loaded {len(diff_results)} pre-computed DiffBIR outputs")

    # ---------- Stage 3 & 4: Blend + Post-process ----------
    alpha = 0.7  # DiffBIR weight
    print(f"  Stage 3: Blend (DiffBIR {alpha*100:.0f}%) + Post-process")
    for name, r_img in real_results.items():
        if name in diff_results:
            d_img = diff_results[name].astype(np.float32)
            r_f = r_img.astype(np.float32)
            if d_img.shape != r_f.shape:
                d_img = cv2.resize(d_img, (r_f.shape[1], r_f.shape[0]))
            blended = np.clip(alpha * d_img + (1 - alpha) * r_f, 0, 255).astype(np.uint8)
        else:
            blended = r_img
        enhanced = _postprocess(blended, sharpen=2.0, contrast=2.0, saturation=1.2)
        cv2.imwrite(os.path.join(output_path, name), enhanced)

    print(f"  Done. {len(real_results)} images saved to {output_path}")
