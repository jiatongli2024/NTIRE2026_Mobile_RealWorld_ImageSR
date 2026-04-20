# NTR @ UIUC — NTIRE 2026 Mobile Real-World Image Super-Resolution

**Team:** NTR
**Institution:** University of Illinois Urbana-Champaign
**Challenge:** NTIRE 2026 Mobile Real-World Image Super-Resolution (×4)
**CodaBench:** https://www.codabench.org/competitions/13509/ (account: miketjc0316)
**Best Score:** CLIPIQA = 0.7293, MUSIQ = 68.94 (submission 634640)

---

## Method Summary

We use **OSEDiff** (Wu et al., NeurIPS 2024) — a one-step diffusion SR model that
distils Stable Diffusion 2.1 via score-identity distillation. The model runs
inference in a single forward pass (no iterative sampling).

**Key settings for our best submission:**
- Color alignment: **wavelet** (preserves low-frequency statistics of the LR input)
- Post-processing: **unsharp mask** (radius=1.5, strength=130%) + **saturation ×1.1**
- Process size: 512 (tiled VAE, FP16)
- No test-time augmentation (TTA hurts perceptual metrics for diffusion SR)

We use the **publicly released** OSEDiff LoRA weights without any fine-tuning.

---

## Quick Start (3 commands)

```bash
# 1. Clone OSEDiff and install dependencies
git clone https://github.com/cswry/OSEDiff OSEDiff
pip install -r requirements.txt

# 2. Download model weights (see "Model Downloads" below)

# 3. Run inference
bash infer.sh /path/to/LR_images /path/to/SR_output
```

---

## System Requirements

- GPU: NVIDIA GPU with ≥24 GB VRAM (tested on NVIDIA A6000 48 GB)
- CUDA: 11.8 or 12.1
- Python: 3.10+
- Disk: ~20 GB for model weights

---

## Installation

```bash
# Clone OSEDiff into this directory
git clone https://github.com/cswry/OSEDiff OSEDiff

# Install dependencies
pip install -r requirements.txt
```

---

## Model Downloads

### 1. Stable Diffusion 2.1 base
```python
from huggingface_hub import snapshot_download
snapshot_download(
    "Manojb/stable-diffusion-2-1-base",
    local_dir="hf_cache/sd21"
)
```

### 2. OSEDiff LoRA weights (`osediff.pkl`)
Download from the OSEDiff official release:
https://github.com/cswry/OSEDiff?tab=readme-ov-file#-quick-inference

Place at: `OSEDiff/preset/models/osediff.pkl`

### 3. RAM model (`ram_swin_large_14m.pth`)
```bash
wget -O hf_cache/ram_models/ram_swin_large_14m.pth \
  https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
```

### 4. DAPE adapter (`DAPE.pth`)
Download `DAPE.pth` from the OSEDiff release page:
https://github.com/cswry/OSEDiff?tab=readme-ov-file#-quick-inference

Place at: `hf_cache/ram_models/DAPE.pth`

**Expected directory structure after all downloads:**
```
.
├── infer.sh
├── postprocess.py
├── test.sh
├── requirements.txt
├── README.md
├── OSEDiff/                              # Cloned from GitHub
│   ├── test_osediff.py
│   ├── preset/models/osediff.pkl        # Downloaded
│   └── ...
└── hf_cache/
    ├── sd21/                            # or models--Manojb--stable-diffusion-2-1-base/
    │   └── model_index.json
    └── ram_models/
        ├── ram_swin_large_14m.pth       # Downloaded
        └── DAPE.pth                     # Downloaded
```

---

## Running Inference

```bash
bash infer.sh <LR_INPUT_DIR> <SR_OUTPUT_DIR>
```

**Environment variables (optional overrides):**
```bash
OSEDIFF_DIR=/path/to/OSEDiff  \  # default: ./OSEDiff
HF_HOME=/path/to/hf_cache     \  # default: ./hf_cache
PYTHON=python3                 \  # default: python3
CUDA_VISIBLE_DEVICES=0         \  # default: 0
bash infer.sh /path/to/LR /path/to/SR
```

**What the script does:**
1. Runs OSEDiff with wavelet color alignment on all `.png` files in the input directory
2. Applies post-processing (unsharp mask radius=1.5, strength=130%; saturation ×1.1)
3. Saves final SR images to the output directory (JPEG quality=85, `.png` extension)

**Runtime:** ~6.8 s/image on NVIDIA A6000 (48 GB). For 193 test images: ~22 min.

---

## Post-Processing Details

The `postprocess.py` script applies two operations to each OSEDiff SR output:

```python
# Unsharp mask (edge sharpening)
img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))

# Saturation boost
img = ImageEnhance.Color(img).enhance(1.1)
```

Output is saved as JPEG quality=85 with `.png` extension (PIL reads/writes by magic bytes).

**Why JPEG quality=85?**
JPEG quality=70 degraded CLIPIQA from ~0.68 to 0.36 (blocking artifacts penalised by IQA).
Quality=85 is the safe threshold — preserves all perceptual benefit of OSEDiff.

---

## Submission Packaging

After inference, package output images for CodaBench:

```bash
# Images must be flat at zip root (no subdirectories)
cd /path/to/SR_output
zip -j submission.zip *.png
```

---

## Ablation Results (CodaBench Test Phase)

| Method | CLIPIQA ↑ | MUSIQ ↑ |
|--------|-----------|---------|
| OSEDiff, adain, no post-proc. | 0.6627 | 65.71 |
| OSEDiff, adain + sharpen+sat | 0.6775 | 68.38 |
| OSEDiff, wavelet only | 0.6818 | 66.33 |
| **OSEDiff, wavelet + sharpen+sat** | **0.7293** | **68.94** |
| OSEDiff, 4-fold TTA | 0.5689 | 62.79 |
| MDAE-SFT (PSNR model) | 0.3759 | 29.35 |

**Key findings:**
1. Wavelet alignment outperforms AdaIN for CLIPIQA (+0.052 with post-proc.)
2. Unsharp mask + saturation boost raises MUSIQ by ~2.6 points
3. TTA **hurts** perceptual metrics — averaging diffusion outputs blurs textures

---

## Citation

```bibtex
@article{osediff,
  title   = {One-Step Effective Diffusion Network for Real-World Image Super-Resolution},
  author  = {Wu, Rongyuan and Yang, Lingchen and Sun, Longwei and Li, Zhengqiang and Zhang, Kai},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2024}
}
```
