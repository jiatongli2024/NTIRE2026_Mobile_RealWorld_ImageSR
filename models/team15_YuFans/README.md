# Team 20 — YuFans: DiffBIR-RealESRGAN Perceptual Blend

## Setup

```bash
# Clone official repo
git clone https://github.com/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR
cd NTIRE2026_Mobile_RealWorld_ImageSR

# Copy this folder to models/
cp -r team20_BlendSR models/

# Download RealESRGAN weights
mkdir -p model_zoo/team20_BlendSR
wget -O model_zoo/team20_BlendSR/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Clone DiffBIR and download weights
cd model_zoo/team20_BlendSR
git clone https://github.com/XPixelGroup/DiffBIR.git
pip install -r DiffBIR/requirements.txt

# OR: use pre-computed DiffBIR outputs
# Place DiffBIR test outputs in model_zoo/team20_BlendSR/diffbir_outputs/
```

## Run

```bash
python test.py --test_dir /path/to/test_lr --save_dir results --model_id 20
```

## Dependencies
- torch >= 2.0
- torchvision
- opencv-python-headless
- numpy
- (For DiffBIR: diffusers, accelerate, xformers)
