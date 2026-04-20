#!/usr/bin/env python3
"""
Post-processing for NTR @ UIUC OSEDiff SR outputs.
Applies unsharp mask (sharpening) + saturation enhancement.
Saves as JPEG quality=85 with original .png extension.

Usage:
    python postprocess.py --input_dir <DIR> --output_dir <OUT_DIR>
"""

import argparse
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance


def postprocess(img: Image.Image) -> Image.Image:
    """Apply unsharp mask + saturation boost."""
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))
    img = ImageEnhance.Color(img).enhance(1.1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    print(f"Post-processing {len(images)} images from {input_dir} -> {output_dir}")

    for i, src in enumerate(images):
        img = Image.open(src).convert("RGB")
        img = postprocess(img)
        dst = output_dir / src.name  # keep same filename
        # Save as JPEG q=85 but with original extension
        img.save(str(dst), format="JPEG", quality=85)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(images)}] {src.name}")

    print(f"Done. {len(images)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
