import glob
import os
from typing import Dict

import cv2
import torch

from .model import RRDBNet


def _load_checkpoint(model_path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    return checkpoint


def _extract_state_dict(checkpoint: Dict[str, torch.Tensor]):
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _resolve_model_path(model_dir):
    if os.path.isdir(model_dir):
        candidates = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
        if candidates:
            return candidates[0]
        txt_candidates = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
        if txt_candidates:
            return _resolve_model_path(txt_candidates[0])
        raise FileNotFoundError(f"No checkpoint or descriptor found under: {model_dir}")

    if model_dir.endswith(".txt"):
        with open(model_dir, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    _, value = line.split("=", 1)
                    line = value.strip()
                if not os.path.isabs(line):
                    line = os.path.abspath(os.path.join(os.path.dirname(model_dir), line))
                return line
        raise ValueError(f"No checkpoint path found in descriptor: {model_dir}")

    return model_dir


def _resolve_input_dir(input_path):
    lq_dir = os.path.join(input_path, "LQ")
    return lq_dir if os.path.isdir(lq_dir) else input_path


def _list_images(input_path):
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(input_path, pattern)))
    return sorted(set(image_paths))


def _read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return tensor


def _write_image(image_tensor, output_path):
    image = image_tensor.squeeze(0).detach().float().cpu().clamp_(0, 1).numpy()
    image = (image.transpose(1, 2, 0) * 255.0).round().astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)


def main(model_dir, input_path, output_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    checkpoint_path = _resolve_model_path(model_dir)
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = _extract_state_dict(checkpoint)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    input_dir = _resolve_input_dir(input_path)
    image_paths = _list_images(input_dir)
    os.makedirs(output_path, exist_ok=True)

    with torch.inference_mode():
        for image_path in image_paths:
            image = _read_image(image_path).to(device)
            output = model(image)
            save_path = os.path.join(output_path, os.path.basename(image_path))
            _write_image(output, save_path)
