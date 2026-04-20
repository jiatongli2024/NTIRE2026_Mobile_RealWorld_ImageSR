import glob
import os

import torch
from PIL import Image
from torchvision import transforms

from .team18_SAFMN_Deep15 import SAFMN_Deep15


DEFAULT_CHECKPOINT_NAME = "team18_SAFMN_Deep15.pth"
UPSCALE = 4


def resolve_checkpoint_path(model_dir, checkpoint_name=DEFAULT_CHECKPOINT_NAME):
    if model_dir is None:
        raise ValueError("model_dir must be provided.")
    if os.path.isfile(model_dir):
        return model_dir
    return os.path.join(model_dir, checkpoint_name)


def load_model(model_dir, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(model_dir)

    model = SAFMN_Deep15(num_in_ch=3, num_out_ch=3, dim=40, num_blocks=15, upscale=UPSCALE)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def list_input_images(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path):
        return [input_path]

    image_paths = []
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        image_paths.extend(glob.glob(os.path.join(input_path, ext.upper())))
    image_paths.sort()
    return image_paths


@torch.no_grad()
def main(model_dir, input_path, output_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_dir=model_dir, device=device)
    os.makedirs(output_path, exist_ok=True)

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    image_paths = list_input_images(input_path)
    for path in image_paths:
        lr_pil = Image.open(path).convert("RGB")
        ori_width, ori_height = lr_pil.size
        lr_tensor = to_tensor(lr_pil).unsqueeze(0).to(device)

        sr_tensor = model(lr_tensor).clamp(0, 1)
        sr_pil = to_pil(sr_tensor[0].cpu())

        target_size = (ori_width * UPSCALE, ori_height * UPSCALE)
        if sr_pil.size != target_size:
            sr_pil = sr_pil.resize(target_size, Image.BICUBIC)

        sr_pil.save(os.path.join(output_path, os.path.basename(path)))

