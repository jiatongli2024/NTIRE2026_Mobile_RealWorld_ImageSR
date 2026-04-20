"""I/O wrapper for Mobile_Ntire_SRCB model loading and inference."""

import glob
import os

import torch
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import UNet2DConditionModel

from .model import create_srnet


DEFAULT_UNET_PATH = "stable-diffusion-2-1-base"
DEFAULT_CHECKPOINT_NAME = "weights.pth"


def resolve_checkpoint_path(model_dir, checkpoint_name=DEFAULT_CHECKPOINT_NAME):
    if model_dir is None:
        raise ValueError("model_dir must be provided.")
    if os.path.isfile(model_dir):
        return model_dir
    return os.path.join(model_dir, checkpoint_name)


def load_model(model_dir, device=None, unet_path=None, checkpoint_name=DEFAULT_CHECKPOINT_NAME, mixed_precision="fp32"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(model_dir, checkpoint_name)
    unet_path = resolve_checkpoint_path(model_dir, DEFAULT_UNET_PATH)

    config = UNet2DConditionModel.load_config(unet_path, subfolder="unet")
    unet = UNet2DConditionModel.from_config(config)    

    model = create_srnet(unet, checkpoint_path=checkpoint_path)
    model = model.to(device)

    if mixed_precision == "fp16" and device.type == "cuda":
        model = model.to(dtype=torch.float16)

    model.eval()
    return model


def list_input_images(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_paths = []
    if os.path.isfile(input_path):
        with open(input_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                image_paths.append(line.split()[0])
    else:
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(input_path, ext)))
            image_paths.extend(glob.glob(os.path.join(input_path, ext.upper())))
        image_paths.sort()

    return image_paths


def preprocess_image(path, device):
    lr_pil = Image.open(path).convert("RGB")
    ori_width, ori_height = lr_pil.size

    new_height = max((ori_height // 16) * 16, 16)
    new_width = max((ori_width // 16) * 16, 16)
    if new_height != ori_height or new_width != ori_width:
        lr_pil = lr_pil.resize((new_width, new_height), Image.BICUBIC)

    lr_tensor = transforms.ToTensor()(lr_pil).to(device).unsqueeze(0)
    return lr_tensor, (ori_width, ori_height)


def save_output(sr_tensor, output_path, original_size):
    expected_width = original_size[0] * 4
    expected_height = original_size[1] * 4

    sr_pil = transforms.ToPILImage()(sr_tensor[0].cpu())
    if sr_pil.size != (expected_width, expected_height):
        sr_pil = sr_pil.resize((expected_width, expected_height), Image.BICUBIC)
    sr_pil.save(output_path)


def run(model, input_path, output_path, device):
    os.makedirs(output_path, exist_ok=True)
    image_paths = list_input_images(input_path)

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Processing images"):
            lr_tensor, original_size = preprocess_image(path, device)
            sr_tensor = model(lr_tensor)
            save_output(sr_tensor, os.path.join(output_path, os.path.basename(path)), original_size)

    return image_paths


def main(model_dir, input_path, output_path, device=None, unet_path=None, checkpoint_name=DEFAULT_CHECKPOINT_NAME, mixed_precision="fp32"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_dir=model_dir,
        device=device,
        unet_path=unet_path,
        checkpoint_name=checkpoint_name,
        mixed_precision=mixed_precision,
    )
    return run(model, input_path, output_path, device)
