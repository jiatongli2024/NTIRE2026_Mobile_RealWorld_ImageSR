import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import utils_logger
from utils import utils_image as util

from .model import IMAG2006


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def adain_color_fix(target, source):
    to_tensor = transforms.ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    to_image = transforms.ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image


def wavelet_color_fix(target, source):
    to_tensor = transforms.ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    to_image = transforms.ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    if len(size) != 4:
        raise ValueError("The input feature should be a 4D tensor.")
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def wavelet_blur(image, radius):
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None]
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image, levels=5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += image - low_freq
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat, style_feat):
    content_high_freq, _content_low_freq = wavelet_decomposition(content_feat)
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    return content_high_freq + style_low_freq


def resolve_device(device):
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = str(device) if device is not None else "cuda"

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA is not available. Falling back to CPU.")
        return "cpu"
    return device_str


def list_input_images(input_path):
    if input_path.lower().endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    if os.path.isdir(input_path):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(files)

    return [input_path]


def resolve_output_name(image_path):
    src_name = os.path.basename(image_path)
    stem = src_name.split(".")[0]
    return stem + ".png"


def _load_config(model_dir):
    if model_dir is None:
        config_path = None
    elif os.path.isdir(model_dir):
        for name in ("config.json", "config.yaml", "config.yml"):
            cand = os.path.join(model_dir, name)
            if os.path.exists(cand):
                config_path = cand
                break
        else:
            config_path = None
    else:
        config_path = model_dir if os.path.exists(model_dir) else None

    cfg = {}
    if config_path:
        if config_path.endswith((".yml", ".yaml")):
            try:
                import yaml
            except Exception as exc:
                raise RuntimeError("YAML config requires PyYAML installed.") from exc
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

    cfg.setdefault("prompt", "")
    cfg.setdefault("mid_timestep", 273)
    cfg.setdefault("process_size", 512)
    cfg.setdefault("upscale", 4)
    cfg.setdefault("align_method", "adain")
    cfg.setdefault("weight_dtype", "bf16")
    cfg.setdefault("png_compress_level", 0)

    missing = [k for k in ("sd_path", "lora_path") if not cfg.get(k)]
    if missing:
        raise ValueError(
            f"Missing required config fields: {missing}. "
            "Provide config.json or set IMAG2006_SD_PATH/IMAG2006_LORA_PATH."
        )

    cfg["mid_timestep"] = int(cfg["mid_timestep"])
    cfg["process_size"] = int(cfg["process_size"])
    cfg["upscale"] = int(cfg["upscale"])
    cfg["png_compress_level"] = int(cfg["png_compress_level"])
    return cfg


def _resolve_weight_dtype(value):
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return torch.bfloat16
    if isinstance(value, str):
        key = value.lower().replace("torch.", "")
        mapping = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if key in mapping:
            return mapping[key]
    raise ValueError(f"Unsupported weight_dtype: {value}")


def _build_prompt_embeds(prompt_text, cfg, device, weight_dtype):
    prompt_embeds_path = cfg.get("prompt_embeds_path", "")
    if prompt_embeds_path and os.path.exists(prompt_embeds_path):
        prompt_embeds = torch.load(prompt_embeds_path, map_location="cpu")
        return prompt_embeds.to(device=device, dtype=weight_dtype)

    from transformers import AutoTokenizer, CLIPTextModel
    from diffusers.training_utils import free_memory

    tokenizer = AutoTokenizer.from_pretrained(cfg["sd_path"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        cfg["sd_path"], subfolder="text_encoder"
    ).to(device=device, dtype=weight_dtype)
    text_encoder.eval()

    prompt_batch = [prompt_text]
    with torch.inference_mode():
        prompt_embeds = [
            text_encoder(
                tokenizer(
                    caption,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device=text_encoder.device)
            )[0]
            for caption in prompt_batch
        ]
        prompt_embeds = torch.concat(prompt_embeds, dim=0)

    del tokenizer
    del text_encoder
    free_memory()
    return prompt_embeds


def _apply_color_fix(output_pil, input_pil, align_method):
    if align_method == "adain":
        return adain_color_fix(target=output_pil, source=input_pil)
    if align_method == "wavelet":
        return wavelet_color_fix(target=output_pil, source=input_pil)
    return output_pil


def run_inference(net_sr, prompt_embeds, cfg, data_path, save_path, device):
    image_names = list_input_images(data_path)
    if len(image_names) == 0:
        raise ValueError("No input images found.")

    util.mkdir(save_path)

    tile_size = cfg["process_size"] // 8
    tile_overlap = tile_size // 2

    for image_name in tqdm(image_names, desc="Inference"):
        input_image = Image.open(image_name).convert("RGB")
        ori_width, ori_height = input_image.size
        rscale = cfg["upscale"]
        resize_flag = False

        if ori_width < cfg["process_size"] // rscale or ori_height < cfg["process_size"] // rscale:
            scale = (cfg["process_size"] // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))

        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        out_name = resolve_output_name(image_name)
        out_path = os.path.join(save_path, out_name)

        with torch.inference_mode():
            lq_img = TF.to_tensor(input_image).unsqueeze(0).to(
                device=device,
                dtype=cfg["weight_dtype"],
            ) * 2 - 1

            output_image = net_sr(
                lq_img=lq_img,
                prompt_embeds=prompt_embeds,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                verbose=True,
            )

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clamp(output_image, 0, 1).float()
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        output_pil = _apply_color_fix(output_pil, input_image, cfg["align_method"])

        if resize_flag:
            output_pil = output_pil.resize(
                (int(cfg["upscale"] * ori_width), int(cfg["upscale"] * ori_height))
            )

        final_w = ori_width * cfg["upscale"]
        final_h = ori_height * cfg["upscale"]
        output_pil = output_pil.resize((final_w, final_h), Image.LANCZOS)

        output_pil.save(out_path, format="PNG", compress_level=cfg["png_compress_level"])


def main(model_dir, input_path, output_path, device=None):
    utils_logger.logger_info("NTIRE2026-ImageSRx4", log_path="NTIRE2026-ImageSRx4.log")

    if torch.cuda.is_available():
        torch.cuda.current_device()
        torch.cuda.empty_cache()

    cfg = _load_config(model_dir)
    device_str = resolve_device(device)
    weight_dtype = _resolve_weight_dtype(cfg.get("weight_dtype"))
    cfg["weight_dtype"] = weight_dtype

    prompt_text = cfg.get("prompt", "")
    prompt_embeds = _build_prompt_embeds(prompt_text, cfg, device_str, weight_dtype)

    net_sr = IMAG2006(
        sd_path=cfg["sd_path"],
        lora_path=cfg["lora_path"],
        mid_timestep=cfg["mid_timestep"],
        device=device_str,
        weight_dtype=weight_dtype,
    )

    run_inference(net_sr, prompt_embeds, cfg, input_path, output_path, device_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMAG2006 OMGSR-S Inference")
    parser.add_argument("--model_dir", type=str, required=True, help="Config path or directory")
    parser.add_argument("--input_path", type=str, required=True, help="Input image/dir/txt")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu")
    args = parser.parse_args()

    main(args.model_dir, args.input_path, args.output_path, args.device)
