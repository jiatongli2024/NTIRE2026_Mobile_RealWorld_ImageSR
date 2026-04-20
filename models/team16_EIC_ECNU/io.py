import os
import glob
import torch
import math
from PIL import Image
from torchvision import transforms
import torch.nn.functional as tF
import torchvision.transforms.functional as F
from tqdm import tqdm

from .infer.omgsr_s_infer_model_multi_lora import OMGSR_S_Infer
from .infer.wavelet_color_fix import adain_color_fix

@torch.no_grad()
def judge_bicubic_or_unknown(input_image, device="cpu"):
    """
    input_image: PIL.Image
    Returns:
        is_unknown: bool
        score: float
        info: dict
    """
    x = F.to_tensor(input_image).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,3,H,W]
    _, _, h, w = x.shape

    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    gray = tF.interpolate(gray, size=(128, 128), mode="bilinear", align_corners=False).clamp(0, 1)

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    gx = tF.conv2d(gray, sobel_x, padding=1)
    gy = tF.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)

    grad_mean = float(mag.mean().item())
    grad_p95 = float(torch.quantile(mag.flatten(), 0.95).item())

    x0 = gray - gray.mean(dim=(2, 3), keepdim=True)
    A = torch.abs(torch.fft.fftshift(torch.fft.fft2(x0), dim=(-2, -1))) + 1e-8

    H = W = 128
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cy, cx = H // 2, W // 2
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max()
    rmax_check = 1 if (h > H and w > W) else -1

    low_mask = (rr <= 0.15 * rmax)
    high_mask = (rr > 0.35 * rmax)

    low_e = float(A[0, 0][low_mask].mean().item())
    high_e = float(A[0, 0][high_mask].mean().item())
    fft_high_low_ratio = high_e / (low_e + 1e-8)

    log_h = math.log(h + 1e-6)
    log_w = math.log(w + 1e-6)

    score = (
        +1.2 * grad_mean
        +1.0 * grad_p95
        +1.5 * fft_high_low_ratio
        +0.20 * log_h
        +0.25 * log_w
        +0.10
    ) * rmax_check

    is_unknown = score > 0.0

    info = {
        "grad_mean": grad_mean,
        "grad_p95": grad_p95,
        "fft_high_low_ratio": fft_high_low_ratio,
        "log_h": log_h,
        "log_w": log_w,
        "h": h,
        "w": w,
        "score": score,
        "label": "unknown" if is_unknown else "bicubic",
    }
    return is_unknown, score, info

def omgsr_inference(model_dir, input_path, output_path, device):
    process_size = 512
    upscale = 4
    mid_timestep = 273
    weight_dtype = torch.float16
    align_method = 'adain'

    sd_path = '/home/NTIRE26/data2/NTIRE2026/Mobile/NTIRE2026_Mobile_RealWorld_ImageSR/pretrained/stable-diffusion-2-1-base'
    # sd_path = os.path.join(model_dir, 'stable-diffusion-2-1-base')
    lora_path_unknown = os.path.join(model_dir, "lora_unknown")
    lora_path_bicubic = os.path.join(model_dir, "lora_bicubic")
    embeds_path = os.path.join(model_dir, "empty_embeds.pt")

    if not os.path.exists(embeds_path):
        raise FileNotFoundError(f"Cannot find {embeds_path}! Please ensure it is in your model_zoo folder.")
        
    prompt_embeds = torch.load(embeds_path).to(device, dtype=weight_dtype)
    print("Successfully loaded pre-computed empty prompt embeddings.")

    net_sr = OMGSR_S_Infer(
        sd_path=sd_path,
        lora_path_unknown=lora_path_unknown,
        lora_path_bicubic=lora_path_bicubic,
        mid_timestep=mid_timestep,
        device=device,
        weight_dtype=weight_dtype
    )

    os.makedirs(output_path, exist_ok=True)
    image_names = sorted(glob.glob(os.path.join(input_path, "*.png")) + 
                         glob.glob(os.path.join(input_path, "*.jpg")) + 
                         glob.glob(os.path.join(input_path, "*.jpeg")))

    bicubic_images = []
    unknown_images = []
    for image_name in image_names:
        with Image.open(image_name) as img:
            input_image = img.convert("RGB")
            is_unknown, score, info = judge_bicubic_or_unknown(input_image, device="cpu")

        if is_unknown:
            unknown_images.append(image_name)
        else:
            bicubic_images.append(image_name)
                
    print(f"Total images: {len(image_names)} (bicubic: {len(bicubic_images)}, Unknown: {len(unknown_images)})")

    def process_image_group(img_list, is_bicubic_group):
        if not img_list:
            return

        net_sr.merge_current_lora(is_bicubic=is_bicubic_group)
        
        group_name = "bicubic" if is_bicubic_group else "Unknown"
        for image_name in tqdm(img_list, desc=f"Processing {group_name} Images"):
            input_image = Image.open(image_name).convert('RGB')
            ori_width, ori_height = input_image.size
            
            rscale = upscale
            resize_flag = False

            if ori_width < process_size // rscale or ori_height < process_size // rscale:
                scale = (process_size // rscale) / min(ori_width, ori_height)
                input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
                resize_flag = True

            input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
            new_width = input_image.width - input_image.width % 8
            new_height = input_image.height - input_image.height % 8
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
            bname = os.path.basename(image_name).split('.')[0] + ".png"
            
            tile_size = process_size // 8
            tile_overlap = tile_size // 2

            with torch.no_grad():
                lq_img = F.to_tensor(input_image).unsqueeze(0).to(device=device, dtype=weight_dtype) * 2 - 1
                output_image, _ = net_sr(lq_img, prompt_embeds, tile_size, tile_overlap)

            output_image = output_image * 0.5 + 0.5
            output_image = torch.clip(output_image, 0, 1).float()
            output_pil = transforms.ToPILImage()(output_image[0].cpu())

            if align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            
            target_width = ori_width * rscale
            target_height = ori_height * rscale
            if resize_flag or output_pil.size != (target_width, target_height):
                output_pil = output_pil.resize((target_width, target_height), Image.LANCZOS)
                
            output_pil.save(os.path.join(output_path, bname))
            
        net_sr.unmerge_current_lora()

    process_image_group(bicubic_images, is_bicubic_group=True)
    process_image_group(unknown_images, is_bicubic_group=False)