import os.path
import logging
import torch
import argparse
import json
import glob
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
import cv2
from torchvision.transforms import ToTensor, ToPILImage
from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util
import torch.cuda
import argparse
from PIL import Image
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import json
import numpy as np
from models.team04_TODSR.pipeline import pipelinesd21
def check_image_size(x, padder_size=8):
    # 获取图像的宽高
    width, height = x.size
    padder_size = padder_size
    # 计算需要填充的高度和宽度
    mod_pad_h = (padder_size - height % padder_size) % padder_size
    mod_pad_w = (padder_size - width % padder_size) % padder_size
    x_np = np.array(x)
    # 使用 ImageOps.expand 进行填充
    x_padded = cv2.copyMakeBorder(x_np, top=0, bottom=mod_pad_h, left=0, right=mod_pad_w, borderType=cv2.BORDER_REPLICATE)

    x = Image.fromarray(x_padded)
    # x = x.resize((width + mod_pad_w, height + mod_pad_h))
    
    return x, width, height, width + mod_pad_w, height + mod_pad_h

def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output
def main(model_dir, input_path, output_path, device=None):
    utils_logger.logger_info("NTIRE2026-Mobile Real-World Image Super-Resolution", log_path="NTIRE2026-Mobile Real-World Image Super-Resolution_TODSR.log")
    logger = logging.getLogger("NTIRE2026-Mobile Real-World Image Super-Resolution")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {device}')

    sd_path="model_zoo/team04_TODSR/stable-diffusion-2-1-base"
    model=pipelinesd21(sd_path)
    model = model.to(device)
    model.set_eval(model_dir)
    os.makedirs(output_path, exist_ok=True)
    with torch.inference_mode():
                prompt_embeds = [
                        model.text_encoder(
                            model.tokenizer(
                                "", max_length=model.tokenizer.model_max_length,
                                padding="max_length", truncation=True, return_tensors="pt"
                            ).input_ids.to(device)
                        )[0]
                    ]
                prompt_embeds=torch.concat(prompt_embeds, dim=0).to(device)

    exist_file = os.listdir(output_path)

    with torch.no_grad():
        for file_name in sorted(os.listdir(input_path)):
            img_name, ext = os.path.splitext(file_name)
            if ext == ".json":
                continue
            
            if f"{img_name}.png" in exist_file:
                print(f"{img_name}.png exist")
                continue
            else:
                print(img_name)

            image = Image.open(os.path.join(input_path,file_name)).convert('RGB')


            # step 2: Restorationtext 
            w, h = image.size
            w *= 4
            h *= 4
            image = image.resize((w, h), Image.LANCZOS)
            input_image, width_init, height_init, width_now, height_now = check_image_size(image)
            

            gen_image,_ = model(lq_img=input_image, prompt_embeds = prompt_embeds,height = height_now, width=width_now,)
            path = os.path.join(output_path, img_name+'.png')
            cropped_image = gen_image[0].crop((0, 0, width_init, height_init))

            out_image = wavelet_color_fix(cropped_image, image)

            out_image.save(path)
