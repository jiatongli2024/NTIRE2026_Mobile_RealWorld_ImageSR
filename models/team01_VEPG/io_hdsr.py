import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from .model_hdsr import HDSR_eval

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import glob


def _normalize_device(device):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)
    return device


def hd_sr(args, device=None):
    device = _normalize_device(device)
    if not hasattr(args, "default"):
        args.default = True
    model = HDSR_eval(args, device=device)
    model.set_eval()
    model.vae.train()


    if os.path.isdir(args.test_dir):
        image_names = sorted(glob.glob(f'{args.test_dir}/*.jpg') + glob.glob(f'{args.test_dir}/*.png')) #*.png
    else:
        image_names = [args.test_dir]

    # Make the output directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    time_records = []
    for image_name in image_names:
        # Ensure the input image is a multiple of 8
        test_dir = Image.open(image_name).convert('RGB')
        ori_width, ori_height = test_dir.size
        print("ori size", (ori_width, ori_height))
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            test_dir = test_dir.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        test_dir = test_dir.resize((test_dir.size[0] * rscale, test_dir.size[1] * rscale))
        new_width = test_dir.width - test_dir.width % 8
        new_height = test_dir.height - test_dir.height % 8
        
        if new_height != ori_height * rscale or new_width != ori_width * rscale:
            resize_flag = True

        test_dir = test_dir.resize((new_width, new_height), Image.LANCZOS)
        print("size", (new_height, new_width))
        bname = os.path.basename(image_name)

        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(test_dir).unsqueeze(0).to(device) * 2 - 1
            validation_prompt = ""
            output_image = model(args.default, c_t, prompt=validation_prompt)


        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=test_dir)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=test_dir)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(args.save_dir, bname))

    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', '-i', type=str, default='preset/test_datasets', help="path to the input image")
    parser.add_argument('--save_dir', '-o', type=str, default='experiments/test', help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str, default='models--stabilityai--stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str, default='', help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="wavelet")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=256) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    


    args = parser.parse_args()
    if not hasattr(args, "default"):
        args.default = True

    # Call the processing function
    hd_sr(args)
