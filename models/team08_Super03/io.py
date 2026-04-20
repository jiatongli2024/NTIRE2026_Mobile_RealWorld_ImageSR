import os
import glob
import json
import copy
import logging
import argparse
import torch
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline
from diffusers.models.autoencoders.vae import Decoder
from .model import Net
# -----------------------------

from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util

def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # --------------------------------
        # test the image as a whole 
        # --------------------------------
        _, _, h0, w0 = img_lq.shape
        pad_h = (64 - h0 % 64) % 64
        pad_w = (64 - w0 % 64) % 64
        # pad: (left, right, top, bottom)
        img_lq_pad = F.pad(img_lq, (0, pad_w, 0, pad_h), mode="reflect")

        output = model(img_lq_pad)

        scale_h = output.shape[-2] // img_lq_pad.shape[-2]
        scale_w = output.shape[-1] // img_lq_pad.shape[-1]
        output = output[:, :, :h0 * scale_h, :w0 * scale_w]
        
    else:
        # --------------------------------
        # test the image tile by tile
        # --------------------------------
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                
                _, _, ph, pw = in_patch.shape
                pad_ph = (64 - ph % 64) % 64
                pad_pw = (64 - pw % 64) % 64
                in_patch_pad = F.pad(in_patch, (0, pad_pw, 0, pad_ph), mode="reflect")
                
                out_patch_pad = model(in_patch_pad)
                
                pscale_h = out_patch_pad.shape[-2] // in_patch_pad.shape[-2]
                pscale_w = out_patch_pad.shape[-1] // in_patch_pad.shape[-1]
                out_patch = out_patch_pad[:, :, :ph * pscale_h, :pw * pscale_w]
                
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

def run(model, data_path, save_path, tile, device):
    data_range = 1.0
    sf = 4
    border = sf

    if data_path.endswith('/'):  # solve when path ends with /
        data_path = data_path[:-1]
    # scan all the jpg and png images
    input_img_list = sorted(glob.glob(os.path.join(data_path, '*.[jpJP][pnPN]*[gG]')))
    # save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    for i, img_lr in enumerate(input_img_list):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_lr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)


        img_lr = img_lr * 2 - 1

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        img_sr = forward(img_lr, model, tile)
        
        img_sr = (img_sr + 1) / 2
        img_sr = img_sr.clamp(0, 1)

        img_sr = util.tensor2uint(img_sr, data_range)
        util.imsave(img_sr, os.path.join(save_path, img_name+ext))


def main(model_dir, input_path, output_path, device=None):
    utils_logger.logger_info("NTIRE2024-ImageSRx4", log_path="NTIRE2024-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2024-ImageSRx4")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {device}')

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    
    
    #download sd2-1

    # from modelscope import snapshot_download
    # model_id = snapshot_download('stabilityai/stable-diffusion-2-1-base')
    # Local: model_id = "/autodl-fs/data/stable-diffusion-2-1"
    model_id = "/data/pretrained/stable-diffusion-2-1-base"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, local_files_only=True).to(device)
    unet = pipe.unet

    ckpt_halfdecoder = torch.load("./model_zoo/team11_Super03/halfDecoder.ckpt", weights_only=False)
    decoder = Decoder(in_channels=4,
                out_channels=3,
                up_block_types=["UpDecoderBlock2D" for _ in range(4)],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=2, 
                norm_num_groups=32, 
                act_fn="silu", 
                norm_type="group", 
                mid_block_add_attention=True).to(device)
    
    decoder_ckpt = {}
    for k,v in ckpt_halfdecoder["state_dict"].items():
        if "decoder" in k:
            new_k = k.replace("decoder.", "")
            decoder_ckpt[new_k] = v
    decoder.load_state_dict(decoder_ckpt, strict=True)

    base_model = Net(unet, copy.deepcopy(decoder)).to(device)


    ckpt = torch.load(model_dir, weights_only=False, map_location=device)
    new_state = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    base_model.load_state_dict(new_state, strict=False)

    model = torch.nn.Sequential(
        base_model,
        *decoder.up_blocks,
        decoder.conv_norm_out,
        decoder.conv_act,
        decoder.conv_out,
    ).to(device)

    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    run(model, input_path, output_path, tile, device)