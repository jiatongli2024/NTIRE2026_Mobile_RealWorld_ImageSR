import inspect
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import cv2
import random
import numpy as np
from einops import rearrange
from PIL import Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer
)
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models import ImageProjection


from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin

import PIL.Image
from peft import LoraConfig, PeftModel
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules, ModulesToSaveWrapper
from peft.utils.other import transpose
from peft.tuners.tuners_utils import BaseTunerLayer
from types import SimpleNamespace
from diffusers.utils.import_utils import is_xformers_available
if is_invisible_watermark_available():
    from .watermark import StableDiffusionXLWatermarker

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
from diffusers import DDPMScheduler
import glob

from models.team04_TODSR.autoencoder_kl import AutoencoderKL
from models.team04_TODSR.unet_2d_condition import UNet2DConditionModel
def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None


import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""
def initialize_vae(vae,lora_rank, return_lora_module_names=False, pretrained_model_path=None):
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder_pix = []
    l_target_modules_encoder_sem = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
    
    lora_conf_encoder_pix = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_pix)
    lora_conf_encoder_sem = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_sem)
    
    vae = PeftModel(
        model=vae,
        peft_config=lora_conf_encoder_pix,
        adapter_name="default_encoder_pix",
    )
    vae.add_adapter(adapter_name="default_encoder_sem", peft_config=lora_conf_encoder_sem)
    return vae, l_target_modules_encoder_pix,l_target_modules_encoder_sem

def initialize_vae_duallora(vae,lora_rank, return_lora_module_names=False, pretrained_model_path=None):
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder_pix = []
    l_target_modules_encoder_sem = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
    
    lora_conf_encoder_pix = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_pix)
    lora_conf_encoder_sem = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_sem)
    
    vae.add_adapter(lora_conf_encoder_pix, adapter_name="default_encoder_pix")
    vae.add_adapter(lora_conf_encoder_sem, adapter_name="default_encoder_sem")
    return vae, l_target_modules_encoder_pix,l_target_modules_encoder_sem


def initialize_vae_singlelora(vae,lora_rank, return_lora_module_names=False, pretrained_model_path=None):
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))

            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))

    
    lora_conf_encoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    
    vae.add_adapter(lora_conf_encoder,adapter_name="default_encoder")
    return vae, l_target_modules_encoder

def initialize_unet(unet,rank_pix, rank_sem, return_lora_module_names=False, pretrained_model_path=None):
    # unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix = [], [], []
    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv_shortcut", "conv", "conv1", "conv2", "conv_in", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n ):
                l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n ):
                l_target_modules_decoder_pix.append(n.replace(".weight",""))
                l_target_modules_decoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others_pix.append(n.replace(".weight",""))
                l_modules_others_sem.append(n.replace(".weight",""))
                break

    lora_conf_encoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_pix)
    lora_conf_decoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_pix)
    lora_conf_others_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_modules_others_pix)
    lora_conf_encoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_sem)
    lora_conf_decoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_sem)
    lora_conf_others_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_modules_others_sem)

    unet.add_adapter(lora_conf_encoder_pix, adapter_name="default_encoder_pix")
    unet.add_adapter(lora_conf_decoder_pix, adapter_name="default_decoder_pix")
    unet.add_adapter(lora_conf_others_pix, adapter_name="default_others_pix")
    unet.add_adapter(lora_conf_encoder_sem, adapter_name="default_encoder_sem")
    unet.add_adapter(lora_conf_decoder_sem, adapter_name="default_decoder_sem")
    unet.add_adapter(lora_conf_others_sem, adapter_name="default_others_sem")

    if return_lora_module_names:
        return unet, l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix, l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem
    else:
        return unet
    


class  TODSR_Pipeline(
    torch.nn.Module
):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        DDPM_scheduler: DDPMScheduler,
        args=None,
    ):
        super().__init__()

        self.vae=vae
        self.text_encoder=text_encoder
        self.tokenizer=tokenizer
        self.unet=unet
        self.scheduler=scheduler
        self.DDPM_scheduler = DDPM_scheduler
        self.image_processor = VaeImageProcessor(self.vae.config.scaling_factor)
    def set_eval(self,model_dir):
        unet_adapter_path=model_dir+"/model.pkl"
        vae_adapter_path=model_dir+"/model_vae.pkl"

        """Set models to evaluation mode."""
        self._load_pretrained_weights(unet_adapter_path,vae_adapter_path)
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
    def _load_pretrained_weights(self, unet_adapter_path,vae_adapter_path):
        """Load pretrained weights and initialize LoRA adapters."""
        print("Loading pretrained weights...")
        sd = torch.load(unet_adapter_path)
        vae=torch.load(vae_adapter_path)
        self._load_and_save_ckpt_from_state_dict_iqa(sd)
        self.load_vae_lora_single(vae)
        self.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix','default_encoder_sem', 'default_decoder_sem', 'default_others_sem','default_encoder_iqa', 'default_decoder_iqa', 'default_others_iqa'])
        self.vae.set_adapter(['default_encoder'])
        print("Merging LoRA adapters...")
        self.unet.merge_and_unload()
        self.vae.merge_and_unload()

    def _load_and_save_ckpt_from_state_dict_iqa(self, sd):
        """Load checkpoint and initialize LoRA adapters."""
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"])
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"])
        self.lora_conf_others_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"])

        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")

        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix, \
        self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
        sd["unet_lora_encoder_modules_pix"], sd["unet_lora_decoder_modules_pix"], sd["unet_lora_others_modules_pix"], \
            sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"]

        self.lora_conf_encoder_iqa = LoraConfig(r=sd["lora_rank_unet_iqa"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_iqa"])
        self.lora_conf_decoder_iqa = LoraConfig(r=sd["lora_rank_unet_iqa"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_iqa"])
        self.lora_conf_others_iqa = LoraConfig(r=sd["lora_rank_unet_iqa"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_iqa"])

        self.unet.add_adapter(self.lora_conf_encoder_iqa, adapter_name="default_encoder_iqa")
        self.unet.add_adapter(self.lora_conf_decoder_iqa, adapter_name="default_decoder_iqa")
        self.unet.add_adapter(self.lora_conf_others_iqa, adapter_name="default_others_iqa")

        self.lora_unet_modules_encoder_iqa, self.lora_unet_modules_decoder_iqa, self.lora_unet_others_iqa= \
            sd["unet_lora_encoder_modules_iqa"], sd["unet_lora_decoder_modules_iqa"], sd["unet_lora_others_modules_iqa"] 

        for n, p in self.unet.named_parameters():
            if "lora" in n :
                p.data.copy_(sd["state_dict_unet"][n])
                # print(n,p.data)   
        
    def load_vae_lora_single(self, sd):
        self.lora_conf_encoder = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",target_modules=sd["vae_lora_encoder_modules"])
        self.vae.add_adapter(self.lora_conf_encoder, adapter_name="default_encoder")
        self.lora_vae_modules_encoder =sd["vae_lora_encoder_modules"]

        for n, p in self.vae.named_parameters():
            if "lora" in n :
                p.data.copy_(sd["state_dict_vae"][n])
                # print(n,p.data)

    def forward(
        self,
        lq_img,
        prompt_embeds,height, width,
        **kwargs,
    ):
        
        device = torch.device('cuda')
        self.device=device
        
        lq_img_list = [lq_img]
        lq_img = self.image_processor.preprocess(lq_img_list, height=height, width=width).to(device, dtype = self.unet.dtype)      
        lq_latents = self.vae.encode(lq_img).latent_dist.sample()
        latents = lq_latents * self.vae.config.scaling_factor
        latent_model_input=latents

        with torch.no_grad(): 
            timestep =torch.tensor(273, device=latent_model_input.device).long()    
            alphas_cumprod = self.DDPM_scheduler.alphas_cumprod.to(dtype=torch.float32)
            alpha_prod_t = alphas_cumprod[timestep]
            model_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds,).sample 
            x_denoised =(latent_model_input - torch.sqrt(1-alpha_prod_t) * model_pred) / torch.sqrt(alpha_prod_t)
        image = (self.vae.decode(x_denoised/ self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = self.image_processor.postprocess(image, output_type="pil")
        return output_image,image
    

    def set_encoder_tile_settings(self, 
                         denoise_encoder_tile_sample_min_size = 1024, 
                         denoise_encoder_sample_overlap_factor = 0.25, 
                         vae_sample_size=1024, 
                         vae_tile_overlap_factor = 0.25):
        
        self.vae.config.sample_size = vae_sample_size
        self.vae.tile_overlap_factor = vae_tile_overlap_factor

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()
        

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()
        

def pipelinesd21(sd_path):
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet").to(dtype=torch.float16)
    DDPM_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
    
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        print('enable_xformers_memory_efficient_attention')
    else:
        raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    pipe = TODSR_Pipeline(
        vae =vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        DDPM_scheduler=DDPM_scheduler,
        args=None,
    )

    return pipe