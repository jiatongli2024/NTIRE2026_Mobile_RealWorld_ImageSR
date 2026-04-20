"""
NTIRE 2026 Super-Resolution Model
Merged model containing: SRNet (based on UNet), EightLayerConv, and TAESD Decoder
"""
import torch
import types
from torch import nn
import torch.nn.functional as F

# =============================================================================
# forward
# =============================================================================
def MyUNet2DConditionModel_SD_forward(self, x):
    global skip
    x = self.conv_in(x)
    skip = [x]
    x = self.body(x)
    return x

def MyCrossAttnDownBlock2D_SD_forward(self, x):
    for i in range(2):
        x = self.resnets[i](x)
        # print(x.shape)
        x = self.attentions[i](x)
        skip.append(x)
    if self.downsamplers is not None:
        x = self.downsamplers[0](x)
        skip.append(x)
    return x

def MyCrossAttnUpBlock2D_SD_forward(self, x):
    for i in range(3):
        x = self.resnets[i](torch.cat([x, skip.pop()], dim=1))
        x = self.attentions[i](x)
    if self.upsamplers is not None:
        x = self.upsamplers[0](x)
    return x

def MyDownBlock2D_SD_forward(self, x):
    for i in range(2):
        x = self.resnets[i](x)
        skip.append(x)
    return x

def MyUNetMidBlock2DCrossAttn_SD_forward(self, x):
    x = self.resnets[0](x)
    x = self.attentions[0](x)
    x = self.resnets[1](x)
    return x

def MyUpBlock2D_SD_forward(self, x):
    # import pdb;pdb.set_trace()
    for i in range(3):
        x = self.resnets[i](torch.cat([x, skip.pop()], dim=1))
    x = self.upsamplers[0](x)
    return x

def MyResnetBlock2D_SD_forward(self, x_in):
    x = self.norm1(x_in)
    x = self.nonlinearity(x)
    x = self.conv1(x)
    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.conv2(x)
    if self.in_channels == self.out_channels:
        return x + x_in
    return x + self.conv_shortcut(x_in)

def MyTransformer2DModel_SD_forward(self, x_in):
    b, c, h, w = x_in.shape
    x = self.norm(x_in)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    x = self.proj_in(x)
    for block in self.transformer_blocks:
        x = x + block.attn1(block.norm1(x))
        x = x + block.ff(block.norm3(x))
    x = self.proj_out(x)
    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x + x_in


# =============================================================================
# TAESD Decoder Components
# =============================================================================

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out, use_midblock_gn=False):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            conv1x1, n_gn = lambda n_in, n_out: nn.Conv2d(n_in, n_out, 1, bias=False), n_in*4
            self.pool = nn.Sequential(conv1x1(n_in, n_gn), nn.GroupNorm(4, n_gn), nn.ReLU(inplace=True), conv1x1(n_gn, n_in))
    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))

def Decoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


# =============================================================================
# EightLayerConv: 8-layer convolution network with residual connection
# =============================================================================

class EightLayerConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(EightLayerConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.silu1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.silu2 = nn.SiLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.silu3 = nn.SiLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.silu4 = nn.SiLU(inplace=True)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.silu5 = nn.SiLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.silu6 = nn.SiLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 128, 3, padding=1)
        self.silu7 = nn.SiLU(inplace=True)
        self.conv8 = nn.Conv2d(128, out_channels, 3, padding=1)
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.silu1(x)
        x = self.conv2(x)
        x = self.silu2(x)
        x = self.conv3(x)
        x = self.silu3(x)
        x = self.conv4(x)
        x = self.silu4(x)
        x = self.conv5(x)
        x = self.silu5(x)
        x = self.conv6(x)
        x = self.silu6(x)
        x = self.conv7(x)
        x = self.silu7(x)
        x = self.conv8(x)
        return x + identity


# =============================================================================
# Net: SR model based on modified UNet
# =============================================================================

from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, \
                                                  CrossAttnUpBlock2D, \
                                                  DownBlock2D, \
                                                  UpBlock2D, \
                                                  UNetMidBlock2DCrossAttn
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D

# from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, \
#                                                   CrossAttnUpBlock2D, \
#                                                   DownBlock2D, \
#                                                   UpBlock2D, \
#                                                   UNetMidBlock2DCrossAttn
# from diffusers.models.resnet import ResnetBlock2D
# from diffusers.models.transformer_2d import Transformer2DModel
# from diffusers.models.attention import BasicTransformerBlock
# from diffusers.models.downsampling import Downsample2D
# from diffusers.models.upsampling import Upsample2D


def find_parent(model, module_name):
    components = module_name.split(".")
    parent = model
    for comp in components[:-1]:
        parent = getattr(parent, comp)
    return parent, components[-1]

def halve_channels(model):
    for name, module in model.named_modules():
        if hasattr(module, "pruned"):
            continue
        if isinstance(module, nn.Conv2d):
            in_channels = int(module.in_channels * 0.5)
            out_channels = int(module.out_channels * 0.5)
            new_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=module.kernel_size,
                                 stride=module.stride,
                                 padding=module.padding,
                                 dilation=module.dilation,
                                 groups=module.groups,
                                 bias=module.bias is not None)
            with torch.no_grad():
                new_conv.weight.copy_(module.weight[:out_channels, :in_channels])
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias[:out_channels])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_conv)
            new_conv.pruned = True
        elif isinstance(module, nn.Linear):
            in_features = int(module.in_features * 0.5)
            out_features = int(module.out_features * 0.5)
            new_linear = nn.Linear(in_features=in_features,
                                   out_features=out_features,
                                   bias=module.bias is not None)
            with torch.no_grad():
                new_linear.weight.copy_(module.weight[:out_features, :in_features])
                if module.bias is not None:
                    new_linear.bias.copy_(module.bias[:out_features])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_linear)
            new_linear.pruned = True
        elif isinstance(module, nn.GroupNorm):
            num_channels = int(module.num_channels * 0.5)
            for num_groups in [32, 24, 16, 12, 8, 6, 4, 2, 1]:
                if num_channels % num_groups == 0:
                    break
            new_gn = nn.GroupNorm(num_groups=num_groups,
                                  num_channels=num_channels,
                                  eps=module.eps,
                                  affine=module.affine)
            with torch.no_grad():
                new_gn.weight.copy_(module.weight[:num_channels])
                new_gn.bias.copy_(module.bias[:num_channels])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_gn)
            new_gn.pruned = True
        elif isinstance(module, nn.LayerNorm):
            normalized_shape = int(module.normalized_shape[0] * 0.5)
            new_ln = nn.LayerNorm(normalized_shape, 
                                  eps=module.eps, 
                                  elementwise_affine=module.elementwise_affine)
            with torch.no_grad():
                new_ln.weight.copy_(module.weight[:normalized_shape])
                new_ln.bias.copy_(module.bias[:normalized_shape])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_ln)
            new_ln.pruned = True
        elif isinstance(module, Downsample2D) or isinstance(module, Upsample2D):
            module.channels = int(module.channels * 0.5)


class Net(nn.Module):
    """SR model based on modified UNet architecture"""
    def __init__(self, unet):
        super().__init__()
        del unet.time_embedding
        new_conv_in = nn.Conv2d(24, 320, 3, padding=1)
        new_conv_in.weight.data = unet.conv_in.weight.data.repeat(1, 6, 1, 1)
        new_conv_in.bias.data = unet.conv_in.bias.data
        unet.conv_in = new_conv_in
        new_conv_out = nn.Conv2d(320, 8, 3, padding=1)
        new_conv_out.weight.data = unet.conv_out.weight.data.repeat(2, 1, 1, 1)[:8]
        new_conv_out.bias.data = unet.conv_out.bias.data.repeat(2,)[:8]
        unet.conv_out = new_conv_out
        def ResnetBlock2D_remove_time_emb_proj(module):
            if isinstance(module, ResnetBlock2D):
                del module.time_emb_proj
        unet.apply(ResnetBlock2D_remove_time_emb_proj)
        def BasicTransformerBlock_remove_cross_attn(module):
            if isinstance(module, BasicTransformerBlock):
                del module.attn2, module.norm2
        unet.apply(BasicTransformerBlock_remove_cross_attn)
        def set_inplace_to_true(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True
        unet.apply(set_inplace_to_true)
        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(MyCrossAttnDownBlock2D_SD_forward, module)
            elif isinstance(module, DownBlock2D):
                module.forward = types.MethodType(MyDownBlock2D_SD_forward, module)
            elif isinstance(module, UNetMidBlock2DCrossAttn):
                module.forward = types.MethodType(MyUNetMidBlock2DCrossAttn_SD_forward, module)
            elif isinstance(module, UpBlock2D):
                module.forward = types.MethodType(MyUpBlock2D_SD_forward, module)
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(MyCrossAttnUpBlock2D_SD_forward, module)
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(MyResnetBlock2D_SD_forward, module)
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(MyTransformer2DModel_SD_forward, module)
        unet.apply(replace_forward_methods)
        unet.forward = types.MethodType(MyUNet2DConditionModel_SD_forward, unet)
        halve_channels(unet)
        unet.body = nn.Sequential(
            *unet.down_blocks,
            unet.mid_block,
            *unet.up_blocks,
            unet.conv_norm_out,
            unet.conv_act,
            unet.conv_out,
        )
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),        
            unet
        )
    
    def forward(self, x):
        return self.body(x)


# =============================================================================
# SRNet: Complete Super-Resolution Network (Net + EightLayerConv + TAESD Decoder)
# =============================================================================

class SRNet(nn.Module):
    """
    Complete Super-Resolution Network
    Combines: Net (UNet-based SR) + EightLayerConv (refinement) + TAESD Decoder (latent to image)
    
    All weights can be loaded from a single combined checkpoint or from separate checkpoints.
    """
    def __init__(self, unet=None, latent_channels=4):
        super().__init__()
        
        # Store unet reference for Net initialization
        self._unet = unet
        self.latent_channels = latent_channels
        
        # These will be initialized when unet is provided
        self.net = None
        self.refine = None
        self.decoder = None
        
    def _initialize_modules(self, unet, checkpoint_path):
        """Initialize all modules when unet is available"""
        if self.net is not None:
            return  # Already initialized
            
        # Main SR network (based on UNet)
        self.net = Net(unet)
        
        # 8-layer convolution for refinement
        self.refine = EightLayerConv(in_channels=4, out_channels=4)
        
        # TAESD decoder for latent to image conversion
        self.decoder = Decoder(latent_channels=self.latent_channels)
        self.load_weights(checkpoint_path)
    
    def load_weights(self, checkpoint_path=None):
        """
        Load weights from a single combined checkpoint or from separate checkpoints.
        
        Args:
            checkpoint_path: Path to combined checkpoint containing all weights
            net_path: Path to Net weights (best.pth)
            conv_path: Path to EightLayerConv weights (best_conv8_v2.pth)
            decoder_path: Path to TAESD decoder weights (taesd_decoder.pth)
        """
        if checkpoint_path is not None:
            # Load combined checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.net.load_state_dict(checkpoint['net'])
            self.refine.load_state_dict(checkpoint['refine'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            print(f"Loaded combined weights from {checkpoint_path}")
    
    def save_combined_weights(self, save_path):
        """Save all weights to a single combined checkpoint"""
        checkpoint = {
            'net': self.net.state_dict(),
            'refine': self.refine.state_dict(),
            'decoder': self.decoder.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"Saved combined weights to {save_path}")
    
    def forward(self, x):
        """
        Forward pass: LR image -> SR image
        
        Args:
            x: Input LR image tensor [B, 3, H, W] in range [0, 1]
        
        Returns:
            SR image tensor [B, 3, 4*H, 4*W] in range [0, 1]
        """
        # Net: SR in latent space
        z0 = self.net(x)  # Output: [B, 4, H/4, W/4] latent
        
        # EightLayerConv: Refine latent
        z0_refined = self.refine(z0)
        
        # TAESD Decoder: latent to image (4x upscale)
        sr_image = self.decoder(z0_refined)
        
        return sr_image.clamp(0, 1)


def create_srnet(unet, checkpoint_path):
    """
    Factory function to create and initialize SRNet.
    
    Args:
        unet: Pretrained UNet2DConditionModel
        checkpoint_path: Path to combined checkpoint
        net_path, conv_path, decoder_path: Paths to separate checkpoints
    
    Returns:
        SRNet model ready for inference
    """
    model = SRNet(unet=unet)
    model._initialize_modules(unet, checkpoint_path)  

    
    return model