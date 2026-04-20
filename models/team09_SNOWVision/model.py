"""SNOWVision Mobile SR Network — x4 Super-Resolution for NNAPI Deployment.

Hybrid-Group backbone with high-frequency bias extraction and gated artifact
suppression. All operators target Android NNAPI FP16 compatibility.

Architecture (1.13M parameters, <160 ms on Dimensity 8400):
    Input  -> ShallowExtract -> 3x ResidualGroups (80 blocks total)
           -> DenseRefiner + ChannelGate -> ArtifactSuppressor (4 blocks)
           -> HighFrequencyBias (2 blocks) -> Upsampler (PS x2+x2)
           + GlobalSkip -> Output
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReparamDepthwise(nn.Module):
    """Multi-branch depthwise conv fused to single 3x3 at export."""

    def __init__(self, channels: int, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.channels = channels
        if deploy:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=True)
        else:
            self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1,
                                     groups=channels, bias=True)
            self.conv1x1 = nn.Conv2d(channels, channels, 1,
                                     groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.conv(x)
        return self.conv3x3(x) + self.conv1x1(x) + x

    def fuse(self):
        if self.deploy:
            return
        w3 = self.conv3x3.weight.data.clone()
        b3 = self.conv3x3.bias.data.clone()
        w1 = F.pad(self.conv1x1.weight.data.clone(), [1, 1, 1, 1])
        b1 = self.conv1x1.bias.data.clone()
        identity = torch.zeros_like(w3)
        for i in range(self.channels):
            identity[i, 0, 1, 1] = 1.0
        self.deploy = True
        self.conv = nn.Conv2d(self.channels, self.channels, 3, padding=1,
                              groups=self.channels, bias=True)
        self.conv.weight.data = w3 + w1 + identity
        self.conv.bias.data = b3 + b1
        del self.conv3x3, self.conv1x1


class GatedRefinementBlock(nn.Module):
    """Gated convolution block for artifact suppression (NAFNet-style)."""

    def __init__(self, dim: int, use_reparam: bool = True, deploy: bool = False):
        super().__init__()
        self.dw = (ReparamDepthwise(dim, deploy) if use_reparam
                   else nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True))
        self.pw_feat = nn.Conv2d(dim, dim, 1, bias=True)
        self.pw_gate = nn.Conv2d(dim, dim, 1, bias=True)
        self.pw_out = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw(x)
        return shortcut + self.pw_out(self.pw_feat(x) * torch.sigmoid(self.pw_gate(x)))


class HighFrequencyBias(nn.Module):
    """Extracts high-frequency detail via local-minus-smooth decomposition."""

    def __init__(self, dim: int, use_reparam: bool = True, deploy: bool = False):
        super().__init__()
        self.dw_local = (ReparamDepthwise(dim, deploy) if use_reparam
                         else nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True))
        self.pw_local = nn.Conv2d(dim, dim, 1, bias=True)
        self.act = nn.ReLU6()
        self.dw_smooth = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                              dtype=torch.float32) / 16.0
        self.dw_smooth.weight.data = kernel.unsqueeze(0).unsqueeze(0).expand(
            dim, 1, 3, 3).clone()
        self.dw_smooth.weight.requires_grad = False
        self.pw_out = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.act(self.pw_local(self.dw_local(x)))
        high_freq = local_feat - self.dw_smooth(x)
        return x + self.pw_out(high_freq)


class ResidualBlock(nn.Module):
    """Depthwise-separable residual block with inverted bottleneck."""

    def __init__(self, dim: int, kernel_size: int = 3, expansion: int = 2):
        super().__init__()
        hidden = dim * expansion
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2,
                            groups=dim, bias=True)
        self.pw1 = nn.Conv2d(dim, hidden, 1, bias=True)
        self.act = nn.ReLU6()
        self.pw2 = nn.Conv2d(hidden, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(self.act(self.pw1(self.dw(x))))


class ResidualGroup(nn.Module):
    """Stack of residual blocks with a tail projection."""

    def __init__(self, dim: int, depth: int, kernel_size: int = 3, expansion: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(dim, kernel_size, expansion) for _ in range(depth)])
        self.conv_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.conv_pw = nn.Conv2d(dim, dim, 1, bias=True)
        self.conv_act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_pw(self.conv_act(self.conv_dw(self.blocks(x))))


class ChannelGate(nn.Module):
    """Lightweight squeeze-and-excitation channel attention."""

    def __init__(self, dim: int, reduction: int = 4, depth: int = 2):
        super().__init__()
        mid = max(dim // reduction, 8)
        layers = [nn.Conv2d(dim, mid, 1, bias=True), nn.ReLU6()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(mid, mid, 1, bias=True), nn.ReLU6()]
        layers.append(nn.Conv2d(mid, dim, 1, bias=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.body(x.mean(dim=(2, 3), keepdim=True)))


def _icnr_init(conv: nn.Conv2d, scale: int = 2):
    with torch.no_grad():
        sub = conv.weight.shape[0] // (scale * scale)
        nn.init.kaiming_normal_(conv.weight[:sub])
        for i in range(1, scale * scale):
            conv.weight[i * sub:(i + 1) * sub] = conv.weight[:sub]


class PixelShuffleUpsampler(nn.Module):
    """Cascaded x2+x2 PixelShuffle upsampler."""

    def __init__(self, dim: int, out_channels: int = 3):
        super().__init__()
        self.ps_conv1 = nn.Conv2d(dim, dim * 4, 3, padding=1, bias=True)
        self.ps1 = nn.PixelShuffle(2)
        self.act1 = nn.ReLU6()
        self.ps_conv2 = nn.Conv2d(dim, dim * 4, 3, padding=1, bias=True)
        self.ps2 = nn.PixelShuffle(2)
        self.act2 = nn.ReLU6()
        self.c_out = nn.Conv2d(dim, out_channels, 3, padding=1, bias=True)
        _icnr_init(self.ps_conv1, 2)
        _icnr_init(self.ps_conv2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.ps1(self.ps_conv1(x)))
        x = self.act2(self.ps2(self.ps_conv2(x)))
        return self.c_out(x)


@dataclass
class NetworkConfig:
    in_channels: int = 3
    embed_dim: int = 48
    depths: Tuple[int, ...] = (24, 34, 22)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    expansion: int = 2
    upscale: int = 4
    num_refinement_blocks: int = 4
    num_hf_blocks: int = 2
    use_reparam: bool = True
    deploy: bool = False


class MobileHGSR(nn.Module):
    def __init__(self, cfg: NetworkConfig = NetworkConfig()):
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim

        self.conv_first = nn.Conv2d(cfg.in_channels, dim, 3, padding=1, bias=True)

        self.groups = nn.ModuleList([
            ResidualGroup(dim, depth=d, kernel_size=k, expansion=cfg.expansion)
            for d, k in zip(cfg.depths, cfg.kernel_sizes)
        ])

        self.conv_after = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=True), nn.ReLU6(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True), nn.ReLU6(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True), nn.ReLU6(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True), nn.ReLU6(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True), nn.ReLU6(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
        )
        self.se_gate = ChannelGate(dim, reduction=4, depth=4)

        self.naf_tail = nn.ModuleList([
            GatedRefinementBlock(dim, cfg.use_reparam, cfg.deploy)
            for _ in range(cfg.num_refinement_blocks)
        ])

        self.lhfb_blocks = nn.ModuleList([
            HighFrequencyBias(dim, cfg.use_reparam, cfg.deploy)
            for _ in range(cfg.num_hf_blocks)
        ])

        self.upsampler = PixelShuffleUpsampler(dim, cfg.in_channels)

        self.skip_conv1 = nn.Conv2d(cfg.in_channels, cfg.in_channels * 4, 1, bias=True)
        self.skip_ps1 = nn.PixelShuffle(2)
        self.skip_conv2 = nn.Conv2d(cfg.in_channels, cfg.in_channels * 4, 1, bias=True)
        self.skip_ps2 = nn.PixelShuffle(2)
        with torch.no_grad():
            for sc in [self.skip_conv1, self.skip_conv2]:
                sc.weight.zero_()
                for c in range(cfg.in_channels):
                    sc.weight[c * 4:(c + 1) * 4, c, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.skip_ps2(self.skip_conv2(self.skip_ps1(self.skip_conv1(x))))
        feat = self.conv_first(x)
        for group in self.groups:
            feat = group(feat)
        feat = self.se_gate(self.conv_after(feat)) + feat
        for block in self.naf_tail:
            feat = block(feat)
        for block in self.lhfb_blocks:
            feat = block(feat)
        return self.upsampler(feat) + base

    def fuse_reparam(self):
        for block in self.naf_tail:
            if isinstance(block.dw, ReparamDepthwise):
                block.dw.fuse()
        for block in self.lhfb_blocks:
            if isinstance(block.dw_local, ReparamDepthwise):
                block.dw_local.fuse()


def mobilehgsr(deploy: bool = False) -> MobileHGSR:
    return MobileHGSR(NetworkConfig(deploy=deploy))
