import torch
from basicsr.archs import build_network


def build_model(device: torch.device | str = "cuda"):
    net_opt = dict(
        type="PLKSR_Rep",
        dim=64,
        n_blocks=12,
        upscaling_factor=4,
        ccm_type="DCCM",
        kernel_size=17,
        split_ratio=0.25,
        use_ea=True,
    )
    net = build_network(net_opt)
    net = net.to(device)
    net.eval()
    return net
