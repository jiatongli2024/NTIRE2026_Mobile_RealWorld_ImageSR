import os
import socket
import subprocess
from datetime import datetime
import torch

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"
def save_checkpoint(
    save_path,
    model,
    optimizer,
    lr_scheduler=None,
    epoch=0,
    step=0,
    train_args=None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_args": train_args or {},
        "meta": {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "git_commit": get_git_commit(),
        }
    }

    if lr_scheduler is not None:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()

    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Saved to {save_path}")