import glob
import json
import os
from types import SimpleNamespace

from .io_hdsr import hd_sr

_DEFAULTS = {
    "test_dir": None,
    "save_dir": None,
    "pretrained_model_path": "/data/wangcongyu/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06",
    "pretrained_path": "/data/wangcongyu/OSEDiff-main/NTIRE2026_Mobile_RealWorld_ImageSR-main/model_zoo/team02_er.pkl",
    "seed": 42,
    "process_size": 512,
    "upscale": 4,
    "align_method": "adain",
    "vae_decoder_tiled_size": 224,
    "vae_encoder_tiled_size": 1024,
    "latent_tiled_size": 256,
    "latent_tiled_overlap": 32,
    "mixed_precision": "fp16",
    "default": True,
}

_ENV_MAP = {
    "pretrained_model_path": "HDSR_PRETRAINED_MODEL_PATH",
    "pretrained_path": "HDSR_PRETRAINED_PATH",
    "mixed_precision": "HDSR_MIXED_PRECISION",
    "align_method": "HDSR_ALIGN_METHOD",
    "process_size": "HDSR_PROCESS_SIZE",
    "upscale": "HDSR_UPSCALE",
    "vae_decoder_tiled_size": "HDSR_VAE_DECODER_TILED_SIZE",
    "vae_encoder_tiled_size": "HDSR_VAE_ENCODER_TILED_SIZE",
    "latent_tiled_size": "HDSR_LATENT_TILED_SIZE",
    "latent_tiled_overlap": "HDSR_LATENT_TILED_OVERLAP",
    "default": "HDSR_DEFAULT",
}

_ENV_CAST = {
    "process_size": int,
    "upscale": int,
    "vae_decoder_tiled_size": int,
    "vae_encoder_tiled_size": int,
    "latent_tiled_size": int,
    "latent_tiled_overlap": int,
}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _load_config_file(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    if path.endswith(".yml") or path.endswith(".yaml"):
        try:
            import yaml
        except Exception:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_config(model_dir):
    if not model_dir:
        return {}
    if os.path.isfile(model_dir):
        if model_dir.endswith((".json", ".yml", ".yaml")):
            return _load_config_file(model_dir)
        return {}
    if os.path.isdir(model_dir):
        for name in ("config.json", "config.yml", "config.yaml"):
            path = os.path.join(model_dir, name)
            if os.path.isfile(path):
                return _load_config_file(path)
    return {}


def _resolve_pretrained_path(model_dir, current_value):
    if current_value and current_value != _DEFAULTS["pretrained_path"]:
        return current_value
    if model_dir:
        if os.path.isfile(model_dir):
            if model_dir.endswith((".json", ".yml", ".yaml")):
                return current_value
            return model_dir
        if os.path.isdir(model_dir):
            candidates = []
            candidates.extend(sorted(glob.glob(os.path.join(model_dir, "*.pkl"))))
            candidates.extend(sorted(glob.glob(os.path.join(model_dir, "pretrain", "*.pkl"))))
            if candidates:
                return candidates[0]
    local_fallback = os.path.join(os.path.dirname(__file__), "pretrain", "model_6001.pkl")
    if os.path.isfile(local_fallback):
        return local_fallback
    return current_value


def _apply_env_overrides(args_dict):
    for key, env_name in _ENV_MAP.items():
        if env_name in os.environ:
            value = os.environ[env_name]
            cast = _ENV_CAST.get(key)
            if cast is not None:
                try:
                    value = cast(value)
                except Exception:
                    pass
            if key == "default":
                value = _as_bool(value)
            args_dict[key] = value


def main(model_dir, input_path, output_path, device=None):
    args_dict = dict(_DEFAULTS)
    args_dict.update(_load_config(model_dir))
    _apply_env_overrides(args_dict)

    args_dict["test_dir"] = input_path
    args_dict["save_dir"] = output_path
    args_dict["default"] = _as_bool(args_dict.get("default", True))
    args_dict["pretrained_path"] = _resolve_pretrained_path(model_dir, args_dict.get("pretrained_path"))

    args = SimpleNamespace(**args_dict)
    hd_sr(args, device=device)
