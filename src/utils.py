"""Shared utilities for Dual-JEPA Split-Latent RetNet Singing Voice Separation."""

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from src.audio.stft import STFTProcessor
from src.models.retnet_jepa import SpectrogramRetNetEncoder, StemRetNetJEPA
from src.models.separator import SplitLatentRetNetUNet


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device(cfg_device: str = "cuda") -> torch.device:
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_stft_processor(cfg: Dict[str, Any]) -> STFTProcessor:
    a_cfg = cfg["audio"]
    return STFTProcessor(
        n_fft=a_cfg["n_fft"],
        hop_length=a_cfg["hop_length"],
        n_freq_bins=a_cfg["n_freq_bins"],
    )


def build_stem_jepa(cfg: Dict[str, Any], retnet_mode: str | None = None) -> StemRetNetJEPA:
    m_cfg = cfg["model"]
    a_cfg = cfg["audio"]
    j_cfg = cfg["jepa_pretrain"]
    mode = retnet_mode or m_cfg["retnet_mode"]

    enc = SpectrogramRetNetEncoder(
        n_freq_bins=a_cfg["n_freq_bins"],
        crop_frames=a_cfg["crop_frames"],
        patch_size=tuple(m_cfg["patch_size"]),
        d_model=m_cfg["d_model"],
        num_layers=m_cfg["num_layers"],
        num_heads=m_cfg["num_heads"],
        ffn_mult=m_cfg["ffn_mult"],
        mode=mode,
        dropout=m_cfg["dropout"],
    )
    return StemRetNetJEPA(
        encoder=enc,
        predictor_dim=m_cfg["predictor_dim"],
        predictor_layers=m_cfg["predictor_layers"],
        num_heads=m_cfg["num_heads"],
        mask_ratio=j_cfg["mask_ratio"],
        lambda_var=j_cfg["lambda_var"],
    )


def build_unet_separator(cfg: Dict[str, Any], retnet_mode: str | None = None) -> SplitLatentRetNetUNet:
    m_cfg = cfg["model"]
    a_cfg = cfg["audio"]
    mode = retnet_mode or m_cfg["retnet_mode"]
    return SplitLatentRetNetUNet(
        n_freq_bins=a_cfg["n_freq_bins"],
        crop_frames=a_cfg["crop_frames"],
        patch_size=tuple(m_cfg["patch_size"]),
        d_model=m_cfg["d_model"],
        num_layers=m_cfg["num_layers"],
        decoder_layers=m_cfg["decoder_layers"],
        num_heads=m_cfg["num_heads"],
        ffn_mult=m_cfg["ffn_mult"],
        unet_channels=m_cfg.get("unet_channels", 64),
        mode=mode,
        dropout=m_cfg["dropout"],
    )
