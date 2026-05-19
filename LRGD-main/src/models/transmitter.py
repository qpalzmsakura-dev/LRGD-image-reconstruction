"""
Channel-Aware Transmitter for CA-LRGD.

Drop-in replacement for src/models/transmitter.py in the original LRGD project.
Innovation points:
1) Channel-aware adaptive sampling ratio.
2) Saliency + edge joint importance sampling mask.
3) Metadata output for receiver-side adaptive low-rank diffusion.

The original public APIs are preserved: Transmitter(cfg)(image) still works.
For stronger transmitter-side adaptation, call Transmitter(cfg)(image, channel_snr=cfg.channel.snr)
if your main.py can pass the channel SNR.
"""

import os
import sys
import time
from typing import Any, Optional

import cv2
import numpy as np
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from scipy.ndimage import sobel

from utils.image_utils import canny, compress_image_to_quality_jpeg
from utils.mask_utils import apply_mask_to_image, generate_saliency_based_sampling_mask
from utils.walsh_cs_utils import walsh_hadamard_encode

from .third_party_models.clip_ci import load_ci_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def _cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    """Safely read a nested key from OmegaConf/DictConfig/dict/object."""
    cur = cfg
    for key in dotted_key.split("."):
        try:
            if isinstance(cur, dict):
                cur = cur[key]
            else:
                cur = getattr(cur, key)
        except Exception:
            return default
    return cur


def _normalize_map(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64)
    min_v, max_v = float(np.min(x)), float(np.max(x))
    if max_v - min_v < eps:
        return np.ones_like(x, dtype=np.float64)
    return (x - min_v) / (max_v - min_v + eps)


def _snr_reliability(snr_db: Optional[float], low_snr: float, high_snr: float) -> float:
    """
    Map SNR to [0, 1]. 0 means very unreliable channel; 1 means reliable channel.
    """
    if snr_db is None:
        return 1.0
    if high_snr <= low_snr:
        return 1.0
    return float(np.clip((float(snr_db) - low_snr) / (high_snr - low_snr), 0.0, 1.0))


def generate_channel_aware_sampling_mask(
    image: Image.Image,
    sampling_rate: float,
    snr_db: Optional[float] = None,
    low_snr: float = 5.0,
    high_snr: float = 25.0,
    seed: Optional[int] = None,
) -> tuple[Image.Image, dict]:
    """
    Generate an importance mask by combining spectral saliency and edge/gradient cues.

    Return convention follows the original LRGD mask_utils implementation:
    - black pixels are kept/transmitted;
    - white pixels are removed/masked.
    """
    if not (0 <= sampling_rate <= 1):
        raise ValueError("sampling_rate must be between 0 and 1")

    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    reliability = _snr_reliability(snr_db, low_snr, high_snr)
    stress = 1.0 - reliability  # larger under poor channel condition

    # Saliency map: same spectral residual idea as the baseline, with safe fallback.
    try:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(img_bgr)
        if not success:
            raise RuntimeError("cv2 saliency failed")
        saliency_map = _normalize_map(saliency_map)
    except Exception:
        # Fallback: luminance gradient map.
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gx = sobel(gray, axis=0)
        gy = sobel(gray, axis=1)
        saliency_map = _normalize_map(np.sqrt(gx**2 + gy**2))

    # Edge / structure map. Under poor SNR, we bias the sparse pixels towards structure.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edge_map = cv2.Canny(gray, 80, 180).astype(np.float64) / 255.0
    gx = sobel(gray, axis=0)
    gy = sobel(gray, axis=1)
    gradient_map = _normalize_map(np.sqrt(gx**2 + gy**2))

    saliency_weight = 0.65 - 0.20 * stress
    edge_weight = 0.25 + 0.25 * stress
    gradient_weight = 0.10 + 0.10 * stress
    uniform_floor = 0.02

    score = (
        saliency_weight * saliency_map
        + edge_weight * edge_map
        + gradient_weight * gradient_map
        + uniform_floor
    )
    score = np.maximum(score, 1e-12)
    prob = score / np.sum(score)

    num_samples = int(round(float(sampling_rate) * h * w))
    num_samples = max(1, min(num_samples, h * w))
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(h * w, size=num_samples, replace=False, p=prob.reshape(-1))

    sampling_mask = np.zeros((h, w), dtype=bool)
    sampling_mask.flat[sampled_indices] = True

    # Original apply_mask_to_image keeps black and removes white.
    remove_mask = np.where(sampling_mask, 0, 255).astype(np.uint8)
    meta = {
        "sampling_rate": float(sampling_rate),
        "channel_snr_db": None if snr_db is None else float(snr_db),
        "channel_reliability": reliability,
        "saliency_weight": float(saliency_weight),
        "edge_weight": float(edge_weight),
        "gradient_weight": float(gradient_weight),
        "num_samples": int(num_samples),
    }
    return Image.fromarray(remove_mask, mode="L"), meta


class Transmitter(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Transmitter, self).__init__()
        self.cs_method: str = cfg.cs_method
        self.sampling_rate: float = float(cfg.sampling.rate)
        self.if_enable_clip: bool = bool(cfg.clip_model.enable)
        self.if_enable_clip_neg: bool = bool(cfg.clip_model.enable_negative)
        self.if_enable_contour: bool = bool(cfg.contour.enable)
        self.canny_threshold1: float = cfg.contour.canny_threshold1
        self.canny_threshold2: float = cfg.contour.canny_threshold2
        self.ci_fast_mode: bool = bool(cfg.clip_model.fast_mode)

        # CA-LRGD adaptive transmitter settings. Missing config fields use safe defaults.
        self.adaptive_enable: bool = bool(_cfg_get(cfg, "adaptive.enable", True))
        self.adaptive_rate_mode: str = str(_cfg_get(cfg, "adaptive.rate_mode", "snr_inverse"))
        self.low_snr: float = float(_cfg_get(cfg, "adaptive.low_snr", 5.0))
        self.high_snr: float = float(_cfg_get(cfg, "adaptive.high_snr", 25.0))
        self.min_sampling_rate: float = float(
            _cfg_get(cfg, "adaptive.min_sampling_rate", max(0.01, self.sampling_rate * 0.70))
        )
        self.max_sampling_rate: float = float(
            _cfg_get(cfg, "adaptive.max_sampling_rate", min(0.50, self.sampling_rate * 1.50))
        )
        self.mask_seed: Optional[int] = _cfg_get(cfg, "adaptive.mask_seed", None)
        self.default_channel_snr: Optional[float] = _cfg_get(cfg, "adaptive.default_snr", None)
        self.last_adaptive_meta: dict = {}

        logger.info(f"CS method: {self.cs_method}")
        logger.info(f"Base sampling rate: {self.sampling_rate}")
        logger.info(
            "CA-LRGD adaptive transmitter: enable={}, rate_mode={}, min_rate={}, max_rate={}, low_snr={}, high_snr={}".format(
                self.adaptive_enable,
                self.adaptive_rate_mode,
                self.min_sampling_rate,
                self.max_sampling_rate,
                self.low_snr,
                self.high_snr,
            )
        )

        # Load the captioning/interrogation model.
        if self.if_enable_clip:
            self.ci = load_ci_model(cfg.clip_model.model_name)

    def forward(self, image, channel_snr: Optional[float] = None) -> tuple[dict, dict]:
        """
        Existing calls Transmitter(cfg)(image) still work.
        New adaptive call: Transmitter(cfg)(image, channel_snr=snr_db).
        """
        if isinstance(image, list):
            return [self.forward(img, channel_snr=channel_snr) for img in image]

        if channel_snr is None:
            channel_snr = self.default_channel_snr

        times: dict = {}

        start_time = time.time()
        desc, neg_desc = self.generate_clip_description(image)
        times["text"] = time.time() - start_time

        start_time = time.time()
        contour = self.generate_contour(image)
        times["edge"] = time.time() - start_time

        effective_sampling_rate = self.get_effective_sampling_rate(channel_snr)
        adaptive_meta = {
            "channel_snr_db": None if channel_snr is None else float(channel_snr),
            "base_sampling_rate": float(self.sampling_rate),
            "effective_sampling_rate": float(effective_sampling_rate),
            "channel_reliability": _snr_reliability(channel_snr, self.low_snr, self.high_snr),
            "adaptive_enable": bool(self.adaptive_enable),
        }

        ret = {
            "trans_method": self.cs_method,
            "clip_description": desc,
            "clip_neg_description": neg_desc,
            "contour": contour,
            "channel_snr_db": channel_snr,
            "effective_sampling_rate": effective_sampling_rate,
            "adaptive_meta": adaptive_meta,
        }

        if self.cs_method == "pixel":
            start_time = time.time()
            ret["sampled_pixels"] = self.sample_pixels(image, channel_snr=channel_snr)
            ret["adaptive_meta"].update(self.last_adaptive_meta)
            times["sparse"] = time.time() - start_time
        elif self.cs_method == "jpeg":
            ret["image"] = compress_image_to_quality_jpeg(
                image, round(effective_sampling_rate * 100)
            )
            ret["compression_ratio"] = effective_sampling_rate
        elif self.cs_method == "walsh":
            compressed_data, metadata = walsh_hadamard_encode(image, effective_sampling_rate)
            metadata["compressed_data"] = compressed_data
            metadata["channel_snr_db"] = channel_snr
            metadata["adaptive_meta"] = adaptive_meta
            ret["walsh_metadata"] = metadata
        else:
            raise ValueError(f"Unknown CS method: {self.cs_method}")

        return ret, times

    def get_effective_sampling_rate(self, channel_snr: Optional[float] = None) -> float:
        if not self.adaptive_enable or self.adaptive_rate_mode == "fixed":
            return float(self.sampling_rate)

        reliability = _snr_reliability(channel_snr, self.low_snr, self.high_snr)
        if self.adaptive_rate_mode == "snr_inverse":
            # Poorer channel -> more sparse observations/redundancy.
            rate = self.max_sampling_rate - reliability * (self.max_sampling_rate - self.min_sampling_rate)
        elif self.adaptive_rate_mode == "snr_direct":
            # More reliable channel -> transmit more details.
            rate = self.min_sampling_rate + reliability * (self.max_sampling_rate - self.min_sampling_rate)
        else:
            rate = self.sampling_rate
        return float(np.clip(rate, 0.001, 1.0))

    def sample_pixels(self, image, channel_snr: Optional[float] = None):
        sampling_rate = self.get_effective_sampling_rate(channel_snr)
        if self.adaptive_enable:
            random_mask, meta = generate_channel_aware_sampling_mask(
                image,
                sampling_rate,
                snr_db=channel_snr,
                low_snr=self.low_snr,
                high_snr=self.high_snr,
                seed=self.mask_seed,
            )
            self.last_adaptive_meta = meta
        else:
            random_mask = generate_saliency_based_sampling_mask(image, sampling_rate)
            self.last_adaptive_meta = {"sampling_rate": float(sampling_rate)}
        masked_image = apply_mask_to_image(image, random_mask)
        return masked_image

    def generate_clip_description(self, image):
        if not self.if_enable_clip:
            return None, None

        if self.ci_fast_mode:
            desc = self.ci.interrogate_fast(image)
        else:
            desc = self.ci.interrogate_classic(image)
        neg_desc = (
            self.ci.interrogate_negative(image) if self.if_enable_clip_neg else None
        )
        return desc, neg_desc

    def generate_contour(self, image):
        if not self.if_enable_contour:
            return None

        contour = canny(image, self.canny_threshold1, self.canny_threshold2)
        contour = contour.convert("1")  # Convert to binary image
        return contour
