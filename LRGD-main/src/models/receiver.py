"""
Channel-Aware Receiver for CA-LRGD.

Drop-in replacement for src/models/receiver.py in the original LRGD project.
Innovation points:
1) Receiver reads channel_snr_db injected by Channel.
2) Dynamically adjusts low-rank rank according to channel reliability.
3) Dynamically adjusts low-rank guidance start, text guidance, and ControlNet scale.
"""

import os
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from utils.image_utils import inpaint_image
from utils.walsh_cs_utils import walsh_hadamard_decode

from .third_party_models.guide_sdcnp import LowRankGuidedSDCNP
from .third_party_models.load_sd_pipe import load_sd_pipe
from .traditional.interpolate import interpolate_sparse_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def _cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
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


def _snr_reliability(snr_db: Optional[float], low_snr: float, high_snr: float) -> float:
    if snr_db is None:
        return 1.0
    if high_snr <= low_snr:
        return 1.0
    return float(np.clip((float(snr_db) - low_snr) / (high_snr - low_snr), 0.0, 1.0))


class ChannelAwareLRGDPolicy:
    """
    Small deterministic policy for channel-aware diffusion.

    Design intuition:
    - Low SNR: the sparse guide is noisy, so use a lower rank to suppress corrupted
      high-frequency details, but rely more on protected text/edge constraints.
    - High SNR: use a higher rank and stronger visual guide to preserve details.
    """

    def __init__(
        self,
        low_snr: float = 5.0,
        high_snr: float = 25.0,
        low_rank_scale: float = 0.60,
        high_rank_scale: float = 1.25,
        low_guidance_scale: float = 0.55,
        high_guidance_scale: float = 1.00,
        low_control_scale: float = 1.25,
        high_control_scale: float = 1.00,
        low_text_guidance: float = 8.5,
        high_text_guidance: float = 7.5,
    ):
        self.low_snr = low_snr
        self.high_snr = high_snr
        self.low_rank_scale = low_rank_scale
        self.high_rank_scale = high_rank_scale
        self.low_guidance_scale = low_guidance_scale
        self.high_guidance_scale = high_guidance_scale
        self.low_control_scale = low_control_scale
        self.high_control_scale = high_control_scale
        self.low_text_guidance = low_text_guidance
        self.high_text_guidance = high_text_guidance

    def reliability(self, snr_db: Optional[float]) -> float:
        return _snr_reliability(snr_db, self.low_snr, self.high_snr)

    def rank(self, base_rank: Optional[int], snr_db: Optional[float]) -> Optional[int]:
        if base_rank is None:
            return None
        rel = self.reliability(snr_db)
        scale = self.low_rank_scale + rel * (self.high_rank_scale - self.low_rank_scale)
        return max(1, int(round(float(base_rank) * scale)))

    def guidance_sampling_rate(self, sampling_ratio: float, snr_db: Optional[float]) -> float:
        rel = self.reliability(snr_db)
        scale = self.low_guidance_scale + rel * (self.high_guidance_scale - self.low_guidance_scale)
        return float(np.clip(float(sampling_ratio) * scale, 0.001, 1.0))

    def controlnet_scale(self, snr_db: Optional[float]) -> float:
        rel = self.reliability(snr_db)
        return float(self.low_control_scale + rel * (self.high_control_scale - self.low_control_scale))

    def text_guidance_scale(self, snr_db: Optional[float]) -> float:
        rel = self.reliability(snr_db)
        return float(self.low_text_guidance + rel * (self.high_text_guidance - self.low_text_guidance))


class Receiver(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Receiver, self).__init__()
        self.method: str = cfg.method
        if self.method == "sd":
            self.num_inference_step: int = cfg.stable_diffusion.num_inference_step or 20
            self.rank: int = cfg.stable_diffusion.rank
            self.info_percent: float = cfg.stable_diffusion.info_percent
            self.rank_percent: float = cfg.stable_diffusion.rank_percent
            self.ortho_projection: bool = cfg.stable_diffusion.ortho_projection
            self.smooth_transition: bool = cfg.stable_diffusion.smooth_transition
            self.generator = torch.Generator(device="cpu").manual_seed(1)
            logger.info(f"Number of inference steps: {self.num_inference_step}")
            logger.info(f"Base rank: {self.rank}")
        else:
            logger.info(f"*** Receiver method: {self.method} ***")

        # CA-LRGD receiver adaptation settings. Missing fields use safe defaults.
        self.channel_aware_enable: bool = bool(_cfg_get(cfg, "channel_aware.enable", True))
        self.ca_policy = ChannelAwareLRGDPolicy(
            low_snr=float(_cfg_get(cfg, "channel_aware.low_snr", 5.0)),
            high_snr=float(_cfg_get(cfg, "channel_aware.high_snr", 25.0)),
            low_rank_scale=float(_cfg_get(cfg, "channel_aware.low_rank_scale", 0.60)),
            high_rank_scale=float(_cfg_get(cfg, "channel_aware.high_rank_scale", 1.25)),
            low_guidance_scale=float(_cfg_get(cfg, "channel_aware.low_guidance_scale", 0.55)),
            high_guidance_scale=float(_cfg_get(cfg, "channel_aware.high_guidance_scale", 1.00)),
            low_control_scale=float(_cfg_get(cfg, "channel_aware.low_control_scale", 1.25)),
            high_control_scale=float(_cfg_get(cfg, "channel_aware.high_control_scale", 1.00)),
            low_text_guidance=float(_cfg_get(cfg, "channel_aware.low_text_guidance", 8.5)),
            high_text_guidance=float(_cfg_get(cfg, "channel_aware.high_text_guidance", 7.5)),
        )
        logger.info(f"CA-LRGD receiver channel-aware enable: {self.channel_aware_enable}")

        # Load the model.
        if cfg.stable_diffusion.enable:
            load_pipe = {
                "sd1.5": load_sd_pipe,
            }[cfg.stable_diffusion.model_name]
            self.pipe: LowRankGuidedSDCNP = load_pipe(
                method=cfg.stable_diffusion.method,
                use_lcm_lora=cfg.stable_diffusion.use_lcm_lora,
            )
            logger.info(f"Loaded model: {cfg.stable_diffusion.model_name}")
        else:
            self.pipe = None

    def freeze_pipe(self):
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.controlnet.requires_grad_(False)

    def _get_channel_snr(self, transmitter_output: dict) -> Optional[float]:
        snr = transmitter_output.get("channel_snr_db", None)
        if snr is None:
            meta = transmitter_output.get("adaptive_meta", {}) or {}
            snr = meta.get("channel_snr_db", None)
        try:
            return None if snr is None else float(snr)
        except Exception:
            return None

    def forward(self, transmitter_output, get_inpaint=False):
        if isinstance(transmitter_output, list):
            return [self.forward(i, get_inpaint) for i in transmitter_output]

        times: dict = {}

        trans_method: str = transmitter_output["trans_method"]
        clip_desc: str = transmitter_output["clip_description"]
        clip_neg_desc: str = transmitter_output["clip_neg_description"]
        contour: Image.Image = transmitter_output["contour"]
        snr_db: Optional[float] = self._get_channel_snr(transmitter_output)

        if trans_method == "pixel":
            sampled_pixels: Image.Image = transmitter_output["sampled_pixels"]
            sampling_ratio: float = 1 - transparent_pixel_ratio(sampled_pixels)
            w, h = sampled_pixels.width, sampled_pixels.height
        elif trans_method == "jpeg":
            low_quality_image: Image.Image = transmitter_output["image"]
            sampling_ratio = transmitter_output["compression_ratio"]
            w, h = low_quality_image.width, low_quality_image.height
        elif trans_method == "walsh":
            walsh_metadata = transmitter_output["walsh_metadata"]
            low_quality_image = walsh_hadamard_decode(
                walsh_metadata["compressed_data"],
                walsh_metadata,
            )
            sampling_ratio = walsh_metadata["sampling_ratio"]
            h, w = walsh_metadata["original_size"]
        else:
            raise ValueError(f"Unknown trans_method: {trans_method}")

        if self.method == "bi":  # bilinear interpolation
            assert (
                trans_method == "pixel"
            ), "Only pixel cs mode is supported for Bilinear Interpolation"
            image = low_quality_image = interpolate_sparse_image(sampled_pixels)

        elif self.method == "jpeg":  # jpeg compression
            assert trans_method == "jpeg", "Only jpeg cs mode is supported for JPEG"
            image = low_quality_image

        elif self.method == "walsh":
            assert (
                trans_method == "walsh"
            ), "Only walsh cs mode is supported for Walsh Hadamard"
            image = low_quality_image

        elif self.method == "sd":
            if trans_method == "pixel":
                start_time = time.time()
                low_quality_image = inpaint_image(sampled_pixels)
                times["pre-inpaint"] = time.time() - start_time

            if self.pipe is not None:
                start_time = time.time()
                with torch.autocast("cuda"):
                    if self.channel_aware_enable:
                        adaptive_rank = self.ca_policy.rank(self.rank, snr_db)
                        adaptive_sampling_rate = self.ca_policy.guidance_sampling_rate(
                            sampling_ratio, snr_db
                        )
                        control_scale = self.ca_policy.controlnet_scale(snr_db)
                        text_guidance_scale = self.ca_policy.text_guidance_scale(snr_db)
                    else:
                        adaptive_rank = self.rank
                        adaptive_sampling_rate = sampling_ratio
                        control_scale = 1.0
                        text_guidance_scale = 7.5

                    times["channel_snr_db"] = -1 if snr_db is None else snr_db
                    times["adaptive_rank"] = -1 if adaptive_rank is None else adaptive_rank
                    times["adaptive_sampling_rate"] = adaptive_sampling_rate
                    times["controlnet_scale"] = control_scale
                    times["text_guidance_scale"] = text_guidance_scale

                    logger.info(
                        "CA-LRGD receiver policy: snr={}, rank={}, guide_start={:.4f}, control_scale={:.3f}, text_guidance={:.3f}".format(
                            snr_db,
                            adaptive_rank,
                            adaptive_sampling_rate,
                            control_scale,
                            text_guidance_scale,
                        )
                    )

                    if isinstance(self.pipe, LowRankGuidedSDCNP):
                        image = self.pipe(
                            clip_desc or "",
                            negative_prompt=clip_neg_desc or "",
                            guidance_scale=text_guidance_scale,
                            rank=adaptive_rank,
                            info_percent=self.info_percent,
                            rank_percent=self.rank_percent,
                            ortho_projection=self.ortho_projection,
                            smooth_transition=self.smooth_transition,
                            num_inference_steps=self.num_inference_step,
                            generator=self.generator,
                            image=contour,
                            guide_image=low_quality_image,
                            sampling_rate=adaptive_sampling_rate,
                            controlnet_conditioning_scale=control_scale,
                        ).images[0]
                    else:
                        assert (
                            trans_method == "pixel"
                        ), "Only pixel cs mode is supported for this model"
                        # reverse the mask
                        mask = Image.eval(
                            sampled_pixels.split()[-1], lambda a: 255 if a == 0 else 0
                        ).convert("L")
                        image = self.pipe(
                            clip_desc or "",
                            negative_prompt=clip_neg_desc or "",
                            guidance_scale=text_guidance_scale,
                            num_inference_steps=self.num_inference_step,
                            generator=self.generator,
                            image=low_quality_image,
                            mask_image=mask,
                            control_image=contour,
                            controlnet_conditioning_scale=control_scale,
                        ).images[0]

                    image = image.resize((w, h))

                times["diffusion"] = time.time() - start_time
            else:
                image = low_quality_image
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if get_inpaint:
            return image, low_quality_image, times
        return image, times


def transparent_pixel_ratio(image: Image.Image) -> float:
    if image.mode != "RGBA":
        raise ValueError("Input image must be in RGBA format")

    image_array = np.array(image)
    alpha_channel = image_array[:, :, 3]
    transparent_pixels = np.sum(alpha_channel == 0)
    total_pixels = alpha_channel.size
    ratio = transparent_pixels / total_pixels

    return ratio
