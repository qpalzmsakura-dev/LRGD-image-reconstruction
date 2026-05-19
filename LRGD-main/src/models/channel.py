"""
Channel with Unequal Error Protection (UEP) for CA-LRGD.

Drop-in replacement for src/models/channel.py in the original LRGD project.
Innovation points:
1) Adds channel_snr_db metadata to received packets.
2) Applies stream-specific SNR boost for text / edge / sparse streams.
3) Keeps the original Channel(cfg)(input) API.
"""

import io
import sys
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from utils.qam import qam16ModulationImage, qam16ModulationString, qam16ModulationTensor


# These keys are exempt from adding noise. They are control/metadata rather than payload.
noise_exempt_keys = [
    "trans_method",
    "compression_ratio",
    "original_size",
    "padded_shape",
    "block_size",
    "sampling_ratio",
    "n_channels",
    "blocks_h",
    "blocks_w",
    "channel_snr_db",
    "effective_sampling_rate",
    "adaptive_meta",
    "uep_meta",
]

TEXT_KEYS = {"clip_description", "clip_neg_description"}
EDGE_KEYS = {"contour", "edge", "control_image"}
SPARSE_KEYS = {"sampled_pixels", "image", "compressed_data", "walsh_metadata"}


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


class Channel(nn.Module):
    def __init__(self, cfg: DictConfig = None):
        super(Channel, self).__init__()
        if cfg is None:
            cfg = DictConfig(dict(channel_type="none", snr=None))
        self.chan_type = cfg.channel_type  # "awgn", "none"
        self.snr = cfg.snr
        assert self.chan_type in ["none", "awgn"], "Only AWGN channel is supported."

        # CA-LRGD UEP settings. Defaults are active but conservative.
        self.uep_enable: bool = bool(_cfg_get(cfg, "uep.enable", True))
        self.text_snr_boost: float = float(_cfg_get(cfg, "uep.text_snr_boost", 25.0))
        self.edge_snr_boost: float = float(_cfg_get(cfg, "uep.edge_snr_boost", 8.0))
        self.sparse_snr_boost: float = float(_cfg_get(cfg, "uep.sparse_snr_boost", 0.0))
        logger.info("Built {} channel, SNR {} dB.".format(cfg.channel_type, cfg.snr))
        logger.info(
            "CA-LRGD UEP: enable={}, text_boost={} dB, edge_boost={} dB, sparse_boost={} dB".format(
                self.uep_enable,
                self.text_snr_boost,
                self.edge_snr_boost,
                self.sparse_snr_boost,
            )
        )

    def _boosted_snr(self, key: Optional[str], snr_db: Optional[float]) -> Optional[float]:
        if snr_db is None or not self.uep_enable or key is None:
            return snr_db
        if key in TEXT_KEYS:
            return float(snr_db) + self.text_snr_boost
        if key in EDGE_KEYS:
            return float(snr_db) + self.edge_snr_boost
        if key in SPARSE_KEYS:
            return float(snr_db) + self.sparse_snr_boost
        return snr_db

    def gaussian_noise_layer(
        self,
        input: str | Image.Image | torch.Tensor,
        snr_db: Optional[float] = None,
    ):
        if snr_db is None:
            snr_db = self.snr
        if snr_db is None or snr_db >= 100:
            return input

        if isinstance(input, str):
            return qam16ModulationString(input, snr_db=snr_db)

        elif isinstance(input, Image.Image):
            return qam16ModulationImage(input, snr_db=snr_db)

        elif isinstance(input, torch.Tensor):
            return qam16ModulationTensor(input, snr_db=snr_db)

        elif isinstance(input, np.ndarray):
            return qam16ModulationTensor(
                torch.from_numpy(input), snr_db=snr_db
            ).numpy()

        else:
            return input

    def calculate_size_KB(self, data, sparse=False):
        """Calculate the size of the input data in KB."""

        def get_byte_size(obj, sparse=False):
            if obj is None:
                return 0
            if isinstance(obj, int):
                return sys.getsizeof(obj)
            if isinstance(obj, float):
                return sys.getsizeof(obj)
            if isinstance(obj, str):
                return len(obj.encode("utf-8"))
            if isinstance(obj, bytes):
                return sys.getsizeof(obj)
            if isinstance(obj, tuple):
                return sum(get_byte_size(v) for v in obj)
            if isinstance(obj, list):
                return sum(get_byte_size(v) for v in obj)
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            if isinstance(obj, torch.Tensor):
                return obj.element_size() * obj.nelement()
            if isinstance(obj, Image.Image):
                buffer = io.BytesIO()
                if sparse:
                    obj.save(buffer, format="WebP", lossless=True)
                else:
                    obj.save(buffer, format="PNG", optimize=True)
                return len(buffer.getvalue())
            if isinstance(obj, dict):
                return sum(get_byte_size(v, sparse) for _, v in obj.items())
            raise ValueError(f"Unsupported input type: {type(obj)}")

        size = get_byte_size(data, sparse)
        return size / 1024

    def forward(
        self, input: list | tuple | dict
    ) -> Tuple[dict, dict] | List[Tuple[dict, dict]]:
        if isinstance(input, list):
            return [self.forward(i) for i in input]
        if isinstance(input, tuple):
            return tuple(self.forward(i) for i in input)

        # Calculate the KB size of the input.
        sparse = {k: False for k in input.keys()}
        sparse["sampled_pixels"] = True
        input_sizes = dict(total=self.calculate_size_KB(input, sparse=sparse))
        if "clip_description" in input.keys():
            input_sizes["text"] = self.calculate_size_KB(
                input["clip_description"], sparse=sparse
            )
        if "contour" in input.keys():
            input_sizes["edge"] = self.calculate_size_KB(input["contour"], sparse=sparse)
        if "sampled_pixels" in input.keys():
            input_sizes["sparse"] = self.calculate_size_KB(
                input["sampled_pixels"], sparse=sparse
            )

        received = self.qam(input)
        if isinstance(received, dict):
            received["channel_snr_db"] = self.snr
            received["uep_meta"] = {
                "enable": bool(self.uep_enable),
                "text_snr_boost": float(self.text_snr_boost),
                "edge_snr_boost": float(self.edge_snr_boost),
                "sparse_snr_boost": float(self.sparse_snr_boost),
            }
        return received, input_sizes

    def qam(
        self,
        obj: None | dict | list | tuple | str | Image.Image | torch.Tensor,
        snr_override: Optional[float] = None,
        parent_key: Optional[str] = None,
    ):
        """Add QAM/AWGN noise to the input object with optional stream-specific SNR."""
        if obj is None:
            return None

        effective_snr = self._boosted_snr(parent_key, self.snr if snr_override is None else snr_override)
        if effective_snr is None or effective_snr >= 100:
            return obj

        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in noise_exempt_keys:
                    out[k] = v
                else:
                    out[k] = self.qam(v, snr_override=effective_snr, parent_key=k)
            return out
        elif isinstance(obj, list):
            return [self.qam(i, snr_override=effective_snr, parent_key=parent_key) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self.qam(i, snr_override=effective_snr, parent_key=parent_key) for i in obj)

        if self.chan_type == "none":
            return obj
        elif self.chan_type == "awgn":
            return self.gaussian_noise_layer(obj, snr_db=effective_snr)
        else:
            raise ValueError(f"Unsupported channel type {self.chan_type}")
