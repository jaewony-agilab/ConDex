"""The judge's image preprocessing, applied by us before any backend sees an image.

Qwen3-VL cuts images into 16 px patches merged 2x2, so it resizes every image to multiples of 32
within [min_pixels, max_pixels] (`smart_resize`, same algorithm as the HF Qwen2-VL/Qwen3-VL image
processor). We do that resize ourselves, once, with the same rule: the processor then has nothing
left to change, and the plain/overlay/Claude/LoRA paths all see the same pixels. `check_processor`
confirms this against the real HF processor on the cluster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class JudgeInput:
    factor: int = 32  # patch 16 x merge 2 (Qwen3-VL)
    min_pixels: int = 65536  # 256 * 256
    max_pixels: int = 262144  # 512 * 512: our cap, sets the judge's input resolution


def smart_resize(height: int, width: int, cfg: JudgeInput = JudgeInput()) -> tuple[int, int]:
    f = cfg.factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too large: {height}x{width}")
    h = max(f, round(height / f) * f)
    w = max(f, round(width / f) * f)
    if h * w > cfg.max_pixels:
        beta = math.sqrt(height * width / cfg.max_pixels)
        h = max(f, math.floor(height / beta / f) * f)
        w = max(f, math.floor(width / beta / f) * f)
    elif h * w < cfg.min_pixels:
        beta = math.sqrt(cfg.min_pixels / (height * width))
        h = math.ceil(height * beta / f) * f
        w = math.ceil(width * beta / f) * f
    return h, w


def to_model_input(image: np.ndarray, cfg: JudgeInput = JudgeInput()) -> np.ndarray:
    h, w = smart_resize(image.shape[0], image.shape[1], cfg)
    if (h, w) == image.shape[:2]:
        return np.asarray(image)
    return np.asarray(Image.fromarray(np.asarray(image)).resize((w, h), Image.BICUBIC))


def model_scale(height: int, width: int, cfg: JudgeInput = JudgeInput()) -> float:
    """Render pixels per model-input pixel (shorter sides)."""
    h, w = smart_resize(height, width, cfg)
    return min(height, width) / min(h, w)


def check_processor(model_id: str, image: np.ndarray, cfg: JudgeInput = JudgeInput()) -> dict:
    """Push a pre-resized image through the real HF processor; report the grid it used and the
    max abs difference between our pixels and the processor's (de-normalized) pixels."""
    from transformers import AutoProcessor  # cluster only

    x = to_model_input(image, cfg)
    proc = AutoProcessor.from_pretrained(model_id)
    ip = proc.image_processor
    out = ip(images=[Image.fromarray(x)], return_tensors="np")
    t, gh, gw = (int(v) for v in out["image_grid_thw"][0])
    p, m = ip.patch_size, ip.merge_size
    tp = getattr(ip, "temporal_patch_size", 2)
    pv = np.asarray(out["pixel_values"], np.float32)
    # inverse of the processor's patch flattening: (t, gh/m, gw/m, m, m, C, T, p, p)
    pv = pv.reshape(t, gh // m, gw // m, m, m, 3, tp, p, p)
    img = pv.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(t * tp, 3, gh * p, gw * p)[0]
    mean = np.asarray(ip.image_mean, np.float32)[:, None, None]
    std = np.asarray(ip.image_std, np.float32)[:, None, None]
    rec = np.clip((img * std + mean) * 255.0, 0, 255).transpose(1, 2, 0)
    same_size = rec.shape[:2] == x.shape[:2]
    return {
        "ours_hw": list(x.shape[:2]),
        "processor_hw": [gh * p, gw * p],
        "same_size": bool(same_size),
        "max_abs_diff": float(np.abs(rec - x.astype(np.float32)).max()) if same_size else None,
        "reconstructed": rec.astype(np.uint8),
    }
