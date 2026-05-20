from __future__ import annotations

import io

import numpy as np
from PIL import Image


def apply_jpeg_artifacts(image_float: np.ndarray, quality: int) -> np.ndarray:
    image_u8 = (np.clip(image_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    src = Image.fromarray(image_u8, mode="RGB")
    buf = io.BytesIO()
    src.save(buf, format="JPEG", quality=quality, optimize=False)
    buf.seek(0)
    restored = Image.open(buf).convert("RGB")
    return np.asarray(restored, dtype=np.float64) / 255.0


def smooth_jpeg_highlight_artifacts(
    rgb: np.ndarray,
    *,
    strength: float = 0.55,
    blur_fn=None,
    periodic_x: bool = True,
) -> np.ndarray:
    """Re-low-pass small-scale highlights after JPEG to reduce blocking/aliasing."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return rgb
    lin = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    if blur_fn is None:
        from starsky_gen.nebula import _blur_separable_xy

        blur_fn = _blur_separable_xy
    luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    hot = np.clip((luma - 0.55) / 0.38, 0.0, 1.0) ** 1.15
    fine = luma - blur_fn(luma, passes=1, periodic_x=periodic_x)
    coarse = luma - blur_fn(luma, passes=3, periodic_x=periodic_x)
    blocky = np.clip(np.abs(fine) - np.abs(coarse) * 0.35, 0.0, None)
    gate = hot * np.clip(blocky / (np.percentile(blocky[blocky > 0], 92) + 1e-6 if np.any(blocky > 0) else 1.0), 0.0, 1.0)
    smooth_l = blur_fn(luma, passes=2, periodic_x=periodic_x)
    blend = luma * (1.0 - gate * s * 0.65) + smooth_l * (gate * s * 0.65)
    scale = np.divide(blend, np.maximum(luma, 1e-8), out=np.ones_like(luma), where=luma > 1e-8)
    return np.clip(lin * scale[..., np.newaxis], 0.0, 1.0)
