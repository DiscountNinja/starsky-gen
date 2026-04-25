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
