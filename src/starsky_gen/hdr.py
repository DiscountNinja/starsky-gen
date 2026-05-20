"""HDR buffer dtype helpers (32-bit float layers for canvas and stars)."""

from __future__ import annotations

import numpy as np

HDR_DTYPE = np.float32


def hdr_zeros(shape: tuple[int, ...] | int, *sizes: int) -> np.ndarray:
    if sizes:
        shape = (shape, *sizes)  # type: ignore[assignment]
    return np.zeros(shape, dtype=HDR_DTYPE)


def as_hdr(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=HDR_DTYPE)


def ensure_hdr(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == HDR_DTYPE:
        return arr
    return arr.astype(HDR_DTYPE, copy=False)
