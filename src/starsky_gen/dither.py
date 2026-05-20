"""Blue-noise dither before 8-bit output (reduces banding in gradients)."""

from __future__ import annotations

import numpy as np

_TILE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def blue_noise_tile(size: int = 64, *, seed: int = 42) -> np.ndarray:
    """Spectrally shaped noise tile in [0, 1] (blue-ish, tiled at save time)."""
    key = (size, seed)
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]
    rng = np.random.default_rng(seed)
    n = max(16, int(size))
    white = rng.standard_normal((n, n))
    spec = np.fft.fftshift(np.fft.fft2(white))
    cy, cx = n // 2, n // 2
    yy, xx = np.ogrid[:n, :n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r0, r1 = n * 0.06, n * 0.42
    mask = np.clip((r - r0) / max(r1 - r0, 1e-6), 0.0, 1.0)
    field = np.fft.ifft2(np.fft.ifftshift(spec * mask)).real
    field -= float(field.min())
    span = float(field.max()) + 1e-9
    out = (field / span).astype(np.float64)
    _TILE_CACHE[key] = out
    return out


def apply_blue_noise_dither_u8(
    rgb: np.ndarray,
    *,
    strength: float = 1.0,
    tile_seed: int = 42,
) -> np.ndarray:
    """Add tiled blue-noise before uint8 quantize (luma-only, preserves chroma)."""
    s = float(strength)
    lin = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    if s < 1e-6:
        return (lin * 255.0 + 0.5).astype(np.uint8)
    h, w, _ = lin.shape
    tile = blue_noise_tile(64, seed=tile_seed)
    th, tw = tile.shape
    yy = np.arange(h, dtype=np.int64)[:, None] % th
    xx = np.arange(w, dtype=np.int64)[None, :] % tw
    bn = tile[yy, xx]
    lu = (
        lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
    )
    lu_d = np.clip(lu * 255.0 + (bn - 0.5) * s, 0.0, 255.0)
    ratio = lin / np.maximum(lu[..., np.newaxis], 1e-8)
    out = ratio * (lu_d[..., np.newaxis] / 255.0)
    return np.clip(out * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
