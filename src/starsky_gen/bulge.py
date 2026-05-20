"""Asymmetric galactic bulge: multi-lobe longitude profile × mid-plane band."""

from __future__ import annotations

import numpy as np

from starsky_gen.placement import galactic_midplane_mask
from starsky_gen.procedural_noise import fbm2d, gaussian_blur_pil


def _wrap_dlon(xn: np.ndarray, cx: float) -> np.ndarray:
    d = np.abs(xn - cx)
    return np.minimum(d, 1.0 - d) * 2.0


def render_bulge_layer(
    width: int,
    height: int,
    *,
    bulge_n: float = 1.4,
    bulge_scale: float = 0.14,
    bulge_intensity: float = 0.42,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
    band_lat_sigma: float = 0.10,
    rng: np.random.Generator | None = None,
    extinction_mod: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-lobe warm core along the plane; dimmed where dust extinction is high."""
    _ = bulge_n
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    xn = (xx + 0.5) / width
    mid = galactic_midplane_mask(
        height,
        width,
        lat_sigma=band_lat_sigma,
        band_rotation_deg=band_rotation_deg,
        band_curvature_amp=band_curvature_amp,
    )
    rng = rng if rng is not None else np.random.default_rng(0)
    center_lon = float(rng.uniform(0.44, 0.56))
    n_lobes = int(rng.integers(2, 5))
    prof = np.zeros((height, width), dtype=np.float64)
    for _ in range(n_lobes):
        cx = float(np.clip(center_lon + rng.uniform(-0.14, 0.14), 0.30, 0.70))
        lon_sigma = max(float(bulge_scale) * rng.uniform(1.4, 2.6), 0.08)
        amp = float(rng.uniform(0.55, 1.15))
        skew = float(rng.uniform(-0.22, 0.22))
        dlon = _wrap_dlon(xn, cx)
        lobe = amp * np.exp(-((dlon / lon_sigma) ** 2))
        lobe = lobe * (1.0 + skew * np.tanh((xn - cx) * 4.0))
        prof = np.maximum(prof, lobe)
    prof = prof * mid
    prof = gaussian_blur_pil(prof, 8.0, periodic_x=True)
    prof = gaussian_blur_pil(prof, 4.5, periodic_x=True)
    hi = float(np.percentile(prof, 98.0)) + 1e-8
    prof = np.clip(prof / hi, 0.0, 1.0) ** 1.12
    ch_b, cw_b = max(6, height // 32), max(10, width // 24)
    cavity = fbm2d(rng, ch_b, cw_b, base_scale=0.22, octaves=4, periodic_x=True)
    from starsky_gen.procedural_noise import _resize_bilinear

    cavity = _resize_bilinear(cavity, height, width, periodic_x=True)
    prof = np.clip(prof * (0.52 + 0.48 * cavity), 0.0, 1.0)
    y_norm = (yy - (height - 1) * 0.5) / max((height - 1) * 0.5, 1.0)
    razor = 1.0 - 0.44 * np.exp(-((y_norm / 0.09) ** 2))
    prof = np.clip(prof * razor, 0.0, 1.0)
    dlon_ref = _wrap_dlon(xn, center_lon)
    t = np.clip(dlon_ref / (max(float(bulge_scale) * 2.8, 0.14)), 0.0, 1.0) ** 1.18
    center = np.array([1.0, 0.98, 0.94], dtype=np.float64)
    outer = np.array([0.90, 0.86, 0.80], dtype=np.float64)
    rgb = outer * (1.0 - prof[..., np.newaxis] * 0.22) + center * prof[..., np.newaxis] * 0.78
    rgb = rgb * (1.0 - t[..., np.newaxis] * 0.35) + outer * t[..., np.newaxis] * 0.35
    out = rgb * prof[..., np.newaxis] * float(bulge_intensity)
    if extinction_mod is not None:
        ext = np.clip(np.asarray(extinction_mod, dtype=np.float64), 0.0, 1.0)
        if ext.shape == (height, width):
            vis = np.clip(1.0 - ext**1.25, 0.08, 1.0)
            out = out * vis[..., np.newaxis]
    return out.astype(np.float64)
