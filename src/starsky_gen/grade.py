"""Post-grade: Bayer noise, vignette, color grade, dodge/burn."""

from __future__ import annotations

import numpy as np

from starsky_gen.color_science import apply_white_balance, rec709_luma


def apply_bayer_chromatic_noise(
    rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 0.012,
) -> np.ndarray:
    """Subtle Bayer-pattern noise; gated so dark sky does not turn into color grain."""
    h, w, _ = rgb.shape
    luma = rec709_luma(rgb)
    gate = np.clip((luma - 0.012) / 0.06, 0.0, 1.0) ** 1.4
    if float(np.max(gate)) < 1e-6:
        return rgb
    yy, xx = np.mgrid[0:h, 0:w]
    phase = ((yy & 1) << 1) | (xx & 1)
    n = rng.normal(0.0, strength, size=(h, w, 3))
    delta = np.zeros((h, w, 3), dtype=np.float64)
    delta[..., 0] = np.where(phase == 0, n[..., 0], 0.0)
    delta[..., 1] = np.where(phase == 1, n[..., 1], 0.0)
    delta[..., 2] = np.where(phase == 2, n[..., 2], 0.0)
    return np.clip(rgb + delta * gate[..., np.newaxis] * luma[..., np.newaxis], 0.0, None)


def apply_vignette_field_curvature(
    rgb: np.ndarray,
    *,
    strength: float = 1.0,
    desat_edges: float = 0.06,
) -> np.ndarray:
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = (xx / w - 0.5) * 2.0
    yn = (yy / h - 0.5) * 2.0
    r2 = xn * xn + yn * yn
    vig = 1.0 - 0.22 * strength * r2
    out = rgb * vig[..., np.newaxis]
    l = rec709_luma(out)
    mean_l = np.mean(l, axis=(0, 1), keepdims=True)
    sat_scale = 1.0 - desat_edges * strength * r2[..., np.newaxis]
    chroma = out - l[..., np.newaxis]
    out = mean_l + chroma * sat_scale
    return np.clip(out, 0.0, None)


def apply_atmospheric_scatter(rgb: np.ndarray, *, strength: float = 0.025) -> np.ndarray:
    l = rec709_luma(rgb)
    thresh = float(np.quantile(l, 0.999))
    mask = (l >= thresh).astype(np.float64)
    if mask.sum() < 4:
        return rgb
    from scipy.ndimage import gaussian_filter

    glow = gaussian_filter(l * mask, sigma=6.0)
    warm = np.array([1.0, 0.96, 0.92], dtype=np.float64)
    return rgb + glow[..., np.newaxis] * warm * strength


def apply_color_grade(
    rgb: np.ndarray,
    bulge_mask: np.ndarray,
    *,
    global_desat: float = 0.04,
) -> np.ndarray:
    l = rec709_luma(rgb)
    mean_l = np.mean(l, axis=(0, 1), keepdims=True)
    chroma = rgb - l[..., np.newaxis]
    out = mean_l + chroma * (1.0 - global_desat)
    warm = np.array([1.04, 1.02, 0.98], dtype=np.float64)
    cool = np.array([0.98, 0.99, 1.02], dtype=np.float64)
    return apply_white_balance(out, warm, cool, bulge_mask)


def apply_dodge_burn_lanes(
    rgb: np.ndarray,
    lane_mask: np.ndarray,
    *,
    amp: float = 0.06,
) -> np.ndarray:
    """Emphasize dust lane crests via local luma offset."""
    l = rec709_luma(rgb)
    from scipy.ndimage import gaussian_filter

    blur = gaussian_filter(l, sigma=2.5)
    detail = l - blur
    m = np.clip(lane_mask, 0.0, 1.0)
    boost = detail * m * amp
    out = rgb + boost[..., np.newaxis]
    return np.clip(out, 0.0, None)
