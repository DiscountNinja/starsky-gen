"""HDR tone mapping: ACES fitted, filmic, asinh helpers."""

from __future__ import annotations

import numpy as np

from starsky_gen.color_science import rec709_luma, remap_luma_preserving_chroma


def aces_fitted(x: np.ndarray) -> np.ndarray:
    """Stephen Hill ACES fitted curve on scalar array."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, None)


def tone_map_aces_rgb(
    rgb: np.ndarray,
    *,
    exposure: float = 1.0,
    disk_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Chroma-preserving ACES on linear RGB."""
    x = np.asarray(rgb, dtype=np.float64) * float(exposure)
    l = rec709_luma(x)
    l_t = aces_fitted(l)
    scale = np.where(l > 1e-8, l_t / (l + 1e-8), 0.0)
    out = x * scale[..., np.newaxis]
    if disk_weight is not None:
        w = np.asarray(disk_weight, dtype=np.float64)
        if w.ndim == 2:
            w = w[..., np.newaxis]
        out = x * (1.0 - w) + out * w
    return np.clip(out, 0.0, None)


def tone_map_filmic_luma(l: np.ndarray, shoulder: float = 0.85) -> np.ndarray:
    k = float(shoulder)
    l = np.clip(l, 0.0, None)
    return l * (1.0 + k * l) / (1.0 + l)


def tone_map_asinh_luma(l: np.ndarray, gain: float) -> np.ndarray:
    g = max(float(gain), 1e-6)
    return np.arcsinh(l * g) / np.arcsinh(g)


def asinh_linear_stretch_luma(
    luma: np.ndarray,
    *,
    gain: float,
    q: float = 1.0,
    shadow: float = 0.0,
    midtone_exposure: float = 1.0,
    toe_strength: float = 0.0,
) -> np.ndarray:
    """Linear HDR asinh: asinh((L - shadow) * gain / Q) / asinh(gain / Q).

    ``midtone_exposure`` lifts mids before stretch; ``toe_strength`` softens shadow toe.
    """
    g = max(float(gain), 1e-6)
    qv = max(float(q), 1e-6)
    lu = np.maximum(np.asarray(luma, dtype=np.float64), 0.0)
    me = float(np.clip(midtone_exposure, 0.5, 2.0))
    if abs(me - 1.0) > 1e-6:
        mid_w = np.exp(-((lu - 0.22) ** 2) / (2.0 * 0.14**2))
        lu = lu * (1.0 + (me - 1.0) * mid_w)
    x = np.maximum(lu - float(shadow), 0.0) * g / qv
    denom = np.arcsinh(g / qv) + 1e-12
    out = np.arcsinh(x) / denom
    ts = float(np.clip(toe_strength, 0.0, 1.0))
    if ts > 1e-6:
        toe = np.clip((0.12 - out) / 0.12, 0.0, 1.0) ** 1.35
        out = out + toe * ts * 0.045 * (1.0 - out)
    return np.clip(out, 0.0, None)


_LINEAR_ACESCCT_BREAK = 0.0078125  # 2^-7


def linear_to_acescct(luma: np.ndarray) -> np.ndarray:
    """ACEScct log encoding (luma)."""
    x = np.maximum(np.asarray(luma, dtype=np.float64), 0.0)
    return np.where(
        x > _LINEAR_ACESCCT_BREAK,
        (np.log2(x + 1e-12) + 9.72) / 17.52,
        -0.35828648 + x / _LINEAR_ACESCCT_BREAK * 0.001,
    )


def acescct_to_linear(acescct: np.ndarray) -> np.ndarray:
    t = np.asarray(acescct, dtype=np.float64)
    return np.where(
        t > -0.35828648,
        np.power(2.0, t * 17.52 - 9.72),
        np.maximum((t + 0.35828648) * _LINEAR_ACESCCT_BREAK / 0.001, 0.0),
    )


def apply_acescct_cinematic_grade(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float = 0.55,
    shoulder: float = 0.12,
) -> np.ndarray:
    """Final cinematic rolloff via ACEScct log contrast (chroma preserved)."""
    s = float(np.clip(strength, 0.0, 1.5))
    if s < 1e-6:
        return rgb
    w = np.broadcast_to(disk_w, rgb.shape[:2])[..., np.newaxis]
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    t = linear_to_acescct(lu)
    sh = float(shoulder)
    t_grade = t * (1.0 + s * sh * (0.48 - t * t))
    t_grade = t_grade - s * 0.04 * np.clip(t_grade + 0.35, 0.0, None) ** 2
    l_new = np.clip(acescct_to_linear(t_grade), 0.0, None)
    toned = remap_luma_preserving_chroma(lin, l_new)
    return np.maximum(rgb * (1.0 - w) + toned * w, 0.0)


def disk_filmic_rgb_grade(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    shoulder: float = 0.85,
) -> np.ndarray:
    w = np.broadcast_to(disk_w, rgb.shape[:2])[..., np.newaxis]
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    l_new = tone_map_filmic_luma(lu, shoulder=shoulder)
    toned = remap_luma_preserving_chroma(lin, l_new)
    return np.maximum(rgb * (1.0 - w) + toned * w, 0.0)


def disk_reinhard_rgb_grade(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    k: float = 0.34,
) -> np.ndarray:
    w = np.broadcast_to(disk_w, rgb.shape[:2])[..., np.newaxis]
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    l_new = lu / (1.0 + float(k) * lu)
    toned = remap_luma_preserving_chroma(lin, l_new)
    return np.maximum(rgb * (1.0 - w) + toned * w, 0.0)
