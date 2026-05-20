"""Unify unresolved diffuse and resolved star color / display grade relationships."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starsky_gen.color_science import rec709_luma, remap_luma_preserving_chroma

_DEFAULT_WARM_DISK = np.array([1.02, 0.94, 0.86], dtype=np.float64)


@dataclass(frozen=True)
class DiskPhotonExposure:
    """Shared linear exposure reference for resolved + unresolved photons in the plane."""

    plane_median_luma: float
    plane_p90_luma: float
    linear_gain: float
    chroma: np.ndarray


def estimate_disk_photon_exposure(
    stars: np.ndarray,
    disk_w: np.ndarray,
    *,
    chroma: np.ndarray | None = None,
) -> DiskPhotonExposure:
    """Derive one exposure law from the star layer (resolved population)."""
    s = np.maximum(np.asarray(stars, dtype=np.float64), 0.0)
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    lu = rec709_luma(s)
    gate = dw > 0.26
    pos = lu[gate & (lu > 1e-10)]
    if pos.size < 16:
        return DiskPhotonExposure(
            plane_median_luma=0.08,
            plane_p90_luma=0.22,
            linear_gain=1.0,
            chroma=normalize_chroma_rgb(chroma),
        )
    med = float(np.median(pos))
    p90 = float(np.percentile(pos, 90.0))
    target_med = 0.11
    gain = float(np.clip(target_med / max(med, 1e-8), 0.55, 1.65))
    if chroma is None:
        chroma = disk_chroma_from_star_layer(stars, disk_w)
    return DiskPhotonExposure(
        plane_median_luma=med,
        plane_p90_luma=p90,
        linear_gain=gain,
        chroma=normalize_chroma_rgb(chroma),
    )


def apply_shared_photon_exposure(
    rgb: np.ndarray,
    exposure: DiskPhotonExposure,
    disk_w: np.ndarray,
    *,
    strength: float,
    diffuse_only: bool = False,
    diffuse_blur: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the same linear exposure gain to canvas or stars (diffuse can be masked)."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return np.asarray(rgb, dtype=np.float64)
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if diffuse_only and diffuse_blur is not None:
        lu_b = np.clip(np.asarray(diffuse_blur, dtype=np.float64), 0.0, None)
        hf = np.clip(lu - lu_b, 0.0, None)
        pointness = np.clip(hf / np.maximum(lu, 1e-6), 0.0, 1.0) ** 0.88
        mask = np.clip(dw * (1.0 - pointness * 0.92), 0.0, 1.0) * s
    else:
        mask = dw * s
    gain = 1.0 + (float(exposure.linear_gain) - 1.0) * mask
    new_lu = lu * gain
    return np.maximum(0.0, remap_luma_preserving_chroma(lin, new_lu))


def normalize_chroma_rgb(rgb: np.ndarray | None) -> np.ndarray:
    if rgb is None:
        return _DEFAULT_WARM_DISK.copy()
    c = np.maximum(np.asarray(rgb, dtype=np.float64).reshape(3), 0.0)
    m = float(np.max(c))
    if m < 1e-8:
        return _DEFAULT_WARM_DISK.copy()
    return (c / m).astype(np.float64)


def unresolved_speckle_rgb(
    *,
    warmth: float,
    disk_chroma: np.ndarray | None = None,
) -> np.ndarray:
    """Blend neutral unresolved tint toward disk star chroma (reduces warm+cool stacks)."""
    w = float(np.clip(warmth, 0.0, 1.0))
    neutral = np.array([0.992, 0.994, 0.996], dtype=np.float64)
    warm = normalize_chroma_rgb(disk_chroma if disk_chroma is not None else _DEFAULT_WARM_DISK)
    return (neutral * (1.0 - w) + warm * w).astype(np.float64)


def disk_chroma_from_star_layer(
    stars: np.ndarray,
    disk_w: np.ndarray,
    *,
    luma_percentile: float = 70.0,
    min_samples: int = 48,
) -> np.ndarray:
    """Mean chromaticity of in-plane stars (reference for unresolved / diffuse)."""
    s = np.maximum(np.asarray(stars, dtype=np.float64), 0.0)
    h, w = int(s.shape[0]), int(s.shape[1])
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw.ndim == 1:
        dw = dw[:, None]
    if dw.shape != (h, w):
        dw = np.broadcast_to(dw, (h, w))
    lu = rec709_luma(s)
    gate = dw > 0.28
    if not bool(np.any(gate)):
        return _DEFAULT_WARM_DISK.copy()
    lu_g = lu[gate]
    p = float(np.percentile(lu_g[lu_g > 1e-10], np.clip(luma_percentile, 40.0, 92.0)))
    mask = gate & (lu >= max(p * 0.55, 1e-8))
    if int(np.count_nonzero(mask)) < min_samples:
        mask = gate & (lu > 1e-10)
    if int(np.count_nonzero(mask)) < 8:
        return _DEFAULT_WARM_DISK.copy()
    rgb = s[mask]
    lu_m = lu[mask][:, np.newaxis]
    chroma = rgb / np.maximum(lu_m, 1e-8)
    mean = np.mean(chroma, axis=0)
    return normalize_chroma_rgb(mean)


def ism_lift_rgb(
    *,
    chroma_lock: float,
    disk_chroma: np.ndarray | None,
) -> np.ndarray:
    """Warm ISM lift color: locked to stars vs legacy gold+cool mix."""
    lock = float(np.clip(chroma_lock, 0.0, 1.0))
    star_warm = normalize_chroma_rgb(disk_chroma if disk_chroma is not None else _DEFAULT_WARM_DISK)
    legacy = np.array([0.55, 0.42, 0.26], dtype=np.float64)
    legacy_cool = np.array([0.22, 0.24, 0.32], dtype=np.float64)
    warm = legacy * (1.0 - lock) + star_warm * lock
    # Reduce independent cool shoulder that causes cyan islands after ACES.
    _ = legacy_cool
    return warm.astype(np.float64)


def harmonize_diffuse_canvas_chroma(
    canvas: np.ndarray,
    target_chroma: np.ndarray,
    disk_w: np.ndarray,
    diffuse_blur: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    """Pull low-frequency (diffuse) canvas chroma toward star reference in the plane."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return canvas
    target = normalize_chroma_rgb(target_chroma)
    lin = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    lu_blur = np.clip(np.asarray(diffuse_blur, dtype=np.float64), 0.0, None)
    hf = np.clip(lu - lu_blur, 0.0, None)
    pointness = np.clip(hf / np.maximum(lu, 1e-6), 0.0, 1.0) ** 0.9
    mask = np.clip(dw * (1.0 - pointness * 0.9), 0.0, 1.0) * s
    if float(np.max(mask)) < 1e-6:
        return lin
    tr = normalize_chroma_rgb(target)
    tr_lu = float(np.dot(tr, np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)))
    tr_ratio = tr / max(tr_lu, 1e-8)
    in_ratio = lin / np.maximum(lu[..., np.newaxis], 1e-6)
    pulled_ratio = in_ratio * (1.0 - mask[..., np.newaxis]) + tr_ratio.reshape(1, 1, 3) * mask[
        ..., np.newaxis
    ]
    out = np.maximum(pulled_ratio * lu[..., np.newaxis], 0.0)
    return out.astype(np.float64)


def soften_diffuse_chroma_toward(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    target_chroma: np.ndarray,
    *,
    strength: float = 0.38,
) -> np.ndarray:
    """Pull diffuse gas RGB toward one chroma reference (reduces red/blue patchwork)."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return np.asarray(rgb, dtype=np.float64)
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw.ndim == 1:
        dw = dw[:, None]
    h, w_img = int(lin.shape[0]), int(lin.shape[1])
    if dw.shape != (h, w_img):
        dw = np.broadcast_to(dw, (h, w_img))
    w = np.clip(dw * s, 0.0, 1.0)
    tr = normalize_chroma_rgb(target_chroma)
    tr_lu = float(np.dot(tr, np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)))
    tr_rgb = tr / max(tr_lu, 1e-8)
    gray = lu[..., np.newaxis]
    target_rgb = gray * tr_rgb.reshape(1, 1, 3)
    out = lin * (1.0 - w[..., np.newaxis]) + target_rgb * w[..., np.newaxis]
    return np.maximum(0.0, out).astype(np.float64)


def matched_star_display_stretch_gain(
    star_hdr: np.ndarray,
    scene_hdr: np.ndarray,
    disk_w: np.ndarray,
    base_gain: float,
) -> float:
    """Align star percentile-asinh stretch with in-plane scene HDR (split-grade path)."""
    s = np.maximum(np.asarray(star_hdr, dtype=np.float64), 0.0)
    c = np.maximum(np.asarray(scene_hdr, dtype=np.float64), 0.0)
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0) > 0.32
    if not bool(np.any(dw)):
        return float(base_gain)
    lu_s = rec709_luma(s)[dw]
    lu_c = rec709_luma(c)[dw]
    pos_s = lu_s[lu_s > 1e-10]
    pos_c = lu_c[lu_c > 1e-10]
    if pos_s.size < 16 or pos_c.size < 16:
        return float(base_gain)
    p_s = float(np.percentile(pos_s, 99.2))
    p_c = float(np.percentile(pos_c, 99.0))
    if p_s < 1e-8:
        return float(base_gain)
    ratio = float(np.clip(p_c / p_s, 0.35, 2.8))
    gain = float(base_gain) * (ratio**0.55)
    return float(np.clip(gain, 4.0, 24.0))
