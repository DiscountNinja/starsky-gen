"""Display post-FX: DOF, bloom, god rays, core grade, lightwrap, color stratification."""

from __future__ import annotations

import numpy as np

from starsky_gen.color_science import rec709_luma
from starsky_gen.nebula import _blur_separable_xy, _blur_x_only, _resize_bilinear


def _blur_luma(
    luma: np.ndarray,
    passes: int,
    *,
    periodic_x: bool,
    axis_x: bool = False,
) -> np.ndarray:
    f = luma.astype(np.float64)
    if axis_x:
        return _blur_x_only(f, passes=passes, periodic_x=periodic_x)
    return _blur_separable_xy(f, passes=passes, periodic_x=periodic_x)


def depth_map_from_disk(
    disk_w: np.ndarray,
    height: int,
    width: int,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
) -> np.ndarray:
    """0 = near/sharp (midplane), 1 = far/soft (halo and outer sky)."""
    far = np.clip(1.0 - disk_w, 0.0, 1.0)
    cr_h = max(24, height // 32)
    cr_w = max(32, width // 24)
    n = rng.normal(0.0, 1.0, size=(cr_h, cr_w))
    n = _resize_bilinear(n, height, width, periodic_x=periodic_x)
    n = _blur_separable_xy(n, passes=2, periodic_x=periodic_x)
    n = (n - np.mean(n)) / (np.std(n) + 1e-6)
    return np.clip(far * 0.82 + np.clip(n * 0.08 + 0.5, 0.0, 1.0) * 0.18, 0.0, 1.0)


def apply_depth_of_field(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    strength: float = 0.38,
    max_sigma_px: float = 5.0,
    periodic_x: bool = True,
) -> np.ndarray:
    s = float(strength)
    if s < 1e-6:
        return rgb
    d = np.clip(np.asarray(depth, dtype=np.float64) * s, 0.0, 1.0)
    if float(np.max(d)) < 1e-6:
        return rgb
    p_lo = max(1, int(round(max_sigma_px * 0.22)))
    p_mid = max(2, int(round(max_sigma_px * 0.52)))
    p_hi = max(3, int(round(max_sigma_px * 0.95)))
    luma = rec709_luma(rgb)
    b_lo = _blur_luma(luma, p_lo, periodic_x=periodic_x)
    b_mid = _blur_luma(luma, p_mid, periodic_x=periodic_x)
    b_hi = _blur_luma(luma, p_hi, periodic_x=periodic_x)
    target_l = luma * (1.0 - d) + b_lo * d * 0.28 + b_mid * d * 0.38 + b_hi * d * 0.34
    scale = np.divide(target_l, luma, out=np.ones_like(luma), where=luma > 1e-9)
    return np.clip(rgb * scale[..., np.newaxis], 0.0, None)


def apply_global_s_curve(rgb: np.ndarray, *, strength: float = 0.24) -> np.ndarray:
    s = float(strength)
    if s < 1e-6:
        return rgb
    lin = np.maximum(rgb, 0.0)
    l = rec709_luma(lin)
    t = l - 0.5
    curve = 0.5 + t * (1.0 + 0.42 * s * (0.28 - t * t))
    scale = np.divide(np.clip(curve, 0.0, 1.0), l, out=np.ones_like(l), where=l > 1e-9)
    return np.clip(lin * scale[..., np.newaxis], 0.0, 1.0)


def apply_split_toning(
    rgb: np.ndarray,
    *,
    strength: float = 0.14,
    warm_high: tuple[float, float, float] = (1.04, 1.02, 0.97),
    cool_shadow: tuple[float, float, float] = (0.96, 0.99, 1.05),
) -> np.ndarray:
    s = float(strength)
    if s < 1e-6:
        return rgb
    l = rec709_luma(rgb)
    hi = np.clip((l - 0.42) / 0.48, 0.0, 1.0) ** 1.1
    sh = np.clip((0.38 - l) / 0.38, 0.0, 1.0) ** 1.05
    wh = np.array(warm_high, dtype=np.float64)
    cs = np.array(cool_shadow, dtype=np.float64)
    tint = 1.0 + (wh - 1.0)[None, None, :] * (hi[..., None] * s) + (cs - 1.0)[None, None, :] * (
        sh[..., None] * s
    )
    return np.clip(rgb * tint, 0.0, 1.0)


def apply_local_contrast(
    rgb: np.ndarray,
    *,
    strength: float = 0.09,
    periodic_x: bool = True,
) -> np.ndarray:
    s = float(strength)
    if s < 1e-6:
        return rgb
    l = rec709_luma(rgb)
    blur = _blur_luma(l, 5, periodic_x=periodic_x)
    detail = l - blur
    gate = np.clip(1.0 - np.abs(detail) * 4.5, 0.15, 1.0)
    boosted = l + detail * s * 0.85 * gate
    scale = np.divide(boosted, l, out=np.ones_like(l), where=l > 1e-9)
    return np.clip(rgb * scale[..., np.newaxis], 0.0, 1.0)


def apply_core_local_contrast(
    rgb: np.ndarray,
    core_mask: np.ndarray,
    *,
    strength: float = 0.11,
    periodic_x: bool = True,
) -> np.ndarray:
    """Mid-frequency CLAHE-like lift on nebula core only."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    m = np.clip(core_mask, 0.0, 1.0)
    if float(np.max(m)) < 1e-6:
        return rgb
    l = rec709_luma(rgb)
    blur_coarse = _blur_luma(l, 6, periodic_x=periodic_x)
    blur_fine = _blur_luma(l, 2, periodic_x=periodic_x)
    detail = (blur_fine - blur_coarse) * m
    gate = np.clip(1.0 - np.abs(detail) * 3.2, 0.12, 1.0)
    boosted = l + detail * s * 1.1 * gate
    scale = np.divide(boosted, l, out=np.ones_like(l), where=l > 1e-9)
    out = rgb * scale[..., np.newaxis]
    return np.clip(rgb * (1.0 - m[..., np.newaxis]) + out * m[..., np.newaxis], 0.0, 1.0)


def apply_tri_scale_bloom(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float = 0.18,
    threshold: float = 0.90,
    mix_tight: float = 0.50,
    mix_mid: float = 0.25,
    mix_wide: float = 0.10,
    neb_luma: np.ndarray | None = None,
    periodic_x: bool = True,
) -> np.ndarray:
    """Three-scale thresholded bloom: tight hot core, medium glow, wide haze.

    A mild high-frequency mask breaks up the halo so it reads textured, not a flat glow.
    """
    s = float(strength)
    if s < 1e-6:
        return rgb
    l = rec709_luma(rgb)
    thr = float(np.clip(threshold, 0.5, 0.99))
    hot = np.clip((l - thr) / max(1.0 - thr, 0.04), 0.0, 1.0) ** 1.15
    hot *= np.clip(disk_w, 0.0, 1.0)
    if float(np.max(hot)) < 1e-8:
        return rgb
    b_t = _blur_luma(hot, 1, periodic_x=periodic_x)
    b_m = _blur_luma(hot, 2, periodic_x=periodic_x)
    b_w = _blur_luma(hot, 3, periodic_x=periodic_x)
    glow = b_t * mix_tight + b_m * mix_mid + b_w * mix_wide
    hf = np.clip(hot - b_t, 0.0, 1.0)
    h, w_img = hot.shape
    yy, xx = np.mgrid[0:h, 0:w_img].astype(np.float64)
    tex = 0.62 + 0.38 * (0.5 + 0.5 * np.sin(xx * 0.047) * np.cos(yy * 0.039))
    glow = glow * (0.70 + 0.22 * hf + 0.08 * tex)
    if neb_luma is not None:
        nl = np.clip(np.broadcast_to(neb_luma, (h, w_img)), 0.0, 1.0)
        nl_s = _blur_luma(nl, 2, periodic_x=periodic_x)
        structure = np.clip(nl - nl_s, 0.0, 1.0)
        preserve = 0.50 + 0.50 * (1.0 - structure**0.85)
        glow = glow * preserve
    local = np.maximum(
        np.sum(rgb * glow[..., np.newaxis], axis=(0, 1)) / (float(glow.sum()) + 1e-8),
        1e-8,
    )
    local = local / np.maximum(np.max(local), 1e-8)
    bloom_rgb = local.reshape(1, 1, 3) * glow[..., np.newaxis]
    return np.clip(rgb + bloom_rgb * s * 0.042, 0.0, 1.0)


def apply_display_bloom(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float = 0.18,
    threshold: float = 0.90,
    periodic_x: bool = True,
) -> np.ndarray:
    return apply_tri_scale_bloom(
        rgb,
        disk_w,
        strength=strength,
        threshold=threshold,
        periodic_x=periodic_x,
    )


def apply_god_rays(
    rgb: np.ndarray,
    hot_mask: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float = 0.12,
    periodic_x: bool = True,
) -> np.ndarray:
    """Radial-ish shafts from bright core (low-res stretched blur)."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    h, w = hot_mask.shape
    hot = np.clip(hot_mask, 0.0, 1.0) * disk_w
    mass = float(hot.sum()) + 1e-8
    if mass < 1e-6:
        return rgb
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy = float((yy * hot).sum() / mass)
    cx = float((xx * hot).sum() / mass)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 2.0
    radial = hot / (r**0.55)
    radial = _blur_luma(radial, 4, periodic_x=periodic_x)
    radial = _blur_luma(radial, 8, periodic_x=periodic_x)
    radial = np.clip(radial * (1.0 / (float(np.max(radial)) + 1e-8)), 0.0, 1.0)
    tint = np.array([1.0, 0.92, 0.84], dtype=np.float64)
    add = radial[..., np.newaxis] * tint * s * 0.09
    return np.clip(rgb + add * 0.65, 0.0, 1.0)


def apply_core_dodge_burn(
    rgb: np.ndarray,
    neb_luma: np.ndarray,
    disk_w: np.ndarray,
    *,
    burn_strength: float = 0.14,
    dodge_strength: float = 0.06,
) -> np.ndarray:
    """Darken under brightest core; slight dodge on core rim for depth."""
    b = float(burn_strength)
    d = float(dodge_strength)
    if b < 1e-6 and d < 1e-6:
        return rgb
    nl = np.clip(neb_luma, 0.0, 1.0)
    core = np.clip((nl - 0.35) / 0.55, 0.0, 1.0) ** 1.2 * disk_w
    rim = _blur_separable_xy(core, passes=2, periodic_x=True)
    rim = np.clip(rim - core, 0.0, 1.0)
    out = rgb * (1.0 - core[..., np.newaxis] * b * 0.12)
    out = out + rim[..., np.newaxis] * d * 0.04
    return np.clip(out, 0.0, 1.0)


def apply_color_stratification(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float = 0.12,
) -> np.ndarray:
    """Warm near core, cooler magenta/blue in halo (modest saturation)."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    t = np.clip(disk_w, 0.0, 1.0)
    warm = np.array([1.05, 1.02, 0.94], dtype=np.float64)
    cool = np.array([0.96, 0.97, 1.06], dtype=np.float64)
    tint = warm[None, None, :] * t[..., np.newaxis] + cool[None, None, :] * (1.0 - t[..., np.newaxis])
    factor = 1.0 + (tint - 1.0) * s
    out = rgb * factor
    l = rec709_luma(out)
    mean_l = np.mean(l, axis=(0, 1), keepdims=True)
    chroma = out - l[..., np.newaxis]
    sat = 1.0 - s * 0.35 * (1.0 - t[..., np.newaxis])
    return np.clip(mean_l + chroma * sat, 0.0, 1.0)


def apply_lightwrap_stars(
    stars: np.ndarray,
    canvas: np.ndarray,
    core_mask: np.ndarray,
    *,
    strength: float = 0.09,
    periodic_x: bool = True,
) -> np.ndarray:
    """Bleed bright core onto nearby star layer."""
    s = float(strength)
    if s < 1e-6:
        return stars
    l = rec709_luma(canvas)
    wrap = _blur_luma(np.clip(l * core_mask, 0.0, 1.0), 3, periodic_x=periodic_x)
    warm = np.array([1.0, 0.94, 0.88], dtype=np.float64)
    gate = np.clip(core_mask * 1.4, 0.0, 1.0)[..., np.newaxis]
    return np.maximum(stars, stars + wrap[..., np.newaxis] * warm * s * 0.22 * gate)


def apply_directional_motion_blur(
    rgb: np.ndarray,
    *,
    strength: float = 0.06,
    periodic_x: bool = True,
) -> np.ndarray:
    """Subtle blur along galactic longitude (rotation axis)."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    passes = max(1, int(round(s * 24.0)))
    l = rec709_luma(rgb)
    b = _blur_luma(l, passes, periodic_x=periodic_x, axis_x=True)
    blend = np.clip(s * 0.55, 0.0, 0.42)
    target = l * (1.0 - blend) + b * blend
    scale = np.divide(target, l, out=np.ones_like(l), where=l > 1e-9)
    return np.clip(rgb * scale[..., np.newaxis], 0.0, None)


def apply_darken_sky(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float = 0.55,
) -> np.ndarray:
    """Push latitudes away from the disk toward true black."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    sky = np.clip(1.0 - np.clip(disk_w, 0.0, 1.0) * 1.05, 0.0, 1.0) ** 1.35
    return np.clip(rgb * (1.0 - sky[..., np.newaxis] * s), 0.0, 1.0)


def apply_band_display_highlight_cap(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    knee: float = 0.44,
    cap: float = 0.68,
) -> np.ndarray:
    """Rolloff display highlights in the galactic plane (after bloom, before B/W stretch)."""
    out = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lu = rec709_luma(out)
    w = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0) ** 1.05
    kn = float(np.clip(knee, 0.32, 0.58))
    cp = float(np.clip(cap, 0.52, 0.82))
    excess = np.maximum(lu - kn, 0.0)
    lu_new = kn + excess / (1.0 + excess * (3.2 + w * 2.8))
    lu_new = np.minimum(lu_new, cp)
    scale = np.divide(lu_new, lu, out=np.ones_like(lu), where=lu > 1e-9)
    eff = 1.0 - w[..., np.newaxis] * (1.0 - scale[..., np.newaxis])
    return np.clip(out * eff, 0.0, 1.0)


def apply_display_contrast_finish(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    black_point: float = 0.035,
    white_point: float = 0.96,
    sky_darken: float = 0.52,
) -> np.ndarray:
    """Final black/white stretch; slightly stronger in the disk band."""
    out = apply_darken_sky(rgb, disk_w, strength=sky_darken)
    bp = float(black_point)
    wp = max(float(white_point), bp + 0.05)
    l = rec709_luma(np.clip(out, 0.0, None))
    w = 0.55 + 0.45 * np.clip(disk_w, 0.0, 1.0)
    knee = 0.58
    excess = np.maximum(l - knee, 0.0)
    l_roll = knee + excess / (1.0 + excess * (2.6 + w * 2.4))
    roll_scale = np.divide(l_roll, l, out=np.ones_like(l), where=l > 1e-9)
    out = np.clip(
        out * (1.0 - w[..., np.newaxis] * (1.0 - roll_scale[..., np.newaxis])),
        0.0,
        1.0,
    )
    l = rec709_luma(out)
    dw = np.clip(disk_w, 0.0, 1.0)
    sky = np.clip(1.0 - dw**1.12, 0.0, 1.0)
    bp_eff = bp * (1.0 - w * 0.35)
    scaled = np.clip((l - bp_eff) / max(wp - bp, 1e-6), 0.0, 1.0)
    scale_full = np.divide(scaled, l, out=np.ones_like(l), where=l > 1e-9)
    scale = 1.0 + sky * (scale_full - 1.0)
    return np.clip(out * scale[..., np.newaxis], 0.0, 1.0)


def apply_shadow_lift(rgb: np.ndarray, *, lift: float = 0.018) -> np.ndarray:
    lf = float(lift)
    if lf < 1e-8:
        return rgb
    l = rec709_luma(rgb)
    sh = np.clip((0.22 - l) / 0.22, 0.0, 1.0) ** 1.05
    return np.clip(rgb + sh[..., np.newaxis] * lf, 0.0, 1.0)


def apply_masked_density_sharpen(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    density_map: np.ndarray | None,
    *,
    sigma_px: float,
    amp_faint: float,
    amp_midplane: float,
    knee: float,
    periodic_x: bool = True,
) -> np.ndarray:
    """Unsharp on disk midplane / faint nebula structure; excludes sky via density × disk masks."""
    if sigma_px < 1e-5 or (amp_faint < 1e-7 and amp_midplane < 1e-7):
        return rgb
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lum = rec709_luma(lin)
    dw = np.clip(np.broadcast_to(disk_w, lum.shape), 0.0, 1.0)
    dens = np.clip(np.broadcast_to(density_map, lum.shape) if density_map is not None else dw, 0.0, 1.0)
    # Keep sharpening off halo / empty sky.
    sky_excl = np.clip((dw - 0.08) / 0.42, 0.0, 1.0) ** 1.15
    structure = dens * sky_excl * (0.35 + 0.65 * dw)
    passes = int(np.clip(round(float(sigma_px)), 2, 8))
    blurred = _blur_luma(lum, passes, periodic_x=periodic_x)
    hi = lum - blurred
    m_faint = structure * (lum < float(knee)) * float(amp_faint)
    mid_bell = np.exp(-((lum - 0.26) ** 2) / (2.0 * 0.14**2))
    m_mid = structure * mid_bell * dw * float(amp_midplane)
    boosted = lum + hi * (m_faint + m_mid)
    scale = np.divide(boosted, np.maximum(lum, 3e-6), out=np.ones_like(lum), where=lum > 3e-6)
    return np.clip(lin * scale[..., np.newaxis], 0.0, None)
