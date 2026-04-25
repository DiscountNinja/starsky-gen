from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from starsky_gen.config import NebulaMode, ProjectionMode, RenderConfig
from starsky_gen.nebula import (
    _blur_rgb_separable_xy,
    _blur_separable_xy,
    _blur_x_only,
    _blur_y_only,
    _resize_bilinear,
    generate_nebula,
)
from starsky_gen.postprocess import apply_jpeg_artifacts
from starsky_gen.projections import cubemap_faces_from_equirect, sph_to_equirect_xy
from starsky_gen.reference_stars import paint_reference_anchors
from starsky_gen.starfield import (
    STAR_COLOR_NAMES,
    STAR_SIZE_NAMES,
    catalog_stats,
    paint_star,
    rgb_from_bv,
    reroll_stars_in_dark_lanes,
    sample_cluster_star_catalog,
    sample_star_catalog,
    size_radius,
    star_color,
)


def _background_plane(
    rng: np.random.Generator,
    height: int,
    width: int,
    enabled: bool,
    black_background: bool,
) -> np.ndarray:
    if black_background:
        return np.zeros((height, width, 3), dtype=np.float64)

    y = np.linspace(-1.0, 1.0, height)[:, None]
    disk = np.exp(-(y**2) / 0.22)
    falloff = (1.0 - disk) ** 0.9

    if enabled:
        base = 0.065 + disk * 0.06 - falloff * 0.03
    else:
        # Even without gradient mode enabled, keep a soft non-black backdrop.
        base = 0.048 + disk * 0.026 - falloff * 0.018

    # Mix fine noise and sparse speckle so some grain reads like faint stars / sensor floor.
    noise = rng.normal(0.0, 0.0135, size=(height, width))
    blue_noise = rng.normal(0.0, 0.0165, size=(height, width))
    speckle_gate = rng.random((height, width))
    speckle_core = np.where(speckle_gate > 0.9935, rng.uniform(0.05, 0.13, size=(height, width)), 0.0)
    speckle_halo = np.where(speckle_gate > 0.985, rng.uniform(0.006, 0.022, size=(height, width)), 0.0)
    speckle = speckle_core + speckle_halo
    value = np.clip(np.repeat(base, width, axis=1) + noise + speckle, 0.02, 0.26)
    # Slight per-render tint variation avoids a single fixed background color.
    tint_shift = rng.uniform(-0.004, 0.004)
    blue = np.clip(value + 0.015 + blue_noise + tint_shift, 0.035, 0.25)
    red = np.clip(value - 0.006 - tint_shift * 0.5, 0.02, 0.19)
    green = np.clip(value - 0.002, 0.02, 0.20)
    return np.stack([red, green, blue], axis=2)


def _galactic_disk_weight(height: int, sigma: float = 0.46) -> np.ndarray:
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
    return np.exp(-((yy**2) / (sigma**2)))


def _apply_long_exposure_look(
    canvas: np.ndarray,
    rng: np.random.Generator,
    *,
    sky_w: np.ndarray,
    band: np.ndarray,
    disk_w: np.ndarray,
) -> np.ndarray:
    """Mimic real stacked wide-field frames: uneven sky, amp-style corner lift, asymmetric vignette."""
    h, w, _ = canvas.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    sky_m = np.clip(sky_w[..., None], 0.0, 1.0)
    # Blend so the bright disk is barely touched (mostly high latitudes / corners).
    mask = np.clip(0.88 * sky_m + 0.12 * (1.0 - disk_w[..., None]), 0.0, 1.0)

    ang = float(rng.uniform(0.0, 6.283185307179586))
    ca, sa = np.cos(ang), np.sin(ang)
    saw = xx * ca + yy * sa
    saw_n = np.clip(saw / max(abs(ca) + abs(sa), 0.25), -1.0, 1.0)
    warm = np.array([1.0, 0.94, 0.86], dtype=np.float64)
    cool = np.array([0.90, 0.93, 1.04], dtype=np.float64)
    t = (0.5 + 0.5 * saw_n)[..., None]
    rgb_tilt = warm * (1.0 - t) + cool * t
    sky_lift = (0.012 + 0.034 * sky_w[..., None]) * (0.52 + 0.48 * saw_n[..., None])
    out = canvas + sky_lift * rgb_tilt * mask

    cx = float(rng.uniform(-0.62, 0.62))
    cy = float(rng.uniform(-0.62, 0.62))
    wx = float(rng.uniform(0.38, 0.72))
    wy = float(rng.uniform(0.42, 0.78))
    glow = np.exp(-(((xx - cx) ** 2) / wx + ((yy - cy) ** 2) / wy))
    glow *= np.clip(sky_w * (1.0 - band * 0.52), 0.0, 1.0)
    amp = np.array([0.018, 0.014, 0.011], dtype=np.float64) * float(rng.uniform(0.88, 1.32))
    out = out + glow[..., None] * amp

    ox = float(rng.uniform(-0.14, 0.14))
    oy = float(rng.uniform(-0.14, 0.14))
    ax = float(rng.uniform(0.82, 1.18))
    ay = float(rng.uniform(0.82, 1.18))
    rad = np.clip((xx - ox) ** 2 * ax + (yy - oy) ** 2 * ay, 0.0, 1.35)
    vig = 1.0 - (0.022 + 0.018 * rng.random()) * sky_w * np.clip(rad, 0.0, 1.25)
    vig = np.clip(vig, 0.965, 1.0)
    out = out * (mask * vig[..., None] + (1.0 - mask))

    return np.clip(out, 0.0, 1.0)


def _apply_galactic_disk_luminance_envelope(
    canvas: np.ndarray,
    rng: np.random.Generator,
    *,
    disk_w: np.ndarray,
) -> np.ndarray:
    """Brighter near seeded galactic-longitude center along the band; darker toward equirect poles."""
    h, w, _ = canvas.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    sy = float(rng.uniform(0.26, 0.38))
    vert = 0.74 + 0.26 * np.exp(-((yy**2) / (sy**2)))
    gc_x = float(np.clip(rng.normal(0.0, 0.12), -0.40, 0.40))
    sx = float(rng.uniform(0.088, 0.19))
    amp = float(rng.uniform(0.09, 0.175))
    bulge = 1.0 + amp * np.exp(-(((xx - gc_x) ** 2) / (sx**2 + 1e-9)))
    skew = float(rng.uniform(-0.13, 0.13))
    bulge = bulge * (1.0 + skew * np.tanh((xx - gc_x) * 2.6))
    bulge = np.clip(bulge, 0.96, 1.24)
    plane_gate = 0.52 + 0.48 * np.exp(-((yy**2) / 0.50))
    horiz = 1.0 + (bulge - 1.0) * plane_gate
    scale = np.clip(vert * horiz, 0.72, 1.20)
    # Ease off in the outer disk halo so grade stacks gently with disk_w-based passes later.
    ease = 0.62 + 0.38 * disk_w
    scale = 1.0 + (scale - 1.0) * ease
    return np.clip(canvas * scale[..., None], 0.0, 1.0)


def _soft_knee_star_layer(
    star_img: np.ndarray, disk_w: np.ndarray, *, knee: float = 0.36, strength: float = 1.65
) -> None:
    """Reduce clipped white mush where many disk stars overlap (in-place)."""
    lum = np.mean(np.clip(star_img, 0.0, None), axis=2)
    excess = np.maximum(0.0, lum - knee)
    factor = 1.0 / (1.0 + strength * excess)
    w = disk_w * 0.90 + 0.10
    star_img *= (w * factor + (1.0 - w))[..., None]


def _luma_tone_map_disk(rgb: np.ndarray, disk_w: np.ndarray, *, k: float = 0.52) -> np.ndarray:
    """Reinhard-style luma roll-off in the disk only, preserving hue."""
    lin = np.clip(rgb, 0.0, None)
    luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    l_new = luma / (1.0 + k * luma)
    scale = np.divide(l_new, luma, out=np.ones_like(luma), where=luma > 1e-8)
    adjusted = lin * scale[..., None]
    w = disk_w[..., None]
    return np.clip(rgb * (1.0 - w) + adjusted * w, 0.0, 1.0)


def _disk_gamma_lift(rgb: np.ndarray, disk_w: np.ndarray, *, gamma: float = 0.93) -> np.ndarray:
    """Slight gamma in the disk only: lifts shadow detail (photo print / sensor response)."""
    w = disk_w[..., None]
    lifted = np.clip(rgb, 0.0, 1.0) ** gamma
    return np.clip(rgb * (1.0 - w) + lifted * w, 0.0, 1.0)


def _disk_photo_grade(rgb: np.ndarray, disk_w: np.ndarray) -> np.ndarray:
    """Local disk grade: mild toe lift + soft S-curve on luma (chrominance roughly preserved)."""
    w = disk_w[..., None]
    lin = np.clip(rgb, 0.0, 1.0)
    luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    toe = np.clip((0.055 - luma) / 0.055, 0.0, 1.0) ** 0.62
    l_lift = np.clip(luma + 0.018 * toe * disk_w, 0.0, 1.0)
    t = l_lift - 0.5
    l_curve = np.clip(0.5 + t * (1.0 + 0.36 * (0.25 - t * t)), 0.0, 1.0)
    scale = np.divide(l_curve, l_lift, out=np.ones_like(l_lift), where=l_lift > 1e-9)
    graded = np.clip(lin * scale[..., None], 0.0, 1.0)
    return np.clip(rgb * (1.0 - w) + graded * w, 0.0, 1.0)


def _dust_scattered_backlight(ext: np.ndarray, *, galaxy_streak: bool) -> np.ndarray:
    """Warm interstellar glow in thick dust (avoids pure black cutouts)."""
    thick = np.clip(1.0 - ext, 0.0, 1.0)
    warm = np.array([0.16, 0.11, 0.075], dtype=np.float64)
    brown = np.array([0.085, 0.055, 0.038], dtype=np.float64)
    if galaxy_streak:
        s = (thick**1.12) * 0.056
        s2 = (thick**1.65) * 0.026
    else:
        s = (thick**1.18) * 0.034
        s2 = (thick**1.7) * 0.015
    return warm * s[..., None] + brown * s2[..., None]


def _dust_rim_light(ext: np.ndarray, *, galaxy_streak: bool) -> np.ndarray:
    """Warm light on extinction gradients (cloud edges / partial transparency)."""
    sm = _blur_separable_xy(ext, passes=2 if galaxy_streak else 1)
    rim = np.clip(sm - ext, 0.0, 1.0)
    rim = _blur_separable_xy(rim, passes=1)
    rim = np.clip(rim**0.88, 0.0, 1.0)
    warm = np.array([0.28, 0.19, 0.13], dtype=np.float64)
    mag = np.array([0.34, 0.07, 0.22], dtype=np.float64)
    amp = 0.052 if galaxy_streak else 0.030
    return rim[..., None] * (warm + mag * 0.30) * amp


def _dust_volume_mottle(
    rng: np.random.Generator, ext: np.ndarray, *, galaxy_streak: bool
) -> np.ndarray:
    """Low-contrast noise inside thick dust (internal structure, not a flat mask)."""
    thick = np.clip(1.0 - ext, 0.0, 1.0)
    n = rng.normal(0.0, 1.0, size=ext.shape)
    n = _blur_separable_xy(n, passes=1)
    brown = np.array([0.10, 0.086, 0.078], dtype=np.float64)
    w = (thick**1.22)[..., None]
    amp = 0.010 if galaxy_streak else 0.008
    return n[..., None] * brown * w * amp


def _apply_extinction_to_canvas(
    canvas: np.ndarray,
    ext: np.ndarray,
    *,
    galaxy_streak: bool,
    rng_mottle: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply per-pixel extinction (caller feathers `ext` for galaxy streak); redden in thick dust."""
    if galaxy_streak:
        ext = np.clip(ext * 0.972 + 0.022, 0.0, 1.0)
    out = canvas * ext[..., None]
    out = out + _dust_scattered_backlight(ext, galaxy_streak=galaxy_streak)
    out = out + _dust_rim_light(ext, galaxy_streak=galaxy_streak)
    if rng_mottle is not None:
        out = out + _dust_volume_mottle(rng_mottle, ext, galaxy_streak=galaxy_streak)
    if galaxy_streak:
        d = np.clip((1.0 - ext) ** 0.9, 0.0, 1.0)
        out[..., 0] *= 1.0 + 0.068 * d
        out[..., 1] *= 1.0 - 0.012 * d
        out[..., 2] *= 1.0 - 0.082 * d
    return np.clip(out, 0.0, 1.0)


def _extinction_from_dust_and_lane(
    dust_occlusion: np.ndarray,
    lane_ext: np.ndarray,
    cfg: RenderConfig,
) -> np.ndarray:
    base_extinction_strength = 0.52 if cfg.nebula_mode == NebulaMode.galaxy_streak else 0.36
    extinction_strength = base_extinction_strength * cfg.nebula_tuning.dust_strength
    lane_boost = np.clip(
        lane_ext * (0.78 + 0.32 * cfg.nebula_tuning.dust_strength),
        0.0,
        1.0,
    )
    lane_k = 0.44 + 0.26 * cfg.nebula_tuning.dust_strength if cfg.nebula_mode == NebulaMode.galaxy_streak else 0.0
    return np.clip(
        1.0 - dust_occlusion * extinction_strength - lane_boost * lane_k,
        0.10,
        1.0,
    )


def _apply_separated_disk_sky_grain(
    canvas: np.ndarray,
    rng: np.random.Generator,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Disk: large-scale noise only (below star PSF scale). Sky: finer read-noise. Foreground stars add after."""
    yy = np.linspace(-1.0, 1.0, height)[:, None]
    band = np.exp(-((yy**2) / 0.55))
    sky_w = np.clip(1.0 - band, 0.0, 1.0) ** 0.38
    gn = rng.normal(0.0, 0.016, size=(height, width))
    gn_lp = _blur_separable_xy(_blur_separable_xy(gn, passes=3), passes=3)
    disk_sm = np.stack(
        [gn_lp * 0.97, gn_lp * 1.0, gn_lp * 1.02],
        axis=2,
    ).astype(np.float64)
    out = np.clip(canvas + disk_sm * band[..., None] * 0.040, 0.0, 1.0)
    grain_hf = rng.normal(0.0, 0.0125, size=(height, width, 3)).astype(np.float64)
    hf_gate = (0.32 + 0.74 * sky_w)[..., None]
    out = np.clip(out + grain_hf * hf_gate * 0.80, 0.0, 1.0)
    gg = rng.normal(0.0, 0.0088, size=(height, width, 3)).astype(np.float64)
    neutral_g = np.array([0.34, 0.35, 0.36], dtype=np.float64)
    neutral_g /= float(np.sum(neutral_g))
    g_strength = (0.34 + 0.66 * sky_w)[..., None] * 0.76
    return np.clip(out + gg * neutral_g * g_strength, 0.0, 1.0)


def _band_micro_ripple(
    rgb: np.ndarray, rng: np.random.Generator, disk_w: np.ndarray, *, strength: float = 0.011
) -> np.ndarray:
    """High-frequency luma ripples in the disk (unresolved star / grain texture)."""
    h, w, _ = rgb.shape
    a = rng.normal(0.0, 1.0, size=(h, w))
    b = rng.normal(0.0, 0.65, size=(h, w))
    # Small shifts on both axes reduce row/column grain in the disk.
    b = np.roll(b, int(rng.integers(-3, 4)), axis=1)
    b = np.roll(b, int(rng.integers(-2, 3)), axis=0)
    fine = (a + b * 0.55) * strength
    neutral = np.array([0.34, 0.36, 0.40], dtype=np.float64)
    neutral /= np.sum(neutral)
    return np.clip(rgb + disk_w[..., None] * fine[..., None] * neutral, 0.0, 1.0)


def _chromatic_fringe_disk(rgb: np.ndarray, rng: np.random.Generator, disk_w: np.ndarray) -> np.ndarray:
    """Subtle red/blue separation on brightest disk pixels (photo aberration hint)."""
    peak = np.max(rgb, axis=2)
    gate = disk_w * np.clip((peak - 0.58) / 0.34, 0.0, 1.0)
    r_n = rng.normal(0.0, 1.0, size=peak.shape)
    u = rng.random(peak.shape)
    out = rgb.copy()
    out[..., 0] = np.clip(out[..., 0] + gate * (r_n * 0.007 + 0.0035), 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] - gate * (r_n * 0.0065 + 0.003), 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + gate * (u - 0.5) * 0.0032, 0.0, 1.0)
    return out


def _disk_halation_soft(rgb: np.ndarray, disk_w: np.ndarray, *, strength: float = 0.028) -> np.ndarray:
    """Wide, warm halation from bright disk regions (lens / long exposure feel)."""
    lum = np.mean(np.clip(rgb, 0.0, 1.0), axis=2)
    lb = _blur_separable_xy(lum, passes=2)
    halo = np.clip((lb - lum) * strength, 0.0, 0.09) * disk_w
    wh = np.array([1.0, 0.94, 0.86], dtype=np.float64)
    return np.clip(rgb + halo[..., None] * wh, 0.0, 1.0)


def _empty_star_stats() -> dict[str, dict[str, int]]:
    return {
        "color_counts": {n: 0 for n in STAR_COLOR_NAMES},
        "size_counts": {n: 0 for n in STAR_SIZE_NAMES},
    }


def _merge_star_stats(a: dict[str, dict[str, int]], b: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    out = _empty_star_stats()
    for name in STAR_COLOR_NAMES:
        out["color_counts"][name] = a["color_counts"].get(name, 0) + b["color_counts"].get(name, 0)
    for name in STAR_SIZE_NAMES:
        out["size_counts"][name] = a["size_counts"].get(name, 0) + b["size_counts"].get(name, 0)
    return out


def _add_stars_from_catalog(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: RenderConfig,
    catalog: dict[str, np.ndarray],
    *,
    foreground_layer: bool,
    galaxy_disk_cool_stars: bool = False,
    point_disk_stars: bool = False,
    plane_psf_elongation: bool = False,
    cluster_layer: bool = False,
) -> dict[str, dict[str, int]]:
    xs, ys = sph_to_equirect_xy(catalog["lon"], catalog["lat"], cfg.width, cfg.height)
    has_bv = "bv" in catalog
    for i in range(xs.shape[0]):
        color_name = STAR_COLOR_NAMES[int(catalog["color_idx"][i])]
        size_name = STAR_SIZE_NAMES[int(catalog["size_idx"][i])]
        radius = size_radius(rng, size_name)

        u_lum = float(rng.random())
        if not foreground_layer and not cluster_layer:
            # Power toward faint magnitudes so the disk reads as stars, not uniform ISO grain.
            if u_lum < 0.78:
                lum = 0.03 + (u_lum / 0.78) ** 2.35 * 0.58
            else:
                lum = 0.61 + ((u_lum - 0.78) / 0.22) ** 1.05 * 0.52
            if rng.random() < 0.011:
                lum *= float(rng.uniform(1.25, 2.85))
            lum = float(np.clip(lum, 0.024, 2.4))
        else:
            lum = 1.0

        bv_w: float | None = float(catalog["bv"][i]) if has_bv else None

        # Keep large stars predominantly hot / blue (pull B–V).
        if has_bv and bv_w is not None and radius >= 6 and bv_w > -0.06 and rng.random() < 0.78:
            bv_w = float(rng.uniform(-0.32, 0.05))

        if not has_bv and radius >= 6 and color_name != "blue" and rng.random() < 0.78:
            color_name = "blue"

        # Red / cool giants: usually small on screen.
        if has_bv and bv_w is not None and bv_w >= 0.95:
            if radius > 3 and rng.random() < 0.9:
                radius = int(rng.integers(1, 4))
            else:
                radius = min(radius, 4)
        elif color_name == "red":
            if radius > 3 and rng.random() < 0.9:
                radius = int(rng.integers(1, 4))
            else:
                radius = min(radius, 4)

        # Yellow–orange: uncommon as large disks.
        if has_bv and bv_w is not None and 0.42 <= bv_w < 0.95 and radius > 3 and rng.random() < 0.88:
            radius = int(rng.integers(1, 4))
        elif color_name == "yellow" and radius > 3 and rng.random() < 0.88:
            radius = int(rng.integers(1, 4))

        if point_disk_stars and not foreground_layer and not cluster_layer and cfg.features.galaxy_view:
            plane_gate = float(np.exp(-((catalog["lat"][i] / 0.34) ** 2)))
            if plane_gate > 0.30:
                if rng.random() < 0.34:
                    radius = 1
                elif radius == 1 and lum > 0.45 and rng.random() < 0.52:
                    radius = 2

        if cluster_layer:
            radius = 1 if rng.random() < 0.90 else 2

        if has_bv and bv_w is not None:
            color = rgb_from_bv(bv_w, catalog["jitter"][i])
            if bv_w >= 1.0:
                color = color * 0.52
        else:
            color = star_color(color_name, catalog["jitter"][i])
            if color_name == "red":
                color = color * 0.5
        if cluster_layer:
            color = color * rng.uniform(0.48, 0.82)
        elif foreground_layer:
            color = color * rng.uniform(0.84, 1.18)
        else:
            color = color * lum * float(rng.uniform(0.88, 1.12))
        if not cluster_layer and cfg.features.galaxy_view and not foreground_layer:
            hue = float(rng.uniform(-0.12, 0.12))
            hue_scale = 0.22 if has_bv else 0.55
            color = color * np.array(
                [
                    1.0 + hue * hue_scale,
                    1.0 - 0.46 * hue * hue_scale,
                    1.0 - 0.56 * hue * hue_scale,
                ],
                dtype=np.float64,
            )
            color = np.clip(color, 0.0, 1.0)
        if (
            not cluster_layer
            and cfg.features.galaxy_view
            and not foreground_layer
            and radius <= 3
        ):
            uo = rng.random()
            if uo < 0.0028:
                color = np.clip(
                    np.array([0.52, 0.74, 1.12], dtype=np.float64) * float(rng.uniform(0.92, 1.08)),
                    0.0,
                    1.0,
                )
            elif uo < 0.0044:
                color = np.clip(
                    np.array([0.64, 0.28, 0.095], dtype=np.float64) * float(rng.uniform(0.88, 1.06)),
                    0.0,
                    1.0,
                )
        warm_disk_star = (not has_bv and color_name in ("yellow", "red")) or (
            has_bv and bv_w is not None and bv_w >= 0.42
        )
        if warm_disk_star and not foreground_layer and not cluster_layer and cfg.features.galaxy_view:
            plane_d = float(np.exp(-((catalog["lat"][i] / 0.38) ** 2)))
            color = color * (1.0 - 0.22 * plane_d * rng.uniform(0.88, 1.0))
        if cfg.features.depth and not foreground_layer:
            lat01 = np.clip((catalog["lat"][i] + np.pi / 2.0) / np.pi, 0.0, 1.0)
            # Smooth rolloff avoids a hard brightness contour line.
            depth_scale = 0.22 + 0.78 * (lat01**0.85)
            color = color * depth_scale
        if foreground_layer:
            color = color * rng.uniform(0.98, 1.08)
        elif galaxy_disk_cool_stars and cfg.features.galaxy_view and not cluster_layer:
            plane = float(np.exp(-((catalog["lat"][i] / 0.34) ** 2)))
            cool = np.array([0.94, 0.97, 1.05], dtype=np.float64)
            color = color * ((1.0 - 0.26 * plane) * 1.0 + 0.26 * plane * cool)
        if not foreground_layer and cfg.features.galaxy_view and not cluster_layer:
            plane_star = float(np.exp(-((catalog["lat"][i] / 0.36) ** 2)))
            x_n0 = float(xs[i]) / max(cfg.width - 1, 1)
            core_pre = float(
                np.exp(-((catalog["lat"][i] / 0.26) ** 2)) * np.exp(-(((x_n0 - 0.5) / 0.22) ** 2))
            )
            halo_w = 1.0 - plane_star
            color = color * (1.0 - 0.08 * halo_w)
            blue_halo = np.array([0.90, 0.94, 1.06], dtype=np.float64)
            warm_disk = np.array([1.04, 0.99, 0.92], dtype=np.float64)
            halo_mix = blue_halo * (1.0 - 0.55 * core_pre) + warm_disk * (0.55 * core_pre)
            color = color * (halo_w * halo_mix + (1.0 - halo_w))
            color = np.clip(color, 0.0, 1.0)
        if not foreground_layer and not cluster_layer and cfg.features.galaxy_view:
            x_n2 = float(xs[i]) / max(cfg.width - 1, 1)
            core_gold = float(
                np.exp(-((catalog["lat"][i] / 0.26) ** 2)) * np.exp(-(((x_n2 - 0.5) / 0.22) ** 2))
            )
            warm_core = np.array([0.165, 0.120, 0.062], dtype=np.float64)
            color = np.clip(color + warm_core * (core_gold * 1.88), 0.0, 1.0)
            gw = 0.72 * core_gold
            gold_shift = np.array([1.16, 0.94, 0.70], dtype=np.float64)
            color = np.clip(color * ((1.0 - gw) + gw * gold_shift), 0.0, 1.0)
            if core_gold > 0.12:
                wc = float(np.clip((core_gold - 0.12) / 0.55, 0.0, 1.0))
                color = np.clip(color * (1.0 + wc * np.array([0.045, 0.012, -0.038], dtype=np.float64)), 0.0, 1.0)
        if foreground_layer and cfg.features.galaxy_view and not cluster_layer:
            x_n = float(xs[i]) / max(cfg.width - 1, 1)
            bulge_proxy = float(
                np.exp(-((catalog["lat"][i] / 0.26) ** 2)) * np.exp(-(((x_n - 0.5) / 0.20) ** 2))
            )
            if bulge_proxy > 0.48 and radius > 2 and rng.random() < 0.68:
                radius = min(radius, int(rng.choice([2, 2, 3])))
            if bulge_proxy > 0.58 and radius > 1 and rng.random() < 0.42:
                radius = 1
        if not foreground_layer and cfg.features.galaxy_view and not cluster_layer:
            z_disk = float(np.exp(-((catalog["lat"][i] / 0.252) ** 2)) ** 1.12)
            color = color * (0.80 + 0.20 * z_disk)
        plat: float | None = (
            float(catalog["lat"][i]) if plane_psf_elongation and cfg.features.galaxy_view else None
        )
        paint_star(img, int(xs[i]), int(ys[i]), radius, color, rng, galactic_lat=plat)
    np.clip(img, 0.0, 1.0, out=img)
    stats = catalog_stats(catalog)
    return {"color_counts": stats.color_counts, "size_counts": stats.size_counts}


def _save_image(img: np.ndarray, path: Path, fmt: str, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")
    if fmt == "jpg":
        pil.save(path, format="JPEG", quality=quality)
    else:
        pil.save(path, format="PNG")


def render_single(
    cfg: RenderConfig,
    generation_index: int,
    on_pass_complete: Callable[[], None] | None = None,
) -> tuple[dict[str, Path], dict[str, dict[str, int]]]:
    seed = (cfg.seed or 0) + generation_index
    seed_seq = np.random.SeedSequence(seed)
    rng_bg, rng_stars_bg, rng_clusters, rng_stars_fg, rng_nebula, rng_post, rng_chroma = [
        np.random.default_rng(s) for s in seed_seq.spawn(7)
    ]
    canvas = _background_plane(
        rng=rng_bg,
        height=cfg.height,
        width=cfg.width,
        enabled=cfg.features.background_gradient,
        black_background=cfg.features.black_background,
    )
    if on_pass_complete:
        on_pass_complete()

    stats = _empty_star_stats()
    stars_bg = np.zeros_like(canvas)
    disk_w = _galactic_disk_weight(cfg.height)
    density_scale = 1.0 + (0.36 if cfg.features.galaxy_view else 0.0)
    band_boost = 1.08 if cfg.features.galaxy_view else 1.0

    nebula_bundle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    if (
        cfg.features.nebula
        and cfg.nebula_mode == NebulaMode.galaxy_streak
        and cfg.features.stars
    ):
        neb0, neb_emit0, dust0, lane0 = generate_nebula(
            rng_nebula, cfg.nebula_mode, cfg.height, cfg.width, cfg.nebula_tuning
        )
        ext0 = _extinction_from_dust_and_lane(dust0, lane0, cfg)
        nebula_bundle = (neb0, neb_emit0, dust0, lane0, ext0)

    if cfg.features.stars:
        cat_bg = sample_star_catalog(
            rng_stars_bg,
            cfg.width,
            cfg.height,
            density_scale,
            layer="background",
            galactic_band_boost=band_boost,
            latitude_color_bias=True,
        )
        if nebula_bundle is not None and cfg.features.galaxy_view:
            reroll_stars_in_dark_lanes(
                cat_bg,
                rng_stars_bg,
                cfg.width,
                cfg.height,
                nebula_bundle[3],
            )
        stats = _add_stars_from_catalog(
            stars_bg,
            rng_stars_bg,
            cfg,
            cat_bg,
            foreground_layer=False,
            galaxy_disk_cool_stars=True,
            point_disk_stars=cfg.features.galaxy_view,
            plane_psf_elongation=cfg.features.galaxy_view,
            cluster_layer=False,
        )
        if cfg.features.galaxy_view:
            cat_cl = sample_cluster_star_catalog(
                rng_clusters, cfg.width, cfg.height, density_scale
            )
            stats_cl = _add_stars_from_catalog(
                stars_bg,
                rng_clusters,
                cfg,
                cat_cl,
                foreground_layer=False,
                galaxy_disk_cool_stars=False,
                point_disk_stars=False,
                plane_psf_elongation=True,
                cluster_layer=True,
            )
            stats = _merge_star_stats(stats, stats_cl)
        if cfg.features.galaxy_view:
            _soft_knee_star_layer(stars_bg, disk_w)
            paint_reference_anchors(stars_bg, rng_stars_bg, cfg)
            np.multiply(stars_bg, 0.80, out=stars_bg, casting="unsafe")
        if on_pass_complete:
            on_pass_complete()
    canvas = np.clip(canvas + stars_bg, 0.0, 1.0)

    ext_paint_for_fg: np.ndarray | None = None
    if cfg.features.nebula:
        if nebula_bundle is not None:
            neb, neb_emit, dust_occlusion, lane_ext, extinction = nebula_bundle
        else:
            neb, neb_emit, dust_occlusion, lane_ext = generate_nebula(
                rng_nebula, cfg.nebula_mode, cfg.height, cfg.width, cfg.nebula_tuning
            )
            extinction = _extinction_from_dust_and_lane(dust_occlusion, lane_ext, cfg)
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            # Balance x/y: heavy y-only preserved longitude stripes in extinction → vertical star columns.
            ext_paint = _blur_separable_xy(extinction, passes=3)
            ext_paint = _blur_x_only(ext_paint, passes=3)
            ext_paint = _blur_y_only(ext_paint, passes=2)
            ext_paint = _blur_separable_xy(ext_paint, passes=2)
            ext_paint = _blur_y_only(ext_paint, passes=2)
        else:
            ext_paint = extinction
        ext_paint_for_fg = ext_paint
        canvas = _apply_extinction_to_canvas(
            canvas,
            ext_paint,
            galaxy_streak=cfg.nebula_mode == NebulaMode.galaxy_streak,
            rng_mottle=rng_nebula,
        )
        neb_luma = np.mean(neb, axis=2)
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            neb_luma = np.clip(neb_luma + np.mean(neb_emit, axis=2) * 0.32, 0.0, 1.0)
        neb_peaks = np.max(neb, axis=2)
        # Non-uniform transparency, but keep nebula emissive so color bands remain visible.
        alpha_floor = 0.22 if cfg.nebula_mode == NebulaMode.galaxy_streak else 0.18
        neb_alpha = np.clip(alpha_floor + (neb_luma**0.66) * (0.68 + 0.52 * neb_peaks), 0.12, 0.90)
        # Low-frequency multiplier — full-res white noise reads as glitter on smooth gas.
        _sr, _sc = max(2, cfg.height // 22), max(2, cfg.width // 22)
        neb_struct = _resize_bilinear(rng_nebula.random((_sr, _sc)), cfg.height, cfg.width)
        neb_struct = _blur_separable_xy(neb_struct, passes=4)
        neb_struct = np.clip(0.972 + 0.045 * neb_struct, 0.965, 1.03)[..., None]
        neb_contrib = neb * neb_alpha[..., None] * neb_struct
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            neb_contrib *= np.array([1.04, 1.00, 1.04], dtype=np.float64)
            # Brighter nebula where stars are less extincted (gas reads behind dust lanes).
            neb_contrib *= 0.94 * (0.88 + 0.12 * ext_paint[..., None])
            # Mild chroma lift in the disk band (counteracts brown stacking from extinction + grading).
            lin = np.clip(neb_contrib, 0.0, 1.0)
            luma = np.mean(lin, axis=2, keepdims=True)
            boosted = np.clip(luma + (lin - luma) * 1.30, 0.0, 1.0)
            yy_s = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float64)[:, None]
            band_s = np.exp(-((yy_s**2) / 0.54))[..., None]
            neb_contrib = np.clip(lin * (1.0 - band_s * 0.26) + boosted * (band_s * 0.34), 0.0, 1.0)
        canvas = np.clip(canvas + neb_contrib, 0.0, 1.0)
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            emit_luma = np.mean(neb_emit, axis=2)
            emit_peaks = np.max(neb_emit, axis=2)
            emit_chr = np.clip(
                np.max(neb_emit, axis=2) - np.min(neb_emit, axis=2),
                0.0,
                1.0,
            )
            emit_metric = np.clip(
                np.maximum(emit_luma * 1.15, emit_chr * 1.65),
                0.0,
                1.0,
            )
            emit_alpha = np.clip(
                0.16 + (emit_metric**0.55) * (0.58 + 0.48 * emit_peaks),
                0.08,
                0.78,
            )
            emit_coarse = _resize_bilinear(
                rng_nebula.random((max(2, cfg.height // 18), max(2, cfg.width // 18))),
                cfg.height,
                cfg.width,
            )
            emit_struct = np.clip(
                0.978 + 0.038 * _blur_separable_xy(emit_coarse, passes=3),
                0.97,
                1.018,
            )[..., None]
            emit_contrib = neb_emit * emit_alpha[..., None] * emit_struct
            emit_contrib *= 1.02 * (0.92 + 0.08 * ext_paint[..., None])
            emit_contrib *= np.array([1.06, 0.97, 1.05], dtype=np.float64)
            peak_gate = np.clip((emit_peaks - 0.06) * 2.2, 0.0, 1.0)[..., None]
            emit_contrib *= 1.0 + peak_gate * 0.22
            em_add = emit_contrib * 1.05
            canvas = np.clip(canvas + em_add + _blur_rgb_separable_xy(em_add, passes=2) * 0.12, 0.0, 1.0)
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            # Low-frequency gold / magenta haze keyed to bright gas + clearer sightlines (moderate).
            gas_w = np.clip(neb_luma * (0.22 + 0.78 * ext_paint), 0.0, 1.0)
            haze = _blur_separable_xy(gas_w, passes=4)
            yy_h = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float64)[:, None]
            band_h = np.exp(-((yy_h**2) / 0.56))
            gold_h = np.array([0.74, 0.56, 0.12], dtype=np.float64)
            mag_h = np.array([0.48, 0.06, 0.34], dtype=np.float64)
            cool_scatter = np.array([0.09, 0.13, 0.22], dtype=np.float64)
            canvas = np.clip(
                canvas
                + (haze * band_h)[..., None] * gold_h * 0.074
                + (haze * band_h)[..., None] * cool_scatter * 0.017
                + band_h[..., None] * np.array([0.14, 0.10, 0.04], dtype=np.float64) * 0.018
                + (np.clip(haze, 0.0, 1.0) ** 1.02 * band_h)[..., None] * mag_h * 0.026,
                0.0,
                1.0,
            )
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            yy0 = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float64)[:, None]
            band_air = np.exp(-((yy0**2) / 0.78))
            air_rgb = np.array([0.012, 0.011, 0.014], dtype=np.float64)
            canvas = np.clip(canvas + band_air[..., None] * air_rgb * 1.15, 0.0, 1.0)
            h, w = cfg.height, cfg.width
            gray = np.clip(
                np.mean(neb_contrib, axis=2) + np.mean(emit_contrib, axis=2) * 0.40,
                0.0,
                1.0,
            )
            bh, bw = max(3, h // 14), max(3, w // 18)
            bloom_small = _resize_bilinear(gray, bh, bw)
            bloom = _resize_bilinear(bloom_small, h, w)
            bloom = np.clip(bloom**1.18, 0.0, 1.0)
            yy = np.linspace(-1.0, 1.0, h)[:, None]
            band_w = np.exp(-((yy**2) / 0.62))
            warm_bloom = np.array([0.56, 0.42, 0.18], dtype=np.float64)
            canvas = np.clip(canvas + (bloom * band_w)[..., None] * warm_bloom * 0.062, 0.0, 1.0)
        if on_pass_complete:
            on_pass_complete()

    if cfg.features.galaxy_view:
        canvas = _apply_separated_disk_sky_grain(
            canvas, rng_post, height=cfg.height, width=cfg.width
        )

    if cfg.features.stars:
        stars_fg = np.zeros_like(canvas)
        cat_fg = sample_star_catalog(
            rng_stars_fg,
            cfg.width,
            cfg.height,
            density_scale,
            layer="foreground",
            latitude_color_bias=False,
        )
        stats_fg = _add_stars_from_catalog(
            stars_fg,
            rng_stars_fg,
            cfg,
            cat_fg,
            foreground_layer=True,
            galaxy_disk_cool_stars=False,
            point_disk_stars=False,
            plane_psf_elongation=cfg.features.galaxy_view,
            cluster_layer=False,
        )
        stats = _merge_star_stats(stats, stats_fg)
        if cfg.features.galaxy_view:
            _soft_knee_star_layer(stars_fg, disk_w, knee=0.42, strength=1.25)
            if (
                ext_paint_for_fg is not None
                and cfg.nebula_mode == NebulaMode.galaxy_streak
                and cfg.features.nebula
            ):
                stars_fg *= ext_paint_for_fg[..., None]
            np.multiply(stars_fg, 0.84, out=stars_fg, casting="unsafe")
        canvas = np.clip(canvas + stars_fg, 0.0, 1.0)
        if on_pass_complete:
            on_pass_complete()

    if cfg.features.galaxy_view:
        yy = np.linspace(-1.0, 1.0, cfg.height)[:, None]
        band = np.exp(-((yy**2) / 0.55))
        sky_w = np.clip(1.0 - band, 0.0, 1.0) ** 0.38
        if cfg.features.long_exposure_look:
            canvas = _apply_long_exposure_look(canvas, rng_post, sky_w=sky_w, band=band, disk_w=disk_w)
        canvas = _apply_galactic_disk_luminance_envelope(canvas, rng_post, disk_w=disk_w)
        canvas = _luma_tone_map_disk(canvas, disk_w, k=0.44)
        canvas = _disk_gamma_lift(canvas, disk_w, gamma=0.91)
        canvas = _disk_photo_grade(canvas, disk_w)
        yy_z = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float64)[:, None]
        zodiac_plane = np.exp(-((yy_z**2) / 0.016))
        zodiac_rgb = np.array([0.0095, 0.0090, 0.0082], dtype=np.float64)
        canvas = np.clip(canvas + (zodiac_plane * 0.20)[..., None] * zodiac_rgb, 0.0, 1.0)
        canvas = _band_micro_ripple(canvas, rng_post, disk_w, strength=0.0042)
        canvas = _chromatic_fringe_disk(canvas, rng_chroma, disk_w)
        canvas = _disk_halation_soft(canvas, disk_w, strength=0.056)
        canvas = np.clip(canvas, 0.0, 1.0)
        if on_pass_complete:
            on_pass_complete()

    if cfg.features.jpeg_artifact_pass and cfg.output_format == "jpg":
        canvas = apply_jpeg_artifacts(canvas, cfg.quality)
        if on_pass_complete:
            on_pass_complete()

    saved: dict[str, Path] = {}
    ext = cfg.output_format.value
    base_name = f"{cfg.output_base_name}_{generation_index:04d}"

    if cfg.projection_mode in {ProjectionMode.equirectangular, ProjectionMode.both}:
        eq_path = cfg.output_dir / f"{base_name}_equirect.{ext}"
        _save_image(canvas, eq_path, ext, cfg.quality)
        saved["equirectangular"] = eq_path
        if on_pass_complete:
            on_pass_complete()

    if cfg.projection_mode in {ProjectionMode.cubemap, ProjectionMode.both}:
        faces = cubemap_faces_from_equirect(canvas, cfg.cubemap_face_size)
        if on_pass_complete:
            on_pass_complete()
        for face_name, face_img in faces.items():
            face_path = cfg.output_dir / f"{base_name}_cube_{face_name}.{ext}"
            _save_image(face_img, face_path, ext, cfg.quality)
            if on_pass_complete:
                on_pass_complete()
        saved["cubemap"] = cfg.output_dir

    return saved, stats
