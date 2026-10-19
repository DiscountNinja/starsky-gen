"""Subtle optical & sensor effects (display-scale, photographic)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import map_coordinates

from starsky_gen.color_science import apply_camera_response_linear, rec709_luma
from starsky_gen.nebula import _blur_separable_xy, _resize_bilinear

if TYPE_CHECKING:
    from starsky_gen.config import FeatureConfig


def inject_sensor_noise(
    rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    shot_scale: float,
    read_sigma: float,
    space: Literal["linear", "display"] = "linear",
) -> np.ndarray:
    """Poisson shot noise (Gaussian approx) + Gaussian read noise.

    *linear* — before tone map (variance ∝ signal in scene units).
    *display* — after tone map (camera pipeline; gated off near-black sky).
    """
    if shot_scale <= 1e-9 and read_sigma <= 1e-10:
        return rgb
    out = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    if shot_scale > 1e-9:
        if space == "display":
            lu = rec709_luma(out)
            sky_gate = np.clip((lu - 0.012) / 0.08, 0.0, 1.0) ** 1.35
            sig = np.sqrt(np.clip(lu * shot_scale, 0.0, None)) * sky_gate
            noise = rng.normal(0.0, 1.0, size=lu.shape) * sig
            scale = np.divide(
                np.maximum(lu + noise, 0.0),
                np.maximum(lu, 1e-10),
                out=np.ones_like(lu),
                where=lu > 1e-10,
            )
            out = out * scale[..., np.newaxis]
        else:
            sig = np.sqrt(np.clip(out * shot_scale, 0.0, None)) + 1e-8
            out = out + rng.normal(0.0, sig)
    if read_sigma > 1e-10:
        out = out + rng.normal(0.0, read_sigma, size=out.shape)
    return np.maximum(out, 0.0)


def apply_lens_vignette(
    rgb: np.ndarray,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Slight vignette: corners darker; blue attenuates more than red (lens + sensor stack)."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = (xx / max(w - 1, 1) - 0.5) * 2.0
    yn = (yy / max(h - 1, 1) - 0.5) * 2.0
    r2 = np.clip(xn * xn + yn * yn, 0.0, 1.8)
    vig_r = 1.0 - 0.17 * s * r2
    vig_g = 1.0 - 0.21 * s * r2
    vig_b = 1.0 - 0.28 * s * r2
    gains = np.stack([vig_r, vig_g, vig_b], axis=-1)
    return np.clip(rgb * gains, 0.0, None)


def _add_gaussian_blob(
    out: np.ndarray,
    gx: float,
    gy: float,
    *,
    sigma: float,
    amp: float,
    tint: np.ndarray,
    periodic_x: bool,
) -> None:
    h, w, _ = out.shape
    half = int(np.clip(round(sigma * 3.2), 3, 16))
    yy = np.arange(-half, half + 1, dtype=np.float64)
    xx = np.arange(-half, half + 1, dtype=np.float64)
    X, Y = np.meshgrid(xx, yy)
    stamp = np.exp(-(X * X + Y * Y) / (2.0 * sigma * sigma)) * float(amp)
    for j, dy in enumerate(yy):
        yi = int(np.round(gy + dy))
        if yi < 0 or yi >= h:
            continue
        for i, dx in enumerate(xx):
            xi_f = gx + dx
            if periodic_x:
                xi = int(np.mod(np.round(xi_f), w))
            else:
                xi = int(np.round(xi_f))
                if xi < 0 or xi >= w:
                    continue
            out[yi, xi, :] = np.maximum(0.0, out[yi, xi, :] + stamp[j, i] * tint)


def paint_hero_lens_ghosts(
    rgb: np.ndarray,
    heroes: list | tuple,
    rng: np.random.Generator,
    *,
    strength: float = 0.055,
    periodic_x: bool = True,
) -> np.ndarray:
    """1–2 soft offset ghost blobs + short spike accent per hero star."""
    s = float(np.clip(strength, 0.0, 0.35))
    if s < 1e-6 or not heroes:
        return rgb
    out = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    h, w, _ = out.shape
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    for hero in heroes:
        x = float(getattr(hero, "x", 0.0))
        y = float(getattr(hero, "y", 0.0))
        flux = float(getattr(hero, "flux", 1.0))
        rgb_t = getattr(hero, "rgb", (1.0, 0.95, 0.9))
        tint = np.clip(np.asarray(rgb_t, dtype=np.float64).reshape(-1)[:3], 0.0, 1.5)
        if tint.shape[0] < 3:
            tint = np.array([1.0, 0.95, 0.9], dtype=np.float64)
        amp = s * float(np.clip(0.55 + 0.12 * np.log10(max(flux, 1.0)), 0.35, 1.35))
        vx, vy = cx - x, cy - y
        vn = float(np.hypot(vx, vy)) + 1e-6
        g1x = (x + vx * float(rng.uniform(0.18, 0.34))) % float(w) if periodic_x else float(
            np.clip(x + vx * float(rng.uniform(0.18, 0.34)), 0.0, w - 1)
        )
        g1y = float(np.clip(y + vy * float(rng.uniform(0.18, 0.34)), 0.0, h - 1))
        g2x = x - (vx / vn) * float(rng.uniform(12.0, 36.0))
        g2y = y - (vy / vn) * float(rng.uniform(8.0, 28.0))
        if periodic_x:
            g2x = g2x % float(w)
        else:
            g2x = float(np.clip(g2x, 0.0, w - 1))
        g2y = float(np.clip(g2y, 0.0, h - 1))
        _add_gaussian_blob(
            out, g1x, g1y, sigma=float(rng.uniform(2.8, 5.5)), amp=amp * 0.55, tint=tint, periodic_x=periodic_x
        )
        _add_gaussian_blob(
            out, g2x, g2y, sigma=float(rng.uniform(1.8, 3.6)), amp=amp * 0.28, tint=tint, periodic_x=periodic_x
        )
        ang = float(rng.uniform(0.0, np.pi))
        length = float(rng.uniform(10.0, 22.0))
        ts = np.linspace(-length, length, int(length * 2) + 1)
        for t in ts:
            fall = float(np.exp(-(abs(t) / max(length, 1e-3)) ** 1.6))
            for arm_scale, arm_amp in ((1.0, 0.42), (0.55, 0.26)):
                a = ang if arm_scale >= 0.99 else ang + 0.5 * np.pi
                px = x + np.cos(a) * t * arm_scale
                py = y + np.sin(a) * t * arm_scale
                if periodic_x:
                    pxi = int(np.mod(np.round(px), w))
                else:
                    if px < 0 or px >= w:
                        continue
                    pxi = int(np.round(px))
                if py < 0 or py >= h:
                    continue
                pyi = int(np.round(py))
                out[pyi, pxi, :] = np.maximum(0.0, out[pyi, pxi, :] + fall * amp * arm_amp * tint)
    return out


def apply_thin_film_lens_flare(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 0.032,
    periodic_x: bool = True,
) -> np.ndarray:
    """Thin-film interference tint on lens flare / saturated cores (angle-varying RGB)."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    lin = np.maximum(rgb.astype(np.float64), 0.0)
    h, w, _ = lin.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    ang = np.arctan2(yy - cy, xx - cx) + float(rng.uniform(0.0, 6.283185307179586))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    # First-order thin-film fringes (approximate).
    fringe = 0.5 + 0.5 * np.sin(ang * 3.2 + phase)
    film_rgb = np.stack(
        [
            0.55 + 0.45 * np.sin(ang * 2.1 + phase),
            0.50 + 0.50 * np.cos(ang * 2.4 + phase * 0.7),
            0.60 + 0.40 * np.sin(ang * 1.8 + phase * 1.3),
        ],
        axis=-1,
    )
    film_rgb = np.clip(film_rgb, 0.35, 1.25)
    lu = rec709_luma(lin)
    hot = np.clip((lu - 0.62) / 0.32, 0.0, 1.0) ** 1.2
    dw = np.clip(np.broadcast_to(disk_w, hot.shape), 0.0, 1.0)
    gate = hot * dw * fringe
    halo_l = _blur_separable_xy(gate, passes=3, periodic_x=periodic_x)
    halo_l = _blur_separable_xy(halo_l, passes=2, periodic_x=periodic_x)
    return np.clip(lin + halo_l[..., np.newaxis] * film_rgb * s * 0.11, 0.0, None)


def apply_wavelength_halation(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float = 0.038,
    sat_threshold: float = 0.68,
    periodic_x: bool = True,
) -> np.ndarray:
    """Soft wavelength-dependent halos on near-saturated nebula / bright disk (R wide, B tight)."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    lin = np.maximum(rgb.astype(np.float64), 0.0)
    lu = rec709_luma(lin)
    hot = np.clip((lu - sat_threshold) / max(1.0 - sat_threshold, 0.08), 0.0, 1.0) ** 1.15
    dw = np.clip(np.broadcast_to(disk_w, hot.shape), 0.0, 1.0)
    hot = hot * dw
    if float(np.max(hot)) < 1e-8:
        return lin
    halo_l = _blur_separable_xy(hot, passes=4, periodic_x=periodic_x)
    halo_l = _blur_separable_xy(halo_l, passes=3, periodic_x=periodic_x)
    local_rgb = np.maximum(
        np.sum(lin * halo_l[..., np.newaxis], axis=(0, 1)) / (float(np.sum(halo_l)) + 1e-8),
        1e-8,
    )
    local_rgb = local_rgb / np.maximum(np.max(local_rgb), 1e-8)
    halo_rgb = local_rgb.reshape(1, 1, 3) * halo_l[..., np.newaxis]
    return np.clip(lin + halo_rgb * s * 0.07, 0.0, None)


def apply_lateral_chromatic_aberration(
    rgb: np.ndarray,
    *,
    strength: float = 1.0,
    max_shift_px: float = 0.22,
    periodic_x: bool = True,
) -> np.ndarray:
    """Sub-pixel lateral CA toward frame edges, strongest on bright cores."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    lin = np.maximum(rgb.astype(np.float64), 0.0)
    h, w, _ = lin.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    xn = (xx - cx) / max(cx, 1.0)
    yn = (yy - cy) / max(cy, 1.0)
    r = np.sqrt(xn * xn + yn * yn)
    edge = np.clip((r - 0.35) / 0.75, 0.0, 1.0) ** 1.12
    lu = rec709_luma(lin)
    bright = np.clip((lu - 0.52) / 0.42, 0.0, 1.0) ** 1.15
    wgt = edge * bright * s * 0.65
    if float(np.max(wgt)) < 1e-8:
        return lin
    ux = xn / (r + 1e-6)
    uy = yn / (r + 1e-6)
    shift = wgt * float(max_shift_px)
    dy_r = yy + uy * shift
    dx_r = xx + ux * shift
    dy_b = yy - uy * shift
    dx_b = xx - ux * shift
    if periodic_x:
        dx_r = np.mod(dx_r, w)
        dx_b = np.mod(dx_b, w)
    else:
        dx_r = np.clip(dx_r, 0.0, w - 1.0)
        dx_b = np.clip(dx_b, 0.0, w - 1.0)
    dy_r = np.clip(dy_r, 0.0, h - 1.0)
    dy_b = np.clip(dy_b, 0.0, h - 1.0)
    red_s = map_coordinates(lin[..., 0], [dy_r, dx_r], order=1, mode="nearest")
    blu_s = map_coordinates(lin[..., 2], [dy_b, dx_b], order=1, mode="nearest")
    out = lin.copy()
    out[..., 0] = lin[..., 0] * (1.0 - wgt) + red_s * wgt
    out[..., 2] = lin[..., 2] * (1.0 - wgt) + blu_s * wgt
    return np.clip(out, 0.0, None)


def apply_film_grain_display(
    rgb: np.ndarray,
    rng: np.random.Generator,
    strength: float,
    *,
    periodic_x: bool = True,
) -> np.ndarray:
    """Film grain after tone map: luma-shaped (strongest in mid-tones) + coarse/fine spatial mix."""
    from starsky_gen.dither import blue_noise_tile

    s = float(strength)
    if s <= 1e-7:
        return rgb
    lin = np.maximum(rgb.astype(np.float64), 0.0)
    h, wd, _ = lin.shape
    tile = blue_noise_tile(64, seed=int(rng.integers(0, 2**31 - 1)))
    th, tw = tile.shape
    yy = np.arange(h, dtype=np.int64)[:, None] % th
    xx = np.arange(wd, dtype=np.int64)[None, :] % tw
    n_fine = (tile[yy, xx] - 0.5) * 2.0
    n_fine = _blur_separable_xy(n_fine, passes=1, periodic_x=periodic_x)
    cr_h, cr_w = max(8, h // 16), max(8, wd // 16)
    coarse = rng.normal(0.0, 1.0, size=(cr_h, cr_w))
    coarse = _resize_bilinear(coarse, h, wd, periodic_x=periodic_x)
    coarse = _blur_separable_xy(coarse, passes=2, periodic_x=periodic_x)
    n = 0.62 * n_fine + 0.38 * coarse
    n = (n - np.mean(n)) / (np.std(n) + 1e-6)
    lu = rec709_luma(lin)
    # Mid-tone bell; suppress near-black sky and clipped highlights.
    mid_bell = np.exp(-((lu - 0.28) ** 2) / (2.0 * 0.16**2))
    sky_gate = np.clip((lu - 0.012) / 0.10, 0.0, 1.0) ** 1.25
    hi_gate = 1.0 - np.clip((lu - 0.82) / 0.15, 0.0, 1.0) ** 1.1
    gate = np.sqrt(np.clip(lu, 0.0, 1.0)) * mid_bell * sky_gate * hi_gate
    grain_l = lu * (1.0 + n * s * gate * 1.15)
    scale = np.divide(
        grain_l,
        np.maximum(lu, 1e-10),
        out=np.ones_like(lu),
        where=lu > 1e-10,
    )
    return np.clip(lin * scale[..., np.newaxis], 0.0, 1.0)


def apply_isp_linear_chain(
    rgb: np.ndarray,
    *,
    strength: float = 1.0,
    wb_strength: float = 0.38,
) -> np.ndarray:
    """Post–linear-noise ISP: white balance + demosaic-like low-pass gamut mapping."""
    s = float(np.clip(strength, 0.0, 1.5))
    if s < 1e-6:
        return rgb
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    mean = np.mean(lin, axis=(0, 1))
    gray = float(np.mean(mean))
    wb = 1.0 + (gray / (mean + 1e-6) - 1.0) * float(wb_strength) * s
    lin = lin * wb.reshape(1, 1, 3)
    # Demosaic-like: per-channel mild cross-talk blur (Bayer-ish low-pass).
    ch = [lin[..., c] for c in range(3)]
    blur = [_blur_separable_xy(c, passes=1, periodic_x=False) for c in ch]
    mix = 0.62 * s
    lin = np.stack(
        [
            ch[0] * (1.0 - mix) + blur[0] * mix * 0.92 + blur[1] * mix * 0.05 + blur[2] * mix * 0.03,
            ch[1] * (1.0 - mix) + blur[1] * mix * 0.90 + blur[0] * mix * 0.05 + blur[2] * mix * 0.05,
            ch[2] * (1.0 - mix) + blur[2] * mix * 0.92 + blur[1] * mix * 0.05 + blur[0] * mix * 0.03,
        ],
        axis=-1,
    )
    lin = apply_camera_response_linear(lin, exposure=1.0)
    return np.clip(lin, 0.0, None)


def apply_optical_display_pass(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    features: FeatureConfig,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
) -> np.ndarray:
    """Lens vignette, thin-film flare, halation, lateral CA (post tone map; grain applied later)."""
    canvas = np.maximum(rgb.astype(np.float64), 0.0)
    canvas = apply_lens_vignette(canvas, strength=features.vignette_strength)
    canvas = apply_thin_film_lens_flare(
        canvas,
        disk_w,
        rng,
        strength=features.lens_flare_thin_film_strength,
        periodic_x=periodic_x,
    )
    canvas = apply_wavelength_halation(
        canvas,
        disk_w,
        strength=features.halation_strength,
        periodic_x=periodic_x,
    )
    canvas = apply_lateral_chromatic_aberration(
        canvas,
        strength=features.chromatic_aberration_strength,
        periodic_x=periodic_x,
    )
    return canvas
