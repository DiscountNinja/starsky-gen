"""CPU-side FBM / ridged noise, domain warp, carve, and edge-preserving dust alpha."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter


def _resize_bilinear(
    field: np.ndarray, out_h: int, out_w: int, *, periodic_x: bool = False
) -> np.ndarray:
    """2D bilinear resize; optional torus wrap on X for equirect seams."""
    a = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    if periodic_x and a.shape[1] >= 2:
        a = np.concatenate([a, a[:, :1]], axis=1)
    im = Image.fromarray(a, mode="F")
    out = np.asarray(im.resize((out_w, out_h), Image.Resampling.BILINEAR), dtype=np.float64)
    return np.clip(out, 0.0, 1.0)


def gaussian_blur_pil(
    field: np.ndarray, radius: float, *, periodic_x: bool = False
) -> np.ndarray:
    """Gaussian blur via uint8 L path (Pillow rejects mode F for GaussianBlur)."""
    a = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
    if periodic_x and a.shape[1] >= 2:
        a = np.concatenate([a, a[:, :1]], axis=1)
    u8 = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    im = Image.fromarray(u8, mode="L")
    r = max(0.01, float(radius))
    out8 = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=r)), dtype=np.float64) / 255.0
    if periodic_x and out8.shape[1] > field.shape[1]:
        out8 = out8[:, : field.shape[1]]
    return np.clip(out8, 0.0, 1.0)


def bilateral_edge_preserve(
    field: np.ndarray,
    *,
    spatial_radius: float = 1.5,
    range_sigma: float = 0.08,
    periodic_x: bool = False,
) -> np.ndarray:
    """Approximate bilateral: edge-stopping weight from range kernel + Gaussian smooth."""
    f = np.clip(field.astype(np.float64), 0.0, 1.0)
    smooth = gaussian_blur_pil(f, spatial_radius, periodic_x=periodic_x)
    fine = gaussian_blur_pil(f, spatial_radius * 0.45, periodic_x=periodic_x)
    g = np.abs(f - fine)
    rs = max(range_sigma, 1e-5)
    w = np.exp(-((g / rs) ** 2))
    w = np.clip(w, 0.0, 1.0)
    out = f * w + smooth * (1.0 - w)
    return np.clip(out, 0.0, 1.0)


def bilinear_sample(
    field: np.ndarray,
    yi: np.ndarray,
    xi: np.ndarray,
    *,
    periodic_x: bool = False,
) -> np.ndarray:
    """Sample `field` (H,W) at fractional indices `yi`, `xi` same shape."""
    h, w = field.shape
    xi = np.clip(xi, 0.0, w - 1 - 1e-7)
    yi = np.clip(yi, 0.0, h - 1 - 1e-7)
    x0 = np.floor(xi).astype(np.int64)
    y0 = np.floor(yi).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    tx = (xi - x0.astype(np.float64)).astype(np.float64)
    ty = (yi - y0.astype(np.float64)).astype(np.float64)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    if periodic_x:
        x0m = np.mod(x0, w).astype(np.int64)
        x1m = np.mod(x1, w).astype(np.int64)
    else:
        x0m = np.clip(x0, 0, w - 1)
        x1m = np.clip(x1, 0, w - 1)
    a00 = field[y0c, x0m]
    a01 = field[y0c, x1m]
    a10 = field[y1c, x0m]
    a11 = field[y1c, x1m]
    a0 = a00 * (1.0 - tx) + a01 * tx
    a1 = a10 * (1.0 - tx) + a11 * tx
    return a0 * (1.0 - ty) + a1 * ty


def _octave_layer(
    rng: np.random.Generator,
    height: int,
    width: int,
    cells_h: int,
    cells_w: int,
    *,
    periodic_x: bool,
) -> np.ndarray:
    cells_h = max(2, cells_h)
    cells_w = max(2, cells_w)
    coarse = rng.random((cells_h, cells_w))
    return _resize_bilinear(coarse, height, width, periodic_x=periodic_x)


def fbm2d(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    base_scale: float,
    octaves: int,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    periodic_x: bool = False,
    elongate_along_x: float = 1.0,
    blur_coarse_more: bool = False,
) -> np.ndarray:
    """Value FBM; `base_scale` is cycles-per-image-width (~0.003 = very coarse)."""
    acc = np.zeros((height, width), dtype=np.float64)
    amp_sum = 0.0
    amp = 1.0
    fmul = 1.0
    for o in range(max(1, octaves)):
        f_eff = base_scale * fmul
        f_eff = max(f_eff, 2.0 / max(width, height))
        raw_cw = int(width * f_eff / max(elongate_along_x, 1e-6))
        raw_ch = int(height * f_eff)
        cells_w = max(2, min(width, raw_cw))
        cells_h = max(2, min(height, raw_ch))
        layer = _octave_layer(rng, height, width, cells_h, cells_w, periodic_x=periodic_x)
        if blur_coarse_more:
            # Blur coarse octaves more than fine — reduces speckle / repeating sand.
            radius = max(0.35, 2.8 / (o + 1.15))
            layer = gaussian_blur_pil(layer, radius, periodic_x=periodic_x)
        acc += amp * layer
        amp_sum += amp
        amp *= gain
        fmul *= lacunarity
    return acc / max(amp_sum, 1e-9)


def smooth_perlin2d(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    base_scale: float,
    periodic_x: bool = False,
    elongate_along_x: float = 1.0,
) -> np.ndarray:
    """Very smooth Perlin-like base (2 octaves, heavy blur)."""
    field = fbm2d(
        rng,
        height,
        width,
        base_scale=base_scale,
        octaves=2,
        lacunarity=2.0,
        gain=0.55,
        periodic_x=periodic_x,
        elongate_along_x=elongate_along_x,
        blur_coarse_more=True,
    )
    return gaussian_blur_pil(field, 2.2, periodic_x=periodic_x)


def ridged_fbm2d(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    base_scale: float,
    octaves: int,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    periodic_x: bool = False,
    elongate_along_x: float = 1.0,
) -> np.ndarray:
    """Ridged multifractal: per octave ridge = 1 - |2n-1| on octave noise n in [0,1]."""
    acc = np.zeros((height, width), dtype=np.float64)
    amp_sum = 0.0
    amp = 1.0
    fmul = 1.0
    for _ in range(max(1, octaves)):
        f_eff = base_scale * fmul
        f_eff = max(f_eff, 2.0 / max(width, height))
        raw_cw = int(width * f_eff / max(elongate_along_x, 1e-6))
        raw_ch = int(height * f_eff)
        cells_w = max(2, min(width, raw_cw))
        cells_h = max(2, min(height, raw_ch))
        layer = _octave_layer(rng, height, width, cells_h, cells_w, periodic_x=periodic_x)
        ridge = 1.0 - np.abs(layer * 2.0 - 1.0)
        acc += amp * ridge
        amp_sum += amp
        amp *= gain
        fmul *= lacunarity
    return acc / max(amp_sum, 1e-9)


def domain_warp_displacement(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    warp_scale: float,
    periodic_x: bool,
    elongate_along_x: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-frequency (wx, wy) in roughly [-1, 1] for pixel-space offsets."""
    f_eff = max(warp_scale, 4.0 / max(width, height))
    raw_cw = int(width * f_eff / max(elongate_along_x, 1e-6))
    raw_ch = int(height * f_eff)
    cells_w = max(2, min(width, raw_cw))
    cells_h = max(2, min(height, raw_ch))
    wx = _octave_layer(rng, height, width, cells_h, cells_w, periodic_x=periodic_x)
    wy = _octave_layer(rng, height, width, cells_h, cells_w, periodic_x=periodic_x)
    wx = (wx - 0.5) * 2.0
    wy = (wy - 0.5) * 2.0
    return wx, wy


def resample_warped(
    field: np.ndarray,
    disp_x_px: np.ndarray,
    disp_y_px: np.ndarray,
    *,
    periodic_x: bool,
) -> np.ndarray:
    h, w = field.shape
    yi = np.linspace(0.0, h - 1, h, dtype=np.float64)[:, None] + disp_y_px
    xi = np.linspace(0.0, w - 1, w, dtype=np.float64)[None, :] + disp_x_px
    return bilinear_sample(field, yi, xi, periodic_x=periodic_x)


def selective_subtract_carve(
    field: np.ndarray,
    *,
    shift_x_px: float,
    blur_passes: int,
    strength: float,
    periodic_x: bool,
    blur_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    """D - strength * shift_blur(D) for channel-like carving (blur_fn = separable blur)."""
    shifted = np.roll(field, int(round(shift_x_px)), axis=1)
    if periodic_x and abs(round(shift_x_px)) > 0:
        pass
    elif not periodic_x and round(shift_x_px) != 0:
        shifted = np.clip(shifted, 0.0, 1.0)
    blurred = blur_fn(shifted, passes=max(1, blur_passes), periodic_x=periodic_x)
    out = field - strength * blurred
    return np.clip(out, 0.0, 1.0)


def combine_cloud_layers(
    base: np.ndarray,
    ridged: np.ndarray,
    fine: np.ndarray,
    band_mask: np.ndarray,
    *,
    w_base: float,
    w_ridge: float,
    w_fine: float,
) -> np.ndarray:
    """Weighted combine (legacy helper for distant/full)."""
    d = w_base * base + w_ridge * ridged + w_fine * fine
    d = np.clip(d, 0.0, 1.0) * np.clip(band_mask, 0.0, 1.0)
    lo = float(np.quantile(d, 0.02))
    hi = float(np.quantile(d, 0.995))
    if hi <= lo + 1e-9:
        return np.clip(d, 0.0, 1.0)
    d = (d - lo) / (hi - lo)
    return np.clip(d, 0.0, 1.0)


def _pct_stretch(a: np.ndarray, lo_pct: float = 1.5, hi_pct: float = 99.2) -> np.ndarray:
    vlo = float(np.percentile(a, lo_pct))
    vhi = float(np.percentile(a, hi_pct))
    if vhi <= vlo + 1e-9:
        return np.clip(a, 0.0, 1.0)
    return np.clip((a - vlo) / (vhi - vlo), 0.0, 1.0)


def assemble_galaxy_dust_alpha(
    base: np.ndarray,
    filaments: np.ndarray,
    fine: np.ndarray,
    band_mask: np.ndarray,
    *,
    periodic_x: bool,
) -> dict[str, np.ndarray]:
    """Recipe: band-masked base + ridged carve + fine detail, smoothstep, bilateral."""
    band = np.clip(band_mask, 0.0, 1.0)
    b = np.clip(base * band, 0.0, 1.0)
    f_lin = np.clip(filaments * band, 0.0, 1.0)
    # Sharper ridge response (dark lanes read darker between bright filaments).
    f = np.clip((f_lin * 1.14 - 0.035), 0.0, 1.0)
    f = np.clip(f**0.88, 0.0, 1.0)
    fil_dense = f
    blurred_f = gaussian_blur_pil(f, 2.5, periodic_x=periodic_x)
    carve = np.clip(f - blurred_f * 0.45, 0.0, 1.0)
    carve = np.clip(carve**1.2, 0.0, 1.0)
    d_raw = np.clip(fine * band, 0.0, 1.0)
    clump_hf = _pct_stretch(np.clip(fine * band * 1.35 - 0.08, 0.0, 1.0) ** 1.4, 5.0, 97.0)
    # Coarse scales blurred more than fine detail (anti-speckle).
    b_soft = gaussian_blur_pil(b, 2.8, periodic_x=periodic_x)
    d_soft = gaussian_blur_pil(d_raw, 0.55, periodic_x=periodic_x)
    b_n = _pct_stretch(b_soft, 2.0, 99.0)
    d_n = (_pct_stretch(d_soft, 2.0, 98.5) * 0.92 + 0.04) * (1.0 + 0.12 * clump_hf)
    dust_raw = np.clip(0.62 * b_n + 1.0 * carve + 0.045 * d_n, 0.0, 1.0)
    lo = float(np.percentile(dust_raw, 4.0))
    hi = float(np.percentile(dust_raw, 98.5))
    if hi <= lo + 1e-9:
        dust_alpha = np.clip(dust_raw, 0.0, 1.0)
    else:
        dust_alpha = np.clip((dust_raw - lo) / (hi - lo), 0.0, 1.0)
    dust_alpha = bilateral_edge_preserve(
        dust_alpha, spatial_radius=1.5, range_sigma=0.08, periodic_x=periodic_x
    )
    return {
        "base_masked": b,
        "fil_dense": fil_dense,
        "carve": carve,
        "detail": d_raw,
        "dust_raw": dust_raw,
        "dust_alpha": dust_alpha,
    }


def curl_noise2d(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    base_scale: float,
    periodic_x: bool,
    elongate_along_x: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Divergence-free 2D curl field (vx, vy) from Perlin potential gradients."""
    pot = smooth_perlin2d(
        rng,
        height,
        width,
        base_scale=base_scale,
        periodic_x=periodic_x,
        elongate_along_x=elongate_along_x,
    )
    if periodic_x:
        dpx = (np.roll(pot, -1, axis=1) - np.roll(pot, 1, axis=1)) * 0.5
    else:
        dpx = np.zeros_like(pot)
        dpx[:, 1:-1] = (pot[:, 2:] - pot[:, :-2]) * 0.5
        dpx[:, 0] = pot[:, 1] - pot[:, 0]
        dpx[:, -1] = pot[:, -1] - pot[:, -2]
    dpy = np.zeros_like(pot)
    dpy[1:-1, :] = (pot[2:, :] - pot[:-2, :]) * 0.5
    dpy[0, :] = pot[1, :] - pot[0, :]
    dpy[-1, :] = pot[-1, :] - pot[-2, :]
    vx = np.clip(-dpy, -1.0, 1.0)
    vy = np.clip(dpx, -1.0, 1.0)
    return vx, vy


def log_spiral_arm_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    arms: int = 2,
    pitch: float = 0.24,
    strength: float = 0.55,
    phase: float = 0.0,
) -> np.ndarray:
    """Logarithmic spiral arm density in [-1,1]² longitude/latitude coordinates."""
    r = np.sqrt(x**2 + (y * 2.8) ** 2) + 1e-4
    theta = np.arctan2(y, x)
    spiral = np.cos(float(arms) * theta - np.log(r + 0.08) / max(float(pitch), 0.08) + float(phase))
    arm = np.clip(0.5 + 0.5 * spiral, 0.0, 1.0)
    disk = np.exp(-((y**2) / 0.42))
    return np.clip(arm * disk * float(strength) + (1.0 - float(strength)) * disk, 0.0, 1.0)


def sersic_bulge_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_sersic: float = 2.0,
    effective_radius: float = 0.32,
    strength: float = 0.48,
    center_x: float = 0.0,
) -> np.ndarray:
    """Elliptical Sérsic bulge envelope for arm density modulation."""
    dx = x - float(center_x)
    r = np.sqrt(dx**2 + (y * 1.35) ** 2) / max(float(effective_radius), 0.05)
    n = max(float(n_sersic), 0.5)
    b_n = 2.0 * n - 1.0 / 3.0 + 0.009877 * (n - 0.8)
    prof = np.exp(-b_n * (np.power(np.clip(r, 0.0, 8.0), 1.0 / n) - 1.0))
    return np.clip(prof * float(strength) + (1.0 - float(strength)), 0.0, 1.0)


def galaxy_arm_density_modulator(
    height: int,
    width: int,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    spiral_strength: float,
    n_sersic: float = 1.6,
    periodic_x: bool,
) -> np.ndarray:
    """Combine Sérsic bulge + logarithmic spiral arms for gas/dust density."""
    cx = float(rng.uniform(-0.18, 0.18))
    arms = int(rng.integers(2, 4))
    phase = float(rng.uniform(0.0, 6.283185307179586))
    spiral = log_spiral_arm_mask(
        x, y, arms=arms, pitch=float(rng.uniform(0.18, 0.30)), strength=spiral_strength, phase=phase
    )
    bulge = sersic_bulge_mask(x, y, n_sersic=n_sersic, strength=0.42 + 0.35 * spiral_strength, center_x=cx)
    mod = np.clip(spiral * 0.62 + bulge * 0.38, 0.0, 1.0)
    mod = gaussian_blur_pil(mod, 1.8, periodic_x=periodic_x)
    return np.clip(0.55 + 0.45 * mod, 0.0, 1.0)


def fractal_turbulence_bands(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    n_bands: int,
    periodic_x: bool,
    elongate_along_x: float,
) -> list[np.ndarray]:
    """1–3 octave-scale turbulence layers (FBM + curl advection) with distinct variance."""
    bands: list[np.ndarray] = []
    scales = (0.0048, 0.011, 0.028)[: max(1, min(int(n_bands), 3))]
    gains = (0.42, 0.55, 0.68)
    for i, sc in enumerate(scales):
        rng_i = np.random.default_rng(int(rng.integers(0, 2**31)))
        turb = fbm2d(
            rng_i,
            height,
            width,
            base_scale=sc,
            octaves=3 + i,
            lacunarity=2.05,
            gain=gains[i],
            periodic_x=periodic_x,
            elongate_along_x=elongate_along_x,
            blur_coarse_more=(i == 0),
        )
        vx, vy = curl_noise2d(
            rng_i, height, width, base_scale=sc * 2.2, periodic_x=periodic_x, elongate_along_x=elongate_along_x
        )
        warp_amp = 0.04 * (1.0 + 0.25 * i)
        yy = np.arange(height, dtype=np.float64)[:, None]
        xx = np.arange(width, dtype=np.float64)[None, :]
        yi = np.clip(yy + vy * height * warp_amp, 0, height - 1)
        xi = xx + vx * width * warp_amp
        if periodic_x:
            xi = np.mod(xi, width)
        else:
            xi = np.clip(xi, 0, width - 1)
        warped = bilinear_sample(turb, yi, xi, periodic_x=periodic_x)
        v = float(rng.uniform(0.85, 1.15))
        bands.append(np.clip(warped * v, 0.0, 1.0))
    return bands


def build_galaxy_streak_noise_stack(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool,
    elongate_along_x: float,
    fine_mix: float = 0.78,
    spiral_strength: float = 0.0,
    turbulence_octaves: int = 3,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Multi-scale Perlin + ridged FBM + fine FBM + optional spiral/curl turbulence."""
    el = max(elongate_along_x, 1e-3)
    rng_warp = np.random.default_rng(int(rng.integers(0, 2**31)))
    rng_base = np.random.default_rng(int(rng.integers(0, 2**31)))
    rng_ridge = np.random.default_rng(int(rng.integers(0, 2**31)))
    rng_fine = np.random.default_rng(int(rng.integers(0, 2**31)))
    # Two decorrelated low fields → curl-like perpendicular mixing (twist along plane).
    n1, n2 = domain_warp_displacement(
        rng_warp, height, width, warp_scale=0.01, periodic_x=periodic_x, elongate_along_x=el
    )
    u1, u2 = domain_warp_displacement(
        rng_warp, height, width, warp_scale=0.011, periodic_x=periodic_x, elongate_along_x=el * 1.08
    )
    cx = np.clip((n1 - u2) * 0.55, -1.0, 1.0)
    cy = np.clip((n2 + u1) * 0.55, -1.0, 1.0)
    # Anisotropic filament warp: stronger along X (galactic longitude), milder across band.
    fil_amp_x = 0.06 * float(width) * 0.175
    fil_amp_y = 0.06 * float(height) * 0.072 * (1.55 / el)
    fx = cx * fil_amp_x
    fy = cy * fil_amp_y
    fxf = fx * 0.58
    fyf = fy * 0.58

    perlin_base = smooth_perlin2d(
        rng_base,
        height,
        width,
        base_scale=0.0032,
        periodic_x=periodic_x,
        elongate_along_x=el,
    )
    fbm_body = fbm2d(
        rng_base,
        height,
        width,
        base_scale=0.0055,
        octaves=3,
        lacunarity=2.0,
        gain=0.50,
        periodic_x=periodic_x,
        elongate_along_x=el,
        blur_coarse_more=True,
    )
    base = np.clip(perlin_base * 0.55 + fbm_body * 0.45, 0.0, 1.0)
    base = gaussian_blur_pil(base, 1.6, periodic_x=periodic_x)
    ridge_src = ridged_fbm2d(
        rng_ridge,
        height,
        width,
        base_scale=0.014,
        octaves=5,
        lacunarity=2.0,
        gain=0.75,
        periodic_x=periodic_x,
        elongate_along_x=el,
    )
    filaments = resample_warped(ridge_src, fx, fy, periodic_x=periodic_x)
    fine_src = fbm2d(
        rng_fine,
        height,
        width,
        base_scale=0.08,
        octaves=5,
        lacunarity=2.0,
        gain=0.42,
        periodic_x=periodic_x,
        elongate_along_x=el * 1.05,
        blur_coarse_more=False,
    )
    fine = resample_warped(fine_src, fxf, fyf, periodic_x=periodic_x)
    fine = gaussian_blur_pil(fine, 0.45, periodic_x=periodic_x) * float(np.clip(fine_mix, 0.35, 1.0))

    density_mod = np.ones((height, width), dtype=np.float64)
    if spiral_strength > 1e-5 and x_grid is not None and y_grid is not None:
        density_mod = galaxy_arm_density_modulator(
            height,
            width,
            x_grid,
            y_grid,
            rng,
            spiral_strength=float(spiral_strength),
            periodic_x=periodic_x,
        )
    turb_bands = fractal_turbulence_bands(
        rng,
        height,
        width,
        n_bands=turbulence_octaves,
        periodic_x=periodic_x,
        elongate_along_x=el,
    )
    if turb_bands:
        wts = (0.38, 0.34, 0.28)[: len(turb_bands)]
        turb_mix = sum(w * b for w, b in zip(wts, turb_bands, strict=False))
        turb_mix = turb_mix / max(sum(wts), 1e-9)
        base = np.clip(base * (0.72 + 0.28 * turb_mix) * density_mod, 0.0, 1.0)
        filaments = np.clip(filaments * (0.78 + 0.22 * turb_mix) * density_mod, 0.0, 1.0)
        fine = np.clip(fine * (0.82 + 0.18 * turb_bands[-1]) * density_mod, 0.0, 1.0)
    else:
        base = np.clip(base * density_mod, 0.0, 1.0)
        filaments = np.clip(filaments * density_mod, 0.0, 1.0)
        fine = np.clip(fine * density_mod, 0.0, 1.0)

    return {
        "base": base,
        "ridged": filaments,
        "fine": fine,
        "warp_x": cx,
        "warp_y": cy,
        "density_mod": density_mod,
        "turbulence": turb_bands[0] if turb_bands else base,
    }


def build_simple_noise_stack(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool,
    preset: str,
) -> dict[str, np.ndarray]:
    """Lighter stack for `distant` / `full` (suppress high-frequency vs galaxy)."""
    elong = 1.15 if preset == "distant" else 1.35
    base = fbm2d(
        rng,
        height,
        width,
        base_scale=0.006,
        octaves=3,
        gain=0.55,
        periodic_x=periodic_x,
        elongate_along_x=elong,
    )
    wx, wy = domain_warp_displacement(
        rng, height, width, warp_scale=0.02, periodic_x=periodic_x, elongate_along_x=elong
    )
    amp_x = 4.0 if preset == "distant" else 5.5
    amp_y = 3.0 if preset == "distant" else 4.0
    ridge_src = ridged_fbm2d(
        rng,
        height,
        width,
        base_scale=0.035,
        octaves=4,
        gain=0.55,
        periodic_x=periodic_x,
        elongate_along_x=elong,
    )
    ridged = resample_warped(ridge_src, wx * amp_x, wy * amp_y, periodic_x=periodic_x)
    fine = fbm2d(
        rng,
        height,
        width,
        base_scale=0.11,
        octaves=4,
        gain=0.42,
        periodic_x=periodic_x,
        elongate_along_x=elong * 1.05,
    )
    return {"base": base, "ridged": ridged, "fine": fine, "warp_x": wx, "warp_y": wy}
