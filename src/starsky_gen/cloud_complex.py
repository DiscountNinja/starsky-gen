"""Full-FOV cloud-complex envelopes for scene_mode=nebula (no galactic band)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from starsky_gen.procedural_noise import (
    _resize_bilinear,
    domain_warp_displacement,
    ridged_fbm2d,
)


def _gauss_float(
    field: np.ndarray,
    sigma: float,
    *,
    periodic_x: bool = True,
) -> np.ndarray:
    """Isotropic float gaussian: wrap on X, edge on Y (equirect-safe).

    Uses equal-sigma 1D passes — avoids PIL uint8 barcodes and the old 2D
    wrap-pad path that left a mild vertical bias (gx/gy < 1).
    """
    a = np.asarray(field, dtype=np.float64)
    sig = float(max(sigma, 1e-3))
    x_mode = "wrap" if periodic_x else "nearest"
    out = gaussian_filter1d(a, sig, axis=1, mode=x_mode)
    out = gaussian_filter1d(out, sig, axis=0, mode="nearest")
    return out


def _destreak_columns(
    field: np.ndarray,
    *,
    strength: float = 0.85,
    blur_sigma: float = 10.0,
    periodic_x: bool = True,
) -> np.ndarray:
    """Subtract longitude column-comb (vertical barcode) while keeping 2D structure."""
    a = np.asarray(field, dtype=np.float64)
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return a
    col = np.mean(a, axis=0)
    x_mode = "wrap" if periodic_x else "nearest"
    col_lp = gaussian_filter1d(col, float(max(blur_sigma, 1.0)), mode=x_mode)
    comb = (col - col_lp)[None, :]
    return a - s * comb


def _domain_warp_xy(
    field: np.ndarray,
    rng: np.random.Generator,
    *,
    amp_px: float = 6.0,
    warp_scale: float = 0.045,
    periodic_x: bool = True,
) -> np.ndarray:
    """2D domain warp so ridged filaments curve into webs, not circles."""
    a = np.asarray(field, dtype=np.float64)
    h, w = a.shape
    wx, wy = domain_warp_displacement(
        rng, h, w, warp_scale=warp_scale, periodic_x=periodic_x
    )
    wx = (wx - wx.mean()) / (wx.std() + 1e-8)
    wy = (wy - wy.mean()) / (wy.std() + 1e-8)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    amp = float(amp_px)
    xs = xx + wx * amp
    ys = yy + wy * amp * 0.85
    if periodic_x:
        xs = np.mod(xs, float(w))
    else:
        xs = np.clip(xs, 0.0, float(w - 1))
    ys = np.clip(ys, 0.0, float(h - 1))

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = np.clip(y0 + 1, 0, h - 1)
    if periodic_x:
        x1 = np.mod(x1, w)
        x0 = np.mod(x0, w)
    else:
        x1 = np.clip(x1, 0, w - 1)
        x0 = np.clip(x0, 0, w - 1)
    tx = xs - np.floor(xs)
    ty = ys - np.floor(ys)
    v00 = a[y0, x0]
    v01 = a[y0, x1]
    v10 = a[y1, x0]
    v11 = a[y1, x1]
    return (1.0 - ty) * ((1.0 - tx) * v00 + tx * v01) + ty * ((1.0 - tx) * v10 + tx * v11)


def _domain_warp_x(
    field: np.ndarray,
    rng: np.random.Generator,
    *,
    amp_px: float = 1.6,
    sigma: float = 9.0,
    periodic_x: bool = True,
) -> np.ndarray:
    """Break residual column coherence with a soft horizontal sample warp."""
    a = np.asarray(field, dtype=np.float64)
    h, w = a.shape
    disp = _gauss_float(rng.standard_normal((h, w)), sigma, periodic_x=periodic_x)
    disp = (disp - disp.mean()) / (disp.std() + 1e-8)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    xs = xx + disp * float(amp_px)
    if periodic_x:
        xs = np.mod(xs, float(w))
    else:
        xs = np.clip(xs, 0.0, float(w - 1))
    x0 = np.floor(xs).astype(np.int64)
    x1 = x0 + 1
    if periodic_x:
        x1 = np.mod(x1, w)
        x0 = np.mod(x0, w)
    else:
        x1 = np.clip(x1, 0, w - 1)
    t = xs - np.floor(xs)
    y_idx = yy.astype(np.int64)
    return (1.0 - t) * a[y_idx, x0] + t * a[y_idx, x1]


def _filtered_white_fbm(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    octaves: int = 5,
    gain: float = 0.55,
    base_sigma: float = 18.0,
    periodic_x: bool = True,
) -> np.ndarray:
    """Isotropic multi-scale soft noise from filtered white noise (no value-grid lattice)."""
    h, w = int(height), int(width)
    acc = np.zeros((h, w), dtype=np.float64)
    amp_sum = 0.0
    amp = 1.0
    sigma = float(base_sigma)
    for _ in range(max(1, octaves)):
        n = rng.standard_normal((h, w)).astype(np.float64)
        layer = _gauss_float(n, sigma, periodic_x=periodic_x)
        lo = float(np.percentile(layer, 1.0))
        hi = float(np.percentile(layer, 99.0))
        layer = np.clip((layer - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        layer = _destreak_columns(
            layer, strength=0.55, blur_sigma=max(6.0, sigma * 0.45), periodic_x=periodic_x
        )
        layer = np.clip(layer, 0.0, 1.0)
        acc += amp * layer
        amp_sum += amp
        amp *= gain
        sigma = max(1.15, sigma * 0.50)
    out = acc / max(amp_sum, 1e-9)
    return _destreak_columns(out, strength=0.70, blur_sigma=8.0, periodic_x=periodic_x)


def _ridged_filament_field(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    base_scale: float,
    octaves: int,
    periodic_x: bool,
    contrast: float = 1.0,
) -> np.ndarray:
    """Coarse ridged FBM upsampled + destreaked into filament ridges."""
    h, w = int(height), int(width)
    ch = max(12, h // 6)
    cw = max(16, w // 6)
    ridge = ridged_fbm2d(
        rng, ch, cw, base_scale=base_scale, octaves=octaves, periodic_x=periodic_x
    )
    ridge = _resize_bilinear(ridge, h, w, periodic_x=periodic_x)
    ridge = _destreak_columns(ridge, strength=0.65, blur_sigma=7.0, periodic_x=periodic_x)
    # Sharpen ridge contrast: midtones toward 0, peaks toward 1.
    c = float(np.clip(contrast, 0.6, 2.2))
    ridge = np.clip((ridge - 0.42) / 0.48, 0.0, 1.0) ** (1.0 / c)
    return np.clip(ridge, 0.0, 1.0)


def build_cloud_complex_envelope(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool = True,
    style: str = "balanced",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (structure_weight, emission_rgb, dust_occlusion) in [0,1].

    Soft FBM body modulated by ridged/warped filaments — spider-web nebula gas
    instead of geometric rings and blobs. ``style``: subtle|balanced|dramatic.
    """
    h, w = int(height), int(width)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    style_key = str(style).strip().lower()
    # (structure_gain, freckle_gain, chroma_gain, ridge_contrast)
    style_presets: dict[str, tuple[float, float, float, float]] = {
        "subtle": (0.90, 0.78, 0.82, 0.95),
        "balanced": (1.0, 1.0, 1.0, 1.15),
        "dramatic": (1.14, 1.42, 1.28, 1.55),
    }
    struct_g, freckle_g, chroma_g, ridge_c = style_presets.get(
        style_key, style_presets["balanced"]
    )

    body = _filtered_white_fbm(
        rng,
        h,
        w,
        octaves=5,
        base_sigma=max(14.0, 0.040 * float(w)),
        periodic_x=periodic_x,
    )
    body = np.clip((body - 0.28) / 0.55, 0.0, 1.0) ** 0.88

    # Primary structure: multi-scale ridged filaments with 2D domain warp.
    fil_lo = _ridged_filament_field(
        rng,
        h,
        w,
        base_scale=0.11,
        octaves=5,
        periodic_x=periodic_x,
        contrast=ridge_c,
    )
    fil_hi = _ridged_filament_field(
        rng,
        h,
        w,
        base_scale=0.28,
        octaves=4,
        periodic_x=periodic_x,
        contrast=ridge_c * 1.05,
    )
    filaments = np.clip(0.62 * fil_lo + 0.48 * fil_hi, 0.0, 1.0)
    filaments = _domain_warp_xy(
        filaments,
        rng,
        amp_px=max(5.0, 0.012 * float(w)),
        warp_scale=0.040,
        periodic_x=periodic_x,
    )
    filaments = np.clip(
        _destreak_columns(filaments, strength=0.72, blur_sigma=8.0, periodic_x=periodic_x),
        0.0,
        1.0,
    )
    # Second finer warp pass for web-like branching.
    filaments = _domain_warp_xy(
        filaments,
        rng,
        amp_px=max(2.5, 0.006 * float(w)),
        warp_scale=0.085,
        periodic_x=periodic_x,
    )
    filaments = np.clip(filaments, 0.0, 1.0)

    # Optional low-weight geometric accents (never dominate — avoids oval rings).
    accent_w = 0.0 if style_key == "dramatic" else 0.10
    arcs = np.zeros((h, w), dtype=np.float64)
    cores = np.zeros((h, w), dtype=np.float64)
    if accent_w > 1e-6:
        n_arcs = int(rng.integers(1, 3))
        for _ in range(n_arcs):
            cx = float(rng.uniform(-0.85, 0.85))
            cy = float(rng.uniform(-0.55, 0.55))
            r0 = float(rng.uniform(0.28, 0.72))
            thick = float(rng.uniform(0.08, 0.16))
            dx = np.minimum(np.abs(xx - cx), 2.0 - np.abs(xx - cx))
            dy = yy - cy
            rad = np.sqrt(dx * dx + dy * dy)
            ring = np.exp(-(((rad - r0) / thick) ** 2))
            arcs = np.maximum(arcs, ring * float(rng.uniform(0.45, 0.85)))
        arcs = _gauss_float(arcs, 1.2, periodic_x=periodic_x)
        n_cores = int(rng.integers(1, 3))
        for _ in range(n_cores):
            cx = float(rng.uniform(-0.9, 0.9))
            cy = float(rng.uniform(-0.7, 0.7))
            sig = float(rng.uniform(0.10, 0.20))
            dx = np.minimum(np.abs(xx - cx), 2.0 - np.abs(xx - cx))
            blob = np.exp(-((dx**2 + (yy - cy) ** 2) / (2.0 * sig**2)))
            cores = np.maximum(cores, blob * float(rng.uniform(0.45, 0.80)))
        cores = _gauss_float(cores, 1.0, periodic_x=periodic_x)

    cool_lobes = _filtered_white_fbm(
        rng,
        h,
        w,
        octaves=3,
        base_sigma=max(10.0, 0.028 * float(w)),
        periodic_x=periodic_x,
    )
    cool_lobes = np.clip((cool_lobes - 0.42) / 0.48, 0.0, 1.0) ** 1.15
    cool_lobes = np.clip(cool_lobes * (0.45 + 0.55 * filaments), 0.0, 1.0)

    dust_hf = _filtered_white_fbm(
        rng,
        h,
        w,
        octaves=4,
        base_sigma=max(4.5, 0.012 * float(w)),
        periodic_x=periodic_x,
    )
    freckles = np.clip((dust_hf - 0.48) / 0.42, 0.0, 1.0) ** 1.40
    # Dark lanes along filament voids (spider-web carve).
    lane_carve = np.clip((1.0 - filaments) * freckles, 0.0, 1.0)

    # Body modulated by filaments — primary look is web/ridges, not rings.
    structure = body * (0.35 + 0.65 * filaments)
    structure = np.clip(structure + accent_w * 0.55 * arcs + accent_w * 0.45 * cores, 0.0, 1.0)
    structure = np.clip(structure * struct_g, 0.0, 1.0)
    # Soft full-sky floor so poles keep faint gas.
    structure = np.clip(structure * 0.90 + body * 0.14, 0.0, 1.0)
    freckle_mix = 0.42 * freckle_g
    structure = np.clip(structure * (1.0 - freckle_mix * lane_carve), 0.0, 1.0)
    structure = _gauss_float(structure, 0.70, periodic_x=periodic_x)
    structure = np.clip(
        _destreak_columns(structure, strength=0.80, blur_sigma=9.0, periodic_x=periodic_x),
        0.0,
        1.0,
    )
    structure = np.clip(
        _domain_warp_x(structure, rng, amp_px=1.5, sigma=10.0, periodic_x=periodic_x),
        0.0,
        1.0,
    )
    structure = gaussian_filter1d(
        structure, 0.50, axis=1, mode="wrap" if periodic_x else "nearest"
    )
    structure = np.clip(structure, 0.0, 1.0)

    # Multi-hue emission keyed off filament + body (not ring masks).
    ha = np.clip(0.58 * filaments + 0.28 * body + 0.12 * cores, 0.0, 1.0)
    oiii = np.clip(0.55 * cool_lobes + 0.35 * filaments * (1.0 - 0.35 * freckles), 0.0, 1.0)
    dust_pink = np.clip(0.50 * body + 0.38 * filaments * (1.0 - freckles), 0.0, 1.0)
    emit_r = structure * (0.30 * dust_pink + 0.50 * ha + 0.14 * oiii)
    emit_g = structure * (0.18 * dust_pink + 0.24 * ha + 0.46 * oiii)
    emit_b = structure * (0.12 * dust_pink + 0.12 * ha + 0.58 * oiii + 0.22 * cool_lobes)
    emission = np.stack([emit_r, emit_g, emit_b], axis=2)
    emission = np.clip(
        emission * float(rng.uniform(0.90, 1.10)) * (0.78 * chroma_g),
        0.0,
        1.0,
    )
    for c in range(3):
        emission[:, :, c] = _destreak_columns(
            emission[:, :, c], strength=0.75, blur_sigma=9.0, periodic_x=periodic_x
        )
        emission[:, :, c] = _domain_warp_x(
            emission[:, :, c], rng, amp_px=1.1, sigma=8.0, periodic_x=periodic_x
        )
        emission[:, :, c] = gaussian_filter1d(
            emission[:, :, c], 0.40, axis=1, mode="wrap" if periodic_x else "nearest"
        )
    emission = np.clip(emission, 0.0, 1.0)

    dust_occ = np.clip(lane_carve * freckle_g * (0.26 + 0.38 * structure), 0.0, 0.74)
    dust_occ = _gauss_float(dust_occ, 0.90, periodic_x=periodic_x)
    dust_occ = np.clip(
        _destreak_columns(dust_occ, strength=0.70, blur_sigma=8.0, periodic_x=periodic_x),
        0.0,
        0.74,
    )
    return structure.astype(np.float64), emission.astype(np.float64), dust_occ.astype(np.float64)
