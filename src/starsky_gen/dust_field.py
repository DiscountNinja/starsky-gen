"""Fractal extinction masks: ridged multifractal → erosion → plane-stretched lanes.

Extinction must track morph absorption HF: band_transmission_from_morph_absorption adds hp_m to T
when hp_m>0 (absorption peak). Subtracting hp_m inverted puff peaks into clear windows and
smoothed turbulent clouds in the composite (see generator morphology-primary path).
"""

from __future__ import annotations

import numpy as np

from starsky_gen.procedural_noise import (
    _resize_bilinear,
    fbm2d,
    gaussian_blur_pil,
    ridged_fbm2d,
)


def _blur_y_only_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Elongate filaments along galactic latitude (vertical in equirect)."""
    sig = max(float(sigma), 0.15)
    h = int(field.shape[0])
    rad = max(1, min(int(round(sig * 2.8)), max(1, h // 2 - 1)))
    yy = np.arange(-rad, rad + 1, dtype=np.float64)
    ker = np.exp(-(yy**2) / (2.0 * sig**2))
    ker /= float(np.sum(ker)) + 1e-14
    out = np.empty_like(field, dtype=np.float64)
    for j in range(field.shape[1]):
        out[:, j] = np.convolve(field[:, j], ker, mode="same")
    return out


def _blur_x_only_field(field: np.ndarray, sigma: float, *, periodic_x: bool) -> np.ndarray:
    """Light longitude smoothing; periodic wrap on X."""
    sig = max(float(sigma), 0.12)
    rad = max(1, int(round(sig * 2.5)))
    xx = np.arange(-rad, rad + 1, dtype=np.float64)
    ker = np.exp(-(xx**2) / (2.0 * sig**2))
    ker /= float(np.sum(ker)) + 1e-14
    out = np.empty_like(field, dtype=np.float64)
    h = field.shape[0]
    for i in range(h):
        row = field[i]
        if periodic_x:
            pad = np.concatenate([row[-rad:], row, row[:rad]])
            conv = np.convolve(pad, ker, mode="same")
            out[i] = conv[rad : rad + row.size]
        else:
            out[i] = np.convolve(row, ker, mode="same")
    return out


def build_filament_erosion_map(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool = True,
) -> np.ndarray:
    """Erosion-style absorption identity: long filaments, broken rivers (not soft cloud)."""
    h, w = int(height), int(width)
    ch, cw = max(12, h // 22), max(18, w // 18)
    ridge_lon = ridged_fbm2d(rng, ch, cw, base_scale=0.065, octaves=6, periodic_x=periodic_x)
    ridge_lon = _resize_bilinear(ridge_lon, h, w, periodic_x=periodic_x)
    valley_lon = np.clip(1.0 - ridge_lon, 0.0, 1.0)
    valley_lon = _blur_x_only_field(valley_lon, 2.8, periodic_x=periodic_x)
    valley_lon = _blur_y_only_field(valley_lon, 0.55)
    wide_lon = gaussian_blur_pil(valley_lon, 2.6, periodic_x=periodic_x)
    rivers_lon = np.clip(wide_lon - valley_lon * 0.88, 0.0, 1.0) ** 1.72

    rng2 = np.random.default_rng(int(rng.integers(0, 2**31)))
    ridge_brk = ridged_fbm2d(rng2, ch, cw, base_scale=0.11, octaves=5, periodic_x=periodic_x)
    ridge_brk = _resize_bilinear(ridge_brk, h, w, periodic_x=periodic_x)
    valley_brk = np.clip(1.0 - ridge_brk, 0.0, 1.0)
    valley_brk = _blur_x_only_field(valley_brk, 0.42, periodic_x=periodic_x)
    valley_brk = _blur_y_only_field(valley_brk, 0.85)
    wide_brk = gaussian_blur_pil(valley_brk, 1.4, periodic_x=periodic_x)
    fractures = np.clip(wide_brk - valley_brk * 0.82, 0.0, 1.0) ** 1.62

    coarse = fbm2d(rng, max(6, ch // 2), max(8, cw // 2), base_scale=0.18, octaves=3, periodic_x=periodic_x)
    coarse = _resize_bilinear(coarse, h, w, periodic_x=periodic_x)
    coarse = _blur_y_only_field(coarse, 3.2)
    coarse = np.clip((coarse - 0.48) / 0.38, 0.0, 1.0) ** 2.1

    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    plane = np.exp(-((yy**2) / 0.52))
    out = np.clip(
        np.maximum(rivers_lon * 0.72, fractures * 0.58) + coarse * 0.22,
        0.0,
        1.0,
    )
    out *= plane
    lo = float(np.percentile(out, 4.0))
    hi = float(np.percentile(out, 96.5))
    if hi > lo + 1e-8:
        out = np.clip((out - lo) / (hi - lo), 0.0, 1.0)
    return np.clip(out**2.05, 0.0, 1.0).astype(np.float64)


def compress_absorption_opacity(
    absorption: np.ndarray,
    *,
    gamma: float = 1.42,
    mix: float = 1.0,
) -> np.ndarray:
    """Convex opacity compression: 1-(1-a)^γ deepens lanes, leaves faint haze (γ>1)."""
    g = float(np.clip(gamma, 1.0, 2.8))
    mix_f = float(np.clip(mix, 0.0, 1.0))
    a = np.clip(np.asarray(absorption, dtype=np.float64), 0.0, 1.0)
    if g <= 1.001 or mix_f < 1e-6:
        return a.astype(np.float64)
    compressed = 1.0 - (1.0 - a) ** g
    return np.clip(a * (1.0 - mix_f) + compressed * mix_f, 0.0, 1.0).astype(np.float64)


def transmission_from_absorption_map(
    absorption: np.ndarray,
    *,
    void_floor: float = 0.035,
    sharpness: float = 2.2,
    opacity_gamma: float = 1.42,
) -> np.ndarray:
    """Map erosion absorption [0,1] → transmission with sharp fractured falloff."""
    a = compress_absorption_opacity(absorption, gamma=opacity_gamma)
    vf = float(np.clip(void_floor, 0.01, 0.2))
    sh = float(np.clip(sharpness, 1.2, 5.8))
    return np.clip(vf + (1.0 - vf) * (1.0 - a**sh), vf, 1.0).astype(np.float64)


def reinforce_absorption_edges(
    transmission: np.ndarray,
    absorption: np.ndarray,
    *,
    edge_gain: float = 0.42,
    periodic_x: bool = True,
) -> np.ndarray:
    """Sharpen lane rims: fractured absorption, not smoky gradients."""
    t = np.clip(np.asarray(transmission, dtype=np.float64), 0.0, 1.0)
    a = np.clip(np.asarray(absorption, dtype=np.float64), 0.0, 1.0)
    if periodic_x:
        gx = 0.5 * (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1))
    else:
        gx = np.gradient(a, axis=1)
    gy = np.gradient(a, axis=0)
    grad = np.sqrt(gx**2 + gy**2 + 1e-12)
    g93 = float(np.percentile(grad, 93.0))
    edge = np.clip(grad / (g93 + 1e-6), 0.0, 1.0) ** 1.38
    carve = edge * float(np.clip(edge_gain, 0.0, 1.2)) * (a**1.12)
    return np.clip(t * (1.0 - carve), 0.01, 1.0).astype(np.float64)


def build_fractal_extinction_field(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool = True,
    plane_stretch_x: float = 2.35,
    erosion_power: float = 1.72,
    ridge_weight: float = 0.68,
) -> np.ndarray:
    """Absorption map [0,1]: filament erosion primary, legacy multifractal fill."""
    filament = build_filament_erosion_map(rng, height, width, periodic_x=periodic_x)
    ch = max(8, height // 48)
    cw = max(12, width // 64)
    ridge = ridged_fbm2d(rng, ch, cw, base_scale=0.09, octaves=4, periodic_x=periodic_x)
    ridge = _resize_bilinear(ridge, height, width, periodic_x=periodic_x)
    mf = np.clip((1.0 - np.abs(ridge * 2.0 - 1.0)) ** float(erosion_power), 0.0, 1.0)
    sx = max(float(plane_stretch_x), 1.0)
    fill = gaussian_blur_pil(mf, 0.55 / sx, periodic_x=periodic_x)
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
    plane = np.exp(-((yy**2) / 0.45))
    fill = np.clip(fill * plane * 0.35, 0.0, 1.0)
    out = np.clip(np.maximum(filament, fill * float(ridge_weight)), 0.0, 1.0)
    return out.astype(np.float64)


def build_band_disruption_field(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool = True,
    n_voids: int | None = None,
) -> np.ndarray:
    """Asymmetric voids, vertical gouges, and band breaks (1 = deep lane / near disappearance)."""
    h, w = int(height), int(width)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    band = np.exp(-((yy**2) / 0.62))

    ch, cw = max(6, h // 32), max(10, w // 24)
    gouge = ridged_fbm2d(rng, ch, cw, base_scale=0.08, octaves=5, periodic_x=periodic_x)
    gouge = _resize_bilinear(gouge, h, w, periodic_x=periodic_x)
    gouge = gaussian_blur_pil(gouge, 0.55, periodic_x=periodic_x)
    gouge = np.clip((gouge - 0.42) / 0.48, 0.0, 1.0) ** 1.65

    vert_cut = fbm2d(rng, ch, cw, base_scale=0.16, octaves=4, periodic_x=periodic_x)
    vert_cut = _resize_bilinear(vert_cut, h, w, periodic_x=periodic_x)
    vert_cut = gaussian_blur_pil(vert_cut, 1.35, periodic_x=periodic_x)
    vert_cut = np.clip((vert_cut - 0.38) / 0.52, 0.0, 1.0) ** 1.45
    vert_cut *= band

    voids = np.zeros((h, w), dtype=np.float64)
    nv = int(n_voids if n_voids is not None else rng.integers(5, 11))
    for i in range(nv):
        sub = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + (i + 1) * 7919)
        shard = ridged_fbm2d(
            sub,
            max(4, ch // 2),
            max(6, cw // 2),
            base_scale=0.11 + 0.03 * i,
            octaves=4,
            periodic_x=periodic_x,
        )
        shard = _resize_bilinear(shard, h, w, periodic_x=periodic_x)
        shard = _blur_y_only_field(shard, float(sub.uniform(2.0, 4.0)))
        shard = _blur_x_only_field(shard, float(sub.uniform(0.22, 0.52)), periodic_x=periodic_x)
        thr = float(sub.uniform(0.58, 0.76))
        carve = np.clip((shard - thr) / max(1.0 - thr, 1e-8), 0.0, 1.0) ** 1.42
        voids = np.maximum(voids, carve * float(sub.uniform(0.58, 0.94)))
    voids *= band

    wobble = fbm2d(rng, max(4, h // 48), max(6, w // 40), base_scale=0.2, octaves=3, periodic_x=periodic_x)
    wobble = _resize_bilinear(wobble, h, w, periodic_x=periodic_x)
    band_break = np.exp(-(((yy + (wobble - 0.5) * 0.52) ** 2) / (0.48 + 0.28 * wobble)))
    band_hole = np.clip(band * (1.0 - band_break * 0.88), 0.0, 1.0)
    band_hole = np.clip((band_hole - 0.55) / 0.38, 0.0, 1.0) ** 1.25

    out = np.clip(
        np.maximum(gouge * 0.62, np.maximum(voids, vert_cut * 0.58)) + band_hole * 0.42,
        0.0,
        1.0,
    )
    lo = float(np.percentile(out, 3.0))
    hi = float(np.percentile(out, 97.5))
    if hi > lo + 1e-8:
        out = np.clip((out - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float64)


def carve_extinction_discontinuities(
    transmission: np.ndarray,
    disruption: np.ndarray,
    *,
    strength: float = 1.0,
    void_floor: float = 0.035,
) -> np.ndarray:
    """Drive transmission toward void_floor in disrupted regions (ugly dark cuts, not smooth strip)."""
    t = np.clip(np.asarray(transmission, dtype=np.float64), 0.0, 1.0)
    d = np.clip(np.asarray(disruption, dtype=np.float64), 0.0, 1.0)
    s = float(np.clip(strength, 0.0, 2.0))
    vf = float(np.clip(void_floor, 0.01, 0.2))
    if s < 1e-6:
        return t
    w = np.clip(d * s, 0.0, 1.0) ** 1.72
    carved = t * (1.0 - w * 0.97)
    target = vf + (1.0 - vf) * (1.0 - w**1.08)
    out = np.minimum(carved, target)
    return np.clip(out, vf, 1.0).astype(np.float64)


def emission_clearance_from_extinction(
    ext_paint: np.ndarray,
    *,
    floor: float = 0.10,
    span: float = 0.52,
    power: float = 1.35,
) -> np.ndarray:
    """Visibility for emission/haze: ~0 in lanes, ~1 in clear sightlines."""
    e = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
    clear = np.clip((e - floor) / max(span, 1e-6), 0.0, 1.0)
    return np.clip(clear**float(power), 0.0, 1.0).astype(np.float64)


def attenuate_rgb_column_comb(
    rgb: np.ndarray,
    band_weight: np.ndarray | None,
    *,
    strength: float = 0.58,
    periodic_x: bool = True,
) -> np.ndarray:
    """Remove vertical longitude combs from diffuse RGB (luma-preserving)."""
    from starsky_gen.color_science import rec709_luma, remap_luma_preserving_chroma

    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    lu_fix = attenuate_column_comb(lu, band_weight, strength=strength, periodic_x=periodic_x)
    return np.maximum(0.0, remap_luma_preserving_chroma(lin, lu_fix)).astype(np.float64)


def attenuate_column_comb(
    field: np.ndarray,
    band_weight: np.ndarray | None,
    *,
    strength: float = 0.48,
    periodic_x: bool = True,
) -> np.ndarray:
    """Remove longitude-column comb from coarse-grid upscales (vertical pillars in equirect)."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return np.clip(field, 0.0, 1.0).astype(np.float64)
    a = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
    h, w = a.shape
    if band_weight is None:
        gate = np.ones((h, w), dtype=bool)
    else:
        gate = _band_host_gate(band_weight, (h, w)) > 0.20
        if not bool(np.any(gate)):
            return a
    col = np.mean(a, axis=0, keepdims=True)
    col_2d = np.broadcast_to(col, (h, w)).copy()
    col_blur = _blur_x_only_field(col_2d, 14.0, periodic_x=periodic_x)
    comb = (col_2d - col_blur) * s
    out = a.copy()
    out[gate] = np.clip(a[gate] - comb[gate], 0.0, 1.0)
    return out.astype(np.float64)


def deepen_center_band_transmission(
    transmission: np.ndarray,
    band_weight: np.ndarray | None,
    *,
    void_floor: float,
    strength: float = 0.72,
    periodic_x: bool = True,
) -> np.ndarray:
    """Extra opacity in the galactic core so dust swallows the bright band center."""
    s = float(np.clip(strength, 0.0, 1.2))
    if s < 1e-6:
        return np.clip(transmission, void_floor, 1.0).astype(np.float64)
    vf = float(np.clip(void_floor, 0.004, 0.12))
    t = np.clip(np.asarray(transmission, dtype=np.float64), vf, 1.0)
    h, w = t.shape
    gate = _band_host_gate(band_weight, (h, w))
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, np.newaxis]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[np.newaxis, :]
    core = np.exp(-(xx**2) / 0.14) * np.exp(-((yy * 0.94) ** 2) / 0.10)
    core = np.clip(core * gate, 0.0, 1.0) ** 0.82
    carve = core * s * np.clip((1.0 - t) ** 0.55, 0.08, 1.0)
    out = np.clip(t * (1.0 - carve * 0.48), vf, 1.0)
    # Do not subtract transmission HF — that turns vertical plumes into soft curtains.
    return out.astype(np.float64)


def band_transmission_from_morph_absorption(
    filament_trans: np.ndarray,
    morph_absorption: np.ndarray,
    band_weight: np.ndarray | None,
    *,
    void_floor: float,
    clear_max: float = 0.18,
    periodic_x: bool = True,
    filament_detail: float = 0.22,
    puff_punch_mask: np.ndarray | None = None,
    coupling_weight: np.ndarray | None = None,
    opacity_gamma: float = 1.42,
) -> np.ndarray:
    """Morph absorption drives T globally — soft coupling, not a hard in-band mask."""
    vf = float(np.clip(void_floor, 0.004, 0.12))
    t_clear = float(np.clip(clear_max, vf + 0.04, 0.26))
    fil = np.clip(np.asarray(filament_trans, dtype=np.float64), vf, 1.0)
    morph = np.clip(np.asarray(morph_absorption, dtype=np.float64), 0.0, 1.0)
    h, w = fil.shape
    w_field = _extinction_coupling_gate(
        band_weight, (h, w), coupling_weight, periodic_x=periodic_x
    )
    if puff_punch_mask is not None:
        pm = np.clip(np.asarray(puff_punch_mask, dtype=np.float64), 0.0, 1.0)
        if pm.shape != (h, w):
            pm = _resize_bilinear(pm, h, w, periodic_x=periodic_x)
        w_field = np.clip(np.maximum(w_field, pm * 0.68), 0.0, 1.2)
    else:
        pm = None
    lo_a = float(np.percentile(morph, 4.0))
    hi_a = float(np.percentile(morph, 93.0))
    if hi_a <= lo_a + 1e-8:
        hi_a = lo_a + 0.15
    morph_o = compress_absorption_opacity(morph, gamma=opacity_gamma)
    norm = np.clip((morph_o - lo_a) / (hi_a - lo_a), 0.0, 1.0)
    target = vf + (1.0 - norm**2.55) * (t_clear - vf)
    sig = float(np.clip(max(h, w) * 0.014, 1.8, 16.0))
    med_m = gaussian_blur_pil(morph, sig, periodic_x=periodic_x)
    hp_m = np.clip(morph - med_m * 0.74, -0.45, 0.45)
    med_f = gaussian_blur_pil(fil, sig, periodic_x=periodic_x)
    hp_f = np.clip(fil - med_f * 0.76, -0.35, 0.35)
    fd = float(np.clip(filament_detail, 0.0, 0.45))
    punch = np.clip(pm, 0.0, 1.0) if pm is not None else np.zeros((h, w), dtype=np.float64)
    morph_dark = target - hp_m * (0.86 + 0.48 * punch)
    # Morphology coupling only in the disk — off-band keeps filament transmission (clear sky).
    if band_weight is not None:
        bw = np.clip(np.asarray(band_weight, dtype=np.float64), 0.0, 1.0)
        if bw.ndim == 1:
            bw = bw[:, None]
        if bw.shape != (h, w):
            bw = np.broadcast_to(bw, (h, w))
        plane = np.clip(bw**1.35, 0.0, 1.0)
    else:
        plane = np.clip(w_field, 0.0, 1.0)
    w_clip = np.clip(w_field * plane, 0.0, 1.0)
    out = np.clip(
        fil * (1.0 - w_clip) + morph_dark * w_clip + hp_f * fd * w_clip * 0.32,
        vf,
        t_clear,
    )
    if band_weight is not None:
        sky = np.clip(1.0 - plane, 0.0, 1.0)
        out = np.clip(out * (1.0 - sky) + sky * 0.995, vf, 1.0)
    return out.astype(np.float64)


def gas_clearance_from_extinction(
    ext_paint: np.ndarray,
    *,
    floor: float = 0.05,
    power: float = 1.02,
    min_clear: float = 0.26,
    band_weight: np.ndarray | None = None,
) -> np.ndarray:
    """ISM gas/haze visibility when morphology extinction pins T near void_floor in the band."""
    if band_weight is not None:
        return band_relative_clearance(
            ext_paint,
            band_weight,
            min_clear=float(min_clear),
            power=float(power),
            p_lo=8.0,
            p_hi=92.0,
        )
    e = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
    lo = float(np.percentile(e, 10.0))
    hi = float(np.percentile(e, 90.0))
    adaptive_floor = min(float(floor), lo)
    span = max(hi - adaptive_floor, 0.06)
    clear = np.clip((e - adaptive_floor) / span, 0.0, 1.0) ** float(power)
    if min_clear > 1e-6:
        clear = np.clip(np.maximum(clear, min_clear), 0.0, 1.0)
    return clear.astype(np.float64)


def band_relative_clearance(
    ext_paint: np.ndarray,
    band_weight: np.ndarray | None = None,
    *,
    p_lo: float = 12.0,
    p_hi: float = 88.0,
    min_clear: float = 0.10,
    power: float = 1.05,
) -> np.ndarray:
    """Sightline clearance relative to in-band transmission span (T is capped low in the plane)."""
    e = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
    h, w = e.shape
    if band_weight is not None:
        bw = np.clip(np.asarray(band_weight, dtype=np.float64), 0.0, 1.0)
        if bw.ndim == 1:
            bw = bw[:, None]
        if bw.shape != (h, w):
            bw = np.broadcast_to(bw, (h, w))
        sample = e[bw > 0.22]
    else:
        sample = e.ravel()
    if sample.size < 32:
        sample = e.ravel()
    lo = float(np.percentile(sample, p_lo))
    hi = float(np.percentile(sample, p_hi))
    span = max(hi - lo, 0.05)
    clear = np.clip((e - lo) / span, 0.0, 1.0) ** float(power)
    if min_clear > 1e-6 and band_weight is not None:
        bw = np.clip(np.asarray(band_weight, dtype=np.float64), 0.0, 1.0)
        if bw.ndim == 1:
            bw = bw[:, None]
        if bw.shape != (h, w):
            bw = np.broadcast_to(bw, (h, w))
        disk = np.clip((bw - 0.12) / 0.88, 0.0, 1.0)
        clear = np.clip(clear * (1.0 - disk) + np.maximum(clear, min_clear) * disk, 0.0, 1.0)
    return clear.astype(np.float64)


def stretch_absorption_contrast(
    absorption: np.ndarray,
    *,
    contrast: float = 1.2,
    p_lo: float = 4.0,
    p_hi: float = 91.0,
) -> np.ndarray:
    """Percentile stretch so dust lanes stay asymmetric (not flattened by max-normalize)."""
    a = np.clip(np.asarray(absorption, dtype=np.float64), 0.0, 1.0)
    lo = float(np.percentile(a, p_lo))
    hi = float(np.percentile(a, p_hi))
    if hi <= lo + 1e-8:
        return a
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    gamma = 1.0 / max(float(contrast), 0.55)
    return np.clip(out**gamma, 0.0, 1.0).astype(np.float64)


def fragment_lane_absorption(
    absorption: np.ndarray,
    disruption: np.ndarray,
    latent_turb: np.ndarray,
    *,
    periodic_x: bool,
    strength: float = 0.72,
) -> np.ndarray:
    """Roughen lane edges and add filament texture — no multiplicative hole punching."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return np.clip(absorption, 0.0, 1.0)
    a = np.clip(np.asarray(absorption, dtype=np.float64), 0.0, 1.0)
    h, w = a.shape
    scale = float(max(h, w))
    med_sig = float(np.clip(scale * 0.024, 4.0, 38.0))
    soft = gaussian_blur_pil(a, med_sig, periodic_x=periodic_x)
    lane_hi = np.clip(a - soft * 0.72, 0.0, 1.0) ** 1.08
    d = np.clip(np.asarray(disruption, dtype=np.float64), 0.0, 1.0)
    t = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
    filament = np.maximum(d * (0.48 + 0.52 * t), lane_hi * 0.42)
    filament = _blur_y_only_field(filament, 1.05)
    filament = _blur_x_only_field(filament, 0.72, periodic_x=periodic_x)
    ch, cw = max(6, h // 32), max(8, w // 20)
    edge_n = fbm2d(
        np.random.default_rng(int(h * 31 + w * 17) & 0xFFFFFFFF),
        ch,
        cw,
        base_scale=0.20,
        octaves=4,
        periodic_x=periodic_x,
    )
    edge_n = _resize_bilinear(edge_n, h, w, periodic_x=periodic_x)
    edge_n = _blur_x_only_field(edge_n, 0.45, periodic_x=periodic_x)
    roughen = np.clip(1.0 + (edge_n - 0.5) * s * 0.38, 0.72, 1.24)
    out = np.clip(
        np.maximum(a * roughen, filament * s * 0.58) * (0.62 + 0.38 * soft),
        0.0,
        1.0,
    )
    return out.astype(np.float64)


def apply_turbulent_cloud_breakup(
    transmission: np.ndarray,
    dust_absorption: np.ndarray,
    latent_turb: np.ndarray,
    latent_ridge: np.ndarray,
    *,
    void_floor: float,
    periodic_x: bool = True,
    strength: float = 0.62,
    band_weight: np.ndarray | None = None,
    coupling_weight: np.ndarray | None = None,
) -> np.ndarray:
    """2-D turbulent extinction breakup (ridged + turb), not vertical combing or round blobs."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return np.clip(transmission, void_floor, 1.0).astype(np.float64)
    vf = float(np.clip(void_floor, 0.004, 0.12))
    t = np.clip(np.asarray(transmission, dtype=np.float64), vf, 1.0)
    dust = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    turb = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
    ridge = np.clip(np.asarray(latent_ridge, dtype=np.float64), 0.0, 1.0)
    h, w = t.shape
    scale = float(max(h, w))
    ch, cw = max(8, h // 26), max(12, w // 16)
    rng = np.random.default_rng(int(h * 19 + w * 23) & 0xFFFFFFFF)
    ridged = ridged_fbm2d(rng, ch, cw, base_scale=0.16, octaves=5, periodic_x=periodic_x)
    ridged = _resize_bilinear(ridged, h, w, periodic_x=periodic_x)
    ridge_fil = np.clip(1.0 - np.abs(ridge * 2.0 - 1.0), 0.0, 1.0) ** 1.12
    raw = np.clip(ridged * 0.52 + turb * 0.38 + ridge_fil * 0.28 + dust * 0.22, 0.0, 1.0)
    med_sig = float(np.clip(scale * 0.012, 1.0, 12.0))
    raw_hp = np.clip(raw - gaussian_blur_pil(raw, med_sig, periodic_x=periodic_x) * 0.78, 0.0, 1.0)
    raw_hp = _blur_x_only_field(raw_hp, 0.55, periodic_x=periodic_x)
    raw_hp = _blur_y_only_field(raw_hp, 0.85)
    lo = float(np.percentile(raw_hp, 62.0))
    hi = float(np.percentile(raw_hp, 96.5))
    if hi > lo + 1e-8:
        raw_hp = np.clip((raw_hp - lo) / (hi - lo), 0.0, 1.0) ** 1.28
    band_gate = _extinction_coupling_gate(
        band_weight, (h, w), coupling_weight, periodic_x=periodic_x
    )
    carve = raw_hp * s * 0.34 * np.clip((1.0 - t) ** 0.48, 0.12, 1.0) * band_gate
    out = np.clip(t * (1.0 - carve), vf, 1.0).astype(np.float64)
    return out.astype(np.float64)


def _band_host_gate(band_weight: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Concentrate dust microstructure in the galactic disk (suppress off-band turbulence)."""
    if band_weight is None:
        return np.ones(shape, dtype=np.float64)
    bw = np.clip(np.asarray(band_weight, dtype=np.float64), 0.0, 1.0)
    if bw.shape != shape:
        from starsky_gen.procedural_noise import _resize_bilinear

        bw = _resize_bilinear(bw, shape[0], shape[1], periodic_x=True)
    return np.clip(0.015 + 0.985 * bw**0.88, 0.0, 1.0).astype(np.float64)


def _feathered_band_host_gate(
    band_weight: np.ndarray | None,
    shape: tuple[int, int],
    *,
    periodic_x: bool = True,
) -> np.ndarray:
    """Soft latitudinal disk gate — avoids railroad edges from sharp disk_weight × extinction."""
    if band_weight is None:
        return np.ones(shape, dtype=np.float64)
    from starsky_gen.structure_envelope import soften_band_envelope

    soft = soften_band_envelope(
        band_weight, shape, periodic_x=periodic_x, lat_blur_sigma=18.0, power=0.54
    )
    return np.clip(0.08 + 0.92 * soft**0.78, 0.0, 1.0).astype(np.float64)


def _extinction_coupling_gate(
    band_weight: np.ndarray | None,
    shape: tuple[int, int],
    coupling_weight: np.ndarray | None = None,
    *,
    floor: float = 0.30,
    periodic_x: bool = True,
) -> np.ndarray:
    """Blend disk host with morphology coupling so extinction is not rail-locked."""
    if band_weight is None and coupling_weight is None:
        return np.ones(shape, dtype=np.float64)
    h, w = shape
    f = float(np.clip(floor, 0.22, 0.42))
    base = np.full((h, w), f, dtype=np.float64)
    if band_weight is not None:
        bg = _feathered_band_host_gate(band_weight, shape, periodic_x=periodic_x)
        base = np.clip(f + (1.0 - f) * bg, f, 1.0)
    if coupling_weight is not None:
        cw = np.clip(np.asarray(coupling_weight, dtype=np.float64), 0.0, 1.35)
        if cw.shape != shape:
            cw = _resize_bilinear(cw, h, w, periodic_x=periodic_x)
        base = np.clip(np.maximum(base, f + (1.0 - f) * cw * 0.84), f, 1.15)
    return base.astype(np.float64)


def flatten_offband_dust_absorption(
    dust_absorption: np.ndarray,
    band_weight: np.ndarray | None,
    *,
    periodic_x: bool = True,
    ceiling: float = 0.42,
    soften: float = 0.74,
) -> np.ndarray:
    """Compress halo dust without blurring the bright band into the poles."""
    del periodic_x  # kept for API stability
    a = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    bg = _band_host_gate(band_weight, a.shape)
    halo = bg < 0.26
    if not bool(np.any(halo)):
        return a.astype(np.float64)
    out = a.copy()
    s = float(np.clip(soften, 0.45, 0.95))
    cap = float(np.clip(ceiling, 0.28, 0.62))
    out[halo] = np.clip(a[halo] * s + 0.08 * (1.0 - bg[halo]), 0.06, cap)
    return out.astype(np.float64)


def remap_band_absorption_contrast(
    dust_absorption: np.ndarray,
    band_weight: np.ndarray | None,
    *,
    floor: float = 0.10,
    span: float = 0.68,
    p_lo: float = 8.0,
    p_hi: float = 86.0,
) -> np.ndarray:
    """Band-local histogram remap — keeps lane contrast without a white clipped core."""
    a = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    bg = _band_host_gate(band_weight, a.shape)
    on = bg > 0.18
    if not bool(np.any(on)):
        return a.astype(np.float64)
    out = a.copy()
    sub = a[on]
    lo = float(np.percentile(sub, p_lo))
    hi = float(np.percentile(sub, p_hi))
    if hi > lo + 1e-8:
        sub = np.clip(floor + (sub - lo) / (hi - lo) * span, 0.06, floor + span)
    else:
        sub = np.clip(sub, 0.06, floor + span)
    out[on] = sub
    return out.astype(np.float64)


def enrich_dust_absorption_microstructure(
    dust_absorption: np.ndarray,
    rng: np.random.Generator,
    latent_ridge: np.ndarray,
    latent_turb: np.ndarray,
    *,
    periodic_x: bool = True,
    strength: float = 0.92,
    band_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Independent small/fine dust scales — constructive compression, not nested mega blur."""
    s = float(np.clip(strength, 0.0, 1.5))
    if s < 1e-6:
        return np.clip(dust_absorption, 0.0, 1.0).astype(np.float64)
    s_eff = float(np.clip(0.85 + 0.15 * s, 0.85, 1.5))
    a = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    ridge = np.clip(np.asarray(latent_ridge, dtype=np.float64), 0.0, 1.0)
    turb = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
    h, w = a.shape
    scale = float(max(h, w))
    mega_sig = float(np.clip(scale * 0.062, 8.0, 68.0))
    med_sig = float(np.clip(scale * 0.022, 3.0, 26.0))
    fine_sig = float(np.clip(scale * 0.0050, 0.72, 7.2))
    mega = gaussian_blur_pil(a, mega_sig, periodic_x=periodic_x)
    med_blur = gaussian_blur_pil(a, med_sig, periodic_x=periodic_x)
    ch_m, cw_m = max(8, h // 26), max(12, w // 18)
    ch_s, cw_s = max(10, h // 16), max(14, w // 11)
    ch_f, cw_f = max(12, h // 12), max(16, w // 9)
    seed = (
        int(abs(hash((h, w, float(np.mean(ridge)), float(np.mean(turb))))) % (2**31 - 1))
        ^ 0xD057
    )
    sub = np.random.default_rng(seed)
    med_n = ridged_fbm2d(sub, ch_m, cw_m, base_scale=0.17, octaves=5, periodic_x=periodic_x)
    med_n = _resize_bilinear(med_n, h, w, periodic_x=periodic_x)
    small_n = fbm2d(sub, ch_s, cw_s, base_scale=0.27, octaves=5, periodic_x=periodic_x)
    small_n = _resize_bilinear(small_n, h, w, periodic_x=periodic_x)
    fine_n = ridged_fbm2d(sub, ch_f, cw_f, base_scale=0.36, octaves=5, periodic_x=periodic_x)
    fine_n = _resize_bilinear(fine_n, h, w, periodic_x=periodic_x)
    medium = np.clip(med_n * (0.56 + 0.44 * a) + med_blur * 0.10, 0.0, 1.0)
    small_hp = np.clip(a - med_blur * 0.78, 0.0, 1.0)
    small = np.clip(small_n * (0.54 + 0.46 * small_hp), 0.0, 1.0)
    ridge_fil = np.clip(1.0 - np.abs(ridge * 2.0 - 1.0), 0.0, 1.0) ** 1.08
    fine_hp = np.clip(a - gaussian_blur_pil(a, fine_sig, periodic_x=periodic_x) * 0.76, 0.0, 1.0)
    fine = np.clip(fine_n * 0.55 + fine_hp * 0.30 + ridge_fil * 0.15, 0.0, 1.0)
    from starsky_gen.structure_envelope import blend_competitive_scale_hierarchy

    micro = blend_competitive_scale_hierarchy(
        mega,
        medium,
        small,
        fine,
        periodic_x=periodic_x,
        w_mega=0.14,
        w_medium=0.26,
        w_small=0.34,
        w_fine=0.32,
        competition_strength=0.92 * s_eff,
        turbulence_weight=1.0,
    )
    r_hp = np.clip(
        ridge - gaussian_blur_pil(ridge, fine_sig * 1.5, periodic_x=periodic_x) * 0.80,
        0.0,
        1.0,
    )
    t_hp = np.clip(
        turb - gaussian_blur_pil(turb, fine_sig * 1.3, periodic_x=periodic_x) * 0.78,
        0.0,
        1.0,
    )
    micro = np.clip(np.maximum(micro, r_hp * 0.44 + t_hp * 0.36), 0.0, 1.0)
    band_gate = _band_host_gate(band_weight, (h, w))
    band_mask = band_gate > 0.14
    if bool(np.any(band_mask)):
        sub_m = micro[band_mask]
        lo_m = float(np.percentile(sub_m, 48.0))
        hi_m = float(np.percentile(sub_m, 96.0))
    else:
        lo_m = float(np.percentile(micro, 50.0))
        hi_m = float(np.percentile(micro, 96.0))
    if bool(np.any(band_mask)) and hi_m > lo_m + 1e-8:
        sub = np.clip((micro[band_mask] - lo_m) / (hi_m - lo_m), 0.0, 1.0) ** 1.08
        micro = micro.copy()
        micro[band_mask] = sub
    from starsky_gen.structure_envelope import soften_band_envelope

    soft_bw = soften_band_envelope(band_weight, (h, w), periodic_x=periodic_x, lat_blur_sigma=14.0, power=0.54)
    band_gate = np.clip(np.maximum(band_gate, soft_bw * 0.68), 0.0, 1.0)
    micro = np.clip(micro * band_gate, 0.0, 1.0)
    core = np.clip(band_gate**0.92, 0.0, 1.0)
    # Multiplicative roughening works when the band is already bright (additive max() saturates).
    med_ref = float(np.median(micro[band_mask])) if bool(np.any(band_mask)) else 0.5
    mod = np.clip((micro - med_ref) / (float(np.percentile(np.abs(micro - med_ref), 94)) + 1e-6), -1.0, 1.0)
    roughen = s_eff * 0.68 * core
    out = np.clip(a * (1.0 + mod * roughen), 0.04, 0.90)
    tear = np.clip(fine * 0.62 + np.clip(small - medium, -0.35, 0.35) * 0.48, 0.0, 1.0) ** 1.10
    if bool(np.any(band_mask)):
        lo_t = float(np.percentile(tear[band_mask], 58.0))
        hi_t = float(np.percentile(tear[band_mask], 97.0))
    else:
        lo_t = float(np.percentile(tear, 60.0))
        hi_t = float(np.percentile(tear, 97.0))
    if hi_t > lo_t + 1e-8:
        tear = np.clip((tear - lo_t) / (hi_t - lo_t), 0.0, 1.0)
    carve_m = tear * s_eff * 0.48 * core
    out = np.clip(out * (1.0 - carve_m * np.clip(0.35 + 0.65 * a, 0.35, 1.0)), 0.04, 0.90)
    a_med = gaussian_blur_pil(out, med_sig, periodic_x=periodic_x)
    a_hp = np.clip(out - a_med * 0.76, -0.55, 0.55)
    out = np.clip(out + a_hp * s_eff * 0.64 * core, 0.04, 0.90)
    base = np.clip(a, 0.0, 1.0)
    return np.where(band_gate > 0.30, np.clip(out, 0.06, 0.88), base).astype(np.float64)


def carve_dust_transmission_microstructure(
    transmission: np.ndarray,
    dust_absorption: np.ndarray,
    *,
    periodic_x: bool = True,
    strength: float = 0.92,
    void_floor: float = 0.06,
    band_weight: np.ndarray | None = None,
    coupling_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Preserve fine dust filaments in transmission (absorption→trans power law softens edges)."""
    s = float(np.clip(strength, 0.0, 1.5))
    if s < 1e-6:
        return np.clip(transmission, void_floor, 1.0).astype(np.float64)
    s_eff = float(np.clip(0.85 + 0.15 * s, 0.85, 1.5))
    vf = float(np.clip(void_floor, 0.04, 0.14))
    t = np.clip(np.asarray(transmission, dtype=np.float64), vf, 1.0)
    dust = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    scale = float(max(dust.shape))
    med_sig = float(np.clip(scale * 0.018, 2.5, 22.0))
    fine_sig = float(np.clip(scale * 0.0055, 0.75, 7.5))
    med = gaussian_blur_pil(dust, med_sig, periodic_x=periodic_x)
    fine = np.clip(dust - med * 0.76, 0.0, 1.0)
    micro = np.clip(dust - gaussian_blur_pil(dust, fine_sig, periodic_x=periodic_x) * 0.74, 0.0, 1.0)
    detail = np.clip(fine * 0.62 + micro * 0.38, 0.0, 1.0)
    band_gate = _extinction_coupling_gate(
        band_weight, dust.shape, coupling_weight, periodic_x=periodic_x
    )
    band_mask = band_gate > 0.14
    lo = float(np.percentile(detail, 56.0))
    hi = float(np.percentile(detail, 97.0))
    if hi > lo + 1e-8:
        detail = np.clip((detail - lo) / (hi - lo), 0.0, 1.0) ** 1.18
    detail = np.clip(detail * (0.12 + 0.88 * band_gate), 0.0, 1.0)
    t_soft = gaussian_blur_pil(t, fine_sig * 1.4, periodic_x=periodic_x)
    t_hp = np.clip(t_soft - t, 0.0, 1.0)
    lane_w = np.clip((1.0 - t) ** 0.45, 0.06, 1.0) * band_gate
    carve = detail * s_eff * 0.72 * lane_w
    out = np.clip(t * (1.0 - carve) + t_hp * s_eff * 0.14 * lane_w, vf, 1.0)
    t_fine = transmission_from_absorption_map(
        np.clip(dust**1.04, 0.0, 1.0), void_floor=vf, sharpness=4.35
    )
    thr = float(np.percentile(detail[band_mask], 68.0)) if bool(np.any(band_mask)) else float(
        np.percentile(detail, 72.0)
    )
    mask = (detail > thr) & (band_gate > 0.22)
    out = np.minimum(out, np.where(mask, t_fine, out))
    return np.clip(out, vf, 1.0).astype(np.float64)


def compose_constructive_extinction_transmission(
    filament_trans: np.ndarray,
    dust_transmission: np.ndarray,
    *,
    void_floor: float,
    detail_strength: float = 0.72,
    periodic_x: bool = True,
    disk_weight: np.ndarray | None = None,
    coupling_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Carve filament lanes with independent dust small/fine scales (not a re-blurred clone).

    Uses d_mega/d_med blurs for lane context; d_fine and hp carve preserve morph puff HF on T.
    """
    vf = float(np.clip(void_floor, 0.004, 0.12))
    t = np.clip(np.asarray(filament_trans, dtype=np.float64), vf, 1.0)
    d = np.clip(np.asarray(dust_transmission, dtype=np.float64), vf, 1.0)
    s = float(np.clip(detail_strength, 0.0, 1.0))
    if s < 1e-6:
        return t.astype(np.float64)
    scale = float(max(d.shape))
    mega_sig = float(np.clip(scale * 0.042, 6.0, 52.0))
    med_sig = float(np.clip(scale * 0.014, 2.0, 18.0))
    fine_sig = float(np.clip(scale * 0.0045, 0.55, 5.5))
    d_mega = gaussian_blur_pil(d, mega_sig, periodic_x=periodic_x)
    d_med = gaussian_blur_pil(d, med_sig, periodic_x=periodic_x)
    band_gate = _extinction_coupling_gate(
        disk_weight, d.shape, coupling_weight, periodic_x=periodic_x
    )
    d_small = np.clip(d - d_med * 0.80, 0.0, 1.0)
    d_fine = np.clip(d - gaussian_blur_pil(d, fine_sig, periodic_x=periodic_x) * 0.78, 0.0, 1.0)
    band_mask = band_gate > 0.14
    if bool(np.any(band_mask)):
        sub_s = d_small[band_mask]
        lo = float(np.percentile(sub_s, 54.0))
        hi = float(np.percentile(sub_s, 97.5))
    else:
        lo = float(np.percentile(d_small, 55.0))
        hi = float(np.percentile(d_small, 97.5))
    if hi > lo + 1e-8:
        d_small = np.clip((d_small - lo) / (hi - lo), 0.0, 1.0) ** 1.12
    if bool(np.any(band_mask)):
        sub_f = d_fine[band_mask]
        lo_f = float(np.percentile(sub_f, 60.0))
        hi_f = float(np.percentile(sub_f, 98.0))
    else:
        lo_f = float(np.percentile(d_fine, 62.0))
        hi_f = float(np.percentile(d_fine, 98.0))
    if hi_f > lo_f + 1e-8:
        d_fine = np.clip((d_fine - lo_f) / (hi_f - lo_f), 0.0, 1.0) ** 1.18
    d_small = np.clip(d_small * (0.15 + 0.85 * band_gate), 0.0, 1.0)
    d_fine = np.clip(d_fine * (0.15 + 0.85 * band_gate), 0.0, 1.0)
    lane_w = np.clip((1.0 - t) ** 0.50, 0.05, 1.0) * band_gate
    detail = np.clip(d_small * (1.0 + 0.42 * d_med) + d_fine * (1.0 + 0.38 * d_small), 0.0, 1.0)
    band_boost = 1.0 + 0.22 * band_gate
    s_eff = float(np.clip(0.88 + 0.12 * s, 0.88, 1.5))
    out = np.clip(t * (1.0 - detail * s_eff * 0.58 * lane_w * band_boost), vf, 1.0)
    out = np.clip(out + d_fine * s_eff * 0.12 * lane_w, vf, 1.0)
    if periodic_x:
        gx = 0.5 * (np.roll(d_mega, -1, axis=1) - np.roll(d_mega, 1, axis=1))
    else:
        gx = np.gradient(d_mega, axis=1)
    gy = np.gradient(d_mega, axis=0)
    grad = np.sqrt(gx**2 + gy**2 + 1e-12)
    g93 = float(np.quantile(grad, 0.93))
    gn = np.clip(grad / (g93 + 1e-6), 0.0, 1.0)
    out = np.clip(out * (1.0 - 0.12 * gn * s * lane_w), vf, 1.0)
    frag = np.clip((d_small - d_med) * s * 0.16, -0.18, 0.18)
    out = np.clip(out * (1.0 - frag * lane_w), vf, 1.0)
    return np.minimum(out, t).astype(np.float64)


def build_morphology_extinction_transmission(
    galactic,
    *,
    void_floor: float = 0.008,
    filament_strength: float = 1.28,
    discontinuity_strength: float = 1.55,
    absorption_contrast: float = 1.62,
    extinction_strength: float = 1.72,
    disk_weight: np.ndarray | None = None,
    periodic_x: bool = True,
    lane_carve_boost: float = 0.74,
    lane_fragment_strength: float = 0.72,
    fine_texture_strength: float = 0.48,
    brutal_mask: np.ndarray | None = None,
    brutal_survival_floor: float = 0.05,
    puff_punch_mask: np.ndarray | None = None,
    morph_coupling: np.ndarray | None = None,
    opacity_gamma: float = 1.42,
) -> np.ndarray:
    """Filament-first extinction — smooth dust maps modulate lanes, not round blobs."""
    h, w = galactic.dust_absorption.shape
    scale = float(max(h, w))
    fil_s = float(np.clip(filament_strength, 0.0, 1.75))
    disc_s = float(np.clip(discontinuity_strength, 0.0, 2.0))
    ext_k = float(np.clip(extinction_strength, 0.5, 2.5))
    absorption = np.zeros((h, w), dtype=np.float64)
    if galactic.extinction_maps is not None:
        em = galactic.extinction_maps
        absorption = np.maximum(absorption, em.erosion * (1.18 * fil_s))
        fract = np.clip(em.fractal_dark, 0.0, 1.0)
        fract_soft = gaussian_blur_pil(
            fract, float(np.clip(scale * 0.04, 5.0, 48.0)), periodic_x=periodic_x
        )
        fract_lane = np.clip(fract - fract_soft * 0.78, 0.0, 1.0) ** 1.18
        absorption = np.maximum(absorption, fract_lane * (1.02 * fil_s))
        disc = np.clip(em.disruption, 0.0, 1.0)
        disc_soft = gaussian_blur_pil(
            disc, float(np.clip(scale * 0.04, 5.0, 48.0)), periodic_x=periodic_x
        )
        disc_lane = np.clip(disc - disc_soft * 0.78, 0.0, 1.0) ** 1.22
        absorption = np.maximum(absorption, disc_lane * (1.12 * min(disc_s, 1.85)))
    dust_src = np.clip(
        np.asarray(
            getattr(galactic, "dust_absorption_morph", galactic.dust_absorption),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
    dust_a = stretch_absorption_contrast(
        dust_src, contrast=absorption_contrast * 0.88, p_lo=5.0, p_hi=92.0
    )
    absorption = np.clip(
        absorption * (0.58 + 0.42 * dust_a * ext_k) + dust_a * (0.20 * ext_k),
        0.0,
        1.0,
    )
    ridge_dark = np.clip(
        1.0 - np.abs(galactic.latent_ridge * 2.0 - 1.0), 0.0, 1.0
    ) ** 1.38
    absorption = np.maximum(absorption, ridge_dark * 0.32)
    void_w = np.clip(galactic.void_mask, 0.0, 1.0)
    v_soft = gaussian_blur_pil(void_w, float(np.clip(scale * 0.05, 6.0, 56.0)), periodic_x=periodic_x)
    void_lane = np.clip(void_w - v_soft * 0.78, 0.0, 1.0) ** 1.22
    absorption = np.maximum(absorption, void_lane * 0.10)
    soft_a = gaussian_blur_pil(absorption, float(np.clip(scale * 0.036, 6.0, 44.0)), periodic_x=periodic_x)
    lane_a = np.clip(absorption - soft_a * 0.74, 0.0, 1.0) ** 1.22
    absorption = np.clip(lane_a * 0.62 + absorption * 0.38, 0.0, 1.0)
    absorption = _blur_x_only_field(absorption, 1.15, periodic_x=periodic_x)
    absorption = _blur_y_only_field(absorption, 0.35)
    if galactic.extinction_maps is not None:
        absorption = fragment_lane_absorption(
            absorption,
            galactic.extinction_maps.disruption,
            galactic.latent_turb,
            periodic_x=periodic_x,
            strength=lane_fragment_strength,
        )
    if morph_coupling is not None:
        mc = np.clip(np.asarray(morph_coupling, dtype=np.float64), 0.0, 1.35)
        if mc.shape != (h, w):
            mc = _resize_bilinear(mc, h, w, periodic_x=periodic_x)
        absorption = absorption * (0.74 + 0.26 * np.clip(mc**0.58, 0.38, 1.0))
    elif disk_weight is not None:
        dw_soft = _feathered_band_host_gate(disk_weight, (h, w), periodic_x=periodic_x)
        absorption = absorption * (0.82 + 0.18 * dw_soft**0.55)
    absorption = stretch_absorption_contrast(
        absorption, contrast=absorption_contrast * 1.14, p_lo=2.0, p_hi=88.0
    )
    vf = float(np.clip(void_floor, 0.004, 0.10))
    og = float(np.clip(opacity_gamma, 1.0, 2.6))
    trans = transmission_from_absorption_map(
        absorption, void_floor=vf, sharpness=4.25, opacity_gamma=og
    )
    if galactic.extinction_maps is not None and float(discontinuity_strength) > 1e-6:
        trans = carve_extinction_discontinuities(
            trans,
            galactic.extinction_maps.disruption,
            strength=float(discontinuity_strength) * 1.12,
            void_floor=vf,
        )
    trans = reinforce_absorption_edges(
        trans,
        absorption,
        edge_gain=0.88 * fil_s,
        periodic_x=periodic_x,
    )
    trans = finalize_morphology_extinction_transmission(
        trans,
        disk_weight,
        void_floor=vf,
        lane_carve_boost=lane_carve_boost,
        periodic_x=periodic_x,
        morph_coupling=morph_coupling,
    )
    dw_host = disk_weight if disk_weight is not None else galactic.disk_weight
    trans = apply_fine_extinction_texture(
        trans,
        dust_src,
        galactic.latent_turb,
        void_floor=vf,
        periodic_x=periodic_x,
        strength=fine_texture_strength,
        band_weight=dw_host,
        coupling_weight=morph_coupling,
    )
    trans = apply_turbulent_cloud_breakup(
        trans,
        dust_src,
        galactic.latent_turb,
        galactic.latent_ridge,
        void_floor=vf,
        periodic_x=periodic_x,
        strength=float(fine_texture_strength) * 0.92,
        band_weight=dw_host,
        coupling_weight=morph_coupling,
    )
    if brutal_mask is not None:
        from starsky_gen.structure_envelope import apply_brutal_erasure_transmission

        trans = apply_brutal_erasure_transmission(
            trans,
            brutal_mask,
            survival_floor=float(brutal_survival_floor),
            periodic_x=periodic_x,
        )
    detail_s = float(np.clip(0.62 + fine_texture_strength * 0.62, 0.0, 1.0))
    dust_t_src = np.clip(
        np.asarray(
            getattr(galactic, "dust_transmission_morph", galactic.dust_transmission),
            dtype=np.float64,
        ),
        vf,
        1.0,
    )
    trans = compose_constructive_extinction_transmission(
        trans,
        dust_t_src,
        void_floor=vf,
        detail_strength=detail_s,
        periodic_x=periodic_x,
        disk_weight=disk_weight if disk_weight is not None else galactic.disk_weight,
        coupling_weight=morph_coupling,
    )
    dw_host = disk_weight if disk_weight is not None else galactic.disk_weight
    trans = band_transmission_from_morph_absorption(
        trans,
        dust_src,
        dw_host,
        void_floor=vf,
        clear_max=0.22,
        periodic_x=periodic_x,
        filament_detail=0.34,
        puff_punch_mask=puff_punch_mask,
        coupling_weight=morph_coupling,
        opacity_gamma=og,
    )
    trans = deepen_center_band_transmission(
        trans,
        dw_host,
        void_floor=vf,
        strength=0.22,
        periodic_x=periodic_x,
    )
    trans = carve_dust_transmission_microstructure(
        trans,
        dust_src,
        periodic_x=periodic_x,
        strength=float(np.clip(0.72 + fine_texture_strength * 0.55, 0.0, 1.35)),
        void_floor=vf,
        band_weight=dw_host,
        coupling_weight=morph_coupling,
    )
    if dw_host is not None:
        from starsky_gen.structure_envelope import apply_disk_weight_pole_falloff

        dw = np.clip(np.asarray(dw_host, dtype=np.float64), 0.0, 1.0)
        if dw.ndim == 1:
            dw = dw[:, None]
        if dw.shape != trans.shape:
            dw = np.broadcast_to(dw, trans.shape)
        sky = np.clip(1.0 - apply_disk_weight_pole_falloff(dw, h, sigma=0.34, power=1.18), 0.0, 1.0)
        trans = np.clip(trans * (1.0 - sky) + sky * 0.998, vf, 1.0)
    return trans


def apply_fine_extinction_texture(
    transmission: np.ndarray,
    dust_absorption: np.ndarray,
    latent_turb: np.ndarray,
    *,
    void_floor: float,
    periodic_x: bool = True,
    strength: float = 0.48,
    band_weight: np.ndarray | None = None,
    coupling_weight: np.ndarray | None = None,
) -> np.ndarray:
    """~1–5 px extinction events on top of macro lanes (fine dust silhouette)."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return np.clip(transmission, void_floor, 1.0).astype(np.float64)
    vf = float(np.clip(void_floor, 0.004, 0.12))
    t = np.clip(np.asarray(transmission, dtype=np.float64), vf, 1.0)
    dust = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    turb = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
    ridge = np.clip(1.0 - np.abs(dust * 2.0 - 1.0), 0.0, 1.0) ** 1.15
    raw = np.maximum(dust * (0.38 + 0.62 * turb), ridge * 0.42)
    h, w = raw.shape
    scale = float(max(h, w))
    fine_sig = float(np.clip(scale * 0.00155, 0.28, 2.8))
    coarse = gaussian_blur_pil(raw, fine_sig * 2.4, periodic_x=periodic_x)
    fine = np.clip(raw - coarse, 0.0, 1.0)
    lo = float(np.percentile(fine, 64.0))
    hi = float(np.percentile(fine, 97.2))
    if hi > lo + 1e-8:
        fine = np.clip((fine - lo) / (hi - lo), 0.0, 1.0) ** 1.32
    band_gate = _extinction_coupling_gate(
        band_weight, (h, w), coupling_weight, periodic_x=periodic_x
    )
    carve = fine * s * 1.34 * np.clip((1.0 - t) ** 0.52, 0.08, 1.0) * band_gate
    t = np.clip(t * (1.0 - carve), vf, 1.0)
    turb_mix = np.clip(turb * 0.62 + ridge * 0.38, 0.0, 1.0)
    wisp = np.clip(turb_mix - gaussian_blur_pil(turb_mix, fine_sig * 1.6, periodic_x=periodic_x), 0.0, 1.0)
    wisp = np.clip(wisp**1.12, 0.0, 1.0) * s * 0.34 * band_gate
    return np.clip(t * (1.0 - wisp * np.clip((1.0 - t) ** 0.42, 0.1, 1.0)), vf, 1.0).astype(np.float64)


def finalize_morphology_extinction_transmission(
    transmission: np.ndarray,
    disk_weight: np.ndarray | None,
    *,
    void_floor: float,
    lane_carve_boost: float = 0.74,
    periodic_x: bool = True,
    morph_coupling: np.ndarray | None = None,
) -> np.ndarray:
    """Extra asymmetric lane carve on top of filament/disruption stack."""
    _ = periodic_x
    vf = float(np.clip(void_floor, 0.004, 0.12))
    out = np.asarray(transmission, dtype=np.float64)
    dark = np.clip(1.0 - out, 0.0, 1.0)
    scale = float(max(dark.shape))
    soft = gaussian_blur_pil(dark, float(np.clip(scale * 0.020, 3.0, 32.0)), periodic_x=periodic_x)
    lane = np.clip(dark - soft * 0.82, 0.0, 1.0) ** 1.35
    if morph_coupling is not None:
        mc = np.clip(np.asarray(morph_coupling, dtype=np.float64), 0.0, 1.35)
        if mc.shape != dark.shape:
            mc = _resize_bilinear(mc, dark.shape[0], dark.shape[1], periodic_x=periodic_x)
        lane = lane * (0.48 + 0.52 * np.clip(mc**0.62, 0.35, 1.0))
    elif disk_weight is not None:
        dw_soft = _feathered_band_host_gate(disk_weight, dark.shape, periodic_x=periodic_x)
        lane = lane * (0.52 + 0.48 * dw_soft**0.65)
    boost = float(np.clip(lane_carve_boost, 0.0, 0.98))
    return np.clip(out * (1.0 - lane * boost * 0.72), vf, 1.0).astype(np.float64)


def apply_missing_region_extinction(
    canvas: np.ndarray,
    transmission: np.ndarray,
    *,
    void_floor: float = 0.004,
    missing_boost: float = 0.52,
) -> np.ndarray:
    """Second-pass multiplicative mask: unresolved → extincted → missing regions."""
    ext = np.clip(np.asarray(transmission, dtype=np.float64), 0.0, 1.0)
    vf = float(np.clip(void_floor, 0.002, 0.12))
    boost = float(np.clip(missing_boost, 0.0, 0.95))
    dark = np.clip(1.0 - ext, 0.0, 1.0) ** 2.42
    h, w = dark.shape
    scale = float(max(h, w))
    d_soft = _blur_x_only_field(
        dark, float(np.clip(scale * 0.06, 4.0, 64.0)), periodic_x=True
    )
    dark_lane = np.clip(dark - d_soft * 0.72, 0.0, 1.0) ** 1.28
    dark_mix = np.clip(dark_lane * 0.68 + dark * 0.32, 0.0, 1.0)
    extra = np.clip(1.0 - dark_mix * boost, vf, 1.0)
    out = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    if out.ndim == 3:
        return np.maximum(0.0, out * extra[..., np.newaxis])
    return np.maximum(0.0, out * extra)


def blend_extinction_fields(
    procedural_ext: np.ndarray,
    fractal_ext: np.ndarray,
    *,
    fractal_weight: float = 0.48,
) -> np.ndarray:
    """Merge nebula lane extinction with fractal dust mask (max = darker)."""
    p = np.clip(np.asarray(procedural_ext, dtype=np.float64), 0.0, 1.0)
    f = np.clip(np.asarray(fractal_ext, dtype=np.float64), 0.0, 1.0)
    w = float(np.clip(fractal_weight, 0.0, 1.0))
    return np.clip(np.maximum(p * (1.0 - w * 0.35), f * w + p * (1.0 - w)), 0.0, 1.0)
