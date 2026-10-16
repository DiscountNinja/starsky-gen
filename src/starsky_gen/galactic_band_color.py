"""Final display-space galactic band color grade (gas + diffuse, not stars-only)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from starsky_gen.color_science import rec709_luma, remap_luma_preserving_chroma
from starsky_gen.dust_field import band_relative_clearance


def _ratio(rgb: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    c = np.maximum(np.asarray(rgb, dtype=np.float64).reshape(3), 0.0)
    lu = float(np.dot(c, np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)))
    return (c / max(lu, 1e-8)).astype(np.float64)


# Stellar-population palette (Rec.709 luminance-normalized ratios).
_GOLD_OLD = _ratio((1.04, 0.92, 0.72))
_GOLD_WHITE = _ratio((1.02, 0.99, 0.92))
_GOLD_WARM = _ratio((1.01, 0.98, 0.88))
_DUST_BROWN = _ratio((0.46, 0.42, 0.36))
_DUST_BLACK = _ratio((0.12, 0.11, 0.10))
_HII_RED = _ratio((1.42, 0.28, 0.16))
_HII_ORANGE = _ratio((1.28, 0.52, 0.22))
_COOL_SHELL = _ratio((0.72, 0.82, 1.08))
_WARM_CORE = _ratio((1.18, 0.78, 0.48))
_GOLD_AMBER = _ratio((1.22, 0.92, 0.42))


def _sparse_peak_mask(
    field: np.ndarray,
    plane: np.ndarray,
    *,
    percentile: float = 99.0,
    power: float = 3.2,
    min_plane: float = 0.20,
) -> np.ndarray:
    """Keep only rare peaks (Milky-Way-like sparse H II), not broad SF/nebula fields."""
    f = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0) * np.clip(plane, 0.0, 1.0)
    gate = plane > float(min_plane)
    if not bool(np.any(gate)):
        return np.zeros_like(f, dtype=np.float64)
    thr = float(np.percentile(f[gate], float(np.clip(percentile, 90.0, 99.8))))
    span = max(1.0 - thr, 0.04)
    peaks = np.clip((f - thr) / span, 0.0, 1.0) ** float(power)
    return peaks * gate.astype(np.float64)


def _structure_lane_weight(
    gas_struct: np.ndarray,
    lane_dark: np.ndarray,
    plane: np.ndarray,
    blur_fn: Callable[..., np.ndarray],
    *,
    periodic_x: bool,
    inner_bias: float,
    xx: np.ndarray | None = None,
) -> np.ndarray:
    """Morphology-driven lane weight — avoids a smooth Gaussian donut at band center."""
    med = np.clip(blur_fn(gas_struct, passes=2, periodic_x=periodic_x), 0.0, 1.0)
    hf = np.clip((gas_struct - med * 0.78) * 2.5, 0.0, 1.0) ** 1.06
    struct = np.clip(0.30 * gas_struct + 0.70 * hf, 0.0, 1.0)
    if xx is not None:
        lane_mid = np.exp(-(xx**2) / 0.13) * lane_dark
        smooth_blob = lane_mid * (1.0 - hf) ** 1.4
        struct = np.clip(struct * (1.0 - smooth_blob * 0.68), 0.0, 1.0)
    bias = float(np.clip(inner_bias, 0.0, 1.2))
    return np.clip(struct * plane * (0.64 + 0.36 * bias), 0.0, 1.12)


def _extinction_gas_structure(
    ext: np.ndarray,
    dust: np.ndarray,
    lane_dark: np.ndarray,
    blur_fn: Callable[..., np.ndarray],
    *,
    periodic_x: bool,
    latent_turb: np.ndarray | None = None,
    gas_texture: np.ndarray | None = None,
) -> np.ndarray:
    """High-frequency extinction filaments visible in the gas (not a smooth brown wash)."""
    ext_dark = np.clip(1.0 - np.clip(ext, 0.0, 1.0), 0.0, 1.0)
    med = np.clip(blur_fn(ext_dark, passes=2, periodic_x=periodic_x), 0.0, 1.0)
    hf_lane = np.clip((ext_dark - med * 0.80) * 3.6, 0.0, 1.0) ** 1.10
    med_fine = np.clip(blur_fn(ext_dark, passes=1, periodic_x=periodic_x), 0.0, 1.0)
    hf_micro = np.clip((ext_dark - med_fine * 0.68) * 4.8, 0.0, 1.0) ** 1.18
    dust_r = np.clip(dust, 0.0, 1.0) ** 0.92
    struct = np.clip(
        lane_dark * 0.38
        + dust_r * 0.32
        + hf_lane * 0.52
        + hf_micro * 0.88
        + lane_dark * dust_r * 0.28,
        0.0,
        1.0,
    ) ** 0.84
    if latent_turb is not None:
        turb = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
        if turb.shape != struct.shape:
            turb = np.broadcast_to(turb, struct.shape)
        t_med = np.clip(blur_fn(turb, passes=1, periodic_x=periodic_x), 0.0, 1.0)
        t_hf = np.clip((turb - t_med * 0.74) * 3.2, 0.0, 1.0) ** 1.12
        struct = np.clip(struct * 0.72 + t_hf * 0.42 + hf_micro * t_hf * 0.35, 0.0, 1.0)
    if gas_texture is not None:
        gt = np.clip(np.asarray(gas_texture, dtype=np.float64), 0.0, 1.0)
        if gt.shape != struct.shape:
            gt = np.broadcast_to(gt, struct.shape)
        g_med = np.clip(blur_fn(gt, passes=1, periodic_x=periodic_x), 0.0, 1.0)
        g_hf = np.clip((gt - g_med * 0.70) * 3.4, 0.0, 1.0) ** 1.08
        struct = np.clip(np.maximum(struct, g_hf * 0.78), 0.0, 1.0)
    return struct


def _apply_fluffy_micro_gas_texture(
    out: np.ndarray,
    *,
    gas_struct: np.ndarray,
    rel_clear: np.ndarray,
    turb: np.ndarray,
    plane: np.ndarray,
    pointness: np.ndarray,
    blur_fn: Callable[..., np.ndarray],
    rng: np.random.Generator,
    strength: float,
    periodic_x: bool,
) -> np.ndarray:
    """Puffy cleared pockets + tight micro contrast in the cloud body."""
    st = float(np.clip(strength, 0.0, 1.3))
    if st < 1e-6:
        return out
    lu = rec709_luma(np.maximum(out, 0.0))
    med = np.clip(blur_fn(gas_struct, passes=1, periodic_x=periodic_x), 0.0, 1.0)
    hf = np.clip((gas_struct - med * 0.70) * 3.0, 0.0, 1.0) ** 1.06
    t_med = np.clip(blur_fn(turb, passes=2, periodic_x=periodic_x), 0.0, 1.0)
    t_hf = np.clip((turb - t_med * 0.76) * 2.8, 0.0, 1.0) ** 1.08
    n_micro = rng.random(gas_struct.shape, dtype=np.float64)
    n_micro = np.clip(n_micro * 1.8 - 0.55, 0.0, 1.0)
    body = np.clip((lu - 0.048) / 0.045, 0.0, 1.0)
    gate = plane * (1.0 - pointness * 0.88) * body
    puff = rel_clear * (0.28 + 0.72 * t_hf) * hf * gate * st
    puff_lu = np.clip(puff * 0.11, 0.0, 0.075)
    micro_dark = hf * (0.42 + 0.58 * (1.0 - rel_clear)) * n_micro * gate * st * 0.38
    out_lu = np.clip(lu + puff_lu, 0.0, 1.0)
    out_lu = np.clip(out_lu * (1.0 - micro_dark), 0.0, 1.0)
    return remap_luma_preserving_chroma(out, out_lu)


def _turbulent_black_dust(
    out: np.ndarray,
    *,
    ext: np.ndarray,
    lane_dark: np.ndarray,
    dust: np.ndarray,
    void_w: np.ndarray,
    turb: np.ndarray,
    plane: np.ndarray,
    pointness: np.ndarray,
    strength: float,
    blur_fn: Callable[..., np.ndarray],
    rng: np.random.Generator,
    periodic_x: bool,
    xx: np.ndarray | None = None,
    inner_bias: float = 1.0,
) -> np.ndarray:
    """Carve turbulent near-black dust into the gas using extinction structure."""
    st = float(np.clip(strength, 0.0, 1.35))
    if st < 1e-6:
        return out
    lu_out = rec709_luma(np.maximum(out, 0.0))
    gas_struct = _extinction_gas_structure(
        ext, dust, lane_dark, blur_fn, periodic_x=periodic_x, latent_turb=turb
    )
    struct_w = _structure_lane_weight(
        gas_struct,
        lane_dark,
        plane,
        blur_fn,
        periodic_x=periodic_x,
        inner_bias=inner_bias,
        xx=xx,
    )
    obscure = (
        gas_struct
        * (0.18 + 0.82 * np.clip(void_w, 0.0, 1.0))
        * (0.35 + 0.65 * np.clip(turb, 0.0, 1.0))
        * (0.42 + 0.58 * struct_w)
        * plane
    )
    obscure = np.clip(obscure**0.72, 0.0, 1.0)
    if float(np.max(obscure)) < 1e-6:
        return out

    n_micro = np.clip(rng.random(obscure.shape, dtype=np.float64) * 1.9 - 0.52, 0.0, 1.0)
    n_fine = blur_fn(rng.random(obscure.shape, dtype=np.float64), passes=1, periodic_x=periodic_x)
    n_coarse = blur_fn(rng.random(obscure.shape, dtype=np.float64), passes=3, periodic_x=periodic_x)
    noise = (
        n_micro * 0.32
        + np.clip(n_fine * 1.45 - 0.38, 0.0, 1.0) * 0.38
        + np.clip(n_coarse * 1.2 - 0.32, 0.0, 1.0) * 0.30
    )
    obscure = obscure * (0.14 + 0.86 * noise)
    fluff_pocket = np.clip(turb - blur_fn(turb, passes=2, periodic_x=periodic_x) * 0.8, 0.0, 1.0)
    obscure = np.clip(obscure * (0.78 + 0.42 * fluff_pocket) - obscure * fluff_pocket * 0.18, 0.0, 1.0)

    # Gas body: allow crushing bright ISM, not sky or stars.
    ism = np.clip((lu_out - 0.032) / 0.048, 0.0, 1.0)
    ism = ism * np.clip(1.0 - (lu_out - 0.88) / 0.12, 0.0, 1.0) ** 0.65
    carve = obscure * ism * (1.0 - pointness * 0.97) * st
    if float(np.max(carve)) < 1e-6:
        return out

    # Crush toward true black — not proportional brown dimming.
    crush = np.clip(carve, 0.0, 1.0)
    target_lu = np.clip(lu_out * (1.0 - crush) ** 2.35, 0.0, None)
    target_lu = np.minimum(
        target_lu,
        0.006 + lu_out * (1.0 - crush) * 0.05,
    )
    black_rgb = _DUST_BLACK.reshape(1, 1, 3) * target_lu[..., np.newaxis]
    mix = crush[..., np.newaxis]
    return np.clip(out * (1.0 - mix) + black_rgb * mix, 0.0, 1.0)


def _display_micro_fields(
    ext: np.ndarray,
    dust: np.ndarray | None,
    turb: np.ndarray | None,
    morph_gas: np.ndarray | None,
    *,
    periodic_x: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-scale + fine-scale structure for display sculpting (minimal blur)."""
    from starsky_gen.procedural_noise import gaussian_blur_pil

    h, w = int(ext.shape[0]), int(ext.shape[1])
    scale = float(max(h, w))
    sig_ultra = float(np.clip(scale * 0.0009, 0.14, 1.25))
    sig_fine = float(np.clip(scale * 0.0024, 0.32, 2.6))
    sig_lane = float(np.clip(scale * 0.0095, 1.0, 9.0))

    ext_dark = np.clip(1.0 - np.clip(ext, 0.0, 1.0), 0.0, 1.0)
    med_u = gaussian_blur_pil(ext_dark, sig_ultra, periodic_x=periodic_x)
    med_f = gaussian_blur_pil(ext_dark, sig_fine, periodic_x=periodic_x)
    med_l = gaussian_blur_pil(ext_dark, sig_lane, periodic_x=periodic_x)
    micro = np.clip((ext_dark - med_u) * 7.5, 0.0, 1.0) ** 1.02
    fine = np.clip((ext_dark - med_f) * 4.2, 0.0, 1.0) ** 1.04
    lane = np.clip((ext_dark - med_l) * 2.6, 0.0, 1.0) ** 1.06

    if dust is not None:
        d = np.clip(np.asarray(dust, dtype=np.float64), 0.0, 1.0)
        if d.shape != ext.shape:
            d = np.broadcast_to(d, ext.shape)
        d_u = gaussian_blur_pil(d, sig_ultra, periodic_x=periodic_x)
        d_f = gaussian_blur_pil(d, sig_fine, periodic_x=periodic_x)
        micro = np.clip(np.maximum(micro, (d - d_u) * 6.8), 0.0, 1.0)
        fine = np.clip(np.maximum(fine, (d - d_f) * 3.8), 0.0, 1.0)

    if turb is not None:
        t = np.clip(np.asarray(turb, dtype=np.float64), 0.0, 1.0)
        if t.shape != ext.shape:
            t = np.broadcast_to(t, ext.shape)
        t_f = gaussian_blur_pil(t, sig_fine, periodic_x=periodic_x)
        t_hf = np.clip((t - t_f) * 4.5, 0.0, 1.0) ** 1.08
        fine = np.clip(np.maximum(fine, t_hf * 0.72), 0.0, 1.0)

    if morph_gas is not None:
        g = np.clip(np.asarray(morph_gas, dtype=np.float64), 0.0, 1.0)
        if g.shape != ext.shape:
            g = np.broadcast_to(g, ext.shape)
        g_u = gaussian_blur_pil(g, sig_ultra, periodic_x=periodic_x)
        g_hf = np.clip((g - g_u) * 5.5, 0.0, 1.0) ** 1.05
        micro = np.clip(np.maximum(micro, g_hf), 0.0, 1.0)

    return micro.astype(np.float64), fine.astype(np.float64), lane.astype(np.float64)


def apply_band_display_microstructure(
    canvas: np.ndarray,
    ext_paint: np.ndarray,
    disk_w: np.ndarray,
    *,
    dust_absorption: np.ndarray | None = None,
    latent_turb: np.ndarray | None = None,
    morph_gas: np.ndarray | None = None,
    strength: float = 1.05,
    periodic_x: bool = True,
) -> np.ndarray:
    """Final display sculpt: visible micro filaments and fluffy cloud breakup in the gas."""
    st = float(np.clip(strength, 0.0, 1.5))
    if st < 1e-6:
        return np.asarray(canvas, dtype=np.float64)
    lin = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    h, w = int(lin.shape[0]), int(lin.shape[1])
    ext = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
    if ext.shape != (h, w):
        ext = np.broadcast_to(ext, (h, w))
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw.ndim == 1:
        dw = dw[:, None]
    if dw.shape != (h, w):
        dw = np.broadcast_to(dw, (h, w))
    plane = np.clip(dw**1.05, 0.0, 1.0)

    micro, fine, lane = _display_micro_fields(
        ext,
        dust_absorption,
        latent_turb,
        morph_gas,
        periodic_x=periodic_x,
    )
    rel_clear = band_relative_clearance(ext, dw, min_clear=0.14, power=0.88)

    lu = rec709_luma(lin)
    from starsky_gen.procedural_noise import gaussian_blur_pil

    scale = float(max(h, w))
    sig_pt = float(np.clip(scale * 0.0028, 0.35, 3.2))
    lu_blur = gaussian_blur_pil(lu, sig_pt, periodic_x=periodic_x)
    pointness = np.clip((lu - lu_blur) / np.maximum(lu, 1e-6), 0.0, 1.0) ** 0.86
    gate = plane * (1.0 - pointness * 0.90)

    composite = np.clip(micro * (0.50 + 0.50 * fine) * (0.40 + 0.60 * lane), 0.0, 1.0)
    puff = np.clip(fine * (1.0 - micro * 0.50) * rel_clear, 0.0, 1.0)

    on = gate > 0.25
    ref = float(np.median(composite[on])) if bool(np.any(on)) else 0.42
    signed = (composite - ref) * gate * st
    lu_sculpt = np.clip(lu + signed * 0.52, 0.0, 1.0)
    lu_sculpt = np.clip(
        lu_sculpt + puff * gate * st * 0.16 - composite * (1.0 - rel_clear) * gate * st * 0.10,
        0.0,
        1.0,
    )

    ratio = lin / np.maximum(lu[..., np.newaxis], 1e-6)
    out = np.maximum(ratio * lu_sculpt[..., np.newaxis], 0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float64)


def apply_band_luma_separation(
    canvas: np.ndarray,
    ext_paint: np.ndarray,
    disk_w: np.ndarray,
    *,
    dust_absorption: np.ndarray | None = None,
    latent_turb: np.ndarray | None = None,
    morph_gas: np.ndarray | None = None,
    strength: float = 1.12,
    periodic_x: bool = True,
) -> np.ndarray:
    """Expand local contrast in the gas — visible gaps between clouds and lanes."""
    st = float(np.clip(strength, 0.0, 1.6))
    if st < 1e-6:
        return np.asarray(canvas, dtype=np.float64)
    lin = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    h, w = int(lin.shape[0]), int(lin.shape[1])
    ext = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
    if ext.shape != (h, w):
        ext = np.broadcast_to(ext, (h, w))
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw.ndim == 1:
        dw = dw[:, None]
    if dw.shape != (h, w):
        dw = np.broadcast_to(dw, (h, w))
    plane = np.clip(dw**1.05, 0.0, 1.0)

    micro, fine, lane = _display_micro_fields(
        ext,
        dust_absorption,
        latent_turb,
        morph_gas,
        periodic_x=periodic_x,
    )
    rel_clear = band_relative_clearance(ext, dw, min_clear=0.12, power=0.82)
    composite = np.clip(micro * (0.48 + 0.52 * fine) * (0.38 + 0.62 * lane), 0.0, 1.0)

    lu = rec709_luma(lin)
    from starsky_gen.procedural_noise import gaussian_blur_pil

    scale = float(max(h, w))
    sig_loc = float(np.clip(scale * 0.011, 0.75, 9.0))
    sig_pt = float(np.clip(scale * 0.0028, 0.35, 3.2))
    local = gaussian_blur_pil(lu, sig_loc, periodic_x=periodic_x)
    lu_blur = gaussian_blur_pil(lu, sig_pt, periodic_x=periodic_x)
    pointness = np.clip((lu - lu_blur) / np.maximum(lu, 1e-6), 0.0, 1.0) ** 0.86
    gate = plane * (1.0 - pointness * 0.90)

    dev = lu - local
    sep = gate * (0.40 + 0.60 * composite) * st
    lu_sep = local + dev * (1.0 + sep * 0.85)

    dark_gap = composite * (1.0 - rel_clear) ** 1.05 * gate * st * 0.22
    bright_gap = (1.0 - composite) * rel_clear * fine * gate * st * 0.18
    lu_sep = np.clip(lu_sep * (1.0 - dark_gap) + bright_gap, 0.0, 1.0)

    return remap_luma_preserving_chroma(lin, np.clip(lu_sep, 0.0, 1.0))


def suppress_gold_chroma_islands(
    canvas: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float = 0.85,
    periodic_x: bool = True,
    blur_fn: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """Strip jagged gold chroma dashes — especially on band shoulders.

    Midplane ``disk_weight`` alone is the wrong gate: shoulder flecks sit where
    plane is low, so a plane-gated kill misses them. Use a wide band host and
    extra shoulder pressure.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return np.asarray(canvas, dtype=np.float64)
    out = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    h, w = int(out.shape[0]), int(out.shape[1])
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw.ndim == 1:
        dw = dw[:, None]
    if dw.shape != (h, w):
        dw = np.broadcast_to(dw, (h, w))
    if blur_fn is None:
        from starsky_gen.nebula import _blur_separable_xy

        blur_fn = _blur_separable_xy
    from starsky_gen.structure_envelope import soften_band_envelope

    # Wide host so shoulder latitudes still get the kill.
    from starsky_gen.structure_envelope import midplane_gold_lat_gate

    wide = soften_band_envelope(dw, (h, w), periodic_x=periodic_x, lat_blur_sigma=22.0, power=0.42)
    wide = np.clip(wide, 0.0, 1.0)
    core = np.clip(dw**1.35, 0.0, 1.0)
    lat_core = midplane_gold_lat_gate(h, sigma=0.10, power=1.70)
    if lat_core.shape != (h, w):
        lat_core = np.broadcast_to(lat_core, (h, w))
    # Shoulder pressure: band envelope but off the razor midplane (lat + disk).
    shoulder = np.clip(wide * (1.0 - core * 0.70) * (1.0 - lat_core * 0.85), 0.0, 1.0)
    lu = rec709_luma(out)
    lu_blur = blur_fn(lu, passes=2, periodic_x=periodic_x)
    pointness = np.clip((lu - lu_blur) / np.maximum(lu, 1e-6), 0.0, 1.0) ** 0.9
    star_protect = np.clip(pointness / 0.07, 0.0, 1.0) ** 1.45
    gold_ex = np.clip(out[..., 0] - np.maximum(out[..., 1], out[..., 2]) * 1.04, 0.0, None)
    gold_soft = blur_fn(gold_ex, passes=5, periodic_x=periodic_x)
    gold_hf = np.clip(gold_ex - gold_soft * 0.72, 0.0, None)
    body = np.clip((lu - 0.04) / 0.18, 0.0, 1.0)
    gate = (0.55 * wide + 0.90 * shoulder) * (1.0 - star_protect) * body
    # HF streaks + any residual gold excess on shoulders.
    kill = (gold_hf * 1.15 + gold_ex * shoulder * 0.55) * gate * s
    kill = np.clip(kill, 0.0, gold_ex)
    out[..., 0] = np.maximum(0.0, out[..., 0] - kill)
    out[..., 1] = out[..., 1] + kill * 0.48
    out[..., 2] = out[..., 2] + kill * 0.22
    # Soft midplane: compress remaining gold peaks toward warm (not yellow).
    gold_ex2 = np.clip(out[..., 0] - np.maximum(out[..., 1], out[..., 2]) * 1.04, 0.0, None)
    peak = np.clip(gold_ex2 - gold_soft * 0.90, 0.0, None)
    peak_kill = peak * (0.40 + 0.60 * wide) * (1.0 - star_protect) * 0.70 * s
    out[..., 0] = np.maximum(0.0, out[..., 0] - peak_kill)
    out[..., 1] = out[..., 1] + peak_kill * 0.40

    # Thin horizontal dashes (1–2 px tall gold runs): kill when above/below are cooler.
    gold_ex3 = np.clip(out[..., 0] - np.maximum(out[..., 1], out[..., 2]) * 1.04, 0.0, None)
    above = np.roll(gold_ex3, 1, axis=0)
    below = np.roll(gold_ex3, -1, axis=0)
    above[0, :] = gold_ex3[0, :]
    below[-1, :] = gold_ex3[-1, :]
    vert_max = np.maximum(above, below)
    thin = np.clip(gold_ex3 - vert_max * 1.15, 0.0, None)
    thin_gate = (1.0 - star_protect) * np.clip(wide + shoulder, 0.0, 1.0) * body
    thin_kill = thin * thin_gate * (1.15 * s)
    out[..., 0] = np.maximum(0.0, out[..., 0] - thin_kill)
    out[..., 1] = out[..., 1] + thin_kill * 0.50
    out[..., 2] = out[..., 2] + thin_kill * 0.25
    return np.clip(out, 0.0, 1.0).astype(np.float64)


def apply_core_multi_scatter_glow(
    canvas: np.ndarray,
    *,
    opaque_core: np.ndarray,
    disk_w: np.ndarray,
    void_mask: np.ndarray | None,
    strength: float,
    periodic_x: bool,
    blur_fn: Callable[..., np.ndarray],
    density_g: np.ndarray | None = None,
    structure_survival: np.ndarray | None = None,
    gold_population: np.ndarray | None = None,
) -> np.ndarray:
    """Cheap multi-scatter: density + gold-weighted amber cores × disk_weight.

    Intensity uses soft ``density_g`` (primary) and soft ``gold_population_weight``
    multiplicatively so the older core is brighter and gold-tinted. Sharp survival
    is only a secondary soft cue.
    """
    s = float(np.clip(strength, 0.0, 0.55))
    if s < 1e-6:
        return canvas
    h, w = int(opaque_core.shape[0]), int(opaque_core.shape[1])
    if density_g is not None:
        dens = np.clip(np.asarray(density_g, dtype=np.float64), 0.0, 1.0)
        if dens.shape != (h, w):
            dens = np.broadcast_to(dens, (h, w))
        dens = blur_fn(dens, passes=5, periodic_x=periodic_x)
    else:
        dens = blur_fn(np.clip(np.asarray(opaque_core, dtype=np.float64), 0.0, 1.0), passes=5, periodic_x=periodic_x)
    # Survival almost unused — density_G drives soft volumetric highlights, not hard islands.
    if structure_survival is not None:
        surv = np.clip(np.asarray(structure_survival, dtype=np.float64), 0.0, 1.35) / 1.35
        if surv.shape != (h, w):
            surv = np.broadcast_to(surv, (h, w))
        surv_soft = blur_fn(surv, passes=7, periodic_x=periodic_x)
        dens = np.clip(0.97 * dens + 0.03 * surv_soft, 0.0, 1.0)
    opq = blur_fn(np.clip(np.asarray(opaque_core, dtype=np.float64), 0.0, 1.0), passes=3, periodic_x=periodic_x)
    dens = np.clip(0.92 * dens + 0.08 * opq, 0.0, 1.0)

    gold = np.zeros((h, w), dtype=np.float64)
    if gold_population is not None:
        gold = np.clip(np.asarray(gold_population, dtype=np.float64), 0.0, 1.0)
        if gold.shape != (h, w):
            gold = np.broadcast_to(gold, (h, w))
        gold = blur_fn(gold, passes=4, periodic_x=periodic_x)
        # Tint driver = soft density + gold (avoids streaky gold-only hotspots).
        gold = np.clip(0.55 * dens + 0.45 * gold, 0.0, 1.0)
        gold = gold / max(float(np.percentile(gold, 99.5)), 1e-6)
        gold = np.clip(gold, 0.0, 1.0) ** 1.15  # compress peaks

    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw.ndim == 1:
        dw = dw[:, None]
    if dw.shape != (h, w):
        dw = np.broadcast_to(dw, (h, w))
    # Tight disk + latitude falloff so gold dies on visual shoulders.
    from starsky_gen.structure_envelope import midplane_gold_lat_gate

    dw_tight = blur_fn(np.clip(dw, 0.0, 1.0), passes=2, periodic_x=periodic_x) ** 2.35
    lat = midplane_gold_lat_gate(h, sigma=0.095, power=1.90)
    if lat.shape != (h, w):
        lat = np.broadcast_to(lat, (h, w))
    gate = dens * dw_tight * lat * (0.28 + 0.72 * gold) * dens
    if void_mask is not None:
        void_w = np.clip(np.asarray(void_mask, dtype=np.float64), 0.0, 1.0)
        if void_w.shape == gate.shape:
            gate = gate * (1.0 - 0.82 * blur_fn(void_w, passes=2, periodic_x=periodic_x))
    soft = blur_fn(gate, passes=4, periodic_x=periodic_x)
    soft = soft / max(float(np.percentile(soft, 99.5)), 1e-6)
    glow = np.clip(soft, 0.0, 1.0) ** 1.18
    # Soft warm core — lower peak so it reads amber, not bright streaks.
    tint = (
        _GOLD_AMBER.reshape(1, 1, 3) * gold[..., np.newaxis] * 0.72
        + _WARM_CORE.reshape(1, 1, 3) * (1.0 - gold[..., np.newaxis]) * 0.48
    )
    add = glow[..., np.newaxis] * tint * (0.028 * s)
    return np.clip(np.asarray(canvas, dtype=np.float64) + add, 0.0, 1.0)


def apply_galactic_band_color_grade(
    canvas: np.ndarray,
    disk_w: np.ndarray,
    ext_paint: np.ndarray,
    *,
    dust_absorption: np.ndarray | None = None,
    star_formation: np.ndarray | None = None,
    void_mask: np.ndarray | None = None,
    latent_turb: np.ndarray | None = None,
    hii_hint: np.ndarray | None = None,
    dust_lane_weight: np.ndarray | None = None,
    puff_shell_weight: np.ndarray | None = None,
    stellar_density: np.ndarray | None = None,
    structure_survival: np.ndarray | None = None,
    gold_population: np.ndarray | None = None,
    strength: float = 0.88,
    inner_bias: float = 0.72,
    anti_blue: float = 0.38,
    void_turbulence: float = 0.32,
    hii_strength: float = 0.06,
    dust_black_strength: float = 1.12,
    gas_fluff_strength: float = 0.78,
    micro_display_strength: float = 1.32,
    separation_strength: float = 0.0,
    multi_scatter_strength: float = 0.22,
    black_preserve_floor: float = 0.028,
    gas_texture: np.ndarray | None = None,
    periodic_x: bool = True,
    blur_fn: Callable[..., np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Re-map diffuse band color: warm gold/white ISM, brown/black lanes, sparse H II.

    Targets gas and continuum in the plane; resolved stars are mostly preserved via
    a point-source mask. Soft black lift below ``black_preserve_floor`` is allowed for
    cinematic skyboxes (voids stay cooler, plane warmer).
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return np.asarray(canvas, dtype=np.float64)
    lin = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    h, w = int(lin.shape[0]), int(lin.shape[1])
    dw_pre = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw_pre.ndim == 1:
        dw_pre = dw_pre[:, None]
    if dw_pre.shape != (h, w):
        dw_pre = np.broadcast_to(dw_pre, (h, w))
    plane_pre = np.clip(dw_pre**1.05, 0.0, 1.0)
    exc_b = np.clip(
        lin[..., 2] - np.maximum(lin[..., 0], lin[..., 1]) * 0.94,
        0.0,
        None,
    )
    if float(np.max(exc_b)) > 1e-6:
        sh = plane_pre * 0.38
        lin = lin.copy()
        lin[..., 2] = np.maximum(0.0, lin[..., 2] - exc_b * sh)
        lin[..., 0] = lin[..., 0] + exc_b * sh * 0.34
        lin[..., 1] = lin[..., 1] + exc_b * sh * 0.20
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw.ndim == 1:
        dw = dw[:, None]
    if dw.shape != (h, w):
        dw = np.broadcast_to(dw, (h, w))
    ext = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
    if ext.shape != (h, w):
        ext = np.broadcast_to(ext, (h, w))

    if blur_fn is None:
        from starsky_gen.nebula import _blur_separable_xy

        blur_fn = _blur_separable_xy

    lu = rec709_luma(lin)
    lu_blur = np.clip(blur_fn(lu, passes=2, periodic_x=periodic_x), 0.0, None)
    hf = np.clip(lu - lu_blur, 0.0, None)
    pointness = np.clip(hf / np.maximum(lu, 1e-6), 0.0, 1.0) ** 0.88
    bright_keep = np.clip((lu - 0.26) / 0.16, 0.0, 1.0) * np.clip(pointness / 0.10, 0.0, 1.0)
    plane = np.clip(dw**1.05, 0.0, 1.0)

    bf = float(np.clip(black_preserve_floor, 0.02, 0.14))
    luma_gate = np.clip((lu - bf) / max(0.12 - bf, 0.04), 0.0, 1.0) ** 1.35
    star_protect = np.clip(pointness / 0.08, 0.0, 1.0) ** 1.35
    diffuse_w = (
        np.clip(
            plane
            * luma_gate
            * (1.0 - pointness * 0.98)
            * (1.0 - bright_keep * 0.98)
            * (1.0 - star_protect * 0.92),
            0.0,
            1.0,
        )
        * s
    )

    rel_clear = band_relative_clearance(ext, dw, min_clear=0.14, power=0.90)
    lane_dark = np.clip(1.0 - rel_clear, 0.0, 1.0) ** 1.14

    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]

    sf = (
        np.clip(np.asarray(star_formation, dtype=np.float64), 0.0, 1.0)
        if star_formation is not None
        else np.zeros((h, w), dtype=np.float64)
    )
    if sf.shape != (h, w):
        sf = np.broadcast_to(sf, (h, w))
    dust = (
        np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
        if dust_absorption is not None
        else lane_dark
    )
    if dust.shape != (h, w):
        dust = np.broadcast_to(dust, (h, w))
    void_w = (
        np.clip(np.asarray(void_mask, dtype=np.float64), 0.0, 1.0)
        if void_mask is not None
        else lane_dark * 0.5
    )
    if void_w.shape != (h, w):
        void_w = np.broadcast_to(void_w, (h, w))
    turb = (
        np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
        if latent_turb is not None
        else void_w
    )
    if turb.shape != (h, w):
        turb = np.broadcast_to(turb, (h, w))

    gas_struct = _extinction_gas_structure(
        ext,
        dust,
        lane_dark,
        blur_fn,
        periodic_x=periodic_x,
        latent_turb=turb,
        gas_texture=gas_texture,
    )
    struct_w = _structure_lane_weight(
        gas_struct,
        lane_dark,
        plane,
        blur_fn,
        periodic_x=periodic_x,
        inner_bias=inner_bias,
        xx=xx,
    )

    lane_arch = (
        np.clip(np.asarray(dust_lane_weight, dtype=np.float64), 0.0, 1.0)
        if dust_lane_weight is not None
        else lane_dark
    )
    if lane_arch.shape != (h, w):
        lane_arch = np.broadcast_to(lane_arch, (h, w))
    shell_arch = (
        np.clip(np.asarray(puff_shell_weight, dtype=np.float64), 0.0, 1.0)
        if puff_shell_weight is not None
        else np.clip(plane * (1.0 - lane_dark), 0.0, 1.0)
    )
    if shell_arch.shape != (h, w):
        shell_arch = np.broadcast_to(shell_arch, (h, w))

    w_warm = (
        (rel_clear**0.55 + 0.34 * plane)
        * (1.0 - sf * 0.45)
        * plane
        * (0.58 + 0.28 * struct_w + 0.22 * shell_arch)
    )
    dens_soft = plane
    if stellar_density is not None:
        dens_soft = np.clip(np.asarray(stellar_density, dtype=np.float64), 0.0, 1.0)
        if dens_soft.shape != (h, w):
            dens_soft = np.broadcast_to(dens_soft, (h, w))
        # Soft volumetric driver — avoid hard survival-like islands in grade.
        dens_soft = blur_fn(dens_soft, passes=5, periodic_x=periodic_x)

    gold_w = np.zeros((h, w), dtype=np.float64)
    if gold_population is not None:
        gold_w = np.clip(np.asarray(gold_population, dtype=np.float64), 0.0, 1.0)
        if gold_w.shape != (h, w):
            gold_w = np.broadcast_to(gold_w, (h, w))
        gold_w = blur_fn(gold_w, passes=4, periodic_x=periodic_x)
        gold_w = np.clip(0.55 * dens_soft + 0.45 * gold_w, 0.0, 1.0)
        gold_w = gold_w / max(float(np.percentile(gold_w, 99.5)), 1e-6)
        gold_w = np.clip(gold_w, 0.0, 1.0) ** 1.18
    # Razor disk + latitude gate — shoulder latitudes must stay cool/red, not gold.
    from starsky_gen.structure_envelope import midplane_gold_lat_gate

    lat = midplane_gold_lat_gate(h, sigma=0.095, power=1.90)
    if lat.shape != (h, w):
        lat = np.broadcast_to(lat, (h, w))
    gold_w = gold_w * np.clip(plane**3.05, 0.0, 1.0) * lat

    # Soft amber spine; modest extra weight in central ~35% of tightened gold_w.
    nz_g = gold_w > 1e-5
    gold_core = np.zeros_like(gold_w)
    if bool(np.any(nz_g)):
        thr_g = float(np.percentile(gold_w[nz_g], 65.0))
        gold_core = np.clip((gold_w - thr_g) / max(1.0 - thr_g, 0.08), 0.0, 1.0)
    w_core = np.clip(
        lane_arch
        * (0.30 + 0.70 * dens_soft)
        * (0.35 + 0.65 * dust)
        * (0.25 + 0.85 * gold_w)
        * (1.0 + 0.18 * gold_core),
        0.0,
        1.0,
    ) * 0.38
    w_cool = np.clip(shell_arch * rel_clear**0.42 * (1.0 - lane_arch * 0.55), 0.0, 1.0) * 0.55

    # Light contrast + soft blur for intentional red/orange nebula clumps.
    w_hii = np.zeros((h, w), dtype=np.float64)
    red_src = np.clip(gas_struct * dens_soft * (0.45 + 0.55 * dust) * plane, 0.0, 1.0)
    if hii_hint is not None:
        hh = np.clip(np.asarray(hii_hint, dtype=np.float64), 0.0, 1.0)
        if hh.ndim == 3:
            hh = rec709_luma(hh)
        if hh.shape != (h, w):
            hh = np.broadcast_to(hh, (h, w))
        red_src = np.clip(np.maximum(red_src, hh * plane), 0.0, 1.0)
    if structure_survival is not None:
        surv = np.clip(np.asarray(structure_survival, dtype=np.float64), 0.0, 1.35) / 1.35
        if surv.shape == (h, w):
            red_src = red_src * (0.45 + 0.55 * blur_fn(surv, passes=5, periodic_x=periodic_x))
    red_src = red_src * (1.0 - 0.55 * gold_w)  # allow more red next to warm core
    gate = red_src > 1e-5
    if bool(np.any(gate)):
        thr = float(np.percentile(red_src[gate], 70.0))
        clumped = np.clip((red_src - thr) / max(1.0 - thr, 0.08), 0.0, 1.0) ** 1.50
        clumped = blur_fn(clumped, passes=4, periodic_x=periodic_x)
        w_hii = clumped * dens_soft * float(np.clip(hii_strength, 0.0, 0.55)) * 1.55

    w_dust = (
        np.clip(gas_struct * (0.55 + 0.45 * dust) * struct_w * (0.35 + 0.65 * void_w), 0.0, 1.0)
        * 0.52
    )
    # Stronger void windows through the plane.
    w_void = lane_dark * void_w * turb * struct_w * (0.48 + 0.72 * lane_dark) * 0.78

    lane_d = np.clip(lane_dark, 0.0, 1.0)
    dust_rgb = (0.42 * _DUST_BROWN + 0.58 * _DUST_BLACK) * (1.0 - lane_d[..., np.newaxis]) + (
        0.58 * _DUST_BROWN + 0.42 * _DUST_BLACK
    ) * lane_d[..., np.newaxis]
    # Richer red-orange around dense dust; gold amber in core.
    hii_rgb = (
        _HII_RED.reshape(1, 1, 3) * (0.55 + 0.45 * (1.0 - gold_w[..., np.newaxis]))
        + _HII_ORANGE.reshape(1, 1, 3) * (0.35 * (1.0 - gold_w[..., np.newaxis]))
    )
    core_rgb = (
        _GOLD_AMBER.reshape(1, 1, 3) * gold_w[..., np.newaxis]
        + _WARM_CORE.reshape(1, 1, 3) * (1.0 - gold_w[..., np.newaxis]) * 0.65
    )

    w_sum = w_warm + w_core + w_cool + w_hii + w_dust + w_void + 1e-6
    # Prefer soft warm over saturated amber (amber HF reads as gold streak islands).
    gold_mix = 0.24 * _GOLD_OLD + 0.34 * _GOLD_WHITE + 0.42 * _GOLD_AMBER
    tr = (
        w_warm[..., np.newaxis] * gold_mix
        + w_core[..., np.newaxis] * core_rgb
        + w_cool[..., np.newaxis] * _COOL_SHELL
        + w_hii[..., np.newaxis] * hii_rgb
        + w_dust[..., np.newaxis] * dust_rgb
        + w_void[..., np.newaxis] * _DUST_BLACK
    ) / w_sum[..., np.newaxis]

    in_ratio = lin / np.maximum(lu[..., np.newaxis], 1e-6)
    blue_excess = np.clip(lin[..., 2] / np.maximum(lin[..., 0], 1e-6) - 1.05, 0.0, 0.55)
    blend = diffuse_w[..., np.newaxis] * (
        0.62
        + 0.38 * np.clip(lu / 0.40, 0.0, 1.0)[..., np.newaxis]
        + 0.14 * blue_excess[..., np.newaxis]
    )
    graded_ratio = in_ratio * (1.0 - blend) + tr * blend
    out = np.maximum(graded_ratio * lu[..., np.newaxis], 0.0)

    dust_mix = np.clip((w_dust + w_void) / w_sum, 0.0, 1.0) * diffuse_w
    if float(np.max(dust_mix)) > 1e-6:
        dark_lu = np.minimum(lu, lu * (1.0 - dust_mix * (0.34 + 0.24 * struct_w)))
        out = remap_luma_preserving_chroma(out, dark_lu)

    ab = float(np.clip(anti_blue, 0.0, 1.0))
    if ab > 1e-6:
        tone_gate = diffuse_w * luma_gate * (1.0 - np.clip(lane_dark * 0.88 + void_w * 0.28, 0.0, 1.0))
        cool = np.clip(
            out[..., 2] - (out[..., 0] * 0.48 + out[..., 1] * 0.44),
            0.0,
            None,
        )
        out[..., 2] = np.maximum(
            0.0,
            out[..., 2] - cool * ab * tone_gate * 0.72,
        )
        warm_lift = np.clip(
            (out[..., 0] + out[..., 1]) * 0.5 - out[..., 2],
            0.0,
            None,
        )
        out[..., 0] = out[..., 0] + warm_lift * ab * tone_gate * 0.06
        out[..., 1] = out[..., 1] + warm_lift * ab * tone_gate * 0.05
        out[..., 2] = np.maximum(0.0, out[..., 2] - warm_lift * ab * tone_gate * 0.04)

    vt = float(np.clip(void_turbulence, 0.0, 1.0))
    if vt > 1e-6 and rng is not None:
        void_gate = np.clip(w_void * lane_dark * plane, 0.0, 1.0)
        lu_mid = rec709_luma(out)
        ism = np.clip((lu_mid - 0.032) / 0.05, 0.0, 1.0)
        void_gate = void_gate * ism * (1.0 - pointness * 0.85)
        if float(np.max(void_gate)) > 1e-5:
            noise = rng.random((h, w), dtype=np.float64)
            noise = blur_fn(noise, passes=1, periodic_x=periodic_x)
            noise = np.clip(noise * 1.4 - 0.35, 0.0, 1.0)
            crush_w = void_gate * vt * (0.10 + 0.22 * noise)
            crush_lu = np.minimum(lu_mid, lu_mid * (1.0 - crush_w * 0.38))
            crush_lu = np.minimum(crush_lu, 0.02 + lu_mid * (1.0 - crush_w) * 0.12)
            black_rgb = _DUST_BLACK.reshape(1, 1, 3) * crush_lu[..., np.newaxis]
            mix = crush_w[..., np.newaxis]
            out = out * (1.0 - mix) + black_rgb * mix

    if rng is not None and float(gas_fluff_strength) > 1e-6:
        out = _apply_fluffy_micro_gas_texture(
            out,
            gas_struct=gas_struct,
            rel_clear=rel_clear,
            turb=turb,
            plane=plane,
            pointness=pointness,
            blur_fn=blur_fn,
            rng=rng,
            strength=float(gas_fluff_strength) * s,
            periodic_x=periodic_x,
        )

    if rng is not None:
        out = _turbulent_black_dust(
            out,
            ext=ext,
            lane_dark=lane_dark,
            dust=dust,
            void_w=void_w,
            turb=turb,
            plane=plane,
            pointness=pointness,
            strength=float(dust_black_strength) * s,
            blur_fn=blur_fn,
            rng=rng,
            periodic_x=periodic_x,
            xx=xx,
            inner_bias=inner_bias,
        )

    micro_s = float(micro_display_strength) * s
    if micro_s > 1e-6:
        out = apply_band_display_microstructure(
            out,
            ext,
            dw,
            dust_absorption=dust,
            latent_turb=turb,
            morph_gas=gas_texture,
            strength=micro_s,
            periodic_x=periodic_x,
        )

    sep_s = float(separation_strength) * s
    if sep_s > 1e-6:
        out = apply_band_luma_separation(
            out,
            ext,
            dw,
            dust_absorption=dust,
            latent_turb=turb,
            morph_gas=gas_texture,
            strength=sep_s,
            periodic_x=periodic_x,
        )

    # Aggressive void carve: dark windows through the plane (soft, not binary).
    # Keep resolved bright points out of the carve so stars stay crisp.
    void_carve = np.clip(void_w * (0.55 + 0.45 * lane_dark) * plane, 0.0, 1.0)
    void_carve = blur_fn(void_carve, passes=2, periodic_x=periodic_x)
    void_carve = void_carve * (1.0 - star_protect) * (1.0 - bright_keep)
    out = np.maximum(out * (1.0 - 0.42 * void_carve[..., np.newaxis]), 0.0)

    # Midplane multi-scatter: density + gold_population × disk_weight (amber core).
    opaque_core = np.clip((1.0 - ext) * (0.45 + 0.55 * lane_arch) * (0.30 + 0.70 * turb), 0.0, 1.0)
    out = apply_core_multi_scatter_glow(
        out,
        opaque_core=opaque_core,
        disk_w=dw,
        void_mask=void_w,
        strength=float(multi_scatter_strength) * s,
        periodic_x=periodic_x,
        blur_fn=blur_fn,
        density_g=stellar_density,
        structure_survival=structure_survival,
        gold_population=gold_population,
    )

    # Outside gold core, pull back red saturation so the center reads distinctly gold.
    # Diffuse-only: never desaturate resolved bright points.
    red_ex = np.clip(out[..., 0] - np.maximum(out[..., 1], out[..., 2]) * 1.02, 0.0, None)
    desat = diffuse_w * (1.0 - gold_w) * 0.38 * (1.0 - star_protect) * (1.0 - bright_keep)
    out[..., 0] = np.maximum(0.0, out[..., 0] - red_ex * desat)
    out[..., 1] = out[..., 1] + red_ex * desat * 0.22
    out[..., 2] = out[..., 2] + red_ex * desat * 0.08

    # Kill jagged HF gold chroma islands: keep soft warm core, strip streak excess.
    # Also catch shoulder/lane flecks (near-zero stars) that survive as thin horizontal dashes.
    gold_ex = np.clip(
        out[..., 0] - np.maximum(out[..., 1], out[..., 2]) * 1.05,
        0.0,
        None,
    )
    gold_soft = blur_fn(gold_ex, passes=4, periodic_x=periodic_x)
    gold_hf = np.clip(gold_ex - gold_soft * 0.85, 0.0, None)
    lu_now = rec709_luma(out)
    lu_body = np.clip((lu_now - 0.08) / 0.22, 0.0, 1.0)
    hf_gate = (
        (0.35 + 0.65 * diffuse_w)
        * (1.0 - star_protect)
        * (1.0 - bright_keep)
        * np.clip(plane**0.65, 0.0, 1.0)
        * lu_body
    )
    kill = gold_hf * hf_gate * (0.92 * s)
    out[..., 0] = np.maximum(0.0, out[..., 0] - kill)
    out[..., 1] = out[..., 1] + kill * 0.40
    # Replace remaining gold excess with soft midplane warmth (no HF dashes).
    gold_ex2 = np.clip(
        out[..., 0] - np.maximum(out[..., 1], out[..., 2]) * 1.05,
        0.0,
        None,
    )
    replace = np.clip(gold_ex2 - gold_soft, 0.0, None) * hf_gate * 0.70 * s
    out[..., 0] = np.maximum(0.0, out[..., 0] - replace)
    out[..., 1] = out[..., 1] + replace * 0.55
    out[..., 2] = out[..., 2] + replace * 0.18

    # Subtle black lift off-plane only — keep cinematic voids dark (not grey haze).
    lu_out = rec709_luma(out)
    lift = np.clip((0.010 - lu_out) / 0.010, 0.0, 1.0) * (1.0 - plane * 0.78)
    cool_void = np.array([0.006, 0.007, 0.011], dtype=np.float64).reshape(1, 1, 3)
    out = out + lift[..., np.newaxis] * cool_void

    return np.clip(out, 0.0, 1.0).astype(np.float64)
