"""Spectral emission composition, CCM dust attenuation helpers, and Mie-like forward scatter."""

from __future__ import annotations

import numpy as np

from starsky_gen.color_science import ccm_transmission_from_av

# Normalized line + continuum spectra in linear sRGB (Hα red, [O III] green-blue).
_LINE_HALPHA = np.array([1.0, 0.06, 0.22], dtype=np.float64)
_LINE_HALPHA = _LINE_HALPHA / np.max(_LINE_HALPHA)
# Bluish-green [O III] (muted vs Hα to avoid neon cyan in the band).
_LINE_OIII = np.array([0.14, 0.62, 0.52], dtype=np.float64)
_LINE_OIII = _LINE_OIII / np.max(_LINE_OIII)
_LINE_OIII_CONT = np.array([0.55, 0.58, 0.56], dtype=np.float64)
_LINE_SII = np.array([0.82, 0.10, 0.08], dtype=np.float64)
_LINE_SII = _LINE_SII / np.max(_LINE_SII)
_REFL_CONTINUUM = np.array([0.72, 0.78, 0.95], dtype=np.float64)
_WARM_CONTINUUM = np.array([0.88, 0.84, 0.72], dtype=np.float64)


def _blur_rgb_separable(
    rgb: np.ndarray,
    passes: int,
    *,
    periodic_x: bool,
    blur_fn,
) -> np.ndarray:
    return np.stack(
        [blur_fn(rgb[:, :, c], passes=passes, periodic_x=periodic_x) for c in range(3)],
        axis=2,
    )


def build_hii_compact_mask(
    ha_mask: np.ndarray,
    ha_hot: np.ndarray,
    ha_core: np.ndarray,
    band_envelope: np.ndarray,
    streak: np.ndarray,
    clumps: np.ndarray,
    *,
    spot_blobs: np.ndarray | None = None,
) -> np.ndarray:
    """Compact clumpy H II regions (small, not sheet-saturated)."""
    clump_u = np.clip(clumps, 0.0, 1.0)
    mid_var = clump_u * (1.0 - clump_u) * 4.0
    compact = (
        np.clip(ha_hot, 0.0, 1.0) ** 2.2
        * band_envelope
        * np.clip(streak, 0.0, 1.0)
        * (0.38 + 0.62 * ha_core)
        * (0.42 + 0.58 * clump_u)
        * (0.88 + 0.12 * mid_var)
    )
    knots = np.clip(ha_mask, 0.0, 1.0) ** 2.7 * band_envelope * 0.38
    compact = np.clip(compact + knots, 0.0, 1.0)
    if spot_blobs is not None:
        compact = np.clip(compact + np.clip(spot_blobs, 0.0, 1.0) * 0.55, 0.0, 1.0)
    return np.clip(compact, 0.0, 1.0)


def build_reflection_diffuse_mask(
    zone_map: np.ndarray,
    band_envelope: np.ndarray,
    streak: np.ndarray,
    vio_mask: np.ndarray,
    activity_map: np.ndarray,
    *,
    blur_fn,
    periodic_x: bool,
) -> np.ndarray:
    """Smooth reflection-nebula continuum (no sharp H II knots)."""
    smooth = np.clip(zone_map, 0.0, 1.0) * band_envelope * np.clip(streak, 0.0, 1.0)
    smooth = blur_fn(smooth, passes=5, periodic_x=periodic_x)
    smooth = np.clip(smooth**0.92, 0.0, 1.0)
    refl = np.clip(vio_mask * 0.35 + smooth * 0.65, 0.0, 1.0)
    refl *= 0.55 + 0.45 * np.clip(activity_map, 0.0, 1.0)
    return blur_fn(refl, passes=3, periodic_x=periodic_x)


def compose_line_emission_rgb(
    hii_mask: np.ndarray,
    oiii_mask: np.ndarray,
    sii_mask: np.ndarray,
    *,
    halpha_saturation: float = 0.5,
    patch_strength: float = 0.95,
    cloud_gain: float = 1.0,
    emit_cap: float = 0.72,
) -> np.ndarray:
    """Continuum-free emission lines in linear RGB; capped to avoid uniform saturation."""
    ps = float(np.clip(patch_strength, 0.4, 2.0))
    cg = float(np.clip(cloud_gain, 0.5, 2.0))
    sat = float(np.clip(halpha_saturation, 0.2, 1.0))
    ha = _LINE_HALPHA * sat
    ha_neutral = float(np.mean(ha))
    # Blend continuum into emission so Hα does not clump as saturated red patches.
    ha = ha_neutral * (1.0 - 0.55 * sat) + ha * (0.45 + 0.55 * sat)

    hii = np.clip(hii_mask, 0.0, 1.0)
    o3 = np.clip(oiii_mask, 0.0, 1.0) * hii * 0.85
    s2 = np.clip(sii_mask, 0.0, 1.0) * hii * 0.55
    o3_line = _LINE_OIII * 0.55 + _LINE_OIII_CONT * 0.45

    emit = (
        hii[..., np.newaxis] * ha * (0.42 + 0.14 * ps) * cg
        + o3[..., np.newaxis] * o3_line * (0.10 + 0.05 * ps) * cg
        + s2[..., np.newaxis] * _LINE_SII * (0.10 + 0.04 * ps) * cg
    )
    clump_w = np.clip(hii**1.35, 0.0, 1.0)
    warm = 1.0 + 0.06 * clump_w
    emit[..., 0] *= warm
    emit[..., 1] *= 1.0 + 0.02 * clump_w
    emit[..., 2] *= 1.0 - 0.04 * clump_w
    emit = np.clip(emit, 0.0, emit_cap)
    per_ch = np.max(emit, axis=(0, 1), keepdims=True)
    emit = emit / np.maximum(per_ch / emit_cap, 1.0)
    return np.clip(emit, 0.0, emit_cap)


def compose_reflection_continuum_rgb(
    refl_mask: np.ndarray,
    warm_mix: np.ndarray | None = None,
    *,
    strength: float = 1.0,
    cap: float = 0.55,
) -> np.ndarray:
    """Diffuse reflection nebula continuum (smooth, bluish)."""
    m = np.clip(refl_mask, 0.0, 1.0)
    rgb = m[..., np.newaxis] * _REFL_CONTINUUM * float(strength)
    if warm_mix is not None:
        wm = np.clip(warm_mix, 0.0, 1.0)[..., np.newaxis]
        rgb = rgb * (1.0 - 0.22 * wm) + wm * _WARM_CONTINUUM * 0.18 * float(strength)
    return np.clip(rgb, 0.0, cap)


def av_map_from_transmission(ext: np.ndarray, *, av_scale: float) -> np.ndarray:
    """Scalar transmission ext in [floor,1] → visual extinction A_V map."""
    t = np.clip(np.asarray(ext, dtype=np.float64), 1e-4, 1.0)
    return np.clip(-2.5 * np.log10(t), 0.0, 1.0) * float(max(av_scale, 0.0))


def extinction_av_scale_for_lane_depth(
    *,
    transmission_floor: float = 0.14,
    lane_mag_at_floor: float = 1.5,
) -> float:
    """Scale A_V so darkest transmission (lane floor) ≈ ``lane_mag_at_floor`` magnitudes (0.5–2 typical)."""
    t = float(np.clip(transmission_floor, 1e-4, 1.0))
    raw_av = -2.5 * float(np.log10(t))
    return float(lane_mag_at_floor) / max(raw_av, 1e-6)


def apply_ccm_extinction_linear(
    rgb: np.ndarray,
    ext: np.ndarray,
    *,
    av_scale: float,
    rv: float = 3.1,
) -> np.ndarray:
    """Multiply linear RGB by CCM transmission derived from extinction/transmission field."""
    trans = ccm_transmission_from_av(av_map_from_transmission(ext, av_scale=av_scale), rv=rv)
    return np.maximum(np.asarray(rgb, dtype=np.float64) * trans, 0.0)


def henyey_greenstein(mu: np.ndarray | float, g: float) -> np.ndarray | float:
    """Henyey–Greenstein phase function P(cos θ); ``mu`` is cos(angle to forward axis)."""
    g = float(g)
    mu_arr = np.asarray(mu, dtype=np.float64)
    denom = (1.0 + g * g - 2.0 * g * mu_arr) ** 1.5
    p = (1.0 - g * g) / (4.0 * np.pi * np.maximum(denom, 1e-12))
    if np.isscalar(mu):
        return float(p)
    return p


def _shift_field_y(
    field: np.ndarray,
    dy_px: float,
    *,
    periodic_x: bool,
) -> np.ndarray:
    """Fractional vertical shift (positive dy moves content down)."""
    f = np.asarray(field, dtype=np.float64)
    h, w = f.shape[:2]
    if abs(dy_px) < 1e-6:
        return f.copy()
    yy = np.arange(h, dtype=np.float64)[:, None]
    xx = np.arange(w, dtype=np.float64)[None, :]
    src_y = yy - dy_px
    y0 = np.floor(src_y).astype(np.int64)
    ty = src_y - y0
    y1 = np.clip(y0 + 1, 0, h - 1)
    y0 = np.clip(y0, 0, h - 1)
    if periodic_x:
        x0 = np.mod(xx.astype(np.int64), w)
        x1 = (x0 + 1) % w
        tx = xx - np.floor(xx)
    else:
        x0 = np.clip(np.floor(xx).astype(np.int64), 0, w - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        tx = xx - x0
    if f.ndim == 3:
        z00 = f[y0, x0]
        z01 = f[y0, x1]
        z10 = f[y1, x0]
        z11 = f[y1, x1]
        zb0 = z00 * (1.0 - tx) + z01 * tx
        zb1 = z10 * (1.0 - tx) + z11 * tx
        return zb0 * (1.0 - ty) + zb1 * ty
    z00 = f[y0, x0]
    z01 = f[y0, x1]
    z10 = f[y1, x0]
    z11 = f[y1, x1]
    zb0 = z00 * (1.0 - tx) + z01 * tx
    zb1 = z10 * (1.0 - tx) + z11 * tx
    return zb0 * (1.0 - ty) + zb1 * ty


def integrate_volume_scatter_hg(
    rgb: np.ndarray,
    emission: np.ndarray,
    *,
    optical_depth: np.ndarray | None = None,
    albedo: float = 0.62,
    g_forward: float = 0.68,
    g_back: float = -0.32,
    steps: int = 4,
    step_px: float = 2.4,
    strength: float = 0.08,
    periodic_x: bool = True,
    blur_fn,
) -> np.ndarray:
    """Short-path single-scatter integration with dual-lobe Henyey–Greenstein phase."""
    s = float(np.clip(strength, 0.0, 0.45))
    if s < 1e-6:
        return rgb
    lin = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, None)
    emit = np.clip(np.asarray(emission, dtype=np.float64), 0.0, None)
    if emit.ndim == 3:
        emit = np.mean(emit, axis=2)
    gate = np.clip((emit - 0.10) / 0.52, 0.0, 1.0) ** 1.28
    if float(np.max(gate)) < 1e-8:
        return lin
    tau = np.clip(optical_depth, 0.0, 4.0) if optical_depth is not None else np.full_like(emit, 0.35)
    sigma_s = float(np.clip(albedo, 0.15, 0.95))
    acc = np.zeros_like(lin)
    n_steps = int(np.clip(steps, 1, 8))
    sp = float(max(step_px, 0.5))
    # Importance weights: forward lobe along −Y (light behind plane scattering toward viewer).
    mu_fwd = np.linspace(0.55, 0.95, n_steps, dtype=np.float64)
    mu_back = np.linspace(-0.85, -0.35, n_steps, dtype=np.float64)
    w_fwd = henyey_greenstein(mu_fwd, g_forward)
    w_back = henyey_greenstein(mu_back, g_back)
    w_fwd = w_fwd / (np.sum(w_fwd) + 1e-12)
    w_back = w_back / (np.sum(w_back) + 1e-12)
    src = lin * gate[..., np.newaxis]
    emit_w = np.clip(emit * gate, 0.0, None)
    for k in range(n_steps):
        path = float(k + 1)
        trans = np.exp(-tau * path * 0.22)
        # Incident radiance along path (shifted emission drives forward/back lobes).
        e_fwd = _shift_field_y(emit_w, -sp * path, periodic_x=periodic_x)
        e_back = _shift_field_y(emit_w, sp * path * 0.45, periodic_x=periodic_x)
        path_w = (0.55 + 0.45 * e_fwd + 0.28 * e_back)[..., np.newaxis]
        blur_p_f = max(2, int(round(sp * path * 0.85)))
        blur_p_b = max(1, int(round(sp * path * 0.55)))
        sc_f = _blur_rgb_separable(src * path_w, blur_p_f, periodic_x=periodic_x, blur_fn=blur_fn)
        sc_b = _blur_rgb_separable(src * (0.75 + 0.25 * e_back[..., np.newaxis]), blur_p_b, periodic_x=periodic_x, blur_fn=blur_fn)
        acc += (sc_f * float(w_fwd[k]) + sc_b * float(w_back[k])) * trans[..., np.newaxis]
    warm_fwd = np.array([1.03, 0.99, 0.94], dtype=np.float64)
    cool_back = np.array([0.97, 0.99, 1.02], dtype=np.float64)
    halo = acc * sigma_s * s
    halo[..., 0] *= 0.58 * warm_fwd[0] + 0.42 * cool_back[0]
    halo[..., 1] *= 0.58 * warm_fwd[1] + 0.42 * cool_back[1]
    halo[..., 2] *= 0.58 * warm_fwd[2] + 0.42 * cool_back[2]
    soft = _blur_rgb_separable(halo, 2, periodic_x=periodic_x, blur_fn=blur_fn)
    return np.clip(lin + halo * 0.72 + soft * 0.28, 0.0, None)


def forward_scatter_hg_volume(
    rgb: np.ndarray,
    hot_luma: np.ndarray | None = None,
    *,
    optical_depth: np.ndarray | None = None,
    strength: float = 0.07,
    g_forward: float = 0.68,
    g_back: float = -0.32,
    steps: int = 4,
    periodic_x: bool = True,
    blur_fn,
) -> np.ndarray:
    """Physically motivated volumetric scatter (HG dual-lobe short-path integration)."""
    lin = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, None)
    if hot_luma is None:
        hot_luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    if optical_depth is None and hot_luma is not None:
        optical_depth = np.clip(1.0 - hot_luma, 0.08, 1.0) * 0.85
    return integrate_volume_scatter_hg(
        lin,
        hot_luma,
        optical_depth=optical_depth,
        g_forward=g_forward,
        g_back=g_back,
        steps=steps,
        strength=strength,
        periodic_x=periodic_x,
        blur_fn=blur_fn,
    )


def dust_lane_multiscatter_fill(
    rgb: np.ndarray,
    ext: np.ndarray,
    ambient_luma: np.ndarray,
    *,
    strength: float = 0.05,
    periodic_x: bool = True,
    blur_fn,
    plane_gate: np.ndarray | None = None,
) -> np.ndarray:
    """Low-cost 2nd-order scatter: bleed ambient/nebula light back into thick dust lanes."""
    s = float(np.clip(strength, 0.0, 0.2))
    if s < 1e-6:
        return rgb
    lin = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, None)
    t = np.clip(1.0 - np.asarray(ext, dtype=np.float64), 0.0, 1.0)
    if plane_gate is not None:
        pg = np.clip(np.asarray(plane_gate, dtype=np.float64), 0.0, 1.0)
        t = t * pg
    thick = np.clip(t**1.05, 0.0, 1.0)
    if float(np.max(thick)) < 1e-8:
        return lin
    amb = np.clip(np.asarray(ambient_luma, dtype=np.float64), 0.0, None)
    if plane_gate is not None:
        amb = amb * pg
    # Screen-space: wide ambient halo sampled into lane interiors.
    amb_wide = blur_fn(amb, passes=5, periodic_x=periodic_x)
    amb_mid = blur_fn(amb, passes=2, periodic_x=periodic_x)
    scatter_l = np.clip(amb_wide * 0.62 + amb_mid * 0.38, 0.0, None)
    warm = np.array([0.22, 0.17, 0.12], dtype=np.float64)
    cool = np.array([0.11, 0.12, 0.14], dtype=np.float64)
    tint = warm * (1.0 - thick[..., None]) + cool * thick[..., None]
    fill = scatter_l[..., None] * tint * thick[..., None] * s
    # Edge-weighted second pass along lane rims (forward scatter into trenches).
    edge = np.clip(blur_fn(t, passes=2, periodic_x=periodic_x) - t, 0.0, 1.0)
    rim = blur_fn(edge * amb, passes=2, periodic_x=periodic_x)
    fill = fill + rim[..., None] * warm * (s * 0.42)
    return np.clip(lin + fill, 0.0, None)


def forward_scatter_mie(
    rgb: np.ndarray,
    hot_luma: np.ndarray | None = None,
    *,
    strength: float = 0.07,
    periodic_x: bool = True,
    blur_fn,
    optical_depth: np.ndarray | None = None,
    g_forward: float = 0.68,
    g_back: float = -0.32,
    steps: int = 4,
) -> np.ndarray:
    """Volumetric forward/back scatter (HG); kept name for call-site compatibility."""
    return forward_scatter_hg_volume(
        rgb,
        hot_luma,
        optical_depth=optical_depth,
        strength=strength,
        g_forward=g_forward,
        g_back=g_back,
        steps=steps,
        periodic_x=periodic_x,
        blur_fn=blur_fn,
    )
