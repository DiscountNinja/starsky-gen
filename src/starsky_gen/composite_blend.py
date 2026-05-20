"""Linear-space compositing: additive gas, emission screen cores, luma-preserving tone helpers."""

from __future__ import annotations

import numpy as np

from starsky_gen.color_science import rec709_luma, remap_luma_preserving_chroma
from starsky_gen.hdr import HDR_DTYPE, ensure_hdr


def percentile_asinh_luma_stretch(
    luma: np.ndarray,
    *,
    stretch_gain: float,
    p_lo: float = 1.5,
    p_hi: float = 99.4,
    filmic_shoulder: float = 0.16,
) -> np.ndarray:
    """Percentile normalize + asinh on luminance only (not RGB)."""
    lu = np.maximum(np.asarray(luma, dtype=np.float64), 0.0)
    positive = lu[lu > 1e-12]
    lo = float(np.percentile(positive, p_lo)) if positive.size else 0.0
    hi = float(np.percentile(positive, p_hi)) if positive.size else 1.0
    span = max(hi - lo, 1e-8)
    norm = np.clip((lu - lo) / span, 0.0, 1.0)
    g = max(float(stretch_gain), 1.0)
    l_disp = np.arcsinh(norm * g) / np.arcsinh(g)
    l_disp = l_disp / (1.0 + filmic_shoulder * l_disp**2)
    hot = np.clip((lu - hi * 0.62) / max(hi * 0.38, 1e-8), 0.0, 1.0)
    l_disp = np.maximum(l_disp, hot * 0.56)
    return l_disp


def normalize_star_stack_luma_preserve_chroma(
    stars: np.ndarray,
    *,
    percentile: float = 98.5,
    target: float = 0.82,
) -> np.ndarray:
    """Gentle global luma normalize on the star layer (chroma preserved)."""
    s = np.maximum(stars.astype(np.float64), 0.0)
    lu = rec709_luma(s)
    positive = lu[lu > 1e-12]
    if positive.size < 16:
        return s.astype(HDR_DTYPE)
    ref = float(np.percentile(positive, float(np.clip(percentile, 90.0, 99.5))))
    scale = float(target) / max(ref, 1e-8)
    scale = float(np.clip(scale, 0.55, 1.35))
    return remap_luma_preserving_chroma(s, lu * scale).astype(HDR_DTYPE)


def stars_hdr_to_display(
    stars: np.ndarray,
    *,
    stretch_gain: float = 11.0,
    output_gain: float = 1.10,
    emit_cap: float = 1.0,
    peak_percentile: float = 99.75,
    faint_desat: float = 0.0,
    bright_chroma_desat: float = 0.0,
) -> np.ndarray:
    """Map HDR star layer to display with chroma-safe percentile asinh."""
    s = np.maximum(stars.astype(np.float64), 0.0)
    g = float(stretch_gain)
    if g <= 0.0:
        g = auto_star_display_stretch_gain(s, peak_percentile=peak_percentile, target_peak=0.98)
    lu = rec709_luma(s)
    l_disp = percentile_asinh_luma_stretch(
        lu,
        stretch_gain=g,
        p_hi=float(np.clip(peak_percentile, 90.0, 99.95)),
    )
    out = remap_luma_preserving_chroma(s, l_disp)
    out = np.clip(out * output_gain, 0.0, emit_cap)
    desat = float(np.clip(faint_desat, 0.0, 0.25))
    if desat > 1e-6:
        lu_out = rec709_luma(out)
        faint_w = np.clip(1.0 - lu_out / 0.38, 0.0, 1.0) ** 1.25
        gray = lu_out[..., np.newaxis]
        out = out * (1.0 - faint_w[..., np.newaxis] * desat) + gray * (faint_w[..., np.newaxis] * desat)
    bright_desat = float(np.clip(bright_chroma_desat, 0.0, 0.2))
    if bright_desat > 1e-6:
        lu_out = rec709_luma(out)
        spec_w = np.clip((lu_out - 0.76) / 0.22, 0.0, 1.0) ** 1.12
        if float(np.max(spec_w)) > 1e-8:
            gray = lu_out[..., np.newaxis]
            out = out * (1.0 - spec_w[..., np.newaxis] * bright_desat) + gray * (
                spec_w[..., np.newaxis] * bright_desat
            )
    return np.clip(out, 0.0, emit_cap).astype(HDR_DTYPE)


def soft_knee_star_peaks(stars: np.ndarray, *, strength: float, percentile: float = 99.95) -> np.ndarray:
    """Gentle compression on star display peaks before add-max over nebula."""
    s = float(np.clip(strength, 0.0, 0.65))
    if s < 1e-6:
        return stars
    lu = rec709_luma(stars)
    positive = lu[lu > 1e-8]
    if positive.size < 32:
        return stars
    cap = float(np.percentile(positive, float(np.clip(percentile, 99.5, 99.98))))
    knee = cap * 0.86
    hot = np.clip((lu - knee) / max(cap - knee, 1e-6), 0.0, 1.0) ** 1.15
    scale = 1.0 - hot * s * 0.42
    return remap_luma_preserving_chroma(stars, lu * scale).astype(HDR_DTYPE)


def auto_star_display_stretch_gain(
    stars: np.ndarray,
    *,
    peak_percentile: float = 99.75,
    target_peak: float = 0.98,
    gain_lo: float = 4.0,
    gain_hi: float = 28.0,
) -> float:
    """Pick percentile-asinh gain so the top ``peak_percentile`` of star luma ≈ display white."""
    s = np.maximum(stars.astype(np.float64), 0.0)
    lu = rec709_luma(s)
    positive = lu[lu > 1e-12]
    if positive.size < 8:
        return 10.0

    def peak_out(g: float) -> float:
        out = stars_hdr_to_display(s, stretch_gain=g, output_gain=1.0)
        return float(np.percentile(rec709_luma(out), peak_percentile))

    g_lo, g_hi = float(gain_lo), float(gain_hi)
    p_lo, p_hi = peak_out(g_lo), peak_out(g_hi)
    if p_hi < target_peak:
        return g_hi
    if p_lo > target_peak:
        return g_lo
    for _ in range(18):
        g_mid = 0.5 * (g_lo + g_hi)
        if peak_out(g_mid) > target_peak:
            g_hi = g_mid
        else:
            g_lo = g_mid
    return 0.5 * (g_lo + g_hi)


def recompute_hdr_asinh_gain(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    base_gain: float,
    *,
    ref_percentile: float = 99.4,
    ref_linear: float = 0.06,
    gain_min: float = 0.02,
    gain_max: float = 0.22,
) -> float:
    """Boost linear asinh gain when post-extinction canvas is dim — preserves faint nebula detail."""
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    h, w_img = lin.shape[:2]
    w = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if w.ndim == 1:
        w = w[:, None]
    if w.shape[0] != h or w.shape[1] != w_img:
        w = np.broadcast_to(w, (h, w_img))
    lu = rec709_luma(lin)
    mask = w > 0.12
    sample = lu[mask] if np.any(mask) else lu.ravel()
    positive = sample[sample > 1e-10]
    if positive.size == 0:
        return float(base_gain)
    ref = float(np.percentile(positive, ref_percentile))
    rel = float(ref_linear / max(ref, 1e-6))
    if ref > 0.32:
        rel *= (0.32 / ref) ** 0.85
    return float(np.clip(float(base_gain) * rel, gain_min, gain_max))


def composite_stars_over_display_canvas(
    canvas: np.ndarray,
    star_disp: np.ndarray,
    band_gate: np.ndarray,
    *,
    add_scale: float = 0.85,
    band_chroma_desat: float = 0.22,
    band_brightness_scale: float = 0.72,
    band_chroma_adapt: float = 0.38,
    max_blend_in_band: float = 0.45,
    peak_soft_clamp: float = 0.32,
) -> np.ndarray:
    """Add tone-mapped stars over graded nebula without a blue fringe on the disk.

    Stars-only or nebula-only passes look fine; the artifact is the *combination* — cool
    catalog stars stacked on a warm/emitting canvas. Desaturate, dim, and pull star chroma
    toward the local canvas color in the bright plane.
    """
    base = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    stars = np.maximum(np.asarray(star_disp, dtype=np.float64), 0.0)
    gate = np.clip(np.asarray(band_gate, dtype=np.float64), 0.0, 1.0)
    if gate.ndim == 1:
        gate = gate[:, None]
    h, w_img = base.shape[:2]
    gate = np.broadcast_to(gate, (h, w_img))

    lu_s = rec709_luma(stars)
    lu_b = rec709_luma(base)
    gray = lu_s[..., np.newaxis]
    desat = float(np.clip(band_chroma_desat, 0.0, 0.45))
    stars_adj = stars * (1.0 - gate[..., np.newaxis] * desat) + gray * (gate[..., np.newaxis] * desat)
    bright = float(np.clip(band_brightness_scale, 0.35, 1.0))
    stars_adj = stars_adj * (1.0 - gate[..., np.newaxis] * (1.0 - bright))

    adapt = float(np.clip(band_chroma_adapt, 0.0, 0.65))
    if adapt > 1e-6:
        w = gate[..., np.newaxis] * adapt
        denom = np.maximum(lu_b, 1e-6)[..., np.newaxis]
        canvas_chroma = base / denom
        canvas_chroma = canvas_chroma / (
            np.maximum(np.max(canvas_chroma, axis=2, keepdims=True), 1e-6)
        )
        target = canvas_chroma * lu_s[..., np.newaxis]
        stars_adj = stars_adj * (1.0 - w) + target * w

    stars_adj = soft_knee_star_peaks(stars_adj, strength=peak_soft_clamp)

    add_s = float(np.clip(add_scale, 0.0, 1.0))
    out = np.clip(base + stars_adj * add_s, 0.0, 1.0)

    mb = float(np.clip(max_blend_in_band, 0.0, 1.0))
    peak_w = (1.0 - mb * gate)[..., np.newaxis]
    out = np.maximum(out, base * peak_w + stars_adj * (1.0 - peak_w))
    return np.clip(out, 0.0, 1.0)


def composite_add_gas(
    canvas: np.ndarray,
    gas: np.ndarray,
    visibility: np.ndarray | None = None,
) -> np.ndarray:
    """Additive continuum in linear HDR (visibility modulates radiance, not Porter–Duff α)."""
    base = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    layer = np.maximum(np.asarray(gas, dtype=np.float64), 0.0)
    if visibility is not None:
        vis = np.clip(np.asarray(visibility, dtype=np.float64), 0.0, 1.0)
        if vis.ndim == 2:
            vis = vis[..., np.newaxis]
        layer = layer * vis
    return ensure_hdr(base + layer)


def composite_emission_add_screen(
    canvas: np.ndarray,
    emission: np.ndarray,
    hot_mask: np.ndarray,
    *,
    add_strength: float = 1.0,
    core_screen_mix: float = 0.72,
    core_threshold: float = 0.22,
) -> np.ndarray:
    """Additive emission; bright cores use screen blend (physically glowing gas, not α-over).

    Future option for hot cores: linear add then per-channel maximum, e.g.
    ``np.maximum(base + emit, base)`` (or ``np.maximum(added, emit)``), instead of
    screen — can punch H II peaks harder without α-over; tune vs screen in previews.
    """
    base = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    emit = np.maximum(np.asarray(emission, dtype=np.float64), 0.0) * float(add_strength)
    added = base + emit
    # HDR screen: 1 - (1-a)(1-b)  (see docstring for add+maximum alternative)
    screen = 1.0 - (1.0 - base) * (1.0 - emit)
    hot = np.clip(np.asarray(hot_mask, dtype=np.float64), 0.0, 1.0)
    if hot.ndim == 2:
        hot = hot[..., np.newaxis]
    core = np.clip((hot - core_threshold) / max(1.0 - core_threshold, 1e-6), 0.0, 1.0) ** 0.85
    mix = core * float(core_screen_mix)
    out = added * (1.0 - mix) + np.maximum(added, screen) * mix
    return ensure_hdr(np.maximum(out, base))


def composite_emission_chroma_preserve(
    canvas: np.ndarray,
    emission: np.ndarray,
    *,
    strength: float = 1.0,
    core_screen_mix: float = 0.78,
) -> np.ndarray:
    """Add emission without luma-only remapping (for off-band H II after plane grade)."""
    emit = np.maximum(np.asarray(emission, dtype=np.float64), 0.0)
    hot = rec709_luma(emit)
    return composite_emission_add_screen(
        canvas,
        emit,
        hot,
        add_strength=strength,
        core_screen_mix=core_screen_mix,
        core_threshold=0.10,
    )


def reinhard_luma_preserving(rgb: np.ndarray, *, k: float = 0.52) -> np.ndarray:
    """Global Reinhard on luminance with chroma reprojection."""
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    l_new = lu / (1.0 + float(k) * lu)
    return remap_luma_preserving_chroma(lin, l_new)
