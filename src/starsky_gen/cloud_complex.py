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


def _contrast_stretch(
    field: np.ndarray,
    *,
    lo_pct: float = 20.0,
    hi_pct: float = 96.0,
    power: float = 0.85,
) -> np.ndarray:
    """Normalize to [0,1] between percentiles, then apply a contrast power."""
    a = np.asarray(field, dtype=np.float64)
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    out = np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.clip(out ** float(power), 0.0, 1.0)


def _downsample_blur2d(
    field: np.ndarray,
    *,
    scale: float = 0.38,
    blur_sigma: float = 1.35,
    periodic_x: bool = True,
) -> np.ndarray:
    """Downsample → soft blur → upsample so fine noise reads as soft cloud mass."""
    a = np.asarray(field, dtype=np.float64)
    h, w = int(a.shape[0]), int(a.shape[1])
    hs = max(8, int(round(h * float(scale))))
    ws = max(8, int(round(w * float(scale))))
    small = _resize_bilinear(a, hs, ws, periodic_x=periodic_x)
    small = _gauss_float(small, blur_sigma, periodic_x=periodic_x)
    return _resize_bilinear(np.clip(small, 0.0, 1.0), h, w, periodic_x=periodic_x)


# Primary hero targets. Short aliases (red/gold/blue/magenta) map to long names.
# ``red`` / ``classic-red`` match display hex #FF1818 (same soft path as custom).
NEBULA_HERO_COLOR_PRESETS: dict[str, tuple[float, float, float]] = {
    "classic-red": (1.00, 24.0 / 255.0, 24.0 / 255.0),
    "red": (1.00, 24.0 / 255.0, 24.0 / 255.0),
    "gold-core": (1.00, 0.72, 0.28),
    "gold": (1.00, 0.72, 0.28),
    "cool-blue": (0.32, 0.72, 1.00),
    "blue": (0.32, 0.72, 1.00),
    "magenta-accent": (0.95, 0.28, 0.82),
    "magenta": (0.95, 0.28, 0.82),
    "neutral": (0.92, 0.90, 0.88),
}

# Canonical key for each alias (used for variation siblings / logging).
_NEBULA_HERO_CANONICAL: dict[str, str] = {
    "classic-red": "red",
    "red": "red",
    "gold-core": "gold",
    "gold": "gold",
    "cool-blue": "blue",
    "blue": "blue",
    "magenta-accent": "magenta",
    "magenta": "magenta",
    "neutral": "neutral",
}

# Sibling hues for sparse local variation around the primary target.
# Red (#FF1818 family): warm amber cores + deeper maroon skirts — no magenta wash.
_NEBULA_VARIATION_SIBLINGS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "red": ((1.00, 0.28, 0.12), (0.82, 0.10, 0.10)),
    "gold": ((1.00, 0.45, 0.25), (0.55, 0.70, 0.95)),
    "blue": ((0.55, 0.90, 1.00), (0.75, 0.45, 0.95)),
    "magenta": ((1.00, 0.40, 0.55), (0.40, 0.55, 1.00)),
    "neutral": ((1.00, 0.85, 0.70), (0.70, 0.78, 0.95)),
    "custom": ((1.00, 0.65, 0.30), (0.45, 0.60, 1.00)),
}


def _normalize_hero_rgb(rgb: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    arr = np.clip(np.asarray(rgb, dtype=np.float64).reshape(3), 0.0, None)
    peak = float(np.max(arr)) + 1e-8
    return (arr / peak).astype(np.float64)


def _chroma_from_hero(hero_rgb: np.ndarray) -> np.ndarray:
    """Peak-normalized hero → Rec.709-luma-unit chroma (preserves brightness on tint)."""
    primary = np.asarray(hero_rgb, dtype=np.float64).reshape(3)
    primary = primary / (float(np.max(primary)) + 1e-8)
    lu = float(0.2126 * primary[0] + 0.7152 * primary[1] + 0.0722 * primary[2]) + 1e-8
    return (primary / lu).astype(np.float64)


def _tint_preserve_luma(luma: np.ndarray, chroma_rgb: np.ndarray) -> np.ndarray:
    """Apply chroma so Rec.709 luma stays ``luma``."""
    c = np.asarray(chroma_rgb, dtype=np.float64).reshape(3)
    return c.reshape(1, 1, 3) * np.asarray(luma, dtype=np.float64)[..., None]


def parse_nebula_hero_color(value: str | None) -> tuple[np.ndarray, str]:
    """Parse ``--nebula-hero-color`` into (rgb[3], canonical_key).

    Accepts:
    - Presets: ``classic-red``/``red``, ``gold-core``/``gold``, ``cool-blue``/``blue``,
      ``magenta-accent``/``magenta``, ``neutral``
    - Hex: ``#FC5656``, ``FC5656``, or ``0xFC5656`` (bash: quote ``'#FC5656'`` or omit ``#``)
    - RGB: ``R,G,B`` (0–1 or 0–255)
    """
    if value is None or not str(value).strip():
        rgb = _normalize_hero_rgb(NEBULA_HERO_COLOR_PRESETS["magenta"])
        return rgb, "magenta"
    s = str(value).strip()
    key = s.lower().replace("_", "-")

    if key in NEBULA_HERO_COLOR_PRESETS:
        canon = _NEBULA_HERO_CANONICAL.get(key, key)
        return _normalize_hero_rgb(NEBULA_HERO_COLOR_PRESETS[key]), canon

    # Hex: #RRGGBB, 0xRRGGBB, or RRGGBB (unquoted # is a bash comment — prefer quote/omit).
    hex_s = s
    if hex_s.lower().startswith("0x"):
        hex_s = hex_s[2:]
    elif hex_s.startswith("#"):
        hex_s = hex_s[1:]
    if len(hex_s) == 6 and all(c in "0123456789abcdefABCDEF" for c in hex_s):
        r = int(hex_s[0:2], 16) / 255.0
        g = int(hex_s[2:4], 16) / 255.0
        b = int(hex_s[4:6], 16) / 255.0
        rgb = _normalize_hero_rgb((r, g, b))
        # Exact preset match → canonical key (so #FF1818 == red finish siblings).
        for pname, prgb in NEBULA_HERO_COLOR_PRESETS.items():
            if np.allclose(rgb, _normalize_hero_rgb(prgb), atol=1e-4):
                return rgb, _NEBULA_HERO_CANONICAL.get(pname, pname)
        return rgb, "custom"

    # RGB triple
    if "," in s:
        parts = [p.strip() for p in s.replace(" ", "").split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"nebula hero color RGB must be R,G,B (got {value!r}); "
                "or use a preset / hex (#RRGGBB, RRGGBB, or 0xRRGGBB)"
            )
        try:
            nums = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"nebula hero color RGB must be numeric (got {value!r})") from exc
        if max(nums) > 1.0 + 1e-9:
            nums = [n / 255.0 for n in nums]
        return _normalize_hero_rgb(tuple(nums)), "custom"

    raise ValueError(
        f"unknown nebula hero color {value!r}; use red|gold|blue|magenta|neutral "
        "(or classic-red|gold-core|cool-blue|magenta-accent), hex "
        "(#RRGGBB / RRGGBB / 0xRRGGBB; quote '#…' in bash), or R,G,B"
    )


def resolve_nebula_hero_rgb(
    *,
    preset: str = "magenta",
    custom_rgb: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, str]:
    """Return (rgb[3], canonical_key). Prefer ``parse_nebula_hero_color`` for CLI strings."""
    if custom_rgb is not None:
        return _normalize_hero_rgb(custom_rgb), "custom"
    return parse_nebula_hero_color(preset)


def parse_hero_rgb_string(value: str | None) -> tuple[float, float, float] | None:
    """Back-compat: parse RGB / hex / preset into a float RGB tuple."""
    if value is None or not str(value).strip():
        return None
    rgb, _ = parse_nebula_hero_color(value)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def _apply_hero_color_and_variation(
    rgb: np.ndarray,
    gate: np.ndarray,
    rng: np.random.Generator,
    *,
    hero_rgb: np.ndarray,
    preset_key: str = "magenta",
    amount: float = 0.62,
    variation: float = 0.38,
    periodic_x: bool = True,
) -> np.ndarray:
    """Tint toward primary hero color; sparse gold/cool only in a few dense cores."""
    em = np.asarray(rgb, dtype=np.float64)
    luma = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    p95 = float(np.percentile(luma, 95)) + 1e-8
    dens = np.clip(luma / p95, 0.0, 1.0)
    dens = dens * (0.35 + 0.65 * np.clip(gate, 0.0, 1.0))
    primary = _chroma_from_hero(hero_rgb)
    sib_a, sib_b = _NEBULA_VARIATION_SIBLINGS.get(
        preset_key, _NEBULA_VARIATION_SIBLINGS["custom"]
    )
    gold = _chroma_from_hero(sib_a)
    cool = _chroma_from_hero(sib_b)
    # Soft spatial mix field — thresholded so only a few knots pick up siblings.
    mix = _filtered_white_fbm(
        rng,
        int(em.shape[0]),
        int(em.shape[1]),
        octaves=3,
        base_sigma=max(10.0, 0.035 * float(em.shape[1])),
        periodic_x=periodic_x,
    )
    mix = _gauss_float(mix, 2.2, periodic_x=periodic_x)
    mix = np.clip((mix - 0.55) / 0.35, 0.0, 1.0) ** 1.35
    # Sparse core pockets only (bulk stays on the hero preset).
    core_pocket = np.clip(dens**1.45 * mix, 0.0, 1.0)
    rim = np.clip(dens**1.10 * (1.0 - mix) * np.clip((dens - 0.35) / 0.45, 0.0, 1.0), 0.0, 1.0)
    local = (
        primary.reshape(1, 1, 3)
        + (gold.reshape(1, 1, 3) - primary.reshape(1, 1, 3))
        * (float(variation) * 0.70 * core_pocket[..., None])
        + (cool.reshape(1, 1, 3) - primary.reshape(1, 1, 3))
        * (float(variation) * 0.35 * rim[..., None])
    )
    local = np.clip(local, 0.0, None)
    # Preserve luma while replacing chroma toward local target.
    tinted = local * luma[..., None]
    amt = float(amount) * dens
    out = em * (1.0 - amt[..., None]) + tinted * amt[..., None]
    # Extra saturation in densest cores — keep hero hue from washing to white.
    gray = luma[..., None]
    core_sat = np.clip((dens - 0.28) / 0.72, 0.0, 1.0)[..., None]
    out = out + (out - gray) * core_sat * 0.85
    return np.clip(out, 0.0, 1.0)


def _tighten_emission_hue(
    rgb: np.ndarray,
    gate: np.ndarray,
    hero_rgb: np.ndarray,
    *,
    strength: float = 0.55,
    sat_cap: float = 0.88,
    preset_key: str = "magenta",
) -> np.ndarray:
    """Soft hue pull + saturation clamp; kills magenta/pink outliers toward hero."""
    em = np.asarray(rgb, dtype=np.float64)
    luma = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    p95 = float(np.percentile(luma, 95)) + 1e-8
    dens = np.clip(luma / p95, 0.0, 1.0)
    dens = dens * (0.25 + 0.75 * np.clip(gate, 0.0, 1.0))
    chroma = _chroma_from_hero(hero_rgb)
    s = float(np.clip(strength, 0.0, 1.0))
    amt = s * np.clip(dens**0.70, 0.0, 1.0)
    tinted = _tint_preserve_luma(luma, chroma)
    out = em * (1.0 - amt[..., None]) + tinted * amt[..., None]
    # Soft saturation clamp (avoid neon blowout / pastel pink wash).
    peak = np.max(out, axis=2, keepdims=True) + 1e-8
    floor = np.min(out, axis=2, keepdims=True)
    sat = np.clip((peak - floor) / peak, 0.0, 1.0)
    cap = float(np.clip(sat_cap, 0.35, 1.0))
    over = np.maximum(sat - cap, 0.0)
    if float(np.max(over)) > 1e-6:
        pull = over / (sat + 1e-8)
        gray = luma[..., None]
        out = out * (1.0 - 0.55 * pull) + gray * (0.55 * pull)
    _ = preset_key  # shared finish path for all presets (incl. red / custom hex).
    return np.clip(out, 0.0, 1.0)


def _soften_core_falloff(
    rgb: np.ndarray,
    *,
    power: float = 0.82,
    blur_sigma: float = 1.65,
    mix: float = 0.42,
    periodic_x: bool = True,
) -> np.ndarray:
    """Gentler bright-core → midtone gradient (power curve + light core blur)."""
    em = np.asarray(rgb, dtype=np.float64)
    luma = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    p = float(np.clip(power, 0.45, 1.0))
    soft_l = np.power(np.clip(luma, 0.0, 1.0), p)
    scale = soft_l / (luma + 1e-8)
    out = em * scale[..., None]
    # Extra blur only where cores were hottest — spreads clip into midtones.
    p92 = float(np.percentile(luma, 92)) + 1e-8
    core = np.clip((luma - p92 * 0.85) / max(1.0 - p92 * 0.85, 1e-6), 0.0, 1.0)
    core = _gauss_float(core, 2.4, periodic_x=periodic_x)
    m = float(np.clip(mix, 0.0, 1.0))
    for c in range(min(3, out.shape[2])):
        blurred = _gauss_float(out[:, :, c], float(blur_sigma), periodic_x=periodic_x)
        out[:, :, c] = (1.0 - m * core) * out[:, :, c] + (m * core) * blurred
    return np.clip(out, 0.0, 1.0)


def _restore_hero_chroma(
    rgb: np.ndarray,
    gate: np.ndarray,
    hero_rgb: np.ndarray,
    *,
    strength: float = 0.62,
) -> np.ndarray:
    """Re-lean dense regions toward hero hue after soft blur / base add wash."""
    em = np.asarray(rgb, dtype=np.float64)
    luma = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    p95 = float(np.percentile(luma, 95)) + 1e-8
    dens = np.clip(luma / p95, 0.0, 1.0)
    dens = dens * (0.25 + 0.75 * np.clip(gate, 0.0, 1.0))
    tinted = _tint_preserve_luma(luma, _chroma_from_hero(hero_rgb))
    amt = float(strength) * np.clip(dens**0.75, 0.0, 1.0)
    out = em * (1.0 - amt[..., None]) + tinted * amt[..., None]
    gray = luma[..., None]
    out = out + (out - gray) * dens[..., None] * (0.55 * float(strength))
    return np.clip(out, 0.0, 1.0)


def _lift_dense_emission(
    rgb: np.ndarray,
    *,
    gain: float = 2.4,
    black_pct: float = 55.0,
) -> np.ndarray:
    """Raise intensity in surviving dense regions; crush near-black point."""
    em = np.asarray(rgb, dtype=np.float64)
    luma = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    black = float(np.percentile(luma, black_pct))
    # Soft black crush then gain into the mid/high end of surviving mass.
    lifted = np.clip(em - black, 0.0, None)
    denom = max(float(np.percentile(lifted, 99.5)) + 1e-8, 1e-6)
    lifted = np.clip(lifted / denom * float(gain), 0.0, 1.0)
    # Preserve soft skirts of gated mass (avoid hard clip to pure black).
    mask = (luma > black * 0.35).astype(np.float64)[..., None]
    out = np.maximum(lifted, em * 0.22 * mask)
    return np.clip(out, 0.0, 1.0)


def _hero_boost_emission(
    rgb: np.ndarray,
    gate: np.ndarray,
    *,
    strength: float = 0.55,
    periodic_x: bool = True,
) -> np.ndarray:
    """Further boost one/two densest regions so a readable hero complex appears."""
    em = np.asarray(rgb, dtype=np.float64)
    g = np.asarray(gate, dtype=np.float64)
    luma = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    score = _gauss_float(np.clip(luma * (0.45 + 0.55 * g), 0.0, 1.0), 4.5, periodic_x=periodic_x)
    thr = float(np.percentile(score, 97.2))
    hero = np.clip((score - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)
    hero = _gauss_float(hero, 3.5, periodic_x=periodic_x)
    hero = np.clip(hero ** 0.80, 0.0, 1.0)
    boost = 1.0 + float(strength) * hero
    # Saturation punch in hero cores (keep hue, raise chroma).
    gray = luma[..., None]
    sat_boost = em + (em - gray) * (hero[..., None] * 0.55)
    out = np.clip(sat_boost * boost[..., None], 0.0, 1.0)
    # Mild peak normalize — avoid stretching mids into blown-out white.
    peak = float(np.percentile(out, 99.85)) + 1e-8
    return np.clip(out / max(peak, 1.08), 0.0, 1.0)


def _soft_highlight_rolloff(
    rgb: np.ndarray,
    *,
    knee: float = 0.42,
    strength: float = 0.72,
) -> np.ndarray:
    """Compress highlights while preserving chroma (tames blown-out neon cores)."""
    em = np.asarray(rgb, dtype=np.float64)
    luma = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    k = float(np.clip(knee, 0.15, 0.85))
    s = float(np.clip(strength, 0.0, 1.0))
    over = np.maximum(luma - k, 0.0)
    # Soft shoulder: pull excess back toward the knee.
    compressed = luma - s * over * (over / (over + (1.0 - k) + 1e-8))
    scale = compressed / (luma + 1e-8)
    out = em * scale[..., None]
    # Channel peak shoulder — red-orange heroes clip R long before Rec.709 luma.
    peak = np.max(out, axis=2)
    pk = float(np.clip(knee + 0.12, 0.35, 0.92))
    p_over = np.maximum(peak - pk, 0.0)
    p_comp = peak - s * p_over * (p_over / (p_over + (1.0 - pk) + 1e-8))
    p_scale = p_comp / (peak + 1e-8)
    return np.clip(out * p_scale[..., None], 0.0, 1.0)


def _compose_cloud_layer_interaction(
    structure: np.ndarray,
    emission: np.ndarray,
    dust_occ: np.ndarray,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
    style: str = "balanced",
    hero_rgb: np.ndarray | None = None,
    hero_preset: str = "magenta",
    macro: np.ndarray | None = None,
    shell: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-layer emission under a shared large-scale silhouette.

    Soft dark vista (+ optional Barnard-scale shells) under gated hero; ``macro``
    keys amp/lanes. Composite is additive then clip.
    """
    style_key = str(style).strip().lower()
    dramatic = style_key == "dramatic"
    dust_sigma = 2.4 if dramatic else 2.0
    struct_sigma = 2.8 if dramatic else 2.2
    dust_blur = _gauss_float(dust_occ, dust_sigma, periodic_x=periodic_x)
    dust_gate = _contrast_stretch(dust_blur, lo_pct=18.0, hi_pct=97.0, power=0.72)
    # Strong dust drive — sparse dust should nearly kill hero emission.
    dust_gate = np.clip(dust_gate ** 0.82, 0.0, 1.0)

    struct_soft = _gauss_float(structure, struct_sigma, periodic_x=periodic_x)
    struct_soft = _contrast_stretch(struct_soft, lo_pct=22.0, hi_pct=96.0, power=0.90)
    # Overlap: product + min so only co-located dense regions survive as hero.
    overlap = np.minimum(dust_gate, struct_soft)
    gate = np.clip(dust_gate * (0.28 + 0.72 * struct_soft) * (0.45 + 0.55 * overlap), 0.0, 1.0)

    if macro is not None:
        macro_s = np.clip(np.asarray(macro, dtype=np.float64), 0.0, 1.0)
    else:
        macro_s = np.clip(struct_soft, 0.0, 1.0)
    # Edge strength: preserve fine detail on silhouettes / knots only.
    edge_s = np.abs(struct_soft - _gauss_float(struct_soft, 6.5, periodic_x=periodic_x))
    edge_s = _contrast_stretch(edge_s, lo_pct=35.0, hi_pct=99.0, power=0.85)

    if hero_rgb is not None:
        target = np.asarray(hero_rgb, dtype=np.float64).reshape(3)
        target = target / (float(np.max(target)) + 1e-8)
        preset_key = str(hero_preset).strip().lower() or "magenta"
    else:
        target, preset_key = resolve_nebula_hero_rgb(preset=hero_preset)

    # --- Vista underlayer: expansive dark cloud envelope (fills most of equirect) ---
    # Deliberately NOT hard-gated to macro — hero owns the silhouette; vista is the sky.
    struct_wide = _gauss_float(
        np.asarray(structure, dtype=np.float64),
        36.0 if dramatic else 28.0,
        periodic_x=periodic_x,
    )
    dust_wide = _gauss_float(dust_blur, 22.0 if dramatic else 16.0, periodic_x=periodic_x)
    sb = _contrast_stretch(struct_wide, lo_pct=1.0, hi_pct=82.0, power=0.32)
    db = _contrast_stretch(dust_wide, lo_pct=1.5, hi_pct=86.0, power=0.36)
    # Wide soft field + high pedestal so most of the FOV carries immersive volume.
    vista_density = np.clip(0.48 + 0.52 * sb * (0.40 + 0.60 * db), 0.0, 1.0)
    vista_density = np.clip(vista_density ** 0.36, 0.0, 1.0)
    # Soft macro bias — surrounds stay lit (inside the gas, not island-on-void).
    vista_density = np.clip(vista_density * (0.82 + 0.18 * macro_s), 0.0, 1.0)
    # Barnard-scale filamentary shells — faint arcs through the surround.
    if shell is not None:
        sh = np.clip(np.asarray(shell, dtype=np.float64), 0.0, 1.0)
        # Shells read strongest off the hero core (loop around / through the vista).
        shell_w = 0.72 if dramatic else 0.58
        vista_density = np.clip(
            vista_density + shell_w * sh * (0.45 + 0.55 * (1.0 - 0.65 * macro_s)),
            0.0,
            1.0,
        )
    base_lane = _gauss_float(dust_wide, 11.0 if dramatic else 8.5, periodic_x=periodic_x)
    # Mild lane carve — never eat enveloping gas into black corridors.
    lane_carve_k = 0.12 if dramatic else 0.08
    vista_density = np.clip(vista_density * (1.0 - lane_carve_k * base_lane), 0.0, 1.0)
    vista_density = _gauss_float(vista_density, 22.0 if dramatic else 16.0, periodic_x=periodic_x)
    vista_density = _downsample_blur2d(
        vista_density,
        scale=0.10 if dramatic else 0.14,
        blur_sigma=3.6,
        periodic_x=periodic_x,
    )
    # Soft cloudy fill structure (not flat color) — LF noise + blurred density.
    h_v, w_v = int(structure.shape[0]), int(structure.shape[1])
    # Equirect Y → poles: top/bottom rows must stay enveloped (no dark caps).
    # Wider than |y|^1.45 so caps and near-cap belts both carry structured fill.
    yy = np.linspace(-1.0, 1.0, h_v, dtype=np.float64)[:, None]
    pole_w = np.clip(np.abs(yy) ** 1.05, 0.0, 1.0)
    # Cap belt: outer ~18% of height gets an extra structured envelope.
    pole_cap = np.clip((np.abs(yy) - 0.64) / 0.36, 0.0, 1.0) ** 0.85
    fill_noise = _filtered_white_fbm(
        rng,
        h_v,
        w_v,
        octaves=5,
        gain=0.66,
        base_sigma=max(20.0, 0.085 * float(w_v)),
        periodic_x=periodic_x,
    )
    fill_noise = _gauss_float(fill_noise, 5.0 if dramatic else 4.2, periodic_x=periodic_x)
    fill_noise = _contrast_stretch(fill_noise, lo_pct=2.5, hi_pct=97.5, power=0.42)
    # Extra soft LF pass — modest amplitude so surround feels like extended gas.
    fill_lf = _filtered_white_fbm(
        rng,
        h_v,
        w_v,
        octaves=3,
        gain=0.58,
        base_sigma=max(36.0, 0.14 * float(w_v)),
        periodic_x=periodic_x,
    )
    fill_lf = _gauss_float(fill_lf, 8.0 if dramatic else 6.5, periodic_x=periodic_x)
    fill_lf = _contrast_stretch(fill_lf, lo_pct=5.0, hi_pct=95.0, power=0.55)
    # Blurred structure residue — same morphology family as hero, much softer.
    struct_fill = _gauss_float(
        np.asarray(structure, dtype=np.float64),
        22.0 if dramatic else 16.0,
        periodic_x=periodic_x,
    )
    struct_fill = _contrast_stretch(struct_fill, lo_pct=2.0, hi_pct=90.0, power=0.42)
    struct_fill = _downsample_blur2d(
        struct_fill, scale=0.14 if dramatic else 0.18, blur_sigma=2.6, periodic_x=periodic_x
    )
    # Wide cloudy swing so soft fill reads as lobed gas, not a narrow pedestal.
    # Noise dominates cavities (hero holes) so fill stays cloudy where structure is thin.
    cavity = np.clip(1.0 - struct_fill, 0.0, 1.0)
    fill_cloud = np.clip(
        0.10
        + 0.48 * fill_noise
        + 0.22 * fill_lf
        + 0.16 * struct_fill
        + 0.12 * vista_density
        + 0.36 * cavity * fill_noise
        + 0.22 * pole_w * fill_lf
        + 0.28 * pole_cap * (0.45 * fill_noise + 0.55 * fill_lf),
        0.0,
        1.0,
    )
    # Light soften only — keep mid-scale cloudy lobes readable in the fill.
    fill_cloud = _gauss_float(fill_cloud, 3.6 if dramatic else 3.0, periodic_x=periodic_x)
    fill_cloud = _contrast_stretch(fill_cloud, lo_pct=2.0, hi_pct=98.0, power=0.72)
    # Far / off-macro regions: slightly darker + desaturated (deeper volume).
    # Ease far-darken toward poles so equirect caps stay filled.
    far = np.clip(1.0 - 0.40 * macro_s, 0.0, 1.0)
    far = _gauss_float(far, 12.0, periodic_x=periodic_x)
    # Kill far-darken on caps so poles aren't systematically emptier than mid-lats.
    far = far * (1.0 - 0.85 * pole_w) * (1.0 - 0.90 * pole_cap)
    deeper = np.clip(
        target * np.array([0.88, 0.72, 0.65], dtype=np.float64),
        0.0,
        1.0,
    )
    near_tint = np.clip(0.96 * target + 0.04 * deeper, 0.0, 1.0)
    far_tint = np.clip(0.72 * target + 0.28 * deeper, 0.0, 1.0)
    far_lu = float(0.2126 * far_tint[0] + 0.7152 * far_tint[1] + 0.0722 * far_tint[2])
    far_tint = np.clip(0.62 * far_tint + 0.38 * far_lu, 0.0, 1.0)
    base_tint = (
        near_tint.reshape(1, 1, 3) * (1.0 - 0.45 * far[..., None])
        + far_tint.reshape(1, 1, 3) * (0.45 * far[..., None])
    )
    # Dimmer than hero — soft fill covers voids; stay faint so clumps still punch.
    # Weak macro gate so poles/cavities stay covered without a mid-grey wash.
    vista_amp0 = 0.46 if dramatic else 0.34
    vista_amp_map = vista_amp0 * (0.86 + 0.14 * macro_s) * (0.18 + 0.82 * fill_cloud)
    # Cavities + poles still carry readable fill amp (not crushed to flat dark).
    vista_amp_map = vista_amp_map * (0.75 + 0.25 * cavity) * (1.0 - 0.12 * far)
    vista_amp_map = vista_amp_map * (1.0 + 0.42 * pole_w + 0.55 * pole_cap)
    base_rgb = base_tint * fill_cloud[..., None] * vista_amp_map[..., None]
    # Structured ambient from the same cloudy field (never flat wash).
    ambient_k = 0.024 if dramatic else 0.016
    base_rgb = base_rgb + base_tint * (
        ambient_k
        * (
            0.30
            + 0.45 * fill_cloud
            + 0.28 * cavity * fill_noise
            + 0.42 * pole_w * fill_lf
            + 0.55 * pole_cap * fill_cloud
        )
    )[..., None]
    # Hard cloudy floor — faint full-sky underlayer; pole caps get a stronger seal.
    floor_lu = 0.078 if dramatic else 0.058
    floor_k = floor_lu * (1.0 + 0.55 * pole_cap)
    base_floor = base_tint * (
        floor_k * (0.50 + 0.50 * fill_cloud) * (0.85 + 0.45 * pole_w + 0.35 * pole_cap)
    )[..., None]
    base_rgb = np.maximum(base_rgb, base_floor)
    # Light soften only — keep mid-scale cloudy variation readable.
    for c in range(3):
        soft = _gauss_float(base_rgb[:, :, c], 5.5 if dramatic else 4.5, periodic_x=periodic_x)
        base_rgb[:, :, c] = 0.70 * base_rgb[:, :, c] + 0.30 * soft
    base_rgb = np.clip(base_rgb, 0.0, 1.0)

    # --- Hero layer: gated, colored, punched ---
    em = np.asarray(emission, dtype=np.float64)
    em = np.clip(em * gate[..., None], 0.0, 1.0)
    # Soften more in broad flats; keep sharper on edges / dense knots.
    scale = 0.28 if dramatic else 0.34
    flat_w = np.clip((1.0 - edge_s) * (0.35 + 0.65 * macro_s), 0.0, 1.0)
    for c in range(min(3, em.shape[2])):
        soft = _downsample_blur2d(
            em[:, :, c], scale=scale, blur_sigma=1.75, periodic_x=periodic_x
        )
        mix = 0.35 + 0.50 * flat_w
        em[:, :, c] = np.clip((1.0 - mix) * em[:, :, c] + mix * soft, 0.0, 1.0)
    em = np.clip(em, 0.0, 1.0)

    # Extinction dust: coarse lanes dominate; fine edge is secondary.
    dust_hi = _contrast_stretch(dust_blur, lo_pct=10.0, hi_pct=98.0, power=0.58)
    edge = np.abs(dust_blur - _gauss_float(dust_blur, 7.5, periodic_x=periodic_x))
    edge = _contrast_stretch(edge, lo_pct=30.0, hi_pct=99.0, power=0.80)
    dust_out = np.clip(0.72 * dust_hi + 0.22 * edge + 0.18 * dust_gate * struct_soft, 0.0, 0.94)
    # Boost dust in macro cavities so fingers cut emission as connected silhouettes.
    cavity = np.clip(1.0 - macro_s, 0.0, 1.0)
    dust_out = np.clip(dust_out + 0.28 * cavity * dust_hi, 0.0, 0.94)
    dust_out = _gauss_float(dust_out, 1.6 if dramatic else 1.25, periodic_x=periodic_x)
    dust_out = np.clip(dust_out, 0.0, 0.94)

    # Wider soft dark lanes — corridors through gas (shared for all hero colors).
    lane_k = 0.58 if dramatic else 0.44
    dust_lane = _gauss_float(dust_out, 4.2 if dramatic else 3.2, periodic_x=periodic_x)
    em_pre = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    em_dens = np.clip(em_pre / (float(np.percentile(em_pre, 95)) + 1e-8), 0.0, 1.0)
    lane_k_eff = lane_k * (1.0 + 0.38 * em_dens)
    transmit = np.clip(1.0 - lane_k_eff * dust_lane, 0.18, 1.0)
    em = np.clip(em * transmit[..., None], 0.0, 1.0)

    # Soft mid-scale structure imprint (shared; no red-only HF path).
    struct_blur = _gauss_float(
        np.asarray(structure, dtype=np.float64),
        7.5 if dramatic else 6.0,
        periodic_x=periodic_x,
    )
    detail = np.abs(np.asarray(structure, dtype=np.float64) - struct_blur)
    detail = _contrast_stretch(detail, lo_pct=22.0, hi_pct=96.0, power=0.82)
    detail = _gauss_float(detail, 1.6, periodic_x=periodic_x)
    imprint = np.clip(0.62 + 0.36 * detail, 0.45, 1.12)
    imprint = np.clip(
        imprint * (0.55 + 0.45 * em_dens) + (1.0 - em_dens) * 0.96,
        0.50,
        1.12,
    )
    em = np.clip(em * imprint[..., None], 0.0, 1.0)

    # Photogenic finish — shared soft path (preset red == custom #FF1818).
    gain = 2.35 if dramatic else 1.95
    em = _lift_dense_emission(em, gain=gain, black_pct=48.0 if dramatic else 44.0)
    em = np.clip(em * (0.32 + 0.68 * macro_s)[..., None], 0.0, 1.0)
    var = 0.22 if dramatic else 0.16
    amt = 0.90 if dramatic else 0.82
    em = _apply_hero_color_and_variation(
        em,
        gate,
        rng,
        hero_rgb=target,
        preset_key=preset_key,
        amount=amt,
        variation=var,
        periodic_x=periodic_x,
    )
    em = _tighten_emission_hue(
        em,
        gate,
        target,
        strength=0.52 if dramatic else 0.44,
        sat_cap=0.90,
        preset_key=preset_key,
    )
    soft_sig = 2.85 if dramatic else 2.25
    for c in range(min(3, em.shape[2])):
        soft = _gauss_float(em[:, :, c], soft_sig, periodic_x=periodic_x)
        mix = 0.58 + 0.28 * flat_w
        em[:, :, c] = np.clip((1.0 - mix) * em[:, :, c] + mix * soft, 0.0, 1.0)
    boost_gate = np.clip(gate * (0.35 + 0.65 * macro_s), 0.0, 1.0)
    em = _hero_boost_emission(
        em,
        boost_gate,
        strength=0.58 if dramatic else 0.44,
        periodic_x=periodic_x,
    )
    em = _soften_core_falloff(
        em,
        power=0.78,
        blur_sigma=2.15 if dramatic else 1.75,
        mix=0.55 if dramatic else 0.44,
        periodic_x=periodic_x,
    )
    fin_scale = 0.42 if dramatic else 0.50
    for c in range(min(3, em.shape[2])):
        soft = _downsample_blur2d(
            em[:, :, c], scale=fin_scale, blur_sigma=1.35, periodic_x=periodic_x
        )
        mix = 0.45 + 0.40 * flat_w
        em[:, :, c] = np.clip((1.0 - mix) * em[:, :, c] + mix * soft, 0.0, 1.0)
    edge_sig = 1.85 if dramatic else 1.55
    for c in range(min(3, em.shape[2])):
        soft = _gauss_float(em[:, :, c], edge_sig, periodic_x=periodic_x)
        mix = 0.35 + 0.40 * flat_w
        em[:, :, c] = np.clip((1.0 - mix) * em[:, :, c] + mix * soft, 0.0, 1.0)
    em = np.clip(em, 0.0, 1.0)

    # Additive composite: dark vista under punched hero cores.
    em = np.clip(base_rgb + em, 0.0, 1.0)
    # Re-assert cloudy floor after hero add (lanes/boost must not punch black holes).
    em = np.maximum(em, base_floor)
    # Soften hero↔fill transition band so silhouettes don't read as hard cutouts.
    em_lu0 = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    p42 = float(np.percentile(em_lu0, 42))
    p82 = float(np.percentile(em_lu0, 82))
    span = max(p82 - p42, 1e-6)
    trans_w = np.clip((em_lu0 - p42) / span, 0.0, 1.0)
    trans_w = (4.0 * trans_w * (1.0 - trans_w)) ** 1.15
    trans_w = _gauss_float(trans_w, 3.2, periodic_x=periodic_x)
    for c in range(min(3, em.shape[2])):
        soft = _gauss_float(em[:, :, c], 4.2 if dramatic else 3.4, periodic_x=periodic_x)
        mix = 0.28 * trans_w if dramatic else 0.22 * trans_w
        em[:, :, c] = np.clip((1.0 - mix) * em[:, :, c] + mix * soft, 0.0, 1.0)
    em = _soft_highlight_rolloff(em, knee=0.48, strength=0.55)
    em = _restore_hero_chroma(
        em,
        gate,
        target,
        strength=0.72 if dramatic else 0.62,
    )
    em = _tighten_emission_hue(
        em,
        gate,
        target,
        strength=0.42,
        sat_cap=0.90,
        preset_key=preset_key,
    )

    # Structure imprint follows gate and macro; preserve peak readability.
    struct_out = np.clip(structure * (0.40 + 0.60 * gate) * (0.35 + 0.65 * macro_s), 0.0, 1.0)
    sp = float(np.percentile(struct_out, 99.5)) + 1e-8
    if sp < 0.55:
        struct_out = np.clip(struct_out * (0.62 / sp), 0.0, 1.0)
    return struct_out, em, dust_out


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


def _barnard_shell_field(
    rng: np.random.Generator,
    height: int,
    width: int,
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    n_arcs: int = 2,
    periodic_x: bool = True,
) -> np.ndarray:
    """Large incomplete filamentary shells (Barnard's Loop–scale Hα arcs)."""
    h, w = int(height), int(width)
    shells = np.zeros((h, w), dtype=np.float64)
    n = max(1, int(n_arcs))
    for i in range(n):
        # Wide arcs centered near the FOV; secondary arcs offset.
        cx = float(rng.uniform(-0.35, 0.35)) + (0.25 if i > 0 else 0.0) * float(
            rng.choice([-1.0, 1.0])
        )
        cy = float(rng.uniform(-0.25, 0.20))
        r0 = float(rng.uniform(0.55, 0.95)) * (1.0 if i == 0 else float(rng.uniform(0.72, 0.92)))
        thick = float(rng.uniform(0.045, 0.090)) * (1.15 if i == 0 else 0.85)
        dx = np.minimum(np.abs(xx - cx), 2.0 - np.abs(xx - cx))
        rad = np.sqrt(dx * dx + (yy - cy) ** 2)
        ring = np.exp(-(((rad - r0) / max(thick, 1e-4)) ** 2))
        # Incomplete C-arc (not a closed ring).
        ang = np.arctan2(yy - cy, np.sin(np.pi * dx))
        span = float(rng.uniform(1.6, 3.4))
        center = float(rng.uniform(-np.pi, np.pi))
        dang = np.abs((ang - center + np.pi) % (2.0 * np.pi) - np.pi)
        arc_gate = np.clip(1.0 - dang / max(span * 0.5, 1e-4), 0.0, 1.0) ** 0.65
        amp = float(rng.uniform(0.72, 1.0)) if i == 0 else float(rng.uniform(0.38, 0.62))
        shells = np.maximum(shells, ring * arc_gate * amp)
    # Filamentary break-up along the shell (not a smooth cartoon ring).
    break_up = _filtered_white_fbm(
        rng,
        h,
        w,
        octaves=4,
        base_sigma=max(8.0, 0.022 * float(w)),
        periodic_x=periodic_x,
    )
    break_up = _contrast_stretch(break_up, lo_pct=20.0, hi_pct=96.0, power=0.75)
    shells = np.clip(shells * (0.35 + 0.65 * break_up), 0.0, 1.0)
    shells = _domain_warp_xy(
        shells,
        rng,
        amp_px=max(4.0, 0.010 * float(w)),
        warp_scale=0.035,
        periodic_x=periodic_x,
    )
    shells = _gauss_float(shells, 2.2, periodic_x=periodic_x)
    return np.clip(shells, 0.0, 1.0)


def build_cloud_complex_envelope(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool = True,
    style: str = "balanced",
    hero_color: str = "magenta",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (structure_weight, emission_rgb, dust_occlusion) in [0,1].

    Soft FBM body modulated by ridged/warped filaments — spider-web nebula gas
    instead of geometric rings and blobs. ``style``: subtle|balanced|dramatic.
    ``hero_color``: preset (red/gold/blue/magenta/neutral or long names), ``#hex``, or ``R,G,B``.
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

    # Large-scale silhouette: 1–2 explicit lobes + soft LF FBM skirts.
    # Explicit Gaussians guarantee a readable distant shape (FBM alone stays islandy).
    macro = np.zeros((h, w), dtype=np.float64)
    n_lobes = 2 if style_key == "dramatic" else 1
    if style_key == "subtle":
        n_lobes = 1
    for i in range(n_lobes):
        lx = float(rng.uniform(-0.55, 0.55))
        ly = float(rng.uniform(-0.35, 0.30))
        lsig = float(rng.uniform(0.28, 0.48)) * (1.15 if i == 0 else 0.72)
        dx = np.minimum(np.abs(xx - lx), 2.0 - np.abs(xx - lx))
        lobe = np.exp(-((dx**2 + (yy - ly) ** 2) / (2.0 * lsig**2)))
        amp = float(rng.uniform(0.85, 1.0)) if i == 0 else float(rng.uniform(0.40, 0.62))
        macro = np.maximum(macro, lobe * amp)
    skirt = _filtered_white_fbm(
        rng,
        h,
        w,
        octaves=3,
        base_sigma=max(24.0, 0.070 * float(w)),
        periodic_x=periodic_x,
    )
    skirt = _gauss_float(skirt, 12.0 if style_key == "dramatic" else 9.0, periodic_x=periodic_x)
    skirt = _contrast_stretch(skirt, lo_pct=15.0, hi_pct=94.0, power=0.70)
    macro = np.clip(0.78 * macro + 0.32 * skirt * macro, 0.0, 1.0)
    macro = _gauss_float(macro, 8.0 if style_key == "dramatic" else 6.0, periodic_x=periodic_x)
    # Hard stretch: cavities near zero, dominant lobes near one.
    macro = _contrast_stretch(
        macro,
        lo_pct=5.0,
        hi_pct=90.0,
        power=0.48 if style_key == "dramatic" else 0.58,
    )
    macro = np.clip(macro ** (1.55 if style_key == "dramatic" else 1.25), 0.0, 1.0)
    macro = np.clip(macro, 0.0, 1.0)

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

    # FOV-scale Hα shells (Barnard's Loop–like) — faint vista arcs, not hero mass.
    n_shell = {"subtle": 1, "balanced": 2, "dramatic": 2}.get(style_key, 2)
    shell = _barnard_shell_field(
        rng,
        h,
        w,
        yy,
        xx,
        n_arcs=n_shell,
        periodic_x=periodic_x,
    )
    # Prefer shells wrapping the macro complex rather than sitting on its peak.
    shell = np.clip(shell * (0.40 + 0.60 * (1.0 - 0.55 * macro)), 0.0, 1.0)

    # Hero complex: one dominant core + Barnard-loop–like arc (Orion-style read).
    # Balanced keeps a lighter accent; dramatic pushes a clear hero body.
    arcs = np.zeros((h, w), dtype=np.float64)
    cores = np.zeros((h, w), dtype=np.float64)
    hero_w = {"subtle": 0.22, "balanced": 0.55, "dramatic": 1.0}.get(style_key, 0.55)
    if hero_w > 1e-6:
        # Off-center hero so the plate does not read as a bullseye.
        hx = float(rng.uniform(-0.42, 0.42))
        hy = float(rng.uniform(-0.28, 0.22))
        # Primary emission body.
        hsig = float(rng.uniform(0.14, 0.22)) * (1.08 if style_key == "dramatic" else 1.0)
        dxh = np.minimum(np.abs(xx - hx), 2.0 - np.abs(xx - hx))
        hero_core = np.exp(-((dxh**2 + (yy - hy) ** 2) / (2.0 * hsig**2)))
        hero_core = np.clip(hero_core ** 0.92, 0.0, 1.0)
        # Amplify cores where macro is already bright at the hero seed.
        yi = int(np.clip(round((hy + 1.0) * 0.5 * (h - 1)), 0, h - 1))
        xi = int(np.clip(round((hx + 1.0) * 0.5 * (w - 1)), 0, w - 1))
        macro_here = float(macro[yi, xi])
        core_boost = 0.55 + 0.70 * macro_here
        cores = np.maximum(
            cores, hero_core * float(rng.uniform(0.72, 0.95)) * hero_w * core_boost
        )
        # Secondary compact knot near the hero (M42-like companion).
        kx = hx + float(rng.uniform(-0.12, 0.12))
        ky = hy + float(rng.uniform(0.06, 0.18))
        ksig = float(rng.uniform(0.045, 0.085))
        dxk = np.minimum(np.abs(xx - kx), 2.0 - np.abs(xx - kx))
        knot = np.exp(-((dxk**2 + (yy - ky) ** 2) / (2.0 * ksig**2)))
        cores = np.maximum(
            cores, knot * float(rng.uniform(0.55, 0.85)) * hero_w * (0.60 + 0.55 * macro_here)
        )
        cores = np.clip(cores * (0.35 + 0.75 * macro), 0.0, 1.0)
        cores = _gauss_float(cores, 1.15, periodic_x=periodic_x)
        # Wide incomplete arc around the hero (Barnard's Loop–like).
        r0 = float(rng.uniform(0.32, 0.52))
        thick = float(rng.uniform(0.055, 0.10))
        rad = np.sqrt(dxh * dxh + (yy - hy) ** 2)
        ring = np.exp(-(((rad - r0) / thick) ** 2))
        # Fade one side so it reads as an arc, not a closed oval.
        ang = np.arctan2(yy - hy, np.sin(np.pi * dxh))
        arc_gate = 0.35 + 0.65 * np.clip(np.cos(ang - float(rng.uniform(-0.4, 0.4))), 0.0, 1.0)
        arcs = np.maximum(
            arcs,
            ring * arc_gate * float(rng.uniform(0.55, 0.88)) * hero_w * (0.50 + 0.60 * macro_here),
        )
        arcs = np.clip(arcs * (0.30 + 0.80 * macro), 0.0, 1.0)
        arcs = _gauss_float(arcs, 1.35, periodic_x=periodic_x)
        # Optional faint secondary wisps (subtle only).
        if style_key == "subtle":
            n_extra = int(rng.integers(0, 2))
            for _ in range(n_extra):
                cx = float(rng.uniform(-0.85, 0.85))
                cy = float(rng.uniform(-0.55, 0.55))
                r1 = float(rng.uniform(0.22, 0.55))
                th1 = float(rng.uniform(0.08, 0.14))
                dx = np.minimum(np.abs(xx - cx), 2.0 - np.abs(xx - cx))
                rad2 = np.sqrt(dx * dx + (yy - cy) ** 2)
                arcs = np.maximum(
                    arcs,
                    np.exp(-(((rad2 - r1) / th1) ** 2)) * float(rng.uniform(0.25, 0.45)) * 0.35,
                )

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
    # Edge mask: freckles enrich silhouettes/knots, not broad flats.
    struct_pre = np.clip(body * (0.32 + 0.58 * filaments), 0.0, 1.0)
    edge_pre = np.abs(struct_pre - _gauss_float(struct_pre, 5.5, periodic_x=periodic_x))
    edge_pre = _contrast_stretch(edge_pre, lo_pct=40.0, hi_pct=99.0, power=0.85)
    freckle_edge = np.clip(0.18 + 0.82 * np.maximum(edge_pre, 0.55 * cores + 0.35 * arcs), 0.0, 1.0)
    freckles = np.clip(freckles * freckle_edge, 0.0, 1.0)
    # Dark lanes along filament voids (spider-web carve) — edge-weighted.
    lane_carve = np.clip((1.0 - filaments) * freckles, 0.0, 1.0)

    # Body + filaments, with hero core/arc as the readable anchor (not soup).
    structure = body * (0.32 + 0.58 * filaments)
    structure = np.clip(
        structure
        + 0.72 * hero_w * cores
        + 0.58 * hero_w * arcs
        + 0.22 * filaments * cores
        + 0.38 * shell,
        0.0,
        1.0,
    )
    structure = np.clip(structure * struct_g, 0.0, 1.0)
    # Macro silhouette owns the distant shape (dramatic stronger); keep peak mass.
    macro_floor = 0.05 if style_key == "dramatic" else 0.12
    structure = np.clip(structure * (macro_floor + (1.0 - macro_floor) * macro), 0.0, 1.0)
    # Soft full-sky floor so poles keep faint gas (weaker under dramatic so hero wins).
    floor_w = 0.12 if style_key == "dramatic" else 0.14
    yy_s = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    pole_lift = np.clip(np.abs(yy_s) ** 1.05, 0.0, 1.0)
    pole_cap_s = np.clip((np.abs(yy_s) - 0.64) / 0.36, 0.0, 1.0) ** 0.85
    structure = np.clip(
        structure * 0.86
        + body * floor_w * (0.70 + 0.30 * macro)
        + body * 0.14 * pole_lift * (0.55 + 0.45 * macro)
        + body * 0.16 * pole_cap_s * (0.65 + 0.35 * macro),
        0.0,
        1.0,
    )
    # Renormalize so hero cores still reach a readable peak.
    sp = float(np.percentile(structure, 99.5)) + 1e-8
    if sp < 0.72:
        structure = np.clip(structure * (0.78 / sp), 0.0, 1.0)
    # Dust lanes carve harder through the hero complex; freckles muted in flats.
    freckle_mix = 0.28 * freckle_g
    hero_lane = np.clip(freckles * (0.35 + 0.65 * cores + 0.40 * arcs), 0.0, 1.0)
    structure = np.clip(
        structure * (1.0 - freckle_mix * lane_carve) * (1.0 - 0.28 * hero_w * hero_lane),
        0.0,
        1.0,
    )
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

    # Multi-hue emission: Hα-heavy hero, cooler OIII lobes in the web.
    ha = np.clip(
        0.42 * filaments + 0.22 * body + 0.55 * cores + 0.38 * arcs + 0.48 * shell,
        0.0,
        1.0,
    )
    oiii = np.clip(0.55 * cool_lobes + 0.35 * filaments * (1.0 - 0.35 * freckles), 0.0, 1.0)
    dust_pink = np.clip(0.50 * body + 0.38 * filaments * (1.0 - freckles) + 0.22 * cores, 0.0, 1.0)
    emit_r = structure * (0.28 * dust_pink + 0.58 * ha + 0.12 * oiii)
    emit_g = structure * (0.16 * dust_pink + 0.28 * ha + 0.42 * oiii)
    emit_b = structure * (0.10 * dust_pink + 0.14 * ha + 0.52 * oiii + 0.22 * cool_lobes)
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

    # Coarse connected extinction: cavities + inverse-filaments, not only freckles.
    cavity = np.clip(1.0 - macro, 0.0, 1.0)
    dust_coarse = _filtered_white_fbm(
        rng,
        h,
        w,
        octaves=3,
        base_sigma=max(16.0, 0.048 * float(w)),
        periodic_x=periodic_x,
    )
    dust_coarse = _gauss_float(dust_coarse, 8.0 if style_key == "dramatic" else 6.5, periodic_x=periodic_x)
    dust_coarse = _contrast_stretch(dust_coarse, lo_pct=12.0, hi_pct=96.0, power=0.60)
    dust_coarse = np.clip(
        dust_coarse * (0.45 + 0.55 * cavity) * (0.40 + 0.60 * (1.0 - filaments)),
        0.0,
        1.0,
    )
    dust_occ = np.clip(
        0.62 * dust_coarse
        + 0.22 * lane_carve * freckle_g * (0.26 + 0.38 * structure)
        + 0.20 * hero_w * hero_lane * (0.18 + 0.28 * structure)
        + 0.28 * cavity * dust_coarse,
        0.0,
        0.88,
    )
    dust_occ = _gauss_float(dust_occ, 2.4 if style_key == "dramatic" else 1.8, periodic_x=periodic_x)
    dust_occ = np.clip(
        _destreak_columns(dust_occ, strength=0.70, blur_sigma=8.0, periodic_x=periodic_x),
        0.0,
        0.88,
    )
    hero_target, hero_key = parse_nebula_hero_color(str(hero_color))
    structure, emission, dust_occ = _compose_cloud_layer_interaction(
        structure,
        emission,
        dust_occ,
        rng,
        periodic_x=periodic_x,
        style=style_key,
        hero_rgb=hero_target,
        hero_preset=hero_key,
        macro=macro,
        shell=shell,
    )
    return structure.astype(np.float64), emission.astype(np.float64), dust_occ.astype(np.float64)
