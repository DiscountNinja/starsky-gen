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
NEBULA_HERO_COLOR_PRESETS: dict[str, tuple[float, float, float]] = {
    "classic-red": (1.00, 0.30, 0.34),
    "red": (1.00, 0.30, 0.34),
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

# Sibling hues for local variation around the primary target.
_NEBULA_VARIATION_SIBLINGS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "red": ((1.00, 0.55, 0.22), (0.70, 0.35, 0.55)),
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


def parse_nebula_hero_color(value: str | None) -> tuple[np.ndarray, str]:
    """Parse ``--nebula-hero-color`` into (rgb[3], canonical_key).

    Accepts:
    - Presets: ``classic-red``/``red``, ``gold-core``/``gold``, ``cool-blue``/``blue``,
      ``magenta-accent``/``magenta``, ``neutral``
    - Hex: ``#FC5656`` or ``FC5656``
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

    # Hex: #RRGGBB or RRGGBB
    hex_s = s[1:] if s.startswith("#") else s
    if len(hex_s) == 6 and all(c in "0123456789abcdefABCDEF" for c in hex_s):
        r = int(hex_s[0:2], 16) / 255.0
        g = int(hex_s[2:4], 16) / 255.0
        b = int(hex_s[4:6], 16) / 255.0
        return _normalize_hero_rgb((r, g, b)), "custom"

    # RGB triple
    if "," in s:
        parts = [p.strip() for p in s.replace(" ", "").split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"nebula hero color RGB must be R,G,B (got {value!r}); "
                "or use a preset / #hex"
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
        "(or classic-red|gold-core|cool-blue|magenta-accent), #RRGGBB, or R,G,B"
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
    """Tint toward primary hero color with local gold/cool variation."""
    em = np.asarray(rgb, dtype=np.float64)
    luma = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    p95 = float(np.percentile(luma, 95)) + 1e-8
    dens = np.clip(luma / p95, 0.0, 1.0)
    dens = dens * (0.35 + 0.65 * np.clip(gate, 0.0, 1.0))
    primary = np.asarray(hero_rgb, dtype=np.float64).reshape(3)
    primary = primary / (float(np.max(primary)) + 1e-8)
    sib_a, sib_b = _NEBULA_VARIATION_SIBLINGS.get(
        preset_key, _NEBULA_VARIATION_SIBLINGS["custom"]
    )
    gold = np.asarray(sib_a, dtype=np.float64)
    cool = np.asarray(sib_b, dtype=np.float64)
    # Soft spatial mix field for local hue (not crunchy pixel noise).
    mix = _filtered_white_fbm(
        rng,
        int(em.shape[0]),
        int(em.shape[1]),
        octaves=3,
        base_sigma=max(10.0, 0.035 * float(em.shape[1])),
        periodic_x=periodic_x,
    )
    mix = _gauss_float(mix, 2.2, periodic_x=periodic_x)
    mix = np.clip((mix - 0.35) / 0.45, 0.0, 1.0)
    # Core leans gold sibling; skirts lean cooler.
    edge = np.clip(1.0 - dens, 0.0, 1.0)
    local = (
        primary.reshape(1, 1, 3) * (1.0 - float(variation))
        + gold.reshape(1, 1, 3) * (float(variation) * 0.55 * mix[..., None])
        + cool.reshape(1, 1, 3) * (float(variation) * 0.45 * edge[..., None] * (1.0 - mix[..., None]))
    )
    local = local / (np.max(local, axis=2, keepdims=True) + 1e-8)
    # Preserve luma while replacing chroma toward local target.
    tinted = local * luma[..., None]
    amt = float(amount) * dens
    out = em * (1.0 - amt[..., None]) + tinted * amt[..., None]
    # Extra saturation only in densest cores.
    gray = luma[..., None]
    core_sat = np.clip((dens - 0.45) / 0.55, 0.0, 1.0)[..., None]
    out = out + (out - gray) * core_sat * 0.55
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
    sat_boost = em + (em - gray) * (hero[..., None] * 0.65)
    out = np.clip(sat_boost * boost[..., None], 0.0, 1.0)
    # Soft peak normalize — preserve relative hero punch without washing to white.
    peak = float(np.percentile(out, 99.85)) + 1e-8
    return np.clip(out / max(peak, 0.92), 0.0, 1.0)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gate emission by dust×structure overlap, then photogenic finish.

    Finish: intensity lift, hero color + local hue variation, soft volumetric
    blur, extinction lane carve, denser-core punch, and a final downsample-blur.
    """
    style_key = str(style).strip().lower()
    dramatic = style_key == "dramatic"
    dust_sigma = 2.4 if dramatic else 2.0
    struct_sigma = 2.8 if dramatic else 2.2
    dust_blur = _gauss_float(dust_occ, dust_sigma, periodic_x=periodic_x)
    dust_gate = _contrast_stretch(dust_blur, lo_pct=18.0, hi_pct=97.0, power=0.72)
    # Strong dust drive — sparse dust should nearly kill emission.
    dust_gate = np.clip(dust_gate ** 0.82, 0.0, 1.0)

    struct_soft = _gauss_float(structure, struct_sigma, periodic_x=periodic_x)
    struct_soft = _contrast_stretch(struct_soft, lo_pct=22.0, hi_pct=96.0, power=0.90)
    # Overlap: product + min so only co-located dense regions survive.
    overlap = np.minimum(dust_gate, struct_soft)
    gate = np.clip(dust_gate * (0.28 + 0.72 * struct_soft) * (0.45 + 0.55 * overlap), 0.0, 1.0)

    em = np.asarray(emission, dtype=np.float64)
    em = np.clip(em * gate[..., None], 0.0, 1.0)
    # Soften gated emission into cloud bodies (per-channel downsample-blur).
    scale = 0.32 if dramatic else 0.38
    for c in range(min(3, em.shape[2])):
        em[:, :, c] = _downsample_blur2d(
            em[:, :, c], scale=scale, blur_sigma=1.55, periodic_x=periodic_x
        )
    em = np.clip(em, 0.0, 1.0)

    # Extinction dust: wider dynamic range + edge emphasis for lane/edge darkening.
    dust_hi = _contrast_stretch(dust_blur, lo_pct=12.0, hi_pct=98.0, power=0.65)
    edge = np.abs(dust_blur - _gauss_float(dust_blur, 5.5, periodic_x=periodic_x))
    edge = _contrast_stretch(edge, lo_pct=30.0, hi_pct=99.0, power=0.80)
    dust_out = np.clip(0.62 * dust_hi + 0.38 * edge + 0.22 * dust_gate * struct_soft, 0.0, 0.94)
    dust_out = _gauss_float(dust_out, 1.1, periodic_x=periodic_x)
    dust_out = np.clip(dust_out, 0.0, 0.94)

    # Soft dark lanes (blurred dust) — corridors through gas, not cellular pits.
    lane_k = 0.48 if dramatic else 0.36
    dust_lane = _gauss_float(dust_out, 2.6, periodic_x=periodic_x)
    transmit = np.clip(1.0 - lane_k * dust_lane, 0.28, 1.0)
    em = np.clip(em * transmit[..., None], 0.0, 1.0)

    # Photogenic finish on surviving mass only.
    gain = 3.25 if dramatic else 2.65
    em = _lift_dense_emission(em, gain=gain, black_pct=50.0 if dramatic else 46.0)
    if hero_rgb is not None:
        target = np.asarray(hero_rgb, dtype=np.float64).reshape(3)
        target = target / (float(np.max(target)) + 1e-8)
        preset_key = str(hero_preset).strip().lower() or "magenta"
    else:
        target, preset_key = resolve_nebula_hero_rgb(preset=hero_preset)
    em = _apply_hero_color_and_variation(
        em,
        gate,
        rng,
        hero_rgb=target,
        preset_key=preset_key,
        amount=0.72 if dramatic else 0.58,
        variation=0.42 if dramatic else 0.34,
        periodic_x=periodic_x,
    )
    # Extra soft volumetric pass — heal crunch after lane carve.
    soft_sig = 2.55 if dramatic else 2.05
    for c in range(min(3, em.shape[2])):
        soft = _gauss_float(em[:, :, c], soft_sig, periodic_x=periodic_x)
        em[:, :, c] = np.clip(0.22 * em[:, :, c] + 0.78 * soft, 0.0, 1.0)
    em = _hero_boost_emission(
        em,
        gate,
        strength=1.05 if dramatic else 0.70,
        periodic_x=periodic_x,
    )
    # Final downsample-blur so edges feel volumetric (less crunchy).
    fin_scale = 0.48 if dramatic else 0.55
    for c in range(min(3, em.shape[2])):
        soft = _downsample_blur2d(
            em[:, :, c], scale=fin_scale, blur_sigma=1.25, periodic_x=periodic_x
        )
        em[:, :, c] = np.clip(0.40 * em[:, :, c] + 0.60 * soft, 0.0, 1.0)
    em = np.clip(em, 0.0, 1.0)

    # Structure keeps a soft imprint of the gate so density maps match emission.
    struct_out = np.clip(structure * (0.40 + 0.60 * gate), 0.0, 1.0)
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
        cores = np.maximum(cores, hero_core * float(rng.uniform(0.72, 0.95)) * hero_w)
        # Secondary compact knot near the hero (M42-like companion).
        kx = hx + float(rng.uniform(-0.12, 0.12))
        ky = hy + float(rng.uniform(0.06, 0.18))
        ksig = float(rng.uniform(0.045, 0.085))
        dxk = np.minimum(np.abs(xx - kx), 2.0 - np.abs(xx - kx))
        knot = np.exp(-((dxk**2 + (yy - ky) ** 2) / (2.0 * ksig**2)))
        cores = np.maximum(cores, knot * float(rng.uniform(0.55, 0.85)) * hero_w)
        cores = _gauss_float(cores, 1.15, periodic_x=periodic_x)
        # Wide incomplete arc around the hero (Barnard's Loop–like).
        r0 = float(rng.uniform(0.32, 0.52))
        thick = float(rng.uniform(0.055, 0.10))
        rad = np.sqrt(dxh * dxh + (yy - hy) ** 2)
        ring = np.exp(-(((rad - r0) / thick) ** 2))
        # Fade one side so it reads as an arc, not a closed oval.
        ang = np.arctan2(yy - hy, np.sin(np.pi * dxh))
        arc_gate = 0.35 + 0.65 * np.clip(np.cos(ang - float(rng.uniform(-0.4, 0.4))), 0.0, 1.0)
        arcs = np.maximum(arcs, ring * arc_gate * float(rng.uniform(0.55, 0.88)) * hero_w)
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
    # Dark lanes along filament voids (spider-web carve).
    lane_carve = np.clip((1.0 - filaments) * freckles, 0.0, 1.0)

    # Body + filaments, with hero core/arc as the readable anchor (not soup).
    structure = body * (0.32 + 0.58 * filaments)
    structure = np.clip(
        structure + 0.72 * hero_w * cores + 0.58 * hero_w * arcs + 0.22 * filaments * cores,
        0.0,
        1.0,
    )
    structure = np.clip(structure * struct_g, 0.0, 1.0)
    # Soft full-sky floor so poles keep faint gas (weaker under dramatic so hero wins).
    floor_w = 0.08 if style_key == "dramatic" else 0.14
    structure = np.clip(structure * 0.90 + body * floor_w, 0.0, 1.0)
    # Dust lanes carve harder through the hero complex.
    freckle_mix = 0.42 * freckle_g
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
    ha = np.clip(0.42 * filaments + 0.22 * body + 0.55 * cores + 0.38 * arcs, 0.0, 1.0)
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

    dust_occ = np.clip(
        lane_carve * freckle_g * (0.26 + 0.38 * structure)
        + hero_w * hero_lane * (0.18 + 0.28 * structure),
        0.0,
        0.78,
    )
    dust_occ = _gauss_float(dust_occ, 0.90, periodic_x=periodic_x)
    dust_occ = np.clip(
        _destreak_columns(dust_occ, strength=0.70, blur_sigma=8.0, periodic_x=periodic_x),
        0.0,
        0.78,
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
    )
    return structure.astype(np.float64), emission.astype(np.float64), dust_occ.astype(np.float64)
