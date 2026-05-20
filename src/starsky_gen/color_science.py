"""sRGB/linear transforms, Planck star colors, extinction, and blend helpers."""

from __future__ import annotations

import numpy as np

# CIE 1931 2° CMF sampled at 10 nm (380–780 nm) — compact tables
_WL = np.arange(380.0, 790.0, 10.0, dtype=np.float64)
_CIE_X = np.interp(
    _WL,
    [380, 440, 490, 510, 580, 645, 780],
    [0.0014, 0.343, 0.239, 0.0093, 0.873, 0.318, 0.0001],
)
_CIE_Y = np.interp(
    _WL,
    [380, 440, 490, 510, 580, 645, 780],
    [0.0000, 0.038, 0.323, 0.086, 0.872, 0.319, 0.0000],
)
_CIE_Z = np.interp(
    _WL,
    [380, 440, 490, 510, 580, 645, 780],
    [0.0065, 1.747, 4.702, 0.110, 0.018, 0.000, 0.000],
)

# sRGB D65 primaries (linear RGB from XYZ)
_SRGB_FROM_XYZ = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

# OBAFGKM effective temperatures (K) and sampling weights
_SPECTRAL_TEFF = np.array([42000.0, 15000.0, 8500.0, 6500.0, 5500.0, 4000.0, 3000.0], dtype=np.float64)
_SPECTRAL_WEIGHTS = np.array([0.0005, 0.007, 0.062, 0.13, 0.24, 0.37, 0.1905], dtype=np.float64)
_SPECTRAL_WEIGHTS /= _SPECTRAL_WEIGHTS.sum()


def _blackbody_rgb_raw(teff_k: float) -> np.ndarray:
    h = 6.62607015e-34
    c = 299792458.0
    k_b = 1.380649e-23
    wl_m = _WL * 1e-9
    spec = (2.0 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k_b * teff_k)) - 1.0)
    x = float(np.sum(spec * _CIE_X))
    y = float(np.sum(spec * _CIE_Y))
    z = float(np.sum(spec * _CIE_Z))
    if y < 1e-30:
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)
    xyz = np.array([x / y, 1.0, z / y], dtype=np.float64)
    rgb = _SRGB_FROM_XYZ @ xyz
    rgb = np.clip(rgb, 0.0, None)
    m = float(np.max(rgb))
    if m > 1e-12:
        rgb /= m
    return rgb.astype(np.float64)


_TEFF_LUT = np.linspace(3000.0, 40000.0, 256, dtype=np.float64)
_BLACKBODY_LUT = np.zeros((256, 3), dtype=np.float64)
for i, teff in enumerate(_TEFF_LUT):
    _BLACKBODY_LUT[i] = _blackbody_rgb_raw(float(teff))


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.0031308, x * 12.92, (1.0 + a) * np.power(np.clip(x, 0.0, None), 1.0 / 2.4) - a)


def blackbody_rgb(teff_k: float) -> np.ndarray:
    t = float(np.clip(teff_k, _TEFF_LUT[0], _TEFF_LUT[-1]))
    u = (t - _TEFF_LUT[0]) / (_TEFF_LUT[-1] - _TEFF_LUT[0]) * (_BLACKBODY_LUT.shape[0] - 1)
    i0 = int(np.floor(u))
    i1 = min(i0 + 1, _BLACKBODY_LUT.shape[0] - 1)
    f = u - i0
    return ((1.0 - f) * _BLACKBODY_LUT[i0] + f * _BLACKBODY_LUT[i1]).astype(np.float64)


def teff_to_bv(teff_k: float) -> float:
    """Approximate B–V from effective temperature (solar ~5778 K → ~0.65)."""
    t = float(np.clip(teff_k, 2800.0, 50000.0))
    if t > 9000.0:
        return float(np.clip(0.0 - 0.0004 * (t - 5778.0), -0.35, 0.1))
    return float(np.clip(0.65 + 0.00085 * (5778.0 - t), -0.2, 2.0))


def spectral_class_sample(rng: np.random.Generator) -> tuple[float, str]:
    idx = int(rng.choice(len(_SPECTRAL_TEFF), p=_SPECTRAL_WEIGHTS))
    names = ("O", "B", "A", "F", "G", "K", "M")
    teff = float(_SPECTRAL_TEFF[idx] * rng.uniform(0.92, 1.08))
    return teff, names[idx]


def sample_teffective_array(n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(_SPECTRAL_TEFF), size=n, p=_SPECTRAL_WEIGHTS)
    jitter = rng.uniform(0.92, 1.08, size=n)
    return (_SPECTRAL_TEFF[idx] * jitter).astype(np.float64)


def sample_teffective_for_placement(
    n: int,
    lat: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """IMF sample with disk warmth: fewer B-type pinpricks, more F/G/K in the plane."""
    lat = np.asarray(lat, dtype=np.float64)
    if lat.shape[0] != n:
        n = int(lat.shape[0])
    band_w = np.exp(-((lat / 0.38) ** 2))
    halo = np.clip(1.0 - band_w, 0.0, 1.0)[:, np.newaxis]
    disk = band_w[:, np.newaxis]
    w_halo = _SPECTRAL_WEIGHTS * np.array([0.55, 0.72, 0.92, 1.0, 1.05, 1.08, 1.06])
    w_disk = _SPECTRAL_WEIGHTS * np.array([0.12, 0.28, 0.62, 1.28, 1.58, 1.42, 1.18])
    weights = halo * w_halo + disk * w_disk
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    u = rng.random(n)
    cum = np.cumsum(weights, axis=1)
    idx = (u[:, np.newaxis] > cum).sum(axis=1)
    idx = np.clip(idx, 0, len(_SPECTRAL_TEFF) - 1)
    jitter = rng.uniform(0.92, 1.08, size=n)
    teff = (_SPECTRAL_TEFF[idx] * jitter).astype(np.float64)
    u2 = rng.random(n)
    rare_hot = (u2 < 0.007) & (band_w > 0.28)
    warm_disk = (u2 < 0.14) & (band_w > 0.32) & ~rare_hot
    teff = np.where(rare_hot, rng.uniform(12000.0, 26000.0, size=n), teff)
    teff = np.where(warm_disk, rng.uniform(4800.0, 7000.0, size=n), teff)
    return teff.astype(np.float64)


def adjust_teffective_for_population(
    teff: np.ndarray,
    gold_population: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Young patches → hotter Teff; mature patches → cooler (pairs with gold paint gating)."""
    t = np.asarray(teff, dtype=np.float64)
    gp = np.clip(np.asarray(gold_population, dtype=np.float64), 0.0, 1.0)
    if t.shape[0] != gp.shape[0]:
        return t
    u = rng.random(t.shape[0])
    young = gp < 0.30
    mature = gp > 0.58
    out = t.copy()
    hot_p = np.clip(0.44 + 0.48 * (1.0 - gp), 0.0, 0.92)
    hot_young = young & (t < 9600.0) & (u < hot_p)
    out = np.where(hot_young, rng.uniform(9200.0, 19500.0, size=t.shape[0]), out)
    u2 = rng.random(t.shape[0])
    cool_p = np.clip(0.38 + 0.58 * np.clip(gp - 0.42, 0.0, 1.0), 0.0, 0.88)
    cool_mature = mature & (out > 7600.0) & (u2 < cool_p)
    out = np.where(cool_mature, rng.uniform(4000.0, 6200.0, size=t.shape[0]), out)
    return out.astype(np.float64)


def adjust_bv_for_population(
    bv: np.ndarray,
    gold_population: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Young → bluer (lower B–V); old gold patches → warmer (higher B–V)."""
    b = np.asarray(bv, dtype=np.float64)
    gp = np.clip(np.asarray(gold_population, dtype=np.float64), 0.0, 1.0)
    if b.shape[0] != gp.shape[0]:
        return b
    young = gp < 0.28
    old = gp > 0.55
    jitter = rng.normal(0.0, 0.04, size=b.shape[0])
    shift = np.zeros_like(b)
    shift = np.where(young, shift - (0.10 + 0.22 * (1.0 - gp)), shift)
    shift = np.where(old, shift + (0.08 + 0.32 * gp), shift)
    return np.clip(b + shift + jitter, -0.35, 2.1).astype(np.float64)


def warm_teffective_for_core_bulge(
    teff_k: float,
    core_gold: float,
    rng: np.random.Generator,
) -> float:
    """Remap hot spectral types to F/G/K in the bulge (blue-white rare in core)."""
    cg = float(np.clip(core_gold, 0.0, 1.0))
    if cg < 0.10:
        return float(teff_k)
    t = float(teff_k)
    if cg > 0.50 and t > 6200.0:
        return float(rng.uniform(4400.0, 6000.0))
    if cg > 0.30 and t > 7200.0 and rng.random() < 0.82 + 0.15 * cg:
        return float(rng.uniform(4600.0, 6400.0))
    if cg > 0.18 and t > 8800.0 and rng.random() < 0.65 * cg:
        return float(rng.uniform(4800.0, 6600.0))
    return t


# Slight cool bias off-plane; disk stars use per-star warm shift in generator.
_CAMERA_WB = np.array([0.992, 1.0, 1.028], dtype=np.float64)
_CAMERA_SHOULDER = 0.14


def apply_camera_response_linear(
    rgb: np.ndarray,
    *,
    exposure: float = 1.0,
    white_balance: np.ndarray | None = None,
) -> np.ndarray:
    """Map linear blackbody/spectral RGB through a simple camera response (still linear)."""
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0) * float(max(exposure, 0.0))
    wb = _CAMERA_WB if white_balance is None else np.asarray(white_balance, dtype=np.float64).reshape(3)
    lin = lin * wb
    lin = lin / (1.0 + _CAMERA_SHOULDER * lin)
    return np.clip(lin, 0.0, None)


def star_chromatic_perturb_weight(
    mag: float,
    *,
    mag_bright: float,
    mag_faint: float,
    cutoff_mag: float = 9.8,
    gamma: float = 4.0,
) -> float:
    """0 = no hue/RGB jitter (faint); 1 = full perturbation (bright).

    Isolated sub-pixel stars with even tiny hue offsets read as saturated
    chromatic speckles; suppress perturbation below the cutoff magnitude.
    """
    if float(mag) >= float(cutoff_mag):
        return 0.0
    span = max(float(cutoff_mag) - float(mag_bright), 1e-6)
    bright_u = float(np.clip((float(cutoff_mag) - float(mag)) / span, 0.0, 1.0))
    return float(bright_u**gamma)


def attenuate_chroma_multipliers(
    factors: tuple[float, float, float],
    weight: float,
) -> tuple[float, float, float]:
    """Blend per-channel RGB factors toward neutral (1, 1, 1)."""
    w = float(np.clip(weight, 0.0, 1.0))
    return tuple(1.0 + (float(f) - 1.0) * w for f in factors)


def star_rgb_from_teffective(
    teff_k: float,
    jitter: np.ndarray,
    *,
    camera: bool = True,
) -> np.ndarray:
    """Planck → XYZ → linear sRGB with optional camera response (reduced neutral wash)."""
    rgb = blackbody_rgb(teff_k) + np.asarray(jitter, dtype=np.float64) * 0.35
    rgb = np.maximum(rgb, 0.0)
    peak = float(np.max(rgb))
    if peak > 1e-10:
        rgb = rgb / peak
    rgb = rgb * 0.975 + float(np.mean(rgb)) * 0.025
    if camera:
        rgb = apply_camera_response_linear(rgb)
    return np.clip(rgb, 0.0, 1.0)


def star_rgb_spectral(
    teff_k: float,
    jitter: np.ndarray,
    *,
    camera: bool = True,
) -> np.ndarray:
    """Alias for spectral star color path (Teff blackbody + camera)."""
    return star_rgb_from_teffective(teff_k, jitter, camera=camera)


def apply_white_balance(
    rgb: np.ndarray,
    warm_gain: np.ndarray,
    cool_gain: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Per-pixel mix warm/cool multipliers by mask in [0,1]."""
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim == 2:
        m = m[..., np.newaxis]
    w = np.asarray(warm_gain, dtype=np.float64).reshape(1, 1, 3)
    c = np.asarray(cool_gain, dtype=np.float64).reshape(1, 1, 3)
    out = rgb * (1.0 - m) * c + rgb * m * w
    return np.clip(out, 0.0, None)


# Effective wavelengths (Å) for linear sRGB R, G, B channels.
_CCM_WL_ANGSTROM = np.array([7000.0, 5500.0, 4400.0], dtype=np.float64)


def ccm89_albedo_per_ebv(wavelength_angstrom: float, rv: float = 3.1) -> float:
    """A(λ) / E(B-V) from Cardelli, Clayton & Mathis (1989), ApJ 345, 245."""
    lam = float(wavelength_angstrom)
    if lam <= 0.0:
        return 0.0
    x = 1.0e4 / lam - 1.82
    if x < 0.3:
        a, b = 0.0, 0.0
    elif x < 1.1:
        a = 0.574 * x**1.61
        b = -0.527 * x**1.61
    elif x < 3.3:
        y = x - 1.82
        a = (
            1.0
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
        )
        b = (
            1.41338 * y
            + 2.28338 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
        )
    else:
        y = x - 5.9
        a = 1.752 - 0.316 * x - 0.104 / ((x - 4.0) + 0.609 / ((x - 4.67) ** 2 + 1e-12))
        b = (
            -3.090
            + 1.825 * x
            + 1.206 / ((x - 4.62) ** 2 + 1e-12)
            + 0.213 / ((x - 3.98) ** 2 + 1e-12)
        )
    return float(a + b / rv)


_CCM_AL_EBV = np.array(
    [ccm89_albedo_per_ebv(float(w), 3.1) for w in _CCM_WL_ANGSTROM],
    dtype=np.float64,
)


def ccm_transmission_from_av(av_map: np.ndarray, *, rv: float = 3.1) -> np.ndarray:
    """Per-pixel linear RGB transmission T = 10^(-0.4 * A(λ)), A = (Aλ/E(B-V))·(A_V/R_V)."""
    av = np.clip(np.asarray(av_map, dtype=np.float64), 0.0, 12.0)
    if av.ndim == 2:
        ebv = av / rv
        al = _CCM_AL_EBV if abs(rv - 3.1) < 1e-6 else np.array(
            [ccm89_albedo_per_ebv(float(w), rv) for w in _CCM_WL_ANGSTROM],
            dtype=np.float64,
        )
        return np.power(10.0, -0.4 * al * ebv[..., np.newaxis])
    raise ValueError("av_map must be 2D")


def ccm89_transmission(rgb: np.ndarray, a_v: float, *, rv: float = 3.1) -> np.ndarray:
    """Apply CCM extinction: multiply linear RGB by 10^(-0.4 * A(λ)), A = (Aλ/EBV) * E(B-V)."""
    av = float(np.clip(a_v, 0.0, 12.0))
    if av < 1e-8:
        return np.asarray(rgb, dtype=np.float64)
    ebv = av / rv
    al_ebv = np.array([ccm89_albedo_per_ebv(w, rv) for w in _CCM_WL_ANGSTROM], dtype=np.float64)
    att = np.power(10.0, -0.4 * al_ebv * ebv)
    out = np.asarray(rgb, dtype=np.float64) * att
    return np.clip(out, 0.0, None)


def ccm89_v_band_transmission(a_v: float, *, rv: float = 3.1) -> float:
    """Flux attenuation at ~5500 Å for a given A_V."""
    av = float(np.clip(a_v, 0.0, 12.0))
    if av < 1e-8:
        return 1.0
    al_v = ccm89_albedo_per_ebv(5500.0, rv)
    return float(10.0 ** (-0.4 * al_v * av / rv))


def extinction_redden(rgb: np.ndarray, a_v: float, *, rv: float = 3.1) -> np.ndarray:
    """Dust reddening via CCM89; ``a_v`` is visual extinction A_V in magnitudes."""
    return ccm89_transmission(rgb, a_v, rv=rv)


def extinction_from_transmission(transmission: float, *, strength: float = 1.0) -> float:
    """Map a 0–1 transmission field sample to A_V (scaled by strength)."""
    t = float(np.clip(transmission, 1e-4, 1.0))
    return float(-2.5 * np.log10(t) * max(strength, 0.0))


def rec709_luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def remap_luma_preserving_chroma(
    rgb: np.ndarray,
    luma_out: np.ndarray,
    *,
    floor: float = 1e-10,
) -> np.ndarray:
    """Set Rec.709 luminance while preserving per-pixel RGB ratios (chromaticity)."""
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    l_out = np.maximum(np.asarray(luma_out, dtype=np.float64), 0.0)
    l_in = np.maximum(rec709_luma(lin), floor)
    scale = np.divide(l_out, l_in, out=np.zeros_like(l_out), where=l_in > floor)
    return (lin * scale[..., np.newaxis]).astype(np.float64)


def blend_darken_preserve_contrast(
    base: np.ndarray,
    factor: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    chroma_preserve: float = 0.72,
) -> np.ndarray:
    """Darken by factor while partially preserving local chroma vs luma."""
    b = np.asarray(base, dtype=np.float64)
    f = np.asarray(factor, dtype=np.float64)
    if f.ndim == 2:
        f = f[..., np.newaxis]
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        if m.ndim == 2:
            m = m[..., np.newaxis]
        f = 1.0 + (f - 1.0) * m
    l0 = rec709_luma(b)
    l1 = np.clip(l0 * np.squeeze(f, axis=-1) if f.shape[-1] == 1 else l0 * f[..., 0], 0.0, None)
    scale = np.where(l0 > 1e-8, l1 / (l0 + 1e-8), 0.0)
    if f.ndim == 3 and f.shape[-1] == 1:
        out = b * scale[..., np.newaxis]
    else:
        out = b * f
    # Chroma preservation: blend toward original chroma ratio
    mean_c = np.mean(out, axis=-1, keepdims=True)
    mean_b = np.mean(b, axis=-1, keepdims=True)
    chroma_b = b - mean_b
    chroma_out = out - mean_c
    out = mean_c + chroma_b * chroma_preserve + chroma_out * (1.0 - chroma_preserve)
    return np.clip(out, 0.0, None)
