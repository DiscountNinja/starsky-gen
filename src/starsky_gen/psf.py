"""ROI Moffat core + Gaussian halo stamping for HDR stars (periodic X)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypedDict

import numpy as np

from starsky_gen.color_science import (
    attenuate_chroma_multipliers,
    star_chromatic_perturb_weight,
)


class PsfTier(Enum):
    """Depth / optical tier for stellar PSF stamping."""

    BACKGROUND = "background"
    MID = "mid"
    FOREGROUND = "foreground"


class MoffatParams(TypedDict):
    sigma_x: float
    sigma_y: float
    theta: float
    beta: float


def moffat_fwhm_px(sigma: float, beta: float) -> float:
    """Moffat FWHM in pixels from scale σ and β."""
    b = max(float(beta), 1.05)
    s = max(float(sigma), 1e-6)
    return 2.0 * s * float(np.sqrt(2.0 ** (1.0 / b) - 1.0))


def moffat_sigma_from_fwhm(fwhm_px: float, beta: float) -> float:
    """Invert ``moffat_fwhm_px`` for target FWHM."""
    b = max(float(beta), 1.05)
    f = max(float(fwhm_px), 1e-6)
    return f / (2.0 * float(np.sqrt(2.0 ** (1.0 / b) - 1.0)))


@dataclass(frozen=True)
class PsfTuning:
    sigma_core_min: float = 0.42
    sigma_core_max: float = 2.05
    beta_default: float = 3.0
    beta_min: float = 2.5
    beta_max: float = 4.0
    fwhm_base_px: float = 1.32
    fwhm_ref_mag: float = 8.0
    fwhm_mag_coeff: float = 0.06
    halo_sigma_bright: float = 2.8
    halo_sigma_faint: float = 1.6
    halo_amp: float = 0.012
    halo_flux_floor: float = 52.0
    ultra_halo_flux_cut: float = 420.0
    fwhm_bright_jitter_sigma: float = 0.22
    fwhm_faint_jitter_half: float = 0.06
    bloom_sigma: float = 2.2
    bloom_amp: float = 0.010
    m_bloom_cut: float = 7.8
    mag_radius_gamma: float = 0.10
    max_half_px: int = 28
    mag_bright: float = 8.0
    mag_faint: float = 20.0
    spike_amp: float = 0.06
    spike_length_px: float = 18.0
    core_sigma_scale: float = 0.36
    core_beta: float = 3.2
    core_saturation_flux: float = 165.0
    core_flux_frac_bright: float = 0.58
    core_flux_frac_faint: float = 0.14
    shelf_sigma_scale: float = 0.68
    shelf_flux_frac: float = 0.16
    wing_flux_frac: float = 0.008
    wing_sigma_px: float = 5.5
    wide_halo_sigma_px: float = 6.5
    exp_halo_scale_px: float = 9.0
    chromatic_sigma_blue: float = 0.88
    chromatic_sigma_red: float = 1.10
    spike_bleed_amp: float = 0.085
    spike_bleed_length_px: float = 22.0
    faint_core_sigma_scale: float = 0.72
    tier_background_sigma_scale: float = 0.68
    tier_mid_sigma_scale: float = 1.0
    tier_foreground_sigma_scale: float = 1.12
    tier_background_max_half_px: int = 14
    tier_mid_max_half_px: int = 22
    tier_foreground_max_half_px: int = 32
    sensor_aperture_mm: float = 50.0
    sensor_focal_mm: float = 85.0
    ultra_bloom_flux_mult: float = 1.12
    core_clamp_bright_scale: float = 0.62


def flux_from_mag(mag: float, ref_mag: float) -> float:
    """Approximate Vega-linear flux in arbitrary HDR units."""
    return float(np.clip(10 ** (-0.4 * float(mag - ref_mag)), 1e-8, 1e7))


def resolve_psf_tier(
    *,
    layer: Literal["background", "mid", "foreground"] | None = None,
    tier: PsfTier | None = None,
) -> PsfTier:
    if tier is not None:
        return tier
    if layer == "foreground":
        return PsfTier.FOREGROUND
    if layer == "mid":
        return PsfTier.MID
    return PsfTier.BACKGROUND


def aperture_sigma_scale(tuning: PsfTuning | None = None) -> float:
    """PSF width scales ~√(f/#) from aperture and focal length (reference 50 mm @ f/1.7)."""
    t = PsfTuning() if tuning is None else tuning
    f_num = max(float(t.sensor_focal_mm) / max(float(t.sensor_aperture_mm), 1e-3), 0.5)
    ref_f = 85.0 / 50.0
    return float(np.sqrt(f_num / ref_f))


def _tier_scales(tier: PsfTier, t: PsfTuning) -> tuple[float, int]:
    if tier == PsfTier.FOREGROUND:
        return float(t.tier_foreground_sigma_scale), int(t.tier_foreground_max_half_px)
    if tier == PsfTier.MID:
        return float(t.tier_mid_sigma_scale), int(t.tier_mid_max_half_px)
    return float(t.tier_background_sigma_scale), int(t.tier_background_max_half_px)


def chromatic_sigma_scales(
    *,
    bv: float | None = None,
    teff_k: float | None = None,
    tuning: PsfTuning | None = None,
) -> tuple[float, float]:
    """Blue stars slightly sharper; red stars slightly broader (σx, σy scales)."""
    t = PsfTuning() if tuning is None else tuning
    if teff_k is not None:
        blue_u = float(np.clip((float(teff_k) - 3800.0) / 9000.0, 0.0, 1.0))
    else:
        bvv = 0.65 if bv is None else float(bv)
        blue_u = float(np.clip((0.88 - bvv) / 1.15, 0.0, 1.0))
    sx = float(t.chromatic_sigma_blue + (1.0 - blue_u) * (t.chromatic_sigma_red - t.chromatic_sigma_blue))
    sy = sx * 0.98
    return sx, sy


def _mag_span_fraction(mag: float, mag_bright: float, mag_faint: float) -> float:
    """0 = brightest end of range, 1 = faintest."""
    span = float(max(mag_faint - mag_bright, 1e-6))
    return float(np.clip((float(mag) - mag_bright) / span, 0.0, 1.0))


def moffat_params_from_mag_and_flux(
    mag: float,
    *,
    flux: float,
    galactic_lat_rad: float | None,
    rng: np.random.Generator,
    tuning: PsfTuning | None = None,
    tier: PsfTier = PsfTier.BACKGROUND,
    fwhm_scale: float | None = None,
) -> tuple[MoffatParams, int, float, float]:
    """Core Moffat params, patch half-span, optional halo (sigma_px, rel_amp)."""
    t = PsfTuning() if tuning is None else tuning
    tier_sigma, tier_max_half = _tier_scales(tier, t)
    ap_scale = aperture_sigma_scale(t)
    faint_u = _mag_span_fraction(mag, t.mag_bright, t.mag_faint)
    bright_u = 1.0 - faint_u
    beta = float(
        np.clip(
            t.core_beta + bright_u * 0.25 + faint_u * (t.beta_default - t.core_beta),
            t.beta_min,
            t.beta_max,
        )
    )
    fs = float(fwhm_scale) if fwhm_scale is not None else sample_psf_fwhm_scale(
        rng, mag, flux, tuning=t
    )
    mag_nudge = 1.0 + t.fwhm_mag_coeff * max(0.0, float(t.fwhm_ref_mag) - float(mag)) * 0.35
    fwhm_px = float(t.fwhm_base_px) * fs * mag_nudge
    if bright_u > 0.42:
        jitter_sig = float(t.fwhm_bright_jitter_sigma) * (0.25 + 0.45 * bright_u)
        fwhm_px *= float(rng.lognormal(0.0, jitter_sig))
    elif faint_u > 0.35:
        jh = float(t.fwhm_faint_jitter_half)
        fwhm_px *= float(rng.uniform(1.0 - jh, 1.0 + jh))
    sx = moffat_sigma_from_fwhm(fwhm_px, beta)
    sy = sx * float(np.clip(rng.uniform(0.90, 0.98), 0.82, 1.0))
    if galactic_lat_rad is not None:
        plane = float(np.exp(-((galactic_lat_rad / 0.41) ** 2)))
        sx *= 1.0 + 0.04 * plane
        sy /= 1.0 + 0.02 * plane
    flux_scale = float(np.clip(flux, 1e-6, None)) ** float(t.mag_radius_gamma)
    sx *= flux_scale ** (0.012 + 0.010 * faint_u)
    sy *= flux_scale ** (0.010 + 0.009 * faint_u)
    sx *= 1.0 + 0.012 * bright_u
    sy *= 1.0 + 0.010 * bright_u
    sx *= tier_sigma * ap_scale
    sy *= tier_sigma * ap_scale
    if tier == PsfTier.BACKGROUND:
        sx *= float(t.faint_core_sigma_scale)
        sy *= float(t.faint_core_sigma_scale)
    sx *= float(rng.uniform(0.92, 1.08))
    sy *= float(rng.uniform(0.92, 1.08))
    theta = float(rng.uniform(-0.55, 0.55))
    max_half = min(int(t.max_half_px), tier_max_half)
    pr = int(
        np.clip(
            3.0 + sx * np.sqrt(max(beta, 2.0)) * 2.15,
            4 if tier == PsfTier.BACKGROUND else 6,
            max_half,
        )
    )
    halo_sigma = float(t.halo_sigma_faint + bright_u * (t.halo_sigma_bright - t.halo_sigma_faint))
    halo_sigma = min(halo_sigma, 3.4)
    return (
        {"sigma_x": sx, "sigma_y": sy, "theta": theta, "beta": beta},
        pr,
        halo_sigma,
        0.0,
    )


def psf_halo_tiers(
    flux: float,
    mag: float,
    tuning: PsfTuning | None = None,
) -> tuple[float, float, float, bool]:
    """Halo tiers: tight inner exp (0.5–1%), noise-masked outer wing (0.2–0.6%), ultra wide frac."""
    t = PsfTuning() if tuning is None else tuning
    if flux < t.halo_flux_floor * 0.85:
        return 0.0, 0.0, 0.0, False
    bright_u = 1.0 - _mag_span_fraction(mag, t.mag_bright, t.mag_faint)
    logf = float(np.log10(max(flux, 1.0)))
    inner_rel = float(0.005 + 0.005 * bright_u)
    inner_rel = min(inner_rel, 0.010)
    outer_rel = float(0.002 + 0.004 * np.clip((logf - 1.05) / 1.25, 0.0, 1.0))
    outer_rel = min(outer_rel, 0.006)
    wide_frac = 0.0
    if flux >= t.ultra_halo_flux_cut:
        wide_frac = float(0.006 + 0.008 * np.clip((logf - 2.50) / 0.80, 0.0, 1.0))
        wide_frac = min(wide_frac, 0.014)
    allow_bloom = flux >= t.ultra_halo_flux_cut * 1.4 and mag < t.m_bloom_cut
    return inner_rel, outer_rel, wide_frac, allow_bloom


def build_normalized_moffat_kernel(half: int, params: MoffatParams) -> np.ndarray:
    """Discrete Moffat on an odd (2*half+1)² grid, normalized so sum(weights)=1."""
    w = half * 2 + 1
    yy = np.linspace(-half, half, w, dtype=np.float64)
    xx = np.linspace(-half, half, w, dtype=np.float64)
    xg = xx[None, :]
    yg = yy[:, None]
    st = np.sin(params["theta"])
    ct = np.cos(params["theta"])
    xr = xg * ct + yg * st
    yr = -xg * st + yg * ct
    rn2 = ((xr / max(params["sigma_x"], 0.055)) ** 2) + (
        (yr / max(params["sigma_y"], 0.055)) ** 2
    )
    k = (1.0 + rn2) ** (-params["beta"])
    s = np.sum(np.maximum(k, 0.0)) + 1e-14
    return np.maximum(k / s, 0.0).astype(np.float64)


def build_normalized_gaussian_kernel(half: int, sigma: float) -> np.ndarray:
    """Odd-sized Gaussian kernel normalized to sum 1."""
    w = half * 2 + 1
    yy = np.linspace(-half, half, w, dtype=np.float64)
    xx = np.linspace(-half, half, w, dtype=np.float64)
    xg = xx[None, :]
    yg = yy[:, None]
    sg = max(float(sigma), 0.12)
    k = np.exp(-(xg**2 + yg**2) / (2.0 * sg**2))
    s = np.sum(k) + 1e-14
    return np.maximum(k / s, 0.0).astype(np.float64)


kernel_cache_moffat: dict[tuple[int, int, float, float, float], np.ndarray] = {}
kernel_cache_gauss: dict[tuple[int, float], np.ndarray] = {}


def _moffat_key(half: int, params: MoffatParams) -> tuple[int, int, float, float, float]:
    th = round(float(params["theta"]) * 18.0) / 18.0
    return (
        half,
        int(round(params["beta"] * 10)),
        round(params["sigma_x"] * 8) / 8,
        round(params["sigma_y"] * 8) / 8,
        th,
    )


def normalized_moffat_kernel_cached(half: int, params: MoffatParams) -> np.ndarray:
    ky = _moffat_key(half, params)
    k = kernel_cache_moffat.get(ky)
    if k is None:
        k = build_normalized_moffat_kernel(half, params)
        kernel_cache_moffat[ky] = k
    return k


def normalized_gaussian_kernel_cached(half: int, sigma: float) -> np.ndarray:
    sg = round(float(sigma) * 4) / 4
    ky = (half, sg)
    k = kernel_cache_gauss.get(ky)
    if k is None:
        k = build_normalized_gaussian_kernel(half, sg)
        kernel_cache_gauss[ky] = k
    return k


def build_normalized_exponential_kernel(half: int, scale_px: float) -> np.ndarray:
    """Faint exponential halo for very bright stars."""
    w = half * 2 + 1
    yy = np.linspace(-half, half, w, dtype=np.float64)
    xx = np.linspace(-half, half, w, dtype=np.float64)
    r = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2)
    sc = max(float(scale_px), 0.8)
    k = np.exp(-r / sc)
    s = np.sum(k) + 1e-14
    return np.maximum(k / s, 0.0).astype(np.float64)


kernel_cache_exp: dict[tuple[int, float], np.ndarray] = {}


def normalized_exponential_kernel_cached(half: int, scale_px: float) -> np.ndarray:
    sc = round(float(scale_px) * 2) / 2
    ky = (half, sc)
    k = kernel_cache_exp.get(ky)
    if k is None:
        k = build_normalized_exponential_kernel(half, sc)
        kernel_cache_exp[ky] = k
    return k


def build_spike_kernel(half: int, length_px: float, theta: float, amp: float) -> np.ndarray:
    """Four faint diffraction arms."""
    w = half * 2 + 1
    yy = np.linspace(-half, half, w, dtype=np.float64)
    xx = np.linspace(-half, half, w, dtype=np.float64)
    xg = xx[None, :]
    yg = yy[:, None]
    k = np.zeros((w, w), dtype=np.float64)
    for ang in (theta, theta + np.pi / 2):
        ct, st = np.cos(ang), np.sin(ang)
        proj = np.abs(xg * ct + yg * st)
        arm = np.exp(-(proj**2) / (2.0 * (0.35 + length_px * 0.02) ** 2))
        arm *= np.exp(-proj / max(length_px, 2.0))
        k += arm
    s = np.sum(k) + 1e-14
    return (k / s * amp).astype(np.float64)


def build_asymmetric_spike_bleed_kernel(
    half: int,
    length_px: float,
    theta: float,
    *,
    spike_amp: float,
    bleed_amp: float,
    bleed_elong: float = 2.6,
) -> np.ndarray:
    """Asymmetric diffraction spikes + saturated sensor bleed along primary axis."""
    w = half * 2 + 1
    yy = np.linspace(-half, half, w, dtype=np.float64)
    xx = np.linspace(-half, half, w, dtype=np.float64)
    xg = xx[None, :]
    yg = yy[:, None]
    k = np.zeros((w, w), dtype=np.float64)
    arm_w = (1.0, 0.72, 0.88, 0.64)
    for i, ang in enumerate((theta, theta + np.pi / 2, theta + np.pi, theta + 1.5 * np.pi)):
        ct, st = np.cos(ang), np.sin(ang)
        proj = xg * ct + yg * st
        arm = np.exp(-(proj**2) / (2.0 * (0.28 + length_px * 0.018) ** 2))
        arm *= np.exp(-np.abs(proj) / max(length_px, 2.0))
        k += arm * arm_w[i % 4]
    ct, st = np.cos(theta), np.sin(theta)
    xr = xg * ct + yg * st
    yr = -xg * st + yg * ct
    bleed = np.exp(-(xr**2) / (2.0 * (length_px * 0.22 * bleed_elong) ** 2))
    bleed *= np.exp(-(yr**2) / (2.0 * (length_px * 0.09) ** 2))
    k += bleed * bleed_amp
    s = np.sum(k) + 1e-14
    return (k / s * spike_amp).astype(np.float64)


def stamp_psf_patch(
    canvas: np.ndarray,
    yi: int,
    xi: int,
    kern: np.ndarray,
    amplitude_rgb: np.ndarray,
    *,
    periodic_x: bool,
    subpixel_dx: float = 0.0,
    subpixel_dy: float = 0.0,
) -> None:
    """Add kernel * amplitude_rgb (per-channel) into float RGB canvas."""
    ampl = np.ascontiguousarray(amplitude_rgb, dtype=np.float64).reshape((1, 1, -1))
    patch = kern[..., np.newaxis] * ampl
    h, wd, _cc = canvas.shape
    kh, kw = kern.shape
    cy = kh // 2
    cx = kw // 2
    iy0 = float(yi - cy) + float(subpixel_dy)
    ix0 = float(xi - cx) + float(subpixel_dx)
    yi_row = iy0 + np.arange(kh, dtype=np.float64)
    xi_col = ix0 + np.arange(kw, dtype=np.float64)
    yy_grid, xx_grid = np.meshgrid(yi_row, xi_col, indexing="ij")
    vy = (yy_grid >= 0) & (yy_grid < h)
    yy_i = np.clip(np.round(yy_grid).astype(np.int64), 0, h - 1)
    if periodic_x:
        xp_grid = np.mod(np.round(xx_grid).astype(np.int64), wd)
        ii, jj = np.nonzero(vy)
        canvas[yy_i[ii, jj], xp_grid[ii, jj]] += patch[ii, jj]
    else:
        sel = vy & (xx_grid >= 0) & (xx_grid < wd)
        ii, jj = np.nonzero(sel)
        canvas[yy_i[ii, jj], np.clip(np.round(xx_grid[ii, jj]).astype(np.int64), 0, wd - 1)] += patch[
            ii, jj
        ]


def _split_psf_flux(
    flux: float,
    mag: float,
    t: PsfTuning,
    *,
    tier: PsfTier = PsfTier.BACKGROUND,
) -> tuple[float, float, float]:
    """Partition flux into tight saturated core, shelf, and extended wing."""
    faint_u = _mag_span_fraction(mag, t.mag_bright, t.mag_faint)
    bright_u = 1.0 - faint_u
    core_frac = float(t.core_flux_frac_faint + bright_u * (t.core_flux_frac_bright - t.core_flux_frac_faint))
    shelf_frac = float(t.shelf_flux_frac * (0.35 + 0.65 * bright_u))
    core_cap = float(t.core_saturation_flux)
    if bright_u > 0.55:
        core_cap *= float(t.core_clamp_bright_scale + (1.0 - t.core_clamp_bright_scale) * (1.0 - bright_u))
    if tier == PsfTier.FOREGROUND:
        core_frac = min(core_frac * 1.12, 0.52)
        core_cap *= 1.08
    elif tier == PsfTier.BACKGROUND:
        core_frac *= 0.88
        core_cap *= 0.92
    core_flux = min(float(flux) * core_frac, core_cap)
    shelf_flux = float(flux) * shelf_frac
    wing_floor = float(flux) * (0.10 + 0.14 * faint_u)
    wing_flux = max(float(flux) - core_flux - shelf_flux, wing_floor)
    return core_flux, shelf_flux, wing_flux


class StarPsfStyle(Enum):
    """Per-star optical family (wide-field stacks mix these)."""

    PINPRICK = "pinprick"
    STANDARD = "standard"
    SOFT_SEEING = "soft_seeing"
    SATURATED = "saturated"
    EXTINCTED = "extincted"
    STACK_TWIN = "stack_twin"


def sample_psf_fwhm_scale(
    rng: np.random.Generator,
    mag: float,
    flux: float,
    *,
    tuning: PsfTuning | None = None,
) -> float:
    """Angular size lottery: many tiny, few medium, rare bloated/saturated (weakly tied to mag/flux)."""
    t = PsfTuning() if tuning is None else tuning
    u = float(rng.random())
    logf = float(np.log10(max(float(flux), 1.0)))
    bright_u = 1.0 - _mag_span_fraction(mag, t.mag_bright, t.mag_faint)
    if u < 0.84:
        return float(rng.uniform(0.42, 0.68))
    if u < 0.95:
        return float(rng.uniform(0.86, 0.95))
    if float(mag) >= 10.0:
        return float(rng.uniform(0.86, 0.95))
    if u < 0.985:
        scale = float(rng.uniform(0.96, 1.18))
    else:
        if logf < 1.75 and bright_u < 0.55:
            scale = float(rng.uniform(0.96, 1.18))
        else:
            scale = float(rng.uniform(1.18, 1.55))
    if bright_u > 0.72 and u > 0.90 and logf > 2.0:
        scale = max(scale, float(rng.uniform(1.22, 1.62)))
    return scale


@dataclass(frozen=True)
class StarPsfVariety:
    """Stamp-time PSF modifiers sampled per star."""

    style: StarPsfStyle = StarPsfStyle.STANDARD
    fwhm_scale: float = 1.0
    sigma_scale: float = 1.0
    halo_scale: float = 1.0
    bloom_scale: float = 1.0
    core_sat_scale: float = 1.0
    wing_scale: float = 1.0
    chroma_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    stack_dx: float = 0.0
    stack_dy: float = 0.0
    stack_flux_frac: float = 0.0


def sample_star_psf_variety(
    rng: np.random.Generator,
    mag: float,
    flux: float,
    *,
    teff_k: float | None = None,
    bv: float | None = None,
    extinction_t: float = 0.0,
    local_density: float = 0.0,
    galactic_lat_rad: float | None = None,
    psf_environment: float = 1.0,
    layer: Literal["background", "mid", "foreground"] | None = None,
    tuning: PsfTuning | None = None,
) -> StarPsfVariety:
    """Pick an optical family and scales so stars are not one uniform PSF regime."""
    t = PsfTuning() if tuning is None else tuning
    faint_u = _mag_span_fraction(mag, t.mag_bright, t.mag_faint)
    bright_u = 1.0 - faint_u
    ext_u = float(np.clip(extinction_t, 0.0, 1.0))
    dens_u = float(np.clip(local_density, 0.0, 1.0))
    hot = (teff_k is not None and float(teff_k) > 8800.0) or (bv is not None and float(bv) < 0.18)
    plane_u = 0.0
    if galactic_lat_rad is not None:
        plane_u = float(np.exp(-((float(galactic_lat_rad) / 0.41) ** 2)))

    w = np.array(
        [
            0.38 + 0.14 * bright_u + (0.16 if hot else 0.0) - 0.14 * ext_u,
            0.40 - 0.10 * bright_u,
            0.06 + 0.10 * ext_u + 0.06 * dens_u + 0.04 * plane_u,
            0.03 + 0.10 * bright_u,
            0.06 + 0.18 * ext_u + 0.05 * dens_u,
            0.03 + 0.05 * bright_u + 0.03 * plane_u,
        ],
        dtype=np.float64,
    )
    w = np.maximum(w, 1e-6)
    w /= float(np.sum(w))
    env_u = float(np.clip(psf_environment, 0.35, 1.35))
    fg_u = 1.0 if layer == "foreground" else (0.55 if layer == "mid" else 0.0)
    scores = w.copy()
    scores[0] += bright_u * 0.22 + (1.0 - env_u) * 0.04
    scores[1] += (1.0 - bright_u) * 0.12 + dens_u * 0.08
    scores[2] += ext_u * 0.18 + (1.0 - env_u) * 0.10
    scores[3] += bright_u * 0.20
    scores[4] += ext_u * 0.22 + dens_u * 0.10
    scores[5] += plane_u * 0.08 + fg_u * 0.12
    scores /= float(np.sum(scores))
    tie = 0.12
    if rng.random() < tie:
        style = list(StarPsfStyle)[int(rng.choice(len(StarPsfStyle), p=w))]
    else:
        style = list(StarPsfStyle)[int(np.argmax(scores))]

    fwhm_scale = sample_psf_fwhm_scale(rng, mag, flux, tuning=t)
    sigma_scale = 1.0
    halo_scale = 1.0
    bloom_scale = 1.0
    core_sat_scale = 1.0
    wing_scale = 1.0
    stack_dx = 0.0
    stack_dy = 0.0
    stack_flux_frac = 0.0

    if style == StarPsfStyle.PINPRICK:
        sigma_scale = float(rng.uniform(0.82, 0.96))
        halo_scale = float(rng.uniform(0.78, 0.92))
        bloom_scale = 0.0
    elif style == StarPsfStyle.SOFT_SEEING:
        sigma_scale = float(rng.uniform(1.12, 1.44))
        halo_scale = float(rng.uniform(1.10, 1.38))
        bloom_scale = float(rng.uniform(0.90, 1.22))
        wing_scale = float(rng.uniform(1.15, 1.45))
    elif style == StarPsfStyle.SATURATED:
        fwhm_scale = max(fwhm_scale, float(rng.uniform(1.42, 2.05)))
        sigma_scale = float(rng.uniform(0.92, 1.08))
        halo_scale = float(rng.uniform(1.05, 1.28))
        bloom_scale = float(rng.uniform(1.65, 2.55))
        core_sat_scale = float(rng.uniform(1.15, 1.42))
    elif style == StarPsfStyle.EXTINCTED:
        sigma_scale = float(rng.uniform(1.18, 1.62))
        halo_scale = float(rng.uniform(1.14, 1.48))
        bloom_scale = float(rng.uniform(0.55, 0.88))
        core_sat_scale = float(rng.uniform(0.72, 0.88))
        wing_scale = float(rng.uniform(1.08, 1.32))
    elif style == StarPsfStyle.STACK_TWIN:
        sigma_scale = float(rng.uniform(0.90, 1.10))
        halo_scale = float(rng.uniform(0.92, 1.08))
        stack_dx = float(rng.uniform(-0.52, 0.52))
        stack_dy = float(rng.uniform(-0.48, 0.48))
        stack_flux_frac = float(rng.uniform(0.30, 0.54))
    else:
        sigma_scale = float(rng.uniform(0.84, 1.18))
        halo_scale = float(rng.uniform(0.86, 1.20))
        bloom_scale = float(rng.uniform(0.72, 1.28))

    cj = 0.011 if style != StarPsfStyle.EXTINCTED else 0.008
    chroma = (
        float(np.clip(1.0 + rng.normal(0.0, cj), 0.90, 1.10)),
        float(np.clip(1.0 + rng.normal(0.0, cj * 0.85), 0.90, 1.10)),
        float(np.clip(1.0 + rng.normal(0.0, cj * 1.12), 0.90, 1.10)),
    )
    if style == StarPsfStyle.EXTINCTED:
        chroma = (chroma[0] * 1.045, chroma[1] * 1.012, chroma[2] * 0.958)
    elif hot and style == StarPsfStyle.PINPRICK:
        chroma = (chroma[0] * 0.985, chroma[1] * 0.992, chroma[2] * 1.018)

    chroma_w = star_chromatic_perturb_weight(mag, mag_bright=t.mag_bright, mag_faint=t.mag_faint)
    chroma = attenuate_chroma_multipliers(chroma, chroma_w**2)

    return StarPsfVariety(
        style=style,
        fwhm_scale=fwhm_scale,
        sigma_scale=sigma_scale,
        halo_scale=halo_scale,
        bloom_scale=bloom_scale,
        core_sat_scale=core_sat_scale,
        wing_scale=wing_scale,
        chroma_rgb=chroma,
        stack_dx=stack_dx,
        stack_dy=stack_dy,
        stack_flux_frac=stack_flux_frac,
    )


def _hero_point_star(
    mag: float,
    flux: float,
    *,
    teff_k: float | None,
    bv: float | None,
    tuning: PsfTuning,
) -> bool:
    """True for bright hot/blue stars (used for variety weighting, not auto PSF path)."""
    if float(flux) < tuning.halo_flux_floor * 0.75:
        return False
    if float(mag) > 8.2:
        return False
    if teff_k is not None and float(teff_k) > 8800.0:
        return True
    if bv is not None and float(bv) < 0.18:
        return True
    return False


def stamp_star_psf(
    canvas: np.ndarray,
    yi: int,
    xi: int,
    spectral_rgb: np.ndarray,
    flux: float,
    mag: float,
    *,
    galactic_lat_rad: float | None,
    rng: np.random.Generator,
    tuning: PsfTuning | None = None,
    periodic_x: bool = True,
    subpixel_dx: float = 0.0,
    subpixel_dy: float = 0.0,
    field_angle_rad: float | None = None,
    add_spikes: bool = False,
    bv: float | None = None,
    teff_k: float | None = None,
    tier: PsfTier = PsfTier.BACKGROUND,
    layer: Literal["background", "mid", "foreground"] | None = None,
    variety: StarPsfVariety | None = None,
) -> None:
    """Tight saturated core + shelf Moffat + Gaussian wings/halo/bloom/spikes."""
    t = PsfTuning() if tuning is None else tuning
    psf_tier = resolve_psf_tier(layer=layer, tier=tier)
    v = variety if variety is not None else StarPsfVariety()
    style = v.style
    hero = style == StarPsfStyle.PINPRICK
    params, hr, halo_sigma, halo_rel = moffat_params_from_mag_and_flux(
        mag,
        flux=flux,
        galactic_lat_rad=galactic_lat_rad,
        rng=rng,
        tuning=t,
        tier=psf_tier,
        fwhm_scale=float(v.fwhm_scale),
    )
    if field_angle_rad is not None:
        elong = 1.0 + 0.06 * min(abs(field_angle_rad), 1.2)
        params = dict(params)
        params["sigma_x"] = float(params["sigma_x"]) * elong
        params["sigma_y"] = float(params["sigma_y"]) / elong
    csx, csy = chromatic_sigma_scales(bv=bv, teff_k=teff_k, tuning=t)
    params = dict(params)
    params["sigma_x"] = float(params["sigma_x"]) * csx * float(v.sigma_scale)
    params["sigma_y"] = float(params["sigma_y"]) * csy * float(v.sigma_scale)
    halo_sigma *= float(v.halo_scale)
    spec = spectral_rgb.astype(np.float64)
    cr, cg, cb = v.chroma_rgb
    spec = spec * np.array([cr, cg, cb], dtype=np.float64)
    core_flux, shelf_flux, wing_flux = _split_psf_flux(flux, mag, t, tier=psf_tier)
    hero_seeing_flux = 0.0
    if hero:
        # Avoid single-pixel core saturation; most light in a small seeing disk.
        core_flux = min(float(flux) * 0.38, float(t.core_saturation_flux) * 0.55)
        hero_seeing_flux = float(flux) * 0.44
        shelf_flux = 0.0
        wing_flux = max(float(flux) - core_flux - hero_seeing_flux, float(flux) * 0.04)
    elif style == StarPsfStyle.SATURATED:
        core_flux = min(
            float(flux) * 0.56,
            float(t.core_saturation_flux) * float(v.core_sat_scale),
        )
        shelf_flux = float(flux) * 0.18
        wing_flux = max(float(flux) - core_flux - shelf_flux, float(flux) * 0.06)
    elif style == StarPsfStyle.EXTINCTED:
        core_flux *= float(v.core_sat_scale)
        shelf_flux *= 1.22
        wing_flux *= float(v.wing_scale)
    elif style == StarPsfStyle.SOFT_SEEING:
        shelf_flux *= 1.18
        wing_flux *= float(v.wing_scale)

    core_scale = t.core_sigma_scale * (0.88 if hero else 1.0)
    # Tight saturated core (small σ, high β for a flat-top shelf).
    core_params: MoffatParams = {
        "sigma_x": max(float(params["sigma_x"]) * core_scale, 0.48 if hero else 0.38),
        "sigma_y": max(float(params["sigma_y"]) * core_scale, 0.46 if hero else 0.36),
        "theta": params["theta"],
        "beta": float(np.clip(t.core_beta, t.beta_min, 3.2)),
    }
    h_core = int(
        np.clip(
            3.0 + core_params["sigma_x"] * np.sqrt(max(core_params["beta"], 2.0)) * 1.65,
            4,
            min(12, t.max_half_px),
        )
    )
    k_core = normalized_moffat_kernel_cached(h_core, core_params)
    stamp_psf_patch(
        canvas,
        yi,
        xi,
        k_core,
        spec * core_flux,
        periodic_x=periodic_x,
        subpixel_dx=subpixel_dx,
        subpixel_dy=subpixel_dy,
    )

    amp = spec * (wing_flux if not hero else max(wing_flux, float(flux) * 0.08))
    if not hero:
        # Intermediate shelf (non-linear bridge between core and wings).
        shelf_params: MoffatParams = {
            "sigma_x": max(float(params["sigma_x"]) * t.shelf_sigma_scale, 0.48),
            "sigma_y": max(float(params["sigma_y"]) * t.shelf_sigma_scale, 0.46),
            "theta": params["theta"],
            "beta": float(params["beta"]),
        }
        h_shelf = int(
            np.clip(
                4.0 + shelf_params["sigma_x"] * np.sqrt(max(shelf_params["beta"], 2.0)) * 1.85,
                5,
                min(16, t.max_half_px),
            )
        )
        k_shelf = normalized_moffat_kernel_cached(h_shelf, shelf_params)
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            k_shelf,
            spec * shelf_flux,
            periodic_x=periodic_x,
            subpixel_dx=subpixel_dx,
            subpixel_dy=subpixel_dy,
        )

        kern = normalized_moffat_kernel_cached(hr, params)
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            kern,
            amp,
            periodic_x=periodic_x,
            subpixel_dx=subpixel_dx,
            subpixel_dy=subpixel_dy,
        )
    if flux >= t.halo_flux_floor * 0.35 and not hero:
        blue_u = 0.5
        if teff_k is not None:
            blue_u = float(np.clip((float(teff_k) - 3800.0) / 9000.0, 0.0, 1.0))
        elif bv is not None:
            blue_u = float(np.clip((0.88 - float(bv)) / 1.15, 0.0, 1.0))
        wing_rgb = spec * float(flux) * t.wing_flux_frac * float(v.wing_scale)
        spec_n = spec / (np.maximum(np.max(spec), 1e-8))
        for sig_scale, chrom_bias in (
            (0.90 + 0.08 * blue_u, np.array([0.96, 0.98, 1.02], dtype=np.float64)),
            (1.05 + 0.10 * (1.0 - blue_u), np.array([1.02, 0.99, 0.97], dtype=np.float64)),
        ):
            sig = float(t.wing_sigma_px) * sig_scale
            wh = int(np.clip(round(sig * 1.75), 5, 22))
            ek = normalized_exponential_kernel_cached(wh, sig)
            wing_tint = spec_n * chrom_bias
            wing_tint = wing_tint / (np.maximum(np.max(wing_tint), 1e-8))
            stamp_psf_patch(
                canvas,
                yi,
                xi,
                ek,
                wing_rgb * wing_tint,
                periodic_x=periodic_x,
                subpixel_dx=subpixel_dx,
                subpixel_dy=subpixel_dy,
            )
    elif hero and hero_seeing_flux > 1e-6:
        # Pin-point core + thin seeing disk (σ≈0.9–1.2 px), not wide halos.
        sig = max(float(params["sigma_x"]) * 0.62, 0.78)
        wh = int(np.clip(round(sig * 1.95), 3, 6))
        ek = normalized_gaussian_kernel_cached(wh, sig)
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            ek,
            spec * hero_seeing_flux,
            periodic_x=periodic_x,
            subpixel_dx=subpixel_dx,
            subpixel_dy=subpixel_dy,
        )
    inner_rel, outer_rel, wide_frac, allow_bloom = psf_halo_tiers(flux, mag, t)
    inner_rel *= float(v.halo_scale)
    outer_rel *= float(v.halo_scale)
    wide_frac *= float(v.halo_scale)
    if hero:
        inner_rel = 0.0
        outer_rel = 0.0
        wide_frac = 0.0
        allow_bloom = False
    elif style == StarPsfStyle.SATURATED and flux >= t.halo_flux_floor * 0.45:
        allow_bloom = True
    if psf_tier == PsfTier.FOREGROUND and allow_bloom:
        allow_bloom = True
    elif psf_tier == PsfTier.BACKGROUND:
        allow_bloom = allow_bloom and mag < t.m_bloom_cut - 0.4
    noise_mod = float(rng.uniform(0.78, 1.22))
    if inner_rel > 1e-6:
        inner_scale = max(0.65, halo_sigma * 0.32)
        eh = int(np.clip(round(inner_scale * 2.4), 3, 9))
        ek = normalized_exponential_kernel_cached(eh, inner_scale)
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            ek,
            amp * inner_rel,
            periodic_x=periodic_x,
            subpixel_dx=subpixel_dx,
            subpixel_dy=subpixel_dy,
        )
    if outer_rel > 1e-6:
        oh = int(np.clip(round(halo_sigma * 2.15), 5, 14))
        ok = normalized_gaussian_kernel_cached(oh, halo_sigma * 1.18)
        wing_mask = noise_mod * float(rng.uniform(0.82, 1.18))
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            ok,
            amp * outer_rel * wing_mask,
            periodic_x=periodic_x,
            subpixel_dx=subpixel_dx,
            subpixel_dy=subpixel_dy,
        )
    if wide_frac > 1e-6:
        wh = int(np.clip(round(t.wide_halo_sigma_px * 1.25), 10, min(24, t.max_half_px + 6)))
        wk = normalized_gaussian_kernel_cached(wh, t.wide_halo_sigma_px)
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            wk,
            spec * float(flux) * wide_frac * noise_mod,
            periodic_x=periodic_x,
            subpixel_dx=subpixel_dx,
            subpixel_dy=subpixel_dy,
        )
    if allow_bloom:
        bloom_amp = t.bloom_amp * noise_mod * max(float(v.bloom_scale), 0.0)
        bloom_sig = t.bloom_sigma
        if psf_tier == PsfTier.FOREGROUND and not hero:
            bloom_amp *= float(t.ultra_bloom_flux_mult)
        if hero:
            bloom_amp *= 0.55
            bloom_sig *= 0.72
        elif style == StarPsfStyle.EXTINCTED:
            bloom_sig *= 1.12
        bh = int(np.clip(round(bloom_sig * (1.15 if psf_tier == PsfTier.FOREGROUND and not hero else 1.0)), 3, 9))
        bk = normalized_gaussian_kernel_cached(bh, bloom_sig)
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            bk,
            amp * bloom_amp,
            periodic_x=periodic_x,
            subpixel_dx=subpixel_dx,
            subpixel_dy=subpixel_dy,
        )
    do_spikes = add_spikes or (
        psf_tier == PsfTier.FOREGROUND and mag < t.m_bloom_cut and flux >= t.ultra_halo_flux_cut * 0.85
    )
    if do_spikes:
        sh = int(np.clip(round(t.spike_bleed_length_px), 10, t.max_half_px + 12))
        sk = build_asymmetric_spike_bleed_kernel(
            sh,
            t.spike_bleed_length_px,
            params["theta"],
            spike_amp=t.spike_amp,
            bleed_amp=t.spike_bleed_amp,
        )
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            sk,
            amp * 0.42,
            periodic_x=periodic_x,
            subpixel_dx=subpixel_dx,
            subpixel_dy=subpixel_dy,
        )
    if v.stack_flux_frac > 1e-6:
        twin_flux = float(flux) * float(v.stack_flux_frac)
        twin_core = min(twin_flux * 0.55, float(t.core_saturation_flux) * 0.48)
        twin_see = twin_flux - twin_core
        sig_t = max(float(params["sigma_x"]) * 0.58, 0.72)
        wh_t = int(np.clip(round(sig_t * 1.85), 3, 6))
        ek_t = normalized_gaussian_kernel_cached(wh_t, sig_t)
        tdx = float(subpixel_dx) + float(v.stack_dx)
        tdy = float(subpixel_dy) + float(v.stack_dy)
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            k_core,
            spec * twin_core,
            periodic_x=periodic_x,
            subpixel_dx=tdx,
            subpixel_dy=tdy,
        )
        stamp_psf_patch(
            canvas,
            yi,
            xi,
            ek_t,
            spec * twin_see,
            periodic_x=periodic_x,
            subpixel_dx=tdx,
            subpixel_dy=tdy,
        )


# Back-compat alias
stamp_star_moffat = stamp_star_psf
