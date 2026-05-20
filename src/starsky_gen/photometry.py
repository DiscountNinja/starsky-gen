"""Stellar IMF sampling and mass–luminosity → apparent magnitude projection."""

from __future__ import annotations

import numpy as np


def sample_salpeter_mass(
    n: int,
    rng: np.random.Generator,
    *,
    alpha: float = 2.35,
    m_lo: float = 0.10,
    m_hi: float = 50.0,
) -> np.ndarray:
    """Sample stellar mass (M☉) from dN/dM ∝ M^-α on [m_lo, m_hi]."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    lo = float(max(m_lo, 1e-4))
    hi = float(max(m_hi, lo * 1.001))
    u = rng.random(n)
    if abs(alpha - 1.0) < 1e-8:
        return lo * (hi / lo) ** u
    a = 1.0 - float(alpha)
    lo_a, hi_a = lo**a, hi**a
    return np.clip((u * (hi_a - lo_a) + lo_a) ** (1.0 / a), lo, hi).astype(np.float64)


def mass_to_abs_mag_main_sequence(mass_solar: np.ndarray) -> np.ndarray:
    """Rough ZAMS / main-sequence M_V from mass (M☉)."""
    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), 0.08)
    log_l = np.where(
        m < 0.43,
        2.0 * np.log10(m) - 0.35,
        -0.55 + 3.75 * np.log10(m),
    )
    return (4.83 - 2.5 * log_l).astype(np.float64)


def mass_to_abs_mag_giant(mass_solar: np.ndarray) -> np.ndarray:
    """Bright-giant / red-giant branch (rare tail)."""
    m = np.maximum(np.asarray(mass_solar, dtype=np.float64), 0.9)
    log_l = 0.85 + 1.65 * np.log10(m)
    return (4.83 - 2.5 * log_l).astype(np.float64)


def sample_distance_modulus(
    n: int,
    rng: np.random.Generator,
    *,
    mean: float = 11.4,
    sigma: float = 1.65,
    mu_min: float = 7.8,
    mu_max: float = 16.2,
) -> np.ndarray:
    """Galactic-disk distance modulus μ (mag) for apparent-magnitude projection."""
    mu = rng.normal(float(mean), float(sigma), size=n)
    return np.clip(mu, float(mu_min), float(mu_max)).astype(np.float64)


def _trim_ultra_bright_mags(
    mags: np.ndarray,
    rng: np.random.Generator,
    *,
    mag_bright: float,
    mag_faint: float,
    magnitude_ultra_cut: float,
    max_ultra_bright_stars: int,
) -> np.ndarray:
    """Keep at most ``max_ultra_bright_stars`` below ``magnitude_ultra_cut``."""
    ml = float(min(mag_bright, mag_faint))
    mh = float(max(mag_bright, mag_faint))
    cap = float(magnitude_ultra_cut)
    if max_ultra_bright_stars < 1:
        return mags.astype(np.float64)
    bright_idx = np.flatnonzero(mags < cap)
    excess = int(bright_idx.shape[0]) - int(max_ultra_bright_stars)
    if excess <= 0:
        return np.clip(mags, ml, mh).astype(np.float64)
    losers = bright_idx[np.argsort(mags[bright_idx])][:excess]
    eps = float((mh - ml) * 1e-6 + 5e-4)
    lo = float(np.minimum(cap + eps, mh - eps))
    mags = mags.copy()
    mags[losers] = rng.uniform(lo, mh, size=excess)
    return np.clip(mags, ml, mh).astype(np.float64)


def sample_apparent_magnitudes_lf(
    n_stars: int,
    rng: np.random.Generator,
    *,
    mag_bright: float,
    mag_faint: float,
    giant_fraction: float = 0.048,
    salpeter_alpha: float = 2.35,
    distance_modulus_mean: float = 11.4,
    distance_modulus_sigma: float = 1.65,
    magnitude_ultra_cut: float = 6.5,
    max_ultra_bright_stars: int = 6,
    lf_power_slope: float = 0.52,
) -> np.ndarray:
    """Salpeter IMF + distance modulus + Schechter-like bright tail (dN/dm power law)."""
    n = int(n_stars)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    ml = float(min(mag_bright, mag_faint))
    mh = float(max(mag_bright, mag_faint))
    k = np.log(10.0) * float(lf_power_slope)

    def _draw_truncated_power(n_draw: int, a: float, b: float) -> np.ndarray:
        if abs(k) < 1e-14:
            return rng.uniform(a, b, size=n_draw)
        u = rng.random(n_draw)
        ec_lo, ec_hi = np.exp(k * a), np.exp(k * b)
        return np.clip(np.log(u * (ec_hi - ec_lo) + ec_lo) / k, a, b)

    mags = _draw_truncated_power(n, ml, mh)
    is_giant = rng.random(n) < float(np.clip(giant_fraction, 0.0, 0.25))
    m_ms = sample_salpeter_mass(n, rng, alpha=salpeter_alpha, m_lo=0.12, m_hi=12.0)
    m_gi = sample_salpeter_mass(n, rng, alpha=1.75, m_lo=1.2, m_hi=35.0)
    mass = np.where(is_giant, m_gi, m_ms)
    m_abs = np.where(
        is_giant,
        mass_to_abs_mag_giant(mass),
        mass_to_abs_mag_main_sequence(mass),
    )
    mu = sample_distance_modulus(
        n, rng, mean=distance_modulus_mean, sigma=distance_modulus_sigma
    )
    imf_mags = np.clip(m_abs + mu, ml - 2.0, mh + 0.5)
    # Blend IMF draw with LF power law (physical mass–luminosity + field counts).
    blend = rng.uniform(0.38, 0.62, size=n)
    mags = mags * (1.0 - blend) + imf_mags * blend
    bright_draw = rng.random(n) < 0.022
    if np.any(bright_draw):
        hi_b = max(ml + 1e-4, float(magnitude_ultra_cut) + 0.35)
        mags[bright_draw] = rng.uniform(ml, hi_b, int(np.sum(bright_draw)))
    mags = np.clip(mags, ml, mh)
    return _trim_ultra_bright_mags(
        mags,
        rng,
        mag_bright=ml,
        mag_faint=mh,
        magnitude_ultra_cut=magnitude_ultra_cut,
        max_ultra_bright_stars=max_ultra_bright_stars,
    )


def sample_apparent_magnitudes_imf(
    n_stars: int,
    rng: np.random.Generator,
    *,
    mag_bright: float,
    mag_faint: float,
    giant_fraction: float = 0.048,
    salpeter_alpha: float = 2.35,
    distance_modulus_mean: float = 11.4,
    distance_modulus_sigma: float = 1.65,
    magnitude_ultra_cut: float = 6.5,
    max_ultra_bright_stars: int = 6,
) -> np.ndarray:
    """Broken IMF → absolute mag + distance modulus → apparent V-like magnitudes."""
    n = int(n_stars)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    ml = float(min(mag_bright, mag_faint))
    mh = float(max(mag_bright, mag_faint))

    is_giant = rng.random(n) < float(np.clip(giant_fraction, 0.0, 0.25))
    m_ms = sample_salpeter_mass(
        n, rng, alpha=salpeter_alpha, m_lo=0.12, m_hi=12.0
    )
    m_gi = sample_salpeter_mass(n, rng, alpha=1.75, m_lo=1.2, m_hi=35.0)
    mass = np.where(is_giant, m_gi, m_ms)
    m_abs = np.where(
        is_giant,
        mass_to_abs_mag_giant(mass),
        mass_to_abs_mag_main_sequence(mass),
    )
    mu = sample_distance_modulus(
        n,
        rng,
        mean=distance_modulus_mean,
        sigma=distance_modulus_sigma,
    )
    mags = np.clip(m_abs + mu, ml - 2.0, mh + 0.5)

    # Blend a few sub-giants into the bright end
    bright_draw = rng.random(n) < 0.018
    if np.any(bright_draw):
        hi_b = max(ml + 1e-4, float(magnitude_ultra_cut) + 0.35)
        mags[bright_draw] = rng.uniform(ml, hi_b, int(np.sum(bright_draw)))

    mags = np.clip(mags, ml, mh)
    return _trim_ultra_bright_mags(
        mags,
        rng,
        mag_bright=ml,
        mag_faint=mh,
        magnitude_ultra_cut=magnitude_ultra_cut,
        max_ultra_bright_stars=max_ultra_bright_stars,
    )
