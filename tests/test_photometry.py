"""IMF / apparent-magnitude sampling tests."""

import numpy as np

from starsky_gen.photometry import (
    sample_apparent_magnitudes_imf,
    sample_apparent_magnitudes_lf,
    sample_salpeter_mass,
)
from starsky_gen.starfield import sample_cluster_star_catalog


def test_salpeter_mass_in_range() -> None:
    rng = np.random.default_rng(0)
    m = sample_salpeter_mass(500, rng, alpha=2.35, m_lo=0.2, m_hi=20.0)
    assert m.shape == (500,)
    assert float(m.min()) >= 0.2 - 1e-9
    assert float(m.max()) <= 20.0 + 1e-9


def test_lf_magnitudes_within_bounds() -> None:
    rng = np.random.default_rng(5)
    mags = sample_apparent_magnitudes_lf(
        3000,
        rng,
        mag_bright=8.0,
        mag_faint=20.0,
        magnitude_ultra_cut=6.5,
        max_ultra_bright_stars=6,
    )
    assert mags.shape == (3000,)
    assert float(mags.min()) >= 8.0 - 1e-6
    assert int(np.sum(mags < 6.5)) <= 6


def test_imf_magnitudes_within_bounds() -> None:
    rng = np.random.default_rng(1)
    mags = sample_apparent_magnitudes_imf(
        2000,
        rng,
        mag_bright=8.0,
        mag_faint=20.0,
        giant_fraction=0.05,
        magnitude_ultra_cut=6.5,
        max_ultra_bright_stars=8,
    )
    assert mags.shape == (2000,)
    assert float(mags.min()) >= 8.0 - 1e-6
    assert float(mags.max()) <= 20.0 + 1e-6
    assert int(np.sum(mags < 6.5)) <= 8


def test_cluster_catalog_photometry_and_spread() -> None:
    rng = np.random.default_rng(2)
    cat = sample_cluster_star_catalog(
        rng,
        1024,
        512,
        1.0,
        attach_apparent_mag=True,
        use_imf_magnitudes=True,
        imf_giant_fraction=0.12,
    )
    assert "phot_mag" in cat
    mags = cat["phot_mag"]
    assert mags.size > 80
    assert float(np.std(mags)) > 0.35
    assert float(np.percentile(mags, 10)) < float(np.percentile(mags, 90))


def test_cluster_radial_mag_gradient_single() -> None:
    """One synthetic cluster: center stars are brighter (lower m)."""
    rng = np.random.default_rng(3)
    cl_lon, cl_lat = 1.0, 0.05
    n = 120
    lon = (cl_lon + rng.normal(0, 0.02, n)) % (2 * np.pi)
    lat = cl_lat + rng.normal(0, 0.015, n)
    dlon = np.minimum(np.abs(lon - cl_lon), 2 * np.pi - np.abs(lon - cl_lon))
    r2 = (dlon / 0.02) ** 2 + ((lat - cl_lat) / 0.015) ** 2
    mags = 12.0 - 1.1 * np.exp(-r2 / 2.2)
    inner = mags[r2 < np.quantile(r2, 0.25)]
    outer = mags[r2 > np.quantile(r2, 0.75)]
    assert float(np.mean(inner)) < float(np.mean(outer))
