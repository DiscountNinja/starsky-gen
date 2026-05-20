"""Placement density and Poisson-disk tests."""

import numpy as np

from starsky_gen.placement import (
    latitudinal_density,
    rho_disk_sech2,
    sample_galactic_disk_lon_lat_v2,
    sample_lon_lat_poisson,
)
from starsky_gen.projections import sph_to_equirect_xy


def test_sech2_peak_at_zero() -> None:
    center = float(rho_disk_sech2(np.array([0.0]), 0.19)[0])
    wing = float(rho_disk_sech2(np.array([0.5]), 0.19)[0])
    assert center > wing


def test_disk_halo_mixture() -> None:
    lat = np.array([0.0, 0.55])
    d = latitudinal_density(lat, halo_fraction=0.0)
    h = latitudinal_density(lat, halo_fraction=0.5)
    assert float(d[0]) >= float(h[0]) * 0.85
    assert float(h[1]) > float(d[1]) * 1.05


def test_poisson_sample_count() -> None:
    rng = np.random.default_rng(0)
    lon, lat = sample_lon_lat_poisson(40, 256, 128, rng, n_bright=5)
    assert lon.shape[0] >= 35
    assert lat.shape[0] == lon.shape[0]


def test_v2_placement_finite() -> None:
    rng = np.random.default_rng(2)
    lon, lat = sample_galactic_disk_lon_lat_v2(rng, 80, 320, 160)
    assert np.all(np.isfinite(lon))
    assert np.all(np.abs(lat) <= np.pi / 2)


def test_poisson_avoids_equirect_pole_rows() -> None:
    """Steradian weighting: lower star density in top/bottom equirect rows."""
    rng = np.random.default_rng(17)
    h, w = 256, 512
    lon, lat = sample_lon_lat_poisson(2500, w, h, rng, n_bright=40)
    _, ys = sph_to_equirect_xy(lon, lat, w, h)
    edge = max(4, h // 14)
    mid = (ys >= edge) & (ys < h - edge)
    edge_frac = float(np.sum(~mid)) / float(h)
    mid_frac = float(np.sum(mid)) / float(h)
    assert edge_frac < mid_frac * 0.55
