"""Hierarchical star placement: density field → associations → individuals."""

import numpy as np

from starsky_gen.placement import (
    build_master_density_field,
    pick_association_peaks,
    sample_hierarchical_poisson_field,
)


def test_master_density_normalized() -> None:
    rng = np.random.default_rng(1)
    d = build_master_density_field(128, 64, rng, periodic_x=True)
    assert d.shape == (64, 128)
    assert float(d.max()) <= 1.0 + 1e-6
    assert float(d.min()) > 0.0


def test_association_peaks_separated() -> None:
    rng = np.random.default_rng(2)
    d = build_master_density_field(256, 128, rng, periodic_x=True)
    rows, cols = pick_association_peaks(d, rng, n_peaks=6, min_sep_px=24.0)
    assert rows.size >= 2
    assert cols.size == rows.size


def test_hierarchical_sample_count() -> None:
    rng = np.random.default_rng(4)
    lon, lat = sample_hierarchical_poisson_field(rng, 400, 256, 128)
    assert lon.shape == lat.shape
    assert lon.size == 400


def test_poisson_placement_exact_count() -> None:
    from starsky_gen.placement import sample_lon_lat_poisson

    rng = np.random.default_rng(5)
    lon, lat = sample_lon_lat_poisson(1200, 512, 256, rng, min_sep_faint_px=3.0, n_bright=40)
    assert lon.size == lat.size == 1200
