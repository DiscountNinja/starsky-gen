"""Fractal extinction and emission clearance masks."""

import numpy as np

from starsky_gen.dust_field import (
    build_band_disruption_field,
    build_filament_erosion_map,
    build_fractal_extinction_field,
    carve_extinction_discontinuities,
    compress_absorption_opacity,
    emission_clearance_from_extinction,
    reinforce_absorption_edges,
    transmission_from_absorption_map,
)


def test_opacity_gamma_deepens_lanes_preserves_weak_haze() -> None:
    a = np.linspace(0.05, 0.95, 21, dtype=np.float64)
    c = compress_absorption_opacity(a, gamma=1.45)
    assert float(c[2]) < float(a[10])
    assert float(c[-1]) > float(a[-1]) * 1.02
    t_lin = transmission_from_absorption_map(a, void_floor=0.02, sharpness=3.2, opacity_gamma=1.0)
    t_gamma = transmission_from_absorption_map(a, void_floor=0.02, sharpness=3.2, opacity_gamma=1.45)
    assert float(t_gamma[-3]) < float(t_lin[-3])
    assert float(t_gamma[2]) >= float(t_lin[2]) * 0.99


def test_filament_erosion_has_sharp_transmission() -> None:
    rng = np.random.default_rng(11)
    erosion = build_filament_erosion_map(rng, 80, 160, periodic_x=True)
    trans = transmission_from_absorption_map(erosion, void_floor=0.04, sharpness=2.3)
    assert float(trans.min()) < 0.15
    grad = np.abs(np.diff(trans, axis=1))
    assert float(np.percentile(grad, 97.0)) > 0.04


def test_reinforce_edges_darkens_rims() -> None:
    erosion = np.zeros((32, 64), dtype=np.float64)
    erosion[16, 30:34] = 1.0
    t0 = np.full((32, 64), 0.6, dtype=np.float64)
    t1 = reinforce_absorption_edges(t0, erosion, edge_gain=0.5, periodic_x=True)
    assert float(t1.min()) < float(t0.min())


def test_fractal_extinction_bounded() -> None:
    rng = np.random.default_rng(3)
    dark = build_fractal_extinction_field(rng, 64, 128, periodic_x=True)
    assert dark.shape == (64, 128)
    assert float(dark.min()) >= 0.0
    assert float(dark.max()) <= 1.0


def test_disruption_carves_deep_voids() -> None:
    rng = np.random.default_rng(7)
    disrupt = build_band_disruption_field(rng, 96, 192, periodic_x=True)
    ext = np.full((96, 192), 0.72, dtype=np.float64)
    carved = carve_extinction_discontinuities(ext, disrupt, strength=1.0, void_floor=0.04)
    assert float(carved.min()) < 0.12
    assert float(np.mean(carved < 0.15)) > 0.02


def test_emission_clearance_low_in_dark_lanes() -> None:
    ext = np.linspace(0.05, 0.95, 32, dtype=np.float64)[None, :]
    ext = np.broadcast_to(ext, (16, 32))
    clear = emission_clearance_from_extinction(ext, power=1.4)
    assert float(clear[0, 0]) < float(clear[0, -1])
