"""Regression tests for procedural cloud noise and nebula dust output."""

import numpy as np

from starsky_gen.config import NebulaMode, NebulaTuningConfig
from starsky_gen.nebula import generate_nebula
from starsky_gen import procedural_noise as pn


def test_combine_cloud_layers_bounded() -> None:
    rng = np.random.default_rng(0)
    h, w = 32, 48
    b = pn.fbm2d(rng, h, w, base_scale=0.02, octaves=2, periodic_x=False)
    r = pn.ridged_fbm2d(rng, h, w, base_scale=0.04, octaves=2, periodic_x=False)
    f = pn.fbm2d(rng, h, w, base_scale=0.08, octaves=2, periodic_x=False)
    band = np.exp(-((np.linspace(-1.0, 1.0, h)[:, None]) ** 2) / 0.5)
    out = pn.combine_cloud_layers(b, r, f, band, w_base=0.3, w_ridge=0.5, w_fine=0.2)
    assert out.shape == (h, w)
    assert float(np.min(out)) >= 0.0
    assert float(np.max(out)) <= 1.0 + 1e-9


def test_galaxy_streak_nebula_has_mid_frequency_detail() -> None:
    """Gas should not collapse to a single smooth gradient (regression for over-blur)."""
    rng = np.random.default_rng(99)
    neb, _emit, _dust, _lane = generate_nebula(
        rng,
        NebulaMode.galaxy_streak,
        128,
        256,
        NebulaTuningConfig(),
        progress_cb=None,
    )
    from starsky_gen.nebula import _blur_separable_xy

    luma = np.mean(neb, axis=2)
    soft = _blur_separable_xy(luma, passes=4, periodic_x=True)
    # Coarse blur should not erase most structure (over-smoothed gas fails this).
    assert float(np.std(luma)) > float(np.std(soft)) * 1.05
    gx = np.diff(luma, axis=1)
    assert float(np.percentile(np.abs(gx), 92)) > 0.004


def test_galaxy_streak_dust_occlusion_stats() -> None:
    rng = np.random.default_rng(12345)
    _neb, _emit, dust, lane = generate_nebula(
        rng,
        NebulaMode.galaxy_streak,
        96,
        192,
        NebulaTuningConfig(),
        progress_cb=None,
    )
    assert dust.shape == (96, 192)
    assert lane.shape == (96, 192)
    m = float(np.mean(dust))
    assert 0.10 < m < 0.90, m
    assert float(np.max(dust)) <= 1.0 + 1e-9


def test_debug_layer_base_short_circuits() -> None:
    rng = np.random.default_rng(7)
    neb, emit, dust, lane = generate_nebula(
        rng,
        NebulaMode.galaxy_streak,
        48,
        64,
        NebulaTuningConfig(debug_pass="layer_base"),
    )
    assert neb.shape == (48, 64, 3)
    assert np.allclose(emit, 0.0)
    assert dust.shape == (48, 64)
    assert lane.shape == (48, 64)
    assert np.allclose(neb[..., 0], neb[..., 1]) and np.allclose(neb[..., 1], neb[..., 2])


def test_distant_mode_runs() -> None:
    rng = np.random.default_rng(99)
    neb, emit, dust, lane = generate_nebula(
        rng, NebulaMode.distant, 64, 64, NebulaTuningConfig(), progress_cb=None
    )
    assert neb.shape == (64, 64, 3)
    assert dust.shape == (64, 64)


def test_noise_stack_galaxy_shapes() -> None:
    rng = np.random.default_rng(3)
    s = pn.build_galaxy_streak_noise_stack(
        rng, 40, 80, periodic_x=True, elongate_along_x=1.7
    )
    for k in ("base", "ridged", "fine", "warp_x", "warp_y"):
        assert s[k].shape == (40, 80), k


def test_assemble_galaxy_dust_alpha_keys() -> None:
    rng = np.random.default_rng(1)
    h, w = 48, 64
    s = pn.build_galaxy_streak_noise_stack(rng, h, w, periodic_x=True, elongate_along_x=1.6)
    band = np.exp(-((np.linspace(-1.0, 1.0, h)[:, None]) ** 2) / (2 * 0.13**2))
    pack = pn.assemble_galaxy_dust_alpha(s["base"], s["ridged"], s["fine"], band, periodic_x=True)
    for k in ("carve", "dust_alpha", "fil_dense"):
        assert k in pack
        assert pack[k].shape == (h, w)
