"""Composite & blending helpers (linear gas add, emission screen, luma tone)."""

import numpy as np

from starsky_gen.composite_blend import (
    composite_add_gas,
    composite_emission_add_screen,
    percentile_asinh_luma_stretch,
    recompute_hdr_asinh_gain,
    remap_luma_preserving_chroma,
    reinhard_luma_preserving,
    stars_hdr_to_display,
)
from starsky_gen.color_science import rec709_luma


def test_remap_luma_preserves_hue_direction():
    rgb = np.array([[[0.2, 0.5, 0.9]]], dtype=np.float64)
    l_out = np.array([[0.4]], dtype=np.float64)
    out = remap_luma_preserving_chroma(rgb, l_out)
    assert np.isclose(rec709_luma(out), 0.4, rtol=1e-5)
    ratios_in = rgb[0, 0] / rgb[0, 0, 1]
    ratios_out = out[0, 0] / out[0, 0, 1]
    np.testing.assert_allclose(ratios_in, ratios_out, rtol=1e-5)


def test_percentile_asinh_only_affects_luma_mapping():
    lu = np.linspace(0.0, 2.0, 50)
    stretched = percentile_asinh_luma_stretch(lu, stretch_gain=8.0)
    assert stretched.shape == lu.shape
    assert stretched[0] >= 0.0
    assert stretched[-1] > stretched[10]


def test_stars_hdr_to_display_preserves_blue_bias():
    stars = np.zeros((4, 4, 3), dtype=np.float32)
    stars[2, 2] = [0.05, 0.08, 0.35]
    out = stars_hdr_to_display(stars, stretch_gain=10.0)
    assert out[2, 2, 2] > out[2, 2, 0]


def test_composite_add_gas_is_additive_not_alpha_over_black():
    canvas = np.zeros((2, 2, 3), dtype=np.float32)
    gas = np.full((2, 2, 3), 0.5, dtype=np.float64)
    vis = np.full((2, 2), 0.5, dtype=np.float64)
    out = composite_add_gas(canvas, gas, visibility=vis)
    np.testing.assert_allclose(out[0, 0], [0.25, 0.25, 0.25], rtol=1e-5)


def test_emission_screen_brighter_than_add_alone_on_hot_core():
    base = np.full((3, 3, 3), 0.3, dtype=np.float64)
    emit = np.zeros((3, 3, 3), dtype=np.float64)
    emit[1, 1] = [0.9, 0.2, 0.15]
    hot = np.zeros((3, 3), dtype=np.float64)
    hot[1, 1] = 0.95
    add_only = base + emit
    screened = composite_emission_add_screen(base, emit, hot, core_screen_mix=0.9)
    assert rec709_luma(screened)[1, 1] >= rec709_luma(add_only)[1, 1] - 1e-6


def test_recompute_gain_boosts_when_canvas_dim():
    dim = np.full((8, 8, 3), 0.01, dtype=np.float64)
    bright = np.full((8, 8, 3), 0.8, dtype=np.float64)
    disk_w = np.ones((8, 8), dtype=np.float64)
    g_dim = recompute_hdr_asinh_gain(dim, disk_w, base_gain=1.0)
    g_bright = recompute_hdr_asinh_gain(bright, disk_w, base_gain=1.0)
    assert g_dim > g_bright


def test_attenuate_rgb_column_comb_reduces_lon_pillars() -> None:
    from starsky_gen.dust_field import attenuate_rgb_column_comb

    h, w = 48, 96
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.12)), (h, w))
    col = np.sin(np.linspace(0, 12 * np.pi, w))[None, :] * 0.08 + 0.42
    rgb = np.stack([col, col * 1.02, col * 0.96], axis=2)
    out = attenuate_rgb_column_comb(rgb, disk, strength=0.7, periodic_x=True)
    comb_in = float(np.std(np.mean(rgb[:, :, 0], axis=0)))
    comb_out = float(np.std(np.mean(out[:, :, 0], axis=0)))
    assert comb_out < comb_in * 0.72


def test_reinhard_preserves_chroma_direction():
    rgb = np.array([[[0.1, 0.4, 0.9]]], dtype=np.float64)
    out = reinhard_luma_preserving(rgb, k=0.5)
    assert out[0, 0, 2] > out[0, 0, 0]
