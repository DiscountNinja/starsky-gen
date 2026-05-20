"""Suggested photometry / tone-map starting parameters."""

import numpy as np

from starsky_gen.composite_blend import auto_star_display_stretch_gain, recompute_hdr_asinh_gain
from starsky_gen.nebula_physics import extinction_av_scale_for_lane_depth
from starsky_gen.psf import PsfTuning, moffat_fwhm_px, moffat_params_from_mag_and_flux, moffat_sigma_from_fwhm
from starsky_gen.tone_map import asinh_linear_stretch_luma


def test_moffat_fwhm_mag_law() -> None:
    tune = PsfTuning(fwhm_base_px=2.0, fwhm_ref_mag=8.0, fwhm_mag_coeff=0.2)
    rng = np.random.default_rng(0)
    p_b, _, _, _ = moffat_params_from_mag_and_flux(
        6.0, flux=100.0, galactic_lat_rad=0.0, rng=rng, tuning=tune
    )
    p_f, _, _, _ = moffat_params_from_mag_and_flux(
        14.0, flux=1.0, galactic_lat_rad=0.0, rng=rng, tuning=tune
    )
    fwhm_b = moffat_fwhm_px(p_b["sigma_x"], p_b["beta"])
    fwhm_f = moffat_fwhm_px(p_f["sigma_x"], p_f["beta"])
    assert fwhm_b > fwhm_f
    expect_b = 2.0 * (1.0 + 0.2 * 2.0)
    assert abs(fwhm_b - expect_b) < 1.05


def test_asinh_linear_gain_q() -> None:
    lu = np.array([0.0, 0.05, 0.2], dtype=np.float64)
    lo = asinh_linear_stretch_luma(lu, gain=0.05, q=1.0)
    hi = asinh_linear_stretch_luma(lu, gain=0.15, q=1.0)
    assert hi[-1] > lo[-1]


def test_extinction_av_scale_lane_range() -> None:
    scale = extinction_av_scale_for_lane_depth(transmission_floor=0.14, lane_mag_at_floor=1.5)
    raw = -2.5 * np.log10(0.14)
    assert abs(scale * raw - 1.5) < 0.05


def test_auto_star_stretch_targets_bright_tail() -> None:
    stars = np.zeros((32, 32, 3), dtype=np.float32)
    stars[16, 16] = [8.0, 8.0, 9.0]
    stars[10, 12] = [0.5, 0.5, 0.55]
    g = auto_star_display_stretch_gain(stars, peak_percentile=99.0, target_peak=0.95)
    assert 4.0 <= g <= 28.0


def test_recompute_linear_asinh_gain_boosts_dim_canvas() -> None:
    dim = np.full((8, 8, 3), 0.02, dtype=np.float64)
    bright = np.full((8, 8, 3), 0.35, dtype=np.float64)
    disk_w = np.ones((8, 8))
    g_dim = recompute_hdr_asinh_gain(dim, disk_w, base_gain=0.08)
    g_bright = recompute_hdr_asinh_gain(bright, disk_w, base_gain=0.08)
    assert g_dim > g_bright
