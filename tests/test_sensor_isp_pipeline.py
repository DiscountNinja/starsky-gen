"""Sensor optics, ISP, tone map, and output pipeline tests."""

import numpy as np

from starsky_gen.config import FeatureConfig
from starsky_gen.optical_effects import (
    apply_isp_linear_chain,
    apply_thin_film_lens_flare,
)
from starsky_gen.postfx import apply_masked_density_sharpen
from starsky_gen.postprocess import apply_jpeg_artifacts, smooth_jpeg_highlight_artifacts
from starsky_gen.tone_map import apply_acescct_cinematic_grade, asinh_linear_stretch_luma
from starsky_gen.tone_grade import apply_galaxy_scene_tone, mild_disk_asinh_grade


def test_asinh_midtone_and_toe() -> None:
    lu = np.array([0.02, 0.15, 0.45, 0.9])
    base = asinh_linear_stretch_luma(lu, gain=0.08, q=1.0)
    lifted = asinh_linear_stretch_luma(lu, gain=0.08, q=1.0, midtone_exposure=1.2, toe_strength=0.3)
    assert float(lifted[2]) > float(base[2])
    assert float(lifted[0]) >= float(base[0])


def test_acescct_grade_runs() -> None:
    rgb = np.clip(np.random.default_rng(0).random((24, 48, 3)) * 0.6 + 0.05, 0.0, 1.0)
    disk_w = np.ones((24, 48))
    out = apply_acescct_cinematic_grade(rgb, disk_w, strength=0.5)
    assert out.shape == rgb.shape


def test_isp_chain_finite() -> None:
    rgb = np.full((16, 32, 3), 0.2)
    out = apply_isp_linear_chain(rgb, strength=1.0)
    assert np.all(np.isfinite(out))


def test_thin_film_adds_color_on_hot() -> None:
    rng = np.random.default_rng(1)
    rgb = np.zeros((32, 32, 3))
    rgb[16, 16] = [0.95, 0.9, 0.88]
    disk_w = np.ones((32, 32))
    out = apply_thin_film_lens_flare(rgb, disk_w, rng, strength=0.06, periodic_x=True)
    assert float(np.std(out[14:18, 14:18])) > float(np.std(rgb[14:18, 14:18]))


def test_masked_sharpen_skips_empty_sky() -> None:
    rgb = np.full((32, 48, 3), 0.06)
    disk_w = np.exp(-((np.linspace(-1, 1, 32)[:, None] ** 2) / 0.5))
    dens = disk_w * 0.8
    out = apply_masked_density_sharpen(
        rgb,
        disk_w,
        dens,
        sigma_px=2.0,
        amp_faint=0.15,
        amp_midplane=0.1,
        knee=0.2,
        periodic_x=True,
    )
    # High latitude (sky): little change
    assert float(np.mean(np.abs(out[0, 24] - rgb[0, 24]))) < 0.01


def test_scene_tone_acescct_mode() -> None:
    feats = FeatureConfig(galaxy_tone_curve="acescct")
    rgb = np.clip(np.random.default_rng(2).random((12, 24, 3)) * 0.4, 0.02, None)
    out = apply_galaxy_scene_tone(rgb, np.ones((12, 24)), feats, None, periodic_x=True)
    assert np.all(out >= 0.0)


def test_jpeg_then_highlight_smooth() -> None:
    rng = np.random.default_rng(3)
    rgb = np.clip(rng.random((64, 64, 3)) * 0.5 + 0.2, 0.0, 1.0)
    rgb[30:34, 30:34] = 0.95
    jpg = apply_jpeg_artifacts(rgb, quality=35)
    smooth = smooth_jpeg_highlight_artifacts(jpg, strength=0.6, periodic_x=True)
    assert smooth.shape == jpg.shape
