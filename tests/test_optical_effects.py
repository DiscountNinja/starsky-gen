"""Optical & sensor effects."""

import numpy as np

from starsky_gen.optical_effects import (
    apply_film_grain_display,
    apply_lateral_chromatic_aberration,
    apply_lens_vignette,
    apply_wavelength_halation,
    inject_sensor_noise,
)


def test_inject_sensor_noise_increases_variance() -> None:
    rng = np.random.default_rng(0)
    rgb = np.full((32, 48, 3), 0.25)
    out = inject_sensor_noise(
        rgb, rng, shot_scale=0.01, read_sigma=0.002, space="display"
    )
    assert float(np.var(out - rgb)) > 1e-8


def test_lens_vignette_darkens_corners_redder() -> None:
    rgb = np.full((64, 64, 3), 0.5)
    out = apply_lens_vignette(rgb, strength=1.0)
    assert out[0, 0, 2] < out[0, 0, 0]
    assert out[32, 32, 0] > out[0, 0, 0]


def test_halation_adds_on_bright_core() -> None:
    rgb = np.zeros((32, 32, 3))
    rgb[16, 16] = [0.92, 0.88, 0.85]
    disk_w = np.ones((32, 32))
    out = apply_wavelength_halation(rgb, disk_w, strength=0.08, periodic_x=True)
    assert float(out[16, 17, 0]) > float(rgb[16, 17, 0])


def test_chromatic_aberration_separates_channels_at_edge() -> None:
    rgb = np.zeros((64, 64, 3))
    rgb[4, 4] = [0.9, 0.85, 0.8]
    out = apply_lateral_chromatic_aberration(rgb, strength=1.2, periodic_x=True)
    assert out.shape == rgb.shape
    assert float(np.max(np.abs(out - rgb))) > 1e-5


def test_film_grain_low_amplitude() -> None:
    rng = np.random.default_rng(1)
    rgb = np.full((32, 32, 3), 0.35)
    out = apply_film_grain_display(rgb, rng, 0.006, periodic_x=True)
    assert float(np.mean(np.abs(out - rgb))) < 0.02
