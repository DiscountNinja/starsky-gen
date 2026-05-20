"""Anti-synthetic-artifact helpers (jitter, dither, nebula noise)."""

import numpy as np

from starsky_gen.dither import apply_blue_noise_dither_u8, blue_noise_tile
from starsky_gen.placement import _poisson_disk_on_grid, pixels_to_lon_lat
from starsky_gen.procedural_noise import build_galaxy_streak_noise_stack


def test_blue_noise_tile_bounded() -> None:
    t = blue_noise_tile(32, seed=7)
    assert t.shape == (32, 32)
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def test_dither_changes_quantization() -> None:
    ramp = np.linspace(0.1, 0.9, 64, dtype=np.float64)
    rgb = np.stack([ramp.reshape(8, 8)] * 3, axis=-1)
    plain = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    dithered = apply_blue_noise_dither_u8(rgb, strength=1.0)
    assert not np.array_equal(plain, dithered)


def test_dither_preserves_chroma_on_ramp() -> None:
    ramp = np.linspace(0.2, 0.8, 64, dtype=np.float64)
    base = np.stack([ramp.reshape(8, 8) * c for c in (1.02, 0.98, 0.94)], axis=-1)
    out = apply_blue_noise_dither_u8(base, strength=0.8).astype(np.float64) / 255.0
    in_ratio = base[4, 4] / max(float(np.mean(base[4, 4])), 1e-8)
    out_ratio = out[4, 4] / max(float(np.mean(out[4, 4])), 1e-8)
    np.testing.assert_allclose(in_ratio, out_ratio, rtol=0.08, atol=0.06)


def test_poisson_returns_fractional_pixels() -> None:
    rng = np.random.default_rng(0)
    density = np.ones((32, 48))
    rows, cols = _poisson_disk_on_grid(12, density, rng, min_sep_px=4.0)
    assert rows.dtype == np.float64
    assert np.any(np.abs(rows - np.round(rows)) > 0.05)


def test_galaxy_noise_stack_keys() -> None:
    rng = np.random.default_rng(1)
    s = build_galaxy_streak_noise_stack(rng, 64, 128, periodic_x=True, elongate_along_x=1.5, fine_mix=0.7)
    assert "base" in s and "fine" in s
    assert float(np.std(s["fine"])) < float(np.std(s["base"])) * 2.5
