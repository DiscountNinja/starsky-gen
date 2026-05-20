"""Galaxy spiral / turbulence structure helpers."""

import numpy as np

from starsky_gen.nebula_physics import dust_lane_multiscatter_fill
from starsky_gen.procedural_noise import (
    galaxy_arm_density_modulator,
    log_spiral_arm_mask,
)


def _blur_stub(field: np.ndarray, passes: int = 1, periodic_x: bool = False) -> np.ndarray:
    _ = passes, periodic_x
    out = field.copy()
    for _ in range(2):
        out = (
            np.roll(out, 1, 0)
            + np.roll(out, -1, 0)
            + np.roll(out, 1, 1)
            + np.roll(out, -1, 1)
            + out
        ) / 5.0
    return out


def test_log_spiral_varies_with_longitude() -> None:
    y = np.linspace(-1.0, 1.0, 64)[:, None]
    x = np.linspace(-1.0, 1.0, 128)[None, :]
    m = log_spiral_arm_mask(x, y, arms=2, strength=0.8)
    assert float(np.std(m)) > 0.04


def test_density_modulator_in_range() -> None:
    rng = np.random.default_rng(0)
    h, w = 64, 128
    y = np.linspace(-1.0, 1.0, h)[:, None]
    x = np.linspace(-1.0, 1.0, w)[None, :]
    mod = galaxy_arm_density_modulator(h, w, x, y, rng, spiral_strength=0.6, periodic_x=True)
    assert mod.shape == (h, w)
    assert float(mod.min()) >= 0.0
    assert float(mod.max()) <= 1.0 + 1e-6


def test_multiscatter_fills_dark_lanes() -> None:
    rgb = np.zeros((32, 32, 3))
    rgb[8:24, 8:24] = 0.4
    ext = np.ones((32, 32))
    ext[10:22, 10:22] = 0.12
    amb = np.full((32, 32), 0.35)
    out = dust_lane_multiscatter_fill(rgb, ext, amb, strength=0.12, blur_fn=_blur_stub)
    dark = out[15, 15]
    assert float(np.mean(dark)) > 0.02
