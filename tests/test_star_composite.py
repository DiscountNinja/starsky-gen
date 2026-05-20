"""Star-over-nebula composite avoids canvas-only dim + full blue stars."""

import numpy as np

from starsky_gen.composite_blend import (
    composite_stars_over_display_canvas,
    soft_knee_star_peaks,
)


def test_band_composite_desaturates_plane_stars() -> None:
    h, w = 64, 128
    canvas = np.full((h, w, 3), 0.55, dtype=np.float64)
    stars = np.zeros((h, w, 3), dtype=np.float64)
    stars[:, : w // 2] = np.array([0.15, 0.35, 0.95], dtype=np.float64)
    gate = np.zeros((h, w), dtype=np.float64)
    gate[:, w // 2 :] = 1.0
    out = composite_stars_over_display_canvas(
        canvas,
        stars,
        gate,
        add_scale=1.0,
        band_chroma_desat=0.30,
        band_brightness_scale=0.70,
    )
    left_b = float(np.mean(out[:, 10, 2] / np.maximum(out[:, 10, 0], 1e-6)))
    right_b = float(np.mean(out[:, w - 10, 2] / np.maximum(out[:, w - 10, 0], 1e-6)))
    assert left_b > right_b + 0.15


def test_soft_knee_reduces_star_peak() -> None:
    stars = np.zeros((64, 64, 3), dtype=np.float64)
    for dy, dx in ((28, 30), (32, 32), (36, 34)):
        stars[dy, dx] = [0.2, 0.35, 0.95]
    stars += 0.02
    soft = soft_knee_star_peaks(stars, strength=0.45)
    assert float(np.max(soft)) < float(np.max(stars)) * 0.98
