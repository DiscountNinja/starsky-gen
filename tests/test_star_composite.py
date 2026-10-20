"""Star-over-nebula composite avoids canvas-only dim + full blue stars."""

import numpy as np

from starsky_gen.composite_blend import (
    composite_stars_over_display_canvas,
    soft_knee_star_peaks,
)
from starsky_gen.generator import (
    _composite_nebula_heroes_on_top,
    _hero_star_protect_mask,
    _nebula_emission_star_modulate,
)
from starsky_gen.hero_stars import HeroStarRecord


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


def test_nebula_emission_modulate_spares_heroes() -> None:
    """Regular stars dim in dense gas; hero mask stays full brightness."""
    h, w = 48, 96
    stars = np.full((h, w, 3), 0.40, dtype=np.float64)
    dens = np.ones((h, w), dtype=np.float64)
    heroes = [
        HeroStarRecord(x=48.0, y=24.0, rgb=(1.0, 0.9, 0.85), flux=8.0),
    ]
    mask = _hero_star_protect_mask(h, w, heroes, radius_px=5.0)
    assert float(mask.max()) > 0.5
    out = _nebula_emission_star_modulate(
        stars, dens, hero_mask=mask, dim_k=0.40, floor=0.50, warm=False
    )
    # Far from hero: attenuated.
    assert float(out[2, 2].mean()) < 0.40 * 0.72
    # Hero core: near full.
    assert float(out[24, 48].mean()) > 0.38


def test_composite_nebula_heroes_on_top_raises_local_peak() -> None:
    """Post-nebula hero paint sits above a dim canvas without softening."""
    h, w = 64, 128
    canvas = np.full((h, w, 3), 0.12, dtype=np.float64)
    heroes = [
        HeroStarRecord(x=64.0, y=32.0, rgb=(1.0, 0.95, 0.9), flux=12.0),
    ]
    rng = np.random.default_rng(0)
    out = _composite_nebula_heroes_on_top(
        canvas,
        heroes,
        rng,
        width=w,
        height=h,
        spike_max=0,
        ghost_strength=0.0,
        periodic_x=True,
    )
    assert float(out[32, 64].max()) > 0.55
    # Far field unchanged (heroes on top, not a wash).
    assert float(out[2, 2].mean()) < 0.15
