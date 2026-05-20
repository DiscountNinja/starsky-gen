"""Tone mapping & grading (scene linear + display finish)."""

import numpy as np

from starsky_gen.color_science import rec709_luma
from starsky_gen.config import FeatureConfig
from starsky_gen.tone_grade import (
    apply_galaxy_scene_tone,
    apply_localized_core_scurve_dodge_burn,
    apply_neutral_white_balance,
    mild_disk_asinh_grade,
)


def test_mild_asinh_preserves_faint_relative_order() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.float64)
    rgb[2, 2] = [0.02, 0.025, 0.03]
    rgb[4, 4] = [0.35, 0.32, 0.30]
    disk_w = np.ones((8, 8))
    mild = mild_disk_asinh_grade(rgb, disk_w, gain=0.06, curvature=0.65, q=1.0)
    strong = mild_disk_asinh_grade(rgb, disk_w, gain=0.06, curvature=1.0, q=1.0)
    l_m = rec709_luma(mild)
    l_s = rec709_luma(strong)
    # Lower curvature lifts faint arms less aggressively than full gain.
    assert l_m[2, 2] < l_s[2, 2]
    assert l_m[4, 4] <= l_s[4, 4] + 0.02


def test_localized_core_scurve_changes_core_not_sky() -> None:
    h, w = 32, 48
    rgb = np.full((h, w, 3), 0.08)
    rgb[14:18, 20:28] = 0.55
    neb = np.zeros((h, w))
    neb[14:18, 20:28] = 0.8
    disk_w = np.exp(-((np.linspace(-1, 1, h)[:, None] ** 2) / 0.5))
    out = apply_localized_core_scurve_dodge_burn(
        rgb, neb, disk_w, scurve_strength=0.25, periodic_x=True
    )
    assert float(np.mean(np.abs(out[2, 2] - rgb[2, 2]))) < 1e-4
    assert float(np.mean(np.abs(out[16, 24] - rgb[16, 24]))) > 1e-4


def test_neutral_wb_moves_toward_gray() -> None:
    rgb = np.full((4, 4, 3), [0.5, 0.4, 0.3])
    out = apply_neutral_white_balance(rgb, strength=0.8)
    assert np.std(out[0, 0]) < np.std(rgb[0, 0])


def test_scene_tone_runs_with_defaults() -> None:
    rgb = np.clip(np.random.default_rng(0).random((16, 32, 3)) * 0.5, 0.01, None)
    disk_w = np.ones((16, 32))
    feats = FeatureConfig()
    out = apply_galaxy_scene_tone(rgb, disk_w, feats, None, periodic_x=True)
    assert out.shape == rgb.shape
    assert np.all(out >= 0.0)
