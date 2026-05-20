"""Disk radiance unification between unresolved and resolved stars."""

import numpy as np

from starsky_gen.disk_radiance import (
    apply_shared_photon_exposure,
    disk_chroma_from_star_layer,
    estimate_disk_photon_exposure,
    harmonize_diffuse_canvas_chroma,
    ism_lift_rgb,
    unresolved_speckle_rgb,
)


def test_unresolved_warmth_interpolates_chroma() -> None:
    cold = unresolved_speckle_rgb(warmth=0.0)
    hot = unresolved_speckle_rgb(warmth=1.0, disk_chroma=np.array([1.0, 0.8, 0.5]))
    assert float(hot[2]) < float(cold[2])


def test_ism_lift_locks_to_star_chroma() -> None:
    star = np.array([1.0, 0.85, 0.6])
    locked = ism_lift_rgb(chroma_lock=1.0, disk_chroma=star)
    loose = ism_lift_rgb(chroma_lock=0.0, disk_chroma=star)
    assert abs(float(locked[1]) - 0.85) < abs(float(loose[1]) - 0.85)


def test_harmonize_pulls_diffuse_not_points() -> None:
    h, w = 64, 128
    disk_w = np.exp(-((np.linspace(-1, 1, h)[:, None] ** 2) / 0.4))
    canvas = np.zeros((h, w, 3), dtype=np.float64)
    canvas[32, 64] = np.array([2.0, 1.8, 1.6])
    canvas += disk_w[..., None] * np.array([0.4, 0.35, 0.55])
    target = np.array([1.0, 0.9, 0.75])
    lu_blur = np.ones((h, w)) * 0.35
    out = harmonize_diffuse_canvas_chroma(
        canvas, target, disk_w, lu_blur, strength=0.9
    )
    assert float(out[32, 64, 2]) < float(canvas[32, 64, 2])
    assert float(np.mean(out)) > 0.01


def test_shared_photon_exposure_raises_diffuse_not_isolated_points() -> None:
    h, w = 48, 96
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk_w = np.broadcast_to(np.exp(-(yy**2) / 0.35), (h, w))
    canvas = disk_w[..., None] * np.array([0.06, 0.05, 0.07])
    canvas[24, 48] = np.array([1.5, 1.4, 1.3])
    stars = np.zeros_like(canvas)
    band = disk_w > 0.35
    stars[band] = np.array([0.14, 0.13, 0.12])
    stars[24, 48] = np.array([0.9, 0.85, 0.8])
    exp = estimate_disk_photon_exposure(stars, disk_w)
    assert abs(exp.linear_gain - 1.0) > 0.05
    lu_blur = np.ones((h, w)) * 0.05
    band_mask = disk_w > 0.3
    mean_before = float(np.mean(canvas[band_mask]))
    out = apply_shared_photon_exposure(
        canvas, exp, disk_w, strength=0.8, diffuse_only=True, diffuse_blur=lu_blur
    )
    assert abs(float(np.mean(out[band_mask])) - mean_before) > 1e-4
    assert float(out[24, 48, 0]) < float(canvas[24, 48, 0]) * 1.02


def test_disk_chroma_from_stars() -> None:
    stars = np.zeros((32, 64, 3), dtype=np.float64)
    disk_w = np.ones((32, 64))
    stars[16, 32] = np.array([1.2, 0.9, 0.5])
    c = disk_chroma_from_star_layer(stars, disk_w)
    assert float(c[0]) > float(c[2])
