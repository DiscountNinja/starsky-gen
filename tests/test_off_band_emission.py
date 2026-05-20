"""Off-band H II mask decoupled from galactic band."""

import numpy as np

from starsky_gen.galactic_structure import build_galactic_morphology
from starsky_gen.generator import _build_off_band_hii_layer
from starsky_gen.structure_envelope import (
    build_off_band_mask,
    build_warped_emission_blob,
    localize_emission_clouds,
)


def test_off_band_mask_peaks_away_from_plane() -> None:
    h, w = 64, 128
    dw = np.zeros((h, w), dtype=np.float64)
    dw[h // 2 - 2 : h // 2 + 3, :] = 1.0
    off = build_off_band_mask(dw, None, h, decouple_strength=1.0)
    assert float(off[h // 2, w // 2]) < float(off[4, w // 2]) * 0.65
    assert float(np.max(off)) > 0.35


def test_off_band_mask_uses_vertical_extent() -> None:
    m = build_galactic_morphology(96, 48, np.random.default_rng(9))
    off = build_off_band_mask(
        m.disk_weight, m.vertical_extent, m.height, decouple_strength=1.0
    )
    assert float(np.std(off[m.disk_weight < 0.5])) > 0.05


def test_off_band_hii_layer_non_square() -> None:
    """Blob placement must index (h, w) grids, not 1-D yy/xx ravels."""
    m = build_galactic_morphology(1024, 2048, np.random.default_rng(42))
    dw = m.disk_weight
    layer = _build_off_band_hii_layer(
        m,
        dw,
        strength=2.0,
        periodic_x=True,
        blob_count=5,
        hii_seed=424242,
    )
    assert layer.shape == (m.height, m.width, 3)
    assert float(np.max(layer)) > 1e-6


def test_off_band_hii_seed_stable() -> None:
    m = build_galactic_morphology(128, 256, np.random.default_rng(1))
    dw = m.disk_weight
    a = _build_off_band_hii_layer(
        m, dw, strength=2.0, periodic_x=True, blob_count=4, hii_seed=99
    )
    b = _build_off_band_hii_layer(
        m, dw, strength=2.0, periodic_x=True, blob_count=4, hii_seed=99
    )
    c = _build_off_band_hii_layer(
        m, dw, strength=2.0, periodic_x=True, blob_count=4, hii_seed=100
    )
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_localize_emission_clouds_shrinks_footprint() -> None:
    h, w = 128, 256
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    wash = np.clip(0.35 + 0.25 * yy**2, 0.0, 1.0)
    mask = np.clip(1.0 - np.abs(yy) * 0.55, 0.0, 1.0)
    out = localize_emission_clouds(wash, mask, periodic_x=True)
    active = mask > 0.2
    frac_before = float(np.mean(wash[active] > 0.25))
    frac_after = float(np.mean(out[active] > 0.25))
    assert frac_after < frac_before * 0.55


def test_warped_emission_blob_not_radially_uniform() -> None:
    h, w = 128, 256
    rng = np.random.default_rng(7)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    cy, cx = 0.15, -0.2
    blob = build_warped_emission_blob(
        rng, h, w, center_y=cy, center_x=cx, extent_y=0.14, extent_x=0.10, support_mask=np.ones((h, w)), periodic_x=True
    )
    ang = np.arctan2(yy - cy, xx - cx)
    bins = [blob[(ang >= a) & (ang < a + np.pi / 4)] for a in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    means = [float(np.mean(b)) if b.size > 8 else 0.0 for b in bins]
    assert max(means) > min(means) * 1.12
