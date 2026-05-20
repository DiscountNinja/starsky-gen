"""Depth layering: faint cull preserves bright stars."""

import numpy as np

from starsky_gen.starfield import cull_faint_resolved_stars, sample_star_catalog


def test_cull_faint_keeps_ultra_bright() -> None:
    rng = np.random.default_rng(0)
    n = 800
    mags = np.concatenate(
        [
            rng.uniform(6.0, 7.5, size=12),
            rng.uniform(13.0, 19.0, size=n - 12),
        ]
    )
    cat = {
        "lon": rng.uniform(0, 6.28, size=n),
        "lat": rng.normal(0, 0.2, size=n),
        "color_idx": np.zeros(n, dtype=np.int64),
        "size_idx": np.zeros(n, dtype=np.int64),
        "jitter": rng.normal(0, 0.01, size=(n, 3)),
        "bv": rng.normal(0.3, 0.4, size=n),
        "phot_mag": mags,
    }
    out = cull_faint_resolved_stars(
        cat,
        rng,
        mag_faint_floor=12.0,
        dropout_strength=0.65,
        mag_faint=20.0,
        magnitude_ultra_cut=6.8,
    )
    kept = out["phot_mag"]
    assert kept.size < n * 0.82
    assert np.sum(kept < 7.2) >= 8
    assert float(np.median(kept)) > 13.5


def test_foreground_density_scale_reduces_count() -> None:
    rng = np.random.default_rng(1)
    dense = sample_star_catalog(
        rng, 1920, 1080, 1.0, layer="foreground", foreground_star_density_scale=1.0
    )
    sparse = sample_star_catalog(
        rng, 1920, 1080, 1.0, layer="foreground", foreground_star_density_scale=0.65
    )
    assert sparse["lon"].size < dense["lon"].size * 0.88
