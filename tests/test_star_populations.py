"""Star populations: cosmic, halo, galactic overdensity."""

from __future__ import annotations

import numpy as np

from starsky_gen.galactic_structure import build_galactic_morphology
from starsky_gen.starfield import (
    inject_galactic_overdensity_stars,
    sample_halo_star_catalog,
    sample_isotropic_cosmic_catalog,
    sample_star_catalog,
)


def test_cosmic_catalog_isotropic_latitudes() -> None:
    rng = np.random.default_rng(42)
    cat = sample_isotropic_cosmic_catalog(rng, 1024, 512, 1.0, anchor_count=4)
    lat = cat["lat"]
    assert lat.size > 500
    assert float(np.std(lat)) > 0.45
    assert float(np.percentile(np.abs(lat), 95)) > 0.75


def test_halo_catalog_broader_than_disk() -> None:
    rng = np.random.default_rng(7)
    cat = sample_halo_star_catalog(rng, 1024, 512, 1.0, halo_lat_sigma=0.55)
    assert cat["lon"].size > 40
    assert float(np.std(cat["lat"])) > 0.08


def test_inject_overdensity_adds_stars() -> None:
    rng = np.random.default_rng(99)
    morph = build_galactic_morphology(512, 256, rng)
    base = sample_star_catalog(rng, 512, 256, 1.0, galactic_structure=morph)
    n0 = int(base["lon"].size)
    out = inject_galactic_overdensity_stars(
        base, rng, morph, 512, 256, count=120, attach_apparent_mag=True
    )
    assert int(out["lon"].size) == n0 + 120
    assert "teff" in out


def test_cull_with_dust_visibility_after_inject() -> None:
    from starsky_gen.galactic_structure import attach_dust_visibility_to_catalog
    from starsky_gen.starfield import cull_faint_resolved_stars

    rng = np.random.default_rng(42)
    morph = build_galactic_morphology(512, 256, rng)
    cat = sample_star_catalog(
        rng, 512, 256, 1.0, galactic_structure=morph, attach_apparent_mag=True
    )
    attach_dust_visibility_to_catalog(cat, morph, 512, 256)
    cat = inject_galactic_overdensity_stars(
        cat, rng, morph, 512, 256, count=80, attach_apparent_mag=True
    )
    attach_dust_visibility_to_catalog(cat, morph, 512, 256)
    culled = cull_faint_resolved_stars(
        cat,
        rng,
        galactic_structure=morph,
        mag_faint_floor=13.4,
        dropout_strength=0.58,
        mag_faint=20.0,
        magnitude_ultra_cut=6.5,
        magnitude_ref_mag=9.35,
        width=512,
        height=256,
    )
    assert culled["phot_mag"].size <= cat["phot_mag"].size
    if "dust_visibility" in culled:
        assert culled["dust_visibility"].shape[0] == culled["lon"].shape[0]
