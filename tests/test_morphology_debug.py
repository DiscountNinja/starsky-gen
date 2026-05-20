"""Morphology fields, deposit-driven unresolved, and render profile wiring.

Visual validation::

    PYTHONPATH=src python3 -m starsky_gen.cli generate \\
        --debug-layers --grayscale-morphology --seed 42

Inspect ``{output}_layers/morphology_gray.png``, ``unresolved_deposit.png``,
``dropout_mask.png``, and ``extinction.png``.
"""

import numpy as np

from starsky_gen.color_science import star_chromatic_perturb_weight
from starsky_gen.galactic_structure import (
    GalacticMorphology,
    apply_unresolved_field_to_canvas,
    build_galactic_morphology,
    deposit_catalog_unresolved_flux,
    morphology_grayscale_preview,
    resolved_keep_probability,
)
from starsky_gen.psf import flux_from_mag, sample_psf_fwhm_scale


def test_seed_perturb_changes_morphology_maps() -> None:
    m0 = build_galactic_morphology(96, 48, np.random.default_rng(100), seed_perturb_scale=1.0)
    m1 = build_galactic_morphology(96, 48, np.random.default_rng(101), seed_perturb_scale=1.0)
    m_same = build_galactic_morphology(96, 48, np.random.default_rng(100), seed_perturb_scale=1.0)
    assert not np.allclose(m0.star_formation, m1.star_formation, atol=0.02)
    assert not np.allclose(m0.dust_absorption, m1.dust_absorption, atol=0.02)
    assert not np.allclose(m0.cluster_prob, m1.cluster_prob, atol=0.02)
    assert np.allclose(m0.star_formation, m_same.star_formation, rtol=0.02)


def test_seed_perturb_scale_zero_skips_epsilon() -> None:
    off_a = build_galactic_morphology(64, 32, np.random.default_rng(7), seed_perturb_scale=0.0)
    off_b = build_galactic_morphology(64, 32, np.random.default_rng(7), seed_perturb_scale=0.0)
    on = build_galactic_morphology(64, 32, np.random.default_rng(7), seed_perturb_scale=1.0)
    assert np.allclose(off_a.cluster_prob, off_b.cluster_prob)
    assert not np.allclose(off_a.star_formation, on.star_formation, atol=0.02)


def test_resolve_weight_inherits_from_unresolved() -> None:
    rng = np.random.default_rng(12)
    m = build_galactic_morphology(96, 48, rng, drop_strength=0.55)
    u = m.unresolved_prior
    w = m.resolve_weight
    hi_u = u > float(np.percentile(u, 96.0))
    lo_u = u < float(np.percentile(u, 40.0))
    assert float(w[hi_u].mean()) <= float(w[lo_u].mean()) + 0.12
    rate = m.unresolved_speckle_rate()
    assert float(rate[hi_u].mean()) > float(rate[lo_u].mean())


def test_morphology_has_resolve_weight_and_void() -> None:
    rng = np.random.default_rng(7)
    m = build_galactic_morphology(128, 64, rng)
    assert isinstance(m, GalacticMorphology)
    assert m.resolve_weight.shape == (64, 128)
    assert m.unresolved_coarse.shape == (64, 128)
    assert float(m.void_mask.max()) <= 1.0
    assert m.extinction_maps is not None


def test_void_reduces_peak_density() -> None:
    m0 = build_galactic_morphology(
        128, 64, np.random.default_rng(8), void_strength=0.0, macro_void_count=0
    )
    m1 = build_galactic_morphology(
        128, 64, np.random.default_rng(8), void_strength=0.85, macro_void_count=0
    )
    assert float(m1.void_mask.mean()) > float(m0.void_mask.mean())
    hi_void = m1.void_mask > 0.35
    if np.any(hi_void):
        assert float(m1.stellar_density[hi_void].mean()) < float(
            m0.stellar_density[hi_void].mean()
        )


def test_faint_chroma_suppressed_harder() -> None:
    w = star_chromatic_perturb_weight(11.0, mag_bright=8.0, mag_faint=20.0)
    assert w == 0.0
    w2 = star_chromatic_perturb_weight(8.0, mag_bright=8.0, mag_faint=20.0)
    assert w2 > 0.15


def test_grayscale_preview_bounded() -> None:
    rng = np.random.default_rng(9)
    m = build_galactic_morphology(64, 32, rng)
    g = morphology_grayscale_preview(m)
    assert g.shape == (32, 64)
    assert float(g.max()) <= 1.0 + 1e-6


def test_discontinuity_cut_lowers_g_in_disrupted_pixels() -> None:
    rng = np.random.default_rng(11)
    m0 = build_galactic_morphology(128, 64, rng, discontinuity_cut_strength=0.0)
    m1 = build_galactic_morphology(128, 64, rng, discontinuity_cut_strength=0.85)
    assert m0.extinction_maps is not None and m1.extinction_maps is not None
    disc = m1.extinction_maps.disruption
    hi = disc > float(np.percentile(disc, 88.0))
    assert float(m1.stellar_density[hi].mean()) < float(m0.stellar_density[hi].mean())


def test_batch_deposit_increases_accum_and_skip_mask() -> None:
    rng = np.random.default_rng(21)
    m = build_galactic_morphology(96, 48, rng, drop_strength=0.62)
    n = 400
    lon = rng.uniform(0.0, 2.0 * np.pi, size=n)
    lat = rng.uniform(-0.35, 0.35, size=n)
    mags = rng.uniform(11.0, 18.0, size=n)
    catalog = {
        "lon": lon,
        "lat": lat,
        "phot_mag": mags,
        "dust_visibility": np.full(n, 0.92),
    }
    skip = deposit_catalog_unresolved_flux(
        catalog,
        m,
        rng,
        width=96,
        height=48,
        magnitude_ref_mag=9.35,
        mag_bright=8.0,
        mag_faint=20.0,
        dropout_strength=0.62,
    )
    assert skip.sum() > 10
    assert float(m.unresolved_accum.sum()) > 1e-6


def test_cull_increases_unresolved_accum() -> None:
    from starsky_gen.starfield import cull_faint_resolved_stars

    rng = np.random.default_rng(22)
    m = build_galactic_morphology(96, 48, rng)
    n = 300
    catalog = {
        "lon": rng.uniform(0.0, 2.0 * np.pi, size=n),
        "lat": rng.uniform(-0.3, 0.3, size=n),
        "phot_mag": rng.uniform(13.0, 17.5, size=n),
        "dust_visibility": np.ones(n),
    }
    before = float(m.unresolved_accum.sum())
    cull_faint_resolved_stars(
        catalog,
        rng,
        mag_faint_floor=13.4,
        dropout_strength=0.78,
        mag_faint=20.0,
        magnitude_ultra_cut=6.5,
        galactic_structure=m,
        magnitude_ref_mag=9.35,
        width=96,
        height=48,
    )
    assert float(m.unresolved_accum.sum()) > before


def test_psf_fwhm_lottery_favors_tiny_at_faint_mag() -> None:
    rng = np.random.default_rng(33)
    scales = [
        sample_psf_fwhm_scale(rng, 14.0, flux_from_mag(14.0, 9.35))
        for _ in range(200)
    ]
    tiny = sum(1 for s in scales if s < 0.72)
    assert tiny >= 140


def test_deposit_primary_unresolved_path_runs() -> None:
    rng = np.random.default_rng(44)
    m = build_galactic_morphology(64, 32, rng)
    m.unresolved_accum[16, 32] = 2.5
    canvas = np.zeros((32, 64, 3), dtype=np.float64)
    out = apply_unresolved_field_to_canvas(
        canvas, rng, m, texture_strength=1.0, deposit_primary=True
    )
    assert float(out.max()) > 0.0
