"""Tests for integrated galactic structure field."""

import numpy as np

from starsky_gen.galactic_structure import (
    GalacticStructure,
    apply_unresolved_field_to_canvas,
    attach_dust_visibility_to_catalog,
    build_galactic_structure,
    resolved_keep_probability,
    sample_foreground_lon_lat_from_structure,
    sample_hierarchical_from_structure,
)
from starsky_gen.starfield import cull_faint_resolved_stars, reroll_stars_in_dark_lanes


def test_merge_extinction_preserves_morph_dust_maps() -> None:
    m = build_galactic_structure(128, 256, np.random.default_rng(11), dust_micro_strength=1.0)
    morph_a = m.dust_absorption_morph.copy()
    morph_t = m.dust_transmission_morph.copy()
    ext = np.clip(m.dust_transmission * 0.25, 0.02, 1.0)
    m.merge_nebula_extinction(ext)
    assert np.allclose(m.dust_absorption_morph, morph_a)
    assert np.allclose(m.dust_transmission_morph, morph_t)
    on = m.disk_weight > 0.35
    assert float(m.dust_absorption[on].max()) > float(morph_a[on].max()) + 0.05


def test_fine_puff_field_has_high_frequency_breakup() -> None:
    from starsky_gen.debug_trace import highpass_band_stats
    from starsky_gen.structure_envelope import build_fine_puff_field

    puff = build_fine_puff_field(
        np.random.default_rng(7), 512, 1024, periodic_x=True, strength=1.0
    )
    assert float(highpass_band_stats(puff)["hp_std"]) > 0.03


def test_morphology_turbulent_gas_field_has_band_variance() -> None:
    from starsky_gen.structure_envelope import build_morphology_turbulent_gas_field

    m = build_galactic_structure(512, 1024, np.random.default_rng(42), dust_micro_strength=1.0)
    ext = np.clip(m.dust_transmission_morph * 0.85 + 0.08, 0.05, 0.32)
    gas = build_morphology_turbulent_gas_field(
        m, ext, periodic_x=True, rng=np.random.default_rng(42)
    )
    on = m.disk_weight > 0.35
    assert float(gas[on].std()) > 0.012
    assert float(gas[on].max()) > float(gas[on].min()) + 0.08


def test_dust_micro_concentrated_in_galactic_band() -> None:
    hi = build_galactic_structure(128, 256, np.random.default_rng(21), dust_micro_strength=1.0)
    lo = build_galactic_structure(128, 256, np.random.default_rng(21), dust_micro_strength=0.0)
    delta = np.abs(hi.dust_absorption - lo.dust_absorption)
    on = hi.disk_weight > 0.38
    off = hi.disk_weight < 0.28
    assert int(on.sum()) > 100 and int(off.sum()) > 100
    assert float(delta[on].mean()) > float(delta[off].mean()) * 1.5
    assert float(delta[on].mean()) > 0.04


def test_dust_transmission_microstructure_increases_edge_energy() -> None:
    from starsky_gen.procedural_noise import gaussian_blur_pil

    def _band_hp_std(morph) -> float:
        soft = gaussian_blur_pil(morph.dust_transmission, 2.5, periodic_x=True)
        hp = np.abs(morph.dust_transmission - soft)
        on = morph.disk_weight > 0.35
        return float(np.std(hp[on])) if bool(np.any(on)) else 0.0

    ratios: list[float] = []
    for seed in (9, 17, 31, 42):
        hi = build_galactic_structure(128, 256, np.random.default_rng(seed), dust_micro_strength=1.0)
        lo = build_galactic_structure(128, 256, np.random.default_rng(seed), dust_micro_strength=0.0)
        ratios.append(_band_hp_std(hi) / max(_band_hp_std(lo), 1e-9))
    assert float(np.median(ratios)) >= 1.15


def test_build_galactic_structure_shapes() -> None:
    rng = np.random.default_rng(0)
    g = build_galactic_structure(128, 64, rng)
    assert g.stellar_density.shape == (64, 128)
    assert g.dust_transmission.shape == (64, 128)
    assert g.unresolved_prior.shape == (64, 128)
    assert float(g.stellar_density.max()) <= 1.0 + 1e-6


def test_hierarchical_from_structure_count() -> None:
    rng = np.random.default_rng(1)
    st = build_galactic_structure(256, 128, rng)
    lon, lat = sample_hierarchical_from_structure(rng, 300, st)
    assert lon.size == lat.size == 300


def test_foreground_uses_structure() -> None:
    rng = np.random.default_rng(2)
    st = build_galactic_structure(256, 128, rng)
    lon, lat = sample_foreground_lon_lat_from_structure(rng, 80, st)
    assert lon.size == 80


def test_reroll_attaches_visibility_not_teleport() -> None:
    rng = np.random.default_rng(3)
    st = build_galactic_structure(128, 64, rng)
    lon0 = np.array([1.0, 2.0, 3.0])
    cat = {"lon": lon0.copy(), "lat": np.zeros(3)}
    ext = np.ones((64, 128)) * 0.2
    reroll_stars_in_dark_lanes(
        cat, rng, 128, 64, ext, galactic_structure=st
    )
    assert np.allclose(cat["lon"], lon0)
    assert "dust_visibility" in cat


def test_resolved_keep_brighter_more_likely() -> None:
    p_faint = resolved_keep_probability(
        g=0.8,
        dust_t=0.9,
        disk_w=0.9,
        mag=14.0,
        mag_bright=8.0,
        mag_faint=20.0,
        dropout_strength=0.5,
        mid_layer=False,
    )
    p_bright = resolved_keep_probability(
        g=0.8,
        dust_t=0.9,
        disk_w=0.9,
        mag=8.5,
        mag_bright=8.0,
        mag_faint=20.0,
        dropout_strength=0.5,
        mid_layer=False,
    )
    assert p_bright > p_faint


def test_unresolved_deposit_and_render() -> None:
    rng = np.random.default_rng(4)
    st = build_galactic_structure(64, 32, rng)
    st.deposit_unresolved(10, 20, 0.05)
    canvas = np.zeros((32, 64, 3), dtype=np.float64)
    out = apply_unresolved_field_to_canvas(canvas, rng, st, texture_strength=1.0)
    assert float(out.max()) > 0.0


def test_cull_deposits_to_structure() -> None:
    rng = np.random.default_rng(5)
    st = build_galactic_structure(128, 64, rng)
    n = 200
    cat = {
        "lon": rng.uniform(0, 2 * np.pi, n),
        "lat": rng.normal(0, 0.2, n),
        "phot_mag": rng.uniform(11.0, 19.0, n),
    }
    u0 = float(st.unresolved_accum.sum())
    cull_faint_resolved_stars(
        cat,
        rng,
        mag_faint_floor=10.0,
        dropout_strength=0.9,
        mag_faint=20.0,
        magnitude_ultra_cut=6.0,
        galactic_structure=st,
        width=128,
        height=64,
    )
    assert float(st.unresolved_accum.sum()) >= u0
