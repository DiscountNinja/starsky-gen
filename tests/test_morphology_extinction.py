"""Morphology-driven extinction should mask asymmetrically."""

import numpy as np

from starsky_gen.dust_field import (
    apply_missing_region_extinction,
    band_relative_clearance,
    build_morphology_extinction_transmission,
    compose_constructive_extinction_transmission,
    emission_clearance_from_extinction,
    fragment_lane_absorption,
    gas_clearance_from_extinction,
)
from starsky_gen.generator import _apply_band_dark_patches
from starsky_gen.galactic_structure import (
    build_galactic_morphology,
    local_field_variance_stretch,
    resolved_keep_probability,
)


def test_morphology_extinction_has_deep_lanes() -> None:
    rng = np.random.default_rng(31)
    m = build_galactic_morphology(128, 64, rng, seed_perturb_scale=1.0)
    ext = build_morphology_extinction_transmission(
        m,
        void_floor=0.016,
        filament_strength=1.15,
        discontinuity_strength=1.25,
        absorption_contrast=1.25,
        extinction_strength=1.22,
    )
    assert float(ext.min()) <= 0.05
    dark = ext < 0.05
    clear = ext > 0.08
    assert int(dark.sum()) > 120
    assert int(clear.sum()) > 60
    assert float(ext[dark].mean()) < float(ext[clear].mean()) * 0.55


def test_fragment_lane_absorption_breaks_smooth_blobs() -> None:
    h, w = 128, 256
    rng = np.random.default_rng(4)
    blob = np.zeros((h, w), dtype=np.float64)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    blob += np.exp(-((yy**2) / 0.08)) * 0.85
    d = rng.random((h, w))
    t = rng.random((h, w))
    out = fragment_lane_absorption(blob, d, t, periodic_x=True, strength=0.8)
    gx_o = float(np.abs(np.diff(out, axis=1)).mean())
    gx_b = float(np.abs(np.diff(blob, axis=1)).mean())
    assert gx_o > gx_b * 1.02
    assert float(np.mean(out)) >= float(np.mean(blob)) * 0.55


def test_missing_region_extinction_darkens_lanes() -> None:
    canvas = np.ones((32, 48, 3), dtype=np.float64) * 0.8
    ext = np.linspace(0.9, 0.01, 48, dtype=np.float64)[np.newaxis, :]
    ext = np.broadcast_to(ext, (32, 48))
    out = apply_missing_region_extinction(canvas, ext, void_floor=0.004, missing_boost=0.55)
    assert float(out[:, 0].mean()) > float(out[:, -1].mean()) * 1.4


def test_local_variance_stretch_preserves_mean() -> None:
    rng = np.random.default_rng(7)
    field = rng.random((32, 48))
    out = local_field_variance_stretch(field, variance=1.4)
    assert abs(float(np.mean(out)) - float(np.mean(field))) < 0.02
    assert float(np.std(out)) > float(np.std(field)) * 1.1


def test_obliteration_crushes_survival_in_dense_dusty_patches() -> None:
    rng = np.random.default_rng(11)
    m = build_galactic_morphology(96, 48, rng, obliteration_strength=0.9)
    obl = m.obliteration_mask
    assert float(obl.max()) > 0.35
    p_clear = resolved_keep_probability(
        g=0.85,
        dust_t=0.9,
        disk_w=0.8,
        mag=12.0,
        mag_bright=8.0,
        mag_faint=18.0,
        dropout_strength=0.5,
        mid_layer=False,
        resolve_weight=0.75,
        obliteration=0.05,
    )
    p_oblit = resolved_keep_probability(
        g=0.85,
        dust_t=0.9,
        disk_w=0.8,
        mag=12.0,
        mag_bright=8.0,
        mag_faint=18.0,
        dropout_strength=0.5,
        mid_layer=False,
        resolve_weight=0.75,
        obliteration=0.92,
    )
    assert p_oblit < p_clear * 0.55
    assert float(np.mean(obl[obl > 0.12])) > 0.22


def test_generation_phase_changes_morphology() -> None:
    m0 = build_galactic_morphology(64, 32, np.random.default_rng(5), generation_phase=0.0)
    m1 = build_galactic_morphology(
        64, 32, np.random.default_rng(5), generation_phase=2.1, regional_chaos=0.5
    )
    on = m0.disk_weight > 0.35
    assert bool(np.any(on))
    assert not np.allclose(m0.dust_absorption[on], m1.dust_absorption[on], atol=0.002)


def test_turbulent_cloud_breakup_adds_lane_texture() -> None:
    from starsky_gen.dust_field import apply_turbulent_cloud_breakup

    rng = np.random.default_rng(3)
    h, w = 96, 160
    trans = np.ones((h, w), dtype=np.float64) * 0.72
    dust = rng.random((h, w))
    turb = rng.random((h, w))
    ridge = rng.random((h, w))
    out = apply_turbulent_cloud_breakup(
        trans, dust, turb, ridge, void_floor=0.01, strength=0.75
    )
    gx = float(np.abs(np.diff(out, axis=1)).mean())
    gx_in = float(np.abs(np.diff(trans, axis=1)).mean())
    assert gx > gx_in * 1.08
    assert float(out.min()) < float(trans.min()) * 0.92


def test_extinction_diverges_from_dust_transmission() -> None:
    rng = np.random.default_rng(17)
    m = build_galactic_morphology(128, 256, rng, seed_perturb_scale=1.0)
    dust = m.dust_transmission
    ext = build_morphology_extinction_transmission(
        m,
        void_floor=0.016,
        filament_strength=1.15,
        fine_texture_strength=0.64,
    )
    corr = float(np.corrcoef(ext.ravel(), dust.ravel())[0, 1])
    assert corr < 0.985
    assert float(np.mean(ext < 0.05)) >= float(np.mean(dust < 0.12)) * 0.35


def test_compose_constructive_extinction_adds_fragmentation() -> None:
    from starsky_gen.procedural_noise import gaussian_blur_pil

    h, w = 96, 192
    rng = np.random.default_rng(2)
    dust = np.clip(0.06 + rng.random((h, w)) * 0.9, 0.06, 1.0)
    fil = gaussian_blur_pil(dust, 18.0, periodic_x=True)
    out = compose_constructive_extinction_transmission(
        fil, dust, void_floor=0.02, detail_strength=0.8, periodic_x=True
    )
    gx_out = float(np.abs(np.diff(out, axis=1)).mean())
    gx_fil = float(np.abs(np.diff(fil, axis=1)).mean())
    assert gx_out > gx_fil * 1.05
    assert not np.allclose(out, dust, atol=0.02)


def test_band_transmission_from_morph_absorption_spreads() -> None:
    from starsky_gen.dust_field import band_transmission_from_morph_absorption

    rng = np.random.default_rng(3)
    h, w = 64, 128
    fil = np.full((h, w), 0.008, dtype=np.float64)
    morph = np.clip(0.08 + rng.random((h, w)) * 0.72, 0.0, 1.0)
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.12)), (h, w)).copy()
    out = band_transmission_from_morph_absorption(
        fil, morph, disk, void_floor=0.004, clear_max=0.28
    )
    on = disk > 0.35
    assert float(out[on].std()) > 0.012
    assert float(out[on].max()) <= 0.30
    assert float(np.mean(out[on] <= 0.02)) < 0.55
    assert float(np.mean(out[on] >= 0.22)) < 0.58


def test_gas_clearance_keeps_band_visibility_when_ext_at_floor() -> None:
    ext = np.full((64, 128), 0.048, dtype=np.float64)
    ext[20:44, 40:90] = np.linspace(0.05, 0.22, 50, dtype=np.float64)[None, :]
    yy = np.linspace(-1, 1, 64, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.12)), ext.shape).copy()
    ext_flat = np.full((64, 128), 0.048, dtype=np.float64)
    legacy = emission_clearance_from_extinction(ext_flat, floor=0.05, power=1.02)
    gas = gas_clearance_from_extinction(ext, floor=0.05, power=0.92, min_clear=0.32, band_weight=disk)
    on = disk > 0.35
    assert float(legacy[on].mean()) < 0.01
    assert float(gas[on].mean()) >= 0.28
    assert float(gas[on].std()) > 0.08


def test_band_relative_clearance_spans_capped_transmission() -> None:
    ext = np.full((64, 128), 0.048, dtype=np.float64)
    ext[20:44, 40:90] = np.linspace(0.05, 0.22, 50, dtype=np.float64)[None, :]
    yy = np.linspace(-1, 1, 64, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.12)), ext.shape).copy()
    legacy = np.clip(0.06 + 0.94 * ext**1.28, 0.0, 1.0)
    rel = band_relative_clearance(ext, disk, min_clear=0.14)
    on = disk > 0.35
    assert float(rel[on].mean()) > float(legacy[on].mean()) * 1.8
    assert float(rel[on].max()) > 0.55


def test_band_dark_patches_keep_majority_of_band_bright() -> None:
    rng = np.random.default_rng(9)
    m = build_galactic_morphology(96, 192, rng)
    ext = build_morphology_extinction_transmission(m)
    h, w = ext.shape
    canvas = np.ones((h, w, 3), dtype=np.float64) * 0.42
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.12)), (h, w)).copy()
    out = _apply_band_dark_patches(
        canvas,
        ext,
        disk,
        m,
        strength=0.58,
        periodic_x=True,
        morph_primary=True,
    )
    lu = np.mean(out, axis=2)
    on = disk > 0.35
    assert float(np.mean(lu[on] < 0.22)) < 0.42


def test_morphology_extinction_varies_by_seed() -> None:
    m0 = build_galactic_morphology(96, 48, np.random.default_rng(1))
    m1 = build_galactic_morphology(96, 48, np.random.default_rng(2))
    e0 = build_morphology_extinction_transmission(m0)
    e1 = build_morphology_extinction_transmission(m1)
    assert not np.allclose(e0, e1, atol=0.05)
