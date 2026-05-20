"""Vertical structure envelope, brutal erasure, longitude asymmetry."""

import numpy as np

from starsky_gen.galactic_structure import build_galactic_morphology, resolved_keep_probability
from starsky_gen.procedural_noise import fbm2d
from starsky_gen.structure_envelope import (
    apply_brutal_erasure_transmission,
    apply_variable_band_thickness,
    build_asymmetric_band_bleed_envelope,
    build_disk_mesoscale_thickness_field,
    build_hii_emission_hierarchy,
    build_emission_morphology_field,
    build_gold_population_field,
    build_ism_scale_hierarchy,
    build_longitude_asymmetry,
    build_morphology_ism_rgb,
    build_vertical_structure_envelope,
    latitude_plane_gate,
    reinject_cloud_microstructure,
    reinject_vertical_dust_structure,
    seam_safe_lon_weights,
    wisp_vertical_bleed_gate,
)


def test_latitude_plane_gate_suppresses_poles() -> None:
    g = latitude_plane_gate(64, sigma=0.40, power=1.0)
    assert g.shape == (64, 1)
    assert float(g[0, 0]) < 0.10
    assert float(g[31, 0]) > 0.95
    assert float(g[-1, 0]) < 0.10


def test_disk_mesoscale_thickness_varies_scale_height() -> None:
    h, w = 128, 256
    yy = np.linspace(-1, 1, h)[:, None]
    dw = np.broadcast_to(np.exp(-((yy**2) / 0.12)), (h, w))
    rng = np.random.default_rng(17)
    meso, shear = build_disk_mesoscale_thickness_field(
        dw, rng, periodic_x=True, strength=0.62
    )
    assert float(np.std(meso[dw > 0.35])) > 0.12
    assert float(np.min(meso)) < -0.08
    assert float(np.max(meso)) > 0.10
    assert float(np.max(np.abs(shear[dw > 0.35]))) > 0.006


def test_hii_emission_hierarchy_has_meso_complexes() -> None:
    from starsky_gen.procedural_noise import gaussian_blur_pil

    h, w = 256, 512
    yy = np.linspace(-1, 1, h)[:, None]
    host = np.exp(-((yy**2) / 0.12)) * 0.85
    rng = np.random.default_rng(99)
    out = build_hii_emission_hierarchy(host, rng, periodic_x=True, strength=1.0)
    scale = float(max(h, w))
    mega_blur = gaussian_blur_pil(out, scale * 0.10, periodic_x=True)
    meso = np.clip(out - mega_blur, 0.0, 1.0)
    assert float(np.std(out)) > float(np.std(mega_blur)) * 0.55
    assert float(np.std(meso)) > 0.025
    peaks = int(np.sum((meso > 0.18) & (host > 0.3)))
    assert peaks > 60
    h, w = 128, 256
    yy = np.linspace(-1, 1, h)[:, None]
    dw = np.broadcast_to(np.exp(-((yy**2) / 0.11)), (h, w))
    rng = np.random.default_rng(42)
    out, kind, meso = apply_variable_band_thickness(
        dw,
        rng,
        band_lat_sigma=0.12,
        jitter_strength=0.92,
        band_curvature_amp=0.08,
        thickness_asymmetry=0.45,
        mesoscale_strength=0.55,
        periodic_x=True,
    )
    assert kind in ("s", "u", "w", "flat")
    mid = h // 2
    upper = out[: mid - 4, :]
    lower = out[mid + 4 :, :]
    upper_fwhm = float(np.mean(np.sum(upper > 0.42, axis=0)))
    lower_fwhm = float(np.mean(np.sum(lower > 0.42, axis=0)))
    ratio = max(upper_fwhm, lower_fwhm) / (min(upper_fwhm, lower_fwhm) + 1e-6)
    assert ratio > 1.22
    from starsky_gen.structure_envelope import sample_band_lat_curve

    lat_curve, _ = sample_band_lat_curve(rng, w, band_curvature_amp=0.08)
    assert float(np.std(lat_curve)) > 0.010


def test_asymmetric_band_bleed_breaks_parallel_rim() -> None:
    h, w = 64, 128
    yy = np.linspace(-1, 1, h)[:, None]
    dw = np.exp(-((yy**2) / 0.10))
    rng = np.random.default_rng(11)
    env = build_asymmetric_band_bleed_envelope(dw, h, w, rng, bleed_strength=0.85)
    gy = np.gradient(env, axis=0)
    rim = (dw[:, 0] > 0.25) & (dw[:, 0] < 0.72)
    assert float(np.std(env[rim, :])) > 0.028
    assert float(np.max(env) - float(np.min(env[rim, :]))) > 0.115


def test_wisp_vertical_bleed_gate_has_halo_outside_plane() -> None:
    h, w = 64, 128
    pg = latitude_plane_gate(h, sigma=0.48, power=0.96)
    bg = latitude_plane_gate(h, sigma=0.74, power=0.84)
    wisp, halo = wisp_vertical_bleed_gate(h, w, plane_gate=pg, bleed_gate=bg)
    assert wisp.shape == (h, w)
    assert halo.shape == (h, w)
    mid = h // 2
    rim = h // 4
    assert float(wisp[mid, 0]) >= float(pg[mid, 0]) * 0.98
    assert float(np.max(halo)) > 0.08
    assert float(halo[rim, 0]) > 0.12
    assert float(wisp[rim, 0]) > float(pg[rim, 0]) * 0.85


def test_reinject_vertical_dust_raises_offband_and_hp() -> None:
    rng = np.random.default_rng(7)
    h, w = 64, 128
    dust = np.full((h, w), 0.22, dtype=np.float64)
    ve = fbm2d(rng, 16, 32, base_scale=0.2, octaves=3)
    ve = np.clip(ve, 0.0, 1.0)
    from starsky_gen.procedural_noise import _resize_bilinear

    ve = _resize_bilinear(ve, h, w)
    turb = _resize_bilinear(fbm2d(rng, 16, 32, base_scale=0.25, octaves=3), h, w)
    ridge = _resize_bilinear(fbm2d(rng, 16, 32, base_scale=0.18, octaves=3), h, w)
    dw = np.exp(-((np.linspace(-1, 1, h)[:, None] ** 2) / 0.12))
    surv = np.clip(0.4 + 0.6 * ve, 0.0, 1.35)
    out = reinject_vertical_dust_structure(
        dust, ve, surv, turb, ridge, dw, strength=0.9
    )
    yy = np.linspace(-1, 1, h)[:, None]
    off = np.broadcast_to(np.abs(yy) > 0.35, (h, w))
    assert float(np.mean(out[off])) > float(np.mean(dust[off])) + 0.02
    assert float(np.std(out - dust)) > 0.01


def test_structure_survival_extends_off_plane() -> None:
    rng = np.random.default_rng(19)
    m = build_galactic_morphology(
        96,
        48,
        rng,
        vertical_extent_strength=0.85,
        structure_host_latitude_scale=2.0,
    )
    h, w = m.height, m.width
    yy = np.linspace(-1, 1, h)[:, None]
    band_core = np.broadcast_to(np.exp(-((yy**2) / 0.08)), (h, w))
    off_plane = m.structure_survival[band_core < 0.2]
    in_band = m.structure_survival[band_core > 0.7]
    assert float(np.mean(off_plane)) > 0.08
    assert float(np.mean(in_band)) > float(np.mean(off_plane)) * 1.2


def test_brutal_erasure_drives_low_transmission() -> None:
    trans = np.ones((32, 48)) * 0.75
    smooth = np.zeros((32, 48))
    smooth[10:18, 20:30] = 0.90
    out_smooth = apply_brutal_erasure_transmission(trans, smooth, survival_floor=0.05)
    assert float(out_smooth[15, 25]) > 0.40
    filament = np.zeros((32, 48))
    filament[15, 18:30] = 1.0
    filament[14, 20:28] = 0.78
    out_fil = apply_brutal_erasure_transmission(trans, filament, survival_floor=0.05)
    assert float(out_fil[15, 24]) <= 0.10
    assert float(out_fil[2, 2]) > 0.5


def test_longitude_asymmetry_right_hotter() -> None:
    lon = build_longitude_asymmetry(128, strength=0.9)
    left = float(np.mean(lon[:, :32]))
    right = float(np.mean(lon[:, -32:]))
    assert right > left * 1.12


def test_seam_guard_tapers_edges() -> None:
    w = seam_safe_lon_weights(256, guard_frac=0.06)
    assert float(w[0, 0]) < float(w[0, 128]) * 0.5


def test_gold_population_has_young_and_old_patches() -> None:
    m = build_galactic_morphology(96, 48, np.random.default_rng(23))
    gp = m.gold_population_weight
    in_band = gp[m.structure_survival > 0.12]
    assert float(np.std(in_band)) > 0.08
    assert float(np.percentile(in_band, 12)) < 0.26
    assert float(np.percentile(in_band, 88)) > float(np.percentile(in_band, 12)) * 1.42


def test_brutal_survival_below_soft_obliteration() -> None:
    p_soft = resolved_keep_probability(
        g=0.8,
        dust_t=0.85,
        disk_w=0.75,
        mag=11.0,
        mag_bright=8.0,
        mag_faint=18.0,
        dropout_strength=0.5,
        mid_layer=False,
        resolve_weight=0.7,
        obliteration=0.25,
    )
    p_brutal = resolved_keep_probability(
        g=0.8,
        dust_t=0.85,
        disk_w=0.75,
        mag=11.0,
        mag_bright=8.0,
        mag_faint=18.0,
        dropout_strength=0.5,
        mid_layer=False,
        resolve_weight=0.7,
        obliteration=0.25,
        brutal_erasure=0.9,
    )
    assert p_brutal <= 0.08
    assert p_brutal < p_soft * 0.55


def test_reinject_cloud_microstructure_adds_detail() -> None:
    rng = np.random.default_rng(3)
    field = rng.random((128, 256))
    hier = build_ism_scale_hierarchy(field, rng, 128, 256, strength=0.95, periodic_x=True, turbulence_weight=1.5)
    out = reinject_cloud_microstructure(
        hier, periodic_x=True, detail_mix=0.55, rng=rng, channel_salt=2
    )
    gx_h = float(np.abs(np.diff(hier, axis=1)).mean())
    gx_o = float(np.abs(np.diff(out, axis=1)).mean())
    assert gx_o >= gx_h * 0.90
    assert float(np.max(out) - np.max(hier)) >= -0.02


def test_ism_scale_hierarchy_has_four_scales() -> None:
    from starsky_gen.procedural_noise import gaussian_blur_pil

    rng = np.random.default_rng(5)
    field = rng.random((256, 512))
    out = build_ism_scale_hierarchy(field, rng, 256, 512, strength=0.9, periodic_x=True, turbulence_weight=1.2)
    mega = gaussian_blur_pil(field, 40.0, periodic_x=True)
    med = gaussian_blur_pil(field, 14.0, periodic_x=True)
    assert out.shape == field.shape
    assert float(np.std(out - mega)) > 0.02
    corr_mega_med = float(np.corrcoef(mega.ravel(), med.ravel())[0, 1])
    corr_out_mega = float(np.corrcoef(out.ravel(), mega.ravel())[0, 1])
    assert corr_out_mega < corr_mega_med + 0.08
    grad_out = float(np.abs(np.diff(out, axis=1)).mean())
    grad_mega = float(np.abs(np.diff(mega, axis=1)).mean())
    assert grad_out > grad_mega * 1.06


def test_competitive_scale_hierarchy_beats_constructive_blend() -> None:
    from starsky_gen.procedural_noise import gaussian_blur_pil
    from starsky_gen.structure_envelope import blend_competitive_scale_hierarchy

    rng = np.random.default_rng(21)
    field = rng.random((128, 256))
    h, w = field.shape
    scale = float(max(h, w))
    mega = gaussian_blur_pil(field, scale * 0.085, periodic_x=True)
    med_blur = gaussian_blur_pil(field, scale * 0.028, periodic_x=True)
    from starsky_gen.structure_envelope import _independent_turbulence_fields

    med_n, small_n, fine_n = _independent_turbulence_fields(rng, h, w, periodic_x=True)
    medium = np.clip(med_n * 0.6 + med_blur * 0.12, 0.0, 1.0)
    small = np.clip(small_n * 0.7, 0.0, 1.0)
    fine = np.clip(fine_n * 0.65, 0.0, 1.0)
    constructive = np.clip(
        mega * 0.45
        + medium * 0.25 * (1.0 + 0.5 * mega)
        + small * 0.20 * (1.0 + medium)
        + fine * 0.10 * (1.0 + small),
        0.0,
        1.0,
    )
    competitive = blend_competitive_scale_hierarchy(
        mega, medium, small, fine, periodic_x=True, competition_strength=1.0
    )
    grad_c = float(np.abs(np.diff(constructive, axis=1)).mean())
    grad_x = float(np.abs(np.diff(competitive, axis=1)).mean())
    assert grad_x > grad_c * 1.04
    assert float(np.std(competitive - mega)) > float(np.std(constructive - mega)) * 0.95


def test_ism_constructive_hierarchy_less_mega_locked() -> None:
    from starsky_gen.procedural_noise import gaussian_blur_pil

    rng = np.random.default_rng(19)
    field = rng.random((128, 256))
    out = build_ism_scale_hierarchy(field, rng, 128, 256, strength=0.95, periodic_x=True, turbulence_weight=1.2)
    mega = gaussian_blur_pil(field, 40.0, periodic_x=True)
    corr_out_mega = float(np.corrcoef(out.ravel(), mega.ravel())[0, 1])
    assert corr_out_mega < 0.92
    grad_out = float(np.abs(np.diff(out, axis=1)).mean())
    grad_mega = float(np.abs(np.diff(mega, axis=1)).mean())
    assert grad_out > grad_mega * 1.12


def test_ism_hierarchy_less_single_scale_than_blur_stack() -> None:
    from starsky_gen.procedural_noise import gaussian_blur_pil

    rng = np.random.default_rng(11)
    field = rng.random((128, 256))
    h, w = field.shape
    scale = float(max(h, w))
    mega = gaussian_blur_pil(field, scale * 0.085, periodic_x=True)
    med = gaussian_blur_pil(field, scale * 0.028, periodic_x=True)
    old_single = np.clip(mega * 0.68 + med * 0.26, 0.0, 1.0)
    out = build_ism_scale_hierarchy(field, rng, h, w, strength=0.95, periodic_x=True, turbulence_weight=1.5)
    assert float(np.std(out - old_single)) > 0.04


def test_morphology_ism_rgb_has_multi_scale_detail() -> None:
    m = build_galactic_morphology(128, 256, np.random.default_rng(8))
    rgb, luma = build_morphology_ism_rgb(
        m, np.random.default_rng(8), m.height, m.width, hierarchy_strength=0.96, periodic_x=True
    )
    assert rgb.shape == (m.height, m.width, 3)
    assert luma.shape == (m.height, m.width)
    gx = float(np.abs(np.diff(luma, axis=1)).mean())
    assert gx > 0.003
    assert float(np.std(luma)) > 0.06


def test_emission_morphology_tracks_clear_sight() -> None:
    m = build_galactic_morphology(64, 128, np.random.default_rng(3))
    ext = np.clip(m.dust_transmission, 0.02, 1.0)
    emit = build_emission_morphology_field(
        ext,
        m.star_formation,
        m.structure_survival,
        m.dust_absorption,
        periodic_x=True,
        rng=np.random.default_rng(3),
    )
    assert emit.shape == ext.shape
    assert float(np.std(emit)) > 0.03
    clear = ext > 0.55
    dark = ext < 0.12
    if int(clear.sum()) > 8 and int(dark.sum()) > 8:
        assert float(emit[clear].mean()) > float(emit[dark].mean()) * 1.35
