"""Tests for color_science module."""

import numpy as np

from starsky_gen.color_science import (
    apply_camera_response_linear,
    attenuate_chroma_multipliers,
    blackbody_rgb,
    blend_darken_preserve_contrast,
    ccm89_albedo_per_ebv,
    ccm89_transmission,
    extinction_redden,
    linear_to_srgb,
    sample_teffective_array,
    sample_teffective_for_placement,
    spectral_class_sample,
    warm_teffective_for_core_bulge,
    srgb_to_linear,
    star_chromatic_perturb_weight,
    star_rgb_from_teffective,
    teff_to_bv,
)


def test_srgb_roundtrip() -> None:
    x = np.linspace(0.0, 1.0, 17)
    y = linear_to_srgb(srgb_to_linear(x))
    assert np.allclose(x, y, atol=0.002)


def test_hotter_star_bluer_than_cool() -> None:
    hot = blackbody_rgb(12000.0)
    cool = blackbody_rgb(3800.0)
    assert hot[2] / (hot[0] + 1e-6) > cool[2] / (cool[0] + 1e-6)
    assert cool[0] / (cool[2] + 1e-6) > hot[0] / (hot[2] + 1e-6)


def test_teff_lut_monotonic_blue() -> None:
    temps = [3000.0, 5500.0, 10000.0, 30000.0]
    blues = [blackbody_rgb(t)[2] for t in temps]
    assert blues[-1] > blues[0]


def test_spectral_class_sample() -> None:
    rng = np.random.default_rng(0)
    classes = {spectral_class_sample(rng)[1] for _ in range(200)}
    assert "G" in classes or "K" in classes


def test_teffective_star_color() -> None:
    rng = np.random.default_rng(1)
    rgb = star_rgb_from_teffective(5778.0, rng.normal(0, 0.01, 3))
    assert rgb.shape == (3,)
    assert float(rgb.min()) >= 0.0


def test_spectral_star_less_neutral_than_wash() -> None:
    rng = np.random.default_rng(9)
    hot = star_rgb_from_teffective(12000.0, rng.normal(0, 0.01, 3), camera=True)
    cool = star_rgb_from_teffective(3500.0, rng.normal(0, 0.01, 3), camera=True)
    assert float(np.std(hot)) > 0.04
    assert hot[2] / (hot[0] + 1e-6) > cool[2] / (cool[0] + 1e-6)


def test_camera_response_linear_finite() -> None:
    rgb = np.array([2.0, 1.5, 1.0])
    out = apply_camera_response_linear(rgb)
    assert np.all(np.isfinite(out))
    assert float(out.max()) < float(rgb.max())


def test_teffective_to_bv_solar() -> None:
    bv = teff_to_bv(5778.0)
    assert 0.4 < bv < 0.9


def test_blend_darken_finite() -> None:
    base = np.ones((8, 8, 3)) * 0.5
    factor = np.full((8, 8), 0.4)
    out = blend_darken_preserve_contrast(base, factor)
    assert out.shape == base.shape
    assert float(np.mean(out)) < float(np.mean(base))


def test_ccm89_blue_extinguished_more_than_red() -> None:
    al_b = ccm89_albedo_per_ebv(4400.0)  # B channel
    al_r = ccm89_albedo_per_ebv(7000.0)  # R channel
    assert al_b > al_r


def test_ccm_reddens_white_star() -> None:
    white = np.array([1.0, 1.0, 1.0])
    out = ccm89_transmission(white, a_v=2.0)
    assert out[0] > out[2]
    assert float(np.mean(out)) < 1.0


def test_extinction_redden_uses_av() -> None:
    rgb = np.array([0.5, 0.5, 0.5])
    assert float(np.mean(extinction_redden(rgb, 0.0).astype(float))) == 0.5


def test_faint_star_chroma_perturb_suppressed() -> None:
    w_bright = star_chromatic_perturb_weight(8.0, mag_bright=8.0, mag_faint=20.0)
    w_mid = star_chromatic_perturb_weight(10.0, mag_bright=8.0, mag_faint=20.0)
    w_faint = star_chromatic_perturb_weight(13.0, mag_bright=8.0, mag_faint=20.0)
    assert w_bright > w_mid >= 0.0
    assert w_faint == 0.0
    chroma = attenuate_chroma_multipliers((1.08, 0.94, 1.02), 0.0)
    assert chroma == (1.0, 1.0, 1.0)


def test_core_bulge_remaps_hot_teffective() -> None:
    rng = np.random.default_rng(3)
    warm = [
        warm_teffective_for_core_bulge(15000.0, 0.65, rng)
        for _ in range(200)
    ]
    cool = [
        warm_teffective_for_core_bulge(15000.0, 0.05, rng)
        for _ in range(50)
    ]
    assert float(np.median(warm)) < 6500.0
    assert float(np.median(cool)) > 12000.0


def test_disk_placement_skews_warmer_than_global() -> None:
    rng = np.random.default_rng(7)
    n = 8000
    lat_disk = np.zeros(n, dtype=np.float64)
    lat_halo = np.full(n, 0.85, dtype=np.float64)
    teff_disk = sample_teffective_for_placement(n, lat_disk, rng)
    teff_halo = sample_teffective_for_placement(n, lat_halo, rng)
    teff_global = sample_teffective_array(n, rng)
    assert float(np.mean(teff_disk < 7500.0)) > float(np.mean(teff_global < 7500.0))
    assert float(np.mean(teff_disk < 7500.0)) > float(np.mean(teff_halo < 7500.0))
