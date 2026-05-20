"""Photoreal magnitude sampling + PSF kernel tests."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats as scipy_stats

from starsky_gen.config import FeatureConfig, OutputFormat, ProjectionMode, RenderConfig
from starsky_gen.generator import render_single
from starsky_gen.psf import (
    PsfTuning,
    build_normalized_gaussian_kernel,
    build_normalized_moffat_kernel,
    flux_from_mag,
    moffat_params_from_mag_and_flux,
    psf_halo_tiers,
    stamp_star_psf,
)
from starsky_gen.starfield import sample_apparent_magnitudes, sample_star_catalog


def test_magnitude_sampling_matches_powerlaw_cdf() -> None:
    rng = np.random.default_rng(0)
    mb, mf = 8.0, 20.0
    alpha = 0.60
    mags = sample_apparent_magnitudes(
        24_500,
        rng,
        mag_bright=mb,
        mag_faint=mf,
        magnitude_log_slope=alpha,
        magnitude_ultra_cut=6.5,
        max_ultra_bright_stars=12,
    )
    hist, edges = np.histogram(mags, bins=np.linspace(mb, mf, num=41))
    centers = 0.5 * (edges[1:] + edges[:-1])
    krate = np.log(10.0) * alpha
    w = centers * 0.0 + (edges[1] - edges[0])
    expect = np.exp(krate * centers)
    pdf = expect / np.sum(expect * w)
    expect_counts = pdf * hist.sum() * w
    valid = hist > 8
    chi2 = np.sum(((hist[valid] - expect_counts[valid]) ** 2) / (expect_counts[valid] + 1e-6))
    p = float(scipy_stats.chi2.sf(chi2, max(int(np.sum(valid)) - 1, 1)))
    assert hist.sum() == mags.size
    assert float(mags.min()) >= mb - 1e-6
    assert float(mags.max()) <= mf + 1e-6
    assert p > 1e-4


def test_moffat_and_gaussian_kernels_normalized() -> None:
    rng = np.random.default_rng(3)
    tune = PsfTuning(mag_bright=8.0, mag_faint=20.0)
    p, hr, hs, ha = moffat_params_from_mag_and_flux(
        11.5, flux=flux_from_mag(11.5, 9.35), galactic_lat_rad=0.1, rng=rng, tuning=tune
    )
    km = build_normalized_moffat_kernel(hr, p)
    assert np.isclose(np.sum(km), 1.0, rtol=1e-10, atol=8e-5)
    hh = int(round(hs * 2.2))
    kg = build_normalized_gaussian_kernel(hh, hs)
    assert np.isclose(np.sum(kg), 1.0, rtol=1e-10, atol=8e-5)


def test_psf_halo_tiers_ultra_bright_wider() -> None:
    tune = PsfTuning()
    g_mid, w_mid, _, _ = psf_halo_tiers(120.0, 11.0, tune)
    g_hi, w_hi, _, bloom = psf_halo_tiers(900.0, 6.0, tune)
    assert g_mid <= 0.016
    assert w_mid < w_hi
    assert w_hi <= 0.016
    assert bloom


def test_dual_psf_stamp_increases_local_flux() -> None:
    rng = np.random.default_rng(9)
    canvas = np.zeros((64, 64, 3), dtype=np.float64)
    rgb = np.array([0.8, 0.85, 0.95])
    stamp_star_psf(canvas, 32, 32, rgb, flux=120.0, mag=5.5, galactic_lat_rad=0.0, rng=rng)
    peak = float(np.max(canvas))
    assert peak > 0.5


def test_band_density_increases_catalog_size() -> None:
    rng = np.random.default_rng(1)
    c1 = sample_star_catalog(rng, 512, 256, 1.0, band_star_density_scale=1.0)
    c3 = sample_star_catalog(rng, 512, 256, 1.0, band_star_density_scale=3.0)
    assert c3["lon"].shape[0] > c1["lon"].shape[0] * 2.4


def test_crowding_raises_band_luma(tmp_path: Path) -> None:
    base = FeatureConfig(
        jpeg_artifact_pass=False,
        photoreal_stars=True,
        nebula=False,
        reference_anchors=False,
        band_star_density_scale=1.0,
    )
    dense = base.model_copy(update={"band_star_density_scale": 3.0})
    cfg_lo = RenderConfig(
        width=256,
        height=128,
        output_base_name="crowd_lo",
        output_dir=tmp_path,
        generations=1,
        seed=42,
        projection_mode=ProjectionMode.equirectangular,
        output_format=OutputFormat.png,
        features=base,
    )
    cfg_hi = cfg_lo.model_copy(update={"output_base_name": "crowd_hi", "features": dense})
    p_lo, _ = render_single(cfg_lo, 0)
    p_hi, _ = render_single(cfg_hi, 0)
    img_lo = np.mean(np.asarray(Image.open(p_lo["equirectangular"]).convert("RGB")), axis=2)
    img_hi = np.mean(np.asarray(Image.open(p_hi["equirectangular"]).convert("RGB")), axis=2)
    cy = img_lo.shape[0] // 2
    band = slice(cy - 16, cy + 16)
    assert float(np.mean(img_hi[band])) > float(np.mean(img_lo[band])) * 1.04


def test_photoreal_small_render_finite(tmp_path: Path) -> None:
    cfg = RenderConfig(
        width=288,
        height=144,
        output_base_name="photoreal_smoke",
        output_dir=tmp_path,
        generations=1,
        seed=7,
        projection_mode=ProjectionMode.equirectangular,
        output_format=OutputFormat.png,
        features=FeatureConfig(
            jpeg_artifact_pass=False,
            photoreal_stars=True,
            galaxy_tone_curve="asinh",
            nebula=False,
            reference_anchors=False,
        ),
    )
    paths, stats = render_single(cfg, 0)
    img = np.asarray(Image.open(paths["equirectangular"]).convert("RGB"))
    assert img.shape[:2] == (144, 288)
    assert np.any(img > 0)
    assert "color_counts" in stats
