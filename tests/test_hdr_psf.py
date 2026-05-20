"""HDR dtype and physical PSF structure tests."""

import numpy as np

from starsky_gen.hdr import HDR_DTYPE, as_hdr, hdr_zeros
from starsky_gen.psf import (
    PsfTier,
    PsfTuning,
    StarPsfStyle,
    StarPsfVariety,
    flux_from_mag,
    moffat_params_from_mag_and_flux,
    stamp_star_psf,
)


def test_hdr_zeros_float32() -> None:
    a = hdr_zeros(8, 8, 3)
    assert a.dtype == HDR_DTYPE
    assert as_hdr(a).dtype == HDR_DTYPE


def test_physical_psf_core_peak_and_wings() -> None:
    rng = np.random.default_rng(4)
    canvas = hdr_zeros(96, 96, 3)
    rgb = np.array([0.85, 0.90, 1.0], dtype=np.float64)
    flux = 220.0
    stamp_star_psf(
        canvas,
        48,
        48,
        rgb,
        flux=flux,
        mag=6.0,
        galactic_lat_rad=0.0,
        rng=rng,
        tuning=PsfTuning(),
        variety=StarPsfVariety(style=StarPsfStyle.STANDARD, fwhm_scale=1.05),
    )
    peak = float(np.max(canvas))
    assert peak > 1.0
    patch = canvas[40:56, 40:56]
    assert float(np.sum(patch > peak * 0.05)) >= 3


def test_fwhm_scale_controls_size_not_flux_alone() -> None:
    rng = np.random.default_rng(7)
    tune = PsfTuning()
    ref = 9.35
    flux_bright = flux_from_mag(6.0, ref)
    flux_faint = flux_from_mag(16.0, ref)
    p_tiny_bright, _, _, _ = moffat_params_from_mag_and_flux(
        6.0,
        flux=flux_bright,
        galactic_lat_rad=0.0,
        rng=rng,
        tuning=tune,
        fwhm_scale=0.58,
    )
    p_blob_faint, _, _, _ = moffat_params_from_mag_and_flux(
        16.0,
        flux=flux_faint,
        galactic_lat_rad=0.0,
        rng=rng,
        tuning=tune,
        fwhm_scale=1.65,
    )
    assert p_blob_faint["sigma_x"] > p_tiny_bright["sigma_x"]


def test_fwhm_lottery_mostly_tiny() -> None:
    from starsky_gen.psf import sample_psf_fwhm_scale

    rng = np.random.default_rng(99)
    tune = PsfTuning()
    scales = [
        sample_psf_fwhm_scale(rng, 10.0, 50.0, tuning=tune) for _ in range(500)
    ]
    tiny = sum(1 for s in scales if s < 0.72)
    assert tiny >= 380
    assert max(scales) <= 0.96


def test_psf_tier_foreground_wider_than_background() -> None:
    rng = np.random.default_rng(11)
    tune = PsfTuning()
    flux = 180.0
    p_bg, _, _, _ = moffat_params_from_mag_and_flux(
        7.0,
        flux=flux,
        galactic_lat_rad=0.0,
        rng=rng,
        tuning=tune,
        tier=PsfTier.BACKGROUND,
    )
    p_fg, _, _, _ = moffat_params_from_mag_and_flux(
        7.0,
        flux=flux,
        galactic_lat_rad=0.0,
        rng=rng,
        tuning=tune,
        tier=PsfTier.FOREGROUND,
    )
    assert p_fg["sigma_x"] > p_bg["sigma_x"]
