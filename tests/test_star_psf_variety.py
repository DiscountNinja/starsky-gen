"""Per-star PSF optical families."""

import numpy as np

from starsky_gen.hdr import hdr_zeros
from starsky_gen.psf import (
    PsfTuning,
    StarPsfStyle,
    StarPsfVariety,
    sample_star_psf_variety,
    stamp_star_psf,
)


def _footprint_px(img: np.ndarray, th: float) -> int:
    lu = np.mean(img, axis=2)
    peak = float(lu.max())
    return int(np.sum(lu > th * peak)) if peak > 1e-8 else 0


def test_sample_variety_covers_multiple_styles() -> None:
    rng = np.random.default_rng(42)
    styles: set[StarPsfStyle] = set()
    for _ in range(400):
        v = sample_star_psf_variety(
            rng,
            7.5,
            180.0,
            teff_k=9500.0,
            bv=0.0,
            extinction_t=0.35,
            local_density=0.6,
        )
        styles.add(v.style)
    assert StarPsfStyle.PINPRICK in styles
    assert StarPsfStyle.SOFT_SEEING in styles or StarPsfStyle.EXTINCTED in styles
    assert len(styles) >= 4


def test_pinprick_tighter_than_soft_seeing() -> None:
    rng = np.random.default_rng(0)
    tune = PsfTuning()
    rgb = np.array([0.8, 0.9, 1.0], dtype=np.float64)
    pin = hdr_zeros(64, 64, 3)
    soft = hdr_zeros(64, 64, 3)
    stamp_star_psf(
        pin,
        32,
        32,
        rgb,
        flux=240.0,
        mag=6.5,
        galactic_lat_rad=0.0,
        rng=rng,
        tuning=tune,
        variety=StarPsfVariety(
            style=StarPsfStyle.PINPRICK,
            fwhm_scale=0.62,
            sigma_scale=0.9,
            halo_scale=0.85,
        ),
    )
    stamp_star_psf(
        soft,
        32,
        32,
        rgb,
        flux=240.0,
        mag=6.5,
        galactic_lat_rad=0.0,
        rng=rng,
        tuning=tune,
        variety=StarPsfVariety(
            style=StarPsfStyle.SOFT_SEEING,
            fwhm_scale=1.48,
            sigma_scale=1.32,
            halo_scale=1.28,
            wing_scale=1.3,
        ),
    )
    assert _footprint_px(soft, 0.06) >= _footprint_px(pin, 0.06) + 1


def test_stack_twin_adds_offset_energy() -> None:
    rng = np.random.default_rng(1)
    tune = PsfTuning()
    rgb = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    single = hdr_zeros(48, 48, 3)
    twin = hdr_zeros(48, 48, 3)
    stamp_star_psf(
        single,
        24,
        24,
        rgb,
        flux=120.0,
        mag=8.0,
        galactic_lat_rad=None,
        rng=rng,
        tuning=tune,
        variety=StarPsfVariety(style=StarPsfStyle.STANDARD),
    )
    stamp_star_psf(
        twin,
        24,
        24,
        rgb,
        flux=120.0,
        mag=8.0,
        galactic_lat_rad=None,
        rng=rng,
        tuning=tune,
        variety=StarPsfVariety(
            style=StarPsfStyle.STACK_TWIN,
            stack_dx=0.42,
            stack_dy=-0.35,
            stack_flux_frac=0.45,
        ),
    )
    assert float(np.sum(twin)) > float(np.sum(single)) * 1.12
