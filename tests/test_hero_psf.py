"""Hero (bright blue) stars: compact pin-point PSF."""

import numpy as np

from starsky_gen.hdr import hdr_zeros
from starsky_gen.psf import (
    PsfTuning,
    StarPsfStyle,
    StarPsfVariety,
    _hero_point_star,
    stamp_star_psf,
)


def test_hero_point_detects_hot_star() -> None:
    t = PsfTuning()
    assert _hero_point_star(6.5, 200.0, teff_k=12000.0, bv=-0.05, tuning=t)
    assert not _hero_point_star(12.0, 200.0, teff_k=12000.0, bv=-0.05, tuning=t)


def test_hero_stamp_tighter_than_default_bright() -> None:
    rng = np.random.default_rng(0)
    tune = PsfTuning()
    rgb = np.array([0.75, 0.88, 1.0], dtype=np.float64)
    flux = 280.0
    hero = hdr_zeros(64, 64, 3)
    blob = hdr_zeros(64, 64, 3)
    stamp_star_psf(
        hero,
        32,
        32,
        rgb,
        flux=flux,
        mag=6.0,
        galactic_lat_rad=0.0,
        teff_k=11000.0,
        bv=-0.02,
        rng=rng,
        tuning=tune,
        variety=StarPsfVariety(style=StarPsfStyle.PINPRICK, fwhm_scale=0.64),
    )
    stamp_star_psf(
        blob,
        32,
        32,
        rgb,
        flux=flux,
        mag=6.0,
        galactic_lat_rad=0.0,
        teff_k=5500.0,
        bv=0.65,
        rng=rng,
        tuning=tune,
        variety=StarPsfVariety(style=StarPsfStyle.SATURATED, fwhm_scale=1.55),
    )
    def radius_at(img: np.ndarray, th: float) -> float:
        lu = np.mean(img, axis=2)
        peak = float(lu.max())
        if peak < 1e-8:
            return 0.0
        mask = lu > th * peak
        if not np.any(mask):
            return 0.0
        ys, xs = np.where(mask)
        cy, cx = 32, 32
        return float(np.max(np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)))

    def footprint_px(img: np.ndarray, th: float) -> int:
        lu = np.mean(img, axis=2)
        peak = float(lu.max())
        return int(np.sum(lu > th * peak)) if peak > 1e-8 else 0

    assert float(np.sum(blob)) > float(np.sum(hero)) * 0.85
    assert footprint_px(hero, 0.12) <= footprint_px(blob, 0.06)
