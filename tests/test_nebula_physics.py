"""Nebula spectral composition and CCM dust attenuation tests."""

import numpy as np

from starsky_gen.color_science import ccm_transmission_from_av
from starsky_gen.nebula_physics import (
    apply_ccm_extinction_linear,
    compose_line_emission_rgb,
    forward_scatter_hg_volume,
    forward_scatter_mie,
    henyey_greenstein,
)


def test_ccm_transmission_blue_dimmer_than_red() -> None:
    av = np.full((8, 8), 2.0)
    trans = ccm_transmission_from_av(av)
    assert trans.shape == (8, 8, 3)
    assert float(np.mean(trans[..., 2])) < float(np.mean(trans[..., 0]))


def test_ccm_extinction_preserves_channels() -> None:
    rgb = np.ones((4, 4, 3)) * 0.8
    ext = np.full((4, 4), 0.35)
    out = apply_ccm_extinction_linear(rgb, ext, av_scale=2.5)
    assert out.shape == rgb.shape
    assert float(np.mean(out)) < float(np.mean(rgb))


def test_line_emission_capped_and_red_halpha() -> None:
    hii = np.zeros((16, 16))
    hii[8, 8] = 1.0
    emit = compose_line_emission_rgb(hii, hii * 0.5, hii * 0.3, emit_cap=0.7)
    assert float(emit.max()) <= 0.71
    assert float(emit[8, 8, 0]) > float(emit[8, 8, 2])


def _blur_stub(field: np.ndarray, passes: int = 1, periodic_x: bool = False) -> np.ndarray:
    """Minimal box blur so volumetric scatter spreads energy in tests."""
    _ = periodic_x
    f = np.asarray(field, dtype=np.float64)
    out = f.copy()
    for _ in range(max(1, int(passes))):
        if out.ndim == 2:
            acc = (
                np.roll(out, 1, 0)
                + np.roll(out, -1, 0)
                + np.roll(out, 1, 1)
                + np.roll(out, -1, 1)
                + out
            ) / 5.0
        else:
            acc = np.stack([_blur_stub(out[:, :, c], passes=1) for c in range(3)], axis=2)
        out = acc
    return out


def test_forward_scatter_increases_local_energy() -> None:
    rgb = np.zeros((32, 32, 3))
    rgb[16, 16] = [2.0, 1.8, 1.6]
    hot = np.zeros((32, 32))
    hot[16, 16] = 2.0
    out = forward_scatter_mie(rgb, hot, strength=0.12, blur_fn=_blur_stub)
    assert float(out.sum()) > float(rgb.sum())


def test_henyey_greenstein_forward_peak() -> None:
    g = 0.7
    p_fwd = float(henyey_greenstein(0.95, g))
    p_back = float(henyey_greenstein(-0.5, g))
    assert p_fwd > p_back


def test_hg_volume_scatter_soft_halo() -> None:
    rgb = np.zeros((48, 48, 3))
    rgb[24, 24] = [1.5, 1.4, 1.2]
    hot = np.zeros((48, 48))
    hot[24, 24] = 1.4
    out = forward_scatter_hg_volume(rgb, hot, strength=0.14, blur_fn=_blur_stub)
    ring = out[20:28, 20:28].sum()
    assert ring > float(rgb[20:28, 20:28].sum())
