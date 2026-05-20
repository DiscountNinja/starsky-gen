"""Unresolved speckle should be occluded under thick gas, not additive-through."""

import numpy as np

from starsky_gen.generator import _mute_speckle_under_nebula


def test_mute_speckle_under_nebula_reduces_bright_gas() -> None:
    canvas = np.ones((32, 48, 3), dtype=np.float64) * 0.5
    neb = np.zeros((32, 48), dtype=np.float64)
    neb[8:24, 10:38] = 0.85
    gas_mask = np.ones((32, 48), dtype=np.float64)
    out = _mute_speckle_under_nebula(
        canvas,
        neb,
        gas_mask=gas_mask,
        ext_paint=np.ones((32, 48)) * 0.7,
        strength=0.8,
    )
    assert float(out[16, 24].mean()) < float(canvas[16, 24].mean()) * 0.55
    assert float(out[2, 4].mean()) > float(out[16, 24].mean())
