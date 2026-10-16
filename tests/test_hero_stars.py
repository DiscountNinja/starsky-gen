"""Hero star selection, lens ghosts, and midplane gold core gate."""

from __future__ import annotations

import numpy as np

from starsky_gen.hero_stars import (
    HeroStarRecord,
    attach_hero_flags,
    hero_min_separation_ok,
    select_hero_indices,
)
from starsky_gen.optical_effects import paint_hero_lens_ghosts
from starsky_gen.structure_envelope import midplane_gold_lat_gate


def _toy_catalog(n: int = 80, *, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Mostly near midplane, a few off-plane.
    lat = rng.normal(0.0, 0.08, size=n)
    lat[0:3] = rng.uniform(0.18, 0.32, size=3) * rng.choice([-1.0, 1.0], size=3)
    return {
        "lon": rng.uniform(0.0, 2.0 * np.pi, size=n),
        "lat": lat,
        "phot_mag": rng.uniform(4.5, 14.0, size=n),
    }


def test_select_hero_indices_count_and_separation() -> None:
    w, h = 256, 128
    cat = _toy_catalog(120, seed=11)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    dw = np.exp(-(yy**2) / 0.12)
    dw = np.broadcast_to(dw, (h, w)).copy()
    idx = select_hero_indices(
        cat,
        width=w,
        height=h,
        disk_weight=dw,
        count=6,
        min_sep_frac=0.10,
        off_plane_slots=2,
        rng=np.random.default_rng(0),
    )
    assert idx.shape[0] == 6
    attach_hero_flags(cat, idx)
    assert int(np.sum(cat["_hero"])) == 6
    from starsky_gen.projections import sph_to_equirect_xy_float

    xf, _ = sph_to_equirect_xy_float(cat["lon"][idx], cat["lat"][idx], w, h)
    assert hero_min_separation_ok(xf, w, min_sep_frac=0.08)


def test_hero_rim_preference_vs_midplane_only() -> None:
    """Rim scoring should place heroes farther from the equator than razor-midplane picks."""
    w, h = 320, 160
    rng = np.random.default_rng(42)
    n = 200
    # Dense midplane + a clear rim population.
    lat_mid = rng.normal(0.0, 0.04, size=n // 2)
    lat_rim = rng.choice([-1.0, 1.0], size=n // 2) * rng.uniform(0.12, 0.26, size=n // 2)
    lat = np.concatenate([lat_mid, lat_rim])
    cat = {
        "lon": rng.uniform(0.0, 2.0 * np.pi, size=n),
        "lat": lat,
        "phot_mag": np.concatenate(
            [rng.uniform(5.0, 9.0, size=n // 2), rng.uniform(4.5, 8.5, size=n // 2)]
        ),
    }
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    dw = np.broadcast_to(np.exp(-(yy**2) / 0.14), (h, w)).copy()
    rim_idx = select_hero_indices(
        cat, width=w, height=h, disk_weight=dw, count=6, off_plane_slots=1, rng=None
    )
    # Synthetic midplane-only score (legacy preference).
    lat_abs = np.abs(cat["lat"])
    bright = np.clip(
        10.0 ** (-0.4 * (cat["phot_mag"] - float(np.median(cat["phot_mag"])))), 0.05, 40.0
    )
    mid_w = np.exp(-((lat_abs / 0.12) ** 2))
    mid_score = bright * (0.25 + 0.75 * mid_w)
    mid_idx = np.argsort(-mid_score)[:6]
    assert float(np.mean(np.abs(cat["lat"][rim_idx]))) > float(
        np.mean(np.abs(cat["lat"][mid_idx]))
    )


def test_paint_hero_lens_ghosts_bounded() -> None:
    rng = np.random.default_rng(2)
    rgb = np.zeros((64, 128, 3), dtype=np.float64)
    heroes = [
        HeroStarRecord(x=20.0, y=32.0, rgb=(0.9, 0.85, 1.0), flux=12.0),
        HeroStarRecord(x=90.0, y=30.0, rgb=(1.0, 0.9, 0.7), flux=8.0),
    ]
    out = paint_hero_lens_ghosts(rgb, heroes, rng, strength=0.08, periodic_x=True)
    assert np.all(np.isfinite(out))
    assert float(np.max(out)) < 0.55
    assert float(np.mean(out)) > 0.0


def test_gold_core_percentile_gate_smoke() -> None:
    h, w = 96, 192
    lat = midplane_gold_lat_gate(h, sigma=0.09, power=2.0)
    gp = np.broadcast_to(lat, (h, w)).astype(np.float64) * 0.8
    # Add weak shoulder noise that must not enter the top-35% core.
    gp = gp + 0.02 * np.random.default_rng(1).random((h, w))
    gold_w = np.clip(gp * lat, 0.0, 1.0)
    nz = gold_w > 1e-5
    thr = float(np.percentile(gold_w[nz], 65.0))
    core = gold_w >= thr
    # Core should concentrate near the equator.
    mid = slice(h // 2 - 4, h // 2 + 4)
    shoulder = slice(0, h // 6)
    assert float(np.mean(core[mid])) > float(np.mean(core[shoulder])) * 3.0
