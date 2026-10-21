"""Composite-path sandwich: lane dropout, rifts, post-star protect, type competition."""

from __future__ import annotations

import numpy as np

from starsky_gen.config import FeatureConfig, LookTarget, RenderConfig, finalize_render_config
from starsky_gen.galactic_structure import compute_resolve_weight_field
from starsky_gen.generator import (
    apply_dark_rifts_to_extinction,
    apply_post_star_lane_protect,
    build_dark_rift_mask,
)
from starsky_gen.postfx import apply_tri_scale_bloom
from starsky_gen.structure_envelope import (
    _softmax_compete_channels,
    build_structure_architecture_weights,
)


def test_lane_gated_resolve_weight_drops_in_dust_lane() -> None:
    h, w = 32, 64
    g = np.ones((h, w)) * 0.8
    d = np.ones((h, w)) * 0.35  # opaque-ish
    disk = np.ones((h, w))
    lane = np.zeros((h, w))
    lane[h // 2 - 2 : h // 2 + 2, :] = 1.0
    base = compute_resolve_weight_field(g, d, disk, drop_strength=0.2)
    gated = compute_resolve_weight_field(
        g,
        d,
        disk,
        drop_strength=0.2,
        dust_lane_weight=lane,
        lane_dropout_boost=0.85,
        lane_dropout_power=1.1,
    )
    mid = slice(h // 2 - 2, h // 2 + 2)
    shell = slice(0, 4)
    assert float(gated[mid].mean()) < float(base[mid].mean()) * 0.85
    assert float(gated[shell].mean()) > float(gated[mid].mean())


def test_softmax_competition_increases_channel_contrast() -> None:
    a = np.full((8, 8), 0.55)
    b = np.full((8, 8), 0.45)
    soft = _softmax_compete_channels([a, b], competition=0.0)
    hard = _softmax_compete_channels([a, b], competition=0.9)
    soft_gap = float(np.mean(np.abs(soft[0] - soft[1])))
    hard_gap = float(np.mean(np.abs(hard[0] - hard[1])))
    assert hard_gap > soft_gap


def test_architecture_competition_changes_dust_lane() -> None:
    rng = np.random.default_rng(3)
    h, w = 48, 96
    disk = np.exp(-((np.linspace(-1, 1, h)[:, None]) ** 2) / 0.12)
    disk = np.broadcast_to(disk, (h, w))
    ve = rng.random((h, w)) * 0.6
    surv = rng.random((h, w))
    puff = rng.random((h, w))
    turb = rng.random((h, w))
    blend = build_structure_architecture_weights(
        disk, ve, surv, puff, turb, strength=1.0, type_competition=0.0, periodic_x=True
    )
    compete = build_structure_architecture_weights(
        disk, ve, surv, puff, turb, strength=1.0, type_competition=0.8, periodic_x=True
    )
    # Competition should not collapse all mass to zero.
    assert float(compete.dust_lane_weight.max()) > 0.05
    assert float(np.mean(np.abs(compete.dust_lane_weight - blend.dust_lane_weight))) > 1e-4


def test_dark_rifts_carve_transmission() -> None:
    rng = np.random.default_rng(5)
    h, w = 40, 80
    disk = np.broadcast_to(np.exp(-((np.linspace(-1, 1, h)[:, None]) ** 2) / 0.1), (h, w))
    lon = np.linspace(0.6, 1.4, w)
    mask = build_dark_rift_mask(
        h, w, rng, disk, lon, coverage=0.25, strength=1.0, periodic_x=True
    )
    assert float(mask.max()) > 0.05
    ext = np.ones((h, w)) * 0.75
    out = apply_dark_rifts_to_extinction(ext, mask, void_floor=0.06)
    assert float(out.mean()) < float(ext.mean())
    sel = mask > (0.35 * float(mask.max()))
    assert bool(np.any(sel))
    assert float(out[sel].mean()) < float(ext[sel].mean())


def test_post_star_lane_protect_darkens_midplane() -> None:
    h, w = 48, 96
    canvas = np.full((h, w, 3), 0.55)
    ext = np.ones((h, w)) * 0.85
    ext[h // 2 - 2 : h // 2 + 2, :] = 0.12
    lane = np.zeros((h, w))
    lane[h // 2 - 2 : h // 2 + 2, :] = 1.0
    disk = np.ones((h, w))
    out = apply_post_star_lane_protect(
        canvas,
        ext_paint=ext,
        disk_w=disk,
        dust_lane_weight=lane,
        rift_mask=None,
        strength=1.10,
        periodic_x=True,
        hero_luma_floor=0.95,
    )
    mid = out[h // 2, w // 2].mean()
    shell = out[4, w // 2].mean()
    assert mid < shell
    assert mid < 0.55


def test_star_layer_bloom_periodic_seam() -> None:
    h, w = 32, 64
    rgb = np.zeros((h, w, 3))
    rgb[:, 0, :] = 0.98
    rgb[:, -1, :] = 0.98
    disk = np.ones((h, w))
    out = apply_tri_scale_bloom(
        rgb,
        disk,
        strength=0.1,
        threshold=0.86,
        mix_tight=0.62,
        mix_mid=0.25,
        mix_wide=0.05,
        periodic_x=True,
    )
    seam = float(np.mean(np.abs(out[:, 0, :] - out[:, -1, :])))
    assert seam < 0.1


def test_finalize_cinematic_enables_sandwich_hooks() -> None:
    cfg = finalize_render_config(
        RenderConfig(features=FeatureConfig(look_target=LookTarget.cinematic))
    )
    f = cfg.features
    assert f.lane_star_dropout_boost >= 0.50
    assert f.post_star_lane_protect_strength >= 0.40
    assert f.dark_rift_strength >= 0.70
    assert f.star_layer_bloom_strength >= 0.10
    # Soft cloud volume: moderate type competition (not razor lane/shell sandwich).
    assert 0.30 <= f.structure_type_competition <= 0.50
    assert f.nebula_haze_strength <= 0.30
    assert f.long_exposure_look is False
