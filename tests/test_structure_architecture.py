"""Structure architecture weight plan — band-wide turbulent drive."""

from __future__ import annotations

import numpy as np


def test_architecture_drive_high_in_band_core() -> None:
    from starsky_gen.structure_envelope import (
        build_structure_architecture_weights,
        build_vertical_structure_envelope,
    )

    h, w = 256, 512
    rng = np.random.default_rng(19)
    yy = np.linspace(-1, 1, h)[:, None]
    disk_w = np.broadcast_to(np.exp(-((yy**2) / 0.10)), (h, w))
    disk_w = disk_w / (float(np.max(disk_w)) + 1e-8)
    g = np.clip(disk_w * 0.6 + rng.random((h, w)) * 0.08, 0.0, 1.0)
    surv, ve, _ = build_vertical_structure_envelope(
        h, w, rng, g, disk_w, band_lat_sigma=0.10, extent_strength=0.72, host_latitude_scale=1.85, periodic_x=True
    )
    pf = rng.random((h, w))
    turb = rng.random((h, w))
    arch = build_structure_architecture_weights(
        disk_w, ve, surv, pf, turb, band_core_drive=0.52, strength=1.0, periodic_x=True
    )
    core = disk_w > 0.55
    rim = (disk_w > 0.12) & (disk_w < 0.45)
    assert float(np.mean(arch.turbulence_drive[core])) > 0.32
    assert float(np.mean(arch.turbulence_drive[core])) >= float(np.mean(arch.turbulence_drive[rim])) * 0.85


def test_architecture_pipeline_changes_dust() -> None:
    from starsky_gen.structure_envelope import (
        apply_structure_architecture_pipeline,
        build_structure_architecture_weights,
        build_vertical_structure_envelope,
    )

    h, w = 128, 256
    rng = np.random.default_rng(3)
    yy = np.linspace(-1, 1, h)[:, None]
    disk_w = np.broadcast_to(np.exp(-((yy**2) / 0.11)), (h, w))
    g = disk_w.copy()
    surv, ve, _ = build_vertical_structure_envelope(
        h, w, rng, g, disk_w, band_lat_sigma=0.10, extent_strength=0.8, host_latitude_scale=1.85, periodic_x=True
    )
    pf = rng.random((h, w)) * 0.5 + 0.2
    turb = rng.random((h, w))
    arch = build_structure_architecture_weights(
        disk_w, ve, surv, pf, turb, strength=1.0, periodic_x=True
    )
    dust = np.full((h, w), 0.42, dtype=np.float64)
    out = apply_structure_architecture_pipeline(
        dust,
        arch,
        ve,
        surv,
        pf,
        rng,
        turb,
        turb,
        disk_w,
        periodic_x=True,
        reinject_strength=0.9,
        puff_inject_strength=0.5,
        cloud_blend_strength=1.0,
    )
    on = disk_w > 0.35
    assert float(np.std(out[on])) > float(np.std(dust[on])) * 1.04


def test_architecture_type_map_rgb_shape() -> None:
    from starsky_gen.structure_envelope import (
        ARCHITECTURE_TYPE_COLORS,
        build_structure_architecture_type_map,
        build_structure_architecture_weights,
    )

    h, w = 64, 128
    rng = np.random.default_rng(1)
    disk_w = np.broadcast_to(np.exp(-((np.linspace(-1, 1, h)[:, None] ** 2) / 0.12)), (h, w))
    ve = rng.random((h, w))
    surv = np.full((h, w), 0.8)
    arch = build_structure_architecture_weights(
        disk_w, ve, surv, rng.random((h, w)), rng.random((h, w)), strength=1.0, periodic_x=True
    )
    rgb = build_structure_architecture_type_map(arch)
    assert rgb.shape == (h, w, 3)
    assert float(np.max(rgb)) <= 1.0 + 1e-9
    assert len(ARCHITECTURE_TYPE_COLORS) >= 5
