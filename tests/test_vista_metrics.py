"""Vista acceptance metrics and BandPlan sandwich authority."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from starsky_gen.band_plan import apply_band_plan_lane_protect, build_band_plan
from starsky_gen.config import (
    FeatureConfig,
    LookTarget,
    NebulaMode,
    OutputFormat,
    ProjectionMode,
    RenderConfig,
    finalize_render_config,
)
from starsky_gen.generator import render_single
from starsky_gen.vista_metrics import (
    assert_cinematic_vista,
    measure_vista_equirect,
)


def _synthetic_cinematic_band(h: int = 64, w: int = 128) -> np.ndarray:
    """Warm midplane + cooler shells + void poles (glow-tolerant cinematic silhouette)."""
    rgb = np.full((h, w, 3), 0.04)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    mid = np.exp(-(yy**2) / 0.04)
    shell = np.clip(np.exp(-(yy**2) / 0.16) - mid * 0.55, 0.0, 1.0)
    # Warm core glow (allowed midplane light) + cooler shell tint.
    rgb += mid[..., np.newaxis] * np.array([0.42, 0.22, 0.10])
    rgb += shell[..., np.newaxis] * np.array([0.18, 0.24, 0.32])
    # Longitude tint variation for chroma_spread.
    xx = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
    rgb[..., 0] += 0.06 * mid[:, 0:1] * np.sin(2.0 * np.pi * xx)
    rgb[..., 2] += 0.05 * shell[:, 0:1] * np.cos(2.0 * np.pi * xx)
    return np.clip(rgb, 0.0, 1.0)


def test_measure_vista_glow_tolerant_structure() -> None:
    rgb = _synthetic_cinematic_band()
    m = measure_vista_equirect(rgb)
    assert m["plane_luma"] > m["void_luma"]
    assert m["chroma_spread"] > 0.01
    assert_cinematic_vista(m)


def test_assert_cinematic_vista_fails_flat_field() -> None:
    rgb = np.full((48, 96, 3), 0.35)
    m = measure_vista_equirect(rgb)
    with pytest.raises(AssertionError):
        assert_cinematic_vista(m)


def test_band_plan_lane_protect_soft_shoulders() -> None:
    h, w = 48, 96
    canvas = np.full((h, w, 3), 0.55)
    ext = np.ones((h, w)) * 0.85
    ext[h // 2 - 2 : h // 2 + 2, :] = 0.10
    lane = np.zeros((h, w))
    lane[h // 2 - 2 : h // 2 + 2, :] = 1.0
    disk = np.ones((h, w))
    plan = build_band_plan(
        height=h,
        width=w,
        disk_w=disk,
        ext_paint=ext,
        dust_lane_weight=lane,
        rift_mask=None,
        periodic_x=True,
    )
    # Soft shoulders: protect should extend beyond hard lane rows.
    assert float(plan.protect_mask[h // 2 - 6, w // 2]) > 0.02
    out = apply_band_plan_lane_protect(canvas, plan, strength=0.55, hero_luma_floor=0.95)
    assert float(out[h // 2, w // 2].mean()) < 0.55


def test_documentary_finalize_sandwich_knobs_zero() -> None:
    f = finalize_render_config(
        RenderConfig(features=FeatureConfig(look_target=LookTarget.documentary))
    ).features
    assert f.lane_star_dropout_boost == 0.0
    assert f.post_star_lane_protect_strength == 0.0
    assert f.dark_rift_strength == 0.0
    assert f.star_layer_bloom_strength == 0.0
    assert f.structure_type_competition == 0.0


def test_band_plan_exports_expected_maps() -> None:
    h, w = 24, 48
    disk = np.broadcast_to(np.exp(-((np.linspace(-1, 1, h)[:, None]) ** 2) / 0.1), (h, w))
    ext = np.clip(1.0 - 0.7 * disk, 0.05, 1.0)
    rift = np.zeros((h, w))
    rift[h // 2, w // 4] = 1.0
    plan = build_band_plan(
        height=h,
        width=w,
        disk_w=disk,
        ext_paint=ext,
        rift_mask=rift,
        periodic_x=True,
    )
    assert plan.lane_weight.shape == (h, w)
    assert plan.shell_weight.shape == (h, w)
    assert plan.protect_mask.shape == (h, w)
    assert float(plan.rift_mask.max()) > 0.0
    assert float(plan.opaque.max()) > 0.0


def test_cinematic_seeded_equirect_vista_smoke(tmp_path: Path) -> None:
    """Seeded full-path smoke: cinematic equirect must keep band structure + color."""
    cfg = RenderConfig(
        width=320,
        height=160,
        output_base_name="vista_glow_smoke",
        output_dir=tmp_path,
        generations=1,
        seed=7,
        projection_mode=ProjectionMode.equirectangular,
        output_format=OutputFormat.png,
        nebula_mode=NebulaMode.galaxy_streak,
        features=FeatureConfig(
            look_target=LookTarget.cinematic,
            jpeg_artifact_pass=False,
            photoreal_stars=True,
            galaxy_view=True,
            nebula=True,
            reference_anchors=False,
        ),
    )
    paths, _ = render_single(cfg, 0)
    img = np.asarray(Image.open(paths["equirectangular"]).convert("RGB"), dtype=np.float64) / 255.0
    metrics = measure_vista_equirect(img)
    assert metrics["height"] == 160 and metrics["width"] == 320
    # Glow-tolerant: soft under-glow can flatten mid vs shell means; plane/void + chroma carry structure.
    assert_cinematic_vista(metrics, min_band_contrast=0.02, min_chroma_spread=0.008)
