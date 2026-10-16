"""Cinematic vs documentary look-target presets and wrap-safe defaults."""

from __future__ import annotations

import numpy as np

from starsky_gen.config import (
    FeatureConfig,
    LookTarget,
    RenderConfig,
    _CINEMATIC_LOOK_UPDATE,
    _DOCUMENTARY_LOOK_UPDATE,
    apply_look_target_preset,
    finalize_render_config,
)
from starsky_gen.optical_effects import apply_film_grain_display
from starsky_gen.postfx import apply_tri_scale_bloom
from starsky_gen.projections import cubemap_faces_from_equirect


def test_default_feature_config_is_cinematic() -> None:
    feat = FeatureConfig()
    assert feat.look_target == LookTarget.cinematic
    assert feat.long_exposure_look is False
    assert feat.film_grain_strength <= 0.01
    assert feat.display_bloom_strength >= 0.09
    assert feat.bloom_mix_wide <= 0.08
    assert feat.galaxy_tone_curve == "filmic"
    assert feat.sensor_noise_stage == "off"
    assert feat.structure_inner_dust_lane_strength >= 0.82
    assert feat.galactic_band_dust_black_strength >= 0.90
    assert feat.galactic_anchor_star_count >= 30
    assert feat.m_bloom_cut >= 8.5
    assert feat.lane_star_dropout_boost >= 0.5
    assert feat.post_star_lane_protect_strength >= 0.4
    assert feat.dark_rift_strength >= 0.7
    assert feat.star_layer_bloom_strength >= 0.10
    assert feat.star_psf_bloom_scale >= 1.0
    assert feat.nebula_haze_strength <= 0.30
    assert feat.background_star_density_scale <= 0.35
    assert feat.plane_star_gate_power >= 2.3


def test_finalize_cinematic_keeps_sparse_background_density() -> None:
    """MW recognition must not floor cosmic density high (photogenic voids)."""
    cfg = RenderConfig(
        features=FeatureConfig(
            look_target=LookTarget.cinematic,
            milky_way_recognition_preset=True,
            cosmic_star_density_scale=1.0,
            background_star_density_scale=0.30,
        )
    )
    out = finalize_render_config(cfg).features
    assert out.background_star_density_scale <= 0.35
    assert out.cosmic_star_density_scale <= 1.05
    # Effective off-plane scale stays sparse (~70% cut vs ~1.0 path).
    assert out.cosmic_star_density_scale * out.background_star_density_scale <= 0.40


def test_finalize_cinematic_reinforces_mw_sandwich() -> None:
    cfg = RenderConfig(features=FeatureConfig(look_target=LookTarget.cinematic))
    out = finalize_render_config(cfg).features
    assert out.long_exposure_look is False
    assert out.structure_architecture_strength >= 1.12
    assert out.structure_band_core_drive >= 0.68
    assert out.structure_inner_dust_lane_strength >= 0.85
    assert out.galactic_band_dust_black_strength >= 0.75
    assert out.galactic_band_separation_strength >= 1.18
    assert out.band_dark_patch_strength >= 0.55
    assert out.display_bloom_strength >= 0.09
    assert out.bloom_threshold <= 0.88


def test_cinematic_preset_scalars() -> None:
    cfg = RenderConfig(
        features=FeatureConfig(
            look_target=LookTarget.cinematic,
            # Deliberately documentary-ish scalars before preset:
            long_exposure_look=False,
            film_grain_strength=0.04,
            display_bloom_strength=0.01,
            galaxy_tone_curve="asinh",
            sensor_noise_stage="both",
        )
    )
    out = apply_look_target_preset(cfg).features
    for key, expected in _CINEMATIC_LOOK_UPDATE.items():
        assert getattr(out, key) == expected, key


def test_documentary_preset_restores_photo_stack() -> None:
    cfg = RenderConfig(features=FeatureConfig(look_target=LookTarget.documentary))
    out = apply_look_target_preset(cfg).features
    for key, expected in _DOCUMENTARY_LOOK_UPDATE.items():
        assert getattr(out, key) == expected, key
    assert out.long_exposure_look is True
    assert out.film_grain_strength == 0.04
    assert out.galaxy_tone_curve == "asinh"


def test_documentary_finalize_zeros_sandwich_knobs() -> None:
    """Documentary look recipe must not leave cinematic sandwich leftovers."""
    cfg = finalize_render_config(
        RenderConfig(features=FeatureConfig(look_target=LookTarget.documentary))
    )
    f = cfg.features
    assert f.lane_star_dropout_boost == 0.0
    assert f.post_star_lane_protect_strength == 0.0
    assert f.dark_rift_strength == 0.0
    assert f.star_layer_bloom_strength == 0.0
    assert f.structure_type_competition == 0.0


def test_cinematic_preserves_explicit_long_exposure_override() -> None:
    cfg = RenderConfig(
        features=FeatureConfig(
            look_target=LookTarget.cinematic,
            long_exposure_look=True,
            long_exposure_star_trails=True,
        )
    )
    out = apply_look_target_preset(cfg).features
    assert out.long_exposure_look is True
    assert out.long_exposure_star_trails is True
    assert out.display_bloom_strength == _CINEMATIC_LOOK_UPDATE["display_bloom_strength"]


def test_finalize_applies_look_then_mw_preset() -> None:
    cfg = RenderConfig(features=FeatureConfig(look_target=LookTarget.cinematic))
    out = finalize_render_config(cfg)
    assert out.features.look_target == LookTarget.cinematic
    assert out.features.long_exposure_look is False
    assert out.wrap_safe is True


def test_bloom_periodic_wrap_continuity() -> None:
    """Higher cinematic bloom must stay seam-safe under periodic_x blur."""
    h, w = 48, 96
    rgb = np.full((h, w, 3), 0.08)
    rgb[:, 0, :] = 0.95
    rgb[:, -1, :] = 0.95
    disk_w = np.ones((h, w))
    bloom = _CINEMATIC_LOOK_UPDATE
    out = apply_tri_scale_bloom(
        rgb,
        disk_w,
        strength=float(bloom["display_bloom_strength"]),
        threshold=float(bloom["bloom_threshold"]),
        mix_tight=float(bloom["bloom_mix_tight"]),
        mix_mid=float(bloom["bloom_mix_mid"]),
        mix_wide=float(bloom["bloom_mix_wide"]),
        periodic_x=True,
    )
    seam = float(np.mean(np.abs(out[:, 0, :] - out[:, -1, :])))
    assert seam < 0.08


def test_film_grain_near_zero_cinematic_amplitude() -> None:
    rng = np.random.default_rng(3)
    rgb = np.full((32, 32, 3), 0.4)
    grain = float(FeatureConfig().film_grain_strength)
    out = apply_film_grain_display(rgb, rng, grain, periodic_x=True)
    assert float(np.mean(np.abs(out - rgb))) < 0.01


def test_wrap_safe_projection_smoke() -> None:
    assert RenderConfig().wrap_safe is True
    h, w = 64, 128
    img = np.random.default_rng(7).random((h, w, 3))
    img[:, 0, :] = img[:, -1, :]
    faces = cubemap_faces_from_equirect(img, face_size=16)
    assert set(faces.keys()) == {"px", "nx", "py", "ny", "pz", "nz"}
