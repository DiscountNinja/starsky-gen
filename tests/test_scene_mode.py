"""scene_mode normal|nebula finalize wiring and full-sky heroes."""

from __future__ import annotations

import numpy as np

from starsky_gen.cloud_complex import build_cloud_complex_envelope
from starsky_gen.config import (
    FeatureConfig,
    IsolateLayer,
    LookTarget,
    NebulaMode,
    OutputFormat,
    ProjectionMode,
    RenderConfig,
    SceneMode,
    finalize_render_config,
    use_mw_band,
)
from starsky_gen.generator import render_single
from starsky_gen.hero_stars import select_hero_indices
from starsky_gen.placement import (
    EQUIRECT_LAT_POLE_MARGIN_FULL_SKY,
    EQUIRECT_POLE_TAPER_FRAC,
    EQUIRECT_POLE_TAPER_FRAC_FULL_SKY,
    apply_equirect_steradian_weights,
    build_equirect_density_map,
    pixels_to_lon_lat,
    soft_squash_lat,
)


def test_finalize_nebula_scene_skips_mw_band_and_keeps_grade() -> None:
    cfg = RenderConfig(
        scene_mode=SceneMode.nebula,
        nebula_mode=NebulaMode.galaxy_streak,
        features=FeatureConfig(
            look_target=LookTarget.cinematic,
            milky_way_recognition_preset=True,
            galaxy_view=True,
        ),
    )
    out = finalize_render_config(cfg)
    assert out.scene_mode == SceneMode.nebula
    assert out.nebula_mode == NebulaMode.full
    assert use_mw_band(out) is False
    assert out.features.galaxy_view is True
    assert out.features.nebula is True
    assert out.features.milky_way_recognition_preset is False
    assert out.features.band_ism_dominance == 0.0
    assert 0.62 <= out.features.star_shine_through_strength <= 0.78
    assert 5 <= out.features.hero_star_count <= 8
    assert out.features.spike_max_count >= 6
    assert out.features.star_layer_bloom_strength >= 0.32


def test_finalize_nebula_forces_galaxy_view_even_if_cli_disabled() -> None:
    cfg = RenderConfig(
        scene_mode=SceneMode.nebula,
        features=FeatureConfig(look_target=LookTarget.cinematic, galaxy_view=False),
    )
    out = finalize_render_config(cfg)
    assert out.features.galaxy_view is True
    assert use_mw_band(out) is False


def test_finalize_normal_keeps_mw_recognition() -> None:
    cfg = RenderConfig(
        scene_mode=SceneMode.normal,
        features=FeatureConfig(look_target=LookTarget.cinematic),
    )
    out = finalize_render_config(cfg)
    assert use_mw_band(out) is True
    assert out.features.milky_way_recognition_preset is True
    assert out.nebula_mode == NebulaMode.galaxy_streak


def test_full_sky_heroes_without_disk_weight() -> None:
    w, h = 256, 128
    rng = np.random.default_rng(9)
    n = 100
    cat = {
        "lon": rng.uniform(0.0, 2.0 * np.pi, size=n),
        "lat": rng.uniform(-1.0, 1.0, size=n),
        "phot_mag": rng.uniform(4.0, 12.0, size=n),
    }
    struct = rng.random((h, w))
    idx = select_hero_indices(
        cat,
        width=w,
        height=h,
        disk_weight=None,
        count=6,
        placement="full_sky",
        structure_weight=struct,
        rng=rng,
    )
    assert idx.shape[0] == 6
    # Full-sky should not force near-equator clustering.
    lat_abs = np.abs(cat["lat"][idx])
    assert float(np.median(lat_abs)) > 0.08


def test_nebula_isotropic_density_has_no_midplane_peak() -> None:
    """Nebula mode density must not recreate a galactic-band midplane."""
    from starsky_gen.placement import build_isotropic_equirect_density_map, build_master_density_field

    h, w = 256, 512
    rng = np.random.default_rng(11)
    iso = build_isotropic_equirect_density_map(w, h, rng)
    band = build_equirect_density_map(
        w,
        h,
        band_lat_sigma=0.10,
        disk_height=0.19,
        halo_fraction=0.22,
        lon_bulge_amp=0.35,
    )
    # Mid vs high-latitude mean: isotropic stays near-flat; band map peaks at equator.
    iso_mid = float(iso[h // 2].mean())
    iso_hi = float(iso[h // 8].mean())
    band_mid = float(band[h // 2].mean())
    band_hi = float(band[h // 8].mean())
    assert iso_mid / max(iso_hi, 1e-8) < 1.35
    assert band_mid / max(band_hi, 1e-8) > 2.5
    master = build_master_density_field(w, h, rng, isotropic_field=True)
    m_mid = float(master[h // 2].mean())
    m_hi = float(master[h // 8].mean())
    assert m_mid / max(m_hi, 1e-8) < 1.40
    soft = apply_equirect_steradian_weights(
        iso, pole_taper_frac=EQUIRECT_POLE_TAPER_FRAC_FULL_SKY, pole_lift=0.22
    )
    assert float(soft[: max(2, h // 40)].mean()) > 0.05


def test_cloud_complex_dramatic_richer_than_subtle() -> None:
    rng = np.random.default_rng(42)
    _, em_s, du_s = build_cloud_complex_envelope(rng, 128, 256, style="subtle")
    rng = np.random.default_rng(42)
    _, em_d, du_d = build_cloud_complex_envelope(rng, 128, 256, style="dramatic")
    # Dramatic should push more dust occlusion and more chroma separation.
    assert float(du_d.mean()) >= float(du_s.mean()) * 0.95
    chroma_s = float(np.std(em_s, axis=2).mean())
    chroma_d = float(np.std(em_d, axis=2).mean())
    assert chroma_d >= chroma_s * 0.95
    # Not a flat salmon field: blue channel has real mass.
    assert float(em_d[..., 2].mean()) > 0.02


def test_soft_squash_lat_spreads_near_pole() -> None:
    """Hard clip piles mass on one latitude; soft squash keeps a spread."""
    hi = np.pi / 2 - EQUIRECT_LAT_POLE_MARGIN_FULL_SKY
    near = np.linspace(hi - 0.08, np.pi / 2, 200)
    soft = soft_squash_lat(near, pole_margin=EQUIRECT_LAT_POLE_MARGIN_FULL_SKY)
    assert float(np.std(soft)) > 0.01
    assert float(soft.max()) < hi + 1e-6
    assert float(soft.min()) > -hi - 1e-6


def test_pixels_to_lon_lat_soft_poles_avoid_edge_shelf() -> None:
    h, w = 64, 128
    rng = np.random.default_rng(7)
    # All samples on bottom equirect row — would form a shelf under hard clip.
    rows = np.full(80, h - 1, dtype=np.int32)
    cols = rng.integers(0, w, size=80)
    _, lat = pixels_to_lon_lat(
        rows, cols, w, h, rng, pole_margin=EQUIRECT_LAT_POLE_MARGIN_FULL_SKY, soft_poles=True
    )
    # After nudge + soft squash, latitudes are not a single collapsed value.
    assert float(np.std(lat)) > 0.005
    yn = 0.5 + lat / np.pi
    assert float(yn.min()) > 1.0 / float(h)  # off absolute bottom pixel center


def test_isotropic_full_sky_avoids_equirect_y_border_shelf() -> None:
    """Nebula isotropic catalogs fill near poles without an empty-cap + shelf ring."""
    from starsky_gen.projections import sph_to_equirect_xy
    from starsky_gen.starfield import sample_isotropic_cosmic_catalog, sample_star_catalog

    w, h = 512, 256
    rng = np.random.default_rng(42)
    cos = sample_isotropic_cosmic_catalog(
        rng, w, h, 1.0, pole_margin=EQUIRECT_LAT_POLE_MARGIN_FULL_SKY
    )
    _, ys = sph_to_equirect_xy(cos["lon"], cos["lat"], w, h)
    assert int(ys.min()) >= 1
    assert int(ys.max()) <= h - 2
    counts = np.bincount(ys, minlength=h).astype(np.float64)
    assert float(counts.max()) / max(float(cos["lon"].size), 1.0) < 0.04
    # Poles populated: outermost octiles should not be empty.
    assert float(counts[: h // 8].sum()) > 0.02 * float(cos["lon"].size)
    assert float(counts[-h // 8 :].sum()) > 0.02 * float(cos["lon"].size)

    bg = sample_star_catalog(
        np.random.default_rng(7),
        w,
        h,
        0.85,
        layer="background",
        latitude_color_bias=False,
        isotropic_field=True,
        pole_margin=EQUIRECT_LAT_POLE_MARGIN_FULL_SKY,
        pole_taper_frac=0.0,
        cluster_strength=0.15,
        placement_asymmetry=0.35,
    )
    _, ys_bg = sph_to_equirect_xy(bg["lon"], bg["lat"], w, h)
    assert int(ys_bg.min()) >= 1
    assert int(ys_bg.max()) <= h - 2
    c_bg = np.bincount(ys_bg, minlength=h).astype(np.float64)
    # Asymmetry must not re-clip to the MW ±0.12 margin (~20px empty caps).
    assert int(c_bg[:6].sum()) > 0
    assert int(c_bg[-6:].sum()) > 0
    assert float(c_bg.max()) / max(float(ys_bg.size), 1.0) < 0.05
    # No empty-cap-then-shelf: first 8% of rows should hold a fair share of stars.
    assert float(c_bg[: max(4, h // 12)].sum()) > 0.015 * float(ys_bg.size)


def test_cloud_complex_filament_contrast() -> None:
    """Filamentary structure has higher spatial contrast than a soft body-only field."""
    rng = np.random.default_rng(11)
    struct, _, _ = build_cloud_complex_envelope(rng, 96, 192, style="dramatic")
    # Local contrast: std of high-pass residual should be material.
    blur = np.mean(struct)
    contrast = float(np.std(struct - blur))
    assert contrast > 0.08
    assert float(struct.mean()) > 0.05
    # Not a single oval blob: multiple local peaks across the frame.
    row_peaks = int(np.sum(struct.max(axis=1) > 0.35))
    assert row_peaks > 8


def test_nebula_scene_honors_no_stars_and_zero_density() -> None:
    cfg = RenderConfig(
        scene_mode=SceneMode.nebula,
        features=FeatureConfig(
            look_target=LookTarget.cinematic,
            stars=False,
            hero_star_enabled=True,
            nebula=True,
            background_star_density_scale=0.0,
        ),
    )
    out = finalize_render_config(cfg).features
    assert out.stars is False
    assert out.hero_star_enabled is True
    assert out.nebula is True
    assert out.background_star_density_scale == 0.0


def test_nebula_scene_honors_no_nebula() -> None:
    cfg = RenderConfig(
        scene_mode=SceneMode.nebula,
        features=FeatureConfig(look_target=LookTarget.cinematic, nebula=False, stars=True),
    )
    out = finalize_render_config(cfg).features
    assert out.nebula is False


def test_nebula_stars_only_no_midplane_haze(tmp_path) -> None:
    """--mode nebula --no-nebula --black-background must not paint a MW haze band."""
    cfg = RenderConfig(
        width=256,
        height=128,
        seed=42,
        generations=1,
        output_dir=tmp_path,
        output_base_name="stars",
        output_format=OutputFormat.png,
        projection_mode=ProjectionMode.equirectangular,
        scene_mode=SceneMode.nebula,
        nebula_mode=NebulaMode.full,
        features=FeatureConfig(
            look_target=LookTarget.cinematic,
            stars=True,
            hero_star_enabled=True,
            nebula=False,
            black_background=True,
            background_gradient=False,
        ),
    )
    saved, _ = render_single(finalize_render_config(cfg), 0)
    from PIL import Image

    img = np.asarray(Image.open(saved["equirectangular"]), dtype=np.float64) / 255.0
    lum = img.mean(axis=2)
    h = lum.shape[0]
    row = lum.mean(axis=1)
    # True black void between stars (grade must not lift floor to mid-grey haze).
    assert float(np.median(lum)) < 0.08
    assert float(np.percentile(lum, 10)) < 0.02
    # Midplane continuum (not star peaks) stays near black — no soft grey ribbon.
    mid = lum[h // 2 - 4 : h // 2 + 4]
    assert float(np.percentile(mid, 25)) < 0.05


def test_isolate_layer_heroes_smoke(tmp_path) -> None:
    cfg = RenderConfig(
        width=256,
        height=128,
        seed=3,
        generations=1,
        output_dir=tmp_path,
        output_base_name="iso",
        output_format=OutputFormat.png,
        projection_mode=ProjectionMode.equirectangular,
        scene_mode=SceneMode.nebula,
        nebula_mode=NebulaMode.full,
        features=FeatureConfig(
            look_target=LookTarget.cinematic,
            stars=False,
            hero_star_enabled=True,
            hero_star_count=4,
            nebula=False,
            black_background=True,
            background_gradient=False,
            isolate_layer=IsolateLayer.heroes,
        ),
    )
    saved, _stats = render_single(cfg, generation_index=0)
    assert "equirectangular" in saved
    assert saved["equirectangular"].is_file()
