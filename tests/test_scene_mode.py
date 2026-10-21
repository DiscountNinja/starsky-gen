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
    # Interactive Layer: sparse dense-core glow×emission mix in nebula scene.
    assert abs(out.features.nebula_interact_density - 0.62) < 1e-6
    assert abs(out.features.nebula_interact_strength - 0.28) < 1e-6


def test_finalize_normal_weaker_nebula_interact() -> None:
    cfg = RenderConfig(
        scene_mode=SceneMode.normal,
        features=FeatureConfig(look_target=LookTarget.cinematic),
    )
    out = finalize_render_config(cfg)
    assert abs(out.features.nebula_interact_density - 0.65) < 1e-6
    assert abs(out.features.nebula_interact_strength - 0.12) < 1e-6


def test_nebula_interact_cli_override_survives_finalize() -> None:
    cfg = RenderConfig(
        scene_mode=SceneMode.nebula,
        features=FeatureConfig(
            look_target=LookTarget.cinematic,
            nebula_interact_density=0.30,
            nebula_interact_strength=0.50,
        ),
    )
    out = finalize_render_config(cfg)
    assert abs(out.features.nebula_interact_density - 0.30) < 1e-6
    assert abs(out.features.nebula_interact_strength - 0.50) < 1e-6


def test_nebula_interact_glow_rgb_gates_and_preserves_peak() -> None:
    from starsky_gen.generator import _nebula_interact_glow_rgb

    star = np.array([1.0, 0.85, 0.70], dtype=np.float64)
    em = np.zeros((8, 8, 3), dtype=np.float64)
    # Bright magenta emission at sample pixel (dense-core gate).
    em[3, 4] = np.array([1.0, 0.22, 0.95], dtype=np.float64)
    cold = _nebula_interact_glow_rgb(star, em, 0, 0, density=0.62, strength=0.28)
    assert float(np.max(np.abs(cold - star / np.max(star)))) < 1e-5
    hot = _nebula_interact_glow_rgb(star, em, 3, 4, density=0.62, strength=0.28)
    assert float(hot[0]) > float(cold[0]) * 0.9
    assert float(hot[2]) > float(cold[2])
    assert abs(float(np.max(hot)) - 1.0) < 1e-5
    # Mid-bright emission should stay mostly spectral (sparse-core gate).
    em[5, 5] = np.array([0.40, 0.10, 0.35], dtype=np.float64)
    mid = _nebula_interact_glow_rgb(star, em, 5, 5, density=0.62, strength=0.28)
    assert float(np.max(np.abs(mid - star / np.max(star)))) < 0.08


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
    # Dramatic may concentrate color into fewer cores; allow near-parity.
    assert chroma_d >= chroma_s * 0.85
    # Not a flat salmon field: blue channel has real mass in bright cores.
    assert float(np.percentile(em_d[..., 2], 95)) > 0.04


def test_cloud_complex_dramatic_has_hero_peak() -> None:
    """Dramatic style concentrates emission into a readable hero body (not flat soup)."""
    rng = np.random.default_rng(42)
    struct, em, dust = build_cloud_complex_envelope(rng, 192, 384, style="dramatic")
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    p99 = float(np.percentile(lum, 99))
    med = float(np.median(lum))
    # Soft base raises the median; hero cores must still clearly outshine it.
    assert p99 > med * 1.8
    assert float(struct.max()) > 0.55
    # Dust carves through bright structure (lanes present).
    assert float(dust.max()) > 0.12
    assert float(dust.mean()) > 0.02


def test_cloud_complex_two_layer_fills_and_punches() -> None:
    """Dark vista envelopes most of the plate; hero cores still punch above it."""
    rng = np.random.default_rng(42)
    _, em, _ = build_cloud_complex_envelope(rng, 160, 320, style="dramatic")
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    # Immersive envelope: nearly all pixels carry dim volume (inside the gas).
    assert float(np.mean(lum > 0.025)) > 0.90
    assert float(np.percentile(lum, 10)) > 0.025
    assert float(np.min(lum)) > 0.02
    # Hierarchy: sparse bright cores, not a wall-to-wall bright fill.
    assert float(np.mean(lum > 0.35)) < 0.45
    assert float(np.percentile(lum, 99)) > float(lum.mean()) * 1.6
    assert float(np.percentile(lum, 95)) > 0.16
    # Hero well above soft vista median.
    assert float(np.percentile(lum, 99)) > float(np.percentile(lum, 50)) * 1.8


def test_cloud_complex_immersive_envelope_hue() -> None:
    """Soft surround stays in hero hue family (not neutral black voids)."""
    rng = np.random.default_rng(42)
    _, em, _ = build_cloud_complex_envelope(
        rng, 160, 320, style="dramatic", hero_color="red"
    )
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    soft = (lum > np.percentile(lum, 15)) & (lum < np.percentile(lum, 45))
    assert np.any(soft)
    assert float(em[soft, 0].mean()) > float(em[soft, 1].mean()) * 1.5
    assert float(em[soft, 0].mean()) > float(em[soft, 2].mean()) * 1.5
    # Ambient pedestal: darkest percentile not crushed to pure zero.
    assert float(np.percentile(lum, 5)) > 0.004


def test_cloud_complex_soft_fill_has_cloudy_variation() -> None:
    """Underlayer fill has soft cloudy structure — not a flat color wash."""
    from starsky_gen.cloud_complex import _gauss_float
    from starsky_gen.procedural_noise import _resize_bilinear

    rng = np.random.default_rng(42)
    _, em, _ = build_cloud_complex_envelope(
        rng, 192, 384, style="dramatic", hero_color="gold"
    )
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    # Dim surround (below hero peaks) still has spatial variation.
    soft = lum < np.percentile(lum, 55)
    assert np.any(soft)
    soft_lum = np.where(soft, lum, np.nan)
    # Coarse map of soft surround should not be flat.
    fill = np.nan_to_num(soft_lum, nan=float(np.nanmean(lum[soft])))
    coarse = _resize_bilinear(fill, 24, 48, periodic_x=True)
    assert float(coarse.max() - coarse.min()) > 0.050
    # Mid-scale residual in soft zone (cloudy, not constant).
    blur = _gauss_float(lum, 10.0, periodic_x=True)
    hp = np.abs(lum - blur)
    assert float(hp[soft].mean()) > 0.009
    # Soft zone relative contrast — lobes readable, not a narrow pedestal.
    soft_med = float(np.nanmedian(lum[soft])) + 1e-8
    assert float(np.std(lum[soft])) / soft_med > 0.18
    # Hero still clearly brighter than soft fill median.
    assert float(np.percentile(lum, 99)) > soft_med * 2.0


def test_cloud_complex_pole_coverage_no_black_caps() -> None:
    """Equirect top/bottom (sphere poles) stay enveloped — not systematically empty."""
    rng = np.random.default_rng(42)
    _, em, _ = build_cloud_complex_envelope(
        rng, 192, 384, style="dramatic", hero_color="gold"
    )
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    h = lum.shape[0]
    edge = max(4, h // 12)
    mid0, mid1 = h // 3, (2 * h) // 3
    pole = np.concatenate([lum[:edge, :], lum[-edge:, :]], axis=0)
    mid = lum[mid0:mid1, :]
    # Poles carry real fill (not pure black pockets).
    assert float(np.percentile(pole, 5)) > 0.045
    assert float(np.mean(pole < 0.03)) < 0.03
    # Poles not much emptier than mid-latitudes (no dark caps).
    assert float(np.median(pole)) > float(np.median(mid)) * 0.65
    assert float(np.mean(lum < 0.025)) < 0.02
    assert float(np.min(lum)) > 0.025
    # Cap rows (outer ~12%) track mid-latitude soft fill — enclosed sphere.
    cap = max(3, h // 16)
    caps = np.concatenate([lum[:cap, :], lum[-cap:, :]], axis=0)
    assert float(np.median(caps)) > float(np.percentile(mid, 20)) * 0.70


def test_cloud_complex_macro_silhouette_and_dust_lanes() -> None:
    """Large-scale contrast + connected dust (not hundreds of equal islands)."""
    rng = np.random.default_rng(42)
    struct, em, dust = build_cloud_complex_envelope(rng, 192, 384, style="dramatic")
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    # Large-scale: low-res downsample still has strong contrast.
    from starsky_gen.procedural_noise import _resize_bilinear

    coarse = _resize_bilinear(lum, 24, 48, periodic_x=True)
    assert float(coarse.max() - coarse.min()) > 0.18
    assert float(np.percentile(coarse, 90)) > float(np.percentile(coarse, 20)) + 0.10
    # Dust has range and dark mass outside the brightest structure.
    assert float(dust.max() - dust.min()) > 0.40
    bright = struct > np.percentile(struct, 85)
    dark = struct < np.percentile(struct, 25)
    if np.any(bright) and np.any(dark):
        assert float(dust[dark].mean()) >= float(dust[bright].mean()) * 0.85


def test_cloud_complex_barnard_shell_arcs_in_vista() -> None:
    """FOV-scale shell arcs contribute elongated structure beyond a single blob."""
    from starsky_gen.cloud_complex import _barnard_shell_field

    rng = np.random.default_rng(42)
    h, w = 128, 256
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    shell = _barnard_shell_field(rng, h, w, yy, xx, n_arcs=2, periodic_x=True)
    assert float(shell.max()) > 0.35
    # Incomplete arc: some longitudes much brighter than others (not a flat fill).
    col = shell.mean(axis=0)
    assert float(np.percentile(col, 90)) > float(np.percentile(col, 20)) + 0.02
    # Full envelope still has wide soft coverage with shell mixed in.
    rng = np.random.default_rng(42)
    _, em, _ = build_cloud_complex_envelope(rng, 160, 320, style="dramatic", hero_color="red")
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    assert float(np.mean(lum > 0.015)) > 0.85


def test_cloud_layer_interaction_gates_emission_to_dust_overlap() -> None:
    """Hero cores track dust×structure; soft base fills voids; dust has range."""
    rng = np.random.default_rng(42)
    struct, em, dust = build_cloud_complex_envelope(rng, 160, 320, style="dramatic")
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    # Soft base + sparse hero: median below bright tail; hot cores not wall-to-wall.
    assert float(np.percentile(lum, 50)) < float(np.percentile(lum, 95)) * 0.75
    assert float(np.mean(lum > 0.35)) < 0.40
    assert float(np.percentile(lum, 95)) > 0.22
    # Warm chroma in bright cores (saturation finish).
    hot = lum > np.percentile(lum, 92)
    if np.any(hot):
        assert float(em[hot, 0].mean()) >= float(em[hot, 2].mean()) * 0.85
        assert float(struct[hot].mean()) > float(struct.mean()) * 0.85
    # Dust carves cavities (higher outside bright cores) with usable range.
    cool = lum < np.percentile(lum, 30)
    if np.any(hot) and np.any(cool):
        assert float(dust[cool].mean()) >= float(dust[hot].mean()) * 0.90
    assert float(dust.max() - dust.min()) > 0.35
    assert float(np.percentile(dust, 90)) > float(np.percentile(dust, 20)) + 0.12


def test_nebula_hero_color_presets_shift_chroma() -> None:
    from starsky_gen.cloud_complex import parse_nebula_hero_color

    gold, key = parse_nebula_hero_color("gold")
    assert key == "gold"
    assert float(gold[0]) > float(gold[2])
    gold2, key2 = parse_nebula_hero_color("gold-core")
    assert key2 == "gold"
    assert np.allclose(gold, gold2)
    red, key = parse_nebula_hero_color("red")
    assert key == "red"
    # Preset red is #FF1818 (R dominant; G≈B).
    assert float(red[0]) > float(red[1]) * 8.0
    assert abs(float(red[1]) - float(red[2])) < 0.02
    ff1818, key_ff = parse_nebula_hero_color("#FF1818")
    assert key_ff == "red"  # exact preset hex maps to canonical red
    assert np.allclose(red, ff1818, atol=1e-6)
    assert np.allclose(red, parse_nebula_hero_color("FF1818")[0], atol=1e-6)
    assert parse_nebula_hero_color("0xFF1818")[1] == "red"
    deff, key_de = parse_nebula_hero_color("#DEFFB8")
    assert key_de == "custom"
    assert float(deff[1]) > float(deff[0])
    blue, key = parse_nebula_hero_color("cool-blue")
    assert key == "blue"
    assert float(blue[2]) > float(blue[0])
    hex_rgb, key = parse_nebula_hero_color("#FC5656")
    assert key == "custom"
    assert float(hex_rgb[0]) > float(hex_rgb[1])
    trip, key = parse_nebula_hero_color("255,80,40")
    assert key == "custom"
    assert float(trip[0]) > float(trip[1])

    rng = np.random.default_rng(7)
    _, em_g, _ = build_cloud_complex_envelope(
        rng, 96, 192, style="dramatic", hero_color="gold"
    )
    rng = np.random.default_rng(7)
    _, em_b, _ = build_cloud_complex_envelope(
        rng, 96, 192, style="dramatic", hero_color="blue"
    )
    rng = np.random.default_rng(7)
    _, em_r, _ = build_cloud_complex_envelope(
        rng, 96, 192, style="dramatic", hero_color="red"
    )
    lum_g = em_g.mean(axis=2)
    lum_b = em_b.mean(axis=2)
    lum_r = em_r.mean(axis=2)
    hot_g = lum_g > np.percentile(lum_g, 90)
    hot_b = lum_b > np.percentile(lum_b, 90)
    hot_r = lum_r > np.percentile(lum_r, 90)
    ratio_g = float(em_g[hot_g, 0].mean() / max(em_g[hot_g, 2].mean(), 1e-6))
    ratio_b = float(em_b[hot_b, 0].mean() / max(em_b[hot_b, 2].mean(), 1e-6))
    assert ratio_g > ratio_b
    # Blue cores must read blue (not washed beige/white).
    assert float(em_b[hot_b, 2].mean()) > float(em_b[hot_b, 0].mean()) * 1.15
    chroma_b = float(np.mean(em_b[hot_b].max(axis=1) - em_b[hot_b].min(axis=1)))
    assert chroma_b > 0.22
    # #FF1818-family cores (R well above G and B; G≈B).
    assert float(em_r[hot_r, 0].mean()) > float(em_r[hot_r, 1].mean()) * 2.0
    assert float(em_r[hot_r, 0].mean()) > float(em_r[hot_r, 2].mean()) * 2.0
    assert abs(float(em_r[hot_r, 1].mean()) - float(em_r[hot_r, 2].mean())) < float(
        em_r[hot_r, 0].mean()
    ) * 0.35


def test_red_emission_hue_tight_and_vista_matches() -> None:
    """Red preset: tight hero chroma, soft peaks, underlayer same hue family."""
    rng = np.random.default_rng(42)
    _, em, _ = build_cloud_complex_envelope(
        rng, 160, 320, style="dramatic", hero_color="red"
    )
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    hot = lum > np.percentile(lum, 94)
    soft = (lum > np.percentile(lum, 40)) & (lum < np.percentile(lum, 70))
    assert np.any(hot) and np.any(soft)
    # Hot cores stay #FF1818 red (R dominant; G≈B, not magenta).
    assert float(em[hot, 0].mean()) > float(em[hot, 1].mean()) * 2.0
    assert float(em[hot, 2].mean()) < float(em[hot, 1].mean()) * 1.35
    # Soft underlayer tracks the same hue (not brown/neutral gray).
    assert float(em[soft, 0].mean()) > float(em[soft, 1].mean()) * 1.6
    assert float(em[soft, 0].mean()) > float(em[soft, 2].mean()) * 1.6
    # Peaks compressed: very few near-white luminances.
    assert float(np.mean(lum > 0.72)) < 0.02


def test_red_preset_matches_custom_ff1818_envelope() -> None:
    """Named red and custom #FF1818 share finish path — envelopes stay close."""
    rng = np.random.default_rng(42)
    _, em_r, _ = build_cloud_complex_envelope(
        rng, 128, 256, style="dramatic", hero_color="red"
    )
    rng = np.random.default_rng(42)
    _, em_c, _ = build_cloud_complex_envelope(
        rng, 128, 256, style="dramatic", hero_color="FF1818"
    )
    assert float(np.mean(np.abs(em_r - em_c))) < 0.04
    lum_r = 0.2126 * em_r[..., 0] + 0.7152 * em_r[..., 1] + 0.0722 * em_r[..., 2]
    lum_c = 0.2126 * em_c[..., 0] + 0.7152 * em_c[..., 1] + 0.0722 * em_c[..., 2]
    assert abs(float(lum_r.mean()) - float(lum_c.mean())) < 0.02


def test_red_emission_keeps_structure_inside_hot_cores() -> None:
    """Red emission must not erase filament relief inside bright mass (vs flat slab)."""
    from starsky_gen.cloud_complex import _gauss_float
    from starsky_gen.procedural_noise import _resize_bilinear

    rng = np.random.default_rng(42)
    _, em, dust = build_cloud_complex_envelope(
        rng, 192, 384, style="dramatic", hero_color="red"
    )
    lum = 0.2126 * em[..., 0] + 0.7152 * em[..., 1] + 0.0722 * em[..., 2]
    # Large-scale contrast still present after compose.
    coarse = _resize_bilinear(lum, 24, 48, periodic_x=True)
    assert float(coarse.max() - coarse.min()) > 0.12
    assert float(np.percentile(coarse, 90)) > float(np.percentile(coarse, 20)) + 0.06
    # Soft mid-scale relief inside hot cores (not HF burned-film worm texture).
    blur = _gauss_float(lum, 6.5, periodic_x=True)
    hp = np.abs(lum - blur)
    hot = lum > np.percentile(lum, 90)
    assert np.any(hot)
    rel = hp / (lum + 1e-8)
    assert float(rel[hot].mean()) > 0.008
    assert float(np.std(lum[hot]) / (float(lum[hot].mean()) + 1e-8)) > 0.012
    # Softness: high-frequency residual stays below mid-scale (not crunchy).
    hf = np.abs(lum - _gauss_float(lum, 1.4, periodic_x=True))
    assert float(hf[hot].mean()) < float(hp[hot].mean()) * 1.85
    # Dust still carves through bright emission.
    assert float(dust.max()) > 0.35
    if np.any(hot):
        cool = lum < np.percentile(lum, 30)
        if np.any(cool):
            assert float(dust[cool].mean()) >= float(dust[hot].mean()) * 0.75


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
    lum.mean(axis=1)
    # True black void between stars (grade must not lift floor to mid-grey haze).
    assert float(np.median(lum)) < 0.08
    assert float(np.percentile(lum, 10)) < 0.02
    # Midplane continuum (not star peaks) stays near black — no soft grey ribbon.
    mid = lum[h // 2 - 4 : h // 2 + 4]
    assert float(np.percentile(mid, 25)) < 0.05


def test_no_haze_crushes_soft_midplane_wash(tmp_path) -> None:
    """--no-stars --no-haze keeps continuum structure without haze lifting the floor."""
    from PIL import Image

    def _mid_soft(path) -> tuple[float, float]:
        img = np.asarray(Image.open(path), dtype=np.float64) / 255.0
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        h = lum.shape[0]
        mid = lum[h // 2 - 4 : h // 2 + 4]
        return float(np.percentile(mid, 25)), float(mid.std())

    base = dict(
        width=256,
        height=128,
        seed=42,
        generations=1,
        output_dir=tmp_path,
        output_format=OutputFormat.png,
        projection_mode=ProjectionMode.equirectangular,
        scene_mode=SceneMode.normal,
        nebula_mode=NebulaMode.galaxy_streak,
    )
    feat = dict(
        look_target=LookTarget.cinematic,
        stars=False,
        hero_star_enabled=False,
        nebula=True,
        galaxy_view=True,
        black_background=True,
        background_gradient=False,
    )
    with_haze, _ = render_single(
        finalize_render_config(
            RenderConfig(
                **base,
                output_base_name="haze_on",
                features=FeatureConfig(**feat, nebula_haze=True),
            )
        ),
        0,
    )
    no_haze, _ = render_single(
        finalize_render_config(
            RenderConfig(
                **base,
                output_base_name="haze_off",
                features=FeatureConfig(**feat, nebula_haze=False),
            )
        ),
        0,
    )
    soft_on, _ = _mid_soft(with_haze["equirectangular"])
    soft_off, std_off = _mid_soft(no_haze["equirectangular"])
    # Continuum tube stays visible without haze; haze must still lift the floor.
    assert soft_off < soft_on
    assert soft_off < 0.32
    # Structure remains (not a second flat wash at a lower level).
    assert std_off > 0.02


def test_feature_nebula_haze_defaults_on() -> None:
    assert FeatureConfig().nebula_haze is True
    assert FeatureConfig(nebula_haze=False).nebula_haze is False


def test_no_stars_band_only_has_midplane_structure(tmp_path) -> None:
    """--no-stars keeps the MW band pipeline with midplane luminance and no sparse star peaks."""
    from PIL import Image

    cfg = finalize_render_config(
        RenderConfig(
            width=256,
            height=128,
            seed=42,
            generations=1,
            output_dir=tmp_path,
            output_base_name="band_only",
            output_format=OutputFormat.png,
            projection_mode=ProjectionMode.equirectangular,
            scene_mode=SceneMode.normal,
            nebula_mode=NebulaMode.galaxy_streak,
            features=FeatureConfig(
                look_target=LookTarget.cinematic,
                stars=False,
                hero_star_enabled=False,
                nebula=True,
                galaxy_view=True,
                black_background=True,
                background_gradient=False,
            ),
        )
    )
    assert cfg.features.stars is False
    assert use_mw_band(cfg) is True
    saved, _ = render_single(cfg, 0)
    img = np.asarray(Image.open(saved["equirectangular"]), dtype=np.float64) / 255.0
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    h, w = lum.shape
    mid = lum[h // 2 - 4 : h // 2 + 4]
    poles = np.concatenate([lum[: max(2, h // 16)], lum[-max(2, h // 16) :]])
    assert float(np.percentile(mid, 50)) > 0.02
    # Dust lanes + emission raise midplane contrast vs a flat cream slab.
    assert float(mid.std()) > 0.035
    assert float(np.percentile(poles, 90)) < float(np.percentile(mid, 50)) * 0.85
    # Dark rivers darker than bright pockets, but not binary crushed holes.
    assert float(np.percentile(mid, 15)) < float(np.percentile(mid, 85)) * 0.78
    assert float(np.percentile(mid, 15)) > 0.03
    # No bright sparse star cores off-plane (unresolved grit / catalog paint).
    # Soft continuum sheath may glow above mid-grey; star points sit much brighter.
    off = lum[h // 5 : 2 * h // 5]
    assert float(np.percentile(off, 99.5)) < 0.65


def test_no_galaxy_view_skips_mw_cream_ribbon(tmp_path) -> None:
    """--no-galaxy-view: use_mw_band false; midplane is not a MW cream ribbon."""
    from PIL import Image

    def _mid_p25(path) -> float:
        img = np.asarray(Image.open(path), dtype=np.float64) / 255.0
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        h = lum.shape[0]
        mid = lum[h // 2 - 4 : h // 2 + 4]
        return float(np.percentile(mid, 25))

    base = dict(
        width=256,
        height=128,
        seed=42,
        generations=1,
        output_dir=tmp_path,
        output_format=OutputFormat.png,
        projection_mode=ProjectionMode.equirectangular,
        scene_mode=SceneMode.normal,
        nebula_mode=NebulaMode.galaxy_streak,
    )
    feat_common = dict(
        look_target=LookTarget.cinematic,
        stars=True,
        hero_star_enabled=False,
        nebula=True,
        black_background=True,
        background_gradient=False,
        photoreal_stars=True,
    )
    off_cfg = finalize_render_config(
        RenderConfig(
            **base,
            output_base_name="no_gv",
            features=FeatureConfig(**feat_common, galaxy_view=False),
        )
    )
    band_feat = {**feat_common, "galaxy_view": True, "stars": False, "hero_star_enabled": False}
    on_cfg = finalize_render_config(
        RenderConfig(
            **base,
            output_base_name="with_gv",
            features=FeatureConfig(**band_feat),
        )
    )
    assert use_mw_band(off_cfg) is False
    assert use_mw_band(on_cfg) is True
    off_saved, _ = render_single(off_cfg, 0)
    on_saved, _ = render_single(on_cfg, 0)
    soft_off = _mid_p25(off_saved["equirectangular"])
    soft_on = _mid_p25(on_saved["equirectangular"])
    # Stars-only midplane soft floor stays well below band-only continuum.
    assert soft_off < soft_on * 0.45
    assert soft_off < 0.08


def test_no_galaxy_view_photoreal_star_cores(tmp_path) -> None:
    """Stars without MW band still get photoreal cores (not soft grey discs)."""
    from PIL import Image

    cfg = finalize_render_config(
        RenderConfig(
            width=256,
            height=128,
            seed=42,
            generations=1,
            output_dir=tmp_path,
            output_base_name="stars_solo",
            output_format=OutputFormat.png,
            projection_mode=ProjectionMode.equirectangular,
            scene_mode=SceneMode.normal,
            nebula_mode=NebulaMode.galaxy_streak,
            features=FeatureConfig(
                look_target=LookTarget.cinematic,
                stars=True,
                hero_star_enabled=False,
                nebula=True,
                galaxy_view=False,
                black_background=True,
                background_gradient=False,
                photoreal_stars=True,
            ),
        )
    )
    assert cfg.features.background_star_density_scale >= 0.85
    assert cfg.features.star_layer_bloom_strength >= 0.32
    saved, _ = render_single(cfg, 0)
    img = np.asarray(Image.open(saved["equirectangular"]), dtype=np.float64) / 255.0
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    # HDR display stretch + bloom (same path as nebula --no-nebula --black-background).
    assert float(lum.max()) > 0.85
    assert float(np.percentile(lum, 99.5)) > 0.35
    assert float(np.percentile(lum, 99.9)) > 0.55


def test_no_galaxy_view_stars_match_nebula_no_nebula_brightness(tmp_path) -> None:
    """--no-galaxy-view black plate should land near nebula --no-nebula star punch."""
    from PIL import Image

    def _p995(path) -> float:
        img = np.asarray(Image.open(path), dtype=np.float64) / 255.0
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        return float(np.percentile(lum, 99.5))

    base = dict(
        width=256,
        height=128,
        seed=42,
        generations=1,
        output_dir=tmp_path,
        output_format=OutputFormat.png,
        projection_mode=ProjectionMode.equirectangular,
    )
    no_gv, _ = render_single(
        finalize_render_config(
            RenderConfig(
                **base,
                output_base_name="nogv",
                scene_mode=SceneMode.normal,
                nebula_mode=NebulaMode.galaxy_streak,
                features=FeatureConfig(
                    look_target=LookTarget.cinematic,
                    stars=True,
                    hero_star_enabled=False,
                    nebula=True,
                    galaxy_view=False,
                    black_background=True,
                    background_gradient=False,
                    photoreal_stars=True,
                ),
            )
        ),
        0,
    )
    neb, _ = render_single(
        finalize_render_config(
            RenderConfig(
                **base,
                output_base_name="neb",
                scene_mode=SceneMode.nebula,
                nebula_mode=NebulaMode.full,
                features=FeatureConfig(
                    look_target=LookTarget.cinematic,
                    stars=True,
                    hero_star_enabled=False,
                    nebula=False,
                    black_background=True,
                    background_gradient=False,
                    photoreal_stars=True,
                ),
            )
        ),
        0,
    )
    p_gv = _p995(no_gv["equirectangular"])
    p_neb = _p995(neb["equirectangular"])
    # Same order of magnitude — not soft pinpricks vs punched cores.
    assert p_gv > 0.35
    assert p_gv > p_neb * 0.35
    assert p_gv < p_neb * 2.5


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


def test_isolate_background_stars_and_nebula_gas_differ(tmp_path) -> None:
    """Isolate plates are usable: stars vs gas differ; haze on/off gas plates differ."""
    from PIL import Image

    def _lu(path):
        img = np.asarray(Image.open(path), dtype=np.float64) / 255.0
        return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    base = dict(
        width=256,
        height=128,
        seed=42,
        generations=1,
        output_dir=tmp_path,
        output_format=OutputFormat.png,
        projection_mode=ProjectionMode.equirectangular,
        scene_mode=SceneMode.normal,
        nebula_mode=NebulaMode.galaxy_streak,
    )
    stars_saved, _ = render_single(
        finalize_render_config(
            RenderConfig(
                **base,
                output_base_name="iso_stars",
                features=FeatureConfig(
                    look_target=LookTarget.cinematic,
                    stars=True,
                    hero_star_enabled=False,
                    nebula=True,
                    galaxy_view=True,
                    black_background=True,
                    background_gradient=False,
                    isolate_layer=IsolateLayer.background_stars,
                ),
            )
        ),
        0,
    )
    gas_on, _ = render_single(
        finalize_render_config(
            RenderConfig(
                **base,
                output_base_name="iso_gas_on",
                features=FeatureConfig(
                    look_target=LookTarget.cinematic,
                    stars=False,
                    hero_star_enabled=False,
                    nebula=True,
                    galaxy_view=True,
                    nebula_haze=True,
                    black_background=True,
                    background_gradient=False,
                    isolate_layer=IsolateLayer.nebula_gas,
                ),
            )
        ),
        0,
    )
    gas_off, _ = render_single(
        finalize_render_config(
            RenderConfig(
                **base,
                output_base_name="iso_gas_off",
                features=FeatureConfig(
                    look_target=LookTarget.cinematic,
                    stars=False,
                    hero_star_enabled=False,
                    nebula=True,
                    galaxy_view=True,
                    nebula_haze=False,
                    black_background=True,
                    background_gradient=False,
                    isolate_layer=IsolateLayer.nebula_gas,
                ),
            )
        ),
        0,
    )
    lu_s = _lu(stars_saved["equirectangular"])
    lu_on = _lu(gas_on["equirectangular"])
    lu_off = _lu(gas_off["equirectangular"])
    assert float(lu_s.max()) > 0.4
    assert float(lu_on.mean()) > 1e-4
    assert float(lu_off.mean()) > 1e-4
    # Haze on/off isolate plates must not be identical (structure-gate changes gas).
    assert float(np.mean(np.abs(lu_on - lu_off))) > 0.002
    # Star plate is not the same as gas plate.
    assert float(np.mean(np.abs(lu_s - lu_on))) > 0.015
