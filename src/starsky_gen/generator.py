from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Literal

# #region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[2] / ".cursor" / "debug-408793.log"


def _dbg_band_stats(
    canvas: np.ndarray,
    disk_w: np.ndarray | None,
    *,
    center_row: int | None = None,
) -> dict[str, float]:
    lu = rec709_luma(np.maximum(np.asarray(canvas, dtype=np.float64), 0.0))
    h = int(lu.shape[0])
    cr = int(h // 2) if center_row is None else int(center_row)
    cr = int(np.clip(cr, 0, max(0, h - 1)))
    row = lu[max(0, cr - 2) : min(h, cr + 3), :]
    stats: dict[str, float] = {
        "center_row_min": float(np.min(row)),
        "center_row_mean": float(np.mean(row)),
        "center_row_max": float(np.max(row)),
        "center_row_frac_lt_0.35": float(np.mean(row < 0.35)),
    }
    if disk_w is not None:
        dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
        if dw.ndim == 1:
            dw = dw[:, None]
        if dw.shape[0] == h:
            band = dw[:, 0] > 0.35
            if bool(np.any(band)):
                b = lu[band]
                stats.update(
                    {
                        "band_p50": float(np.percentile(b, 50)),
                        "band_p95": float(np.percentile(b, 95)),
                        "band_p99": float(np.percentile(b, 99)),
                        "band_frac_gt_0.95": float(np.mean(b > 0.95)),
                    }
                )
    return stats


def _dbg_hist(
    label: str,
    field: np.ndarray,
    disk_w: np.ndarray | None,
    *,
    hypothesis_id: str = "BG",
    run_id: str | None = None,
) -> None:
    """Min/p50/p95/max for full frame, sky (disk<0.15), and band (disk>0.35)."""
    import os

    arr = np.clip(np.asarray(field, dtype=np.float64), 0.0, None)
    if arr.ndim == 3:
        arr = rec709_luma(arr)
    flat = arr.ravel()
    stats: dict[str, float] = {
        "min": float(np.min(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
    }
    if disk_w is not None:
        h = int(arr.shape[0])
        dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
        if dw.ndim == 1:
            dw = dw[:, None]
        if dw.shape[0] == h:
            sky = dw[:, 0] < 0.15
            band = dw[:, 0] > 0.35
            if bool(np.any(sky)):
                s = arr[sky]
                stats.update(
                    {
                        "sky_min": float(np.min(s)),
                        "sky_p50": float(np.percentile(s, 50)),
                        "sky_p95": float(np.percentile(s, 95)),
                        "sky_mean": float(np.mean(s)),
                    }
                )
            if bool(np.any(band)):
                b = arr[band]
                stats.update(
                    {
                        "band_min": float(np.min(b)),
                        "band_p50": float(np.percentile(b, 50)),
                        "band_p95": float(np.percentile(b, 95)),
                    }
                )
    _dbg_log(hypothesis_id, "generator.py:_dbg_hist", label, stats, run_id=run_id)


def _dbg_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    *,
    run_id: str | None = None,
) -> None:
    import os

    if run_id is None:
        run_id = os.environ.get("STK_DEBUG_RUN", "pre")
    payload = {
        "sessionId": "408793",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
    }
    try:
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass


# #endregion

import numpy as np
from PIL import Image

from starsky_gen.bulge import render_bulge_layer
from starsky_gen.dust_field import (
    apply_missing_region_extinction,
    build_band_disruption_field,
    build_filament_erosion_map,
    build_fractal_extinction_field,
    build_morphology_extinction_transmission,
    carve_extinction_discontinuities,
    attenuate_rgb_column_comb,
    band_relative_clearance,
    emission_clearance_from_extinction,
    gas_clearance_from_extinction,
    reinforce_absorption_edges,
    transmission_from_absorption_map,
)
from starsky_gen.catalog_data import (
    fit_procedural_stats_from_catalog,
    load_catalog_subset,
    load_luminance_overlay,
    merge_catalog_positions,
)
from starsky_gen.color_science import (
    ccm89_v_band_transmission,
    extinction_from_transmission,
    extinction_redden,
    rec709_luma,
    remap_luma_preserving_chroma,
    star_chromatic_perturb_weight,
    star_rgb_from_teffective,
    warm_teffective_for_core_bulge,
)
from starsky_gen.hdr import HDR_DTYPE, as_hdr, ensure_hdr
from starsky_gen.config import FeatureConfig, NebulaMode, ProjectionMode, RenderConfig
from starsky_gen.dither import apply_blue_noise_dither_u8
from starsky_gen.optical_effects import (
    apply_film_grain_display,
    apply_isp_linear_chain,
    inject_sensor_noise,
)
from starsky_gen.postfx import (
    apply_color_stratification,
    apply_depth_of_field,
    apply_directional_motion_blur,
    apply_lightwrap_stars,
    apply_masked_density_sharpen,
    depth_map_from_disk,
)
from starsky_gen.postfx import apply_band_display_highlight_cap
from starsky_gen.tone_grade import apply_galaxy_display_finish, apply_galaxy_linear_grade_pipeline
from starsky_gen.nebula import (
    _blur_rgb_separable_xy,
    _blur_separable_xy,
    _blur_x_only,
    _blur_y_only,
    _resize_bilinear,
    generate_nebula,
)
from starsky_gen.galactic_band_color import (
    apply_band_display_microstructure,
    apply_band_luma_separation,
    apply_galactic_band_color_grade,
)
from starsky_gen.disk_radiance import (
    apply_shared_photon_exposure,
    disk_chroma_from_star_layer,
    estimate_disk_photon_exposure,
    harmonize_diffuse_canvas_chroma,
    soften_diffuse_chroma_toward,
    ism_lift_rgb,
    matched_star_display_stretch_gain,
    unresolved_speckle_rgb,
)
from starsky_gen.composite_blend import (
    composite_add_gas,
    composite_emission_add_screen,
    composite_emission_chroma_preserve,
    composite_stars_over_display_canvas,
    normalize_star_stack_luma_preserve_chroma,
    stars_hdr_to_display,
)
from starsky_gen.nebula_physics import (
    apply_ccm_extinction_linear,
    dust_lane_multiscatter_fill,
    extinction_av_scale_for_lane_depth,
    forward_scatter_mie,
)
from starsky_gen.postprocess import apply_jpeg_artifacts, smooth_jpeg_highlight_artifacts
from starsky_gen.procedural_noise import gaussian_blur_pil
from starsky_gen.projections import cubemap_faces_from_equirect, sph_to_equirect_xy, sph_to_equirect_xy_float
from starsky_gen.psf import (
    PsfTuning,
    StarPsfVariety,
    flux_from_mag,
    sample_star_psf_variety,
    stamp_star_psf,
)
from starsky_gen.reference_stars import paint_reference_anchors
from starsky_gen.config import RenderProfile
from starsky_gen.galactic_structure import (
    GalacticMorphology,
    GalacticStructure,
    apply_unresolved_field_to_canvas,
    attach_stellar_population_to_catalog,
    build_galactic_morphology,
    compose_inherited_unresolved_field,
    deposit_catalog_unresolved_flux,
    build_galactic_structure,
    morphology_grayscale_preview,
    resolved_keep_probability,
)
from starsky_gen.starfield import (
    STAR_COLOR_NAMES,
    STAR_SIZE_NAMES,
    _paint_thin_diffraction_spikes,
    catalog_stats,
    cull_faint_resolved_stars,
    paint_star,
    rgb_from_bv,
    reroll_stars_in_dark_lanes,
    sample_cluster_star_catalog,
    sample_halo_star_catalog,
    sample_isotropic_cosmic_catalog,
    inject_galactic_overdensity_stars,
    sample_star_catalog,
    size_radius,
    star_color,
)


def _sample_ext_scalar_bilinear(
    sheet: np.ndarray,
    xf: float,
    yf: float,
    *,
    periodic_x: bool,
) -> float:
    """Bilinear sample of grayscale map on equirect; X wraps east–west."""
    map2 = np.asarray(sheet, dtype=np.float64)
    h0, w0 = map2.shape
    y0_i = int(np.clip(np.floor(yf), 0, max(0, h0 - 2)))
    ty = float(np.clip(yf - y0_i, 0.0, 1.0))
    xe = xf % float(w0) if periodic_x else float(np.clip(xf, 0.0, float(max(w0 - 1, 1)) - 1e-12))
    x0_i = int(np.clip(np.floor(xe), 0, w0 - 1))
    x1_i = int((x0_i + 1) % w0) if periodic_x else int(min(x0_i + 1, w0 - 1))
    tx = float(xe - x0_i)

    z00 = map2[y0_i, x0_i]
    z01 = map2[y0_i, x1_i]
    y1_i = min(y0_i + 1, h0 - 1)
    z10 = map2[y1_i, x0_i]
    z11 = map2[y1_i, x1_i]
    zb0 = z00 + tx * (z01 - z00)
    zb1 = z10 + tx * (z11 - z10)
    return float(np.clip(zb0 + ty * (zb1 - zb0), 0.0, 1.0))


def _psf_tuning_from_features(feat: FeatureConfig) -> PsfTuning:
    return PsfTuning(
        sigma_core_min=feat.psf_sigma_core_min,
        sigma_core_max=feat.psf_sigma_core_max,
        beta_default=feat.psf_beta_default,
        fwhm_base_px=feat.psf_fwhm_base_px,
        fwhm_ref_mag=feat.psf_fwhm_ref_mag,
        fwhm_mag_coeff=feat.psf_fwhm_mag_coeff,
        fwhm_bright_jitter_sigma=feat.psf_fwhm_bright_jitter,
        fwhm_faint_jitter_half=feat.psf_fwhm_faint_jitter,
        halo_sigma_bright=feat.psf_halo_sigma_bright,
        halo_amp=feat.psf_halo_amp,
        m_bloom_cut=feat.m_bloom_cut,
        mag_radius_gamma=feat.mag_radius_gamma,
        max_half_px=feat.psf_max_half_px,
        mag_bright=feat.mag_bright_lim,
        mag_faint=feat.mag_faint_lim,
        sensor_aperture_mm=feat.sensor_aperture_mm,
        sensor_focal_mm=feat.sensor_focal_mm,
        bloom_amp=0.022 * float(feat.star_psf_bloom_scale),
        core_sigma_scale=0.36,
        core_flux_frac_bright=0.58,
        shelf_sigma_scale=0.68,
        wing_flux_frac=0.008,
        wide_halo_sigma_px=6.5,
        bloom_sigma=2.2,
        ultra_bloom_flux_mult=1.12,
        tier_foreground_sigma_scale=1.12,
    )


def _placement_kwargs(feat: FeatureConfig) -> dict[str, float | bool]:
    return {
        "use_poisson_placement": feat.galaxy_view,
        "disk_height": feat.disk_height,
        "halo_fraction": feat.halo_fraction,
        "halo_power": feat.halo_power,
        "band_lat_sigma": feat.band_lat_sigma,
        "band_rotation_deg": feat.band_rotation_deg,
        "band_curvature_amp": feat.band_curvature_amp,
        "poisson_min_sep_bright_px": feat.poisson_min_sep_bright_px,
        "poisson_min_sep_faint_px": feat.poisson_min_sep_faint_px,
        "use_spectral_teffective": feat.use_spectral_teffective,
        "placement_asymmetry": feat.placement_asymmetry,
        "cluster_strength": feat.cluster_placement_strength,
        "use_imf_magnitudes": feat.use_imf_magnitudes,
        "imf_giant_fraction": feat.imf_giant_fraction,
        "hierarchical_star_placement": feat.hierarchical_star_placement,
        "population_gradient_strength": feat.stellar_age_gradient_strength,
    }


def _midplane_unsharp(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    amp: float,
    periodic_x: bool,
) -> np.ndarray:
    if amp <= 1e-6:
        return rgb
    luma = np.mean(rgb, axis=2)
    blur = _blur_separable_xy(luma, passes=2, periodic_x=periodic_x)
    detail = luma - blur
    w = np.clip(disk_w, 0.0, 1.0)
    out = rgb + detail[..., np.newaxis] * w[..., np.newaxis] * amp
    return np.clip(out, 0.0, None)


def _bulge_warmth_scalar(lon: float, lat: float, width: int) -> float:
    """Peak near galactic center (equirect x≈0.5, lat≈0)."""
    x_n = float((lon / (2.0 * np.pi)) % 1.0)
    dx = min(abs(x_n - 0.5), 1.0 - abs(x_n - 0.5)) * 2.0
    lat_w = float(np.exp(-((lat / 0.24) ** 2)))
    lon_w = float(np.exp(-((dx / 0.20) ** 2)))
    return lat_w * lon_w


def _deepen_extinction_lanes(
    ext: np.ndarray,
    disk_w: np.ndarray,
    *,
    periodic_x: bool,
) -> np.ndarray:
    """Anisotropic dark-lane mask: tight along longitude, elongated across latitude."""
    dark = np.clip(1.0 - ext, 0.0, 1.0)
    dark = _blur_x_only(dark, passes=1, periodic_x=periodic_x)
    dark = _blur_y_only(dark, passes=3)
    dark = np.clip(dark**1.18, 0.0, 1.0)
    lo = float(np.percentile(dark, 58.0))
    hi = float(np.percentile(dark, 92.0))
    if hi <= lo + 1e-8:
        lane = dark
    else:
        lane = np.clip((dark - lo) / (hi - lo), 0.0, 1.0)
    erode = np.clip(dark - _blur_separable_xy(dark, passes=1, periodic_x=periodic_x), 0.0, 1.0)
    dilate = np.clip(_blur_separable_xy(dark, passes=2, periodic_x=periodic_x) - dark, 0.0, 1.0)
    lane = np.clip(lane * (0.82 + 0.18 * (dilate - erode * 0.72)), 0.0, 1.0)
    lane = lane**1.62 * disk_w
    out = ext * (1.0 - lane * 0.78)
    return np.clip(out, 0.018, 1.0)


def _deepen_extinction_lanes_strong(
    ext: np.ndarray,
    disk_w: np.ndarray,
    *,
    periodic_x: bool,
) -> np.ndarray:
    """Extra lane carve when extinction_first_nebula is enabled."""
    out = _deepen_extinction_lanes(ext, disk_w, periodic_x=periodic_x)
    dark = np.clip(1.0 - out, 0.0, 1.0)
    dark = np.clip(dark**1.35, 0.0, 1.0) * disk_w
    return np.clip(out * (1.0 - dark * 0.52), 0.012, 1.0)


def _dust_lane_local_contrast(
    rgb: np.ndarray,
    ext: np.ndarray,
    disk_w: np.ndarray,
    *,
    amp: float,
    periodic_x: bool,
) -> np.ndarray:
    if amp < 1e-6:
        return rgb
    lum = np.clip(
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2],
        0.0,
        None,
    )
    thick = np.clip(1.0 - ext, 0.0, 1.0)
    m = disk_w * (thick > 0.22) * (lum < 0.42)
    blur_l = _blur_separable_xy(lum, passes=4, periodic_x=periodic_x)
    blur_s = _blur_separable_xy(lum, passes=1, periodic_x=periodic_x)
    hi = lum - blur_s
    lo = blur_l - lum
    detail = (hi * 0.65 - lo * 0.35) * amp * m
    boosted = lum + detail
    scale = boosted / np.maximum(lum, 3e-6)
    return np.maximum(rgb * scale[..., None], 0.0)


def _dust_lane_forward_scatter(
    rgb: np.ndarray,
    ext: np.ndarray,
    neb_luma: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float,
    periodic_x: bool,
    plane_gate: np.ndarray | None = None,
) -> np.ndarray:
    """Soft forward-scatter rims on dust lane borders (1–3% of local nebula brightness)."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    pg = np.clip(np.asarray(plane_gate, dtype=np.float64), 0.0, 1.0) if plane_gate is not None else 1.0
    thick = np.clip(1.0 - ext, 0.0, 1.0) * pg
    edge = np.clip(_blur_separable_xy(thick, passes=2, periodic_x=periodic_x) - thick, 0.0, 1.0)
    edge = _blur_separable_xy(edge, passes=1, periodic_x=periodic_x)
    rim = edge * np.clip(neb_luma, 0.0, 1.0) * np.clip(disk_w, 0.0, 1.0) * pg
    rim = _blur_separable_xy(rim, passes=2, periodic_x=periodic_x)
    tint = np.array([1.02, 0.99, 0.96], dtype=np.float64)
    return np.clip(rgb + rim[..., np.newaxis] * tint * s, 0.0, None)


def _emission_mask_local_contrast(
    rgb: np.ndarray,
    emit_luma: np.ndarray,
    emit_peaks: np.ndarray,
    *,
    amp: float,
    periodic_x: bool,
) -> np.ndarray:
    """Masked high-pass (unsharp) inside bright emission only (clump detail, not global punch)."""
    a = float(amp)
    if a < 1e-6:
        return rgb
    mask = np.clip((emit_peaks - 0.05) / 0.32, 0.0, 1.0) ** 1.05
    if float(np.max(mask)) < 1e-8:
        return rgb
    blur_s = _blur_separable_xy(emit_luma, passes=1, periodic_x=periodic_x)
    blur_m = _blur_separable_xy(emit_luma, passes=4, periodic_x=periodic_x)
    detail = (emit_luma - blur_s) * 0.78 + (blur_m - emit_luma) * 0.14
    lu = np.clip(
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2],
        0.0,
        None,
    )
    boosted = lu + detail * a * mask
    scale = boosted / np.maximum(lu, 3e-6)
    return np.maximum(rgb * scale[..., np.newaxis], 0.0)


def _disk_filmic_luma_grade(
    rgb: np.ndarray, disk_w: np.ndarray, *, shoulder: float
) -> np.ndarray:
    w = disk_w[..., None]
    lin = np.maximum(rgb, 0.0)
    lu = (
        0.2126 * lin[..., 0]
        + 0.7152 * lin[..., 1]
        + 0.0722 * lin[..., 2]
    )
    k = float(max(shoulder, 0.1))
    l_new = lu * (1.0 + k * lu) / (1.0 + lu)
    scale = np.divide(l_new, np.maximum(lu, 7e-8), out=np.ones_like(lu), where=lu > 8e-7)
    toned = lin * scale[..., None]
    return rgb * (1.0 - w) + toned * w


def _faint_disk_unsharp(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    sigma_px: float,
    amp: float,
    knee: float,
    periodic_x: bool,
) -> np.ndarray:
    if sigma_px < 1e-5 or amp < 1e-7:
        return rgb
    lum = np.clip(
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2],
        0.0,
        None,
    )
    passes = int(np.clip(round(float(sigma_px)), 2, 8))
    base = lum * disk_w + np.mean(lum) * (1.0 - disk_w)
    blurred = _blur_separable_xy(base.astype(np.float64), passes=passes, periodic_x=periodic_x)
    hi = lum - blurred
    m = disk_w * (lum < knee)
    boosted = lum + hi * amp * m
    s = boosted / np.maximum(lum, 3e-6)
    return rgb * s[..., None]


def _canvas_add_linear(canvas: np.ndarray, delta: np.ndarray, *, galaxy_hdr: bool) -> np.ndarray:
    """Linear add for Milky Way cinematic stack; display path clips after tone mapping."""
    if galaxy_hdr:
        return ensure_hdr(np.maximum(0.0, canvas + delta))
    return ensure_hdr(np.clip(canvas + delta, 0.0, 1.0))


def _hdr_stars_to_display(
    stars: np.ndarray,
    disk_w: np.ndarray,
    *,
    stretch_gain: float = 11.0,
    peak_percentile: float = 99.75,
    faint_desat: float = 0.0,
    display_cap: float = 1.0,
    output_gain: float = 1.0,
) -> np.ndarray:
    """Separate star tone map: luma-only percentile asinh, chroma reprojected (no RGB normalize)."""
    _ = disk_w
    return stars_hdr_to_display(
        stars,
        stretch_gain=stretch_gain,
        peak_percentile=peak_percentile,
        faint_desat=faint_desat,
        emit_cap=display_cap,
        output_gain=output_gain,
    )


def _periodic_lon_grid_xx(xx: np.ndarray) -> np.ndarray:
    """Longitude coordinate on [-1,1] with x=-1 and x=+1 identified (C^1 at the equirect seam)."""
    return (1.0 / np.pi) * np.arctan2(np.sin(np.pi * xx), np.cos(np.pi * xx))


def _wrap_lon_delta_xx_minus_a(xx: np.ndarray, a: float) -> np.ndarray:
    """Signed longitude difference vs scalar anchor `a` on [-1,1] with wrap."""
    return _periodic_lon_grid_xx(xx - a)


def _background_plane(
    rng: np.random.Generator,
    height: int,
    width: int,
    enabled: bool,
    black_background: bool,
    texture_strength: float = 1.0,
) -> np.ndarray:
    if black_background:
        return np.zeros((height, width, 3), dtype=HDR_DTYPE)

    y = np.linspace(-1.0, 1.0, height)[:, None]
    disk = np.exp(-(y**2) / 0.22)
    falloff = (1.0 - disk) ** 0.9

    if enabled:
        base = 0.038 + disk * 0.042 - falloff * 0.022
    else:
        base = 0.032 + disk * 0.022 - falloff * 0.014

    # Full-resolution filtered noise floor only (avoid coarse upsampled-lattice artifacts).
    noise_hf = rng.normal(0.0, 1.0, size=(height, width))
    noise_mf = _blur_separable_xy(noise_hf, passes=1, periodic_x=True)
    noise_lf = _blur_separable_xy(noise_mf, passes=2, periodic_x=True)
    t = float(np.clip(texture_strength, 0.0, 2.0))
    pole = np.exp(-(y**2) / 0.32)
    noise = (noise_hf * 0.0014 * t + noise_mf * 0.0018 * t + noise_lf * 0.0018 * t) * pole
    blue_noise = (noise_mf * 0.0022 + noise_lf * 0.0020) * t * pole

    # Sparse full-resolution background points (faint star bed), with tiny halo.
    sky_gate = np.clip(1.0 - disk, 0.0, 1.0)
    p_core = (0.00038 + 0.00085 * rng.random((height, width))) * sky_gate * t
    core = np.where(
        rng.random((height, width)) < p_core,
        (0.012 + 0.034 * rng.random((height, width))) * (0.60 + 0.40 * t),
        0.0,
    )
    halo = _blur_separable_xy(core, passes=1, periodic_x=True)
    speckle = np.clip(core + halo * 0.12, 0.0, 0.038)
    value = np.clip(np.repeat(base, width, axis=1) + noise + speckle, 0.0, 0.24)
    # Slight per-render tint variation avoids a single fixed background color.
    tint_shift = rng.uniform(-0.004, 0.004)
    blue = np.clip(value + 0.010 + blue_noise + tint_shift, 0.0, 0.20)
    red = np.clip(value - 0.003 - tint_shift * 0.5, 0.0, 0.16)
    green = np.clip(value - 0.001, 0.0, 0.17)
    return np.stack([red, green, blue], axis=2).astype(HDR_DTYPE)


def _save_layer_u8(gray: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    Image.fromarray((g * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _export_morphology_debug_layers(
    cfg: RenderConfig,
    base_name: str,
    *,
    galactic: GalacticMorphology | None,
    stars_bg: np.ndarray,
    stars_mid: np.ndarray,
    stars_fg: np.ndarray,
    ext_paint: np.ndarray | None,
    rng_post: np.random.Generator,
) -> None:
    if not cfg.features.debug_export_layers and not cfg.features.debug_grayscale_morphology:
        return
    out_dir = cfg.output_dir / f"{base_name}_layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    if galactic is not None:
        maps = galactic.derived_maps()
        _save_layer_u8(maps["stellar_density"], out_dir / "density_G.png")
        # Morphology-only dust (pre-extinction merge). dust_D = T (bright = clear).
        dust_a_m = maps.get("dust_absorption_morph", maps["dust_absorption"])
        dust_t_m = maps.get("dust_transmission_morph", maps["dust_transmission"])
        _save_layer_u8(dust_t_m, out_dir / "dust_D.png")
        _save_layer_u8(dust_a_m, out_dir / "dust_A.png")
        _save_layer_u8(np.clip(1.0 - dust_t_m, 0.0, 1.0), out_dir / "dust_opacity.png")
        _save_layer_u8(maps["dust_absorption"], out_dir / "dust_A_effective.png")
        _save_layer_u8(maps["dust_transmission"], out_dir / "dust_D_effective.png")
        _save_layer_u8(galactic.disk_weight, out_dir / "disk_weight.png")
        if galactic.disk_thickness_modulation is not None:
            mod_vis = np.clip(
                0.5 + 0.5 * galactic.disk_thickness_modulation, 0.0, 1.0
            )
            _save_layer_u8(mod_vis, out_dir / "disk_thickness_mod.png")
        _save_layer_u8(maps["unresolved_total"], out_dir / "unresolved_U.png")
        _save_layer_u8(maps["resolve_weight"], out_dir / "resolve_W.png")
        _save_layer_u8(1.0 - maps["resolve_weight"], out_dir / "dropout_mask.png")
        _save_layer_u8(maps["void_mask"], out_dir / "void_mask.png")
        if "obliteration_mask" in maps:
            _save_layer_u8(maps["obliteration_mask"], out_dir / "obliteration_mask.png")
        if "brutal_erasure_mask" in maps:
            _save_layer_u8(maps["brutal_erasure_mask"], out_dir / "brutal_erasure_mask.png")
        if "structure_survival" in maps:
            _save_layer_u8(maps["structure_survival"], out_dir / "structure_survival.png")
        if "gold_population_weight" in maps:
            _save_layer_u8(maps["gold_population_weight"], out_dir / "gold_population_weight.png")
        acc = galactic.unresolved_accum
        acc_vis = acc / (float(np.percentile(acc, 99.5)) + 1e-8)
        _save_layer_u8(np.clip(acc_vis, 0.0, 1.0), out_dir / "unresolved_deposit.png")
        if cfg.features.debug_grayscale_morphology:
            _save_layer_u8(morphology_grayscale_preview(galactic), out_dir / "morphology_gray.png")
    if ext_paint is not None:
        _save_layer_u8(ext_paint, out_dir / "extinction.png")
    star_stack = np.maximum(0.0, stars_bg + stars_mid * 0.92 + stars_fg)
    luma = np.clip(rec709_luma(star_stack), 0.0, 1.0)
    _save_layer_u8(luma, out_dir / "resolved_stars.png")
    if galactic is not None:
        unresolved_only = np.zeros_like(star_stack)
        unresolved_only = compose_inherited_unresolved_field(
            unresolved_only,
            rng_post,
            galactic,
            texture_strength=cfg.features.background_texture_strength
            * cfg.features.unresolved_background_strength,
            periodic_x=bool(cfg.wrap_safe),
            deposit_primary=cfg.features.unresolved_deposit_primary,
        )
        _save_layer_u8(rec709_luma(unresolved_only), out_dir / "unresolved_only.png")


def _galactic_disk_weight(height: int, sigma: float = 0.46) -> np.ndarray:
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
    return np.exp(-((yy**2) / (sigma**2)))


def _apply_long_exposure_look(
    canvas: np.ndarray,
    rng: np.random.Generator,
    *,
    sky_w: np.ndarray,
    band: np.ndarray,
    disk_w: np.ndarray,
    vignette_strength: float = 1.0,
) -> np.ndarray:
    """Mimic real stacked wide-field frames: uneven sky, amp-style corner lift, asymmetric vignette."""
    h, w, _ = canvas.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    xx_p = _periodic_lon_grid_xx(xx)
    sky_m = np.clip(sky_w[..., None], 0.0, 1.0)
    # Blend so the bright disk is barely touched (mostly high latitudes / corners).
    mask = np.clip(0.88 * sky_m + 0.12 * (1.0 - disk_w[..., None]), 0.0, 1.0)

    ang = float(rng.uniform(0.0, 6.283185307179586))
    ca, sa = np.cos(ang), np.sin(ang)
    saw = xx_p * ca + yy * sa
    saw_n = np.clip(saw / max(abs(ca) + abs(sa), 0.25), -1.0, 1.0)
    warm = np.array([1.0, 0.94, 0.86], dtype=np.float64)
    cool = np.array([0.90, 0.93, 1.04], dtype=np.float64)
    t = (0.5 + 0.5 * saw_n)[..., None]
    rgb_tilt = warm * (1.0 - t) + cool * t
    sky_lift = (0.012 + 0.034 * sky_w[..., None]) * (0.52 + 0.48 * saw_n[..., None])
    out = canvas + sky_lift * rgb_tilt * mask

    cx = float(rng.uniform(-0.62, 0.62))
    cy = float(rng.uniform(-0.62, 0.62))
    wx = float(rng.uniform(0.38, 0.72))
    wy = float(rng.uniform(0.42, 0.78))
    dx_c = _wrap_lon_delta_xx_minus_a(xx, cx)
    glow = np.exp(-(((dx_c**2)) / wx + ((yy - cy) ** 2) / wy))
    glow *= np.clip(sky_w * (1.0 - band * 0.52), 0.0, 1.0)
    amp = np.array([0.018, 0.014, 0.011], dtype=np.float64) * float(rng.uniform(0.88, 1.32))
    out = out + glow[..., None] * amp

    ox = float(rng.uniform(-0.14, 0.14))
    oy = float(rng.uniform(-0.14, 0.14))
    ax = float(rng.uniform(0.82, 1.18))
    ay = float(rng.uniform(0.82, 1.18))
    dx_o = _wrap_lon_delta_xx_minus_a(xx, ox)
    rad = np.clip((dx_o**2) * ax + (yy - oy) ** 2 * ay, 0.0, 1.35)
    vscale = float(np.clip(vignette_strength, 0.0, 2.5))
    vig = 1.0 - (0.022 + 0.018 * rng.random()) * vscale * sky_w * np.clip(rad, 0.0, 1.25)
    vig = np.clip(vig, 0.955, 1.0)
    out = out * (mask * vig[..., None] + (1.0 - mask))

    return np.maximum(out, 0.0)


def _apply_galactic_disk_luminance_envelope(
    canvas: np.ndarray,
    rng: np.random.Generator,
    *,
    disk_w: np.ndarray,
    strength_scale: float = 1.0,
) -> np.ndarray:
    """Brighter near seeded galactic-longitude center along the band; darker toward equirect poles."""
    h, w, _ = canvas.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    sy = float(rng.uniform(0.26, 0.38))
    vert = 0.74 + 0.26 * np.exp(-((yy**2) / (sy**2)))
    gc_x = float(np.clip(rng.normal(0.0, 0.12), -0.40, 0.40))
    sx = float(rng.uniform(0.088, 0.19))
    amp = float(rng.uniform(0.09, 0.175)) * float(np.clip(strength_scale, 0.35, 1.25))
    lu_med = float(np.median(rec709_luma(np.maximum(canvas, 0.0))[np.clip(disk_w, 0, 1) > 0.28]))
    if lu_med > 0.16:
        amp *= float(np.clip(1.0 - (lu_med - 0.16) / 0.32, 0.25, 1.0))
    dx_gc = _wrap_lon_delta_xx_minus_a(xx, gc_x)
    bulge = 1.0 + amp * np.exp(-((dx_gc**2) / (sx**2 + 1e-9)))
    skew = float(rng.uniform(-0.13, 0.13))
    bulge = bulge * (1.0 + skew * np.tanh(dx_gc * 2.6))
    center_notch = np.exp(-((dx_gc**2) / 0.07)) * np.exp(-((yy**2) / 0.055))
    bulge = 1.0 + (bulge - 1.0) * np.clip(1.0 - center_notch * 0.28, 0.55, 1.0)
    bulge = np.clip(bulge, 0.96, 1.06 if lu_med > 0.18 else 1.10)
    plane_gate = 0.52 + 0.48 * np.exp(-((yy**2) / 0.50))
    horiz = 1.0 + (bulge - 1.0) * plane_gate
    scale = np.clip(vert * horiz, 0.72, 1.06 if lu_med > 0.18 else 1.20)
    # Ease off in the outer disk halo so grade stacks gently with disk_w-based passes later.
    ease = 0.62 + 0.38 * disk_w
    scale = 1.0 + (scale - 1.0) * ease
    # #region agent log
    _cr = int(canvas.shape[0] // 2)
    _row_scale = scale[_cr, :]
    _dbg_log(
        "A",
        "generator.py:_apply_galactic_disk_luminance_envelope",
        "disk envelope scale along center row",
        {
            "lu_med": lu_med,
            "amp": amp,
            "center_notch_peak": float(np.max(center_notch)),
            "scale_row_min": float(np.min(_row_scale)),
            "scale_row_max": float(np.max(_row_scale)),
            "scale_row_mean": float(np.mean(_row_scale)),
            "bulge_row_min": float(np.min(bulge[_cr, :])),
            "bulge_row_max": float(np.max(bulge[_cr, :])),
        },
    )
    # #endregion
    return np.maximum(canvas * scale[..., None], 0.0)


def _soft_knee_star_layer(
    star_img: np.ndarray, disk_w: np.ndarray, *, knee: float = 0.52, strength: float = 0.95
) -> None:
    """Reduce clipped white mush where many disk stars overlap (in-place)."""
    lum = np.mean(np.clip(star_img, 0.0, None), axis=2)
    excess = np.maximum(0.0, lum - knee)
    factor = 1.0 / (1.0 + strength * excess)
    w = disk_w * 0.90 + 0.10
    star_img *= (w * factor + (1.0 - w))[..., None]


def _luma_tone_map_disk(rgb: np.ndarray, disk_w: np.ndarray, *, k: float = 0.52) -> np.ndarray:
    """Reinhard-style luma roll-off in the disk only, preserving hue."""
    lin = np.clip(rgb, 0.0, None)
    luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    l_new = luma / (1.0 + k * luma)
    scale = np.divide(l_new, luma, out=np.ones_like(luma), where=luma > 1e-8)
    adjusted = lin * scale[..., None]
    w = disk_w[..., None]
    return np.maximum(rgb * (1.0 - w) + adjusted * w, 0.0)


def _disk_gamma_lift(rgb: np.ndarray, disk_w: np.ndarray, *, gamma: float = 0.93) -> np.ndarray:
    """Slight gamma in the disk only: lifts shadow detail (photo print / sensor response)."""
    w = disk_w[..., None]
    lifted = np.maximum(rgb, 0.0) ** gamma
    return np.maximum(0.0, rgb * (1.0 - w) + lifted * w)


def _disk_photo_grade(rgb: np.ndarray, disk_w: np.ndarray) -> np.ndarray:
    """Local disk grade: mild toe lift + soft S-curve on luma (chrominance roughly preserved)."""
    w = disk_w[..., None]
    lin = np.maximum(rgb, 0.0)
    luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    toe = np.clip((0.055 - luma) / 0.055, 0.0, 1.0) ** 0.62
    l_lift = np.clip(luma + 0.018 * toe * disk_w, 0.0, 1.0)
    t = l_lift - 0.5
    l_curve = np.clip(0.5 + t * (1.0 + 0.36 * (0.25 - t * t)), 0.0, 1.0)
    scale = np.divide(l_curve, l_lift, out=np.ones_like(l_lift), where=l_lift > 1e-9)
    graded = np.maximum(lin * scale[..., None], 0.0)
    return np.maximum(rgb * (1.0 - w) + graded * w, 0.0)


def _dust_scattered_backlight(
    ext: np.ndarray,
    *,
    galaxy_streak: bool,
    plane_gate: np.ndarray | None = None,
) -> np.ndarray:
    """Warm interstellar glow in thick dust (avoids pure black cutouts)."""
    pg = np.clip(np.asarray(plane_gate, dtype=np.float64), 0.0, 1.0) if plane_gate is not None else 1.0
    thick = np.clip(1.0 - ext, 0.0, 1.0) * pg
    warm = np.array([0.16, 0.11, 0.075], dtype=np.float64)
    brown = np.array([0.085, 0.055, 0.038], dtype=np.float64)
    if galaxy_streak:
        s = (thick**1.28) * 0.019
        s2 = (thick**1.78) * 0.007
    else:
        s = (thick**1.18) * 0.034
        s2 = (thick**1.7) * 0.015
    return warm * s[..., None] + brown * s2[..., None]


def _dust_rim_light(
    ext: np.ndarray,
    *,
    galaxy_streak: bool,
    plane_gate: np.ndarray | None = None,
) -> np.ndarray:
    """Warm light on extinction gradients (cloud edges / partial transparency)."""
    pbx = bool(galaxy_streak)
    pg = np.clip(np.asarray(plane_gate, dtype=np.float64), 0.0, 1.0) if plane_gate is not None else 1.0
    sm = _blur_separable_xy(ext, passes=2 if galaxy_streak else 1, periodic_x=pbx)
    rim = np.clip(sm - ext, 0.0, 1.0) * pg
    rim = _blur_separable_xy(rim, passes=1, periodic_x=pbx)
    rim = np.clip(rim**0.88, 0.0, 1.0)
    warm = np.array([0.28, 0.19, 0.13], dtype=np.float64)
    mag = np.array([0.34, 0.07, 0.22], dtype=np.float64)
    amp = 0.028 if galaxy_streak else 0.034
    return rim[..., None] * (warm + mag * 0.30) * amp


def _dust_volume_mottle(
    rng: np.random.Generator,
    ext: np.ndarray,
    *,
    galaxy_streak: bool,
    plane_gate: np.ndarray | None = None,
) -> np.ndarray:
    """Low-contrast noise inside thick dust (internal structure, not a flat mask)."""
    pg = np.clip(np.asarray(plane_gate, dtype=np.float64), 0.0, 1.0) if plane_gate is not None else 1.0
    thick = np.clip(1.0 - ext, 0.0, 1.0) * pg
    n = rng.normal(0.0, 1.0, size=ext.shape)
    n = _blur_separable_xy(n, passes=1, periodic_x=bool(galaxy_streak))
    brown = np.array([0.10, 0.086, 0.078], dtype=np.float64)
    w = (thick**1.22)[..., None]
    amp = 0.010 if galaxy_streak else 0.008
    return n[..., None] * brown * w * amp


def _apply_extinction_to_canvas(
    canvas: np.ndarray,
    ext: np.ndarray,
    *,
    galaxy_streak: bool,
    rng_mottle: np.random.Generator | None = None,
    av_scale: float = 2.65,
    rv: float = 3.1,
    fill_suppress: float = 1.0,
    plane_gate: np.ndarray | None = None,
) -> np.ndarray:
    """CCM per-channel attenuation in linear space, plus optional dust backlight/rim."""
    if galaxy_streak:
        ext = np.clip(ext * 0.992 + 0.008, 0.0, 1.0)
    out = apply_ccm_extinction_linear(canvas, ext, av_scale=av_scale, rv=rv)
    fs = float(np.clip(fill_suppress, 0.0, 1.0))
    if fs > 1e-6:
        ext_c = np.clip(np.asarray(ext, dtype=np.float64), 0.0, 1.0)
        dark = np.clip(1.0 - ext_c, 0.0, 1.0)
        dark_soft = _blur_separable_xy(dark, passes=1, periodic_x=bool(galaxy_streak))
        lane_edge = np.clip(dark - dark_soft * 0.78, 0.0, 1.0) ** 1.08
        fill_w = np.clip(dark**1.10 * (0.30 + 0.70 * lane_edge), 0.0, 1.0) * fs
        fill_3 = fill_w[..., np.newaxis]
        out = out + _dust_scattered_backlight(
            ext_c, galaxy_streak=galaxy_streak, plane_gate=plane_gate
        ) * fill_3
        out = out + _dust_rim_light(ext_c, galaxy_streak=galaxy_streak, plane_gate=plane_gate) * fill_3
        if rng_mottle is not None:
            out = out + _dust_volume_mottle(
                rng_mottle, ext_c, galaxy_streak=galaxy_streak, plane_gate=plane_gate
            ) * (fill_3 * 0.85)
    return np.maximum(out, 0.0)


def _extinction_from_dust_and_lane(
    dust_occlusion: np.ndarray,
    lane_ext: np.ndarray,
    cfg: RenderConfig,
) -> np.ndarray:
    base_extinction_strength = 0.56 if cfg.nebula_mode == NebulaMode.galaxy_streak else 0.38
    extinction_strength = (
        base_extinction_strength * cfg.nebula_tuning.dust_strength * cfg.features.dust_opacity
    )
    lane_boost = np.clip(
        lane_ext * (0.78 + 0.32 * cfg.nebula_tuning.dust_strength),
        0.0,
        1.0,
    )
    lane_k = 0.34 + 0.20 * cfg.nebula_tuning.dust_strength if cfg.nebula_mode == NebulaMode.galaxy_streak else 0.0
    floor = 0.06 if cfg.nebula_mode == NebulaMode.galaxy_streak else 0.10
    return np.clip(
        1.0 - dust_occlusion * extinction_strength - lane_boost * lane_k,
        floor,
        1.0,
    )


def _extinction_gradient_mag(ext: np.ndarray, *, periodic_x: bool) -> np.ndarray:
    if periodic_x:
        gx = 0.5 * (np.roll(ext, -1, axis=1) - np.roll(ext, 1, axis=1))
    else:
        gx = np.gradient(ext, axis=1)
    gy = np.gradient(ext, axis=0)
    return np.sqrt(np.clip(gx, -1e6, 1e6) ** 2 + np.clip(gy, -1e6, 1e6) ** 2 + 1e-12)


def _guided_extinction_refine(ext: np.ndarray, *, periodic_x: bool, smoke_fill: float = 0.38) -> np.ndarray:
    """Edge-aware blend; low smoke_fill keeps fractured absorption (not cloudy smear)."""
    g = _extinction_gradient_mag(ext, periodic_x=periodic_x)
    g93 = float(np.quantile(g, 0.93))
    gn = np.clip(g / (g93 + 1e-6), 0.0, 1.0)
    sm = _blur_separable_xy(ext, passes=1, periodic_x=periodic_x)
    w = np.clip(float(smoke_fill) * gn, 0.0, 0.28)
    out = ext * (1.0 - w) + sm * w
    return np.clip(out, 0.0, 1.0)


def _apply_separated_disk_sky_grain(
    canvas: np.ndarray,
    rng: np.random.Generator,
    *,
    height: int,
    width: int,
    periodic_x: bool,
    texture_strength: float = 1.0,
    speckle_scale: float = 1.0,
) -> np.ndarray:
    t = float(np.clip(texture_strength, 0.0, 2.0))
    sp = float(np.clip(speckle_scale, 0.0, 2.0))
    """Crisp unresolved-star texture: denser in galactic band, sparse in outer sky."""
    yy = np.linspace(-1.0, 1.0, height)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float64)[None, :]
    band = np.exp(-((yy**2) / 0.55))
    sky_w = np.clip(1.0 - band, 0.0, 1.0) ** 0.38
    # Seed-driven bulge placement/shape so center concentration is not static between seeds.
    bulge_cx = float(np.clip(rng.normal(0.0, 0.20), -0.62, 0.62))
    bulge_w = float(rng.uniform(0.16, 0.36))
    dx_b = _wrap_lon_delta_xx_minus_a(xx, bulge_cx)
    bulge = np.exp(-((dx_b**2) / bulge_w))
    if rng.random() < 0.55:
        bulge2_cx = float(np.clip(bulge_cx + rng.normal(0.0, 0.24), -0.9, 0.9))
        bulge2_w = float(rng.uniform(0.22, 0.52))
        dx_b2 = _wrap_lon_delta_xx_minus_a(xx, bulge2_cx)
        bulge2 = np.exp(-((dx_b2**2) / bulge2_w))
        bulge = np.clip(np.maximum(bulge, bulge2 * float(rng.uniform(0.42, 0.78))), 0.0, 1.0)

    # 2-D low-frequency floor (separable blur on white noise reads as vertical streaks in equirect).
    from starsky_gen.procedural_noise import _resize_bilinear, fbm2d

    ch, cw = max(4, height // 28), max(6, width // 22)
    gn_lp = fbm2d(rng, ch, cw, base_scale=0.12, octaves=3, periodic_x=periodic_x)
    gn_lp = _resize_bilinear(gn_lp, height, width, periodic_x=periodic_x)
    gn_lp = (gn_lp - 0.5) * 0.026
    disk_sm = np.stack(
        [gn_lp * 0.97, gn_lp * 1.0, gn_lp * 1.02],
        axis=2,
    ).astype(np.float64)
    out = np.maximum(0.0, canvas + disk_sm * band[..., None] * (0.006 * t))

    # Full-res point process: crisp micro-speckles everywhere with center/band weighting.
    core_center = np.clip((band**0.58) * (0.74 + 0.26 * bulge), 0.0, 1.0)
    sky_gate = np.clip((sky_w - 0.10) / 0.90, 0.0, 1.0)

    p_core = (
        (0.00032 + 0.00092 * rng.random((height, width))) * sky_gate
        + (0.00180 + 0.00480 * rng.random((height, width))) * core_center
    )
    p_core *= t * sp
    core_amp = (
        (0.003 + 0.010 * rng.random((height, width))) * sky_gate
        + (0.004 + 0.014 * rng.random((height, width))) * core_center
    )
    speckle_core = np.where(rng.random((height, width)) < p_core, core_amp, 0.0)

    # Near-zero halo: keep unresolved texture crisp, not smeared (pole-gated to avoid vertical bleed).
    from starsky_gen.structure_envelope import latitude_plane_gate

    pole_gate = latitude_plane_gate(height, sigma=0.34, power=1.2)
    speckle_halo = _blur_separable_xy(speckle_core * pole_gate, passes=1, periodic_x=periodic_x)
    speckle_map = np.clip(speckle_core * 1.20 + speckle_halo * 0.01, 0.0, 0.052 * (0.65 + 0.35 * t))

    # Ultra-fine "dust" of unresolved stars across the whole frame, stronger in the galactic band.
    micro_p = (
        (0.0038 + 0.0058 * rng.random((height, width))) * sky_gate
        + (0.0080 + 0.0150 * rng.random((height, width))) * core_center
    )
    micro_p *= t * sp
    micro_amp = (
        (0.0012 + 0.0048 * rng.random((height, width))) * sky_gate
        + (0.0015 + 0.0058 * rng.random((height, width))) * core_center
    )
    micro_core = np.where(rng.random((height, width)) < micro_p, micro_amp, 0.0)
    speckle_map = np.clip(speckle_map + micro_core, 0.0, 0.060 * (0.65 + 0.35 * t))

    # Extra pin-point layer: tiny, high-density, mostly single-pixel points.
    pin_p = (
        (0.0040 + 0.0060 * rng.random((height, width))) * sky_gate
        + (0.0090 + 0.0160 * rng.random((height, width))) * core_center
    )
    pin_p *= t * sp
    pin_amp = (
        (0.0009 + 0.0028 * rng.random((height, width))) * sky_gate
        + (0.0011 + 0.0034 * rng.random((height, width))) * core_center
    )
    pin_core = np.where(rng.random((height, width)) < pin_p, pin_amp, 0.0)
    speckle_map = np.clip(speckle_map + pin_core, 0.0, 0.062 * (0.65 + 0.35 * t))

    # Unresolved speckle stays near-neutral: tiny luminance cannot carry hue jitter.
    ct = rng.normal(0.0, 1.0, size=(height, width))
    r_mul = np.clip(1.0 + ct * 0.0025, 0.98, 1.02)
    g_mul = np.clip(1.0 + ct * 0.0010, 0.99, 1.01)
    b_mul = np.clip(1.0 - ct * 0.0028, 0.98, 1.02)
    star_rgb_jitter = np.stack([r_mul, g_mul, b_mul], axis=2)
    neutral_star_rgb = np.array([0.992, 0.992, 0.992], dtype=np.float64)
    speckle_rgb = speckle_map[..., None] * star_rgb_jitter * neutral_star_rgb

    # PSF-size variance: mostly sharp points, with a minority of slightly broader stars.
    psf_gate = rng.random((height, width))
    spread_seed = np.where(psf_gate < 0.12, speckle_map, 0.0)
    spread = _blur_separable_xy(spread_seed, passes=1, periodic_x=periodic_x)
    speckle_rgb = np.clip(
        speckle_rgb + spread[..., None] * np.array([0.90, 0.92, 0.96], dtype=np.float64) * (0.16 * t),
        0.0,
        0.14 * (0.70 + 0.30 * t),
    )

    # Sparse brighter anchor stars with tiny halos to match real-frame star hierarchy.
    anchor_p = (
        (0.00003 + 0.00008 * rng.random((height, width))) * sky_gate
        + (0.00010 + 0.00020 * rng.random((height, width))) * core_center
    )
    anchor_p *= t
    anchor_amp = (
        (0.030 + 0.090 * rng.random((height, width))) * sky_gate
        + (0.050 + 0.140 * rng.random((height, width))) * core_center
    )
    anchor_core = np.where(rng.random((height, width)) < anchor_p, anchor_amp, 0.0)
    anchor_halo = _blur_separable_xy(anchor_core, passes=1, periodic_x=periodic_x)
    anchor_rgb = (
        anchor_core[..., None] * np.array([0.98, 0.99, 1.00], dtype=np.float64)
        + anchor_halo[..., None] * np.array([0.90, 0.92, 0.97], dtype=np.float64) * 0.14
    )

    out = np.maximum(
        0.0,
        out + (speckle_rgb + anchor_rgb) * (0.30 + 0.54 * sky_gate + 0.96 * core_center)[..., None] * t,
    )
    return out


def _mute_speckle_under_nebula(
    canvas: np.ndarray,
    neb_luma: np.ndarray,
    *,
    gas_mask: np.ndarray,
    ext_paint: np.ndarray | None = None,
    emit_luma: np.ndarray | None = None,
    strength: float = 0.62,
) -> np.ndarray:
    """Attenuate background speckle where ISM should block unresolved light (not show through gas)."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return canvas
    gas = np.clip(np.asarray(neb_luma, dtype=np.float64), 0.0, 1.0)
    if emit_luma is not None:
        gas = np.maximum(gas, np.clip(np.asarray(emit_luma, dtype=np.float64), 0.0, 1.0) * 0.42)
    gm = np.clip(np.asarray(gas_mask, dtype=np.float64), 0.0, 1.0)
    thick = np.clip(gas**1.08 * gm, 0.0, 1.0)
    thick = np.where(thick > np.quantile(thick, 0.58), thick, thick * 0.35)
    if ext_paint is not None:
        ext = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
        lo = float(np.percentile(ext, 12.0))
        hi = float(np.percentile(ext, 88.0))
        rel = np.clip((ext - lo) / max(hi - lo, 0.06), 0.0, 1.0)
        occ = thick * (0.32 + 0.68 * rel)
    else:
        occ = thick
    occ = np.clip(occ**0.82, 0.0, 0.90)
    return np.maximum(0.0, canvas * (1.0 - occ * s)[..., np.newaxis])


def _band_micro_ripple(
    rgb: np.ndarray,
    rng: np.random.Generator,
    disk_w: np.ndarray,
    *,
    strength: float = 0.011,
    neb_luma: np.ndarray | None = None,
) -> np.ndarray:
    """High-frequency luma ripples in the disk (unresolved star / grain texture)."""
    h, w, _ = rgb.shape
    gate = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if neb_luma is not None:
        nl = np.clip(np.asarray(neb_luma, dtype=np.float64), 0.0, 1.0)
        gate = gate * np.clip(1.0 - nl**0.68 * 0.82, 0.12, 1.0)
    a = rng.normal(0.0, 1.0, size=(h, w))
    b = rng.normal(0.0, 0.65, size=(h, w))
    # Small shifts on both axes reduce row/column grain in the disk.
    b = np.roll(b, int(rng.integers(-3, 4)), axis=1)
    b = np.roll(b, int(rng.integers(-2, 3)), axis=0)
    fine = (a + b * 0.55) * strength
    neutral = np.array([0.34, 0.36, 0.40], dtype=np.float64)
    neutral /= np.sum(neutral)
    return np.maximum(0.0, rgb + gate[..., None] * fine[..., None] * neutral)


def _add_galactic_cloud_body(
    canvas: np.ndarray,
    rng: np.random.Generator,
    *,
    neb_luma: np.ndarray,
    ext_paint: np.ndarray,
    disk_w: np.ndarray,
    periodic_x: bool,
    strength: float = 1.0,
) -> np.ndarray:
    """Add broad continuum clouds so the Milky Way reads as volume, not only stars."""
    h, w, _ = canvas.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    band_gate = np.exp(-((yy**2) / 0.42))
    clear = np.clip(ext_paint, 0.0, 1.0)
    dusty = np.clip(1.0 - clear, 0.0, 1.0)

    neb_raw = np.clip(neb_luma, 0.0, 1.0)
    body = _blur_separable_xy(neb_raw * band_gate, passes=3, periodic_x=periodic_x)
    body = _blur_x_only(body, passes=2, periodic_x=periodic_x)
    body_hi = np.clip(neb_raw - _blur_separable_xy(neb_raw, passes=2, periodic_x=periodic_x), 0.0, 1.0)
    body = np.clip(body**0.88 + body_hi * 0.22, 0.0, 1.0)

    # Very low-frequency envelope keeps cloud masses coherent across longitude.
    env_small = _resize_bilinear(
        rng.random((max(2, h // 24), max(2, w // 20))),
        h,
        w,
        periodic_x=periodic_x,
    )
    env = _blur_separable_xy(env_small, passes=4, periodic_x=periodic_x)
    env = np.clip(0.82 + 0.32 * env, 0.80, 1.14)

    # Multi-scale breakup prevents smooth, "airbrushed" blobs.
    coarse = _resize_bilinear(
        rng.random((max(2, h // 18), max(2, w // 16))), h, w, periodic_x=periodic_x
    )
    coarse = _blur_separable_xy(coarse, passes=3, periodic_x=periodic_x)
    fine = _resize_bilinear(
        rng.random((max(2, h // 40), max(2, w // 36))), h, w, periodic_x=periodic_x
    )
    fine = _blur_separable_xy(fine, passes=1, periodic_x=periodic_x)
    breakup = np.clip(0.78 + 0.34 * coarse + (fine - 0.5) * 0.32, 0.70, 1.28)

    # Add soft edge wisps from extinction gradients so dust has feathered boundaries.
    clear_sm = _blur_separable_xy(clear, passes=2, periodic_x=periodic_x)
    edge = np.clip(clear_sm - clear, 0.0, 1.0)
    edge = _blur_separable_xy(edge, passes=1, periodic_x=periodic_x)
    edge = np.clip(edge**0.9, 0.0, 1.0)

    flow = _resize_bilinear(
        rng.random((max(2, h // 20), max(2, w // 14))), h, w, periodic_x=periodic_x
    )
    flow = _blur_separable_xy(flow, passes=2, periodic_x=periodic_x)
    # Break long linear lane artifacts with curved, low-frequency flow modulation.
    yyf = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xxf = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    xxf_p = _periodic_lon_grid_xx(xxf)
    ph = float(rng.uniform(0.0, 6.283185307179586))
    wave = 0.5 + 0.5 * np.sin(2.1 * xxf_p + 1.3 * yyf + ph)
    flow_mod = np.clip(0.82 + 0.22 * flow + 0.12 * wave, 0.72, 1.16)

    cloud_w = np.clip(body * (0.62 + 0.38 * clear) * env * breakup * flow_mod * band_gate, 0.0, 1.0)
    cloud_w = np.clip(cloud_w + edge * band_gate * 0.24, 0.0, 1.0)
    warm = np.array([0.19, 0.17, 0.14], dtype=np.float64)
    cool = np.array([0.09, 0.11, 0.15], dtype=np.float64)
    cloud_rgb = cloud_w[..., None] * (warm * 0.74 + cool * 0.26)
    cloud_rgb *= (0.66 + 0.34 * disk_w)[..., None]

    # Dust silhouettes should still read as textured warm haze, not pure black cuts.
    dust_fill = _blur_separable_xy(dusty * band_gate, passes=3, periodic_x=periodic_x)
    dust_mod = _blur_separable_xy(
        np.clip((coarse * 0.72 + fine * 0.28), 0.0, 1.0), passes=1, periodic_x=periodic_x
    )
    dust_rgb = dust_fill[..., None] * np.array([0.07, 0.052, 0.038], dtype=np.float64) * (0.34 + 0.24 * dust_mod[..., None])
    s = float(np.clip(strength, 0.25, 2.5))
    return np.maximum(0.0, canvas + (cloud_rgb * 0.82 + dust_rgb) * s)


def _band_render_envelope(
    height: int,
    width: int,
    *,
    band_bleed_env: np.ndarray | None,
    bleed_gate: np.ndarray | None,
    plane_gate: np.ndarray | None,
    lat_sigma: float = 0.52,
) -> np.ndarray:
    """Latitudinal envelope for haze/cloud/bloom — feathered disk, not a parallel railroad track."""
    bg = bleed_gate if bleed_gate is not None else (
        plane_gate if plane_gate is not None else 1.0
    )
    if band_bleed_env is not None:
        return np.clip(np.asarray(band_bleed_env, dtype=np.float64) * bg, 0.0, 1.45)
    yy = np.linspace(-1.0, 1.0, int(height), dtype=np.float64)[:, None]
    return np.clip(np.exp(-((yy**2) / float(lat_sigma))) * bg, 0.0, 1.0)


def _structure_host_gate(
    galactic: GalacticMorphology | None,
    height: int,
    width: int,
) -> np.ndarray:
    """Band hosts structure; envelope can extend above/below the plane."""
    if galactic is not None:
        from starsky_gen.structure_envelope import latitude_plane_gate

        host = np.clip(galactic.structure_survival, 0.0, 1.35)
        return host * latitude_plane_gate(int(height), sigma=0.42, power=1.08)
    yy = np.linspace(-1.0, 1.0, int(height), dtype=np.float64)[:, None]
    return np.exp(-((yy**2) / 0.50))


def _add_morphology_band_nebula_body(
    canvas: np.ndarray,
    galactic: GalacticMorphology,
    disk_w: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
    hierarchy_strength: float = 0.82,
    white_brightness: float = 1.22,
    periodic_x: bool,
) -> np.ndarray:
    """In-band ISM pass only: white bright, gold warm, black dust (single additive pass).

    Red H II is accumulated separately in ``off_band_layer`` (decoupled from the band).
    """
    from starsky_gen.structure_envelope import build_morphology_ism_rgb

    h, w = canvas.shape[0], canvas.shape[1]
    rgb, _ = build_morphology_ism_rgb(
        galactic,
        rng,
        h,
        w,
        hierarchy_strength=hierarchy_strength,
        white_brightness=white_brightness,
        periodic_x=periodic_x,
    )
    amp = float(np.clip(strength, 0.0, 4.0)) * 0.72
    return np.maximum(0.0, canvas + rgb * amp)


def _off_band_emission_mask(
    disk_w: np.ndarray,
    galactic: GalacticMorphology | None,
    height: int,
    *,
    decouple_strength: float = 1.0,
    band_lat_sigma: float = 0.12,
    use_vertical_extent: bool = False,
) -> np.ndarray:
    """Latitude/disk mask for red H II — decoupled from band host envelope."""
    from starsky_gen.structure_envelope import build_off_band_mask

    ve = galactic.vertical_extent if galactic is not None else None
    return build_off_band_mask(
        disk_w,
        ve,
        height,
        band_lat_sigma=band_lat_sigma,
        decouple_strength=decouple_strength,
        use_vertical_extent=use_vertical_extent,
    )


_OFF_BAND_HII_VARIANTS: tuple[np.ndarray, ...] = (
    np.array([1.42, 0.24, 0.14], dtype=np.float64),  # classic Hα red
    np.array([1.05, 0.38, 0.26], dtype=np.float64),  # Hα + OIII blend
    np.array([0.88, 0.52, 0.36], dtype=np.float64),  # warm pink H II
)


def _build_off_band_hii_layer(
    galactic: GalacticMorphology,
    disk_w: np.ndarray,
    *,
    strength: float,
    periodic_x: bool,
    band_lat_sigma: float = 0.12,
    blob_count: int = 3,
    diffuse_weight: float = 0.08,
    hii_seed: int,
) -> np.ndarray:
    """Off-band H II: 2–3 regional red nebulae outside the galactic band (seed-locked)."""
    from starsky_gen.structure_envelope import (
        build_hii_emission_hierarchy,
        build_turbulent_hii_emission_cloud,
        derive_nebula_rng,
    )

    h, w = galactic.height, galactic.width
    s = float(np.clip(strength, 0.0, 3.5))
    if s < 1e-6:
        return np.zeros((h, w, 3), dtype=np.float64)
    off = _off_band_emission_mask(
        disk_w,
        galactic,
        h,
        decouple_strength=1.12,
        band_lat_sigma=max(float(band_lat_sigma) * 1.15, 0.10),
        use_vertical_extent=True,
    )
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    from starsky_gen.structure_envelope import build_hii_near_band_placement_score

    sf = np.clip(galactic.star_formation, 0.0, 1.0)
    clump = np.clip(sf * (0.55 + 0.45 * galactic.latent_turb), 0.0, 1.0)
    score_place = build_hii_near_band_placement_score(
        disk_w, sf, h, band_lat_sigma=band_lat_sigma, band_weight=0.78
    )
    score_place = np.clip(score_place * (0.38 + 0.62 * clump), 0.0, 1.0)
    sig_pb = max(float(band_lat_sigma), 0.05)
    plane_support = np.exp(-((yy / sig_pb) ** 2))
    layer = np.zeros((h, w, 3), dtype=np.float64)
    n_blobs = int(np.clip(blob_count, 0, 8))
    dw_frac = float(np.clip(diffuse_weight, 0.0, 0.45))
    field_rng = derive_nebula_rng(hii_seed, "hii_regional")
    regional = build_hii_emission_hierarchy(
        np.clip(score_place * (0.40 + 0.60 * clump), 0.0, 1.0),
        field_rng,
        periodic_x=periodic_x,
        disk_weight=disk_w,
        strength=0.88,
    )
    layer = np.maximum(
        layer,
        regional[..., np.newaxis]
        * np.array([1.18, 0.30, 0.24], dtype=np.float64)
        * (0.16 + 0.07 * s),
    )
    if n_blobs > 0:
        flat = regional.ravel()
        place_w = score_place.ravel()
        used: list[tuple[int, int]] = []
        for i in range(n_blobs):
            blob_rng = derive_nebula_rng(hii_seed, "off_band", i)
            cand = np.flatnonzero((flat > 0.12) & (place_w > 0.10))
            if cand.size < 1:
                cand = np.flatnonzero(flat > 0.06)
            if cand.size < 1:
                cand = np.flatnonzero(place_w > 1e-4)
            if cand.size < 1:
                row_probs = score_place[:, 0]
                row_probs = row_probs / (float(row_probs.sum()) + 1e-12)
                iy = int(blob_rng.choice(h, p=row_probs))
                ix = int(blob_rng.integers(0, w))
            else:
                wts = flat[cand].astype(np.float64)
                wts = wts / (float(wts.sum()) + 1e-12)
                pick = int(blob_rng.choice(cand, p=wts))
                iy, ix = divmod(pick, w)
                for _ in range(6):
                    if all(abs(iy - u[0]) > h // 10 or abs(ix - u[1]) > w // 8 for u in used):
                        break
                    pick = int(blob_rng.choice(cand, p=wts))
                    iy, ix = divmod(pick, w)
            used.append((iy, ix))
            cy = float(yy[iy, 0])
            cx = float(xx[0, ix])
            wy = float(blob_rng.uniform(0.05, 0.11))
            wx = float(blob_rng.uniform(0.05, 0.10))
            if i % 3 == 1:
                wy *= 1.18
                wx *= 0.92
            elif i % 3 == 2:
                wy *= 0.90
                wx *= 1.14
            cloud = build_turbulent_hii_emission_cloud(
                blob_rng,
                h,
                w,
                center_y=cy,
                center_x=cx,
                extent_y=wy,
                extent_x=wx,
                support_mask=np.clip(
                    plane_support * 0.62 + off * 0.28, 0.0, 1.0
                ) * float(blob_rng.uniform(0.78, 1.0)),
                periodic_x=periodic_x,
            )
            spot = cloud
            color = _OFF_BAND_HII_VARIANTS[i % len(_OFF_BAND_HII_VARIANTS)]
            amp_i = (0.40 + 0.16 * s) * float(blob_rng.uniform(0.95, 1.18))
            layer = np.maximum(layer, spot[..., np.newaxis] * color * amp_i)
        if dw_frac > 0.06:
            scale = float(max(h, w))
            mega_sig = float(np.clip(scale * 0.095, 18.0, 140.0))
            wash = gaussian_blur_pil(np.max(layer, axis=2), mega_sig * 0.18, periodic_x=periodic_x)
            layer = np.clip(
                layer
                + wash[..., np.newaxis]
                * plane_support[..., np.newaxis]
                * dw_frac
                * 0.05,
                0.0,
                1.45,
            )
    falloff = np.clip(plane_support[..., np.newaxis] * 0.68 + off[..., np.newaxis] * 0.22 + 0.14, 0.0, 1.0)
    layer = np.clip(layer * falloff**1.04, 0.0, 1.45)
    return layer.astype(np.float64)


def _composite_off_band_layer(
    canvas: np.ndarray,
    off_band_layer: np.ndarray,
    *,
    strength: float = 1.0,
    support_mask: np.ndarray | None = None,
    periodic_x: bool = True,
    localize: bool = True,
) -> np.ndarray:
    """Late add of H II emission (skips plane chroma harmonize when used after grade)."""
    layer = np.maximum(np.asarray(off_band_layer, dtype=np.float64), 0.0)
    if float(np.max(layer)) < 1e-8:
        return canvas
    if localize and support_mask is not None:
        layer = _localize_off_band_rgb(layer, support_mask, periodic_x=periodic_x)
    return composite_emission_chroma_preserve(
        canvas, layer, strength=float(np.clip(strength, 0.0, 3.5)), core_screen_mix=0.82
    )


def _composite_red_hii_late(
    canvas: np.ndarray,
    off_band_layer: np.ndarray | None,
    band_hii_layer: np.ndarray | None,
    *,
    strength: float = 1.0,
    periodic_x: bool = True,
    disk_w: np.ndarray | None = None,
) -> np.ndarray:
    """Add all H II RGB after grade — do not gate by off-band mask (that erased visible red)."""
    s = float(np.clip(strength, 0.0, 4.0))
    if disk_w is not None:
        lu = rec709_luma(np.maximum(np.asarray(canvas, dtype=np.float64), 0.0))
        h, w_img = lu.shape
        dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
        if dw.ndim == 1:
            dw = dw[:, None]
        if dw.shape != (h, w_img):
            dw = np.broadcast_to(dw, (h, w_img))
        plane = lu[dw > 0.22]
        if plane.size > 64:
            p95 = float(np.percentile(plane, 95))
            hot = float(np.clip((p95 - 0.42) / 0.30, 0.0, 1.0))
            s *= float(np.clip(1.0 - hot * 0.92, 0.06, 1.0))
    s = float(np.clip(s, 0.0, 0.32))
    # #region agent log
    _dbg_log(
        "E",
        "generator.py:_composite_red_hii_late",
        "effective late H II strength",
        {"config_strength": strength, "effective_s": s},
    )
    # #endregion
    out = canvas
    if band_hii_layer is not None and float(np.max(band_hii_layer)) > 1e-8:
        out = _composite_off_band_layer(
            out,
            band_hii_layer,
            strength=s * 0.38,
            support_mask=None,
            periodic_x=periodic_x,
            localize=False,
        )
    if off_band_layer is not None and float(np.max(off_band_layer)) > 1e-8:
        out = _composite_off_band_layer(
            out,
            off_band_layer,
            strength=s * 0.55,
            support_mask=None,
            periodic_x=periodic_x,
            localize=False,
        )
    return out


def _localize_off_band_rgb(
    layer: np.ndarray,
    support_mask: np.ndarray,
    *,
    periodic_x: bool,
    peak_percentile: float = 86.0,
) -> np.ndarray:
    """Compact off-band H II: suppress diffuse tails before composite."""
    from starsky_gen.structure_envelope import localize_emission_clouds

    rgb = np.maximum(np.asarray(layer, dtype=np.float64), 0.0)
    m = np.clip(np.asarray(support_mask, dtype=np.float64), 0.0, 1.0)
    lu = rec709_luma(rgb)
    loc = localize_emission_clouds(
        lu,
        m,
        periodic_x=periodic_x,
        peak_percentile=peak_percentile,
        tail_floor=0.04,
    )
    scale = np.clip(loc / (lu + 1e-6), 0.0, 1.0)
    return np.clip(rgb * scale[..., np.newaxis] * m[..., np.newaxis], 0.0, 1.25)


def _morphology_continuum_guide(
    morph_gas_struct: np.ndarray,
    ext_paint: np.ndarray,
    *,
    morph_ism_luma: np.ndarray | None = None,
    dust_absorption_morph: np.ndarray | None = None,
    periodic_x: bool = True,
) -> np.ndarray:
    """HF-preserving guide for haze/cloud/glow when morphology layers are primary.

    Post-gas passes default to neb_luma (smooth procedural). When morph-primary, drive haze/cloud
    from morph maps so tight turbulent structure is not re-washed after composite_add_gas.
    """
    g = np.clip(np.asarray(morph_gas_struct, dtype=np.float64), 0.0, 1.0)
    if morph_ism_luma is not None:
        g = np.clip(g * 0.50 + np.clip(morph_ism_luma, 0.0, 1.0) * 0.50, 0.0, 1.0)
    dark = np.clip(1.0 - np.clip(ext_paint, 0.0, 1.0), 0.0, 1.0) ** 1.04
    if dust_absorption_morph is not None:
        da = np.clip(np.asarray(dust_absorption_morph, dtype=np.float64), 0.0, 1.0)
        med_d = _blur_separable_xy(da, passes=1, periodic_x=periodic_x)
        lane = np.clip(da - med_d * 0.74, 0.0, 1.0) ** 1.14
        g = np.clip(np.maximum(g, lane * 0.62 + dark * 0.38), 0.0, 1.0)
    else:
        g = np.clip(np.maximum(g, dark * 0.52), 0.0, 1.0)
    scale = float(max(g.shape))
    sig = float(np.clip(scale * 0.0048, 0.35, 4.8))
    from starsky_gen.procedural_noise import gaussian_blur_pil

    med_g = gaussian_blur_pil(g, sig, periodic_x=periodic_x)
    hp = np.clip(g - med_g * 0.76, 0.0, 1.0) ** 1.06
    return np.clip(g * 0.26 + hp * 0.74, 0.0, 1.0).astype(np.float64)


def _reinforce_morph_canvas_texture(
    canvas: np.ndarray,
    morph_gas_struct: np.ndarray,
    disk_w: np.ndarray,
    *,
    plane_gate: np.ndarray | None = None,
    strength: float = 0.82,
) -> np.ndarray:
    """Re-seat puffy morphology on the graded canvas (layers had it; wash removed it)."""
    s = float(np.clip(strength, 0.0, 1.4))
    if s < 1e-6:
        return canvas
    g = np.clip(np.asarray(morph_gas_struct, dtype=np.float64), 0.0, 1.0)
    gate = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if plane_gate is not None:
        gate = gate * np.clip(np.asarray(plane_gate, dtype=np.float64), 0.0, 1.0)
    from starsky_gen.procedural_noise import gaussian_blur_pil

    scale = float(max(g.shape))
    sig = float(np.clip(scale * 0.0035, 0.4, 4.5))
    med = gaussian_blur_pil(g, sig, periodic_x=True)
    hp = np.clip(g - med * 0.72, 0.0, 1.0) ** 1.02
    puff = hp * gate
    void = (1.0 - hp) * gate
    warm = np.array([0.20, 0.16, 0.12], dtype=np.float64)
    out = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    out = out + puff[..., np.newaxis] * warm * (0.46 * s)
    out = out * (1.0 - void[..., np.newaxis] * (0.17 * s))
    return np.maximum(0.0, out)


def _apply_extinction_band_display_wash(
    canvas: np.ndarray,
    ext_paint: np.ndarray,
    disk_w: np.ndarray,
    *,
    strength: float = 0.62,
) -> np.ndarray:
    """Screen a grey band wash with lane darks keyed to extinction (matches extinction.png)."""
    s = float(np.clip(strength, 0.0, 1.2))
    if s < 1e-6:
        return canvas
    rel = band_relative_clearance(ext_paint, disk_w, min_clear=0.16, power=0.92)
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw.ndim == 1:
        dw = dw[:, None]
    h, w = rel.shape
    if dw.shape != (h, w):
        dw = np.broadcast_to(dw, (h, w))
    host = np.clip(dw**1.05, 0.0, 1.0)
    wash_lu = np.clip((0.34 + 0.38 * rel) * host, 0.0, 0.78)
    wash_rgb = wash_lu[..., np.newaxis] * np.array([1.0, 0.995, 0.99], dtype=np.float64)
    lin = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    blend = np.clip(host * s, 0.0, 1.0)[..., np.newaxis]
    return np.maximum(lin, lin + wash_rgb * blend * (1.0 - lin))


def _sculpt_morphology_lane_contrast(
    canvas: np.ndarray,
    ext_paint: np.ndarray,
    disk_w: np.ndarray,
    *,
    plane_gate: np.ndarray | None,
    strength: float,
) -> np.ndarray:
    """Late multiplicative lane carve on the finished band (filament ext, not blob caps)."""
    s = float(np.clip(strength, 0.0, 1.2))
    if s < 1e-6:
        return canvas
    ext = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    from starsky_gen.structure_envelope import soften_band_envelope

    gate = soften_band_envelope(dw, dw.shape, periodic_x=True, lat_blur_sigma=16.0, power=0.54)
    if plane_gate is not None:
        gate = gate * np.clip(np.asarray(plane_gate, dtype=np.float64), 0.0, 1.0)
    rel = band_relative_clearance(ext, disk_w, min_clear=0.12, power=1.0)
    dark = np.clip(1.0 - rel, 0.0, 1.0) ** 1.22
    h, w = dark.shape
    row_mean = np.mean(dark, axis=1, keepdims=True)
    dark = np.clip(dark - row_mean * 0.62, 0.0, 1.0) ** 1.06
    d_soft = _blur_separable_xy(dark, passes=1, periodic_x=True)
    dark = np.clip(dark + (dark - d_soft) * 0.32, 0.0, 1.0) ** 1.10
    carve = np.clip(dark * gate * s, 0.0, 0.52)
    return np.maximum(0.0, canvas * (1.0 - carve[..., np.newaxis]))


def _compress_band_highlights(
    canvas: np.ndarray,
    band_gate: np.ndarray,
    *,
    knee: float = 0.36,
    compress: float = 0.72,
    disk_w: np.ndarray | None = None,
) -> np.ndarray:
    """Soft-knee + shoulder luma compression in the galactic band (keeps chroma, stops white clip)."""
    g = np.clip(np.asarray(band_gate, dtype=np.float64), 0.0, 1.0)
    if disk_w is not None:
        core = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0) ** 1.28
        g = np.clip(g + core * 0.28, 0.0, 1.35)
    if float(np.max(g)) < 1e-6:
        return canvas
    lin = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    kn = float(np.clip(knee, 0.14, 0.52))
    cp = float(np.clip(compress, 0.35, 2.2))
    span = max(1.0 - kn, 0.08)
    over = np.maximum(lu - kn, 0.0)
    # Shoulder: excess / (1 + excess * strength) — preserves sub-knee, crushes hot core.
    strength = (1.8 + cp * 5.5) * np.clip(g, 0.0, 1.35)
    rolloff = over / (1.0 + over / span * strength)
    lu_new = np.clip(kn + rolloff, 0.0, None)
    out = remap_luma_preserving_chroma(lin, lu_new)
    return np.maximum(0.0, out)


def _limit_band_linear_headroom(
    canvas: np.ndarray,
    disk_w: np.ndarray,
    *,
    percentile: float = 99.0,
    target_peak: float = 0.48,
) -> np.ndarray:
    """Scale down linear HDR when the stack runs hot (before asinh / display tone)."""
    lin = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    positive = lu[lu > 1e-8]
    if positive.size < 64:
        return lin
    peak = float(np.percentile(positive, float(np.clip(percentile, 96.0, 99.6))))
    tgt = float(np.clip(target_peak, 0.32, 0.62))
    if peak <= tgt + 1e-6:
        return lin
    scale = tgt / peak
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    h, w_img = int(lin.shape[0]), int(lin.shape[1])
    if dw.shape != (h, w_img):
        if dw.ndim == 2 and dw.shape[1] == 1:
            dw = np.broadcast_to(dw, (h, w_img))
        elif dw.size == h:
            dw = np.broadcast_to(dw.reshape(h, 1), (h, w_img))
    plane = np.clip(0.55 + 0.45 * dw**1.02, 0.55, 1.0)[..., np.newaxis]
    eff = scale * plane + scale * (1.0 - plane) * 0.82
    return np.maximum(0.0, lin * eff)


def finalize_band_linear_hdr(
    canvas: np.ndarray,
    disk_w: np.ndarray,
    *,
    knee: float = 0.30,
    compress: float = 1.35,
    peak_percentile: float = 98.5,
    peak_target: float = 0.36,
    plane_luma_cap: float = 0.42,
) -> np.ndarray:
    """Last linear-HDR pass before tone map: shoulder, global scale, hard plane luma cap."""
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    h, w_img = int(canvas.shape[0]), int(canvas.shape[1])
    if dw.shape != (h, w_img):
        if dw.ndim == 2 and dw.shape[1] == 1:
            dw = np.broadcast_to(dw, (h, w_img))
        elif dw.size == h:
            dw = np.broadcast_to(dw.reshape(h, 1), (h, w_img))
    gate = np.clip(dw, 0.0, 1.0)
    lin = np.maximum(np.asarray(canvas, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    cap = float(np.clip(plane_luma_cap, 0.28, 0.72))
    plane = np.clip(dw**1.08, 0.0, 1.0)
    band = plane > 0.2
    sample = lu[band] if bool(np.any(band)) else lu.ravel()
    peak = float(np.percentile(sample, float(np.clip(peak_percentile, 96.0, 99.6))))
    p95 = float(np.percentile(sample, 95.0)) if sample.size > 32 else peak
    tgt = float(np.clip(peak_target, 0.32, 0.62))
    # When the plane is already dim, full compress + headroom flattens it to uniform fog.
    if p95 > 0.52:
        out = _compress_band_highlights(
            canvas,
            gate,
            knee=knee,
            compress=compress,
            disk_w=dw,
        )
        lin = np.maximum(np.asarray(out, dtype=np.float64), 0.0)
        lu = rec709_luma(lin)
        if peak > tgt + 1e-6:
            hot = np.clip((lu - tgt * 0.92) / max(peak - tgt * 0.92, 1e-6), 0.0, 1.0)
            scale = np.where(lu > tgt, tgt / np.maximum(lu, 1e-8), 1.0)
            eff = 1.0 - (hot * plane)[..., np.newaxis] * (1.0 - scale[..., np.newaxis])
            lin = np.maximum(0.0, lin * eff)
    hot_only = np.clip((lu - cap * 0.92) / max(cap * 0.08, 1e-6), 0.0, 1.0)
    scale = np.where(lu > cap, cap / np.maximum(lu, 1e-8), 1.0)
    eff = 1.0 - (hot_only * plane)[..., np.newaxis] * (1.0 - scale[..., np.newaxis])
    out = np.maximum(0.0, lin * eff).astype(np.float64)
    _fin = _dbg_band_stats(out, disk_w)
    _lu_out = rec709_luma(out)
    if bool(np.any(band)):
        _fin["band_std"] = float(np.std(_lu_out[band]))
    # #region agent log
    _dbg_log(
        "F",
        "generator.py:finalize_band_linear_hdr",
        "linear HDR finalize",
        {
            "peak_target": peak_target,
            "plane_luma_cap": plane_luma_cap,
            "peak_in": peak,
            **_fin,
        },
    )
    # #endregion
    return out.astype(np.float64)


def _apply_band_ism_dominance(
    canvas: np.ndarray,
    *,
    disk_w: np.ndarray,
    neb_luma: np.ndarray,
    galactic: GalacticMorphology | None,
    dominance: float,
    periodic_x: bool,
    disk_chroma: np.ndarray | None = None,
    chroma_lock: float = 0.0,
    extinction: np.ndarray | None = None,
    preserve_detail: bool = False,
) -> np.ndarray:
    """Floor diffuse band radiance in linear HDR so ISM reads above point stars."""
    dom = float(np.clip(dominance, 0.0, 2.0))
    if dom < 1e-6:
        return canvas
    h, w_img = int(canvas.shape[0]), int(canvas.shape[1])
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    if dw.ndim == 1:
        dw = dw[:, None]
    if dw.shape != (h, w_img):
        dw = np.broadcast_to(dw, (h, w_img))
    nl = np.clip(np.asarray(neb_luma, dtype=np.float64), 0.0, 1.0)
    gate = _structure_host_gate(galactic, h, w_img)
    gate = np.clip(dw * gate, 0.0, 1.0)
    lu_now = rec709_luma(np.maximum(np.asarray(canvas, dtype=np.float64), 0.0))
    band = gate > 0.22
    if bool(np.any(band)):
        med = float(np.median(lu_now[band]))
        if med > 0.20:
            dom *= float(np.clip(1.0 - (med - 0.20) / 0.45, 0.0, 1.0))
    if dom < 1e-6:
        return canvas
    if preserve_detail:
        body = np.clip(nl * gate, 0.0, 1.0) ** 0.98
    else:
        body = _blur_separable_xy(nl * gate, passes=3, periodic_x=periodic_x)
        body = _blur_x_only(body, passes=1, periodic_x=periodic_x)
        body = np.clip(body**0.76, 0.0, 1.0)
    if galactic is not None:
        sf = np.clip(galactic.star_formation, 0.0, 1.0)
        body = np.clip(body * (0.62 + 0.38 * sf), 0.0, 1.0)
        dust = np.clip(galactic.dust_absorption, 0.0, 1.0)
        dust_b = _blur_separable_xy(dust * gate, passes=2, periodic_x=periodic_x)
        body = np.clip(body * (1.0 - 0.42 * dust_b), 0.0, 1.0)
    lu = rec709_luma(canvas)
    target = (0.05 + 0.22 * body) * dom
    lift_cap = 0.14 if preserve_detail else 0.24
    deficit = np.clip((target - lu) * gate, 0.0, lift_cap)
    if extinction is not None:
        rel_clear = band_relative_clearance(extinction, disk_w, min_clear=0.14, power=1.02)
        deficit = deficit * np.clip(0.14 + 0.86 * rel_clear, 0.14, 1.0)
    warm = ism_lift_rgb(chroma_lock=chroma_lock, disk_chroma=disk_chroma)
    rgb = deficit[..., np.newaxis] * warm
    screen = np.maximum(canvas, 0.0)
    blend = np.clip(deficit * 0.72, 0.0, 0.78)[..., np.newaxis]
    out = screen + rgb * (1.0 - screen * 0.42) * blend
    return np.maximum(0.0, out)


def _apply_band_dark_patches(
    canvas: np.ndarray,
    ext_paint: np.ndarray,
    disk_w: np.ndarray,
    galactic: GalacticMorphology | None,
    *,
    strength: float,
    periodic_x: bool,
    morph_primary: bool = False,
) -> np.ndarray:
    """Interrupt the luminous band with extinction-keyed dark patches (not a smooth glow)."""
    s = float(np.clip(strength, 0.0, 1.2))
    if s < 1e-6:
        return canvas
    ext = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    gate = _structure_host_gate(galactic, canvas.shape[0], canvas.shape[1])
    gate = np.clip(dw * gate, 0.0, 1.0)
    rel_clear = band_relative_clearance(ext, disk_w, min_clear=0.10, power=1.0)
    dark = np.clip(1.0 - rel_clear, 0.0, 1.0) ** 1.28
    if galactic is not None and not morph_primary:
        void_w = np.clip(galactic.void_mask, 0.0, 1.0)
        dust_a = np.clip(galactic.dust_absorption, 0.0, 1.0)
        brutal = np.clip(galactic.brutal_erasure_mask, 0.0, 1.0)
        dark = np.maximum(dark, void_w * void_w * 0.42)
        dark = np.maximum(dark, dust_a * dust_a * 0.30)
        dark = np.maximum(dark, brutal * brutal * 0.28)
    dark_sm = _blur_separable_xy(dark * gate, passes=2, periodic_x=periodic_x)
    dark = np.clip(dark + (dark - dark_sm) * 0.28, 0.0, 1.0) ** 1.08
    carve = np.clip(dark * gate * s, 0.0, 0.78)
    floor = np.clip(1.0 - carve * 0.92, 0.06, 1.0)
    out = np.maximum(0.0, canvas * floor[..., np.newaxis])
    # #region agent log
    _cr = int(canvas.shape[0] // 2)
    _dbg_log(
        "B",
        "generator.py:_apply_band_dark_patches",
        "dark patch floor on center row",
        {
            "strength": s,
            "rel_clear_band_p50": float(np.percentile(rel_clear[gate > 0.18], 50))
            if bool(np.any(gate > 0.18))
            else float(np.percentile(rel_clear, 50)),
            "floor_row_min": float(np.min(floor[_cr, :])),
            "floor_row_mean": float(np.mean(floor[_cr, :])),
            "carve_row_max": float(np.max(carve[_cr, :])),
            **_dbg_band_stats(out, disk_w, center_row=_cr),
        },
    )
    # #endregion
    return out


def _attenuate_stars_for_plane(
    stars: np.ndarray,
    disk_w: np.ndarray,
    *,
    plane_scale: float,
) -> np.ndarray:
    """Lower star radiance where the disk is bright so nebula continuum can dominate."""
    ps = float(np.clip(plane_scale, 0.15, 1.0))
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    atten = np.clip(ps + (1.0 - ps) * (1.0 - dw), ps, 1.0)
    return np.maximum(0.0, stars * atten[..., np.newaxis])


def _empty_star_stats() -> dict[str, dict[str, int]]:
    return {
        "color_counts": {n: 0 for n in STAR_COLOR_NAMES},
        "size_counts": {n: 0 for n in STAR_SIZE_NAMES},
    }


def _merge_star_stats(a: dict[str, dict[str, int]], b: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    out = _empty_star_stats()
    for name in STAR_COLOR_NAMES:
        out["color_counts"][name] = a["color_counts"].get(name, 0) + b["color_counts"].get(name, 0)
    for name in STAR_SIZE_NAMES:
        out["size_counts"][name] = a["size_counts"].get(name, 0) + b["size_counts"].get(name, 0)
    return out


def _paint_asymmetric_halo(
    img: np.ndarray,
    x: int,
    y: int,
    halo_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float,
) -> None:
    """Small anisotropic halo around a sharp star core."""
    h, w, _ = img.shape
    if h <= 0 or w <= 0:
        return
    rx = int(rng.integers(6, 11))
    ry = int(rng.integers(3, 7))
    dx = int(rng.integers(-3, 4))
    dy = int(rng.integers(-2, 3))
    lx0 = max(0, x - rx)
    lx1 = min(w, x + rx + 1)
    ly0 = max(0, y - ry)
    ly1 = min(h, y + ry + 1)
    if lx0 >= lx1 or ly0 >= ly1:
        return
    xx = np.arange(lx0, lx1, dtype=np.float64)[None, :]
    yy = np.arange(ly0, ly1, dtype=np.float64)[:, None]
    sx = float(rng.uniform(2.4, 4.6))
    sy = float(rng.uniform(1.3, 2.8))
    core = np.exp(-(((xx - (x + dx)) ** 2) / (sx**2) + ((yy - (y + dy)) ** 2) / (sy**2)))
    tail = np.exp(-(((xx - (x - dx * 0.7)) ** 2) / ((sx * 1.75) ** 2) + ((yy - (y - dy * 0.7)) ** 2) / ((sy * 1.55) ** 2)))
    halo = np.clip(core * 0.72 + tail * 0.28, 0.0, 1.0) * float(np.clip(strength, 0.0, 1.0))
    img[ly0:ly1, lx0:lx1, :] = np.maximum(
        img[ly0:ly1, lx0:lx1, :] + halo[..., None] * halo_rgb[None, None, :],
        0.0,
    )


def _add_stars_from_catalog(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: RenderConfig,
    catalog: dict[str, np.ndarray],
    *,
    foreground_layer: bool,
    galaxy_disk_cool_stars: bool = False,
    point_disk_stars: bool = False,
    plane_psf_elongation: bool = False,
    cluster_layer: bool = False,
    mid_layer: bool = False,
    extinction_for_stars: np.ndarray | None = None,
    disk_w: np.ndarray | None = None,
    galactic_structure: GalacticMorphology | None = None,
    progress_cb: Callable[[float], None] | None = None,
    trail_subframes: int = 1,
    trail_step_px: float = 0.0,
    trail_angle_rad: float | None = None,
    population: Literal["cosmic", "halo", "galactic"] = "galactic",
) -> dict[str, dict[str, int]]:
    xs, ys = sph_to_equirect_xy(catalog["lon"], catalog["lat"], cfg.width, cfg.height)
    xf_coords, yf_coords = sph_to_equirect_xy_float(catalog["lon"], catalog["lat"], cfg.width, cfg.height)
    has_bv = "bv" in catalog
    n = max(1, int(xs.shape[0]))
    if progress_cb is not None:
        progress_cb(0.0)
    next_report = 0.05
    psf_tune = _psf_tuning_from_features(cfg.features)
    redden_s = float(cfg.features.star_reddening_strength)
    bulge_s = float(cfg.features.bulge_warmth_strength)
    spectral_stars = bool(cfg.features.use_spectral_teffective)
    spike_budget = int(cfg.features.spike_max_count)
    spikes_used = 0
    cx = cfg.width * 0.5
    cy = cfg.height * 0.5
    drop_max = float(cfg.features.disk_star_density_dropout)
    use_structure_split = (
        population == "galactic"
        and galactic_structure is not None
        and cfg.features.galaxy_view
        and not foreground_layer
        and not cluster_layer
        and drop_max > 1e-6
    )
    deposit_primary = bool(cfg.features.unresolved_deposit_primary)
    resolved_skip = catalog.get("_resolved_skip")
    spike_flux_cut = float("inf")
    if "phot_mag" in catalog and cfg.features.galaxy_view and cfg.features.photoreal_stars:
        mags = catalog["phot_mag"].astype(np.float64)
        ref = float(cfg.features.magnitude_ref_mag)
        fluxes = np.clip(10.0 ** (-0.4 * (mags - ref)), 1e-8, 1e7)
        spike_flux_cut = float(np.percentile(fluxes, 99.95))
    dust_vis_arr = catalog.get("dust_visibility")
    for i in range(xs.shape[0]):
        yi_px = int(np.clip(int(ys[i]), 0, cfg.height - 1))
        xi_px = int(xs[i]) % cfg.width
        if use_structure_split and resolved_skip is not None and bool(resolved_skip[i]):
            if progress_cb is not None:
                frac = float(i + 1) / float(n)
                if frac >= next_report or i + 1 == n:
                    progress_cb(min(frac, 1.0))
                    next_report += 0.05
            continue
        if use_structure_split and not deposit_primary:
            dw_arr = galactic_structure.disk_weight if galactic_structure is not None else disk_w
            dw = float(dw_arr[yi_px, xi_px]) if dw_arr is not None else 0.5
            g_loc = galactic_structure.sample_stellar(yi_px, xi_px)
            d_loc = galactic_structure.sample_dust(yi_px, xi_px)
            rw = galactic_structure.sample_resolve_weight(yi_px, xi_px)
            obl = galactic_structure.sample_obliteration(yi_px, xi_px)
            brutal = galactic_structure.sample_brutal_erasure(yi_px, xi_px)
            mag_early = (
                float(catalog["phot_mag"][i])
                if "phot_mag" in catalog and cfg.features.photoreal_stars
                else None
            )
            p_keep = resolved_keep_probability(
                g=g_loc,
                dust_t=d_loc,
                disk_w=dw,
                mag=mag_early,
                mag_bright=cfg.features.mag_bright_lim,
                mag_faint=cfg.features.mag_faint_lim,
                dropout_strength=drop_max,
                mid_layer=mid_layer,
                resolve_weight=rw,
                obliteration=obl,
                brutal_erasure=brutal,
            )
            if rng.random() >= p_keep:
                if mag_early is not None and cfg.features.photoreal_stars:
                    deferred_flux = float(
                        flux_from_mag(mag_early, cfg.features.magnitude_ref_mag)
                    )
                    if dust_vis_arr is not None:
                        deferred_flux *= float(dust_vis_arr[i])
                    galactic_structure.deposit_unresolved(yi_px, xi_px, deferred_flux)
                if progress_cb is not None:
                    frac = float(i + 1) / float(n)
                    if frac >= next_report or i + 1 == n:
                        progress_cb(min(frac, 1.0))
                        next_report += 0.05
                continue
        photoreal_paint = (
            cfg.features.photoreal_stars
            and ("phot_mag" in catalog)
            and (
                population in ("cosmic", "halo")
                or (
                    cfg.features.galaxy_view
                    and (cluster_layer or mid_layer or (not foreground_layer))
                )
            )
        )
        color_name = STAR_COLOR_NAMES[int(catalog["color_idx"][i])]
        size_name = STAR_SIZE_NAMES[int(catalog["size_idx"][i])]
        radius = size_radius(rng, size_name)

        u_lum = float(rng.random())
        if mid_layer:
            if photoreal_paint:
                lum = 1.0
            elif u_lum < 0.78:
                lum = 0.028 + (u_lum / 0.78) ** 2.2 * 0.58
            else:
                lum = 0.608 + ((u_lum - 0.78) / 0.22) ** 0.88 * 0.72
            lum = float(np.clip(lum * rng.uniform(0.82, 1.38), 0.022, 3.2))
        elif population == "cosmic":
            if photoreal_paint:
                lum = 1.0
            elif u_lum < 0.96:
                lum = 0.003 + (u_lum / 0.96) ** 3.6 * 0.07
            else:
                lum = 0.073 + ((u_lum - 0.96) / 0.04) ** 0.9 * 0.35
            if u_lum > 0.992:
                lum *= float(rng.uniform(3.0, 9.5))
            lum = float(np.clip(lum * 0.72, 0.002, 5.0))
        elif population == "halo":
            if photoreal_paint:
                lum = 1.0
            elif u_lum < 0.84:
                lum = 0.014 + (u_lum / 0.84) ** 2.35 * 0.42
            else:
                lum = 0.434 + ((u_lum - 0.84) / 0.16) ** 0.95 * 0.55
            if u_lum > 0.978 and rng.random() < 0.018:
                lum *= float(rng.uniform(1.4, 3.2))
            lum = float(np.clip(lum * 0.92, 0.010, 2.8))
        elif not foreground_layer and not cluster_layer:
            # Power toward faint magnitudes; heavier tail for rare bright "hero" stars (cinematic).
            if photoreal_paint:
                lum = 1.0
            elif u_lum < 0.86:
                lum = 0.012 + (u_lum / 0.86) ** 3.2 * 0.42
            else:
                lum = 0.432 + ((u_lum - 0.86) / 0.14) ** 0.88 * 0.72
            r_boost = float(rng.random())
            if r_boost < 0.038:
                lum *= float(rng.uniform(2.2, 8.5))
            elif r_boost < 0.058:
                lum *= float(rng.uniform(1.45, 4.2))
            elif r_boost < 0.072:
                lum *= float(rng.uniform(1.15, 2.6))
            lum = float(np.clip(lum, 0.010, 8.0))
        elif foreground_layer:
            if u_lum < 0.93:
                lum = float(rng.uniform(0.88, 1.22))
            else:
                lum = float(rng.uniform(1.38, 2.45))
        else:
            lum = 1.0

        bv_w: float | None = float(catalog["bv"][i]) if has_bv else None
        plane_w = float(np.exp(-((catalog["lat"][i] / 0.36) ** 2)))
        x_n_star = float(xs[i]) / max(cfg.width - 1, 1)
        core_gold = float(
            np.exp(-((catalog["lat"][i] / 0.32) ** 2))
            * np.exp(-(((x_n_star - 0.5) / 0.28) ** 2))
        )
        if (
            cfg.features.galaxy_view
            and population == "galactic"
            and not cluster_layer
            and not foreground_layer
        ):
            if "gold_population" in catalog:
                pop_gold = float(catalog["gold_population"][i])
            elif galactic_structure is not None:
                pop_gold = galactic_structure.sample_gold_population(yi_px, xi_px)
            else:
                pop_gold = 1.0
            core_gold *= float(np.clip(pop_gold, 0.0, 1.0))

        # Large disk stars: gold in core; hot blue only off-plane / rare.
        if has_bv and bv_w is not None and radius >= 6:
            if core_gold > 0.38 and bv_w < 0.75 and rng.random() < 0.72:
                bv_w = float(rng.uniform(0.52, 1.05))
            elif plane_w > 0.42 and bv_w < 0.62 and rng.random() < 0.48:
                bv_w = float(rng.uniform(0.42, 0.98))
            elif plane_w < 0.28 and core_gold < 0.15 and bv_w > -0.02 and rng.random() < 0.14:
                bv_w = float(rng.uniform(-0.18, 0.10))

        if (
            not has_bv
            and radius >= 6
            and color_name != "blue"
            and core_gold < 0.18
            and plane_w < 0.26
            and rng.random() < 0.12
        ):
            color_name = "blue"

        # Red / cool giants: usually small on screen.
        if has_bv and bv_w is not None and bv_w >= 0.95:
            if radius > 3 and rng.random() < 0.9:
                radius = int(rng.integers(1, 4))
            else:
                radius = min(radius, 4)
        elif color_name == "red":
            if radius > 3 and rng.random() < 0.9:
                radius = int(rng.integers(1, 4))
            else:
                radius = min(radius, 4)

        # Yellow–orange: uncommon as large disks.
        if has_bv and bv_w is not None and 0.42 <= bv_w < 0.95 and radius > 3 and rng.random() < 0.88:
            radius = int(rng.integers(1, 4))
        elif color_name == "yellow" and radius > 3 and rng.random() < 0.88:
            radius = int(rng.integers(1, 4))

        if point_disk_stars and not foreground_layer and not cluster_layer and cfg.features.galaxy_view:
            plane_gate = float(np.exp(-((catalog["lat"][i] / 0.34) ** 2)))
            if plane_gate > 0.30:
                if rng.random() < 0.22:
                    radius = 1
                elif radius == 1 and (not photoreal_paint) and lum > 0.45 and rng.random() < 0.52:
                    radius = 2

        if cluster_layer:
            radius = 1 if rng.random() < 0.90 else 2

        chroma_w = 1.0
        if photoreal_paint and "phot_mag" in catalog:
            chroma_w = star_chromatic_perturb_weight(
                float(catalog["phot_mag"][i]),
                mag_bright=cfg.features.mag_bright_lim,
                mag_faint=cfg.features.mag_faint_lim,
            )
        elif (
            not foreground_layer
            and not cluster_layer
            and cfg.features.galaxy_view
            and not photoreal_paint
        ):
            chroma_w = float(np.clip((lum - 0.12) / 0.55, 0.0, 1.0) ** 2.2)

        star_jitter = catalog["jitter"][i] * (chroma_w**3)
        if "teff" in catalog and cfg.features.use_spectral_teffective:
            teff_paint = float(catalog["teff"][i])
            if cfg.features.galaxy_view and not cluster_layer and core_gold > 0.08:
                teff_paint = warm_teffective_for_core_bulge(teff_paint, core_gold, rng)
            color = star_rgb_from_teffective(teff_paint, star_jitter, camera=True)
        elif has_bv and bv_w is not None:
            color = rgb_from_bv(bv_w, star_jitter)
            if bv_w >= 1.0:
                color = color * 0.52
        else:
            color = star_color(color_name, star_jitter)
            if color_name == "red":
                color = color * 0.5
        # Hot stars: gold in core; cool blue only off-plane and rare.
        if not cluster_layer and cfg.features.galaxy_view and photoreal_paint:
            if has_bv and bv_w is not None and bv_w <= 0.12 and rng.random() < 0.14:
                teff_hot = float(catalog["teff"][i]) if "teff" in catalog else 0.0
                if spectral_stars and teff_hot > 9000.0:
                    if core_gold > 0.22:
                        gold_hot = np.array([1.16, 0.96, 0.72], dtype=np.float64)
                        mix = float(np.clip(0.22 + 0.55 * core_gold, 0.22, 0.72))
                        color = np.clip(color * (1.0 - mix) + gold_hot * mix, 0.0, 1.0)
                    elif plane_w < 0.32 and rng.random() < 0.35:
                        cool_hot = np.array([0.86, 0.93, 1.14], dtype=np.float64)
                        color = np.clip(color * 0.88 + cool_hot * 0.12, 0.0, 1.0)
                else:
                    warm_hot = np.array([0.98, 0.94, 0.86], dtype=np.float64)
                    color = np.clip(color * 0.88 + warm_hot * 0.12, 0.0, 1.0)
        if cluster_layer:
            color = color * rng.uniform(0.58, 0.98)
        elif foreground_layer:
            color = color * lum * float(rng.uniform(0.78, 1.15))
        elif photoreal_paint:
            color = color * float(rng.uniform(0.88, 1.12))
        else:
            color = color * lum * float(rng.uniform(0.88, 1.12))
        if not cluster_layer and cfg.features.galaxy_view and not foreground_layer and chroma_w > 1e-4:
            hue = float(rng.uniform(-0.12, 0.12))
            if spectral_stars:
                hue_scale = 0.10
                blue_pull = 0.34 * (1.0 - 0.62 * plane_w)
            else:
                hue_scale = 0.22 if has_bv else 0.55
                blue_pull = 0.56 * (1.0 - 0.55 * plane_w)
            hue_scale *= chroma_w
            gold_pull = (0.22 + 0.38 * core_gold) * plane_w * chroma_w
            color = color * np.array(
                [
                    1.0 + hue * hue_scale + gold_pull * max(hue, 0.0),
                    1.0 - 0.46 * hue * hue_scale,
                    1.0 - blue_pull * hue * hue_scale - gold_pull * max(-hue, 0.0),
                ],
                dtype=np.float64,
            )
            color = np.clip(color, 0.0, 1.0)
        allow_rare_tint = chroma_w > 0.6
        if photoreal_paint and "phot_mag" in catalog:
            allow_rare_tint = float(catalog["phot_mag"][i]) < 9.0 and chroma_w > 0.6
        if (
            not cluster_layer
            and cfg.features.galaxy_view
            and not foreground_layer
            and radius <= 3
            and allow_rare_tint
        ):
            uo = rng.random()
            if uo < 0.0012 and core_gold < 0.20:
                color = np.clip(
                    np.array([0.78, 0.86, 0.98], dtype=np.float64) * float(rng.uniform(0.92, 1.06)),
                    0.0,
                    1.0,
                )
            elif uo < 0.0044:
                color = np.clip(
                    np.array([0.64, 0.28, 0.095], dtype=np.float64) * float(rng.uniform(0.88, 1.06)),
                    0.0,
                    1.0,
                )
        warm_disk_star = (not has_bv and color_name in ("yellow", "red")) or (
            has_bv and bv_w is not None and bv_w >= 0.42
        )
        if warm_disk_star and not foreground_layer and not cluster_layer and cfg.features.galaxy_view:
            plane_d = float(np.exp(-((catalog["lat"][i] / 0.38) ** 2)))
            color = color * (1.0 - 0.22 * plane_d * rng.uniform(0.88, 1.0))
        if cfg.features.depth and not foreground_layer:
            lat01 = np.clip((catalog["lat"][i] + np.pi / 2.0) / np.pi, 0.0, 1.0)
            # Smooth rolloff avoids a hard brightness contour line.
            depth_scale = 0.22 + 0.78 * (lat01**0.85)
            color = color * depth_scale
        if galaxy_disk_cool_stars and cfg.features.galaxy_view and not cluster_layer and not foreground_layer:
            plane = float(np.exp(-((catalog["lat"][i] / 0.34) ** 2)))
            cool = np.array([0.98, 0.99, 1.01], dtype=np.float64)
            color = color * ((1.0 - 0.04 * plane) * 1.0 + 0.04 * plane * cool)
        if not foreground_layer and cfg.features.galaxy_view and not cluster_layer:
            plane_star = float(np.exp(-((catalog["lat"][i] / 0.36) ** 2)))
            x_n0 = float(xs[i]) / max(cfg.width - 1, 1)
            core_pre = float(
                np.exp(-((catalog["lat"][i] / 0.26) ** 2)) * np.exp(-(((x_n0 - 0.5) / 0.22) ** 2))
            )
            halo_w = 1.0 - plane_star
            color = color * (1.0 - 0.08 * halo_w)
            neutral_disk = np.array([0.98, 0.98, 0.99], dtype=np.float64)
            if spectral_stars:
                warm_disk = np.array([1.04, 0.99, 0.92], dtype=np.float64)
                halo_mix_strength = 0.22
            else:
                warm_disk = np.array([1.05, 0.98, 0.90], dtype=np.float64)
                halo_mix_strength = 0.38
            halo_mix = neutral_disk * (1.0 - 0.55 * core_pre) + warm_disk * (0.55 * core_pre)
            color = color * (halo_w * halo_mix_strength * halo_mix + (1.0 - halo_w * halo_mix_strength))
            color = np.clip(color, 0.0, 1.0)
        if not foreground_layer and not cluster_layer and cfg.features.galaxy_view:
            gold_scale = 1.42 if (photoreal_paint and spectral_stars) else 1.0
            if spectral_stars and "teff" in catalog:
                teff_i = float(catalog["teff"][i])
                if teff_i > 9000.0 and core_gold > 0.30:
                    gold_scale *= 1.38
                elif teff_i > 11000.0 and core_gold < 0.12:
                    gold_scale *= 0.40
            warm_core = np.array([0.32, 0.21, 0.042], dtype=np.float64)
            color = np.clip(color + warm_core * (core_gold * 3.6 * gold_scale), 0.0, 1.0)
            gw = float(np.clip(1.08 * core_gold * gold_scale, 0.0, 0.96))
            gold_shift = np.array([1.24, 0.92, 0.58], dtype=np.float64)
            color = np.clip(color * ((1.0 - gw) + gw * gold_shift), 0.0, 1.0)
            if core_gold > 0.08:
                wc = float(np.clip((core_gold - 0.08) / 0.40, 0.0, 1.0)) * gold_scale
                color = np.clip(
                    color * (1.0 + wc * np.array([0.14, 0.018, -0.16], dtype=np.float64)),
                    0.0,
                    1.0,
                )
            if core_gold > 0.38:
                mg = float(np.clip((core_gold - 0.38) / 0.52, 0.0, 1.0)) * gold_scale
                gold_ref = np.array([1.22, 0.86, 0.48], dtype=np.float64)
                color = np.clip(color * (1.0 - mg * 0.88) + gold_ref * (mg * 0.88), 0.0, 1.0)
        if foreground_layer and cfg.features.galaxy_view and not cluster_layer:
            x_n = float(xs[i]) / max(cfg.width - 1, 1)
            bulge_proxy = float(
                np.exp(-((catalog["lat"][i] / 0.26) ** 2)) * np.exp(-(((x_n - 0.5) / 0.20) ** 2))
            )
            if bulge_proxy > 0.48 and radius > 2 and rng.random() < 0.68:
                radius = min(radius, int(rng.choice([2, 2, 3])))
            if bulge_proxy > 0.58 and radius > 1 and rng.random() < 0.42:
                radius = 1
        if not foreground_layer and cfg.features.galaxy_view and not cluster_layer:
            z_disk = float(np.exp(-((catalog["lat"][i] / 0.252) ** 2)) ** 1.12)
            z_floor, z_span = (0.88, 0.12) if spectral_stars else (0.80, 0.20)
            color = color * (z_floor + z_span * z_disk)
        eff_bulge_s = bulge_s * (0.45 if spectral_stars else 1.0)
        if (
            photoreal_paint
            and eff_bulge_s > 1e-6
            and not foreground_layer
            and not cluster_layer
        ):
            bw = _bulge_warmth_scalar(float(catalog["lon"][i]), float(catalog["lat"][i]), cfg.width)
            if bw > 0.08:
                warm = (
                    np.array([1.08, 1.02, 0.94], dtype=np.float64)
                    if spectral_stars
                    else np.array([1.14, 1.02, 0.88], dtype=np.float64)
                )
                color = np.clip(
                    color * ((1.0 - bw * eff_bulge_s) + warm * (bw * eff_bulge_s)), 0.0, 1.0
                )
        plat: float | None = (
            float(catalog["lat"][i]) if plane_psf_elongation and cfg.features.galaxy_view else None
        )
        jitter = float(cfg.features.star_position_jitter_px)
        xi_f = float(xf_coords[i]) + float(rng.uniform(-jitter, jitter))
        yi_f = float(yf_coords[i]) + float(rng.uniform(-jitter, jitter))
        xi = int(np.floor(xi_f))
        yi = int(np.floor(yi_f))
        spx = xi_f - xi
        spy = yi_f - yi
        if (
            extinction_for_stars is not None
            and redden_s > 1e-6
            and not photoreal_paint
        ):
            et = _sample_ext_scalar_bilinear(
                extinction_for_stars,
                float(xf_coords[i]),
                float(yf_coords[i]),
                periodic_x=True,
            )
            a_v = extinction_from_transmission(et, strength=redden_s)
            color = extinction_redden(np.maximum(color, 0.0), a_v, rv=cfg.features.extinction_r_v)
        if photoreal_paint:
            mag = float(catalog["phot_mag"][i])
            flux_val = flux_from_mag(mag, cfg.features.magnitude_ref_mag)
            spectral = np.maximum(color.astype(np.float64), 0.0)
            if extinction_for_stars is not None and redden_s > 1e-6:
                et = _sample_ext_scalar_bilinear(
                    extinction_for_stars,
                    float(xf_coords[i]),
                    float(yf_coords[i]),
                    periodic_x=True,
                )
                a_v = extinction_from_transmission(et, strength=redden_s)
                spectral = extinction_redden(spectral, a_v, rv=cfg.features.extinction_r_v)
                flux_val *= ccm89_v_band_transmission(a_v)
            if eff_bulge_s > 1e-6:
                bw = _bulge_warmth_scalar(float(catalog["lon"][i]), float(catalog["lat"][i]), cfg.width)
                flux_val *= 1.0 + 0.22 * bw * eff_bulge_s
            if not foreground_layer and not cluster_layer:
                span = float(cfg.features.mag_faint_lim - cfg.features.mag_bright_lim)
                faint_u = float(np.clip((mag - cfg.features.mag_bright_lim) / max(span, 1e-6), 0.0, 1.0))
                flux_val *= 1.0 - 0.14 * faint_u
            if not foreground_layer:
                scatter = float(cfg.features.star_flux_scatter_sigma)
                if scatter > 1e-6 and photoreal_paint:
                    flux_val *= 1.0 + float(rng.normal(0.0, scatter))
                elif photoreal_paint:
                    flux_val *= float(rng.uniform(0.92, 1.08))
                else:
                    flux_val *= float(rng.uniform(0.86, 1.16))
            if mid_layer:
                flux_val *= float(rng.uniform(0.94, 1.24))
            if cluster_layer:
                flux_val *= float(rng.uniform(1.22, 1.68))
            if dust_vis_arr is not None:
                flux_val *= float(dust_vis_arr[i])
            if galactic_structure is not None:
                flux_val *= 0.88 + 0.12 * galactic_structure.sample_psf_env(yi_px, xi_px)
            field_ang = math.atan2(float(yf_coords[i] - cy), float(xf_coords[i] - cx))
            do_spike = spikes_used < spike_budget and flux_val >= spike_flux_cut
            if do_spike:
                spikes_used += 1
            star_bv = float(catalog["bv"][i]) if "bv" in catalog else None
            star_teff = float(catalog["teff"][i]) if "teff" in catalog else None
            if foreground_layer:
                psf_layer: Literal["background", "mid", "foreground"] = "foreground"
            elif mid_layer:
                psf_layer = "mid"
            else:
                psf_layer = "background"
            psf_variety: StarPsfVariety | None = None
            if cfg.features.star_psf_variety:
                ext_t = 0.0
                if extinction_for_stars is not None:
                    ext_t = _sample_ext_scalar_bilinear(
                        extinction_for_stars,
                        float(xf_coords[i]),
                        float(yf_coords[i]),
                        periodic_x=True,
                    )
                local_d = 0.0
                psf_env = 1.0
                if galactic_structure is not None:
                    local_d = galactic_structure.sample_stellar(yi_px, xi_px)
                    psf_env = galactic_structure.sample_psf_env(yi_px, xi_px)
                elif disk_w is not None:
                    local_d = float(disk_w[yi_px, xi_px])
                psf_variety = sample_star_psf_variety(
                    rng,
                    mag,
                    flux_val,
                    teff_k=star_teff,
                    bv=star_bv,
                    extinction_t=ext_t,
                    local_density=local_d,
                    galactic_lat_rad=plat,
                    psf_environment=psf_env,
                    layer=psf_layer,
                    tuning=psf_tune,
                )
            n_trail = max(1, int(trail_subframes))
            use_trail = (
                n_trail > 1
                and trail_angle_rad is not None
                and trail_step_px > 1e-6
                and not foreground_layer
            )
            trail_cos = math.cos(trail_angle_rad) if use_trail and trail_angle_rad is not None else 0.0
            trail_sin = math.sin(trail_angle_rad) if use_trail and trail_angle_rad is not None else 0.0
            stamp_flux = flux_val / float(n_trail) if use_trail else flux_val
            for ti in range(n_trail):
                t_off = 0.0
                if use_trail and n_trail > 1:
                    t_off = (float(ti) / float(n_trail - 1) - 0.5) * 2.0
                tdx = trail_cos * trail_step_px * t_off
                tdy = trail_sin * trail_step_px * t_off
                stamp_star_psf(
                    img,
                    yi,
                    xi,
                    spectral,
                    stamp_flux,
                    mag,
                    galactic_lat_rad=plat,
                    rng=rng,
                    tuning=psf_tune,
                    periodic_x=True,
                    subpixel_dx=spx + tdx,
                    subpixel_dy=spy + tdy,
                    field_angle_rad=field_ang,
                    add_spikes=do_spike and ti == n_trail // 2,
                    bv=star_bv,
                    teff_k=star_teff,
                    layer=psf_layer,
                    variety=psf_variety,
                )
        else:
            paint_star(img, int(np.round(xi_f)), int(np.round(yi_f)), radius, color, rng, galactic_lat=plat)
        if (
            cfg.features.galaxy_view
            and not foreground_layer
            and not cluster_layer
            and core_gold > 0.18
            and ((has_bv and bv_w is not None and bv_w >= 0.28) or (not has_bv and color_name in ("yellow", "red")))
        ):
            halo_strength = float(np.clip((core_gold - 0.18) / 0.52, 0.0, 1.0)) * float(rng.uniform(0.07, 0.18))
            # Keep halo warmer but dull/desaturated so the bright star core remains the focal point.
            halo_rgb = np.array([0.16, 0.145, 0.13], dtype=np.float64) * float(rng.uniform(0.90, 1.05))
            _paint_asymmetric_halo(img, xi, yi, halo_rgb, rng, strength=halo_strength)
        if progress_cb is not None:
            frac = float(i + 1) / float(n)
            if frac >= next_report or i + 1 == n:
                progress_cb(min(frac, 1.0))
                next_report += 0.05
    np.maximum(img, 0.0, out=img)
    stats = catalog_stats(catalog)
    return {"color_counts": stats.color_counts, "size_counts": stats.size_counts}


def _save_image(
    img: np.ndarray,
    path: Path,
    fmt: str,
    quality: int,
    *,
    dither_strength: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = apply_blue_noise_dither_u8(img, strength=dither_strength)
    pil = Image.fromarray(u8, mode="RGB")
    if fmt == "jpg":
        pil.save(path, format="JPEG", quality=quality)
    else:
        pil.save(path, format="PNG")


def _enforce_horizontal_wrap(img: np.ndarray, *, seam_width: int | None = None) -> np.ndarray:
    """Only close the endpoint columns; avoid wide seam strips that roll into visible bands."""
    _ = seam_width  # Legacy arg kept for API compatibility.
    _, w, _ = img.shape
    if w < 2:
        return img
    seam = 0.5 * (img[:, 0, :] + img[:, -1, :])
    img[:, 0, :] = seam
    img[:, -1, :] = seam
    return np.clip(img, 0.0, 1.0)


def render_single(
    cfg: RenderConfig,
    generation_index: int,
    on_pass_complete: Callable[[str, float], None] | Callable[[str], None] | Callable[[], None] | None = None,
) -> tuple[dict[str, Path], dict[str, dict[str, int]]]:
    def _notify_stage(stage: str, progress: float = 1.0) -> None:
        if on_pass_complete is None:
            return
        try:
            on_pass_complete(stage, float(np.clip(progress, 0.0, 1.0)))  # type: ignore[misc]
        except TypeError:
            try:
                on_pass_complete(stage)  # type: ignore[misc]
            except TypeError:
                on_pass_complete()  # type: ignore[operator]

    seed = (cfg.seed or 0) + generation_index
    seed_seq = np.random.SeedSequence(seed)
    (
        rng_bg,
        rng_stars_bg,
        rng_clusters,
        rng_stars_fg,
        rng_nebula,
        rng_post,
        rng_chroma,
        rng_cosmic,
        rng_halo,
    ) = [np.random.default_rng(s) for s in seed_seq.spawn(9)]
    rng_stars_mid = np.random.default_rng(int(rng_stars_bg.integers(0, 2**63)))
    wrap_lon_blur_x = bool(cfg.wrap_safe)
    _notify_stage("background", 0.0)
    canvas = ensure_hdr(
        _background_plane(
            rng=rng_bg,
            height=cfg.height,
            width=cfg.width,
            enabled=cfg.features.background_gradient,
            black_background=cfg.features.black_background,
            texture_strength=cfg.features.background_texture_strength,
        )
    )
    _notify_stage("background", 1.0)
    # #region agent log
    import os as _os_dbg

    if _os_dbg.environ.get("STK_ABLATION"):
        _yy_bg = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float64)[:, None]
        _dw_bg = np.exp(-((_yy_bg**2) / 0.22))
        _dbg_hist("canvas_after_background", canvas, _dw_bg, hypothesis_id="BG")
    # #endregion

    stats = _empty_star_stats()
    stars_cosmic = np.zeros_like(canvas, dtype=HDR_DTYPE)
    stars_halo = np.zeros_like(canvas, dtype=HDR_DTYPE)
    stars_bg = np.zeros_like(canvas, dtype=HDR_DTYPE)
    stars_mid = np.zeros_like(canvas, dtype=HDR_DTYPE)
    stars_fg = np.zeros_like(canvas, dtype=HDR_DTYPE)
    galactic: GalacticMorphology | None = None
    if cfg.features.galaxy_view and cfg.features.stars:
        _pk = _placement_kwargs(cfg.features)
        gen_phase = float(generation_index) * 0.173
        perturb_scale = float(cfg.features.morphology_seed_perturb_scale) * (
            1.0 + 0.07 * (generation_index % 7)
        )
        galactic = build_galactic_morphology(
            cfg.width,
            cfg.height,
            rng_stars_bg,
            disk_height=float(_pk["disk_height"]),
            halo_fraction=float(_pk["halo_fraction"]),
            halo_power=float(_pk["halo_power"]),
            band_lat_sigma=float(_pk["band_lat_sigma"]),
            band_rotation_deg=float(_pk["band_rotation_deg"]),
            band_curvature_amp=float(_pk["band_curvature_amp"]),
            band_thickness_asymmetry=float(cfg.features.band_thickness_asymmetry),
            disk_mesoscale_thickness_strength=float(
                cfg.features.disk_mesoscale_thickness_strength
            ),
            cluster_strength=float(_pk["cluster_strength"]),
            turbulence_strength=float(_pk.get("turbulence_strength", 0.85)),
            void_strength=float(cfg.features.morphology_void_strength),
            scar_strength=float(cfg.features.morphology_scar_strength),
            placement_asymmetry=float(_pk["placement_asymmetry"]),
            drop_strength=float(cfg.features.disk_star_density_dropout),
            discontinuity_cut_strength=float(
                cfg.features.morphology_discontinuity_cut_strength
            ),
            macro_void_count=int(cfg.features.morphology_macro_void_count),
            periodic_x=wrap_lon_blur_x,
            seed_perturb_scale=perturb_scale,
            sf_perturb=float(cfg.features.morphology_sf_perturb),
            dust_perturb=float(cfg.features.morphology_dust_perturb),
            dust_micro_strength=float(cfg.features.morphology_dust_micro_strength),
            cluster_perturb=float(cfg.features.morphology_cluster_perturb),
            local_variance=float(cfg.features.morphology_local_variance),
            obliteration_strength=float(cfg.features.morphology_obliteration_strength),
            regional_chaos=float(cfg.features.morphology_regional_chaos),
            generation_phase=gen_phase,
            vertical_extent_strength=float(cfg.features.structure_vertical_extent),
            structure_host_latitude_scale=float(cfg.features.structure_host_latitude_scale),
            longitude_asymmetry_strength=float(cfg.features.longitude_asymmetry_strength),
            brutal_erasure_strength=float(cfg.features.extinction_brutal_erasure_strength),
            seam_guard_strength=float(cfg.features.seam_guard_strength),
            disaster_peak_count=3,
            gold_population_patchiness=float(cfg.features.stellar_gold_population_patchiness),
        )
        disk_w = galactic.disk_weight
    else:
        disk_w = _galactic_disk_weight(cfg.height)
    if disk_w is not None:
        _dw0 = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
        if _dw0.ndim == 1:
            _dw0 = _dw0[:, None]
        if _dw0.shape != (cfg.height, cfg.width):
            disk_w = np.broadcast_to(_dw0, (cfg.height, cfg.width))
        else:
            disk_w = _dw0
    depth_scene: np.ndarray | None = None
    if cfg.features.galaxy_view and cfg.features.depth_of_field_strength > 1e-6:
        depth_scene = depth_map_from_disk(
            disk_w, cfg.height, cfg.width, rng_post, periodic_x=wrap_lon_blur_x
        )
    density_scale = 1.0 + (0.36 if cfg.features.galaxy_view else 0.0)
    band_boost = 1.03 if cfg.features.galaxy_view else 1.0
    if cfg.features.stars and cfg.features.galaxy_view and cfg.features.cosmic_star_enabled:
        _notify_stage("stars cosmic", 0.0)
        cat_cosmic = sample_isotropic_cosmic_catalog(
            rng_cosmic,
            cfg.width,
            cfg.height,
            density_scale,
            density_scale_mult=cfg.features.cosmic_star_density_scale,
            attach_apparent_mag=cfg.features.photoreal_stars,
            anchor_count=int(cfg.features.cosmic_anchor_count),
        )
        stats_cosmic = _add_stars_from_catalog(
            stars_cosmic,
            rng_cosmic,
            cfg,
            cat_cosmic,
            foreground_layer=False,
            population="cosmic",
            progress_cb=lambda p: _notify_stage("stars cosmic", p),
        )
        stats = _merge_star_stats(stats, stats_cosmic)
        np.multiply(stars_cosmic, 0.78, out=stars_cosmic, casting="unsafe")
        canvas = _canvas_add_linear(canvas, stars_cosmic, galaxy_hdr=True)
        _notify_stage("stars cosmic", 1.0)
    stars_above_nebula = bool(
        cfg.features.galaxy_view
        and cfg.features.stars
        and cfg.features.nebula
        and cfg.nebula_mode == NebulaMode.galaxy_streak
        and (
            cfg.features.split_star_display_grade
            or cfg.features.stars_after_display_grade
        )
    )
    if cfg.features.galaxy_view and cfg.features.unified_linear_grade:
        stars_above_nebula = False
    ism_dom = float(cfg.features.band_ism_dominance) if cfg.features.galaxy_view else 0.0
    radiance_unify = float(cfg.features.disk_radiance_unify_strength) if cfg.features.galaxy_view else 0.0
    photon_unify = float(cfg.features.photon_exposure_unify_strength) if cfg.features.galaxy_view else 0.0
    disk_chroma_ref: np.ndarray | None = None
    disk_exposure = None
    trail_subframes = 1
    trail_step_px = 0.0
    trail_angle_rad: float | None = None
    if (
        cfg.features.galaxy_view
        and cfg.features.long_exposure_look
        and cfg.features.long_exposure_star_trails
        and cfg.features.stars
    ):
        trail_subframes = int(cfg.features.long_exposure_subframes)
        trail_step_px = float(cfg.features.long_exposure_trail_step_px)
        trail_angle_rad = float(rng_post.uniform(0.0, 2.0 * math.pi))
    grade_neb_luma: np.ndarray | None = None
    morph_ism_rgb: np.ndarray | None = None
    morph_ism_luma: np.ndarray | None = None
    morph_ism_layers: object | None = None
    morph_puff_punch: np.ndarray | None = None
    morph_cont_guide: np.ndarray | None = None
    morph_gas_struct: np.ndarray | None = None
    off_band_layer: np.ndarray | None = None
    band_hii_layer: np.ndarray | None = None
    if cfg.features.galaxy_view and cfg.features.nebula:
        off_band_layer = np.zeros((cfg.height, cfg.width, 3), dtype=np.float64)
        band_hii_layer = np.zeros((cfg.height, cfg.width, 3), dtype=np.float64)

    nebula_bundle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    if (
        cfg.features.nebula
        and cfg.nebula_mode == NebulaMode.galaxy_streak
        and cfg.features.stars
    ):
        _notify_stage("nebula prep", 0.0)
        neb0, neb_emit0, dust0, lane0 = generate_nebula(
            rng_nebula,
            cfg.nebula_mode,
            cfg.height,
            cfg.width,
            cfg.nebula_tuning,
            progress_cb=lambda p: _notify_stage("nebula prep", 0.10 + p * 0.75),
            galaxy_features=cfg.features if cfg.features.galaxy_view else None,
        )
        if (
            cfg.features.luminance_map_path is not None
            and cfg.features.catalog_mode == "luminance_overlay"
        ):
            lum_ov = load_luminance_overlay(
                cfg.features.luminance_map_path, cfg.width, cfg.height
            )
            if lum_ov is not None:
                neb0 = np.clip(neb0 * (0.62 + 0.38 * lum_ov[..., np.newaxis]), 0.0, 1.0)
        ext0 = _extinction_from_dust_and_lane(dust0, lane0, cfg)
        _notify_stage("nebula prep", 0.90)
        nebula_bundle = (neb0, neb_emit0, dust0, lane0, ext0)
        _notify_stage("nebula prep", 1.0)

    if cfg.features.stars:
        _notify_stage("stars background prep", 0.0)
        _gal_frac = (
            float(cfg.features.galactic_star_density_fraction)
            if (
                cfg.features.galaxy_view
                and (cfg.features.cosmic_star_enabled or cfg.features.halo_star_enabled)
            )
            else 1.0
        )
        phot_kw = dict(
            attach_apparent_mag=cfg.features.photoreal_stars and cfg.features.galaxy_view,
            band_star_density_scale=(
                cfg.features.band_star_density_scale * _gal_frac
                if cfg.features.galaxy_view
                else 1.0
            ),
            foreground_star_density_scale=cfg.features.foreground_star_density_scale,
            photometry_mag_bright=cfg.features.mag_bright_lim,
            photometry_mag_faint=cfg.features.mag_faint_lim,
            photometry_slope=cfg.features.magnitude_log_slope,
            photometry_ultra_cut=cfg.features.magnitude_ultra_cut,
            max_ultra_bright_stars=int(cfg.features.galactic_anchor_star_count),
            **_placement_kwargs(cfg.features),
        )
        faint_cull_kw = dict(
            mag_faint_floor=cfg.features.resolved_faint_mag_floor,
            dropout_strength=cfg.features.resolved_faint_dropout,
            mag_faint=cfg.features.mag_faint_lim,
            magnitude_ultra_cut=cfg.features.magnitude_ultra_cut,
        )
        if cfg.features.catalog_mode == "stats_seed":
            cat_sub = load_catalog_subset(seed)
            slope_hint, _bv_hint = fit_procedural_stats_from_catalog(cat_sub)
            phot_kw["photometry_slope"] = slope_hint
        ext_for_star_paint = (
            nebula_bundle[4] if (nebula_bundle is not None and phot_kw["attach_apparent_mag"]) else None
        )
        if cfg.features.galaxy_view and cfg.features.halo_star_enabled:
            _notify_stage("stars halo", 0.0)
            cat_halo = sample_halo_star_catalog(
                rng_halo,
                cfg.width,
                cfg.height,
                density_scale,
                density_scale_mult=cfg.features.halo_star_density_scale,
                halo_lat_sigma=float(cfg.features.halo_lat_sigma),
                attach_apparent_mag=bool(phot_kw["attach_apparent_mag"]),
                max_ultra_bright_stars=4,
            )
            stats_halo = _add_stars_from_catalog(
                stars_halo,
                rng_halo,
                cfg,
                cat_halo,
                foreground_layer=False,
                galaxy_disk_cool_stars=True,
                point_disk_stars=False,
                population="halo",
                progress_cb=lambda p: _notify_stage("stars halo", p),
            )
            stats = _merge_star_stats(stats, stats_halo)
            _notify_stage("stars halo", 1.0)
        if galactic is not None and nebula_bundle is not None:
            galactic.merge_nebula_extinction(nebula_bundle[4])
        _notify_stage("stars background prep", 0.08)
        cat_bg = sample_star_catalog(
            rng_stars_bg,
            cfg.width,
            cfg.height,
            density_scale,
            layer="background",
            galactic_band_boost=band_boost,
            latitude_color_bias=True,
            galactic_structure=galactic,
            **phot_kw,
        )
        _notify_stage("stars background prep", 0.42)
        if cfg.features.catalog_mode == "positions":
            cat_sub = load_catalog_subset(seed)
            cat_bg = merge_catalog_positions(cat_sub, cat_bg, rng_stars_bg, cfg.features.catalog_blend)
        _notify_stage("stars background prep", 0.50)
        if nebula_bundle is not None and cfg.features.galaxy_view:
            reroll_stars_in_dark_lanes(
                cat_bg,
                rng_stars_bg,
                cfg.width,
                cfg.height,
                nebula_bundle[4],
                galactic_structure=galactic,
            )
        if galactic is not None and cfg.features.galaxy_view:
            attach_stellar_population_to_catalog(
                cat_bg, galactic, cfg.width, cfg.height, rng_stars_bg
            )
        if cfg.features.galaxy_view and cfg.features.photoreal_stars:
            cat_bg = cull_faint_resolved_stars(
                cat_bg,
                rng_stars_bg,
                galactic_structure=galactic,
                magnitude_ref_mag=cfg.features.magnitude_ref_mag,
                width=cfg.width,
                height=cfg.height,
                **faint_cull_kw,
            )
        if (
            galactic is not None
            and cfg.features.galaxy_view
            and int(cfg.features.galactic_overdensity_star_count) > 0
        ):
            cat_bg = inject_galactic_overdensity_stars(
                cat_bg,
                rng_stars_bg,
                galactic,
                cfg.width,
                cfg.height,
                count=int(cfg.features.galactic_overdensity_star_count),
                attach_apparent_mag=bool(phot_kw.get("attach_apparent_mag", False)),
                mag_bright=float(phot_kw.get("photometry_mag_bright", 9.0)) - 0.8,
                mag_faint=float(phot_kw.get("photometry_mag_faint", 18.0)) * 0.82,
                anchor_mag_bright=4.8,
                anchor_mag_faint=7.6,
            )
            attach_stellar_population_to_catalog(
                cat_bg, galactic, cfg.width, cfg.height, rng_stars_bg
            )
            from starsky_gen.galactic_structure import attach_dust_visibility_to_catalog

            attach_dust_visibility_to_catalog(cat_bg, galactic, cfg.width, cfg.height)
        if (
            galactic is not None
            and cfg.features.unresolved_deposit_primary
            and cfg.features.photoreal_stars
            and float(cfg.features.disk_star_density_dropout) > 1e-6
        ):
            deposit_catalog_unresolved_flux(
                cat_bg,
                galactic,
                rng_stars_bg,
                width=cfg.width,
                height=cfg.height,
                magnitude_ref_mag=cfg.features.magnitude_ref_mag,
                mag_bright=cfg.features.mag_bright_lim,
                mag_faint=cfg.features.mag_faint_lim,
                dropout_strength=float(cfg.features.disk_star_density_dropout),
                mid_layer=False,
            )
        _notify_stage("stars background prep", 1.0)
        _notify_stage("stars background", 0.0)
        _trail_kw = dict(
            trail_subframes=trail_subframes,
            trail_step_px=trail_step_px,
            trail_angle_rad=trail_angle_rad,
        )
        stats = _add_stars_from_catalog(
            stars_bg,
            rng_stars_bg,
            cfg,
            cat_bg,
            foreground_layer=False,
            galaxy_disk_cool_stars=True,
            point_disk_stars=cfg.features.galaxy_view,
            plane_psf_elongation=cfg.features.galaxy_view,
            cluster_layer=False,
            extinction_for_stars=ext_for_star_paint,
            disk_w=disk_w if cfg.features.galaxy_view else None,
            galactic_structure=galactic,
            progress_cb=lambda p: _notify_stage("stars background", p * 0.78),
            **_trail_kw,
        )
        if cfg.features.galaxy_view:
            cluster_kw = {
                k: phot_kw[k]
                for k in (
                    "attach_apparent_mag",
                    "band_star_density_scale",
                    "photometry_mag_bright",
                    "photometry_mag_faint",
                    "photometry_slope",
                    "photometry_ultra_cut",
                    "max_ultra_bright_stars",
                    "use_imf_magnitudes",
                )
            }
            cluster_kw["imf_giant_fraction"] = cfg.features.imf_cluster_giant_fraction
            cat_cl = sample_cluster_star_catalog(
                rng_clusters,
                cfg.width,
                cfg.height,
                density_scale,
                galactic_structure=galactic,
                **cluster_kw,
            )
            stats_cl = _add_stars_from_catalog(
                stars_bg,
                rng_clusters,
                cfg,
                cat_cl,
                foreground_layer=False,
                galaxy_disk_cool_stars=False,
                point_disk_stars=False,
                plane_psf_elongation=True,
                cluster_layer=True,
                extinction_for_stars=ext_for_star_paint,
                disk_w=None,
                galactic_structure=galactic,
                progress_cb=lambda p: _notify_stage("stars background", 0.78 + p * 0.18),
                **_trail_kw,
            )
            stats = _merge_star_stats(stats, stats_cl)
        if cfg.features.galaxy_view and cfg.features.star_midlayer_scale > 1e-4:
            _notify_stage("stars mid", 0.0)
            mid_kw = dict(phot_kw)
            mid_kw["band_star_density_scale"] = (
                float(phot_kw.get("band_star_density_scale", 1.0))
                * float(cfg.features.star_midlayer_scale)
            )
            cat_mid = sample_star_catalog(
                rng_stars_mid,
                cfg.width,
                cfg.height,
                density_scale,
                layer="mid",
                galactic_band_boost=band_boost,
                latitude_color_bias=True,
                galactic_structure=galactic,
                **mid_kw,
            )
            if cfg.features.photoreal_stars:
                cat_mid = cull_faint_resolved_stars(
                    cat_mid,
                    rng_stars_mid,
                    galactic_structure=galactic,
                    magnitude_ref_mag=cfg.features.magnitude_ref_mag,
                    width=cfg.width,
                    height=cfg.height,
                    **faint_cull_kw,
                )
            if galactic is not None:
                attach_stellar_population_to_catalog(
                    cat_mid, galactic, cfg.width, cfg.height, rng_stars_mid
                )
            if (
                galactic is not None
                and cfg.features.unresolved_deposit_primary
                and cfg.features.photoreal_stars
                and float(cfg.features.disk_star_density_dropout) > 1e-6
            ):
                deposit_catalog_unresolved_flux(
                    cat_mid,
                    galactic,
                    rng_stars_mid,
                    width=cfg.width,
                    height=cfg.height,
                    magnitude_ref_mag=cfg.features.magnitude_ref_mag,
                    mag_bright=cfg.features.mag_bright_lim,
                    mag_faint=cfg.features.mag_faint_lim,
                    dropout_strength=float(cfg.features.disk_star_density_dropout),
                    mid_layer=True,
                )
            stats_mid = _add_stars_from_catalog(
                stars_mid,
                rng_stars_mid,
                cfg,
                cat_mid,
                foreground_layer=False,
                galaxy_disk_cool_stars=False,
                point_disk_stars=False,
                plane_psf_elongation=cfg.features.galaxy_view,
                cluster_layer=False,
                mid_layer=True,
                extinction_for_stars=ext_for_star_paint,
                disk_w=disk_w if cfg.features.galaxy_view else None,
                galactic_structure=galactic,
                progress_cb=lambda p: _notify_stage("stars mid", p),
                **_trail_kw,
            )
            stats = _merge_star_stats(stats, stats_mid)
            _notify_stage("stars mid", 1.0)
        if cfg.features.galaxy_view:
            bg_knee = 0.38 if stars_above_nebula else 0.95
            _soft_knee_star_layer(
                stars_bg, disk_w, knee=0.72 if stars_above_nebula else 0.52, strength=bg_knee
            )
            paint_reference_anchors(stars_bg, rng_stars_bg, cfg)
            # Brighter star field; extra lift when composited above nebula.
            if stars_above_nebula and cfg.features.photoreal_stars:
                bg_scale = 1.14
            elif cfg.features.photoreal_stars and cfg.features.use_spectral_teffective:
                bg_scale = 1.02
            else:
                bg_scale = 0.92 if cfg.features.photoreal_stars else 0.80
            bg_scale *= float(np.clip(1.0 - 0.14 * ism_dom, 0.78, 1.0))
            np.multiply(stars_bg, bg_scale, out=stars_bg, casting="unsafe")
            if cfg.features.star_midlayer_scale > 1e-4:
                mid_knee = 0.48 if stars_above_nebula else 0.72
                _soft_knee_star_layer(stars_mid, disk_w, knee=0.48, strength=mid_knee)
                np.multiply(stars_mid, 0.94, out=stars_mid, casting="unsafe")
        mb = float(cfg.features.motion_blur_strength)
        if mb > 1e-6 and cfg.features.galaxy_view and trail_subframes <= 1:
            stars_bg = apply_directional_motion_blur(stars_bg, strength=mb, periodic_x=wrap_lon_blur_x)
        dof_s = float(cfg.features.depth_of_field_strength)
        dof_px = float(cfg.features.depth_of_field_max_px)
        if depth_scene is not None and dof_s > 1e-6 and cfg.features.galaxy_view:
            stars_bg = apply_depth_of_field(
                stars_bg,
                np.clip(depth_scene * 0.88, 0.0, 1.0),
                strength=dof_s * 0.85,
                max_sigma_px=dof_px * 0.75,
                periodic_x=wrap_lon_blur_x,
            )
            if cfg.features.star_midlayer_scale > 1e-4:
                stars_mid = apply_depth_of_field(
                    stars_mid,
                    np.clip(depth_scene * 0.45, 0.0, 1.0),
                    strength=dof_s * 0.45,
                    max_sigma_px=dof_px * 0.45,
                    periodic_x=wrap_lon_blur_x,
                )
        _notify_stage("stars background", 1.0)
        if radiance_unify > 1e-6 and disk_w is not None:
            star_ref = np.maximum(0.0, stars_bg + stars_mid * 0.85)
            disk_chroma_ref = disk_chroma_from_star_layer(star_ref, disk_w)

    # Inherited unresolved luminous field on canvas before dust/extinction (resolved stars later).
    # Morphology-smoothing choke point: continuum blur + speckle σ-pyramid on g×T prior (see
    # galactic_structure.compose_inherited_unresolved_field). preserve_morph_hf when dust-primary.
    if cfg.features.galaxy_view and galactic is not None:
        _notify_stage("unresolved field", 0.0)
        grain_strength = (
            cfg.features.background_texture_strength
            * cfg.features.unresolved_background_strength
        )
        unresolved_rgb = unresolved_speckle_rgb(
            warmth=float(cfg.features.unresolved_spectral_warmth) * radiance_unify,
            disk_chroma=disk_chroma_ref,
        )
        canvas = compose_inherited_unresolved_field(
            canvas,
            rng_post,
            galactic,
            texture_strength=grain_strength,
            periodic_x=wrap_lon_blur_x,
            deposit_primary=cfg.features.unresolved_deposit_primary,
            unresolved_rgb=unresolved_rgb,
            preserve_morph_hf=bool(cfg.features.morphology_dust_primary),
        )
        _notify_stage("unresolved field", 1.0)

    gv_hdr = bool(cfg.features.galaxy_view)

    ext_paint_for_fg: np.ndarray | None = None
    if cfg.features.nebula:
        _notify_stage("nebula clouds/dust", 0.08)
        if nebula_bundle is not None:
            neb, neb_emit, dust_occlusion, lane_ext, extinction = nebula_bundle
        else:
            neb, neb_emit, dust_occlusion, lane_ext = generate_nebula(
                rng_nebula,
                cfg.nebula_mode,
                cfg.height,
                cfg.width,
                cfg.nebula_tuning,
                progress_cb=lambda p: _notify_stage("nebula clouds/dust", 0.08 + p * 0.18),
                galaxy_features=cfg.features if cfg.features.galaxy_view else None,
            )
            if (
                cfg.features.luminance_map_path is not None
                and cfg.features.catalog_mode == "luminance_overlay"
            ):
                lum_ov = load_luminance_overlay(
                    cfg.features.luminance_map_path, cfg.width, cfg.height
                )
                if lum_ov is not None:
                    neb = np.clip(neb * (0.62 + 0.38 * lum_ov[..., np.newaxis]), 0.0, 1.0)
            extinction = _extinction_from_dust_and_lane(dust_occlusion, lane_ext, cfg)
        _notify_stage("nebula clouds/dust", 0.30)
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            dust_primary = bool(
                galactic is not None and cfg.features.morphology_dust_primary
            )
            vf = float(cfg.features.extinction_void_floor)
            fil_s = float(cfg.features.extinction_filament_strength)
            disc_s = float(cfg.features.extinction_discontinuity_strength)
            if dust_primary and galactic is not None:
                from starsky_gen.structure_envelope import (
                    build_extinction_coupling_field,
                    build_structure_morph_host,
                    morphology_puff_punch_mask,
                )

                morph_puff_punch = morphology_puff_punch_mask(
                    galactic.dust_absorption_morph,
                    rng_nebula,
                    periodic_x=wrap_lon_blur_x,
                )
                morph_host = build_structure_morph_host(
                    disk_w if disk_w is not None else galactic.disk_weight,
                    galactic.vertical_extent,
                    periodic_x=wrap_lon_blur_x,
                )
                morph_coupling = build_extinction_coupling_field(
                    morph_host,
                    galactic.dust_absorption_morph,
                    galactic.vertical_extent,
                    morph_puff_punch,
                    periodic_x=wrap_lon_blur_x,
                )
                ext_paint = build_morphology_extinction_transmission(
                    galactic,
                    void_floor=vf,
                    filament_strength=fil_s,
                    discontinuity_strength=disc_s,
                    absorption_contrast=float(cfg.features.morphology_absorption_contrast),
                    extinction_strength=float(cfg.features.morphology_extinction_strength),
                    disk_weight=disk_w,
                    periodic_x=wrap_lon_blur_x,
                    lane_carve_boost=float(cfg.features.morphology_extinction_lane_carve),
                    lane_fragment_strength=float(cfg.features.morphology_lane_fragment_strength),
                    fine_texture_strength=float(cfg.features.extinction_fine_texture_strength),
                    brutal_mask=galactic.brutal_erasure_mask,
                    brutal_survival_floor=float(cfg.features.extinction_brutal_survival_floor),
                    puff_punch_mask=morph_puff_punch,
                    morph_coupling=morph_coupling,
                    opacity_gamma=float(cfg.features.extinction_opacity_gamma),
                )
                if not bool(cfg.features.morphology_dust_primary):
                    neb_lane = _blur_y_only(extinction, passes=1)
                    ext_paint = np.minimum(ext_paint, neb_lane)
            else:
                ext_paint = _blur_y_only(extinction, passes=1)
                ext_paint = _blur_x_only(ext_paint, passes=1, periodic_x=wrap_lon_blur_x)
                smoke_fill = 0.04 if galactic is not None else 0.14
                ext_paint = _guided_extinction_refine(
                    ext_paint, periodic_x=wrap_lon_blur_x, smoke_fill=smoke_fill
                )
                ext_paint = _deepen_extinction_lanes(ext_paint, disk_w, periodic_x=wrap_lon_blur_x)
                if cfg.features.extinction_first_nebula:
                    if galactic is not None and galactic.extinction_maps is not None:
                        em = galactic.extinction_maps
                        erosion = em.erosion
                        fractal_dark = em.fractal_dark
                        disruption = em.disruption
                    else:
                        erosion = build_filament_erosion_map(
                            rng_nebula,
                            cfg.height,
                            cfg.width,
                            periodic_x=wrap_lon_blur_x,
                        )
                        fractal_dark = build_fractal_extinction_field(
                            rng_nebula,
                            cfg.height,
                            cfg.width,
                            periodic_x=wrap_lon_blur_x,
                        )
                        disruption = build_band_disruption_field(
                            rng_nebula,
                            cfg.height,
                            cfg.width,
                            periodic_x=wrap_lon_blur_x,
                        )
                    og = float(cfg.features.extinction_opacity_gamma)
                    fil_trans = transmission_from_absorption_map(
                        erosion,
                        void_floor=vf,
                        sharpness=2.35,
                        opacity_gamma=og,
                    )
                    ext_paint = np.minimum(
                        ext_paint,
                        np.clip(vf + (fil_trans - vf) * min(fil_s, 1.25), vf, 1.0),
                    )
                    trans_frac = transmission_from_absorption_map(
                        fractal_dark,
                        void_floor=vf,
                        sharpness=2.0,
                        opacity_gamma=og,
                    )
                    ext_paint = np.minimum(ext_paint, trans_frac)
                    ext_paint = _deepen_extinction_lanes_strong(
                        ext_paint, disk_w, periodic_x=wrap_lon_blur_x
                    )
                    ext_paint = reinforce_absorption_edges(
                        ext_paint,
                        erosion,
                        edge_gain=0.48 * fil_s,
                        periodic_x=wrap_lon_blur_x,
                    )
                    if disc_s > 1e-6:
                        ext_paint = carve_extinction_discontinuities(
                            ext_paint,
                            disruption,
                            strength=disc_s,
                            void_floor=vf,
                        )
        else:
            ext_paint = extinction
        ext_paint_for_fg = ext_paint
        # #region agent log
        if disk_w is not None:
            _dbg_hist(
                "ext_paint_transmission",
                ext_paint,
                disk_w,
                hypothesis_id="BG",
            )
        # #endregion
        if galactic is not None:
            galactic.merge_nebula_extinction(ext_paint)
        dust_primary = bool(
            galactic is not None
            and cfg.features.morphology_dust_primary
            and cfg.nebula_mode == NebulaMode.galaxy_streak
        )
        plane_gate: np.ndarray | None = None
        bleed_gate: np.ndarray | None = None
        soft_band_env: np.ndarray | None = None
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            from starsky_gen.structure_envelope import latitude_plane_gate

            plane_gate = latitude_plane_gate(cfg.height, sigma=0.42, power=1.02)
            bleed_gate = latitude_plane_gate(cfg.height, sigma=0.50, power=0.94)
        wisp_gate: np.ndarray | None = None
        wisp_halo: np.ndarray | None = None
        if (
            cfg.nebula_mode == NebulaMode.galaxy_streak
            and plane_gate is not None
            and bleed_gate is not None
        ):
            from starsky_gen.structure_envelope import wisp_vertical_bleed_gate

            _ve = galactic.vertical_extent if galactic is not None else None
            _surv = galactic.structure_survival if galactic is not None else None
            wisp_gate, wisp_halo = wisp_vertical_bleed_gate(
                cfg.height,
                cfg.width,
                plane_gate=plane_gate,
                bleed_gate=bleed_gate,
                vertical_extent=_ve,
                structure_survival=_surv,
                periodic_x=wrap_lon_blur_x,
            )
            bleed_gate = np.clip(np.maximum(bleed_gate, wisp_gate * 0.94), 0.0, 1.0)
        band_bleed_env: np.ndarray | None = None
        core_void_mask: np.ndarray | None = None
        nebula_master_seed = int(seed) + int(generation_index) * 9973
        if cfg.nebula_mode == NebulaMode.galaxy_streak and disk_w is not None:
            from starsky_gen.structure_envelope import (
                build_core_void_mask,
                build_realistic_core_carve_mask,
                derive_nebula_rng,
            )

            _cv_rng = derive_nebula_rng(nebula_master_seed, "core_void")
            core_void_mask = build_core_void_mask(
                disk_w,
                _cv_rng,
                periodic_x=wrap_lon_blur_x,
                void_strength=0.76,
            )
            _carve_rng = derive_nebula_rng(nebula_master_seed, "core_carve")
            core_void_mask = np.clip(
                core_void_mask
                * build_realistic_core_carve_mask(
                    disk_w,
                    _carve_rng,
                    periodic_x=wrap_lon_blur_x,
                    carve_strength=0.78,
                ),
                0.26,
                1.0,
            )
            from starsky_gen.structure_envelope import (
                build_asymmetric_band_bleed_envelope,
                build_fine_puff_field,
            )

            _bleed_puff = build_fine_puff_field(
                np.random.default_rng(int(seed) + generation_index * 991 + 0xBAEB),
                cfg.height,
                cfg.width,
                periodic_x=wrap_lon_blur_x,
                strength=1.0,
                center_boost=1.0,
            )
            band_bleed_env = build_asymmetric_band_bleed_envelope(
                disk_w,
                cfg.height,
                cfg.width,
                rng_nebula,
                lon_asymmetry=galactic.lon_asymmetry if galactic is not None else None,
                puff_field=_bleed_puff,
                morph_absorption=(
                    galactic.dust_absorption_morph if galactic is not None else None
                ),
                periodic_x=wrap_lon_blur_x,
            )
        morph_haze_scale = 1.0
        morph_gas_scale = 1.0
        morph_emit_scale = 1.0
        emit_floor = float(cfg.features.extinction_void_floor)
        emit_morph: np.ndarray | None = None
        if galactic is not None and cfg.nebula_mode == NebulaMode.galaxy_streak:
            from starsky_gen.structure_envelope import build_emission_morphology_field

            emit_morph = build_emission_morphology_field(
                ext_paint,
                galactic.star_formation,
                galactic.structure_survival,
                galactic.dust_absorption,
                periodic_x=wrap_lon_blur_x,
                rng=np.random.default_rng(int(seed) + generation_index * 27109 + 0x51A0CE11),
            )
        if dust_primary:
            morph_haze_scale = float(cfg.features.morphology_nebula_haze_scale)
            morph_gas_scale = float(cfg.features.morphology_nebula_gas_scale)
            morph_emit_scale = float(cfg.features.morphology_nebula_emit_scale)
            emit_floor = max(emit_floor, 0.05)
        emit_clear = emission_clearance_from_extinction(
            ext_paint,
            floor=emit_floor,
            power=cfg.features.emission_extinction_gate_power,
        )
        emit_mask = emit_clear
        gas_mask = emit_clear
        if dust_primary:
            emit_gate_p = float(cfg.features.morphology_nebula_emit_gate_power)
            emit_mask = np.clip(emit_clear**emit_gate_p, 0.0, 1.0)
            if plane_gate is not None:
                emit_mask = emit_mask * plane_gate
            # Gas/haze stay visible in the band; lanes gate emission, not the whole ISM.
            gas_mask = gas_clearance_from_extinction(
                ext_paint,
                floor=emit_floor,
                power=0.92,
                min_clear=0.32,
                band_weight=disk_w,
            )
            # Keep gas clearance off-band for wispy morph bleed (plane_gate clips wisps).
        lane_av = extinction_av_scale_for_lane_depth(
            transmission_floor=cfg.features.extinction_transmission_floor,
            lane_mag_at_floor=cfg.features.extinction_lane_mag_max,
        )
        av_scale = (
            lane_av
            * cfg.nebula_tuning.dust_strength
            * cfg.features.dust_opacity
            * (1.05 if cfg.nebula_mode == NebulaMode.galaxy_streak else 0.92)
        )
        fill_suppress = 1.0
        if dust_primary:
            av_scale *= float(cfg.features.morphology_extinction_av_scale) * 0.82
            fill_suppress = float(cfg.features.morphology_extinction_fill_suppress) * 0.88
        canvas = _apply_extinction_to_canvas(
            canvas,
            ext_paint,
            galaxy_streak=cfg.nebula_mode == NebulaMode.galaxy_streak,
            rng_mottle=rng_nebula,
            av_scale=av_scale,
            rv=cfg.features.extinction_r_v,
            fill_suppress=fill_suppress,
            plane_gate=plane_gate,
        )
        if float(cfg.features.morphology_missing_region_boost) > 1e-6:
            miss_boost = float(cfg.features.morphology_missing_region_boost)
            if dust_primary:
                miss_boost *= 0.88
            canvas = apply_missing_region_extinction(
                canvas,
                ext_paint,
                void_floor=vf,
                missing_boost=miss_boost,
            )
        _notify_stage("nebula clouds/dust", 0.52)
        neb_rad_scale = 1.0
        if cfg.features.galaxy_view:
            neb_rad_scale = (
                float(cfg.features.band_nebula_radiance_scale)
                * float(cfg.features.nebula_color_strength)
                * float(np.clip(0.88 + 0.22 * ism_dom, 0.88, 1.35))
            )
        if dust_primary and galactic is not None and cfg.features.galaxy_view:
            from starsky_gen.structure_envelope import (
                build_morphology_ism_layers,
                derive_nebula_rng,
            )

            ism_rng = derive_nebula_rng(nebula_master_seed, "ism_layers")
            morph_ism_layers = build_morphology_ism_layers(
                galactic,
                ism_rng,
                cfg.height,
                cfg.width,
                hierarchy_strength=float(cfg.features.diffuse_scale_hierarchy),
                white_brightness=float(cfg.features.morphology_white_brightness),
                detail_strength=float(cfg.features.morphology_detail_strength),
                periodic_x=wrap_lon_blur_x,
                texture_seed=nebula_master_seed ^ 0x15A0D,
            )
            morph_ism_rgb = morph_ism_layers.combined_rgb
            morph_ism_luma = morph_ism_layers.luma
            if morph_puff_punch is None:
                morph_puff_punch = morph_ism_layers.puff_punch_mask
            body_s = neb_rad_scale * morph_haze_scale * (0.52 + 0.22 * ism_dom)
            ism_gate = wisp_gate if wisp_gate is not None else (
                plane_gate if plane_gate is not None else 1.0
            )
            if disk_w is not None:
                from starsky_gen.structure_envelope import soften_band_envelope

                soft_band_env = soften_band_envelope(
                    disk_w, (cfg.height, cfg.width), periodic_x=wrap_lon_blur_x
                )
                host = np.clip(soft_band_env, 0.0, 1.0)
                ism_gate = np.clip(ism_gate * (0.52 + 0.48 * host), 0.0, 1.0)
                if wisp_halo is not None:
                    ism_gate = np.clip(ism_gate + wisp_halo * 0.42, 0.0, 1.35)
            gate3 = ism_gate[..., np.newaxis]
            core_v = (
                core_void_mask[..., np.newaxis]
                if core_void_mask is not None
                else 1.0
            )
            canvas = np.maximum(
                0.0,
                canvas
                + morph_ism_layers.gold_emit_rgb * body_s * 0.36 * gate3 * core_v
                + morph_ism_layers.white_rgb * body_s * 0.12 * gate3 * core_v,
            )
            absorb_l = np.clip(np.max(morph_ism_layers.absorption_rgb, axis=2), 0.0, 1.0)
            canvas = np.maximum(
                0.0,
                canvas * (1.0 - (absorb_l * 0.14 * ism_gate)[..., np.newaxis]),
            )
            if off_band_layer is not None:
                hii_seed = nebula_master_seed ^ 0xE011C0DE
                hii_off = _build_off_band_hii_layer(
                    galactic,
                    disk_w if disk_w is not None else np.ones((cfg.height, cfg.width)),
                    strength=float(cfg.features.off_band_emission_strength),
                    periodic_x=wrap_lon_blur_x,
                    band_lat_sigma=float(cfg.features.band_lat_sigma),
                    blob_count=int(cfg.features.off_band_hii_blob_count),
                    diffuse_weight=float(cfg.features.off_band_hii_diffuse_weight),
                    hii_seed=hii_seed,
                )
                off_band_layer += hii_off
                from starsky_gen.structure_envelope import (
                    build_scene_red_hii_spots,
                    derive_nebula_rng,
                )

                scene_red = build_scene_red_hii_spots(
                    derive_nebula_rng(hii_seed, "scene_red_hdr"),
                    cfg.height,
                    cfg.width,
                    disk_w if disk_w is not None else np.ones((cfg.height, cfg.width)),
                    galactic.star_formation,
                    spot_count=int(cfg.features.scene_red_hii_spot_count),
                    strength=float(cfg.features.scene_red_hii_strength),
                    band_lat_sigma=float(cfg.features.band_lat_sigma),
                    periodic_x=wrap_lon_blur_x,
                    hii_seed=hii_seed,
                )
                off_band_layer += scene_red
                from starsky_gen.structure_envelope import build_band_hii_patches

                band_hii_layer = build_band_hii_patches(
                    derive_nebula_rng(hii_seed, "band_hii_hdr"),
                    cfg.height,
                    cfg.width,
                    disk_w if disk_w is not None else np.ones((cfg.height, cfg.width)),
                    galactic.star_formation,
                    patch_count=int(cfg.features.band_hii_patch_count),
                    strength=float(cfg.features.band_hii_strength),
                    periodic_x=wrap_lon_blur_x,
                    hii_seed=hii_seed,
                )
        neb_luma = np.mean(neb, axis=2)
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            neb_luma = np.clip(neb_luma + np.mean(neb_emit, axis=2) * 0.32, 0.0, 1.0)
        if cfg.features.galaxy_view:
            grade_neb_luma = neb_luma.copy()
        neb_peaks = np.max(neb, axis=2)
        # Visibility modulates gas radiance (additive in linear HDR), not Porter–Duff α-blend.
        _dw_vis = (
            np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
            if disk_w is not None
            else np.zeros((cfg.height, cfg.width), dtype=np.float64)
        )
        if _dw_vis.ndim == 1:
            _dw_vis = _dw_vis[:, None]
        if _dw_vis.shape != (cfg.height, cfg.width):
            _dw_vis = np.broadcast_to(_dw_vis, (cfg.height, cfg.width))
        # Softer vertical gate so gas fills the band envelope, not a razor-thin row.
        density_gate = np.clip(_dw_vis**1.38, 0.0, 1.0)
        gas_visibility = np.clip(
            (neb_luma**0.48) * (0.82 + 0.62 * neb_peaks) * density_gate,
            0.0,
            1.0,
        )
        _gas_vis_gate = (
            wisp_gate
            if dust_primary and wisp_gate is not None
            else plane_gate
        )
        if _gas_vis_gate is not None:
            gas_visibility = np.clip(gas_visibility * _gas_vis_gate, 0.0, 1.0)
        if dust_primary and galactic is not None:
            from starsky_gen.structure_envelope import build_morphology_turbulent_gas_field

            morph_gas_struct = build_morphology_turbulent_gas_field(
                galactic,
                ext_paint,
                periodic_x=wrap_lon_blur_x,
                rng=np.random.default_rng(int(seed) + generation_index * 44017 + 0xC10D),
            )
            morph_cont_guide = _morphology_continuum_guide(
                morph_gas_struct,
                ext_paint,
                morph_ism_luma=morph_ism_luma,
                dust_absorption_morph=galactic.dust_absorption_morph,
                periodic_x=wrap_lon_blur_x,
            )
            neb_struct = np.clip(1.0 + 0.22 * morph_gas_struct * density_gate, 0.88, 1.08)
        elif dust_primary:
            from starsky_gen.procedural_noise import _resize_bilinear as _rz, fbm2d

            _ch, _cw = max(4, cfg.height // 28), max(6, cfg.width // 24)
            neb_struct = fbm2d(rng_nebula, _ch, _cw, base_scale=0.14, octaves=3, periodic_x=wrap_lon_blur_x)
            neb_struct = _rz(neb_struct, cfg.height, cfg.width, periodic_x=wrap_lon_blur_x)
            neb_struct = np.clip(0.978 + 0.028 * neb_struct, 0.972, 1.02)
        else:
            _sr, _sc = max(2, cfg.height // 22), max(2, cfg.width // 22)
            neb_struct = _resize_bilinear(
                rng_nebula.random((_sr, _sc)),
                cfg.height,
                cfg.width,
                periodic_x=wrap_lon_blur_x,
            )
            neb_struct = _blur_separable_xy(neb_struct, passes=4, periodic_x=wrap_lon_blur_x)
            neb_struct = np.clip(0.972 + 0.045 * neb_struct, 0.965, 1.03)
        gas_visibility = np.clip(gas_visibility * neb_struct, 0.0, 1.0)
        neb_gas = np.maximum(neb, 0.0).astype(np.float64) * neb_rad_scale
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            gas_tint = np.array([1.01, 1.00, 1.012], dtype=np.float64)
            if stars_above_nebula:
                gas_tint = np.array([1.008, 1.00, 1.004], dtype=np.float64)
            neb_gas *= gas_tint
            if dust_primary and galactic is not None:
                sf = np.clip(galactic.star_formation, 0.0, 1.0)
                void_w = np.clip(galactic.void_mask, 0.0, 1.0)
                gas_presence = np.clip(
                    0.28 + 0.72 * sf * (1.0 - 0.42 * void_w),
                    0.0,
                    1.0,
                )
                if soft_band_env is not None:
                    host = np.clip(soft_band_env, 0.0, 1.0)
                    if wisp_gate is not None:
                        host = np.clip(np.maximum(host, wisp_gate * 0.78), 0.0, 1.0)
                    gas_presence = np.clip(
                        gas_presence * (0.54 + 0.46 * host), 0.0, 1.0
                    )
                elif disk_w is not None:
                    dw = np.clip(disk_w, 0.0, 1.0)
                    gas_presence = np.clip(gas_presence * (0.38 + 0.62 * dw), 0.0, 1.0)
                elif wisp_gate is not None:
                    gas_presence = np.clip(gas_presence * wisp_gate, 0.0, 1.0)
                if galactic is not None:
                    gas_rel = band_relative_clearance(
                        ext_paint, disk_w, min_clear=0.22, power=0.88
                    )
                    dust_open = np.clip(
                        1.0 - galactic.dust_absorption_morph * 0.88, 0.10, 1.0
                    ) ** 0.98
                    lane_sil = np.clip(
                        0.20 + 0.72 * gas_rel * (0.42 + 0.58 * dust_open),
                        0.18,
                        1.0,
                    )
                else:
                    lane_sil = np.clip(
                        0.18
                        + 0.82
                        * band_relative_clearance(ext_paint, disk_w, min_clear=0.20),
                        0.16,
                        1.0,
                    )
                proc_w = float(np.clip(0.34 + 0.42 * morph_gas_scale, 0.28, 0.72))
                neb_proc_base = neb_gas
                if morph_ism_rgb is not None and morph_ism_luma is not None:
                    from starsky_gen.structure_envelope import build_fine_puff_field

                    morph_gas_s = neb_rad_scale * (1.18 + 0.48 * ism_dom)
                    struct_g = (
                        morph_gas_struct
                        if morph_gas_struct is not None
                        else np.clip(morph_ism_luma, 0.0, 1.0)
                    )
                    puff_gas = build_fine_puff_field(
                        np.random.default_rng(int(seed) + generation_index * 8803 + 0x50FF),
                        cfg.height,
                        cfg.width,
                        periodic_x=wrap_lon_blur_x,
                        strength=1.0,
                        center_boost=1.0,
                    )
                    puff_mod = np.clip(0.38 + 1.22 * puff_gas**1.06, 0.0, 1.85)
                    morph_lu = np.clip(morph_ism_luma, 0.0, 1.0)
                    morph_mix = (
                        morph_ism_rgb
                        * morph_gas_s
                        * struct_g[..., np.newaxis]
                        * (0.72 + 0.28 * puff_mod[..., np.newaxis])
                    )
                    if disk_chroma_ref is not None:
                        _tr = disk_chroma_ref / max(
                            float(
                                np.dot(
                                    disk_chroma_ref,
                                    np.array([0.2126, 0.7152, 0.0722], dtype=np.float64),
                                )
                            ),
                            1e-8,
                        )
                        morph_mix = morph_lu[..., np.newaxis] * _tr.reshape(1, 1, 3) * (
                            0.62 + 0.38 * struct_g[..., np.newaxis]
                        ) + morph_mix * 0.22
                    _morph_gate = (
                        wisp_gate if wisp_gate is not None else plane_gate
                    )
                    if _morph_gate is not None:
                        morph_mix = morph_mix * _morph_gate[..., np.newaxis]
                    morph_w = float(np.clip(1.06 - 0.22 * morph_gas_scale, 0.88, 1.12))
                    proc_tail = (
                        neb_proc_base
                        * proc_w
                        * gas_presence[..., np.newaxis]
                        * lane_sil[..., np.newaxis]
                    )
                    if morph_gas_struct is not None:
                        proc_tail = proc_tail * (
                            0.42 + 0.58 * morph_gas_struct[..., np.newaxis]
                        )
                    neb_gas = morph_mix * morph_w * 1.02 + proc_tail * 0.34
                    if wisp_halo is not None and morph_cont_guide is not None:
                        wisp_rgb = np.clip(
                            morph_cont_guide[..., np.newaxis]
                            * np.array([0.94, 0.90, 0.86], dtype=np.float64),
                            0.0,
                            1.0,
                        )
                        neb_gas = neb_gas + (
                            wisp_rgb
                            * wisp_halo[..., np.newaxis]
                            * neb_rad_scale
                            * morph_gas_scale
                            * 0.14
                        )
                else:
                    neb_gas *= (
                        proc_w
                        * gas_presence[..., np.newaxis]
                        * lane_sil[..., np.newaxis]
                    )
                    if morph_gas_struct is not None:
                        neb_gas = neb_gas * (
                            0.35 + 0.65 * morph_gas_struct[..., np.newaxis]
                        )
                _neb_gas_gate = (
                    wisp_gate if wisp_gate is not None else plane_gate
                )
                if _neb_gas_gate is not None:
                    neb_gas = neb_gas * _neb_gas_gate[..., np.newaxis]
                gas_visibility = np.clip(
                    gas_visibility * gas_presence * gas_mask,
                    0.0,
                    1.0,
                )
                gas_rel_vis = band_relative_clearance(
                    ext_paint, disk_w, min_clear=0.30, power=0.85
                )
                host_vis = (
                    np.clip(soft_band_env, 0.0, 1.0)
                    if soft_band_env is not None
                    else np.clip(disk_w, 0.0, 1.0)
                )
                gas_visibility = np.clip(
                    np.maximum(
                        gas_visibility,
                        (0.44 + 0.56 * gas_rel_vis) * gas_presence * host_vis,
                    ),
                    0.0,
                    1.0,
                )
                neb_gas = neb_gas * (1.22 + 0.18 * morph_gas_scale)
                if morph_gas_struct is not None and galactic is not None:
                    puff_clear = np.clip(
                        (1.0 - galactic.dust_absorption_morph) ** 1.02, 0.06, 1.0
                    )
                    puff_vis = np.clip(
                        morph_gas_struct * 0.72 + puff_clear * 0.48, 0.0, 1.0
                    )
                    puff_boost = puff_vis * gas_presence * 1.05
                    if wisp_halo is not None:
                        puff_boost = np.clip(
                            np.maximum(puff_boost, puff_vis * wisp_halo * 0.72),
                            0.0,
                            1.0,
                        )
                    gas_visibility = np.clip(
                        np.maximum(gas_visibility, puff_boost),
                        0.0,
                        1.0,
                    )
                elif morph_ism_luma is not None:
                    gas_visibility = np.clip(
                        np.maximum(
                            gas_visibility,
                            morph_ism_luma * 0.52 * gas_presence,
                        ),
                        0.0,
                        1.0,
                    )
            elif dust_primary:
                neb_gas *= morph_gas_scale * np.clip(
                    0.55 + 0.45 * ext_paint[..., None], 0.55, 1.0
                )
                gas_visibility = np.clip(gas_visibility * gas_mask, 0.0, 1.0)
            else:
                neb_gas *= 1.08 * (0.90 + 0.10 * ext_paint[..., None])
        if cfg.features.galaxy_view and disk_w is not None and cfg.nebula_mode == NebulaMode.galaxy_streak:
            neb_gas = attenuate_rgb_column_comb(
                neb_gas,
                disk_w,
                strength=0.62 if dust_primary else 0.48,
                periodic_x=wrap_lon_blur_x,
            )
            chroma_tgt = disk_chroma_ref
            if chroma_tgt is None and dust_primary:
                chroma_tgt = disk_chroma_from_star_layer(
                    np.maximum(0.0, stars_bg + stars_mid * 0.85), disk_w
                )
            if chroma_tgt is not None:
                neb_gas = soften_diffuse_chroma_toward(
                    neb_gas,
                    disk_w,
                    chroma_tgt,
                    strength=0.58 if dust_primary else 0.32,
                )
        gas_occ = float(cfg.features.unresolved_gas_occlusion)
        if gas_occ > 1e-6 and cfg.features.galaxy_view:
            canvas = _mute_speckle_under_nebula(
                canvas,
                neb_luma,
                gas_mask=gas_mask,
                ext_paint=ext_paint,
                emit_luma=np.mean(neb_emit, axis=2),
                strength=gas_occ,
            )
        canvas = composite_add_gas(canvas, neb_gas, visibility=gas_visibility)
        # #region agent log
        import os as _os_gas

        if _os_gas.environ.get("STK_ABLATION") and disk_w is not None:
            _dbg_hist("canvas_after_gas", canvas, disk_w, hypothesis_id="A")
            if ext_paint is not None:
                _dbg_hist("ext_paint", ext_paint, disk_w, hypothesis_id="A")
        # #endregion
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            emit_luma = np.mean(neb_emit, axis=2)
            emit_peaks = np.max(neb_emit, axis=2)
            emit_layer = np.maximum(neb_emit, 0.0).astype(np.float64) * neb_rad_scale
            emit_scale = (
                float(cfg.features.nebula_emit_scale_with_stars)
                if stars_above_nebula
                else 1.0
            )
            emit_layer *= emit_scale * morph_emit_scale * (1.02 * (0.92 + 0.08 * ext_paint[..., None]))
            if (
                dust_primary
                and morph_ism_layers is not None
                and cfg.features.galaxy_view
            ):
                emit_layer = np.maximum(
                    emit_layer,
                    morph_ism_layers.red_hii_rgb * neb_rad_scale * 0.72,
                )
            if dust_primary and galactic is not None:
                emit_boost = np.clip(
                    0.35 + 0.65 * galactic.star_formation * (1.0 - 0.35 * galactic.void_mask),
                    0.25,
                    1.0,
                )
                emit_layer *= emit_boost[..., np.newaxis]
            if emit_morph is not None:
                emit_layer *= emit_morph[..., np.newaxis]
            emit_layer *= emit_mask[..., np.newaxis]
            emit_tint_band = np.array([1.06, 0.96, 0.94], dtype=np.float64)
            if stars_above_nebula:
                emit_tint_band = np.array([1.03, 0.98, 0.97], dtype=np.float64)
            emit_tint_off = np.array([1.28, 0.34, 0.22], dtype=np.float64)
            dw_emit = (
                np.clip(disk_w, 0.0, 1.0)
                if disk_w is not None
                else np.ones((cfg.height, cfg.width), dtype=np.float64)
            )
            off_emit_gate = _off_band_emission_mask(
                dw_emit,
                galactic,
                cfg.height,
                decouple_strength=0.58,
                band_lat_sigma=float(cfg.features.band_lat_sigma),
            )
            band_emit_gate = np.clip(1.0 - off_emit_gate**0.92, 0.0, 1.0) * dw_emit
            wing = np.clip(band_emit_gate * off_emit_gate * 3.6, 0.0, 1.0)
            ha_band = np.array([1.22, 0.40, 0.30], dtype=np.float64)
            band_tint = emit_tint_band * (1.0 - wing[..., np.newaxis]) + ha_band * wing[..., np.newaxis]
            emit_band = emit_layer * band_emit_gate[..., np.newaxis] * band_tint
            emit_off = (
                emit_layer
                * np.clip(off_emit_gate**1.72, 0.0, 1.0)[..., np.newaxis]
                * emit_tint_off
                * float(cfg.features.off_band_emit_scale)
            )
            if off_band_layer is not None:
                emit_off_loc = _localize_off_band_rgb(
                    np.maximum(emit_off, 0.0),
                    off_emit_gate,
                    periodic_x=wrap_lon_blur_x,
                    peak_percentile=80.0,
                )
                off_band_layer += emit_off_loc
            emit_hot = np.clip(
                np.maximum(emit_peaks, emit_luma * 1.1),
                0.0,
                1.0,
            )
            emit_hot_band = emit_hot * band_emit_gate
            canvas = composite_emission_add_screen(
                canvas,
                emit_band,
                emit_hot_band,
                add_strength=1.0,
                core_screen_mix=0.68,
            )
            emit_unsharp = float(cfg.features.emission_unsharp_amount) * float(
                cfg.features.nebula_color_strength
            )
            canvas = _emission_mask_local_contrast(
                canvas,
                emit_luma,
                emit_peaks,
                amp=emit_unsharp,
                periodic_x=wrap_lon_blur_x,
            )
            scatter_s = (
                0.042
                * float(cfg.features.nebula_color_strength)
                * float(cfg.features.volumetric_scatter_strength)
            )
            if stars_above_nebula:
                scatter_s *= 0.72
            if dust_primary:
                scatter_s *= float(np.clip(np.mean(gas_mask) * 0.85 + 0.22, 0.22, 0.72))
            else:
                scatter_s *= float(np.clip(np.mean(emit_clear) * 1.15 + 0.12, 0.15, 1.0))
            scatter_luma = emit_luma * gas_mask
            scatter_od = np.clip(1.0 - ext_paint, 0.06, 1.0) * 1.15
            if plane_gate is not None:
                scatter_luma = scatter_luma * (
                    bleed_gate if bleed_gate is not None else plane_gate
                )
                scatter_od = scatter_od * plane_gate
            canvas = forward_scatter_mie(
                canvas,
                scatter_luma,
                strength=scatter_s,
                optical_depth=scatter_od,
                g_forward=cfg.features.volumetric_g_forward,
                g_back=cfg.features.volumetric_g_back,
                periodic_x=wrap_lon_blur_x,
                blur_fn=_blur_separable_xy,
            )
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            core_v2 = (
                core_void_mask
                if core_void_mask is not None
                else np.ones((cfg.height, cfg.width), dtype=np.float64)
            )
            _yy_axis = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float64)[:, None]
            axis_carve = 1.0 - 0.38 * np.exp(-((_yy_axis / 0.058) ** 2))
            cont_sup = (
                float(cfg.features.morphology_continuum_suppress)
                if dust_primary
                else 1.0
            )
            # Haze/cloud/glow: smooth neb_luma re-washes morph unless cont_luma uses morph guide.
            # Haze: legacy keys to clear sightlines; morphology keys to dust rims + ISM peaks.
            cont_luma = neb_luma
            if dust_primary and morph_cont_guide is not None:
                cont_luma = np.clip(
                    np.maximum(neb_luma * 0.12, morph_cont_guide * 1.02),
                    0.0,
                    1.0,
                )
            if dust_primary:
                dust_w = np.clip(1.0 - ext_paint, 0.0, 1.0) ** 0.92
                gas_w = np.clip(cont_luma * (0.44 + 0.56 * dust_w), 0.0, 1.0)
            else:
                gas_w = np.clip(cont_luma * (0.22 + 0.78 * ext_paint), 0.0, 1.0)
            if emit_morph is not None:
                gas_w = np.clip(gas_w * emit_morph, 0.0, 1.0)
            pg = plane_gate if plane_gate is not None else 1.0
            bg = bleed_gate if bleed_gate is not None else pg
            gas_band = gas_w * bg
            haze_soft = _blur_separable_xy(gas_band, passes=3, periodic_x=wrap_lon_blur_x)
            haze_hi = np.clip(gas_band - haze_soft, 0.0, 1.0)
            haze = np.clip(
                haze_soft * (0.28 if dust_primary else 0.70)
                + haze_hi * (0.52 if dust_primary else 0.12)
                + gas_band * 0.08,
                0.0,
                1.0,
            )
            band_h = _band_render_envelope(
                cfg.height,
                cfg.width,
                band_bleed_env=band_bleed_env,
                bleed_gate=bg,
                plane_gate=pg,
            )
            haze_s = float(cfg.features.nebula_haze_strength) * morph_haze_scale * cont_sup * 0.88
            if dust_primary:
                haze_s *= 0.36
            haze_warm = 0.55 if dust_primary else 1.0
            gold_h = np.array([0.48, 0.46, 0.40], dtype=np.float64) * haze_warm
            mag_h = np.array([0.32, 0.09, 0.24], dtype=np.float64) * (0.65 if dust_primary else 1.0)
            cool_scatter = np.array([0.09, 0.10, 0.12], dtype=np.float64)
            gc = gas_mask[..., np.newaxis]
            canvas = _canvas_add_linear(
                canvas,
                (
                    (haze * band_h)[..., None] * gold_h * (0.020 if stars_above_nebula else 0.030)
                    + (haze * band_h)[..., None] * cool_scatter * 0.012
                    + (band_h * core_v2 * axis_carve)[..., None]
                    * np.array([0.08, 0.078, 0.076], dtype=np.float64)
                    * 0.012
                    + (np.clip(haze, 0.0, 1.0) ** 1.02 * band_h)[..., None] * mag_h * 0.006
                )
                * haze_s
                * gc,
                galaxy_hdr=gv_hdr,
            )
            # Restore broad gold/white luminous cloud band across the galactic plane.
            if dust_primary:
                dust_w = np.clip(1.0 - ext_paint, 0.0, 1.0) ** 0.95
                cloud_src = np.clip(cont_luma * (0.52 + 0.48 * dust_w), 0.0, 1.0)
            else:
                cloud_src = np.clip(cont_luma * (0.34 + 0.66 * ext_paint**1.18), 0.0, 1.0)
            if emit_morph is not None:
                cloud_src = np.clip(cloud_src * emit_morph, 0.0, 1.0)
            band_cloud = _blur_separable_xy(cloud_src * bg, passes=2, periodic_x=wrap_lon_blur_x)
            cloud_hi = np.clip(cloud_src * bg - band_cloud, 0.0, 1.0)
            band_cloud = np.clip(band_cloud**0.92 + cloud_hi * 0.26, 0.0, 1.0)
            plane_bw = _band_render_envelope(
                cfg.height,
                cfg.width,
                band_bleed_env=band_bleed_env,
                bleed_gate=bg,
                plane_gate=pg,
            )
            gold_band_rgb = np.array([0.52, 0.46, 0.36], dtype=np.float64)
            white_band_rgb = np.array([0.76, 0.74, 0.70], dtype=np.float64)
            cloud_mix = np.clip(0.44 + 0.56 * band_cloud, 0.0, 1.0)
            warm_white = gold_band_rgb * (1.0 - cloud_mix[..., None]) + white_band_rgb * cloud_mix[..., None]
            canvas = _canvas_add_linear(
                canvas,
                (band_cloud * plane_bw * core_v2 * axis_carve)[..., None]
                * warm_white
                * ((0.028 if stars_above_nebula else 0.034) if dust_primary else (0.072 if stars_above_nebula else 0.094))
                * haze_s
                * gc,
                galaxy_hdr=gv_hdr,
            )
            cloud_scale = (1.28 + 0.22 * ism_dom) if dust_primary else (1.15 + 0.20 * ism_dom)
            if dust_primary and morph_cont_guide is not None:
                cloud_scale *= 0.32
            if cont_sup > 0.06:
                _cloud_luma = (
                    morph_cont_guide
                    if dust_primary and morph_cont_guide is not None
                    else neb_luma
                )
                canvas = _add_galactic_cloud_body(
                    canvas,
                    rng_nebula,
                    neb_luma=_cloud_luma,
                    ext_paint=ext_paint,
                    disk_w=disk_w,
                    periodic_x=wrap_lon_blur_x,
                    strength=cloud_scale * cont_sup,
                )
            # Broad unresolved-light lift for the Milky Way body (closer to photographic continuum).
            if dust_primary:
                dust_w = np.clip(1.0 - ext_paint, 0.0, 1.0) ** 0.90
                glow_src = np.clip(cont_luma * (0.46 + 0.54 * dust_w), 0.0, 1.0)
            else:
                glow_src = np.clip(neb_luma * (0.26 + 0.74 * ext_paint**1.22), 0.0, 1.0)
            if emit_morph is not None:
                glow_src = np.clip(glow_src * emit_morph, 0.0, 1.0)
            disk_glow = _blur_separable_xy(glow_src * bg, passes=2, periodic_x=wrap_lon_blur_x)
            glow_hi = np.clip(glow_src * bg - disk_glow, 0.0, 1.0)
            disk_glow = np.clip(disk_glow**0.86 + glow_hi * 0.20, 0.0, 1.0)
            band_g = _band_render_envelope(
                cfg.height,
                cfg.width,
                band_bleed_env=band_bleed_env,
                bleed_gate=bg,
                plane_gate=pg,
            )
            glow_rgb = np.array([0.19, 0.18, 0.17], dtype=np.float64)
            canvas = _canvas_add_linear(
                canvas,
                (disk_glow * band_g)[..., None]
                * glow_rgb
                * (0.12 if dust_primary else (0.14 if stars_above_nebula else 0.22))
                * haze_s
                * gc,
                galaxy_hdr=gv_hdr,
            )
        if cfg.nebula_mode == NebulaMode.galaxy_streak:
            band_air = (
                plane_gate
                if plane_gate is not None
                else np.exp(-((np.linspace(-1.0, 1.0, cfg.height, dtype=np.float64)[:, None] ** 2) / 0.40))
            )
            air_rgb = np.array([0.012, 0.011, 0.014], dtype=np.float64)
            canvas = _canvas_add_linear(canvas, band_air[..., None] * air_rgb * 1.15, galaxy_hdr=gv_hdr)
            h, w = cfg.height, cfg.width
            gray = np.clip(
                rec709_luma(neb_gas) * gas_visibility + emit_luma * 0.40,
                0.0,
                1.0,
            )
            if plane_gate is not None:
                gray = gray * plane_gate
            bh, bw = max(3, h // 14), max(3, w // 18)
            bloom_small = _resize_bilinear(gray, bh, bw, periodic_x=wrap_lon_blur_x)
            bloom = _resize_bilinear(bloom_small, h, w, periodic_x=wrap_lon_blur_x)
            bloom = np.clip(bloom**1.18, 0.0, 1.0)
            band_w = _band_render_envelope(
                h,
                w,
                band_bleed_env=band_bleed_env,
                bleed_gate=bleed_gate if bleed_gate is not None else plane_gate,
                plane_gate=plane_gate,
                lat_sigma=0.40,
            )
            warm_bloom = np.array([0.40, 0.38, 0.34], dtype=np.float64)
            canvas = _canvas_add_linear(
                canvas,
                (bloom * band_w)[..., None] * warm_bloom * 0.016 * haze_s * gc,
                galaxy_hdr=gv_hdr,
            )
        if cfg.features.galaxy_view and cfg.features.bulge_intensity > 1e-4:
            bulge_layer = render_bulge_layer(
                cfg.width,
                cfg.height,
                bulge_n=cfg.features.bulge_n,
                bulge_scale=cfg.features.bulge_scale,
                bulge_intensity=cfg.features.bulge_intensity,
                band_rotation_deg=cfg.features.band_rotation_deg,
                band_curvature_amp=cfg.features.band_curvature_amp,
                band_lat_sigma=cfg.features.band_lat_sigma,
                rng=rng_nebula,
                extinction_mod=1.0 - np.clip(ext_paint, 0.0, 1.0) if cfg.features.extinction_first_nebula else None,
            )
            # Soft-add bulge so it tints the nebula core instead of forming bright lumps.
            bulge_layer = bulge_layer * (0.55 + 0.45 * np.clip(disk_w, 0.0, 1.0)[..., np.newaxis])
            if core_void_mask is not None:
                bulge_layer = bulge_layer * core_void_mask[..., np.newaxis]
            canvas = _canvas_add_linear(canvas, bulge_layer * 0.48, galaxy_hdr=gv_hdr)
        if cfg.nebula_mode == NebulaMode.galaxy_streak and cfg.features.galaxy_view:
            canvas = _dust_lane_local_contrast(
                canvas,
                ext_paint,
                disk_w,
                amp=cfg.features.lane_contrast_amp,
                periodic_x=wrap_lon_blur_x,
            )
            thick_pg = np.clip(1.0 - ext_paint, 0.0, 1.0) * (
                plane_gate if plane_gate is not None else 1.0
            )
            rim = np.clip(
                _blur_separable_xy(thick_pg, passes=1, periodic_x=wrap_lon_blur_x) - thick_pg,
                0.0,
                1.0,
            )
            rim = _blur_separable_xy(
                rim * neb_luma * disk_w * (plane_gate if plane_gate is not None else 1.0),
                passes=1,
                periodic_x=wrap_lon_blur_x,
            )
            canvas = np.clip(
                canvas + rim[..., np.newaxis] * np.array([1.02, 0.99, 0.96]) * 0.024,
                0.0,
                None,
            )
            canvas = _dust_lane_forward_scatter(
                canvas,
                ext_paint,
                neb_luma,
                disk_w,
                strength=0.026 * float(cfg.features.nebula_color_strength),
                periodic_x=wrap_lon_blur_x,
                plane_gate=plane_gate,
            )
            ms = float(cfg.features.dust_multiscatter_strength)
            if dust_primary:
                ms *= float(np.clip(np.mean(gas_mask) * 0.45 + 0.06, 0.06, 0.26))
            if ms > 1e-6:
                if dust_primary:
                    ms *= 0.55
                amb_luma = np.clip(
                    rec709_luma(canvas) + neb_luma * 0.42 + emit_luma * 0.18,
                    0.0,
                    None,
                )
                canvas = dust_lane_multiscatter_fill(
                    canvas,
                    ext_paint,
                    amb_luma,
                    strength=ms,
                    periodic_x=wrap_lon_blur_x,
                    blur_fn=_blur_separable_xy,
                    plane_gate=plane_gate,
                )
        if (
            dust_primary
            and ext_paint_for_fg is not None
            and disk_w is not None
            and cfg.nebula_mode == NebulaMode.galaxy_streak
        ):
            canvas = _sculpt_morphology_lane_contrast(
                canvas,
                ext_paint_for_fg,
                disk_w,
                plane_gate=plane_gate,
                strength=0.62,
            )
        if cfg.features.galaxy_view and cfg.features.nebula and disk_w is not None:
            ism_lift_luma = (
                morph_ism_luma
                if morph_ism_luma is not None and dust_primary
                else (grade_neb_luma if grade_neb_luma is not None else neb_luma)
            )
            canvas = _apply_band_ism_dominance(
                canvas,
                disk_w=disk_w,
                neb_luma=ism_lift_luma,
                galactic=galactic,
                dominance=ism_dom,
                periodic_x=wrap_lon_blur_x,
                disk_chroma=disk_chroma_ref,
                chroma_lock=float(cfg.features.ism_lift_chroma_lock) * radiance_unify,
                extinction=ext_paint_for_fg,
                preserve_detail=morph_ism_luma is not None and dust_primary,
            )
            if (
                dust_primary
                and morph_gas_struct is not None
                and disk_w is not None
            ):
                canvas = _reinforce_morph_canvas_texture(
                    canvas,
                    morph_gas_struct,
                    disk_w,
                    plane_gate=plane_gate,
                    strength=1.48,
                )
            if radiance_unify > 1e-4 and disk_w is not None and cfg.features.stars:
                if disk_chroma_ref is None:
                    disk_chroma_ref = disk_chroma_from_star_layer(
                        np.maximum(0.0, stars_bg + stars_mid * 0.85), disk_w
                    )
                lu_blur = _blur_separable_xy(rec709_luma(canvas), passes=4, periodic_x=wrap_lon_blur_x)
                # harmonize_diffuse blurs luma before chroma reprojection — lowers band HF on canvas.
                unify_s = radiance_unify * (0.06 if dust_primary else 0.88)
                canvas = harmonize_diffuse_canvas_chroma(
                    canvas,
                    disk_chroma_ref,
                    disk_w,
                    lu_blur,
                    strength=unify_s,
                )
            if (
                not bool(cfg.features.off_band_late_composite)
                and float(cfg.features.off_band_emission_strength) > 1e-6
            ):
                canvas = _composite_red_hii_late(
                    canvas,
                    off_band_layer,
                    band_hii_layer,
                    strength=float(cfg.features.off_band_emission_strength),
                    periodic_x=wrap_lon_blur_x,
                    disk_w=disk_w,
                )
            if ext_paint_for_fg is not None:
                patch_s = float(cfg.features.band_dark_patch_strength)
                if dust_primary:
                    patch_s *= 0.78
                canvas = _apply_band_dark_patches(
                    canvas,
                    ext_paint_for_fg,
                    disk_w,
                    galactic,
                    strength=patch_s,
                    periodic_x=wrap_lon_blur_x,
                    morph_primary=dust_primary,
                )
        _notify_stage("nebula clouds/dust", 1.0)

    if (
        cfg.features.stars
        and cfg.features.galaxy_view
        and cfg.features.halo_star_enabled
        and float(np.max(stars_halo)) > 1e-8
    ):
        halo_add = stars_halo.astype(np.float64, copy=False)
        if ext_paint_for_fg is not None:
            ext_k = np.clip(np.asarray(ext_paint_for_fg, dtype=np.float64), 0.0, 1.0)
            halo_add = halo_add * np.clip(0.18 + 0.82 * ext_k, 0.12, 1.0)[..., np.newaxis]
        canvas = _canvas_add_linear(canvas, halo_add * 0.90, galaxy_hdr=gv_hdr)

    if cfg.features.stars and not stars_above_nebula:
        stars_bg_add = stars_bg
        stars_mid_add = stars_mid
        if cfg.features.galaxy_view and disk_w is not None:
            plane_scale = float(cfg.features.band_star_plane_scale) * float(
                np.clip(1.12 - 0.42 * ism_dom, 0.35, 1.0)
            )
            stars_bg_add = _attenuate_stars_for_plane(stars_bg, disk_w, plane_scale=plane_scale)
            if cfg.features.star_midlayer_scale > 1e-4:
                stars_mid_add = _attenuate_stars_for_plane(
                    stars_mid, disk_w, plane_scale=min(1.0, plane_scale + 0.12)
                )
        canvas = _canvas_add_linear(canvas, stars_bg_add, galaxy_hdr=gv_hdr)
        if cfg.features.galaxy_view and cfg.features.star_midlayer_scale > 1e-4:
            canvas = _canvas_add_linear(canvas, stars_mid_add * 0.92, galaxy_hdr=gv_hdr)

    if cfg.features.galaxy_view and galactic is None and not stars_above_nebula:
        grain_strength = (
            cfg.features.background_texture_strength
            * cfg.features.unresolved_background_strength
        )
        canvas = _apply_separated_disk_sky_grain(
            canvas,
            rng_post,
            height=cfg.height,
            width=cfg.width,
            periodic_x=wrap_lon_blur_x,
            texture_strength=grain_strength,
            speckle_scale=0.95,
        )

    if cfg.features.stars:
        _notify_stage("stars foreground prep", 0.0)
        cat_fg = sample_star_catalog(
            rng_stars_fg,
            cfg.width,
            cfg.height,
            density_scale,
            layer="foreground",
            latitude_color_bias=False,
            foreground_star_density_scale=cfg.features.foreground_star_density_scale,
            galactic_structure=galactic,
        )
        _notify_stage("stars foreground prep", 1.0)
        _notify_stage("stars foreground", 0.0)
        stats_fg = _add_stars_from_catalog(
            stars_fg,
            rng_stars_fg,
            cfg,
            cat_fg,
            foreground_layer=True,
            galaxy_disk_cool_stars=False,
            point_disk_stars=False,
            plane_psf_elongation=cfg.features.galaxy_view,
            cluster_layer=False,
            extinction_for_stars=ext_paint_for_fg,
            galactic_structure=galactic,
            progress_cb=lambda p: _notify_stage("stars foreground", p),
        )
        stats = _merge_star_stats(stats, stats_fg)
        if cfg.features.galaxy_view:
            fg_knee = 0.62 if stars_above_nebula else 1.05
            _soft_knee_star_layer(stars_fg, disk_w, knee=0.42, strength=fg_knee)
            fg_scale = 1.08 if (stars_above_nebula and cfg.features.use_spectral_teffective) else (
                0.98 if cfg.features.use_spectral_teffective else 0.90
            )
            np.multiply(stars_fg, fg_scale, out=stars_fg, casting="unsafe")
        if not stars_above_nebula:
            stars_fg_add = stars_fg
            if cfg.features.galaxy_view and disk_w is not None:
                stars_fg_add = _attenuate_stars_for_plane(
                    stars_fg,
                    disk_w,
                    plane_scale=min(1.0, float(cfg.features.band_star_plane_scale) + 0.18),
                )
            canvas = _canvas_add_linear(canvas, stars_fg_add, galaxy_hdr=gv_hdr)
        _notify_stage("stars foreground", 0.88)
        _notify_stage("stars foreground", 1.0)

    render_profile = cfg.features.render_profile
    use_linear_grade = render_profile in (RenderProfile.physical_grade, RenderProfile.full)
    use_display_finish = render_profile == RenderProfile.full
    use_star_display_desat = render_profile == RenderProfile.full

    if cfg.features.galaxy_view:
        _notify_stage("grade/color", 0.0)
        if photon_unify > 1e-4 and disk_w is not None and cfg.features.stars:
            star_ref = np.maximum(0.0, stars_bg + stars_mid * 0.85 + stars_fg * 0.65)
            if disk_chroma_ref is None:
                disk_chroma_ref = disk_chroma_from_star_layer(star_ref, disk_w)
            disk_exposure = estimate_disk_photon_exposure(
                star_ref, disk_w, chroma=disk_chroma_ref
            )
            lu_blur_exp = _blur_separable_xy(
                rec709_luma(canvas), passes=4, periodic_x=wrap_lon_blur_x
            )
            canvas = apply_shared_photon_exposure(
                canvas,
                disk_exposure,
                disk_w,
                strength=photon_unify,
                diffuse_only=True,
                diffuse_blur=lu_blur_exp,
            )
        if (
            radiance_unify > 1e-4
            and disk_w is not None
            and cfg.features.stars
            and not stars_above_nebula
        ):
            if disk_chroma_ref is None:
                disk_chroma_ref = disk_chroma_from_star_layer(
                    np.maximum(0.0, stars_bg + stars_mid * 0.85 + stars_fg * 0.65),
                    disk_w,
                )
            lu_blur = _blur_separable_xy(rec709_luma(canvas), passes=4, periodic_x=wrap_lon_blur_x)
            canvas = harmonize_diffuse_canvas_chroma(
                canvas,
                disk_chroma_ref,
                disk_w,
                lu_blur,
                strength=radiance_unify * 0.75,
            )
        dof_s = float(cfg.features.depth_of_field_strength)
        if cfg.features.depth_blur_nebula:
            dof_s = float(cfg.features.depth_of_field_strength)
            dof_px = float(cfg.features.depth_of_field_max_px)
            if depth_scene is not None and dof_s > 1e-6:
                canvas = apply_depth_of_field(
                    canvas,
                    np.clip(depth_scene * 0.58, 0.0, 1.0),
                    strength=dof_s * 0.28,
                    max_sigma_px=dof_px * 0.45,
                    periodic_x=wrap_lon_blur_x,
                )
        yy = np.linspace(-1.0, 1.0, cfg.height)[:, None]
        band = np.exp(-((yy**2) / 0.55))
        sky_w = np.clip(1.0 - band, 0.0, 1.0) ** 0.38
        if cfg.features.long_exposure_look:
            canvas = _apply_long_exposure_look(
                canvas,
                rng_post,
                sky_w=sky_w,
                band=band,
                disk_w=disk_w,
                vignette_strength=cfg.features.vignette_strength,
            )
        env_scale = 0.68 if stars_above_nebula else 1.0
        canvas = _apply_galactic_disk_luminance_envelope(
            canvas, rng_post, disk_w=disk_w, strength_scale=env_scale
        )
        if core_void_mask is not None:
            canvas = np.maximum(
                0.0,
                canvas * (0.88 + 0.12 * core_void_mask[..., np.newaxis]),
            )
        noise_stage = cfg.features.sensor_noise_stage
        shot_n = cfg.features.sensor_shot_noise_scale
        read_n = cfg.features.sensor_read_noise_sigma
        density_sharp = grade_neb_luma
        if density_sharp is None:
            density_sharp = np.clip(disk_w, 0.0, 1.0)
        elif ext_paint_for_fg is not None:
            density_sharp = np.clip(
                grade_neb_luma * 0.72 + ext_paint_for_fg * 0.28,
                0.0,
                1.0,
            )
        canvas = apply_masked_density_sharpen(
            canvas,
            disk_w,
            density_sharp,
            sigma_px=cfg.features.faint_unsharp_sigma_px,
            amp_faint=cfg.features.faint_unsharp_amp,
            amp_midplane=cfg.features.midplane_unsharp_amp,
            knee=cfg.features.faint_unsharp_luma_knee,
            periodic_x=wrap_lon_blur_x,
        )
        yy_z = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float64)[:, None]
        zodiac_plane = np.exp(-((yy_z**2) / 0.040))
        if disk_w is not None:
            zodiac_plane = zodiac_plane * np.clip(disk_w, 0.0, 1.0) ** 0.72
        if core_void_mask is not None:
            zodiac_plane = zodiac_plane * core_void_mask
        zodiac_rgb = np.array([0.0095, 0.0090, 0.0082], dtype=np.float64)
        canvas = np.maximum(0.0, canvas + (zodiac_plane * 0.14)[..., None] * zodiac_rgb)
        canvas = _band_micro_ripple(
            canvas,
            rng_post,
            disk_w,
            strength=0.0042,
            neb_luma=grade_neb_luma,
        )
        if disk_w is not None and cfg.nebula_mode == NebulaMode.galaxy_streak:
            # #region agent log
            _lu_pre = rec709_luma(np.maximum(canvas, 0.0))
            _dwf = np.clip(disk_w, 0.0, 1.0)
            if _dwf.ndim == 1:
                _dwf = _dwf[:, None]
            _bm = _dwf[:, 0] > 0.35
            _pre_fin: dict[str, float] = {}
            if bool(np.any(_bm)):
                _b = _lu_pre[_bm]
                _pre_fin = {
                    "band_p50": float(np.percentile(_b, 50)),
                    "band_p95": float(np.percentile(_b, 95)),
                    "band_std": float(np.std(_b)),
                }
            if ext_paint_for_fg is not None and bool(np.any(_bm)):
                _pre_fin["ext_mean_band"] = float(
                    np.mean(np.clip(ext_paint_for_fg, 0.0, 1.0)[_bm])
                )
            _dbg_log(
                "F",
                "generator.py:pre_finalize",
                "linear canvas before finalize",
                _pre_fin,
            )
            # #endregion
            canvas = finalize_band_linear_hdr(
                canvas,
                disk_w,
                knee=0.42,
                compress=1.15,
                peak_percentile=99.2,
                peak_target=float(cfg.features.band_hdr_peak_target),
                plane_luma_cap=float(cfg.features.band_plane_luma_cap),
            )
        lane_edge = None
        if ext_paint_for_fg is not None:
            lane_edge = _extinction_gradient_mag(ext_paint_for_fg, periodic_x=wrap_lon_blur_x)
        _display_kw = dict(
            neb_luma=grade_neb_luma,
            lane_edge=lane_edge,
            periodic_x=wrap_lon_blur_x,
        )
        def _inject_post_isp_noise(c: np.ndarray) -> np.ndarray:
            if noise_stage in ("linear", "both"):
                return inject_sensor_noise(
                    c,
                    rng_post,
                    shot_scale=shot_n,
                    read_sigma=read_n,
                    space="linear",
                )
            return c

        if stars_above_nebula and use_linear_grade:
            canvas = apply_galaxy_linear_grade_pipeline(
                canvas,
                disk_w,
                cfg.features,
                rng_post,
                grade_neb_luma=grade_neb_luma,
                periodic_x=wrap_lon_blur_x,
                star_overlay_pending=True,
                lane_edge=lane_edge,
                inject_noise_fn=_inject_post_isp_noise,
            )
            # #region agent log
            _dbg_log(
                "D",
                "generator.py:post_grade_pipeline",
                "after linear grade (pre-stars)",
                _dbg_band_stats(canvas, disk_w),
            )
            # #endregion
            # Nebula HDR already went through morphology extinction; re-modulating the
            # graded display canvas here washed out stars and the band (see log G).
            if (
                ext_paint_for_fg is not None
                and disk_w is not None
                and not stars_above_nebula
            ):
                _t = np.clip(np.asarray(ext_paint_for_fg, dtype=np.float64), 0.0, 1.0)
                _pl = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
                if _pl.ndim == 1:
                    _pl = _pl[:, None]
                _ext_mod = 0.26 + 0.74 * _t**0.88
                canvas = np.maximum(
                    0.0,
                    canvas
                    * (1.0 - _pl[..., np.newaxis] + _pl[..., np.newaxis] * _ext_mod[..., np.newaxis]),
                )
                _dbg_log(
                    "G",
                    "generator.py:post_grade_extinction",
                    "extinction modulation on display canvas",
                    _dbg_band_stats(canvas, disk_w),
                )
            elif ext_paint_for_fg is not None and stars_above_nebula and disk_w is not None:
                _pl = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
                if _pl.ndim == 1:
                    _pl = _pl[:, None]
                _rel = band_relative_clearance(
                    ext_paint_for_fg, disk_w, min_clear=0.12, power=0.92
                )
                _ext_mod = np.clip(0.58 + 0.42 * _rel, 0.52, 1.0)
                canvas = np.maximum(
                    0.0,
                    canvas
                    * (1.0 - _pl[..., np.newaxis] + _pl[..., np.newaxis] * _ext_mod[..., np.newaxis]),
                )
                _dbg_log(
                    "G",
                    "generator.py:post_grade_extinction",
                    "relative extinction on display canvas (stars above nebula)",
                    _dbg_band_stats(canvas, disk_w),
                )
                if not cfg.features.galactic_band_color_grade:
                    canvas = _apply_extinction_band_display_wash(
                        canvas,
                        ext_paint_for_fg,
                        disk_w,
                        strength=float(cfg.features.band_dark_patch_strength) * 1.05,
                    )
            if cfg.features.stars:
                yy_band = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float64)[:, None]
                band_gate = np.exp(-((yy_band**2) / 0.50)) * np.clip(disk_w, 0.0, 1.0)
                star_stack = np.maximum(0.0, stars_bg + stars_mid * 0.92 + stars_fg)
                if disk_w is not None:
                    plane_scale = float(cfg.features.band_star_plane_scale) * float(
                        np.clip(1.12 - 0.42 * ism_dom, 0.35, 1.0)
                    )
                    star_stack = _attenuate_stars_for_plane(
                        star_stack, disk_w, plane_scale=plane_scale
                    )
                if disk_exposure is not None and photon_unify > 1e-4:
                    star_stack = apply_shared_photon_exposure(
                        star_stack, disk_exposure, disk_w, strength=photon_unify * 0.92
                    )
                # #region agent log
                def _star_luma_stats(layer: np.ndarray, label: str) -> None:
                    lu = rec709_luma(np.maximum(layer, 0.0))
                    pos = lu[lu > 1e-6]
                    if int(pos.size) < 16:
                        return
                    band_m = band_gate > 0.22 if disk_w is not None else np.ones_like(lu, dtype=bool)
                    bp = lu[band_m]
                    _dbg_log(
                        "WASH",
                        "generator.py:star_display_path",
                        label,
                        {
                            "p50": float(np.percentile(pos, 50)),
                            "p95": float(np.percentile(pos, 95)),
                            "max": float(np.max(pos)),
                            "band_p95": float(np.percentile(bp, 95)) if bp.size > 16 else 0.0,
                            "ism_dom": float(ism_dom),
                        },
                    )

                _star_luma_stats(star_stack, "star_stack_pre_norm")
                # #endregion
                star_stack = normalize_star_stack_luma_preserve_chroma(
                    star_stack, percentile=98.8, target=0.90
                )
                # #region agent log
                _star_luma_stats(star_stack, "star_stack_post_norm")
                # #endregion
                if ext_paint_for_fg is not None and not stars_above_nebula:
                    ext_k = np.clip(np.asarray(ext_paint_for_fg, dtype=np.float64), 0.0, 1.0)
                    star_stack = star_stack * np.clip(
                        0.38 + 0.62 * ext_k**0.92, 0.28, 1.0
                    )[..., np.newaxis]
                    # #region agent log
                    _star_luma_stats(star_stack, "star_stack_post_ext")
                    # #endregion
                elif ext_paint_for_fg is not None and stars_above_nebula:
                    rel_clear = band_relative_clearance(
                        ext_paint_for_fg, disk_w, min_clear=0.12, power=0.95
                    )
                    star_stack = star_stack * np.clip(
                        0.44 + 0.56 * rel_clear, 0.38, 1.0
                    )[..., np.newaxis]
                    # #region agent log
                    _star_luma_stats(star_stack, "star_stack_post_ext_rel")
                    # #endregion
                spectral = bool(cfg.features.use_spectral_teffective)
                desat_mul = 1.0 if use_star_display_desat else 0.0
                faint_desat = float(cfg.features.faint_star_chroma_desat) * (0.5 if spectral else 1.0) * desat_mul
                band_adapt = float(cfg.features.star_band_chroma_adapt) * (0.2 if spectral else 1.0) * desat_mul
                band_desat = float(cfg.features.star_band_chroma_desat) * (0.5 if spectral else 1.0) * desat_mul
                peak_clamp = float(cfg.features.star_peak_soft_clamp) * (0.45 if spectral else 1.0)
                stretch_g = float(cfg.features.star_display_stretch_gain)
                out_gain = 1.10
                if cfg.features.split_star_match_scene_tone and radiance_unify > 1e-4:
                    stretch_g = matched_star_display_stretch_gain(
                        star_stack,
                        canvas,
                        disk_w,
                        stretch_g,
                    )
                    out_gain = float(np.clip(1.0 + 0.08 * radiance_unify, 0.92, 1.06))
                star_disp = _hdr_stars_to_display(
                    star_stack,
                    disk_w,
                    stretch_gain=stretch_g,
                    peak_percentile=cfg.features.star_stretch_peak_percentile,
                    faint_desat=faint_desat,
                    display_cap=cfg.features.star_display_white_cap,
                    output_gain=out_gain,
                )
                star_add = float(cfg.features.star_composite_add_scale) * float(
                    np.clip(1.0 - 0.16 * ism_dom, 0.72, 1.0)
                )
                star_bright = float(cfg.features.star_band_brightness_scale) * float(
                    np.clip(1.0 - 0.20 * ism_dom, 0.72, 1.0)
                )
                blend_in_band = float(np.clip(0.38 - 0.10 * ism_dom, 0.24, 0.42))
                canvas = composite_stars_over_display_canvas(
                    canvas,
                    star_disp,
                    band_gate,
                    add_scale=min(0.98, star_add),
                    band_chroma_desat=band_desat,
                    band_brightness_scale=star_bright,
                    band_chroma_adapt=band_adapt,
                    peak_soft_clamp=peak_clamp,
                    max_blend_in_band=blend_in_band,
                )
                # #region agent log
                _dbg_log(
                    "E",
                    "generator.py:post_star_composite",
                    "after star composite",
                    _dbg_band_stats(canvas, disk_w),
                )
                # #endregion
        else:
            _medium_only_grade = bool(cfg.features.galaxy_view and not cfg.features.stars)
            if cfg.features.optics_before_tone_map and use_linear_grade:
                canvas = apply_galaxy_linear_grade_pipeline(
                    canvas,
                    disk_w,
                    cfg.features,
                    rng_post,
                    grade_neb_luma=grade_neb_luma,
                    periodic_x=wrap_lon_blur_x,
                    lane_edge=lane_edge,
                    star_overlay_pending=_medium_only_grade,
                    inject_noise_fn=_inject_post_isp_noise,
                )
            else:
                if cfg.features.isp_chain_strength > 1e-6:
                    canvas = apply_isp_linear_chain(
                        canvas,
                        strength=cfg.features.isp_chain_strength,
                    )
                canvas = apply_galaxy_scene_tone(
                    canvas,
                    disk_w,
                    cfg.features,
                    grade_neb_luma,
                    periodic_x=wrap_lon_blur_x,
                )
                canvas = _inject_post_isp_noise(canvas)
                if use_display_finish:
                    canvas = apply_galaxy_display_finish(
                        canvas,
                        disk_w,
                        cfg.features,
                        rng_post,
                        star_overlay_pending=_medium_only_grade,
                        **_display_kw,
                    )
        if (
            bool(cfg.features.off_band_late_composite)
            and float(cfg.features.off_band_emission_strength) > 1e-6
        ):
            lu_pre = rec709_luma(canvas)
            _band_pre_hii = _dbg_band_stats(canvas, disk_w)
            _late_band_hii = band_hii_layer
            if bool(cfg.features.galactic_band_color_grade):
                _late_band_hii = None
            canvas = _composite_red_hii_late(
                canvas,
                off_band_layer,
                _late_band_hii,
                strength=float(cfg.features.off_band_emission_strength),
                periodic_x=wrap_lon_blur_x,
                disk_w=disk_w,
            )
            # #region agent log
            _post = _dbg_band_stats(canvas, disk_w)
            _dbg_log(
                "E",
                "generator.py:post_late_hii",
                "late H II composite delta",
                {
                    "pre": _band_pre_hii,
                    "post": _post,
                    "band_p50_delta": _post.get("band_p50", 0.0)
                    - _band_pre_hii.get("band_p50", 0.0),
                },
            )
            # #endregion
        if use_display_finish and cfg.features.film_grain_strength > 1e-6:
            canvas = apply_film_grain_display(
                canvas,
                rng_post,
                cfg.features.film_grain_strength,
                periodic_x=wrap_lon_blur_x,
            )
        if (
            cfg.features.galaxy_view
            and disk_w is not None
            and cfg.nebula_mode == NebulaMode.galaxy_streak
        ):
            canvas = apply_band_display_highlight_cap(
                canvas,
                disk_w,
                knee=0.40,
                cap=float(cfg.features.band_display_peak_cap),
            )
        if (
            cfg.features.galactic_band_color_grade
            and cfg.features.galaxy_view
            and disk_w is not None
            and ext_paint_for_fg is not None
            and cfg.nebula_mode == NebulaMode.galaxy_streak
        ):
            _hii_hint = None
            if band_hii_layer is not None and float(np.max(band_hii_layer)) > 1e-8:
                _hii_hint = rec709_luma(np.maximum(band_hii_layer, 0.0))
            _dust_m = None
            _sf_m = None
            _void_m = None
            _turb_m = None
            if galactic is not None:
                _dust_m = galactic.dust_absorption_morph
                _sf_m = galactic.star_formation
                _void_m = galactic.void_mask
                _turb_m = galactic.latent_turb
            canvas = apply_galactic_band_color_grade(
                canvas,
                disk_w,
                ext_paint_for_fg,
                dust_absorption=_dust_m,
                star_formation=_sf_m,
                void_mask=_void_m,
                latent_turb=_turb_m,
                hii_hint=_hii_hint,
                strength=float(cfg.features.galactic_band_color_grade_strength),
                dust_black_strength=float(cfg.features.galactic_band_dust_black_strength),
                gas_fluff_strength=float(cfg.features.galactic_band_gas_fluff_strength),
                micro_display_strength=float(
                    cfg.features.galactic_band_micro_display_strength
                ),
                separation_strength=0.0,
                gas_texture=morph_gas_struct,
                periodic_x=wrap_lon_blur_x,
                blur_fn=_blur_separable_xy,
                rng=rng_post,
            )
            if morph_gas_struct is not None and disk_w is not None:
                canvas = _reinforce_morph_canvas_texture(
                    canvas,
                    morph_gas_struct,
                    disk_w,
                    plane_gate=plane_gate,
                    strength=1.35,
                )
            if ext_paint_for_fg is not None and disk_w is not None:
                canvas = _sculpt_morphology_lane_contrast(
                    canvas,
                    ext_paint_for_fg,
                    disk_w,
                    plane_gate=plane_gate,
                    strength=1.02,
                )
                canvas = apply_band_luma_separation(
                    canvas,
                    ext_paint_for_fg,
                    disk_w,
                    dust_absorption=_dust_m,
                    latent_turb=_turb_m,
                    morph_gas=morph_gas_struct,
                    strength=float(cfg.features.galactic_band_separation_strength),
                    periodic_x=wrap_lon_blur_x,
                )
                canvas = apply_band_display_microstructure(
                    canvas,
                    ext_paint_for_fg,
                    disk_w,
                    dust_absorption=_dust_m,
                    latent_turb=_turb_m,
                    morph_gas=morph_gas_struct,
                    strength=float(cfg.features.galactic_band_micro_display_strength)
                    * 0.55,
                    periodic_x=wrap_lon_blur_x,
                )
        # #region agent log
        _dbg_log(
            "D",
            "generator.py:pre_save",
            "final canvas before clip",
            _dbg_band_stats(canvas, disk_w),
        )
        if core_void_mask is not None:
            _cr = int(canvas.shape[0] // 2)
            _dbg_log(
                "C",
                "generator.py:pre_save",
                "core_void_mask on center row",
                {
                    "void_row_min": float(np.min(core_void_mask[_cr, :])),
                    "void_row_mean": float(np.mean(core_void_mask[_cr, :])),
                    "void_row_max": float(np.max(core_void_mask[_cr, :])),
                },
            )
        if galactic is not None and galactic.disk_thickness_modulation is not None:
            _meso = np.asarray(galactic.disk_thickness_modulation, dtype=np.float64)
            _cr = int(canvas.shape[0] // 2)
            _dbg_log(
                "C",
                "generator.py:pre_save",
                "mesoscale thickness on center row",
                {
                    "meso_row_min": float(np.min(_meso[_cr, :])),
                    "meso_row_max": float(np.max(_meso[_cr, :])),
                    "meso_row_mean": float(np.mean(_meso[_cr, :])),
                },
            )
        # #endregion
        if disk_w is not None:
            _sky_crush = np.clip(1.0 - np.clip(disk_w, 0.0, 1.0) ** 1.12, 0.0, 1.0)
            if _sky_crush.ndim == 1:
                _sky_crush = _sky_crush[:, None]
            canvas = np.maximum(
                0.0,
                canvas * (1.0 - _sky_crush[..., np.newaxis] * 0.92),
            )
        # #region agent log
        import os as _os_abl

        if _os_abl.environ.get("STK_ABLATION") and disk_w is not None:
            _dbg_hist("canvas_pre_save", canvas, disk_w, hypothesis_id="BG")
        # #endregion
        canvas = ensure_hdr(np.clip(canvas, 0.0, 1.0))
        _notify_stage("grade/color", 1.0)

    if not cfg.features.galaxy_view:
        canvas = ensure_hdr(np.clip(canvas, 0.0, 1.0))

    if cfg.features.jpeg_artifact_pass and cfg.output_format == "jpg":
        canvas = apply_jpeg_artifacts(canvas, cfg.quality)
        if cfg.features.jpeg_highlight_smooth > 1e-6:
            canvas = smooth_jpeg_highlight_artifacts(
                canvas,
                strength=cfg.features.jpeg_highlight_smooth,
                blur_fn=_blur_separable_xy,
                periodic_x=wrap_lon_blur_x if cfg.wrap_safe else False,
            )
        _notify_stage("jpeg artifacts", 1.0)

    if cfg.wrap_safe:
        canvas = _enforce_horizontal_wrap(canvas)

    saved: dict[str, Path] = {}
    ext = cfg.output_format.value
    base_name = f"{cfg.output_base_name}_{generation_index:04d}"

    if cfg.projection_mode in {ProjectionMode.equirectangular, ProjectionMode.both}:
        eq_path = cfg.output_dir / f"{base_name}_equirect.{ext}"
        _save_image(
            canvas,
            eq_path,
            ext,
            cfg.quality,
            dither_strength=cfg.features.blue_noise_dither_strength,
        )
        saved["equirectangular"] = eq_path
        _notify_stage("save equirectangular", 1.0)

    if cfg.features.galaxy_view and (
        cfg.features.debug_export_layers or cfg.features.debug_grayscale_morphology
    ):
        _export_morphology_debug_layers(
            cfg,
            base_name,
            galactic=galactic,
            stars_bg=stars_bg,
            stars_mid=stars_mid,
            stars_fg=stars_fg if cfg.features.stars else np.zeros_like(canvas),
            ext_paint=ext_paint_for_fg,
            rng_post=rng_post,
        )

    if cfg.projection_mode in {ProjectionMode.cubemap, ProjectionMode.both}:
        faces = cubemap_faces_from_equirect(canvas, cfg.cubemap_face_size)
        _notify_stage("project cubemap", 1.0)
        for face_name, face_img in faces.items():
            face_path = cfg.output_dir / f"{base_name}_cube_{face_name}.{ext}"
            _save_image(
                face_img,
                face_path,
                ext,
                cfg.quality,
                dither_strength=cfg.features.blue_noise_dither_strength,
            )
            _notify_stage(f"save cube {face_name}", 1.0)
        saved["cubemap"] = cfg.output_dir


    return saved, stats
