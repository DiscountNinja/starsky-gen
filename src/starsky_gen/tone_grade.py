"""Scene-linear tone map + display finish for galaxy_view grading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from starsky_gen.color_science import rec709_luma
from starsky_gen.composite_blend import (
    recompute_hdr_asinh_gain,
    reinhard_luma_preserving,
    remap_luma_preserving_chroma,
)
from starsky_gen.grade import apply_atmospheric_scatter, apply_dodge_burn_lanes
from starsky_gen.optical_effects import apply_isp_linear_chain, apply_lens_vignette, apply_optical_display_pass
from starsky_gen.nebula import _blur_separable_xy
from starsky_gen.postfx import (
    apply_band_display_highlight_cap,
    apply_display_contrast_finish,
    apply_global_s_curve,
    apply_god_rays,
    apply_local_contrast,
    apply_shadow_lift,
    apply_split_toning,
    apply_tri_scale_bloom,
)

from starsky_gen.tone_map import (
    apply_acescct_cinematic_grade,
    asinh_linear_stretch_luma,
    disk_filmic_rgb_grade,
    disk_reinhard_rgb_grade,
)

if TYPE_CHECKING:
    from starsky_gen.config import FeatureConfig


def mild_disk_asinh_grade(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    *,
    gain: float,
    q: float = 1.0,
    curvature: float = 0.70,
    midtone_exposure: float = 1.0,
    toe_strength: float = 0.0,
) -> np.ndarray:
    """Mild linear asinh on disk luma (gain ≈ 0.02–0.2, Q ≈ 0.5–1.5)."""
    w = np.broadcast_to(disk_w, rgb.shape[:2])[..., np.newaxis]
    lin = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lu = rec709_luma(lin)
    gg = float(max(gain * max(curvature, 0.25), 1e-4))
    l_new = asinh_linear_stretch_luma(
        lu,
        gain=gg,
        q=q,
        midtone_exposure=midtone_exposure,
        toe_strength=toe_strength,
    )
    l_new = l_new / (1.0 + 0.24 * np.clip(l_new - 0.32, 0.0, None) ** 2.0)
    toned = remap_luma_preserving_chroma(lin, l_new)
    return np.maximum(rgb * (1.0 - w) + toned * w, 0.0)


def apply_localized_core_scurve_dodge_burn(
    rgb: np.ndarray,
    neb_luma: np.ndarray,
    disk_w: np.ndarray,
    *,
    scurve_strength: float = 0.20,
    dodge_strength: float = 0.06,
    burn_strength: float = 0.10,
    radius_passes: int = 2,
    periodic_x: bool = True,
) -> np.ndarray:
    """Small-radius S-curve dodge/burn on the bulge core (depth), not global contrast."""
    sc = float(scurve_strength)
    bd = float(burn_strength)
    dd = float(dodge_strength)
    if sc < 1e-6 and bd < 1e-6 and dd < 1e-6:
        return rgb
    lin = np.maximum(rgb.astype(np.float64), 0.0)
    lu = rec709_luma(lin)
    nl = np.clip(neb_luma, 0.0, 1.0)
    dw = np.clip(disk_w, 0.0, 1.0)
    core = np.clip((nl - 0.30) / 0.58, 0.0, 1.0) ** 1.12 * dw
    passes = int(np.clip(radius_passes, 1, 6))
    core_soft = _blur_separable_xy(core, passes=passes, periodic_x=periodic_x)
    rim = np.clip(_blur_separable_xy(core, passes=passes + 1, periodic_x=periodic_x) - core, 0.0, 1.0)
    # Localized S-curve on luma inside core mask.
    t = lu - 0.5
    l_curve = np.clip(0.5 + t * (1.0 + sc * (0.26 - t * t)), 0.0, 1.0)
    m = core_soft[..., np.newaxis]
    lu_out = lu * (1.0 - core_soft) + l_curve * core_soft
    out = remap_luma_preserving_chroma(lin, lu_out)
    # Burn under brightest core; mild rim dodge (global dodge reduced vs earlier builds).
    peak = np.clip((lu - 0.55) / 0.35, 0.0, 1.0) * core
    out = out * (1.0 - peak[..., np.newaxis] * bd * 0.20)
    out = out + rim[..., np.newaxis] * dd * 0.028
    return np.clip(lin * (1.0 - m) + out * m, 0.0, 1.0)


def apply_neutral_white_balance(
    rgb: np.ndarray,
    *,
    strength: float = 0.42,
) -> np.ndarray:
    """Gray-world nudge toward neutral (subtle); avoids strong warm/cool cast before split-tone."""
    s = float(strength)
    if s < 1e-6:
        return rgb
    lin = np.maximum(rgb.astype(np.float64), 0.0)
    mean = np.mean(lin, axis=(0, 1))
    gray = float(np.mean(mean))
    target = np.array([gray, gray, gray], dtype=np.float64)
    corr = 1.0 + (target / (mean + 1e-6) - 1.0) * s
    return np.clip(lin * corr.reshape(1, 1, 3), 0.0, 1.0)


def apply_galaxy_scene_tone(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    features: FeatureConfig,
    neb_luma: np.ndarray | None,
    *,
    periodic_x: bool,
) -> np.ndarray:
    """Scene-referred tone: mild HDR compression + localized core grade (no global punch)."""
    canvas = np.maximum(rgb.astype(np.float64), 0.0)
    h, w_img = canvas.shape[:2]
    dw = np.broadcast_to(disk_w, (h, w_img))

    if features.galaxy_tone_curve == "asinh":
        tone_gain = recompute_hdr_asinh_gain(
            canvas,
            dw,
            features.asinh_stretch_gain,
        )
        canvas = mild_disk_asinh_grade(
            canvas,
            dw,
            gain=tone_gain,
            q=features.asinh_stretch_q,
            curvature=features.disk_asinh_curvature,
            midtone_exposure=features.asinh_midtone_exposure,
            toe_strength=features.asinh_toe_strength,
        )
    elif features.galaxy_tone_curve == "filmic":
        canvas = disk_filmic_rgb_grade(canvas, dw, shoulder=features.filmic_shoulder)
    elif features.galaxy_tone_curve == "aces":
        from starsky_gen.tone_map import tone_map_aces_rgb

        canvas = tone_map_aces_rgb(
            canvas,
            exposure=features.aces_exposure,
            disk_weight=dw,
        )
    elif features.galaxy_tone_curve == "acescct":
        canvas = apply_acescct_cinematic_grade(
            canvas,
            dw,
            strength=features.acescct_grade_strength,
            shoulder=features.acescct_shoulder,
        )
    else:
        canvas = disk_reinhard_rgb_grade(canvas, dw, k=0.34)

    # Local core grade before global filmic / reinhard rolloff.
    if neb_luma is not None:
        canvas = apply_localized_core_scurve_dodge_burn(
            canvas,
            neb_luma,
            dw,
            scurve_strength=features.core_local_scurve_strength,
            dodge_strength=features.core_dodge_strength,
            burn_strength=features.core_burn_strength,
            radius_passes=features.core_local_scurve_radius_passes,
            periodic_x=periodic_x,
        )
        if features.core_clahe_strength > 1e-6:
            from starsky_gen.postfx import apply_core_local_contrast

            core_mask = np.clip((neb_luma - 0.28) / 0.62, 0.0, 1.0) * dw
            canvas = apply_core_local_contrast(
                canvas,
                core_mask,
                strength=features.core_clahe_strength,
                periodic_x=periodic_x,
            )

    rein_k = 0.38
    if float(getattr(features, "band_ism_dominance", 0.0)) > 0.95:
        rein_k = 0.44
    canvas = reinhard_luma_preserving(canvas, k=rein_k)
    lu = rec709_luma(canvas)
    hi = np.clip((lu - 0.40) / 0.38, 0.0, 1.0) ** 1.15
    comp = np.maximum(canvas, 0.0) ** 1.22
    m = (dw * hi)[..., np.newaxis]
    canvas = np.maximum(canvas * (1.0 - m) + comp * m, 0.0)
    # #region agent log
    import json
    import time
    from pathlib import Path

    _lp = Path(__file__).resolve().parents[2] / ".cursor" / "debug-408793.log"
    _bm = np.clip(dw, 0, 1) > 0.35
    if bool(np.any(_bm)):
        _b = lu[_bm]
        try:
            with _lp.open("a", encoding="utf-8") as _fh:
                _fh.write(
                    json.dumps(
                        {
                            "sessionId": "408793",
                            "hypothesisId": "H",
                            "location": "tone_grade.py:apply_galaxy_scene_tone",
                            "message": "post tone map band stats",
                            "data": {
                                "p50": float(np.percentile(_b, 50)),
                                "p95": float(np.percentile(_b, 95)),
                                "std": float(np.std(_b)),
                            },
                            "timestamp": int(time.time() * 1000),
                            "runId": __import__("os").environ.get("STK_DEBUG_RUN", "pre"),
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
    # #endregion
    return canvas


def apply_galaxy_linear_grade_pipeline(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    features: FeatureConfig,
    rng: np.random.Generator,
    *,
    grade_neb_luma: np.ndarray | None,
    periodic_x: bool = True,
    star_overlay_pending: bool = False,
    lane_edge: np.ndarray | None = None,
    inject_noise_fn=None,
) -> np.ndarray:
    """Linear scene → optional lens vignette → tone map → ISP → noise → display finish."""
    canvas = np.maximum(rgb.astype(np.float64), 0.0)
    if features.optics_before_tone_map and float(features.vignette_strength) > 1e-6:
        canvas = apply_lens_vignette(
            canvas, strength=float(features.vignette_strength) * 0.42
        )
    canvas = apply_galaxy_scene_tone(
        canvas,
        disk_w,
        features,
        grade_neb_luma,
        periodic_x=periodic_x,
    )
    isp_s = float(features.isp_chain_strength)
    if star_overlay_pending:
        isp_s *= 0.28
    if isp_s > 1e-6:
        canvas = apply_isp_linear_chain(
            canvas,
            strength=isp_s,
        )
    if inject_noise_fn is not None:
        canvas = inject_noise_fn(canvas)
    return apply_galaxy_display_finish(
        canvas,
        disk_w,
        features,
        rng,
        neb_luma=grade_neb_luma,
        lane_edge=lane_edge,
        periodic_x=periodic_x,
        star_overlay_pending=star_overlay_pending,
    )


def apply_galaxy_display_finish(
    rgb: np.ndarray,
    disk_w: np.ndarray,
    features: FeatureConfig,
    rng: np.random.Generator,
    *,
    neb_luma: np.ndarray | None = None,
    lane_edge: np.ndarray | None = None,
    periodic_x: bool = True,
    star_overlay_pending: bool = False,
) -> np.ndarray:
    """Display finish: grade → optical/sensor (post tone-map) → bloom → B/W stretch.

    When ``star_overlay_pending``, skip strong global contrast (stars composited after).
    """
    canvas = np.maximum(rgb.astype(np.float64), 0.0)
    h, w_img = canvas.shape[:2]
    dw = np.broadcast_to(disk_w, (h, w_img))

    canvas = apply_neutral_white_balance(canvas, strength=features.neutral_wb_strength)
    canvas = apply_split_toning(
        canvas,
        strength=features.split_toning_strength,
        warm_high=(1.01, 1.005, 1.03),
        cool_shadow=(0.99, 0.995, 1.02),
    )
    lift = float(features.shadow_lift)
    if star_overlay_pending:
        lift *= 0.35
    canvas = apply_shadow_lift(canvas, lift=lift)
    scurve = 0.0 if star_overlay_pending else features.filmic_s_curve_strength
    loc_ctr = 0.0 if star_overlay_pending else features.local_contrast_strength * 0.40
    if scurve > 1e-6:
        canvas = apply_global_s_curve(canvas, strength=scurve)
    if loc_ctr > 1e-6:
        canvas = apply_local_contrast(
            canvas,
            strength=loc_ctr,
            periodic_x=periodic_x,
        )
    scatter = 0.018 if not star_overlay_pending else 0.0
    if scatter > 1e-6:
        canvas = apply_atmospheric_scatter(canvas, strength=scatter)
    canvas = apply_optical_display_pass(
        canvas,
        dw,
        features,
        rng,
        periodic_x=periodic_x,
    )
    if lane_edge is not None:
        lane_amp = float(features.lane_contrast_amp) * (1.65 if star_overlay_pending else 1.0)
        canvas = apply_dodge_burn_lanes(canvas, lane_edge, amp=lane_amp)
    bloom_s = float(features.display_bloom_strength)
    if star_overlay_pending:
        bloom_s *= 0.22
        if neb_luma is not None:
            canvas = apply_localized_core_scurve_dodge_burn(
                canvas,
                neb_luma,
                dw,
                scurve_strength=features.core_local_scurve_strength * 0.55,
                dodge_strength=features.core_dodge_strength * 0.35,
                burn_strength=features.core_burn_strength * 0.85,
                radius_passes=max(2, features.core_local_scurve_radius_passes),
                periodic_x=periodic_x,
            )
    canvas = apply_tri_scale_bloom(
        canvas,
        dw,
        strength=bloom_s,
        threshold=features.bloom_threshold,
        mix_tight=features.bloom_mix_tight,
        mix_mid=features.bloom_mix_mid,
        mix_wide=features.bloom_mix_wide,
        neb_luma=neb_luma,
        periodic_x=periodic_x,
    )
    if neb_luma is not None and features.god_rays_strength > 1e-5:
        hot = np.clip((neb_luma - 0.55) / 0.35, 0.0, 1.0) ** 1.2
        canvas = apply_god_rays(
            canvas,
            hot,
            dw,
            strength=features.god_rays_strength,
            periodic_x=periodic_x,
        )
    lu_fin = rec709_luma(canvas)
    hot_spec = np.clip((lu_fin - 0.70) / 0.28, 0.0, 1.0) ** 1.15
    desat_amt = float(np.clip(features.highlight_chroma_desat, 0.0, 0.14))
    if star_overlay_pending:
        desat_amt *= 0.45
    luma_g = np.mean(canvas, axis=2, keepdims=True)
    chroma_scale = 0.992 - hot_spec * desat_amt
    canvas = np.maximum(0.0, luma_g + (canvas - luma_g) * chroma_scale[..., np.newaxis])
    positive = lu_fin[lu_fin > 1e-6]
    if positive.size > 64 and desat_amt > 1e-6:
        tail = float(np.percentile(positive, 99.95))
        ultra = np.clip((lu_fin - tail * 0.94) / max(tail * 0.06, 1e-6), 0.0, 1.0) ** 1.1
        canvas = canvas * (1.0 - ultra[..., np.newaxis] * desat_amt * 0.55) + luma_g * (
            ultra[..., np.newaxis] * desat_amt * 0.55
        )
    if features.final_acescct_rolloff > 1e-6 and features.galaxy_tone_curve != "acescct":
        canvas = apply_acescct_cinematic_grade(
            canvas,
            dw,
            strength=features.final_acescct_rolloff,
            shoulder=features.acescct_shoulder,
        )
    cap_peak = float(getattr(features, "band_display_peak_cap", 0.58))
    cap_knee = 0.40 if star_overlay_pending else 0.44
    canvas = apply_band_display_highlight_cap(
        canvas, dw, knee=cap_knee, cap=cap_peak
    )
    if star_overlay_pending:
        return apply_display_contrast_finish(
            canvas,
            dw,
            black_point=features.display_black_point * 0.65,
            white_point=min(0.94, features.display_white_point + 0.01),
            sky_darken=min(0.92, features.sky_darken_strength * 1.12),
        )
    return apply_display_contrast_finish(
        canvas,
        dw,
        black_point=features.display_black_point,
        white_point=features.display_white_point,
        sky_darken=features.sky_darken_strength,
    )
