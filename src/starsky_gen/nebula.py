from __future__ import annotations

from typing import Callable

import numpy as np

from starsky_gen import procedural_noise as _pn
from starsky_gen.color_science import blend_darken_preserve_contrast
from starsky_gen.config import FeatureConfig, NebulaMode, NebulaTuningConfig
from starsky_gen.nebula_physics import (
    build_hii_compact_mask,
    build_reflection_diffuse_mask,
    compose_line_emission_rgb,
    compose_reflection_continuum_rgb,
    forward_scatter_mie,
)
from starsky_gen.placement import galactic_midplane_mask


def _wrap_dx(x: np.ndarray, a: float | np.ndarray) -> np.ndarray:
    """Signed longitude difference on [-1,1] with x=-1 and x=+1 identified (equirect wrap)."""
    da = np.asanyarray(a, dtype=np.float64)
    return np.mod(x - da + 1.0, 2.0) - 1.0


def _wrap_x11_scalar(v: float) -> float:
    """Wrap scalar longitude from any real value into [-1,1)."""
    return float(np.mod(v + 1.0, 2.0) - 1.0)


def _resize_bilinear(field: np.ndarray, out_h: int, out_w: int, *, periodic_x: bool = False) -> np.ndarray:
    """Resize a 2D array with separable linear interpolation.

    When ``periodic_x`` is True, the first and last source columns are treated as
    adjacent on a torus so equirectangular longitude wraps without a seam.
    """
    src_h, src_w = field.shape
    y_src = np.arange(src_h, dtype=np.float64)
    x_src = np.arange(src_w, dtype=np.float64)
    y_dst = np.linspace(0.0, src_h - 1, out_h, dtype=np.float64)
    x_dst = np.linspace(0.0, src_w - 1, out_w, dtype=np.float64)

    if periodic_x and src_w >= 2:
        field_x = np.concatenate([field, field[:, :1]], axis=1)
        x_src = np.arange(src_w + 1, dtype=np.float64)
        src_w_eff = src_w + 1
    else:
        field_x = field
        src_w_eff = src_w

    row_interp = np.empty((out_h, src_w_eff), dtype=np.float64)
    for x in range(src_w_eff):
        row_interp[:, x] = np.interp(y_dst, y_src, field_x[:, x])

    out = np.empty((out_h, out_w), dtype=np.float64)
    for y in range(out_h):
        out[y, :] = np.interp(x_dst, x_src, row_interp[y, :])
    return out


def _blur_separable_xy(
    field: np.ndarray, passes: int = 1, *, alternate: bool = True, periodic_x: bool = False
) -> np.ndarray:
    """Small separable blur; alternate pass order (x/y vs y/x) to avoid row/column streak buildup."""
    out = np.clip(field.astype(np.float64, copy=False), 0.0, 1.0)
    k0, k1, k2 = 0.22, 0.56, 0.22
    x_mode = "wrap" if periodic_x else "edge"
    for pi in range(passes):
        y_first = alternate and (pi % 2 == 1)
        if y_first:
            padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
            out = k0 * padded[:-2, :] + k1 * padded[1:-1, :] + k2 * padded[2:, :]
            padded = np.pad(out, ((0, 0), (1, 1)), mode=x_mode)
            out = k0 * padded[:, :-2] + k1 * padded[:, 1:-1] + k2 * padded[:, 2:]
        else:
            padded = np.pad(out, ((0, 0), (1, 1)), mode=x_mode)
            out = k0 * padded[:, :-2] + k1 * padded[:, 1:-1] + k2 * padded[:, 2:]
            padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
            out = k0 * padded[:-2, :] + k1 * padded[1:-1, :] + k2 * padded[2:, :]
    return np.clip(out, 0.0, 1.0)


def _blur_rgb_separable_xy(rgb: np.ndarray, passes: int = 2, *, periodic_x: bool = False) -> np.ndarray:
    """Separable blur on each channel (gas RGB is 3D; _blur_separable_xy is 2D-only)."""
    return np.stack(
        [_blur_separable_xy(rgb[:, :, c], passes=passes, periodic_x=periodic_x) for c in range(3)],
        axis=2,
    )


def _detail_blend_after_blur(
    rgb: np.ndarray,
    *,
    passes: int = 3,
    detail_mix: float = 0.50,
    chroma_boost: float = 1.18,
    periodic_x: bool = False,
) -> np.ndarray:
    """Blur for cohesion, then blend mid/high frequencies back (avoids airbrushed gas)."""
    soft = _blur_rgb_separable_xy(rgb, passes=passes, periodic_x=periodic_x)
    wide = _blur_rgb_separable_xy(rgb, passes=max(2, passes - 1), periodic_x=periodic_x)
    high = rgb - wide
    out = np.clip(soft + high * float(detail_mix), 0.0, 1.0)
    lm = np.mean(out, axis=2, keepdims=True)
    return np.clip(lm + (out - lm) * float(chroma_boost), 0.0, 1.0)


def _blur_y_only(field: np.ndarray, passes: int = 2) -> np.ndarray:
    """Vertical-only blur to soften comb-like edges after strong horizontal stretching."""
    out = np.clip(field.astype(np.float64, copy=False), 0.0, 1.0)
    k0, k1, k2 = 0.15, 0.70, 0.15
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = k0 * padded[:-2, :] + k1 * padded[1:-1, :] + k2 * padded[2:, :]
    return np.clip(out, 0.0, 1.0)


def _blur_x_only(field: np.ndarray, passes: int = 2, *, periodic_x: bool = False) -> np.ndarray:
    """Horizontal-only blur to merge column-coherent dust into cloud-like masses."""
    out = np.clip(field.astype(np.float64, copy=False), 0.0, 1.0)
    k0, k1, k2 = 0.15, 0.70, 0.15
    x_mode = "wrap" if periodic_x else "edge"
    for _ in range(passes):
        padded = np.pad(out, ((0, 0), (1, 1)), mode=x_mode)
        out = k0 * padded[:, :-2] + k1 * padded[:, 1:-1] + k2 * padded[:, 2:]
    return np.clip(out, 0.0, 1.0)


def _smooth_noise(
    rng: np.random.Generator,
    height: int,
    width: int,
    octaves: int = 4,
    *,
    periodic_x: bool = False,
) -> np.ndarray:
    noise = np.zeros((height, width), dtype=np.float64)
    amp_sum = 0.0
    for o in range(octaves):
        scale = 2 ** (o + 2)
        sample_h = max(2, height // scale)
        sample_w = max(2, width // scale)
        coarse = rng.random((sample_h, sample_w))
        layer = _resize_bilinear(coarse, height, width, periodic_x=periodic_x)
        amp = 0.6 ** o
        noise += layer * amp
        amp_sum += amp
    return noise / max(amp_sum, 1e-6)


def _contrast_curve(field: np.ndarray, low: float, high: float, gamma: float) -> np.ndarray:
    """Normalize and reshape contrast with a soft gamma curve."""
    t = np.clip((field - low) / max(high - low, 1e-6), 0.0, 1.0)
    return t**gamma


def _structural_lane_scaffold(
    rng: np.random.Generator,
    x: np.ndarray,
    y: np.ndarray,
    band_gate: np.ndarray,
    *,
    sign: float,
    lane_count: int = 3,
    periodic_x: bool = False,
) -> np.ndarray:
    """Build coherent dust-lane spines with path samples (distance-field blend)."""
    scaffold = np.zeros_like(band_gate, dtype=np.float64)
    n_samples = max(48, int(x.shape[1] * 0.20))
    t = np.linspace(-1.0, 1.0, n_samples, dtype=np.float64)

    for _ in range(lane_count):
        phase = float(rng.uniform(0.0, 6.283185307179586))
        phase2 = float(rng.uniform(0.0, 6.283185307179586))
        y0 = float(rng.normal(0.0, 0.10))
        slope = float(rng.uniform(-0.30, 0.30)) * float(sign)
        amp1 = float(rng.uniform(0.040, 0.105))
        amp2 = float(rng.uniform(0.020, 0.070))
        f1 = float(rng.uniform(0.9, 1.8))
        f2 = float(rng.uniform(1.9, 3.6))
        x_wig = float(rng.uniform(0.03, 0.16))
        y_width = float(rng.uniform(0.038, 0.092))

        x_path = np.clip(t + x_wig * np.sin(2.0 * t + phase2), -1.08, 1.08)
        y_path = (
            y0
            + slope * t
            + amp1 * np.sin(f1 * 3.14159 * t + phase)
            + amp2 * np.sin(f2 * 3.14159 * t + phase2)
        )
        y_path = np.clip(y_path, -0.45, 0.45)

        lane = np.zeros_like(scaffold)
        for i in range(n_samples):
            px = float(x_path[i])
            py = float(y_path[i])
            # Broaden points in x and keep y narrow to produce long, coherent trenches.
            wx = float(rng.uniform(0.040, 0.070))
            wy = y_width * float(rng.uniform(0.88, 1.22))
            dx = _wrap_dx(x, px) if periodic_x else (x - px)
            d = (dx / wx) ** 2 + ((y - py) / wy) ** 2
            lane = np.maximum(lane, np.exp(-d))

        lane = _blur_x_only(
            _blur_separable_xy(lane, passes=2, periodic_x=periodic_x),
            passes=2,
            periodic_x=periodic_x,
        )
        # Branch/break map avoids parallel uninterrupted bars.
        branch = _contrast_curve(
            _smooth_noise(rng, x.shape[0], x.shape[1], octaves=3, periodic_x=periodic_x),
            0.60,
            0.99,
            1.45,
        )
        branch = _blur_separable_xy(branch, passes=1, periodic_x=periodic_x)
        lane *= 0.72 + 0.28 * branch
        lane = np.clip(_contrast_curve(lane, 0.14, 0.995, 1.18), 0.0, 1.0)
        scaffold = np.maximum(scaffold, lane)

    scaffold = np.clip(scaffold * band_gate, 0.0, 1.0)
    scaffold = _blur_x_only(
        _blur_separable_xy(scaffold, passes=2, periodic_x=periodic_x),
        passes=3,
        periodic_x=periodic_x,
    )
    breakup = _contrast_curve(
        _smooth_noise(rng, x.shape[0], x.shape[1], octaves=2, periodic_x=periodic_x),
        0.35,
        0.96,
        0.92,
    )
    breakup = _blur_separable_xy(breakup, passes=2, periodic_x=periodic_x)
    scaffold = scaffold * (0.78 + 0.22 * breakup)
    return np.clip(_contrast_curve(scaffold, 0.10, 0.995, 1.02), 0.0, 1.0)


def generate_nebula(
    rng: np.random.Generator,
    mode: NebulaMode,
    height: int,
    width: int,
    tuning: NebulaTuningConfig,
    progress_cb: Callable[[float], None] | None = None,
    *,
    galaxy_features: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gf = galaxy_features
    morph_primary = bool(gf is not None and gf.morphology_dust_primary)
    band_lat_sigma = float(gf.band_lat_sigma) if gf is not None else 0.10
    band_rot = float(gf.band_rotation_deg) if gf is not None else 2.5
    band_curve = float(gf.band_curvature_amp) if gf is not None else 0.04
    nebula_color_strength = float(gf.nebula_color_strength) if gf is not None else 1.0
    nebula_fine_mix = float(gf.nebula_fine_noise_mix) if gf is not None else 0.78
    nebula_spiral_strength = float(gf.nebula_spiral_strength) if gf is not None else 0.52
    nebula_turbulence_octaves = int(gf.nebula_turbulence_octaves) if gf is not None else 3
    bulge_desat = float(gf.bulge_desat) if gf is not None else 0.35
    halpha_sat = float(tuning.emit_halpha_saturation)
    emit_patch_s = float(tuning.emit_patch_strength)

    def _p(v: float) -> None:
        if progress_cb is not None:
            progress_cb(float(np.clip(v, 0.0, 1.0)))

    _p(0.02)
    # Presets provide coarse artistic direction; scalar knobs provide fine control.
    # Keep these values synchronized with the tuning guide in `NebulaTuningConfig`.
    # Practical baseline while tuning:
    # - subtle: less dust and softer extinction.
    # - balanced: production default for "good out of the box".
    # - dramatic: heavier dust with stronger extinction contrast.
    # If defaults in config shift, revisit these multipliers in the same change.
    style_presets: dict[str, tuple[float, float, float]] = {
        "subtle": (0.92, 0.82, 0.78),
        "balanced": (1.0, 1.0, 1.0),
        "dramatic": (1.12, 1.24, 1.30),
    }
    style_cloud, style_coverage, style_strength = style_presets[tuning.style]
    cloud_gain = style_cloud * tuning.cloud_continuity
    coverage_gain = style_coverage * tuning.dust_coverage
    strength_gain = style_strength * tuning.dust_strength

    wrap_x = mode == NebulaMode.galaxy_streak

    def _sn(octaves: int = 4) -> np.ndarray:
        return _smooth_noise(rng, height, width, octaves=octaves, periodic_x=wrap_x)

    def _rz(field: np.ndarray) -> np.ndarray:
        return _resize_bilinear(field, height, width, periodic_x=wrap_x)

    n = _sn(octaves=5)
    warp = _sn(octaves=2)
    wx = (warp - 0.5) * 0.8
    wy = (_sn(octaves=2) - 0.5) * 0.8
    y_idx = np.clip((np.arange(height)[:, None] + wy * height).astype(int), 0, height - 1)
    x_idx = (np.arange(width)[None, :] + wx * width).astype(int)
    if wrap_x:
        x_idx = np.mod(x_idx, width)
    else:
        x_idx = np.clip(x_idx, 0, width - 1)
    n = n[y_idx, x_idx]
    _p(0.08)

    color_field = _sn(octaves=2)
    color_mix = np.clip(color_field * 1.2, 0.0, 1.0)[..., None]
    y = np.linspace(-1.0, 1.0, height)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)[None, :]
    # Seed-driven global longitude offset for cloud/lane center anchors.
    lon_bias = float(rng.uniform(-0.82, 0.82)) if wrap_x else 0.0
    dust_structure = np.zeros((height, width), dtype=np.float64)
    core_band = np.ones((height, width), dtype=np.float64)
    lane_extinction = np.zeros((height, width), dtype=np.float64)
    lane_scaffold = np.zeros((height, width), dtype=np.float64)
    # galaxy_streak only: (dust_rgb, emit_rgb) after shared color ops, for separate nebula passes.
    split_nebula_rgb: tuple[np.ndarray, np.ndarray] | None = None
    galaxy_dust_pack: dict[str, np.ndarray] | None = None

    if mode == NebulaMode.distant:
        stk = _pn.build_simple_noise_stack(rng, height, width, periodic_x=False, preset="distant")
        band_m = np.exp(-((y**2) / 0.9))
        n_like = _pn.combine_cloud_layers(
            stk["base"], stk["ridged"], stk["fine"], band_m, w_base=0.42, w_ridge=0.36, w_fine=0.22
        )
        cx = rng.uniform(-0.5, 0.5)
        cy = rng.uniform(-0.4, 0.4)
        blob = np.exp(-(((x - cx) ** 2) / 0.18 + ((y - cy) ** 2) / 0.10))
        mask = (n_like * 0.7 + blob * 0.9).clip(0.0, 1.0) ** 1.6
        tint_a = np.array([0.30, 0.18, 0.38], dtype=np.float64)
        tint_b = np.array([0.58, 0.34, 0.64], dtype=np.float64)
        color = tint_a * (1.0 - color_mix) + tint_b * color_mix
        _p(0.48)
    elif mode == NebulaMode.full:
        stk = _pn.build_simple_noise_stack(rng, height, width, periodic_x=False, preset="full")
        broad = np.exp(-((y**2) / 0.9))
        n_like = _pn.combine_cloud_layers(
            stk["base"], stk["ridged"], stk["fine"], broad, w_base=0.38, w_ridge=0.40, w_fine=0.22
        )
        mask = (n_like * 0.9 + broad * 0.35).clip(0.0, 1.0) ** 1.2
        tint_a = np.array([0.26, 0.16, 0.34], dtype=np.float64)
        tint_b = np.array([0.54, 0.31, 0.60], dtype=np.float64)
        color = tint_a * (1.0 - color_mix) + tint_b * color_mix
        _p(0.48)
    else:
        # Off-center bulge: optional per seed so not every generation has a pronounced core.
        sign = -1.0 if rng.random() < 0.5 else 1.0
        bulge_on = rng.random() < 0.68
        if bulge_on:
            cx_band = _wrap_x11_scalar(
                float(np.clip(rng.normal(sign * rng.uniform(0.10, 0.36), 0.12), -0.55, 0.55)) + lon_bias
            )
            core_band = np.exp(-(_wrap_dx(x, cx_band) ** 2) / rng.uniform(0.22, 0.36))
        else:
            cx_band = _wrap_x11_scalar(float(rng.uniform(-0.25, 0.25)) + lon_bias)
            # Broad, weak center hint only; avoids fixed-looking bulge each render.
            core_band = np.exp(-(_wrap_dx(x, cx_band) ** 2) / rng.uniform(0.70, 1.05)) * rng.uniform(0.24, 0.48)
        _p(0.14)
        band_warp = (_sn(octaves=1) - 0.5) * 0.32
        streak = np.exp(-(((y - band_warp) ** 2) / 0.48))
        broad_band = np.exp(-((y**2) / 1.08))
        outer_band = np.exp(-((y**2) / 1.72))
        bleed_haze = (_sn(octaves=1) * 0.55 + 0.45).clip(0.45, 1.0)
        dark_lanes = (1.0 - _sn(octaves=3) * 0.7).clip(0.20, 1.0)
        patch = _sn(octaves=2)
        patch = np.where(patch > 0.58, 1.0, patch * 0.35)
        density_breaks = (0.55 + patch * 0.75).clip(0.30, 1.25)
        sn_mv = _sn(octaves=2)
        micro_voids = 1.0 - np.clip((sn_mv - 0.52) / 0.38, 0.0, 1.0) * 0.22
        # Coarse clumps provide large-scale galactic structure.
        clump_coarse = _rz(rng.random((max(2, height // 28), max(2, width // 28))))
        clump_mid = _rz(rng.random((max(2, height // 14), max(2, width // 14))))
        clumps = _contrast_curve(clump_coarse * 0.65 + clump_mid * 0.35, 0.24, 0.92, 0.74)
        # Thickness map creates denser bulges in parts of the galactic band.
        th_sz = max(2, min(height // 22, width // 22))
        thickness_map = _rz(rng.random((th_sz, th_sz)))
        thickness_map = _contrast_curve(thickness_map, 0.34, 0.90, 0.88)
        # Slow horizontal modulation keeps the whole band active without uniform flatness.
        longitudinal_1d = _smooth_noise(rng, 1, width, octaves=3, periodic_x=True)[0]
        x_line = np.linspace(-1.0, 1.0, width, dtype=np.float64)
        long_skew = float(rng.uniform(0.02, 0.10)) * float(rng.choice([-1.0, 1.0])) * x_line
        longitudinal_1d = longitudinal_1d + long_skew + float(rng.uniform(-0.04, 0.04))
        longitudinal = _contrast_curve(longitudinal_1d[None, :], 0.18, 0.92, 0.90)
        long_warp2d = np.clip(_sn(octaves=2), 0.0, 1.0)
        # Strong 2D modulation — pure 1D longitude reads as vertical stripes in mask → dust columns.
        longitudinal = np.clip(longitudinal * (0.42 + 0.58 * long_warp2d), 0.0, 1.0)
        long_scr = np.clip(_sn(octaves=4), 0.0, 1.0)
        longitudinal = np.clip(longitudinal * (0.72 + 0.28 * long_scr), 0.0, 1.0)
        long_break = np.clip(_sn(octaves=5), 0.0, 1.0)
        longitudinal = np.clip(longitudinal * (0.68 + 0.32 * long_break), 0.0, 1.0)
        longitudinal = _blur_separable_xy(longitudinal, passes=2, periodic_x=wrap_x)
        elong_plane = 1.52 + 0.30 * cloud_gain + float(rng.uniform(0.0, 0.22))
        noise_stack = _pn.build_galaxy_streak_noise_stack(
            rng,
            height,
            width,
            periodic_x=wrap_x,
            elongate_along_x=elong_plane,
            fine_mix=nebula_fine_mix,
            spiral_strength=nebula_spiral_strength,
            turbulence_octaves=nebula_turbulence_octaves,
            x_grid=x,
            y_grid=y,
        )
        dp_dbg = tuning.debug_pass
        if dp_dbg in (
            "layer_base",
            "layer_filaments",
            "layer_fine",
            "layer_carve",
            "layer_dust_alpha",
            "mask_only",
            "warp_vectors",
        ):
            band_lat_dbg = np.exp(-((y**2) / (2.0 * 0.13**2)))
            band_m_dbg = np.clip(streak * 0.72 + broad_band * 0.28, 0.0, 1.0) * band_lat_dbg
            pack_dbg = None
            if dp_dbg in ("layer_carve", "layer_dust_alpha"):
                pack_dbg = _pn.assemble_galaxy_dust_alpha(
                    noise_stack["base"],
                    noise_stack["ridged"],
                    noise_stack["fine"],
                    band_m_dbg,
                    periodic_x=wrap_x,
                )
            if dp_dbg == "mask_only":
                plane = np.clip(band_m_dbg, 0.0, 1.0)
                neb_dbg = np.stack([plane, plane, plane], axis=2)
            elif dp_dbg == "warp_vectors":
                wx_d, wy_d = noise_stack["warp_x"], noise_stack["warp_y"]
                r = np.clip(0.5 + 0.5 * wx_d, 0.0, 1.0)
                g = np.clip(0.5 + 0.5 * wy_d, 0.0, 1.0)
                neb_dbg = np.stack([r, g, np.zeros_like(r)], axis=2)
                plane = np.clip(0.5 * (r + g), 0.0, 1.0)
            elif dp_dbg == "layer_carve":
                plane = np.clip(pack_dbg["carve"], 0.0, 1.0)  # type: ignore[union-attr]
                neb_dbg = np.stack([plane, plane, plane], axis=2)
            elif dp_dbg == "layer_dust_alpha":
                plane = np.clip(pack_dbg["dust_alpha"], 0.0, 1.0)  # type: ignore[union-attr]
                neb_dbg = np.stack([plane, plane, plane], axis=2)
            else:
                key = {"layer_base": "base", "layer_filaments": "ridged", "layer_fine": "fine"}[dp_dbg]
                plane = np.clip(_contrast_curve(noise_stack[key], 0.05, 0.995, 0.92), 0.0, 1.0)
                neb_dbg = np.stack([plane, plane, plane], axis=2)
            emit_dbg = np.zeros((height, width, 3), dtype=np.float64)
            lane_dbg = np.zeros((height, width), dtype=np.float64)
            return neb_dbg, emit_dbg, plane, lane_dbg
        # Break pure column correlation (1D longitude × blur_x reads as vertical comb in gas/dust).
        destripe = _blur_separable_xy(_sn(octaves=3), passes=2, periodic_x=wrap_x)
        destripe_mix = 0.88 if morph_primary else 0.54
        longitudinal = np.clip(
            longitudinal * (1.0 - destripe_mix) + destripe * destripe_mix, 0.0, 1.0
        )
        longitudinal = _blur_separable_xy(longitudinal, passes=2, periodic_x=wrap_x)
        # Broad activity zones restore large-scale non-uniformity along longitude.
        activity_map = _rz(rng.random((max(2, height // 12), max(2, width // 11))))
        activity_map = _contrast_curve(activity_map, 0.18, 0.92, 0.72)
        activity_map = _blur_separable_xy(activity_map, passes=1, periodic_x=wrap_x)
        # Sparse macro structure: occasional cavities and dense knots.
        macro_voids = _rz(rng.random((max(2, height // 24), max(2, width // 14))))
        macro_voids = _contrast_curve(macro_voids, 0.76, 0.99, 1.60)
        heavy_spots = _rz(rng.random((max(2, height // 13), max(2, width // 9))))
        heavy_spots = _contrast_curve(heavy_spots, 0.60, 0.95, 0.95)
        heavy_spots = _blur_separable_xy(heavy_spots, passes=1, periodic_x=wrap_x)
        # Bridge field links neighboring clumps into more continuous band segments.
        bridge_map = _rz(rng.random((max(2, height // 12), max(2, width // 7))))
        bridge_map = _contrast_curve(bridge_map, 0.18, 0.90, 0.78)
        bridge_map = _blur_separable_xy(bridge_map, passes=1, periodic_x=wrap_x)
        _p(0.26)

        # Lane texture: finer in x than before to avoid tall vertical cell boundaries (stripe artifacts).
        lane_coarse = rng.random((max(2, height // 52), max(2, width // 12)))
        lane_noise = _rz(lane_coarse)
        lane_warp = (_sn(octaves=1) - 0.5) * 0.12
        lane_noise = np.clip(lane_noise * (0.90 + 0.20 * lane_warp), 0.0, 1.0)
        lane_noise = _blur_x_only(lane_noise, passes=3, periodic_x=wrap_x)
        lane_noise = _blur_y_only(lane_noise, passes=1)
        lane_cut = _contrast_curve(lane_noise, 0.40, 0.89, 1.24)
        # Flow-modulated lane continuity: keeps long coherent trenches, suppresses cell-like breakup.
        lane_flow = _blur_separable_xy(_sn(octaves=2), passes=2, periodic_x=wrap_x)
        lane_shear = np.sin(
            2.6 * x + 0.95 * y + float(rng.uniform(0.0, 6.283185307179586))
        ) * 0.5 + 0.5
        lane_cont = np.clip(
            0.62 + lane_flow * 0.30 + lane_shear * 0.08,
            0.56,
            1.0,
        )
        lane_cut *= lane_cont
        scaffold_gate = np.clip(streak * 0.90 + broad_band * 0.10, 0.0, 1.0)
        lane_scaffold = _structural_lane_scaffold(
            rng,
            x,
            y,
            scaffold_gate,
            sign=sign,
            lane_count=int(rng.integers(2, 5)),
            periodic_x=True,
        )
        _p(0.36)
        lane_mix_map = _contrast_curve(_sn(octaves=2), 0.18, 0.94, 0.90)
        lane_mix_map = _blur_separable_xy(lane_mix_map, passes=2, periodic_x=wrap_x)
        lane_mix_w = np.clip(0.14 + 0.20 * lane_mix_map, 0.10, 0.36)
        lane_cut = np.clip(
            lane_cut * (1.0 - lane_mix_w) + lane_scaffold * lane_mix_w * (0.90 + 0.08 * strength_gain),
            0.0,
            1.0,
        )
        filament_gate = _rz(rng.random((max(2, height // 14), max(2, width // 8))))
        filament_gate = _contrast_curve(filament_gate, 0.64, 0.98, 1.45)
        filament_gate = _blur_separable_xy(filament_gate, passes=1, periodic_x=wrap_x)
        break_mask = _rz(rng.random((max(2, height // 14), max(2, width // 7))))
        break_mask = _contrast_curve(break_mask, 0.58, 0.96, 1.35)
        break_mask = _blur_separable_xy(break_mask, passes=1, periodic_x=wrap_x)
        # Keep only portions of lanes to avoid "everywhere" continuous streaking.
        lane_cut *= 0.56 + filament_gate * 0.56
        lane_cut *= 1.0 - break_mask * 0.24
        lane_cut = np.clip(lane_cut, 0.0, 1.0)
        # One thick, coherent dust lane (silhouette) offset from the band center.
        dom_x = _wrap_x11_scalar(float(np.clip(rng.normal(sign * rng.uniform(0.14, 0.42), 0.16), -0.68, 0.68)) + lon_bias)
        dom_y = float(np.clip(rng.normal(0.0, 0.048), -0.15, 0.15))
        sig_x = rng.uniform(0.52, 1.02)
        sig_y = rng.uniform(0.102, 0.175)
        dominant_ridge = np.exp(-((_wrap_dx(x, dom_x) ** 2) / sig_x + ((y - dom_y) ** 2) / sig_y))
        dominant_ridge *= np.clip(streak * 0.62 + broad_band * 0.38, 0.0, 1.0) * (0.52 + 0.48 * filament_gate)
        dominant_ridge = _contrast_curve(dominant_ridge, 0.38, 0.995, 1.92)
        lane_cut = np.clip(lane_cut + dominant_ridge * (0.66 + 0.24 * strength_gain), 0.0, 1.0)
        lane_cut = _blur_separable_xy(lane_cut, passes=1, periodic_x=wrap_x)
        lane_cut = _blur_x_only(lane_cut, passes=3, periodic_x=wrap_x)
        lane_cut = _blur_y_only(lane_cut, passes=1)
        lane_cut = _blur_separable_xy(lane_cut, passes=1, periodic_x=wrap_x)
        lane_cut = _blur_separable_xy(lane_cut, passes=2, periodic_x=wrap_x)
        lane_cut = _blur_y_only(lane_cut, passes=1)
        lane_shape = np.maximum(lane_cut, lane_scaffold * 0.56)
        dark_lane_factor = np.clip(1.0 - lane_shape * 0.78, 0.08, 1.0)

        band_envelope = (streak * 0.70 + broad_band * 0.22 + outer_band * 0.08).clip(0.0, 1.0)
        # Coarse volume field suppresses speckle and makes clouds read as connected masses.
        volume_soft = _rz(rng.random((max(2, height // 16), max(2, width // 14))))
        volume_soft = _contrast_curve(volume_soft, 0.18, 0.90, 0.88)
        volume_soft = _blur_separable_xy(volume_soft, passes=2, periodic_x=wrap_x)
        # Extra very-low-frequency envelope so the disk reads as one lit volume, not same-scale clumps.
        vol_mega = _rz(rng.random((max(2, height // 52), max(2, width // 20))))
        vol_mega = _contrast_curve(vol_mega, 0.10, 0.88, 0.78)
        vol_mega = _blur_separable_xy(vol_mega, passes=3, periodic_x=wrap_x)
        vol_mega = _blur_x_only(vol_mega, passes=2, periodic_x=wrap_x)
        volume_soft = np.clip(volume_soft * 0.72 + vol_mega * 0.28, 0.0, 1.0)
        # Warped `n` is 5-octave fBm — keep a sharper mid-scale share so clouds are not one smooth sheet.
        n_smooth = _blur_separable_xy(n, passes=1, periodic_x=wrap_x)
        fine_density = (
            n_smooth * 0.24
            + _sn(octaves=3) * 0.34
            + volume_soft * 0.42
        )
        base = band_envelope * bleed_haze * (
            0.30 + fine_density * 0.24 + clumps * 0.82 + thickness_map * 0.68 + volume_soft * 0.94
        )
        base = np.maximum(base, band_envelope * bridge_map * (0.82 + 0.12 * cloud_gain))
        # Shallower longitude gain (less column-driven contrast in the volume mask).
        if morph_primary:
            base *= 0.97 + longitudinal * (0.02 + 0.02 * long_warp2d)
        else:
            base *= 0.94 + longitudinal * (0.07 + 0.06 * long_warp2d)
        # Vary large-scale activity contrast per seed to diversify cloud morphology.
        act_k = float(rng.uniform(0.44, 0.72))
        base *= (1.0 - act_k) + activity_map * act_k
        heavy_k = float(rng.uniform(0.16, 0.42))
        base *= 0.82 + heavy_spots * (heavy_k + 0.08 * cloud_gain)
        x_ramp = 1.0 + float(rng.uniform(0.02, 0.08)) * (x * float(rng.choice([-1.0, 1.0])))
        x_ramp_2d = np.clip(
            (x_ramp * 0.65 + 0.35 * (0.94 + 0.12 * np.clip(_sn(octaves=3), 0.0, 1.0)))
            * (0.90 + 0.10 * np.clip(_sn(octaves=2), 0.0, 1.0)),
            0.90,
            1.10,
        )
        base *= x_ramp_2d
        # Blend medium cloud mass into lane carving so lanes look embedded in volumes, not hard cuts.
        lane_mass = np.clip(
            _blur_separable_xy(volume_soft * band_envelope, passes=2, periodic_x=wrap_x), 0.0, 1.0
        )
        lane_mass_k = float(rng.uniform(0.16, 0.40))
        detail = dark_lanes * density_breaks * micro_voids * dark_lane_factor * (0.74 + lane_mass_k * lane_mass)
        detail *= 0.84 + (1.0 - activity_map) * 0.28
        mask = (base * (0.86 + 0.05 * cloud_gain) + detail * (0.040 - 0.013 * (cloud_gain - 1.0))).clip(0.0, 1.0)
        mask = mask * 0.76 + volume_soft * (0.10 + 0.05 * cloud_gain) + bridge_map * (0.11 + 0.05 * cloud_gain)
        if not morph_primary:
            mask *= 1.0 - macro_voids * (0.18 + 0.08 * coverage_gain)
        mask = _contrast_curve(mask, 0.22, 0.96, 0.92)
        mask = _blur_separable_xy(mask, passes=1, periodic_x=wrap_x)
        mask = _blur_x_only(mask, passes=1, periodic_x=wrap_x)
        mask_hi = np.clip(mask - _blur_separable_xy(mask, passes=2, periodic_x=wrap_x), 0.0, 1.0)
        mask = np.clip(mask + mask_hi * 0.38, 0.0, 1.0)
        mid_mask = galactic_midplane_mask(
            height,
            width,
            lat_sigma=band_lat_sigma,
            band_rotation_deg=band_rot,
            band_curvature_amp=band_curve,
        )
        band_lat = mid_mask
        band_m_combined = np.clip(streak * 0.68 + broad_band * 0.32, 0.0, 1.0) * band_lat
        galaxy_dust_pack = _pn.assemble_galaxy_dust_alpha(
            noise_stack["base"],
            noise_stack["ridged"],
            noise_stack["fine"],
            band_m_combined,
            periodic_x=wrap_x,
        )
        D2 = _blur_separable_xy(galaxy_dust_pack["dust_alpha"], passes=1, periodic_x=wrap_x)
        dust_structure = np.clip(
            D2 * (0.58 + 0.10 * cloud_gain)
            + lane_cut * (0.14 + 0.06 * coverage_gain)
            + lane_scaffold * (0.10 + 0.04 * strength_gain)
            + macro_voids * (0.0 if morph_primary else (0.14 + 0.10 * coverage_gain))
            + filament_gate * (0.12 + 0.04 * cloud_gain)
            + (1.0 - mask) * 0.09
            + dominant_ridge * (0.14 + 0.10 * strength_gain)
            + lane_mass * (0.08 + 0.05 * cloud_gain),
            0.0,
            1.0,
        )
        _p(0.48)
        # Great Rift–style trench: one broad, tilted dark lane through the disk (not mirrored).
        rx = _wrap_x11_scalar(float(np.clip(rng.normal(sign * rng.uniform(0.12, 0.44), 0.14), -0.62, 0.62)) + lon_bias)
        ry = float(np.clip(rng.normal(0.0, 0.055), -0.12, 0.12))
        wx_r = rng.uniform(0.14, 0.32)
        wy_r = rng.uniform(0.56, 1.05)
        band_gate = np.clip(streak * 0.86 + broad_band * 0.14, 0.0, 1.0)
        rift_wx = (_sn(octaves=2) - 0.5) * 0.11 * band_gate
        rift_wy = (_sn(octaves=2) - 0.5) * 0.09 * band_gate
        rift_oval = np.exp(
            -((_wrap_dx(x + rift_wx, rx) ** 2) / wx_r + ((y + rift_wy - ry) ** 2) / wy_r)
        ) * band_gate
        tilt = float(rng.uniform(-0.42, 0.42))
        x_rift = _wrap_dx(x + rift_wx * 0.85, rx)
        xc = x_rift * np.cos(tilt) + (y + rift_wy * 0.85 - ry) * np.sin(tilt)
        yc = -x_rift * np.sin(tilt) + (y + rift_wy * 0.85 - ry) * np.cos(tilt)
        rift_fil = np.exp(-((xc**2) / (0.24 + 0.10 * streak)) - ((yc**2) / 0.68)) * band_gate
        rift_wisp = _contrast_curve(_sn(octaves=4), 0.32, 0.94, 1.28)
        rift_wisp_soft = _blur_separable_xy(rift_wisp, passes=1, periodic_x=wrap_x)
        rift_combo = np.clip(
            _contrast_curve(
                (rift_oval * 0.62 + rift_fil * 0.52) * (0.48 + 0.52 * rift_wisp_soft),
                0.08,
                0.995,
                1.42,
            ),
            0.0,
            1.0,
        )
        # Macro rift field: broad coherent trench that survives later blurs/extinction shaping.
        rift_macro = _blur_separable_xy(rift_combo, passes=5, periodic_x=wrap_x)
        rift_macro = _blur_x_only(rift_macro, passes=4, periodic_x=wrap_x)
        rift_macro = _blur_y_only(rift_macro, passes=1)
        rift_macro = np.clip(_contrast_curve(rift_macro, 0.10, 0.98, 1.12), 0.0, 1.0)
        dust_structure = np.clip(
            dust_structure + rift_combo * (0.52 + 0.21 * strength_gain),
            0.0,
            1.0,
        )
        swirl = _sn(octaves=3)
        diag = np.sin(2.12 * x + 1.55 * y + float(rng.uniform(0.0, 6.283185307179586))) * 0.5 + 0.5
        dust_asym = 1.0 + float(rng.uniform(0.04, 0.09)) * (0.74 * (swirl - 0.5) + 0.26 * (diag - 0.5))
        dust_asym = np.clip(dust_asym, 0.96, 1.04)
        dust_structure = np.clip(dust_structure * dust_asym, 0.0, 1.0)
        flow_ph = float(rng.uniform(0.0, 6.283185307179586))
        flow_mod = 0.965 + 0.035 * (
            0.5 + 0.5 * np.sin(1.55 * x * float(sign) + 2.05 * y + flow_ph)
        )
        dust_structure = np.clip(dust_structure * flow_mod, 0.0, 1.0)
        lob_x0 = _wrap_x11_scalar(float(sign * rng.uniform(0.38, 0.66)) + lon_bias)
        lob_y0 = float(rng.uniform(-0.11, 0.11))
        lob_wx = rng.uniform(0.36, 0.68)
        lob_wy = rng.uniform(0.54, 0.98)
        lob_px = (_sn(octaves=3) - 0.5) * 0.16 * band_gate
        lob_py = (_sn(octaves=3) - 0.5) * 0.13 * band_gate
        side_core = np.exp(
            -((_wrap_dx(x + lob_px, lob_x0) ** 2) / lob_wx) - (((y + lob_py - lob_y0) ** 2) / lob_wy)
        ) * band_gate
        lob_wisp = _contrast_curve(_sn(octaves=4), 0.38, 0.96, 1.42)
        side_lobe = np.clip(_contrast_curve(side_core * (0.18 + 0.82 * lob_wisp), 0.14, 0.994, 1.78), 0.0, 1.0)
        dust_structure = np.clip(
            dust_structure + side_lobe * (0.16 + 0.10 * strength_gain),
            0.0,
            1.0,
        )
        _p(0.62)
        # Extra mid-scale diffuse clouds (soft, band-aligned) so extinction + nebula read dustier.
        mid_wisp = _sn(octaves=3) * np.clip(
            streak * 0.72 + broad_band * 0.28, 0.0, 1.0
        )
        mid_wisp = _contrast_curve(mid_wisp * (0.45 + 0.55 * filament_gate), 0.28, 0.95, 1.12)
        mid_wisp = _blur_separable_xy(mid_wisp, passes=2, periodic_x=wrap_x)
        dust_structure = np.clip(
            dust_structure + mid_wisp * (0.12 + 0.08 * coverage_gain + 0.04 * cloud_gain),
            0.0,
            1.0,
        )
        plate = _rz(rng.random((max(2, height // 42), max(2, width // 9))))
        plate = _contrast_curve(plate, 0.18, 0.88, 0.82)
        plate *= np.clip(streak * 0.75 + broad_band * 0.25, 0.0, 1.0)
        plate = _blur_x_only(_blur_separable_xy(plate, passes=1, periodic_x=wrap_x), passes=1, periodic_x=wrap_x)
        dust_structure = np.clip(
            dust_structure + plate * (0.08 + 0.05 * coverage_gain + 0.02 * cloud_gain),
            0.0,
            1.0,
        )
        dust_structure = _blur_separable_xy(dust_structure, passes=1, periodic_x=wrap_x)
        rift_feather = 0.93 + 0.12 * _sn(octaves=2)
        dust_structure = np.clip(dust_structure * rift_feather, 0.0, 1.0)
        ds_merge = _blur_separable_xy(dust_structure, passes=2, periodic_x=wrap_x)
        dust_structure = np.clip(0.48 * dust_structure + 0.52 * ds_merge, 0.0, 1.0)
        band_soft = np.clip(streak * 0.92 + broad_band * 0.08, 0.0, 1.0)
        river = _blur_x_only(
            _blur_y_only(dust_structure * band_soft, passes=1),
            passes=1,
            periodic_x=wrap_x,
        )
        dust_structure = np.clip(0.78 * dust_structure + 0.22 * river, 0.0, 1.0)
        dust_structure = _blur_x_only(dust_structure, passes=2, periodic_x=wrap_x)
        dust_structure = _blur_y_only(dust_structure, passes=1)
        dust_structure = np.clip(dust_structure**1.02, 0.0, 1.0)
        # Macro elliptical dust patches read as giant circles when morphology extinction is primary — skip.
        band_gate_m = np.clip(streak * 0.90 + broad_band * 0.10, 0.0, 1.0)
        mega_on = False if morph_primary else (rng.random() < 0.34)
        if not mega_on:
            dust_structure = _blur_x_only(dust_structure, passes=1, periodic_x=wrap_x)
        # Anchor large dust patches in the disk wings (bulge longitude cx_band keeps warm gas).
        dust_lon0 = float(sign * rng.uniform(0.32, 0.58))
        cx1 = _wrap_x11_scalar(float(np.clip(cx_band + dust_lon0 + rng.normal(0.0, 0.055), -0.86, 0.86)))
        cy1 = float(np.clip(rng.normal(float(sign * 0.035), 0.095), -0.20, 0.20))
        sx1 = float(rng.uniform(0.32, 0.62))
        sy1 = float(rng.uniform(0.11, 0.23))
        th1 = float(rng.uniform(-0.42, 0.42))
        c1, s1 = np.cos(th1), np.sin(th1)
        dx1, dy1 = _wrap_dx(x, cx1), y - cy1
        xr1 = dx1 * c1 + dy1 * s1
        yr1 = -dx1 * s1 + dy1 * c1
        mega_a = np.exp(-((xr1**2) / sx1 + (yr1**2) / sy1)) * band_gate_m
        if mega_on and rng.random() < 0.52:
            dust_lon1 = float(-sign * rng.uniform(0.24, 0.52))
            cx2 = _wrap_x11_scalar(float(np.clip(cx_band + dust_lon1 + rng.normal(0.0, 0.075), -0.88, 0.88)))
            if abs(cx2 - cx1) < 0.18:
                cx2 = _wrap_x11_scalar(float(np.clip(cx2 + float(sign) * 0.36, -0.88, 0.88)))
            cy2 = float(np.clip(rng.normal(float(-sign * 0.032), 0.095), -0.20, 0.20))
            sx2 = float(rng.uniform(0.24, 0.52))
            sy2 = float(rng.uniform(0.11, 0.23))
            w2 = float(rng.uniform(0.38, 0.65))
            th2 = float(rng.uniform(-0.42, 0.42))
            c2, s2 = np.cos(th2), np.sin(th2)
            dx2, dy2 = _wrap_dx(x, cx2), y - cy2
            xr2 = dx2 * c2 + dy2 * s2
            yr2 = -dx2 * s2 + dy2 * c2
            mega_b = np.exp(-((xr2**2) / sx2 + (yr2**2) / sy2)) * band_gate_m * w2
            mega_raw = np.maximum(mega_a, mega_b) if mega_on else np.zeros_like(mega_a)
        else:
            mega_raw = mega_a if mega_on else np.zeros_like(mega_a)
        # Break macro patch into cloud-like fragments when present.
        if mega_on:
            mega_brk = _contrast_curve(_sn(octaves=3), 0.28, 0.94, 0.96)
            mega_brk = _blur_separable_xy(mega_brk, passes=1, periodic_x=wrap_x)
            mega_raw *= 0.48 + 0.52 * mega_brk
        mega_env = _blur_separable_xy(mega_raw, passes=5, periodic_x=wrap_x)
        mega_env = _blur_x_only(mega_env, passes=3, periodic_x=wrap_x)
        mega_env = _blur_y_only(mega_env, passes=2)
        mega_env = np.clip(_contrast_curve(mega_env, 0.10, 0.90, 0.82), 0.0, 1.0) ** float(rng.uniform(0.91, 0.98))
        if mega_on:
            mega_frag = _blur_separable_xy(_contrast_curve(_sn(octaves=2), 0.20, 0.94, 0.92), passes=1, periodic_x=wrap_x)
            mega_env *= np.clip(0.62 + 0.38 * mega_frag, 0.50, 1.0)
        if mega_on:
            gate_w = np.clip(mega_env, 0.0, 1.0) ** (0.22 + 0.07 * cloud_gain)
            mega_add = 0.17 + 0.05 * coverage_gain + 0.034 * strength_gain
            dust_structure = np.clip(
                dust_structure * (0.30 + 0.70 * gate_w) + mega_env * mega_add,
                0.0,
                1.0,
            )
            refill = _blur_separable_xy(
                _contrast_curve(_sn(octaves=3), 0.18, 0.92, 0.88), passes=1, periodic_x=wrap_x
            )
            dust_structure = np.clip(
                dust_structure * (1.0 - mega_env * 0.42) + refill * mega_env * (0.04 + 0.03 * cloud_gain),
                0.0,
                1.0,
            )
        dust_structure = _blur_x_only(dust_structure, passes=2, periodic_x=wrap_x)
        dust_structure = _blur_y_only(dust_structure, passes=1)
        dust_structure = np.clip(dust_structure**1.04, 0.0, 1.0)
        # Thin dust only in the tight bulge nucleus (keep a high floor so lanes stay visible).
        if morph_primary:
            warm_carve = np.clip(streak**1.35, 0.0, 1.0)
            warm_carve = _blur_x_only(warm_carve, passes=1, periodic_x=wrap_x)
        else:
            warm_carve = np.clip((core_band**1.42) * (streak**1.02), 0.0, 1.0)
            warm_carve = _blur_separable_xy(warm_carve, passes=2, periodic_x=wrap_x)
        warm_cut = 0.16 if morph_primary else 0.55
        dust_structure = np.clip(
            dust_structure * (0.56 + 0.44 * ((1.0 - warm_cut * warm_carve) ** 0.82)),
            0.0,
            1.0,
        )
        # Add a little extra dust in the band shoulders only (do not scale down the midplane — that was hiding dust).
        shoulder = np.clip(4.2 * streak * (1.0 - streak), 0.0, 1.0) ** 0.52
        shoulder = _blur_separable_xy(shoulder, passes=1, periodic_x=wrap_x)
        dust_structure = np.clip(dust_structure + shoulder * (0.10 + 0.06 * coverage_gain), 0.0, 1.0)
        # Diffuse dust veil: stronger low-frequency mass across the plane, still avoiding vertical over-thickening.
        veil_field = _contrast_curve(_sn(octaves=2), 0.12, 0.92, 0.86)
        veil_field = _blur_separable_xy(veil_field, passes=2, periodic_x=wrap_x)
        wide_band = np.clip(streak * 0.28 + broad_band * 0.72, 0.0, 1.0)
        long_veil = _blur_separable_xy(longitudinal, passes=2, periodic_x=wrap_x)
        long_veil = np.clip(0.76 + 0.24 * long_veil, 0.72, 1.0)
        dust_structure = np.clip(
            dust_structure + veil_field * wide_band * long_veil * (0.10 + 0.07 * coverage_gain),
            0.0,
            1.0,
        )
        dust_structure = _blur_x_only(dust_structure, passes=5, periodic_x=wrap_x)
        dust_structure = _blur_separable_xy(dust_structure, passes=2, periodic_x=wrap_x)
        dust_structure = np.clip(dust_structure**0.985, 0.0, 1.0)
        # Keep dense dust volumetric (photo-like) rather than fully crushed to black.
        ds_lift = np.clip(_blur_separable_xy(dust_structure, passes=2, periodic_x=wrap_x), 0.0, 1.0)
        dust_structure = np.clip(dust_structure * 0.95 + ds_lift * 0.05, 0.0, 1.0)
        # Galactic band palette tuned toward dusty gas instead of ember-like fire tones.
        cloud_mix = (_sn(octaves=1) * 0.78 + n_smooth * 0.22).clip(0.0, 1.0)
        zone_map = _sn(octaves=2)
        zone_map = (zone_map * 0.85 + cloud_mix * 0.15).clip(0.0, 1.0)
        # Continuum cloud body: warm yellow/white dominant, with minimal reddish bias.
        black = np.array([0.022, 0.022, 0.026], dtype=np.float64)
        warm_mid = np.array([0.40, 0.34, 0.28], dtype=np.float64)
        gold = np.array([0.66, 0.62, 0.52], dtype=np.float64)
        warm_white = np.array([0.82, 0.80, 0.76], dtype=np.float64)

        t1 = np.clip(zone_map / 0.55, 0.0, 1.0)[..., None]
        t2 = np.clip((zone_map - 0.55) / 0.28, 0.0, 1.0)[..., None]
        t3 = np.clip((zone_map - 0.83) / 0.17, 0.0, 1.0)[..., None]
        color = black * (1.0 - t1) + warm_mid * t1
        color = color * (1.0 - t2) + gold * t2
        color = color * (1.0 - t3) + warm_white * t3
        rust_mix = np.clip(_sn(octaves=2), 0.0, 1.0)[..., None]
        # Keep tiny warm variation only; avoid a global red-brown cast.
        color = color * (1.0 - 0.045 * rust_mix) + np.array([0.34, 0.30, 0.24], dtype=np.float64) * (0.045 * rust_mix)
        # Per-pixel hue drift in dust tones (brown / tan / brick variation).
        hue_j = (_sn(octaves=2) - 0.5)[..., None]
        color = color + hue_j * np.array([0.026, 0.019, -0.010], dtype=np.float64) * band_envelope[..., None]
        # Mild desaturation — keep enough chroma for gold / magenta / H II to read in final grade.
        lum = np.mean(color, axis=2, keepdims=True)
        color = color * 0.70 + lum * 0.30
        # Central bulge: subtle warm lift (reference is mostly neutral with a soft core glow).
        plane_for_bulge = streak[..., None]
        bulge_w = (core_band[..., None] ** 0.82) * (0.10 + 0.90 * (plane_for_bulge**0.80))
        cream = np.array([0.90, 0.88, 0.82], dtype=np.float64)
        color = color * (1.0 + 0.16 * bulge_w) + cream * bulge_w * 0.12
        lum_bulge = np.mean(color, axis=2, keepdims=True)
        color = color * (1.0 - 0.12 * bulge_w) + lum_bulge * (0.12 * bulge_w)
        color = color * (1.0 - bulge_desat * bulge_w) + lum_bulge * (bulge_desat * bulge_w)
        lum_c = np.mean(color, axis=2, keepdims=True)
        color = lum_c + (color - lum_c) * nebula_color_strength
        # Narrow inner hotspots: unresolved knots inside the bright band (continuum only).
        inner_hot = np.clip(
            _blur_separable_xy(
                (core_band * np.clip(streak, 0.0, 1.0)) ** 1.32,
                passes=2,
                periodic_x=wrap_x,
            ),
            0.0,
            1.0,
        )
        hot_rgb = np.array([0.12, 0.11, 0.095], dtype=np.float64)
        color = np.clip(color + inner_hot[..., None] * hot_rgb * (0.085 + 0.035 * cloud_gain), 0.0, 1.0)
        _p(0.74)
        # Softer brown transition into bright gold at dust silhouettes (volume in front of backlight).
        ds_chroma = np.clip(
            _blur_separable_xy(dust_structure, passes=3, periodic_x=wrap_x) ** 0.48, 0.0, 1.0
        )[..., None]
        gold_ink = np.array([0.14, 0.12, 0.10], dtype=np.float64)
        color = np.clip(
            color * (1.0 - 0.018 * ds_chroma) + gold_ink * (0.06 * ds_chroma) * band_envelope[..., None],
            0.0,
            1.0,
        )

        # --- Spectral emission: compact H II (lines) vs smooth reflection continuum ---
        ha_field = _sn(octaves=4)
        ha_spots = _rz(rng.random((max(2, height // 28), max(2, width // 28))))
        ha_mask = ha_field * 0.55 + ha_spots * 0.45
        ha_mask = _contrast_curve(ha_mask, 0.62, 0.995, 2.25)
        ha_mask *= band_envelope * (0.30 + 0.70 * np.clip(clumps, 0.0, 1.0))
        ha_cloud = np.clip(_sn(octaves=3), 0.0, 1.0)
        ha_lon = _blur_separable_xy(longitudinal, passes=2, periodic_x=wrap_x) * 0.55 + long_warp2d * 0.45
        ha_mask *= (0.48 + 0.52 * ha_lon) * (0.82 + 0.38 * ha_cloud)
        ha_core = np.clip(
            _blur_separable_xy((core_band * streak) ** 0.58, passes=1, periodic_x=wrap_x), 0.0, 1.0
        )
        ha_mask *= 0.50 + 0.50 * ha_core
        ha_grad = np.clip(
            np.abs(ha_mask - _blur_separable_xy(ha_mask, passes=3, periodic_x=wrap_x)),
            0.0,
            1.0,
        ) ** 0.58
        ha_hot = (
            np.clip(ha_mask, 0.0, 1.0) ** 2.35
            * band_envelope
            * np.clip(streak, 0.0, 1.0)
            * (0.32 + 0.68 * ha_core)
            * (0.45 + 0.55 * np.clip(clumps, 0.0, 1.0))
        )
        spot_scalar = np.zeros((height, width), dtype=np.float64)
        n_spots = int(rng.integers(4, 8))
        for _ in range(n_spots):
            cx_s = _wrap_x11_scalar(float(rng.normal(0.0, 0.48)))
            cy_s = float(np.clip(rng.normal(0.0, 0.14), -0.38, 0.38))
            wx_s = float(rng.uniform(0.008, 0.038))
            wy_s = float(rng.uniform(0.012, 0.055))
            dx_s = _wrap_dx(x, cx_s)
            cloud = np.exp(-((dx_s**2) / wx_s + ((y - cy_s) ** 2) / wy_s))
            cloud *= np.clip(streak * 0.54 + broad_band * 0.46, 0.0, 1.0)
            cloud = np.clip(cloud**1.22, 0.0, 1.0)
            spot_scalar = np.clip(spot_scalar + cloud * float(rng.uniform(0.18, 0.36)), 0.0, 1.0)
        spot_scalar = _blur_separable_xy(spot_scalar, passes=2, periodic_x=wrap_x)
        hii_compact = build_hii_compact_mask(
            ha_mask,
            ha_hot,
            ha_core,
            band_envelope,
            streak,
            clumps,
            spot_blobs=spot_scalar,
        )
        oiii_mask = np.clip(
            hii_compact
            * np.clip((zone_map - 0.42) / 0.48, 0.0, 1.0)
            * (0.22 + 0.78 * ha_grad),
            0.0,
            1.0,
        )
        oiii_mask = _contrast_curve(oiii_mask * _sn(octaves=1), 0.22, 0.94, 1.18)
        sii_mask = _blur_separable_xy(
            hii_compact * band_envelope * np.clip(streak, 0.0, 1.0),
            passes=2,
            periodic_x=wrap_x,
        )
        sii_mask = np.clip(sii_mask * (1.0 + 0.28 * ha_grad), 0.0, 1.0)
        ha_sat = float(np.clip(halpha_sat * (1.32 if morph_primary else 1.0), 0.2, 1.0))
        emit_cap = 0.82 if morph_primary else 0.68
        emit_rgb = compose_line_emission_rgb(
            hii_compact,
            oiii_mask,
            sii_mask,
            halpha_saturation=ha_sat,
            patch_strength=emit_patch_s * (1.18 if morph_primary else 1.0),
            cloud_gain=cloud_gain,
            emit_cap=emit_cap,
        )
        emit_rgb *= nebula_color_strength
        if morph_primary:
            emit_rgb = np.clip(emit_rgb * np.array([1.10, 0.88, 0.86], dtype=np.float64), 0.0, emit_cap)
        _p(0.86)
        # Shadow-side teal (scattered light / O-association shadows) in darker dust.
        teal_field = _sn(octaves=3)
        teal_mask = _contrast_curve(teal_field, 0.71, 0.997, 2.25)
        teal_mask *= band_envelope * np.clip(1.0 - zone_map, 0.0, 1.0) * (0.28 + 0.72 * np.clip(clumps, 0.0, 1.0))
        teal_mask *= 0.42 + 0.58 * ha_grad
        teal_mask *= 0.62 + 0.38 * np.clip(dust_structure, 0.0, 1.0) ** 0.38
        teal_rgb = np.array([0.10, 0.16, 0.17], dtype=np.float64)
        color = color + teal_mask[..., None] * teal_rgb * (0.28 + 0.10 * cloud_gain)
        # Smooth reflection-nebula continuum (diffuse, not line emission).
        vio_field = _rz(rng.random((max(2, height // 26), max(2, width // 26))))
        vio_mask = _contrast_curve(vio_field * 0.55 + _sn(octaves=2) * 0.45, 0.78, 0.998, 2.5)
        vio_cloud = np.clip(_sn(octaves=2), 0.0, 1.0)
        vio_mask *= band_envelope * ha_lon * (0.22 + 0.78 * activity_map) * (0.74 + 0.52 * vio_cloud)
        refl_mask = build_reflection_diffuse_mask(
            zone_map,
            band_envelope,
            streak,
            vio_mask,
            activity_map,
            blur_fn=_blur_separable_xy,
            periodic_x=wrap_x,
        )
        reflection_rgb = compose_reflection_continuum_rgb(
            refl_mask,
            warm_mix=zone_map,
            strength=0.80 + 0.14 * coverage_gain,
            cap=0.48,
        )
        color = np.clip(color + reflection_rgb, 0.0, 1.0)
        # Tight deep-red molecular filaments on dust continuum only.
        fil_field = _sn(octaves=4)
        fil_mask = _contrast_curve(fil_field, 0.76, 0.999, 3.1)
        fil_mask *= band_envelope * dark_lanes * (0.35 + 0.65 * filament_gate)
        fil_rgb = np.array([0.50, 0.10, 0.075], dtype=np.float64)
        fil_add = fil_mask[..., None] * fil_rgb * (0.26 + 0.09 * strength_gain)
        color = color + fil_add
        # Cooler diffuse gas in the disk; warm cream stays concentrated in core × band.
        away_core = np.clip(1.0 - np.minimum(1.0, core_band * streak * 1.55), 0.0, 1.0)[..., None]
        cool_disk = np.array([0.96, 0.98, 1.02], dtype=np.float64)
        cool_mul = 1.0 + (cool_disk - 1.0) * away_core * 0.10
        color = color * cool_mul
        # Large-scale brown-screen mottling (blur longitude so RGB drift is not vertical stripes).
        long_soft = _blur_separable_xy(longitudinal, passes=2, periodic_x=wrap_x)
        lon_tint = (long_soft - 0.5) * 0.09
        r_ch = np.clip(1.0 + lon_tint, 0.88, 1.12)
        g_ch = np.clip(1.0 - 0.035 * lon_tint, 0.88, 1.12)
        b_ch = np.clip(1.0 - 0.07 * np.abs(lon_tint), 0.88, 1.12)
        lon_rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
        lon_brk = np.clip(_sn(octaves=2), 0.0, 1.0)
        lon_mul = lon_rgb * (0.90 + 0.14 * np.clip(clumps, 0.0, 1.0))[..., None] * (0.88 + 0.24 * lon_brk)[..., None]
        color = color * lon_mul

        # Mid-scale gold + magenta glow (foreground gas — moderate saturation).
        low_h, low_w = max(2, height // 20), max(2, width // 18)
        gold_blob = _blur_separable_xy(_rz(rng.random((low_h, low_w))), passes=3, periodic_x=wrap_x)
        mag_blob = _blur_separable_xy(
            _rz(rng.random((max(2, height // 22), max(2, width // 16)))),
            passes=3,
            periodic_x=wrap_x,
        )
        gold_blob = _contrast_curve(gold_blob, 0.22, 0.90, 0.82) * band_envelope * (0.32 + 0.68 * np.clip(streak, 0.0, 1.0))
        mag_lon = _blur_separable_xy(longitudinal, passes=2, periodic_x=wrap_x) * 0.5 + long_scr * 0.5
        mag_blob = (
            _contrast_curve(mag_blob, 0.28, 0.94, 1.06)
            * band_envelope
            * mag_lon
            * (0.26 + 0.74 * np.clip(clumps, 0.0, 1.0))
        )
        gold_mid = gold_blob[..., None] * np.array([0.72, 0.62, 0.38], dtype=np.float64) * 0.085
        mag_mid = mag_blob[..., None] * np.array([0.42, 0.08, 0.30], dtype=np.float64) * 0.048
        color = np.clip(color + gold_mid, 0.0, 1.0)
        # Split veils so warm dust (gold) and H II (pink–magenta) stay separate spectrally.
        veil_gold = _blur_separable_xy(
            gold_blob * 1.35 * np.clip(streak, 0.0, 1.0),
            passes=2,
            periodic_x=wrap_x,
        )
        veil_hi_g = np.clip(veil_gold - _blur_separable_xy(veil_gold, passes=2, periodic_x=wrap_x), 0.0, 1.0)
        veil_gold = np.clip(veil_gold * 0.68 + veil_hi_g * 0.42, 0.0, 1.0)
        veil_hii = _blur_separable_xy(
            mag_blob * (0.85 + 0.15 * np.clip(activity_map, 0.0, 1.0)) * np.clip(streak, 0.0, 1.0),
            passes=2,
            periodic_x=wrap_x,
        )
        veil_hi_h = np.clip(veil_hii - _blur_separable_xy(veil_hii, passes=2, periodic_x=wrap_x), 0.0, 1.0)
        veil_hii = np.clip(veil_hii * 0.65 + veil_hi_h * 0.45, 0.0, 1.0)
        vg_add = veil_gold[..., None] * np.array([0.55, 0.50, 0.40], dtype=np.float64) * 0.038
        vh_add = veil_hii[..., None] * np.array([0.38, 0.10, 0.30], dtype=np.float64) * 0.017
        color = np.clip(color + vg_add, 0.0, 1.0)

        # Light merge on line emission only (preserve compact H II structure).
        emit_smooth = _blur_rgb_separable_xy(emit_rgb, passes=1, periodic_x=wrap_x)
        emit_rgb = np.clip(emit_rgb * 0.88 + emit_smooth * 0.12, 0.0, 0.68)

        dust_rgb = np.clip(color - emit_rgb, 0.0, 1.0)
        split_nebula_rgb = (dust_rgb, emit_rgb)

        mask *= 0.94 + 0.22 * core_band

    density_field = _sn(octaves=2)
    density_field = (density_field - 0.5) * (0.09 if split_nebula_rgb is not None else 0.15)
    mask *= 1.0 + density_field

    temp_shift = _sn(octaves=2)[..., None]
    warm = np.array([1.02, 0.94, 0.90], dtype=np.float64)
    cool = np.array([0.88, 0.92, 1.04], dtype=np.float64)
    temp = warm * temp_shift + cool * (1.0 - temp_shift)
    temp_sat = 0.985 if mode == NebulaMode.galaxy_streak else 0.9
    mul_te = temp * temp_sat
    if split_nebula_rgb is not None:
        d_sp, e_sp = split_nebula_rgb
        d_sp = d_sp * mul_te
        emit_ch = np.array([1.03, 0.995, 1.04], dtype=np.float64)
        e_sp = np.clip(e_sp * mul_te * emit_ch, 0.0, 1.0)
        color = np.clip(d_sp + e_sp, 0.0, 1.0)
        split_nebula_rgb = (d_sp, e_sp)
    else:
        color *= mul_te

    if split_nebula_rgb is not None:
        threshold = 0.518 + rng.uniform(-0.028, 0.028)
    else:
        threshold = 0.50 + rng.uniform(-0.05, 0.05)
    mask = np.clip((mask - threshold) * 1.48, 0.0, 1.0)
    glow = np.exp(-((1.0 - mask) * 3.0))
    mask *= 0.85 + 0.25 * glow
    micro = _sn(octaves=2)
    if split_nebula_rgb is not None:
        mask *= 0.982 + 0.038 * micro
    else:
        mask *= 0.965 + 0.07 * micro
    emission = mask**1.18
    if split_nebula_rgb is not None:
        d_sp, e_sp = split_nebula_rgb
        lobe_d = 0.80 + 0.52 * emission[..., None]
        lobe_e = 0.78 + 0.44 * emission[..., None]
        d_sp = d_sp * lobe_d
        e_sp = e_sp * lobe_e
        color = np.clip(d_sp + e_sp, 0.0, 1.0)
        split_nebula_rgb = (d_sp, e_sp)
    else:
        color = color * (0.74 + 0.42 * emission[..., None])

    dust = _sn(octaves=2)
    dust = np.clip((dust - 0.48) * 2.2, 0.0, 1.0)
    dust = dust**1.7
    dd_sz = max(2, min(height // 22, width // 22))
    dense_dust = _rz(rng.random((dd_sz, dd_sz)))
    dense_dust = _contrast_curve(dense_dust, 0.74, 0.995, 1.65)
    bp_sz = max(2, min(height // 20, width // 20))
    black_pockets = _rz(rng.random((bp_sz, bp_sz)))
    black_pockets = _contrast_curve(black_pockets, 0.58, 0.94, 1.35)
    mask -= dust * (0.18 + 0.10 * coverage_gain)
    if mode == NebulaMode.galaxy_streak:
        if not morph_primary:
            mask -= black_pockets * (0.10 + 0.08 * coverage_gain)
        mask -= dense_dust * (0.18 + 0.11 * coverage_gain)
    else:
        mask -= black_pockets * (0.16 + 0.12 * coverage_gain)
        mask -= dense_dust * (0.22 + 0.14 * coverage_gain)
    if mode == NebulaMode.galaxy_streak:
        pocket_dim = 0.0 if morph_primary else (0.085 + 0.042)
        dense_dim = 0.088 + 0.052 * strength_gain + 0.042
    else:
        pocket_dim = 0.10
        dense_dim = 0.10 + 0.06 * strength_gain
    pocket_dim_mul = 1.0 - (black_pockets * pocket_dim + dense_dust * dense_dim)[..., None]
    if split_nebula_rgb is not None:
        d_sp, e_sp = split_nebula_rgb
        d_sp *= pocket_dim_mul
        # Brown dust pockets dim continuum strongly; line gas keeps more chroma (was erasing H II).
        e_sp *= 0.58 + 0.42 * pocket_dim_mul
        color = np.clip(d_sp + e_sp, 0.0, 1.0)
        split_nebula_rgb = (d_sp, e_sp)
    else:
        color *= pocket_dim_mul
    mask = np.clip(mask, 0.0, 1.0)

    mask_tex = _sn(octaves=2)
    mask_tex = (mask_tex - 0.5) * 0.024
    mask *= 1.0 + mask_tex * 0.055
    _p(0.92)

    base_glow = _sn(octaves=2)
    base_glow = (base_glow - 0.5) * 0.065
    if mode == NebulaMode.galaxy_streak:
        # Slightly lifted floor so diffuse disk / unresolved-star haze reads above pure black.
        background = np.array([0.024, 0.022, 0.030], dtype=np.float64)
        if morph_primary:
            background = np.array([0.006, 0.005, 0.007], dtype=np.float64)
    else:
        background = np.array([0.014, 0.017, 0.024], dtype=np.float64)
    if split_nebula_rgb is not None:
        d_sp, e_sp = split_nebula_rgb
        if morph_primary and mode == NebulaMode.galaxy_streak:
            bg_gate = np.clip(streak[..., None] ** 1.08, 0.0, 1.0)
            d_sp = d_sp + (base_glow[..., None] + background) * bg_gate
        else:
            d_sp = d_sp + base_glow[..., None] + background
        color = np.clip(d_sp + e_sp, 0.0, 1.0)
        split_nebula_rgb = (d_sp, e_sp)
    else:
        color += base_glow[..., None]
        color += background
    _p(0.96)
    if mode == NebulaMode.galaxy_streak:
        dust_structure = _blur_separable_xy(dust_structure, passes=2, periodic_x=wrap_x)
    dust_str_w = (0.56 + 0.26 * coverage_gain) if mode == NebulaMode.galaxy_streak else (0.28 + 0.14 * coverage_gain)
    bp_for_ext = black_pockets
    dd_for_ext = dense_dust
    if mode == NebulaMode.galaxy_streak:
        bp_for_ext = _blur_x_only(
            _blur_y_only(
                _blur_separable_xy(black_pockets, passes=1, periodic_x=wrap_x),
                passes=2,
            ),
            passes=3,
            periodic_x=wrap_x,
        )
        dd_for_ext = _blur_x_only(
            _blur_y_only(
                _blur_separable_xy(dense_dust, passes=2, periodic_x=wrap_x),
                passes=3,
            ),
            passes=3,
            periodic_x=wrap_x,
        )
        rift_for_ext = _blur_x_only(
            _blur_separable_xy(rift_macro, passes=2, periodic_x=wrap_x),
            passes=2,
            periodic_x=wrap_x,
        )
        scaffold_for_ext = _blur_x_only(
            _blur_separable_xy(lane_scaffold, passes=2, periodic_x=wrap_x),
            passes=3,
            periodic_x=wrap_x,
        )
        lane_low = _blur_x_only(
            _blur_y_only(_blur_separable_xy(dust_structure, passes=2, periodic_x=wrap_x), passes=1),
            passes=1,
            periodic_x=wrap_x,
        )
        lane_low = np.clip(_contrast_curve(lane_low, 0.06, 0.97, 1.32), 0.0, 1.0)
    dust_occlusion = _contrast_curve(
        dust * 0.14
        + bp_for_ext * (0.28 + 0.12 * coverage_gain)
        + dd_for_ext * (0.78 + 0.32 * coverage_gain)
        + dust_structure * dust_str_w,
        0.30,
        0.997,
        1.88 + 0.22 * strength_gain,
    )
    if mode == NebulaMode.galaxy_streak:
        # Extinction should be driven by coherent lane fields, not pixel-scale dust texture.
        if morph_primary:
            ds_hi = np.clip(
                dust_structure
                - _blur_separable_xy(dust_structure, passes=1, periodic_x=wrap_x) * 0.82,
                0.0,
                1.0,
            ) ** 1.06
            turb_cloud = _contrast_curve(_sn(octaves=4), 0.20, 0.94, 1.22)
            turb_cloud *= np.clip(streak * 0.72 + broad_band * 0.28, 0.0, 1.0)
            turb_cloud = _blur_x_only(
                _blur_y_only(turb_cloud, passes=1),
                passes=1,
                periodic_x=wrap_x,
            )
            dust_occlusion = _contrast_curve(
                lane_low * (0.68 + 0.24 * strength_gain)
                + rift_for_ext * (0.76 + 0.32 * strength_gain)
                + scaffold_for_ext * (0.28 + 0.12 * strength_gain)
                + ds_hi * (0.26 + 0.10 * strength_gain)
                + turb_cloud * (0.32 + 0.14 * coverage_gain),
                0.12,
                0.994,
                2.08 + 0.26 * strength_gain,
            )
        else:
            dust_occlusion = _contrast_curve(
                dust * 0.04
                + bp_for_ext * (0.12 + 0.06 * coverage_gain)
                + dd_for_ext * (0.22 + 0.12 * coverage_gain)
                + dust_structure * (0.14 + 0.08 * coverage_gain)
                + lane_low * (0.78 + 0.30 * strength_gain)
                + rift_for_ext * (0.82 + 0.36 * strength_gain)
                + scaffold_for_ext * (0.32 + 0.14 * strength_gain),
                0.12,
                0.994,
                2.05 + 0.24 * strength_gain,
            )
        vert_lump = _contrast_curve(_sn(octaves=4), 0.18, 0.94, 1.05)
        band_wobble = (vert_lump - 0.5) * 0.34
        band_width = 0.62 + 0.38 * _contrast_curve(_sn(octaves=3), 0.22, 0.90, 0.92)
        band_bias = np.exp(-(((y + band_wobble) ** 2) / np.maximum(band_width, 0.28)))
        if morph_primary:
            center_bias = np.clip(0.58 + 0.42 * np.clip(streak, 0.0, 1.0), 0.45, 1.0)
            dust_occlusion = np.clip(
                dust_occlusion * (0.22 + 0.78 * band_bias) * (0.68 + 0.32 * center_bias)
                + rift_for_ext * (0.18 + 0.20 * strength_gain)
                + scaffold_for_ext * (0.08 + 0.10 * strength_gain),
                0.0,
                1.0,
            )
        else:
            cx_off = float(sign * rng.uniform(0.0, 0.22))
            cy_off = float(rng.uniform(-0.12, 0.12))
            skew = float(sign * rng.uniform(0.08, 0.26))
            xw = _wrap_dx(x, cx_off + skew * y) if wrap_x else (x - cx_off - skew * y)
            yw = y - cy_off
            radial_core = np.exp(-((xw**2) / 0.48) - ((yw**2) / (1.05 + 0.55 * vert_lump)))
            lump = vert_lump
            center_bias = np.clip(0.32 + 0.68 * radial_core * (0.40 + 0.60 * lump), 0.18, 1.0)
            xskew_off = float(sign * rng.uniform(0.08, 0.20)) * y
            xskew = _wrap_dx(x, xskew_off) if wrap_x else (x - xskew_off)
            dense_center = np.exp(-((xskew**2) / 0.32)) * (0.48 + 0.52 * lump)
            dense_dust = dense_dust * (0.28 + 0.72 * dense_center)
            dust_occlusion = np.clip(
                dust_occlusion * (0.18 + 0.82 * band_bias) * (0.52 + 0.48 * center_bias)
                + dense_dust * (0.05 + 0.06 * strength_gain)
                + rift_for_ext * (0.16 + 0.18 * strength_gain)
                + scaffold_for_ext * (0.06 + 0.08 * strength_gain),
                0.0,
                1.0,
            )
        dust_occlusion = np.clip(dust_occlusion**1.14, 0.0, 1.0)
        dust_occlusion = _blur_y_only(dust_occlusion, passes=1)
        dust_occlusion = _blur_x_only(dust_occlusion, passes=1, periodic_x=wrap_x)
        dust_occlusion = np.clip(dust_occlusion, 0.0, 1.0)
        lane_extinction = np.clip(
            _contrast_curve(
                lane_low * 0.56 + rift_for_ext * 0.34 + scaffold_for_ext * 0.18,
                0.02,
                0.993,
                1.52,
            ),
            0.0,
            1.0,
        )
        if mega_on:
            mega_align = np.clip(_blur_separable_xy(mega_env, passes=1, periodic_x=wrap_x), 0.0, 1.0)
            lane_extinction = np.clip(lane_extinction * (0.72 + 0.28 * mega_align), 0.0, 1.0)
        lane_extinction = _blur_y_only(lane_extinction, passes=1)
        lane_extinction = _blur_x_only(lane_extinction, passes=1, periodic_x=wrap_x)
        lane_extinction = np.clip(lane_extinction**1.12, 0.0, 1.0)
    if split_nebula_rgb is not None:
        dust_rgb, emit_rgb = split_nebula_rgb
        m_shell = np.clip((mask[..., None] * 0.86 + 0.078) * 0.96, 0.0, 1.0)
        # Vertical taper: stronger on continuum dust; ionized gas can sit slightly "above" the plane.
        if morph_primary:
            streak_s = np.clip(streak, 0.0, 1.0)
            shoulder = np.exp(-((y**2) / 0.58))
            band_taper_d = np.clip(streak_s**1.02 + shoulder * 0.14, 0.0, 1.0)[..., None]
            band_taper_e = np.clip(streak_s**0.98 + shoulder * 0.10, 0.0, 1.0)[..., None]
            band_taper_d = band_taper_d * band_lat[..., None]
            band_taper_e = band_taper_e * (0.88 + 0.12 * band_lat[..., None])
        else:
            band_taper_d = (0.72 + 0.28 * np.clip(streak, 0.0, 1.0) ** 1.05)[..., None]
            band_taper_e = (0.84 + 0.16 * np.clip(streak, 0.0, 1.0) ** 1.02)[..., None]
        m_dust = m_shell * band_taper_d
        m_emit = m_shell * band_taper_e
        # Build dual-field dust model:
        # - occluder field controls silhouettes/extinction agreement
        # - emissive continuum field is synthesized separately (not just blurred occluder RGB)
        dust_gas = _blur_rgb_separable_xy(dust_rgb, passes=2, periodic_x=wrap_x)
        dust_diffuse = _blur_rgb_separable_xy(dust_rgb, passes=4, periodic_x=wrap_x)
        neb_occ = (dust_rgb * 0.32 + dust_gas * 0.38 + dust_diffuse * 0.18) * m_dust
        cont_raw = np.clip(
            mask
            * (0.58 + 0.42 * np.clip(streak, 0.0, 1.0))
            * (0.62 + 0.38 * np.clip(activity_map, 0.0, 1.0))
            * (0.52 + 0.48 * np.clip(clumps, 0.0, 1.0)),
            0.0,
            1.0,
        )
        cont_seed = _blur_separable_xy(cont_raw, passes=1, periodic_x=wrap_x)
        cont_noise = _blur_separable_xy(_sn(octaves=3), passes=1, periodic_x=wrap_x)
        cont_shape = np.clip(
            cont_seed * (0.70 + 0.30 * cont_noise),
            0.0,
            1.0,
        )
        cont_hi = np.clip(
            cont_raw - _blur_separable_xy(cont_raw, passes=3, periodic_x=wrap_x),
            0.0,
            1.0,
        )
        cont_shape = np.clip(cont_shape + cont_hi * 0.48, 0.0, 1.0)
        galaxy_boost = (1.62 if morph_primary else 1.85) if gf is not None else 1.0
        cont_rgb = (
            cont_shape[..., None]
            * np.array([0.16, 0.152, 0.142], dtype=np.float64)
            * galaxy_boost
        )
        if tuning.debug_pass == "occluder_only":
            neb_dbg = np.clip(neb_occ, 0.0, 1.0)
            emit_dbg = np.zeros((height, width, 3), dtype=np.float64)
            return neb_dbg, emit_dbg, dust_occlusion, lane_extinction
        if tuning.debug_pass == "continuum_only":
            band_vis = np.clip(streak[..., None], 0.0, 1.0) ** 1.02
            neb_dbg = np.clip(cont_rgb * 2.85 * band_vis, 0.0, 1.0)
            emit_dbg = np.zeros((height, width, 3), dtype=np.float64)
            return neb_dbg, emit_dbg, dust_occlusion, lane_extinction
        neb = np.clip(neb_occ + cont_rgb * (0.68 if gf is not None else 0.54), 0.0, 1.0)
        if gf is not None:
            neb = np.clip(neb * 1.22, 0.0, 1.0)
        # Pass 2: H II — moderate merge so knots read as regions, not single pixels.
        emit_gas = _blur_rgb_separable_xy(emit_rgb, passes=1, periodic_x=wrap_x)
        emit_mul = 1.28 if gf is not None else 1.06
        neb_emit = np.clip((emit_rgb * 0.82 + emit_gas * 0.18) * m_emit * emit_mul, 0.0, 1.0)
        neb_emit = forward_scatter_mie(
            neb_emit,
            np.mean(neb_emit, axis=2),
            strength=0.052,
            periodic_x=wrap_x,
            blur_fn=_blur_separable_xy,
        )
        if galaxy_dust_pack is not None:
            fthin = np.clip(0.52 - galaxy_dust_pack["fil_dense"], 0.0, 1.0)
            boost = (fthin**1.25) * np.clip(streak, 0.0, 1.0) * (0.032 + 0.022 * cloud_gain)
            neb_emit = np.clip(
                neb_emit
                + boost[..., None] * np.array([0.48, 0.44, 0.90], dtype=np.float64),
                0.0,
                1.0,
            )

        dust_for_neb = np.clip(
            _blur_separable_xy(dust_structure, passes=2, periodic_x=wrap_x) ** 0.36, 0.0, 1.0
        )
        neb *= (0.80 + 0.20 * (1.0 - 0.78 * dust_for_neb))[..., None]
        neb_emit *= (0.90 + 0.10 * (1.0 - 0.55 * dust_for_neb))[..., None]
        # Enforce lane silhouettes in visible gas: RGB must obey the same macro lanes as extinction.
        lane_macro = np.clip(
            _blur_x_only(
                _blur_separable_xy(lane_extinction, passes=2, periodic_x=wrap_x),
                passes=3,
                periodic_x=wrap_x,
            ),
            0.0,
            1.0,
        )
        lane_deep = np.clip(_contrast_curve(lane_macro, 0.20, 0.995, 1.10), 0.0, 1.0)
        ext_gate = np.clip(1.0 - lane_macro, 0.0, 1.0)
        ext_gate_d = np.clip(ext_gate**1.28, 0.03, 1.0)
        ext_gate_e = np.clip(ext_gate**0.96, 0.07, 1.0)
        neb = blend_darken_preserve_contrast(neb, ext_gate_d, mask=lane_macro)
        neb_emit = blend_darken_preserve_contrast(neb_emit, ext_gate_e, mask=lane_macro * 0.85)
        # Keep deepest occluder lanes dark: almost no continuum fill in the strongest trenches.
        deep_block_d = np.clip(1.0 - lane_deep * 0.86, 0.10, 1.0)
        deep_block_e = np.clip(1.0 - lane_deep * 0.56, 0.28, 1.0)
        neb *= deep_block_d[..., None]
        neb_emit *= deep_block_e[..., None]

        plane_glow = np.clip(streak**1.08, 0.0, 1.0)
        neb *= (0.93 + 0.16 * plane_glow)[..., None]
        neb_emit *= (0.96 + 0.14 * plane_glow)[..., None]

        y_band = np.exp(-((y**2) / (0.44 if morph_primary else 0.92)))
        if morph_primary:
            y_band = np.clip(y_band * (0.82 + 0.18 * band_lat), 0.0, 1.0)
        lum_neb = np.mean(neb, axis=2)
        gsz_y = max(10, height // 10)
        gsz_x = max(20, width // 16)
        sm = _resize_bilinear(lum_neb * y_band, gsz_y, gsz_x, periodic_x=True)
        horiz = _resize_bilinear(sm, height, width, periodic_x=True)
        horiz = _blur_separable_xy(horiz, passes=2, periodic_x=wrap_x)
        horiz = np.clip(horiz**1.06, 0.0, 1.0)
        warm_h = np.array([0.98, 0.78, 0.14], dtype=np.float64)
        neb = neb + horiz[..., None] * warm_h * (0.030 + 0.011 * cloud_gain) * ext_gate_d[..., None] * deep_block_d[..., None]

        floor_warp = np.clip(_sn(octaves=2), 0.0, 1.0)
        diffuse_floor = y_band * (0.016 + 0.013 * _sn(octaves=1))
        diffuse_floor *= 0.80 + 0.40 * floor_warp
        floor_rgb = np.array([0.085, 0.076, 0.048], dtype=np.float64)
        neb = np.clip(
            neb + diffuse_floor[..., None] * floor_rgb * (0.60 + 0.40 * ext_gate_d[..., None]) * deep_block_d[..., None],
            0.0,
            1.0,
        )

        lane_w = _contrast_curve(dust_structure, 0.05, 0.995, 1.14)
        lk = (0.76 + 0.44 * strength_gain) * (0.12 + 0.88 * core_band)
        neb *= np.clip(1.0 - lk[..., None] * lane_w[..., None] * 0.92, 0.24, 1.0)
        neb_emit *= np.clip(1.0 - lk[..., None] * lane_w[..., None] * 0.40, 0.58, 1.0)

        ds_blur = _blur_separable_xy(dust_structure, passes=2, periodic_x=wrap_x)
        dust_edge = np.clip(dust_structure - 0.86 * ds_blur, 0.0, 1.0) ** 1.05
        dust_edge = _blur_separable_xy(dust_edge, passes=1, periodic_x=wrap_x)
        rim = dust_edge * np.clip(streak * 0.92 + broad_band * 0.08, 0.0, 1.0)
        rim_rgb = np.array([0.62, 0.46, 0.18], dtype=np.float64)
        rim_mag = np.array([0.42, 0.08, 0.28], dtype=np.float64)
        neb = np.clip(
            neb + rim[..., None] * rim_rgb * (0.070 + 0.024 * cloud_gain) * ext_gate_d[..., None] * deep_block_d[..., None],
            0.0,
            1.0,
        )
        neb_emit = np.clip(
            neb_emit + rim[..., None] * rim_mag * (0.030 + 0.012 * cloud_gain) * ext_gate_e[..., None] * deep_block_e[..., None],
            0.0,
            1.0,
        )

        ripple_a = _sn(octaves=2)
        ripple_b = _blur_separable_xy(_sn(octaves=4), passes=2, periodic_x=wrap_x)
        ripple = (ripple_a * 0.68 + ripple_b * 0.32 - 0.5) * 2.0
        ripple = np.clip(ripple, -0.075, 0.075) * y_band
        neb = np.clip(
            neb
            + ripple[..., None]
            * np.array([0.020, 0.020, 0.026], dtype=np.float64)
            * ext_gate_d[..., None]
            * deep_block_d[..., None],
            0.0,
            1.0,
        )
        turb = _blur_separable_xy(_sn(octaves=3), passes=2, periodic_x=wrap_x)
        neb = np.clip(
            neb
            + (turb - 0.5)[..., None]
            * np.array([0.009, 0.009, 0.012], dtype=np.float64)
            * y_band[..., None]
            * ext_gate_d[..., None],
            0.0,
            1.0,
        )

        em_lum_d = np.clip(np.mean(neb, axis=2), 0.0, 1.0)
        em_lum_e = np.clip(np.mean(neb_emit, axis=2), 0.0, 1.0)
        lift_d = (em_lum_d**0.62)[..., None] * y_band[..., None]
        lift_e = (em_lum_e**0.58)[..., None] * y_band[..., None]
        streak_m = np.clip(streak[..., None], 0.0, 1.0)
        neb = np.clip(neb + lift_d * np.array([0.065, 0.052, 0.038], dtype=np.float64), 0.0, 1.0)
        neb_emit = np.clip(
            neb_emit
            + lift_e * np.array([0.42, 0.06, 0.38], dtype=np.float64) * streak_m
            + lift_e * np.array([0.16, 0.03, 0.14], dtype=np.float64) * (1.0 - streak_m * 0.35),
            0.0,
            1.0,
        )

        neb = _detail_blend_after_blur(neb, passes=2, detail_mix=0.62, chroma_boost=1.28, periodic_x=wrap_x)
        neb_emit = _detail_blend_after_blur(
            neb_emit, passes=2, detail_mix=0.54, chroma_boost=1.14, periodic_x=wrap_x
        )
        lm_d = np.mean(neb, axis=2, keepdims=True)
        neb = np.clip(lm_d + (neb - lm_d) * (1.14 if morph_primary else 1.26), 0.0, 1.0)
        lm_e = np.mean(neb_emit, axis=2, keepdims=True)
        neb_emit = np.clip(lm_e + (neb_emit - lm_e) * 1.06, 0.0, 1.0)
        neb_emit = np.clip(neb_emit * 1.05, 0.0, 1.0)
        if morph_primary:
            from starsky_gen.structure_envelope import latitude_plane_gate

            plane_cut = latitude_plane_gate(height, sigma=0.58, power=0.88)
            neb = neb * plane_cut[..., np.newaxis]
            neb_emit = neb_emit * plane_cut[..., np.newaxis]
        _p(1.0)
        return (
            np.clip(neb, 0.0, 1.0),
            np.clip(neb_emit, 0.0, 1.0),
            dust_occlusion,
            lane_extinction,
        )
    emit_empty = np.zeros((height, width, 3), dtype=np.float64)
    _p(1.0)
    return color * (mask[..., None] * 0.38 + 0.05), emit_empty, dust_occlusion, lane_extinction
