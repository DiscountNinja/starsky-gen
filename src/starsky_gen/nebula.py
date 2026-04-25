from __future__ import annotations

import numpy as np

from starsky_gen.config import NebulaMode, NebulaTuningConfig


def _resize_bilinear(field: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D array with separable linear interpolation."""
    src_h, src_w = field.shape
    y_src = np.arange(src_h, dtype=np.float64)
    x_src = np.arange(src_w, dtype=np.float64)
    y_dst = np.linspace(0.0, src_h - 1, out_h, dtype=np.float64)
    x_dst = np.linspace(0.0, src_w - 1, out_w, dtype=np.float64)

    row_interp = np.empty((out_h, src_w), dtype=np.float64)
    for x in range(src_w):
        row_interp[:, x] = np.interp(y_dst, y_src, field[:, x])

    out = np.empty((out_h, out_w), dtype=np.float64)
    for y in range(out_h):
        out[y, :] = np.interp(x_dst, x_src, row_interp[y, :])
    return out


def _blur_separable_xy(field: np.ndarray, passes: int = 1, *, alternate: bool = True) -> np.ndarray:
    """Small separable blur; alternate pass order (x/y vs y/x) to avoid row/column streak buildup."""
    out = np.clip(field.astype(np.float64, copy=False), 0.0, 1.0)
    k0, k1, k2 = 0.22, 0.56, 0.22
    for pi in range(passes):
        y_first = alternate and (pi % 2 == 1)
        if y_first:
            padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
            out = k0 * padded[:-2, :] + k1 * padded[1:-1, :] + k2 * padded[2:, :]
            padded = np.pad(out, ((0, 0), (1, 1)), mode="edge")
            out = k0 * padded[:, :-2] + k1 * padded[:, 1:-1] + k2 * padded[:, 2:]
        else:
            padded = np.pad(out, ((0, 0), (1, 1)), mode="edge")
            out = k0 * padded[:, :-2] + k1 * padded[:, 1:-1] + k2 * padded[:, 2:]
            padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
            out = k0 * padded[:-2, :] + k1 * padded[1:-1, :] + k2 * padded[2:, :]
    return np.clip(out, 0.0, 1.0)


def _blur_rgb_separable_xy(rgb: np.ndarray, passes: int = 2) -> np.ndarray:
    """Separable blur on each channel (gas RGB is 3D; _blur_separable_xy is 2D-only)."""
    return np.stack(
        [_blur_separable_xy(rgb[:, :, c], passes=passes) for c in range(3)],
        axis=2,
    )


def _blur_y_only(field: np.ndarray, passes: int = 2) -> np.ndarray:
    """Vertical-only blur to soften comb-like edges after strong horizontal stretching."""
    out = np.clip(field.astype(np.float64, copy=False), 0.0, 1.0)
    k0, k1, k2 = 0.15, 0.70, 0.15
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = k0 * padded[:-2, :] + k1 * padded[1:-1, :] + k2 * padded[2:, :]
    return np.clip(out, 0.0, 1.0)


def _blur_x_only(field: np.ndarray, passes: int = 2) -> np.ndarray:
    """Horizontal-only blur to merge column-coherent dust into cloud-like masses."""
    out = np.clip(field.astype(np.float64, copy=False), 0.0, 1.0)
    k0, k1, k2 = 0.15, 0.70, 0.15
    for _ in range(passes):
        padded = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = k0 * padded[:, :-2] + k1 * padded[:, 1:-1] + k2 * padded[:, 2:]
    return np.clip(out, 0.0, 1.0)


def _smooth_noise(rng: np.random.Generator, height: int, width: int, octaves: int = 4) -> np.ndarray:
    noise = np.zeros((height, width), dtype=np.float64)
    amp_sum = 0.0
    for o in range(octaves):
        scale = 2 ** (o + 2)
        sample_h = max(2, height // scale)
        sample_w = max(2, width // scale)
        coarse = rng.random((sample_h, sample_w))
        layer = _resize_bilinear(coarse, height, width)
        amp = 0.6 ** o
        noise += layer * amp
        amp_sum += amp
    return noise / max(amp_sum, 1e-6)


def _contrast_curve(field: np.ndarray, low: float, high: float, gamma: float) -> np.ndarray:
    """Normalize and reshape contrast with a soft gamma curve."""
    t = np.clip((field - low) / max(high - low, 1e-6), 0.0, 1.0)
    return t**gamma


def generate_nebula(
    rng: np.random.Generator,
    mode: NebulaMode,
    height: int,
    width: int,
    tuning: NebulaTuningConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    n = _smooth_noise(rng, height, width, octaves=5)
    warp = _smooth_noise(rng, height, width, octaves=2)
    wx = (warp - 0.5) * 0.8
    wy = (_smooth_noise(rng, height, width, octaves=2) - 0.5) * 0.8
    y_idx = np.clip((np.arange(height)[:, None] + wy * height).astype(int), 0, height - 1)
    x_idx = np.clip((np.arange(width)[None, :] + wx * width).astype(int), 0, width - 1)
    n = n[y_idx, x_idx]

    color_field = _smooth_noise(rng, height, width, octaves=2)
    color_mix = np.clip(color_field * 1.2, 0.0, 1.0)[..., None]
    y = np.linspace(-1.0, 1.0, height)[:, None]
    x = np.linspace(-1.0, 1.0, width)[None, :]
    dust_structure = np.zeros((height, width), dtype=np.float64)
    core_band = np.ones((height, width), dtype=np.float64)
    lane_extinction = np.zeros((height, width), dtype=np.float64)
    # galaxy_streak only: (dust_rgb, emit_rgb) after shared color ops, for separate nebula passes.
    split_nebula_rgb: tuple[np.ndarray, np.ndarray] | None = None

    if mode == NebulaMode.distant:
        cx = rng.uniform(-0.5, 0.5)
        cy = rng.uniform(-0.4, 0.4)
        blob = np.exp(-(((x - cx) ** 2) / 0.18 + ((y - cy) ** 2) / 0.10))
        mask = (n * 0.7 + blob * 0.9).clip(0.0, 1.0) ** 1.6
        tint_a = np.array([0.30, 0.18, 0.38], dtype=np.float64)
        tint_b = np.array([0.58, 0.34, 0.64], dtype=np.float64)
        color = tint_a * (1.0 - color_mix) + tint_b * color_mix
    elif mode == NebulaMode.full:
        broad = np.exp(-((y**2) / 0.9))
        mask = (n * 0.9 + broad * 0.35).clip(0.0, 1.0) ** 1.2
        tint_a = np.array([0.26, 0.16, 0.34], dtype=np.float64)
        tint_b = np.array([0.54, 0.31, 0.60], dtype=np.float64)
        color = tint_a * (1.0 - color_mix) + tint_b * color_mix
    else:
        # Off-center bulge: skewed normal so the bright core is usually asymmetric.
        sign = -1.0 if rng.random() < 0.5 else 1.0
        cx_band = float(np.clip(rng.normal(sign * rng.uniform(0.10, 0.36), 0.12), -0.55, 0.55))
        core_band = np.exp(-((x - cx_band) ** 2) / 0.28)
        band_warp = (_smooth_noise(rng, height, width, octaves=1) - 0.5) * 0.32
        streak = np.exp(-(((y - band_warp) ** 2) / 0.48))
        broad_band = np.exp(-((y**2) / 1.08))
        outer_band = np.exp(-((y**2) / 1.72))
        bleed_haze = (_smooth_noise(rng, height, width, octaves=1) * 0.55 + 0.45).clip(0.45, 1.0)
        dark_lanes = (1.0 - _smooth_noise(rng, height, width, octaves=3) * 0.7).clip(0.20, 1.0)
        patch = _smooth_noise(rng, height, width, octaves=2)
        patch = np.where(patch > 0.58, 1.0, patch * 0.35)
        density_breaks = (0.55 + patch * 0.75).clip(0.30, 1.25)
        sn_mv = _smooth_noise(rng, height, width, octaves=2)
        micro_voids = 1.0 - np.clip((sn_mv - 0.52) / 0.38, 0.0, 1.0) * 0.22
        # Coarse clumps provide large-scale galactic structure.
        clump_coarse = _resize_bilinear(
            rng.random((max(2, height // 28), max(2, width // 28))), height, width
        )
        clump_mid = _resize_bilinear(
            rng.random((max(2, height // 14), max(2, width // 14))), height, width
        )
        clumps = _contrast_curve(clump_coarse * 0.65 + clump_mid * 0.35, 0.24, 0.92, 0.74)
        # Thickness map creates denser bulges in parts of the galactic band.
        th_sz = max(2, min(height // 22, width // 22))
        thickness_map = _resize_bilinear(rng.random((th_sz, th_sz)), height, width)
        thickness_map = _contrast_curve(thickness_map, 0.34, 0.90, 0.88)
        # Slow horizontal modulation keeps the whole band active without uniform flatness.
        longitudinal_1d = _smooth_noise(rng, 1, width, octaves=3)[0]
        x_line = np.linspace(-1.0, 1.0, width, dtype=np.float64)
        long_skew = float(rng.uniform(0.09, 0.24)) * float(rng.choice([-1.0, 1.0])) * x_line
        longitudinal_1d = longitudinal_1d + long_skew + float(rng.uniform(-0.07, 0.07))
        longitudinal = _contrast_curve(longitudinal_1d[None, :], 0.08, 0.94, 0.74)
        long_warp2d = np.clip(_smooth_noise(rng, height, width, octaves=2), 0.0, 1.0)
        # Strong 2D modulation — pure 1D longitude reads as vertical stripes in mask → dust columns.
        longitudinal = np.clip(longitudinal * (0.42 + 0.58 * long_warp2d), 0.0, 1.0)
        long_scr = np.clip(_smooth_noise(rng, height, width, octaves=4), 0.0, 1.0)
        longitudinal = np.clip(longitudinal * (0.72 + 0.28 * long_scr), 0.0, 1.0)
        long_break = np.clip(_smooth_noise(rng, height, width, octaves=5), 0.0, 1.0)
        longitudinal = np.clip(longitudinal * (0.68 + 0.32 * long_break), 0.0, 1.0)
        longitudinal = _blur_separable_xy(longitudinal, passes=2)
        # Break pure column correlation (1D longitude × blur_x reads as vertical comb in gas/dust).
        destripe = _blur_separable_xy(_smooth_noise(rng, height, width, octaves=3), passes=2)
        longitudinal = np.clip(longitudinal * 0.64 + destripe * 0.36, 0.0, 1.0)
        longitudinal = _blur_separable_xy(longitudinal, passes=2)
        # Broad activity zones restore large-scale non-uniformity along longitude.
        activity_map = _resize_bilinear(
            rng.random((max(2, height // 12), max(2, width // 11))), height, width
        )
        activity_map = _contrast_curve(activity_map, 0.18, 0.92, 0.72)
        activity_map = _blur_separable_xy(activity_map, passes=1)
        # Sparse macro structure: occasional cavities and dense knots.
        macro_voids = _resize_bilinear(
            rng.random((max(2, height // 24), max(2, width // 14))), height, width
        )
        macro_voids = _contrast_curve(macro_voids, 0.76, 0.99, 1.60)
        heavy_spots = _resize_bilinear(
            rng.random((max(2, height // 13), max(2, width // 9))), height, width
        )
        heavy_spots = _contrast_curve(heavy_spots, 0.60, 0.95, 0.95)
        heavy_spots = _blur_separable_xy(heavy_spots, passes=1)
        # Bridge field links neighboring clumps into more continuous band segments.
        bridge_map = _resize_bilinear(
            rng.random((max(2, height // 12), max(2, width // 7))), height, width
        )
        bridge_map = _contrast_curve(bridge_map, 0.18, 0.90, 0.78)
        bridge_map = _blur_separable_xy(bridge_map, passes=1)

        # Lane texture: finer in x than before to avoid tall vertical cell boundaries (stripe artifacts).
        lane_coarse = rng.random((max(2, height // 36), max(2, width // 17)))
        lane_noise = _resize_bilinear(lane_coarse, height, width)
        lane_warp = (_smooth_noise(rng, height, width, octaves=1) - 0.5) * 0.12
        lane_noise = np.clip(lane_noise * (0.90 + 0.20 * lane_warp), 0.0, 1.0)
        lane_noise = _blur_x_only(lane_noise, passes=2)
        lane_noise = _blur_y_only(lane_noise, passes=2)
        lane_cut = _contrast_curve(lane_noise, 0.44, 0.90, 1.18)
        filament_gate = _resize_bilinear(
            rng.random((max(2, height // 14), max(2, width // 8))), height, width
        )
        filament_gate = _contrast_curve(filament_gate, 0.64, 0.98, 1.45)
        filament_gate = _blur_separable_xy(filament_gate, passes=1)
        break_mask = _resize_bilinear(
            rng.random((max(2, height // 14), max(2, width // 7))), height, width
        )
        break_mask = _contrast_curve(break_mask, 0.58, 0.96, 1.35)
        break_mask = _blur_separable_xy(break_mask, passes=1)
        # Keep only portions of lanes to avoid "everywhere" continuous streaking.
        lane_cut *= 0.62 + filament_gate * 0.78
        lane_cut *= 1.0 - break_mask * 0.11
        lane_cut = np.clip(lane_cut, 0.0, 1.0)
        # One thick, coherent dust lane (silhouette) offset from the band center.
        dom_x = float(np.clip(rng.normal(sign * rng.uniform(0.14, 0.42), 0.16), -0.68, 0.68))
        dom_y = float(np.clip(rng.normal(0.0, 0.048), -0.15, 0.15))
        sig_x = rng.uniform(0.52, 1.02)
        sig_y = rng.uniform(0.102, 0.175)
        dominant_ridge = np.exp(-(((x - dom_x) ** 2) / sig_x + ((y - dom_y) ** 2) / sig_y))
        dominant_ridge *= np.clip(streak * 0.62 + broad_band * 0.38, 0.0, 1.0) * (0.52 + 0.48 * filament_gate)
        dominant_ridge = _contrast_curve(dominant_ridge, 0.38, 0.995, 1.92)
        lane_cut = np.clip(lane_cut + dominant_ridge * (0.58 + 0.20 * strength_gain), 0.0, 1.0)
        lane_cut = _blur_separable_xy(lane_cut, passes=1)
        lane_cut = _blur_x_only(lane_cut, passes=2)
        lane_cut = _blur_y_only(lane_cut, passes=2)
        lane_cut = _blur_separable_xy(lane_cut, passes=1)
        lane_cut = _blur_separable_xy(lane_cut, passes=2)
        lane_cut = _blur_y_only(lane_cut, passes=1)
        dark_lane_factor = np.clip(1.0 - lane_cut * 0.84, 0.05, 1.0)

        band_envelope = (streak * 0.70 + broad_band * 0.22 + outer_band * 0.08).clip(0.0, 1.0)
        # Coarse volume field suppresses speckle and makes clouds read as connected masses.
        volume_soft = _resize_bilinear(
            rng.random((max(2, height // 16), max(2, width // 14))), height, width
        )
        volume_soft = _contrast_curve(volume_soft, 0.18, 0.90, 0.88)
        volume_soft = _blur_separable_xy(volume_soft, passes=2)
        # Extra very-low-frequency envelope so the disk reads as one lit volume, not same-scale clumps.
        vol_mega = _resize_bilinear(
            rng.random((max(2, height // 52), max(2, width // 20))), height, width
        )
        vol_mega = _contrast_curve(vol_mega, 0.10, 0.88, 0.78)
        vol_mega = _blur_separable_xy(vol_mega, passes=5)
        vol_mega = _blur_x_only(vol_mega, passes=3)
        volume_soft = np.clip(volume_soft * 0.58 + vol_mega * 0.42, 0.0, 1.0)
        # Warped `n` is 5-octave fBm — too much in the volume mask reads as sand, not clouds.
        n_smooth = _blur_separable_xy(n, passes=2)
        fine_density = (
            n_smooth * 0.14
            + _smooth_noise(rng, height, width, octaves=2) * 0.38
            + volume_soft * 0.48
        )
        base = band_envelope * bleed_haze * (
            0.30 + fine_density * 0.24 + clumps * 0.82 + thickness_map * 0.68 + volume_soft * 0.94
        )
        base = np.maximum(base, band_envelope * bridge_map * (0.82 + 0.12 * cloud_gain))
        # Shallower longitude gain (less column-driven contrast in the volume mask).
        base *= 0.88 + longitudinal * (0.14 + 0.10 * long_warp2d)
        base *= 0.70 + activity_map * 0.58
        base *= 0.84 + heavy_spots * (0.26 + 0.12 * cloud_gain)
        x_ramp = 1.0 + float(rng.uniform(0.08, 0.19)) * (x * float(rng.choice([-1.0, 1.0])))
        x_ramp_2d = np.clip(
            x_ramp * (0.86 + 0.14 * np.clip(_smooth_noise(rng, height, width, octaves=3), 0.0, 1.0)),
            0.85,
            1.22,
        )
        base *= x_ramp_2d
        detail = dark_lanes * density_breaks * micro_voids * dark_lane_factor
        detail *= 0.84 + (1.0 - activity_map) * 0.28
        mask = (base * (0.86 + 0.05 * cloud_gain) + detail * (0.040 - 0.013 * (cloud_gain - 1.0))).clip(0.0, 1.0)
        mask = mask * 0.76 + volume_soft * (0.10 + 0.05 * cloud_gain) + bridge_map * (0.11 + 0.05 * cloud_gain)
        mask *= 1.0 - macro_voids * (0.18 + 0.08 * coverage_gain)
        mask = _contrast_curve(mask, 0.22, 0.96, 0.90)
        mask = _blur_separable_xy(mask, passes=2)
        mask = _blur_separable_xy(mask, passes=3)
        mask = _blur_x_only(mask, passes=2)
        mask = _blur_separable_xy(mask, passes=2)
        mask = _blur_separable_xy(mask, passes=1)
        dust_structure = np.clip(
            lane_cut * (0.40 + 0.13 * coverage_gain)
            + macro_voids * (0.26 + 0.16 * coverage_gain)
            + filament_gate * (0.40 + 0.06 * cloud_gain)
            + (1.0 - mask) * 0.18
            + dominant_ridge * (0.32 + 0.17 * strength_gain),
            0.0,
            1.0,
        )
        # Great Rift–style trench: one broad, tilted dark lane through the disk (not mirrored).
        rx = float(np.clip(rng.normal(sign * rng.uniform(0.12, 0.44), 0.14), -0.62, 0.62))
        ry = float(np.clip(rng.normal(0.0, 0.055), -0.12, 0.12))
        wx_r = rng.uniform(0.14, 0.32)
        wy_r = rng.uniform(0.56, 1.05)
        band_gate = np.clip(streak * 0.86 + broad_band * 0.14, 0.0, 1.0)
        rift_wx = (_smooth_noise(rng, height, width, octaves=2) - 0.5) * 0.11 * band_gate
        rift_wy = (_smooth_noise(rng, height, width, octaves=2) - 0.5) * 0.09 * band_gate
        rift_oval = np.exp(
            -(((x + rift_wx - rx) ** 2) / wx_r + ((y + rift_wy - ry) ** 2) / wy_r)
        ) * band_gate
        tilt = float(rng.uniform(-0.42, 0.42))
        xc = (x + rift_wx * 0.85 - rx) * np.cos(tilt) + (y + rift_wy * 0.85 - ry) * np.sin(tilt)
        yc = -(x + rift_wx * 0.85 - rx) * np.sin(tilt) + (y + rift_wy * 0.85 - ry) * np.cos(tilt)
        rift_fil = np.exp(-((xc**2) / (0.24 + 0.10 * streak)) - ((yc**2) / 0.68)) * band_gate
        rift_wisp = _contrast_curve(_smooth_noise(rng, height, width, octaves=4), 0.32, 0.94, 1.28)
        rift_wisp_soft = _blur_separable_xy(rift_wisp, passes=1)
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
        dust_structure = np.clip(
            dust_structure + rift_combo * (0.52 + 0.21 * strength_gain),
            0.0,
            1.0,
        )
        swirl = _smooth_noise(rng, height, width, octaves=3)
        diag = np.sin(2.12 * x + 1.55 * y + float(rng.uniform(0.0, 6.283185307179586))) * 0.5 + 0.5
        dust_asym = 1.0 + float(rng.uniform(0.04, 0.09)) * (0.74 * (swirl - 0.5) + 0.26 * (diag - 0.5))
        dust_asym = np.clip(dust_asym, 0.96, 1.04)
        dust_structure = np.clip(dust_structure * dust_asym, 0.0, 1.0)
        flow_ph = float(rng.uniform(0.0, 6.283185307179586))
        flow_mod = 0.965 + 0.035 * (
            0.5 + 0.5 * np.sin(1.55 * x * float(sign) + 2.05 * y + flow_ph)
        )
        dust_structure = np.clip(dust_structure * flow_mod, 0.0, 1.0)
        lob_x0 = float(sign * rng.uniform(0.38, 0.66))
        lob_y0 = float(rng.uniform(-0.11, 0.11))
        lob_wx = rng.uniform(0.36, 0.68)
        lob_wy = rng.uniform(0.54, 0.98)
        lob_px = (_smooth_noise(rng, height, width, octaves=3) - 0.5) * 0.16 * band_gate
        lob_py = (_smooth_noise(rng, height, width, octaves=3) - 0.5) * 0.13 * band_gate
        side_core = np.exp(
            -(((x + lob_px - lob_x0) ** 2) / lob_wx) - (((y + lob_py - lob_y0) ** 2) / lob_wy)
        ) * band_gate
        lob_wisp = _contrast_curve(_smooth_noise(rng, height, width, octaves=4), 0.38, 0.96, 1.42)
        side_lobe = np.clip(_contrast_curve(side_core * (0.18 + 0.82 * lob_wisp), 0.14, 0.994, 1.78), 0.0, 1.0)
        dust_structure = np.clip(
            dust_structure + side_lobe * (0.16 + 0.10 * strength_gain),
            0.0,
            1.0,
        )
        # Extra mid-scale diffuse clouds (soft, band-aligned) so extinction + nebula read dustier.
        mid_wisp = _smooth_noise(rng, height, width, octaves=3) * np.clip(
            streak * 0.72 + broad_band * 0.28, 0.0, 1.0
        )
        mid_wisp = _contrast_curve(mid_wisp * (0.45 + 0.55 * filament_gate), 0.28, 0.95, 1.12)
        mid_wisp = _blur_separable_xy(mid_wisp, passes=2)
        dust_structure = np.clip(
            dust_structure + mid_wisp * (0.22 + 0.14 * coverage_gain + 0.08 * cloud_gain),
            0.0,
            1.0,
        )
        # Very coarse plate: large patches (~20–35% of band when combined with other fields).
        plate = _resize_bilinear(
            rng.random((max(2, height // 42), max(2, width // 9))), height, width
        )
        plate = _contrast_curve(plate, 0.18, 0.88, 0.82)
        plate *= np.clip(streak * 0.75 + broad_band * 0.25, 0.0, 1.0)
        plate = _blur_separable_xy(plate, passes=2)
        dust_structure = np.clip(
            dust_structure + plate * (0.20 + 0.11 * coverage_gain + 0.06 * cloud_gain),
            0.0,
            1.0,
        )
        dust_structure = _blur_separable_xy(dust_structure, passes=1)
        rift_feather = 0.93 + 0.12 * _smooth_noise(rng, height, width, octaves=2)
        dust_structure = np.clip(dust_structure * rift_feather, 0.0, 1.0)
        ds_merge = _blur_separable_xy(dust_structure, passes=3)
        dust_structure = np.clip(0.28 * dust_structure + 0.72 * ds_merge, 0.0, 1.0)
        band_soft = np.clip(streak * 0.92 + broad_band * 0.08, 0.0, 1.0)
        river = _blur_separable_xy(dust_structure * band_soft, passes=4)
        dust_structure = np.clip(0.55 * dust_structure + 0.45 * river, 0.0, 1.0)
        # Heavy y-only blur preserves x-sharp modulation → vertical "comb"; merge in x first, light y feather.
        dust_structure = _blur_x_only(dust_structure, passes=3)
        dust_structure = _blur_separable_xy(dust_structure, passes=3)
        dust_structure = _blur_y_only(dust_structure, passes=2)
        dust_structure = np.clip(dust_structure**0.94, 0.0, 1.0)
        # One or two large disk patches: gate fine dust without a solid black "hole".
        band_gate_m = np.clip(streak * 0.90 + broad_band * 0.10, 0.0, 1.0)
        # Anchor large dust patches in the disk wings (bulge longitude cx_band keeps warm gas).
        dust_lon0 = float(sign * rng.uniform(0.32, 0.58))
        cx1 = float(np.clip(cx_band + dust_lon0 + rng.normal(0.0, 0.055), -0.86, 0.86))
        cy1 = float(np.clip(rng.normal(float(sign * 0.035), 0.095), -0.20, 0.20))
        sx1 = float(rng.uniform(0.32, 0.62))
        sy1 = float(rng.uniform(0.11, 0.23))
        th1 = float(rng.uniform(-0.42, 0.42))
        c1, s1 = np.cos(th1), np.sin(th1)
        dx1, dy1 = x - cx1, y - cy1
        xr1 = dx1 * c1 + dy1 * s1
        yr1 = -dx1 * s1 + dy1 * c1
        mega_a = np.exp(-((xr1**2) / sx1 + (yr1**2) / sy1)) * band_gate_m
        if rng.random() < 0.58:
            dust_lon1 = float(-sign * rng.uniform(0.24, 0.52))
            cx2 = float(np.clip(cx_band + dust_lon1 + rng.normal(0.0, 0.075), -0.88, 0.88))
            if abs(cx2 - cx1) < 0.18:
                cx2 = float(np.clip(cx2 + float(sign) * 0.36, -0.88, 0.88))
            cy2 = float(np.clip(rng.normal(float(-sign * 0.032), 0.095), -0.20, 0.20))
            sx2 = float(rng.uniform(0.24, 0.52))
            sy2 = float(rng.uniform(0.11, 0.23))
            w2 = float(rng.uniform(0.38, 0.65))
            th2 = float(rng.uniform(-0.42, 0.42))
            c2, s2 = np.cos(th2), np.sin(th2)
            dx2, dy2 = x - cx2, y - cy2
            xr2 = dx2 * c2 + dy2 * s2
            yr2 = -dx2 * s2 + dy2 * c2
            mega_b = np.exp(-((xr2**2) / sx2 + (yr2**2) / sy2)) * band_gate_m * w2
            mega_raw = np.maximum(mega_a, mega_b)
        else:
            mega_raw = mega_a
        mega_env = _blur_separable_xy(mega_raw, passes=5)
        mega_env = _blur_x_only(mega_env, passes=3)
        mega_env = _blur_y_only(mega_env, passes=2)
        mega_env = np.clip(_contrast_curve(mega_env, 0.10, 0.90, 0.82), 0.0, 1.0) ** float(rng.uniform(0.91, 0.98))
        gate_w = np.clip(mega_env, 0.0, 1.0) ** (0.22 + 0.07 * cloud_gain)
        dust_structure = np.clip(
            dust_structure * (0.30 + 0.70 * gate_w)
            + mega_env * (0.17 + 0.05 * coverage_gain + 0.034 * strength_gain),
            0.0,
            1.0,
        )
        dust_structure = _blur_separable_xy(dust_structure, passes=4)
        dust_structure = _blur_x_only(dust_structure, passes=4)
        dust_structure = _blur_y_only(dust_structure, passes=2)
        dust_structure = np.clip(dust_structure**0.995, 0.0, 1.0)
        # Thin dust only in the tight bulge nucleus (keep a high floor so lanes stay visible).
        warm_carve = np.clip((core_band**1.42) * (streak**1.02), 0.0, 1.0)
        warm_carve = _blur_separable_xy(warm_carve, passes=2)
        dust_structure = np.clip(
            dust_structure * (0.56 + 0.44 * ((1.0 - 0.55 * warm_carve) ** 0.82)),
            0.0,
            1.0,
        )
        # Add a little extra dust in the band shoulders only (do not scale down the midplane — that was hiding dust).
        shoulder = np.clip(4.2 * streak * (1.0 - streak), 0.0, 1.0) ** 0.52
        shoulder = _blur_separable_xy(shoulder, passes=1)
        dust_structure = np.clip(dust_structure + shoulder * (0.10 + 0.06 * coverage_gain), 0.0, 1.0)
        dust_structure = _blur_x_only(dust_structure, passes=5)
        dust_structure = _blur_separable_xy(dust_structure, passes=2)
        dust_structure = np.clip(dust_structure**0.985, 0.0, 1.0)
        # Galactic band palette tuned toward dusty gas instead of ember-like fire tones.
        cloud_mix = (_smooth_noise(rng, height, width, octaves=1) * 0.78 + n_smooth * 0.22).clip(0.0, 1.0)
        zone_map = _smooth_noise(rng, height, width, octaves=2)
        zone_map = (zone_map * 0.85 + cloud_mix * 0.15).clip(0.0, 1.0)
        black = np.array([0.022, 0.020, 0.028], dtype=np.float64)
        red = np.array([0.52, 0.13, 0.10], dtype=np.float64)
        gold = np.array([1.0, 0.90, 0.22], dtype=np.float64)
        rust = np.array([0.52, 0.24, 0.13], dtype=np.float64)
        white = np.array([1.0, 0.94, 0.58], dtype=np.float64)

        t1 = np.clip(zone_map / 0.55, 0.0, 1.0)[..., None]
        t2 = np.clip((zone_map - 0.55) / 0.28, 0.0, 1.0)[..., None]
        t3 = np.clip((zone_map - 0.83) / 0.17, 0.0, 1.0)[..., None]
        color = black * (1.0 - t1) + red * t1
        color = color * (1.0 - t2) + gold * t2
        color = color * (1.0 - t3) + white * t3
        rust_mix = np.clip(_smooth_noise(rng, height, width, octaves=2), 0.0, 1.0)[..., None]
        color = color * (1.0 - 0.15 * rust_mix) + rust * (0.15 * rust_mix)
        # Per-pixel hue drift in dust tones (brown / tan / brick variation).
        hue_j = (_smooth_noise(rng, height, width, octaves=2) - 0.5)[..., None]
        color = color + hue_j * np.array([0.055, 0.035, -0.018], dtype=np.float64) * band_envelope[..., None]
        # Mild desaturation — keep enough chroma for gold / magenta / H II to read in final grade.
        lum = np.mean(color, axis=2, keepdims=True)
        color = color * 0.87 + lum * 0.13
        # Central bulge: lift luma and mix toward cream-yellow; slight desaturation in core.
        plane_for_bulge = streak[..., None]
        bulge_w = (core_band[..., None] ** 0.82) * (0.10 + 0.90 * (plane_for_bulge**0.80))
        cream = np.array([0.99, 0.92, 0.58], dtype=np.float64)
        color = color * (1.0 + 0.32 * bulge_w) + cream * bulge_w * 0.30
        lum_bulge = np.mean(color, axis=2, keepdims=True)
        color = color * (1.0 - 0.17 * bulge_w) + lum_bulge * (0.17 * bulge_w)
        # Narrow inner hotspots: unresolved knots inside the bright band (continuum only).
        inner_hot = np.clip(
            _blur_separable_xy((core_band * np.clip(streak, 0.0, 1.0)) ** 1.32, passes=2),
            0.0,
            1.0,
        )
        hot_rgb = np.array([0.18, 0.135, 0.072], dtype=np.float64)
        color = np.clip(color + inner_hot[..., None] * hot_rgb * (0.145 + 0.062 * cloud_gain), 0.0, 1.0)
        # Softer brown transition into bright gold at dust silhouettes (volume in front of backlight).
        ds_chroma = np.clip(_blur_separable_xy(dust_structure, passes=3) ** 0.48, 0.0, 1.0)[..., None]
        gold_ink = np.array([0.20, 0.14, 0.055], dtype=np.float64)
        color = np.clip(color * (1.0 - 0.032 * ds_chroma) + gold_ink * (0.16 * ds_chroma) * band_envelope[..., None], 0.0, 1.0)

        emit_rgb = np.zeros((height, width, 3), dtype=np.float64)
        # Sparse H-alpha style knots (magenta / pink) along the bright band.
        ha_field = _smooth_noise(rng, height, width, octaves=4)
        ha_spots = _resize_bilinear(
            rng.random((max(2, height // 28), max(2, width // 28))), height, width
        )
        ha_mask = ha_field * 0.55 + ha_spots * 0.45
        ha_mask = _contrast_curve(ha_mask, 0.62, 0.998, 2.38)
        ha_mask *= band_envelope * (0.30 + 0.70 * np.clip(clumps, 0.0, 1.0))
        ha_cloud = np.clip(_smooth_noise(rng, height, width, octaves=3), 0.0, 1.0)
        ha_lon = _blur_separable_xy(longitudinal, passes=2) * 0.55 + long_warp2d * 0.45
        ha_mask *= (0.48 + 0.52 * ha_lon) * (0.82 + 0.38 * ha_cloud)
        ha_core = np.clip(_blur_separable_xy((core_band * streak) ** 0.58, passes=1), 0.0, 1.0)
        ha_mask *= 0.50 + 0.50 * ha_core
        ha_grad = np.clip(np.abs(ha_mask - _blur_separable_xy(ha_mask, passes=3)), 0.0, 1.0) ** 0.58
        ha_rgb = np.array([1.0, 0.05, 0.52], dtype=np.float64)
        ha_knot = ha_mask[..., None] * ha_rgb * (1.28 + 0.38 * coverage_gain)
        color = color + ha_knot
        emit_rgb += ha_knot
        ha_wide = _blur_separable_xy(ha_mask * band_envelope, passes=3)
        ha_wide = np.clip(ha_wide**1.05, 0.0, 1.0)
        ha_wide_rgb = ha_wide[..., None] * np.array([0.85, 0.06, 0.48], dtype=np.float64) * (0.22 + 0.09 * cloud_gain)
        color = color + ha_wide_rgb
        emit_rgb += ha_wide_rgb
        # Very faint large-scale pink veil only (sheet was dominating vs reference gold ratio).
        ha_sheet = _blur_separable_xy(ha_mask * band_envelope * np.clip(streak, 0.0, 1.0), passes=5)
        ha_sheet = np.clip(ha_sheet**1.08, 0.0, 1.0)
        ha_sheet_rgb = ha_sheet[..., None] * np.array([0.42, 0.06, 0.32], dtype=np.float64) * (0.055 + 0.02 * cloud_gain)
        color = np.clip(color + ha_sheet_rgb, 0.0, 1.0)
        emit_rgb += ha_sheet_rgb
        # Rare compact H II cores — strong red reference spots (sparse, high contrast).
        ha_hot = (
            np.clip(ha_mask, 0.0, 1.0) ** 2.25
            * band_envelope
            * np.clip(streak, 0.0, 1.0)
            * (0.32 + 0.68 * ha_core)
            * (0.45 + 0.55 * np.clip(clumps, 0.0, 1.0))
        )
        ha_hot_rgb = ha_hot[..., None] * np.array([0.95, 0.03, 0.18], dtype=np.float64) * (0.36 + 0.12 * coverage_gain)
        color = np.clip(color + ha_hot_rgb, 0.0, 1.0)
        emit_rgb += ha_hot_rgb
        # S II–weighted deep red wing (reads beside Hα without neon oversaturation).
        sii_w = _blur_separable_xy(ha_mask * band_envelope * np.clip(streak, 0.0, 1.0), passes=3)
        sii_w = np.clip(sii_w * (1.0 + 0.32 * ha_grad), 0.0, 1.0)
        sii_rgb = sii_w[..., None] * np.array([0.70, 0.08, 0.06], dtype=np.float64) * (0.125 + 0.04 * coverage_gain)
        color = np.clip(color + sii_rgb, 0.0, 1.0)
        emit_rgb += sii_rgb
        # O-III / blue-green line glow on brighter H II edges (photo: not only pink).
        oiii_mask = np.clip(
            ha_mask * np.clip((zone_map - 0.38) / 0.52, 0.0, 1.0) * (0.35 + 0.65 * np.clip(clumps, 0.0, 1.0)),
            0.0,
            1.0,
        )
        oiii_mask *= _smooth_noise(rng, height, width, octaves=1)
        oiii_mask = _contrast_curve(oiii_mask, 0.16, 0.96, 1.22)
        oiii_mask *= 0.18 + 0.82 * ha_grad
        oiii_rgb = np.array([0.08, 0.50, 0.60], dtype=np.float64)
        oiii_add = oiii_mask[..., None] * oiii_rgb * (0.44 + 0.13 * cloud_gain)
        color = color + oiii_add
        emit_rgb += oiii_add
        # Shadow-side teal (scattered light / O-association shadows) in darker dust.
        teal_field = _smooth_noise(rng, height, width, octaves=3)
        teal_mask = _contrast_curve(teal_field, 0.71, 0.997, 2.25)
        teal_mask *= band_envelope * np.clip(1.0 - zone_map, 0.0, 1.0) * (0.28 + 0.72 * np.clip(clumps, 0.0, 1.0))
        teal_mask *= 0.42 + 0.58 * ha_grad
        teal_mask *= 0.62 + 0.38 * np.clip(dust_structure, 0.0, 1.0) ** 0.38
        teal_rgb = np.array([0.08, 0.20, 0.24], dtype=np.float64)
        color = color + teal_mask[..., None] * teal_rgb * (0.58 + 0.16 * cloud_gain)
        # Rare violet–blue reflection wisps.
        vio_field = _resize_bilinear(
            rng.random((max(2, height // 26), max(2, width // 26))), height, width
        )
        vio_mask = _contrast_curve(vio_field * 0.55 + _smooth_noise(rng, height, width, octaves=2) * 0.45, 0.78, 0.998, 2.5)
        vio_cloud = np.clip(_smooth_noise(rng, height, width, octaves=2), 0.0, 1.0)
        vio_mask *= band_envelope * ha_lon * (0.22 + 0.78 * activity_map) * (0.74 + 0.52 * vio_cloud)
        vio_mask *= 0.38 + 0.62 * ha_grad
        vio_rgb = np.array([0.38, 0.26, 0.52], dtype=np.float64)
        vio_add = vio_mask[..., None] * vio_rgb * (0.18 + 0.06 * coverage_gain)
        color = color + vio_add
        emit_rgb += vio_add
        # Tight deep-red molecular filaments (separate from H-alpha pink).
        fil_field = _smooth_noise(rng, height, width, octaves=4)
        fil_mask = _contrast_curve(fil_field, 0.76, 0.999, 3.1)
        fil_mask *= band_envelope * dark_lanes * (0.35 + 0.65 * filament_gate)
        fil_rgb = np.array([0.50, 0.10, 0.075], dtype=np.float64)
        fil_add = fil_mask[..., None] * fil_rgb * (0.26 + 0.09 * strength_gain)
        color = color + fil_add
        emit_rgb += fil_add
        # Cooler diffuse gas in the disk; warm cream stays concentrated in core × band.
        away_core = np.clip(1.0 - np.minimum(1.0, core_band * streak * 1.55), 0.0, 1.0)[..., None]
        cool_disk = np.array([0.92, 0.97, 1.07], dtype=np.float64)
        cool_mul = 1.0 + (cool_disk - 1.0) * away_core * 0.22
        color = color * cool_mul
        emit_rgb *= cool_mul
        # Large-scale brown-screen mottling (blur longitude so RGB drift is not vertical stripes).
        long_soft = _blur_separable_xy(longitudinal, passes=2)
        lon_tint = (long_soft - 0.5) * 0.09
        r_ch = np.clip(1.0 + lon_tint, 0.88, 1.12)
        g_ch = np.clip(1.0 - 0.035 * lon_tint, 0.88, 1.12)
        b_ch = np.clip(1.0 - 0.07 * np.abs(lon_tint), 0.88, 1.12)
        lon_rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
        lon_brk = np.clip(_smooth_noise(rng, height, width, octaves=2), 0.0, 1.0)
        lon_mul = lon_rgb * (0.90 + 0.14 * np.clip(clumps, 0.0, 1.0))[..., None] * (0.88 + 0.24 * lon_brk)[..., None]
        color = color * lon_mul
        emit_rgb *= lon_mul

        # Mid-scale gold + magenta glow (foreground gas — moderate saturation).
        low_h, low_w = max(2, height // 20), max(2, width // 18)
        gold_blob = _blur_separable_xy(_resize_bilinear(rng.random((low_h, low_w)), height, width), passes=3)
        mag_blob = _blur_separable_xy(
            _resize_bilinear(rng.random((max(2, height // 22), max(2, width // 16))), height, width),
            passes=3,
        )
        gold_blob = _contrast_curve(gold_blob, 0.22, 0.90, 0.82) * band_envelope * (0.32 + 0.68 * np.clip(streak, 0.0, 1.0))
        mag_lon = _blur_separable_xy(longitudinal, passes=2) * 0.5 + long_scr * 0.5
        mag_blob = (
            _contrast_curve(mag_blob, 0.28, 0.94, 1.06)
            * band_envelope
            * mag_lon
            * (0.26 + 0.74 * np.clip(clumps, 0.0, 1.0))
        )
        gold_mid = gold_blob[..., None] * np.array([0.96, 0.74, 0.11], dtype=np.float64) * 0.228
        mag_mid = mag_blob[..., None] * np.array([0.52, 0.05, 0.34], dtype=np.float64) * 0.072
        color = np.clip(color + gold_mid + mag_mid, 0.0, 1.0)
        emit_rgb += mag_mid
        # Split veils so warm dust (gold) and H II (pink–magenta) stay separate spectrally.
        veil_gold = _blur_separable_xy(gold_blob * 1.35 * np.clip(streak, 0.0, 1.0), passes=4)
        veil_hii = _blur_separable_xy(
            mag_blob * (0.85 + 0.15 * np.clip(activity_map, 0.0, 1.0)) * np.clip(streak, 0.0, 1.0),
            passes=4,
        )
        vg_add = veil_gold[..., None] * np.array([0.78, 0.56, 0.11], dtype=np.float64) * 0.104
        vh_add = veil_hii[..., None] * np.array([0.48, 0.09, 0.36], dtype=np.float64) * 0.043
        color = np.clip(color + vg_add + vh_add, 0.0, 1.0)
        emit_rgb += vh_add

        # Merge H II into patches; keep more sharp weight so magenta/red is not averaged to brown.
        emit_prev = emit_rgb
        emit_smooth = _blur_rgb_separable_xy(emit_rgb, passes=3)
        emit_rgb = np.clip(emit_rgb * 0.66 + emit_smooth * 0.34, 0.0, 1.0)
        el = np.mean(emit_rgb, axis=2, keepdims=True)
        emit_rgb = np.clip(el + (emit_rgb - el) * 1.12, 0.0, 1.0)
        color = np.clip(color + emit_rgb - emit_prev, 0.0, 1.0)

        dust_rgb = np.clip(color - emit_rgb, 0.0, 1.0)
        split_nebula_rgb = (dust_rgb, emit_rgb)

        mask *= 0.94 + 0.22 * core_band

    density_field = _smooth_noise(rng, height, width, octaves=2)
    density_field = (density_field - 0.5) * (0.09 if split_nebula_rgb is not None else 0.15)
    mask *= 1.0 + density_field

    temp_shift = _smooth_noise(rng, height, width, octaves=2)[..., None]
    warm = np.array([1.05, 0.85, 0.75], dtype=np.float64)
    cool = np.array([0.85, 0.90, 1.05], dtype=np.float64)
    temp = warm * temp_shift + cool * (1.0 - temp_shift)
    temp_sat = 0.992 if mode == NebulaMode.galaxy_streak else 0.9
    mul_te = temp * temp_sat
    if split_nebula_rgb is not None:
        d_sp, e_sp = split_nebula_rgb
        d_sp = d_sp * mul_te
        emit_ch = np.array([1.06, 0.99, 1.08], dtype=np.float64)
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
    micro = _smooth_noise(rng, height, width, octaves=2)
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

    dust = _smooth_noise(rng, height, width, octaves=2)
    dust = np.clip((dust - 0.48) * 2.2, 0.0, 1.0)
    dust = dust**1.7
    dd_sz = max(2, min(height // 22, width // 22))
    dense_dust = _resize_bilinear(rng.random((dd_sz, dd_sz)), height, width)
    dense_dust = _contrast_curve(dense_dust, 0.74, 0.995, 1.65)
    bp_sz = max(2, min(height // 20, width // 20))
    black_pockets = _resize_bilinear(rng.random((bp_sz, bp_sz)), height, width)
    black_pockets = _contrast_curve(black_pockets, 0.58, 0.94, 1.35)
    mask -= dust * (0.18 + 0.10 * coverage_gain)
    if mode == NebulaMode.galaxy_streak:
        mask -= black_pockets * (0.10 + 0.08 * coverage_gain)
        mask -= dense_dust * (0.18 + 0.11 * coverage_gain)
    else:
        mask -= black_pockets * (0.16 + 0.12 * coverage_gain)
        mask -= dense_dust * (0.22 + 0.14 * coverage_gain)
    if mode == NebulaMode.galaxy_streak:
        pocket_dim = 0.085 + 0.042
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

    mask_tex = _smooth_noise(rng, height, width, octaves=2)
    mask_tex = (mask_tex - 0.5) * 0.024
    mask *= 1.0 + mask_tex * 0.055

    base_glow = _smooth_noise(rng, height, width, octaves=2)
    base_glow = (base_glow - 0.5) * 0.065
    if mode == NebulaMode.galaxy_streak:
        # Slightly lifted floor so diffuse disk / unresolved-star haze reads above pure black.
        background = np.array([0.024, 0.022, 0.030], dtype=np.float64)
    else:
        background = np.array([0.014, 0.017, 0.024], dtype=np.float64)
    if split_nebula_rgb is not None:
        d_sp, e_sp = split_nebula_rgb
        d_sp = d_sp + base_glow[..., None] + background
        color = np.clip(d_sp + e_sp, 0.0, 1.0)
        split_nebula_rgb = (d_sp, e_sp)
    else:
        color += base_glow[..., None]
        color += background
    if mode == NebulaMode.galaxy_streak:
        dust_structure = _blur_separable_xy(dust_structure, passes=2)
    dust_str_w = (0.56 + 0.26 * coverage_gain) if mode == NebulaMode.galaxy_streak else (0.28 + 0.14 * coverage_gain)
    bp_for_ext = black_pockets
    dd_for_ext = dense_dust
    if mode == NebulaMode.galaxy_streak:
        bp_for_ext = _blur_x_only(
            _blur_y_only(_blur_separable_xy(black_pockets, passes=1), passes=2), passes=3
        )
        dd_for_ext = _blur_x_only(
            _blur_y_only(_blur_separable_xy(dense_dust, passes=2), passes=3), passes=3
        )
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
        band_bias = np.exp(-((y**2) / 1.15))
        cx_off = float(sign * rng.uniform(0.0, 0.16))
        cy_off = float(rng.uniform(-0.07, 0.07))
        skew = float(sign * rng.uniform(0.05, 0.18))
        xw = x - cx_off - skew * y
        yw = y - cy_off
        radial_core = np.exp(-((xw**2) / 0.52) - ((yw**2) / 1.35))
        lump = _contrast_curve(_smooth_noise(rng, height, width, octaves=4), 0.20, 0.92, 0.86)
        center_bias = np.clip(0.38 + 0.62 * radial_core * (0.48 + 0.52 * lump), 0.26, 1.0)
        xskew = x - float(sign * rng.uniform(0.06, 0.14)) * y
        dense_center = np.exp(-((xskew**2) / 0.36)) * (0.55 + 0.45 * lump)
        dense_dust *= 0.35 + 0.65 * dense_center
        dust_occlusion = np.clip(
            dust_occlusion * (0.24 + 0.76 * band_bias) * (0.70 + 0.30 * center_bias)
            + dense_dust * (0.15 + 0.14 * strength_gain),
            0.0,
            1.0,
        )
        dust_occlusion = np.clip(dust_occlusion * 1.02, 0.0, 1.0)
        dust_occlusion = np.clip(dust_occlusion**0.99, 0.0, 1.0)
        dust_occlusion = _blur_separable_xy(dust_occlusion, passes=2)
        dust_occlusion = _blur_y_only(dust_occlusion, passes=2)
        dust_occlusion = _blur_x_only(dust_occlusion, passes=3)
        dust_occlusion = _blur_separable_xy(dust_occlusion, passes=1)
        dust_occlusion = np.clip(dust_occlusion, 0.0, 1.0)
        # Slightly compress dynamic range so extinction rarely hits the hard floor (less "punched hole").
        dust_occlusion = np.clip(dust_occlusion * 0.962 + 0.020, 0.0, 1.0)
        dust_occlusion = _blur_separable_xy(dust_occlusion, passes=2)
        lane_extinction = np.clip(_contrast_curve(dust_structure, 0.02, 0.995, 1.08), 0.0, 1.0)
        mega_align = np.clip(_blur_separable_xy(mega_env, passes=3), 0.0, 1.0)
        lane_extinction = np.clip(lane_extinction * (0.86 + 0.14 * mega_align), 0.0, 1.0)
        lane_extinction = _blur_separable_xy(lane_extinction, passes=2)
        lane_extinction = _blur_x_only(lane_extinction, passes=3)
        lane_extinction = _blur_y_only(lane_extinction, passes=2)
        lane_extinction = _blur_separable_xy(lane_extinction, passes=1)
    if split_nebula_rgb is not None:
        dust_rgb, emit_rgb = split_nebula_rgb
        m_shell = np.clip((mask[..., None] * 0.86 + 0.078) * 0.96, 0.0, 1.0)
        # Vertical taper: stronger on continuum dust; ionized gas can sit slightly "above" the plane.
        band_taper_d = (0.72 + 0.28 * np.clip(streak, 0.0, 1.0) ** 1.05)[..., None]
        band_taper_e = (0.84 + 0.16 * np.clip(streak, 0.0, 1.0) ** 1.02)[..., None]
        m_dust = m_shell * band_taper_d
        m_emit = m_shell * band_taper_e
        # Pass 1: warm dust / continuum — favor blurred gas for cloud-like continuity.
        dust_gas = _blur_rgb_separable_xy(dust_rgb, passes=4)
        neb = (dust_rgb * 0.30 + dust_gas * 0.70) * m_dust
        # Pass 2: H II — moderate merge so knots read as regions, not single pixels.
        emit_gas = _blur_rgb_separable_xy(emit_rgb, passes=2)
        neb_emit = np.clip((emit_rgb * 0.64 + emit_gas * 0.36) * m_emit * 1.06, 0.0, 1.0)

        dust_for_neb = np.clip(_blur_separable_xy(dust_structure, passes=2) ** 0.36, 0.0, 1.0)
        neb *= (0.80 + 0.20 * (1.0 - 0.78 * dust_for_neb))[..., None]
        neb_emit *= (0.90 + 0.10 * (1.0 - 0.55 * dust_for_neb))[..., None]

        plane_glow = np.clip(streak**1.08, 0.0, 1.0)
        neb *= (0.93 + 0.16 * plane_glow)[..., None]
        neb_emit *= (0.96 + 0.14 * plane_glow)[..., None]

        y_band = np.exp(-((y**2) / 0.92))
        lum_neb = np.mean(neb, axis=2)
        gsz_y = max(10, height // 10)
        gsz_x = max(20, width // 16)
        sm = _resize_bilinear(lum_neb * y_band, gsz_y, gsz_x)
        horiz = _resize_bilinear(sm, height, width)
        horiz = _blur_separable_xy(horiz, passes=2)
        horiz = np.clip(horiz**1.06, 0.0, 1.0)
        warm_h = np.array([0.98, 0.78, 0.14], dtype=np.float64)
        neb = neb + horiz[..., None] * warm_h * (0.145 + 0.055 * cloud_gain)

        floor_warp = np.clip(_smooth_noise(rng, height, width, octaves=2), 0.0, 1.0)
        diffuse_floor = y_band * (0.016 + 0.013 * _smooth_noise(rng, height, width, octaves=1))
        diffuse_floor *= 0.80 + 0.40 * floor_warp
        floor_rgb = np.array([0.12, 0.098, 0.048], dtype=np.float64)
        neb = np.clip(neb + diffuse_floor[..., None] * floor_rgb, 0.0, 1.0)

        lane_w = _contrast_curve(dust_structure, 0.05, 0.995, 1.14)
        lk = (0.76 + 0.44 * strength_gain) * (0.12 + 0.88 * core_band)
        neb *= np.clip(1.0 - lk[..., None] * lane_w[..., None] * 0.92, 0.24, 1.0)
        neb_emit *= np.clip(1.0 - lk[..., None] * lane_w[..., None] * 0.40, 0.58, 1.0)

        ds_blur = _blur_separable_xy(dust_structure, passes=2)
        dust_edge = np.clip(dust_structure - 0.86 * ds_blur, 0.0, 1.0) ** 1.05
        dust_edge = _blur_separable_xy(dust_edge, passes=1)
        rim = dust_edge * np.clip(streak * 0.92 + broad_band * 0.08, 0.0, 1.0)
        rim_rgb = np.array([0.62, 0.46, 0.18], dtype=np.float64)
        rim_mag = np.array([0.42, 0.08, 0.28], dtype=np.float64)
        neb = np.clip(neb + rim[..., None] * rim_rgb * (0.16 + 0.055 * cloud_gain), 0.0, 1.0)
        neb_emit = np.clip(neb_emit + rim[..., None] * rim_mag * (0.055 + 0.024 * cloud_gain), 0.0, 1.0)

        ripple_a = _smooth_noise(rng, height, width, octaves=2)
        ripple_b = _blur_separable_xy(_smooth_noise(rng, height, width, octaves=4), passes=2)
        ripple = (ripple_a * 0.68 + ripple_b * 0.32 - 0.5) * 2.0
        ripple = np.clip(ripple, -0.075, 0.075) * y_band
        neb = np.clip(
            neb + ripple[..., None] * np.array([0.07, 0.07, 0.085], dtype=np.float64),
            0.0,
            1.0,
        )
        turb = _blur_separable_xy(_smooth_noise(rng, height, width, octaves=3), passes=2)
        neb = np.clip(
            neb + (turb - 0.5)[..., None] * np.array([0.022, 0.020, 0.025], dtype=np.float64) * y_band[..., None],
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

        neb = _blur_rgb_separable_xy(neb, passes=5)
        neb_emit = _blur_rgb_separable_xy(neb_emit, passes=2)
        lm_d = np.mean(neb, axis=2, keepdims=True)
        neb = np.clip(lm_d + (neb - lm_d) * 1.20, 0.0, 1.0)
        lm_e = np.mean(neb_emit, axis=2, keepdims=True)
        neb_emit = np.clip(lm_e + (neb_emit - lm_e) * 1.06, 0.0, 1.0)
        neb_emit = np.clip(neb_emit * 1.05, 0.0, 1.0)
        return (
            np.clip(neb, 0.0, 1.0),
            np.clip(neb_emit, 0.0, 1.0),
            dust_occlusion,
            lane_extinction,
        )
    emit_empty = np.zeros((height, width, 3), dtype=np.float64)
    return color * (mask[..., None] * 0.38 + 0.05), emit_empty, dust_occlusion, lane_extinction
