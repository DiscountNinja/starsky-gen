"""Band-hosted structure: vertical extent, brutal erasure, longitude asymmetry, seam guard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starsky_gen.procedural_noise import _resize_bilinear, fbm2d, gaussian_blur_pil, ridged_fbm2d


def derive_nebula_rng(master_seed: int, *labels: str | int) -> np.random.Generator:
    """Deterministic child RNG for nebula/H II — stable across runs for the same master seed."""
    entropy: list[int] = [int(master_seed) & 0xFFFFFFFF]
    for lb in labels:
        if isinstance(lb, str):
            entropy.append(sum((i + 1) * ord(c) for i, c in enumerate(lb)) & 0xFFFFFFFF)
        else:
            entropy.append(int(lb) & 0xFFFFFFFF)
    child = np.random.SeedSequence(entropy).spawn(1)[0]
    return np.random.default_rng(child)


@dataclass
class MorphologyIsmLayers:
    """Split ISM for layered compositing: absorb vs warm emit vs H II."""

    absorption_rgb: np.ndarray
    gold_emit_rgb: np.ndarray
    red_hii_rgb: np.ndarray
    white_rgb: np.ndarray
    puff_punch_mask: np.ndarray
    combined_rgb: np.ndarray
    luma: np.ndarray


@dataclass
class StructureArchitectureWeights:
    """Explicit weight maps that plan where turbulent cloud architecture is driven.

    Used by galactic morphology, extinction coupling, and (later) full-frame nebula so
    puff / vertical extent / reinject share one host instead of rim-only heuristics.

    Channels:
      band_host — permissive ISM host (wider than razor disk_weight).
      turbulence_drive — master mix weight [floor..1] through the band core and plumes.
      puff_weight / vertical_weight / reinject_weight — per-mechanism drive (× turbulence_drive).
      dust_lane_weight — tight mid-plane dust sub-band (Milky-Way dark lane inside the band).
      puff_shell_weight — fluffy shell above/below the dust lane (more puff, not the lane core).
      negative_space_weight — clear pockets in the shell (sightline gaps between clouds).
      survival_weight / turb_latent_weight — auxiliary coupling for nebula/ISM passes.
    """

    band_host: np.ndarray
    turbulence_drive: np.ndarray
    puff_weight: np.ndarray
    vertical_weight: np.ndarray
    reinject_weight: np.ndarray
    dust_lane_weight: np.ndarray
    puff_shell_weight: np.ndarray
    negative_space_weight: np.ndarray
    survival_weight: np.ndarray
    turb_latent_weight: np.ndarray


def soften_band_envelope(
    disk_weight: np.ndarray | None,
    shape: tuple[int, int],
    *,
    periodic_x: bool = True,
    lat_blur_sigma: float = 14.0,
    power: float = 0.58,
) -> np.ndarray:
    """Feather disk host latitudinally — avoids a razor-sharp horizontal 'solid line' in ISM/gas."""
    from starsky_gen.dust_field import _blur_y_only_field

    h, w = int(shape[0]), int(shape[1])
    if disk_weight is None:
        return np.ones((h, w), dtype=np.float64)
    bw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    if bw.shape != (h, w):
        bw = _resize_bilinear(bw, h, w, periodic_x=periodic_x)
    sig_cap = max(4.0, min(28.0, h * 0.22))
    soft = _blur_y_only_field(bw, float(np.clip(lat_blur_sigma, 4.0, sig_cap)))
    soft = np.clip(soft, 0.0, 1.0) ** float(np.clip(power, 0.42, 0.85))
    # Higher floor — razor soft_env zeros read as opaque black bites on black backgrounds.
    return np.clip(0.28 + 0.72 * soft, 0.0, 1.0).astype(np.float64)


def latitude_plane_gate(
    height: int,
    *,
    sigma: float = 0.40,
    power: float = 1.0,
) -> np.ndarray:
    """Suppress plane-locked effects toward equirect poles (stops vertical bleed to frame edges)."""
    sig = float(np.clip(sigma, 0.12, 1.2))
    pw = float(np.clip(power, 0.5, 2.5))
    yy = np.linspace(-1.0, 1.0, int(height), dtype=np.float64)[:, None]
    return np.clip(np.exp(-((yy**2) / sig)) ** pw, 0.0, 1.0).astype(np.float64)


def midplane_gold_lat_gate(
    height: int,
    *,
    sigma: float = 0.095,
    power: float = 1.85,
) -> np.ndarray:
    """Narrow latitude razor for gold — disk_weight alone is too thick on shoulders."""
    sig = float(np.clip(sigma, 0.05, 0.22))
    pw = float(np.clip(power, 1.0, 3.0))
    yy = np.linspace(-1.0, 1.0, int(height), dtype=np.float64)[:, None]
    return np.clip(np.exp(-((yy / sig) ** 2)) ** pw, 0.0, 1.0).astype(np.float64)


def wisp_vertical_bleed_gate(
    height: int,
    width: int,
    *,
    plane_gate: np.ndarray,
    bleed_gate: np.ndarray,
    vertical_extent: np.ndarray | None = None,
    structure_survival: np.ndarray | None = None,
    periodic_x: bool = True,
    halo_power: float = 0.72,
) -> tuple[np.ndarray, np.ndarray]:
    """Wide latitudinal visibility for ISM/gas plus an annular halo outside the tight band gate.

    ``wisp`` is the permissive gate (puffy clouds may extend above/below the disk plane).
    ``halo`` is strongest just outside the tight ``plane_gate`` — wisps bleeding off the band.
    """
    h, w = int(height), int(width)
    pg = np.clip(np.asarray(plane_gate, dtype=np.float64), 0.0, 1.0)
    bg = np.clip(np.asarray(bleed_gate, dtype=np.float64), 0.0, 1.0)
    if pg.ndim == 1 or pg.shape[1] == 1:
        pg = np.broadcast_to(pg.reshape(h, 1) if pg.ndim == 1 else pg, (h, w))
    if bg.ndim == 1 or bg.shape[1] == 1:
        bg = np.broadcast_to(bg.reshape(h, 1) if bg.ndim == 1 else bg, (h, w))
    wisp = np.clip(np.maximum(pg, bg), 0.0, 1.0)
    hp = float(np.clip(halo_power, 0.55, 0.95))
    halo = np.clip(bg - pg * 0.88, 0.0, 1.0) ** hp
    if vertical_extent is not None:
        ve = np.clip(np.asarray(vertical_extent, dtype=np.float64), 0.0, 1.0)
        if ve.shape != (h, w):
            ve = _resize_bilinear(ve, h, w, periodic_x=periodic_x)
        wisp = np.clip(np.maximum(wisp, ve * 0.28), 0.0, 1.0)
        halo = np.clip(np.maximum(halo, ve * 0.22 * (1.0 - pg * 0.78)), 0.0, 1.0)
    if structure_survival is not None:
        surv = np.clip(np.asarray(structure_survival, dtype=np.float64) / 1.35, 0.0, 1.0)
        if surv.shape != (h, w):
            surv = _resize_bilinear(surv, h, w, periodic_x=periodic_x)
        wisp = np.clip(np.maximum(wisp, surv * 0.34), 0.0, 1.0)
        halo = np.clip(np.maximum(halo, surv * 0.26 * (1.0 - pg * 0.82)), 0.0, 1.0)
    pole = latitude_plane_gate(h, sigma=0.50, power=1.02)
    if pole.ndim == 1:
        pole = pole.reshape(h, 1)
    pole = np.broadcast_to(pole, (h, w))
    halo = np.clip(halo * pole, 0.0, 1.0)
    return wisp.astype(np.float64), halo.astype(np.float64)


def sample_band_lat_curve(
    rng: np.random.Generator,
    width: int,
    *,
    band_curvature_amp: float = 0.04,
) -> tuple[np.ndarray, str]:
    """Per-longitude lat center warp: S, U, W, or mild flat (subtle midplane bend)."""
    w = int(width)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float64, endpoint=False)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    amp = float(np.clip(band_curvature_amp * 0.72 + rng.uniform(0.0, 0.018), 0.010, 0.065))
    roll = float(rng.uniform())
    freq = 2.0 * np.pi
    if roll < 0.44:
        curve = amp * 1.05 * np.sin(freq * xx + phase)
        kind = "s"
    elif roll < 0.72:
        curve = amp * 1.18 * (np.sin(freq * xx + phase) ** 2 - 0.5)
        kind = "u"
    elif roll < 0.90:
        curve = amp * (
            0.68 * np.sin(freq * xx + phase) + 0.38 * np.sin(2.0 * freq * xx + phase * 1.17)
        )
        kind = "w"
    else:
        curve = amp * 0.48 * np.sin(freq * xx + phase)
        kind = "flat"
    return curve.reshape(1, -1).astype(np.float64), kind


def clamp_disk_band_thickness(
    disk_weight: np.ndarray,
    *,
    min_half_width_frac: float = 0.072,
    max_half_width_frac: float = 0.155,
    level_frac: float = 0.30,
    periodic_x: bool = True,
) -> np.ndarray:
    """Keep band latitudinal half-width between min/max fractions of frame height."""
    _ = periodic_x
    from starsky_gen.dust_field import _blur_y_only_field

    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = int(dw.shape[0]), int(dw.shape[1])
    min_hw = float(np.clip(min_half_width_frac, 0.032, 0.10)) * h
    max_hw = float(np.clip(max_half_width_frac, min_hw / h + 0.025, 0.20)) * h
    lvl = float(np.clip(level_frac, 0.22, 0.45))
    out = dw.copy()
    yy = np.arange(h, dtype=np.float64)[:, None]
    for j in range(w):
        p = dw[:, j]
        mx = float(np.max(p))
        if mx < 0.07:
            continue
        mask = p >= mx * lvl
        if not bool(np.any(mask)):
            continue
        rows = np.flatnonzero(mask)
        cy = float(np.mean(rows))
        hw = max(float(rows[-1] - rows[0]) * 0.5, 1.0)
        prof = p.copy()
        if hw < min_hw:
            sig = float(np.clip((min_hw - hw) * 0.28, 2.0, 18.0))
            expanded = _blur_y_only_field(prof.reshape(-1, 1), sig).ravel()
            ex_mx = float(np.max(expanded))
            if ex_mx > 1e-8:
                expanded = expanded * (mx / ex_mx)
            prof = np.clip(np.maximum(prof, expanded), 0.0, mx)
        elif hw > max_hw:
            yy_c = yy - cy
            compress = np.exp(-((yy_c / max(max_hw, 2.0)) ** 2))
            prof = prof * compress
            prof = prof / (float(np.max(prof)) + 1e-8) * mx
        out[:, j] = prof
    return np.clip(out * 0.90 + dw * 0.10, 0.0, 1.0).astype(np.float64)


def apply_disk_weight_pole_falloff(
    disk_weight: np.ndarray,
    height: int,
    *,
    sigma: float = 0.40,
    power: float = 1.18,
) -> np.ndarray:
    """Keep the galactic band away from equirect poles (clouds must not clip frame top/bottom)."""
    pole = latitude_plane_gate(int(height), sigma=sigma, power=power)
    if pole.ndim == 1:
        pole = pole.reshape(int(height), 1)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = dw.shape
    if pole.shape[0] != h:
        pole = np.broadcast_to(pole[:h], (h, w))
    elif pole.shape[1] == 1:
        pole = np.broadcast_to(pole, (h, w))
    return np.clip(dw * pole, 0.0, 1.0).astype(np.float64)


def soften_disk_weight_band_rim(
    disk_weight: np.ndarray,
    *,
    periodic_x: bool = True,
    strength: float = 0.74,
) -> np.ndarray:
    """Feather top/bottom band rims — reduces visible horizontal seams on the galactic plane."""
    _ = periodic_x
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return np.clip(disk_weight, 0.0, 1.0).astype(np.float64)
    from starsky_gen.dust_field import _blur_y_only_field

    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = dw.shape
    sig = float(np.clip(h * 0.034, 5.0, 36.0))
    soft = _blur_y_only_field(dw, sig)
    gy = np.abs(np.gradient(dw, axis=0))
    rim = np.clip(gy / (float(np.percentile(gy, 90.0)) + 1e-8), 0.0, 1.0) ** 0.88
    blend = rim * s * 0.48
    return np.clip(dw * (1.0 - blend) + soft * blend, 0.0, 1.0).astype(np.float64)


def build_disk_mesoscale_thickness_field(
    disk_weight: np.ndarray,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
    strength: float = 0.58,
) -> tuple[np.ndarray, np.ndarray]:
    """2D disk scale-height modulation: puff (+), compress/evacuate (−), shear warp.

    Returns ``(thickness_offset, lat_shear)`` where effective thickness uses
    ``sig_y *= clip(1 + thickness_offset, ...)`` — variable thickness, not holes.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = dw.shape
    if s < 1e-6:
        return np.zeros((h, w), dtype=np.float64), np.zeros((h, w), dtype=np.float64)
    salt = int(rng.integers(0, 2**31 - 1))
    ch, cw = max(14, h // 9), max(18, w // 6)
    cell_rng = derive_nebula_rng(salt, "disk_meso_cell")
    ridged = ridged_fbm2d(
        cell_rng,
        ch,
        cw,
        base_scale=0.26,
        octaves=5,
        periodic_x=periodic_x,
        elongate_along_x=1.28,
    )
    ridged = _resize_bilinear(ridged, h, w, periodic_x=periodic_x)
    cells = np.clip(1.0 - np.abs(ridged * 2.0 - 1.0), 0.0, 1.0) ** 1.22
    cellular = (cells - 0.40) * 1.75

    ch2, cw2 = max(10, h // 13), max(14, w // 8)
    basin_rng = derive_nebula_rng(salt, "disk_meso_basin")
    basin = fbm2d(
        basin_rng, ch2, cw2, base_scale=0.15, octaves=4, periodic_x=periodic_x
    )
    basin = _resize_bilinear(basin, h, w, periodic_x=periodic_x)
    basin = (basin - 0.48) * 1.65

    # No constant offset — mesoscale must vanish off the disk (was +0.12 leak).
    gate = np.clip(dw**1.08, 0.0, 1.0)
    meso_raw = np.clip((cellular * 0.58 + basin * 0.42) * gate, -0.62, 0.82) * s
    from starsky_gen.dust_field import _blur_x_only_field, _blur_y_only_field

    gy0 = np.gradient(meso_raw, axis=0)
    gx0 = np.gradient(meso_raw, axis=1)
    lat_shear = np.clip(gy0 * 0.36 + gx0 * 0.13, -0.28, 0.28)
    lat_shear = _blur_y_only_field(
        lat_shear, float(np.clip(h * 0.006, 0.85, 4.5))
    )
    meso = _blur_y_only_field(meso_raw, float(np.clip(h * 0.009, 1.2, 7.0)))
    meso = _blur_x_only_field(meso, 1.1, periodic_x=periodic_x)
    return meso.astype(np.float64), lat_shear.astype(np.float64)


def modulate_disk_weight_longitude(
    disk_weight: np.ndarray,
    lon_asymmetry: np.ndarray,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
    strength: float = 1.0,
) -> np.ndarray:
    """Longitude-varying disk envelope: brighter/thicker on disaster side, quieter opposite."""
    s = float(np.clip(strength, 0.0, 1.35))
    if s < 1e-6:
        return np.clip(disk_weight, 0.0, 1.0).astype(np.float64)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = dw.shape
    lon = np.clip(np.asarray(lon_asymmetry, dtype=np.float64), 0.42, 1.62)
    if lon.ndim == 1:
        lon = np.broadcast_to(lon.reshape(1, -1), (h, w))
    elif lon.shape != (h, w):
        lon = _resize_bilinear(lon, h, w, periodic_x=periodic_x)
    from starsky_gen.dust_field import _blur_x_only_field

    lon = _blur_x_only_field(lon, 2.8, periodic_x=periodic_x)
    ch, cw = max(6, h // 40), max(12, w // 24)
    tilt = fbm2d(rng, ch, cw, base_scale=0.11, octaves=3, periodic_x=periodic_x)
    tilt = _resize_bilinear(tilt, 1, w, periodic_x=periodic_x)[0:1, :]
    tilt = _blur_x_only_field(tilt, 5.0, periodic_x=periodic_x)
    tilt = np.clip((tilt - 0.48) * 1.35, -0.35, 0.35)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    lat_bias = 1.0 + tilt * yy * 0.22 * s
    # Mild lon modulation — strong 0.58+0.42*lon pinched half the band into black silhouettes.
    out = np.clip(dw * (0.82 + 0.18 * lon) * (0.92 + 0.08 * lat_bias), 0.0, 1.0)
    mx = float(np.max(out))
    if mx > 1e-8:
        out = out / mx
    return out.astype(np.float64)


def apply_variable_band_thickness(
    disk_weight: np.ndarray,
    rng: np.random.Generator,
    *,
    band_lat_sigma: float = 0.10,
    jitter_strength: float = 0.62,
    band_curvature_amp: float = 0.04,
    thickness_asymmetry: float = 0.38,
    mesoscale_strength: float = 0.58,
    periodic_x: bool = True,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Longitude-varying thickness + S/U/W lat curve; one side thinner, one thicker."""
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = dw.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    lat_curve, _curve_kind = sample_band_lat_curve(
        rng, w, band_curvature_amp=band_curvature_amp
    )
    meso, meso_shear = build_disk_mesoscale_thickness_field(
        dw, rng, periodic_x=periodic_x, strength=mesoscale_strength
    )
    yy_eff = yy - lat_curve - meso_shear
    ch, cw = max(6, h // 36), max(16, w // 20)
    j = float(np.clip(jitter_strength, 0.0, 1.0))
    lon_top = fbm2d(rng, ch, cw, base_scale=0.12, octaves=4, periodic_x=periodic_x)
    lon_top = _resize_bilinear(lon_top, 1, w, periodic_x=periodic_x)[0:1, :]
    rng_b = np.random.default_rng(int(rng.integers(0, 2**31)) ^ 0xB07A)
    lon_bot = fbm2d(rng_b, ch, cw, base_scale=0.14, octaves=4, periodic_x=periodic_x)
    lon_bot = _resize_bilinear(lon_bot, 1, w, periodic_x=periodic_x)[0:1, :]
    from starsky_gen.dust_field import _blur_x_only_field

    lon_top = _blur_x_only_field(lon_top, 4.2, periodic_x=periodic_x)
    lon_bot = _blur_x_only_field(lon_bot, 4.2, periodic_x=periodic_x)
    sig_base = float(np.clip(band_lat_sigma, 0.04, 0.22)) * 1.28
    # Cloud volume: keep thickness mostly even — large lon/meso bites read as black cutouts.
    thick_top = 0.78 + j * (np.clip(lon_top, 0.0, 1.0) ** 0.75) * 0.22
    thick_bot = 0.80 + j * (np.clip(lon_bot, 0.0, 1.0) ** 0.75) * 0.22
    asym = float(np.clip(thickness_asymmetry, 0.0, 0.72))
    if asym > 1e-6:
        thin_side = 1.0 if float(rng.uniform()) < 0.5 else -1.0
        bias = asym * float(rng.uniform(0.25, 0.50))
        thick_top = np.clip(thick_top * (1.0 + thin_side * bias), 0.55, 1.15)
        thick_bot = np.clip(thick_bot * (1.0 - thin_side * bias), 0.55, 1.15)
    sig_top = np.clip(sig_base * thick_top, 0.055, 0.42)
    sig_bot = np.clip(sig_base * thick_bot, 0.055, 0.44)
    sig_top = np.maximum(sig_top, sig_bot * 0.85)
    lon_warp = fbm2d(rng, ch, cw, base_scale=0.09, octaves=3, periodic_x=periodic_x)
    lon_warp = _resize_bilinear(lon_warp, 1, w, periodic_x=periodic_x)[0:1, :]
    lon_warp = _blur_x_only_field(lon_warp, 6.5, periodic_x=periodic_x)
    lon_scale = 0.94 + 0.10 * np.clip(lon_warp, 0.0, 1.0) ** 0.95
    sig_top = sig_top * lon_scale
    sig_bot = sig_bot * lon_scale
    sig_y = np.where(yy_eff >= 0.0, sig_top, sig_bot)
    sig_y = np.clip(sig_y, sig_base * 0.88, sig_base * 1.14)
    thick_mod = np.clip(1.0 + meso * 0.18, 0.88, 1.14)
    sig_y = sig_y * thick_mod
    band = np.exp(-((yy_eff / np.maximum(sig_y, 0.040)) ** 2))
    row_envelope = np.max(dw, axis=1, keepdims=True)
    row_envelope = row_envelope / (float(np.max(row_envelope)) + 1e-8)
    from starsky_gen.dust_field import _blur_y_only_field

    row_2d = np.broadcast_to(row_envelope, (h, w)).copy()
    row_soft = _blur_y_only_field(row_2d, float(np.clip(h * 0.018, 3.0, 12.0)))
    out = np.clip(band * (0.35 + 0.65 * row_soft), 0.0, 1.0)
    norm = float(np.percentile(dw, 91.0)) + 1e-8
    out = np.clip(out * (0.48 + 0.52 * dw / norm), 0.0, 1.0)
    out = soften_disk_weight_band_rim(out, periodic_x=periodic_x, strength=0.40)
    out = apply_disk_weight_pole_falloff(out, h, sigma=0.38, power=1.22)
    out = clamp_disk_band_thickness(out, periodic_x=periodic_x)
    return (
        (out / (float(np.max(out)) + 1e-8)).astype(np.float64),
        _curve_kind,
        meso.astype(np.float64),
    )


def build_band_lane_shell_profiles(
    disk_weight: np.ndarray,
    *,
    periodic_x: bool = True,
    lane_tightness: float = 0.36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the band into a tight inner dust lane and a puffier shell (sandwich profile).

    Returns ``(dust_lane, puff_shell, band_soft)`` — lane peaks on the warped mid-plane.
    """
    lt = float(np.clip(lane_tightness, 0.18, 0.55))
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = dw.shape
    soft = soften_band_envelope(dw, (h, w), periodic_x=periodic_x, lat_blur_sigma=14.0, power=0.50)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    row_peak = np.max(soft, axis=1, keepdims=True)
    lane_sig = np.clip(row_peak * lt + 0.032, 0.028, 0.12)
    lane = np.exp(-((yy**2) / (2.0 * lane_sig**2))) * soft
    lane = np.clip(lane**1.08, 0.0, 1.0)
    shell = np.clip(soft - lane * 0.78, 0.0, 1.0) ** 0.88
    return (
        lane.astype(np.float64),
        shell.astype(np.float64),
        soft.astype(np.float64),
    )


def _softmax_compete_channels(
    channels: list[np.ndarray],
    *,
    competition: float,
) -> list[np.ndarray]:
    """Sharpen competing architecture channels (0=unchanged, 1=near argmax)."""
    c = float(np.clip(competition, 0.0, 1.0))
    if c < 1e-6 or not channels:
        return channels
    # Temperature: high temp ≈ blend, low ≈ winner-take-most.
    temp = float(np.clip(1.35 - 1.15 * c, 0.18, 1.35))
    stack = np.stack([np.clip(np.asarray(ch, dtype=np.float64), 0.0, 1.0) for ch in channels], axis=0)
    # Stabilize: shift by max, then softmax.
    m = np.max(stack, axis=0, keepdims=True)
    ex = np.exp((stack - m) / temp)
    soft = ex / (np.sum(ex, axis=0, keepdims=True) + 1e-8)
    # Preserve total mass of each pixel's channel sum so morph energy stays similar.
    mass = np.sum(stack, axis=0, keepdims=True) + 1e-8
    out = soft * mass
    # Blend back toward originals so mid competition isn't too harsh.
    blend = c
    return [((1.0 - blend) * stack[i] + blend * out[i]).astype(np.float64) for i in range(stack.shape[0])]


def build_structure_architecture_weights(
    disk_weight: np.ndarray,
    vertical_extent: np.ndarray,
    structure_survival: np.ndarray,
    puff_field: np.ndarray,
    latent_turb: np.ndarray,
    *,
    lon_asymmetry: np.ndarray | None = None,
    erosion: np.ndarray | None = None,
    fractal_dark: np.ndarray | None = None,
    band_core_drive: float = 0.52,
    inner_dust_lane_strength: float = 0.72,
    strength: float = 1.0,
    type_competition: float = 0.0,
    plane_warp_amp: float = 0.07,
    periodic_x: bool = True,
) -> StructureArchitectureWeights:
    """Plan turbulent architecture weights across the full galactic band (not only rims)."""
    st = float(np.clip(strength, 0.0, 1.35))
    core = float(np.clip(band_core_drive, 0.22, 0.82))
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = dw.shape
    ve = np.clip(np.asarray(vertical_extent, dtype=np.float64), 0.0, 1.0)
    if ve.shape != (h, w):
        ve = _resize_bilinear(ve, h, w, periodic_x=periodic_x)
    surv = np.clip(np.asarray(structure_survival, dtype=np.float64) / 1.35, 0.0, 1.0)
    pf = np.clip(np.asarray(puff_field, dtype=np.float64), 0.0, 1.0)
    if pf.shape != (h, w):
        pf = _resize_bilinear(pf, h, w, periodic_x=periodic_x)
    turb = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
    if turb.shape != (h, w):
        turb = _resize_bilinear(turb, h, w, periodic_x=periodic_x)

    # Low-frequency latitude bend so the plane is not a perfectly flat equator.
    warp_a = float(np.clip(plane_warp_amp, 0.0, 0.18))
    if warp_a > 1e-6:
        dw = apply_low_frequency_band_warp(dw, amplitude=warp_a, periodic_x=periodic_x)
        dw = np.clip(dw / max(float(np.max(dw)), 1e-6), 0.0, 1.0)
        ve = apply_low_frequency_band_warp(ve, amplitude=warp_a * 0.85, periodic_x=periodic_x)
        ve = np.clip(ve, 0.0, 1.0)
        pf = apply_low_frequency_band_warp(pf, amplitude=warp_a * 0.70, periodic_x=periodic_x)
        pf = np.clip(pf, 0.0, 1.0)

    lane, shell, soft = build_band_lane_shell_profiles(
        dw, periodic_x=periodic_x, lane_tightness=0.30
    )
    lane_s = float(np.clip(inner_dust_lane_strength, 0.0, 1.25))
    if erosion is not None:
        er = np.clip(np.asarray(erosion, dtype=np.float64), 0.0, 1.0)
        if er.shape != (h, w):
            er = _resize_bilinear(er, h, w, periodic_x=periodic_x)
        fd = (
            np.clip(np.asarray(fractal_dark, dtype=np.float64), 0.0, 1.0)
            if fractal_dark is not None
            else er
        )
        if fd.shape != (h, w):
            fd = _resize_bilinear(fd, h, w, periodic_x=periodic_x)
        dark_tex = np.clip(er * 0.52 + fd * 0.38, 0.0, 1.0)
    else:
        dark_tex = lane * 0.5
    dust_lane = np.clip(lane * (0.42 + 0.58 * dark_tex) * lane_s, 0.0, 1.0)
    negative_space = np.clip(shell * (1.0 - dark_tex * 0.68) ** 1.02, 0.0, 1.0)
    scale = float(max(h, w))
    sig = float(np.clip(scale * 0.009, 0.8, 9.0))
    turb_hp = np.clip(turb - gaussian_blur_pil(turb, sig, periodic_x=periodic_x) * 0.74, 0.0, 1.0)

    band_host = np.clip(
        0.18 + 0.72 * np.maximum(soft, ve * 0.68) + pf**0.92 * 0.38,
        0.0,
        1.42,
    )
    core_drive = np.clip(soft**0.52 * (core + (1.0 - core) * ve * 0.72), 0.0, 1.0)
    plume_drive = np.clip(ve * (0.40 + 0.60 * soft), 0.0, 1.0)
    puff_drive = np.clip(pf**0.86 * (0.34 + 0.66 * soft), 0.0, 1.0)
    surv_drive = np.clip(surv * (0.36 + 0.64 * soft), 0.0, 1.0)
    turb_drive = np.clip(turb_hp * (0.28 + 0.72 * soft) + 0.12 * soft, 0.0, 1.0)

    turbulence_drive = np.clip(
        (
            core_drive * 0.36
            + plume_drive * 0.26
            + puff_drive * 0.22
            + surv_drive * 0.10
            + turb_drive * 0.06
        )
        * st,
        0.0,
        1.0,
    )
    band_floor = np.clip(soft * core + (1.0 - soft) * 0.18, 0.18, 0.62)
    turbulence_drive = np.clip(np.maximum(turbulence_drive, band_floor * st), 0.0, 1.0)
    turbulence_drive = np.clip(
        np.maximum(turbulence_drive, dust_lane * 0.48 + shell * 0.22), 0.0, 1.0
    )

    if lon_asymmetry is not None:
        lon = np.clip(np.asarray(lon_asymmetry, dtype=np.float64), 0.42, 1.62)
        if lon.ndim == 1:
            lon = np.broadcast_to(lon.reshape(1, -1), (h, w))
        elif lon.shape != (h, w):
            lon = _resize_bilinear(lon, h, w, periodic_x=periodic_x)
        turbulence_drive = np.clip(turbulence_drive * (0.82 + 0.22 * (lon - 1.0)), 0.0, 1.0)

    pole = latitude_plane_gate(h, sigma=0.44, power=1.08)
    if pole.ndim == 1:
        pole = pole.reshape(h, 1)
    pole2d = np.broadcast_to(pole, (h, w))
    turbulence_drive = np.clip(turbulence_drive * pole2d, 0.0, 1.0)
    band_host = np.clip(band_host * pole2d, 0.0, 1.42)

    puff_shell_w = np.clip(shell * turbulence_drive, 0.0, 1.0)
    puff_weight = np.clip(
        puff_drive * turbulence_drive * (0.28 + 0.72 * shell) + lane * pf * 0.12,
        0.0,
        1.0,
    )
    vertical_weight = np.clip(
        plume_drive * turbulence_drive * (0.40 + 0.60 * shell), 0.0, 1.0
    )
    reinject_weight = np.clip((surv_drive * 0.55 + plume_drive * 0.45) * turbulence_drive, 0.0, 1.0)
    survival_weight = np.clip(surv * band_host, 0.0, 1.0)
    turb_latent_weight = np.clip(turb_drive * turbulence_drive, 0.0, 1.0)

    competed = _softmax_compete_channels(
        [
            dust_lane,
            puff_weight,
            vertical_weight,
            reinject_weight,
            turb_latent_weight,
            survival_weight,
            negative_space,
            puff_shell_w,
        ],
        competition=float(type_competition),
    )
    (
        dust_lane,
        puff_weight,
        vertical_weight,
        reinject_weight,
        turb_latent_weight,
        survival_weight,
        negative_space,
        puff_shell_w,
    ) = competed

    return StructureArchitectureWeights(
        band_host=band_host.astype(np.float64),
        turbulence_drive=turbulence_drive.astype(np.float64),
        puff_weight=np.clip(puff_weight, 0.0, 1.0).astype(np.float64),
        vertical_weight=np.clip(vertical_weight, 0.0, 1.0).astype(np.float64),
        reinject_weight=np.clip(reinject_weight, 0.0, 1.0).astype(np.float64),
        dust_lane_weight=np.clip(dust_lane, 0.0, 1.0).astype(np.float64),
        puff_shell_weight=np.clip(puff_shell_w, 0.0, 1.0).astype(np.float64),
        negative_space_weight=np.clip(negative_space, 0.0, 1.0).astype(np.float64),
        survival_weight=np.clip(survival_weight, 0.0, 1.0).astype(np.float64),
        turb_latent_weight=np.clip(turb_latent_weight, 0.0, 1.0).astype(np.float64),
    )


# RGB legend for structure_type_map.png debug export (see build_structure_architecture_type_map).
ARCHITECTURE_TYPE_COLORS: dict[str, tuple[float, float, float]] = {
    "dust_lane": (0.72, 0.22, 0.12),
    "puff": (0.13, 0.73, 0.93),
    "vertical": (0.27, 0.87, 0.40),
    "reinject": (0.92, 0.33, 0.67),
    "turb": (0.93, 0.80, 0.13),
    "survival": (0.65, 0.45, 0.28),
    "negative_space": (0.88, 0.90, 0.95),
    "low_drive": (0.12, 0.12, 0.18),
}


def build_structure_architecture_type_map(
    architecture: StructureArchitectureWeights,
    *,
    blend: bool = True,
    low_drive_threshold: float = 0.12,
) -> np.ndarray:
    """RGB type map: which architecture channel dominates each pixel (debug tuning).

    Red-brown=dust_lane, cyan=puff, green=vertical, magenta=reinject, yellow=turb,
    brown=survival, pale=negative_space, dark=low ``turbulence_drive``.
    Weighted blend when ``blend=True`` (mixed types); weights sharpened for clearer edges.
    """
    arch = architecture
    drive = np.clip(arch.turbulence_drive, 0.0, 1.0)
    h, w = drive.shape
    sharp = lambda w: np.clip(np.asarray(w, dtype=np.float64), 0.0, 1.0) ** 1.52

    channels: list[tuple[np.ndarray, tuple[float, float, float]]] = [
        (sharp(arch.dust_lane_weight), ARCHITECTURE_TYPE_COLORS["dust_lane"]),
        (sharp(arch.puff_weight), ARCHITECTURE_TYPE_COLORS["puff"]),
        (sharp(arch.vertical_weight), ARCHITECTURE_TYPE_COLORS["vertical"]),
        (sharp(arch.reinject_weight), ARCHITECTURE_TYPE_COLORS["reinject"]),
        (sharp(arch.turb_latent_weight), ARCHITECTURE_TYPE_COLORS["turb"]),
        (sharp(arch.survival_weight), ARCHITECTURE_TYPE_COLORS["survival"]),
        (sharp(arch.negative_space_weight), ARCHITECTURE_TYPE_COLORS["negative_space"]),
    ]
    if blend:
        rgb = np.zeros((h, w, 3), dtype=np.float64)
        total = np.zeros((h, w), dtype=np.float64)
        for wt, col in channels:
            rgb += wt[..., np.newaxis] * np.asarray(col, dtype=np.float64)
            total += wt
        total = np.maximum(total, 1e-6)
        rgb = rgb / total[..., np.newaxis]
    else:
        stacks = np.stack([c[0] for c in channels], axis=-1)
        dominant = np.argmax(stacks, axis=-1)
        palette = np.asarray([c[1] for c in channels], dtype=np.float64)
        rgb = palette[dominant]
    low = drive < float(np.clip(low_drive_threshold, 0.04, 0.35))
    rgb[low] = np.asarray(ARCHITECTURE_TYPE_COLORS["low_drive"], dtype=np.float64)
    rgb = np.clip(rgb * drive[..., np.newaxis], 0.0, 1.0)
    return rgb.astype(np.float64)


def build_structure_morph_host(
    disk_weight: np.ndarray,
    vertical_extent: np.ndarray | None = None,
    puff_field: np.ndarray | None = None,
    *,
    shape: tuple[int, int] | None = None,
    periodic_x: bool = True,
) -> np.ndarray:
    """Host wider than disk_weight so morphology/puffs are not clipped to a razor band."""
    h, w = shape if shape is not None else disk_weight.shape
    soft = soften_band_envelope(disk_weight, (h, w), periodic_x=periodic_x, lat_blur_sigma=18.0, power=0.50)
    host = np.clip(soft, 0.0, 1.0)
    if vertical_extent is not None:
        ve = np.clip(np.asarray(vertical_extent, dtype=np.float64), 0.0, 1.0)
        if ve.shape != (h, w):
            ve = _resize_bilinear(ve, h, w, periodic_x=periodic_x)
        host = np.clip(np.maximum(host, ve * 0.68), 0.0, 1.35)
    if puff_field is not None:
        pf = np.clip(np.asarray(puff_field, dtype=np.float64), 0.0, 1.0)
        if pf.shape != (h, w):
            pf = _resize_bilinear(pf, h, w, periodic_x=periodic_x)
        host = np.clip(np.maximum(host, pf**1.08 * 0.58), 0.0, 1.42)
    return host.astype(np.float64)


def morphology_puff_punch_mask(
    dust_absorption_morph: np.ndarray,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
) -> np.ndarray:
    """Where puffy clouds punch through the band — passed to extinction for dramatic occlusion."""
    morph = np.clip(np.asarray(dust_absorption_morph, dtype=np.float64), 0.0, 1.0)
    h, w = morph.shape
    puffs = build_fine_puff_field(rng, h, w, periodic_x=periodic_x, strength=1.0, center_boost=0.85)
    scale = float(max(h, w))
    med = gaussian_blur_pil(morph, float(np.clip(scale * 0.014, 1.5, 16.0)), periodic_x=periodic_x)
    hp = np.clip(morph - med * 0.74, 0.0, 1.0) ** 1.06
    return np.clip(puffs * 0.52 + hp * 0.48, 0.0, 1.0).astype(np.float64)


def build_extinction_coupling_field(
    morph_host: np.ndarray,
    dust_absorption_morph: np.ndarray,
    vertical_extent: np.ndarray | None = None,
    puff_punch: np.ndarray | None = None,
    *,
    periodic_x: bool = True,
    floor: float = 0.34,
) -> np.ndarray:
    """Morphology-led weight for extinction — clouds hang near the band but are not rail-clipped."""
    host = np.clip(np.asarray(morph_host, dtype=np.float64), 0.0, 1.42)
    morph = np.clip(np.asarray(dust_absorption_morph, dtype=np.float64), 0.0, 1.0)
    h, w = morph.shape
    scale = float(max(h, w))
    sig = float(np.clip(scale * 0.012, 1.2, 14.0))
    med = gaussian_blur_pil(morph, sig, periodic_x=periodic_x)
    hp = np.clip(morph - med * 0.74, 0.0, 1.0) ** 1.04
    f = float(np.clip(floor, 0.22, 0.48))
    cw = np.clip(f + (1.0 - f) * np.clip(host * 0.70 + hp * 0.88, 0.0, 1.0), f, 1.15)
    if vertical_extent is not None:
        ve = np.clip(np.asarray(vertical_extent, dtype=np.float64), 0.0, 1.0)
        if ve.shape != (h, w):
            ve = _resize_bilinear(ve, h, w, periodic_x=periodic_x)
        cw = np.clip(np.maximum(cw, f + ve * (1.0 - f) * 0.60), f, 1.2)
    if puff_punch is not None:
        pp = np.clip(np.asarray(puff_punch, dtype=np.float64), 0.0, 1.0)
        if pp.shape != (h, w):
            pp = _resize_bilinear(pp, h, w, periodic_x=periodic_x)
        cw = np.clip(np.maximum(cw, f + pp * (1.08 - f)), f, 1.28)
    return cw.astype(np.float64)


def decouple_dust_from_band_gate(
    dust_absorption: np.ndarray,
    morph_host: np.ndarray,
    disk_weight: np.ndarray,
    *,
    periodic_x: bool = True,
    strength: float = 0.72,
    architecture_drive: np.ndarray | None = None,
) -> np.ndarray:
    """Let dust_A bulge off the disk_weight rim where structure host exceeds the soft band."""
    s = float(np.clip(strength, 0.0, 1.2))
    a = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    h, w = a.shape
    soft = soften_band_envelope(disk_weight, (h, w), periodic_x=periodic_x, lat_blur_sigma=16.0, power=0.52)
    host = np.clip(np.asarray(morph_host, dtype=np.float64), 0.0, 1.42)
    punch = np.clip(host - soft * 0.82, 0.0, 1.0) ** 0.92
    if architecture_drive is not None:
        punch = np.clip(punch * np.clip(np.asarray(architecture_drive, dtype=np.float64), 0.0, 1.0), 0.0, 1.0)
    out = a.copy()
    rim = soft < 0.58
    if bool(np.any(rim)):
        lift = np.clip(0.10 + punch * 0.48 * s, 0.0, 0.58)
        out[rim] = np.clip(np.maximum(out[rim], lift[rim] * (0.35 + 0.65 * a[rim])), 0.06, 0.82)
    band = soft >= 0.22
    if bool(np.any(band)):
        lift_b = np.clip(0.08 + punch * 0.32 * s, 0.0, 0.48)
        out[band] = np.clip(np.maximum(out[band], lift_b[band] * (0.42 + 0.58 * a[band])), 0.06, 0.82)
    peak = punch > float(np.percentile(punch[soft > 0.18], 70.0)) if bool(np.any(soft > 0.18)) else 0.45
    if bool(np.any(peak)):
        out[peak] = np.clip(out[peak] + punch[peak] * 0.14 * s, 0.06, 0.85)
    return out.astype(np.float64)


def blend_fluffy_cloud_texture_into_morph_dust(
    dust_absorption: np.ndarray,
    puff_soft: np.ndarray,
    vertical_extent: np.ndarray,
    latent_turb: np.ndarray,
    morph_host: np.ndarray,
    disk_weight: np.ndarray,
    *,
    periodic_x: bool = True,
    strength: float = 0.78,
    architecture: StructureArchitectureWeights | None = None,
) -> np.ndarray:
    """Spread cotton-ball texture through band via architecture weights (or legacy rim gate)."""
    from starsky_gen.procedural_noise import gaussian_blur_pil

    s = float(np.clip(strength, 0.0, 1.25))
    if s < 1e-6:
        return np.clip(dust_absorption, 0.0, 1.0).astype(np.float64)
    a = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    h, w = a.shape
    pf = np.clip(np.asarray(puff_soft, dtype=np.float64), 0.0, 1.0)
    if pf.shape != (h, w):
        pf = _resize_bilinear(pf, h, w, periodic_x=periodic_x)
    ve = np.clip(np.asarray(vertical_extent, dtype=np.float64), 0.0, 1.0)
    if ve.shape != (h, w):
        ve = _resize_bilinear(ve, h, w, periodic_x=periodic_x)
    turb = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
    scale = float(max(h, w))
    sig = float(np.clip(scale * 0.009, 0.8, 9.0))
    turb_hp = np.clip(turb - gaussian_blur_pil(turb, sig, periodic_x=periodic_x) * 0.74, 0.0, 1.0)
    cloud = np.clip(pf**0.90 * 0.58 + ve**0.85 * 0.28 + turb_hp * 0.14, 0.0, 1.0)
    cloud = gaussian_blur_pil(cloud, sig * 1.15, periodic_x=periodic_x)
    target = np.clip(0.22 + cloud * 0.62, 0.08, 0.78)
    if architecture is not None:
        w = np.clip(architecture.turbulence_drive * s * 0.55, 0.0, 1.0)
    else:
        host = np.clip(np.asarray(morph_host, dtype=np.float64), 0.0, 1.42)
        soft = soften_band_envelope(disk_weight, (h, w), periodic_x=periodic_x, lat_blur_sigma=14.0, power=0.52)
        w = np.clip(host * (0.42 + 0.58 * (1.0 - soft**0.78)), 0.0, 1.0) * s * 0.42
    return np.clip(a * (1.0 - w * 0.58) + target * w * 0.42 + a * w * 0.58, 0.06, 0.82).astype(
        np.float64
    )


def apply_inner_dust_subband(
    dust_absorption: np.ndarray,
    architecture: StructureArchitectureWeights,
    *,
    periodic_x: bool = True,
    strength: float = 0.88,
) -> np.ndarray:
    """Raise absorption on the inner dust lane (dark sub-band inside the galactic band)."""
    s = float(np.clip(strength, 0.0, 1.25))
    if s < 1e-6:
        return np.clip(dust_absorption, 0.0, 1.0).astype(np.float64)
    a = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    lane = np.clip(np.asarray(architecture.dust_lane_weight, dtype=np.float64), 0.0, 1.0)
    if lane.shape != a.shape:
        lane = _resize_bilinear(lane, a.shape[0], a.shape[1], periodic_x=periodic_x)
    target = np.clip(0.46 + lane * 0.30, 0.26, 0.82)
    w = lane * s * 0.58
    return np.clip(a * (1.0 - w * 0.36) + target * w * 0.50 + a * w * 0.50, 0.08, 0.86).astype(
        np.float64
    )


def apply_band_negative_space_transmission(
    transmission: np.ndarray,
    architecture: StructureArchitectureWeights,
    *,
    void_floor: float = 0.06,
    strength: float = 0.72,
) -> np.ndarray:
    """Open clear pockets in the puff shell (negative space between clouds)."""
    s = float(np.clip(strength, 0.0, 1.2))
    if s < 1e-6:
        return np.clip(transmission, void_floor, 1.0).astype(np.float64)
    vf = float(np.clip(void_floor, 0.01, 0.2))
    t = np.clip(np.asarray(transmission, dtype=np.float64), vf, 1.0)
    neg = np.clip(np.asarray(architecture.negative_space_weight, dtype=np.float64), 0.0, 1.0)
    if neg.shape != t.shape:
        neg = _resize_bilinear(neg, t.shape[0], t.shape[1], periodic_x=True)
    gap = neg * s * 0.58
    return np.clip(t + gap * (1.0 - t) * 0.62, vf, 1.0).astype(np.float64)


def apply_structure_architecture_pipeline(
    dust_absorption: np.ndarray,
    architecture: StructureArchitectureWeights,
    vertical_extent: np.ndarray,
    structure_survival: np.ndarray,
    puff_field: np.ndarray,
    puff_rng: np.random.Generator,
    latent_turb: np.ndarray,
    latent_ridge: np.ndarray,
    disk_weight: np.ndarray,
    *,
    periodic_x: bool = True,
    reinject_strength: float = 0.88,
    puff_inject_strength: float = 0.42,
    cloud_blend_strength: float = 0.92,
    inner_dust_lane_strength: float = 0.88,
    decouple_primary: float = 0.72,
    decouple_secondary: float = 0.55,
) -> np.ndarray:
    """Apply reinject → decouple → puff inject → cloud blend using shared architecture weights."""
    arch = architecture
    a = reinject_vertical_dust_structure(
        dust_absorption,
        vertical_extent,
        structure_survival,
        latent_turb,
        latent_ridge,
        disk_weight,
        periodic_x=periodic_x,
        strength=reinject_strength,
        drive_weight=arch.reinject_weight,
        vertical_extent_drive=arch.vertical_weight,
    )
    puff_soft = gaussian_blur_pil(
        np.clip(puff_field, 0.0, 1.0),
        float(np.clip(max(a.shape) * 0.011, 1.0, 9.0)),
        periodic_x=periodic_x,
    )
    a = decouple_dust_from_band_gate(
        a,
        arch.band_host,
        disk_weight,
        periodic_x=periodic_x,
        strength=decouple_primary,
        architecture_drive=arch.turbulence_drive,
    )
    a = inject_band_cloud_puffs(
        a,
        arch.band_host,
        puff_rng,
        periodic_x=periodic_x,
        strength=puff_inject_strength,
        host_weight=arch.puff_weight,
    )
    a = decouple_dust_from_band_gate(
        a,
        arch.band_host,
        disk_weight,
        periodic_x=periodic_x,
        strength=decouple_secondary,
        architecture_drive=arch.turbulence_drive,
    )
    a = blend_fluffy_cloud_texture_into_morph_dust(
        a,
        puff_soft,
        puff_field,
        latent_turb,
        arch.band_host,
        disk_weight,
        periodic_x=periodic_x,
        strength=cloud_blend_strength,
        architecture=arch,
    )
    a = apply_inner_dust_subband(
        a, arch, periodic_x=periodic_x, strength=inner_dust_lane_strength
    )
    return np.clip(a, 0.06, 0.88).astype(np.float64)


def build_asymmetric_band_bleed_envelope(
    disk_weight: np.ndarray,
    height: int,
    width: int,
    rng: np.random.Generator,
    *,
    lon_asymmetry: np.ndarray | None = None,
    puff_field: np.ndarray | None = None,
    morph_absorption: np.ndarray | None = None,
    periodic_x: bool = True,
    bleed_strength: float = 0.64,
) -> np.ndarray:
    """Soft, longitude-varying band host for haze/cloud/bloom (replaces sharp exp(-y²) railroad gates)."""
    h, w = int(height), int(width)
    s = float(np.clip(bleed_strength, 0.0, 1.25))
    soft = soften_band_envelope(
        disk_weight, (h, w), periodic_x=periodic_x, lat_blur_sigma=24.0, power=0.46
    )
    env = np.clip(0.12 + 0.88 * soft**0.72, 0.0, 1.0)
    ch_e, cw_e = max(4, h // 28), max(10, w // 18)
    lon_tex = fbm2d(rng, ch_e, cw_e, base_scale=0.14, octaves=3, periodic_x=periodic_x)
    lon_tex = _resize_bilinear(lon_tex, 1, w, periodic_x=periodic_x)[0:1, :]
    env = np.clip(env * (0.90 + 0.14 * lon_tex), 0.0, 1.38)
    if lon_asymmetry is not None:
        lon = np.clip(np.asarray(lon_asymmetry, dtype=np.float64), 0.42, 1.62)
        if lon.ndim == 1:
            lon = np.broadcast_to(lon.reshape(1, -1), (h, w))
        elif lon.shape != (h, w):
            lon = _resize_bilinear(lon, h, w, periodic_x=periodic_x)
        env = np.clip(env * (0.88 + 0.24 * (lon - 1.0) * soft), 0.0, 1.38)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float64, endpoint=False)[None, :]
    ph = float(rng.uniform(0.0, 6.283185307179586))
    wobble = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx * 2.1 + 0.12 * yy + ph))
    rim = np.clip(1.0 - soft, 0.0, 1.0) ** 0.82
    env = np.clip(env + rim * wobble * 0.24 * s, 0.0, 1.38)
    if puff_field is not None:
        pf = np.clip(np.asarray(puff_field, dtype=np.float64), 0.0, 1.0)
        if pf.shape != (h, w):
            pf = _resize_bilinear(pf, h, w, periodic_x=periodic_x)
        peaks = pf**1.14
        thr = float(np.percentile(peaks[soft > 0.22], 68.0)) if bool(np.any(soft > 0.22)) else 0.5
        punch = np.clip((peaks - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0) ** 1.08
        env = np.clip(np.maximum(env, punch * (0.38 + 0.62 * soft) * s), 0.0, 1.48)
    if morph_absorption is not None:
        ma = np.clip(np.asarray(morph_absorption, dtype=np.float64), 0.0, 1.0)
        if ma.shape != (h, w):
            ma = _resize_bilinear(ma, h, w, periodic_x=periodic_x)
        scale = float(max(h, w))
        med = gaussian_blur_pil(ma, float(np.clip(scale * 0.014, 1.5, 16.0)), periodic_x=periodic_x)
        hp = np.clip(ma - med * 0.74, 0.0, 1.0)
        env = np.clip(np.maximum(env, hp * soft * 0.48 * s), 0.0, 1.42)
    env = soften_disk_weight_band_rim(env, periodic_x=periodic_x, strength=0.58)
    return env.astype(np.float64)


def reinject_vertical_dust_structure(
    dust_absorption: np.ndarray,
    vertical_extent: np.ndarray,
    structure_survival: np.ndarray,
    latent_turb: np.ndarray,
    latent_ridge: np.ndarray,
    disk_weight: np.ndarray,
    *,
    periodic_x: bool = True,
    strength: float = 0.78,
    drive_weight: np.ndarray | None = None,
    vertical_extent_drive: np.ndarray | None = None,
) -> np.ndarray:
    """Restore torn edges, cavities, and wispy fragments; optional architecture weight maps."""
    s = float(np.clip(strength, 0.0, 1.35))
    if s < 1e-6:
        return np.clip(dust_absorption, 0.0, 1.0).astype(np.float64)
    a = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    h, w = a.shape
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    if dw.shape != (h, w):
        dw = _resize_bilinear(dw, h, w, periodic_x=periodic_x)
    ve = np.clip(np.asarray(vertical_extent, dtype=np.float64), 0.0, 1.0)
    if ve.shape != (h, w):
        ve = _resize_bilinear(ve, h, w, periodic_x=periodic_x)
    surv = np.clip(np.asarray(structure_survival, dtype=np.float64), 0.0, 1.35)
    turb = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
    ridge = np.clip(np.asarray(latent_ridge, dtype=np.float64), 0.0, 1.0)
    soft_env = soften_band_envelope(dw, (h, w), periodic_x=periodic_x, lat_blur_sigma=20.0, power=0.52)
    off = np.clip(ve * (1.0 - soft_env**0.82), 0.0, 1.0)
    ridge_fil = np.clip(1.0 - np.abs(ridge * 2.0 - 1.0), 0.0, 1.0) ** 1.10
    scale = float(max(h, w))
    med_sig = float(np.clip(scale * 0.016, 2.0, 18.0))
    fine_sig = float(np.clip(scale * 0.0048, 0.65, 7.5))
    ve_med = gaussian_blur_pil(ve, med_sig, periodic_x=periodic_x)
    ve_hp = np.clip(ve - ve_med * 0.76, -0.45, 0.45)
    turb_hp = np.clip(
        turb - gaussian_blur_pil(turb, fine_sig, periodic_x=periodic_x) * 0.74, 0.0, 1.0
    )
    ridge_hp = np.clip(
        ridge_fil - gaussian_blur_pil(ridge_fil, fine_sig * 1.2, periodic_x=periodic_x) * 0.78,
        0.0,
        1.0,
    )
    fragment = np.clip(ve_hp * 0.46 + turb_hp * 0.34 + ridge_hp * 0.28, 0.0, 1.0)
    tear = np.clip(fragment - gaussian_blur_pil(fragment, med_sig, periodic_x=periodic_x) * 0.70, 0.0, 1.0)
    cavity = np.clip(ve_med * 0.55 - fragment * 0.42, 0.0, 1.0) ** 1.18
    plume = off * (0.48 + 0.52 * np.clip(surv / 1.35, 0.0, 1.0))
    band_plume = np.clip(soft_env * (0.40 + 0.60 * ve), 0.0, 1.0)
    plume = np.clip(np.maximum(plume, band_plume * 0.72), 0.0, 1.0)
    if vertical_extent_drive is not None:
        ve_d = np.clip(np.asarray(vertical_extent_drive, dtype=np.float64), 0.0, 1.0)
        if ve_d.shape != (h, w):
            ve_d = _resize_bilinear(ve_d, h, w, periodic_x=periodic_x)
        plume = np.clip(np.maximum(plume, ve_d * (0.38 + 0.62 * soft_env)), 0.0, 1.0)
    mod = s * (0.30 + 0.70 * plume)
    if drive_weight is not None:
        dwt = np.clip(np.asarray(drive_weight, dtype=np.float64), 0.0, 1.0)
        if dwt.shape != (h, w):
            dwt = _resize_bilinear(dwt, h, w, periodic_x=periodic_x)
        mod = mod * (0.28 + 0.72 * dwt)
    out = np.clip(a + plume * mod * 0.24 + fragment * mod * 0.20 + tear * mod * 0.22, 0.06, 0.88)
    out = np.clip(out * (1.0 - cavity * mod * 0.14 * np.clip(a, 0.35, 1.0)), 0.06, 0.88)
    pole = latitude_plane_gate(h, sigma=0.40, power=1.16)
    if pole.ndim == 1:
        pole = pole.reshape(h, 1)
    pole2d = np.broadcast_to(pole, (h, w))
    halo = soft_env < 0.44
    if bool(np.any(halo)):
        bleed = np.clip(0.10 + off * 0.32 + tear * 0.14, 0.0, 0.48) * pole2d
        out[halo] = np.clip(np.maximum(out[halo], bleed[halo] * s * 0.72), 0.08, 0.48)
    out_hp = np.clip(out - gaussian_blur_pil(out, med_sig, periodic_x=periodic_x) * 0.74, -0.35, 0.35)
    core = soft_env > 0.28
    out[core] = np.clip(out[core] + out_hp[core] * s * 0.12, 0.06, 0.88)
    return out.astype(np.float64)


def seam_safe_lon_weights(width: int, *, guard_frac: float = 0.06) -> np.ndarray:
    """Taper fields at equirect longitude seams (x=0 ≡ x=width)."""
    gf = float(np.clip(guard_frac, 0.02, 0.14))
    u = np.linspace(0.0, 1.0, int(width), dtype=np.float64, endpoint=False)
    d = np.minimum(u, 1.0 - u)
    return np.clip(d / gf, 0.12, 1.0)[np.newaxis, :]


def build_longitude_asymmetry(
    width: int,
    *,
    strength: float = 0.88,
) -> np.ndarray:
    """Left quiet → center fractured → right dense disaster (1×W)."""
    s = float(np.clip(strength, 0.0, 1.35))
    if s < 1e-6:
        return np.ones((1, int(width)), dtype=np.float64)
    xx = np.linspace(-1.0, 1.0, int(width), dtype=np.float64)[np.newaxis, :]
    quiet = 0.50 + 0.50 * np.clip((1.0 - xx) / 1.12, 0.0, 1.0) ** 1.08
    disaster = 0.38 + 0.62 * np.clip((xx + 0.10) / 0.90, 0.0, 1.0) ** 1.42
    fractured = 0.58 + 0.42 * np.exp(-(xx**2) / 0.11)
    raw = quiet * (1.0 - 0.48 * fractured) + disaster * fractured * 1.38
    mu = float(np.mean(raw))
    return np.clip(1.0 + (raw / max(mu, 1e-8) - 1.0) * s, 0.42, 1.62).astype(np.float64)


def build_vertical_structure_envelope(
    height: int,
    width: int,
    rng: np.random.Generator,
    stellar_density: np.ndarray,
    disk_weight: np.ndarray,
    *,
    band_lat_sigma: float,
    extent_strength: float,
    host_latitude_scale: float,
    periodic_x: bool,
    mesoscale_field: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """structure_survival, vertical_extent, lat_shift — band hosts structure, does not clip it."""
    h, w = int(height), int(width)
    es = float(np.clip(extent_strength, 0.0, 1.25))
    sig = max(float(band_lat_sigma) * float(host_latitude_scale), 0.06)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    host = np.exp(-((yy**2) / (2.0 * sig**2)))

    ch, cw = max(8, h // 32), max(12, w // 48)
    extent_lo = fbm2d(rng, ch, cw, base_scale=0.13, octaves=4, periodic_x=periodic_x)
    extent_lo = _resize_bilinear(extent_lo, h, w, periodic_x=periodic_x)
    vertical_extent = np.clip(extent_lo**0.90 * (0.30 + 0.70 * host), 0.0, 1.0)
    if mesoscale_field is not None:
        meso = np.clip(np.asarray(mesoscale_field, dtype=np.float64), -0.65, 0.85)
        if meso.shape != (h, w):
            meso = _resize_bilinear(meso, h, w, periodic_x=periodic_x)
        thick_v = np.clip(1.0 + meso * 0.78, 0.34, 1.82)
        vertical_extent = np.clip(vertical_extent * thick_v, 0.0, 1.0)
        meso_col = np.mean(meso, axis=0, keepdims=True)
        sig_col = sig * np.clip(1.0 + meso_col * 0.62, 0.38, 1.75)
        host = np.exp(-((yy**2) / (2.0 * sig_col**2)))

    soft = gaussian_blur_pil(extent_lo, 9.0, periodic_x=periodic_x)
    detail = gaussian_blur_pil(extent_lo, 1.1, periodic_x=periodic_x)
    vert_warp = np.clip(soft - detail, -0.35, 0.35)
    lat_shift = (vertical_extent - 0.5) * es * 0.48 + vert_warp * es * 0.32

    g = np.clip(np.asarray(stellar_density, dtype=np.float64), 0.0, 1.0)
    g_norm = g / (float(np.percentile(g, 92.0)) + 1e-8)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    structure_survival = np.clip(
        host * (0.26 + 0.74 * vertical_extent) * (0.32 + 0.68 * g_norm) * (0.55 + 0.45 * dw),
        0.02,
        1.35,
    )
    off_lift = (
        vertical_extent * np.clip(1.0 - host, 0.0, 1.0) * (0.30 + 0.38 * g_norm)
    )
    structure_survival = np.clip(
        np.maximum(structure_survival, host * dw * 0.34) + off_lift * 1.52,
        0.02,
        1.35,
    )
    return (
        structure_survival.astype(np.float64),
        vertical_extent.astype(np.float64),
        lat_shift.astype(np.float64),
    )


def build_brutal_erasure_mask(
    dust_absorption: np.ndarray,
    void_mask: np.ndarray,
    obliteration: np.ndarray,
    disk_weight: np.ndarray,
    erosion: np.ndarray | None,
    *,
    strength: float,
) -> np.ndarray:
    """Pockets where extinction erases (~5% survival), not softens."""
    s = float(np.clip(strength, 0.0, 1.25))
    if s < 1e-6:
        return np.zeros_like(dust_absorption, dtype=np.float64)
    h, w = int(dust_absorption.shape[0]), int(dust_absorption.shape[1])
    scale = float(max(h, w))
    hp_sig = float(np.clip(scale * 0.04, 5.0, 42.0))

    def _hp(field: np.ndarray) -> np.ndarray:
        f = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
        soft = gaussian_blur_pil(f, hp_sig, periodic_x=True)
        return np.clip(f - soft * 0.80, 0.0, 1.0) ** 1.15

    stack = _hp(dust_absorption)
    stack = np.maximum(stack, _hp(void_mask) * 0.94)
    stack = np.maximum(stack, _hp(obliteration) * 0.90)
    if erosion is not None:
        stack = np.maximum(stack, np.clip(erosion, 0.0, 1.0) * 0.82)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    detail = np.clip(stack, 0.0, 1.0) ** 1.12
    thr = float(np.percentile(detail, 84.0))
    brutal = np.clip((detail - thr) / max(1.0 - thr, 1e-8), 0.0, 1.0) ** 2.05
    return np.clip(brutal * dw * s, 0.0, 1.0).astype(np.float64)


def apply_brutal_erasure_transmission(
    transmission: np.ndarray,
    brutal: np.ndarray,
    *,
    survival_floor: float,
    periodic_x: bool = True,
) -> np.ndarray:
    """Carve torn lanes with soft shoulders (blurred carve), not razor caps."""
    floor = float(np.clip(survival_floor, 0.02, 0.12))
    t = np.clip(np.asarray(transmission, dtype=np.float64), floor, 1.0)
    b = np.clip(np.asarray(brutal, dtype=np.float64), 0.0, 1.0)
    h, w = b.shape
    scale = float(max(h, w))
    b_soft = gaussian_blur_pil(b, float(np.clip(scale * 0.028, 3.0, 36.0)), periodic_x=periodic_x)
    b_lane = np.clip(b - b_soft * 0.76, 0.0, 1.0) ** 1.35
    hp = np.clip(b - b_soft * 0.92, 0.0, 1.0) ** 1.18
    thr_b = float(np.percentile(b, 78.0))
    blob = np.clip(b - thr_b, 0.0, 1.0) ** 1.22
    carve = np.maximum(b_lane, np.maximum(hp * 0.62, blob * 0.42)) * (1.0 - floor) * 0.90
    # Feather carve so deep windows keep soft volumetric shoulders.
    carve = gaussian_blur_pil(
        carve, float(np.clip(scale * 0.010, 1.2, 12.0)), periodic_x=periodic_x
    )
    cap = floor + (1.0 - floor) * (1.0 - b_lane) ** 2.35
    erased = np.clip(t * (1.0 - carve), floor, 1.0)
    return np.minimum(erased, cap).astype(np.float64)


def apply_low_frequency_band_warp(
    field: np.ndarray,
    *,
    amplitude: float = 0.07,
    periodic_x: bool = True,
) -> np.ndarray:
    """Bend a band field in latitude with a wrap-safe low-frequency longitude warp."""
    amp = float(np.clip(amplitude, 0.0, 0.22))
    a = np.asarray(field, dtype=np.float64)
    if amp < 1e-6 or a.ndim != 2:
        return a
    from scipy.ndimage import map_coordinates

    h, w = int(a.shape[0]), int(a.shape[1])
    xx = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float64)
    # Two harmonics: slow primary bend + mild secondary so the plane is not a ruler.
    offset = amp * np.sin(2.0 * np.pi * xx) + 0.42 * amp * np.sin(4.0 * np.pi * xx + 0.85)
    row_shift = offset * (0.5 * float(h - 1))
    yy = np.arange(h, dtype=np.float64)[:, None]
    xx_i = np.arange(w, dtype=np.float64)[None, :]
    src_y = np.clip(yy - row_shift[None, :], 0.0, float(h - 1))
    src_x = np.broadcast_to(xx_i, (h, w)).astype(np.float64)
    if periodic_x:
        src_x = np.mod(src_x, float(w))
    warped = map_coordinates(
        a,
        [src_y.ravel(), src_x.ravel()],
        order=1,
        mode="nearest",
        prefilter=False,
    ).reshape(h, w)
    return np.clip(warped, 0.0, None).astype(np.float64)


def scatter_disaster_lon_peaks(
    field: np.ndarray,
    rng: np.random.Generator,
    *,
    n_peaks: int,
    width: int,
    height: int,
    strength: float,
    periodic_x: bool,
    avoid_seam_frac: float = 0.10,
) -> np.ndarray:
    """Physical hotspots away from the seam — balances accidental edge clusters."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6 or n_peaks < 1:
        return field
    out = np.asarray(field, dtype=np.float64).copy()
    guard = seam_safe_lon_weights(width, guard_frac=avoid_seam_frac)
    margin = max(4, int(width * avoid_seam_frac))
    for _ in range(int(n_peaks)):
        ci = int(rng.integers(margin, max(margin + 1, width - margin)))
        ri = int(rng.integers(height // 5, max(height // 5 + 1, 4 * height // 5)))
        amp = float(rng.uniform(0.42, 0.88)) * s
        out[ri, ci] = max(out[ri, ci], amp)
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
    plane = np.exp(-((yy**2) / 0.38))
    out = out * plane * guard
    out = gaussian_blur_pil(out, 0.9, periodic_x=periodic_x)
    streak = gaussian_blur_pil(out, 4.2, periodic_x=periodic_x)
    out = np.clip(out * 0.55 + (out - streak * 0.42) * 0.65, 0.0, 1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float64)


def build_hii_near_band_placement_score(
    disk_weight: np.ndarray,
    star_formation: np.ndarray,
    height: int,
    *,
    band_lat_sigma: float = 0.12,
    band_weight: float = 0.80,
) -> np.ndarray:
    """Placement weight for H II: strongest on/near the galactic band when disk structure exists."""
    h = int(height)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    sf = np.clip(np.asarray(star_formation, dtype=np.float64), 0.0, 1.0)
    sig = max(float(band_lat_sigma), 0.05)
    on_plane = np.exp(-((yy / sig) ** 2))
    near_plane = np.exp(-((yy / (sig * 2.35)) ** 2))
    in_disk = dw**0.74
    halo = np.clip(near_plane * (1.0 - on_plane * 0.50), 0.0, 1.0) * in_disk * 0.58
    bw = float(np.clip(band_weight, 0.35, 1.0))
    return np.clip(
        on_plane * (0.44 + 0.56 * in_disk) * bw + halo + sf * 0.14,
        0.0,
        1.0,
    ).astype(np.float64)


def build_off_band_mask(
    disk_weight: np.ndarray,
    vertical_extent: np.ndarray | None,
    height: int,
    *,
    band_lat_sigma: float = 0.12,
    decouple_strength: float = 1.0,
    use_vertical_extent: bool = True,
) -> np.ndarray:
    """Where H II / red emission lives off the galactic band (not structure_survival host)."""
    s = float(np.clip(decouple_strength, 0.0, 1.25))
    if s < 1e-6:
        return np.zeros_like(disk_weight, dtype=np.float64)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    yy = np.linspace(-1.0, 1.0, int(height), dtype=np.float64)[:, None]
    sig = max(float(band_lat_sigma), 0.04)
    lat_band = np.exp(-((yy / sig) ** 2))
    off_lat = np.clip(1.0 - lat_band**0.82, 0.0, 1.0)
    off_disk = np.clip(1.0 - dw**0.62, 0.0, 1.0)
    off = np.clip(off_lat * (0.58 + 0.42 * off_disk), 0.0, 1.0)
    if use_vertical_extent and vertical_extent is not None:
        ve = np.clip(np.asarray(vertical_extent, dtype=np.float64), 0.0, 1.0)
        off = np.clip(np.maximum(off, ve * off_lat * 0.18), 0.0, 1.0)
    return np.clip(off**1.18 * s, 0.0, 1.0).astype(np.float64)


_SCENE_RED_COLORS: tuple[np.ndarray, ...] = (
    np.array([1.44, 0.22, 0.14], dtype=np.float64),
    np.array([1.08, 0.36, 0.24], dtype=np.float64),
    np.array([0.90, 0.50, 0.34], dtype=np.float64),
)


def build_band_hii_patches(
    rng: np.random.Generator,
    height: int,
    width: int,
    disk_weight: np.ndarray,
    star_formation: np.ndarray,
    *,
    patch_count: int = 2,
    strength: float = 1.0,
    periodic_x: bool = True,
    hii_seed: int = 0,
) -> np.ndarray:
    """In-band red H II patches along the galactic disk (visible in final grade)."""
    h, w = int(height), int(width)
    s = float(np.clip(strength, 0.0, 3.5))
    n = int(np.clip(patch_count, 0, 5))
    if n < 1 or s < 1e-6:
        return np.zeros((h, w, 3), dtype=np.float64)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    sf = np.clip(np.asarray(star_formation, dtype=np.float64), 0.0, 1.0)
    host = soften_band_envelope(dw, (h, w), periodic_x=periodic_x, lat_blur_sigma=14.0, power=0.56)
    score = np.clip(dw**0.72 * sf**0.85 * host, 0.0, 1.0)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    scale = float(max(h, w))
    float(np.clip(scale * 0.08, 14.0, 100.0))
    regional = build_hii_emission_hierarchy(
        np.clip(score * host, 0.0, 1.0),
        derive_nebula_rng(hii_seed, "band_hii_field"),
        periodic_x=periodic_x,
        disk_weight=dw,
        strength=0.82,
    )
    layer = np.zeros((h, w, 3), dtype=np.float64)
    flat = regional.ravel()
    score_flat = score.ravel()
    for i in range(n):
        spot_rng = derive_nebula_rng(hii_seed, "band_hii", i)
        cand = np.flatnonzero((flat > 0.14) & (score_flat > 0.22))
        if cand.size < 1:
            cand = np.flatnonzero(flat > 0.08)
        if cand.size < 1:
            cand = np.flatnonzero(score_flat > 0.12)
        if cand.size < 1:
            continue
        wts = flat[cand].astype(np.float64)
        wts = wts / (float(wts.sum()) + 1e-12)
        pick = int(spot_rng.choice(cand, p=wts))
        iy, ix = divmod(pick, w)
        cy = float(yy[iy, 0])
        cx = float(xx[0, ix])
        wy = float(spot_rng.uniform(0.05, 0.11))
        wx = float(spot_rng.uniform(0.06, 0.12))
        support = np.clip(score * host * float(spot_rng.uniform(0.88, 1.0)), 0.0, 1.0)
        cloud = build_turbulent_hii_emission_cloud(
            spot_rng,
            h,
            w,
            center_y=cy,
            center_x=cx,
            extent_y=wy,
            extent_x=wx,
            support_mask=support,
            periodic_x=periodic_x,
        )
        color = _SCENE_RED_COLORS[(i + 1) % len(_SCENE_RED_COLORS)]
        amp = (0.62 + 0.24 * s) * float(spot_rng.uniform(1.0, 1.22))
        layer = np.maximum(layer, cloud[..., np.newaxis] * color * amp)
    return np.clip(layer, 0.0, 1.65).astype(np.float64)


def build_scene_red_hii_spots(
    rng: np.random.Generator,
    height: int,
    width: int,
    disk_weight: np.ndarray,
    star_formation: np.ndarray,
    *,
    spot_count: int = 2,
    strength: float = 1.0,
    band_lat_sigma: float = 0.12,
    periodic_x: bool = True,
    hii_seed: int = 0,
) -> np.ndarray:
    """Extra compact red H II nebulae (band + off-band), away from equirect poles."""
    h, w = int(height), int(width)
    s = float(np.clip(strength, 0.0, 3.0))
    n = int(np.clip(spot_count, 0, 6))
    if n < 1 or s < 1e-6:
        return np.zeros((h, w, 3), dtype=np.float64)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    sf = np.clip(np.asarray(star_formation, dtype=np.float64), 0.0, 1.0)
    pole = latitude_plane_gate(h, sigma=0.50, power=1.06)
    if pole.ndim == 1:
        pole = pole.reshape(h, 1)
    pole2d = np.broadcast_to(pole, (h, w))
    score = build_hii_near_band_placement_score(
        dw, sf, h, band_lat_sigma=band_lat_sigma, band_weight=0.82
    ) * pole2d
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    scale = float(max(h, w))
    float(np.clip(scale * 0.09, 16.0, 120.0))
    layer = np.zeros((h, w, 3), dtype=np.float64)
    flat = score.ravel()
    for i in range(n):
        spot_rng = derive_nebula_rng(hii_seed, "scene_red", i)
        cand = np.flatnonzero(flat > 0.12)
        if cand.size < 1:
            row_probs = pole[:, 0]
            row_probs = row_probs / (float(row_probs.sum()) + 1e-12)
            iy = int(spot_rng.choice(h, p=row_probs))
            ix = int(spot_rng.integers(0, w))
        else:
            wts = flat[cand].astype(np.float64)
            wts = wts / (float(wts.sum()) + 1e-12)
            pick = int(spot_rng.choice(cand, p=wts))
            iy, ix = divmod(pick, w)
        cy = float(yy[iy, 0])
        cx = float(xx[0, ix])
        wy = float(spot_rng.uniform(0.08, 0.15))
        wx = float(spot_rng.uniform(0.07, 0.14))
        support = np.clip(score * float(spot_rng.uniform(0.75, 1.0)), 0.0, 1.0)
        cloud = build_turbulent_hii_emission_cloud(
            spot_rng,
            h,
            w,
            center_y=cy,
            center_x=cx,
            extent_y=wy,
            extent_x=wx,
            support_mask=support,
            periodic_x=periodic_x,
        )
        color = _SCENE_RED_COLORS[i % len(_SCENE_RED_COLORS)]
        amp = (0.40 + 0.18 * s) * float(spot_rng.uniform(0.95, 1.20))
        layer = np.maximum(layer, cloud[..., np.newaxis] * color * amp)
    return np.clip(layer, 0.0, 1.65).astype(np.float64)


def localize_emission_clouds(
    field: np.ndarray,
    support_mask: np.ndarray,
    *,
    periodic_x: bool,
    peak_percentile: float = 78.0,
    tail_floor: float = 0.08,
) -> np.ndarray:
    """Keep emission in compact clouds, not a frame-filling radiation wash."""
    f = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
    m = np.clip(np.asarray(support_mask, dtype=np.float64), 0.0, 1.0)
    h, w = f.shape
    if m.shape != (h, w):
        if m.ndim == 2 and m.shape[1] == 1:
            m = np.broadcast_to(m, (h, w))
        elif m.size == h:
            m = np.broadcast_to(m.reshape(h, 1), (h, w))
        else:
            m = np.broadcast_to(m, (h, w))
    scale = float(max(h, w))
    soft = gaussian_blur_pil(f, float(np.clip(scale * 0.028, 2.0, 36.0)), periodic_x=periodic_x)
    peak = np.clip(f - soft * 0.78, 0.0, 1.0) ** 1.12
    out = np.clip(f * m * (0.14 + 0.86 * peak), 0.0, 1.0)
    active = out[m > 0.05]
    if active.size > 64:
        lo = float(np.percentile(active, float(np.clip(peak_percentile, 55.0, 94.0))))
        tf = float(np.clip(tail_floor, 0.0, 0.25))
        out = np.where(out >= lo, out, out * tf)
    return np.clip(out**1.10, 0.0, 1.0).astype(np.float64)


def build_gold_population_field(
    star_formation: np.ndarray,
    stellar_density: np.ndarray,
    latent_turb: np.ndarray,
    latent_ridge: np.ndarray,
    disk_weight: np.ndarray,
    structure_survival: np.ndarray,
    *,
    patchiness: float = 1.28,
) -> np.ndarray:
    """Regional weight for gold bias: high = older population, low = young (no gold wash)."""
    from starsky_gen.galactic_structure import local_field_variance_stretch

    sf = np.clip(np.asarray(star_formation, dtype=np.float64), 0.0, 1.0)
    g = np.clip(np.asarray(stellar_density, dtype=np.float64), 0.0, 1.0)
    turb = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
    ridge = np.clip(np.asarray(latent_ridge, dtype=np.float64), 0.0, 1.0)
    ridge_old = np.clip(1.0 - np.abs(ridge * 2.0 - 1.0), 0.0, 1.0) ** 1.2
    mature = (1.0 - sf) ** 1.42 * (0.32 + 0.68 * g) + ridge_old * 0.38
    active_young = sf**1.18 * (0.52 + 0.48 * (1.0 - g * 0.58))
    patch = 0.40 + 0.60 * turb
    raw = np.clip(mature * patch - active_young * 0.98, 0.0, 1.0) ** 0.74
    legacy = np.clip((1.0 - sf) ** 1.55 * g**0.88 * (0.38 + 0.62 * ridge_old), 0.0, 1.0)
    raw = np.clip(raw + legacy * 0.34, 0.0, 1.0)
    raw = local_field_variance_stretch(
        raw, variance=float(np.clip(patchiness, 0.6, 2.0)) * 0.92, floor=0.0, ceiling=1.0
    )
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    # Prefer midplane density so gold stays in the warm core, not shoulder islands.
    dens = np.clip(np.asarray(stellar_density, dtype=np.float64), 0.0, 1.0)
    raw = np.clip(0.28 * raw + 0.72 * dens * (0.22 + 0.78 * raw), 0.0, 1.0)
    # Only normalize inside the true midplane (not the soft shoulders).
    band_mask = dw > 0.55
    gold = raw.copy()
    if bool(np.any(band_mask)):
        sub = gold[band_mask]
        lo = float(np.percentile(sub, 22.0))
        hi = float(np.percentile(sub, 94.0))
        if hi > lo + 1e-8:
            sub = np.clip((sub - lo) / (hi - lo), 0.0, 1.0) ** 0.95
        gold[band_mask] = sub
    gold = np.where(band_mask, gold, gold * 0.04)
    # Razor disk + latitude falloff — disk_weight alone is still thick on visual shoulders.
    lat = midplane_gold_lat_gate(int(dw.shape[0]), sigma=0.092, power=1.95)
    gold = np.clip(gold * (dw**3.85) * lat, 0.0, 1.0)
    gold = np.clip(gold / max(float(np.percentile(gold, 99.5)), 1e-6), 0.0, 1.0)
    gold = np.clip(gold, 0.0, 1.0) ** 1.35
    return gold.astype(np.float64)


def _field_gradient_magnitude(field: np.ndarray, *, periodic_x: bool) -> np.ndarray:
    """|∇field| with periodic longitude when requested."""
    ext = np.asarray(field, dtype=np.float64)
    if periodic_x:
        gx = 0.5 * (np.roll(ext, -1, axis=1) - np.roll(ext, 1, axis=1))
    else:
        gx = np.gradient(ext, axis=1)
    gy = np.gradient(ext, axis=0)
    return np.sqrt(np.clip(gx, -1e6, 1e6) ** 2 + np.clip(gy, -1e6, 1e6) ** 2 + 1e-12)


def _normalize_gradient_magnitude(grad: np.ndarray) -> np.ndarray:
    g = np.asarray(grad, dtype=np.float64)
    g93 = float(np.quantile(g, 0.93))
    return np.clip(g / (g93 + 1e-6), 0.0, 1.0)


def _independent_turbulence_fields(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decorrelated medium (ridged), small (fbm), fine (ridged) — not blur copies of body."""
    h, w = int(height), int(width)
    ch_m, cw_m = max(4, h // 40), max(6, w // 22)
    ch_s, cw_s = max(6, h // 24), max(8, w // 14)
    ch_f, cw_f = max(8, h // 16), max(12, w // 10)
    medium = ridged_fbm2d(rng, ch_m, cw_m, base_scale=0.14, octaves=4, periodic_x=periodic_x)
    medium = _resize_bilinear(medium, h, w, periodic_x=periodic_x)
    small = fbm2d(rng, ch_s, cw_s, base_scale=0.26, octaves=5, periodic_x=periodic_x)
    small = _resize_bilinear(small, h, w, periodic_x=periodic_x)
    fine = ridged_fbm2d(rng, ch_f, cw_f, base_scale=0.38, octaves=5, periodic_x=periodic_x)
    fine = _resize_bilinear(fine, h, w, periodic_x=periodic_x)
    return (
        np.clip(medium, 0.0, 1.0),
        np.clip(small, 0.0, 1.0),
        np.clip(fine, 0.0, 1.0),
    )


def blend_competitive_scale_hierarchy(
    mega: np.ndarray,
    medium: np.ndarray,
    small: np.ndarray,
    fine: np.ndarray,
    *,
    periodic_x: bool,
    w_mega: float = 0.36,
    w_medium: float = 0.28,
    w_small: float = 0.22,
    w_fine: float = 0.14,
    competition_strength: float = 1.0,
    turbulence_weight: float = 1.0,
) -> np.ndarray:
    """Cooperative per-scale sum + competition (medium vs mega, small vs medium, fine at edges)."""
    c = float(np.clip(competition_strength, 0.0, 1.35))
    tw = float(np.clip(turbulence_weight, 0.35, 2.0))
    c_eff = c * (0.70 + 0.30 * tw)
    mega = np.clip(np.asarray(mega, dtype=np.float64), 0.0, 1.0)
    medium = np.clip(np.asarray(medium, dtype=np.float64), 0.0, 1.0)
    small = np.clip(np.asarray(small, dtype=np.float64), 0.0, 1.0)
    fine = np.clip(np.asarray(fine, dtype=np.float64), 0.0, 1.0)

    med_over_mega = np.clip((medium - mega + 0.06) / 0.44, 0.0, 1.0) ** 1.18
    mega_eff = np.clip(mega * (1.0 - c_eff * 0.64 * med_over_mega), 0.0, 1.0)

    mix = np.clip(
        mega_eff * w_mega + medium * w_medium + small * w_small + fine * w_fine,
        0.0,
        1.0,
    )
    mix = np.clip(mix - c_eff * 0.26 * mega * med_over_mega, 0.0, 1.0)

    frag = np.clip((small - medium) / 0.40, -0.42, 0.42)
    mix = np.clip(mix + c_eff * 0.28 * np.maximum(frag, 0.0), 0.0, 1.0)
    mix = np.clip(mix - c_eff * 0.11 * np.maximum(-frag, 0.0) * medium, 0.0, 1.0)

    bound = _normalize_gradient_magnitude(
        _field_gradient_magnitude(0.56 * small + 0.44 * medium, periodic_x=periodic_x)
    )
    mix = np.clip(mix + c_eff * 0.17 * fine * bound, 0.0, 1.0)
    mix = np.clip(mix - c_eff * 0.08 * fine * (1.0 - bound) * medium**1.04, 0.0, 1.0)

    mega_grad = _normalize_gradient_magnitude(
        _field_gradient_magnitude(mega_eff, periodic_x=periodic_x)
    )
    shear = 0.14 * (0.86 + 0.14 * tw)
    mix = np.clip(mix * (1.0 - shear * c_eff * mega_grad), 0.0, 1.0)
    return mix.astype(np.float64)


def build_ism_scale_hierarchy(
    body: np.ndarray,
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    strength: float,
    periodic_x: bool,
    turbulence_weight: float = 1.0,
    channel_salt: int = 0,
) -> np.ndarray:
    """Constructive four-scale ISM: broad disk + independent scales + shear/fragmentation."""
    s = float(np.clip(strength, 0.0, 1.35))
    if s < 1e-6:
        return np.clip(body, 0.0, 1.0)
    tw = float(np.clip(turbulence_weight, 0.35, 2.0))
    b = np.clip(np.asarray(body, dtype=np.float64), 0.0, 1.0)
    h, w = int(height), int(width)
    scale = float(max(h, w))
    mega_sig = float(np.clip(scale * 0.085, 14.0, 120.0))
    med_sig = float(np.clip(scale * 0.028, 4.5, 42.0))
    small_sig = float(np.clip(scale * 0.009, 1.2, 14.0))
    fine_sig = float(np.clip(scale * 0.0022, 0.35, 3.5))
    salt = int(channel_salt) * 104729
    turb_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + salt + 17)
    med_n, small_n, fine_n = _independent_turbulence_fields(
        turb_rng, h, w, periodic_x=periodic_x
    )
    mega = gaussian_blur_pil(b, mega_sig, periodic_x=periodic_x)
    ch, cw = max(2, h // 52), max(2, w // 28)
    env_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + salt + 31)
    env = fbm2d(env_rng, ch, cw, base_scale=0.11, octaves=3, periodic_x=periodic_x)
    env = _resize_bilinear(env, h, w, periodic_x=periodic_x)
    env = gaussian_blur_pil(env, mega_sig * 0.35, periodic_x=periodic_x)
    mega = np.clip(mega * 0.58 + env * 0.42, 0.0, 1.0)
    med_blur = gaussian_blur_pil(b, med_sig, periodic_x=periodic_x)
    medium = np.clip(med_n * (0.55 + 0.45 * b) + med_blur * 0.12, 0.0, 1.0)
    small_blur = gaussian_blur_pil(b, small_sig, periodic_x=periodic_x)
    small = np.clip(small_n * (0.62 + 0.38 * small_blur), 0.0, 1.0)
    fine_hp = np.clip(b - gaussian_blur_pil(b, fine_sig, periodic_x=periodic_x), 0.0, 1.0)
    fine = np.clip(fine_n * 0.65 + fine_hp * 0.35, 0.0, 1.0)
    tw_boost = float(np.clip((tw - 1.0) * 0.35, 0.0, 0.40))
    mix = blend_competitive_scale_hierarchy(
        mega,
        medium,
        small,
        fine,
        periodic_x=periodic_x,
        w_mega=0.34 - tw_boost * 0.03,
        w_medium=0.28,
        w_small=0.24 + tw_boost * 0.03,
        w_fine=0.14 + tw_boost * 0.02,
        competition_strength=s,
        turbulence_weight=tw,
    )
    detail = np.clip(np.sqrt(np.clip(small * fine, 0.0, 1.0)), 0.0, 1.0)
    mix = np.clip(mix + detail * (0.04 + 0.03 * tw_boost), 0.0, 1.0)
    lo, hi = float(np.percentile(mix, 8.0)), float(np.percentile(mix, 92.0))
    if hi > lo + 1e-8:
        mix = np.clip((mix - lo) / (hi - lo), 0.0, 1.0) ** 0.90
    body_keep = float(np.clip((1.0 - s) * 0.20, 0.0, 0.22))
    return np.clip(b * body_keep + mix * s, 0.0, 1.0).astype(np.float64)


def build_ism_detail_layer(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool,
    channel_salt: int = 0,
) -> np.ndarray:
    """Display-resolution small/fine layer (independent of smooth morphology maps)."""
    h, w = int(height), int(width)
    salt = int(channel_salt) * 104729
    detail_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + salt + 509)
    _, small_n, fine_n = _independent_turbulence_fields(detail_rng, h, w, periodic_x=periodic_x)
    ch, cw = max(8, h // 20), max(12, w // 12)
    micro = ridged_fbm2d(
        detail_rng, ch, cw, base_scale=0.42, octaves=4, periodic_x=periodic_x
    )
    micro = _resize_bilinear(micro, h, w, periodic_x=periodic_x)
    return np.clip(small_n * 0.44 + fine_n * 0.38 + micro * 0.18, 0.0, 1.0).astype(np.float64)


def reinject_cloud_microstructure(
    field: np.ndarray,
    *,
    periodic_x: bool,
    detail_mix: float = 0.48,
    rng: np.random.Generator | None = None,
    channel_salt: int = 0,
) -> np.ndarray:
    """Independent small/fine breakup on top of host (not blur echoes of the same lobe)."""
    dm = float(np.clip(detail_mix, 0.0, 0.92))
    if dm < 1e-6:
        return np.clip(field, 0.0, 1.0)
    b = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
    h, w = b.shape
    scale = float(max(h, w))
    med_sig = float(np.clip(scale * 0.028, 4.5, 42.0))
    small_sig = float(np.clip(scale * 0.009, 1.2, 14.0))
    fine_sig = float(np.clip(scale * 0.0022, 0.35, 3.5))
    if rng is not None:
        detail_rng = np.random.default_rng(
            int(rng.integers(0, 2**31 - 1)) + int(channel_salt) * 7919 + 9031
        )
        _, small_n, fine_n = _independent_turbulence_fields(
            detail_rng, h, w, periodic_x=periodic_x
        )
        small = np.clip(small_n * (0.55 + 0.45 * gaussian_blur_pil(b, small_sig, periodic_x=periodic_x)), 0.0, 1.0)
        fine = np.clip(
            fine_n * 0.62
            + np.clip(b - gaussian_blur_pil(b, fine_sig, periodic_x=periodic_x), 0.0, 1.0) * 0.38,
            0.0,
            1.0,
        )
        medium = np.clip(
            ridged_fbm2d(
                detail_rng,
                max(4, h // 36),
                max(6, w // 20),
                base_scale=0.16,
                octaves=3,
                periodic_x=periodic_x,
            ),
            0.0,
            1.0,
        )
        medium = _resize_bilinear(medium, h, w, periodic_x=periodic_x)
        medium = np.clip(medium * (0.5 + 0.5 * b) + gaussian_blur_pil(b, med_sig, periodic_x=periodic_x) * 0.12, 0.0, 1.0)
    else:
        medium = gaussian_blur_pil(b, med_sig, periodic_x=periodic_x)
        small = gaussian_blur_pil(b, small_sig, periodic_x=periodic_x)
        fine = np.clip(b - gaussian_blur_pil(b, fine_sig, periodic_x=periodic_x), 0.0, 1.0)
    detail = np.clip(small * 0.46 + fine * 0.38 + medium * 0.16, 0.0, 1.0)
    detail_hi = np.clip(
        detail - gaussian_blur_pil(detail, fine_sig * 1.8, periodic_x=periodic_x), 0.0, 1.0
    )
    disagree = np.clip(1.0 - np.abs(b - gaussian_blur_pil(b, med_sig, periodic_x=periodic_x)) * 2.2, 0.0, 1.0)
    amp = dm * (0.72 + 0.28 * disagree)
    return np.clip(b + (detail * 0.32 + detail_hi * 0.78) * amp * 1.22, 0.0, 1.0).astype(np.float64)


def build_cloud_texture_field(
    base: np.ndarray,
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    ridge: np.ndarray | None = None,
    turb: np.ndarray | None = None,
    periodic_x: bool = True,
) -> np.ndarray:
    """Morphology maps + procedural breakup before ISM scale hierarchy."""
    b = np.clip(np.asarray(base, dtype=np.float64), 0.0, 1.0)
    h, w = int(height), int(width)
    ch, cw = max(4, h // 36), max(6, w // 28)
    n1 = fbm2d(rng, ch, cw, base_scale=0.14, octaves=5, periodic_x=periodic_x)
    n1 = _resize_bilinear(n1, h, w, periodic_x=periodic_x)
    n2 = ridged_fbm2d(rng, max(4, ch // 2), max(6, cw // 2), base_scale=0.20, octaves=4, periodic_x=periodic_x)
    n2 = _resize_bilinear(n2, h, w, periodic_x=periodic_x)
    n3 = fbm2d(rng, max(6, ch), max(8, cw), base_scale=0.32, octaves=4, periodic_x=periodic_x)
    n3 = _resize_bilinear(n3, h, w, periodic_x=periodic_x)
    out = np.clip(b * (0.54 + 0.46 * n1) + n2 * 0.28 + n3 * 0.16, 0.0, 1.0)
    if turb is not None:
        t = np.clip(np.asarray(turb, dtype=np.float64), 0.0, 1.0)
        t_hi = np.clip(t - gaussian_blur_pil(t, 2.2, periodic_x=periodic_x) * 0.72, 0.0, 1.0) ** 1.08
        out = np.clip(out * (0.58 + 0.42 * t) + t_hi * 0.14, 0.0, 1.0)
    if ridge is not None:
        r = np.clip(np.asarray(ridge, dtype=np.float64), 0.0, 1.0)
        ridge_dark = np.clip(1.0 - np.abs(r * 2.0 - 1.0), 0.0, 1.0) ** 1.1
        out = np.clip(out * (0.74 + 0.26 * ridge_dark), 0.0, 1.0)
    return out.astype(np.float64)


def build_morphology_ism_layers(
    galactic,
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    hierarchy_strength: float = 0.96,
    white_brightness: float = 1.22,
    detail_strength: float = 1.35,
    periodic_x: bool = True,
    texture_seed: int | None = None,
) -> MorphologyIsmLayers:
    """Split ISM: brown/black absorption, gold/amber emit, red H II — plus puff punch mask."""
    from starsky_gen.galactic_structure import local_field_variance_stretch

    h, w = int(height), int(width)
    sf = np.clip(galactic.star_formation, 0.0, 1.0)
    g = np.clip(galactic.stellar_density, 0.0, 1.0)
    u = np.clip(galactic.unresolved_prior, 0.0, 1.0)
    dust = np.clip(galactic.dust_absorption, 0.0, 1.0)
    turb = np.clip(galactic.latent_turb, 0.0, 1.0)
    ridge = np.clip(galactic.latent_ridge, 0.0, 1.0)
    host = np.clip(galactic.structure_survival, 0.0, 1.35)
    hs = float(np.clip(hierarchy_strength, 0.0, 1.25))
    if texture_seed is not None:
        tex0 = derive_nebula_rng(texture_seed, "ism_tex0")
        tex1 = derive_nebula_rng(texture_seed, "ism_tex1")
        tex2 = derive_nebula_rng(texture_seed, "ism_tex2")
    else:
        tex0 = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + 11)
        tex1 = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + 29)
        tex2 = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + 47)
    body = np.clip(0.38 * sf + 0.28 * g + 0.34 * u, 0.0, 1.0)
    body = build_cloud_texture_field(body, tex0, h, w, ridge=ridge, turb=turb, periodic_x=periodic_x)
    body = build_ism_scale_hierarchy(
        body,
        tex0,
        h,
        w,
        strength=hs * 0.78,
        periodic_x=periodic_x,
        turbulence_weight=1.12,
        channel_salt=0,
    )
    body = np.clip(body**0.86, 0.0, 1.0) * host
    white_base = build_cloud_texture_field(
        np.clip(sf * g * (1.0 - dust * 0.72), 0.0, 1.0) * host,
        tex1,
        h,
        w,
        ridge=ridge,
        turb=turb,
        periodic_x=periodic_x,
    )
    white_base = local_field_variance_stretch(white_base, variance=1.42, floor=0.02, ceiling=1.0)
    white = reinject_cloud_microstructure(
        build_ism_scale_hierarchy(
            white_base,
            tex1,
            h,
            w,
            strength=hs,
            periodic_x=periodic_x,
            turbulence_weight=1.85,
            channel_salt=1,
        ),
        periodic_x=periodic_x,
        detail_mix=0.90,
        rng=rng,
        channel_salt=1,
    )
    # Soft gold body only — no microstructure reinject (avoids jagged shoulder flecks).
    gold = build_ism_scale_hierarchy(
        build_cloud_texture_field(
            np.clip(sf * (0.55 + 0.45 * g), 0.0, 1.0) * host,
            tex2,
            h,
            w,
            turb=turb,
            periodic_x=periodic_x,
        ),
        tex2,
        h,
        w,
        strength=hs * 0.78,
        periodic_x=periodic_x,
        turbulence_weight=1.05,
        channel_salt=2,
    )
    black_base = local_field_variance_stretch(
        np.clip(dust * (0.62 + 0.38 * galactic.obliteration_mask), 0.0, 1.0) * host,
        variance=1.52,
        floor=0.03,
        ceiling=1.0,
    )
    black = reinject_cloud_microstructure(
        build_ism_scale_hierarchy(
            build_cloud_texture_field(black_base, tex0, h, w, ridge=ridge, periodic_x=periodic_x),
            tex0,
            h,
            w,
            strength=hs * 1.10,
            periodic_x=periodic_x,
            turbulence_weight=1.92,
            channel_salt=3,
        ),
        periodic_x=periodic_x,
        detail_mix=0.92,
        rng=rng,
        channel_salt=3,
    )
    wb = float(np.clip(white_brightness, 0.7, 1.8))
    white_rgb = np.clip(
        white[..., np.newaxis] * np.array([0.98, 0.96, 0.88], dtype=np.float64) * (0.62 * wb),
        0.0,
        1.0,
    )
    gp = np.clip(galactic.gold_population_weight, 0.0, 1.0)
    gp_soft = gaussian_blur_pil(
        gp, float(np.clip(max(h, w) * 0.018, 2.0, 16.0)), periodic_x=periodic_x
    )
    dens_for_gold = gaussian_blur_pil(
        np.clip(g, 0.0, 1.0), float(np.clip(max(h, w) * 0.014, 1.5, 12.0)), periodic_x=periodic_x
    )
    gp_soft = np.clip(0.55 * dens_for_gold + 0.45 * gp_soft, 0.0, 1.0)
    gp_soft = gp_soft / max(float(np.percentile(gp_soft, 99.5)), 1e-6)
    gp_soft = np.clip(gp_soft, 0.0, 1.0) ** 1.20
    # Stronger pre-emit blur so gold reads as warm midplane core, not HF streaks.
    gold_soft = gaussian_blur_pil(
        gold, float(np.clip(max(h, w) * 0.024, 2.5, 22.0)), periodic_x=periodic_x
    )
    # Soft body only (extra blur) — body HF must not become midplane gold dashes.
    body_soft = gaussian_blur_pil(
        body, float(np.clip(max(h, w) * 0.018, 2.0, 16.0)), periodic_x=periodic_x
    )
    gold_emit_rgb = np.clip(
        gold_soft[..., np.newaxis] * np.array([1.02, 0.86, 0.48], dtype=np.float64) * 0.12
        + body_soft[..., np.newaxis] * np.array([0.72, 0.64, 0.44], dtype=np.float64) * 0.05,
        0.0,
        1.0,
    )
    # Razor midplane gate — shoulder latitudes must go to ~0.
    soft_dw = soften_band_envelope(galactic.disk_weight, (h, w), periodic_x=periodic_x)
    dw_raw = np.clip(np.asarray(galactic.disk_weight, dtype=np.float64), 0.0, 1.0)
    lat_gate = midplane_gold_lat_gate(h, sigma=0.090, power=2.05)
    dw_gate = np.clip((dw_raw**4.0) * (soft_dw**1.15) * lat_gate, 0.0, 1.0)
    # Modest gold spine boost only where disk is truly high (warm core, no shoulders).
    spine = np.clip(dw_raw**4.2 * lat_gate, 0.0, 1.0)
    gold_emit_rgb = np.clip(
        gold_emit_rgb
        * (0.12 + 0.88 * gp_soft[..., np.newaxis])
        * dw_gate[..., np.newaxis]
        * (0.72 + 0.72 * spine[..., np.newaxis]),
        0.0,
        1.0,
    )
    # Extra boost only in the central ~35% of the tightened gold weight (spine core).
    gold_w = np.clip(gp_soft * lat_gate, 0.0, 1.0)
    nz = gold_w > 1e-5
    core_mask = np.zeros_like(gold_w)
    if bool(np.any(nz)):
        thr_core = float(np.percentile(gold_w[nz], 65.0))
        core_mask = (gold_w >= thr_core).astype(np.float64)
    gold_emit_rgb = np.clip(
        gold_emit_rgb * (1.0 + 0.22 * core_mask[..., np.newaxis] * gold_w[..., np.newaxis]),
        0.0,
        1.0,
    )
    absorption_rgb = np.clip(
        black[..., np.newaxis] * np.array([0.26, 0.24, 0.22], dtype=np.float64) * 0.88,
        0.0,
        1.0,
    )
    hii_host = np.clip(sf * (0.62 + 0.48 * g) * host * (1.0 - dust * 0.58), 0.0, 1.0) ** 1.08
    hii_tex = texture_seed if texture_seed is not None else int(rng.integers(0, 2**31 - 1))
    hii_frag = build_hii_emission_hierarchy(
        hii_host,
        derive_nebula_rng(hii_tex, "hii_morph"),
        periodic_x=periodic_x,
        disk_weight=galactic.disk_weight,
        strength=1.05,
    )
    # Light contrast + soft blur so red/orange H II reads as intentional clumps.
    dens_soft = gaussian_blur_pil(
        np.clip(g, 0.0, 1.0), float(np.clip(max(h, w) * 0.012, 1.2, 12.0)), periodic_x=periodic_x
    )
    red_dens = np.clip(hii_frag * dens_soft * (1.0 - 0.40 * gp_soft), 0.0, 1.0)
    red_gate = red_dens > 1e-5
    if bool(np.any(red_gate)):
        thr = float(np.percentile(red_dens[red_gate], 70.0))
        red_clump = np.clip((red_dens - thr) / max(1.0 - thr, 0.08), 0.0, 1.0) ** 1.50
        red_clump = gaussian_blur_pil(
            red_clump, float(np.clip(max(h, w) * 0.014, 1.4, 12.0)), periodic_x=periodic_x
        )
    else:
        red_clump = red_dens
    red_hii_rgb = np.clip(
        red_clump[..., np.newaxis] * np.array([1.42, 0.34, 0.16], dtype=np.float64) * 1.38,
        0.0,
        1.0,
    )
    soft_env = soften_band_envelope(galactic.disk_weight, (h, w), periodic_x=periodic_x)
    from starsky_gen.dust_field import _band_host_gate

    core = _band_host_gate(galactic.disk_weight, (h, w)) > 0.40
    env = np.ones((h, w), dtype=np.float64)
    rim = (~core) & (soft_env < 0.92)
    if bool(np.any(rim)):
        env[rim] = np.clip(0.28 + 0.72 * soft_env[rim], 0.0, 1.0)
    # HF detail on white/absorption only — never sculpt gold into shoulder flecks.
    scale = float(max(h, w))
    fine_sig = float(np.clip(scale * 0.0022, 0.35, 3.5))
    white_hi = np.clip(
        white - gaussian_blur_pil(white, fine_sig, periodic_x=periodic_x), 0.0, 1.0
    )
    black_hi = np.clip(
        black - gaussian_blur_pil(black, fine_sig, periodic_x=periodic_x), 0.0, 1.0
    )
    detail = build_ism_detail_layer(rng, h, w, periodic_x=periodic_x, channel_salt=9)
    detail = np.clip(
        detail * 0.55 + np.sqrt(np.clip(white_hi * black_hi, 0.0, 1.0)) * 0.45,
        0.0,
        1.0,
    )
    ds = float(np.clip(detail_strength, 0.6, 1.6))
    non_gold = np.maximum(white_rgb - absorption_rgb * 0.42, 0.0)
    lu_ng = np.clip(
        non_gold[..., 0] * 0.2126 + non_gold[..., 1] * 0.7152 + non_gold[..., 2] * 0.0722,
        0.0,
        1.0,
    )
    detail_amp = detail[..., np.newaxis] * lu_ng[..., np.newaxis] * (0.14 * ds)
    non_gold = np.maximum(non_gold * (0.68 + 0.32 * detail[..., np.newaxis]) + detail_amp, 0.0)
    puff_ism = build_fine_puff_field(
        rng, h, w, periodic_x=periodic_x, strength=1.0, center_boost=1.0
    )
    puff_mod = np.clip(0.40 + 1.02 * puff_ism**1.10, 0.0, 1.62)
    mod3 = puff_mod[..., np.newaxis]
    # Keep gold soft — do not re-amplify with puff HF (shoulder fleck source).
    red_hii_rgb = np.clip(red_hii_rgb * mod3, 0.0, 1.0)
    white_rgb = np.clip(white_rgb * mod3, 0.0, 1.0)
    absorption_rgb = np.clip(absorption_rgb * mod3, 0.0, 1.0)
    non_gold = np.clip(non_gold * mod3, 0.0, 1.0)
    rgb = np.maximum(non_gold + gold_emit_rgb, 0.0) * env[..., np.newaxis]
    luma = np.clip(rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722, 0.0, 1.0)
    mu = luma[..., np.newaxis]
    rgb = np.clip(rgb * 0.91 + mu * 0.09, 0.0, 1.0)
    punch = morphology_puff_punch_mask(
        np.clip(galactic.dust_absorption_morph, 0.0, 1.0), rng, periodic_x=periodic_x
    )
    return MorphologyIsmLayers(
        absorption_rgb=absorption_rgb.astype(np.float64),
        gold_emit_rgb=gold_emit_rgb.astype(np.float64),
        red_hii_rgb=red_hii_rgb.astype(np.float64),
        white_rgb=white_rgb.astype(np.float64),
        puff_punch_mask=punch.astype(np.float64),
        combined_rgb=rgb.astype(np.float64),
        luma=luma.astype(np.float64),
    )


def build_morphology_ism_rgb(
    galactic,
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    hierarchy_strength: float = 0.96,
    white_brightness: float = 1.22,
    detail_strength: float = 1.35,
    periodic_x: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Morphology-driven ISM RGB + luma (primary gas when dust_primary)."""
    layers = build_morphology_ism_layers(
        galactic,
        rng,
        height,
        width,
        hierarchy_strength=hierarchy_strength,
        white_brightness=white_brightness,
        detail_strength=detail_strength,
        periodic_x=periodic_x,
    )
    return layers.combined_rgb, layers.luma


def build_fine_puff_field(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    periodic_x: bool = True,
    strength: float = 1.0,
    center_boost: float = 1.0,
) -> np.ndarray:
    """Soft cotton-ball cloud density (FBM body + light detail, not ridged cells)."""
    from starsky_gen.procedural_noise import gaussian_blur_pil

    h, w = int(height), int(width)
    s = float(np.clip(strength, 0.0, 1.5))
    if s < 1e-6:
        return np.zeros((h, w), dtype=np.float64)
    scale = float(max(h, w))
    rng_b = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) ^ 0xF10E)
    ch_a, cw_a = max(96, h // 8), max(56, w // 12)
    base = fbm2d(
        rng,
        ch_a,
        cw_a,
        base_scale=0.14,
        octaves=5,
        periodic_x=periodic_x,
        elongate_along_x=1.35,
    )
    base = _resize_bilinear(base, h, w, periodic_x=periodic_x)
    ch_b, cw_b = max(56, h // 12), max(64, w // 10)
    detail = fbm2d(
        rng_b,
        ch_b,
        cw_b,
        base_scale=0.22,
        octaves=4,
        periodic_x=periodic_x,
        elongate_along_x=1.0,
    )
    detail = _resize_bilinear(detail, h, w, periodic_x=periodic_x)
    rng_c = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) ^ 0xAC7E)
    ch_c, cw_c = max(40, h // 14), max(48, w // 10)
    ridge = ridged_fbm2d(
        rng_c,
        ch_c,
        cw_c,
        base_scale=0.38,
        octaves=3,
        periodic_x=periodic_x,
        elongate_along_x=1.0,
    )
    ridge = _resize_bilinear(ridge, h, w, periodic_x=periodic_x)
    ridge_blur = gaussian_blur_pil(
        ridge, float(np.clip(scale * 0.012, 1.0, 10.0)), periodic_x=periodic_x
    )
    ridge_tex = np.clip((ridge - ridge_blur * 0.82) * 1.1, 0.0, 1.0) ** 1.05 * 0.03
    body = np.clip(base * 0.74 + detail * 0.24 + ridge_tex, 0.0, 1.0) ** 0.92
    sig_body = float(np.clip(scale * 0.0085, 1.0, 7.5))
    puffs = gaussian_blur_pil(body, sig_body, periodic_x=periodic_x)
    detail = np.clip(
        body - gaussian_blur_pil(body, sig_body * 0.55, periodic_x=periodic_x),
        0.0,
        1.0,
    ) ** 0.92
    puffs = np.clip(puffs + detail * 0.32, 0.0, 1.0)
    puffs = gaussian_blur_pil(puffs, sig_body * 0.28, periodic_x=periodic_x)
    cb = float(np.clip(center_boost, 0.0, 1.5))
    if cb > 1e-6:
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[np.newaxis, :]
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, np.newaxis]
        core = np.exp(-(xx**2) / 0.16) * np.exp(-((yy * 0.92) ** 2) / 0.11)
        puffs = np.clip(puffs * (0.48 + 0.52 * core**0.85) ** cb, 0.0, 1.0)
    from starsky_gen.dust_field import attenuate_row_comb

    # Light row-comb only — heavy strength shelved clouds into horizontal ribbons.
    puffs = attenuate_row_comb(puffs, None, strength=0.22)
    return np.clip(puffs * s, 0.0, 1.0).astype(np.float64)


def inject_band_cloud_puffs(
    absorption: np.ndarray,
    band_weight: np.ndarray | None,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
    strength: float = 0.36,
    host_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Add tiny cloud pockets to band dust absorption (constructive, not a flat slab)."""
    from starsky_gen.dust_field import _band_host_gate

    a = np.clip(np.asarray(absorption, dtype=np.float64), 0.0, 1.0)
    h, w = a.shape
    if host_weight is not None:
        hw = np.clip(np.asarray(host_weight, dtype=np.float64), 0.0, 1.0)
        if hw.shape != (h, w):
            hw = _resize_bilinear(hw, h, w, periodic_x=periodic_x)
        gate = hw > 0.14
    else:
        gate = _band_host_gate(band_weight, (h, w)) > 0.20
    if not bool(np.any(gate)):
        return a
    puffs = build_fine_puff_field(rng, h, w, periodic_x=periodic_x, strength=1.0)
    s = float(np.clip(strength, 0.0, 0.75))
    from starsky_gen.procedural_noise import gaussian_blur_pil

    scale = float(max(h, w))
    puff_soft = gaussian_blur_pil(
        puffs, float(np.clip(scale * 0.010, 1.0, 9.0)), periodic_x=periodic_x
    )
    peaks = puff_soft**0.88
    valleys = (1.0 - puff_soft) ** 0.92
    out = a.copy()
    sub_a = a[gate]
    sub_p = peaks[gate]
    sub_v = valleys[gate]
    if host_weight is not None:
        sub_w = np.clip(hw[gate], 0.14, 1.0)
        s_loc = s * (0.40 + 0.60 * sub_w)
    else:
        s_loc = s
    out[gate] = np.clip(sub_a * (1.0 - sub_v * 0.12 * s_loc) + sub_p * 0.14 * s_loc, 0.06, 0.82)
    return out.astype(np.float64)


def build_morphology_turbulent_gas_field(
    galactic,
    ext_paint: np.ndarray,
    *,
    periodic_x: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Billowy cloud density along the plane — soft puff blobs, not hard ribbon cutouts."""
    from starsky_gen.dust_field import band_relative_clearance
    from starsky_gen.procedural_noise import gaussian_blur_pil

    dust = np.clip(np.asarray(galactic.dust_absorption_morph, dtype=np.float64), 0.0, 1.0)
    turb = np.clip(np.asarray(galactic.latent_turb, dtype=np.float64), 0.0, 1.0)
    h, w = dust.shape
    scale = float(max(h, w))
    puff_rng = (
        rng
        if rng is not None
        else np.random.default_rng(int(abs(hash((h, w, float(np.mean(dust))))) % (2**31 - 1)))
    )
    # Multi-scale cotton: large billows + medium clumps + fine fluff.
    puff_lo = build_fine_puff_field(
        puff_rng, h, w, periodic_x=periodic_x, strength=1.0, center_boost=0.35
    )
    puff_lo = gaussian_blur_pil(
        puff_lo, float(np.clip(scale * 0.022, 2.5, 18.0)), periodic_x=periodic_x
    )
    rng_mid = np.random.default_rng(int(puff_rng.integers(0, 2**31 - 1)) ^ 0xC10D)
    puff_mid = build_fine_puff_field(
        rng_mid, h, w, periodic_x=periodic_x, strength=1.0, center_boost=0.25
    )
    puff_mid = gaussian_blur_pil(
        puff_mid, float(np.clip(scale * 0.010, 1.2, 10.0)), periodic_x=periodic_x
    )
    rng_hi = np.random.default_rng(int(puff_rng.integers(0, 2**31 - 1)) ^ 0xF155)
    puff_hi = build_fine_puff_field(
        rng_hi, h, w, periodic_x=periodic_x, strength=0.85, center_boost=0.15
    )
    # Mild dust/turb mottle only — lanes/ridges made hard shelves, not clouds.
    med_d = gaussian_blur_pil(
        dust, float(np.clip(scale * 0.014, 1.5, 14.0)), periodic_x=periodic_x
    )
    dust_mottle = np.clip(dust - med_d * 0.70, 0.0, 1.0) ** 0.85
    med_t = gaussian_blur_pil(
        turb, float(np.clip(scale * 0.012, 1.2, 12.0)), periodic_x=periodic_x
    )
    turb_mottle = np.clip(turb - med_t * 0.72, 0.0, 1.0) ** 0.88
    clouds = np.clip(
        puff_lo * 0.42 + puff_mid * 0.34 + puff_hi * 0.18 + dust_mottle * 0.04 + turb_mottle * 0.02,
        0.0,
        1.0,
    )
    # Emphasize clump peaks so midtones aren't a flat wash (still soft, not binary).
    clouds = np.clip(clouds ** 0.78, 0.0, 1.0)
    soft_env = soften_band_envelope(
        galactic.disk_weight, (h, w), periodic_x=periodic_x, lat_blur_sigma=18.0, power=0.45
    )
    clear = band_relative_clearance(
        ext_paint, galactic.disk_weight, min_clear=0.10, power=0.62
    )
    dust_open = np.clip(1.0 - dust * 0.42, 0.0, 1.0) ** 0.80
    # Density modulates clouds; never punches to empty silhouette.
    out = np.clip(
        clouds * (0.62 + 0.38 * clear) * (0.65 + 0.35 * dust_open),
        0.0,
        1.0,
    )
    # Soft stretch in-band for readable clump contrast without flat tops.
    host = soft_env
    on = host > 0.22
    if bool(np.any(on)):
        lo = float(np.percentile(out[on], 12.0))
        hi = float(np.percentile(out[on], 90.0)) + 1e-8
        stretched = np.clip((out - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** 0.82
        out = np.clip(out * (1.0 - 0.62 * host) + stretched * (0.62 * host), 0.0, 1.0)
    # Luminous floor + soft envelope — keep wisps outside the hard disk core.
    body = np.clip(0.18 + 0.82 * out, 0.0, 1.0)
    wisp_env = np.clip(soft_env ** 0.72, 0.0, 1.0)
    out = body * (0.38 + 0.62 * wisp_env)
    # Light feather only — heavy blur erased billowy clump read.
    out = gaussian_blur_pil(
        out, float(np.clip(scale * 0.0035, 0.6, 4.0)), periodic_x=periodic_x
    )
    return np.clip(out, 0.0, 1.0).astype(np.float64)


def diffuse_scale_hierarchy(
    body: np.ndarray,
    *,
    strength: float,
    periodic_x: bool,
    rng: np.random.Generator | None = None,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper around :func:`build_ism_scale_hierarchy`."""
    b = np.clip(np.asarray(body, dtype=np.float64), 0.0, 1.0)
    h, w = b.shape
    r = rng if rng is not None else np.random.default_rng(0)
    return build_ism_scale_hierarchy(
        b, r, h, w, strength=strength, periodic_x=periodic_x, turbulence_weight=1.0
    )


def build_hii_emission_hierarchy(
    host: np.ndarray,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
    disk_weight: np.ndarray | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """H II emission: weak arm basins, strong mid-scale complexes, knots, fine turbulence.

    Avoids mega→fine coherence (smooth hot filaments) by keeping medium/small scales dominant.
    """
    s = float(np.clip(strength, 0.0, 1.35))
    b = np.clip(np.asarray(host, dtype=np.float64), 0.0, 1.0)
    h, w = b.shape
    if s < 1e-6 or float(np.max(b)) < 1e-8:
        return np.zeros((h, w), dtype=np.float64)
    scale = float(max(h, w))
    salt = int(rng.integers(0, 2**31 - 1))

    ch_m, cw_m = max(10, h // 12), max(14, w // 8)
    med_rng = derive_nebula_rng(salt, "hii_complex")
    med_cell = ridged_fbm2d(
        med_rng,
        ch_m,
        cw_m,
        base_scale=0.24,
        octaves=5,
        periodic_x=periodic_x,
        elongate_along_x=1.42,
    )
    med_cell = _resize_bilinear(med_cell, h, w, periodic_x=periodic_x)
    complexes = np.clip(1.0 - np.abs(med_cell * 2.0 - 1.0), 0.0, 1.0) ** 1.42
    complexes = np.clip(complexes * (0.32 + 0.68 * b), 0.0, 1.0)

    knot_rng = derive_nebula_rng(salt, "hii_knot")
    ch_k, cw_k = max(12, h // 20), max(16, w // 12)
    knots_raw = fbm2d(
        knot_rng, ch_k, cw_k, base_scale=0.30, octaves=5, periodic_x=periodic_x
    )
    knots_raw = _resize_bilinear(knots_raw, h, w, periodic_x=periodic_x)
    knots = np.clip(knots_raw * (0.38 + 0.62 * complexes), 0.0, 1.0) ** 1.08

    med_n, small_n, fine_n = _independent_turbulence_fields(
        derive_nebula_rng(salt, "hii_turb"), h, w, periodic_x=periodic_x
    )
    fine = np.clip(fine_n * (0.48 + 0.52 * knots), 0.0, 1.0)

    mega_sig = float(np.clip(scale * 0.10, 18.0, 130.0))
    mega = gaussian_blur_pil(b, mega_sig, periodic_x=periodic_x) ** 1.18
    if disk_weight is not None:
        soft = soften_band_envelope(disk_weight, (h, w), periodic_x=periodic_x, lat_blur_sigma=16.0)
        mega = np.clip(mega * (0.50 + 0.50 * soft), 0.0, 1.0)

    from starsky_gen.dust_field import _blur_x_only_field

    lon_smooth = _blur_x_only_field(complexes, 2.8, periodic_x=periodic_x)
    complexes_hp = np.clip(complexes - lon_smooth * 0.48, 0.0, 1.0) ** 1.06

    mix = blend_competitive_scale_hierarchy(
        mega,
        complexes_hp,
        knots,
        fine,
        periodic_x=periodic_x,
        w_mega=0.08,
        w_medium=0.40,
        w_small=0.32,
        w_fine=0.20,
        competition_strength=s * 0.95,
        turbulence_weight=1.0,
    )
    mix = np.clip(mix * (0.55 + 0.45 * b), 0.0, 1.0)
    active = mix[mix > 0.04]
    if active.size > 128:
        lo = float(np.percentile(active, 58.0))
        mix = np.where(mix >= lo, mix, mix * 0.10)
    lo_p, hi_p = float(np.percentile(mix, 10.0)), float(np.percentile(mix, 93.0))
    if hi_p > lo_p + 1e-8:
        mix = np.clip((mix - lo_p) / (hi_p - lo_p), 0.0, 1.0) ** 0.92
    return np.clip(mix * s, 0.0, 1.0).astype(np.float64)


def build_emission_morphology_field(
    ext_paint: np.ndarray,
    star_formation: np.ndarray,
    structure_survival: np.ndarray,
    dust_absorption: np.ndarray,
    *,
    periodic_x: bool,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Emission host with decorrelated scales (avoids single-scale spherical glow)."""
    clear = np.clip(np.asarray(ext_paint, dtype=np.float64), 0.0, 1.0)
    sf = np.clip(np.asarray(star_formation, dtype=np.float64), 0.0, 1.0)
    surv = np.clip(np.asarray(structure_survival, dtype=np.float64), 0.0, 1.35)
    dust = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    host = np.clip(clear**0.88 * (0.38 + 0.62 * sf) * (0.52 + 0.48 * surv), 0.0, 1.0)
    h, w = host.shape
    scale = float(max(h, w))
    mega_sig = float(np.clip(scale * 0.09, 16.0, 128.0))
    med_sig = float(np.clip(scale * 0.032, 5.0, 48.0))
    fine_sig = float(np.clip(scale * 0.006, 0.8, 6.0))
    mega = gaussian_blur_pil(host, mega_sig, periodic_x=periodic_x)
    r = rng if rng is not None else np.random.default_rng(0)
    med_n, small_n, fine_n = _independent_turbulence_fields(r, h, w, periodic_x=periodic_x)
    medium = np.clip(med_n * (0.50 + 0.50 * host) + gaussian_blur_pil(host, med_sig, periodic_x=periodic_x) * 0.14, 0.0, 1.0)
    small = np.clip(small_n * (0.45 + 0.55 * sf) * (0.62 + 0.38 * (1.0 - dust)), 0.0, 1.0)
    fine = np.clip(
        fine_n * 0.55
        + np.clip(host - gaussian_blur_pil(host, fine_sig, periodic_x=periodic_x), 0.0, 1.0) * 0.45,
        0.0,
        1.0,
    )
    mix = blend_competitive_scale_hierarchy(
        mega,
        medium,
        small,
        fine,
        periodic_x=periodic_x,
        w_mega=0.12,
        w_medium=0.36,
        w_small=0.32,
        w_fine=0.20,
        competition_strength=0.92,
        turbulence_weight=1.0,
    )
    active = mix[mix > 0.05]
    if active.size > 64:
        lo = float(np.percentile(active, 60.0))
        mix = np.where(mix >= lo, mix, mix * 0.12)
    return np.clip(mix * np.clip(0.5 + 0.5 * np.sqrt(small * fine), 0.0, 1.0), 0.0, 1.0).astype(np.float64)


def build_warped_emission_blob(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    center_y: float,
    center_x: float,
    extent_y: float,
    extent_x: float,
    support_mask: np.ndarray,
    periodic_x: bool,
) -> np.ndarray:
    """Regional emission cloud: soft envelope × independent turbulence (not a radial sphere)."""
    h, w = int(height), int(width)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    wy = max(float(extent_y), 0.04)
    wx = max(float(extent_x), 0.04)
    ch, cw = max(6, h // 32), max(8, w // 24)
    warp = fbm2d(rng, ch, cw, base_scale=0.18, octaves=3, periodic_x=periodic_x)
    warp = _resize_bilinear(warp, h, w, periodic_x=periodic_x)
    wy_l = wy * (0.82 + 0.36 * (warp - 0.5))
    wx_l = wx * (0.82 + 0.36 * (1.0 - warp) * 0.5 + 0.18)
    dist = ((yy - center_y) / wy_l) ** 2 + ((xx - center_x) / wx_l) ** 2
    env = np.exp(-(dist**0.78))
    med_n, small_n, _ = _independent_turbulence_fields(rng, h, w, periodic_x=periodic_x)
    turb = np.clip(med_n * 0.42 + small_n * 0.58, 0.0, 1.0)
    m = np.clip(np.asarray(support_mask, dtype=np.float64), 0.0, 1.0)
    return np.clip(env * (0.22 + 0.78 * turb) * m, 0.0, 1.0).astype(np.float64)


def build_turbulent_hii_emission_cloud(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    center_y: float,
    center_x: float,
    extent_y: float,
    extent_x: float,
    support_mask: np.ndarray,
    periodic_x: bool,
) -> np.ndarray:
    """Regional H II complex: hierarchy inside a soft envelope (not one smooth filament)."""
    h, w = int(height), int(width)
    m = np.clip(np.asarray(support_mask, dtype=np.float64), 0.0, 1.0)
    env = build_warped_emission_blob(
        rng,
        h,
        w,
        center_y=center_y,
        center_x=center_x,
        extent_y=extent_y,
        extent_x=extent_x,
        support_mask=m,
        periodic_x=periodic_x,
    )
    body = build_hii_emission_hierarchy(
        np.clip(env * m, 0.0, 1.0),
        rng,
        periodic_x=periodic_x,
        strength=0.92,
    )
    scale = float(max(h, w))
    puff = build_fine_puff_field(
        rng, h, w, periodic_x=periodic_x, strength=1.0, center_boost=0.55
    )
    cavity = np.clip((1.0 - puff) ** 1.08, 0.0, 1.0)
    body = np.clip(body * (1.0 - cavity * 0.35 * body), 0.0, 1.0)
    soft = gaussian_blur_pil(body, float(np.clip(scale * 0.018, 1.5, 22.0)), periodic_x=periodic_x)
    peak = np.clip(body - soft * 0.62, 0.0, 1.0) ** 1.10
    out = np.clip(body * (0.52 + 0.48 * peak), 0.0, 1.0)
    return np.clip(out * m, 0.0, 1.0).astype(np.float64)


def build_core_void_mask(
    disk_weight: np.ndarray,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
    void_strength: float = 0.62,
) -> np.ndarray:
    """Break up solid galactic-core wash with puffy voids and lane structure."""
    s = float(np.clip(void_strength, 0.0, 1.0))
    if s < 1e-6:
        return np.ones_like(disk_weight, dtype=np.float64)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = dw.shape
    soft = soften_band_envelope(dw, (h, w), periodic_x=periodic_x, lat_blur_sigma=11.0, power=0.56)
    core = np.clip(soft**1.08, 0.0, 1.0)
    puff = build_fine_puff_field(
        rng, h, w, periodic_x=periodic_x, strength=1.0, center_boost=1.15
    )
    ch, cw = max(8, h // 26), max(12, w // 18)
    lane = ridged_fbm2d(rng, ch, cw, base_scale=0.15, octaves=4, periodic_x=periodic_x)
    lane = _resize_bilinear(lane, h, w, periodic_x=periodic_x)
    lane = np.clip(1.0 - np.abs(lane * 2.0 - 1.0), 0.0, 1.0) ** 1.12
    voids = np.clip(puff * 0.62 + lane * 0.38, 0.0, 1.0) ** 1.10
    out = np.clip(1.0 - voids * core * s * 0.86, 0.40, 1.0)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    center_bar = np.exp(-((yy / 0.11) ** 2)) * np.exp(-((xx / 0.15) ** 2))
    out = np.clip(out * (1.0 - center_bar * s * 0.38), 0.38, 1.0)
    scale = float(max(h, w))
    lump = gaussian_blur_pil(out, float(np.clip(scale * 0.016, 2.0, 18.0)), periodic_x=periodic_x)
    return np.clip(out * (0.84 + 0.16 * lump), 0.38, 1.0).astype(np.float64)


def build_realistic_core_carve_mask(
    disk_weight: np.ndarray,
    rng: np.random.Generator,
    *,
    periodic_x: bool = True,
    carve_strength: float = 0.72,
) -> np.ndarray:
    """Suppress axis-aligned orange core bar; seed-locked offset bulge + turbulent breakup."""
    s = float(np.clip(carve_strength, 0.0, 1.0))
    if s < 1e-6:
        return np.ones_like(disk_weight, dtype=np.float64)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    h, w = dw.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    gc_x = float(rng.uniform(-0.20, 0.20))
    gc_y = float(rng.uniform(-0.06, 0.06))
    bulge_peak = np.exp(-(((xx - gc_x) / 0.19) ** 2 + ((yy - gc_y) / 0.12) ** 2))
    ch, cw = max(8, h // 22), max(12, w // 16)
    tex = fbm2d(rng, ch, cw, base_scale=0.17, octaves=5, periodic_x=periodic_x)
    tex = _resize_bilinear(tex, h, w, periodic_x=periodic_x)
    razor = np.exp(-((yy / 0.042) ** 2)) * (0.68 + 0.32 * tex)
    carve = np.clip(bulge_peak * (0.52 + 0.48 * tex) + razor * 0.42, 0.0, 1.0)
    soft = soften_band_envelope(dw, (h, w), periodic_x=periodic_x, lat_blur_sigma=10.0, power=0.58)
    out = np.clip(1.0 - carve * s * 0.62, 0.28, 1.0)
    return np.clip(out * (0.70 + 0.30 * soft), 0.28, 1.0).astype(np.float64)
