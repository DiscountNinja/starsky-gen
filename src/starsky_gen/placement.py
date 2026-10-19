"""Galactic disk/halo density and Poisson-disk star placement on equirect."""

from __future__ import annotations

import math

import numpy as np

from starsky_gen.projections import sph_to_equirect_xy

# Keep sampled |lat| inside this margin of ±π/2 so stars do not pile on equirect rows y≈0/h-1.
# Full-sky nebula vistas use the softer margin to avoid empty polar bands / razor edges.
EQUIRECT_LAT_POLE_MARGIN = 0.12
EQUIRECT_LAT_POLE_MARGIN_FULL_SKY = 0.02
EQUIRECT_POLE_TAPER_FRAC = 0.06
EQUIRECT_POLE_TAPER_FRAC_FULL_SKY = 0.018
# Tiny Y inset (~1.5px) avoids clamp shelves on y=0/h-1 without emptying the poles.
# Soft glow truncation is handled in stamp_psf_patch via an edge fade — not by
# pushing all near-pole mass onto one shelf latitude.
EQUIRECT_POLE_EDGE_PX = 1.5


def equirect_pole_edge_margin(height: int) -> float:
    """Latitude margin for ~1–2 equirect rows — not a paint-safe glow inset."""
    return float(np.pi) * float(EQUIRECT_POLE_EDGE_PX) / float(max(int(height), 2))


# Back-compat aliases (older call sites / tests).
def equirect_paint_safe_guard_px(height: int) -> float:
    return float(EQUIRECT_POLE_EDGE_PX)


def equirect_paint_safe_pole_margin(height: int) -> float:
    return equirect_pole_edge_margin(height)


def lat_equirect_clip_bounds(*, pole_margin: float | None = None) -> tuple[float, float]:
    m = float(EQUIRECT_LAT_POLE_MARGIN if pole_margin is None else pole_margin)
    m = float(np.clip(m, 0.0, 0.35))
    return -np.pi / 2 + m, np.pi / 2 - m


def soft_squash_lat(
    lat: np.ndarray,
    *,
    pole_margin: float | None = None,
) -> np.ndarray:
    """Map latitudes into equirect-safe bounds without pile-up on the clip row.

    Hard ``np.clip`` collapses many near-pole samples onto one latitude → a
    bright horizontal star shelf. Soft tanh squash keeps a continuous spread.
    """
    a = np.asarray(lat, dtype=np.float64)
    lo, hi = lat_equirect_clip_bounds(pole_margin=pole_margin)
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    # Scale so |lat|≈π/2 maps just inside (lo, hi) with smooth tails.
    t = np.tanh(a / max(half * 1.08, 1e-6))
    return mid + half * t


def _nudge_pole_rows(
    rows: np.ndarray,
    height: int,
    rng: np.random.Generator,
    *,
    guard_rows: int = 2,
) -> np.ndarray:
    """Redistribute samples off the outermost equirect rows (avoids shelf)."""
    r = np.asarray(rows, dtype=np.float64).copy()
    h = int(height)
    g = int(max(1, guard_rows))
    if h <= 2 * g + 2:
        return r
    top = r < float(g)
    bot = r > float(h - 1 - g)
    if np.any(top):
        r[top] = rng.uniform(float(g), float(g + 3), size=int(top.sum()))
    if np.any(bot):
        r[bot] = rng.uniform(float(h - 1 - g - 3), float(h - 1 - g), size=int(bot.sum()))
    return np.clip(r, 0.0, float(h - 1))


def _pole_edge_row_taper(
    height: int,
    width: int,
    *,
    guard_rows: int = 4,
    fade_rows: int = 10,
) -> np.ndarray:
    """Near-zero weight on outermost equirect rows; ramps up over ``fade_rows``.

    Keeps an epic full-sky field even without recreating a midplane band.
    """
    h = int(height)
    g = int(max(1, guard_rows))
    fade = int(max(g + 1, fade_rows))
    w_row = np.ones(h, dtype=np.float64)
    if h <= 2 * fade + 2:
        return np.broadcast_to(w_row[:, None], (h, width)).astype(np.float64)
    # Hard mute the absolute shelf rows, then ease in.
    w_row[:g] = 0.0
    w_row[-g:] = 0.0
    ramp = np.linspace(0.0, 1.0, fade - g + 1, dtype=np.float64)[1:]
    n_ramp = int(ramp.size)
    w_row[g : g + n_ramp] = ramp
    w_row[-(g + n_ramp) : -g] = ramp[::-1]
    return np.broadcast_to(w_row[:, None], (h, width)).astype(np.float64)


def equirect_steradian_weights(
    height: int,
    width: int,
    *,
    pole_taper_frac: float = EQUIRECT_POLE_TAPER_FRAC,
    pole_lift: float = 0.0,
) -> np.ndarray:
    """Per-pixel weight ∝ cos|lat| (equal area on the sphere) with pole row fade.

    ``pole_lift`` in [0,1] raises polar weights for full-sky nebula vistas so
    stars are not vacuumed out of the equirect poles.
    """
    yy = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    lat = (yy - 0.5) * np.pi
    w_row = np.clip(np.cos(lat), 0.03, 1.0)
    lift = float(np.clip(pole_lift, 0.0, 1.0))
    if lift > 1e-6:
        w_row = np.clip(w_row * (1.0 - 0.72 * lift) + 0.55 * lift, 0.18, 1.0)
    edge = max(0, int(round(float(pole_taper_frac) * height)))
    if edge > 0:
        ramp_lo = 0.22 if lift > 1e-6 else 0.04
        ramp = np.linspace(ramp_lo, 1.0, edge, dtype=np.float64) ** 1.6
        taper = np.ones(height, dtype=np.float64)
        taper[:edge] = np.minimum(taper[:edge], ramp)
        taper[-edge:] = np.minimum(taper[-edge:], ramp[::-1])
        w_row *= taper
    return np.broadcast_to(w_row[:, None], (height, width)).astype(np.float64)


def apply_equirect_steradian_weights(
    density: np.ndarray,
    *,
    pole_taper_frac: float = EQUIRECT_POLE_TAPER_FRAC,
    pole_lift: float = 0.0,
) -> np.ndarray:
    """Correct pixel-space density for equirectangular area distortion."""
    d = np.asarray(density, dtype=np.float64)
    h, w = d.shape
    out = d * equirect_steradian_weights(
        h, w, pole_taper_frac=pole_taper_frac, pole_lift=pole_lift
    )
    mx = float(np.max(out))
    if mx > 1e-8:
        out = out / mx
    return np.clip(out, 1e-4, 1.0).astype(np.float64)


def rho_disk_sech2(lat_rad: np.ndarray, disk_height: float) -> np.ndarray:
    h = max(float(disk_height), 1e-4)
    z = lat_rad / h
    return (1.0 / np.cosh(z)) ** 2


def rho_halo(lat_rad: np.ndarray, halo_power: float) -> np.ndarray:
    ab = np.abs(lat_rad) + 0.06
    p = max(float(halo_power), 0.35)
    return ab**p


def latitudinal_density(
    lat_rad: np.ndarray,
    *,
    disk_height: float = 0.19,
    halo_fraction: float = 0.22,
    halo_power: float = 1.35,
) -> np.ndarray:
    d = rho_disk_sech2(lat_rad, disk_height)
    h = rho_halo(lat_rad, halo_power)
    f = float(np.clip(halo_fraction, 0.0, 1.0))
    return (1.0 - f) * d + f * h


def build_cluster_density_modulation(
    height: int,
    width: int,
    rng: np.random.Generator,
    *,
    strength: float = 0.72,
    periodic_x: bool = True,
) -> np.ndarray:
    """Worley/FBM hybrid mask: peaks = clusters, valleys = gaps."""
    from starsky_gen.procedural_noise import _resize_bilinear, fbm2d, ridged_fbm2d

    ch, cw = max(8, height // 52), max(12, width // 72)
    a = fbm2d(rng, ch, cw, base_scale=0.09, octaves=4, periodic_x=periodic_x)
    r = ridged_fbm2d(rng, ch, cw, base_scale=0.07, octaves=3, periodic_x=periodic_x)
    a = _resize_bilinear(a, height, width, periodic_x=periodic_x)
    r = _resize_bilinear(r, height, width, periodic_x=periodic_x)
    cells = 1.0 - np.abs(r * 2.0 - 1.0)
    mix = np.clip(a * 0.38 + cells * 0.62, 0.0, 1.0) ** 1.85
    s = float(np.clip(strength, 0.0, 1.5))
    return np.clip(0.10 + mix * (0.38 + 0.58 * s), 0.08, 1.0).astype(np.float64)


def build_isotropic_equirect_density_map(
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    turbulence_strength: float = 0.48,
    cluster_strength: float = 0.38,
    periodic_x: bool = True,
) -> np.ndarray:
    """Full-sky density with no midplane band — mild turbulence + sparse association knots.

    Steradian / pole weights are applied later by ``sample_lon_lat_poisson``.
    """
    from starsky_gen.procedural_noise import _resize_bilinear, fbm2d, ridged_fbm2d

    base = np.ones((height, width), dtype=np.float64)
    ch, cw = max(8, height // 56), max(12, width // 80)
    turb = fbm2d(rng, ch, cw, base_scale=0.11, octaves=3, periodic_x=periodic_x)
    ridge = ridged_fbm2d(rng, ch, cw, base_scale=0.08, octaves=2, periodic_x=periodic_x)
    turb = _resize_bilinear(turb, height, width, periodic_x=periodic_x)
    ridge = _resize_bilinear(ridge, height, width, periodic_x=periodic_x)
    ts = float(np.clip(turbulence_strength, 0.0, 1.5))
    mod = np.clip(
        0.82 + ts * (0.22 * turb + 0.14 * (1.0 - np.abs(ridge * 2.0 - 1.0))),
        0.55,
        1.28,
    )
    d = base * mod
    cs = float(np.clip(cluster_strength, 0.0, 1.5))
    if cs > 1e-5:
        cm = build_cluster_density_modulation(
            height, width, rng, strength=cs, periodic_x=periodic_x
        )
        # High floor so knots vary density without recreating a band.
        d = d * (0.62 + 0.38 * cm)
    d = d / (float(np.max(d)) + 1e-8)
    return np.clip(d, 1e-4, 1.0).astype(np.float64)


def build_master_density_field(
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    disk_height: float = 0.19,
    halo_fraction: float = 0.22,
    halo_power: float = 1.35,
    band_lat_sigma: float = 0.10,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
    turbulence_strength: float = 0.85,
    periodic_x: bool = True,
    isotropic_field: bool = False,
) -> np.ndarray:
    """Large-scale density D: disk × band × fractal turbulence (associations + Poisson).

    ``isotropic_field=True`` skips the MW midplane model for nebula / full-sky vistas.
    """
    from starsky_gen.procedural_noise import _resize_bilinear, fbm2d, ridged_fbm2d

    if isotropic_field:
        return build_isotropic_equirect_density_map(
            width,
            height,
            rng,
            turbulence_strength=min(float(turbulence_strength), 0.55),
            cluster_strength=0.38,
            periodic_x=periodic_x,
        )

    cluster_map = build_cluster_density_modulation(
        height, width, rng, strength=0.92, periodic_x=periodic_x
    )
    base = build_equirect_density_map(
        width,
        height,
        disk_height=disk_height,
        halo_fraction=halo_fraction,
        halo_power=halo_power,
        band_lat_sigma=band_lat_sigma,
        band_rotation_deg=band_rotation_deg,
        band_curvature_amp=band_curvature_amp,
        cluster_map=cluster_map,
    )
    ch, cw = max(8, height // 56), max(12, width // 80)
    turb = fbm2d(rng, ch, cw, base_scale=0.10, octaves=3, periodic_x=periodic_x)
    ridge = ridged_fbm2d(rng, ch, cw, base_scale=0.08, octaves=2, periodic_x=periodic_x)
    turb = _resize_bilinear(turb, height, width, periodic_x=periodic_x)
    ridge = _resize_bilinear(ridge, height, width, periodic_x=periodic_x)
    ts = float(np.clip(turbulence_strength, 0.0, 1.5))
    mod = np.clip(0.72 + ts * (0.28 * turb + 0.18 * (1.0 - np.abs(ridge * 2.0 - 1.0))), 0.45, 1.35)
    d = np.clip(base * mod, 1e-4, None)
    d = d / (float(np.max(d)) + 1e-8)
    return np.clip(d, 1e-4, 1.0).astype(np.float64)


def pick_association_peaks(
    density_map: np.ndarray,
    rng: np.random.Generator,
    *,
    n_peaks: int,
    min_sep_px: float = 28.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima on density field → association (row, col) sites."""
    d = np.asarray(density_map, dtype=np.float64)
    h, w = d.shape
    n_peaks = max(4, int(n_peaks))
    flat = d.ravel()
    n_pix = flat.size
    # Only scan top density candidates (full argsort + per-pixel percentile was O(n²)).
    scan_k = min(n_pix, max(2048, n_peaks * 400))
    if scan_k < n_pix:
        top = np.argpartition(flat, -scan_k)[-scan_k:]
        order = top[np.argsort(flat[top])[::-1]]
    else:
        order = np.argsort(flat)[::-1]
    p55 = float(np.percentile(flat, 55.0))
    rows: list[float] = []
    cols: list[float] = []
    sep2 = float(min_sep_px) ** 2
    for idx in order:
        if len(rows) >= n_peaks:
            break
        if flat[idx] <= p55:
            continue
        r = float(idx // w) + rng.uniform(-0.35, 0.35)
        c = float(idx % w) + rng.uniform(-0.35, 0.35)
        ok = True
        for pr, pc in zip(rows, cols, strict=False):
            dr = r - pr
            dc = c - pc
            if dc > w * 0.5:
                dc -= w
            elif dc < -w * 0.5:
                dc += w
            if dr * dr + dc * dc < sep2:
                ok = False
                break
        if ok:
            rows.append(r)
            cols.append(c)
    if not rows:
        idx = int(order[0])
        rows = [float(idx // w)]
        cols = [float(idx % w)]
    return np.array(rows, dtype=np.float64), np.array(cols, dtype=np.float64)


def sample_hierarchical_poisson_field(
    rng: np.random.Generator,
    n: int,
    width: int,
    height: int,
    *,
    disk_height: float = 0.19,
    halo_fraction: float = 0.22,
    halo_power: float = 1.35,
    band_lat_sigma: float = 0.10,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
    lon_bulge_amp: float = 0.35,
    band_edge_power: float = 1.38,
    pole_margin: float | None = None,
    pole_taper_frac: float = EQUIRECT_POLE_TAPER_FRAC,
    pole_lift: float = 0.0,
    poisson_min_sep_bright_px: float = 14.0,
    poisson_min_sep_faint_px: float = 3.0,
    bright_fraction: float = 0.028,
    association_fraction: float = 0.14,
    density_map: np.ndarray | None = None,
    isotropic_field: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Density field → associations → Poisson individuals (reduces salt-and-pepper)."""
    if density_map is not None:
        master = np.asarray(density_map, dtype=np.float64)
        if master.shape != (height, width):
            raise ValueError(f"density_map shape {master.shape} != ({height}, {width})")
    else:
        master = build_master_density_field(
            width,
            height,
            rng,
            disk_height=disk_height,
            halo_fraction=halo_fraction,
            halo_power=halo_power,
            band_lat_sigma=band_lat_sigma,
            band_rotation_deg=band_rotation_deg,
            band_curvature_amp=band_curvature_amp,
            periodic_x=True,
            isotropic_field=isotropic_field,
        )
    n_base = max(1, int(n * (1.0 - association_fraction)))
    n_bright_tier = min(max(1, int(n_base * bright_fraction)), 160)
    lon_b, lat_b = sample_lon_lat_poisson(
        n_base,
        width,
        height,
        rng,
        density_map=master,
        disk_height=disk_height,
        halo_fraction=halo_fraction,
        halo_power=halo_power,
        band_lat_sigma=band_lat_sigma,
        band_rotation_deg=band_rotation_deg,
        band_curvature_amp=band_curvature_amp,
        lon_bulge_amp=lon_bulge_amp,
        band_edge_power=band_edge_power,
        pole_margin=pole_margin,
        pole_taper_frac=pole_taper_frac,
        pole_lift=pole_lift,
        min_sep_bright_px=poisson_min_sep_bright_px,
        min_sep_faint_px=poisson_min_sep_faint_px,
        n_bright=n_bright_tier,
        cluster_strength=0.0,
        skip_steradian_weights=isotropic_field,
    )
    n_assoc = max(0, n - n_base)
    if n_assoc < 8:
        return lon_b, lat_b
    n_sites = int(np.clip(n_assoc / 18, 8, 42))
    rows, cols = pick_association_peaks(master, rng, n_peaks=n_sites, min_sep_px=28.0)
    lon_parts = [lon_b]
    lat_parts = [lat_b]
    per_site = max(6, n_assoc // max(len(rows), 1))
    for r, c in zip(rows, cols, strict=False):
        n_here = int(rng.integers(max(4, per_site - 4), per_site + 6))
        lon_c, lat_c = pixels_to_lon_lat(
            np.full(n_here, r, dtype=np.float64),
            np.full(n_here, c, dtype=np.float64),
            width,
            height,
            rng,
            pole_margin=pole_margin,
        )
        sig_lon = float(rng.uniform(0.008, 0.038))
        sig_lat = float(rng.uniform(0.006, 0.028))
        lon_c = (lon_c + rng.normal(0.0, sig_lon, size=n_here)) % (2.0 * np.pi)
        lat_lo, lat_hi = lat_equirect_clip_bounds(pole_margin=pole_margin)
        lat_c = np.clip(lat_c + rng.normal(0.0, sig_lat, size=n_here), lat_lo, lat_hi)
        lon_parts.append(lon_c)
        lat_parts.append(lat_c)
    lon = np.concatenate(lon_parts)
    lat = np.concatenate(lat_parts)
    return _normalize_placement_count(
        lon, lat, n, master, width, height, rng, pole_margin=pole_margin
    )


def build_equirect_density_map(
    width: int,
    height: int,
    *,
    disk_height: float = 0.19,
    halo_fraction: float = 0.22,
    halo_power: float = 1.35,
    band_lat_sigma: float = 0.10,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
    lon_bulge_amp: float = 0.35,
    cluster_map: np.ndarray | None = None,
    band_edge_power: float = 1.38,
) -> np.ndarray:
    """2D density D in [0,1] for Poisson-disk and accept-reject."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    xn = (xx + 0.5) / width
    yn = (yy + 0.5) / height
    lat = (yn - 0.5) * np.pi
    lon = (xn - 0.5) * 2.0 * np.pi
    rot = math.radians(float(band_rotation_deg))
    lat_w = lat * math.cos(rot) + lon * math.sin(rot) * 0.12
    lat_w = lat_w + band_curvature_amp * np.sin(2.0 * np.pi * xn)
    rho = latitudinal_density(
        lat_w,
        disk_height=disk_height,
        halo_fraction=halo_fraction,
        halo_power=halo_power,
    )
    band = np.exp(-((lat_w / max(band_lat_sigma * np.pi, 0.02)) ** 2))
    cx = 0.5
    dlon = np.minimum(np.abs(xn - cx), 1.0 - np.abs(xn - cx))
    bulge_lon = np.exp(-((dlon / 0.18) ** 2))
    d = rho * (0.35 + 0.65 * band) * (1.0 + lon_bulge_amp * bulge_lon)
    edge = np.exp(-((lat_w / max(band_lat_sigma * np.pi * 1.15, 0.02)) ** 2)) ** float(band_edge_power)
    d = d * np.clip(edge, 0.08, 1.0)
    if cluster_map is not None:
        cm = np.asarray(cluster_map, dtype=np.float64)
        if cm.shape == (height, width):
            d = d * cm
    d = d / (float(np.max(d)) + 1e-8)
    return np.clip(d, 1e-4, 1.0).astype(np.float64)


def galactic_midplane_mask(
    height: int,
    width: int,
    *,
    lat_sigma: float = 0.10,
    lon_envelope: float = 0.22,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    xn = (xx + 0.5) / width
    yn = (yy + 0.5) / height
    lat = (yn - 0.5) * np.pi
    lon = (xn - 0.5) * 2.0 * np.pi
    rot = math.radians(float(band_rotation_deg))
    lat_w = lat * math.cos(rot) + lon * math.sin(rot) * 0.12
    lat_w = lat_w + band_curvature_amp * np.sin(2.0 * np.pi * xn)
    band = np.exp(-((lat_w / max(lat_sigma * np.pi, 0.02)) ** 2))
    cx = 0.5
    dlon = np.minimum(np.abs(xn - cx), 1.0 - np.abs(xn - cx))
    env = 1.0 + lon_envelope * np.exp(-((dlon / 0.20) ** 2))
    return np.clip(band * env, 0.0, 1.0).astype(np.float64)


def _poisson_disk_on_grid(
    n_target: int,
    density: np.ndarray,
    rng: np.random.Generator,
    *,
    min_sep_px: float,
    max_attempts: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Bridson-style Poisson-disk; returns (row, col) pixel indices."""
    h, w = density.shape
    cell = max(min_sep_px / math.sqrt(2.0), 1.0)
    gw = int(math.ceil(w / cell))
    gh = int(math.ceil(h / cell))
    grid: list[list[tuple[int, int] | None]] = [[None] * gw for _ in range(gh)]
    active: list[tuple[int, int]] = []
    points: list[tuple[float, float]] = []

    def _cell_idx(x: float, y: float) -> tuple[int, int]:
        return int(np.clip(x / cell, 0, gw - 1)), int(np.clip(y / cell, 0, gh - 1))

    def _ok(x: float, y: float) -> bool:
        gi, gj = _cell_idx(x, y)
        for dj in range(-2, 3):
            for di in range(-2, 3):
                ci, cj = gi + di, gj + dj
                if 0 <= ci < gw and 0 <= cj < gh:
                    p = grid[cj][ci]
                    if p is not None:
                        dx = p[0] - x
                        dy = p[1] - y
                        if dx * dx + dy * dy < min_sep_px * min_sep_px:
                            return False
        return True

    # Seed from weighted random
    flat = density.ravel()
    flat /= flat.sum()
    idx0 = int(rng.choice(flat.size, p=flat))
    y0, x0 = divmod(idx0, w)
    points.append((y0, x0))
    active.append((y0, x0))
    gi, gj = _cell_idx(float(x0), float(y0))
    grid[gj][gi] = (x0, y0)

    attempts = 0
    while len(points) < n_target and attempts < n_target * max_attempts * 4:
        attempts += 1
        if not active:
            idx = int(rng.choice(flat.size, p=flat))
            y0, x0 = divmod(idx, w)
            if _ok(float(x0), float(y0)):
                points.append((y0, x0))
                active.append((y0, x0))
                gi, gj = _cell_idx(float(x0), float(y0))
                grid[gj][gi] = (x0, y0)
            continue
        ay, ax = active[int(rng.integers(0, len(active)))]
        placed = False
        for _ in range(max_attempts):
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(min_sep_px, 2.0 * min_sep_px)
            nx = ax + rad * math.cos(ang)
            ny = ay + rad * math.sin(ang)
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if not _ok(nx, ny):
                continue
            iy_f, ix_f = float(ny), float(nx)
            ix_i, iy_i = int(ix_f), int(iy_f)
            if density[iy_i, ix_i] < rng.random() * float(np.max(density)):
                continue
            # Sub-pixel center (avoids integer-grid clumping when projected to the sphere).
            jx = float(rng.uniform(-0.42, 0.42))
            jy = float(rng.uniform(-0.42, 0.42))
            points.append((iy_f + jy, ix_f + jx))
            active.append((int(iy_f), int(ix_f)))
            gi, gj = _cell_idx(nx, ny)
            grid[gj][gi] = (nx, ny)
            placed = True
            break
        if not placed:
            active.remove((ay, ax))

    if len(points) < n_target:
        return _accept_reject_fill(n_target - len(points), density, rng, points)

    rows = np.array([p[0] for p in points[:n_target]], dtype=np.float64)
    cols = np.array([p[1] for p in points[:n_target]], dtype=np.float64)
    return rows, cols


def _accept_reject_lon_lat(
    n: int,
    density: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    pole_margin: float | None = None,
    soft_poles: bool | None = None,
    nudge_pole_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast density-weighted placement for large faint-star counts."""
    flat = np.asarray(density.ravel(), dtype=np.float64)
    flat = flat / (float(flat.sum()) + 1e-14)
    # rng.choice(..., p=flat) on millions of pixels × tens of thousands of stars is very slow.
    cdf = np.cumsum(flat)
    u = rng.random(int(n))
    idx = np.searchsorted(cdf, u * cdf[-1], side="right")
    idx = np.clip(idx, 0, flat.size - 1)
    rows, cols = np.divmod(idx, density.shape[1])
    if nudge_pole_rows:
        rows = _nudge_pole_rows(rows.astype(np.float64), height, rng, guard_rows=2)
    if soft_poles is None:
        soft_poles = (
            pole_margin is not None
            and float(pole_margin) <= EQUIRECT_LAT_POLE_MARGIN_FULL_SKY + 1e-9
        )
    return pixels_to_lon_lat(
        rows.astype(np.int32),
        cols.astype(np.int32),
        width,
        height,
        rng,
        pole_margin=pole_margin,
        soft_poles=bool(soft_poles),
    )


def _accept_reject_fill(
    n_extra: int,
    density: np.ndarray,
    rng: np.random.Generator,
    existing: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = density.shape
    flat = density.ravel()
    flat /= flat.sum()
    rows: list[int] = [int(p[0]) for p in existing]
    cols: list[int] = [int(p[1]) for p in existing]
    for _ in range(n_extra * 8):
        if len(rows) >= len(existing) + n_extra:
            break
        idx = int(rng.choice(flat.size, p=flat))
        y, x = divmod(idx, w)
        rows.append(y)
        cols.append(x)
    r = np.array(rows[: len(existing) + n_extra], dtype=np.int32)
    c = np.array(cols[: len(existing) + n_extra], dtype=np.int32)
    return r, c


def pixels_to_lon_lat(
    rows: np.ndarray,
    cols: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    pole_margin: float | None = None,
    soft_poles: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Map pixel indices to galactic lon/lat with sub-pixel jitter inside cell."""
    cols_f = cols.astype(np.float64)
    rows_f = rows.astype(np.float64)
    if soft_poles:
        rows_f = _nudge_pole_rows(rows_f, height, rng, guard_rows=2)
    xn = (cols_f + rng.uniform(-0.35, 0.35, size=cols_f.size)) / float(width)
    yn = (rows_f + rng.uniform(-0.35, 0.35, size=rows_f.size)) / float(height)
    lon = (xn % 1.0) * 2.0 * np.pi
    if soft_poles:
        lat = soft_squash_lat((yn - 0.5) * np.pi, pole_margin=pole_margin)
    else:
        # Clamp in row space (not lat) with a tiny inset so jitter cannot collapse a
        # whole near-pole band onto one shelf latitude.
        if pole_margin is None:
            y_pad = 0.5 / float(max(height, 2))
        else:
            y_pad = max(0.5 / float(max(height, 2)), float(pole_margin) / float(np.pi))
        yn = np.clip(yn, y_pad, 1.0 - y_pad)
        lat = (yn - 0.5) * np.pi
    return lon, lat


def sample_lon_lat_poisson(
    n: int,
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    density_map: np.ndarray | None = None,
    disk_height: float = 0.19,
    halo_fraction: float = 0.22,
    halo_power: float = 1.35,
    band_lat_sigma: float = 0.10,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
    lon_bulge_amp: float = 0.35,
    band_edge_power: float = 1.38,
    pole_margin: float | None = None,
    pole_taper_frac: float = EQUIRECT_POLE_TAPER_FRAC,
    pole_lift: float = 0.0,
    min_sep_bright_px: float = 14.0,
    min_sep_faint_px: float = 3.0,
    n_bright: int = 0,
    cluster_strength: float = 0.0,
    rng_cluster: np.random.Generator | None = None,
    skip_steradian_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Place n stars: optional bright tier with larger min separation."""
    if density_map is None:
        cluster_map = None
        if cluster_strength > 1e-5 and rng_cluster is not None:
            cluster_map = build_cluster_density_modulation(
                height, width, rng_cluster, strength=cluster_strength, periodic_x=True
            )
        density_map = build_equirect_density_map(
            width,
            height,
            disk_height=disk_height,
            halo_fraction=halo_fraction,
            halo_power=halo_power,
            band_lat_sigma=band_lat_sigma,
            band_rotation_deg=band_rotation_deg,
            band_curvature_amp=band_curvature_amp,
            lon_bulge_amp=lon_bulge_amp,
            band_edge_power=band_edge_power,
            cluster_map=cluster_map,
        )
    dens = np.asarray(density_map, dtype=np.float64)
    if skip_steradian_weights:
        # Flat equirect sampling for nebula "epic field" plates (no midplane from cos|lat|).
        # Mute only the absolute border rows — a deep fade makes empty caps + a shelf ring.
        dens = dens * _pole_edge_row_taper(height, width, guard_rows=1, fade_rows=3)
        mx = float(np.max(dens))
        density_map = np.clip(dens / mx if mx > 1e-8 else dens, 1e-4, 1.0)
    else:
        density_map = apply_equirect_steradian_weights(
            dens,
            pole_taper_frac=pole_taper_frac,
            pole_lift=pole_lift,
        )
    n_bright = min(int(n_bright), n)
    n_faint = n - n_bright
    all_lon: list[np.ndarray] = []
    all_lat: list[np.ndarray] = []
    # Soft pole squash is for MW shelf avoidance. Nebula isotropic: nudge only (no lat squash).
    soft_squash = (not skip_steradian_weights) and (
        pole_margin is not None and float(pole_margin) <= EQUIRECT_LAT_POLE_MARGIN_FULL_SKY + 1e-9
    )
    nudge_only = bool(skip_steradian_weights)
    # Tiny lat margin only — stamp edge-fade handles glow/spike truncation.
    # Do not honor large FULL_SKY margins here: they shelf onto one equirect row.
    if nudge_only:
        place_margin = equirect_pole_edge_margin(height)
        guard_nudge = 1
    else:
        place_margin = pole_margin
        guard_nudge = 2
    if n_bright > 0:
        br, bc = _poisson_disk_on_grid(n_bright, density_map, rng, min_sep_px=min_sep_bright_px)
        if nudge_only:
            br = _nudge_pole_rows(
                br.astype(np.float64), height, rng, guard_rows=guard_nudge
            ).astype(np.int32)
        lon_b, lat_b = pixels_to_lon_lat(
            br, bc, width, height, rng, pole_margin=place_margin, soft_poles=soft_squash
        )
        all_lon.append(lon_b)
        all_lat.append(lat_b)
    if n_faint > 0:
        # Poisson-disk Bridson loop is O(n × attempts); use weighted AR for large catalogs.
        if n_faint > 400:
            lon_f, lat_f = _accept_reject_lon_lat(
                n_faint,
                density_map,
                width,
                height,
                rng,
                pole_margin=place_margin,
                soft_poles=soft_squash,
                nudge_pole_rows=nudge_only,
            )
        else:
            fr, fc = _poisson_disk_on_grid(n_faint, density_map, rng, min_sep_px=min_sep_faint_px)
            if nudge_only:
                fr = _nudge_pole_rows(
                    fr.astype(np.float64), height, rng, guard_rows=guard_nudge
                ).astype(np.int32)
            lon_f, lat_f = pixels_to_lon_lat(
                fr, fc, width, height, rng, pole_margin=place_margin, soft_poles=soft_squash
            )
        all_lon.append(lon_f)
        all_lat.append(lat_f)
    lon = np.concatenate(all_lon) if all_lon else np.zeros(0, dtype=np.float64)
    lat = np.concatenate(all_lat) if all_lat else np.zeros(0, dtype=np.float64)
    return _normalize_placement_count(
        lon,
        lat,
        n,
        density_map,
        width,
        height,
        rng,
        pole_margin=place_margin,
        soft_poles=soft_squash,
        nudge_pole_rows=nudge_only,
    )


def _normalize_placement_count(
    lon: np.ndarray,
    lat: np.ndarray,
    n: int,
    density_map: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    pole_margin: float | None = None,
    soft_poles: bool | None = None,
    nudge_pole_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson-disk may return fewer than ``n``; trim or top up so lon/lat lengths match."""
    target = int(n)
    if lon.size > target:
        pick = rng.choice(lon.size, size=target, replace=False)
        return lon[pick], lat[pick]
    if lon.size < target:
        need = target - int(lon.size)
        lon_e, lat_e = _accept_reject_lon_lat(
            need,
            density_map,
            width,
            height,
            rng,
            pole_margin=pole_margin,
            soft_poles=soft_poles,
            nudge_pole_rows=nudge_pole_rows,
        )
        return np.concatenate([lon, lon_e]), np.concatenate([lat, lat_e])
    return lon, lat


def sample_smooth_poisson_field(
    rng: np.random.Generator,
    n: int,
    width: int,
    height: int,
    *,
    disk_height: float = 0.19,
    halo_fraction: float = 0.22,
    halo_power: float = 1.35,
    band_lat_sigma: float = 0.10,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
    lon_bulge_amp: float = 0.35,
    band_edge_power: float = 1.38,
    pole_margin: float | None = None,
    pole_taper_frac: float = EQUIRECT_POLE_TAPER_FRAC,
    pole_lift: float = 0.0,
    poisson_min_sep_bright_px: float = 14.0,
    poisson_min_sep_faint_px: float = 3.0,
    bright_fraction: float = 0.028,
    cluster_density_strength: float = 0.0,
    density_map: np.ndarray | None = None,
    isotropic_field: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth Poisson-disk field (associations/GCs are a separate catalog layer)."""
    n_bright = min(max(1, int(n * bright_fraction)), 160)
    dens = density_map
    if dens is None and isotropic_field:
        dens = build_isotropic_equirect_density_map(
            width, height, rng, cluster_strength=max(float(cluster_density_strength), 0.28)
        )
    return sample_lon_lat_poisson(
        n,
        width,
        height,
        rng,
        density_map=dens,
        disk_height=disk_height,
        halo_fraction=halo_fraction,
        halo_power=halo_power,
        band_lat_sigma=band_lat_sigma,
        band_rotation_deg=band_rotation_deg,
        band_curvature_amp=band_curvature_amp,
        lon_bulge_amp=lon_bulge_amp,
        band_edge_power=band_edge_power,
        pole_margin=pole_margin,
        pole_taper_frac=pole_taper_frac,
        pole_lift=pole_lift,
        min_sep_bright_px=poisson_min_sep_bright_px,
        min_sep_faint_px=poisson_min_sep_faint_px,
        n_bright=n_bright,
        cluster_strength=0.0 if isotropic_field else cluster_density_strength,
        rng_cluster=None if isotropic_field else (rng if cluster_density_strength > 1e-5 else None),
        skip_steradian_weights=isotropic_field,
    )


def sample_galactic_disk_lon_lat_v2(
    rng: np.random.Generator,
    n: int,
    width: int,
    height: int,
    *,
    disk_height: float = 0.19,
    halo_fraction: float = 0.22,
    halo_power: float = 1.35,
    band_lat_sigma: float = 0.10,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
    poisson_min_sep_bright_px: float = 14.0,
    poisson_min_sep_faint_px: float = 3.0,
    bright_fraction: float = 0.028,
    cluster_strength: float = 0.55,
    stream_perturbation: float = 0.0,
    hierarchical: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Galactic disk stars: hierarchical field+associations or legacy smooth Poisson."""
    if hierarchical:
        lon, lat = sample_hierarchical_poisson_field(
            rng,
            n,
            width,
            height,
            disk_height=disk_height,
            halo_fraction=halo_fraction,
            halo_power=halo_power,
            band_lat_sigma=band_lat_sigma,
            band_rotation_deg=band_rotation_deg,
            band_curvature_amp=band_curvature_amp,
            poisson_min_sep_bright_px=poisson_min_sep_bright_px,
            poisson_min_sep_faint_px=poisson_min_sep_faint_px,
            bright_fraction=bright_fraction,
        )
    else:
        lon, lat = sample_smooth_poisson_field(
            rng,
            n,
            width,
            height,
            disk_height=disk_height,
            halo_fraction=halo_fraction,
            halo_power=halo_power,
            band_lat_sigma=band_lat_sigma,
            band_rotation_deg=band_rotation_deg,
            band_curvature_amp=band_curvature_amp,
            poisson_min_sep_bright_px=poisson_min_sep_bright_px,
            poisson_min_sep_faint_px=poisson_min_sep_faint_px,
            bright_fraction=bright_fraction,
            cluster_density_strength=cluster_strength,
        )
    sp = float(stream_perturbation)
    if sp < 1e-5:
        return lon, lat
    n_cl = int(rng.integers(12, 28))
    cl_lon = rng.uniform(0.0, 2.0 * np.pi, size=n_cl)
    cl_lat = rng.normal(0.0, 0.08, size=n_cl)
    pick = rng.integers(0, n_cl, size=n)
    mask = rng.random(n) < (0.22 * sp)
    if np.any(mask):
        lon[mask] = (cl_lon[pick[mask]] + rng.normal(0, 0.04, mask.sum())) % (2.0 * np.pi)
        lat[mask] = cl_lat[pick[mask]] + rng.normal(0, 0.03, mask.sum())
        lat_lo, lat_hi = lat_equirect_clip_bounds(pole_margin=None)
        lat = np.clip(lat, lat_lo, lat_hi)
    return lon, lat
