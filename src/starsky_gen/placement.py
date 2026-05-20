"""Galactic disk/halo density and Poisson-disk star placement on equirect."""

from __future__ import annotations

import math

import numpy as np

from starsky_gen.projections import sph_to_equirect_xy

# Keep sampled |lat| inside this margin of ±π/2 so stars do not pile on equirect rows y≈0/h-1.
EQUIRECT_LAT_POLE_MARGIN = 0.12


def lat_equirect_clip_bounds() -> tuple[float, float]:
    m = float(EQUIRECT_LAT_POLE_MARGIN)
    return -np.pi / 2 + m, np.pi / 2 - m


def equirect_steradian_weights(
    height: int,
    width: int,
    *,
    pole_taper_frac: float = 0.06,
) -> np.ndarray:
    """Per-pixel weight ∝ cos|lat| (equal area on the sphere) with pole row fade."""
    yy = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    lat = (yy - 0.5) * np.pi
    w_row = np.clip(np.cos(lat), 0.03, 1.0)
    edge = max(1, int(round(pole_taper_frac * height)))
    if edge > 0:
        ramp = np.linspace(0.04, 1.0, edge, dtype=np.float64) ** 1.6
        taper = np.ones(height, dtype=np.float64)
        taper[:edge] = np.minimum(taper[:edge], ramp)
        taper[-edge:] = np.minimum(taper[-edge:], ramp[::-1])
        w_row *= taper
    return np.broadcast_to(w_row[:, None], (height, width)).astype(np.float64)


def apply_equirect_steradian_weights(density: np.ndarray) -> np.ndarray:
    """Correct pixel-space density for equirectangular area distortion."""
    d = np.asarray(density, dtype=np.float64)
    h, w = d.shape
    out = d * equirect_steradian_weights(h, w)
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
) -> np.ndarray:
    """Large-scale density D: disk × band × fractal turbulence (associations + Poisson)."""
    from starsky_gen.procedural_noise import _resize_bilinear, fbm2d, ridged_fbm2d

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
    poisson_min_sep_bright_px: float = 14.0,
    poisson_min_sep_faint_px: float = 3.0,
    bright_fraction: float = 0.028,
    association_fraction: float = 0.14,
    density_map: np.ndarray | None = None,
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
        min_sep_bright_px=poisson_min_sep_bright_px,
        min_sep_faint_px=poisson_min_sep_faint_px,
        n_bright=n_bright_tier,
        cluster_strength=0.0,
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
        )
        sig_lon = float(rng.uniform(0.008, 0.038))
        sig_lat = float(rng.uniform(0.006, 0.028))
        lon_c = (lon_c + rng.normal(0.0, sig_lon, size=n_here)) % (2.0 * np.pi)
        lat_lo, lat_hi = lat_equirect_clip_bounds()
        lat_c = np.clip(lat_c + rng.normal(0.0, sig_lat, size=n_here), lat_lo, lat_hi)
        lon_parts.append(lon_c)
        lat_parts.append(lat_c)
    lon = np.concatenate(lon_parts)
    lat = np.concatenate(lat_parts)
    return _normalize_placement_count(lon, lat, n, master, width, height, rng)


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
    points: list[tuple[int, int]] = []

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
    return pixels_to_lon_lat(rows.astype(np.int32), cols.astype(np.int32), width, height, rng)


def _accept_reject_fill(
    n_extra: int,
    density: np.ndarray,
    rng: np.random.Generator,
    existing: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = density.shape
    flat = density.ravel()
    flat /= flat.sum()
    rows: list[int] = [p[0] for p in existing]
    cols: list[int] = [p[1] for p in existing]
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
) -> tuple[np.ndarray, np.ndarray]:
    """Map pixel indices to galactic lon/lat with sub-pixel jitter inside cell."""
    cols_f = cols.astype(np.float64)
    rows_f = rows.astype(np.float64)
    xn = (cols_f + rng.uniform(-0.35, 0.35, size=cols_f.size)) / float(width)
    yn = (rows_f + rng.uniform(-0.35, 0.35, size=rows_f.size)) / float(height)
    lon = (xn % 1.0) * 2.0 * np.pi
    lat = (yn - 0.5) * np.pi
    lat_lo, lat_hi = lat_equirect_clip_bounds()
    lat = np.clip(lat, lat_lo, lat_hi)
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
    min_sep_bright_px: float = 14.0,
    min_sep_faint_px: float = 3.0,
    n_bright: int = 0,
    cluster_strength: float = 0.0,
    rng_cluster: np.random.Generator | None = None,
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
            cluster_map=cluster_map,
        )
    density_map = apply_equirect_steradian_weights(
        np.asarray(density_map, dtype=np.float64)
    )
    n_bright = min(int(n_bright), n)
    n_faint = n - n_bright
    all_lon: list[np.ndarray] = []
    all_lat: list[np.ndarray] = []
    if n_bright > 0:
        br, bc = _poisson_disk_on_grid(n_bright, density_map, rng, min_sep_px=min_sep_bright_px)
        lon_b, lat_b = pixels_to_lon_lat(br, bc, width, height, rng)
        all_lon.append(lon_b)
        all_lat.append(lat_b)
    if n_faint > 0:
        # Poisson-disk Bridson loop is O(n × attempts); use weighted AR for large catalogs.
        if n_faint > 400:
            lon_f, lat_f = _accept_reject_lon_lat(n_faint, density_map, width, height, rng)
        else:
            fr, fc = _poisson_disk_on_grid(n_faint, density_map, rng, min_sep_px=min_sep_faint_px)
            lon_f, lat_f = pixels_to_lon_lat(fr, fc, width, height, rng)
        all_lon.append(lon_f)
        all_lat.append(lat_f)
    lon = np.concatenate(all_lon) if all_lon else np.zeros(0, dtype=np.float64)
    lat = np.concatenate(all_lat) if all_lat else np.zeros(0, dtype=np.float64)
    return _normalize_placement_count(
        lon, lat, n, density_map, width, height, rng
    )


def _normalize_placement_count(
    lon: np.ndarray,
    lat: np.ndarray,
    n: int,
    density_map: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson-disk may return fewer than ``n``; trim or top up so lon/lat lengths match."""
    target = int(n)
    if lon.size > target:
        pick = rng.choice(lon.size, size=target, replace=False)
        return lon[pick], lat[pick]
    if lon.size < target:
        need = target - int(lon.size)
        lon_e, lat_e = _accept_reject_lon_lat(need, density_map, width, height, rng)
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
    poisson_min_sep_bright_px: float = 14.0,
    poisson_min_sep_faint_px: float = 3.0,
    bright_fraction: float = 0.028,
    cluster_density_strength: float = 0.0,
    density_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth Poisson-disk field (associations/GCs are a separate catalog layer)."""
    n_bright = min(max(1, int(n * bright_fraction)), 160)
    return sample_lon_lat_poisson(
        n,
        width,
        height,
        rng,
        density_map=density_map,
        disk_height=disk_height,
        halo_fraction=halo_fraction,
        halo_power=halo_power,
        band_lat_sigma=band_lat_sigma,
        band_rotation_deg=band_rotation_deg,
        band_curvature_amp=band_curvature_amp,
        min_sep_bright_px=poisson_min_sep_bright_px,
        min_sep_faint_px=poisson_min_sep_faint_px,
        n_bright=n_bright,
        cluster_strength=cluster_density_strength,
        rng_cluster=rng if cluster_density_strength > 1e-5 else None,
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
        lat_lo, lat_hi = lat_equirect_clip_bounds()
        lat = np.clip(lat, lat_lo, lat_hi)
    return lon, lat
