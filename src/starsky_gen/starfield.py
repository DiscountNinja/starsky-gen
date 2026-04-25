from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from starsky_gen.projections import sph_to_equirect_xy

STAR_COLOR_NAMES = ["white", "blue", "yellow", "red"]
STAR_COLOR_WEIGHTS = np.array([0.54, 0.17, 0.16, 0.13], dtype=np.float64)
STAR_SIZE_NAMES = ["tiny", "small", "medium", "large"]
STAR_SIZE_WEIGHTS = np.array([0.758, 0.194, 0.032, 0.016], dtype=np.float64)

# B–V bins for stats / heuristics (rough Johnson B–V).
_BV_BLUE_MAX = 0.12
_BV_WHITE_MAX = 0.42
_BV_YELLOW_MAX = 0.95

BASE_COLORS = {
    "white": np.array([0.93, 0.95, 1.00], dtype=np.float64),
    "blue": np.array([0.72, 0.84, 1.00], dtype=np.float64),
    "yellow": np.array([0.66, 0.64, 0.54], dtype=np.float64),
    "red": np.array([0.31, 0.21, 0.19], dtype=np.float64),
}

_BV_STOPS = np.array([-0.35, -0.1, 0.15, 0.35, 0.58, 0.85, 1.15, 1.6, 2.1], dtype=np.float64)
_RGB_STOPS = np.array(
    [
        [0.62, 0.78, 1.0],
        [0.70, 0.84, 1.0],
        [0.88, 0.93, 1.0],
        [0.94, 0.93, 0.90],
        [0.90, 0.78, 0.58],
        [0.96, 0.64, 0.40],
        [0.82, 0.52, 0.36],
        [0.62, 0.40, 0.30],
        [0.48, 0.30, 0.26],
    ],
    dtype=np.float64,
)


def bv_to_color_idx(bv: np.ndarray) -> np.ndarray:
    """Map B–V to discrete STAR_COLOR_NAMES index for stats."""
    return np.select(
        [
            bv < _BV_BLUE_MAX,
            (bv >= _BV_BLUE_MAX) & (bv < _BV_WHITE_MAX),
            (bv >= _BV_WHITE_MAX) & (bv < _BV_YELLOW_MAX),
        ],
        [1, 0, 2],
        default=3,
    )


def rgb_from_bv(bv: float, jitter: np.ndarray) -> np.ndarray:
    """Continuous stellar color from Johnson B–V (empirical RGB locus)."""
    t = float(np.clip(bv, _BV_STOPS[0], _BV_STOPS[-1]))
    r = float(np.interp(t, _BV_STOPS, _RGB_STOPS[:, 0]))
    g = float(np.interp(t, _BV_STOPS, _RGB_STOPS[:, 1]))
    b = float(np.interp(t, _BV_STOPS, _RGB_STOPS[:, 2]))
    rgb = np.array([r, g, b], dtype=np.float64) + jitter
    neutral = float(np.mean(rgb))
    rgb = rgb * 0.84 + neutral * 0.16
    if bv < 0.06:
        rgb = rgb * np.array([0.96, 0.98, 1.04], dtype=np.float64)
    return np.clip(rgb, 0.0, 1.0)


def _bv_from_color_idx(color_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    centers = np.array([-0.18, 0.28, 0.72, 1.35], dtype=np.float64)
    c = centers[np.clip(color_idx, 0, 3)]
    return np.clip(c + rng.normal(0.0, 0.11, size=color_idx.shape[0]), -0.35, 2.1)


def _sample_bv_latitudes(lat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """B–V distribution: halo/plane skew like real disk + bulge fields."""
    band_w = np.exp(-((lat / 0.40) ** 2))
    halo = np.clip(1.0 - band_w, 0.0, 1.0)
    mean_bv = 0.50 + 0.46 * band_w - 0.14 * halo
    sigma = 0.34 + 0.32 * band_w + 0.11 * halo
    bv = rng.normal(mean_bv, sigma, size=lat.shape[0])
    u = rng.random(lat.shape[0])
    hot = u < 0.052
    bv = np.where(hot, rng.uniform(-0.35, 0.04, size=lat.shape[0]), bv)
    cool = u > 0.948
    bv = np.where(cool, rng.uniform(0.82, 1.95, size=lat.shape[0]), bv)
    return np.clip(bv, -0.35, 2.1)


@dataclass
class StarStats:
    color_counts: dict[str, int]
    size_counts: dict[str, int]


def sample_star_catalog(
    rng: np.random.Generator,
    width: int,
    height: int,
    density_scale: float,
    *,
    layer: Literal["background", "foreground"] = "background",
    galactic_band_boost: float = 1.0,
    latitude_color_bias: bool = True,
) -> dict[str, np.ndarray]:
    area = width * height
    if layer == "foreground":
        n_stars = max(56, int(area / 300 * density_scale))
        size_p = np.array([0.03, 0.12, 0.48, 0.37], dtype=np.float64)
        size_p /= size_p.sum()
    else:
        n_stars = max(120, int(area / 66 * density_scale * galactic_band_boost))
        size_p = STAR_SIZE_WEIGHTS

    if layer == "foreground":
        lon = rng.uniform(0.0, 2.0 * np.pi, size=n_stars)
        core_lat = rng.normal(0.0, 0.30, size=n_stars)
        wing_lat = rng.normal(0.0, 0.52, size=n_stars)
        blend = rng.random(n_stars)
        lat = np.where(blend < 0.72, core_lat, wing_lat)
        lat += rng.normal(0.0, 0.035, size=n_stars)
        lat = np.tanh(lat / 1.15) * (np.pi / 2.0 - 0.02)
    else:
        lon, lat = sample_galactic_disk_lon_lat(rng, n_stars)

    if latitude_color_bias and layer == "background":
        bv = _sample_bv_latitudes(lat, rng)
        color_idx = bv_to_color_idx(bv)
    else:
        color_idx = rng.choice(len(STAR_COLOR_NAMES), size=n_stars, p=STAR_COLOR_WEIGHTS)
        bv = _bv_from_color_idx(color_idx, rng)
    if layer == "foreground":
        bv = np.clip(rng.normal(0.22, 0.38, size=n_stars), -0.35, 1.25)
        color_idx = bv_to_color_idx(bv)
    size_idx = rng.choice(len(STAR_SIZE_NAMES), size=n_stars, p=size_p)
    jitter = rng.normal(0.0, 0.018, size=(n_stars, 3))
    return {
        "lon": lon,
        "lat": lat,
        "color_idx": color_idx,
        "size_idx": size_idx,
        "jitter": jitter,
        "bv": bv,
    }


def sample_galactic_disk_lon_lat(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Oversample: disk plane + many overlapping clumps + streams, then thin to n."""
    overs = 1.78
    m = max(n + 24, int(n * overs) + 32)
    lon = rng.uniform(0.0, 2.0 * np.pi, size=m)
    core_lat = rng.normal(0.0, 0.19, size=m)
    wing_lat = rng.normal(0.0, 0.36, size=m)
    mid_lat = rng.normal(0.0, 0.28, size=m)
    blend = rng.random(m)
    lat = np.where(blend < 0.76, core_lat, np.where(blend < 0.93, wing_lat, mid_lat))
    lat += rng.normal(0.0, 0.052, size=m)
    lat = np.tanh(lat / 1.14) * (np.pi / 2.0 - 0.02)

    n_cl = int(rng.integers(40, 98))
    cl_lon = rng.uniform(0.0, 2.0 * np.pi, size=n_cl)
    cl_lat = rng.normal(0.0, 0.10, size=n_cl)
    cl_lat = np.tanh(cl_lat / 1.12) * (np.pi / 2.0 - 0.02)
    sig_lon = rng.uniform(0.009, 0.076, size=n_cl)
    sig_lat = rng.uniform(0.006, 0.040, size=n_cl)
    cw = rng.uniform(0.32, 1.0, size=n_cl)
    cw /= np.sum(cw)
    assign = rng.choice(n_cl, size=m, p=cw)
    p_cloud = float(rng.uniform(0.50, 0.66))
    is_c = rng.random(m) < p_cloud
    if np.any(is_c):
        idx = np.flatnonzero(is_c)
        aj = assign[is_c]
        lon[idx] = (cl_lon[aj] + rng.normal(0.0, sig_lon[aj], size=idx.size)) % (2.0 * np.pi)
        lat[idx] = cl_lat[aj] + rng.normal(0.0, sig_lat[aj], size=idx.size)
    st = rng.random(m) < 0.07
    if np.any(st):
        st_idx = np.flatnonzero(st)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        lat[st_idx] = lat[st_idx] + 0.13 * np.sin(2.15 * (lon[st_idx] - phase))
    lat = np.tanh(lat / 1.14) * (np.pi / 2.0 - 0.02)

    knot_k = float(rng.uniform(2.1, 4.6))
    knot_ph = float(rng.uniform(0.0, 2.0 * np.pi))
    knot_amp = float(rng.uniform(0.045, 0.12))
    lon = (lon + knot_amp * np.sin(knot_k * lon + knot_ph)) % (2.0 * np.pi)
    lon = (lon + rng.normal(0.0, 0.042, size=m)) % (2.0 * np.pi)

    sigma = 0.188
    disk_w = np.maximum(np.exp(-((lat / sigma) ** 2)) ** 1.52, 0.024)
    for _ in range(2):
        cl_lon = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.uniform(0.18, 0.38))
        dlon = np.abs(lon - cl_lon)
        dlon = np.minimum(dlon, 2.0 * np.pi - dlon)
        disk_w *= 1.0 + amp * np.exp(-((dlon / 0.092) ** 2))
    disk_w = np.maximum(disk_w, 0.016)
    prio = disk_w / (rng.random(m) + 1e-8)
    pick = np.argpartition(-prio, n - 1)[:n]
    return lon[pick], lat[pick]


def reroll_stars_in_dark_lanes(
    catalog: dict[str, np.ndarray],
    rng: np.random.Generator,
    width: int,
    height: int,
    extinction: np.ndarray,
    *,
    exponent: float = 0.84,
    floor: float = 0.11,
    max_passes: int = 7,
) -> None:
    """Move stars out of pixels with strong extinction (fewer stars inside dust lanes)."""
    lon = catalog["lon"]
    lat = catalog["lat"]
    n = int(lon.shape[0])
    for _ in range(max_passes):
        xi, yi = sph_to_equirect_xy(lon, lat, width, height)
        e = extinction[yi, xi]
        p_keep = np.clip(e**exponent, floor, 1.0)
        bad = rng.random(n) >= p_keep
        if not np.any(bad):
            break
        n_new = int(np.count_nonzero(bad))
        lon_new, lat_new = sample_galactic_disk_lon_lat(rng, n_new)
        lon[bad] = lon_new
        lat[bad] = lat_new


def sample_cluster_star_catalog(
    rng: np.random.Generator,
    width: int,
    height: int,
    density_scale: float,
) -> dict[str, np.ndarray]:
    """Dense open clusters and short star streams near the galactic plane (photo realism)."""
    area = width * height
    total = int(np.clip(area / 950.0 * density_scale, 220, 8800))
    n_clusters = int(np.clip(6.0 + area / 340000.0, 6, 32))
    n_clusters = min(n_clusters, max(1, total // 20))

    lon_parts: list[np.ndarray] = []
    lat_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    bv_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    j_parts: list[np.ndarray] = []

    remaining = total
    for ci in range(n_clusters):
        if remaining <= 0:
            break
        cl_lon = float(rng.uniform(0.0, 2.0 * np.pi))
        cl_lat = float(rng.normal(0.0, 0.082))
        sig_lon = float(rng.uniform(0.012, 0.058))
        sig_lat = float(rng.uniform(0.009, 0.038))
        base_per = max(12, total // max(n_clusters, 1))
        lo_n = max(8, int(base_per * 0.35))
        hi_n = max(lo_n + 1, int(base_per * 1.75) + 1)
        n_here = int(rng.integers(lo_n, hi_n))
        n_here = max(8, min(n_here, remaining))
        if ci == n_clusters - 1:
            n_here = remaining
        lon_c = (cl_lon + rng.normal(0.0, sig_lon, size=n_here)) % (2.0 * np.pi)
        lat_c = cl_lat + rng.normal(0.0, sig_lat, size=n_here)
        stream = float(rng.uniform(-0.014, 0.014))
        lat_c = lat_c + stream * np.sin((lon_c - cl_lon) * (2.3 + rng.uniform(-0.4, 0.4)))
        lat_c = np.tanh(lat_c / 1.12) * (np.pi / 2.0 - 0.02)
        bv_c = np.clip(rng.normal(-0.04, 0.22, size=n_here), -0.35, 0.72)
        warm_giant = rng.random(n_here) < 0.085
        bv_c = np.where(warm_giant, rng.uniform(0.72, 1.28, size=n_here), bv_c)
        color_idx = bv_to_color_idx(bv_c)
        size_idx = np.zeros(n_here, dtype=np.int64)
        jitter = rng.normal(0.0, 0.028, size=(n_here, 3))
        lon_parts.append(lon_c)
        lat_parts.append(lat_c)
        c_parts.append(color_idx)
        bv_parts.append(bv_c)
        s_parts.append(size_idx)
        j_parts.append(jitter)
        remaining -= n_here

    if not lon_parts:
        return sample_star_catalog(
            rng, width, height, density_scale * 0.02, layer="background", galactic_band_boost=1.0, latitude_color_bias=False
        )

    return {
        "lon": np.concatenate(lon_parts),
        "lat": np.concatenate(lat_parts),
        "color_idx": np.concatenate(c_parts),
        "size_idx": np.concatenate(s_parts),
        "jitter": np.concatenate(j_parts, axis=0),
        "bv": np.concatenate(bv_parts),
    }


def size_radius(rng: np.random.Generator, size_name: str) -> int:
    if size_name == "tiny":
        return 1
    if size_name == "small":
        return int(rng.integers(2, 4))
    if size_name == "medium":
        return int(rng.integers(3, 5))
    return int(rng.integers(5, 8))


def star_color(color_name: str, jitter: np.ndarray) -> np.ndarray:
    base = BASE_COLORS[color_name].copy()
    color = np.clip(base + jitter, 0.0, 1.0)
    # Keep most colors subtle, but preserve stronger hue separation for blue stars.
    neutral = np.mean(color)
    if color_name == "blue":
        color = color * 0.95 + neutral * 0.05
        color = color * np.array([0.90, 0.95, 1.12], dtype=np.float64)
    else:
        color = color * 0.88 + neutral * 0.12
    return np.clip(color, 0.0, 1.0)


def _paint_thin_diffraction_spikes(
    img: np.ndarray,
    x: int,
    y: int,
    radius: int,
    color_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Rare thin 4-fold spikes (spider / refractor optics), not a second star population."""
    h, w, _ = img.shape
    peak = float(np.max(color_rgb))
    if peak < 0.22:
        return
    if radius >= 5:
        if rng.random() >= 0.52:
            return
        L = int(rng.integers(14, 26))
        amp0 = peak * float(rng.uniform(0.022, 0.058))
    elif radius >= 4:
        if rng.random() >= 0.14:
            return
        L = int(rng.integers(9, 18))
        amp0 = peak * float(rng.uniform(0.012, 0.038))
    else:
        if peak < 0.38 or rng.random() >= 0.065:
            return
        L = int(rng.integers(5, 11))
        amp0 = peak * float(rng.uniform(0.007, 0.022))

    ang = float(rng.uniform(0.0, 0.5 * np.pi))
    for k in range(4):
        th = ang + k * (0.5 * np.pi)
        cth, sth = float(np.cos(th)), float(np.sin(th))
        for t in range(1, L + 1):
            ox = int(round(t * cth))
            oy = int(round(t * sth))
            py, px = y + oy, (x + ox) % w
            if py < 0 or py >= h:
                continue
            fall = 1.0 / ((0.55 + float(t)) ** 1.15)
            tip = float(t) / float(max(L, 1))
            chrom = np.array([0.96 - 0.05 * tip, 0.98 - 0.02 * tip, 1.0 + 0.10 * tip], dtype=np.float64)
            img[py, px] += color_rgb * (amp0 * fall) * chrom


def paint_star(
    img: np.ndarray,
    x: int,
    y: int,
    radius: int,
    color_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    galactic_lat: float | None = None,
) -> None:
    h, w, _ = img.shape
    if radius <= 1:
        if 0 <= x < w and 0 <= y < h:
            u_b = float(rng.random())
            if u_b < 0.58:
                brightness = float(rng.uniform(0.12, 0.44))
            elif u_b < 0.90:
                brightness = float(rng.uniform(0.36, 0.82))
            else:
                brightness = float(rng.uniform(0.72, 1.18))
            color = color_rgb * float(rng.uniform(0.84, 1.0))
            img[y, x] += color * brightness
            # Occasional tight optic cross on saturated pinpoints (rare vs field).
            if (
                brightness * float(np.max(color_rgb)) > 0.62
                and rng.random() < 0.0045
            ):
                sp = float(np.max(color_rgb)) * brightness * float(rng.uniform(0.06, 0.14))
                for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
                    px = (x + ox) % w
                    py = y + oy
                    if 0 <= py < h:
                        img[py, px] += color_rgb * sp * (0.55 if abs(ox) + abs(oy) == 2 else 1.0)
        return

    bright_star = radius >= 5
    asym_x = rng.uniform(0.82, 1.18) if bright_star else rng.uniform(0.88, 1.12)
    asym_y = rng.uniform(0.82, 1.18) if bright_star else rng.uniform(0.88, 1.12)
    if galactic_lat is not None and radius > 1:
        plane_f = float(np.exp(-((galactic_lat / 0.38) ** 2)))
        asym_x *= 1.0 + 0.12 * plane_f
        asym_y *= 1.0 / (1.0 + 0.08 * plane_f)
    tilt = rng.uniform(-0.28, 0.28)
    core = radius * 0.32
    edge_noise_scale = rng.uniform(0.06, 0.16) if bright_star else rng.uniform(0.04, 0.12)
    glow_jitter_scale = rng.uniform(0.05, 0.12)
    dx = rng.uniform(-0.3, 0.3)
    dy = rng.uniform(-0.3, 0.3)

    for oy in range(-radius, radius + 1):
        py = y + oy
        if py < 0 or py >= h:
            continue
        for ox in range(-radius, radius + 1):
            px = (x + ox) % w
            tx = ox * np.cos(tilt) - oy * np.sin(tilt) + dx
            ty = ox * np.sin(tilt) + oy * np.cos(tilt) + dy
            theta = np.arctan2(ty, tx)
            edge_ruffle = 1.0 + np.sin(theta * 4.0 + tilt * 7.0) * edge_noise_scale
            local_radius = max(1.0, radius * edge_ruffle)
            d = np.sqrt((tx / asym_x) ** 2 + (ty / asym_y) ** 2)
            if d > local_radius:
                continue
            core_sigma = max(core * 0.62, 0.22)
            base = np.exp(-(d**2) / (2 * (core_sigma**2)))
            if radius <= 2:
                halo = 0.0
            else:
                halo_sigma = max(radius * 0.36, 0.85)
                halo_boost = 0.006 if radius == 3 else (0.005 if radius == 4 else (0.016 if radius >= 5 else 0.0))
                halo_w = 0.038 + 0.025 * min(radius, 7) + halo_boost
                halo = np.exp(-(d**2) / (2 * halo_sigma**2)) * halo_w
            glow_jitter = 1.0 + rng.uniform(-glow_jitter_scale, glow_jitter_scale)
            intensity = min(1.0, (base * 0.98 + halo) * glow_jitter)
            if d < core * 0.40:
                intensity = min(1.0, intensity * 1.06)
            if d > core:
                edge_cool = np.array([0.78, 0.86, 1.0])
                pixel_color = color_rgb * 0.93 + edge_cool * 0.07
            else:
                pixel_color = color_rgb
            img[py, px] += pixel_color * intensity

    if bright_star and rng.random() < 0.44:
        sp = float(rng.uniform(0.058, 0.12)) * float(np.max(color_rgb))
        for ox, oy in ((1, 0), (-1, 0), (2, 0), (-2, 0), (0, 1), (0, -1), (0, 2), (0, -2)):
            px = (x + ox) % w
            py = y + oy
            if 0 <= py < h:
                img[py, px] += color_rgb * sp

    if radius >= 4 and rng.random() < 0.32:
        L = int(min(max(radius + 3, 7), 15))
        vig = float(np.max(color_rgb)) * float(rng.uniform(0.011, 0.026))
        for ox in range(-L, L + 1):
            if ox == 0:
                continue
            wx = 1.0 / (1.0 + abs(ox) * 0.62)
            px = (x + ox) % w
            if 0 <= y < h:
                img[y, px] += color_rgb * vig * wx
        for oy in range(-L, L + 1):
            if oy == 0:
                continue
            wy = 1.0 / (1.0 + abs(oy) * 0.62)
            py = y + oy
            if 0 <= py < h:
                img[py, x % w] += color_rgb * vig * wy

    if radius >= 3:
        _paint_thin_diffraction_spikes(img, x, y, radius, color_rgb, rng)


def catalog_stats(catalog: dict[str, np.ndarray]) -> StarStats:
    cidx = bv_to_color_idx(catalog["bv"]) if "bv" in catalog else catalog["color_idx"]
    sidx = catalog["size_idx"]
    color_counts = {name: int(np.sum(cidx == i)) for i, name in enumerate(STAR_COLOR_NAMES)}
    size_counts = {name: int(np.sum(sidx == i)) for i, name in enumerate(STAR_SIZE_NAMES)}
    return StarStats(color_counts=color_counts, size_counts=size_counts)
