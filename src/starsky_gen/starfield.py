from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from starsky_gen.galactic_structure import GalacticMorphology

GalacticStructure = GalacticMorphology

from starsky_gen.color_science import (
    sample_teffective_for_placement,
    teff_to_bv,
)
from starsky_gen.photometry import sample_apparent_magnitudes_lf
from starsky_gen.placement import (
    EQUIRECT_LAT_POLE_MARGIN_FULL_SKY,
    EQUIRECT_POLE_TAPER_FRAC,
    equirect_pole_edge_margin,
    lat_equirect_clip_bounds,
    sample_hierarchical_poisson_field,
    sample_smooth_poisson_field,
    soft_squash_lat,
)
from starsky_gen.projections import sph_to_equirect_xy

STAR_COLOR_NAMES = ["white", "blue", "yellow", "red"]
STAR_COLOR_WEIGHTS = np.array([0.58, 0.20, 0.16, 0.06], dtype=np.float64)
STAR_SIZE_NAMES = ["tiny", "small", "medium", "large"]
# Tiny-heavy cinematic hierarchy: vast faint texture; rare medium/large glow anchors.
STAR_SIZE_WEIGHTS = np.array([0.90, 0.085, 0.012, 0.003], dtype=np.float64)

_LAT_CLIP_LO, _LAT_CLIP_HI = lat_equirect_clip_bounds()
_MAX_EQUIRECT_LAT = _LAT_CLIP_HI


def _lat_clip_pair(*, pole_margin: float | None = None) -> tuple[float, float, float]:
    """Return (lat_lo, lat_hi, max_abs_lat) for equirect sampling."""
    lo, hi = lat_equirect_clip_bounds(pole_margin=pole_margin)
    return float(lo), float(hi), float(hi)


def _bound_lat(lat: np.ndarray, *, pole_margin: float | None = None) -> np.ndarray:
    """Keep latitudes inside equirect-safe bounds.

    Full-sky / nebula plates use a hard clip (soft tanh squash piles near-pole
    mass onto one shelf latitude a few dozen rows in from the Y border).
    MW-band placements keep the soft squash for the wider default margin.
    Prefer ``_sample_isotropic_lat`` for new sphere samples so mass is not
    collapsed onto the clip latitude.
    """
    if pole_margin is not None and float(pole_margin) <= EQUIRECT_LAT_POLE_MARGIN_FULL_SKY + 1e-9:
        lo, hi, _ = _lat_clip_pair(pole_margin=pole_margin)
        return np.clip(lat, lo, hi)
    if pole_margin is not None:
        return soft_squash_lat(lat, pole_margin=pole_margin)
    lo, hi, _ = _lat_clip_pair(pole_margin=pole_margin)
    return np.clip(lat, lo, hi)


def _sample_isotropic_lat(
    rng: np.random.Generator,
    n: int,
    *,
    pole_margin: float | None = None,
    height: int | None = None,
) -> np.ndarray:
    """Uniform-on-sphere latitudes, conditioned to stay off equirect pole rows.

    Samples ``sin(lat) ~ U(sin(lo), sin(hi))`` so density stays isotropic on the
    allowed band without clipping a pile onto one shelf latitude.
    When ``height`` is set, use a ~1.5px edge margin (not a large glow inset) so
    poles stay populated; stamp edge-fade softens truncated bloom.
    """
    # Prefer ~1.5px edge when height is known — large FULL_SKY margins shelf at low res.
    if height is not None and height > 1:
        margin = equirect_pole_edge_margin(height)
    elif pole_margin is None:
        margin = EQUIRECT_LAT_POLE_MARGIN_FULL_SKY
    else:
        margin = pole_margin
    lo, hi, _ = _lat_clip_pair(pole_margin=margin)
    u_lo = float(np.sin(lo))
    u_hi = float(np.sin(hi))
    if u_hi <= u_lo + 1e-12:
        return np.full(int(n), 0.0, dtype=np.float64)
    u = rng.uniform(u_lo, u_hi, size=int(n))
    return np.arcsin(u).astype(np.float64)

# B–V bins for stats / heuristics (rough Johnson B–V).
_BV_BLUE_MAX = 0.12
_BV_WHITE_MAX = 0.42
_BV_YELLOW_MAX = 0.95

BASE_COLORS = {
    "white": np.array([0.93, 0.95, 1.00], dtype=np.float64),
    "blue": np.array([0.82, 0.88, 0.98], dtype=np.float64),
    "yellow": np.array([0.72, 0.70, 0.62], dtype=np.float64),
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
    """B–V distribution: warm disk, rare hot blue, cool halo."""
    band_w = np.exp(-((lat / 0.40) ** 2))
    halo = np.clip(1.0 - band_w, 0.0, 1.0)
    mean_bv = 0.44 + 0.34 * band_w + 0.24 * halo
    sigma = 0.32 + 0.28 * band_w + 0.12 * halo
    bv = rng.normal(mean_bv, sigma, size=lat.shape[0])
    u = rng.random(lat.shape[0])
    hot = (u < 0.012) & (band_w < 0.48)
    bv = np.where(hot, rng.uniform(-0.22, 0.08, size=lat.shape[0]), bv)
    warm = (u < 0.16) & (band_w > 0.32)
    bv = np.where(warm, rng.uniform(0.58, 1.12, size=lat.shape[0]), bv)
    cool = u > 0.952
    bv = np.where(cool, rng.uniform(0.82, 1.95, size=lat.shape[0]), bv)
    return np.clip(bv, -0.35, 2.1)


def sample_apparent_magnitudes(
    n_stars: int,
    rng: np.random.Generator,
    *,
    mag_bright: float,
    mag_faint: float,
    magnitude_log_slope: float,
    magnitude_ultra_cut: float,
    max_ultra_bright_stars: int,
) -> np.ndarray:
    """Sample apparent V-like magnitudes with dN/dm ∝ 10^(α·m) between numeric endpoints."""
    ml = float(min(mag_bright, mag_faint))
    mh = float(max(mag_bright, mag_faint))
    k = np.log(10.0) * float(magnitude_log_slope)

    def _trunc_power_draw(n_draw: int) -> np.ndarray:
        if abs(k) < 1e-14:
            return rng.uniform(ml, mh, size=n_draw)
        u = rng.random(n_draw)
        ec_lo = np.exp(k * ml)
        ec_hi = np.exp(k * mh)
        m_draw = np.log(u * (ec_hi - ec_lo) + ec_lo) / k
        return np.clip(m_draw, ml, mh)

    mags = _trunc_power_draw(n_stars)
    cap = float(magnitude_ultra_cut)
    if max_ultra_bright_stars < 1:
        return mags.astype(np.float64)

    bright_idx = np.flatnonzero(mags < cap)
    excess = int(bright_idx.shape[0]) - int(max_ultra_bright_stars)
    if excess <= 0:
        return np.clip(mags, ml, mh).astype(np.float64)

    losers = bright_idx[np.argsort(mags[bright_idx])][:excess]
    eps = float((mh - ml) * 1e-6 + 5e-4)
    lo = float(np.minimum(cap + eps, mh - eps))

    def _cond_draw(n_draw: int, a: float) -> np.ndarray:
        if a >= mh - eps:
            a2 = mh - eps
        else:
            a2 = a
        if abs(k) < 1e-14:
            return rng.uniform(max(a2, ml), mh, size=n_draw)
        u = rng.random(n_draw)
        ec_lo = np.exp(k * a2)
        ec_hi = np.exp(k * mh)
        out = np.log(u * (ec_hi - ec_lo) + ec_lo) / k
        return np.clip(out, a2, mh)

    mags[losers] = _cond_draw(excess, lo)
    return np.clip(mags, ml, mh).astype(np.float64)


@dataclass
class StarStats:
    color_counts: dict[str, int]
    size_counts: dict[str, int]


def cull_faint_resolved_stars(
    catalog: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    mag_faint_floor: float,
    dropout_strength: float,
    mag_faint: float,
    magnitude_ultra_cut: float,
    galactic_structure: GalacticStructure | None = None,
    magnitude_ref_mag: float = 12.0,
    width: int = 0,
    height: int = 0,
) -> dict[str, np.ndarray]:
    """Cull faint resolved stars; culled flux deposits into unresolved field U."""
    if "phot_mag" not in catalog or float(dropout_strength) < 1e-6:
        return catalog
    mags = catalog["phot_mag"].astype(np.float64)
    n = int(mags.size)
    if n < 1:
        return catalog
    floor = float(mag_faint_floor)
    mh = float(mag_faint)
    cap = float(magnitude_ultra_cut)
    span = max(mh - floor, 1e-6)
    ultra_keep = mags < cap
    bright_keep = mags < floor
    u = np.clip((mags - floor) / span, 0.0, 1.0)
    p_drop = float(dropout_strength) * (u**1.35)
    keep = ultra_keep | bright_keep | (rng.random(n) >= p_drop)
    if galactic_structure is not None and width > 0 and height > 0 and np.any(~keep):
        from starsky_gen.psf import flux_from_mag

        drop_idx = np.flatnonzero(~keep)
        xi, yi = sph_to_equirect_xy(
            catalog["lon"][drop_idx], catalog["lat"][drop_idx], width, height
        )
        dust_vis = catalog.get("dust_visibility")
        dust_vis_len = int(dust_vis.shape[0]) if dust_vis is not None else 0
        for j, idx in enumerate(drop_idx):
            flux = float(flux_from_mag(float(mags[idx]), magnitude_ref_mag))
            flux *= float(p_drop[idx])
            if dust_vis is not None and idx < dust_vis_len:
                flux *= float(dust_vis[idx])
            elif galactic_structure is not None:
                flux *= float(
                    galactic_structure.sample_dust(int(yi[j]), int(xi[j]) % width)
                )
            galactic_structure.deposit_unresolved(int(yi[j]), int(xi[j]), flux)
    if bool(np.all(keep)):
        return catalog
    return {k: v[keep] for k, v in catalog.items()}


def _merge_catalog_append(
    base: dict[str, np.ndarray], extra: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if int(extra["lon"].size) < 1:
        return base
    if int(base["lon"].size) < 1:
        return extra
    out: dict[str, np.ndarray] = {}
    for key in set(base.keys()) | set(extra.keys()):
        if key in base and key in extra:
            out[key] = np.concatenate([base[key], extra[key]], axis=0)
        elif key in base:
            out[key] = base[key]
        else:
            out[key] = extra[key]
    return out


def sample_isotropic_cosmic_catalog(
    rng: np.random.Generator,
    width: int,
    height: int,
    density_scale: float,
    *,
    density_scale_mult: float = 1.0,
    attach_apparent_mag: bool = False,
    photometry_mag_bright: float = 11.0,
    photometry_mag_faint: float = 20.8,
    photometry_ultra_cut: float = 8.8,
    max_ultra_bright_stars: int = 5,
    anchor_count: int = 6,
    anchor_mag_bright: float = 4.8,
    anchor_mag_faint: float = 8.4,
    pole_margin: float | None = None,
) -> dict[str, np.ndarray]:
    """Population A: uniform sphere — independent of galactic morphology."""
    area = width * height
    n = max(
        48,
        int(area / 40.0 * density_scale * float(np.clip(density_scale_mult, 0.05, 2.5))),
    )
    lon = rng.uniform(0.0, 2.0 * np.pi, size=n)
    # Isotropic on the sphere, kept off equirect pole rows (no clip shelf).
    lat = _sample_isotropic_lat(rng, n, pole_margin=pole_margin, height=height)
    # Almost all tiny/cool — sparse cinematic voids, not a busy field.
    size_p = np.array([0.94, 0.055, 0.004, 0.001], dtype=np.float64)
    size_p /= size_p.sum()
    color_idx = rng.choice(len(STAR_COLOR_NAMES), size=n, p=np.array([0.78, 0.16, 0.05, 0.01]))
    bv = np.clip(rng.normal(0.12, 0.18, size=n), -0.28, 0.55)
    size_idx = rng.choice(len(STAR_SIZE_NAMES), size=n, p=size_p)
    jitter = rng.normal(0.0, 0.012, size=(n, 3))
    keys: dict[str, np.ndarray] = {
        "lon": lon,
        "lat": lat,
        "color_idx": color_idx,
        "size_idx": size_idx,
        "jitter": jitter,
        "bv": bv,
        "population": np.full(n, 0, dtype=np.int8),
    }
    if attach_apparent_mag:
        keys["phot_mag"] = sample_apparent_magnitudes_lf(
            n,
            rng,
            mag_bright=photometry_mag_bright,
            mag_faint=photometry_mag_faint,
            giant_fraction=0.012,
            magnitude_ultra_cut=photometry_ultra_cut,
            max_ultra_bright_stars=max_ultra_bright_stars,
        )
    na = int(np.clip(anchor_count, 0, 24))
    if na > 0:
        lon_a = rng.uniform(0.0, 2.0 * np.pi, size=na)
        lat_a = _sample_isotropic_lat(rng, na, pole_margin=pole_margin, height=height)
        anchor: dict[str, np.ndarray] = {
            "lon": lon_a,
            "lat": lat_a,
            "color_idx": rng.choice(len(STAR_COLOR_NAMES), size=na, p=np.array([0.48, 0.18, 0.26, 0.08])),
            "size_idx": rng.choice(
                len(STAR_SIZE_NAMES),
                size=na,
                p=np.array([0.02, 0.12, 0.38, 0.48], dtype=np.float64),
            ),
            "jitter": rng.normal(0.0, 0.01, size=(na, 3)),
            "bv": np.clip(rng.normal(0.18, 0.35, size=na), -0.35, 0.85),
            "population": np.full(na, 0, dtype=np.int8),
        }
        if attach_apparent_mag:
            anchor["phot_mag"] = rng.uniform(
                float(anchor_mag_bright),
                float(anchor_mag_faint),
                size=na,
            )
        keys = _merge_catalog_append(keys, anchor)
    return keys


def sample_halo_star_catalog(
    rng: np.random.Generator,
    width: int,
    height: int,
    density_scale: float,
    *,
    density_scale_mult: float = 1.0,
    halo_lat_sigma: float = 0.52,
    attach_apparent_mag: bool = False,
    photometry_mag_bright: float = 9.5,
    photometry_mag_faint: float = 20.0,
    photometry_ultra_cut: float = 7.8,
    max_ultra_bright_stars: int = 4,
    pole_margin: float | None = None,
    isotropic_field: bool = False,
) -> dict[str, np.ndarray]:
    """Population B: thick disk / halo — broad plane bias, older warmer stars."""
    area = width * height
    sig = float(np.clip(halo_lat_sigma, 0.28, 1.05))
    n = max(
        24,
        int(area / 195.0 * density_scale * float(np.clip(density_scale_mult, 0.05, 2.0))),
    )
    lon = rng.uniform(0.0, 2.0 * np.pi, size=n)
    if isotropic_field:
        lat = _sample_isotropic_lat(rng, n, pole_margin=pole_margin, height=height)
    else:
        lat = rng.normal(0.0, sig, size=n)
        lat += rng.normal(0.0, 0.06, size=n)
        lat = _bound_lat(np.tanh(lat * 1.08) * (np.pi * 0.5), pole_margin=pole_margin)
    size_p = np.array([0.88, 0.09, 0.02, 0.01], dtype=np.float64)
    size_p /= size_p.sum()
    bv = np.clip(rng.normal(0.78, 0.26, size=n), 0.22, 1.65)
    warm = rng.random(n) < 0.22
    bv = np.where(warm, rng.uniform(0.95, 1.45, size=n), bv)
    color_idx = bv_to_color_idx(bv)
    size_idx = rng.choice(len(STAR_SIZE_NAMES), size=n, p=size_p)
    jitter = rng.normal(0.0, 0.014, size=(n, 3))
    keys: dict[str, np.ndarray] = {
        "lon": lon,
        "lat": lat,
        "color_idx": color_idx,
        "size_idx": size_idx,
        "jitter": jitter,
        "bv": bv,
        "population": np.full(n, 1, dtype=np.int8),
    }
    if attach_apparent_mag:
        keys["phot_mag"] = sample_apparent_magnitudes_lf(
            n,
            rng,
            mag_bright=photometry_mag_bright,
            mag_faint=photometry_mag_faint,
            giant_fraction=0.065,
            magnitude_ultra_cut=photometry_ultra_cut,
            max_ultra_bright_stars=max_ultra_bright_stars,
        )
    return keys


def inject_galactic_overdensity_stars(
    catalog: dict[str, np.ndarray],
    rng: np.random.Generator,
    galactic_structure: GalacticMorphology,
    width: int,
    height: int,
    *,
    count: int = 320,
    attach_apparent_mag: bool = True,
    mag_bright: float = 9.2,
    mag_faint: float = 15.8,
    anchor_mag_bright: float = 5.4,
    anchor_mag_faint: float = 8.2,
    anchor_fraction: float = 0.14,
) -> dict[str, np.ndarray]:
    """Place extra resolved stars on morphology overdensity peaks (high → absurd tail)."""
    n = int(np.clip(count, 0, 4000))
    if n <= 0:
        return catalog
    peak = np.clip(
        galactic_structure.stellar_density * galactic_structure.cluster_prob,
        0.0,
        1.0,
    )
    flat = peak.ravel()
    if float(np.max(flat)) < 1e-8:
        return catalog
    weights = np.clip(flat**2.4, 1e-12, None)
    weights /= float(np.sum(weights)) + 1e-12
    pick = rng.choice(flat.size, size=n, replace=True, p=weights)
    yi, xi = np.divmod(pick, peak.shape[1])
    yi = yi.astype(np.float64) + rng.uniform(-0.42, 0.42, size=n)
    xi = xi.astype(np.float64) + rng.uniform(-0.42, 0.42, size=n)
    from starsky_gen.placement import pixels_to_lon_lat

    lon, lat = pixels_to_lon_lat(yi, xi, width, height, rng)
    size_p = np.array([0.04, 0.14, 0.38, 0.44], dtype=np.float64)
    size_p /= size_p.sum()
    size_idx = rng.choice(len(STAR_SIZE_NAMES), size=n, p=size_p)
    color_idx = rng.choice(len(STAR_COLOR_NAMES), size=n, p=STAR_COLOR_WEIGHTS)
    jitter = rng.uniform(-0.04, 0.04, size=(n, 3))
    bv = rng.uniform(-0.05, 0.55, size=n)
    teff = sample_teffective_for_placement(n, lat, rng)
    extra: dict[str, np.ndarray] = {
        "lon": lon.astype(np.float64),
        "lat": lat.astype(np.float64),
        "color_idx": color_idx.astype(np.int32),
        "size_idx": size_idx.astype(np.int32),
        "bv": bv.astype(np.float64),
        "teff": teff.astype(np.float64),
        "jitter": jitter.astype(np.float64),
    }
    if attach_apparent_mag:
        n_anchor = int(np.clip(n * float(anchor_fraction), 0, n))
        mags = rng.uniform(float(mag_bright), float(mag_faint), size=n).astype(np.float64)
        if n_anchor > 0:
            anchor_idx = rng.choice(n, size=n_anchor, replace=False)
            mags[anchor_idx] = rng.uniform(
                float(anchor_mag_bright), float(anchor_mag_faint), size=n_anchor
            )
        extra["phot_mag"] = mags
    return _merge_catalog_append(catalog, extra)


def sample_star_catalog(
    rng: np.random.Generator,
    width: int,
    height: int,
    density_scale: float,
    *,
    layer: Literal["background", "mid", "foreground"] = "background",
    galactic_band_boost: float = 1.0,
    latitude_color_bias: bool = True,
    attach_apparent_mag: bool = False,
    band_star_density_scale: float = 1.0,
    foreground_star_density_scale: float = 1.0,
    photometry_mag_bright: float = 8.0,
    photometry_mag_faint: float = 20.0,
    photometry_slope: float = 0.6,
    photometry_ultra_cut: float = 6.5,
    max_ultra_bright_stars: int = 6,
    use_poisson_placement: bool = True,
    disk_height: float = 0.19,
    halo_fraction: float = 0.22,
    halo_power: float = 1.35,
    band_lat_sigma: float = 0.10,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
    poisson_min_sep_bright_px: float = 14.0,
    poisson_min_sep_faint_px: float = 3.0,
    use_spectral_teffective: bool = True,
    placement_asymmetry: float = 0.0,
    cluster_strength: float = 0.72,
    use_imf_magnitudes: bool = True,
    imf_giant_fraction: float = 0.048,
    hierarchical_star_placement: bool = True,
    galactic_structure: GalacticStructure | None = None,
    population_gradient_strength: float = 0.72,
    lon_bulge_amp: float = 0.35,
    band_edge_power: float = 1.38,
    pole_margin: float | None = None,
    pole_taper_frac: float = EQUIRECT_POLE_TAPER_FRAC,
    pole_lift: float = 0.0,
    isotropic_field: bool = False,
    association_fraction: float = 0.14,
) -> dict[str, np.ndarray]:
    area = width * height
    if layer == "foreground":
        n_stars = max(
            48,
            int(area / 300 * density_scale * float(np.clip(foreground_star_density_scale, 0.35, 1.5))),
        )
        # Sparse bright anchors — still tiny-heavy, but more medium/large than the bed.
        size_p = np.array([0.72, 0.18, 0.07, 0.03], dtype=np.float64)
        size_p /= size_p.sum()
    elif layer == "mid":
        n_stars = max(
            72,
            int(area / 98 * density_scale * galactic_band_boost * band_star_density_scale * 0.55),
        )
        size_p = np.array([0.82, 0.13, 0.04, 0.01], dtype=np.float64)
        size_p /= size_p.sum()
    else:
        n_stars = max(
            120,
            int(area / 66 * density_scale * galactic_band_boost * band_star_density_scale),
        )
        size_p = STAR_SIZE_WEIGHTS

    if layer == "foreground" and galactic_structure is not None:
        from starsky_gen.galactic_structure import sample_foreground_lon_lat_from_structure

        lon, lat = sample_foreground_lon_lat_from_structure(rng, n_stars, galactic_structure)
        lat = _bound_lat(lat, pole_margin=pole_margin)
    elif layer == "foreground":
        lon = rng.uniform(0.0, 2.0 * np.pi, size=n_stars)
        core_lat = rng.normal(0.0, 0.38, size=n_stars)
        wing_lat = rng.normal(0.0, 0.62, size=n_stars)
        blend = rng.random(n_stars)
        lat = np.where(blend < 0.38, core_lat, wing_lat)
        lat += rng.normal(0.0, 0.048, size=n_stars)
        lat = _bound_lat(np.tanh(lat * 1.22 / 1.15) * (np.pi * 0.5), pole_margin=pole_margin)
    elif use_poisson_placement or isotropic_field:
        # isotropic_field must not fall through to sample_galactic_disk_lon_lat
        # (--no-galaxy-view sets poisson off via placement kwargs).
        g_map = (
            None
            if isotropic_field
            else (galactic_structure.stellar_density if galactic_structure is not None else None)
        )
        if (
            hierarchical_star_placement
            and layer == "background"
            and galactic_structure is not None
            and not isotropic_field
        ):
            from starsky_gen.galactic_structure import sample_hierarchical_from_structure

            lon, lat = sample_hierarchical_from_structure(
                rng,
                n_stars,
                galactic_structure,
                poisson_min_sep_bright_px=poisson_min_sep_bright_px,
                poisson_min_sep_faint_px=poisson_min_sep_faint_px,
                population_gradient_strength=population_gradient_strength,
            )
            lat = _bound_lat(lat, pole_margin=pole_margin)
        elif hierarchical_star_placement and layer == "background":
            lon, lat = sample_hierarchical_poisson_field(
                rng,
                n_stars,
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
                pole_margin=pole_margin,
                pole_taper_frac=pole_taper_frac,
                pole_lift=pole_lift,
                poisson_min_sep_bright_px=poisson_min_sep_bright_px,
                poisson_min_sep_faint_px=poisson_min_sep_faint_px,
                isotropic_field=isotropic_field,
                association_fraction=0.06 if isotropic_field else association_fraction,
            )
        else:
            lon, lat = sample_smooth_poisson_field(
                rng,
                n_stars,
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
                pole_margin=pole_margin,
                pole_taper_frac=pole_taper_frac,
                pole_lift=pole_lift,
                poisson_min_sep_bright_px=poisson_min_sep_bright_px,
                poisson_min_sep_faint_px=poisson_min_sep_faint_px,
                cluster_density_strength=0.0 if isotropic_field else cluster_strength * 0.45,
                density_map=g_map,
                isotropic_field=isotropic_field,
            )
    else:
        lon, lat = sample_galactic_disk_lon_lat(rng, n_stars)

    n_placed = int(lon.size)
    if n_placed != int(lat.size):
        n_use = min(n_placed, int(lat.size))
        lon, lat = lon[:n_use], lat[:n_use]
        n_placed = n_use

    asym = float(placement_asymmetry)
    if asym > 1e-5 and layer != "foreground":
        lon = lon + (rng.random(n_placed) - 0.5) * asym * 0.42 * np.sin(lat * 2.05 + rng.uniform(0, 6.28))
        lon = np.mod(lon, 2.0 * np.pi)
        # Default MW clip (±0.12 rad) empties ~20px polar caps and shelves stars on
        # the clip row. Isotropic / full-sky must keep the tiny edge margin only.
        if isotropic_field:
            clip_m = equirect_pole_edge_margin(height)
        elif pole_margin is not None:
            clip_m = float(pole_margin)
        else:
            clip_m = None
        if clip_m is None:
            lat_lo, lat_hi = _LAT_CLIP_LO, _LAT_CLIP_HI
        else:
            lat_lo, lat_hi, _ = _lat_clip_pair(pole_margin=clip_m)
        lat = np.clip(
            lat + (rng.random(n_placed) - 0.5) * asym * 0.14,
            lat_lo,
            lat_hi,
        )

    teff: np.ndarray | None = None
    if latitude_color_bias and layer == "background" and use_spectral_teffective:
        teff = sample_teffective_for_placement(n_placed, lat, rng)
        bv = np.array([teff_to_bv(float(t)) for t in teff], dtype=np.float64)
        color_idx = bv_to_color_idx(bv)
    elif latitude_color_bias and layer == "background":
        bv = _sample_bv_latitudes(lat, rng)
        color_idx = bv_to_color_idx(bv)
    else:
        color_idx = rng.choice(len(STAR_COLOR_NAMES), size=n_placed, p=STAR_COLOR_WEIGHTS)
        bv = _bv_from_color_idx(color_idx, rng)
    if layer == "foreground":
        bv = np.clip(rng.normal(0.22, 0.38, size=n_placed), -0.35, 1.25)
        color_idx = bv_to_color_idx(bv)
    size_idx = rng.choice(len(STAR_SIZE_NAMES), size=n_placed, p=size_p)
    jitter = rng.normal(0.0, 0.018, size=(n_placed, 3))
    keys: dict[str, np.ndarray] = {
        "lon": lon,
        "lat": lat,
        "color_idx": color_idx,
        "size_idx": size_idx,
        "jitter": jitter,
        "bv": bv,
    }
    if teff is not None:
        keys["teff"] = teff
    if attach_apparent_mag and layer in ("background", "mid"):
        if use_imf_magnitudes:
            keys["phot_mag"] = sample_apparent_magnitudes_lf(
                n_placed,
                rng,
                mag_bright=photometry_mag_bright,
                mag_faint=photometry_mag_faint,
                giant_fraction=imf_giant_fraction,
                magnitude_ultra_cut=photometry_ultra_cut,
                max_ultra_bright_stars=max_ultra_bright_stars,
            )
        else:
            keys["phot_mag"] = sample_apparent_magnitudes(
                n_placed,
                rng,
                mag_bright=photometry_mag_bright,
                mag_faint=photometry_mag_faint,
                magnitude_log_slope=photometry_slope,
                magnitude_ultra_cut=photometry_ultra_cut,
                max_ultra_bright_stars=max_ultra_bright_stars,
            )
    return keys


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
    lat = np.tanh(lat / 1.14) * _MAX_EQUIRECT_LAT

    n_cl = int(rng.integers(40, 98))
    cl_lon = rng.uniform(0.0, 2.0 * np.pi, size=n_cl)
    cl_lat = rng.normal(0.0, 0.10, size=n_cl)
    cl_lat = np.tanh(cl_lat / 1.12) * _MAX_EQUIRECT_LAT
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
    lat = np.tanh(lat / 1.14) * _MAX_EQUIRECT_LAT

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
    galactic_structure: GalacticStructure | None = None,
    exponent: float = 0.84,
    floor: float = 0.11,
    max_passes: int = 7,
) -> None:
    """Attach dust visibility from extinction; stars stay correlated (no teleport reroll)."""
    _ = rng, max_passes, exponent, floor
    if galactic_structure is not None:
        galactic_structure.merge_nebula_extinction(extinction)
        from starsky_gen.galactic_structure import attach_dust_visibility_to_catalog

        attach_dust_visibility_to_catalog(catalog, galactic_structure, width, height)
        return
    xi, yi = sph_to_equirect_xy(catalog["lon"], catalog["lat"], width, height)
    yi = np.clip(yi, 0, extinction.shape[0] - 1)
    xi = xi % extinction.shape[1]
    vis = np.clip(extinction[yi, xi] ** 0.84, 0.08, 1.0)
    catalog["dust_visibility"] = vis.astype(np.float64)


def sample_cluster_star_catalog(
    rng: np.random.Generator,
    width: int,
    height: int,
    density_scale: float,
    *,
    attach_apparent_mag: bool = False,
    band_star_density_scale: float = 1.0,
    photometry_mag_bright: float = 8.0,
    photometry_mag_faint: float = 20.0,
    photometry_slope: float = 0.6,
    photometry_ultra_cut: float = 6.5,
    max_ultra_bright_stars: int = 6,
    use_imf_magnitudes: bool = True,
    imf_giant_fraction: float = 0.11,
    galactic_structure: GalacticStructure | None = None,
    full_sky_sites: bool = False,
) -> dict[str, np.ndarray]:
    """Associations, open clusters, and compact GCs with radial magnitude gradients."""
    area = width * height
    total = int(np.clip(area / 640.0 * density_scale, 260, 9600))
    n_open = int(np.clip(5.0 + area / 380000.0, 5, 28))
    n_gc = int(np.clip(1.0 + area / 900000.0, 1, 5))
    n_clusters = n_open + n_gc
    n_clusters = min(n_clusters, max(1, total // 16))

    lon_parts: list[np.ndarray] = []
    lat_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    bv_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    j_parts: list[np.ndarray] = []
    is_gc_flags: list[bool] = []

    cluster_sites: list[tuple[float, float, bool]] = []
    if galactic_structure is not None and not full_sky_sites:
        from starsky_gen.galactic_structure import build_population_placement_maps
        from starsky_gen.placement import pick_association_peaks, pixels_to_lon_lat

        _, peak_map = build_population_placement_maps(
            galactic_structure, gradient_strength=0.78
        )
        rows, cols = pick_association_peaks(
            peak_map, rng, n_peaks=n_clusters, min_sep_px=36.0
        )
        for site_i, (ri, ci_px) in enumerate(zip(rows, cols, strict=False)):
            lon_p, lat_p = pixels_to_lon_lat(
                np.array([ri], dtype=np.float64),
                np.array([ci_px], dtype=np.float64),
                width,
                height,
                rng,
            )
            cluster_sites.append((float(lon_p[0]), float(lat_p[0]), site_i < n_gc))
    else:
        # Nebula / full-sky: uniform on the sphere (no midplane site bias).
        site_margin = EQUIRECT_LAT_POLE_MARGIN_FULL_SKY if full_sky_sites else None
        for site_i in range(n_clusters):
            cl_lon = float(rng.uniform(0.0, 2.0 * np.pi))
            cl_lat = float(
                _sample_isotropic_lat(rng, 1, pole_margin=site_margin, height=height)[0]
            )
            cluster_sites.append((cl_lon, cl_lat, site_i < n_gc))

    remaining = total
    for ci in range(n_clusters):
        if remaining <= 0:
            break
        is_gc = ci < n_gc
        if ci < len(cluster_sites):
            cl_lon, cl_lat, is_gc = cluster_sites[ci]
        elif full_sky_sites:
            cl_lon = float(rng.uniform(0.0, 2.0 * np.pi))
            cl_lat = float(np.arcsin(np.clip(rng.uniform(-1.0, 1.0), -1.0, 1.0)))
        else:
            cl_lon = float(rng.uniform(0.0, 2.0 * np.pi))
            cl_lat = float(rng.normal(0.0, 0.072 if is_gc else 0.082))
        gp_site = 0.45
        if galactic_structure is not None:
            from starsky_gen.projections import sph_to_equirect_xy

            xi_s, yi_s = sph_to_equirect_xy(
                np.array([cl_lon]), np.array([cl_lat]), width, height
            )
            yi_s = int(np.clip(int(yi_s[0]), 0, height - 1))
            xi_s = int(xi_s[0]) % width
            gp_site = float(galactic_structure.gold_population_weight[yi_s, xi_s])
        tight = 0.48 + 0.52 * (1.0 - gp_site)
        if is_gc:
            sig_lon = float(rng.uniform(0.004, 0.014)) * tight
            sig_lat = float(rng.uniform(0.003, 0.011)) * tight
        else:
            sig_lon = float(rng.uniform(0.012, 0.058)) * tight
            sig_lat = float(rng.uniform(0.009, 0.038)) * tight
        base_per = max(12, total // max(n_clusters, 1))
        lo_n = max(10 if is_gc else 8, int(base_per * (0.55 if is_gc else 0.35)))
        hi_n = max(lo_n + 1, int(base_per * (2.4 if is_gc else 1.75)) + 1)
        n_here = int(rng.integers(lo_n, hi_n))
        n_here = max(8, min(int(n_here * (1.14 if not is_gc else 1.06)), remaining))
        if ci == n_clusters - 1:
            n_here = remaining
        lon_c = (cl_lon + rng.normal(0.0, sig_lon, size=n_here)) % (2.0 * np.pi)
        lat_c = cl_lat + rng.normal(0.0, sig_lat, size=n_here)
        if not is_gc:
            stream = float(rng.uniform(-0.014, 0.014))
            lat_c = lat_c + stream * np.sin((lon_c - cl_lon) * (2.3 + rng.uniform(-0.4, 0.4)))
        lat_c = np.tanh(lat_c / 1.12) * _MAX_EQUIRECT_LAT
        bv_c = np.clip(rng.normal(-0.04, 0.22, size=n_here), -0.35, 0.72)
        warm_giant = rng.random(n_here) < (0.14 if is_gc else 0.085)
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
        is_gc_flags.append(is_gc)
        remaining -= n_here

    if not lon_parts:
        return sample_star_catalog(
            rng,
            width,
            height,
            density_scale * 0.02,
            layer="background",
            galactic_band_boost=1.0,
            latitude_color_bias=False,
            attach_apparent_mag=attach_apparent_mag,
            photometry_mag_bright=photometry_mag_bright,
            photometry_mag_faint=photometry_mag_faint,
            photometry_slope=photometry_slope,
            photometry_ultra_cut=photometry_ultra_cut,
            max_ultra_bright_stars=max_ultra_bright_stars,
        )

    lon_a = np.concatenate(lon_parts)
    lat_a = np.concatenate(lat_parts)
    ci = np.concatenate(c_parts)
    si = np.concatenate(s_parts)
    ji = np.concatenate(j_parts, axis=0)
    bvv = np.concatenate(bv_parts)
    out: dict[str, np.ndarray] = {
        "lon": lon_a,
        "lat": lat_a,
        "color_idx": ci,
        "size_idx": si,
        "jitter": ji,
        "bv": bvv,
    }
    if attach_apparent_mag:
        n_here = lon_a.shape[0]
        if use_imf_magnitudes:
            mags = sample_apparent_magnitudes_lf(
                n_here,
                rng,
                mag_bright=photometry_mag_bright,
                mag_faint=photometry_mag_faint,
                giant_fraction=imf_giant_fraction,
                magnitude_ultra_cut=photometry_ultra_cut,
                max_ultra_bright_stars=max_ultra_bright_stars,
            )
        else:
            mags = sample_apparent_magnitudes(
                n_here,
                rng,
                mag_bright=photometry_mag_bright,
                mag_faint=photometry_mag_faint,
                magnitude_log_slope=photometry_slope,
                magnitude_ultra_cut=photometry_ultra_cut,
                max_ultra_bright_stars=max_ultra_bright_stars,
            )
        # Radial gradient: brighter toward cluster/GC center (lower apparent mag).
        mag_grad = np.zeros(n_here, dtype=np.float64)
        idx = 0
        for block_i, lon_block in enumerate(lon_parts):
            n_block = int(lon_block.shape[0])
            if n_block <= 0:
                continue
            sl = slice(idx, idx + n_block)
            is_gc = is_gc_flags[block_i] if block_i < len(is_gc_flags) else False
            lat_block = lat_parts[block_i]
            cl_lon = float(np.mean(lon_block))
            cl_lat = float(np.mean(lat_block))
            dlon = np.minimum(
                np.abs(lon_block - cl_lon), 2.0 * np.pi - np.abs(lon_block - cl_lon)
            )
            sig_lon = float(np.std(lon_block) + 0.008)
            sig_lat = float(np.std(lat_block) + 0.006)
            r2 = (dlon / sig_lon) ** 2 + ((lat_block - cl_lat) / sig_lat) ** 2
            amp = 1.35 if is_gc else 0.95
            mag_grad[sl] = -amp * np.exp(-r2 / (1.8 if is_gc else 2.4))
            idx += n_block
        mags = mags + mag_grad
        mags -= rng.uniform(0.18, 0.72, size=n_here)
        ml = float(min(photometry_mag_bright, photometry_mag_faint))
        mh = float(max(photometry_mag_bright, photometry_mag_faint))
        mags = np.clip(mags, ml, mh)
        out["phot_mag"] = mags
    return out


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
        color = color * 0.90 + neutral * 0.10
        color = color * np.array([0.96, 0.98, 1.02], dtype=np.float64)
    else:
        color = color * 0.94 + neutral * 0.06
    lu = float(np.mean(color))
    color = lu + (color - lu) * 1.22
    return np.clip(color, 0.0, 1.0)


def _paint_thin_diffraction_spikes(
    img: np.ndarray,
    x: int,
    y: int,
    radius: int,
    color_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    force: bool = False,
    flux_scale: float | None = None,
) -> None:
    """Rare thin 4-fold spikes (spider / refractor optics), not a second star population.

    When ``flux_scale`` is set (HDR star flux), spoke energy tracks the core so
    arms survive asinh display stretch; unit-RGB peaks alone wash out.
    """
    h, w, _ = img.shape
    peak = float(np.max(color_rgb))
    if peak < 0.22:
        return
    fs = float(flux_scale) if flux_scale is not None and flux_scale > 0.0 else None
    if radius >= 5:
        if not force and rng.random() >= 0.52:
            return
        # force + display-referred (no flux_scale): longer/brighter arms so spokes
        # survive on graded MW plates (HDR-scale amps wash out post-tonemap).
        _disp_force = bool(force and fs is None)
        L = int(
            rng.integers(
                36 if _disp_force else (16 if force else 14),
                64 if _disp_force else (32 if force else 26),
            )
        )
        if fs is not None:
            amp0 = fs * float(rng.uniform(0.006 if force else 0.0025, 0.014 if force else 0.006))
        else:
            # Display-referred force path needs bright additive arms — warm MW
            # continuum swallows HDR-scale or max-blend spokes.
            amp0 = peak * float(
                rng.uniform(1.6 if _disp_force else 0.022, 2.4 if _disp_force else 0.058)
            )
        _fall_pow = 0.62 if _disp_force else 1.05
        _fall_base = 0.22 if _disp_force else 0.45
    elif radius >= 4:
        if not force and rng.random() >= 0.14:
            return
        L = int(rng.integers(10 if force else 9, 20 if force else 18))
        if fs is not None:
            amp0 = fs * float(rng.uniform(0.004 if force else 0.0018, 0.010 if force else 0.0045))
        else:
            amp0 = peak * float(rng.uniform(0.035 if force else 0.012, 0.08 if force else 0.038))
        _fall_pow = 1.05
        _fall_base = 0.45
    else:
        if not force and (peak < 0.38 or rng.random() >= 0.065):
            return
        L = int(rng.integers(5, 11))
        if fs is not None:
            amp0 = fs * float(rng.uniform(0.002, 0.006))
        else:
            amp0 = peak * float(rng.uniform(0.018 if force else 0.007, 0.045 if force else 0.022))
        _fall_pow = 1.05
        _fall_base = 0.45

    # Unit spectral tint for chroma; energy lives in amp0.
    tint = color_rgb.astype(np.float64)
    tpeak = float(np.max(tint)) + 1e-8
    tint = tint / tpeak

    # Keep spokes inside the frame — truncated arms on y=0/h-1 read as border lines.
    edge_pad = 2
    edge_room = int(min(max(y - edge_pad, 0), max(h - 1 - edge_pad - y, 0)))
    L = int(min(L, max(2, edge_room)))
    if L < 2:
        return
    # Prefer near-axis crosses (classic spider); small jitter only when forced.
    if force and fs is None:
        ang = float(rng.uniform(-0.12, 0.12))
    else:
        ang = float(rng.uniform(0.0, 0.5 * np.pi))
    for k in range(4):
        th = ang + k * (0.5 * np.pi)
        cth, sth = float(np.cos(th)), float(np.sin(th))
        for t in range(1, L + 1):
            ox = int(round(t * cth))
            oy = int(round(t * sth))
            py, px = y + oy, (x + ox) % w
            if py < edge_pad or py >= h - edge_pad:
                continue
            fall = 1.0 / ((_fall_base + float(t)) ** _fall_pow)
            tip = float(t) / float(max(L, 1))
            chrom = np.array([0.96 - 0.05 * tip, 0.98 - 0.02 * tip, 1.0 + 0.10 * tip], dtype=np.float64)
            add = tint * (amp0 * fall) * chrom
            # Additive + 1px neighbor — must exceed continuum to read on MW plates.
            for npy, npx in ((py, px), (py, (px + 1) % w), (min(h - 1, py + 1), px)):
                if npy < edge_pad or npy >= h - edge_pad:
                    continue
                scale = 1.0 if (npy == py and npx == px) else 0.55
                img[npy, npx] = np.clip(img[npy, npx] + add * scale, 0.0, None)


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
    if w > 1:
        off_axis = abs((float(x) / float(max(w - 1, 1))) - 0.5) * 2.0
        coma_w = float(off_axis**1.35)
        if radius >= 3:
            asym_x *= 1.0 + 0.11 * coma_w
            asym_y *= 1.0 / (1.0 + 0.06 * coma_w)
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
                if radius >= 5:
                    halo_w *= 1.22
                elif radius == 4:
                    halo_w *= 1.10
                halo = np.exp(-(d**2) / (2 * halo_sigma**2)) * halo_w
            glow_jitter = 1.0 + rng.uniform(-glow_jitter_scale, glow_jitter_scale)
            base_term = base * 0.98
            denom = base_term + halo + 1e-9
            raw_int = (base_term + halo) * glow_jitter
            if d < core * 0.40:
                raw_int *= 1.06
            peak_cap = 3.6 if radius >= 5 else (2.5 if radius == 4 else (1.45 if radius == 3 else 1.0))
            intensity = float(np.clip(raw_int, 0.0, peak_cap))
            # Core stays chromatic; halo and mid-disk skew toward neutral white (brighter),
            # strongest in the radial middle of the glow — avoids a same-hue blue bloom.
            if d <= core:
                pixel_color = color_rgb
                luma_boost = 1.0
            else:
                h_frac = float(halo / denom)
                span = max(float(local_radius) - core, 1e-6)
                mid_u = float(np.clip((d - core) / span, 0.0, 1.0))
                mid_peak = 4.0 * mid_u * (1.0 - mid_u)
                white_w = float(
                    np.clip(0.03 + 0.14 * h_frac + 0.12 * mid_peak, 0.0, 0.28)
                )
                luma = float(np.clip(np.mean(color_rgb), 0.0, 1.0))
                neutral = np.full(3, min(1.0, luma * 1.07 + 0.025), dtype=np.float64)
                pixel_color = color_rgb * (1.0 - white_w) + neutral * white_w
                luma_boost = 1.0 + 0.085 * white_w
                if radius >= 4:
                    ca = float(mid_u**0.85)
                    pixel_color = pixel_color * np.array([1.0 + 0.038 * ca, 1.0, 1.0 - 0.032 * ca], dtype=np.float64)
                    pixel_color = np.clip(pixel_color, 0.0, 1.0)
            img[py, px] += pixel_color * intensity * luma_boost

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


def cull_catalog_to_plane(
    catalog: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    width: int,
    height: int,
    disk_weight: np.ndarray | None,
    gate_power: float = 2.35,
    keep_floor: float = 0.04,
    min_keep: int = 48,
) -> dict[str, np.ndarray]:
    """Stochastically drop high-latitude stars using disk_weight**power (cinematic plane gate)."""
    from starsky_gen.projections import sph_to_equirect_xy

    n = int(np.asarray(catalog["lon"]).shape[0])
    if n < 1:
        return catalog
    pw = float(np.clip(gate_power, 1.0, 4.5))
    floor = float(np.clip(keep_floor, 0.0, 0.35))
    xs, ys = sph_to_equirect_xy(catalog["lon"], catalog["lat"], int(width), int(height))
    yi = np.clip(ys.astype(np.int64), 0, int(height) - 1)
    xi = np.mod(xs.astype(np.int64), int(width))
    if disk_weight is not None:
        dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
        if dw.shape == (int(height), int(width)):
            host = dw[yi, xi]
        elif dw.ndim == 1 and dw.shape[0] == int(height):
            host = dw[yi]
        else:
            host = np.exp(-((np.asarray(catalog["lat"], dtype=np.float64) / 0.14) ** 2))
    else:
        host = np.exp(-((np.asarray(catalog["lat"], dtype=np.float64) / 0.14) ** 2))
    keep_p = np.clip(host**pw, floor, 1.0)
    keep = rng.random(n) < keep_p
    if int(np.sum(keep)) < int(min_keep) and n >= int(min_keep):
        # Always retain the brightest midplane candidates if cull was too aggressive.
        score = host * keep_p
        if "phot_mag" in catalog:
            score = score * np.clip(10.0 ** (-0.4 * (catalog["phot_mag"] - 12.0)), 0.2, 8.0)
        order = np.argsort(-score)
        keep = np.zeros(n, dtype=bool)
        keep[order[: int(min_keep)]] = True
    if bool(np.all(keep)):
        return catalog
    out: dict[str, np.ndarray] = {}
    for key, val in catalog.items():
        arr = np.asarray(val)
        if arr.shape[0] == n:
            out[key] = arr[keep]
        else:
            out[key] = val
    return out


def catalog_stats(catalog: dict[str, np.ndarray]) -> StarStats:
    cidx = bv_to_color_idx(catalog["bv"]) if "bv" in catalog else catalog["color_idx"]
    sidx = catalog["size_idx"]
    color_counts = {name: int(np.sum(cidx == i)) for i, name in enumerate(STAR_COLOR_NAMES)}
    size_counts = {name: int(np.sum(sidx == i)) for i, name in enumerate(STAR_SIZE_NAMES)}
    return StarStats(color_counts=color_counts, size_counts=size_counts)
