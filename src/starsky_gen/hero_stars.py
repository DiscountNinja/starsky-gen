"""Procedural hero stars: select a few bright band anchors with lens FX."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starsky_gen.projections import sph_to_equirect_xy_float


@dataclass(frozen=True)
class HeroStarRecord:
    """Display-space hero for spikes / lens ghosts."""

    x: float
    y: float
    rgb: tuple[float, float, float]
    flux: float


def select_hero_indices(
    catalog: dict[str, np.ndarray],
    *,
    width: int,
    height: int,
    disk_weight: np.ndarray | None,
    count: int = 6,
    min_sep_frac: float = 0.10,
    off_plane_slots: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Greedy pick ``count`` heroes on band edges / just outside with wrap-aware x sep.

    Prefer rim score (soft plane × low razor midplane) over dense midplane core.
    Up to ``off_plane_slots`` may sit slightly further off-plane for depth.
    """
    n = int(np.asarray(catalog["lon"]).shape[0])
    k = int(max(0, min(count, n)))
    if k < 1:
        return np.zeros(0, dtype=np.int64)

    lon = np.asarray(catalog["lon"], dtype=np.float64)
    lat = np.asarray(catalog["lat"], dtype=np.float64)
    xf, yf = sph_to_equirect_xy_float(lon, lat, int(width), int(height))
    w = int(width)
    h = int(height)
    yi = np.clip(np.round(yf).astype(np.int64), 0, h - 1)
    xi = np.mod(np.round(xf).astype(np.int64), w)

    if "phot_mag" in catalog:
        mag = np.asarray(catalog["phot_mag"], dtype=np.float64)
        bright = np.clip(10.0 ** (-0.4 * (mag - float(np.median(mag)))), 0.05, 40.0)
    else:
        bright = np.ones(n, dtype=np.float64)

    dw = np.ones(n, dtype=np.float64)
    if disk_weight is not None:
        d2 = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
        if d2.shape == (h, w):
            dw = d2[yi, xi]
        elif d2.ndim == 1 and d2.shape[0] == h:
            dw = d2[yi]

    lat_abs = np.abs(lat)
    # Soft plane envelope vs razor midplane — rim peaks at band edges / just outside.
    soft_w = np.exp(-((lat_abs / 0.30) ** 2))
    razor_w = np.exp(-((lat_abs / 0.085) ** 2))
    rim_w = soft_w * (1.0 - 0.88 * razor_w)
    # Slight boost just outside the dense core (|lat| ~ 0.12–0.22).
    edge_bump = np.exp(-(((lat_abs - 0.16) / 0.10) ** 2))
    rim_w = np.clip(rim_w + 0.35 * edge_bump * soft_w, 0.0, 1.0)
    outer_w = np.exp(-((lat_abs / 0.42) ** 2)) * (1.0 - 0.70 * soft_w)

    host = 0.30 + 0.70 * dw
    score_rim = bright * host * (0.20 + 0.80 * rim_w)
    score_outer = bright * host * (0.25 + 0.75 * outer_w)

    order_rim = np.argsort(-score_rim)
    order_outer = np.argsort(-score_outer)
    min_sep = float(np.clip(min_sep_frac, 0.04, 0.25)) * float(w)
    chosen: list[int] = []
    chosen_x: list[float] = []

    def _try_add(idx: int) -> bool:
        x = float(xf[idx])
        for cx in chosen_x:
            dx = abs(x - cx)
            dx = min(dx, float(w) - dx)
            if dx < min_sep:
                return False
        chosen.append(int(idx))
        chosen_x.append(x)
        return True

    n_off = int(np.clip(off_plane_slots, 0, k))
    n_rim = k - n_off
    for idx in order_rim:
        if len(chosen) >= n_rim:
            break
        _try_add(int(idx))
    for idx in order_outer:
        if len(chosen) >= k:
            break
        if int(idx) in chosen:
            continue
        _try_add(int(idx))
    # Fill remaining from rim order if outer pool was sparse.
    for idx in order_rim:
        if len(chosen) >= k:
            break
        if int(idx) in chosen:
            continue
        _try_add(int(idx))

    if rng is not None and len(chosen) > 1:
        # Mild shuffle of selection order for uneven visual rhythm (positions fixed).
        arr = np.asarray(chosen, dtype=np.int64)
        rng.shuffle(arr)
        return arr
    return np.asarray(chosen, dtype=np.int64)


def attach_hero_flags(
    catalog: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Write boolean ``_hero`` mask onto catalog (length = star count)."""
    n = int(np.asarray(catalog["lon"]).shape[0])
    mask = np.zeros(n, dtype=bool)
    if indices.size > 0:
        valid = indices[(indices >= 0) & (indices < n)]
        mask[valid] = True
    catalog["_hero"] = mask
    return catalog


def hero_min_separation_ok(
    xs: np.ndarray,
    width: int,
    *,
    min_sep_frac: float = 0.08,
) -> bool:
    """True if all pairwise wrap-aware x separations meet the floor."""
    pts = np.asarray(xs, dtype=np.float64)
    m = int(pts.shape[0])
    if m < 2:
        return True
    min_sep = float(np.clip(min_sep_frac, 0.02, 0.4)) * float(width)
    for i in range(m):
        for j in range(i + 1, m):
            dx = abs(float(pts[i] - pts[j]))
            dx = min(dx, float(width) - dx)
            if dx < min_sep:
                return False
    return True
