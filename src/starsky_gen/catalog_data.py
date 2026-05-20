"""Optional catalog subsample and luminance overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Synthetic subsample mimicking Gaia-like density (l deg, b deg, G mag, bp-rp)
# Small bundled set for reproducibility without large data files.
_CATALOG_L_DEG = np.array(
    [
        0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330,
        5, 25, 85, 175, 265, 355, 12, 98, 188, 278, 8, 42, 128, 218, 308,
    ],
    dtype=np.float64,
)
_CATALOG_B_DEG = np.array(
    [
        0, 2, -1, 3, -2, 1, 0, -3, 2, 1, -1, 0, 2, -2,
        5, -4, 6, -5, 4, -3, 8, -6, 7, -4, 12, -8, 10, -9, 6,
    ],
    dtype=np.float64,
)
_CATALOG_G = np.array(
    [
        5.5, 7.2, 8.1, 6.8, 9.0, 7.5, 8.8, 9.5, 6.2, 7.8, 8.4, 7.0, 9.2, 8.6,
        6.0, 8.0, 7.3, 9.1, 7.6, 8.2, 5.8, 8.5, 7.9, 9.3, 6.5, 8.7, 7.4, 9.0, 8.1,
    ],
    dtype=np.float64,
)
_CATALOG_BPRP = np.array(
    [
        0.2, 0.5, 0.8, 0.3, 1.0, 0.4, 0.6, 0.9, 0.25, 0.55, 0.7, 0.35, 0.85, 0.65,
        0.15, 0.75, 0.45, 0.95, 0.5, 0.6, 0.1, 0.8, 0.55, 0.9, 0.2, 0.7, 0.4, 0.85, 0.6,
    ],
    dtype=np.float64,
)


def _l_deg_to_lon_rad(l_deg: float) -> float:
    """Galactic l [deg] → renderer lon (GC at π)."""
    return float((l_deg + 180.0) % 360.0) * np.pi / 180.0


def _b_deg_to_lat_rad(b_deg: float) -> float:
    return float(b_deg) * np.pi / 180.0


def load_catalog_subset(
    seed: int,
    *,
    max_stars: int = 200,
) -> dict[str, np.ndarray]:
    """Return lon, lat, mag, teff arrays from bundled subsample."""
    rng = np.random.default_rng(seed)
    n = min(max_stars, _CATALOG_L_DEG.size)
    idx = rng.choice(_CATALOG_L_DEG.size, size=n, replace=n > _CATALOG_L_DEG.size)
    lon = np.array([_l_deg_to_lon_rad(float(_CATALOG_L_DEG[i])) for i in idx])
    lat = np.array([_b_deg_to_lat_rad(float(_CATALOG_B_DEG[i])) for i in idx])
    mag = _CATALOG_G[idx].astype(np.float64)
    bprp = _CATALOG_BPRP[idx].astype(np.float64)
    teff = 4600.0 / (0.92 + bprp) + 2600.0
    return {"lon": lon, "lat": lat, "phot_mag": mag, "teff": teff}


def fit_procedural_stats_from_catalog(
    catalog: dict[str, np.ndarray],
) -> tuple[float, float]:
    """Return (magnitude_log_slope hint, mean_bv hint)."""
    mags = catalog.get("phot_mag")
    if mags is None or mags.size < 8:
        return 0.6, 0.65
    hist, edges = np.histogram(mags, bins=12)
    centers = 0.5 * (edges[1:] + edges[:-1])
    valid = hist > 0
    if valid.sum() < 3:
        return 0.6, 0.65
    slope = 0.55 + 0.02 * float(np.std(mags))
    teff = catalog.get("teff")
    if teff is not None:
        bv_hint = float(np.clip(2.7 - np.log10(np.mean(teff)) * 0.85, 0.3, 1.0))
    else:
        bv_hint = 0.65
    return float(np.clip(slope, 0.35, 0.95)), bv_hint


def merge_catalog_positions(
    catalog: dict[str, np.ndarray],
    procedural: dict[str, np.ndarray],
    rng: np.random.Generator,
    blend: float,
) -> dict[str, np.ndarray]:
    """Replace a fraction of procedural bright stars with catalog entries."""
    b = float(np.clip(blend, 0.0, 1.0))
    if b <= 0 or catalog["lon"].size == 0:
        return procedural
    n_cat = min(int(procedural["lon"].shape[0] * b), catalog["lon"].shape[0])
    if n_cat < 1:
        return procedural
    out = {k: v.copy() for k, v in procedural.items()}
    pick_p = rng.choice(procedural["lon"].shape[0], size=n_cat, replace=False)
    pick_c = rng.choice(catalog["lon"].shape[0], size=n_cat, replace=False)
    out["lon"][pick_p] = catalog["lon"][pick_c]
    out["lat"][pick_p] = catalog["lat"][pick_c]
    if "phot_mag" in catalog and "phot_mag" in out:
        out["phot_mag"][pick_p] = catalog["phot_mag"][pick_c]
    if "teff" in catalog:
        if "teff" not in out:
            out["teff"] = np.full(procedural["lon"].shape[0], 5500.0)
        out["teff"][pick_p] = catalog["teff"][pick_c]
    return out


def load_luminance_overlay(path: Path, width: int, height: int) -> np.ndarray | None:
    """Load grayscale luminance map resized to frame."""
    if not path.is_file():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    img = Image.open(path).convert("L")
    img = img.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return np.clip(arr, 0.0, 1.0)
