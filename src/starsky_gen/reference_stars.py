"""Bright real stars in Galactic (l, b) [deg] for optional sky anchoring.

Longitude is mapped so Galactic center (l≈0) aligns with the renderer equirect
center (lon = π), matching the procedural bulge / dust layout.
RGB hints are linear display weights before `paint_star` accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starsky_gen.config import RenderConfig
from starsky_gen.projections import sph_to_equirect_xy
from starsky_gen.starfield import paint_star


@dataclass(frozen=True)
class GalacticAnchor:
    name: str
    l_deg: float
    b_deg: float
    v_mag: float
    rgb: tuple[float, float, float]


# Approximate Galactic II coordinates and V magnitudes (rounded).
GALACTIC_ANCHORS: tuple[GalacticAnchor, ...] = (
    GalacticAnchor("Sirius", 227.23, -8.90, -1.46, (0.88, 0.90, 1.0)),
    GalacticAnchor("Canopus", 261.01, -52.70, -0.74, (0.92, 0.94, 1.0)),
    GalacticAnchor("Rigil Kentaurus", 315.43, -0.20, -0.01, (0.95, 0.92, 0.88)),
    GalacticAnchor("Arcturus", 15.74, 69.11, -0.05, (0.98, 0.88, 0.62)),
    GalacticAnchor("Vega", 67.45, 19.24, 0.03, (0.82, 0.88, 1.0)),
    GalacticAnchor("Capella", 162.35, 4.28, 0.08, (1.0, 0.96, 0.86)),
    GalacticAnchor("Rigel", 209.75, -38.31, 0.12, (0.78, 0.86, 1.0)),
    GalacticAnchor("Procyon", 213.73, 13.00, 0.34, (0.94, 0.94, 1.0)),
    GalacticAnchor("Achernar", 290.84, -58.81, 0.45, (0.80, 0.88, 1.0)),
    GalacticAnchor("Betelgeuse", 199.50, -9.35, 0.42, (1.0, 0.62, 0.42)),
    GalacticAnchor("Hadar", 315.39, -2.99, 0.61, (0.88, 0.90, 1.0)),
    GalacticAnchor("Altair", 53.17, -8.90, 0.76, (0.90, 0.92, 1.0)),
    GalacticAnchor("Aldebaran", 180.97, -16.18, 0.85, (1.0, 0.78, 0.58)),
    GalacticAnchor("Spica", 316.97, 50.22, 0.97, (0.86, 0.90, 1.0)),
    GalacticAnchor("Antares", 359.90, +4.57, 0.91, (1.0, 0.58, 0.38)),
    GalacticAnchor("Pollux", 192.25, 23.26, 1.14, (1.0, 0.90, 0.72)),
    GalacticAnchor("Fomalhaut", 21.63, -64.89, 1.16, (0.90, 0.92, 1.0)),
    GalacticAnchor("Deneb", 91.02, +1.99, 1.25, (0.88, 0.90, 1.0)),
)


def galactic_lon_to_renderer_lon(l_deg: float) -> float:
    return (np.deg2rad(float(l_deg)) + np.pi) % (2.0 * np.pi)


def _radius_from_v(v_mag: float, rng: np.random.Generator) -> int:
    if v_mag < -0.9:
        return int(rng.integers(5, 9))
    if v_mag < 0.15:
        return int(rng.integers(4, 7))
    if v_mag < 1.05:
        return int(rng.integers(2, 5))
    return int(rng.integers(1, 4))


def paint_reference_anchors(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: RenderConfig,
) -> None:
    if not cfg.features.reference_anchors or not cfg.features.galaxy_view:
        return
    h, w = cfg.height, cfg.width
    for star in GALACTIC_ANCHORS:
        lon = galactic_lon_to_renderer_lon(star.l_deg)
        lat = float(np.clip(np.deg2rad(star.b_deg), -np.pi / 2.0 + 1e-4, np.pi / 2.0 - 1e-4))
        xs, ys = sph_to_equirect_xy(
            np.array([lon], dtype=np.float64),
            np.array([lat], dtype=np.float64),
            w,
            h,
        )
        radius = _radius_from_v(star.v_mag, rng)
        flux = float(0.38 * (10.0 ** (-0.4 * star.v_mag)))
        flux = float(np.clip(flux, 0.10, 1.45))
        color = np.array(star.rgb, dtype=np.float64) * flux
        color = np.clip(color, 0.0, 1.0)
        paint_star(img, int(xs[0]), int(ys[0]), radius, color, rng, galactic_lat=lat)
