"""Authoritative Milky Way sandwich plan for composite stages.

BandPlan is the single structure authority used by dark rifts, post-star lane
protect, and debug exports. Morphology still builds the underlying maps;
composite stages should not reinvent lane/shell/rift geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from starsky_gen.nebula import _blur_separable_xy, _resize_bilinear

if TYPE_CHECKING:
    from starsky_gen.galactic_structure import GalacticMorphology


@dataclass
class BandPlan:
    """Wrap-safe per-pixel sandwich maps (H×W, float64 in [0,1] unless noted)."""

    disk_w: np.ndarray
    lane_weight: np.ndarray
    shell_weight: np.ndarray
    rift_mask: np.ndarray
    opaque: np.ndarray
    host: np.ndarray
    protect_mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.disk_w.shape[0]), int(self.disk_w.shape[1])


def _as_hw(field: np.ndarray, height: int, width: int, *, periodic_x: bool) -> np.ndarray:
    a = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
    if a.ndim == 1:
        a = np.broadcast_to(a.reshape(height, 1), (height, width))
    elif a.shape != (height, width):
        a = _resize_bilinear(a, height, width, periodic_x=periodic_x)
    return np.asarray(a, dtype=np.float64)


def build_band_plan(
    *,
    height: int,
    width: int,
    disk_w: np.ndarray,
    ext_paint: np.ndarray,
    galactic: GalacticMorphology | None = None,
    rift_mask: np.ndarray | None = None,
    dust_lane_weight: np.ndarray | None = None,
    puff_shell_weight: np.ndarray | None = None,
    periodic_x: bool = True,
) -> BandPlan:
    """Build sandwich authority from morphology + finalized extinction (+ optional rifts)."""
    dw = _as_hw(disk_w, height, width, periodic_x=periodic_x)
    opaque = np.clip(1.0 - _as_hw(ext_paint, height, width, periodic_x=periodic_x), 0.0, 1.0)

    yy = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
    mid_gate = np.broadcast_to(np.exp(-(yy**2) / 0.035), (height, width))

    arch = galactic.structure_architecture if galactic is not None else None
    if dust_lane_weight is not None:
        lane = _as_hw(dust_lane_weight, height, width, periodic_x=periodic_x)
    elif arch is not None and getattr(arch, "dust_lane_weight", None) is not None:
        lane = _as_hw(arch.dust_lane_weight, height, width, periodic_x=periodic_x)
    else:
        lane = mid_gate * dw

    if puff_shell_weight is not None:
        shell = _as_hw(puff_shell_weight, height, width, periodic_x=periodic_x)
    elif arch is not None and getattr(arch, "puff_shell_weight", None) is not None:
        shell = _as_hw(arch.puff_shell_weight, height, width, periodic_x=periodic_x)
    else:
        shell = np.clip(dw * (1.0 - mid_gate * 0.85), 0.0, 1.0)

    if arch is not None and getattr(arch, "band_host", None) is not None:
        host = _as_hw(arch.band_host, height, width, periodic_x=periodic_x)
        host = np.clip(host / max(float(np.max(host)), 1e-6), 0.0, 1.0)
    else:
        host = dw

    if rift_mask is None:
        rift = np.zeros((height, width), dtype=np.float64)
    else:
        rift = _as_hw(rift_mask, height, width, periodic_x=periodic_x)

    # Protect mask: lane/midplane × opacity, with rifts winning locally.
    # Soft shoulders (multi-pass blur + gamma lift) so rifts read volumetric, not binary.
    opaque_boost = np.clip(opaque**0.72, 0.0, 1.0)
    protect = np.clip(
        np.maximum(lane**0.85, mid_gate * 0.82) * opaque_boost * (0.55 + 0.45 * dw),
        0.0,
        1.0,
    )
    protect = np.clip(np.maximum(protect, rift * 1.05), 0.0, 1.0)
    protect = np.clip(protect**0.65, 0.0, 1.0)
    protect = _blur_separable_xy(protect, passes=3, periodic_x=periodic_x)
    rift_soft = _blur_separable_xy(np.clip(rift**0.70, 0.0, 1.0), passes=2, periodic_x=periodic_x)

    return BandPlan(
        disk_w=dw.astype(np.float64),
        lane_weight=np.clip(lane, 0.0, 1.0).astype(np.float64),
        shell_weight=np.clip(shell, 0.0, 1.0).astype(np.float64),
        rift_mask=np.clip(rift_soft, 0.0, 1.0).astype(np.float64),
        opaque=opaque.astype(np.float64),
        host=np.clip(host, 0.0, 1.0).astype(np.float64),
        protect_mask=np.clip(protect, 0.0, 1.0).astype(np.float64),
    )


def apply_band_plan_lane_protect(
    canvas: np.ndarray,
    plan: BandPlan,
    *,
    strength: float,
    hero_luma_floor: float = 0.90,
) -> np.ndarray:
    """Re-darken BandPlan protect mask after star composite; spare brightest heroes.

    Strength is intentionally softer when midplane glow is allowed — protect keeps
    structure without crushing core emission.
    """
    from starsky_gen.color_science import rec709_luma

    s = float(np.clip(strength, 0.0, 1.35))
    if s < 1e-6:
        return canvas
    mask = np.clip(plan.protect_mask**0.85, 0.0, 1.0)
    mask = np.clip(np.maximum(mask, plan.rift_mask * 0.92), 0.0, 1.0)
    lu = rec709_luma(canvas)
    hero = np.clip((lu - hero_luma_floor) / max(1.0 - hero_luma_floor, 0.05), 0.0, 1.0) ** 1.35
    protect = 1.0 - 0.45 * hero
    darken = np.clip(mask * s * protect, 0.0, 0.88)
    out = np.maximum(canvas.astype(np.float64) * (1.0 - darken[..., np.newaxis]), 0.0)
    return out.astype(canvas.dtype, copy=False)
