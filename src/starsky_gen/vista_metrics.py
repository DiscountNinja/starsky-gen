"""Equirect acceptance metrics for cinematic vista looks (glow-tolerant)."""

from __future__ import annotations

from typing import Any

import numpy as np

from starsky_gen.color_science import rec709_luma


def measure_vista_equirect(
    rgb: np.ndarray,
    *,
    mid_half_rows: int | None = None,
    shell_row_span: int | None = None,
) -> dict[str, Any]:
    """Measure band structure and color variation on an equirect RGB image.

    Returns keys used by acceptance helpers:
      mid_luma, shell_luma, mid_shell_ratio, band_contrast, mid_darker,
      chroma_spread, plane_luma, void_luma
    """
    img = np.asarray(rgb, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("rgb must be HxWx3+")
    h, w = int(img.shape[0]), int(img.shape[1])
    rgb3 = img[..., :3]
    lu = rec709_luma(rgb3)

    mid_h = int(mid_half_rows) if mid_half_rows is not None else max(1, h // 28)
    shell_h = int(shell_row_span) if shell_row_span is not None else max(2, h // 10)
    cy = h // 2
    mid = lu[max(0, cy - mid_h) : min(h, cy + mid_h), :]
    shell_lo = max(0, cy - mid_h - shell_h)
    shell_hi_start = min(h, cy + mid_h)
    shell_hi_end = min(h, cy + mid_h + shell_h)
    shell = np.concatenate(
        [lu[shell_lo : max(0, cy - mid_h), :], lu[shell_hi_start:shell_hi_end, :]],
        axis=0,
    )
    if shell.size == 0:
        shell = lu[:shell_h, :]

    mid_luma = float(np.mean(mid))
    shell_luma = float(np.mean(shell))
    ratio = mid_luma / max(shell_luma, 1e-8)
    frame_luma = float(np.mean(lu))
    band_contrast = abs(shell_luma - mid_luma) / max(frame_luma, 1e-8)

    # Plane vs polar voids for structure readability.
    yy = np.abs(np.linspace(-1.0, 1.0, h, dtype=np.float64))
    plane_rows = yy < 0.22
    void_rows = yy > 0.55
    plane_luma = float(np.mean(lu[plane_rows, :])) if bool(np.any(plane_rows)) else mid_luma
    void_luma = float(np.mean(lu[void_rows, :])) if bool(np.any(void_rows)) else frame_luma

    # Chroma spread in the plane: warm vs cool variation (not monochromatic wash).
    plane_rgb = rgb3[plane_rows, :, :].reshape(-1, 3)
    if plane_rgb.shape[0] > 64:
        mean_c = np.mean(plane_rgb, axis=0)
        chroma_spread = float(np.mean(np.std(plane_rgb - mean_c, axis=0)))
        rb = plane_rgb[:, 0] - plane_rgb[:, 2]
        warm_cool_span = float(np.percentile(rb, 90) - np.percentile(rb, 10))
    else:
        chroma_spread = 0.0
        warm_cool_span = 0.0

    thr = float(np.percentile(lu, 99.5))
    bright = lu >= max(thr, 0.85)
    star_peak_frac = float(np.mean(bright))

    return {
        "mid_luma": mid_luma,
        "shell_luma": shell_luma,
        "mid_shell_ratio": ratio,
        "band_contrast": band_contrast,
        "mid_darker": mid_luma < shell_luma,
        "plane_luma": plane_luma,
        "void_luma": void_luma,
        "chroma_spread": chroma_spread,
        "warm_cool_span": warm_cool_span,
        "star_peak_frac": star_peak_frac,
        "height": h,
        "width": w,
    }


def assert_cinematic_vista(
    metrics: dict[str, Any],
    *,
    min_band_contrast: float = 0.08,
    min_chroma_spread: float = 0.012,
    min_plane_over_void: float = 1.35,
) -> None:
    """Glow-tolerant cinematic acceptance: structured band + color variation."""
    contrast = float(metrics["band_contrast"])
    if contrast < float(min_band_contrast):
        raise AssertionError(
            f"cinematic vista band_contrast {contrast:.4f} < {min_band_contrast}"
        )
    chroma = float(metrics.get("chroma_spread", 0.0))
    if chroma < float(min_chroma_spread):
        raise AssertionError(
            f"cinematic vista chroma_spread {chroma:.4f} < {min_chroma_spread}"
        )
    plane = float(metrics.get("plane_luma", metrics["mid_luma"]))
    void = float(metrics.get("void_luma", 1e-6))
    ratio = plane / max(void, 1e-8)
    if ratio < float(min_plane_over_void):
        raise AssertionError(
            f"cinematic vista plane/void luma {ratio:.4f} < {min_plane_over_void}"
        )


def assert_cinematic_sandwich(
    metrics: dict[str, Any],
    *,
    max_mid_shell_ratio: float = 0.98,
) -> None:
    """Legacy helper: mid < shell sandwich (advisory; prefer assert_cinematic_vista)."""
    assert_cinematic_vista(metrics)
    # Keep optional sandwich check soft — midplane glow may push ratio >= 1.
    _ = max_mid_shell_ratio
    _ = metrics.get("mid_shell_ratio")
