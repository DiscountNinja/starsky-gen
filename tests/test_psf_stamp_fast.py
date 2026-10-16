"""Vectorized stamp_psf_patch stays bit-identical to the meshgrid reference."""

from __future__ import annotations

import numpy as np

from starsky_gen.psf import (
    MoffatParams,
    normalized_moffat_kernel_cached,
    stamp_psf_patch,
)


def _stamp_meshgrid_reference(
    canvas: np.ndarray,
    yi: int,
    xi: int,
    kern: np.ndarray,
    amplitude_rgb: np.ndarray,
    *,
    periodic_x: bool,
    subpixel_dx: float = 0.0,
    subpixel_dy: float = 0.0,
) -> None:
    ampl = np.ascontiguousarray(amplitude_rgb, dtype=np.float64).reshape((1, 1, -1))
    patch = kern[..., np.newaxis] * ampl
    h, wd, _cc = canvas.shape
    kh, kw = kern.shape
    cy = kh // 2
    cx = kw // 2
    iy0 = float(yi - cy) + float(subpixel_dy)
    ix0 = float(xi - cx) + float(subpixel_dx)
    yi_row = iy0 + np.arange(kh, dtype=np.float64)
    xi_col = ix0 + np.arange(kw, dtype=np.float64)
    yy_grid, xx_grid = np.meshgrid(yi_row, xi_col, indexing="ij")
    vy = (yy_grid >= 0) & (yy_grid < h)
    yy_i = np.clip(np.round(yy_grid).astype(np.int64), 0, h - 1)
    if periodic_x:
        xp_grid = np.mod(np.round(xx_grid).astype(np.int64), wd)
        ii, jj = np.nonzero(vy)
        canvas[yy_i[ii, jj], xp_grid[ii, jj]] += patch[ii, jj]
    else:
        sel = vy & (xx_grid >= 0) & (xx_grid < wd)
        ii, jj = np.nonzero(sel)
        canvas[yy_i[ii, jj], np.clip(np.round(xx_grid[ii, jj]).astype(np.int64), 0, wd - 1)] += patch[
            ii, jj
        ]


def test_stamp_psf_patch_matches_meshgrid_reference() -> None:
    params = MoffatParams(sigma_x=1.15, sigma_y=1.05, theta=0.35, beta=3.0)
    kern = normalized_moffat_kernel_cached(7, params)
    rgb = np.array([1.1, 0.95, 0.8], dtype=np.float64)
    rng = np.random.default_rng(9)
    for periodic in (True, False):
        h, w = (128, 256) if periodic else (96, 96)
        ref = np.zeros((h, w, 3), dtype=np.float64)
        out = np.zeros((h, w, 3), dtype=np.float64)
        ys = rng.integers(0, h, 400)
        xs = rng.integers(0, w, 400)
        dx = rng.uniform(-0.49, 0.49, 400)
        dy = rng.uniform(-0.49, 0.49, 400)
        for yi, xi, ddx, ddy in zip(ys, xs, dx, dy):
            _stamp_meshgrid_reference(
                ref,
                int(yi),
                int(xi),
                kern,
                rgb,
                periodic_x=periodic,
                subpixel_dx=float(ddx),
                subpixel_dy=float(ddy),
            )
            stamp_psf_patch(
                out,
                int(yi),
                int(xi),
                kern,
                rgb,
                periodic_x=periodic,
                subpixel_dx=float(ddx),
                subpixel_dy=float(ddy),
            )
        assert np.allclose(ref, out)
        assert float(np.max(np.abs(ref - out))) == 0.0


def test_noheroes_cli_flag_disables_heroes() -> None:
    from starsky_gen.config import FeatureConfig, LookTarget, RenderConfig, finalize_render_config

    cfg = RenderConfig(
        features=FeatureConfig(look_target=LookTarget.cinematic, hero_star_enabled=False)
    )
    out = finalize_render_config(cfg).features
    assert out.hero_star_enabled is False
