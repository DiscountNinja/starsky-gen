"""Final galactic band color grade (warm ISM, dust lanes, H II)."""

import numpy as np

from starsky_gen.color_science import rec709_luma
from starsky_gen.galactic_band_color import apply_galactic_band_color_grade


def test_band_grade_warms_cool_blue_diffuse() -> None:
    h, w = 48, 96
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.12)), (h, w))
    canvas = np.zeros((h, w, 3), dtype=np.float64)
    canvas[..., 0] = 0.18
    canvas[..., 1] = 0.20
    canvas[..., 2] = 0.34
    canvas *= disk[..., None]
    ext = np.clip(0.12 + disk * 0.55, 0.08, 0.92)
    out = apply_galactic_band_color_grade(
        canvas,
        disk,
        ext,
        dust_absorption=1.0 - ext,
        strength=0.9,
        rng=np.random.default_rng(1),
    )
    on = disk > 0.35
    rin = canvas[on]
    rout = out[on]
    assert float(np.mean(rout[:, 2])) < float(np.mean(rin[:, 2])) * 0.92
    assert float(np.mean(rout[:, 0])) > float(np.mean(rin[:, 0])) * 1.04


def test_band_grade_darkens_extinction_lanes() -> None:
    h, w = 32, 64
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.10)), (h, w))
    lu = 0.42
    canvas = np.full((h, w, 3), lu * 0.95, dtype=np.float64)
    canvas *= disk[..., None]
    ext = np.ones((h, w), dtype=np.float64) * 0.7
    ext[:, w // 3 : 2 * w // 3] = 0.06
    dust = np.zeros((h, w), dtype=np.float64)
    dust[:, w // 3 : 2 * w // 3] = 0.85
    out = apply_galactic_band_color_grade(
        canvas,
        disk,
        ext,
        dust_absorption=dust,
        void_mask=dust,
        latent_turb=dust,
        strength=1.0,
        rng=np.random.default_rng(2),
    )
    lane = (disk > 0.3) & (ext < 0.15)
    clear = (disk > 0.3) & (ext > 0.5)
    assert float(rec709_luma(out[lane]).mean()) < float(rec709_luma(canvas[lane]).mean()) * 0.82
    assert float(np.mean(out[lane, 2])) < float(np.mean(out[lane, 0])) * 0.85


def test_band_grade_turbulent_black_dust() -> None:
    h, w = 48, 96
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.10)), (h, w))
    lu = 0.38
    canvas = np.full((h, w, 3), lu * 0.92, dtype=np.float64)
    canvas *= disk[..., None]
    ext = np.ones((h, w), dtype=np.float64) * 0.65
    ext[:, w // 4 : 3 * w // 4] = 0.05
    dust = np.zeros((h, w), dtype=np.float64)
    dust[:, w // 4 : 3 * w // 4] = 0.92
    void = dust.copy()
    out = apply_galactic_band_color_grade(
        canvas,
        disk,
        ext,
        dust_absorption=dust,
        void_mask=void,
        latent_turb=void,
        strength=1.0,
        dust_black_strength=1.0,
        rng=np.random.default_rng(6),
    )
    lane = (disk > 0.3) & (ext < 0.12)
    assert float(rec709_luma(out[lane]).mean()) < lu * 0.42


def test_structure_lane_weight_prefers_texture_over_smooth_blob() -> None:
    from starsky_gen.galactic_band_color import _structure_lane_weight
    from starsky_gen.nebula import _blur_separable_xy

    h, w = 32, 64
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    plane = np.ones((h, w), dtype=np.float64) * 0.85
    lane = np.broadcast_to(np.exp(-(xx**2) / 0.12), (h, w))
    gas_smooth = lane * 0.92
    rng = np.random.default_rng(7)
    gas_tex = np.clip(gas_smooth + (rng.random((h, w)) - 0.5) * lane * 0.55, 0.0, 1.0)
    kw = dict(
        lane_dark=lane,
        plane=plane,
        blur_fn=_blur_separable_xy,
        periodic_x=True,
        inner_bias=0.55,
        xx=xx,
    )
    w_smooth = _structure_lane_weight(gas_smooth, **kw)
    w_tex = _structure_lane_weight(gas_tex, **kw)
    mid = slice(w // 2 - 4, w // 2 + 5)
    assert float(w_tex[:, mid].mean()) > float(w_smooth[:, mid].mean()) * 1.05


def test_display_microstructure_adds_hf() -> None:
    from starsky_gen.galactic_band_color import apply_band_display_microstructure

    h, w = 48, 96
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.10)), (h, w))
    canvas = np.full((h, w, 3), 0.32, dtype=np.float64)
    canvas *= disk[..., None]
    ext = np.clip(0.08 + disk * 0.55, 0.05, 0.92)
    rng = np.random.default_rng(11)
    ext = np.clip(ext + (rng.random((h, w)) - 0.5) * 0.18 * disk, 0.04, 0.95)
    out = apply_band_display_microstructure(
        canvas,
        ext,
        disk,
        strength=1.1,
        periodic_x=True,
    )
    on = disk > 0.35
    from starsky_gen.procedural_noise import gaussian_blur_pil

    lu_in = rec709_luma(canvas)
    lu_out = rec709_luma(out)
    hf_in = lu_in - gaussian_blur_pil(lu_in, 1.2, periodic_x=True)
    hf_out = lu_out - gaussian_blur_pil(lu_out, 1.2, periodic_x=True)
    assert float(np.std(hf_out[on])) > float(np.std(hf_in[on])) * 1.05


def test_band_luma_separation_expands_local_contrast() -> None:
    from starsky_gen.galactic_band_color import apply_band_luma_separation
    from starsky_gen.procedural_noise import gaussian_blur_pil

    h, w = 48, 96
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.10)), (h, w))
    canvas = np.full((h, w, 3), 0.30, dtype=np.float64) * disk[..., None]
    ext = np.clip(0.1 + disk * 0.5, 0.06, 0.9)
    rng = np.random.default_rng(12)
    ext = np.clip(ext + (rng.random((h, w)) - 0.5) * 0.22 * disk, 0.05, 0.95)
    out = apply_band_luma_separation(canvas, ext, disk, strength=1.15, periodic_x=True)
    on = disk > 0.35
    lu = rec709_luma(canvas)
    lo = rec709_luma(out)
    med = gaussian_blur_pil(lu, 1.2, periodic_x=True)
    hf_in = lu - med
    hf_out = lo - gaussian_blur_pil(lo, 1.2, periodic_x=True)
    assert float(np.std(hf_out[on])) > float(np.std(hf_in[on])) * 1.08


def test_band_grade_preserves_blacks() -> None:
    h, w = 32, 64
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.10)), (h, w))
    canvas = np.zeros((h, w, 3), dtype=np.float64)
    canvas[8:24, 10:54, 0] = 0.02
    canvas[8:24, 10:54, 1] = 0.018
    canvas[8:24, 10:54, 2] = 0.025
    ext = np.clip(0.05 + disk * 0.4, 0.04, 0.9)
    out = apply_galactic_band_color_grade(
        canvas,
        disk,
        ext,
        dust_absorption=1.0 - ext,
        void_mask=1.0 - ext,
        strength=1.0,
        rng=np.random.default_rng(4),
    )
    dark = rec709_luma(canvas) < 0.04
    assert float(np.max(rec709_luma(out[dark]))) < 0.04


def test_band_grade_sparse_hii_not_broad_sf() -> None:
    h, w = 40, 80
    yy = np.linspace(-1, 1, h, dtype=np.float64)[:, None]
    disk = np.broadcast_to(np.exp(-((yy**2) / 0.12)), (h, w))
    canvas = np.full((h, w, 3), 0.28, dtype=np.float64)
    canvas *= disk[..., None]
    ext = np.clip(0.2 + disk * 0.5, 0.1, 0.9)
    sf = np.clip(disk * 0.75 + 0.12, 0.0, 1.0)
    out = apply_galactic_band_color_grade(
        canvas,
        disk,
        ext,
        star_formation=sf,
        strength=0.95,
        hii_strength=0.25,
        rng=np.random.default_rng(5),
    )
    on = disk > 0.35
    red_excess = out[..., 0] - (out[..., 1] + out[..., 2]) * 0.5
    assert float(np.mean(red_excess[on])) < 0.05


def test_band_grade_preserves_bright_points() -> None:
    h, w = 32, 48
    disk = np.ones((h, w), dtype=np.float64) * 0.8
    canvas = np.full((h, w, 3), 0.25, dtype=np.float64)
    canvas[16, 24] = [0.95, 0.92, 0.88]
    ext = np.ones((h, w), dtype=np.float64) * 0.5
    out = apply_galactic_band_color_grade(canvas, disk, ext, strength=0.85, rng=np.random.default_rng(3))
    np.testing.assert_allclose(out[16, 24], canvas[16, 24], rtol=0.12, atol=0.08)
