import numpy as np

from starsky_gen.postfx import (
    apply_depth_of_field,
    apply_display_contrast_finish,
    apply_global_s_curve,
    apply_split_toning,
    depth_map_from_disk,
)


def test_depth_map_and_dof_change_image() -> None:
    h, w = 64, 128
    rng = np.random.default_rng(3)
    disk_w = np.exp(-((np.linspace(-1, 1, h)[:, None] ** 2) / 0.4))
    depth = depth_map_from_disk(disk_w, h, w, rng, periodic_x=True)
    rgb = rng.random((h, w, 3)) * 0.4
    out = apply_depth_of_field(rgb, depth, strength=0.5, periodic_x=True)
    assert out.shape == rgb.shape
    assert float(np.mean(np.abs(out - rgb))) > 1e-5


def test_display_contrast_finish() -> None:
    rgb = np.full((8, 16, 3), 0.2)
    rgb[1, 8] = [0.48, 0.45, 0.42]
    disk = np.full((8, 16), 0.12)
    out = apply_display_contrast_finish(rgb, disk, black_point=0.04, white_point=0.9)
    assert float(out[1, 8, 0]) > float(rgb[1, 8, 0])


def test_s_curve_and_split_toning_bounded() -> None:
    rgb = np.linspace(0.05, 0.9, 48).reshape(4, 4, 3)
    curved = apply_global_s_curve(rgb, strength=0.3)
    toned = apply_split_toning(curved, strength=0.2)
    assert np.all(toned >= 0.0) and np.all(toned <= 1.0)
