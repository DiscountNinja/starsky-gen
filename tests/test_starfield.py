import numpy as np

from starsky_gen.starfield import STAR_COLOR_WEIGHTS, STAR_SIZE_WEIGHTS, sample_star_catalog


def test_star_distribution_close_to_target() -> None:
    rng = np.random.default_rng(1234)
    catalog = sample_star_catalog(
        rng, width=2000, height=1000, density_scale=1.0, latitude_color_bias=False
    )

    color_counts = np.bincount(catalog["color_idx"], minlength=4)
    size_counts = np.bincount(catalog["size_idx"], minlength=4)
    color_freq = color_counts / color_counts.sum()
    size_freq = size_counts / size_counts.sum()

    assert np.all(np.abs(color_freq - STAR_COLOR_WEIGHTS) < 0.03)
    assert np.all(np.abs(size_freq - STAR_SIZE_WEIGHTS) < 0.03)
