import numpy as np

from starsky_gen.starfield import (
    STAR_COLOR_WEIGHTS,
    STAR_SIZE_WEIGHTS,
    cull_catalog_to_plane,
    sample_star_catalog,
)


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


def test_cull_catalog_to_plane_drops_high_latitude() -> None:
    rng = np.random.default_rng(7)
    n = 400
    lat = np.concatenate(
        [
            rng.normal(0.0, 0.06, size=n // 2),
            rng.choice([-1.0, 1.0], size=n // 2) * rng.uniform(0.35, 0.9, size=n // 2),
        ]
    )
    cat = {
        "lon": rng.uniform(0.0, 2.0 * np.pi, size=n),
        "lat": lat,
        "phot_mag": rng.uniform(8.0, 16.0, size=n),
        "size_idx": np.zeros(n, dtype=np.int64),
    }
    h, w = 128, 256
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    dw = np.broadcast_to(np.exp(-(yy**2) / 0.10), (h, w)).copy()
    out = cull_catalog_to_plane(
        cat, rng, width=w, height=h, disk_weight=dw, gate_power=2.45, keep_floor=0.04, min_keep=40
    )
    assert out["lat"].shape[0] < n
    assert out["lat"].shape[0] >= 40
    assert float(np.mean(np.abs(out["lat"]))) < float(np.mean(np.abs(lat)))
