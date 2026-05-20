import numpy as np

from starsky_gen.placement import build_cluster_density_modulation, build_equirect_density_map
from starsky_gen.postfx import apply_tri_scale_bloom


def test_cluster_map_modulates_density() -> None:
    rng = np.random.default_rng(7)
    h, w = 128, 256
    base = build_equirect_density_map(w, h)
    cluster = build_cluster_density_modulation(h, w, rng, strength=0.8)
    mod = build_equirect_density_map(w, h, cluster_map=cluster)
    assert mod.shape == (h, w)
    peak_ratio = float(np.max(mod)) / max(float(np.mean(mod)), 1e-6)
    base_ratio = float(np.max(base)) / max(float(np.mean(base)), 1e-6)
    assert peak_ratio > base_ratio * 1.08


def test_tri_scale_bloom_threshold() -> None:
    rgb = np.zeros((32, 64, 3), dtype=np.float64)
    rgb[16, 32] = [0.98, 0.95, 0.9]
    disk = np.exp(-((np.linspace(-1, 1, 32)[:, None] ** 2) / 0.5))
    out_lo = apply_tri_scale_bloom(rgb, disk, strength=0.2, threshold=0.95)
    out_hi = apply_tri_scale_bloom(rgb, disk, strength=0.2, threshold=0.70)
    assert float(np.mean(out_lo)) < float(np.mean(out_hi))
