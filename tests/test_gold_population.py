"""Regional gold vs young stellar population in the band."""

import numpy as np

from starsky_gen.color_science import (
    adjust_bv_for_population,
    adjust_teffective_for_population,
)
from starsky_gen.galactic_structure import (
    attach_stellar_population_to_catalog,
    build_galactic_morphology,
    build_population_placement_maps,
)


def test_core_gold_gated_by_population_field() -> None:
    m = build_galactic_morphology(64, 32, np.random.default_rng(8))
    assert float(m.gold_population_weight.min()) < 0.15
    assert float(m.gold_population_weight.max()) > 0.42


def test_population_placement_maps_old_cluster_young_scatter() -> None:
    m = build_galactic_morphology(96, 48, np.random.default_rng(11))
    scattered, clustered = build_population_placement_maps(m, gradient_strength=0.85)
    gp = m.gold_population_weight
    band = m.disk_weight > 0.12
    hi = gp > float(np.percentile(gp[band], 72))
    lo = gp < float(np.percentile(gp[band], 28))
    assert float(np.mean(clustered[hi])) > float(np.mean(clustered[lo])) * 1.12
    assert float(np.mean(scattered[lo])) > float(np.mean(scattered[hi])) * 0.92


def test_attach_population_adjusts_teff() -> None:
    m = build_galactic_morphology(64, 32, np.random.default_rng(3))
    rng = np.random.default_rng(3)
    n = 200
    catalog = {
        "lon": rng.uniform(0, 2 * np.pi, n),
        "lat": rng.uniform(-0.35, 0.35, n),
        "teff": np.full(n, 5200.0),
    }
    attach_stellar_population_to_catalog(catalog, m, 64, 32, rng)
    young = catalog["gold_population"] < 0.2
    assert float(np.mean(catalog["teff"][young])) > 5200.0


def test_adjust_teffective_young_hotter() -> None:
    rng = np.random.default_rng(1)
    teff = np.array([5500.0, 5600.0, 5800.0])
    gp = np.array([0.1, 0.15, 0.8])
    out = adjust_teffective_for_population(teff, gp, rng)
    assert float(out[0]) > 8000.0 or float(out[1]) > 8000.0


def test_adjust_bv_young_bluer_old_warmer() -> None:
    rng = np.random.default_rng(2)
    bv = np.array([0.35, 0.40, 0.55])
    gp = np.array([0.12, 0.18, 0.78])
    out = adjust_bv_for_population(bv, gp, rng)
    assert float(out[0]) < float(bv[0])
    assert float(out[2]) > float(bv[2])
