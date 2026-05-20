"""Bulge layer and catalog_data tests."""

import numpy as np

from starsky_gen.bulge import render_bulge_layer
from starsky_gen.catalog_data import (
    load_catalog_subset,
    merge_catalog_positions,
)


def test_bulge_wider_at_center() -> None:
    layer = render_bulge_layer(256, 128, bulge_intensity=0.5, bulge_scale=0.22)
    cy = layer[64, 128]
    edge = layer[64, 32]
    assert float(cy.mean()) > float(edge.mean()) * 1.5


def test_bulge_layer_shape() -> None:
    layer = render_bulge_layer(128, 64, bulge_intensity=0.5)
    assert layer.shape == (64, 128, 3)
    assert float(np.max(layer)) > 0.0


def test_catalog_loader() -> None:
    cat = load_catalog_subset(42, max_stars=10)
    assert cat["lon"].shape[0] == 10
    assert "phot_mag" in cat


def test_catalog_merge() -> None:
    rng = np.random.default_rng(0)
    proc = {
        "lon": rng.uniform(0, 6.28, 20),
        "lat": rng.normal(0, 0.2, 20),
        "phot_mag": rng.uniform(8, 14, 20),
    }
    cat = load_catalog_subset(1, max_stars=5)
    merged = merge_catalog_positions(cat, proc, rng, 0.5)
    assert merged["lon"].shape[0] == 20
