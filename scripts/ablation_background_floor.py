#!/usr/bin/env python3
"""Ablation: which pipeline pass raises the sky background floor? (seed 42, 512x256)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starsky_gen.color_science import rec709_luma  # noqa: E402
from starsky_gen.config import (  # noqa: E402
    FeatureConfig,
    NebulaMode,
    NebulaTuningConfig,
    OutputFormat,
    ProjectionMode,
    RenderConfig,
    RenderProfile,
)
from starsky_gen.generator import render_single  # noqa: E402

LOG_PATH = ROOT / ".cursor" / "debug-408793.log"
OUT_DIR = ROOT / "output" / "ablation_floor"
WIDTH, HEIGHT, SEED = 512, 256, 42


def _sky_band_stats(rgb: np.ndarray) -> dict[str, float]:
    im = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    h = im.shape[0]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float64)
    dw = np.exp(-((y / 0.22) ** 2))[:, None] * np.ones((1, im.shape[1]))
    lu = rec709_luma(im)
    sky = dw[:, 0] < 0.15
    band = dw[:, 0] > 0.35
    out: dict[str, float] = {
        "frame_min": float(np.min(lu)),
        "frame_p50": float(np.percentile(lu, 50)),
        "frame_max": float(np.max(lu)),
    }
    if bool(np.any(sky)):
        s = lu[sky]
        out.update(
            {
                "sky_min": float(np.min(s)),
                "sky_p50": float(np.percentile(s, 50)),
                "sky_p95": float(np.percentile(s, 95)),
                "sky_mean": float(np.mean(s)),
            }
        )
    if bool(np.any(band)):
        b = lu[band]
        out.update(
            {
                "band_p50": float(np.percentile(b, 50)),
                "band_p95": float(np.percentile(b, 95)),
            }
        )
    return out


def _base_features() -> FeatureConfig:
    return FeatureConfig(
        stars=True,
        nebula=True,
        galaxy_view=True,
        background_gradient=True,
        black_background=False,
        render_profile=RenderProfile.full,
        sensor_noise_stage="off",
    )


def _run_variant(name: str, overrides: dict) -> dict:
    feat = _base_features().model_copy(update=overrides)
    cfg = RenderConfig(
        width=WIDTH,
        height=HEIGHT,
        seed=SEED,
        output_dir=OUT_DIR,
        projection=ProjectionMode.equirectangular,
        output_format=OutputFormat.png,
        nebula_mode=NebulaMode.galaxy_streak,
        features=feat,
        nebula=NebulaTuningConfig(),
    )
    os.environ["STK_ABLATION"] = "1"
    os.environ["STK_DEBUG_RUN"] = name
    paths, _ = render_single(cfg, 0)
    im = np.asarray(Image.open(paths["equirectangular"])) / 255.0
    stats = _sky_band_stats(im)
    stats["variant"] = name
    return stats


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants: list[tuple[str, dict]] = [
        ("baseline", {}),
        ("stars_off", {"stars": False}),
        ("mesoscale_off", {"disk_mesoscale_thickness_strength": 0.0}),
        ("microstructure_off", {"morphology_dust_micro_strength": 0.0}),
        ("dark_patch_off", {"band_dark_patch_strength": 0.0}),
        ("late_hii_off", {"off_band_late_composite": False, "off_band_emission_strength": 0.0}),
        (
            "emit_minimal",
            {
                "morphology_nebula_emit_scale": 0.0,
                "band_nebula_radiance_scale": 0.85,
                "nebula_haze_strength": 0.0,
                "off_band_emission_strength": 0.0,
            },
        ),
        ("black_bg", {"black_background": True}),
    ]
    rows: list[dict] = []
    for name, overrides in variants:
        print(f"Running {name}...", flush=True)
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        t0 = time.time()
        try:
            stats = _run_variant(name, overrides)
            stats["elapsed_s"] = round(time.time() - t0, 1)
            rows.append(stats)
            print(f"  sky_p50={stats.get('sky_p50', 0):.4f} sky_mean={stats.get('sky_mean', 0):.4f}")
        except Exception as exc:
            rows.append({"variant": name, "error": str(exc)})
            print(f"  ERROR: {exc}")
    summary_path = OUT_DIR / "ablation_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print(f"Per-pass histograms: {LOG_PATH}")


if __name__ == "__main__":
    main()
