#!/usr/bin/env python3
"""Render A/B/C profile comparison and print basic saturation vs luma stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starsky_gen.config import FeatureConfig, NebulaMode, OutputFormat, RenderConfig, RenderProfile
from starsky_gen.generator import render_single


def _stats(rgb: np.ndarray) -> dict[str, float]:
    x = np.clip(rgb.astype(np.float64), 0.0, 1.0)
    luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    chroma = np.std(x, axis=2)
    return {
        "luma_mean": float(np.mean(luma)),
        "luma_p99": float(np.percentile(luma, 99.0)),
        "sat_mean": float(np.mean(chroma)),
        "sat_p99": float(np.percentile(chroma, 99.0)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare render profiles for one seed.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--out", type=Path, default=Path("output/profile_ab"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for profile in (RenderProfile.physical, RenderProfile.physical_grade, RenderProfile.full):
        cfg = RenderConfig(
            width=args.width,
            height=args.height,
            output_dir=args.out,
            output_base_name=f"profile_{profile.value}",
            generations=1,
            seed=args.seed,
            nebula_mode=NebulaMode.galaxy_streak,
            output_format=OutputFormat.png,
            features=FeatureConfig(
                render_profile=profile,
                debug_export_layers=False,
            ),
        )
        paths, _ = render_single(cfg, 0)
        eq = paths.get("equirectangular")
        if eq is None:
            continue
        img = np.asarray(Image.open(eq).convert("RGB"), dtype=np.float64) / 255.0
        st = _stats(img)
        print(f"{profile.value}: luma_mean={st['luma_mean']:.4f} sat_mean={st['sat_mean']:.4f} sat_p99={st['sat_p99']:.4f}")


if __name__ == "__main__":
    main()
