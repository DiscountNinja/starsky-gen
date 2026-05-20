#!/usr/bin/env python3
"""Compare band high-frequency energy across output PNGs (detect HF wash)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def band_hf_stats(path: Path, *, row_frac: float = 0.5, band_half: float = 0.12) -> dict[str, float]:
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    h = int(im.shape[0])
    y0 = int(h * (row_frac - band_half))
    y1 = int(h * (row_frac + band_half))
    band = im[y0:y1]
    lu = 0.2126 * band[..., 0] + 0.7152 * band[..., 1] + 0.0722 * band[..., 2]
    med = uniform_filter(lu, size=5, mode="wrap")
    hf = lu - med
    return {
        "p50": float(np.percentile(lu, 50)),
        "std": float(np.std(lu)),
        "hf_std": float(np.std(hf)),
        "hf_p90": float(np.percentile(np.abs(hf), 90)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path, default=Path("output"), nargs="?")
    p.add_argument("--glob", default="band_color*_equirect.png")
    args = p.parse_args()
    files = sorted(args.output_dir.glob(args.glob))
    if not files:
        print(f"No files matching {args.glob!r} in {args.output_dir}")
        return
    print(f"{'file':32}  {'p50':>6}  {'std':>6}  {'hf_std':>7}  {'hf_p90':>7}")
    best_hf = 0.0
    best_name = ""
    for f in files:
        s = band_hf_stats(f)
        print(
            f"{f.name:32}  {s['p50']:6.3f}  {s['std']:6.3f}  "
            f"{s['hf_std']:7.4f}  {s['hf_p90']:7.4f}"
        )
        if s["hf_p90"] > best_hf:
            best_hf = s["hf_p90"]
            best_name = f.name
    print(f"\nHighest hf_p90: {best_name} ({best_hf:.4f})")


if __name__ == "__main__":
    main()
