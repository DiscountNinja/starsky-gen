from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from starsky_gen.config import (
    NebulaTuningConfig,
    OutputFormat,
    ProjectionMode,
    RenderConfig,
)
from starsky_gen.generator import render_single


@dataclass(frozen=True)
class Variant:
    name: str
    style: str
    cloud_continuity: float
    dust_coverage: float
    dust_strength: float


def _load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR), dtype=np.float64)
    return np.clip(arr / 255.0, 0.0, 1.0)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _score_match(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_l = _luma(reference)
    can_l = _luma(candidate)

    mse = float(np.mean((reference - candidate) ** 2))

    # Row profile comparison keeps focus on Milky Way band placement/shape.
    ref_row = np.mean(ref_l, axis=1)
    can_row = np.mean(can_l, axis=1)
    ref_row = ref_row / max(float(np.max(ref_row)), 1e-9)
    can_row = can_row / max(float(np.max(can_row)), 1e-9)
    row_mae = float(np.mean(np.abs(ref_row - can_row)))

    # Measure bright-point density away from the disk: avoids blurry grain in the sky.
    h, _w = ref_l.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    off_band = np.exp(-((yy**2) / 0.36)) < 0.42
    ref_bright = float(np.mean((ref_l > np.quantile(ref_l, 0.992)) & off_band))
    can_bright = float(np.mean((can_l > np.quantile(can_l, 0.992)) & off_band))
    bright_err = abs(ref_bright - can_bright)

    # Color temperature balance in the band (reference has neutral-warm core, cooler outskirts).
    band = np.exp(-((yy**2) / 0.55))
    w = np.clip(band, 0.0, 1.0)[..., None]
    ref_rgb_mean = np.sum(reference * w, axis=(0, 1)) / max(float(np.sum(w)), 1e-9)
    can_rgb_mean = np.sum(candidate * w, axis=(0, 1)) / max(float(np.sum(w)), 1e-9)
    color_err = float(np.mean(np.abs(ref_rgb_mean - can_rgb_mean)))

    # Higher is better.
    return -(3.6 * mse + 2.2 * row_mae + 30.0 * bright_err + 2.0 * color_err)


def _variants() -> list[Variant]:
    return [
        Variant("base_balanced", "balanced", 1.22, 0.98, 1.14),
        Variant("denser_dust", "balanced", 1.28, 1.06, 1.20),
        Variant("softer_dust", "balanced", 1.15, 0.92, 1.04),
        Variant("dramatic_continuous", "dramatic", 1.26, 1.02, 1.10),
        Variant("subtle_clean", "subtle", 1.06, 0.90, 0.92),
        Variant("balanced_cloudy", "balanced", 1.34, 1.00, 1.10),
    ]


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _round2_variants(best: Variant) -> list[Variant]:
    # Keep round-2 local search compact so it finishes quickly in interactive runs.
    deltas = [-0.06, 0.0, 0.06]
    out: list[Variant] = []
    seen: set[tuple[float, float, float]] = set()
    for dc in deltas:
        for dv in deltas:
            for ds in deltas:
                c = _clamp(best.cloud_continuity + dc, 0.60, 1.60)
                v = _clamp(best.dust_coverage + dv, 0.50, 1.60)
                s = _clamp(best.dust_strength + ds, 0.50, 1.80)
                key = (round(c, 3), round(v, 3), round(s, 3))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Variant(
                        name=f"r2_c{c:.2f}_v{v:.2f}_s{s:.2f}".replace(".", "p"),
                        style=best.style,
                        cloud_continuity=c,
                        dust_coverage=v,
                        dust_strength=s,
                    )
                )
    return out


def _evaluate(
    variants: list[Variant],
    *,
    reference: np.ndarray,
    output_dir: Path,
    seed: int,
    eval_width: int,
    eval_height: int,
) -> list[tuple[float, Variant, Path]]:
    scored: list[tuple[float, Variant, Path]] = []
    for i, v in enumerate(variants):
        cfg = RenderConfig(
            width=eval_width,
            height=eval_height,
            generations=1,
            seed=seed + i,
            projection_mode=ProjectionMode.equirectangular,
            output_format=OutputFormat.png,
            output_dir=output_dir,
            output_base_name=f"autotune_{v.name}",
            nebula_tuning=NebulaTuningConfig(
                style=v.style,  # type: ignore[arg-type]
                cloud_continuity=v.cloud_continuity,
                dust_coverage=v.dust_coverage,
                dust_strength=v.dust_strength,
            ),
        )
        saved, _stats = render_single(cfg, 0)
        out_path = saved["equirectangular"]
        candidate = _load_rgb(out_path, eval_width, eval_height)
        score = _score_match(reference, candidate)
        scored.append((score, v, out_path))
        print(f"{v.name:20s} score={score: .6f} output={out_path}")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tune Milky Way render settings against a reference image.")
    parser.add_argument("--reference", type=Path, required=True, help="Path to reference image (target look).")
    parser.add_argument("--output-dir", type=Path, default=Path("output/autotune"), help="Where variant outputs are written.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic comparisons.")
    parser.add_argument("--eval-width", type=int, default=1024, help="Evaluation render width.")
    parser.add_argument("--eval-height", type=int, default=512, help="Evaluation render height.")
    parser.add_argument("--top-k", type=int, default=3, help="How many top variants to print.")
    parser.add_argument(
        "--round2",
        action="store_true",
        help="Run a second local search round around the best round-1 variant.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = _load_rgb(args.reference, args.eval_width, args.eval_height)

    print("Round 1:")
    scored = _evaluate(
        _variants(),
        reference=reference,
        output_dir=args.output_dir,
        seed=args.seed,
        eval_width=args.eval_width,
        eval_height=args.eval_height,
    )

    if args.round2 and scored:
        print("\nRound 2:")
        r2_dir = args.output_dir / "round2"
        r2_dir.mkdir(parents=True, exist_ok=True)
        best_r1 = scored[0][1]
        scored = _evaluate(
            _round2_variants(best_r1),
            reference=reference,
            output_dir=r2_dir,
            seed=args.seed + 1000,
            eval_width=args.eval_width,
            eval_height=args.eval_height,
        )

    print("\nTop matches:")
    for rank, (score, v, out_path) in enumerate(scored[: max(1, args.top_k)], start=1):
        print(
            f"{rank}. {v.name} score={score: .6f} style={v.style} "
            f"cloud={v.cloud_continuity:.2f} dust_cov={v.dust_coverage:.2f} "
            f"dust_str={v.dust_strength:.2f} path={out_path}"
        )


if __name__ == "__main__":
    main()
