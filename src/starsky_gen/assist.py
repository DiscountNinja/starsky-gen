from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from starsky_gen.config import NebulaTuningConfig, RenderConfig


@dataclass
class CandidateResult:
    rank: int
    candidate_index: int
    image_path: Path
    score: float
    metrics: dict[str, float]
    config: RenderConfig


def _clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def score_image_guardrails(path: Path) -> tuple[float, dict[str, float]]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    highlight_clip = float(np.mean(np.max(rgb, axis=2) > 0.985))
    shadow_crush = float(np.mean(luma < 0.012))
    seam_delta = float(np.mean(np.abs(rgb[:, 0, :] - rgb[:, -1, :])))

    ch_mean = np.mean(rgb, axis=(0, 1))
    color_cast = float(np.max(ch_mean) - np.min(ch_mean))

    p05 = float(np.quantile(luma, 0.05))
    p95 = float(np.quantile(luma, 0.95))
    contrast = max(0.0, p95 - p05)
    contrast_target = 0.50
    contrast_penalty = float(abs(contrast - contrast_target))

    penalty = (
        highlight_clip * 2.2
        + shadow_crush * 1.6
        + seam_delta * 4.0
        + color_cast * 0.8
        + contrast_penalty * 0.7
    )
    score = 1.0 - penalty
    metrics = {
        "highlight_clip": highlight_clip,
        "shadow_crush": shadow_crush,
        "seam_delta": seam_delta,
        "color_cast": color_cast,
        "contrast": contrast,
        "contrast_penalty": contrast_penalty,
        "penalty": penalty,
    }
    return float(score), metrics


def mutate_render_config(
    base_cfg: RenderConfig,
    rng: np.random.Generator,
    *,
    round_index: int,
    candidate_index: int,
    output_base_name: str,
    seed: int,
) -> RenderConfig:
    cfg = base_cfg.model_copy(deep=True)
    cfg.output_base_name = output_base_name
    cfg.seed = seed

    decay = 1.0 / (1.0 + round_index * 0.16)
    sigma = 0.12 * decay

    style_order = ["subtle", "balanced", "dramatic"]
    style = cfg.nebula_tuning.style
    style_idx = style_order.index(style)
    if rng.random() < (0.16 * decay + 0.04):
        step = -1 if rng.random() < 0.5 else 1
        style_idx = int(np.clip(style_idx + step, 0, len(style_order) - 1))
        style = style_order[style_idx]

    cloud = _clamp(
        cfg.nebula_tuning.cloud_continuity + rng.normal(0.0, sigma),
        0.6,
        1.6,
    )
    coverage = _clamp(
        cfg.nebula_tuning.dust_coverage + rng.normal(0.0, sigma),
        0.5,
        1.6,
    )
    strength = _clamp(
        cfg.nebula_tuning.dust_strength + rng.normal(0.0, sigma),
        0.5,
        1.8,
    )
    long_exposure = cfg.features.long_exposure_look
    if rng.random() < (0.06 * decay):
        long_exposure = not long_exposure

    cfg.nebula_tuning = NebulaTuningConfig(
        style=style,
        cloud_continuity=cloud,
        dust_coverage=coverage,
        dust_strength=strength,
        debug_pass=cfg.nebula_tuning.debug_pass,
    )
    cfg.features.long_exposure_look = long_exposure
    _ = candidate_index
    return cfg


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_jsonable(payload), sort_keys=True) + "\n")
