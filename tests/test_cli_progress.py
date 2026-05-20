"""CLI progress stage list matches generator notifications."""

import re
from pathlib import Path

from starsky_gen.cli import _render_stage_labels
from starsky_gen.config import FeatureConfig, NebulaMode, RenderConfig


def _notify_stages_from_generator() -> set[str]:
    text = Path("src/starsky_gen/generator.py").read_text(encoding="utf-8")
    return set(re.findall(r'_notify_stage\(\s*"([^"]+)"', text))


def test_render_stage_labels_cover_generator_notifications() -> None:
    cfg = RenderConfig(
        features=FeatureConfig(galaxy_view=True, stars=True, nebula=True),
        nebula_mode=NebulaMode.galaxy_streak,
    )
    labels = set(_render_stage_labels(cfg))
    notified = _notify_stages_from_generator()
    optional = {"jpeg artifacts"}
    if cfg.projection_mode.value == "equirectangular":
        optional |= {f"save cube {f}" for f in ("px", "nx", "py", "ny", "pz", "nz")}
        optional.add("project cubemap")
    missing = notified - labels - optional
    assert not missing, f"Stages missing from progress bar: {sorted(missing)}"
    assert "stars mid" in labels
