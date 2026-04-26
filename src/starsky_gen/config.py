from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class NebulaMode(str, Enum):
    distant = "distant"
    full = "full"
    galaxy_streak = "galaxy_streak"


class ProjectionMode(str, Enum):
    equirectangular = "equirectangular"
    cubemap = "cubemap"
    both = "both"


class OutputFormat(str, Enum):
    png = "png"
    jpg = "jpg"


class FeatureConfig(BaseModel):
    stars: bool = True
    depth: bool = True
    nebula: bool = True
    galaxy_view: bool = True
    reference_anchors: bool = Field(
        True,
        description="Paint a small set of bright real stars at approximate Galactic (l,b) positions.",
    )
    background_gradient: bool = True
    black_background: bool = False
    jpeg_artifact_pass: bool = False
    long_exposure_look: bool = Field(
        True,
        description=(
            "Asymmetric stacked-Milky-Way look: tilted sky floor, mild corner glow, off-center vignette "
            "(disable for a cleaner symmetric grade)."
        ),
    )
    background_texture_strength: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="Scale for unresolved-star background texture (0=smooth, 1=default, 2=grittier).",
    )


class NebulaTuningConfig(BaseModel):
    """Artist-facing controls for nebula structure and dust extinction.

    Tuning guide (keep this updated as renderer behavior evolves):
    - Out-of-box baseline: use the defaults in this model.
    - Fast first pass: adjust only `style`; then touch one scalar at a time.
    - `cloud_continuity`: higher = more connected/thicker cloud bands.
    - `dust_coverage`: higher = dust appears in more places.
    - `dust_strength`: higher = stronger star/background dimming in dust.
    Recommended fine-tune ranges after style selection:
    - cloud_continuity: 0.9-1.25
    - dust_coverage: 0.85-1.15
    - dust_strength: 0.80-1.20
    """

    style: Literal["subtle", "balanced", "dramatic"] = Field(
        "balanced",
        description="High-level nebula preset controlling dust/extinction character; start here before scalar tweaks.",
    )
    cloud_continuity: float = Field(
        1.22,
        ge=0.6,
        le=1.6,
        description="Cloud connectivity/thickness along the galactic band; higher values produce denser connected volumes.",
    )
    dust_coverage: float = Field(
        0.98,
        ge=0.5,
        le=1.6,
        description="Fraction of the band eligible for dark dust/void carving; lower values create fewer dust regions.",
    )
    dust_strength: float = Field(
        1.14,
        ge=0.5,
        le=1.8,
        description="Dust extinction strength against stars/background; raise only after coverage feels right.",
    )
    debug_pass: Literal["normal", "occluder_only", "continuum_only"] = Field(
        "normal",
        description=(
            "Diagnostic render for galaxy_streak nebula: `normal` full composite; `occluder_only` shows "
            "dust_rgb blend only; `continuum_only` shows synthetic low-frequency continuum only (boosted for visibility)."
        ),
    )


class RenderConfig(BaseModel):
    width: int = Field(2048, ge=256, le=16384)
    height: int = Field(1024, ge=128, le=8192)
    output_base_name: str = "starsky"
    output_dir: Path = Path("output")
    generations: int = Field(1, ge=1, le=1000)
    seed: int | None = None
    projection_mode: ProjectionMode = ProjectionMode.both
    output_format: OutputFormat = OutputFormat.png
    nebula_mode: NebulaMode = NebulaMode.galaxy_streak
    quality: int = Field(100, ge=50, le=100)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    nebula_tuning: NebulaTuningConfig = Field(
        default_factory=NebulaTuningConfig,
        description="Procedural nebula tuning block used by renderer and config files.",
    )
    cubemap_face_size: int = Field(1024, ge=128, le=4096)
    wrap_safe: Literal[True] = True

    @field_validator("output_base_name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip().replace(" ", "_")
        if not cleaned:
            raise ValueError("output_base_name cannot be empty")
        return cleaned
