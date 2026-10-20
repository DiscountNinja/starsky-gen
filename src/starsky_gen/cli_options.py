"""Shared Typer option aliases for the generate callback and hidden ``generate`` command.

Defaults are applied at the call site (``param: Alias = default``). Option metadata
(help, flags, callbacks, bounds) lives here once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from starsky_gen.config import IsolateLayer, LookTarget, NebulaMode, OutputFormat, ProjectionMode, SceneMode


def validate_nebula_hero_color(ctx: typer.Context, param: typer.CallbackParam, value: str) -> str:
    from starsky_gen.cloud_complex import parse_nebula_hero_color

    _ = ctx, param
    try:
        parse_nebula_hero_color(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return value


WidthOpt = Annotated[int, typer.Option(help="Output width for equirectangular renders.")]
HeightOpt = Annotated[int, typer.Option(help="Output height for equirectangular renders.")]
OutputBaseNameOpt = Annotated[str, typer.Option(help="Base filename prefix.")]
OutputDirOpt = Annotated[
    Path,
    typer.Option("--output-dir", "-o", help="Directory for generated images."),
]
GenerationsOpt = Annotated[
    int,
    typer.Option(
        min=1,
        help="Number of generations to produce (each gets its own seed).",
    ),
]
SeedOpt = Annotated[
    int | None,
    typer.Option(
        help=(
            "Base seed: with --generations N uses seed, seed+1, …, seed+N-1. "
            "Omit for a unique random seed per generation (no repeats in the batch)."
        ),
    ),
]
ProjectionModeOpt = Annotated[
    ProjectionMode,
    typer.Option(help="Projection target: equirectangular, cubemap, or both."),
]
CubeOpt = Annotated[
    bool,
    typer.Option(
        "--cube",
        "-cube",
        help="Shortcut to include cubemap output (switches equirectangular to both).",
    ),
]
OutputFormatOpt = Annotated[OutputFormat, typer.Option(help="Output format (png, jpg, or webp).")]
SceneModeOpt = Annotated[
    SceneMode,
    typer.Option(
        "--mode",
        help="Scene recipe: normal=Milky Way band skybox; nebula=broad cloud-complex vista.",
    ),
]
NebulaModeOpt = Annotated[
    NebulaMode,
    typer.Option(help="Nebula style: distant, full, or galaxy_streak (under --mode normal)."),
]
QualityOpt = Annotated[int, typer.Option(min=50, max=100, help="JPEG/WebP quality (ignored for PNG).")]
CubemapFaceSizeOpt = Annotated[
    int, typer.Option(min=128, max=4096, help="Face size for cubemap outputs.")
]
StarsOpt = Annotated[
    bool,
    typer.Option(help="Background star catalogs (band/mid/cosmic/halo); heroes are separate."),
]
DepthOpt = Annotated[
    bool, typer.Option(help="Dim catalog stars toward the poles (not camera DOF).")
]
NebulaOpt = Annotated[bool, typer.Option(help="Enable nebula gas rendering.")]
GalaxyViewOpt = Annotated[
    bool,
    typer.Option(
        help="Enable Milky Way band under --mode normal (ignored for band under --mode nebula)."
    ),
]
BackgroundGradientOpt = Annotated[
    bool, typer.Option(help="Add non-black galactic background gradient.")
]
BlackBackgroundOpt = Annotated[
    bool,
    typer.Option(help="Force black background (overrides gradient/noise background)."),
]
JpegArtifactPassOpt = Annotated[bool, typer.Option(help="Apply JPEG artifact emulation pass.")]
LookTargetOpt = Annotated[
    LookTarget,
    typer.Option(
        "--look-target",
        help=(
            "cinematic=wrap-safe skybox; documentary=parked photo; "
            "long_exposure=tracked deep field (sharp points, denser faint stars)."
        ),
    ),
]
LongExposureLookOpt = Annotated[
    bool,
    typer.Option(
        "--long-exposure-look/--flat-exposure",
        help=(
            "Advanced override for cinematic only (legacy stacked-sky theater). "
            "Prefer --look-target long_exposure for the tracked deep-field look."
        ),
    ),
]
BackgroundTextureStrengthOpt = Annotated[
    float,
    typer.Option(
        min=0.0,
        max=2.0,
        help="Unresolved-star grain [0–2]: 0=smooth, 1=default, 2=gritty.",
    ),
]
NebulaStyleOpt = Annotated[
    Literal["subtle", "balanced", "dramatic"],
    typer.Option(help="Nebula preset: subtle, balanced, or dramatic."),
]
CloudContinuityOpt = Annotated[
    float,
    typer.Option(min=0.0, max=1.6, help="Connect/thicken cloud volumes [0–1.6]; 0=off."),
]
DustCoverageOpt = Annotated[
    float,
    typer.Option(min=0.0, max=1.6, help="Dark dust area coverage [0–1.6]; 0=none."),
]
DustStrengthOpt = Annotated[
    float,
    typer.Option(min=0.0, max=1.8, help="Dust dimming strength [0–1.8]; 0=none."),
]
NebulaHeroColorOpt = Annotated[
    str,
    typer.Option(
        "--nebula-hero-color",
        help=(
            "Hero nebula tint: red|gold|blue|magenta|neutral (red=#FF1818); "
            "or hex without # (FF1818), 0xFF1818, quoted '#FF1818', or R,G,B. "
            "Bash: never pass bare #… — it is a comment (use FF1818 or '#FF1818')."
        ),
        callback=validate_nebula_hero_color,
    ),
]
NebulaInteractDensityOpt = Annotated[
    float | None,
    typer.Option(
        "--nebula-interact-density",
        min=0.0,
        max=1.0,
        help="Interactive layer density gate [0–1]; higher = only bright emission. Scene default if omitted.",
    ),
]
NebulaInteractStrengthOpt = Annotated[
    float | None,
    typer.Option(
        "--nebula-interact-strength",
        min=0.0,
        max=1.0,
        help="Interactive layer strength [0–1]: star glow lean toward local emission. Scene default if omitted.",
    ),
]
NebulaDebugPassOpt = Annotated[
    Literal["normal", "occluder_only", "continuum_only"],
    typer.Option(
        help="Diagnostic: normal nebula, occluder-only dust RGB, or continuum-only field (galaxy_streak only).",
    ),
]
RenderProfileOpt = Annotated[
    str,
    typer.Option(help="physical=linear only; physical_grade=+grade; full=+display finish/grain."),
]
IsolateLayerOpt = Annotated[
    IsolateLayer,
    typer.Option(
        "--isolate-layer",
        help="Final plate on black: none | background_stars | heroes | nebula_gas.",
    ),
]
DebugExportLayersOpt = Annotated[
    bool,
    typer.Option(
        "--debug-layers",
        help="Write debug layer PNGs (MW morphology and/or cloud structure/emission/dust, extinction, resolved stars).",
    ),
]
DebugGrayscaleMorphologyOpt = Annotated[
    bool,
    typer.Option("--grayscale-morphology", help="Write grayscale morphology diagnostic PNG."),
]
MorphologyVoidStrengthOpt = Annotated[float, typer.Option(min=0.0, max=1.0)]
MorphologyScarStrengthOpt = Annotated[float, typer.Option(min=0.0, max=1.5)]
NoheroesOpt = Annotated[
    bool,
    typer.Option("--noheroes", help="Disable hero stars (spikes/ghosts); independent of --no-stars."),
]
BackgroundDensityOpt = Annotated[
    float | None,
    typer.Option(
        "--background-density",
        min=0.0,
        max=2.0,
        help="Off-plane/cosmic/halo star density [0–2]; blank=look default (~0.30 cinematic).",
    ),
]
SaveAsDefaultOpt = Annotated[
    bool,
    typer.Option(
        "--default",
        help=(
            "Merge explicitly passed flags into .license defaults and exit "
            "(does not render)."
        ),
    ),
]
ShowDefaultsOpt = Annotated[
    bool,
    typer.Option(
        "--show-defaults",
        help="Print custom (.license) and project built-in defaults, then exit.",
    ),
]
GuidedOpt = Annotated[
    bool,
    typer.Option(
        "--guided",
        help=(
            "Force interactive fine-tuning of arguments (even when other flags "
            "are set). Bare starsky-gen with no flags already enters guided; "
            "any other flag uses CLI args + defaults."
        ),
    ),
]
