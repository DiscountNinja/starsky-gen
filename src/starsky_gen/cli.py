from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from starsky_gen.config import (
    FeatureConfig,
    NebulaMode,
    NebulaTuningConfig,
    OutputFormat,
    ProjectionMode,
    RenderConfig,
)
from starsky_gen.generator import render_single

app = typer.Typer(help="Generate realistic starfields for games and skyboxes.")
console = Console()


def _render_pass_count(cfg: RenderConfig) -> int:
    passes = 1  # background pass
    if cfg.features.stars:
        passes += 1
    if cfg.features.nebula:
        passes += 1
    if cfg.features.jpeg_artifact_pass and cfg.output_format == OutputFormat.jpg:
        passes += 1
    if cfg.projection_mode in {ProjectionMode.equirectangular, ProjectionMode.both}:
        passes += 1
    if cfg.projection_mode in {ProjectionMode.cubemap, ProjectionMode.both}:
        passes += 1  # cubemap conversion pass
        passes += 6  # write each cubemap face
    return passes


@app.command()
def generate(
    width: int = typer.Option(2048, help="Output width for equirectangular renders."),
    height: int = typer.Option(1024, help="Output height for equirectangular renders."),
    output_base_name: str = typer.Option("starsky", help="Base filename prefix."),
    output_dir: Path = typer.Option(Path("output"), help="Directory for generated images."),
    generations: int = typer.Option(1, min=1, help="Number of generations to produce."),
    seed: int | None = typer.Option(None, help="Base random seed for deterministic batches."),
    projection_mode: ProjectionMode = typer.Option(
        ProjectionMode.equirectangular,
        help="Projection target: equirectangular, cubemap, or both.",
    ),
    cube: bool = typer.Option(
        False,
        "--cube",
        "-cube",
        help="Shortcut to include cubemap output (switches equirectangular to both).",
    ),
    output_format: OutputFormat = typer.Option(OutputFormat.png, help="Output format (png or jpg)."),
    nebula_mode: NebulaMode = typer.Option(
        NebulaMode.galaxy_streak,
        help="Nebula style: distant, full, or galaxy_streak.",
    ),
    quality: int = typer.Option(100, min=50, max=100, help="JPEG quality (ignored for PNG)."),
    cubemap_face_size: int = typer.Option(1024, min=128, max=4096, help="Face size for cubemap outputs."),
    stars: bool = typer.Option(True, help="Enable star generation."),
    depth: bool = typer.Option(True, help="Enable depth-style intensity falloff."),
    nebula: bool = typer.Option(True, help="Enable nebula rendering."),
    galaxy_view: bool = typer.Option(True, help="Bias star density toward galactic plane."),
    background_gradient: bool = typer.Option(True, help="Add non-black galactic background gradient."),
    black_background: bool = typer.Option(
        False,
        help="Force black background (overrides gradient/noise background).",
    ),
    jpeg_artifact_pass: bool = typer.Option(False, help="Apply JPEG artifact emulation pass."),
    long_exposure_look: bool = typer.Option(
        True,
        "--long-exposure-look/--flat-exposure",
        help="Asymmetric stacked-frame grade (sky tilt, corner glow, vignette); off for symmetric clean output.",
    ),
    nebula_style: Literal["subtle", "balanced", "dramatic"] = typer.Option(
        "balanced",
        help="Nebula preset: subtle, balanced, or dramatic.",
    ),
    cloud_continuity: float = typer.Option(
        1.0,
        min=0.6,
        max=1.6,
        help="Connect/thicken cloud volumes along the galactic band.",
    ),
    dust_coverage: float = typer.Option(
        1.0,
        min=0.5,
        max=1.6,
        help="How much of the band can contain dark dust regions.",
    ),
    dust_strength: float = typer.Option(
        1.0,
        min=0.5,
        max=1.8,
        help="How strongly dust regions dim stars and background.",
    ),
) -> None:
    if cube and projection_mode == ProjectionMode.equirectangular:
        projection_mode = ProjectionMode.both

    cfg = RenderConfig(
        width=width,
        height=height,
        output_base_name=output_base_name,
        output_dir=output_dir,
        generations=generations,
        seed=seed,
        projection_mode=projection_mode,
        output_format=output_format,
        nebula_mode=nebula_mode,
        quality=quality,
        cubemap_face_size=cubemap_face_size,
        nebula_tuning=NebulaTuningConfig(
            style=nebula_style,
            cloud_continuity=cloud_continuity,
            dust_coverage=dust_coverage,
            dust_strength=dust_strength,
        ),
        features=FeatureConfig(
            stars=stars,
            depth=depth,
            nebula=nebula,
            galaxy_view=galaxy_view,
            background_gradient=background_gradient,
            black_background=black_background,
            jpeg_artifact_pass=jpeg_artifact_pass,
            long_exposure_look=long_exposure_look,
        ),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        batch_task = None
        if cfg.generations > 1:
            batch_task = progress.add_task("Batch", total=cfg.generations)

        for i in range(cfg.generations):
            render_task = progress.add_task("Current render", total=_render_pass_count(cfg))
            outputs, stats = render_single(cfg, i, on_pass_complete=lambda: progress.advance(render_task, 1))
            progress.remove_task(render_task)
            if batch_task is not None:
                progress.advance(batch_task, 1)

            console.print(f"Saved generation {i + 1}: {outputs}")
            if stats["color_counts"]:
                console.print(f"  Color counts: {stats['color_counts']}")
                console.print(f"  Size counts: {stats['size_counts']}")


if __name__ == "__main__":
    app()
