from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from starsky_gen.assist import append_jsonl, mutate_render_config, score_image_guardrails
from starsky_gen.config import (
    FeatureConfig,
    NebulaMode,
    NebulaTuningConfig,
    OutputFormat,
    ProjectionMode,
    RenderConfig,
)
from starsky_gen.generator import render_single

app = typer.Typer(
    help="Generate realistic starfields for games and skyboxes.",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def _default_to_generate(ctx: typer.Context) -> None:
    """Run a default render when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(
            generate,
            width=2048,
            height=1024,
            output_base_name="starsky",
            output_dir=Path("output"),
            generations=1,
            seed=None,
            projection_mode=ProjectionMode.equirectangular,
            cube=False,
            output_format=OutputFormat.png,
            nebula_mode=NebulaMode.galaxy_streak,
            quality=100,
            cubemap_face_size=1024,
            stars=True,
            depth=True,
            nebula=True,
            galaxy_view=True,
            background_gradient=True,
            black_background=False,
            jpeg_artifact_pass=False,
            long_exposure_look=True,
            background_texture_strength=1.0,
            nebula_style="balanced",
            cloud_continuity=1.22,
            dust_coverage=0.98,
            dust_strength=1.14,
            nebula_debug_pass="normal",
        )


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
    background_texture_strength: float = typer.Option(
        1.0,
        min=0.0,
        max=2.0,
        help="Scale unresolved-star background texture (0 smooth, 1 default, 2 gritty).",
    ),
    nebula_style: Literal["subtle", "balanced", "dramatic"] = typer.Option(
        "balanced",
        help="Nebula preset: subtle, balanced, or dramatic.",
    ),
    cloud_continuity: float = typer.Option(
        1.22,
        min=0.6,
        max=1.6,
        help="Connect/thicken cloud volumes along the galactic band.",
    ),
    dust_coverage: float = typer.Option(
        0.98,
        min=0.5,
        max=1.6,
        help="How much of the band can contain dark dust regions.",
    ),
    dust_strength: float = typer.Option(
        1.14,
        min=0.5,
        max=1.8,
        help="How strongly dust regions dim stars and background.",
    ),
    nebula_debug_pass: Literal["normal", "occluder_only", "continuum_only"] = typer.Option(
        "normal",
        help="Diagnostic: normal nebula, occluder-only dust RGB, or continuum-only field (galaxy_streak only).",
    ),
) -> None:
    if cube and projection_mode == ProjectionMode.equirectangular:
        projection_mode = ProjectionMode.both
    run_seed = seed if seed is not None else int(np.random.SeedSequence().entropy)

    cfg = RenderConfig(
        width=width,
        height=height,
        output_base_name=output_base_name,
        output_dir=output_dir,
        generations=generations,
        seed=run_seed,
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
            debug_pass=nebula_debug_pass,
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
            background_texture_strength=background_texture_strength,
        ),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if seed is None:
        console.print(f"Using random seed: {run_seed}")

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


@app.command()
def assist(
    width: int = typer.Option(2048, help="Output width for equirectangular renders."),
    height: int = typer.Option(1024, help="Output height for equirectangular renders."),
    output_base_name: str = typer.Option("starsky_assist", help="Base filename prefix."),
    output_dir: Path = typer.Option(Path("output"), help="Directory for generated images."),
    rounds: int = typer.Option(5, min=1, max=100, help="Number of manual-guided rounds."),
    candidates_per_round: int = typer.Option(8, min=2, max=64, help="Candidates generated each round."),
    seed: int | None = typer.Option(None, help="Base random seed for deterministic exploration."),
    output_format: OutputFormat = typer.Option(OutputFormat.png, help="Output format (png or jpg)."),
    nebula_mode: NebulaMode = typer.Option(
        NebulaMode.galaxy_streak,
        help="Nebula style: distant, full, or galaxy_streak.",
    ),
    quality: int = typer.Option(100, min=50, max=100, help="JPEG quality (ignored for PNG)."),
    stars: bool = typer.Option(True, help="Enable star generation."),
    depth: bool = typer.Option(True, help="Enable depth-style intensity falloff."),
    nebula: bool = typer.Option(True, help="Enable nebula rendering."),
    galaxy_view: bool = typer.Option(True, help="Bias star density toward galactic plane."),
    background_gradient: bool = typer.Option(True, help="Add non-black galactic background gradient."),
    black_background: bool = typer.Option(False, help="Force black background."),
    long_exposure_look: bool = typer.Option(
        True,
        "--long-exposure-look/--flat-exposure",
        help="Asymmetric stacked-frame grade toggle.",
    ),
    background_texture_strength: float = typer.Option(
        1.0,
        min=0.0,
        max=2.0,
        help="Scale unresolved-star background texture (0 smooth, 1 default, 2 gritty).",
    ),
    nebula_style: Literal["subtle", "balanced", "dramatic"] = typer.Option(
        "balanced",
        help="Nebula preset: subtle, balanced, or dramatic.",
    ),
    cloud_continuity: float = typer.Option(1.22, min=0.6, max=1.6, help="Cloud continuity."),
    dust_coverage: float = typer.Option(0.98, min=0.5, max=1.6, help="Dust coverage."),
    dust_strength: float = typer.Option(1.14, min=0.5, max=1.8, help="Dust strength."),
    log_file: Path = typer.Option(
        Path("output/assist_log.jsonl"),
        help="JSONL log path for candidates and selections.",
    ),
) -> None:
    cfg = RenderConfig(
        width=width,
        height=height,
        output_base_name=output_base_name,
        output_dir=output_dir,
        generations=1,
        seed=seed,
        projection_mode=ProjectionMode.equirectangular,
        output_format=output_format,
        nebula_mode=nebula_mode,
        quality=quality,
        features=FeatureConfig(
            stars=stars,
            depth=depth,
            nebula=nebula,
            galaxy_view=galaxy_view,
            background_gradient=background_gradient,
            black_background=black_background,
            long_exposure_look=long_exposure_look,
            background_texture_strength=background_texture_strength,
        ),
        nebula_tuning=NebulaTuningConfig(
            style=nebula_style,
            cloud_continuity=cloud_continuity,
            dust_coverage=dust_coverage,
            dust_strength=dust_strength,
        ),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    base_seed = cfg.seed if cfg.seed is not None else int(np.random.SeedSequence().entropy)
    current_cfg = cfg.model_copy(deep=True)
    rng = np.random.default_rng(base_seed)

    append_jsonl(
        log_file,
        {
            "event": "session_start",
            "base_seed": base_seed,
            "rounds": rounds,
            "candidates_per_round": candidates_per_round,
            "base_config": current_cfg.model_dump(mode="json"),
        },
    )

    for round_idx in range(rounds):
        results: list[dict] = []
        console.print(f"\n[bold]Round {round_idx + 1}/{rounds}[/bold]")
        for candidate_idx in range(candidates_per_round):
            cand_base = f"{output_base_name}_r{round_idx + 1:03d}_c{candidate_idx + 1:02d}"
            cand_seed = base_seed + round_idx * 10_000 + candidate_idx
            cand_cfg = mutate_render_config(
                current_cfg,
                rng,
                round_index=round_idx,
                candidate_index=candidate_idx,
                output_base_name=cand_base,
                seed=cand_seed,
            )
            outputs, _ = render_single(cand_cfg, 0)
            image_path = outputs["equirectangular"]
            score, metrics = score_image_guardrails(image_path)
            row = {
                "round": round_idx + 1,
                "candidate_index": candidate_idx,
                "score": score,
                "metrics": metrics,
                "image_path": str(image_path),
                "config": cand_cfg.model_dump(mode="json"),
            }
            results.append(row)
            append_jsonl(log_file, {"event": "candidate", **row})

        ranked = sorted(results, key=lambda item: item["score"], reverse=True)
        table = Table(title=f"Round {round_idx + 1} ranking")
        table.add_column("Rank", justify="right")
        table.add_column("Idx", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Clip", justify="right")
        table.add_column("Crush", justify="right")
        table.add_column("Seam", justify="right")
        table.add_column("Path")
        for rank_idx, item in enumerate(ranked):
            m = item["metrics"]
            table.add_row(
                str(rank_idx),
                str(item["candidate_index"]),
                f"{item['score']:.4f}",
                f"{m['highlight_clip']:.4f}",
                f"{m['shadow_crush']:.4f}",
                f"{m['seam_delta']:.4f}",
                item["image_path"],
            )
        console.print(table)

        selected_rank = typer.prompt(
            "Select winning rank for next round",
            default=0,
            type=int,
        )
        if selected_rank < 0 or selected_rank >= len(ranked):
            raise typer.BadParameter(f"winner rank must be between 0 and {len(ranked) - 1}")

        winner = ranked[selected_rank]
        current_cfg = RenderConfig.model_validate(winner["config"])
        append_jsonl(
            log_file,
            {
                "event": "round_winner",
                "round": round_idx + 1,
                "selected_rank": selected_rank,
                "winner": winner,
            },
        )
        console.print(f"Selected: {winner['image_path']}")

    append_jsonl(
        log_file,
        {
            "event": "session_end",
            "final_config": current_cfg.model_dump(mode="json"),
        },
    )
    console.print(f"\nAssist session complete. Log written to: {log_file}")


if __name__ == "__main__":
    app()
