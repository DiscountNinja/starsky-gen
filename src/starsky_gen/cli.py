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
    RenderProfile,
)
from starsky_gen.generator import render_single

app = typer.Typer(
    help="Generate realistic starfields for games and skyboxes.",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()


def _feature_config_from_cli(
    *,
    stars: bool,
    depth: bool,
    nebula: bool,
    galaxy_view: bool,
    background_gradient: bool,
    black_background: bool,
    jpeg_artifact_pass: bool,
    long_exposure_look: bool,
    background_texture_strength: float,
    render_profile: str = "full",
    debug_export_layers: bool = False,
    debug_grayscale_morphology: bool = False,
    morphology_void_strength: float = 0.72,
    morphology_scar_strength: float = 0.68,
) -> FeatureConfig:
    return FeatureConfig(
        stars=stars,
        depth=depth,
        nebula=nebula,
        galaxy_view=galaxy_view,
        background_gradient=background_gradient,
        black_background=black_background,
        jpeg_artifact_pass=jpeg_artifact_pass,
        long_exposure_look=long_exposure_look,
        background_texture_strength=background_texture_strength,
        render_profile=RenderProfile(render_profile),
        debug_export_layers=debug_export_layers,
        debug_grayscale_morphology=debug_grayscale_morphology,
        morphology_void_strength=morphology_void_strength,
        morphology_scar_strength=morphology_scar_strength,
    )


@app.callback(invoke_without_command=True)
def _default_to_generate(
    ctx: typer.Context,
    width: int = typer.Option(2048, help="Output width for equirectangular renders."),
    height: int = typer.Option(1024, help="Output height for equirectangular renders."),
    output_base_name: str = typer.Option("starsky", help="Base filename prefix."),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir",
        "-o",
        help="Directory for generated images.",
    ),
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
    render_profile: str = typer.Option(
        "full",
        help="Render pipeline: physical | physical_grade | full.",
    ),
    debug_export_layers: bool = typer.Option(
        False, "--debug-layers", help="Write morphology/star/extinction layer PNGs."
    ),
    debug_grayscale_morphology: bool = typer.Option(
        False,
        "--grayscale-morphology",
        help="Write grayscale morphology diagnostic PNG.",
    ),
    morphology_void_strength: float = typer.Option(0.72, min=0.0, max=1.0),
    morphology_scar_strength: float = typer.Option(0.68, min=0.0, max=1.5),
) -> None:
    """Run a default render when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(
            generate,
            width=width,
            height=height,
            output_base_name=output_base_name,
            output_dir=output_dir,
            generations=generations,
            seed=seed,
            projection_mode=projection_mode,
            cube=cube,
            output_format=output_format,
            nebula_mode=nebula_mode,
            quality=quality,
            cubemap_face_size=cubemap_face_size,
            stars=stars,
            depth=depth,
            nebula=nebula,
            galaxy_view=galaxy_view,
            background_gradient=background_gradient,
            black_background=black_background,
            jpeg_artifact_pass=jpeg_artifact_pass,
            long_exposure_look=long_exposure_look,
            background_texture_strength=background_texture_strength,
            nebula_style=nebula_style,
            cloud_continuity=cloud_continuity,
            dust_coverage=dust_coverage,
            dust_strength=dust_strength,
            nebula_debug_pass=nebula_debug_pass,
            render_profile=render_profile,
            debug_export_layers=debug_export_layers,
            debug_grayscale_morphology=debug_grayscale_morphology,
            morphology_void_strength=morphology_void_strength,
            morphology_scar_strength=morphology_scar_strength,
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


def _render_stage_labels(cfg: RenderConfig) -> list[str]:
    labels = ["background"]
    if (
        cfg.features.stars
        and cfg.features.galaxy_view
        and cfg.features.cosmic_star_enabled
    ):
        labels.append("stars cosmic")
    if cfg.features.nebula and cfg.features.stars and cfg.nebula_mode == NebulaMode.galaxy_streak:
        labels.append("nebula prep")
    if cfg.features.stars:
        labels.append("stars background prep")
        if cfg.features.galaxy_view and cfg.features.halo_star_enabled:
            labels.append("stars halo")
        labels.append("stars background")
        if cfg.features.galaxy_view and cfg.features.star_midlayer_scale > 1e-4:
            labels.append("stars mid")
    if cfg.features.nebula:
        labels.append("nebula clouds/dust")
    if cfg.features.galaxy_view and cfg.features.stars:
        labels.append("unresolved field")
    if cfg.features.stars:
        labels.append("stars foreground prep")
        labels.append("stars foreground")
    if cfg.features.galaxy_view:
        labels.append("grade/color")
    if cfg.features.jpeg_artifact_pass and cfg.output_format == OutputFormat.jpg:
        labels.append("jpeg artifacts")
    if cfg.projection_mode in {ProjectionMode.equirectangular, ProjectionMode.both}:
        labels.append("save equirectangular")
    if cfg.projection_mode in {ProjectionMode.cubemap, ProjectionMode.both}:
        labels.append("project cubemap")
        labels.extend([f"save cube {f}" for f in ("px", "nx", "py", "ny", "pz", "nz")])
    return labels


@app.command()
def generate(
    width: int = typer.Option(2048, help="Output width for equirectangular renders."),
    height: int = typer.Option(1024, help="Output height for equirectangular renders."),
    output_base_name: str = typer.Option("starsky", help="Base filename prefix."),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir",
        "-o",
        help="Directory for generated images.",
    ),
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
    render_profile: str = typer.Option(
        "full",
        help="Render pipeline: physical | physical_grade | full.",
    ),
    debug_export_layers: bool = typer.Option(
        False, "--debug-layers", help="Write morphology/star/extinction layer PNGs."
    ),
    debug_grayscale_morphology: bool = typer.Option(
        False,
        "--grayscale-morphology",
        help="Write grayscale morphology diagnostic PNG.",
    ),
    morphology_void_strength: float = typer.Option(0.72, min=0.0, max=1.0),
    morphology_scar_strength: float = typer.Option(0.68, min=0.0, max=1.5),
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
        features=_feature_config_from_cli(
            stars=stars,
            depth=depth,
            nebula=nebula,
            galaxy_view=galaxy_view,
            background_gradient=background_gradient,
            black_background=black_background,
            jpeg_artifact_pass=jpeg_artifact_pass,
            long_exposure_look=long_exposure_look,
            background_texture_strength=background_texture_strength,
            render_profile=render_profile,
            debug_export_layers=debug_export_layers,
            debug_grayscale_morphology=debug_grayscale_morphology,
            morphology_void_strength=morphology_void_strength,
            morphology_scar_strength=morphology_scar_strength,
        ),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if seed is None:
        console.print(f"Using random seed: {run_seed}")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.fields[stage]}"),
        TextColumn("{task.fields[stage_pct]}"),
        BarColumn(),
        TextColumn("{task.completed:.2f}/{task.total:.0f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        batch_task = None
        if cfg.generations > 1:
            batch_task = progress.add_task("Batch", total=cfg.generations)

        for i in range(cfg.generations):
            stages = _render_stage_labels(cfg)
            render_task = progress.add_task(
                "Current render",
                total=max(1, len(stages)),
                stage="starting",
                stage_pct="0.00%",
            )
            stage_index: dict[str, int] = {}
            for si, s in enumerate(stages):
                if s not in stage_index:
                    stage_index[s] = si
            last_overall = 0.0
            last_stage = "starting"

            def _advance_render(stage: str, stage_progress: float = 1.0) -> None:
                nonlocal last_overall, last_stage
                si = stage_index.get(stage, max(0, len(stages) - 1))
                stage_progress = float(np.clip(stage_progress, 0.0, 1.0))
                overall = float(si) + stage_progress
                overall = max(last_overall, min(overall, float(len(stages))))
                delta = overall - last_overall
                if delta > 0:
                    progress.advance(render_task, delta)
                    last_overall = overall
                progress.update(
                    render_task,
                    stage=stage,
                    stage_pct=f"{stage_progress * 100.0:.2f}%",
                )
                # Ensure stage label updates are visible immediately, even when delta == 0.
                if stage != last_stage:
                    progress.refresh()
                    last_stage = stage

            outputs, stats = render_single(cfg, i, on_pass_complete=_advance_render)
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
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir",
        "-o",
        help="Directory for generated images.",
    ),
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
