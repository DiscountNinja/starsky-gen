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
    IsolateLayer,
    LookTarget,
    NebulaMode,
    NebulaTuningConfig,
    OutputFormat,
    ProjectionMode,
    RenderConfig,
    RenderProfile,
    SceneMode,
)
from starsky_gen.generator import render_single
from starsky_gen.guided import (
    GuidedCancelled,
    confirm_start_render,
    print_guided_confirm,
    print_render_preview,
    run_guided_prompts,
    should_run_guided,
)
from starsky_gen.licensing import maybe_prompt_before_generate, record_generations

app = typer.Typer(
    help="Generate cinematic AAA skyboxes and starfields for games.",
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
    look_target: LookTarget,
    long_exposure_look: bool,
    background_texture_strength: float,
    render_profile: str = "full",
    debug_export_layers: bool = False,
    debug_grayscale_morphology: bool = False,
    morphology_void_strength: float = 0.72,
    morphology_scar_strength: float = 0.68,
    hero_star_enabled: bool = True,
    background_star_density_scale: float | None = None,
    isolate_layer: IsolateLayer | str = IsolateLayer.none,
) -> FeatureConfig:
    kw: dict = {}
    if background_star_density_scale is not None:
        kw["background_star_density_scale"] = float(background_star_density_scale)
    iso = isolate_layer
    if not isinstance(iso, IsolateLayer):
        iso = IsolateLayer(str(iso))
    return FeatureConfig(
        stars=stars,
        depth=depth,
        nebula=nebula,
        galaxy_view=galaxy_view,
        look_target=look_target,
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
        hero_star_enabled=hero_star_enabled,
        isolate_layer=iso,
        **kw,
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
    output_format: OutputFormat = typer.Option(
        OutputFormat.png, help="Output format (png, jpg, or webp)."
    ),
    scene_mode: SceneMode = typer.Option(
        SceneMode.normal,
        "--mode",
        help="Scene recipe: normal=Milky Way band skybox; nebula=broad cloud-complex vista.",
    ),
    nebula_mode: NebulaMode = typer.Option(
        NebulaMode.galaxy_streak,
        help="Nebula style: distant, full, or galaxy_streak (under --mode normal).",
    ),
    quality: int = typer.Option(
        100, min=50, max=100, help="JPEG/WebP quality (ignored for PNG)."
    ),
    cubemap_face_size: int = typer.Option(1024, min=128, max=4096, help="Face size for cubemap outputs."),
    stars: bool = typer.Option(
        True, help="Background star catalogs (band/mid/cosmic/halo); heroes are separate."
    ),
    depth: bool = typer.Option(
        True, help="Dim catalog stars toward the poles (not camera DOF)."
    ),
    nebula: bool = typer.Option(True, help="Enable nebula gas rendering."),
    galaxy_view: bool = typer.Option(
        True,
        help="Enable Milky Way band under --mode normal (ignored for band under --mode nebula).",
    ),
    background_gradient: bool = typer.Option(True, help="Add non-black galactic background gradient."),
    black_background: bool = typer.Option(
        False,
        help="Force black background (overrides gradient/noise background).",
    ),
    jpeg_artifact_pass: bool = typer.Option(False, help="Apply JPEG artifact emulation pass."),
    look_target: LookTarget = typer.Option(
        LookTarget.cinematic,
        "--look-target",
        help=(
            "cinematic=wrap-safe skybox; documentary=parked photo; "
            "long_exposure=tracked deep field (sharp points, denser faint stars)."
        ),
    ),
    long_exposure_look: bool = typer.Option(
        False,
        "--long-exposure-look/--flat-exposure",
        help=(
            "Advanced override for cinematic only (legacy stacked-sky theater). "
            "Prefer --look-target long_exposure for the tracked deep-field look."
        ),
    ),
    background_texture_strength: float = typer.Option(
        1.0,
        min=0.0,
        max=2.0,
        help="Unresolved-star grain [0–2]: 0=smooth, 1=default, 2=gritty.",
    ),
    nebula_style: Literal["subtle", "balanced", "dramatic"] = typer.Option(
        "balanced",
        help="Nebula preset: subtle, balanced, or dramatic.",
    ),
    cloud_continuity: float = typer.Option(
        1.22,
        min=0.0,
        max=1.6,
        help="Connect/thicken cloud volumes [0–1.6]; 0=off.",
    ),
    dust_coverage: float = typer.Option(
        0.98,
        min=0.0,
        max=1.6,
        help="Dark dust area coverage [0–1.6]; 0=none.",
    ),
    dust_strength: float = typer.Option(
        1.14,
        min=0.0,
        max=1.8,
        help="Dust dimming strength [0–1.8]; 0=none.",
    ),
    nebula_debug_pass: Literal["normal", "occluder_only", "continuum_only"] = typer.Option(
        "normal",
        help="Diagnostic: normal nebula, occluder-only dust RGB, or continuum-only field (galaxy_streak only).",
    ),
    render_profile: str = typer.Option(
        "full",
        help="physical=linear only; physical_grade=+grade; full=+display finish/grain.",
    ),
    isolate_layer: IsolateLayer = typer.Option(
        IsolateLayer.none,
        "--isolate-layer",
        help="Final plate on black: none | background_stars | heroes | nebula_gas.",
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
    noheroes: bool = typer.Option(
        False,
        "--noheroes",
        help="Disable hero stars (spikes/ghosts); independent of --no-stars.",
    ),
    background_density: float | None = typer.Option(
        None,
        "--background-density",
        min=0.0,
        max=2.0,
        help="Off-plane/cosmic/halo star density [0–2]; blank=look default (~0.30 cinematic).",
    ),
    guided: bool = typer.Option(
        True,
        "--guided/--no-guided",
        help=(
            "Interactive prompts when no other flags are passed (default). "
            "Any CLI flag skips guided and uses args + defaults. "
            "Force with --guided; disable with --no-guided."
        ),
    ),
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
            scene_mode=scene_mode,
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
            look_target=look_target,
            long_exposure_look=long_exposure_look,
            background_texture_strength=background_texture_strength,
            nebula_style=nebula_style,
            cloud_continuity=cloud_continuity,
            dust_coverage=dust_coverage,
            dust_strength=dust_strength,
            nebula_debug_pass=nebula_debug_pass,
            render_profile=render_profile,
            isolate_layer=isolate_layer,
            debug_export_layers=debug_export_layers,
            debug_grayscale_morphology=debug_grayscale_morphology,
            morphology_void_strength=morphology_void_strength,
            morphology_scar_strength=morphology_scar_strength,
            noheroes=noheroes,
            background_density=background_density,
            guided=guided,
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
    output_format: OutputFormat = typer.Option(
        OutputFormat.png, help="Output format (png, jpg, or webp)."
    ),
    scene_mode: SceneMode = typer.Option(
        SceneMode.normal,
        "--mode",
        help="Scene recipe: normal=Milky Way band skybox; nebula=broad cloud-complex vista.",
    ),
    nebula_mode: NebulaMode = typer.Option(
        NebulaMode.galaxy_streak,
        help="Nebula style: distant, full, or galaxy_streak (under --mode normal).",
    ),
    quality: int = typer.Option(
        100, min=50, max=100, help="JPEG/WebP quality (ignored for PNG)."
    ),
    cubemap_face_size: int = typer.Option(1024, min=128, max=4096, help="Face size for cubemap outputs."),
    stars: bool = typer.Option(
        True, help="Background star catalogs (band/mid/cosmic/halo); heroes are separate."
    ),
    depth: bool = typer.Option(
        True, help="Dim catalog stars toward the poles (not camera DOF)."
    ),
    nebula: bool = typer.Option(True, help="Enable nebula gas rendering."),
    galaxy_view: bool = typer.Option(
        True,
        help="Enable Milky Way band under --mode normal (ignored for band under --mode nebula).",
    ),
    background_gradient: bool = typer.Option(True, help="Add non-black galactic background gradient."),
    black_background: bool = typer.Option(
        False,
        help="Force black background (overrides gradient/noise background).",
    ),
    jpeg_artifact_pass: bool = typer.Option(False, help="Apply JPEG artifact emulation pass."),
    look_target: LookTarget = typer.Option(
        LookTarget.cinematic,
        "--look-target",
        help=(
            "cinematic=wrap-safe skybox; documentary=parked photo; "
            "long_exposure=tracked deep field (sharp points, denser faint stars)."
        ),
    ),
    long_exposure_look: bool = typer.Option(
        False,
        "--long-exposure-look/--flat-exposure",
        help=(
            "Advanced override for cinematic only (legacy stacked-sky theater). "
            "Prefer --look-target long_exposure for the tracked deep-field look."
        ),
    ),
    background_texture_strength: float = typer.Option(
        1.0,
        min=0.0,
        max=2.0,
        help="Unresolved-star grain [0–2]: 0=smooth, 1=default, 2=gritty.",
    ),
    nebula_style: Literal["subtle", "balanced", "dramatic"] = typer.Option(
        "balanced",
        help="Nebula preset: subtle, balanced, or dramatic.",
    ),
    cloud_continuity: float = typer.Option(
        1.22,
        min=0.0,
        max=1.6,
        help="Connect/thicken cloud volumes [0–1.6]; 0=off.",
    ),
    dust_coverage: float = typer.Option(
        0.98,
        min=0.0,
        max=1.6,
        help="Dark dust area coverage [0–1.6]; 0=none.",
    ),
    dust_strength: float = typer.Option(
        1.14,
        min=0.0,
        max=1.8,
        help="Dust dimming strength [0–1.8]; 0=none.",
    ),
    nebula_debug_pass: Literal["normal", "occluder_only", "continuum_only"] = typer.Option(
        "normal",
        help="Diagnostic: normal nebula, occluder-only dust RGB, or continuum-only field (galaxy_streak only).",
    ),
    render_profile: str = typer.Option(
        "full",
        help="physical=linear only; physical_grade=+grade; full=+display finish/grain.",
    ),
    isolate_layer: IsolateLayer = typer.Option(
        IsolateLayer.none,
        "--isolate-layer",
        help="Final plate on black: none | background_stars | heroes | nebula_gas.",
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
    noheroes: bool = typer.Option(
        False,
        "--noheroes",
        help="Disable hero stars (spikes/ghosts); independent of --no-stars.",
    ),
    background_density: float | None = typer.Option(
        None,
        "--background-density",
        min=0.0,
        max=2.0,
        help="Off-plane/cosmic/halo star density [0–2]; blank=look default (~0.30 cinematic).",
    ),
    guided: bool = typer.Option(
        True,
        "--guided/--no-guided",
        help=(
            "Interactive prompts when no other flags are passed (default). "
            "Any CLI flag skips guided and uses args + defaults. "
            "Force with --guided; disable with --no-guided."
        ),
    ),
) -> None:
    # Resolve from argv so any real CLI flag skips guided (Typer default stays True).
    _ = guided
    did_guided = False
    if should_run_guided():
        did_guided = True
        guided_defaults = {
            "width": width,
            "height": height,
            "output_base_name": output_base_name,
            "output_dir": output_dir,
            "generations": generations,
            "seed": seed,
            "projection_mode": projection_mode,
            "cube": cube,
            "output_format": output_format,
            "scene_mode": scene_mode,
            "nebula_mode": nebula_mode,
            "quality": quality,
            "cubemap_face_size": cubemap_face_size,
            "stars": stars,
            "depth": depth,
            "nebula": nebula,
            "galaxy_view": galaxy_view,
            "background_gradient": background_gradient,
            "black_background": black_background,
            "jpeg_artifact_pass": jpeg_artifact_pass,
            "look_target": look_target,
            "background_texture_strength": background_texture_strength,
            "nebula_style": nebula_style,
            "cloud_continuity": cloud_continuity,
            "dust_coverage": dust_coverage,
            "dust_strength": dust_strength,
            "nebula_debug_pass": nebula_debug_pass,
            "render_profile": render_profile,
            "isolate_layer": isolate_layer,
            "debug_export_layers": debug_export_layers,
            "debug_grayscale_morphology": debug_grayscale_morphology,
            "morphology_void_strength": morphology_void_strength,
            "morphology_scar_strength": morphology_scar_strength,
            "heroes": not noheroes,
            "background_density": background_density,
        }
        chosen = run_guided_prompts(console, guided_defaults)
        if chosen is None:
            raise typer.Exit(0)
        width = int(chosen["width"])
        height = int(chosen["height"])
        output_base_name = str(chosen["output_base_name"])
        output_dir = Path(chosen["output_dir"])
        generations = int(chosen["generations"])
        seed = chosen["seed"]
        projection_mode = chosen["projection_mode"]
        if not isinstance(projection_mode, ProjectionMode):
            projection_mode = ProjectionMode(str(projection_mode))
        cube = bool(chosen["cube"])
        output_format = chosen["output_format"]
        if not isinstance(output_format, OutputFormat):
            output_format = OutputFormat(str(output_format))
        scene_mode = chosen.get("scene_mode", scene_mode)
        if not isinstance(scene_mode, SceneMode):
            scene_mode = SceneMode(str(scene_mode))
        nebula_mode = chosen["nebula_mode"]
        if not isinstance(nebula_mode, NebulaMode):
            nebula_mode = NebulaMode(str(nebula_mode))
        quality = int(chosen["quality"])
        cubemap_face_size = int(chosen["cubemap_face_size"])
        stars = bool(chosen["stars"])
        depth = bool(chosen["depth"])
        nebula = bool(chosen["nebula"])
        galaxy_view = bool(chosen["galaxy_view"])
        background_gradient = bool(chosen["background_gradient"])
        black_background = bool(chosen["black_background"])
        jpeg_artifact_pass = bool(chosen["jpeg_artifact_pass"])
        look_target = chosen["look_target"]
        if not isinstance(look_target, LookTarget):
            look_target = LookTarget(str(look_target))
        # long_exposure look target is tracked deep-field (no trails); preset clears
        # long_exposure_look. Documentary also keeps the stacked-sky theater off.
        if look_target in (LookTarget.long_exposure, LookTarget.documentary):
            long_exposure_look = False
        background_texture_strength = float(chosen["background_texture_strength"])
        nebula_style = str(chosen["nebula_style"])  # type: ignore[assignment]
        cloud_continuity = float(chosen["cloud_continuity"])
        dust_coverage = float(chosen["dust_coverage"])
        dust_strength = float(chosen["dust_strength"])
        nebula_debug_pass = str(chosen["nebula_debug_pass"])  # type: ignore[assignment]
        render_profile = str(chosen["render_profile"])
        isolate_layer = chosen.get("isolate_layer", IsolateLayer.none)
        if not isinstance(isolate_layer, IsolateLayer):
            isolate_layer = IsolateLayer(str(isolate_layer))
        debug_export_layers = bool(chosen["debug_export_layers"])
        debug_grayscale_morphology = bool(chosen["debug_grayscale_morphology"])
        morphology_void_strength = float(chosen["morphology_void_strength"])
        morphology_scar_strength = float(chosen["morphology_scar_strength"])
        noheroes = not bool(chosen.get("heroes", True))
        background_density = chosen["background_density"]
        if background_density is not None:
            background_density = float(background_density)

        print_guided_confirm(console, chosen)
        try:
            if not confirm_start_render(console):
                console.print("[dim]Discovery only — no render.[/dim]")
                raise typer.Exit(0)
        except GuidedCancelled:
            console.print("[dim]Guided cancelled (no render).[/dim]")
            raise typer.Exit(0)

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
        scene_mode=scene_mode,
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
            look_target=look_target,
            long_exposure_look=long_exposure_look,
            background_texture_strength=background_texture_strength,
            render_profile=render_profile,
            debug_export_layers=debug_export_layers,
            debug_grayscale_morphology=debug_grayscale_morphology,
            morphology_void_strength=morphology_void_strength,
            morphology_scar_strength=morphology_scar_strength,
            hero_star_enabled=not noheroes,
            background_star_density_scale=background_density,
            isolate_layer=isolate_layer,
        ),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if not did_guided:
        print_render_preview(
            console,
            {
                "width": cfg.width,
                "height": cfg.height,
                "output_base_name": cfg.output_base_name,
                "output_dir": cfg.output_dir,
                "generations": cfg.generations,
                "seed": run_seed,
                "projection_mode": cfg.projection_mode,
                "cube": False,
                "output_format": cfg.output_format,
                "scene_mode": cfg.scene_mode,
                "nebula_mode": cfg.nebula_mode,
                "cubemap_face_size": cfg.cubemap_face_size,
                "stars": cfg.features.stars,
                "depth": cfg.features.depth,
                "nebula": cfg.features.nebula,
                "galaxy_view": cfg.features.galaxy_view,
                "background_gradient": cfg.features.background_gradient,
                "black_background": cfg.features.black_background,
                "look_target": cfg.features.look_target,
                "background_texture_strength": cfg.features.background_texture_strength,
                "nebula_style": cfg.nebula_tuning.style,
                "cloud_continuity": cfg.nebula_tuning.cloud_continuity,
                "dust_coverage": cfg.nebula_tuning.dust_coverage,
                "dust_strength": cfg.nebula_tuning.dust_strength,
                "render_profile": cfg.features.render_profile.value
                if hasattr(cfg.features.render_profile, "value")
                else str(cfg.features.render_profile),
                "isolate_layer": cfg.features.isolate_layer,
                "debug_export_layers": cfg.features.debug_export_layers,
                "debug_grayscale_morphology": cfg.features.debug_grayscale_morphology,
                "heroes": cfg.features.hero_star_enabled,
                "background_density": background_density,
            },
        )
    elif seed is None:
        console.print(f"Using random seed: {run_seed}")

    license_state = maybe_prompt_before_generate(console)
    cfg = cfg.model_copy(
        update={
            "image_copyright": license_state.normalized_copyright(),
            "image_license": license_state.normalized_distribution(),
        }
    )

    completed = 0
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
            completed += 1
            progress.remove_task(render_task)
            if batch_task is not None:
                progress.advance(batch_task, 1)

            console.print(f"Saved generation {i + 1}: {outputs}")
            if stats["color_counts"]:
                console.print(f"  Color counts: {stats['color_counts']}")
                console.print(f"  Size counts: {stats['size_counts']}")

    if completed > 0:
        look = getattr(cfg.features, "look_target", None)
        look_s = look.value if hasattr(look, "value") else (str(look) if look is not None else None)
        last_gen_seed = int(run_seed) + max(0, completed - 1)
        record_generations(
            license_state,
            completed,
            seed=last_gen_seed,
            look_target=look_s,
        )


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
    output_format: OutputFormat = typer.Option(
        OutputFormat.png, help="Output format (png, jpg, or webp)."
    ),
    nebula_mode: NebulaMode = typer.Option(
        NebulaMode.galaxy_streak,
        help="Nebula style: distant, full, or galaxy_streak.",
    ),
    quality: int = typer.Option(
        100, min=50, max=100, help="JPEG/WebP quality (ignored for PNG)."
    ),
    stars: bool = typer.Option(True, help="Enable star generation."),
    depth: bool = typer.Option(True, help="Enable depth-style intensity falloff."),
    nebula: bool = typer.Option(True, help="Enable nebula rendering."),
    galaxy_view: bool = typer.Option(True, help="Bias star density toward galactic plane."),
    background_gradient: bool = typer.Option(True, help="Add non-black galactic background gradient."),
    black_background: bool = typer.Option(False, help="Force black background."),
    look_target: LookTarget = typer.Option(
        LookTarget.cinematic,
        "--look-target",
        help=(
            "cinematic=AAA wrap-safe skybox; documentary=parked photo; "
            "long_exposure=tracked deep field (sharp points, denser faint stars)."
        ),
    ),
    long_exposure_look: bool = typer.Option(
        False,
        "--long-exposure-look/--flat-exposure",
        help=(
            "Advanced cinematic override (legacy stacked-sky theater). "
            "Prefer --look-target long_exposure for the tracked deep-field look."
        ),
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
    cloud_continuity: float = typer.Option(1.22, min=0.0, max=1.6, help="Cloud continuity [0–1.6]."),
    dust_coverage: float = typer.Option(0.98, min=0.0, max=1.6, help="Dust coverage [0–1.6]."),
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
            look_target=look_target,
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
