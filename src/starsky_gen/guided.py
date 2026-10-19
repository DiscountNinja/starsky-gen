"""Interactive --guided prompts for generate (newbie setup + flag discovery)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from rich.console import Console

from starsky_gen.config import (
    IsolateLayer,
    LookTarget,
    NebulaMode,
    OutputFormat,
    ProjectionMode,
    SceneMode,
)

FieldKind = Literal[
    "int",
    "float",
    "bool",
    "str",
    "path",
    "enum",
    "optional_int",
    "optional_float",
]
FieldGroup = Literal["core", "advanced"]

# Subcommands that are not option flags (argv after the program name).
_KNOWN_SUBCOMMANDS = frozenset({"generate", "assist"})


def should_run_guided(argv: Sequence[str] | None = None) -> bool:
    """Whether to enter guided prompts for this invocation.

    - Bare ``starsky-gen`` / ``starsky-gen generate`` → guided (newbie default).
    - Any ``-``/``--`` flag other than an explicit ``--guided`` → not guided
      (CLI args + defaults for everything else).
    - ``--guided`` forces guided even when other flags are present.
    - ``--no-guided`` always disables guided.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] in _KNOWN_SUBCOMMANDS:
        tokens = tokens[1:]

    has_no_guided = any(t == "--no-guided" or t.startswith("--no-guided=") for t in tokens)
    has_guided = any(t == "--guided" or t.startswith("--guided=") for t in tokens)
    if has_no_guided:
        return False
    if has_guided:
        return True
    if any(t.startswith("-") for t in tokens):
        return False
    return True


@dataclass(frozen=True)
class GuidedField:
    key: str
    flag: str
    label: str
    help: str
    kind: FieldKind
    default: Any
    group: FieldGroup = "core"
    choices: tuple[str, ...] | None = None
    min_v: float | None = None
    max_v: float | None = None
    # How to render non-default values into the equivalent CLI.
    cli_style: Literal["value", "flag", "neg_flag", "toggle_pair"] = "value"
    # For toggle_pair (long-exposure): affirmative / negative flags.
    flag_on: str | None = None
    flag_off: str | None = None


def _enum_values(enum_cls: type[Enum]) -> tuple[str, ...]:
    return tuple(m.value for m in enum_cls)


STOCK_DEFAULTS: dict[str, Any] = {
    "width": 2048,
    "height": 1024,
    "output_base_name": "starsky",
    "output_dir": Path("output"),
    "generations": 1,
    "seed": None,
    "projection_mode": ProjectionMode.equirectangular,
    "cube": False,
    "output_format": OutputFormat.png,
    "scene_mode": SceneMode.normal,
    "nebula_mode": NebulaMode.galaxy_streak,
    "quality": 100,
    "cubemap_face_size": 1024,
    "stars": True,
    "depth": True,
    "nebula": True,
    "galaxy_view": True,
    "background_gradient": True,
    "black_background": False,
    "jpeg_artifact_pass": False,
    "look_target": LookTarget.cinematic,
    "background_texture_strength": 1.0,
    "nebula_style": "balanced",
    "cloud_continuity": 1.22,
    "dust_coverage": 0.98,
    "dust_strength": 1.14,
    "nebula_debug_pass": "normal",
    "render_profile": "full",
    "isolate_layer": IsolateLayer.none,
    "debug_export_layers": False,
    "debug_grayscale_morphology": False,
    "morphology_void_strength": 0.72,
    "morphology_scar_strength": 0.68,
    "heroes": True,
    "background_density": None,
}


GUIDED_FIELDS: list[GuidedField] = [
    GuidedField(
        "scene_mode",
        "--mode",
        "Scene mode",
        "normal=Milky Way band skybox; nebula=full-sky cloud complex (no galactic band).",
        "enum",
        SceneMode.normal,
        choices=_enum_values(SceneMode),
    ),
    GuidedField(
        "width",
        "--width",
        "Width",
        "Output width for equirectangular renders.",
        "int",
        2048,
        min_v=64,
    ),
    GuidedField(
        "height",
        "--height",
        "Height",
        "Output height for equirectangular renders.",
        "int",
        1024,
        min_v=64,
    ),
    GuidedField(
        "output_base_name",
        "--output-base-name",
        "Output base name",
        "Base filename prefix.",
        "str",
        "starsky",
    ),
    GuidedField(
        "output_dir",
        "--output-dir",
        "Output directory",
        "Directory for generated images.",
        "path",
        Path("output"),
    ),
    GuidedField(
        "generations",
        "--generations",
        "Generations",
        "Number of generations to produce.",
        "int",
        1,
        min_v=1,
    ),
    GuidedField(
        "seed",
        "--seed",
        "Seed",
        "Base random seed for deterministic batches (blank = random).",
        "optional_int",
        None,
    ),
    GuidedField(
        "projection_mode",
        "--projection-mode",
        "Projection mode",
        "Projection target: equirectangular, cubemap, or both.",
        "enum",
        ProjectionMode.equirectangular,
        choices=_enum_values(ProjectionMode),
    ),
    GuidedField(
        "cube",
        "--cube",
        "Include cubemap",
        "Shortcut to include cubemap output (switches equirectangular to both).",
        "bool",
        False,
        cli_style="flag",
    ),
    GuidedField(
        "output_format",
        "--output-format",
        "Output format",
        "Output format (png, jpg, or webp).",
        "enum",
        OutputFormat.png,
        choices=_enum_values(OutputFormat),
    ),
    GuidedField(
        "quality",
        "--quality",
        "Encode quality",
        "JPEG/WebP quality (ignored for PNG).",
        "int",
        100,
        min_v=50,
        max_v=100,
    ),
    GuidedField(
        "cubemap_face_size",
        "--cubemap-face-size",
        "Cubemap face size",
        "Face size for cubemap outputs.",
        "int",
        1024,
        min_v=128,
        max_v=4096,
    ),
    GuidedField(
        "stars",
        "--stars/--no-stars",
        "Background stars",
        "Band/mid/cosmic/halo star catalogs (not hero stars).",
        "bool",
        True,
        cli_style="neg_flag",
        flag_on="--stars",
        flag_off="--no-stars",
    ),
    GuidedField(
        "heroes",
        "--noheroes",
        "Hero stars",
        "Bright heroes with spikes/ghosts (independent of background stars). Off → --noheroes.",
        "bool",
        True,
        cli_style="neg_flag",
        flag_on="--heroes",
        flag_off="--noheroes",
    ),
    GuidedField(
        "depth",
        "--depth/--no-depth",
        "Depth falloff",
        "Dim catalog stars toward the poles (not camera DOF).",
        "bool",
        True,
        cli_style="neg_flag",
        flag_on="--depth",
        flag_off="--no-depth",
    ),
    GuidedField(
        "nebula",
        "--nebula/--no-nebula",
        "Nebula",
        "Enable nebula gas rendering. Off skips nebula-mode/style/dust prompts.",
        "bool",
        True,
        cli_style="neg_flag",
        flag_on="--nebula",
        flag_off="--no-nebula",
    ),
    GuidedField(
        "galaxy_view",
        "--galaxy-view/--no-galaxy-view",
        "Galaxy view",
        "Enable Milky Way band + plane-biased stars (normal mode only; skipped in nebula scene).",
        "bool",
        True,
        cli_style="neg_flag",
        flag_on="--galaxy-view",
        flag_off="--no-galaxy-view",
    ),
    GuidedField(
        "background_gradient",
        "--background-gradient/--no-background-gradient",
        "Background gradient",
        "Add non-black galactic background gradient.",
        "bool",
        True,
        cli_style="neg_flag",
        flag_on="--background-gradient",
        flag_off="--no-background-gradient",
    ),
    GuidedField(
        "black_background",
        "--black-background/--no-black-background",
        "Black background",
        "Force black background (overrides gradient/noise background).",
        "bool",
        False,
        cli_style="flag",
    ),
    GuidedField(
        "look_target",
        "--look-target",
        "Look target",
        "cinematic=wrap-safe skybox; documentary=parked photo grade; "
        "long_exposure=tracked deep field (sharp points, denser faint stars).",
        "enum",
        LookTarget.cinematic,
        choices=_enum_values(LookTarget),
    ),
    GuidedField(
        "background_texture_strength",
        "--background-texture-strength",
        "Background texture strength",
        "Unresolved-star grain [0–2]: 0=smooth, 1=default, 2=gritty.",
        "float",
        1.0,
        min_v=0.0,
        max_v=2.0,
    ),
    GuidedField(
        "nebula_mode",
        "--nebula-mode",
        "Nebula mode",
        "distant=isolated feature; full=immersive fog; galaxy_streak=MW-like plane (normal mode).",
        "enum",
        NebulaMode.galaxy_streak,
        choices=_enum_values(NebulaMode),
    ),
    GuidedField(
        "nebula_style",
        "--nebula-style",
        "Nebula style",
        "Gas look preset: subtle, balanced, or dramatic.",
        "enum",
        "balanced",
        choices=("subtle", "balanced", "dramatic"),
    ),
    GuidedField(
        "cloud_continuity",
        "--cloud-continuity",
        "Cloud continuity",
        "Connect/thicken cloud volumes [0–1.6]; 0=off.",
        "float",
        1.22,
        min_v=0.0,
        max_v=1.6,
    ),
    GuidedField(
        "dust_coverage",
        "--dust-coverage",
        "Dust coverage",
        "How much area can hold dark dust [0–1.6]; 0=none.",
        "float",
        0.98,
        min_v=0.0,
        max_v=1.6,
    ),
    GuidedField(
        "dust_strength",
        "--dust-strength",
        "Dust strength",
        "How strongly dust dims stars/background [0–1.8]; 0=none.",
        "float",
        1.14,
        min_v=0.0,
        max_v=1.8,
    ),
    GuidedField(
        "background_density",
        "--background-density",
        "Background density",
        "Off-plane/cosmic/halo star density [0–2]; cinematic~0.30. Blank=look default. Skip if background stars off.",
        "optional_float",
        None,
        min_v=0.0,
        max_v=2.0,
    ),
    # Advanced
    GuidedField(
        "jpeg_artifact_pass",
        "--jpeg-artifact-pass/--no-jpeg-artifact-pass",
        "JPEG artifact pass",
        "Apply JPEG artifact emulation pass.",
        "bool",
        False,
        group="advanced",
        cli_style="flag",
    ),
    GuidedField(
        "render_profile",
        "--render-profile",
        "Render profile",
        "physical=linear only; physical_grade=+linear grade; full=+display finish/grain.",
        "enum",
        "full",
        group="advanced",
        choices=("physical", "physical_grade", "full"),
    ),
    GuidedField(
        "isolate_layer",
        "--isolate-layer",
        "Isolate layer",
        "Final plate on black: none | background_stars | heroes | nebula_gas.",
        "enum",
        IsolateLayer.none,
        group="advanced",
        choices=_enum_values(IsolateLayer),
    ),
    GuidedField(
        "nebula_debug_pass",
        "--nebula-debug-pass",
        "Nebula debug pass",
        "Diagnostic: normal, occluder_only, or continuum_only.",
        "enum",
        "normal",
        group="advanced",
        choices=("normal", "occluder_only", "continuum_only"),
    ),
    GuidedField(
        "debug_export_layers",
        "--debug-layers",
        "Export debug layers",
        "Write morphology/star/extinction layer PNGs.",
        "bool",
        False,
        group="advanced",
        cli_style="flag",
    ),
    GuidedField(
        "debug_grayscale_morphology",
        "--grayscale-morphology",
        "Grayscale morphology debug",
        "Write grayscale morphology diagnostic PNG.",
        "bool",
        False,
        group="advanced",
        cli_style="flag",
    ),
    GuidedField(
        "morphology_void_strength",
        "--morphology-void-strength",
        "Morphology void strength",
        "Strength of morphology void carving.",
        "float",
        0.72,
        group="advanced",
        min_v=0.0,
        max_v=1.0,
    ),
    GuidedField(
        "morphology_scar_strength",
        "--morphology-scar-strength",
        "Morphology scar strength",
        "Strength of morphology scar features.",
        "float",
        0.68,
        group="advanced",
        min_v=0.0,
        max_v=1.5,
    ),
]


def _default_display(value: Any, kind: FieldKind) -> str:
    if kind == "bool":
        return "yes" if bool(value) else "no"
    if value is None:
        return "(unset)"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _default_markup(value: Any, kind: FieldKind | None = None) -> str:
    """Rich markup: default value in dark green for guided prompts."""
    text = _default_display(value, kind) if kind is not None else str(value)
    return f"[dark_green]{text}[/dark_green]"


def parse_bool(text: str) -> bool | None:
    key = text.strip().lower()
    if key in ("y", "yes", "true", "t", "1", "on"):
        return True
    if key in ("n", "no", "false", "f", "0", "off"):
        return False
    return None


def parse_int(text: str, *, min_v: float | None = None, max_v: float | None = None) -> int | None:
    try:
        v = int(text.strip())
    except (TypeError, ValueError):
        return None
    if min_v is not None and v < min_v:
        return None
    if max_v is not None and v > max_v:
        return None
    return v


def parse_float(text: str, *, min_v: float | None = None, max_v: float | None = None) -> float | None:
    try:
        v = float(text.strip())
    except (TypeError, ValueError):
        return None
    if min_v is not None and v < min_v:
        return None
    if max_v is not None and v > max_v:
        return None
    return v


class GuidedCancelled(Exception):
    """User typed q / cancelled guided mode."""


def _read_line(prompt: str = "> ") -> str:
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError) as exc:
        raise GuidedCancelled from exc


def _check_quit(raw: str) -> None:
    if raw.strip().lower() == "q":
        raise GuidedCancelled


def prompt_size_preset(console: Console, defaults: dict[str, Any]) -> tuple[int, int] | None:
    """Return (width, height), or None to prompt width/height as custom fields.

    Enter keeps the current size (defaults to 2048×1024). Raises GuidedCancelled on q.
    """
    cur_w = int(defaults.get("width", 2048))
    cur_h = int(defaults.get("height", 1024))
    console.print("[bold]Size preset[/bold]")
    console.print("  [dim]flags:[/dim] --width / --height")
    console.print("  1) 2048×1024 (2K equirect)")
    console.print("  2) 4096×2048 (4K skybox)")
    console.print("  3) 8192×4096 (8K skybox)")
    console.print("  4) Custom width × height")
    console.print(
        f"  [dim](current:[/dim] {_default_markup(f'{cur_w}×{cur_h}')}"
        f"[dim]; Enter=keep, q=quit)[/dim]"
    )
    while True:
        raw = _read_line()
        _check_quit(raw)
        if not raw.strip():
            return cur_w, cur_h
        key = raw.strip().lower()
        if key in ("1",):
            return 2048, 1024
        if key in ("2",):
            return 4096, 2048
        if key in ("3",):
            return 8192, 4096
        if key in ("4", "c", "custom"):
            return None
        console.print("[dim]Enter 1–4, Enter to keep current, or q.[/dim]")


def prompt_output_type(
    console: Console, defaults: dict[str, Any]
) -> tuple[ProjectionMode, bool]:
    """Choose generation type: equirect / cubemap / both.

    Sets ``projection_mode`` and clears ``cube`` (mode encodes the choice).
    Enter keeps current. Raises GuidedCancelled on q.
    """
    cur_mode = defaults.get("projection_mode", ProjectionMode.equirectangular)
    if not isinstance(cur_mode, ProjectionMode):
        cur_mode = ProjectionMode(str(cur_mode))
    cur_cube = bool(defaults.get("cube", False))
    if cur_cube and cur_mode == ProjectionMode.equirectangular:
        cur_mode = ProjectionMode.both

    console.print("[bold]Output type[/bold]")
    console.print("  [dim]flags:[/dim] --projection-mode / --cube")
    console.print("  1) Equirectangular only (default)")
    console.print("  2) Cubemap only")
    console.print("  3) Both equirectangular and cubemap")
    console.print(
        f"  [dim](current:[/dim] {_default_markup(cur_mode.value)}"
        f"[dim]; Enter=keep, q=quit)[/dim]"
    )
    while True:
        raw = _read_line()
        _check_quit(raw)
        if not raw.strip():
            return cur_mode, False
        key = raw.strip().lower()
        if key in ("1", "e", "equirect", "equirectangular"):
            return ProjectionMode.equirectangular, False
        if key in ("2", "c", "cube", "cubemap"):
            return ProjectionMode.cubemap, False
        if key in ("3", "b", "both"):
            return ProjectionMode.both, False
        console.print("[dim]Enter 1–3, Enter to keep current, or q.[/dim]")


def prompt_field(
    console: Console,
    spec: GuidedField,
    current: Any,
) -> Any:
    """Prompt one field; Enter keeps current. Raises GuidedCancelled on q."""
    console.print(f"[bold]{spec.label}[/bold]")
    console.print(f"  [dim]flag:[/dim] {spec.flag}")
    console.print(f"  [dim]{spec.help}[/dim]")
    if spec.choices:
        console.print(f"  [dim]choices:[/dim] {', '.join(spec.choices)}")
    console.print(
        f"  [dim](default:[/dim] {_default_markup(current, spec.kind)}"
        f"[dim]; Enter=keep, q=quit)[/dim]"
    )
    while True:
        raw = _read_line()
        _check_quit(raw)
        # Bare Enter / whitespace always keeps the current value (spam-Enter friendly).
        if not raw.strip():
            if spec.kind == "str":
                kept = str(current).strip() if current is not None else ""
                return kept if kept else str(spec.default)
            if spec.kind == "path":
                return Path(current) if current is not None else Path(spec.default)
            return current

        text = raw.strip()
        if spec.kind == "bool":
            parsed = parse_bool(text)
            if parsed is None:
                console.print("[dim]Enter y/n, true/false, 1/0, or Enter.[/dim]")
                continue
            return parsed
        if spec.kind == "int":
            parsed_i = parse_int(text, min_v=spec.min_v, max_v=spec.max_v)
            if parsed_i is None:
                console.print("[dim]Enter a valid integer (within range), or Enter.[/dim]")
                continue
            return parsed_i
        if spec.kind == "optional_int":
            if text.lower() in ("none", "-", "unset"):
                return None
            parsed_i = parse_int(text, min_v=spec.min_v, max_v=spec.max_v)
            if parsed_i is None:
                console.print("[dim]Enter an integer, blank/Enter to leave unset, or q.[/dim]")
                continue
            return parsed_i
        if spec.kind == "float":
            parsed_f = parse_float(text, min_v=spec.min_v, max_v=spec.max_v)
            if parsed_f is None:
                console.print("[dim]Enter a valid number (within range), or Enter.[/dim]")
                continue
            return parsed_f
        if spec.kind == "optional_float":
            if text.lower() in ("none", "-", "unset"):
                return None
            parsed_f = parse_float(text, min_v=spec.min_v, max_v=spec.max_v)
            if parsed_f is None:
                console.print("[dim]Enter a number, blank/Enter to leave unset, or q.[/dim]")
                continue
            return parsed_f
        if spec.kind == "path":
            return Path(text)
        if spec.kind == "str":
            # Non-empty answer required only when typing a replacement; Enter already kept default.
            return text
        if spec.kind == "enum":
            assert spec.choices is not None
            low = text.lower()
            for choice in spec.choices:
                if choice.lower() == low:
                    # Preserve Enum type when default is an Enum.
                    if isinstance(spec.default, Enum):
                        return type(spec.default)(choice)
                    return choice
            console.print(f"[dim]Choose one of: {', '.join(spec.choices)}[/dim]")
            continue
        return text


def prompt_yes_no(
    console: Console,
    question: str,
    *,
    default: bool,
) -> bool:
    """Yes/no with default; raises GuidedCancelled on q."""
    hint = "Y/n" if default else "y/N"
    default_yn = "yes" if default else "no"
    console.print(
        f"[bold]{question}[/bold] "
        f"[dim][{hint}; default[/dim] {_default_markup(default_yn)}[dim]; q=quit][/dim]"
    )
    while True:
        raw = _read_line()
        _check_quit(raw)
        if not raw.strip():
            return default
        parsed = parse_bool(raw)
        if parsed is None:
            console.print("[dim]Enter y/n, Enter, or q.[/dim]")
            continue
        return parsed


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, Enum):
        a = a.value
    if isinstance(b, Enum):
        b = b.value
    if isinstance(a, Path) or isinstance(b, Path):
        return str(a) == str(b)
    if a is None and b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < 1e-9
    return a == b


def format_equivalent_command(
    values: dict[str, Any],
    *,
    stock: dict[str, Any] | None = None,
) -> str:
    """Build ``starsky-gen generate …`` with only non-stock flags."""
    base = stock if stock is not None else STOCK_DEFAULTS
    parts: list[str] = ["starsky-gen", "generate"]
    by_key = {f.key: f for f in GUIDED_FIELDS}
    for key, stock_default in base.items():
        if key not in values:
            continue
        val = values[key]
        if _values_equal(val, stock_default):
            continue
        spec = by_key.get(key)
        if spec is None:
            continue
        if spec.cli_style == "flag":
            if bool(val):
                parts.append(spec.flag.split("/")[0].split()[0] if "/" in spec.flag else spec.flag)
            continue
        if spec.cli_style == "neg_flag":
            on = spec.flag_on or spec.flag.split("/")[0]
            off = spec.flag_off or (
                spec.flag.split("/")[1] if "/" in spec.flag else f"--no-{key.replace('_', '-')}"
            )
            # heroes: only `--noheroes` exists on CLI (no affirmative flag).
            if key == "heroes":
                if not bool(val):
                    parts.append(off)
                continue
            parts.append(on if bool(val) else off)
            continue
        if spec.cli_style == "toggle_pair":
            on = spec.flag_on or "--long-exposure-look"
            off = spec.flag_off or "--flat-exposure"
            parts.append(on if bool(val) else off)
            continue
        # value style
        flag = spec.flag.split("/")[0]
        if isinstance(val, Enum):
            disp = val.value
        elif isinstance(val, Path):
            disp = str(val)
        else:
            disp = str(val)
        # Quote if needed
        if any(ch.isspace() for ch in disp) or disp == "":
            disp = repr(disp)
        parts.append(flag)
        parts.append(disp)
    return " ".join(parts)


def _as_label(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return "(unset)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _scene_intent_line(values: dict[str, Any]) -> str:
    """One-line plain-language description of what the render will look like."""
    scene = values.get("scene_mode", SceneMode.normal)
    scene_s = scene.value if isinstance(scene, Enum) else str(scene)
    stars = bool(values.get("stars", True))
    heroes = bool(values.get("heroes", True))
    # Guided uses heroes=True; CLI path may pass noheroes.
    if "heroes" not in values and "noheroes" in values:
        heroes = not bool(values.get("noheroes"))
    nebula = bool(values.get("nebula", True))
    black = bool(values.get("black_background", False))
    isolate = values.get("isolate_layer", IsolateLayer.none)
    if isinstance(isolate, Enum):
        isolate_s = isolate.value
    else:
        isolate_s = str(isolate or "none")

    if isolate_s not in ("none", ""):
        labels = {
            "background_stars": "background-star plate only (on black)",
            "heroes": "hero-star plate only (on black)",
            "nebula_gas": "nebula-gas plate only (on black)",
        }
        return labels.get(isolate_s, f"isolated {isolate_s} plate")

    parts: list[str] = []
    if scene_s == "nebula":
        parts.append("full-sky cloud-complex scene (no MW band)")
    else:
        galaxy = bool(values.get("galaxy_view", True))
        parts.append("Milky Way band skybox" if galaxy else "skybox without MW band")

    layer_bits: list[str] = []
    if nebula:
        style = _as_label(values.get("nebula_style", "balanced"))
        layer_bits.append(f"nebula gas ({style})")
    else:
        layer_bits.append("no nebula gas")
    if stars and heroes:
        layer_bits.append("background stars + heroes")
    elif stars:
        layer_bits.append("background stars only")
    elif heroes:
        layer_bits.append("hero stars only")
    else:
        layer_bits.append("no stars")
    if black:
        layer_bits.append("black void")
    parts.append(", ".join(layer_bits))
    look = _as_label(values.get("look_target", "cinematic"))
    parts.append(f"{look} look")
    return " · ".join(parts)


def format_guided_summary(values: dict[str, Any]) -> list[str]:
    """Short human-readable lines for the pre-render confirm (what you will get)."""
    w = int(values.get("width", 2048))
    h = int(values.get("height", 1024))
    proj = _as_label(values.get("projection_mode", "equirectangular"))
    cube = bool(values.get("cube", False))
    if cube and proj == "equirectangular":
        proj = "both (--cube)"

    seed = values.get("seed")
    seed_s = _as_label(seed) if seed is not None else "random"
    dens = values.get("background_density")
    dens_s = _as_label(dens) if dens is not None else "look default"

    heroes = bool(values.get("heroes", True))
    if "heroes" not in values and "noheroes" in values:
        heroes = not bool(values.get("noheroes"))

    scene = values.get("scene_mode", SceneMode.normal)
    scene_s = scene.value if isinstance(scene, Enum) else str(scene)
    galaxy = bool(values.get("galaxy_view", True))
    if scene_s == "nebula":
        galaxy_s = "n/a (nebula scene)"
    else:
        galaxy_s = "on (MW band)" if galaxy else "off (no band)"

    lines = [
        f"Intent: {_scene_intent_line(values)}",
        f"Canvas: {w}×{h}  ·  {proj}  ·  "
        f"{_as_label(values.get('output_format', 'png'))}",
        f"Output: {_as_label(values.get('output_dir', 'output'))}/"
        f"{_as_label(values.get('output_base_name', 'starsky'))}  ·  "
        f"{int(values.get('generations', 1))} gen  ·  seed {seed_s}",
        f"Look: {_as_label(values.get('look_target', 'cinematic'))}  ·  "
        f"Scene: {scene_s}  ·  Galaxy view: {galaxy_s}",
        f"Layers: bg stars={'on' if values.get('stars', True) else 'off'}  ·  "
        f"heroes={'on' if heroes else 'off'}  ·  "
        f"nebula={'on' if values.get('nebula', True) else 'off'}",
    ]
    if values.get("nebula", True):
        lines.append(
            f"Nebula: {_as_label(values.get('nebula_mode', 'galaxy_streak'))} / "
            f"{_as_label(values.get('nebula_style', 'balanced'))}  ·  "
            f"continuity {_as_label(values.get('cloud_continuity', 1.22))}  ·  "
            f"dust {_as_label(values.get('dust_coverage', 0.98))}/"
            f"{_as_label(values.get('dust_strength', 1.14))}"
        )
    if values.get("stars", True):
        lines.append(
            f"Star field: density {dens_s}  ·  "
            f"texture {_as_label(values.get('background_texture_strength', 1.0))}  ·  "
            f"depth falloff={'on' if values.get('depth', True) else 'off'}"
        )
    isolate = values.get("isolate_layer", IsolateLayer.none)
    if isolate not in (None, IsolateLayer.none, "none"):
        lines.append(f"Isolate: {_as_label(isolate)} plate on black")
    if values.get("black_background"):
        lines.append("Background: forced black")
    elif not values.get("background_gradient", True):
        lines.append("Background: no gradient")
    if proj in ("cubemap", "both", "both (--cube)"):
        lines.append(f"Cubemap face: {int(values.get('cubemap_face_size', 1024))}px")
    profile = values.get("render_profile", "full")
    if profile not in (None, "full"):
        lines.append(f"Render profile: {_as_label(profile)}")
    if values.get("debug_export_layers") or values.get("debug_grayscale_morphology"):
        bits = []
        if values.get("debug_export_layers"):
            bits.append("debug layers")
        if values.get("debug_grayscale_morphology"):
            bits.append("grayscale morphology")
        lines.append(f"Debug: {', '.join(bits)}")
    return lines


def print_guided_confirm(console: Console, values: dict[str, Any]) -> None:
    """Print summary + equivalent CLI before Start render?"""
    console.print("")
    console.print("[bold]About to render[/bold]")
    for line in format_guided_summary(values):
        console.print(f"  {line}")
    console.print("")
    console.print("[bold]Equivalent command:[/bold]")
    console.print(f"  {format_equivalent_command(values)}")
    console.print("")


def print_render_preview(console: Console, values: dict[str, Any]) -> None:
    """Compact pre-render recap for non-guided CLI runs."""
    console.print("")
    console.print("[bold]About to render[/bold]")
    for line in format_guided_summary(values):
        console.print(f"  {line}")
    console.print("")


def confirm_start_render(console: Console) -> bool:
    return prompt_yes_no(console, "Start render?", default=True)


def should_prompt(spec: GuidedField, answers: dict[str, Any]) -> bool:
    """Skip follow-up prompts when a parent layer/mode is off."""
    key = spec.key
    scene = answers.get("scene_mode", SceneMode.normal)
    if isinstance(scene, str):
        try:
            scene = SceneMode(scene)
        except ValueError:
            scene = SceneMode.normal
    if key == "galaxy_view" and scene == SceneMode.nebula:
        return False
    if key in (
        "nebula_mode",
        "nebula_style",
        "cloud_continuity",
        "dust_coverage",
        "dust_strength",
        "nebula_debug_pass",
    ) and not bool(answers.get("nebula", True)):
        return False
    if key == "background_density" and not bool(answers.get("stars", True)):
        return False
    if key == "depth" and not bool(answers.get("stars", True)) and not bool(
        answers.get("heroes", True)
    ):
        return False
    return True


def run_guided_prompts(
    console: Console,
    defaults: dict[str, Any],
) -> dict[str, Any] | None:
    """Walk size → output type → core → optional advanced.

    Enter accepts the current/default at every step (spam Enter for a default run).
    Returns chosen values, or None if the user quit with q / Ctrl+C.
    Does not ask Start render — caller prints summary + command then confirms.
    """
    values = dict(STOCK_DEFAULTS)
    values.update(defaults)

    console.print(
        "[bold]Guided mode[/bold] — [bold]Enter[/bold] keeps defaults (spam Enter to skip ahead). "
        "Each step shows the CLI flag. Type [bold]q[/bold] anytime to cancel."
    )
    console.print("")

    try:
        size = prompt_size_preset(console, values)
    except GuidedCancelled:
        console.print("[dim]Guided cancelled (no render).[/dim]")
        return None

    skip_wh = size is not None
    if size is not None:
        values["width"], values["height"] = size

    console.print("")
    try:
        mode, cube = prompt_output_type(console, values)
    except GuidedCancelled:
        console.print("[dim]Guided cancelled (no render).[/dim]")
        return None
    values["projection_mode"] = mode
    values["cube"] = cube

    # Scene mode early (after size / output type).
    scene_spec = next(f for f in GUIDED_FIELDS if f.key == "scene_mode")
    console.print("")
    try:
        values["scene_mode"] = prompt_field(
            console, scene_spec, values.get("scene_mode", scene_spec.default)
        )
    except GuidedCancelled:
        console.print("[dim]Guided cancelled (no render).[/dim]")
        return None

    skip_proj = {"projection_mode", "cube", "scene_mode"}
    for spec in GUIDED_FIELDS:
        if spec.group != "core":
            continue
        if skip_wh and spec.key in ("width", "height"):
            continue
        if spec.key in skip_proj:
            continue
        if not should_prompt(spec, values):
            continue
        console.print("")
        try:
            values[spec.key] = prompt_field(console, spec, values.get(spec.key, spec.default))
        except GuidedCancelled:
            console.print("[dim]Guided cancelled (no render).[/dim]")
            return None

    # Nebula scene never uses MW band; keep galaxy_view True for grade/photoreal.
    if values.get("scene_mode") == SceneMode.nebula or values.get("scene_mode") == "nebula":
        values["galaxy_view"] = True

    console.print("")
    try:
        do_adv = prompt_yes_no(console, "Configure advanced options?", default=False)
    except GuidedCancelled:
        console.print("[dim]Guided cancelled (no render).[/dim]")
        return None

    if do_adv:
        for spec in GUIDED_FIELDS:
            if spec.group != "advanced":
                continue
            if not should_prompt(spec, values):
                continue
            console.print("")
            try:
                values[spec.key] = prompt_field(
                    console, spec, values.get(spec.key, spec.default)
                )
            except GuidedCancelled:
                console.print("[dim]Guided cancelled (no render).[/dim]")
                return None

    return values
