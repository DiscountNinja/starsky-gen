"""CLI default persistence: argv detection, kebab↔param map, apply/show helpers."""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

try:
    from click.core import ParameterSource
except ImportError:  # Typer>=0.16 vendors Click as typer._click
    from typer._click.core import ParameterSource

from starsky_gen.config import (
    IsolateLayer,
    LookTarget,
    NebulaMode,
    OutputFormat,
    ProjectionMode,
    SceneMode,
)
from starsky_gen.guided import STOCK_DEFAULTS

# Meta flags never persisted / never treated as generate options.
META_OPTION_NAMES: frozenset[str] = frozenset(
    {
        "default",
        "show-defaults",
        "guided",
        "help",
    }
)

_KNOWN_SUBCOMMANDS = frozenset({"generate", "assist"})

# python_param → kebab store key (matches CLI option names in .license defaults).
PARAM_TO_KEBAB: dict[str, str] = {
    "width": "width",
    "height": "height",
    "output_base_name": "output-base-name",
    "output_dir": "output-dir",
    "generations": "generations",
    "seed": "seed",
    "projection_mode": "projection-mode",
    "cube": "cube",
    "output_format": "output-format",
    "scene_mode": "mode",
    "nebula_mode": "nebula-mode",
    "quality": "quality",
    "cubemap_face_size": "cubemap-face-size",
    "stars": "stars",
    "depth": "depth",
    "nebula": "nebula",
    "galaxy_view": "galaxy-view",
    "background_gradient": "background-gradient",
    "black_background": "black-background",
    "nebula_haze": "haze",
    "jpeg_artifact_pass": "jpeg-artifact-pass",
    "look_target": "look-target",
    "long_exposure_look": "long-exposure-look",
    "background_texture_strength": "background-texture-strength",
    "nebula_style": "nebula-style",
    "cloud_continuity": "cloud-continuity",
    "dust_coverage": "dust-coverage",
    "dust_strength": "dust-strength",
    "nebula_hero_color": "nebula-hero-color",
    "nebula_interact_density": "nebula-interact-density",
    "nebula_interact_strength": "nebula-interact-strength",
    "nebula_debug_pass": "nebula-debug-pass",
    "render_profile": "render-profile",
    "isolate_layer": "isolate-layer",
    "debug_export_layers": "debug-layers",
    "debug_grayscale_morphology": "grayscale-morphology",
    "morphology_void_strength": "morphology-void-strength",
    "morphology_scar_strength": "morphology-scar-strength",
    "noheroes": "noheroes",
    "background_density": "background-density",
}

KEBAB_TO_PARAM: dict[str, str] = {v: k for k, v in PARAM_TO_KEBAB.items()}

# Argv long-option names (without --) → python param. Covers --no-foo / toggle pairs.
FLAG_TO_PARAM: dict[str, str] = {
    "width": "width",
    "height": "height",
    "output-base-name": "output_base_name",
    "output-dir": "output_dir",
    "o": "output_dir",
    "generations": "generations",
    "seed": "seed",
    "projection-mode": "projection_mode",
    "cube": "cube",
    "output-format": "output_format",
    "mode": "scene_mode",
    "nebula-mode": "nebula_mode",
    "quality": "quality",
    "cubemap-face-size": "cubemap_face_size",
    "stars": "stars",
    "no-stars": "stars",
    "depth": "depth",
    "no-depth": "depth",
    "nebula": "nebula",
    "no-nebula": "nebula",
    "galaxy-view": "galaxy_view",
    "no-galaxy-view": "galaxy_view",
    "background-gradient": "background_gradient",
    "no-background-gradient": "background_gradient",
    "black-background": "black_background",
    "no-black-background": "black_background",
    "haze": "nebula_haze",
    "no-haze": "nebula_haze",
    "jpeg-artifact-pass": "jpeg_artifact_pass",
    "no-jpeg-artifact-pass": "jpeg_artifact_pass",
    "look-target": "look_target",
    "long-exposure-look": "long_exposure_look",
    "flat-exposure": "long_exposure_look",
    "background-texture-strength": "background_texture_strength",
    "nebula-style": "nebula_style",
    "cloud-continuity": "cloud_continuity",
    "dust-coverage": "dust_coverage",
    "dust-strength": "dust_strength",
    "nebula-hero-color": "nebula_hero_color",
    "nebula-interact-density": "nebula_interact_density",
    "nebula-interact-strength": "nebula_interact_strength",
    "nebula-debug-pass": "nebula_debug_pass",
    "render-profile": "render_profile",
    "isolate-layer": "isolate_layer",
    "debug-layers": "debug_export_layers",
    "grayscale-morphology": "debug_grayscale_morphology",
    "morphology-void-strength": "morphology_void_strength",
    "morphology-scar-strength": "morphology_scar_strength",
    "noheroes": "noheroes",
    "background-density": "background_density",
}

_ENUM_BY_PARAM: dict[str, type[Enum]] = {
    "projection_mode": ProjectionMode,
    "output_format": OutputFormat,
    "scene_mode": SceneMode,
    "nebula_mode": NebulaMode,
    "look_target": LookTarget,
    "isolate_layer": IsolateLayer,
}


def strip_subcommand(argv: Sequence[str]) -> list[str]:
    tokens = list(argv)
    if tokens and tokens[0] in _KNOWN_SUBCOMMANDS:
        return tokens[1:]
    return tokens


def argv_has_option(name: str, argv: Sequence[str] | None = None) -> bool:
    """True if ``--name`` or ``--name=...`` appears on argv (after optional subcommand)."""
    tokens = strip_subcommand(sys.argv[1:] if argv is None else argv)
    needle = f"--{name}"
    for t in tokens:
        if t == needle or t.startswith(f"{needle}="):
            return True
    return False


def provided_params(argv: Sequence[str] | None = None) -> set[str]:
    """Python param names for generate options explicitly present on argv."""
    tokens = strip_subcommand(sys.argv[1:] if argv is None else argv)
    found: set[str] = set()
    for t in tokens:
        if t == "--" or t in ("-h", "--help"):
            continue
        name: str | None = None
        if t.startswith("--"):
            name = t[2:].split("=", 1)[0]
        elif t.startswith("-") and len(t) == 2:
            # Short form (-o); only -o is mapped.
            name = t[1:]
        elif t.startswith("-") and not t.startswith("--"):
            # Typer also accepts -cube; treat as long-ish.
            name = t.lstrip("-")
        if name is None or name in META_OPTION_NAMES:
            continue
        param = FLAG_TO_PARAM.get(name)
        if param is not None:
            found.add(param)
    return found


def provided_params_from_ctx(ctx: Any) -> set[str]:
    """Python params whose Click/Typer source is not the declared default."""
    found: set[str] = set()
    get_source = getattr(ctx, "get_parameter_source", None)
    if get_source is None:
        return provided_params()
    for param in PARAM_TO_KEBAB:
        try:
            src = get_source(param)
        except Exception:
            continue
        if src is not None and src != ParameterSource.DEFAULT:
            found.add(param)
    return found


def serialize_value(value: Any) -> Any:
    """YAML-friendly encoding for a generate kwarg."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def deserialize_value(param: str, raw: Any) -> Any:
    """Convert a stored default into a generate kwarg value."""
    if raw is None:
        return None
    enum_cls = _ENUM_BY_PARAM.get(param)
    if enum_cls is not None:
        if isinstance(raw, enum_cls):
            return raw
        return enum_cls(str(raw))
    if param == "output_dir":
        return Path(str(raw))
    if param in (
        "stars",
        "depth",
        "nebula",
        "galaxy_view",
        "background_gradient",
        "black_background",
        "nebula_haze",
        "jpeg_artifact_pass",
        "long_exposure_look",
        "cube",
        "debug_export_layers",
        "debug_grayscale_morphology",
        "noheroes",
    ):
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(raw)
    if param in (
        "width",
        "height",
        "generations",
        "quality",
        "cubemap_face_size",
        "seed",
    ):
        if raw is None or raw == "":
            return None if param == "seed" else raw
        return int(raw)
    if param in (
        "background_texture_strength",
        "cloud_continuity",
        "dust_coverage",
        "dust_strength",
        "nebula_interact_density",
        "nebula_interact_strength",
        "morphology_void_strength",
        "morphology_scar_strength",
        "background_density",
    ):
        if raw is None or raw == "":
            return None
        return float(raw)
    return raw


def updates_from_kwargs(kwargs: dict[str, Any], params: set[str]) -> dict[str, Any]:
    """Build kebab defaults map from kwargs for the given python params."""
    out: dict[str, Any] = {}
    for param in sorted(params):
        kebab = PARAM_TO_KEBAB.get(param)
        if kebab is None or param not in kwargs:
            continue
        out[kebab] = serialize_value(kwargs[param])
    return out


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(raw)


def apply_saved_defaults(
    kwargs: dict[str, Any],
    saved: dict[str, Any],
    *,
    provided: set[str] | None = None,
) -> dict[str, Any]:
    """Overlay saved defaults onto kwargs for params not present on argv."""
    present = provided if provided is not None else provided_params()
    out = dict(kwargs)
    for kebab, raw in saved.items():
        if kebab in ("heroes", "noheroes"):
            if "noheroes" in present:
                continue
            # heroes:true → noheroes false; noheroes:true → noheroes true
            if kebab == "heroes":
                out["noheroes"] = not _as_bool(raw)
            else:
                out["noheroes"] = _as_bool(raw)
            continue
        param = KEBAB_TO_PARAM.get(kebab)
        if param is None:
            snake = kebab.replace("-", "_")
            if snake in PARAM_TO_KEBAB:
                param = snake
            else:
                continue
        if param in present:
            continue
        try:
            out[param] = deserialize_value(param, raw)
        except (TypeError, ValueError):
            continue
    return out


def guided_overlay_from_saved(saved: dict[str, Any]) -> dict[str, Any]:
    """Map kebab CLI defaults onto guided keys (heroes instead of noheroes)."""
    overlay: dict[str, Any] = {}
    for kebab, raw in saved.items():
        if kebab in ("noheroes", "heroes"):
            if kebab == "noheroes":
                try:
                    overlay["heroes"] = not bool(deserialize_value("noheroes", raw))
                except (TypeError, ValueError):
                    pass
            else:
                try:
                    if isinstance(raw, bool):
                        overlay["heroes"] = raw
                    else:
                        overlay["heroes"] = str(raw).strip().lower() in (
                            "1",
                            "true",
                            "yes",
                            "y",
                            "on",
                        )
                except (TypeError, ValueError):
                    pass
            continue
        param = KEBAB_TO_PARAM.get(kebab)
        if param is None:
            continue
        # Guided uses the same keys as STOCK_DEFAULTS for most fields.
        guided_key = param
        if param not in STOCK_DEFAULTS and param != "nebula_hero_color":
            # Skip CLI-only knobs guided does not prompt (still applied via kwargs path).
            if param in (
                "long_exposure_look",
                "nebula_interact_density",
                "nebula_interact_strength",
                "nebula_hero_color",
            ):
                continue
        try:
            overlay[guided_key] = deserialize_value(param, raw)
        except (TypeError, ValueError):
            continue
    return overlay


def project_defaults_for_display() -> dict[str, Any]:
    """Built-in project assumptions as kebab → serialized value."""
    out: dict[str, Any] = {}
    # STOCK_DEFAULTS uses heroes; display as noheroes=false for CLI alignment.
    for key, value in STOCK_DEFAULTS.items():
        if key == "heroes":
            out["noheroes"] = serialize_value(not bool(value))
            continue
        kebab = PARAM_TO_KEBAB.get(key, key.replace("_", "-"))
        out[kebab] = serialize_value(value)
    # CLI-only stocks not in STOCK_DEFAULTS.
    out.setdefault("nebula-hero-color", "magenta")
    out.setdefault("long-exposure-look", False)
    out.setdefault("nebula-interact-density", None)
    out.setdefault("nebula-interact-strength", None)
    return out


def format_defaults_value(value: Any) -> str:
    """Plain text for a defaults value (no Rich markup)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_defaults_value_markup(value: Any) -> str:
    """Rich markup for a defaults value: cyan numbers, magenta bools, green strings, dim null."""
    from rich.markup import escape

    if value is None:
        return "[dim]null[/dim]"
    # bool is a subclass of int — check before numeric.
    if isinstance(value, bool):
        return f"[magenta]{'true' if value else 'false'}[/magenta]"
    if isinstance(value, (int, float)):
        return f"[cyan]{escape(str(value))}[/cyan]"
    return f"[green]{escape(str(value))}[/green]"
