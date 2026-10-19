"""CLI default argv parsing, apply overlay, and show helpers."""

from __future__ import annotations

from pathlib import Path

from starsky_gen.cli_defaults import (
    apply_saved_defaults,
    format_defaults_value,
    format_defaults_value_markup,
    guided_overlay_from_saved,
    project_defaults_for_display,
    provided_params,
    serialize_value,
    updates_from_kwargs,
)
from starsky_gen.config import SceneMode
from starsky_gen.guided import STOCK_DEFAULTS


def test_provided_params_from_argv() -> None:
    assert provided_params([]) == set()
    assert provided_params(["--default", "--mode", "nebula"]) == {"scene_mode"}
    assert provided_params(["--no-stars", "--black-background"]) == {
        "stars",
        "black_background",
    }
    assert provided_params(["generate", "--seed", "42", "--guided"]) == {"seed"}
    assert provided_params(["--show-defaults"]) == set()
    assert provided_params(["--noheroes", "--nebula-style", "dramatic"]) == {
        "noheroes",
        "nebula_style",
    }


def test_updates_from_kwargs_only_present() -> None:
    kwargs = {
        "scene_mode": SceneMode.nebula,
        "nebula_style": "dramatic",
        "width": 4096,
        "stars": False,
    }
    updates = updates_from_kwargs(kwargs, {"scene_mode", "nebula_style", "stars"})
    assert updates == {
        "mode": "nebula",
        "nebula-style": "dramatic",
        "stars": False,
    }
    assert "width" not in updates


def test_apply_saved_defaults_respects_argv() -> None:
    kwargs = {
        "scene_mode": SceneMode.normal,
        "nebula_style": "balanced",
        "black_background": False,
        "noheroes": False,
        "width": 2048,
    }
    saved = {
        "mode": "nebula",
        "nebula-style": "dramatic",
        "black-background": True,
        "width": 4096,
    }
    out = apply_saved_defaults(
        kwargs,
        saved,
        provided={"scene_mode"},  # CLI wins for mode
    )
    assert out["scene_mode"] == SceneMode.normal
    assert out["nebula_style"] == "dramatic"
    assert out["black_background"] is True
    assert out["width"] == 4096


def test_apply_heroes_alias() -> None:
    kwargs = {"noheroes": False, "width": 2048}
    out = apply_saved_defaults(kwargs, {"heroes": False}, provided=set())
    assert out["noheroes"] is True
    out2 = apply_saved_defaults(kwargs, {"noheroes": True}, provided=set())
    assert out2["noheroes"] is True


def test_project_defaults_include_stock() -> None:
    project = project_defaults_for_display()
    assert project["mode"] == serialize_value(STOCK_DEFAULTS["scene_mode"])
    assert project["nebula-style"] == "balanced"
    assert project["noheroes"] is False
    assert project["nebula-hero-color"] == "magenta"
    assert format_defaults_value(None) == "null"
    assert format_defaults_value(True) == "true"
    assert format_defaults_value_markup(None) == "[dim]null[/dim]"
    assert format_defaults_value_markup(True) == "[magenta]true[/magenta]"
    assert format_defaults_value_markup(False) == "[magenta]false[/magenta]"
    assert format_defaults_value_markup(2048) == "[cyan]2048[/cyan]"
    assert format_defaults_value_markup(1.22) == "[cyan]1.22[/cyan]"
    assert format_defaults_value_markup("nebula") == "[green]nebula[/green]"


def test_cli_default_save_show_via_typer(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from starsky_gen.cli import app
    from starsky_gen.licensing import load_cli_defaults, load_license

    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["--default", "--mode", "nebula", "--nebula-style", "dramatic", "--black-background"],
    )
    assert result.exit_code == 0, result.output
    assert "Saved defaults" in result.output
    assert "mode: nebula" in result.output
    defaults = load_cli_defaults()
    assert defaults["mode"] == "nebula"
    assert defaults["nebula-style"] == "dramatic"
    assert defaults["black-background"] is True
    # Did not leave unknown legal state unreadable
    assert load_license().cli_defaults["mode"] == "nebula"

    show = runner.invoke(app, ["--show-defaults"])
    assert show.exit_code == 0, show.output
    assert "Custom defaults" in show.output
    assert "mode: nebula" in show.output
    assert "Project defaults" in show.output
    assert "nebula-style: balanced" in show.output  # project stock

    # --default alone does not render; no output images
    assert not list(tmp_path.glob("output/**/*.png"))

    both = runner.invoke(
        app,
        ["--default", "--nebula-hero-color", "gold", "--show-defaults"],
    )
    assert both.exit_code == 0, both.output
    assert "nebula-hero-color: gold" in both.output
    assert "Custom defaults" in both.output


def test_guided_overlay_from_saved() -> None:
    overlay = guided_overlay_from_saved(
        {"mode": "nebula", "noheroes": True, "nebula-style": "dramatic"}
    )
    assert overlay["scene_mode"] == SceneMode.nebula
    assert overlay["heroes"] is False
    assert overlay["nebula_style"] == "dramatic"
