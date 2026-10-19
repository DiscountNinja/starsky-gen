"""Guided CLI parsers, presets, advanced skip, and equivalent command."""

from __future__ import annotations

import pytest
from rich.console import Console

from starsky_gen import guided as guided_mod
from starsky_gen.config import LookTarget, NebulaMode, OutputFormat, ProjectionMode
from starsky_gen.config import SceneMode
from starsky_gen.guided import (
    STOCK_DEFAULTS,
    GuidedCancelled,
    format_equivalent_command,
    format_guided_summary,
    parse_bool,
    parse_float,
    parse_int,
    prompt_output_type,
    prompt_size_preset,
    run_guided_prompts,
    should_prompt,
    should_run_guided,
)


def test_parse_bool() -> None:
    assert parse_bool("y") is True
    assert parse_bool("NO") is False
    assert parse_bool("1") is True
    assert parse_bool("maybe") is None


def test_should_run_guided_from_argv() -> None:
    assert should_run_guided([]) is True
    assert should_run_guided(["generate"]) is True
    assert should_run_guided(["--width", "4096"]) is False
    assert should_run_guided(["generate", "--seed", "1"]) is False
    assert should_run_guided(["--guided"]) is True
    assert should_run_guided(["--guided", "--width", "4096"]) is True
    # Any other flag skips guided (including legacy --no-guided if still passed as unknown argv).
    assert should_run_guided(["--seed", "42"]) is False


def test_prompt_str_enter_keeps_default(monkeypatch: pytest.MonkeyPatch, console: Console) -> None:
    from starsky_gen.guided import GuidedField, prompt_field

    spec = GuidedField(
        "output_base_name",
        "--output-base-name",
        "Output base name",
        "Base filename prefix.",
        "str",
        "starsky",
    )
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "")
    assert prompt_field(console, spec, "starsky") == "starsky"
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "   ")
    assert prompt_field(console, spec, "starsky") == "starsky"


def test_parse_int_and_float_ranges() -> None:
    assert parse_int("42") == 42
    assert parse_int("3", min_v=5) is None
    assert parse_float("1.5", min_v=0.0, max_v=2.0) == 1.5
    assert parse_float("9", max_v=2.0) is None


def test_format_equivalent_command_omits_defaults() -> None:
    values = dict(STOCK_DEFAULTS)
    assert format_equivalent_command(values) == "starsky-gen"

    values["width"] = 4096
    values["height"] = 2048
    values["cube"] = True
    values["heroes"] = False
    values["seed"] = 77
    values["look_target"] = LookTarget.documentary
    cmd = format_equivalent_command(values)
    assert cmd.startswith("starsky-gen")
    assert " generate" not in cmd
    assert "--width 4096" in cmd
    assert "--height 2048" in cmd
    assert "--cube" in cmd
    assert "--noheroes" in cmd
    assert "--heroes" not in cmd
    assert "--seed 77" in cmd
    assert "--look-target documentary" in cmd
    assert "--output-format" not in cmd  # still default png


def test_format_equivalent_command_neg_flags() -> None:
    values = dict(STOCK_DEFAULTS)
    values["stars"] = False
    cmd = format_equivalent_command(values)
    assert "--no-stars" in cmd


def test_format_guided_summary_readable() -> None:
    values = dict(STOCK_DEFAULTS)
    values["width"] = 8192
    values["height"] = 4096
    values["projection_mode"] = ProjectionMode.both
    values["seed"] = 42
    values["heroes"] = False
    values["look_target"] = LookTarget.documentary
    values["nebula"] = False
    values["black_background"] = True
    text = "\n".join(format_guided_summary(values))
    assert "Intent:" in text
    assert "8192×4096" in text
    assert "both" in text
    assert "seed 42" in text
    assert "documentary" in text
    assert "heroes=off" in text
    assert "no nebula gas" in text
    assert "black void" in text or "forced black" in text


def test_should_prompt_skips_dead_followups() -> None:
    answers = dict(STOCK_DEFAULTS)
    answers["scene_mode"] = SceneMode.nebula
    gv = next(f for f in guided_mod.GUIDED_FIELDS if f.key == "galaxy_view")
    assert should_prompt(gv, answers) is False
    answers["scene_mode"] = SceneMode.normal
    answers["nebula"] = False
    dust = next(f for f in guided_mod.GUIDED_FIELDS if f.key == "dust_coverage")
    assert should_prompt(dust, answers) is False
    answers["nebula"] = True
    answers["stars"] = False
    dens = next(f for f in guided_mod.GUIDED_FIELDS if f.key == "background_density")
    assert should_prompt(dens, answers) is False


@pytest.fixture
def console() -> Console:
    return Console(force_terminal=False, quiet=True)


def test_size_preset_4k_and_8k(monkeypatch: pytest.MonkeyPatch, console: Console) -> None:
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "2")
    assert prompt_size_preset(console, dict(STOCK_DEFAULTS)) == (4096, 2048)
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "3")
    assert prompt_size_preset(console, dict(STOCK_DEFAULTS)) == (8192, 4096)


def test_size_preset_enter_keeps_current(monkeypatch: pytest.MonkeyPatch, console: Console) -> None:
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "")
    assert prompt_size_preset(console, dict(STOCK_DEFAULTS)) == (2048, 1024)
    cur = dict(STOCK_DEFAULTS)
    cur["width"], cur["height"] = 4096, 2048
    assert prompt_size_preset(console, cur) == (4096, 2048)


def test_size_preset_custom_returns_none(monkeypatch: pytest.MonkeyPatch, console: Console) -> None:
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "4")
    assert prompt_size_preset(console, dict(STOCK_DEFAULTS)) is None


def test_size_preset_q_cancels(monkeypatch: pytest.MonkeyPatch, console: Console) -> None:
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "q")
    with pytest.raises(GuidedCancelled):
        prompt_size_preset(console, dict(STOCK_DEFAULTS))


def test_output_type_choices(monkeypatch: pytest.MonkeyPatch, console: Console) -> None:
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "2")
    mode, cube = prompt_output_type(console, dict(STOCK_DEFAULTS))
    assert mode == ProjectionMode.cubemap
    assert cube is False
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "3")
    mode, cube = prompt_output_type(console, dict(STOCK_DEFAULTS))
    assert mode == ProjectionMode.both
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "")
    mode, cube = prompt_output_type(console, dict(STOCK_DEFAULTS))
    assert mode == ProjectionMode.equirectangular


def test_run_guided_spam_enter_defaults(
    monkeypatch: pytest.MonkeyPatch, console: Console
) -> None:
    # All Enter: size keep, output type keep, all core keep, advanced no, …
    answers = [""] * 40
    idx = {"i": 0}

    def _reader(prompt: str = "> ") -> str:
        i = idx["i"]
        idx["i"] = i + 1
        return answers[i] if i < len(answers) else "q"

    monkeypatch.setattr(guided_mod, "_read_line", _reader)
    out = run_guided_prompts(console, dict(STOCK_DEFAULTS))
    assert out is not None
    assert out["width"] == 2048
    assert out["height"] == 1024
    assert out["projection_mode"] == ProjectionMode.equirectangular
    assert out["cube"] is False
    assert out["look_target"] == LookTarget.cinematic
    assert out["nebula_mode"] == NebulaMode.galaxy_streak
    assert out["output_format"] == OutputFormat.png
    assert out["jpeg_artifact_pass"] is False
    assert out["morphology_void_strength"] == 0.72


def test_run_guided_q_returns_none(monkeypatch: pytest.MonkeyPatch, console: Console) -> None:
    monkeypatch.setattr(guided_mod, "_read_line", lambda prompt="> ": "q")
    assert run_guided_prompts(console, dict(STOCK_DEFAULTS)) is None


def test_run_guided_8k_both_then_advanced(
    monkeypatch: pytest.MonkeyPatch, console: Console
) -> None:
    answers: list[str] = ["3", "3"]  # 8K, both
    core_skip = {"width", "height", "projection_mode", "cube"}
    core_rest = sum(
        1 for f in guided_mod.GUIDED_FIELDS if f.group == "core" and f.key not in core_skip
    )
    answers.extend([""] * core_rest)
    answers.append("y")
    # advanced: jpeg, render_profile, isolate, nebula_debug, debug_layers=y, grayscale, void, scar
    answers.extend(["", "", "", "", "y", "", "", ""])

    idx = {"i": 0}

    def _reader(prompt: str = "> ") -> str:
        i = idx["i"]
        idx["i"] = i + 1
        if i >= len(answers):
            return "q"
        return answers[i]

    monkeypatch.setattr(guided_mod, "_read_line", _reader)
    out = run_guided_prompts(console, dict(STOCK_DEFAULTS))
    assert out is not None
    assert out["width"] == 8192
    assert out["height"] == 4096
    assert out["projection_mode"] == ProjectionMode.both
    assert out["debug_export_layers"] is True
    cmd = format_equivalent_command(out)
    assert "--width 8192" in cmd
    assert "--projection-mode both" in cmd
    assert "--debug-layers" in cmd
