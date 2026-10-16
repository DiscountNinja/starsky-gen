"""Local .license YAML store, C/O/U + license prompts, and PNG copyright metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from starsky_gen.generator import _save_image
from starsky_gen.licensing import (
    DEFAULT_COPYRIGHT,
    LicenseState,
    apply_prompt_result,
    apply_status_choice,
    copyright_metadata_dict,
    load_license,
    record_generations,
    save_license,
    should_prompt,
)


def test_default_license_is_unknown(tmp_path: Path) -> None:
    state = load_license(tmp_path / ".license")
    assert state.copyright_status == "unknown"
    assert state.distribution_rights == "unknown"
    assert state.run_count == 0
    assert state.last_prompt_at is None
    assert should_prompt(state) is True
    assert state.normalized_copyright() == DEFAULT_COPYRIGHT
    assert state.normalized_distribution() == "unknown"


def test_unknown_always_reprompts(tmp_path: Path) -> None:
    path = tmp_path / ".license"
    state = apply_status_choice(
        LicenseState(run_count=50, last_prompt_at=50),
        "unknown",
    )
    save_license(state, path)
    loaded = load_license(path)
    assert loaded.copyright_status == "unknown"
    assert loaded.distribution_rights == "unknown"
    assert loaded.last_prompt_at is None
    assert should_prompt(loaded) is True


def test_open_mirrors_distribution_rights(tmp_path: Path) -> None:
    path = tmp_path / ".license"
    state = apply_status_choice(LicenseState(run_count=0), "open")
    assert state.distribution_rights == "open"
    assert state.copyright_holder is None
    save_license(state, path)
    loaded = load_license(path)
    assert loaded.copyright_status == "open"
    assert loaded.distribution_rights == "open"
    assert loaded.normalized_copyright() == "open"
    assert loaded.normalized_distribution() == "open"
    assert should_prompt(loaded) is False

    loaded.run_count = 19
    assert should_prompt(loaded) is False
    loaded.run_count = 20
    assert should_prompt(loaded) is True


def test_copyrighted_license_then_holder() -> None:
    cstate = apply_status_choice(
        LicenseState(run_count=0),
        "copyrighted",
        holder="Ada",
        distribution_rights="All Rights Reserved",
        year=2026,
    )
    assert cstate.copyright_status == "copyrighted"
    assert cstate.copyright_holder == "Ada"
    assert cstate.distribution_rights == "All Rights Reserved"
    assert cstate.normalized_copyright() == "© 2026 Ada"
    assert cstate.normalized_distribution() == "All Rights Reserved"


def test_copyrighted_blank_holder_is_year_only() -> None:
    state = apply_status_choice(
        LicenseState(run_count=3),
        "copyrighted",
        holder=None,
        distribution_rights="CC BY 4.0",
        year=2026,
    )
    assert state.normalized_copyright() == "© 2026"
    assert state.normalized_distribution() == "CC BY 4.0"


def test_yaml_roundtrip_and_legacy_json(tmp_path: Path) -> None:
    path = tmp_path / ".license"
    state = LicenseState(
        copyright_status="copyrighted",
        copyright_holder="Studio",
        copyright_year=2026,
        distribution_rights="All Rights Reserved",
        run_count=3,
        last_prompt_at=0,
        seed=42,
        look_target="cinematic",
    )
    save_license(state, path)
    text = path.read_text(encoding="utf-8")
    assert "copyright_status: copyrighted" in text
    assert "distribution_rights: All Rights Reserved" in text
    assert "run_count: 3" in text
    assert "seed: 42" in text
    assert "look_target: cinematic" in text
    loaded = load_license(path)
    assert loaded.copyright_status == "copyrighted"
    assert loaded.copyright_holder == "Studio"
    assert loaded.distribution_rights == "All Rights Reserved"
    assert loaded.seed == 42
    assert loaded.look_target == "cinematic"

    legacy = tmp_path / "legacy.license"
    legacy.write_text(
        '{\n  "copyright": "Acme Labs",\n  "generation_count": 7,\n  "last_prompt_at": 0\n}\n',
        encoding="utf-8",
    )
    migrated = load_license(legacy)
    assert migrated.copyright_status == "copyrighted"
    assert migrated.copyright_holder == "Acme Labs"
    assert migrated.distribution_rights == "unknown"
    assert migrated.run_count == 7


def test_legacy_open_status_mirrors_rights(tmp_path: Path) -> None:
    path = tmp_path / ".license"
    path.write_text(
        "copyright_status: open\ncopyright_holder: null\ncopyright_year: 2026\n"
        "run_count: 1\nlast_prompt_at: 0\n",
        encoding="utf-8",
    )
    loaded = load_license(path)
    assert loaded.copyright_status == "open"
    assert loaded.distribution_rights == "open"


def test_legacy_apply_prompt_result(tmp_path: Path) -> None:
    state = LicenseState(run_count=5, last_prompt_at=None)
    skipped = apply_prompt_result(state, None)
    assert skipped.copyright_status == "unknown"
    assert skipped.distribution_rights == "unknown"
    assert skipped.last_prompt_at is None
    assert should_prompt(skipped) is True

    set_notice = apply_prompt_result(state, "Example")
    assert set_notice.copyright_status == "copyrighted"
    assert set_notice.copyright_holder == "Example"
    assert set_notice.distribution_rights == "unknown"
    assert set_notice.last_prompt_at == 5


def test_record_generations_bumps_count_and_last_gen(tmp_path: Path) -> None:
    path = tmp_path / ".license"
    state = LicenseState(
        copyright_status="copyrighted",
        copyright_holder="Acme",
        copyright_year=2026,
        distribution_rights="All Rights Reserved",
        run_count=2,
        last_prompt_at=0,
    )
    save_license(state, path)
    updated = record_generations(state, 3, path=path, seed=42, look_target="cinematic")
    assert updated.run_count == 5
    assert updated.copyright_holder == "Acme"
    assert updated.distribution_rights == "All Rights Reserved"
    assert updated.seed == 42
    assert updated.look_target == "cinematic"
    assert load_license(path).run_count == 5


def test_copyright_metadata_dict() -> None:
    assert copyright_metadata_dict(None) == {
        "Copyright": "unknown",
        "License": "unknown",
    }
    open_meta = copyright_metadata_dict(LicenseState(copyright_status="open"))
    assert open_meta["Copyright"] == "open"
    assert open_meta["License"] == "open"
    meta = copyright_metadata_dict(
        LicenseState(
            copyright_status="copyrighted",
            copyright_holder="Me",
            copyright_year=2026,
            distribution_rights="CC0",
        )
    )
    assert meta["Copyright"] == "© 2026 Me"
    assert meta["License"] == "CC0"


def test_png_save_embeds_distinct_copyright_and_license(tmp_path: Path) -> None:
    path = tmp_path / "out.png"
    rgb = np.zeros((8, 16, 3), dtype=np.float64)
    _save_image(
        rgb,
        path,
        "png",
        100,
        copyright="© 2026 Test Studio",
        license="All Rights Reserved",
    )
    with Image.open(path) as im:
        assert im.text.get("Copyright") == "© 2026 Test Studio"
        assert im.text.get("License") == "All Rights Reserved"


def test_png_save_defaults_unknown_copyright(tmp_path: Path) -> None:
    path = tmp_path / "out.png"
    rgb = np.zeros((4, 8, 3), dtype=np.float64)
    _save_image(rgb, path, "png", 100, copyright=None, license=None)
    with Image.open(path) as im:
        assert im.text.get("Copyright") == "unknown"
        assert im.text.get("License") == "unknown"


def test_webp_save_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out.webp"
    rgb = np.zeros((8, 16, 3), dtype=np.float64)
    rgb[:, :, 0] = 0.4
    _save_image(rgb, path, "webp", 90, copyright="© 2026 Test", license="CC0")
    assert path.is_file()
    with Image.open(path) as im:
        assert im.format == "WEBP"
        assert im.size == (16, 8)
