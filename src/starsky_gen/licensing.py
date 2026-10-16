"""Local, gitignored copyright / distribution store and gentle pre-generate prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from rich.console import Console

CopyrightStatus = Literal["copyrighted", "open", "unknown"]

LICENSE_FILENAME = ".license"
PROMPT_EVERY_N = 20
DEFAULT_STATUS: CopyrightStatus = "unknown"
VALID_STATUSES: frozenset[str] = frozenset({"copyrighted", "open", "unknown"})


@dataclass
class LicenseState:
    copyright_status: CopyrightStatus = DEFAULT_STATUS
    copyright_holder: str | None = None
    copyright_year: int | None = None
    distribution_rights: str | None = None
    run_count: int = 0
    last_prompt_at: int | None = None
    seed: int | None = None
    look_target: str | None = None

    def resolved_year(self) -> int:
        if self.copyright_year is not None:
            try:
                return int(self.copyright_year)
            except (TypeError, ValueError):
                pass
        return int(datetime.now().year)

    def normalized_copyright(self) -> str:
        """Human-readable notice embedded in PNG/JPEG Copyright metadata."""
        status = _normalize_status(self.copyright_status)
        year = self.resolved_year()
        if status == "open":
            return "open"
        if status == "unknown":
            return "unknown"
        holder = (self.copyright_holder or "").strip()
        if holder:
            return f"© {year} {holder}"
        return f"© {year}"

    def normalized_distribution(self) -> str:
        """Distribution / license notice for PNG License metadata."""
        status = _normalize_status(self.copyright_status)
        if status in ("open", "unknown"):
            return status
        rights = (self.distribution_rights or "").strip()
        return rights if rights else "unknown"


def license_path(cwd: Path | None = None) -> Path:
    root = Path.cwd() if cwd is None else Path(cwd)
    return root / LICENSE_FILENAME


def default_license_state() -> LicenseState:
    return LicenseState(
        copyright_year=int(datetime.now().year),
        distribution_rights="unknown",
    )


def current_year() -> int:
    return int(datetime.now().year)


def _normalize_status(value: object) -> CopyrightStatus:
    if not isinstance(value, str):
        return DEFAULT_STATUS
    key = value.strip().lower()
    if key in ("c", "copyright", "copyrighted", "©"):
        return "copyrighted"
    if key in ("o", "open", "public", "public domain", "pd"):
        return "open"
    if key in ("u", "unknown", ""):
        return "unknown"
    if key in VALID_STATUSES:
        return key  # type: ignore[return-value]
    return DEFAULT_STATUS


def _parse_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _parse_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        text = str(value).strip()
    else:
        text = value.strip()
    if not text:
        return None
    # Drop control-only leftovers (e.g. accidental Esc from an old prompt).
    if not any(ch.isprintable() and not ch.isspace() for ch in text):
        return None
    return text


def _mirror_rights_for_status(status: CopyrightStatus) -> str:
    if status == "open":
        return "open"
    if status == "unknown":
        return "unknown"
    return "unknown"


def _state_from_mapping(raw: dict[str, Any]) -> LicenseState:
    status = _normalize_status(raw.get("copyright_status", raw.get("status", DEFAULT_STATUS)))

    # Legacy JSON used a free-form "copyright" string.
    legacy = raw.get("copyright")
    holder = _parse_optional_str(raw.get("copyright_holder"))
    if holder is None and isinstance(legacy, str) and legacy.strip():
        legacy_text = _parse_optional_str(legacy)
        if legacy_text is None:
            status = "unknown"
        elif legacy_text.lower() in ("unknown", "u"):
            status = "unknown"
        elif legacy_text.lower() in ("open", "o"):
            status = "open"
        else:
            status = "copyrighted"
            # Strip leading © / (c) / year if present so holder stays clean.
            cleaned = re.sub(r"^(?:©|\(c\))\s*", "", legacy_text, flags=re.IGNORECASE)
            cleaned = re.sub(r"^\d{4}\s+", "", cleaned).strip()
            holder = cleaned or legacy_text

    run_count = _parse_optional_int(raw.get("run_count", raw.get("generation_count", 0))) or 0
    run_count = max(0, run_count)
    year = _parse_optional_int(raw.get("copyright_year"))
    if year is None:
        year = current_year()
    last_prompt_at = _parse_optional_int(raw.get("last_prompt_at"))
    seed = _parse_optional_int(raw.get("seed"))
    look_target = _parse_optional_str(raw.get("look_target"))

    rights = _parse_optional_str(raw.get("distribution_rights"))
    if rights is None:
        # Legacy files: open/unknown mirror; copyrighted defaults to unknown.
        rights = _mirror_rights_for_status(status) if status in ("open", "unknown") else "unknown"
    if status in ("open", "unknown"):
        rights = status
        holder = None

    return LicenseState(
        copyright_status=status,
        copyright_holder=holder if status == "copyrighted" else None,
        copyright_year=year,
        distribution_rights=rights,
        run_count=run_count,
        last_prompt_at=last_prompt_at,
        seed=seed,
        look_target=look_target,
    )


def load_license(path: Path | None = None) -> LicenseState:
    p = license_path() if path is None else Path(path)
    if not p.is_file():
        return default_license_state()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return default_license_state()

    raw: Any = None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        raw = None
    if raw is None:
        # Migrate legacy JSON .license if present.
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return default_license_state()
    if not isinstance(raw, dict):
        return default_license_state()
    return _state_from_mapping(raw)


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    # Quote when needed for safe round-trip.
    if text == "" or any(ch in text for ch in ":#{}[]&*!|>'\"%@`,\n\t") or text.strip() != text:
        return json.dumps(text)
    return text


def save_license(state: LicenseState, path: Path | None = None) -> None:
    p = license_path() if path is None else Path(path)
    status = _normalize_status(state.copyright_status)
    year = state.resolved_year()
    holder = (state.copyright_holder or "").strip() if status == "copyrighted" else ""
    rights: str
    if status in ("open", "unknown"):
        rights = status
    else:
        rights = (state.distribution_rights or "").strip() or "unknown"
    payload = (
        "# Local copyright / distribution preferences (gitignored)\n"
        f"copyright_status: {status}  # copyrighted | open | unknown\n"
        f"copyright_holder: {_yaml_scalar(holder if holder else None)}  # optional; copyrighted only\n"
        f"copyright_year: {year}\n"
        f"distribution_rights: {_yaml_scalar(rights)}\n"
        f"run_count: {int(max(0, state.run_count))}\n"
        f"seed: {_yaml_scalar(state.seed)}\n"
        f"look_target: {_yaml_scalar(state.look_target)}\n"
        f"last_prompt_at: {_yaml_scalar(state.last_prompt_at)}\n"
    )
    p.write_text(payload, encoding="utf-8")


def should_prompt(state: LicenseState, *, every_n: int = PROMPT_EVERY_N) -> bool:
    """Unknown always re-prompts; copyrighted/open remind every N runs."""
    if _normalize_status(state.copyright_status) == "unknown":
        return True
    n = max(1, int(every_n))
    if state.last_prompt_at is None:
        return True
    return int(state.run_count) - int(state.last_prompt_at) >= n


def prompt_copyright_status(console: Console, *, current: LicenseState | None = None) -> CopyrightStatus:
    """Letter prompt: C/O/U (or Enter → unknown). Loops until a letter is recognized."""
    cur = current or default_license_state()
    extras: list[str] = []
    if cur.copyright_holder:
        extras.append(f"holder={cur.copyright_holder!r}")
    if cur.distribution_rights and cur.copyright_status == "copyrighted":
        extras.append(f"license={cur.distribution_rights!r}")
    extra = (", " + ", ".join(extras)) if extras else ""
    console.print(
        "[dim]Copyright status for generated images[/dim] "
        f"[dim](current:[/dim] {cur.copyright_status}{extra}"
        f", year={cur.resolved_year()}[dim])[/dim]"
    )
    console.print(
        "[dim]Choose[/dim] [bold]C[/bold]opyrighted / [bold]O[/bold]pen / "
        "[bold]U[/bold]nknown "
        "[dim](Enter = Unknown; unknown will ask again next generate)[/dim]"
    )
    while True:
        try:
            answer = input("(C/O/U) ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("")
            return "unknown"
        if not answer:
            return "unknown"
        status = _normalize_status(answer)
        # Reject free-form junk that collapses to unknown unless they typed U/unknown.
        key = answer.strip().lower()
        if status != "unknown" or key in ("u", "unknown"):
            return status
        console.print("[dim]Please enter C, O, U, or press Enter for Unknown.[/dim]")


def prompt_distribution_rights(console: Console, *, current: str | None = None) -> str:
    """License / distribution rights text. Enter keeps current, else unknown if none."""
    cur = (current or "").strip()
    hint = f" [dim](current: {cur}; Enter keeps it)[/dim]" if cur else ""
    console.print(
        f"[dim]Distribution rights / license{hint}.[/dim] "
        "[dim]Examples: All Rights Reserved, CC0, CC BY 4.0. Enter for unknown if blank:[/dim]"
    )
    try:
        answer = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("")
        return cur if cur else "unknown"
    if answer:
        return answer
    return cur if cur else "unknown"


def prompt_copyright_holder(console: Console, *, current: str | None = None) -> str | None:
    """Optional holder name; Enter leaves blank."""
    hint = f" [dim](current: {current})[/dim]" if (current or "").strip() else ""
    console.print(f"[dim]Optional copyright holder name{hint}. Enter to leave blank:[/dim]")
    try:
        answer = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("")
        return (current or "").strip() or None
    return answer if answer else None


def apply_status_choice(
    state: LicenseState,
    status: CopyrightStatus,
    *,
    holder: str | None = None,
    distribution_rights: str | None = None,
    year: int | None = None,
) -> LicenseState:
    """Apply C/O/U choice. Unknown does not advance last_prompt_at (will re-prompt)."""
    status = _normalize_status(status)
    year_v = int(year) if year is not None else current_year()
    if status == "unknown":
        return LicenseState(
            copyright_status="unknown",
            copyright_holder=None,
            copyright_year=year_v,
            distribution_rights="unknown",
            run_count=int(state.run_count),
            last_prompt_at=None,
            seed=state.seed,
            look_target=state.look_target,
        )
    if status == "open":
        return LicenseState(
            copyright_status="open",
            copyright_holder=None,
            copyright_year=year_v,
            distribution_rights="open",
            run_count=int(state.run_count),
            last_prompt_at=int(state.run_count),
            seed=state.seed,
            look_target=state.look_target,
        )
    # copyrighted — holder=None/blank means no name (© YEAR only).
    cleaned = (holder or "").strip() or None
    rights = (distribution_rights or "").strip() or "unknown"
    return LicenseState(
        copyright_status="copyrighted",
        copyright_holder=cleaned,
        copyright_year=year_v,
        distribution_rights=rights,
        run_count=int(state.run_count),
        last_prompt_at=int(state.run_count),
        seed=state.seed,
        look_target=state.look_target,
    )


def maybe_prompt_before_generate(
    console: Console,
    *,
    path: Path | None = None,
    every_n: int = PROMPT_EVERY_N,
) -> LicenseState:
    """Load `.license`, optionally prompt, persist, and return the active state.

    Flow: C/O/U first. Open/unknown mirrors into distribution_rights (no further prompts).
    Copyrighted asks license/distribution rights, then optional holder name.
    """
    p = license_path() if path is None else Path(path)
    state = load_license(p)
    if not should_prompt(state, every_n=every_n):
        return state

    status = prompt_copyright_status(console, current=state)
    holder: str | None = None
    rights: str | None = None
    if status == "copyrighted":
        rights = prompt_distribution_rights(console, current=state.distribution_rights)
        holder = prompt_copyright_holder(console, current=state.copyright_holder)

    state = apply_status_choice(
        state,
        status,
        holder=holder,
        distribution_rights=rights,
        year=current_year(),
    )
    save_license(state, p)

    if status == "unknown":
        console.print(
            "[dim]Copyright / license: unknown (will ask again next generate).[/dim]"
        )
    elif status == "open":
        console.print("[dim]Copyright / license: open[/dim]")
    else:
        console.print(
            f"[dim]Copyright:[/dim] {state.normalized_copyright()}  "
            f"[dim]License:[/dim] {state.normalized_distribution()}"
        )
    return state


def record_generations(
    state: LicenseState,
    n: int,
    *,
    path: Path | None = None,
    seed: int | None = None,
    look_target: str | None = None,
) -> LicenseState:
    """Bump run_count after successful renders; record last seed / look_target."""
    p = license_path() if path is None else Path(path)
    current = load_license(p)

    # Prefer the in-session status when the file is still default-unknown.
    if current.copyright_status == "unknown" and state.copyright_status != "unknown":
        base = state
    elif current.copyright_status != "unknown":
        base = current
    else:
        base = state if state.copyright_status != "unknown" else current

    status = _normalize_status(base.copyright_status)
    rights: str
    holder: str | None
    if status in ("open", "unknown"):
        rights = status
        holder = None
    else:
        rights = (base.distribution_rights or "").strip() or "unknown"
        holder = base.copyright_holder

    updated = LicenseState(
        copyright_status=status,
        copyright_holder=holder,
        copyright_year=base.copyright_year if base.copyright_year is not None else current_year(),
        distribution_rights=rights,
        run_count=int(current.run_count) + max(0, int(n)),
        last_prompt_at=current.last_prompt_at
        if current.last_prompt_at is not None
        else state.last_prompt_at,
        seed=seed if seed is not None else (state.seed if state.seed is not None else current.seed),
        look_target=(
            look_target
            if look_target is not None
            else (state.look_target if state.look_target is not None else current.look_target)
        ),
    )
    save_license(updated, p)
    return updated


def copyright_metadata_dict(state: LicenseState | str | None) -> dict[str, str]:
    if isinstance(state, LicenseState):
        return {
            "Copyright": state.normalized_copyright(),
            "License": state.normalized_distribution(),
        }
    if isinstance(state, str) and state.strip():
        notice = state.strip()
        return {"Copyright": notice, "License": notice}
    return {"Copyright": "unknown", "License": "unknown"}


# Back-compat aliases used by older tests / imports.
DEFAULT_COPYRIGHT = "unknown"


def apply_prompt_result(state: LicenseState, notice: str | None) -> LicenseState:
    """Legacy helper: free-form notice string → status choice."""
    if notice is None or not str(notice).strip():
        return apply_status_choice(state, "unknown")
    text = str(notice).strip()
    low = text.lower()
    if low in ("unknown", "u"):
        return apply_status_choice(state, "unknown")
    if low in ("open", "o"):
        return apply_status_choice(state, "open")
    return apply_status_choice(
        state,
        "copyrighted",
        holder=text,
        distribution_rights="unknown",
    )
