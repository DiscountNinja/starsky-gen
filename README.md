# starsky-gen

Photo Realistic starfield generator for games, skyboxes, and background assets.

## Features

- Fully flag-driven Python CLI workflow (no interactive prompts).
- Non-uniform, subtle star coloration with weighted distributions:
  - White 65%, Blue 20%, Yellow 10%, Red 5%.
- Weighted star sizes:
  - Tiny (1px) 75%, Small (2-3px) 20%, Medium (4-6px) 4%, Large (7+) 1%.
- Asymmetric stars with cool-edge tinting for dominant white stars.
- Background galactic-plane gradient (no flat black void).
- Nebula modes: `distant`, `full`, `galaxy_streak`.
- Output projections: `equirectangular`, `cubemap`, or `both`.
- Batch generation with per-render and total progress bars.
- Optional JPEG artifact pass for subtle compression character.
- Default background avoids pure black, using a noisy bluish-gray field with galactic-disk weighting.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
starsky-gen generate
```

Example with explicit options:

```bash
starsky-gen generate \
  --projection-mode both \
  --output-format png \
  --nebula-mode galaxy_streak \
  --width 4096 \
  --height 2048 \
  --cubemap-face-size 1024 \
  --output-base-name skybox \
  --output-dir output \
  --generations 4 \
  --seed 42
```

Default behavior:

- `--projection-mode equirectangular` (use `--cube` / `-cube` to include cubemap too)
- `--nebula-mode galaxy_streak`
- `--output-format png`
- `--quality 100` (applies to JPG output)
- `--black-background false` (use `--black-background` only when you explicitly want black)

## Output

- Equirectangular files: `<name>_####_equirect.png|jpg`
- Cubemap files: `<name>_####_cube_{px|nx|py|ny|pz|nz}.png|jpg`

## Testing

```bash
pytest
```

## Open source notes

- Code is structured under `src/starsky_gen`.
- Tests live in `tests`.
- Additional docs are in `docs`.
- Licensed under the MIT License. See 'LICENSE'.
