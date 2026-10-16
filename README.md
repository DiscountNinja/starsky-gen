# starsky-gen

Cinematic AAA skybox / starfield generator for games. Wrap-safe equirectangular and cubemap output is the product contract; photo-documentary look is an opt-in secondary path.
Low resolution sample:
<img width="1024" height="512" alt="starsky-gen" src="https://github.com/user-attachments/assets/5cc44295-65b0-45b1-85ab-c474089b589a" />
## Features

- Fully flag-driven Python CLI workflow (no interactive prompts).
- **Cinematic look by default**: filmic tone, richer bloom, near-zero film grain, flat/wrap-safe exposure.
- Realistic star and Milky Way color with magnitude/PSF stamping (sharp game stars, not stacked-photo noise).
- Non-uniform, subtle star coloration with weighted distributions:
    - White 65%, Blue 20%, Yellow 10%, Red 5%.
- Weighted star sizes:
    - Tiny (1px) 75%, Small (2-3px) 20%, Medium (4-6px) 4%, Large (7+) 1%.
- Asymmetric stars with cool-edge tinting for dominant white stars.
- Background galactic-plane gradient (no flat black void).
- Nebula modes: `distant`, `full`, `galaxy_streak`.
- Output projections: `equirectangular`, `cubemap`, or `both` (seam-aware / wrap-safe).
- Batch generation with per-render and total progress bars.
- Optional `--look-target documentary` for the parked photo/long-exposure grade.
- Optional JPEG artifact pass for subtle compression character.
- Default background avoids pure black, using a noisy bluish-gray field with galactic-disk weighting.

<img width="2048" height="1024" alt="starsky-gen-norm" src="https://github.com/user-attachments/assets/9057f85f-0c4f-4391-ae9a-d8cd625276fd" />

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
starsky-gen
```

Example with explicit options:

```bash
starsky-gen \
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

Optional interactive tuning:

```bash
starsky-gen assist
```

## CLI Reference

```bash
starsky-gen [OPTIONS]
```

Main output and run controls:

- `--width` (int, default `2048`)
- `--height` (int, default `1024`)
- `--output-base-name` (str, default `starsky`)
- `--output-dir` (path, default `output`)
- `--generations` (int >= 1, default `1`)
- `--seed` (int, optional)
- `--projection-mode` (`equirectangular|cubemap|both`, default `equirectangular`)
- `--cube` / `-cube` (shortcut to include cubemap, sets projection to `both`)
- `--output-format` (`png|jpg|webp`, default `png`)
- `--quality` (JPEG/WebP quality `50..100`, default `100`; ignored for PNG)
- `--cubemap-face-size` (`128..4096`, default `1024`)

Feature toggles:

- `--stars/--no-stars`
- `--depth/--no-depth`
- `--nebula/--no-nebula`
- `--galaxy-view/--no-galaxy-view`
- `--background-gradient/--no-background-gradient`
- `--black-background/--no-black-background`
- `--jpeg-artifact-pass/--no-jpeg-artifact-pass`
- `--look-target` (`cinematic|documentary`, default `cinematic`)
- `--long-exposure-look/--flat-exposure` (default flat / wrap-safe; documentary enables long-exposure)

Background texture:

- `--background-texture-strength` (`0.0..2.0`, default `1.0`)

Nebula controls:

- `--nebula-mode` (`distant|full|galaxy_streak`, default `galaxy_streak`)
- `--nebula-style` (`subtle|balanced|dramatic`, default `balanced`)
- `--cloud-continuity` (`0.6..1.6`, default `1.22`)
- `--dust-coverage` (`0.5..1.6`, default `0.98`)
- `--dust-strength` (`0.5..1.8`, default `1.14`)
- `--nebula-debug-pass` (`normal|occluder_only|continuum_only`, default `normal`; galaxy_streak also supports noise-layer diagnostics such as `layer_filaments`, `layer_dust_alpha`, `mask_only`, `warp_vectors`)

Look / galaxy tuning:

- `--look-target cinematic` — AAA wrap-safe vista (filmic, bloom up, grain down; default)
- `--look-target documentary` — parked photo/long-exposure stack (asinh, grain, sensor ISP)
- `--photoreal-stars` / `--classic-stars` (default magnitude+Moffat PSF stars)
- `--mag-bright`, `--mag-faint`, `--band-star-density`
- `--mag-exponent` (magnitude log slope)
- `--asinh-gain`, `--asinh-q` (linear canvas stretch; mainly documentary / asinh curve)
- `--galaxy-tone-curve` (`reinhard|asinh|filmic|aces|acescct`, default `filmic`)
- `--bulge-scale`, `--bulge-intensity`, `--dust-opacity`, `--nebula-color-strength`
- `--emit-halpha-saturation`, `--emit-patch-strength`
- `--split-toning-strength`, `--faint-star-chroma-desat`
- `--disk-star-density-dropout`, `--star-band-chroma-desat`
- `--star-band-brightness-scale`, `--star-band-chroma-adapt`
- `--nebula-emit-scale-with-stars`
- `--star-display-stretch-gain` (`0` = auto white-point match)
- `--no-sensor-noise` (sensor noise off by default under cinematic)

Pipeline / morphology diagnostics:

- `--render-profile` (`physical|physical_grade|full`, default `full`)
- `--debug-layers` (write morphology, extinction, and star-layer PNGs under `<output-dir>/<name>_layers/`, including RGB `structure_type_map.png` — see [docs/usage.md](docs/usage.md#structure-architecture-weight--type-system))
- `--grayscale-morphology` (add `morphology_gray.png` to the layer export)
- `--morphology-void-strength`, `--morphology-scar-strength`

Debug layer export example:

```bash
starsky-gen --seed 42 --debug-layers --output-dir output
# or: python -m starsky_gen.cli --seed 42 --debug-layers -o output
```

Layer folder includes `density_G.png`, `dust_D.png`, `unresolved_U.png`, `resolve_W.png`, `extinction.png`, `obliteration_mask.png`, `brutal_erasure_mask.png`, `structure_survival.png`, `gold_population_weight.png`, and related maps.

Advanced morphology tuning (extinction void floor, off-band red strength, ISM dominance, obliteration, etc.) lives in `FeatureConfig` (`src/starsky_gen/config.py`) and is not exposed as CLI flags yet.

### `assist` command

```bash
starsky-gen assist [OPTIONS]
```

Round/candidate controls:

- `--rounds` (`1..100`, default `5`)
- `--candidates-per-round` (`2..64`, default `8`)
- `--log-file` (path, default `output/assist_log.jsonl`)

Shared rendering controls (subset of main options):

- `--width`, `--height`, `--output-base-name`, `--output-dir`
- `--seed`, `--output-format`, `--nebula-mode`, `--quality`
- `--stars/--no-stars`, `--depth/--no-depth`, `--nebula/--no-nebula`
- `--galaxy-view/--no-galaxy-view`
- `--background-gradient/--no-background-gradient`
- `--black-background/--no-black-background`
- `--long-exposure-look/--flat-exposure`
- `--background-texture-strength`
- `--nebula-style`, `--cloud-continuity`, `--dust-coverage`, `--dust-strength`

Default behavior summary:

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
