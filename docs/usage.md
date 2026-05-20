# Usage Guide

## Basic render

```bash
starsky-gen
```

Defaults: `--nebula-mode galaxy_streak`, `--output-format png`, `--quality 100`, `--projection-mode equirectangular`.

Background is a noisy bluish-gray field that darkens away from the galactic disk unless you pass `--black-background`.

Equivalent module invocation:

```bash
python -m starsky_gen.cli
```

## Common options

```bash
starsky-gen \
  --width 4096 \
  --height 2048 \
  --projection-mode both \
  --cube \
  --output-format png \
  --nebula-mode galaxy_streak \
  --generations 8 \
  --seed 77 \
  --output-dir output \
  --output-base-name skybox
```

Recommended sizes:

- Equirectangular skybox: `4096x2048`
- Cubemap faces: `1024` or `2048` (`--cubemap-face-size`)

Nebula modes:

- `distant` — isolated feature
- `full` — immersive fog
- `galaxy_streak` — Milky-Way-like plane (default)

## Photoreal and display tuning

Galaxy renders use magnitude-based background stars by default (`--photoreal-stars`). Key knobs:

- `--mag-bright`, `--mag-faint`, `--band-star-density`
- `--asinh-gain`, `--asinh-q`, `--galaxy-tone-curve`
- `--disk-star-density-dropout` — thin resolved stars in the bright plane
- `--star-band-chroma-desat`, `--star-band-brightness-scale` — plane star look
- `--render-profile physical` — linear physical composite only (no display finish)
- `--render-profile physical_grade` — physical + unified linear grade, no final display polish
- `--render-profile full` — full pipeline (default)

## Debug / morphology layers

Export intermediate maps for tuning extinction, resolve dropout, and structure:

```bash
starsky-gen --seed 42 --debug-layers -o output
starsky-gen --seed 42 --debug-layers --grayscale-morphology -o output
```

Writes `output/<output-base-name>_layers/` with PNGs such as `density_G.png`, `dust_D.png`, `unresolved_U.png`, `resolve_W.png`, `dropout_mask.png`, `extinction.png`, `obliteration_mask.png`, `brutal_erasure_mask.png`, `structure_survival.png`, `gold_population_weight.png`, and `resolved_stars.png`.

Nebula-only diagnostics:

```bash
starsky-gen --nebula-debug-pass occluder_only
starsky-gen --nebula-debug-pass layer_filaments   # galaxy_streak noise layers
```

## Config-only parameters

Many morphology and photometry fields (e.g. `extinction_void_floor`, `off_band_emission_strength`, `morphology_obliteration_strength`, `band_ism_dominance`) are defined on `FeatureConfig` in `config.py` and are adjusted in Python or by extending the CLI — they are not all exposed as flags yet. See `README.md` for the full CLI reference.

## Interactive search

```bash
starsky-gen assist
```

Rounds of scored candidates with a smaller flag subset; log defaults to `output/assist_log.jsonl`.
