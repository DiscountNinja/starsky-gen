# Usage Guide

## Basic render

```bash
starsky-gen
```

Defaults: `--mode normal`, `--look-target cinematic`, `--nebula-mode galaxy_streak`, `--output-format png`, `--quality 100`, `--projection-mode equirectangular`.

Background is a noisy bluish-gray field that darkens away from the galactic disk unless you pass `--black-background`. Cinematic look keeps grain near zero and bloom/tone elevated for wrap-safe game vistas.

### Scene mode (`--mode`)

- `normal` — Milky Way band skybox (galactic midplane + `nebula-mode` distant/full/galaxy_streak)
- `nebula` — broad cloud-complex vista with **no galactic band**

Under `--mode nebula`, guided skips the Galaxy view prompt (band is already off). Grade/photoreal still use an internal `galaxy_view` path. Layer toggles (`--no-stars`, `--no-nebula`, `--noheroes`, `--background-density 0`) are honored. Use `--nebula-style dramatic` to densify dust and push multi-hue emission on the cloud-complex path.

```bash
starsky-gen --mode normal --seed 1
starsky-gen --mode nebula --seed 1
starsky-gen --mode nebula --nebula-mode full --nebula-style dramatic --seed 42
```

### Persist CLI defaults (`--default` / `--show-defaults`)

Save preferred flags into the local `.license` file (gitignored) under a `defaults` section. Saving exits without rendering:

```bash
starsky-gen --default --mode nebula --nebula-style dramatic --nebula-hero-color gold
starsky-gen --show-defaults   # custom (.license) vs project built-in assumptions
```

Later runs (and guided prompts) use those custom defaults unless you override the same flag on the CLI. Combine `--default` with `--show-defaults` to save and immediately verify.

### Guided / discovery mode (default when no flags)

```bash
starsky-gen                 # guided (no other flags)
starsky-gen --width 4096    # not guided: flag → args + defaults for the rest
starsky-gen --guided --seed 7   # force guided; seed pre-fills the prompt default
```

Walks options interactively when you pass **no** CLI flags (newbie-friendly / discovery). If you pass any flag, guided is skipped and unset options keep their normal (or saved `.license`) defaults. Pass `--guided` to fine-tune interactively even when other flags are set. **Enter** keeps the current/default at every guided step — spam Enter for a full default run.

1. **Size preset** — 2K (2048×1024), 4K (4096×2048), 8K (8192×4096), or custom `--width` / `--height`
2. **Output type** — equirectangular, cubemap, or both (`--projection-mode`)
3. **Scene mode** — `normal` (Milky Way) or `nebula` (broad cloud complex) via `--mode`
4. **Core options** — each prompt shows the matching CLI flag and a short help line
5. **Advanced?** `[y/N]` — optional debug / morphology / profile knobs
6. Prints a short **summary** plus an **equivalent command** you can reuse without `--guided`
7. **Start render?** `[Y/n]` — choose `n` to exit after discovery (no images written)

Type `q` at any prompt to cancel. You can still pass flags with `--guided`; those values become the defaults shown in prompts.

### Local copyright and license (optional)

Before generate runs, starsky-gen may ask for copyright / distribution status to embed in PNG (and JPEG Copyright) metadata:

1. **C**opyrighted / **O**pen / **U**nknown (Enter = Unknown)
2. If **O** or **U** — no further prompts in that session; both Copyright and License metadata use `open` / `unknown`
3. If **C** — asks **distribution rights / license** (free text, e.g. `All Rights Reserved`, `CC BY 4.0`), then an **optional copyright holder** name (Enter → `© YEAR` only)

Once **open** or **copyrighted** is saved, generate does not ask again. **Unknown** (or Enter) re-asks next generate — a light legal nudge, not a recurring interrupt. The local YAML `.license` file stores legal status (optional holder, year, `distribution_rights`, run count, last seed / look target) plus optional CLI `defaults` from `--default`; it is gitignored.

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

Seeds with `--generations N`:

- `--seed 42 --generations 40` → seeds `42` … `81` (start + N−1)
- omit `--seed` → a unique random seed for each generation (no repeats in the batch)

Recommended sizes:

- Equirectangular skybox: `4096x2048`
- Cubemap faces: `1024` or `2048` (`--cubemap-face-size`)

Nebula modes:

- `distant` — isolated feature
- `full` — immersive fog
- `galaxy_streak` — Milky-Way-like plane (default)

## Look target and display tuning

Primary product look is **cinematic** (AAA wrap-safe skybox). Photo and long-exposure looks are opt-in via `--look-target`:

```bash
starsky-gen --look-target cinematic      # wrap-safe skybox (default)
starsky-gen --look-target documentary    # parked photo grade (no star trails)
starsky-gen --look-target long_exposure  # documentary + stacked trails look
```

`--long-exposure-look` / `--flat-exposure` remain as an advanced override for cinematic only; prefer `--look-target long_exposure` for the full trail stack.

Galaxy renders still use magnitude-based PSF stars by default (`--photoreal-stars`) for sharp, realistic color — that is not the same as a documentary camera grade.

### Look recipes

`finalize_render_config()` applies a look recipe, then the Milky Way recognition preset (normal mode). Shared morphology/extinction/PSF stay available; the sandwich *composite* path differs for cinematic vs documentary/long_exposure.

| Knob | Cinematic | Documentary / long_exposure |
|------|-----------|------------------------------|
| `lane_star_dropout_boost` | moderate (~0.55) | **0** |
| `post_star_lane_protect_strength` | soft (~0.48; midplane glow OK) | **0** |
| `dark_rift_strength` | on (~0.78+) | **0** |
| `star_layer_bloom_strength` / `star_psf_bloom_scale` | stronger hero bloom | **0** / lower |
| `structure_type_competition` | competitive types | **0** (soft blend) |
| Tone / grain / bloom | filmic, bloom up, grain down | asinh + ISP stack |
| Star trails | off | off (documentary) / on (long_exposure) |

Cinematic prioritizes **volumetric glow + color hierarchy**. BandPlan still drives protect/rifts; `--debug-layers` exports band-plan PNGs. Documentary/long_exposure zero sandwich leftovers.

### Layer toggles and isolate

- `--stars` / `--no-stars` — **background** catalogs (band/mid/cosmic/halo), not heroes
- `--noheroes` — disable hero stars only (spikes/ghosts); heroes can render with `--no-stars`
- `--nebula` / `--no-nebula` — gas; when off, guided skips nebula-mode/style/dust prompts
- `--background-density` — off-plane/cosmic/halo **star** density `[0–2]` (0 = none); blank = look default
- `--cloud-continuity` `[0–1.6]`, `--dust-coverage` `[0–1.6]`, `--dust-strength` `[0–1.8]` (0 = off/none)
- `--nebula-hero-color` — hero nebula tint (`--mode nebula`): `classic-red`/`red` (`#FF1818`), `gold-core`/`gold`, `cool-blue`/`blue`, `magenta-accent`/`magenta`, `neutral`; or hex (`FF1818`, `0xFF1818`, `'#FF1818'`) / `R,G,B`. In bash, never pass a bare `#…` — the shell treats `#` as a comment (use `FF1818` or quote `'#FF1818'`).
- `--depth` / `--no-depth` — dim catalog stars toward poles (not camera DOF)
- `--render-profile` — `physical` = linear only; `physical_grade` = +grade; `full` = +display finish/grain
- `--isolate-layer` — final plate on black: `none` | `background_stars` | `heroes` | `nebula_gas`

```bash
# Black nebula + heroes only
starsky-gen --mode nebula --black-background --no-stars --nebula \
  --seed 42 --isolate-layer none
# Heroes plate only
starsky-gen --mode nebula --no-stars --isolate-layer heroes --seed 42
```

Key knobs:

- `--look-target` (`cinematic|documentary|long_exposure`)
- `--noheroes` — disable hero stars while keeping background stars
- `--mag-bright`, `--mag-faint`, `--band-star-density`
- `--asinh-gain`, `--asinh-q`, `--galaxy-tone-curve` (`reinhard|asinh|filmic|aces|acescct`; default `filmic`)
- `--disk-star-density-dropout` — thin resolved stars in the bright plane
- `--star-band-chroma-desat`, `--star-band-brightness-scale` — plane star look
- `--render-profile physical|physical_grade|full`

## Debug / morphology layers

Export intermediate maps for tuning extinction, resolve dropout, and structure:

```bash
starsky-gen --seed 42 --debug-layers -o output
starsky-gen --seed 42 --debug-layers --grayscale-morphology -o output
```

Writes `output/<output-base-name>_layers/` with PNGs such as:

- **Any scene:** `extinction.png`, `resolved_stars.png`
- **`--mode nebula`:** `cloud_structure.png`, `cloud_emission.png`, `cloud_dust.png`
- **`--mode normal` (MW morphology):** `density_G.png`, `dust_D.png`, `unresolved_U.png`, `resolve_W.png`, `dropout_mask.png`, and related architecture / band-plan plates when built

```bash
starsky-gen --mode nebula --no-stars --debug-layers --seed 42 -o output
```

### Structure architecture (weight / type system)

Galactic morphology uses two layers:

- **Content** — what clouds look like at different scales (`puff_field`, `vertical_extent`, `latent_turb`, competitive mega/med/small/fine in dust microstructure).
- **Plan** — where each mechanism runs, via `StructureArchitectureWeights` in `structure_envelope.py` (`build_structure_architecture_weights` → `apply_structure_architecture_pipeline`).

**Sandwich model (Milky Way–like):** the band is split into a **tight inner dust lane** (dark sub-band on the mid-plane) and a **puff shell** (fluffy ISM above and below). **Negative space** is carved in the shell as clearer transmission pockets between clouds — not empty black off the band, but gaps inside the luminous layer.

Config knobs (on `FeatureConfig` / generator):

| Field | Role |
|-------|------|
| `structure_architecture_strength` | Global drive for the architecture pipeline (default `1.0`) |
| `structure_band_core_drive` | Minimum turbulent drive inside the band core, not only rims (default `0.52`) |
| `structure_inner_dust_lane_strength` | Inner dust sub-band absorption (default `0.72`) |
| `structure_vertical_extent` | Vertical envelope / plume strength |
| `structure_host_latitude_scale` | How far structure may extend above/below the plane |

**Grayscale architecture layers**

| File | Meaning |
|------|---------|
| `turbulence_architecture.png` | Master drive — where turbulent architecture should show through (bright = active) |
| `architecture_puff_W.png` | Puff (cotton-ball) injection weight |
| `architecture_reinject_W.png` | Reinject (fragments / tears) weight |
| `architecture_dust_lane.png` | Inner dust sub-band weight |
| `architecture_negative_space.png` | Clear pockets in the puff shell |

**`structure_type_map.png` (RGB type map)**

Colored debug view of which architecture **type** dominates each pixel (sharpened weighted blend). Use this when tuning band fluff vs etched cores; a single grayscale weight map does not show type competition.

| Color | Type | Mechanism |
|-------|------|-----------|
| Red-brown | Dust lane | Inner sub-band — dense absorption on mid-plane |
| Cyan | Puff | `build_fine_puff_field` + puff inject (mostly shell) |
| Green | Vertical | `vertical_extent` plumes (shell + plumes) |
| Magenta | Reinject | `reinject_vertical_dust_structure` fragments |
| Yellow | Turb | Latent turbulence high-frequency |
| Brown | Survival | `structure_survival` host |
| Pale gray | Negative space | Clear gaps in puff shell (`apply_band_negative_space_transmission`) |
| Dark blue-gray | Low drive | `turbulence_drive` below threshold — architecture barely applied |

**How to read it**

- Mid-plane should show **red-brown dust lane** (inner sub-band), with **cyan/green** above and below (puff shell) — sandwich, not one muddy blend.
- **Pale patches** in the shell = negative space (good if you want Milky-Way-like gaps between clouds).
- Band **core** should not be mostly dark blue-gray (low drive → flat `dust_A`).
- Compare to `dust_A.png` (morph only) and `dust_A_effective.png` (after extinction merge). If effective is fluffier, the final grade/extinction path is helping; raise `structure_band_core_drive` or `structure_architecture_strength` to push that into morph.
- Matching a Milky Way reference (final composite) requires `extinction.png` and the final equirect, not `disk_weight` or `dust_D` alone.

Scale hierarchy types (mega / medium / small / fine) are **not** shown on this map yet; they live in dust microstructure / ISM hierarchy passes.

## Toward Milky Way (3-second read)

Older **main**-branch renders (`92ac1fe` photoreal pipeline) often passed a quick “is that the Galactic plane?” test. The **failed** morphology branch adds structure fidelity (architecture weights, dust lane, debug maps) but can lose **legibility** if the composite is tuned like a nebula still frame — brown haze, soft edges, stars buried.

Image 3 (reference) reads fast because of **four simultaneous cues**, not one map:

1. **Dark dust sub-band** on the mid-plane (sandwich lane).
2. **Bright star carpet** above and below the lane (not a smooth glow blob).
3. **Warm ISM** with **gaps** (negative space) in the shell.
4. **Sparse red** H II — rare pockets only.

### What regressed vs “good” versions

| Symptom now | Likely cause |
|-------------|----------------|
| Monolithic brown/red haze | `galactic_band_color_grade` + heavy blur in `_apply_band_ism_dominance` when `preserve_detail=False` |
| No obvious lane | Lane must appear in **`extinction.png`** and final grade, not only `dust_A` / type map |
| Stars invisible in band | `band_ism_dominance` + plane star dropout + ISM lift without `preserve_detail` on morph ISM |
| Muddy type map | Types blending; tune `structure_inner_dust_lane_strength`, re-export `structure_type_map.png` |

Structure work (architecture, puff shell, dust lane) is **necessary** but **not sufficient** — the 3-second read is won in the **composite + display** stack.

### Diagnostic order (fastest path)

1. `extinction.png` — dark filamentary lane on mid-plane?
2. `resolved_stars.png` + final equirect — stars visible in shell, lane darker?
3. `structure_type_map.png` — red-brown lane stripe, cyan/green shell, pale gaps?
4. `dust_D_effective.png` vs `dust_A.png` — effective should show lane + fluffy shell

### Milky Way recognition preset (on by default)

For `nebula_mode=galaxy_streak`, `milky_way_recognition_preset=True` (default) applies a tuned floor at render start via `apply_milky_way_recognition_preset()` — stronger lane/architecture, more ISM lift + dark patches, more plane stars. Under `--look-target cinematic` it also allows darker dust lanes (vista sandwich); documentary keeps milder color-grade caps. Set `milky_way_recognition_preset=False` to use only the scalar defaults in `FeatureConfig`.

### Cinematic vista smoke checklist

After retuning look defaults, verify a seeded render still passes a 3-second read:

```bash
starsky-gen --seed 42 --look-target cinematic --debug-layers \
  -o output/vista_check --output-base-name vista
```

Check:

1. Final equirect — dark mid-plane dust lane, warm ISM with gaps (not a smooth sepia slab)
2. Varied star sizes / a few hero blooms (not one white pinprick size)
3. `structure_type_map.png` — red-brown lane + cyan/green puff shell
4. `extinction.png` — filamentary dark lane on the mid-plane
5. Wrap-safe — no frame vignette skew; flat exposure still default

### Suggested knob bundle (Python / `FeatureConfig`)

Defaults now lean toward this bundle; override individual fields or disable the preset above:

| Field | Direction | Why |
|-------|-----------|-----|
| `morphology_dust_primary` | `True` | Morph-led extinction + `build_morphology_ism_layers` with `preserve_detail` |
| `band_ism_dominance` | `0.52`–`0.68` | Diffuse ISM visible, but don’t drown stars |
| `band_dark_patch_strength` | `0.48`–`0.70` | Dark lanes in the luminous band |
| `structure_inner_dust_lane_strength` | `0.82`–`0.95` | Inner dust sub-band in morph |
| `structure_architecture_strength` | `1.12`–`1.20` | Full architecture pipeline |
| `galactic_band_separation_strength` | `1.18`–`1.30` | Cloud/lane gaps (also `apply_band_luma_separation` late in generator) |
| `galactic_band_dust_black_strength` | `0.62`–`0.92` | Near-black dust carve (cinematic vista) |
| `galactic_band_color_grade_strength` | `0.58`–`0.72` | Lower if final image is one brown wash |
| `display_bloom_strength` | `0.09`–`0.11` | Hero/hot-core bloom; keep `bloom_mix_wide` low |
| `galactic_band_gas_fluff_strength` | `0.85`–`1.0` | Puffy shell pockets |
| `disk_star_density_dropout` | `0.35`–`0.48` | Keep stars in the plane |
| `split_star_display_grade` | `True` | Sharper stars on graded nebula |
| `morphology_extinction_lane_carve` | `0.85`+ | Lane shows in `ext_paint` |
| `extinction_void_floor` | `0.006`–`0.012` | Deep lanes without slabbing all detail |

**A/B profiles**

- `--render-profile physical_grade` — drop late display mush; check structure + stars.
- `--render-profile full` — production look; compare to physical_grade to see what display passes cost you.

**Temporary isolation**

- Set `galactic_band_color_grade=False` in config for one render: if it suddenly “passes 3 seconds,” the morph/extinction stack is fine and the **final grade** needs tuning, not more FBM.

### What to build next (code)

1. **Scale type map** (mega/med/small/fine) alongside architecture RGB map.
2. **Couple scale hierarchy → `turbulence_drive`** so fine detail survives all passes.
3. **Export `extinction.png` vs lane mask overlay** in debug layers for side-by-side lane check.

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
