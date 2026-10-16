# Architecture

`starsky-gen` targets **cinematic AAA game skyboxes** (wrap-safe equirect/cubemap). Photo-documentary grading is a secondary `look_target`.

Rendering stages:

1. **Prompt + validation** (`cli.py`, `config.py`) — including `LookTarget` + `finalize_render_config`
2. **Star catalog sampling** (`starfield.py`, `placement.py`, optional `catalog_data.py`)
3. **Color / PSF** (`color_science.py`, `psf.py`)
4. **Background + depth + nebula + bulge compositing** (`generator.py`, `nebula.py`, `bulge.py`, `procedural_noise.py`)
5. **HDR tone map + display finish** (`tone_grade.py`, `tone_map.py`, `grade.py`, `postfx.py`, `optical_effects.py`)
6. **Projection output** (`projections.py`) — seam-aware / `wrap_safe`
7. **JPEG post** (`postprocess.py`)

Galaxy renders composite linear HDR layers (sky, nebula/dust, Sérsic bulge, faint/bright stars), then apply look-target grade: default **filmic + bloom** for cinematic vistas; documentary restores asinh / long-exposure / grain.

The renderer keeps a common spherical sky model and maps it to projection outputs so that seams are naturally reduced and behavior is reproducible by seed.

## Galactic morphology: weight / type plan

`galactic_structure.py` builds a shared latent stack (ridge, turb, disk weight, dust absorption/transmission, unresolved prior). Turbulent cloud **content** (puff, vertical extent, multi-scale noise) is separate from the **plan** that routes it:

1. `build_structure_architecture_weights()` — per-pixel maps: `turbulence_drive`, `puff_weight`, `vertical_weight`, `reinject_weight`, etc.
2. `apply_structure_architecture_pipeline()` — reinject → decouple → puff inject → cloud blend, all gated by those weights.
3. Competitive scale hierarchy (`blend_competitive_scale_hierarchy`) — still supplies local texture inside allowed regions; weights do not replace scales.

Debug export (`--debug-layers`) includes `structure_type_map.png`: an RGB legend of dominant architecture types (cyan puff, green vertical, magenta reinject, yellow turb, brown survival, dark = low drive). See [usage.md — Structure architecture](usage.md#structure-architecture-weight--type-system).
