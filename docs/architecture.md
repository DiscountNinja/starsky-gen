# Architecture

`starsky-gen` is split into small rendering stages:

1. **Prompt + validation** (`cli.py`, `config.py`)
2. **Star catalog sampling** (`starfield.py`, `placement.py`, optional `catalog_data.py`)
3. **Color / PSF** (`color_science.py`, `psf.py`)
4. **Background + depth + nebula + bulge compositing** (`generator.py`, `nebula.py`, `bulge.py`, `procedural_noise.py`)
5. **HDR tone map + grade** (`tone_map.py`, `grade.py`)
6. **Projection output** (`projections.py`)
7. **JPEG post** (`postprocess.py`)

Galaxy renders composite linear HDR layers (sky, nebula/dust, Sérsic bulge, faint/bright stars), then apply ACES/asinh/filmic tone mapping and vignette/Bayer/dodge-burn grading.

The renderer keeps a common spherical sky model and maps it to projection outputs so that seams are naturally reduced and behavior is reproducible by seed.
