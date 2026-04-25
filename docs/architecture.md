# Architecture

`starsky-gen` is split into small rendering stages:

1. **Prompt + validation** (`cli.py`, `config.py`)
2. **Star catalog sampling** (`starfield.py`)
3. **Background + depth + nebula compositing** (`generator.py`, `nebula.py`)
4. **Projection output** (`projections.py`)
5. **Post processing** (`postprocess.py`)

The renderer keeps a common spherical sky model and maps it to projection outputs so that seams are naturally reduced and behavior is reproducible by seed.
