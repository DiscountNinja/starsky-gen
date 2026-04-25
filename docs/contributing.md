# Contributing

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development flow

1. Add or update tests in `tests/`.
2. Keep distribution assumptions explicit and tested.
3. Run:

```bash
pytest
```

## Code style

- Keep comments focused on algorithm intent.
- Avoid over-saturating colors; realism first.
- Preserve seam safety for equirectangular and cubemap outputs.
