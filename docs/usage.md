# Usage Guide

Run with CLI flags:

```bash
starsky-gen generate
```

Default values include `--nebula-mode galaxy_streak`, `--output-format png`, and `--quality 100`.
Background defaults to noisy bluish-gray and darkens away from the galactic disk.
Use `--black-background` only if you explicitly want a black backdrop.

Pass all options explicitly when needed:

```bash
starsky-gen generate \
  --width 4096 \
  --height 2048 \
  --projection-mode both \
  --output-format png \
  --nebula-mode galaxy_streak \
  --generations 8 \
  --seed 77
```

Recommended settings:

- Skybox equirectangular: `4096x2048`
- Cubemap faces: `1024` or `2048`
- Nebula:
  - `distant` for isolated feature
  - `full` for immersive fog
  - `galaxy_streak` for Milky Way-like plane

For reproducible batches, set a fixed seed.
