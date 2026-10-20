"""Resolve per-generation seeds for multi-render batches."""

from __future__ import annotations

import numpy as np


def resolve_batch_seeds(base: int | None, generations: int) -> list[int]:
    """Return one seed per generation.

    - ``base`` set: ``base, base+1, …, base+n-1`` (deterministic sequence).
    - ``base`` None: unique random seeds for the batch (no repeats within the list).
    """
    n = int(generations)
    if n < 1:
        raise ValueError(f"generations must be >= 1 (got {generations!r})")
    if base is not None:
        b = int(base)
        return [b + i for i in range(n)]

    used: set[int] = set()
    out: list[int] = []
    # Bound to signed 31-bit for stable CLI display / cross-platform logs.
    mask = 0x7FFFFFFF
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > n * 1000 + 100:
            raise RuntimeError("failed to allocate unique random seeds")
        s = int(np.random.SeedSequence().entropy) & mask
        if s in used:
            continue
        used.add(s)
        out.append(s)
    return out


def format_batch_seed_summary(base: int | None, seeds: list[int]) -> str:
    """Short human summary for previews / logs."""
    if not seeds:
        return "no seeds"
    if base is not None:
        if len(seeds) == 1:
            return f"seed {seeds[0]}"
        return f"seeds {seeds[0]}–{seeds[-1]} ({len(seeds)} gens, +1 each)"
    if len(seeds) == 1:
        return f"random seed {seeds[0]}"
    return f"{len(seeds)} unique random seeds (no repeats)"
