"""Batch seed resolution: sequential when base set, unique random when not."""

from starsky_gen.batch_seeds import format_batch_seed_summary, resolve_batch_seeds


def test_resolve_batch_seeds_sequential() -> None:
    assert resolve_batch_seeds(42, 1) == [42]
    assert resolve_batch_seeds(42, 40) == list(range(42, 82))
    assert resolve_batch_seeds(42, 40)[-1] == 81


def test_resolve_batch_seeds_unique_random() -> None:
    seeds = resolve_batch_seeds(None, 25)
    assert len(seeds) == 25
    assert len(set(seeds)) == 25
    assert all(isinstance(s, int) and s >= 0 for s in seeds)


def test_format_batch_seed_summary() -> None:
    seq = resolve_batch_seeds(42, 40)
    assert "42" in format_batch_seed_summary(42, seq)
    assert "81" in format_batch_seed_summary(42, seq)
    rnd = resolve_batch_seeds(None, 3)
    assert "unique random" in format_batch_seed_summary(None, rnd)
