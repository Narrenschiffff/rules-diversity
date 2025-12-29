import json
from pathlib import Path

import numpy as np

from rules import config
from rules.cache import load_eval_cache
from rules.eval import bits_from_rule, evaluate_rules_batch


def _dense_three_state_rule() -> np.ndarray:
    """A small fully-connected rule that yields non-empty archetype tags."""
    return bits_from_rule(np.ones((3, 3), dtype=bool))


def _overwrite_cache_missing_fields(cache_path: Path) -> None:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    data["result"].pop("archetype_tags", None)
    data["result"].pop("archetype_hits", None)
    data["result"].pop("archetype_tags_merged", None)
    data["result"].pop("archetype_hits_merged", None)
    cache_path.write_text(json.dumps(data), encoding="utf-8")


def _first_cache_file(cache_dir: Path) -> Path:
    files = sorted(cache_dir.glob("*.json"))
    assert files, "no cache file generated"
    return files[0]


def test_cache_hit_backfills_missing_archetype_fields(tmp_path: Path):
    bits = _dense_three_state_rule()

    # Seed cache.
    first = evaluate_rules_batch(
        n=2,
        k=3,
        bits_list=[bits],
        sym_mode="perm+swap",
        boundary="torus",
        device="cpu",
        enable_spectral=False,
        enable_exact=False,
        cache_dir=tmp_path,
        use_cache=True,
    )[0]
    cache_path = _first_cache_file(tmp_path)

    # Simulate legacy cache without archetype fields.
    _overwrite_cache_missing_fields(cache_path)

    second = evaluate_rules_batch(
        n=2,
        k=3,
        bits_list=[bits],
        sym_mode="perm+swap",
        boundary="torus",
        device="cpu",
        enable_spectral=False,
        enable_exact=False,
        cache_dir=tmp_path,
        use_cache=True,
    )[0]

    assert second.get("archetype_tags", "") != ""
    assert "archetype_hits" in second and second["archetype_hits"]
    assert "archetype_tags_merged" in second
    assert "archetype_hits_merged" in second

    refreshed = load_eval_cache(tmp_path, cache_path.stem)
    assert refreshed is not None
    assert refreshed.get("archetype_tags", "") == second["archetype_tags"]
    assert "archetype_tags_merged" in refreshed
    assert "archetype_hits_merged" in refreshed


def test_refresh_cache_forces_recompute_and_writes_archetypes(tmp_path: Path):
    bits = _dense_three_state_rule()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # First write a valid cache entry.
    _ = evaluate_rules_batch(
        n=2,
        k=3,
        bits_list=[bits],
        sym_mode="perm+swap",
        boundary="open",
        device="cpu",
        enable_spectral=False,
        enable_exact=False,
        cache_dir=cache_dir,
        use_cache=True,
    )[0]
    cache_path = _first_cache_file(cache_dir)

    # Replace result with an empty payload to mimic old cache.
    cache_path.write_text(
        json.dumps({"meta": {"version": config.EVAL_CACHE_VERSION}, "result": {"rows_m": 0}}),
        encoding="utf-8",
    )

    out = evaluate_rules_batch(
        n=2,
        k=3,
        bits_list=[bits],
        sym_mode="perm+swap",
        boundary="open",
        device="cpu",
        enable_spectral=False,
        enable_exact=False,
        cache_dir=cache_dir,
        use_cache=True,
        refresh_cache=True,
    )[0]

    assert "archetype_tags" in out
    assert "archetype_tags_merged" in out
    reloaded = load_eval_cache(cache_dir, cache_path.stem)
    assert reloaded is not None
    assert "archetype_tags" in reloaded
    assert "archetype_tags_merged" in reloaded
