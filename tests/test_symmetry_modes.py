import csv
from pathlib import Path

import numpy as np

from rules.eval import apply_rule_symmetry, bits_from_rule, evaluate_rules_batch
from rules.ga import GAConfig, ga_search_with_batch


def _complete_rule_bits(k: int) -> np.ndarray:
    """All-ones adjacency (including self loops) for quick symmetry checks."""
    return bits_from_rule(np.ones((k, k), dtype=bool))


def test_apply_rule_symmetry_handles_none_perm_exchange():
    bits = _complete_rule_bits(3)

    bits_none, k_none, raw_none, cls_none = apply_rule_symmetry(bits, 3, "none")
    assert k_none == 3
    assert raw_none.tolist() == [0, 1, 2]
    assert cls_none == [1, 1, 1]
    np.testing.assert_array_equal(bits_none, bits)

    bits_perm, k_perm, raw_perm, cls_perm = apply_rule_symmetry(bits, 3, "perm")
    assert k_perm == 3
    assert raw_perm.tolist() == [0, 1, 2]
    assert cls_perm == [1, 1, 1]
    np.testing.assert_array_equal(bits_perm, bits)

    bits_swap, k_swap, raw_swap, cls_swap = apply_rule_symmetry(bits, 3, "perm+swap")
    assert k_swap == 1  # identical rows/cols are merged
    assert raw_swap.tolist() == [0, 0, 0]
    assert cls_swap == [3]
    np.testing.assert_array_equal(bits_swap, np.array([1], dtype=np.uint8))


def test_evaluate_reports_active_states_with_exchange(tmp_path: Path):
    bits = _complete_rule_bits(3)

    out = evaluate_rules_batch(
        n=2,
        k=3,
        bits_list=[bits],
        sym_mode="perm+swap",
        device="cpu",
        enable_spectral=False,
        enable_exact=True,
        cache_dir=tmp_path,
        use_cache=False,
    )[0]

    assert out["k_sym"] == 1
    assert out["sym_mode"] == "perm+swap"
    assert out["active_k"] == 1
    assert out["active_k_raw"] == 3  # merged class expands back to 3 raw states
    assert out["exact_Z"] == 1  # only one legal row on the compressed graph


def test_eval_cache_reuse_does_not_recompute(tmp_path: Path, monkeypatch):
    bits = _complete_rule_bits(2)

    first = evaluate_rules_batch(
        n=2,
        k=2,
        bits_list=[bits],
        sym_mode="perm",
        device="cpu",
        enable_spectral=False,
        enable_exact=True,
        cache_dir=tmp_path,
        use_cache=True,
    )[0]

    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1

    # If cache miss, patched exact counter would raise; cache hit should bypass it.
    import rules.stage1_exact as stage1_exact

    def _boom(*_, **__):
        raise RuntimeError("cache reuse failed")

    monkeypatch.setattr(stage1_exact, "count_patterns_transfer_matrix", _boom)

    second = evaluate_rules_batch(
        n=2,
        k=2,
        bits_list=[bits],
        sym_mode="perm",
        device="cpu",
        enable_spectral=False,
        enable_exact=True,
        cache_dir=tmp_path,
        use_cache=True,
    )[0]
    assert second["active_k"] == first["active_k"]
    assert second["active_k_raw"] == first["active_k_raw"]
    assert second.get("exact_Z") == first.get("exact_Z")


def test_ga_respects_symmetry_modes(tmp_path: Path):
    for sym_mode in ("none", "perm+swap"):
        tag = f"sym_{sym_mode.replace('+', '_')}"
        conf = GAConfig(
            pop_size=4,
            generations=1,
            device="cpu",
            use_lanczos=False,
            enable_exact=False,
            enable_spectral=False,
            fast_eval=True,
            sym_mode=sym_mode,
            cache_dir=tmp_path,
            use_cache=False,
        )
        ga_search_with_batch(n=2, k=2, ga_conf=conf, out_csv_dir=str(tmp_path), run_tag=tag)

        front_path = tmp_path / f"pareto_front_{tag}.csv"
        assert front_path.exists()
        with open(front_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        header = rows[0]
        assert header[0:5] == ["run_tag", "n", "k", "generation", "rule_bits"]
        sym_idx = header.index("sym_mode")
        active_k_idx = header.index("active_k")
        active_k_raw_idx = header.index("active_k_raw")
        # All data rows should carry the requested symmetry label and active_k columns.
        for data in rows[1:]:
            assert data[sym_idx] == sym_mode
            assert data[active_k_idx] != ""  # active_k
            assert data[active_k_raw_idx] != ""  # active_k_raw
