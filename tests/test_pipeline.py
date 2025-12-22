import json
from pathlib import Path

import pytest

from scripts import run_pipeline


def test_bits_from_any_parses_bits_and_pairs():
    bits_cfg = {"bits": "1 1 1"}
    bits = run_pipeline._bits_from_any(bits_cfg, k=2)
    assert bits.tolist() == [1, 1, 1]

    pairs_cfg = {"pairs": [(0, 1)], "allow_self_loops": True}
    bits_pairs = run_pipeline._bits_from_any(pairs_cfg, k=2)
    assert bits_pairs.tolist() == [1, 1, 1]


def test_bits_from_any_invalid_length():
    with pytest.raises(ValueError):
        run_pipeline._bits_from_any({"bits": "10"}, k=2)


def test_run_pipeline_exact_only(tmp_path):
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    args = run_pipeline.build_parser().parse_args(
        [
            "--n",
            "2",
            "--k",
            "2",
            "--rule-bits",
            "111",
            "--no-spectral",
            "--out-dir",
            str(out_dir),
            "--cache-dir",
            str(cache_dir),
            "--run-tag",
            "unit",
            "--heartbeat",
            "0.01",
        ]
    )

    results = run_pipeline.run_pipeline(args)

    assert len(results) == 1
    row = results[0]
    assert row.get("exact_Z") is not None
    assert row.get("boundary") == "torus"

    run_dir = out_dir / "unit"
    csv_path = run_dir / "summary.csv"
    jsonl_path = run_dir / "summary.jsonl"
    assert csv_path.exists()
    assert jsonl_path.exists()

    content = csv_path.read_text(encoding="utf-8")
    assert "exact_Z" in content
    first_line = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_line["rule_bits_canon"] == "111"
    assert first_line["sym_mode"] == "perm"
