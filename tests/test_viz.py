import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from rules import viz


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_viz_handles_symmetry_filter_and_summary(tmp_path: Path):
    # stage1 raw (perm)
    raw = tmp_path / "stage1_pareto_n2_k2_raw.csv"
    _write_csv(
        raw,
        [
            {
                "n": 2,
                "k": 2,
                "rule_count": 2,
                "sum_lambda_powers": 2.0,
                "lambda_max": 1.0,
                "lambda_top2": "(1.0,0.1)",
                "active_k": 1,
                "active_k_raw": 2,
                "k_sym": 1,
                "sym_mode": "perm",
                "boundary": "torus",
            }
        ],
    )

    # stage1 canon (perm+swap)
    canon = tmp_path / "stage1_pareto_n2_k2_canon.csv"
    _write_csv(
        canon,
        [
            {
                "n": 2,
                "k": 2,
                "rule_count": 3,
                "sum_lambda_powers": 3.0,
                "lambda_max": 1.2,
                "lambda_top2": "(1.2,0.3)",
                "active_k": 1,
                "active_k_raw": 3,
                "k_sym": 1,
                "sym_mode": "perm+swap",
                "boundary": "torus",
            }
        ],
    )

    # GA front (perm)
    ga = tmp_path / "pareto_front_demo.csv"
    _write_csv(
        ga,
        [
            {
                "run_tag": "demo",
                "n": 2,
                "k": 2,
                "rule_bits": "111",
                "rule_count": 3,
                "rows_m": 1,
                "lambda_max": 1.1,
                "lambda_top2": "(1.1,0.2)",
                "spectral_gap": 0.9,
                "sum_lambda_powers": 1.1,
                "is_front0": 1,
                "active_k": 2,
                "active_k_raw": 2,
                "k_sym": 2,
                "sym_mode": "perm",
                "lower_bound": 1.1,
                "upper_bound": 1.1,
                "lower_bound_raw": 1.1,
                "upper_bound_raw": 1.1,
                "upper_bound_raw_gersh": 1.1,
                "upper_bound_raw_maxdeg": 1.1,
                "archetype_tags": "",
                "exact_Z": "",
                "trace_exact": "",
                "trace_estimate": "",
                "trace_error": "",
                "trace_error_rel": "",
                "eval_note": "",
            }
        ],
    )

    csvs = [raw, canon, ga]

    # Summary captures symmetry counts and active_k ranges.
    summary = viz.summarize_runs(csvs)
    assert summary[raw.name]["symmetry_counts"]["perm"] == 1
    assert summary[ga.name]["active_k_min"] == 2
    assert summary[canon.name]["symmetry_counts"]["perm+swap"] == 1

    # Filtering by symmetry keeps only matching rows for plots.
    out_dir = tmp_path / "figs"
    paths = viz.plot_all([str(p) for p in csvs], n=2, k=2, out_dir=str(out_dir), sym_filter="perm")
    assert len(paths) == 3
    assert all(Path(p).exists() for p in paths)

    # Filtered summary should drop the perm+swap row.
    summary_perm = viz.summarize_runs(csvs, sym_filter="perm")
    assert canon.name not in summary_perm
    assert summary_perm[raw.name]["symmetry_counts"] == {"perm": 1}
