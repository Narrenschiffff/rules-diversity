import csv
import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

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


def test_entropy_convergence_aggregates_csv(tmp_path: Path):
    csv_a = tmp_path / "n3.csv"
    csv_b = tmp_path / "n4.csv"
    _write_csv(
        csv_a,
        [
            {
                "n": 3,
                "k": 2,
                "boundary": "torus",
                "sym_mode": "perm",
                "rule_bits": "1111",
                "sum_lambda_powers_penalized": 8.0 / 3.0,
                "penalty_factor": 3.0,
            }
        ],
    )
    _write_csv(
        csv_b,
        [
            {
                "n": 4,
                "k": 2,
                "boundary": "torus",
                "sym_mode": "perm",
                "rule_bits": "1111",
                "sum_lambda_powers_penalized": 4.0,
                "penalty_factor": 4.0,
            }
        ],
    )

    series = viz._prepare_entropy_series(
        rule_bits=None,
        k=None,
        n_min=3,
        n_max=4,
        csv_paths=[str(csv_a), str(csv_b)],
        boundary=None,
        sym_mode=None,
        device="cpu",
        use_penalty=True,
        normalize=True,
        read_existing_exact=True,
        read_existing_estimate=True,
    )
    assert series.k == 2
    assert series.boundary == "torus"
    np.testing.assert_allclose(series.n_vals, [3, 4])
    expected = [math.log(8.0 / 3.0) / 3.0, math.log(4.0) / 4.0]
    np.testing.assert_allclose(series.log_vals, expected)

    out_figs = viz.plot_entropy_convergence(
        rule_bits=None,
        k=None,
        n_min=3,
        n_max=4,
        csv_paths=[str(csv_a), str(csv_b)],
        boundary=None,
        sym_mode=None,
        device="cpu",
        out_dir=str(tmp_path / "figs"),
        style="default",
    )
    assert len(out_figs) == 1
    assert Path(out_figs[0]).exists()


def test_entropy_requires_bits_or_rows(tmp_path: Path):
    with pytest.raises(ValueError):
        viz._prepare_entropy_series(
            rule_bits=None,
            k=None,
            n_min=2,
            n_max=3,
            csv_paths=None,
            boundary=None,
            sym_mode=None,
            device="cpu",
            use_penalty=True,
            normalize=True,
            read_existing_exact=True,
            read_existing_estimate=True,
        )


def test_frontier_surface_keypoints_and_contour(tmp_path: Path):
    front_k2 = tmp_path / "front_k2.csv"
    _write_csv(
        front_k2,
        [
            {"n": 2, "k": 2, "rule_count": 1, "objective_penalized": 1.2, "boundary": "torus", "sym_mode": "perm"},
            {"n": 2, "k": 2, "rule_count": 2, "objective_penalized": 2.8, "boundary": "torus", "sym_mode": "perm"},
            {"n": 2, "k": 2, "rule_count": 3, "objective_penalized": 2.9, "boundary": "torus", "sym_mode": "perm"},
            {"n": 3, "k": 2, "rule_count": 1, "objective_penalized": 1.3, "boundary": "torus", "sym_mode": "perm"},
            {"n": 3, "k": 2, "rule_count": 2, "objective_penalized": 3.0, "boundary": "torus", "sym_mode": "perm"},
            {"n": 3, "k": 2, "rule_count": 3, "objective_penalized": 3.4, "boundary": "torus", "sym_mode": "perm"},
        ],
    )
    front_k3 = tmp_path / "front_k3.csv"
    _write_csv(
        front_k3,
        [
            {"n": 2, "k": 3, "rule_count": 1, "objective_penalized": 0.8, "boundary": "torus", "sym_mode": "perm"},
            {"n": 2, "k": 3, "rule_count": 2, "objective_penalized": 1.1, "boundary": "torus", "sym_mode": "perm"},
            {"n": 2, "k": 3, "rule_count": 3, "objective_penalized": 1.4, "boundary": "torus", "sym_mode": "perm"},
            {"n": 3, "k": 3, "rule_count": 1, "objective_penalized": 0.9, "boundary": "torus", "sym_mode": "perm"},
            {"n": 3, "k": 3, "rule_count": 2, "objective_penalized": 1.6, "boundary": "torus", "sym_mode": "perm"},
            {"n": 3, "k": 3, "rule_count": 3, "objective_penalized": 1.9, "boundary": "torus", "sym_mode": "perm"},
        ],
    )

    out_dir = tmp_path / "figs"
    figs, data = viz.plot_frontier_surfaces(
        front_csvs=[str(front_k2), str(front_k3)],
        ks=[2, 3],
        boundary="torus",
        sym_mode="perm",
        metric="objective_penalized",
        plot_types=["contour"],
        out_dir=str(out_dir),
        style="default",
        contour_levels=4,
    )
    assert len(figs) == 1
    assert Path(figs[0]).exists()

    by_k = {d.k: d for d in data}
    assert set(by_k.keys()) == {2, 3}

    k2_points = {p.kind: p for p in by_k[2].key_points}
    assert k2_points["max"].rule_count == pytest.approx(3)
    assert k2_points["max"].n == pytest.approx(3)
    assert k2_points["knee-2Δ"].rule_count == pytest.approx(2)
    assert by_k[2].best_curve_metric[np.nanargmax(by_k[2].best_curve_metric)] == pytest.approx(3.4)

    k3_points = {p.kind: p for p in by_k[3].key_points}
    assert k3_points["knee-2Δ"].rule_count == pytest.approx(2)
    assert k3_points["max"].metric == pytest.approx(1.9)


def test_bucket_band_penalty_alignment():
    rows = [
        {
            "rule_count": 1,
            "sum_lambda_powers_raw": 10.0,
            "sum_lambda_powers_penalized": 2.0,
            "lower_bound_raw": 8.0,
            "upper_bound_raw": 12.0,
            "penalty_factor": 5.0,
        },
        {
            "rule_count": 2,
            "sum_lambda_powers_raw": 6.0,
            "sum_lambda_powers_penalized": 3.0,
            "lower_bound": 2.0,
            "upper_bound": 4.0,
            "penalty_factor": 2.0,
        },
    ]

    xs_pen, est_pen, lo_pen, hi_pen = viz._bucket_best_and_band(rows, use_logy=False, use_penalty=True)
    xs_raw, est_raw, lo_raw, hi_raw = viz._bucket_best_and_band(rows, use_logy=False, use_penalty=False)

    assert est_pen.tolist() == pytest.approx([2.0, 3.0])
    assert lo_pen.tolist() == pytest.approx([8.0 / 5.0, 2.0])
    assert hi_pen.tolist() == pytest.approx([12.0 / 5.0, 4.0])

    assert est_raw.tolist() == pytest.approx([10.0, 6.0])
    assert lo_raw.tolist() == pytest.approx([8.0, 2.0])
    assert hi_raw.tolist() == pytest.approx([12.0, 4.0])
