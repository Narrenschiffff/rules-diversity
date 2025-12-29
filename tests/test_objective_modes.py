import math

import numpy as np
import pytest

from rules.eval import evaluate_rules_batch
from rules.ga import nondominated_sort


def _three_state_diag_bits() -> np.ndarray:
    """Helper to generate a disconnected rule with a non-trivial penalty factor."""
    # k=3 -> diag (3) + upper-tri (3)
    return np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)


def test_evaluate_records_raw_and_penalized_fields():
    bits = _three_state_diag_bits()
    reports = evaluate_rules_batch(
        n=2,
        k=3,
        bits_list=[bits],
        device="cpu",
        use_lanczos=False,
        enable_exact=True,
        use_penalty=False,
    )
    fit = reports[0]

    assert "sum_lambda_powers_raw" in fit and "sum_lambda_powers_penalized" in fit
    assert fit["penalty_factor"] == pytest.approx(2 * fit["rows_m"])
    assert fit["sum_lambda_powers"] == pytest.approx(fit["sum_lambda_powers_raw"])
    assert fit["sum_lambda_powers_penalized"] == pytest.approx(
        fit["sum_lambda_powers_raw"] / fit["penalty_factor"]
    )
    expected_penalized = math.log(max(fit["sum_lambda_powers_raw"], 1e-300)) / (
        2 * max(1, fit["rows_m"])
    )
    assert fit["objective_penalized"] == pytest.approx(expected_penalized)


def test_objective_mode_normalizes_by_rows_and_n():
    bits = _three_state_diag_bits()
    reports = evaluate_rules_batch(
        n=2,
        k=3,
        bits_list=[bits],
        device="cpu",
        use_lanczos=False,
        enable_exact=True,
        objective_mode="logZ_per_nr",
        use_penalty=False,
    )
    fit = reports[0]
    denom = 2 * max(1, fit["rows_m"])
    expected = math.log(max(fit["sum_lambda_powers_raw"], 1e-300))
    assert math.isclose(fit["objective_raw"], expected, rel_tol=1e-6, abs_tol=1e-9)
    expected_pen = expected / denom
    assert math.isclose(fit["objective_penalized"], expected_pen, rel_tol=1e-6, abs_tol=1e-9)


def test_penalty_modes_n_and_rule_count():
    bits = _three_state_diag_bits()
    fit_n = evaluate_rules_batch(
        n=2,
        k=3,
        bits_list=[bits],
        device="cpu",
        use_lanczos=False,
        enable_exact=True,
        penalty_mode="n",
        use_penalty=False,
    )[0]
    assert fit_n["penalty_factor"] == pytest.approx(2.0)
    assert fit_n["sum_lambda_powers_penalized"] == pytest.approx(
        fit_n["sum_lambda_powers_raw"] / fit_n["penalty_factor"]
    )

    fit_rc = evaluate_rules_batch(
        n=2,
        k=3,
        bits_list=[bits],
        device="cpu",
        use_lanczos=False,
        enable_exact=True,
        penalty_mode="n_times_rule_count",
        use_penalty=False,
    )[0]
    assert fit_rc["penalty_factor"] == pytest.approx(2.0 * fit_rc["rule_count"])
    assert fit_rc["sum_lambda_powers_penalized"] == pytest.approx(
        fit_rc["sum_lambda_powers_raw"] / fit_rc["penalty_factor"]
    )


def test_ga_sorting_respects_objective_choice():
    fits = [
        {"rule_count": 1, "objective_penalized": 0.5, "objective_raw": 1.0, "sum_lambda_powers": 0.5},
        {"rule_count": 1, "objective_penalized": 1.0, "objective_raw": 0.2, "sum_lambda_powers": 1.0},
    ]

    fronts_penalized = nondominated_sort(fits, obj_key="objective_penalized")
    assert fronts_penalized[0] == [1]

    fronts_raw = nondominated_sort(fits, obj_key="objective_raw")
    assert fronts_raw[0] == [0]
