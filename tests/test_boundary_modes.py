from pathlib import Path
import sys

import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rules.stage1_exact import (
    count_patterns_backtrack,
    count_patterns_transfer_matrix,
    rule_path_graph,
    enumerate_ring_rows,
    enumerate_ring_rows_fast,
)
from rules.eval import TransferOp


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("k", [2, 3])
def test_open_boundary_counts_match_transfer(n, k):
    R = rule_path_graph(k, allow_self=True)
    backtrack = count_patterns_backtrack(n, k, R, boundary="open")
    transfer = count_patterns_transfer_matrix(n, k, R, boundary="open")
    assert backtrack == transfer


def test_fast_open_rows_match_naive():
    n, k = 3, 3
    R = rule_path_graph(k, allow_self=True)
    fast_rows = np.asarray(enumerate_ring_rows_fast(n, k, R, boundary="open"), dtype=int)
    slow_rows = np.asarray(enumerate_ring_rows(n, k, R, boundary="open"), dtype=int)
    fast_set = sorted(map(tuple, fast_rows.tolist()))
    slow_set = sorted(map(tuple, slow_rows.tolist()))
    assert fast_set == slow_set


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transfer_op_open_matches_exact(n):
    k = 2
    R = rule_path_graph(k, allow_self=True)
    rows = np.asarray(enumerate_ring_rows_fast(n, k, R, boundary="open"), dtype=int)
    op = TransferOp(rows, R, device="cpu", dtype=torch.float64, build_sparse=True)
    vec = torch.ones(rows.shape[0], dtype=torch.float64)
    for _ in range(max(0, n - 1)):
        vec = op.matvec(vec)
    total = float(vec.sum().item())
    expected = count_patterns_transfer_matrix(n, k, R, boundary="open")
    assert np.isclose(total, expected)
