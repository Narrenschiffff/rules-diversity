from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rules.stage1_exact import (
    count_patterns_backtrack,
    count_patterns_transfer_matrix,
    rule_path_graph,
)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("k", [2, 3])
def test_open_boundary_counts_match_transfer(n, k):
    R = rule_path_graph(k, allow_self=True)
    backtrack = count_patterns_backtrack(n, k, R, boundary="open")
    transfer = count_patterns_transfer_matrix(n, k, R, boundary="open")
    assert backtrack == transfer
