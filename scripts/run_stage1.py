# -*- coding: utf-8 -*-
# --- add project root to sys.path (robust for absolute execution) ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # .../rules-diversity
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------------------
from rules.stage1_exact import make_rule_matrix, exact_trace_by_transfer
from rules.logging_setup import setup_logging, get_logger

def main():
    setup_logging()
    logger = get_logger(__name__)
    # 例：k=2，禁止同色相邻（完全二分）；n=4
    k, n = 2, 4
    pairs = [(0,1)]  # 0-1
    R = make_rule_matrix(k, pairs, allow_self_loops=False)
    exact = exact_trace_by_transfer(n, k, R)
    logger.info(f"[stage1] exact trace(T^n) = {exact} for n={n}, k={k}")

if __name__ == "__main__":
    main()
