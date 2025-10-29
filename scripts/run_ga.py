# -*- coding: utf-8 -*-
# --- add project root to sys.path (robust for absolute execution) ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # .../rules-diversity
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------------------
import random, numpy as np, torch, argparse, os
from rules.config import GAConfig
from rules.ga import ga_search_with_batch
from rules.viz import plot_pareto_from_csv
from rules.logging_setup import setup_logging, get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=str, default="(6,3);(6,4)",
                        help="semicolon-separated list of (n,k)")
    parser.add_argument("--out-csv", type=str, default="./out_csv")
    parser.add_argument("--out-fig", type=str, default="./out_fig")
    parser.add_argument("--gens", type=int, default=10)
    parser.add_argument("--pop", type=int, default=24)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trace", type=str, default="hutchpp", choices=["hutchpp","hutch","lanczos_sum","lam_only"])
    args = parser.parse_args()

    setup_logging()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # parse nk list
    nk_list = []
    for token in args.nk.split(";"):
        token = token.strip()
        if token:
            n, k = token.strip("() ").split(",")
            nk_list.append((int(n), int(k)))

    csv_paths = []
    for (n,k) in nk_list:
        conf = GAConfig(pop_size=args.pop, generations=args.gens,
                        device=args.device, trace_mode=args.trace)
        logger.info(f"== Run GA: n={n}, k={k}, device={conf.device} ==")
        pareto, csv_front, csv_gen = ga_search_with_batch(n, k, conf, out_csv_dir=args.out_csv)
        csv_paths.append(csv_front)

    plot_pareto_from_csv(csv_paths, out_dir=args.out_fig, y_log=True)

if __name__ == "__main__":
    main()
