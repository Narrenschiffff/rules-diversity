#!/usr/bin/env python3
"""
Batch runner for rd_cli.py GA on torus boundary for both symmetry modes
(perm / perm+swap), with CUDA-first then CPU fallback. Results are written
under ./results by default (same layout as notebook example).

Usage (from repo root):
    python scripts/run_ga_batch.py

You can override ranges or output root via env vars:
    N_MIN=2 N_MAX=5 OUT_ROOT=./results python scripts/run_ga_batch.py

This script is intentionally simple: it just shells out to rd_cli.py, mirrors
the notebook logic, and retries on CPU if CUDA fails.
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path


def run(cmd: str) -> None:
    print(cmd)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env)
    print(res.stdout)
    if res.stderr:
        print("[stderr]", res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"cmd failed ({res.returncode})")


def ga_torus(n: int, k: int, sym: str, out_root: Path) -> None:
    out_dir = out_root / ("torus_csv" if sym == "perm" else "torus_csv_swap")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = textwrap.dedent(
        f"""
        python scripts/rd_cli.py ga --n {n} --k {k} --generations 16 --pop-size 96 \
          --trace-mode hutchpp --r-vals 4 --hutch-s 32 \
          --sym {sym} --boundary torus --device cuda \
          --out-csv {out_dir.as_posix()}
        """
    ).strip()
    try:
        run(base)
    except RuntimeError as e:
        print(f"[WARN] CUDA failed for (n={n},k={k},sym={sym}), fallback CPU. {e}")
        run(base.replace("--device cuda", "--device cpu"))


def main() -> None:
    out_root = Path(os.environ.get("OUT_ROOT", "./results"))
    out_root.mkdir(parents=True, exist_ok=True)

    n_min = int(os.environ.get("N_MIN", 2))
    n_max = int(os.environ.get("N_MAX", 5))

    # perm
    for n in range(n_min, n_max + 1):
        for k in range(2, 2 * n + 1):
            ga_torus(n, k, sym="perm", out_root=out_root)

    # perm+swap
    for n in range(n_min, n_max + 1):
        for k in range(2, 2 * n + 1):
            ga_torus(n, k, sym="perm+swap", out_root=out_root)


if __name__ == "__main__":
    main()
