#!/usr/bin/env python3
"""
Batch runner for rd_cli.py GA across boundaries (torus/open) and symmetry modes
(perm / perm+swap), with CUDA-first then CPU fallback when applicable. Results
are written under ./results by default (same layout as notebook example).

Usage (from repo root):
    python scripts/run_ga_batch.py

You can override ranges, boundaries, or output root via env vars:
    N_MIN=2 N_MAX=5 BOUNDARIES=torus,open OUT_ROOT=./results python scripts/run_ga_batch.py

Quick open-only sweep (smaller n/k, pure exact path):
    BOUNDARIES=open N_MAX=3 GA_GENS=8 GA_POP=64 python scripts/run_ga_batch.py

Notes:
- open 边界仅走精确计数（自动禁用谱估计），默认在 CPU 上运行，结果落到 open_csv/open_csv_swap。
- torus 边界默认优先 CUDA，失败后回退 CPU，结果落到 torus_csv/torus_csv_swap。

This script is intentionally simple: it just shells out to rd_cli.py, mirrors
the notebook logic, and retries on CPU if CUDA fails.
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path
import sys

# Ensure in-repo execution works without PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rules.bootstrap import ensure_repo_on_path


ROOT = ensure_repo_on_path()
GA_GENS = int(os.environ.get("GA_GENS", 16))
GA_POP = int(os.environ.get("GA_POP", 96))


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


def ga_batch(n: int, k: int, sym: str, boundary: str, out_root: Path) -> None:
    out_dir = out_root / (f"{boundary}_csv" if sym == "perm" else f"{boundary}_csv_swap")
    out_dir.mkdir(parents=True, exist_ok=True)

    # open 边界仅走精确路径，默认使用 CPU；torus 仍优先 CUDA。
    device = "cuda" if boundary == "torus" else "cpu"
    spectral_flags = "" if boundary == "torus" else "--no-spectral --no-lanczos"
    base = textwrap.dedent(
        f"""
        python scripts/rd_cli.py ga --n {n} --k {k} --generations {GA_GENS} --pop-size {GA_POP} \
          --trace-mode hutchpp --r-vals 4 --hutch-s 32 {spectral_flags}\
          --sym {sym} --boundary {boundary} --device {device} \
          --out-csv {out_dir.as_posix()}
        """
    ).strip()
    try:
        run(base)
    except RuntimeError as e:
        if device == "cuda":
            print(
                f"[WARN] CUDA failed for (n={n},k={k},sym={sym},boundary={boundary}), fallback CPU. {e}"
            )
            run(base.replace("--device cuda", "--device cpu"))
        else:
            raise


def main() -> None:
    out_root = Path(os.environ.get("OUT_ROOT", "./results"))
    out_root.mkdir(parents=True, exist_ok=True)

    n_min = int(os.environ.get("N_MIN", 2))
    n_max = int(os.environ.get("N_MAX", 5))
    boundaries = [b.strip() for b in os.environ.get("BOUNDARIES", "torus,open").split(",") if b.strip()]
    if not boundaries:
        boundaries = ["torus"]

    for boundary in boundaries:
        # perm
        for n in range(n_min, n_max + 1):
            for k in range(2, 2 * n + 1):
                ga_batch(n, k, sym="perm", boundary=boundary, out_root=out_root)

        # perm+swap
        for n in range(n_min, n_max + 1):
            for k in range(2, 2 * n + 1):
                ga_batch(n, k, sym="perm+swap", boundary=boundary, out_root=out_root)


if __name__ == "__main__":
    main()
