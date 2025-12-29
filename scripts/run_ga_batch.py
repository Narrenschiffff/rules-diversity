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
- torus/open 均优先 CUDA，失败后回退 CPU；CPU 回退时会自动关闭谱估计（保持稳健）。
- 结果分别落到 torus_csv/torus_csv_swap 与 open_csv/open_csv_swap。

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


def _pick_device(prefer: str = "cuda") -> str:
    device = prefer
    if device == "cuda":
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():  # pragma: no cover - runtime check
                device = "cpu"
        except Exception:  # pragma: no cover - defensive
            device = "cpu"
    return device


def ga_batch(n: int, k: int, sym: str, boundary: str, out_root: Path) -> None:
    out_dir = out_root / (f"{boundary}_csv" if sym == "perm" else f"{boundary}_csv_swap")
    out_dir.mkdir(parents=True, exist_ok=True)

    # torus/open 均优先 CUDA；无 GPU 时回落 CPU，CPU 模式下自动禁用谱估计以求稳。
    device = _pick_device("cuda")
    spectral_flags = "" if device == "cuda" else "--no-spectral --no-lanczos"
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
            # CPU 回退：保持与此前 open 边界一致的精确/稳健策略，关闭谱估计。
            cpu_cmd = base.replace("--device cuda", "--device cpu")
            if "--no-spectral" not in cpu_cmd:
                cpu_cmd += " --no-spectral --no-lanczos"
            run(cpu_cmd)
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
