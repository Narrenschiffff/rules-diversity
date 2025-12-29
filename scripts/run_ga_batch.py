#!/usr/bin/env python3
"""
Batch runner for rd_cli.py GA across boundaries (torus/open), symmetry modes
(perm / perm+swap), and penalty modes (n_times_rule_count / n_times_rows_m),
with CUDA-first then CPU fallback when applicable. Results are written under
./results/<penalty_mode>/... by default (same layout as notebook example).

Usage (from repo root):
    python scripts/run_ga_batch.py

You can override ranges, boundaries, or output root via env vars:
    N_MIN=2 N_MAX=5 BOUNDARIES=torus,open OUT_ROOT=./results python scripts/run_ga_batch.py

Quick open-only sweep (smaller n/k, pure exact path):
    BOUNDARIES=open N_MAX=3 GA_GENS=8 GA_POP=64 python scripts/run_ga_batch.py

Notes:
- torus/open 均优先 CUDA，失败后回退 CPU；CPU 回退时 open 会关闭谱估计，其余场景保留谱估计以便大 n*k 时自动估计。
- 结果分别落到 <penalty_mode>/<boundary>_csv 与 <penalty_mode>/<boundary>_csv_swap。

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
# 默认同时跑两种惩罚模式，可通过环境变量 PENALTY_MODES 覆盖，逗号分隔
PENALTY_MODES = [
    m.strip()
    for m in os.environ.get("PENALTY_MODES", "n_times_rule_count,n_times_rows_m").split(",")
    if m.strip()
]


def run(cmd, *, penalty_mode: str) -> None:
    """Run a command (list form preferred) with penalty env injected."""
    print(cmd if isinstance(cmd, str) else " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    # 将惩罚模式传递给规则评估
    env["RULES_PENALTY_MODE"] = penalty_mode
    res = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, capture_output=True, env=env)
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


def ga_batch(n: int, k: int, sym: str, boundary: str, penalty_mode: str, out_root: Path) -> None:
    penalty_root = out_root / penalty_mode
    out_dir = penalty_root / (f"{boundary}_csv" if sym == "perm" else f"{boundary}_csv_swap")
    out_dir.mkdir(parents=True, exist_ok=True)

    # torus/open 均优先 CUDA；无 GPU 时回落 CPU。
    device = _pick_device("cuda")
    spectral_flags = ""
    # CPU + open 时禁用谱估计以求稳；其它场景保持谱估计以便大规模（n*k>20）自动转估计
    if (device == "cpu") and (boundary == "open"):
        spectral_flags = "--no-spectral --no-lanczos"

    # n*k>20 时倾向估计；未实现时 rd_cli 内部会继续精确
    exact_threshold = "nk<=20"
    # 使用列表避免 Windows 下反斜杠导致的转义/换行问题，路径一律传递 POSIX 风格。
    args = [
        "python",
        "scripts/rd_cli.py",
        "ga",
        "--n", str(n),
        "--k", str(k),
        "--generations", str(GA_GENS),
        "--pop-size", str(GA_POP),
        "--trace-mode", "hutchpp",
        "--r-vals", "4",
        "--hutch-s", "32",
    ]
    if spectral_flags:
        args.extend(spectral_flags.split())
    args.extend([
        "--sym", sym,
        "--boundary", boundary,
        "--device", device,
        "--out-csv", out_dir.as_posix(),
        "--exact-threshold", exact_threshold,
        "--refresh-cache",
    ])
    try:
        run(args, penalty_mode=penalty_mode)
    except RuntimeError as e:
        if device == "cuda":
            print(
                f"[WARN] CUDA failed for (n={n},k={k},sym={sym},boundary={boundary}), fallback CPU. {e}"
            )
            # CPU 回退：open 关闭谱估计，其余保持开启以允许估计路径
            cpu_args = [a if a != "cuda" else "cpu" for a in args]
            if (boundary == "open") and ("--no-spectral" not in cpu_args):
                cpu_args.extend(["--no-spectral", "--no-lanczos"])
            run(cpu_args, penalty_mode=penalty_mode)
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

    for penalty_mode in PENALTY_MODES:
        for boundary in boundaries:
            # perm
            for n in range(n_min, n_max + 1):
                for k in range(2, 2 * n + 1):
                    ga_batch(n, k, sym="perm", boundary=boundary, penalty_mode=penalty_mode, out_root=out_root)

            # perm+swap
            for n in range(n_min, n_max + 1):
                for k in range(2, 2 * n + 1):
                    ga_batch(n, k, sym="perm+swap", boundary=boundary, penalty_mode=penalty_mode, out_root=out_root)


if __name__ == "__main__":
    main()
