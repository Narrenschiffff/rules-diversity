# -*- coding: utf-8 -*-
"""
Visualization utilities:
- Pareto scatter (front-0)
- Growth curve with confidence bands (lower/upper)
- Knee detection & annotation:
  * 'second_diff' — maximize discrete second difference of log-curve
  * 'lcurve'      — geometric knee by distance to chord (L-curve style)
"""

from __future__ import annotations
import csv
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
# Knee detection helpers
# ------------------------------
def _detect_knee_second_diff(xs: List[int], ys: List[float]) -> int:
    """
    On log y, discrete second difference peak.
    Return index in xs/ys of knee; -1 if not found.
    """
    if len(xs) < 3:
        return -1
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    ylog = np.log(np.maximum(y, 1e-300))
    # uniform spacing not guaranteed; use central finite diff with x spacing
    d1 = np.gradient(ylog, x)
    d2 = np.gradient(d1, x)
    idx = int(np.argmax(np.maximum(d2, 0.0)))
    return idx


def _detect_knee_lcurve(xs: List[int], ys: List[float]) -> int:
    """
    L-curve style: knee is point farthest from straight line between endpoints in log-log space.
    """
    if len(xs) < 3:
        return -1
    X = np.log(np.maximum(np.array(xs, dtype=float), 1e-12))
    Y = np.log(np.maximum(np.array(ys, dtype=float), 1e-300))
    x1, y1 = X[0], Y[0]
    x2, y2 = X[-1], Y[-1]
    # distance from point to line
    denom = math.hypot(x2 - x1, y2 - y1) + 1e-18
    dists = np.abs((y2 - y1) * X - (x2 - x1) * Y + x2 * y1 - y2 * x1) / denom
    idx = int(np.argmax(dists))
    if idx == 0 or idx == len(xs) - 1:
        # avoid endpoints; choose next best
        order = np.argsort(dists)[::-1]
        for j in order:
            if 0 < j < len(xs) - 1:
                return int(j)
        return -1
    return idx


def detect_knee(xs: List[int], ys: List[float], method: str = "second_diff") -> int:
    if method == "lcurve":
        return _detect_knee_lcurve(xs, ys)
    return _detect_knee_second_diff(xs, ys)


def _annotate_knee(ax, xs: List[int], ys: List[float], idx: int, label: str):
    if idx <= 0 or idx >= len(xs) - 1:
        return
    ax.scatter([xs[idx]], [ys[idx]], marker="*", s=160, zorder=5)
    ax.annotate(
        f"{label}\n|R|={xs[idx]}",
        xy=(xs[idx], ys[idx]),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=9,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        arrowprops=dict(arrowstyle="-", alpha=0.4),
    )


# ------------------------------
# IO helper
# ------------------------------
def _load_runs(csv_path_fronts: List[str]) -> Dict[str, List[dict]]:
    runs: Dict[str, List[dict]] = {}
    for p in csv_path_fronts:
        with open(p, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                runs.setdefault(row["run_tag"], []).append(row)
    return runs


# ------------------------------
# Plots
# ------------------------------
def plot_pareto_from_csv(csv_path_fronts: List[str], out_dir: str = "./out_fig", y_log: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    runs = _load_runs(csv_path_fronts)

    # 1) Pareto scatter (front-0 only)
    plt.figure()
    for tag, rows in runs.items():
        xs, ys = [], []
        for r in rows:
            if r.get("is_front0", "0") == "1":
                xs.append(int(r["rule_count"]))
                ys.append(float(r["sum_lambda_powers"]))
        if xs:
            plt.scatter(xs, ys, label=tag, alpha=0.85, s=28)
    plt.xlabel("|R| (number of rules)")
    plt.ylabel(r"$\mathrm{trace}(T^n)$ estimate")
    if y_log:
        plt.yscale("log")
    plt.title("Pareto Front (Front-0)")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig1 = os.path.join(out_dir, f"pareto_scatter{'_log' if y_log else ''}.png")
    plt.savefig(fig1, dpi=160)

    # 2) Growth curve + band + knees
    plt.figure()
    for tag, rows in runs.items():
        bucket_best = {}  # |R| -> (est, lo, hi)
        for r in rows:
            if r.get("is_front0", "0") != "1":
                continue
            try:
                Rcnt = int(r["rule_count"])
                est = float(r["sum_lambda_powers"])
                lo = float(r.get("lower_bound", 0.0))
                hi = float(r.get("upper_bound", 0.0))
            except Exception:
                continue
            if (Rcnt not in bucket_best) or (est > bucket_best[Rcnt][0]):
                bucket_best[Rcnt] = (est, lo, hi)

        if bucket_best:
            xs = sorted(bucket_best.keys())
            ests = np.array([bucket_best[x][0] for x in xs], dtype=float)
            los = np.array([bucket_best[x][1] for x in xs], dtype=float)
            his = np.array([bucket_best[x][2] for x in xs], dtype=float)

            plt.plot(xs, ests, marker="o", label=tag)
            valid = np.isfinite(los) & np.isfinite(his) & (his >= los)
            if valid.any():
                plt.fill_between(np.array(xs)[valid], los[valid], his[valid], alpha=0.18)

            # knees
            idx_knee_sd = detect_knee(xs, ests.tolist(), method="second_diff")
            _annotate_knee(plt.gca(), xs, ests.tolist(), idx_knee_sd, f"{tag} knee (2nd-diff)")
            idx_knee_lc = detect_knee(xs, ests.tolist(), method="lcurve")
            _annotate_knee(plt.gca(), xs, ests.tolist(), idx_knee_lc, f"{tag} knee (L-curve)")

    plt.xlabel("|R| (number of rules)")
    plt.ylabel("Best frontier estimate per |R| (with bounds)")
    if y_log:
        plt.yscale("log")
    plt.title("Growth Curve (with confidence bands & knees)")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig2 = os.path.join(out_dir, f"growth_curve{'_log' if y_log else ''}_with_band_knees.png")
    plt.savefig(fig2, dpi=160)

    print(f"[FIG] saved:\n  {fig1}\n  {fig2}")
    return fig1, fig2
