# -*- coding: utf-8 -*-
"""
rules/structures.py
结构原型：生成器 + 识别器 + 扫描与绘图

提供接口（与 rd_cli.py 对齐）：
- scan_archetypes(n,k, types, top_m, out_csv_dir, seed) -> dict
    生成每类原型的一组候选规则，批量评估（λ1、sum_lambda_powers、上下界等），
    输出若干 CSV（每类一个），返回 {type: [csv_paths...]}。
- plot_archetypes(csv_paths, out_dir, style) -> List[str]
    将多类原型进行对比可视化：|R|-trace 散点 + 箱线/小提琴（可选）+ 结构命中特征柱状。

依赖：
- rules.eval: rule_from_bits, bits_from_rule, canonical_bits,
              evaluate_rules_batch, RowsCacheLRU
- rules.viz: apply_style
"""

from __future__ import annotations
import csv, os, math, random
from typing import Dict, List, Tuple, Optional, Set, Iterable

import numpy as np
import matplotlib.pyplot as plt

from .eval import (
    rule_from_bits, bits_from_rule, canonical_bits,
    evaluate_rules_batch, RowsCacheLRU,
)
from .viz import apply_style

__all__ = [
    "proto_selfloop_star", "proto_cycle", "proto_bipartite_with_chords",
    "short_cycle_counts", "is_near_bipartite_with_chord", "recognize_archetypes",
    "scan_archetypes", "plot_archetypes",
]

# ---------------- 生成器 ----------------
def proto_selfloop_star(k: int, center: int = 0, spokes: Optional[List[int]] = None,
                        allow_center_self=True, allow_spokes_self=False) -> np.ndarray:
    if spokes is None:
        spokes = [i for i in range(1, k)]
    R = np.zeros((k, k), dtype=bool)
    for s in spokes:
        R[center, s] = True; R[s, center] = True
    if allow_center_self: R[center, center] = True
    if allow_spokes_self:
        for s in spokes: R[s, s] = True
    return canonical_bits(bits_from_rule(R), k)

def proto_cycle(k: int, cycle_nodes: Optional[List[int]] = None, allow_self=False) -> np.ndarray:
    if cycle_nodes is None:
        cycle_nodes = list(range(k))
    R = np.zeros((k, k), dtype=bool)
    m = len(cycle_nodes)
    for i in range(m):
        u = cycle_nodes[i]; v = cycle_nodes[(i+1) % m]
        R[u, v] = True; R[v, u] = True
    if allow_self:
        for i in cycle_nodes: R[i, i] = True
    return canonical_bits(bits_from_rule(R), k)

def proto_bipartite_with_chords(k: int, A: Optional[Set[int]] = None,
                                num_chords: int = 1, allow_self=False) -> np.ndarray:
    if A is None:
        A = set(range(k//2))
    B = set(range(k)) - A
    A = list(sorted(A)); B = list(sorted(B))
    R = np.zeros((k, k), dtype=bool)
    for a in A:
        for b in B:
            R[a, b] = True; R[b, a] = True
    for i in range(min(num_chords, max(0, len(A)-1))):
        u, v = A[i], A[(i+1) % len(A)]
        R[u, v] = True; R[v, u] = True
    for i in range(min(num_chords, max(0, len(B)-1))):
        u, v = B[i], B[(i+1) % len(B)]
        R[u, v] = True; R[v, u] = True
    if allow_self:
        np.fill_diagonal(R, True)
    return canonical_bits(bits_from_rule(R), k)

# ---------------- 识别器（特征抽取） ----------------
def short_cycle_counts(R: np.ndarray, max_len: int = 5) -> Dict[int, int]:
    k = R.shape[0]
    G = R.copy()
    np.fill_diagonal(G, False)
    res = {1: int(np.trace(R)), 2: 0}
    tri = 0
    for i in range(k):
        for j in range(i+1, k):
            if G[i, j]:
                for l in range(j+1, k):
                    if G[i, l] and G[j, l]:
                        tri += 1
    res[3] = tri
    quad = 0
    for a in range(k):
        for b in range(a+1, k):
            if not G[a, b]: continue
            for c in range(b+1, k):
                if not G[b, c]: continue
                for d in range(c+1, k):
                    if G[c, d] and G[d, a] and not (a == c or b == d):
                        quad += 1
    res[4] = quad
    pent = 0
    for a in range(k):
        for b in range(a+1, k):
            if not G[a, b]: continue
            for c in range(b+1, k):
                if not G[b, c]: continue
                for d in range(c+1, k):
                    if not G[c, d]: continue
                    for e in range(d+1, k):
                        if G[d, e] and G[e, a]:
                            pent += 1
    res[5] = pent
    return res

def is_near_bipartite_with_chord(R: np.ndarray, tol_edges: int = 2) -> bool:
    sc = short_cycle_counts(R, max_len=5)
    odd_cycles = sc.get(3, 0) + sc.get(5, 0)
    return odd_cycles <= tol_edges

def recognize_archetypes(bits_or_R: np.ndarray) -> Dict[str, bool]:
    if bits_or_R.ndim == 1:
        k = int((np.sqrt(8*len(bits_or_R) + 1) - 1) // 2)  # 粗解上三角长度推回 k
        R = rule_from_bits(k, bits_or_R)
    else:
        R = bits_or_R
    k = R.shape[0]
    res: Dict[str, bool] = {}
    deg = R.sum(axis=1)
    res["star_core"] = bool(np.max(deg) >= max(k-1, 2))
    sc = short_cycle_counts(R)
    res["has_tri"] = sc.get(3, 0) > 0
    res["has_quad"] = sc.get(4, 0) > 0
    res["has_pent"] = sc.get(5, 0) > 0
    res["near_bipartite_chord"] = is_near_bipartite_with_chord(R)
    res["selfloop_rich"] = (np.trace(R) >= max(1, k//2))
    return res

# ---------------- 扫描与可视化 ----------------
def _make_proto_pool(k: int, types: List[str], top_m: int, seed: int) -> List[np.ndarray]:
    random.seed(seed)
    cand: List[np.ndarray] = []
    for t in types:
        t = t.strip().lower()
        if t == "star":
            # 中心轮换、辐条子集
            centers = list(range(min(k, max(1, top_m))))
            for c in centers:
                cand.append(proto_selfloop_star(k, center=c, allow_center_self=True, allow_spokes_self=False))
        elif t == "cycle":
            # 允许自环与否两种
            cand.append(proto_cycle(k, allow_self=False))
            cand.append(proto_cycle(k, allow_self=True))
        elif t in ("bip", "bipartite", "near_bip"):
            for chords in range(1, max(2, min(4, top_m))):
                cand.append(proto_bipartite_with_chords(k, A=None, num_chords=chords, allow_self=False))
                cand.append(proto_bipartite_with_chords(k, A=None, num_chords=chords, allow_self=True))
        elif t in ("shortloop", "loop"):
            # 全自环/半自环
            R = np.zeros((k, k), dtype=bool); np.fill_diagonal(R, True)
            cand.append(canonical_bits(bits_from_rule(R), k))
            R2 = R.copy()
            for i in range(0, k, 2):
                R2[i, :] = False; R2[:, i] = False; R2[i, i] = True
            cand.append(canonical_bits(bits_from_rule(R2), k))
        else:
            # 未知类型：跳过
            pass
    # 去重
    uniq: List[np.ndarray] = []
    seen = set()
    for b in cand:
        key = b.tobytes()
        if key not in seen:
            seen.add(key)
            uniq.append(b)
    return uniq[: max(1, top_m * len(types))]

def scan_archetypes(n: int, k: int, types: List[str], top_m: int = 8,
                    out_csv_dir: str = "./results/archetypes", seed: int = 0) -> Dict[str, List[str]]:
    os.makedirs(out_csv_dir, exist_ok=True)
    bits_pool = _make_proto_pool(k, types, top_m, seed)
    # 分类型桶
    type_of = []
    for b in bits_pool:
        R = rule_from_bits(k, b)
        feat = recognize_archetypes(R)
        if feat["star_core"]:
            type_of.append("star")
        elif feat["near_bipartite_chord"]:
            type_of.append("bip")
        elif feat["has_tri"] or feat["has_pent"]:
            type_of.append("cycle")
        else:
            type_of.append("other")

    # 评估（批量）
    rows_lru = RowsCacheLRU(capacity=128)
    outs = evaluate_rules_batch(
        n=n, k=k, bits_list=bits_pool,
        device="cuda" if hasattr(np, "cuda") else "cpu",
        use_lanczos=True, r_vals=3, power_iters=60,
        trace_mode="hutchpp", hutch_s=24,
        lru_rows=rows_lru, max_streams=2,
    )

    # 写 CSV（每类一个）
    buckets: Dict[str, List[Tuple[np.ndarray, Dict]]] = {}
    for b, t, fit in zip(bits_pool, type_of, outs):
        buckets.setdefault(t, []).append((b, fit))

    csv_map: Dict[str, List[str]] = {}
    for t, items in buckets.items():
        path = os.path.join(out_csv_dir, f"arche_{t}_n{n}_k{k}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n", "k", "type", "rule_bits", "rule_count", "rows_m",
                        "lambda_max", "sum_lambda_powers", "lower_bound", "upper_bound",
                        "upper_bound_raw_gersh", "upper_bound_raw_maxdeg"])
            for b, fit in items:
                w.writerow([
                    n, k, t, "".join(map(str, b.tolist())), int(fit.get("rule_count", 0)),
                    int(fit.get("rows_m", 0)),
                    f"{float(fit.get('lambda_max', 0.0)):.6e}",
                    f"{float(fit.get('sum_lambda_powers', 0.0)):.6e}",
                    f"{float(fit.get('lower_bound', 0.0)):.6e}",
                    f"{float(fit.get('upper_bound', 0.0)):.6e}",
                    f"{float(fit.get('upper_bound_raw_gersh', float('nan'))):.6e}",
                    f"{float(fit.get('upper_bound_raw_maxdeg', float('nan'))):.6e}",
                ])
        csv_map.setdefault(t, []).append(path)
    return csv_map

def _load_many(csv_paths: List[str]) -> List[dict]:
    rows: List[dict] = []
    for p in csv_paths:
        if not os.path.exists(p): continue
        with open(p, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows

def plot_archetypes(csv_paths: List[str], out_dir: str = "./out_fig", style: str = "default") -> List[str]:
    apply_style(style)
    os.makedirs(out_dir, exist_ok=True)
    rows = _load_many(csv_paths)
    if not rows:
        print("[archetypes] no rows to plot.")
        return []

    # 1) |R| vs sum_lambda_powers 散点（按 type 着色）
    types = sorted(set(r["type"] for r in rows))
    color_map = {t: i for i, t in enumerate(types)}
    plt.figure()
    for t in types:
        xs = [int(r["rule_count"]) for r in rows if r["type"] == t]
        ys = [float(r["sum_lambda_powers"]) for r in rows if r["type"] == t]
        if not xs: continue
        plt.plot(xs, ys, marker="o", linestyle="None", label=t)
    plt.yscale("log")
    plt.xlabel("|R|")
    plt.ylabel("trace(T^n) estimate")
    plt.title("Archetypes: |R| vs trace")
    plt.legend(frameon=False, ncol=min(3, len(types)))
    plt.tight_layout()
    p1 = os.path.join(out_dir, "archetypes_scatter.png"); plt.savefig(p1, dpi=170)

    # 2) 每类结构的指标分布（箱线）
    data = {}
    for t in types:
        data.setdefault(t, [])
        for r in rows:
            if r["type"] == t:
                try:
                    data[t].append(float(r["sum_lambda_powers"]))
                except Exception:
                    pass
    plt.figure()
    plt.boxplot([data[t] for t in types], labels=types, showfliers=False)
    plt.yscale("log")
    plt.ylabel("trace(T^n) estimate")
    plt.title("Archetypes: trace distribution by type")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "archetypes_box.png"); plt.savefig(p2, dpi=170)

    return [p1, p2]
