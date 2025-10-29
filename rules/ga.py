# -*- coding: utf-8 -*-
"""
NSGA-II (simplified) for rule search with batch evaluation.

改进要点：
- 对 GAConfig 的所有可选字段做归一化（None/非法 -> 默认）。
- 变异/初始化阶段加入最小可行修复（确保至少存在一条边或自环）。
- 逐代 CSV 记录（含 exact 对照占位），日志统一走 logging。
- 与 rules.eval 批量评估对接，显式传入归一化后的参数。
"""

from __future__ import annotations
import csv
import logging
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from .eval import (
    canonical_bits,
    bits_from_rule,
    rule_from_bits,
    evaluate_rules_batch,
    RowsCacheLRU,
)

logger = logging.getLogger(__name__)
__all__ = ["GAConfig", "ga_search_with_batch"]


# -------------------------------
# 工具：安全取值归一化
# -------------------------------
def _int_or(v, default):
    try:
        iv = int(v)
        return iv if iv > 0 else default
    except Exception:
        return default

def _float_or(v, default):
    try:
        fv = float(v)
        return fv
    except Exception:
        return default

def _bool_or(v, default):
    if isinstance(v, bool):
        return v
    if v in (0, 1):
        return bool(v)
    return default

def _device_or(v):
    if isinstance(v, str) and v.lower() in ("cpu", "cuda"):
        return v.lower()
    return "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# 最小可行修复（避免 rows_m=0）
# -------------------------------
def _ensure_minimal_feasible(bits: np.ndarray, k: int) -> np.ndarray:
    R = rule_from_bits(k, bits)
    deg = R.sum(axis=1)
    if int(deg.max()) == 0:  # 无边且无自环
        if k >= 2:
            u, v = 0, 1
            R[u, v] = True
            R[v, u] = True
        else:
            R[0, 0] = True
        return canonical_bits(bits_from_rule(R), k)
    return bits


def mutate(bits: np.ndarray, p_mut: float, k: int) -> np.ndarray:
    m = bits.copy()
    for i in range(m.size):
        if random.random() < p_mut:
            m[i] ^= 1
    return _ensure_minimal_feasible(m, k)


def crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.size < 2:
        return a.copy(), b.copy()
    cut = random.randint(1, a.size - 1)
    c1 = np.concatenate([a[:cut], b[cut:]], axis=0)
    c2 = np.concatenate([b[:cut], a[cut:]], axis=0)
    return c1, c2


def init_population(k: int, pop_size: int, bias_sparse: bool = True) -> List[np.ndarray]:
    L = k + k * (k - 1) // 2
    pop = []
    for _ in range(pop_size):
        if bias_sparse:
            arr = np.zeros(L, dtype=np.uint8)
            # 自环稍微多给一点概率，避免完全不可行
            for i in range(k):
                arr[i] = 1 if random.random() < 0.45 else 0
            idx = k
            for i in range(k):
                for j in range(i + 1, k):
                    arr[idx] = 1 if random.random() < 0.25 else 0
                    idx += 1
        else:
            arr = (np.random.rand(L) < 0.5).astype(np.uint8)
        cand = canonical_bits(arr, k)
        cand = _ensure_minimal_feasible(cand, k)
        pop.append(cand)
    return pop


# ========================================================
# NSGA-II 工具
# ========================================================
def dominates(a, b):
    f1a, f2a = a["rule_count"], -a["sum_lambda_powers"]
    f1b, f2b = b["rule_count"], -b["sum_lambda_powers"]
    return (f1a <= f1b and f2a <= f2b) and (f1a < f1b or f2a < f2b)


def nondominated_sort(pop_fits: List[Dict]) -> List[List[int]]:
    N = len(pop_fits)
    S = [set() for _ in range(N)]
    n_dom = [0] * N
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(pop_fits[p], pop_fits[q]):
                S[p].add(q)
            elif dominates(pop_fits[q], pop_fits[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    fronts.pop()
    return fronts


def crowding_distance(front: List[int], pop_fits: List[Dict]) -> Dict[int, float]:
    if not front:
        return {}
    distances = {i: 0.0 for i in front}
    vals1 = [(i, pop_fits[i]["rule_count"]) for i in front]        # smaller better
    vals2 = [(i, pop_fits[i]["sum_lambda_powers"]) for i in front] # larger better
    for vals, reverse in [(vals1, False), (vals2, True)]:
        vals_sorted = sorted(vals, key=lambda x: x[1], reverse=reverse)
        distances[vals_sorted[0][0]] = float("inf")
        distances[vals_sorted[-1][0]] = float("inf")
        vmin = vals_sorted[-1][1]; vmax = vals_sorted[0][1]
        rng = vmax - vmin if vmax > vmin else 1.0
        for j in range(1, len(vals_sorted) - 1):
            prev_v = vals_sorted[j - 1][1]
            next_v = vals_sorted[j + 1][1]
            distances[vals_sorted[j][0]] += (next_v - prev_v) / rng
    return distances


class GAConfig:
    def __init__(
        self,
        pop_size=32,
        generations=12,
        p_mut=0.08,
        p_cx=0.85,
        elite_keep=6,
        device=None,
        use_lanczos=True,
        r_vals=3,
        power_iters=50,
        trace_mode="hutchpp",
        hutch_s=24,
        lru_rows_capacity=128,
        batch_streams=2,
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.p_mut = p_mut
        self.p_cx = p_cx
        self.elite_keep = elite_keep
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lanczos = use_lanczos
        self.r_vals = r_vals
        self.power_iters = power_iters
        self.trace_mode = trace_mode
        self.hutch_s = hutch_s
        self.lru_rows_capacity = lru_rows_capacity
        self.batch_streams = batch_streams


def _append_front_rows_csv(
    csv_path_front: str,
    tag: str, n: int, k: int,
    pop_bits: List[np.ndarray],
    fits: List[Dict],
    front0_idx: List[int],
) -> None:
    front0_set = set(front0_idx)
    with open(csv_path_front, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i, (bits, fit) in enumerate(zip(pop_bits, fits)):
            rule_count = int(fit.get("rule_count", 0))
            rows_m = int(fit.get("rows_m", 0))
            lam_max = float(fit.get("lambda_max", 0.0))
            sum_lp = float(fit.get("sum_lambda_powers", -1e6))
            active_k = int(fit.get("active_k", 0))
            lb = float(fit.get("lower_bound", 0.0))
            ub = float(fit.get("upper_bound", 0.0))
            lb_raw = float(fit.get("lower_bound_raw", 0.0))
            ub_raw = float(fit.get("upper_bound_raw", 0.0))
            exact_trace = fit.get("exact_trace", "")
            exact_rows_m = fit.get("exact_rows_m", "")

            writer.writerow([
                tag, n, k,
                "".join(map(str, bits.tolist())),
                rule_count, rows_m, f"{lam_max:.6e}",
                f"{sum_lp:.6e}",
                1 if i in front0_set else 0,
                active_k,
                f"{lb:.6e}", f"{ub:.6e}",
                f"{lb_raw:.6e}", f"{ub_raw:.6e}",
                ("" if exact_trace == "" else f"{float(exact_trace):.6e}"),
                ("" if exact_rows_m == "" else str(exact_rows_m)),
            ])


def ga_search_with_batch(n: int, k: int, ga_conf: GAConfig, out_csv_dir: str = "./out_csv", run_tag: str = None):
    os.makedirs(out_csv_dir, exist_ok=True)
    tag = run_tag or f"n{n}_k{k}_{int(time.time())}"
    csv_path_front = os.path.join(out_csv_dir, f"pareto_front_{tag}.csv")
    csv_path_gen = os.path.join(out_csv_dir, f"gen_summary_{tag}.csv")

    # 头
    with open(csv_path_front, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_tag", "n", "k", "rule_bits", "rule_count", "rows_m",
            "lambda_max", "sum_lambda_powers", "is_front0",
            "active_k", "lower_bound", "upper_bound", "lower_bound_raw", "upper_bound_raw",
            "exact_trace", "exact_rows_m",
        ])
    with open(csv_path_gen, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_tag", "n", "k", "generation", "front0_size", "best_sample_R", "best_sample_sumlam",
            "pop_size", "device", "trace_mode",
        ])

    # ===== 归一化 GAConfig =====
    device        = _device_or(getattr(ga_conf, "device", None))
    use_lanczos   = _bool_or(getattr(ga_conf, "use_lanczos", True), True)
    r_vals        = _int_or(getattr(ga_conf, "r_vals", 3), 3)
    power_iters   = _int_or(getattr(ga_conf, "power_iters", 50), 50)
    trace_mode    = getattr(ga_conf, "trace_mode", "hutchpp") or "hutchpp"
    hutch_s       = _int_or(getattr(ga_conf, "hutch_s", 24), 24)
    lru_capacity  = _int_or(getattr(ga_conf, "lru_rows_capacity", 128), 128)
    batch_streams = _int_or(getattr(ga_conf, "batch_streams", 2), 2)
    generations   = _int_or(getattr(ga_conf, "generations", 12), 12)
    elite_keep    = _int_or(getattr(ga_conf, "elite_keep", 6), 6)
    p_cx          = _float_or(getattr(ga_conf, "p_cx", 0.85), 0.85)
    p_mut         = _float_or(getattr(ga_conf, "p_mut", 0.08), 0.08)
    pop_size      = _int_or(getattr(ga_conf, "pop_size", 32), 32)

    logger.info(f"GA start | n={n}, k={k}, device={device}, pop={pop_size}, gens={generations}")

    pop = init_population(k, pop_size, bias_sparse=True)
    cache: Dict[bytes, Dict] = {}
    rows_lru = RowsCacheLRU(capacity=lru_capacity)

    def eval_batch(bits_batch: List[np.ndarray]):
        normed = [canonical_bits(b, k) for b in bits_batch]
        results = [None] * len(normed)
        miss_bits, miss_pos = [], []
        for i, bb in enumerate(normed):
            key = bb.tobytes()
            if key in cache:
                results[i] = cache[key]
            else:
                miss_bits.append(bb); miss_pos.append(i)

        if miss_bits:
            outs = evaluate_rules_batch(
                n, k, miss_bits,
                device=device,
                use_lanczos=use_lanczos,
                r_vals=r_vals,
                power_iters=power_iters,
                trace_mode=trace_mode,
                hutch_s=hutch_s,
                lru_rows=rows_lru,
                max_streams=batch_streams,
            )
            for pos, bb, fit in zip(miss_pos, miss_bits, outs):
                cache[bb.tobytes()] = fit
                results[pos] = fit
        return results, normed

    # ===== 代际循环 =====
    for gen in range(generations):
        fits, pop = eval_batch(pop)
        if not fits:
            logger.warning(f"Empty fitness array at generation {gen}.")
            break

        fronts = nondominated_sort(fits)
        front0 = fronts[0] if fronts else []
        distances = crowding_distance(front0, fits)

        best_points = [(fits[i]["rule_count"], fits[i]["sum_lambda_powers"]) for i in front0] if front0 else []
        best_points = sorted(best_points, key=lambda x: (x[0], -x[1]))
        best_R = best_points[0][0] if best_points else None
        best_sum = best_points[0][1] if best_points else None

        with open(csv_path_gen, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([tag, n, k, gen, len(front0), best_R, best_sum,
                             pop_size, device, trace_mode])

        if best_points:
            preview = ", ".join([f"(|R|={x[0]}, tr≈{x[1]:.3e})" for x in best_points[:5]])
        else:
            preview = "None"
        logger.info(f"[GEN {gen:02d}] front0={len(front0)} | top Pareto (|R|, trace proxy): {preview}")

        _append_front_rows_csv(csv_path_front, tag, n, k, pop_bits=pop, fits=fits, front0_idx=front0)

        # 选择（精英 + 锦标赛）
        new_pop: List[np.ndarray] = []
        if front0:
            elite_sorted = sorted(front0, key=lambda i: distances.get(i, 0.0), reverse=True)
            for i in elite_sorted[:elite_keep]:
                new_pop.append(pop[i].copy())

        layer_of = {}
        for rank, fr in enumerate(fronts):
            for idx in fr:
                layer_of[idx] = rank
        dists_map = {}
        for fr in fronts:
            d = crowding_distance(fr, fits)
            dists_map.update(d)

        def tournament_select():
            a, b = random.sample(range(len(pop)), k=2)
            la, lb = layer_of.get(a, 10**9), layer_of.get(b, 10**9)
            if la < lb:
                return pop[a]
            if lb < la:
                return pop[b]
            da, db = dists_map.get(a, 0.0), dists_map.get(b, 0.0)
            return pop[a] if da >= db else pop[b]

        while len(new_pop) < pop_size:
            if random.random() < p_cx and len(new_pop) + 1 < pop_size:
                p1 = tournament_select()
                p2 = tournament_select()
                c1, c2 = crossover(p1, p2)
                c1 = canonical_bits(mutate(c1, p_mut, k), k)
                c2 = canonical_bits(mutate(c2, p_mut, k), k)
                new_pop.append(c1); new_pop.append(c2)
            else:
                p = tournament_select()
                c = canonical_bits(mutate(p, p_mut, k), k)
                new_pop.append(c)

        pop = new_pop[:pop_size]

    # ===== 最终帕累托 =====
    fits, pop = eval_batch(pop)
    fronts = nondominated_sort(fits)
    pareto_idx = fronts[0] if fronts else []
    pareto = [(pop[i], fits[i]) for i in pareto_idx]
    pareto_sorted = sorted(pareto, key=lambda x: (x[1]["rule_count"], -x[1]["sum_lambda_powers"]))

    logger.info("=== Final Pareto (top 10) ===")
    for i, (bits, fit) in enumerate(pareto_sorted[:10]):
        logger.info(f"[FINAL {i:02d}] |R|={fit['rule_count']:3d}, rows_m={fit['rows_m']:7d}, "
                    f"lambda_max≈{fit['lambda_max']:.4e}, trace≈{fit['sum_lambda_powers']:.4e}")

    return pareto_sorted, csv_path_front, csv_path_gen
