# -*- coding: utf-8 -*-
"""
Batch evaluation core:
- enumerate_ring_rows_fast
- TransferOp (blocked matvec over implicit compatibility)
- power_iteration / lanczos_top_r / Hutch / Hutch++
- 上下界：下界基于特征值，上界使用 Gershgorin / 最大度谱半径（_upper_bound_raw）
- 对外入口 evaluate_rules_batch
- 防御式兜底：iters / r / s / flags 的 None/非法值统一归一化
- 小规模精确计数（stage-1）自动回填 exact_trace / exact_rows_m
"""

from __future__ import annotations
import itertools
import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# =========================
# 通用取值归一化工具
# =========================
def _int_or(v, default: int) -> int:
    try:
        iv = int(v)
        return iv if iv > 0 else default
    except Exception:
        return default

def _float_or(v, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _bool_or(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v in (0, 1):
        return bool(v)
    return default


# =========================
# 规则矩阵 <-> 位串
# =========================
def make_rule_matrix(k: int, allowed_pairs: List[Tuple[int,int]], allow_self_loops=None) -> np.ndarray:
    R = np.zeros((k, k), dtype=bool)
    for (u, v) in allowed_pairs:
        R[u, v] = True
        R[v, u] = True
    if allow_self_loops is not None:
        for i in range(k):
            R[i, i] = bool(allow_self_loops)
    return R

def rule_from_bits(k: int, bits: np.ndarray) -> np.ndarray:
    L_diag = k
    L_upper = k*(k-1)//2
    assert bits.size == L_diag + L_upper
    R = np.zeros((k,k), dtype=bool)
    for i in range(k):
        R[i,i] = bool(bits[i])
    idx = k
    for i in range(k):
        for j in range(i+1, k):
            val = bool(bits[idx]); idx += 1
            R[i,j] = val
            R[j,i] = val
    return R

def bits_from_rule(R: np.ndarray) -> np.ndarray:
    k = R.shape[0]
    out = []
    for i in range(k):
        out.append(1 if R[i,i] else 0)
    for i in range(k):
        for j in range(i+1, k):
            out.append(1 if R[i,j] else 0)
    return np.array(out, dtype=np.uint8)


# =========================
# 对称压缩：规范化
# =========================
def canonical_bits(bits: np.ndarray, k: int) -> np.ndarray:
    R = rule_from_bits(k, bits)
    if k <= 8:
        best = None
        for perm in itertools.permutations(range(k)):
            P = R[np.ix_(perm, perm)]
            cand = bits_from_rule(P)
            if (best is None) or (tuple(cand.tolist()) < tuple(best.tolist())):
                best = cand
        return best
    # k>8：启发式
    deg = R.sum(axis=1).astype(int)
    selfloop = np.diag(R).astype(int)
    adj_str = ["".join('1' if x else '0' for x in row.tolist()) for row in R]
    order = sorted(range(k), key=lambda i: (deg[i], selfloop[i], adj_str[i]), reverse=True)
    P = R[np.ix_(order, order)]
    return bits_from_rule(P)


# =========================
# 行生成（环状合法行）
# =========================
def enumerate_ring_rows_fast(n: int, k: int, R: np.ndarray,
                             batch_limit: int = 200_000) -> np.ndarray:
    assert k <= 16, "当前实现默认 k ≤ 16（更大需改64bit位掩码实现）"
    allowed_next = np.zeros(k, dtype=np.uint32)
    for c in range(k):
        mask = 0
        row = R[c]
        for d in range(k):
            if row[d]:
                mask |= (1 << d)
        allowed_next[c] = np.uint32(mask)

    allowed_prev = np.zeros(k, dtype=np.uint32)
    for d in range(k):
        mask = 0
        col = R[:, d]
        for c in range(k):
            if col[c]:
                mask |= (1 << c)
        allowed_prev[d] = np.uint32(mask)

    out_chunks: List[np.ndarray] = []
    cur_batch: List[List[int]] = []

    def flush_batch():
        nonlocal cur_batch
        if not cur_batch:
            return
        arr = np.array(cur_batch, dtype=np.int16)
        out_chunks.append(arr)
        cur_batch = []

    path = np.empty(n, dtype=np.int16)

    def dfs_build(col: int, last_c: int, first_c: int):
        nonlocal cur_batch
        if col == n:
            if ((int(allowed_next[last_c]) >> first_c) & 1) != 0:
                cur_batch.append(path.copy().tolist())
                if len(cur_batch) >= batch_limit:
                    flush_batch()
            return
        mask = int(allowed_next[last_c])
        if col == n - 1:
            mask &= ((1 << first_c) | int(allowed_prev[first_c]))
        while mask:
            lsb = mask & -mask
            nxt = (lsb.bit_length() - 1)
            path[col] = nxt
            dfs_build(col + 1, nxt, first_c)
            mask ^= lsb

    for s in range(k):
        path[0] = s
        if n == 1:
            if ((int(allowed_next[s]) >> s) & 1) != 0:
                cur_batch.append([s])
            continue
        mask2 = int(allowed_next[s])
        while mask2:
            lsb = mask2 & -mask2
            c2 = (lsb.bit_length() - 1)
            path[1] = c2
            dfs_build(col=2, last_c=c2, first_c=s)
            mask2 ^= lsb

    flush_batch()
    if not out_chunks:
        return np.empty((0, n), dtype=np.int16)
    return np.vstack(out_chunks)


# =========================
# TransferOp
# =========================
class TransferOp:
    def __init__(self, rows_np: np.ndarray, R_np: np.ndarray,
                 device: str = "cuda", dtype: torch.dtype = torch.float64,
                 block_size_j: int = None, block_size_i: int = None):
        self.device = torch.device(device)
        self.dtype = dtype
        self.rows = torch.from_numpy(rows_np.astype(np.int64)).to(self.device)  # (m,n)
        self.m, self.n = self.rows.shape
        self.k = R_np.shape[0]
        self.R = torch.from_numpy(R_np.astype(np.bool_)).to(self.device)        # (k,k)

        if block_size_i is None:
            block_size_i = min(self.m, 2048)
        if block_size_j is None:
            block_size_j = 4096 if self.m >= 10000 else 2048
        if self.m >= 200000:
            block_size_j = 1024
            block_size_i = min(block_size_i, 1024)

        self.block_size_i = int(block_size_i)
        self.block_size_j = int(block_size_j)

    @torch.no_grad()
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape == (self.m,)
        rows = self.rows
        R = self.R
        m, n = self.m, self.n
        y = torch.zeros_like(x, dtype=self.dtype, device=self.device)

        for i0 in range(0, m, self.block_size_i):
            i1 = min(i0 + self.block_size_i, m)
            aL = rows[i0:i1, :].long()
            y_block = torch.zeros((i1 - i0,), dtype=self.dtype, device=self.device)

            for j0 in range(0, m, self.block_size_j):
                j1 = min(j0 + self.block_size_j, m)
                bL = rows[j0:j1, :].long()

                comp = torch.ones((i1 - i0, j1 - j0), dtype=torch.bool, device=self.device)
                for c in range(n):
                    ai = aL[:, c].unsqueeze(1)   # (mi,1)
                    bj = bL[:, c].unsqueeze(0)   # (1,mj)
                    comp &= R[ai, bj]            # (mi,mj)
                    if not comp.any():
                        break

                if comp.any():
                    y_block += comp.to(self.dtype) @ x[j0:j1].to(self.dtype)

            y[i0:i1] += y_block

        return y


# =========================
# 数值算法
# =========================
@torch.no_grad()
def power_iteration(op: TransferOp, iters: int = 60, tol: float = 1e-8,
                    verbose: bool = False, seed: int = 0) -> Tuple[float, torch.Tensor]:
    iters = _int_or(iters, 60)
    tol = _float_or(tol, 1e-8)
    torch.manual_seed(seed)
    m = op.m
    v = torch.randn(m, dtype=op.dtype, device=op.device)
    v = v / (v.norm() + 1e-30)
    last_lambda = None
    for t in range(iters):
        w = op.matvec(v)
        norm_w = w.norm()
        if norm_w.item() == 0.0:
            return 0.0, v
        v = w / norm_w
        lam = (v @ op.matvec(v)).item()
        if verbose and t % 5 == 0:
            logger.debug(f"[power] iter={t}  lambda≈{lam:.6e}")
        if last_lambda is not None and abs(lam - last_lambda) <= tol * max(1.0, abs(lam)):
            return lam, v
        last_lambda = lam
    return last_lambda if last_lambda is not None else 0.0, v

@torch.no_grad()
def lanczos_top_r(op: TransferOp, r: int = 4, iters: int = 80, seed: int = 0,
                  verbose: bool = False) -> List[float]:
    r = _int_or(r, 4)
    iters = _int_or(iters, max(2*r, 20))
    torch.manual_seed(seed)
    m = op.m
    q_prev = torch.zeros(m, dtype=op.dtype, device=op.device)
    q = torch.randn(m, dtype=op.dtype, device=op.device); q /= (q.norm() + 1e-30)

    alphas = []
    betas = [0.0]

    steps = min(r, iters)
    for j in range(steps):
        z = op.matvec(q)
        alpha = (q @ z).item()
        z = z - alpha * q - (betas[-1] * q_prev if j > 0 else 0.0)
        z = z - (z @ q) * q
        if j > 0: z = z - (z @ q_prev) * q_prev

        beta = z.norm().item()
        alphas.append(alpha)
        betas.append(beta)

        if verbose:
            logger.debug(f"[lanczos] j={j} alpha={alpha:.6e} beta={beta:.6e}")
        if beta <= 1e-32:
            break
        q_prev, q = q, z / (beta + 1e-30)

    tdim = len(alphas)
    T_tri = np.zeros((tdim, tdim), dtype=np.float64)
    for i in range(tdim):
        T_tri[i, i] = alphas[i]
        if i + 1 < tdim:
            T_tri[i, i + 1] = betas[i + 1]
            T_tri[i + 1, i] = betas[i + 1]
    evals = np.linalg.eigvalsh(T_tri)
    evals_sorted = np.sort(evals)[::-1]
    return evals_sorted[:r].tolist()

@torch.no_grad()
def apply_T_power(op, v: torch.Tensor, power: int) -> torch.Tensor:
    power = _int_or(power, 1)
    y = v
    for _ in range(power):
        y = op.matvec(y)
    return y

@torch.no_grad()
def estimate_trace_T_power_hutch(op, n_power: int, s: int = 32, seed: int = 0) -> float:
    n_power = _int_or(n_power, 1)
    s = _int_or(s, 32)
    torch.manual_seed(seed)
    m = op.m
    acc = 0.0
    for _ in range(s):
        v = torch.empty(m, dtype=op.dtype, device=op.device).uniform_(-1.0, 1.0).sign()
        Tv = apply_T_power(op, v, n_power)
        acc += (v @ Tv).item()
    return acc / float(s)

@torch.no_grad()
def estimate_trace_T_power_hutchpp(op, n_power: int, s: int = 32, seed: int = 0) -> float:
    n_power = _int_or(n_power, 1)
    s = _int_or(s, 32)
    torch.manual_seed(seed)
    m = op.m
    s1 = max(2, s // 2)
    s2 = s - s1

    Omega = torch.empty(m, s1, dtype=op.dtype, device=op.device).uniform_(-1.0, 1.0).sign()
    Y = torch.zeros(m, s1, dtype=op.dtype, device=op.device)
    for j in range(s1):
        Y[:, j] = apply_T_power(op, Omega[:, j], n_power)

    Q, _ = torch.linalg.qr(Y, mode='reduced')
    trace_lowrank = 0.0
    r_dim = Q.shape[1]
    for j in range(r_dim):
        qj = Q[:, j]
        Tq = apply_T_power(op, qj, n_power)
        trace_lowrank += (qj @ Tq).item()

    trace_res = 0.0
    for _ in range(max(1, s2)):
        z = torch.empty(m, dtype=op.dtype, device=op.device).uniform_(-1.0, 1.0).sign()
        p = Q.T @ z
        w = z - Q @ p
        Tw = apply_T_power(op, w, n_power)
        trace_res += (w @ Tw).item()
    if s2 > 0:
        trace_res /= float(s2)
    return trace_lowrank + trace_res


# =========================
# 图结构指标
# =========================
def _graph_degrees_and_components(R_np: np.ndarray) -> Tuple[np.ndarray, int]:
    k = R_np.shape[0]
    deg = R_np.sum(axis=1).astype(int)  # 自环计度
    adj = (R_np.copy()).astype(bool)
    for i in range(k):
        adj[i, i] = False
    visited = [False]*k
    comps = 0
    for s in range(k):
        if not visited[s]:
            comps += 1
            stack = [s]
            visited[s] = True
            while stack:
                u = stack.pop()
                for v in np.nonzero(adj[u])[0]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
    return deg, comps


# =========================
# LRU 缓存
# =========================
class RowsCacheLRU:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.od = OrderedDict()  # key: bytes(bits), val: np.ndarray rows
    def get(self, key: bytes):
        if key in self.od:
            val = self.od.pop(key)
            self.od[key] = val
            return val
        return None
    def put(self, key: bytes, val: np.ndarray):
        if key in self.od:
            self.od.pop(key)
        elif len(self.od) >= self.capacity:
            self.od.popitem(last=False)
        self.od[key] = val


# =========================
# 上界（raw）：Gershgorin / Δ 上界
# =========================
def _upper_bound_raw(m: int, R_np: np.ndarray, lam_pos: float, n_power: int) -> float:
    """
    不显式构造 T 的可计算上界：
      - Gershgorin 行和上界：ρ(T) ≤ max_i ∑_j |T_ij|（对 0-1 兼容矩阵即最大行和）；
        我们无法直接得到该行和，取保守近似 ≤ m。
      - 最大度谱半径上界：ρ(T) ≤ Δ（对 0-1 矩阵常见），此处取 Δ≈min(m, lam_pos 的正部分) 作为可用尺度。
    综合取：
        upper_raw = m * (min{ m, max(lam_pos, 1) })^n
    若未来实现了对最大行和的抽样近似，可把 delta 换成更紧的估计。
    """
    n_power = _int_or(n_power, 1)
    if m <= 0:
        return 0.0
    # lam_pos 仅用于给出一个可计算的谱半径尺度；至少为 1，至多为 m
    delta = max(1.0, min(float(m), float(lam_pos if lam_pos > 0 else 0.0)))
    return float(m) * (delta ** n_power)


# =========================
# 小规模精确计数（可选）
# =========================
def _maybe_exact_trace(n: int, k: int, R: np.ndarray, rows_m: int,
                       enable: bool = True,
                       n_max: int = 4, k_max: int = 3,
                       rows_m_max: int = 20000) -> Tuple[Optional[int], Optional[int]]:
    """
    条件触发 stage-1 精确计数：
      - n ≤ n_max, k ≤ k_max, rows_m ≤ rows_m_max
      - 存在 rules.stage1_exact，并提供 exact_trace_by_transfer(n,k,R)
    返回：(exact_trace, exact_rows_m)；若跳过或失败则 (None, None)。
    """
    if not enable:
        return None, None
    if not (n <= n_max and k <= k_max and rows_m <= rows_m_max):
        return None, None
    try:
        # 延迟导入，避免常规路径额外依赖
        from rules.stage1_exact import enumerate_ring_rows, build_row_compat_matrix
        rows = enumerate_ring_rows(n, k, R)  # List[List[int]]
        T = build_row_compat_matrix(rows, R) # 0/1 矩阵 (m,m)
        # 直接算 trace(T^n) （m 较小）
        M = np.array(T, dtype=object)
        for _ in range(n - 1):
            M = M @ T
        exact = int(np.trace(M))
        return exact, len(rows)
    except Exception as e:
        logger.debug(f"[exact] skip or failed: {e}")
        return None, None


# =========================
# 评估主入口
# =========================
@torch.no_grad()
def evaluate_rules_batch(n: int,
                         k: int,
                         bits_list: List[np.ndarray],
                         device: str = "cuda",
                         use_lanczos: bool = True,
                         r_vals: int = 3,
                         power_iters: int = 50,
                         trace_mode: str = "hutchpp",
                         hutch_s: int = 24,
                         lru_rows: RowsCacheLRU = None,
                         max_streams: int = 2) -> List[Dict]:
    """
    输出字段：
      rule_count, rows_m, lambda_max, sum_lambda_powers,
      active_k, lower_bound / upper_bound（惩罚后）, lower_bound_raw / upper_bound_raw,
      exact_trace / exact_rows_m（若触发小规模精确计数）
    """
    # 参数兜底
    r_vals      = _int_or(r_vals, 3)
    power_iters = _int_or(power_iters, 50)
    hutch_s     = _int_or(hutch_s, 24)
    max_streams = _int_or(max_streams, 2)
    use_lanczos = _bool_or(use_lanczos, True)

    PENALTY_ALPHA = 1.5  # 温和惩罚强度

    def _adaptive_samples(m: int, base: int) -> int:
        if m < 2_000:
            return max(12, base // 2)
        elif m > 50_000:
            return min(64, base * 2)
        return base

    # ---------- CPU 路径 ----------
    if (device == "cpu") or (not torch.cuda.is_available()):
        outs: List[Dict] = []
        for bits in bits_list:
            R = rule_from_bits(k, bits)
            deg, comps = _graph_degrees_and_components(R)
            rows = enumerate_ring_rows_fast(n, k, R)
            m = int(rows.shape[0])
            if m == 0:
                outs.append({
                    "rule_count": int(bits.sum()),
                    "lambda_max": 0.0,
                    "sum_lambda_powers": -1e300,
                    "rows_m": 0,
                    "active_k": 0,
                    "lower_bound": 0.0, "upper_bound": 0.0,
                    "lower_bound_raw": 0.0, "upper_bound_raw": 0.0,
                    "exact_trace": "", "exact_rows_m": "",
                })
                continue

            used = np.zeros(k, dtype=bool)
            used[np.unique(rows)] = True
            active_k = int(used.sum())

            s_adapt = _adaptive_samples(m, hutch_s)
            op = TransferOp(rows, R, device="cpu", dtype=torch.float64, block_size_j=2048)

            if use_lanczos:
                evals = lanczos_top_r(op, r=r_vals, iters=max(2*r_vals, 20), seed=0, verbose=False)
                lam_max = float(evals[0]) if len(evals) > 0 else 0.0
                lam_pos = max(0.0, lam_max)
                lb_raw = max(lam_pos ** n, sum((max(0.0, lam) ** n) for lam in evals) if len(evals)>0 else 0.0)
                if trace_mode == "lanczos_sum":
                    est = float(sum((max(0.0, lam) ** n) for lam in evals) if len(evals)>0 else 0.0)
                elif trace_mode == "hutch":
                    est = float(estimate_trace_T_power_hutch(op, n_power=n, s=s_adapt, seed=0))
                elif trace_mode == "hutchpp":
                    est = float(estimate_trace_T_power_hutchpp(op, n_power=n, s=s_adapt, seed=0))
                else:
                    est = float(lam_pos ** n)
            else:
                lam_max, _ = power_iteration(op, iters=power_iters, tol=1e-9, verbose=False, seed=0)
                lam_pos = max(0.0, lam_max)
                lb_raw = lam_pos ** n
                est = float(lam_pos ** n)

            ub_raw = _upper_bound_raw(m, R, lam_pos, n_power=n)

            penal = 1.0
            if comps > 1:
                penal *= (0.5 ** (comps - 1))
            if np.any(deg == 0):
                penal *= (active_k / k) ** PENALTY_ALPHA

            # 小规模精确计数（可选）
            exact_trace, exact_rows_m = _maybe_exact_trace(n, k, R, m)

            outs.append({
                "rule_count": int(bits.sum()),
                "lambda_max": lam_max,
                "sum_lambda_powers": est * penal,
                "rows_m": m,
                "active_k": active_k,
                "lower_bound": lb_raw * penal, "upper_bound": ub_raw * penal,
                "lower_bound_raw": lb_raw, "upper_bound_raw": ub_raw,
                "exact_trace": "" if exact_trace is None else exact_trace,
                "exact_rows_m": "" if exact_rows_m is None else exact_rows_m,
            })
        return outs

    # ---------- CUDA 路径 ----------
    streams = [torch.cuda.Stream() for _ in range(max(1, max_streams))]
    outputs: List[Dict] = [None] * len(bits_list)

    for idx, bits in enumerate(bits_list):
        key = bits.tobytes()
        R = rule_from_bits(k, bits)
        deg, comps = _graph_degrees_and_components(R)

        rows = None
        if lru_rows is not None:
            rows = lru_rows.get(key)
        if rows is None:
            rows = enumerate_ring_rows_fast(n, k, R)
            if (lru_rows is not None) and (rows.shape[0] > 0) and (rows.shape[0] <= 1_000_000):
                lru_rows.put(key, rows)

        m = int(rows.shape[0])
        if m == 0:
            outputs[idx] = {
                "rule_count": int(bits.sum()),
                "lambda_max": 0.0,
                "sum_lambda_powers": -1e300,
                "rows_m": 0,
                "active_k": 0,
                "lower_bound": 0.0, "upper_bound": 0.0,
                "lower_bound_raw": 0.0, "upper_bound_raw": 0.0,
                "exact_trace": "", "exact_rows_m": "",
            }
            continue

        used = np.zeros(k, dtype=bool)
        used[np.unique(rows)] = True
        active_k = int(used.sum())

        s_adapt = max(1, _int_or(hutch_s, 24))
        st = streams[idx % len(streams)]
        with torch.cuda.stream(st):
            op = TransferOp(rows, R, device=device, dtype=torch.float64, block_size_j=4096)

            if use_lanczos:
                evals = lanczos_top_r(op, r=_int_or(r_vals, 3), iters=max(2*_int_or(r_vals,3), 20), seed=idx, verbose=False)
                lam_max = float(evals[0]) if len(evals) > 0 else 0.0
                lam_pos = max(0.0, lam_max)
                lb_r = sum((max(0.0, lam) ** n) for lam in evals) if len(evals) > 0 else 0.0
                lb_raw = max(lam_pos ** n, lb_r)
                if trace_mode == "lanczos_sum":
                    est = float(lb_r)
                elif trace_mode == "hutch":
                    est = float(estimate_trace_T_power_hutch(op, n_power=n, s=s_adapt, seed=idx))
                elif trace_mode == "hutchpp":
                    est = float(estimate_trace_T_power_hutchpp(op, n_power=n, s=s_adapt, seed=idx))
                else:
                    est = float(lam_pos ** n)
            else:
                lam_max, _ = power_iteration(op, iters=_int_or(power_iters, 50), tol=1e-9, verbose=False, seed=idx)
                lam_pos = max(0.0, lam_max)
                lb_raw = lam_pos ** n
                est = float(lam_pos ** n)

            ub_raw = _upper_bound_raw(m, R, lam_pos, n_power=n)

            penal = 1.0
            if comps > 1:
                penal *= (0.5 ** (comps - 1))
            if np.any(deg == 0):
                penal *= (active_k / k) ** PENALTY_ALPHA

            exact_trace, exact_rows_m = _maybe_exact_trace(n, k, R, m)

            outputs[idx] = {
                "rule_count": int(bits.sum()),
                "lambda_max": lam_max,
                "sum_lambda_powers": est * penal,
                "rows_m": m,
                "active_k": active_k,
                "lower_bound": lb_raw * penal, "upper_bound": ub_raw * penal,
                "lower_bound_raw": lb_raw, "upper_bound_raw": ub_raw,
                "exact_trace": "" if exact_trace is None else exact_trace,
                "exact_rows_m": "" if exact_rows_m is None else exact_rows_m,
            }

    torch.cuda.synchronize()
    return outputs
