# -*- coding: utf-8 -*-
"""
rules/eval.py
评估与核心算子：
- 规则矩阵编解码（上三角含对角）
- 对称压缩 canonical_bits（规则层面的状态置换对称）
- 行生成器（环状合法行） enumerate_ring_rows_fast
- TransferOp: 不显式构造 T 的 matvec
- 幂迭代 / Lanczos / Hutch/Hutch++
- 结构性指标（度、连通分量）
- 批量评估 evaluate_rules_batch：输出 λ1/λ2、谱间隙、上下界（含 Gershgorin / 最大度谱半径上界）
- 小规模 (n<=4,k<=3) 自动调用 stage-1 精确计数并落盘 exact_Z（可选几何去重接口占位）

依赖：numpy, torch, matplotlib（仅个别函数用, 不在此文件画图）
"""

from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import math
import itertools
import logging
import numpy as np
import torch

from . import config
from .cache import (
    ensure_eval_cache_dir,
    make_eval_cache_key,
    load_eval_cache,
    dump_eval_cache,
)
from .stage1_exact import enumerate_ring_rows, count_patterns_transfer_matrix

# ------------------------------
# 通用小工具
# ------------------------------

def _bool_or(a: Optional[bool], b: bool) -> bool:
    return b if a is None else (a or b)

# ------------------------------
# 规则矩阵编解码
# ------------------------------

def make_rule_matrix(k: int, allowed_pairs: List[Tuple[int,int]], allow_self_loops: Optional[bool]=None) -> np.ndarray:
    R = np.zeros((k,k), dtype=bool)
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
    # diag
    for i in range(k):
        R[i,i] = bool(bits[i])
    # upper
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

# ------------------------------
# 规则层面的状态置换对称：canonical_bits
# ------------------------------

def canonical_bits(bits: np.ndarray, k: int) -> np.ndarray:
    """小 k 完全置换搜索；大 k 启发式。"""
    R = rule_from_bits(k, bits)
    if k <= 8:
        best = None
        for perm in itertools.permutations(range(k)):
            P = R[np.ix_(perm, perm)]
            cand = bits_from_rule(P)
            if (best is None) or (tuple(cand.tolist()) < tuple(best.tolist())):
                best = cand
        return best
    # 启发式：按 (deg, selfloop, 邻接行字典序) 排序
    deg = R.sum(axis=1).astype(int)
    selfloop = np.diag(R).astype(int)
    adj_str = ["".join('1' if x else '0' for x in row.tolist()) for row in R]
    order = sorted(range(k), key=lambda i: (deg[i], selfloop[i], adj_str[i]), reverse=True)
    P = R[np.ix_(order, order)]
    return bits_from_rule(P)


def _exchange_classes(R: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    基于邻接行/列完全一致的“可交换”等价类。返回：
      - raw_to_class: 原始状态 -> 压缩后的类索引
      - class_sizes: 每个等价类的规模
    """
    k = R.shape[0]
    signatures = [tuple(int(x) for x in R[i].tolist()) for i in range(k)]

    sig_to_class: Dict[Tuple[int, ...], int] = {}
    class_members: List[List[int]] = []
    for i, sig in enumerate(signatures):
        if sig in sig_to_class:
            class_members[sig_to_class[sig]].append(i)
        else:
            cid = len(class_members)
            sig_to_class[sig] = cid
            class_members.append([i])

    # 为确定性输出，对等价类按 (signature, first_member) 排序
    order = sorted(range(len(class_members)), key=lambda cid: (signatures[class_members[cid][0]], class_members[cid][0]))
    raw_to_class = np.empty(k, dtype=int)
    class_sizes: List[int] = []
    for new_idx, cid in enumerate(order):
        for m in class_members[cid]:
            raw_to_class[m] = new_idx
        class_sizes.append(len(class_members[cid]))
    return raw_to_class, class_sizes


def apply_rule_symmetry(bits: np.ndarray, k: int, mode: str) -> Tuple[np.ndarray, int, np.ndarray, List[int]]:
    """
    根据对称模式转换规则位串：
      - none: 直接返回
      - perm: canonical_bits（仅置换）
      - perm+swap: 先 canonical_bits，再按等价行/列合并（交换对称）
    返回 (sym_bits, sym_k, raw_to_class, class_sizes)。
    """
    mode = (mode or "perm").lower()
    if mode not in ("none", "perm", "perm+swap", "perm+exchange"):
        raise ValueError(f"unknown symmetry mode: {mode}")

    if mode == "none":
        raw_to_class = np.arange(k, dtype=int)
        return bits.copy(), k, raw_to_class, [1 for _ in range(k)]

    bits_perm = canonical_bits(bits, k)
    if mode == "perm":
        raw_to_class = np.arange(k, dtype=int)
        return bits_perm, k, raw_to_class, [1 for _ in range(k)]

    R = rule_from_bits(k, bits_perm)
    raw_to_class, class_sizes = _exchange_classes(R)
    k2 = int(max(raw_to_class) + 1) if raw_to_class.size > 0 else 0
    reps = []
    for c in range(k2):
        reps.append(int(np.nonzero(raw_to_class == c)[0][0]))
    R_comp = np.zeros((k2, k2), dtype=bool)
    for i, ri in enumerate(reps):
        for j, rj in enumerate(reps):
            R_comp[i, j] = R[ri, rj]
    sym_bits = bits_from_rule(R_comp)
    return sym_bits, k2, raw_to_class, class_sizes

# ------------------------------
# 行生成器（环状合法行）: k ≤ 16 高效版
# ------------------------------

def enumerate_ring_rows_fast(n: int, k: int, R: np.ndarray,
                             batch_limit: int = 200_000) -> np.ndarray:
    assert k <= 16, "enumerate_ring_rows_fast: current implementation assumes k ≤ 16."
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

# ------------------------------
# TransferOp（不显式构造 T）
# ------------------------------

class TransferOp:
    def __init__(self, rows_np: np.ndarray, R_np: np.ndarray,
                 device: str = "cuda", dtype: torch.dtype = torch.float64,
                 block_size_j: Optional[int] = None, block_size_i: Optional[int] = None):
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

# ------------------------------
# 谱估计：幂迭代 / Lanczos / Hutch++
# ------------------------------

@torch.no_grad()
def power_iteration(op: TransferOp, iters: int = 60, tol: float = 1e-8,
                    verbose: bool = False, seed: int = 0) -> Tuple[float, torch.Tensor]:
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
            logging.info(f"[power] iter={t}  lambda≈{lam:.6e}")
        if last_lambda is not None and abs(lam - last_lambda) <= tol * max(1.0, abs(lam)):
            return lam, v
        last_lambda = lam
    return last_lambda if last_lambda is not None else 0.0, v

@torch.no_grad()
def lanczos_top_r(op: TransferOp, r: int = 4, iters: int = 80, seed: int = 0,
                  verbose: bool = False) -> List[float]:
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
            logging.info(f"[lanczos] j={j} alpha={alpha:.6e} beta={beta:.6e}")
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
    y = v
    for _ in range(power):
        y = op.matvec(y)
    return y

@torch.no_grad()
def estimate_trace_T_power_hutch(op, n_power: int, s: int = 32, seed: int = 0) -> float:
    torch.manual_seed(seed)
    m = op.m
    acc = 0.0
    for i in range(s):
        v = torch.empty(m, dtype=op.dtype, device=op.device).uniform_(-1.0, 1.0).sign()
        Tv = apply_T_power(op, v, n_power)
        acc += (v @ Tv).item()
    return acc / float(s)

@torch.no_grad()
def estimate_trace_T_power_hutchpp(op, n_power: int, s: int = 32, seed: int = 0) -> float:
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
    for i in range(max(1, s2)):
        z = torch.empty(m, dtype=op.dtype, device=op.device).uniform_(-1.0, 1.0).sign()
        p = Q.T @ z
        w = z - Q @ p
        Tw = apply_T_power(op, w, n_power)
        trace_res += (w @ Tw).item()
    if s2 > 0:
        trace_res /= float(s2)
    return trace_lowrank + trace_res

# ------------------------------
# 结构性指标
# ------------------------------

def _clustering_coeff(G: np.ndarray) -> float:
    """平均聚类系数（逐点邻接子图三角率均值）。"""
    k = G.shape[0]
    cvals = []
    for u in range(k):
        nbr = np.flatnonzero(G[u])
        d = len(nbr)
        if d < 2:
            cvals.append(0.0)
            continue
        sub = G[np.ix_(nbr, nbr)]
        e = np.triu(sub, 1).sum()
        cvals.append(2.0 * e / (d * (d - 1)))
    return float(np.mean(cvals)) if cvals else 0.0


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


def _graph_spectral_metrics(R_np: np.ndarray) -> Dict:
    """
    构造无向图的谱指标：度矩阵、拉普拉斯（标准/归一化）、
    邻接谱间隙与代数连通度、聚类系数。
    自环从图中移除，仅计入度矩阵的对角。
    """
    k = R_np.shape[0]
    G = R_np.astype(float)
    np.fill_diagonal(G, 0.0)

    deg = G.sum(axis=1)
    D = np.diag(deg)
    L = D - G
    if k == 0:
        return {
            "degree_sequence": [],
            "degree_matrix": [],
            "laplacian": [],
            "laplacian_norm": [],
            "laplacian_alg_conn": 0.0,
            "laplacian_alg_conn_norm": 0.0,
            "adj_lambda1": 0.0,
            "adj_lambda2": 0.0,
            "adj_spectral_gap": 0.0,
            "clustering_coeff": 0.0,
        }

    adj_eigs = np.linalg.eigvalsh(G)
    adj_lambda1 = float(adj_eigs[-1]) if adj_eigs.size > 0 else 0.0
    adj_lambda2 = float(adj_eigs[-2]) if adj_eigs.size >= 2 else 0.0
    adj_gap = adj_lambda1 - adj_lambda2

    lap_eigs = np.linalg.eigvalsh(L)
    laplacian_alg_conn = float(np.sort(lap_eigs)[1]) if lap_eigs.size >= 2 else 0.0

    sqrt_deg = np.sqrt(deg)
    inv_sqrt = np.where(sqrt_deg > 0, 1.0 / sqrt_deg, 0.0)
    L_norm = np.eye(k) - (inv_sqrt[:, None] * G * inv_sqrt[None, :])
    lap_norm_eigs = np.linalg.eigvalsh(L_norm)
    laplacian_alg_conn_norm = float(np.sort(lap_norm_eigs)[1]) if lap_norm_eigs.size >= 2 else 0.0

    clustering = _clustering_coeff(G.astype(bool))

    return {
        "degree_sequence": deg.astype(float).tolist(),
        "degree_matrix": D.astype(float).tolist(),
        "laplacian": L.astype(float).tolist(),
        "laplacian_norm": L_norm.astype(float).tolist(),
        "laplacian_alg_conn": laplacian_alg_conn,
        "laplacian_alg_conn_norm": laplacian_alg_conn_norm,
        "adj_lambda1": adj_lambda1,
        "adj_lambda2": adj_lambda2,
        "adj_spectral_gap": adj_gap,
        "clustering_coeff": clustering,
    }

# ------------------------------
# LRU 行缓存
# ------------------------------

class RowsCacheLRU:
    def __init__(self, capacity=128):
        from collections import OrderedDict
        self.capacity = capacity
        self.od = OrderedDict()
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

# ------------------------------
# 谱半径上界：Gershgorin / 最大度（结构上界，不依赖构造 T）
# ------------------------------

def _upper_bound_raw(R_np: np.ndarray, n: int) -> Dict[str, float]:
    """
    给出 λ(T) 的结构性上界（不显式构造 T）：rho_gersh、rho_maxdeg。
    在 evaluate_rules_batch 中结合 m -> ub_raw_* = m * (rho_*)^n。
    直观：
      - rho_maxdeg ~ max行/列和的安全上界：用 R 的平均度作为尺度（不过分放大）
      - rho_gersh   取 min(k, avg_deg) 的安全上界（避免极端夸大）
    """
    k = R_np.shape[0]
    deg = R_np.sum(axis=1).astype(float)
    deg_mean = float(np.mean(deg)) if k > 0 else 0.0
    rho_maxdeg = max(1.0, deg_mean)      # 不做 n 次方放大，避免过松
    rho_gersh  = max(1.0, min(float(k), deg_mean))
    return {"rho_gersh": rho_gersh, "rho_maxdeg": rho_maxdeg}


def _parse_exact_threshold(threshold) -> Tuple[str, Optional[float]]:
    """
    返回 (mode, limit):
      - mode ∈ {"nk", "rows"}
      - limit 若为 None 表示无穷制
    支持示例："nk<=12"、"rows<=200000"、"500000"（视为 rows）、None。
    """
    mode = "nk"; limit: Optional[float] = None
    if threshold is None:
        return mode, limit
    if isinstance(threshold, (int, float)):
        return "rows", float(threshold)
    s = str(threshold).strip().lower()
    if not s:
        return mode, limit
    import re
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)
    if "row" in s or s.startswith("m"):
        mode = "rows"
    elif "nk" in s:
        mode = "nk"
    try:
        if nums:
            limit = float(nums[0])
    except Exception:
        limit = None
    return mode, limit


def _should_use_exact(enable_exact: bool, threshold, n: int, k_sym: int, rows_m: int) -> Tuple[bool, str]:
    if not enable_exact:
        return False, "exact disabled"
    mode, limit = _parse_exact_threshold(threshold)
    if limit is None:
        return True, ""
    if mode == "nk" and (n * k_sym) > limit:
        return False, f"n*k={n*k_sym} exceeds {limit}"
    if mode == "rows" and rows_m > limit:
        return False, f"rows_m={rows_m} exceeds {limit}"
    return True, ""


def summarize_trace_comparison(fits: List[Dict], warn_rel: float = 0.05, logger=logging.getLogger(__name__)) -> None:
    """输出精确值 vs 估计值摘要，用于 CLI 日志或调试。"""
    rel_errs = []
    for f in fits:
        if "trace_exact" not in f or "trace_estimate" not in f:
            continue
        try:
            exact = float(f.get("trace_exact", float("nan")))
            est = float(f.get("trace_estimate", float("nan")))
        except Exception:
            continue
        if not (np.isfinite(exact) and np.isfinite(est)):
            continue
        if exact == 0:
            continue
        rel_errs.append(abs(est - exact) / max(1e-15, abs(exact)))
    if not rel_errs:
        return
    rel_errs = sorted(rel_errs)
    p50 = rel_errs[len(rel_errs)//2]
    p90 = rel_errs[int(len(rel_errs)*0.9)]
    max_e = rel_errs[-1]
    msg = f"[compare] exact vs est | count={len(rel_errs)}, median={p50:.3e}, p90={p90:.3e}, max={max_e:.3e}"
    if max_e >= warn_rel:
        logger.warning(msg)
    else:
        logger.info(msg)

# ------------------------------
# 批量评估
# ------------------------------

def evaluate_rules_batch(n: int,
                         k: int,
                         bits_list: List[np.ndarray],
                         sym_mode: str = "perm",
                         boundary: str = config.BOUNDARY_MODE,
                         device: str = "cuda",
                         use_lanczos: bool = True,
                         r_vals: int = 3,
                         power_iters: int = 50,
                         trace_mode: str = "hutchpp",
                         hutch_s: int = 24,
                         lru_rows: Optional[RowsCacheLRU] = None,
                         max_streams: int = 2,
                         enable_exact: bool = True,
                         enable_spectral: bool = True,
                         exact_threshold="nk<=12",
                         cache_dir: Optional[Union[str, Path]] = None,
                         use_cache: bool = True,
                         ) -> List[Dict]:
    """
    批量评估规则个体，输出：
      - rows_m, lambda_max, lambda_top2, spectral_gap, sum_lambda_powers
      - active_k（压缩后）、active_k_raw（压缩前）、k_sym/k_raw、sym_mode
      - lower_bound / upper_bound（raw 与惩罚后）
      - 结构上界：upper_bound_raw_gersh / upper_bound_raw_maxdeg
      - archetype_tags（若 rules.structures 存在）
      - exact_Z（启用且阈值允许时）
      - trace_exact / trace_estimate / trace_error（若同时有精确值与估计值）
    """
    PENALTY_ALPHA = 1.5
    def _adaptive_samples(m: int, base: int) -> int:
        if m < 2_000:
            return max(12, base // 2)
        elif m > 50_000:
            return min(64, base * 2)
        return base

    boundary = (boundary or "torus").lower()
    if boundary not in {"torus", "open"}:
        raise ValueError(f"boundary={boundary} not supported")

    cache_root = ensure_eval_cache_dir(cache_dir) if use_cache else None

    # ---------- open 边界：仅精确计数 ----------
    if boundary == "open":
        outs: List[Dict] = []
        for idx, bits in enumerate(bits_list):
            bits_sym, k_sym, raw_to_class, class_sizes = apply_rule_symmetry(bits, k, sym_mode)
            R = rule_from_bits(k_sym, bits_sym)
            rows = enumerate_ring_rows(n=n, k=k_sym, R=R, boundary="open")
            m = len(rows)
            active_classes = np.unique(rows) if m > 0 else np.array([], dtype=int)
            used = np.zeros(k_sym, dtype=bool)
            used[active_classes] = True
            active_k = int(used.sum())
            active_k_raw = int(sum(class_sizes[c] for c in active_classes)) if m > 0 else 0

            cache_key = None
            if cache_root is not None:
                cache_key = make_eval_cache_key(bits_sym, active_k, boundary, sym_mode, n)
                cached = load_eval_cache(cache_root, cache_key)
                if cached is not None:
                    outs.append(cached)
                    continue

            eval_note = ""
            if m == 0:
                Z_exact = 0.0
                eval_note = "open_boundary_rows_m0"
            else:
                Z_exact = float(count_patterns_transfer_matrix(n=n, k=k_sym, R=R, boundary="open"))
                eval_note = "open_boundary_exact"

            fit = {
                "rule_count": int(bits.sum()),
                "lambda_max": 0.0,
                "lambda_top2": (0.0, 0.0),
                "spectral_gap": 0.0,
                "sum_lambda_powers": Z_exact,
                "rows_m": m,
                "active_k": active_k,
                "active_k_raw": active_k_raw,
                "active_k_sym": active_k,
                "k_raw": k,
                "k_sym": k_sym,
                "sym_mode": sym_mode,
                "boundary": boundary,
                "lower_bound": Z_exact, "upper_bound": Z_exact,
                "lower_bound_raw": Z_exact, "upper_bound_raw": Z_exact,
                "upper_bound_raw_gersh": Z_exact, "upper_bound_raw_maxdeg": Z_exact,
                "archetype_hits": {}, "archetype_tags": "",
                "trace_exact": Z_exact, "trace_estimate": Z_exact,
                "trace_error": "", "trace_error_rel": "",
                "eval_note": eval_note,
            }
            if cache_root is not None and cache_key is not None:
                dump_eval_cache(cache_root, cache_key, fit, meta_extra={
                    "boundary": boundary,
                    "sym_mode": sym_mode,
                    "active_k": active_k,
                    "k_sym": k_sym,
                    "n": n,
                })
            outs.append(fit)
        return outs

    # ---------- torus 边界：原有 CPU/GPU 分支 ----------
    if (device == "cpu") or (not torch.cuda.is_available()):
        outs: List[Dict] = []
        for idx, bits in enumerate(bits_list):
            bits_sym, k_sym, raw_to_class, class_sizes = apply_rule_symmetry(bits, k, sym_mode)
            R = rule_from_bits(k_sym, bits_sym)
            deg, comps = _graph_degrees_and_components(R)
            graph_stats = _graph_spectral_metrics(R)

            rows = enumerate_ring_rows_fast(n, k_sym, R)
            m = int(rows.shape[0])
            exact_allowed, exact_reason = _should_use_exact(enable_exact, exact_threshold, n, k_sym, m)
            spectral_allowed = enable_spectral and (m > 0)
            spectral_reason = "" if spectral_allowed else ("rows_m=0" if m == 0 else "spectral disabled")

            active_classes = np.unique(rows) if m > 0 else np.array([], dtype=int)
            used = np.zeros(k_sym, dtype=bool)
            used[active_classes] = True
            active_k = int(used.sum())
            active_k_raw = int(sum(class_sizes[c] for c in active_classes)) if m > 0 else 0

            cache_key = None
            if cache_root is not None:
                cache_key = make_eval_cache_key(bits_sym, active_k, boundary, sym_mode, n)
                cached = load_eval_cache(cache_root, cache_key)
                if cached is not None:
                    outs.append(cached)
                    continue

            trace_exact: Optional[float] = None
            trace_estimate: Optional[float] = None
            trace_error: Optional[float] = None
            trace_error_rel: Optional[float] = None
            eval_notes: List[str] = []
            if not spectral_allowed:
                eval_notes.append(f"spectral:{spectral_reason}")
            if not exact_allowed:
                eval_notes.append(f"exact:{exact_reason}")

            if m == 0:
                # 上界也为 0
                ub_rhos = _upper_bound_raw(R, n)
                fit = {
                    "rule_count": int(bits.sum()),
                    "lambda_max": 0.0,
                    "lambda_top2": (0.0, 0.0),
                    "spectral_gap": 0.0,
                    "sum_lambda_powers": -1e300,
                    "rows_m": 0,
                    "active_k": active_k,
                    "active_k_raw": active_k_raw,
                    "active_k_sym": 0,
                    "k_raw": k,
                    "k_sym": k_sym,
                    "sym_mode": sym_mode,
                    "boundary": boundary,
                    "lower_bound": 0.0, "upper_bound": 0.0,
                    "lower_bound_raw": 0.0, "upper_bound_raw": 0.0,
                    "upper_bound_raw_gersh": 0.0, "upper_bound_raw_maxdeg": 0.0,
                    "archetype_hits": {},
                    "archetype_tags": "",
                    **graph_stats,
                }
                fit.update({
                    "trace_exact": trace_exact if trace_exact is not None else "",
                    "trace_estimate": trace_estimate if trace_estimate is not None else -1e300,
                    "trace_error": trace_error if trace_error is not None else "",
                    "trace_error_rel": trace_error_rel if trace_error_rel is not None else "",
                    "eval_note": "; ".join(eval_notes) if eval_notes else "",
                })
                if cache_root is not None and cache_key is not None:
                    dump_eval_cache(cache_root, cache_key, fit, meta_extra={
                        "boundary": boundary,
                        "sym_mode": sym_mode,
                        "active_k": active_k,
                        "k_sym": k_sym,
                        "n": n,
                    })
                outs.append(fit)
                continue

            s_adapt = _adaptive_samples(m, hutch_s)
            lambda1 = lambda2 = lam_pos = 0.0
            lb_raw = lb = 0.0
            ub_raw = ub = 0.0
            spectral_gap = 0.0
            sum_lp = -1e300

            if spectral_allowed:
                op = TransferOp(rows, R, device="cpu", dtype=torch.float64, block_size_j=2048)

                if use_lanczos:
                    evals = lanczos_top_r(op, r=r_vals, iters=max(2*r_vals, 20), seed=idx, verbose=False)
                    lambda1 = float(evals[0]) if len(evals) > 0 else 0.0
                    lambda2 = float(evals[1]) if len(evals) > 1 else 0.0
                    lam_pos = max(0.0, lambda1)
                    lb_raw = lam_pos ** n
                    lb_r = sum((max(0.0, lam) ** n) for lam in evals) if len(evals) > 0 else 0.0
                    lb_raw = max(lb_raw, lb_r)
                    if trace_mode == "lanczos_sum":
                        est = float(lb_r)
                    elif trace_mode == "hutch":
                        est = float(estimate_trace_T_power_hutch(op, n_power=n, s=s_adapt, seed=idx))
                    elif trace_mode == "hutchpp":
                        est = float(estimate_trace_T_power_hutchpp(op, n_power=n, s=s_adapt, seed=idx))
                    else:
                        est = float(lam_pos ** n)
                else:
                    lambda1, _ = power_iteration(op, iters=power_iters, tol=1e-9, verbose=False, seed=idx)
                    lambda2 = 0.0
                    lam_pos = max(0.0, lambda1)
                    lb_raw = lam_pos ** n
                    est = float(lam_pos ** n)

                ub_raw = float(m) * (lam_pos ** n)
                spectral_gap = max(0.0, lambda1 - lambda2)

                penal = 1.0
                if comps > 1:
                    penal *= (0.5 ** (comps - 1))
                if np.any(deg == 0):
                    penal *= (active_k / k) ** PENALTY_ALPHA

                sum_lp = est * penal
                lb = lb_raw * penal
                ub = ub_raw * penal
                trace_estimate = sum_lp
            else:
                sum_lp = float(trace_exact) if trace_exact is not None else -1e300

            # 结构上界（与 m 结合）
            ub_rhos = _upper_bound_raw(R, n)
            ub_raw_gersh = float(m) * (ub_rhos["rho_gersh"] ** n)
            ub_raw_maxdeg = float(m) * (ub_rhos["rho_maxdeg"] ** n)

            # 原型识别（可选）
            try:
                from .structures import recognize_archetypes
                arc = recognize_archetypes(bits)
                archetype_tags = ";".join([kk for kk,v in arc.items() if v])
                archetype_hits = {k: bool(v) for k, v in arc.items()}
            except Exception:
                archetype_tags = ""
                archetype_hits = {}

            if exact_allowed:
                try:
                    from .stage1_exact import count_patterns_transfer_matrix
                    trace_exact = float(count_patterns_transfer_matrix(n, k_sym, R, return_rows=False))
                except Exception as exc:
                    exact_reason = f"exact failed: {exc.__class__.__name__}"
                    eval_notes.append(f"exact:{exact_reason}")
            # 当谱估计被禁用但 exact 已成功时，沿用精确值作为 sum_lambda_powers，避免 NA。
            if (not spectral_allowed) and (trace_exact is not None):
                sum_lp = trace_exact
                lb = lb_raw = trace_exact
                ub = ub_raw = trace_exact
                trace_estimate = trace_exact

            if trace_estimate is None:
                trace_estimate = sum_lp
            if (trace_exact is not None) and (trace_estimate is not None) and np.isfinite(trace_exact) and np.isfinite(trace_estimate):
                trace_error = trace_estimate - trace_exact
                if trace_exact != 0:
                    trace_error_rel = trace_error / trace_exact

            fit = {
                "rule_count": int(bits.sum()),
                "lambda_max": lambda1,
                "lambda_top2": (lambda1, lambda2),
                "spectral_gap": spectral_gap,
                "sum_lambda_powers": sum_lp,
                "rows_m": m,
                "active_k": active_k,
                "active_k_raw": active_k_raw,
                "active_k_sym": active_k,
                "k_raw": k,
                "k_sym": k_sym,
                "sym_mode": sym_mode,
                "boundary": boundary,
                "lower_bound": lb, "upper_bound": ub,
                "lower_bound_raw": lb_raw, "upper_bound_raw": ub_raw,
                "upper_bound_raw_gersh": ub_raw_gersh, "upper_bound_raw_maxdeg": ub_raw_maxdeg,
                "archetype_hits": archetype_hits,
                "archetype_tags": archetype_tags,
                **graph_stats,
            }

            if trace_exact is not None:
                fit["exact_Z"] = int(trace_exact)
                fit["trace_exact"] = trace_exact
            if trace_estimate is not None:
                fit["trace_estimate"] = trace_estimate
            if trace_error is not None:
                fit["trace_error"] = trace_error
                if trace_error_rel is not None:
                    fit["trace_error_rel"] = trace_error_rel
            if eval_notes:
                fit["eval_note"] = "; ".join(eval_notes)
            if cache_root is not None and cache_key is not None:
                dump_eval_cache(cache_root, cache_key, fit, meta_extra={
                    "boundary": boundary,
                    "sym_mode": sym_mode,
                    "active_k": active_k,
                    "k_sym": k_sym,
                    "n": n,
                })
            outs.append(fit)
        return outs

    # ---------- CUDA 路径 ----------
    streams = [torch.cuda.Stream() for _ in range(max(1, max_streams))]
    outputs: List[Dict] = [None] * len(bits_list)
    rows_lru = lru_rows or RowsCacheLRU(capacity=128)

    for idx, bits in enumerate(bits_list):
        bits_sym, k_sym, raw_to_class, class_sizes = apply_rule_symmetry(bits, k, sym_mode)
        key = (sym_mode + "|").encode("utf-8") + bits_sym.tobytes()
        R = rule_from_bits(k_sym, bits_sym)
        deg, comps = _graph_degrees_and_components(R)
        graph_stats = _graph_spectral_metrics(R)

        rows = rows_lru.get(key)
        if rows is None:
            rows = enumerate_ring_rows_fast(n, k_sym, R)
            if rows.shape[0] > 0 and rows.shape[0] <= 1_000_000:
                rows_lru.put(key, rows)

        m = int(rows.shape[0])
        exact_allowed, exact_reason = _should_use_exact(enable_exact, exact_threshold, n, k_sym, m)
        spectral_allowed = enable_spectral and (m > 0)
        spectral_reason = "" if spectral_allowed else ("rows_m=0" if m == 0 else "spectral disabled")

        active_classes = np.unique(rows) if m > 0 else np.array([], dtype=int)
        used = np.zeros(k_sym, dtype=bool)
        used[active_classes] = True
        active_k = int(used.sum())
        active_k_raw = int(sum(class_sizes[c] for c in active_classes)) if m > 0 else 0

        cache_key = None
        if cache_root is not None:
            cache_key = make_eval_cache_key(bits_sym, active_k, boundary, sym_mode, n)
            cached = load_eval_cache(cache_root, cache_key)
            if cached is not None:
                outputs[idx] = cached
                continue

        trace_exact: Optional[float] = None
        trace_estimate: Optional[float] = None
        trace_error: Optional[float] = None
        trace_error_rel: Optional[float] = None
        eval_notes: List[str] = []
        if not spectral_allowed:
            eval_notes.append(f"spectral:{spectral_reason}")
        if not exact_allowed:
            eval_notes.append(f"exact:{exact_reason}")

        if m == 0:
            ub_rhos = _upper_bound_raw(R, n)
            outputs[idx] = {
                "rule_count": int(bits.sum()),
                "lambda_max": 0.0,
                "lambda_top2": (0.0, 0.0),
                "spectral_gap": 0.0,
                "sum_lambda_powers": -1e300,
                "rows_m": 0,
                "active_k": active_k,
                "active_k_raw": active_k_raw,
                "active_k_sym": 0,
                "k_raw": k,
                "k_sym": k_sym,
                "sym_mode": sym_mode,
                "boundary": boundary,
                "lower_bound": 0.0, "upper_bound": 0.0,
                "lower_bound_raw": 0.0, "upper_bound_raw": 0.0,
                "upper_bound_raw_gersh": 0.0, "upper_bound_raw_maxdeg": 0.0,
                "archetype_hits": {},
                "archetype_tags": "",
                "trace_exact": trace_exact if trace_exact is not None else "",
                "trace_estimate": trace_estimate if trace_estimate is not None else -1e300,
                "trace_error": trace_error if trace_error is not None else "",
                "trace_error_rel": trace_error_rel if trace_error_rel is not None else "",
                "eval_note": "; ".join(eval_notes) if eval_notes else "",
                **graph_stats,
            }
            if cache_root is not None and cache_key is not None:
                dump_eval_cache(cache_root, cache_key, outputs[idx], meta_extra={
                    "boundary": boundary,
                    "sym_mode": sym_mode,
                    "active_k": active_k,
                    "k_sym": k_sym,
                    "n": n,
                })
            continue

        s_adapt = _adaptive_samples(m, hutch_s)
        st = streams[idx % len(streams)]

        lambda1 = lambda2 = lam_pos = 0.0
        lb_raw = lb = 0.0
        ub_raw = ub = 0.0
        spectral_gap = 0.0
        sum_lp = -1e300

        if spectral_allowed:
            with torch.cuda.stream(st):
                op = TransferOp(rows, R, device=device, dtype=torch.float64, block_size_j=4096)

                if use_lanczos:
                    evals = lanczos_top_r(op, r=r_vals, iters=max(2*r_vals, 20), seed=idx, verbose=False)
                    lambda1 = float(evals[0]) if len(evals) > 0 else 0.0
                    lambda2 = float(evals[1]) if len(evals) > 1 else 0.0
                    lam_pos = max(0.0, lambda1)
                    lb_raw = lam_pos ** n
                    lb_r = sum((max(0.0, lam) ** n) for lam in evals) if len(evals) > 0 else 0.0
                    lb_raw = max(lb_raw, lb_r)
                    if trace_mode == "lanczos_sum":
                        est = float(lb_r)
                    elif trace_mode == "hutch":
                        est = float(estimate_trace_T_power_hutch(op, n_power=n, s=s_adapt, seed=idx))
                    elif trace_mode == "hutchpp":
                        est = float(estimate_trace_T_power_hutchpp(op, n_power=n, s=s_adapt, seed=idx))
                    else:
                        est = float(lam_pos ** n)
                else:
                    lambda1, _ = power_iteration(op, iters=power_iters, tol=1e-9, verbose=False, seed=idx)
                    lambda2 = 0.0
                    lam_pos = max(0.0, lambda1)
                    lb_raw = lam_pos ** n
                    est = float(lam_pos ** n)

                ub_raw = float(m) * (lam_pos ** n)
                spectral_gap = max(0.0, lambda1 - lambda2)

                penal = 1.0
                if comps > 1:
                    penal *= (0.5 ** (comps - 1))
                if np.any(deg == 0):
                    penal *= (active_k / k) ** PENALTY_ALPHA

                sum_lp = est * penal
                lb = lb_raw * penal
                ub = ub_raw * penal
                trace_estimate = sum_lp
        else:
            sum_lp = float(trace_exact) if trace_exact is not None else -1e300

            ub_rhos = _upper_bound_raw(R, n)
            ub_raw_gersh = float(m) * (ub_rhos["rho_gersh"] ** n)
            ub_raw_maxdeg = float(m) * (ub_rhos["rho_maxdeg"] ** n)

            try:
                from .structures import recognize_archetypes
                arc = recognize_archetypes(bits)
                archetype_tags = ";".join([kk for kk,v in arc.items() if v])
                archetype_hits = {k: bool(v) for k, v in arc.items()}
            except Exception:
                archetype_tags = ""
                archetype_hits = {}

            if exact_allowed:
                try:
                    from .stage1_exact import count_patterns_transfer_matrix
                    trace_exact = float(count_patterns_transfer_matrix(n, k_sym, R, return_rows=False))
                except Exception as exc:
                    exact_reason = f"exact failed: {exc.__class__.__name__}"
                    eval_notes.append(f"exact:{exact_reason}")
            if (not spectral_allowed) and (trace_exact is not None):
                sum_lp = trace_exact
                lb = lb_raw = trace_exact
                ub = ub_raw = trace_exact
                trace_estimate = trace_exact

            if trace_estimate is None:
                trace_estimate = sum_lp
            if (trace_exact is not None) and (trace_estimate is not None) and np.isfinite(trace_exact) and np.isfinite(trace_estimate):
                trace_error = trace_estimate - trace_exact
                if trace_exact != 0:
                    trace_error_rel = trace_error / trace_exact

            fit = {
                "rule_count": int(bits.sum()),
                "lambda_max": lambda1,
                "lambda_top2": (lambda1, lambda2),
                "spectral_gap": spectral_gap,
                "sum_lambda_powers": sum_lp,
                "rows_m": m,
                "active_k": active_k,
                "active_k_raw": active_k_raw,
                "active_k_sym": active_k,
                "k_raw": k,
                "k_sym": k_sym,
                "sym_mode": sym_mode,
                "boundary": boundary,
                "lower_bound": lb, "upper_bound": ub,
                "lower_bound_raw": lb_raw, "upper_bound_raw": ub_raw,
                "upper_bound_raw_gersh": ub_raw_gersh, "upper_bound_raw_maxdeg": ub_raw_maxdeg,
                "archetype_hits": archetype_hits,
                "archetype_tags": archetype_tags,
                **graph_stats,
            }

            if trace_exact is not None:
                fit["exact_Z"] = int(trace_exact)
                fit["trace_exact"] = trace_exact
            if trace_estimate is not None:
                fit["trace_estimate"] = trace_estimate
            if trace_error is not None:
                fit["trace_error"] = trace_error
                if trace_error_rel is not None:
                    fit["trace_error_rel"] = trace_error_rel
            if eval_notes:
                fit["eval_note"] = "; ".join(eval_notes)

            if cache_root is not None and cache_key is not None:
                dump_eval_cache(cache_root, cache_key, fit, meta_extra={
                    "boundary": boundary,
                    "sym_mode": sym_mode,
                    "active_k": active_k,
                    "k_sym": k_sym,
                    "n": n,
                })

            outputs[idx] = fit

    torch.cuda.synchronize()
    return outputs
