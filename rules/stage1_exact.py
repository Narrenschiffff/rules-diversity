# -*- coding: utf-8 -*-
"""
rules/stage1_exact.py
小规模 (n, k) 精确计数与前沿扫描（同时输出 raw 与 canon 版本）
产出：
  - stage1_all_n{n}_k{k}_raw.csv     （未做置换同构折叠）
  - stage1_pareto_n{n}_k{k}_raw.csv
  - stage1_all_n{n}_k{k}_canon.csv   （按 S_k 置换同构折叠后的代表）
  - stage1_pareto_n{n}_k{k}_canon.csv
列名统一、可复用：
  run_tag, n, k, rule_bits_raw, rule_bits_canon, is_canonical_rep,
  rule_count, Z_exact, rows_m, [arch_* ...]
"""
from __future__ import annotations
from itertools import product
from typing import List, Dict, Tuple, Optional
import os, csv, math
import numpy as np

__all__ = [
    "make_rule_matrix",
    "neighbors_torus",
    "count_patterns_backtrack",
    "enumerate_ring_rows",
    "build_row_compat_matrix",
    "count_patterns_transfer_matrix",
    "cross_check",
    "rule_complete_bipartite",
    "rule_complete_graph",
    "rule_only_self_loops",
    "rule_path_graph",
    "rule_cycle_graph",
    "rule_star_graph",
    "scan_all_rules_exact",
]

# ---------- 依赖（可选） ----------
_HAS_EVAL = False
try:
    from .eval import rule_from_bits as _rule_from_bits_eval
    from .eval import canonical_bits as _canonical_bits_eval  # 期望存在
    _HAS_EVAL = True
except Exception:
    _HAS_EVAL = False

_HAS_STRUCT = False
try:
    from . import structures as _structures
    _HAS_STRUCT = hasattr(_structures, "recognize_archetypes")
except Exception:
    _HAS_STRUCT = False


# ---------- 基础：规则矩阵 & 网格 ----------
def make_rule_matrix(k, allowed_pairs, allow_self_loops=None):
    R = np.zeros((k, k), dtype=bool)
    for (u, v) in allowed_pairs:
        R[u, v] = True
        R[v, u] = True
    if allow_self_loops is not None:
        for i in range(k):
            R[i, i] = bool(allow_self_loops)
    return R

def neighbors_torus(i, j, n):
    return [((i - 1) % n, j),
            ((i + 1) % n, j),
            (i, (j - 1) % n),
            (i, (j + 1) % n)]

# ---------- 回溯计数（用于对照） ----------
def count_patterns_backtrack(n, k, R):
    grid = [[-1] * n for _ in range(n)]
    total = 0
    def ok_left(i, j, v):
        lj = (j - 1) % n; lv = grid[i][lj]
        return True if lv == -1 else bool(R[v, lv])
    def ok_up(i, j, v):
        ui = (i - 1) % n; uv = grid[ui][j]
        return True if uv == -1 else bool(R[v, uv])
    def dfs(pos):
        nonlocal total
        if pos == n*n:
            for i in range(n):
                for j in range(n):
                    v = grid[i][j]
                    if not R[v, grid[i][(j+1)%n]]: return
                    if not R[v, grid[(i+1)%n][j]]: return
            total += 1; return
        i, j = divmod(pos, n)
        for v in range(k):
            if ok_left(i,j,v) and ok_up(i,j,v):
                grid[i][j] = v
                dfs(pos+1)
                grid[i][j] = -1
    dfs(0)
    return total

# ---------- 转移矩阵计数（主力） ----------
def enumerate_ring_rows(n, k, R):
    rows, seq = [], [-1]*n
    def dfs(pos):
        if pos == n:
            if R[seq[-1], seq[0]]: rows.append(seq.copy())
            return
        for v in range(k):
            if pos==0 or R[seq[pos-1], v]:
                seq[pos]=v; dfs(pos+1); seq[pos]=-1
    dfs(0); return rows

def build_row_compat_matrix(rows, R):
    m = len(rows); T = np.zeros((m, m), dtype=np.int64)
    for i,a in enumerate(rows):
        for j,b in enumerate(rows):
            ok = True
            for c in range(len(a)):
                if not R[a[c], b[c]]: ok=False; break
            if ok: T[i,j]=1
    return T

def count_patterns_transfer_matrix(n, k, R, return_rows=False):
    rows = enumerate_ring_rows(n,k,R)
    T = build_row_compat_matrix(rows, R)
    M = np.array(T, dtype=object)
    for _ in range(n-1):
        M = M @ T
    Z = int(np.trace(M))
    if return_rows: return Z, rows, T, M
    return Z

def cross_check(n, k, R, verbose=True):
    a = count_patterns_backtrack(n,k,R)
    b = count_patterns_transfer_matrix(n,k,R)
    if verbose: print(f"[check] n={n},k={k} backtrack={a}, transfer={b}")
    assert a==b, "计数不一致"
    return a

# ---------- 规则构造器（示例） ----------
def rule_complete_graph(k, allow_self=True):
    pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
    return make_rule_matrix(k, pairs, allow_self)

def rule_only_self_loops(k):
    R = np.zeros((k,k), dtype=bool)
    for i in range(k): R[i,i]=True
    return R

def rule_path_graph(k, allow_self=True):
    pairs = [(i, i+1) for i in range(k-1)]
    return make_rule_matrix(k, pairs, allow_self)

def rule_cycle_graph(k, allow_self=True):
    pairs = [(i, (i+1)%k) for i in range(k)]
    return make_rule_matrix(k, pairs, allow_self)

def rule_star_graph(k, center=0, allow_self=True):
    pairs = [(center, j) for j in range(k) if j!=center]
    return make_rule_matrix(k, pairs, allow_self)

def rule_complete_bipartite(k, A=None, allow_self=False):
    if A is None: A = set(range(k//2))
    B = set(range(k)) - A
    pairs = [(a,b) for a in A for b in B]
    return make_rule_matrix(k, pairs, allow_self)

# ---------- 比特编解码 & 同构规范化 ----------
def _bits_length_for_k(k:int)->int:
    # diag k + upper-tri k*(k-1)/2  (无向 + 自环位)
    return k + (k*(k-1))//2

def _rule_from_bits_compact(k:int, bits:np.ndarray)->np.ndarray:
    L = _bits_length_for_k(k); assert bits.size==L
    R = np.zeros((k,k), dtype=bool)
    # diag
    for i in range(k): R[i,i] = bool(bits[i])
    # upper
    p = k
    for i in range(k):
        for j in range(i+1, k):
            b = bool(bits[p]); p+=1
            if b: R[i,j]=R[j,i]=True
    return R

def _rule_from_bits_any(k:int, bits:np.ndarray)->np.ndarray:
    if _HAS_EVAL: return _rule_from_bits_eval(k, bits)
    return _rule_from_bits_compact(k, bits)

def _canonical_bits(bits:np.ndarray, k:int, mode:str="heuristic")->np.ndarray:
    """
    统一入口：
    - 若项目内提供 rules.eval.canonical_bits 则直接用；
    - 否则用轻量启发式：按 (度, 自环, 邻接字典序) 排序重标。
    """
    if _HAS_EVAL:
        try:
            return _canonical_bits_eval(bits, k)
        except Exception:
            pass
    # 轻量启发式
    R = _rule_from_bits_compact(k, bits)
    deg = R.sum(1).astype(int)
    selfloop = np.diag(R).astype(int)
    adj_str = ["".join('1' if x else '0' for x in row.tolist()) for row in R]
    order = sorted(range(k), key=lambda i: (deg[i], selfloop[i], adj_str[i]), reverse=True)
    P = R[np.ix_(order, order)]
    # 重新编码
    L = _bits_length_for_k(k)
    out = np.zeros(L, dtype=np.uint8)
    # diag
    for i in range(k):
        out[i] = 1 if P[i,i] else 0
    p = k
    for i in range(k):
        for j in range(i+1,k):
            out[p] = 1 if P[i,j] else 0; p+=1
    return out

def _enumerate_all_rule_bits_raw(k:int)->List[np.ndarray]:
    L = _bits_length_for_k(k)
    out = []
    for mask in range(1<<L):
        vec = np.fromiter(((mask>>i)&1 for i in range(L)), dtype=np.uint8, count=L)
        out.append(vec)
    return out

# ---------- 前沿 ----------
def _pareto_front_minR_maxZ(items: List[Tuple[int,int]])->List[int]:
    front, idxs = [], range(len(items))
    for i in idxs:
        ri, zi = items[i]
        dom = False
        for j in idxs:
            if i==j: continue
            rj,zj = items[j]
            if (rj<=ri and zj>=zi) and (rj<ri or zj>zi):
                dom=True; break
        if not dom: front.append(i)
    front.sort(key=lambda t:(items[t][0], -items[t][1]))
    return front

# ---------- 扫描（一次遍历，产生 raw 与 canon 两套输出） ----------
def scan_all_rules_exact(n:int,
                         k:int,
                         out_csv_dir:str="./results/out_csv",
                         canonical:bool=True,
                         mark_archetypes:bool=True,
                         save_rows_m:bool=True,
                         run_tag:Optional[str]=None)->Tuple[str,str]:
    """
    返回值保持兼容：返回“canon”版本的 (all_csv, pareto_csv) 路径；
    同时额外写出 *_raw.csv 两个文件。
    """
    os.makedirs(out_csv_dir, exist_ok=True)
    tag = run_tag or f"stage1_n{n}_k{k}"

    raw_rows: List[Dict] = []
    canon_rows_by_key: Dict[bytes, Dict] = {}

    for bits_raw in _enumerate_all_rule_bits_raw(k):
        bits_canon = _canonical_bits(bits_raw, k)
        R = _rule_from_bits_any(k, bits_raw)
        Z, rows_m = count_patterns_transfer_matrix(n,k,R), 0
        if save_rows_m:
            _, rows, *_ = count_patterns_transfer_matrix(n,k,R, return_rows=True)
            rows_m = len(rows)

        row_base = {
            "run_tag": tag,
            "n": n, "k": k,
            "rule_bits_raw": "".join(str(int(x)) for x in bits_raw.tolist()),
            "rule_bits_canon": "".join(str(int(x)) for x in bits_canon.tolist()),
            "rule_count": int(bits_raw.sum()),
            "Z_exact": int(Z),
            "rows_m": int(rows_m),
        }
        if mark_archetypes and _HAS_STRUCT:
            try:
                arch = _structures.recognize_archetypes(bits_raw)
                for k0, v0 in arch.items():
                    row_base[f"arch_{k0}"] = int(bool(v0))
            except Exception:
                pass

        # raw 集直接收集
        raw_rows.append({**row_base, "is_canonical_rep": int(row_base["rule_bits_raw"]==row_base["rule_bits_canon"])})

        # canon 代表：以 canonical bits 做 key，保留（|R| 相同，Z 相同）任一即可；这里收 lex 最小 raw 代表
        key = bits_canon.tobytes()
        prev = canon_rows_by_key.get(key)
        if (prev is None) or (row_base["rule_bits_raw"] < prev["rule_bits_raw"]):
            canon_rows_by_key[key] = {**row_base, "is_canonical_rep": 1}

    # ===== 写 CSV =====
    raw_all = os.path.join(out_csv_dir, f"stage1_all_n{n}_k{k}_raw.csv")
    with open(raw_all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader(); w.writerows(raw_rows)

    raw_objs = [(r["rule_count"], r["Z_exact"]) for r in raw_rows]
    raw_front_idx = _pareto_front_minR_maxZ(raw_objs)
    raw_pareto = [raw_rows[i] for i in raw_front_idx]
    raw_front = os.path.join(out_csv_dir, f"stage1_pareto_n{n}_k{k}_raw.csv")
    with open(raw_front, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader(); w.writerows(raw_pareto)

    canon_rows = list(canon_rows_by_key.values())
    canon_all = os.path.join(out_csv_dir, f"stage1_all_n{n}_k{k}_canon.csv")
    with open(canon_all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(canon_rows[0].keys()))
        w.writeheader(); w.writerows(canon_rows)

    canon_objs = [(r["rule_count"], r["Z_exact"]) for r in canon_rows]
    canon_front_idx = _pareto_front_minR_maxZ(canon_objs)
    canon_pareto = [canon_rows[i] for i in canon_front_idx]
    canon_front = os.path.join(out_csv_dir, f"stage1_pareto_n{n}_k{k}_canon.csv")
    with open(canon_front, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(canon_rows[0].keys()))
        w.writeheader(); w.writerows(canon_pareto)

    # 与原 CLI 保持兼容：返回 canon 两条路径
    return canon_all, canon_front


if __name__ == "__main__":
    n, k = 3, 3
    R = rule_cycle_graph(k, allow_self=True)
    cross_check(n, k, R, verbose=True)
    a, p = scan_all_rules_exact(n=n, k=k, out_csv_dir="./results/out_csv",
                                canonical=True, mark_archetypes=False, save_rows_m=True)
    print("all(canon):", a); print("pareto(canon):", p)
