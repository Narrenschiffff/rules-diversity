# -*- coding: utf-8 -*-
import numpy as np
from itertools import product
from typing import List, Tuple
from .logging_setup import get_logger

logger = get_logger(__name__)

def make_rule_matrix(k: int, allowed_pairs: List[Tuple[int,int]], allow_self_loops=None) -> np.ndarray:
    R = np.zeros((k, k), dtype=bool)
    for (u, v) in allowed_pairs:
        R[u, v] = True
        R[v, u] = True
    if allow_self_loops is not None:
        for i in range(k):
            R[i, i] = bool(allow_self_loops)
    return R

def enumerate_ring_rows(n: int, k: int, R: np.ndarray) -> List[List[int]]:
    rows = []
    seq = [-1] * n
    def ok_adj(a, b): return R[a, b]
    def dfs(pos):
        if pos == n:
            if ok_adj(seq[-1], seq[0]):
                rows.append(seq.copy()); return
            return
        for val in range(k):
            if pos == 0 or ok_adj(seq[pos-1], val):
                seq[pos] = val
                dfs(pos+1)
                seq[pos] = -1
    dfs(0)
    return rows

def build_row_compat_matrix(rows: List[List[int]], R: np.ndarray) -> np.ndarray:
    m = len(rows)
    T = np.zeros((m, m), dtype=np.int64)
    for i in range(m):
        a = rows[i]
        for j in range(m):
            b = rows[j]
            ok = True
            for c in range(len(a)):
                if not R[a[c], b[c]]:
                    ok = False; break
            if ok: T[i, j] = 1
    return T

def exact_trace_by_transfer(n: int, k: int, R: np.ndarray) -> int:
    """精确：trace(T^n)"""
    rows = enumerate_ring_rows(n, k, R)
    T = build_row_compat_matrix(rows, R)
    M = np.array(T, dtype=object)
    for _ in range(n-1):
        M = M @ T
    return int(np.trace(M))
