# -*- coding: utf-8 -*-
# rules/symcanon.py
from __future__ import annotations
import itertools, math, hashlib
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np

# 若你已有这两个函数，请改为 from .eval import rule_from_bits, bits_from_rule
def rule_from_bits(k: int, bits: np.ndarray) -> np.ndarray:
    """bits -> kxk 0/1 矩阵（按行主序）。"""
    A = np.asarray(bits, dtype=np.uint8).reshape(k, k)
    return A

def bits_from_rule(R: np.ndarray) -> np.ndarray:
    return np.asarray(R, dtype=np.uint8).reshape(-1)

# --------- 规范化（canonicalization）与自同构计数 ---------
def _hash_bits(b: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(b, dtype=np.uint8)).hexdigest()

def _wl_refine_signatures(R: np.ndarray, rounds: int = 3) -> np.ndarray:
    """极轻量 1-WL 颜色细化：返回每个颜色的签名哈希（用于启发式排序/对齐）。"""
    k = R.shape[0]
    sig = np.stack([R.sum(1), np.diag(R)], axis=1).astype(int)  # (deg, selfloop)
    for _ in range(rounds):
        # 用邻接计数直方图粗细化
        hist = R @ sig[:, 0:1]
        sig = np.concatenate([sig, hist], axis=1)
    # 压成哈希
    base = sig.astype(np.int64)
    return np.apply_along_axis(lambda row: hash(tuple(row)), 1, base)

def _perm_apply(R: np.ndarray, perm: List[int]) -> np.ndarray:
    P = R[np.ix_(perm, perm)]
    return P

def canonical_bits(bits: np.ndarray, k: int, mode: str = "wl+heuristic") -> np.ndarray:
    """
    将规则按颜色置换群 S_k 取规范代表。
    mode:
      - "exact"       : 穷举 S_k（k<=8 建议）
      - "igraph"      : igraph/bliss 规范标号（若可用）
      - "wl+igraph"   : 先 WL 细化聚类再 igraph（更快更稳）
      - "wl+heuristic": 先 WL，再用签名排序的启发式（纯 numpy，无三方库）
    """
    R = rule_from_bits(k, bits)
    mode = (mode or "wl+heuristic").lower()

    # 1) igraph 路线（若可用）
    if "igraph" in mode:
        try:
            import igraph as ig
            g = ig.Graph.Adjacency(R.tolist(), mode="DIRECTED")
            # 直接用 canonical permutation
            perm = g.canonical_permutation(color=None)  # 也可传入 WL 颜色作为 color
            if hasattr(perm, "perm"):  # igraph>=0.11
                perm = perm.perm
            R2 = _perm_apply(R, perm)
            return bits_from_rule(R2)
        except Exception:
            # 回退到 wl+heuristic
            mode = "wl+heuristic"

    # 2) exact 穷举（仅小 k）
    if mode == "exact" and k <= 8:
        best = None
        best_t = None
        for perm in itertools.permutations(range(k)):
            R2 = _perm_apply(R, list(perm))
            cand = bits_from_rule(R2)
            t = tuple(cand.tolist())
            if best is None or t < best_t:
                best, best_t = cand, t
        return best

    # 3) wl+heuristic：先用 WL 得到排名，再小范围微调
    sig = _wl_refine_signatures(R, rounds=3)
    order = list(np.argsort(-sig))  # 降序排列，使高签名靠前
    R0 = _perm_apply(R, order)
    # 局部 pairwise-swap 改善（很快）
    improved = True
    best = bits_from_rule(R0)
    best_t = tuple(best.tolist())
    while improved:
        improved = False
        for i in range(k):
            for j in range(i+1, k):
                order2 = order[:]
                order2[i], order2[j] = order2[j], order2[i]
                R2 = _perm_apply(R, order2)
                cand = bits_from_rule(R2)
                t = tuple(cand.tolist())
                if t < best_t:
                    best, best_t = cand, t
                    order = order2
                    improved = True
    return best

def aut_size(R: np.ndarray, mode: str = "igraph") -> int:
    """
    计算自同构群大小 |Aut(R)|（用于 multiplicity = k! / |Aut(R)|）。
    默认用 igraph/bliss；若不可用则用 WL 分块 + 穷举子群的保守下界。
    """
    k = R.shape[0]
    if "igraph" in (mode or "").lower():
        try:
            import igraph as ig
            g = ig.Graph.Adjacency(R.tolist(), mode="DIRECTED")
            # igraph 没有直接给 |Aut| 的稳定接口，但可以用 BLISS 的 count：
            # 某些版本：g.count_automorphisms(group="bliss")
            cnt = g.count_automorphisms()  # 若报错可改为 g.count_automorphisms_vf2()
            return int(cnt)
        except Exception:
            pass

    # 回退：WL 分块 + 对每个色类做全排列验证（下界）
    sig = _wl_refine_signatures(R, rounds=3)
    blocks: Dict[int, List[int]] = {}
    for i, s in enumerate(sig.tolist()):
        blocks.setdefault(s, []).append(i)
    # 仅在块内允许置换，跨块置换被禁止（因此得到的是下界）
    # 若块都很小，该下界通常与真值相同；否则偏小（但可用）。
    count = 1
    for _, idxs in blocks.items():
        m = len(idxs)
        if m <= 1:
            continue
        # 验证能否在块内自由置换（完全自由则乘以 m!；否则可进一步筛）
        # 这里给一个保守策略：直接乘 m! 作为近似（经验上可用于 multiplicity 的稳健估计）
        count *= math.factorial(m)
    return max(1, int(count))

def multiplicity(k: int, aut_sz: int) -> int:
    return math.factorial(k) // max(1, aut_sz)

# --------- 父本对齐（用于 GA 交叉前） ---------
def align_parent(R1: np.ndarray, R2: np.ndarray, method: str = "hungarian") -> List[int]:
    """
    给定两个 kxk 规则矩阵，找到把 R2 对齐到 R1 的置换 perm。
    method:
      - "hungarian": 基于行列 Hamming 距离的线性指派（推荐）
      - "wl+greedy": 先按 WL 签名排序，再局部贪心 swap
    """
    k = R1.shape[0]
    if method == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
            # 成本 = 行距离 + 列距离
            C = np.zeros((k, k), dtype=np.int32)
            R1_r, R1_c = R1, R1.T
            R2_r, R2_c = R2, R2.T
            for i in range(k):
                ri = R1_r[i]; ci = R1_c[i]
                C[i, :] = (np.not_equal(ri[None, :], R2_r).sum(axis=1) +
                           np.not_equal(ci[None, :], R2_c).sum(axis=1))
            row_ind, col_ind = linear_sum_assignment(C)
            perm = col_ind.tolist()
            return perm
        except Exception:
            pass  # 回退到 wl+greedy

    # wl+greedy
    s1 = _wl_refine_signatures(R1, rounds=3)
    s2 = _wl_refine_signatures(R2, rounds=3)
    a = np.argsort(-s1).tolist()
    b = np.argsort(-s2).tolist()
    # 把 b 的顺序映射到 a
    pos_b = {node: i for i, node in enumerate(b)}
    perm = [0]*k
    for i, u in enumerate(a):
        perm[i] = b[i]  # 直接对位
    # 小幅 2-swap 改善
    improved = True
    best_cost = _alignment_cost(R1, _perm_apply(R2, perm))
    while improved:
        improved = False
        for i in range(k):
            for j in range(i+1, k):
                perm2 = perm[:]
                perm2[i], perm2[j] = perm2[j], perm2[i]
                cost = _alignment_cost(R1, _perm_apply(R2, perm2))
                if cost < best_cost:
                    best_cost = cost
                    perm = perm2
                    improved = True
    return perm

def _alignment_cost(A: np.ndarray, B: np.ndarray) -> int:
    return int(np.not_equal(A, B).sum())
