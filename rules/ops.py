# -*- coding: utf-8 -*-
import itertools, numpy as np, torch
from typing import List, Tuple, Optional
from collections import OrderedDict
from .logging_setup import get_logger

logger = get_logger(__name__)

def rule_from_bits(k: int, bits: np.ndarray) -> np.ndarray:
    L_diag = k; L_upper = k*(k-1)//2
    assert bits.size == L_diag + L_upper
    R = np.zeros((k,k), dtype=bool)
    for i in range(k): R[i,i] = bool(bits[i])
    idx = k
    for i in range(k):
        for j in range(i+1, k):
            val = bool(bits[idx]); idx += 1
            R[i,j] = val; R[j,i] = val
    return R

def bits_from_rule(R: np.ndarray) -> np.ndarray:
    k = R.shape[0]; out = []
    for i in range(k): out.append(1 if R[i,i] else 0)
    for i in range(k):
        for j in range(i+1, k): out.append(1 if R[i,j] else 0)
    return np.array(out, dtype=np.uint8)

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
    # heuristic for k>8
    deg = R.sum(axis=1).astype(int)
    selfloop = np.diag(R).astype(int)
    adj_str = ["".join('1' if x else '0' for x in row.tolist()) for row in R]
    order = sorted(range(k), key=lambda i: (deg[i], selfloop[i], adj_str[i]), reverse=True)
    P = R[np.ix_(order, order)]
    return bits_from_rule(P)

class RowsCacheLRU:
    def __init__(self, capacity=128):
        self.capacity = capacity; self.od = OrderedDict()
    def get(self, key: bytes):
        if key in self.od:
            v = self.od.pop(key); self.od[key] = v; return v
        return None
    def put(self, key: bytes, val: np.ndarray):
        if key in self.od: self.od.pop(key)
        elif len(self.od) >= self.capacity: self.od.popitem(last=False)
        self.od[key] = val

def enumerate_ring_rows_fast(n: int, k: int, R: np.ndarray, batch_limit: int = 200_000) -> np.ndarray:
    assert k <= 16, "k>16 需改为 64-bit 掩码实现"
    allowed_next = np.zeros(k, dtype=np.uint32)
    allowed_prev = np.zeros(k, dtype=np.uint32)
    for c in range(k):
        mask = 0; row = R[c]
        for d in range(k):
            if row[d]: mask |= (1 << d)
        allowed_next[c] = np.uint32(mask)
    for d in range(k):
        mask = 0; col = R[:, d]
        for c in range(k):
            if col[c]: mask |= (1 << c)
        allowed_prev[d] = np.uint32(mask)

    out_chunks: List[np.ndarray] = []; cur_batch: List[List[int]] = []
    def flush_batch():
        nonlocal cur_batch
        if cur_batch:
            out_chunks.append(np.array(cur_batch, dtype=np.int16))
            cur_batch = []

    path = np.empty(n, dtype=np.int16)
    def dfs_build(col: int, last_c: int, first_c: int):
        nonlocal cur_batch
        if col == n:
            if ((int(allowed_next[last_c]) >> first_c) & 1) != 0:
                cur_batch.append(path.copy().tolist())
                if len(cur_batch) >= batch_limit: flush_batch()
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
            dfs_build(2, c2, s)
            mask2 ^= lsb

    flush_batch()
    return np.empty((0, n), dtype=np.int16) if not out_chunks else np.vstack(out_chunks)

class TransferOp:
    def __init__(self, rows_np: np.ndarray, R_np: np.ndarray,
                 device: str = "cuda", dtype: torch.dtype = torch.float64,
                 block_size_j: Optional[int] = None, block_size_i: Optional[int] = None):
        self.device = torch.device(device); self.dtype = dtype
        self.rows = torch.from_numpy(rows_np.astype(np.int64)).to(self.device)
        self.m, self.n = self.rows.shape; self.k = R_np.shape[0]
        self.R = torch.from_numpy(R_np.astype(np.bool_)).to(self.device)

        if block_size_i is None:
            block_size_i = min(self.m, 2048)
        if block_size_j is None:
            block_size_j = 4096 if self.m >= 10000 else 2048
        if self.m >= 200000:
            block_size_j = 1024; block_size_i = min(block_size_i, 1024)
        self.block_size_i = int(block_size_i); self.block_size_j = int(block_size_j)

    @torch.no_grad()
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape == (self.m,)
        rows = self.rows; R = self.R; m, n = self.m, self.n
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
                    ai = aL[:, c].unsqueeze(1)
                    bj = bL[:, c].unsqueeze(0)
                    comp &= R[ai, bj]
                    if not comp.any(): break
                if comp.any():
                    y_block += comp.to(self.dtype) @ x[j0:j1].to(self.dtype)
            y[i0:i1] += y_block
        return y
