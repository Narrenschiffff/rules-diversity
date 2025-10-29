# -*- coding: utf-8 -*-
import numpy as np, torch
from typing import Tuple, List
from .ops import TransferOp
from .logging_setup import get_logger

logger = get_logger(__name__)

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
            logger.info(f"[power] iter={t}  lambda≈{lam:.6e}")
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
    alphas = []; betas = [0.0]
    steps = min(r, iters)
    for j in range(steps):
        z = op.matvec(q)
        alpha = (q @ z).item()
        z = z - alpha * q - (betas[-1] * q_prev if j > 0 else 0.0)
        z = z - (z @ q) * q
        if j > 0: z = z - (z @ q_prev) * q_prev
        beta = z.norm().item()
        alphas.append(alpha); betas.append(beta)
        if verbose: logger.info(f"[lanczos] j={j} alpha={alpha:.6e} beta={beta:.6e}")
        if beta <= 1e-32: break
        q_prev, q = q, z / (beta + 1e-30)
    tdim = len(alphas)
    T_tri = np.zeros((tdim, tdim), dtype=np.float64)
    for i in range(tdim):
        T_tri[i, i] = alphas[i]
        if i + 1 < tdim: T_tri[i, i+1] = betas[i+1]; T_tri[i+1, i] = betas[i+1]
    evals = np.linalg.eigvalsh(T_tri)
    return np.sort(evals)[::-1][:r].tolist()

@torch.no_grad()
def apply_T_power(op, v: torch.Tensor, power: int) -> torch.Tensor:
    y = v
    for _ in range(power): y = op.matvec(y)
    return y

@torch.no_grad()
def estimate_trace_T_power_hutch(op, n_power: int, s: int = 32, seed: int = 0) -> float:
    torch.manual_seed(seed)
    m = op.m; acc = 0.0
    for i in range(s):
        v = torch.empty(m, dtype=op.dtype, device=op.device).uniform_(-1.0, 1.0).sign()
        Tv = apply_T_power(op, v, n_power); acc += (v @ Tv).item()
    return acc / float(s)

@torch.no_grad()
def estimate_trace_T_power_hutchpp(op, n_power: int, s: int = 32, seed: int = 0) -> float:
    torch.manual_seed(seed)
    m = op.m
    s1 = max(2, s // 2); s2 = s - s1
    Omega = torch.empty(m, s1, dtype=op.dtype, device=op.device).uniform_(-1.0, 1.0).sign()
    Y = torch.zeros(m, s1, dtype=op.dtype, device=op.device)
    for j in range(s1): Y[:, j] = apply_T_power(op, Omega[:, j], n_power)
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    trace_lowrank = 0.0
    for j in range(Q.shape[1]):
        qj = Q[:, j]; Tq = apply_T_power(op, qj, n_power)
        trace_lowrank += (qj @ Tq).item()
    trace_res = 0.0
    for i in range(max(1, s2)):
        z = torch.empty(m, dtype=op.dtype, device=op.device).uniform_(-1.0, 1.0).sign()
        p = Q.T @ z; w = z - Q @ p
        Tw = apply_T_power(op, w, n_power); trace_res += (w @ Tw).item()
    if s2 > 0: trace_res /= float(s2)
    return trace_lowrank + trace_res
