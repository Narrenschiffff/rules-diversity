# -*- coding: utf-8 -*-
"""
NSGA-II (batch) with seeds / heartbeat / parent-aligned crossover (new).

- Canonicalization at init / mutation / crossover / pre-eval.
- Parent alignment: degree -> selfloop -> adjacency-lex stable order.
- Stage1 seeds loading kept here (single source of truth).
- Per-generation append to gen_summary_* and pareto_front_*; heartbeat touch.
"""

from __future__ import annotations
import csv, logging, os, random, time, glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

from . import config
from .eval import (
    canonical_bits, bits_from_rule, rule_from_bits, apply_rule_symmetry,
    evaluate_rules_batch, RowsCacheLRU, summarize_trace_comparison
)
from .utils_io import make_run_tag

logger = logging.getLogger(__name__)
__all__ = ["GAConfig", "ga_search_with_batch"]

# ---------- small utils ----------
def _int_or(v, d): 
    try: return int(v)
    except: return d
def _float_or(v, d):
    try: return float(v)
    except: return d
def _bool_or(v, d):
    if isinstance(v, bool): return v
    if v in (0,1): return bool(v)
    return d
def _device_or(v):
    if isinstance(v, str) and v.lower() in ("cpu","cuda"): return v.lower()
    return "cuda" if torch.cuda.is_available() else "cpu"

def _L_from_k(k: int) -> int:  # bits length = diag + upper-tri
    return k + k*(k-1)//2

def _bits_from_str(s: Optional[str]) -> Optional[np.ndarray]:
    """
    把'010101'转为 np.uint8 数组；空/None/非法串返回 None。
    允许含空格/换行；若出现除{0,1}之外的字符则返回 None。
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # 清理空白
    s = "".join(ch for ch in s if ch in ("0", "1"))
    if not s:
        return None
    try:
        return np.fromiter((1 if ch == "1" else 0 for ch in s), dtype=np.uint8)
    except Exception:
        return None

# ---------- stage1 seeds (single source) ----------
def _load_stage1_seeds(out_csv_dir: str, n: int, k: int, max_seeds: int) -> List[np.ndarray]:
    """
    从 stage1 前沿文件里加载种子（优先 canon，再 raw；都没有就跳过）。
    兼容：stage1_pareto_n{n}_k{k}_raw*.csv / ..._canon*.csv
    同时也容忍含 'rule_bits' 的文件（GA 产物）但不会作为 stage1 种子。
    """
    pats = [
        os.path.join(out_csv_dir, f"stage1_pareto_n{n}_k{k}_canon*.csv"),
        os.path.join(out_csv_dir, f"stage1_pareto_n{n}_k{k}_raw*.csv"),
    ]
    files: List[str] = []
    for p in pats:
        files += glob.glob(p)

    seeds: List[np.ndarray] = []
    seen: set[bytes] = set()

    for fpath in sorted(files):
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                rd = csv.DictReader(fp)
                for r in rd:
                    # 只在这一行选择“一个”字符串；避免对 ndarray 使用 or。
                    s = r.get("rule_bits_canon")
                    if not s:
                        s = r.get("rule_bits_raw")
                    # 兜底：万一上游写了 rule_bits（不应当拿它做 stage1 种子；这里仅容错）
                    if not s:
                        s = r.get("rule_bits")

                    b = _bits_from_str(s)
                    if b is None:
                        continue

                    c = canonical_bits(b, k)
                    key = c.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    seeds.append(c)
                    if len(seeds) >= max_seeds:
                        break
        except Exception:
            logger.exception("[GA] seed load failed: %s", fpath)
        if len(seeds) >= max_seeds:
            break

    return seeds


# ---------- feasibility & parent alignment ----------
def _ensure_minimal_feasible(bits: np.ndarray, k: int) -> np.ndarray:
    R = rule_from_bits(k, bits)
    if int(R.sum()) == 0:
        if k == 1:
            R[0,0] = True
        else:
            R[0,0] = True; R[1,1] = True
    return canonical_bits(bits_from_rule(R), k)

def _stable_node_order(R: np.ndarray) -> List[int]:
    # degree desc, selfloop desc, adjacency row (as '1'/'0') desc
    deg = R.sum(axis=1).astype(int)
    selfloop = np.diag(R).astype(int)
    adj_str = ["".join('1' if x else '0' for x in row.tolist()) for row in R]
    # reverse=True 让“强节点”先排列，便于块对齐
    return sorted(range(R.shape[0]), key=lambda i: (deg[i], selfloop[i], adj_str[i]), reverse=True)

def _remap_bits(bits: np.ndarray, k: int, order: List[int]) -> np.ndarray:
    R = rule_from_bits(k, bits)
    P = R[np.ix_(order, order)]
    return bits_from_rule(P)

def _infer_k_from_bits(bits: np.ndarray) -> int:
    """Infer k from symmetric bit-length L = k + k*(k-1)/2."""
    L = int(bits.size)
    disc = 1 + 8 * L
    k = int((disc ** 0.5 - 1) // 2)
    return k

def _pad_to_len(bits: np.ndarray, L: int) -> np.ndarray:
    if bits.size == L:
        return bits
    out = np.zeros(L, dtype=bits.dtype)
    out[:min(L, bits.size)] = bits[:min(L, bits.size)]
    return out

# ---------- variation operators (with alignment) ----------
def mutate(bits: np.ndarray, p_mut: float, k: int) -> np.ndarray:
    m = bits.copy()
    for i in range(m.size):
        if random.random() < p_mut: m[i] ^= 1
    return _ensure_minimal_feasible(m, k)

def crossover_aligned(a: np.ndarray, b: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """均匀交叉，兼容 perm+swap 压缩后的 k_sym（自动从位长反推 k）。"""
    k_a = _infer_k_from_bits(a); k_b = _infer_k_from_bits(b)
    # 目标位长：覆盖原始 k、两个亲本的位长（含压缩的 k_sym）
    L_target = max(a.size, b.size, _L_from_k(k_a), _L_from_k(k_b), _L_from_k(k))
    a = _pad_to_len(a, L_target)
    b = _pad_to_len(b, L_target)
    k_use = _infer_k_from_bits(np.zeros(L_target, dtype=np.uint8))

    # 构造两亲本共同的稳定顺序，再做均匀交叉（更保留块结构）
    Ra = rule_from_bits(k_use, a); Rb = rule_from_bits(k_use, b)
    order = _stable_node_order(Ra)  # 用 A 的顺序即可
    aa = _remap_bits(a, k_use, order); bb = _remap_bits(b, k_use, order)
    L = aa.size
    mask = np.random.rand(L) < 0.5
    c1 = np.where(mask, aa, bb).astype(np.uint8)
    c2 = np.where(mask, bb, aa).astype(np.uint8)
    c1 = canonical_bits(c1, k_use); c2 = canonical_bits(c2, k_use)
    c1 = _ensure_minimal_feasible(c1, k_use); c2 = _ensure_minimal_feasible(c2, k_use)
    return c1, c2

def init_population(k: int, pop_size: int, bias_sparse: bool = True) -> List[np.ndarray]:
    L = _L_from_k(k)
    pop: List[np.ndarray] = []
    for _ in range(pop_size):
        if bias_sparse:
            arr = np.zeros(L, dtype=np.uint8)
            # 自环更高概率 → 避免 rows_m=0
            for i in range(k): arr[i] = 1 if random.random() < 0.45 else 0
            idx = k
            for i in range(k):
                for j in range(i+1, k):
                    arr[idx] = 1 if random.random() < 0.25 else 0; idx += 1
        else:
            arr = (np.random.rand(L) < 0.5).astype(np.uint8)
        arr = canonical_bits(arr, k)
        arr = _ensure_minimal_feasible(arr, k)
        pop.append(arr)
    return pop

# ---------- NSGA-II essentials ----------
def dominates(a, b):
    f1a, f2a = a["rule_count"], -a["sum_lambda_powers"]
    f1b, f2b = b["rule_count"], -b["sum_lambda_powers"]
    return (f1a <= f1b and f2a <= f2b) and (f1a < f1b or f2a < f2b)

def nondominated_sort(pop_fits: List[Dict]) -> List[List[int]]:
    N = len(pop_fits)
    S = [set() for _ in range(N)]
    n_dom = [0]*N
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p==q: continue
            if dominates(pop_fits[p], pop_fits[q]): S[p].add(q)
            elif dominates(pop_fits[q], pop_fits[p]): n_dom[p] += 1
        if n_dom[p]==0: fronts[0].append(p)
    i=0
    while fronts[i]:
        Q=[]
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q]==0: Q.append(q)
        i += 1
        fronts.append(Q)
    fronts.pop()
    return fronts

def crowding_distance(front: List[int], pop_fits: List[Dict]) -> Dict[int, float]:
    if not front: return {}
    distances = {i:0.0 for i in front}
    vals1 = [(i, pop_fits[i]["rule_count"]) for i in front]        # smaller better
    vals2 = [(i, pop_fits[i]["sum_lambda_powers"]) for i in front] # larger better
    for vals, reverse in [(vals1, False), (vals2, True)]:
        vs = sorted(vals, key=lambda x: x[1], reverse=reverse)
        distances[vs[0][0]] = float("inf"); distances[vs[-1][0]] = float("inf")
        vmin, vmax = vs[-1][1], vs[0][1]
        rng = (vmax - vmin) if vmax>vmin else 1.0
        for j in range(1, len(vs)-1):
            prev_v, next_v = vs[j-1][1], vs[j+1][1]
            distances[vs[j][0]] += (next_v - prev_v) / rng
    return distances

class GAConfig:
    def __init__(self,
                 pop_size=32, generations=12,
                 p_mut=0.08, p_cx=0.85, elite_keep=6,
                 device=None,
                 use_lanczos=True, r_vals=3, power_iters=50,
                 trace_mode="hutchpp", hutch_s=24,
                 lru_rows_capacity=128, batch_streams=2,
                 progress_every=2, fast_eval=False,
                 seed_from_stage1=False, max_stage1_seeds=256,
                 sym_mode: str = "perm",
                 enable_exact: bool = True,
                 enable_spectral: bool = True,
                 exact_threshold: str | int | float = "nk<=12",
                 boundary: str | None = None,
                 cache_dir=None,
                 use_cache: bool = True,
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
        self.progress_every = progress_every
        self.fast_eval = fast_eval
        self.seed_from_stage1 = seed_from_stage1
        self.max_stage1_seeds = max_stage1_seeds
        self.sym_mode = sym_mode
        self.enable_exact = enable_exact
        self.enable_spectral = enable_spectral
        self.exact_threshold = exact_threshold
        self.boundary = (boundary or config.BOUNDARY_MODE)
        self.cache_dir = cache_dir if cache_dir is not None else config.EVAL_CACHE_DIR
        self.use_cache = use_cache

# ---------- CSV appenders ----------
def _append_front_rows_csv(csv_path_front: str, tag: str, n: int, k: int,
                           pop_bits: List[np.ndarray], fits: List[Dict], front0_idx: List[int]) -> None:
    front0_set = set(front0_idx)
    with open(csv_path_front, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for i, (bits, fit) in enumerate(zip(pop_bits, fits)):
            lam_top2 = fit.get("lambda_top2", (0.0,0.0))
            try:
                lam_top2_str = f"({float(lam_top2[0]):.6e},{float(lam_top2[1]):.6e})"
            except Exception:
                lam_top2_str = "(0.000000e+00,0.000000e+00)"
            w.writerow([
                tag, n, k,
                "".join(map(str, bits.tolist())),
                int(fit.get("rule_count", 0)),
                int(fit.get("rows_m", 0)),
                f"{float(fit.get('lambda_max', 0.0)):.6e}",
                lam_top2_str,
                f"{float(fit.get('spectral_gap', 0.0)):.6e}",
                f"{float(fit.get('sum_lambda_powers', -1e300)):.6e}",
                1 if i in front0_set else 0,
                int(fit.get("active_k", 0)),
                int(fit.get("active_k_raw", fit.get("active_k", 0))),
                int(fit.get("k_sym", k)),
                fit.get("sym_mode", "perm"),
                f"{float(fit.get('lower_bound', 0.0)):.6e}",
                f"{float(fit.get('upper_bound', 0.0)):.6e}",
                f"{float(fit.get('lower_bound_raw', 0.0)):.6e}",
                f"{float(fit.get('upper_bound_raw', 0.0)):.6e}",
                f"{float(fit.get('upper_bound_raw_gersh', 0.0)):.6e}",
                f"{float(fit.get('upper_bound_raw_maxdeg', 0.0)):.6e}",
                fit.get("archetype_tags",""),
                ("" if (fit.get("exact_Z","")== "") else str(int(fit.get("exact_Z")))),
                fit.get("trace_exact", ""),
                fit.get("trace_estimate", ""),
                fit.get("trace_error", ""),
                fit.get("trace_error_rel", ""),
                fit.get("eval_note", ""),
            ])

# ---------- main ----------
def ga_search_with_batch(n: int, k: int, ga_conf: GAConfig, out_csv_dir: str="./out_csv", run_tag: str=None):
    os.makedirs(out_csv_dir, exist_ok=True)
    tag = run_tag or make_run_tag(n, k, add_timestamp=False)
    csv_front = os.path.join(out_csv_dir, f"pareto_front_{tag}.csv")
    csv_gen   = os.path.join(out_csv_dir, f"gen_summary_{tag}.csv")
    heartbeat = os.path.join(out_csv_dir, f"heartbeat_{tag}.txt")

    # headers
    with open(csv_front, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["run_tag","n","k","rule_bits","rule_count","rows_m",
                    "lambda_max","lambda_top2","spectral_gap",
                    "sum_lambda_powers","is_front0",
                    "active_k","active_k_raw","k_sym","sym_mode",
                    "lower_bound","upper_bound",
                    "lower_bound_raw","upper_bound_raw",
                    "upper_bound_raw_gersh","upper_bound_raw_maxdeg",
                    "archetype_tags","exact_Z",
                    "trace_exact","trace_estimate","trace_error","trace_error_rel","eval_note"])
    with open(csv_gen, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["run_tag","n","k","generation","front0_size",
                    "best_sample_R","best_sample_sumlam",
                    "pop_size","device","trace_mode","sym_mode"])

    # normalize config
    device      = _device_or(getattr(ga_conf, "device", None))
    use_lanczos = _bool_or(getattr(ga_conf, "use_lanczos", True), True)
    r_vals      = _int_or(getattr(ga_conf, "r_vals", 3), 3)
    power_iters = _int_or(getattr(ga_conf, "power_iters", 50), 50)
    trace_mode  = getattr(ga_conf, "trace_mode", "hutchpp") or "hutchpp"
    hutch_s     = _int_or(getattr(ga_conf, "hutch_s", 24), 24)
    lru_cap     = _int_or(getattr(ga_conf, "lru_rows_capacity", 128), 128)
    max_streams = _int_or(getattr(ga_conf, "batch_streams", 2), 2)
    generations = _int_or(getattr(ga_conf, "generations", 12), 12)
    elite_keep  = _int_or(getattr(ga_conf, "elite_keep", 6), 6)
    p_cx        = _float_or(getattr(ga_conf, "p_cx", 0.85), 0.85)
    p_mut       = _float_or(getattr(ga_conf, "p_mut", 0.08), 0.08)
    pop_size    = _int_or(getattr(ga_conf, "pop_size", 32), 32)
    progress_every   = _int_or(getattr(ga_conf, "progress_every", 2), 2)
    fast_eval       = _bool_or(getattr(ga_conf, "fast_eval", False), False)
    seed_from_stage1= _bool_or(getattr(ga_conf, "seed_from_stage1", False), False)
    max_seeds       = _int_or(getattr(ga_conf, "max_stage1_seeds", 256), 256)
    sym_mode        = getattr(ga_conf, "sym_mode", "perm") or "perm"
    enable_exact    = _bool_or(getattr(ga_conf, "enable_exact", True), True)
    enable_spectral = _bool_or(getattr(ga_conf, "enable_spectral", True), True)
    exact_threshold = getattr(ga_conf, "exact_threshold", "nk<=12")
    boundary        = getattr(ga_conf, "boundary", config.BOUNDARY_MODE) or config.BOUNDARY_MODE
    cache_dir       = getattr(ga_conf, "cache_dir", config.EVAL_CACHE_DIR)
    use_cache       = _bool_or(getattr(ga_conf, "use_cache", True), True)

    if fast_eval or device=="cpu":
        r_vals = min(r_vals, 2); power_iters = min(power_iters, 16); hutch_s = min(hutch_s, 8)

    logger.info(
        f"GA start | n={n}, k={k}, device={device}, sym={sym_mode}, pop={pop_size}, gens={generations}, "
        f"fast_eval={fast_eval}, seed_from_stage1={seed_from_stage1}, exact={enable_exact}, spectral={enable_spectral}, "
        f"exact_th={exact_threshold}, boundary={boundary}, cache_dir={cache_dir}, cache={'on' if use_cache else 'off'}")

    # init population
    pop: List[np.ndarray] = []
    if seed_from_stage1:
        seeds = _load_stage1_seeds(out_csv_dir, n, k, max_seeds)
        random.shuffle(seeds)
        take = min(len(seeds), max(2, pop_size//2))
        pop.extend(seeds[:take])
        logger.info(f"Loaded {len(seeds)} stage1 seeds; take {take}.")
    if len(pop) < pop_size:
        pop.extend(init_population(k, pop_size - len(pop), bias_sparse=True))

    # cache
    cache: Dict[bytes, Dict] = {}
    rows_lru = RowsCacheLRU(capacity=lru_cap)

    def eval_batch(bits_batch: List[np.ndarray]):
        normed = []
        keys = []
        results = [None]*len(bits_batch)
        miss_bits, miss_pos = [], []
        for i, bb in enumerate(bits_batch):
            sym_b, _, _, _ = apply_rule_symmetry(bb, k, sym_mode)
            normed.append(sym_b)
            key = sym_b.tobytes()
            keys.append(key)
            if key in cache:
                results[i] = cache[key]
            else:
                miss_bits.append(bb)
                miss_pos.append(i)
        if miss_bits:
            # 先试主设备（如 CUDA），失败时重试一次主设备，最后才回退 CPU，尽量保持 GPU 路径以提高吞吐。
            try:
                outs = evaluate_rules_batch(n, k, miss_bits,
                                            sym_mode=sym_mode,
                                            boundary=boundary,
                                            device=device, use_lanczos=use_lanczos,
                                            r_vals=r_vals, power_iters=power_iters,
                                            trace_mode=trace_mode, hutch_s=hutch_s,
                                            lru_rows=rows_lru, max_streams=max_streams,
                                            enable_exact=enable_exact,
                                            enable_spectral=enable_spectral,
                                            exact_threshold=exact_threshold,
                                            cache_dir=cache_dir,
                                            use_cache=use_cache)
            except Exception:
                outs = None
                if device == "cuda":
                    logger.warning("evaluate_rules_batch failed on CUDA; retrying on CUDA once", exc_info=True)
                    try:
                        outs = evaluate_rules_batch(n, k, miss_bits,
                                                    sym_mode=sym_mode,
                                                    boundary=boundary,
                                                    device=device, use_lanczos=use_lanczos,
                                                    r_vals=r_vals, power_iters=power_iters,
                                                    trace_mode=trace_mode, hutch_s=hutch_s,
                                                    lru_rows=rows_lru, max_streams=max_streams,
                                                    enable_exact=enable_exact,
                                                    enable_spectral=enable_spectral,
                                                    exact_threshold=exact_threshold,
                                                    cache_dir=cache_dir,
                                                    use_cache=use_cache)
                    except Exception:
                        logger.warning("second CUDA attempt failed; fallback to CPU", exc_info=True)
                if outs is None:
                    outs = evaluate_rules_batch(n, k, miss_bits,
                                                sym_mode=sym_mode,
                                                boundary=boundary,
                                                device="cpu", use_lanczos=use_lanczos,
                                                r_vals=r_vals, power_iters=power_iters,
                                                trace_mode=trace_mode, hutch_s=hutch_s,
                                                lru_rows=rows_lru, max_streams=max_streams,
                                                enable_exact=enable_exact,
                                                enable_spectral=enable_spectral,
                                                exact_threshold=exact_threshold,
                                                cache_dir=cache_dir,
                                                use_cache=use_cache)
            for pos, fit in zip(miss_pos, outs):
                key = keys[pos]
                cache[key] = fit; results[pos] = fit

        # CUDA 在少量环境下可能返回 None（如内核异常）。先尝试再跑一次 CUDA，仍为 None 时才回退 CPU。
        if any(r is None for r in results):
            retry_bits = [bits_batch[i] for i, r in enumerate(results) if r is None]
            retry_pos  = [i for i, r in enumerate(results) if r is None]
            outs = None
            if device == "cuda":
                logger.warning("found None fits on CUDA; retrying those on CUDA")
                try:
                    outs = evaluate_rules_batch(n, k, retry_bits,
                                                sym_mode=sym_mode,
                                                boundary=boundary,
                                                device=device, use_lanczos=use_lanczos,
                                                r_vals=r_vals, power_iters=power_iters,
                                                trace_mode=trace_mode, hutch_s=hutch_s,
                                                lru_rows=rows_lru, max_streams=max_streams,
                                                enable_exact=enable_exact,
                                                enable_spectral=enable_spectral,
                                                exact_threshold=exact_threshold,
                                                cache_dir=cache_dir,
                                                use_cache=use_cache)
                except Exception:
                    logger.warning("CUDA retry for None fits failed; fallback to CPU", exc_info=True)
            if outs is None:
                logger.warning("fallback CPU eval for %d None fits", len(retry_bits))
                outs = evaluate_rules_batch(n, k, retry_bits,
                                            sym_mode=sym_mode,
                                            boundary=boundary,
                                            device="cpu", use_lanczos=use_lanczos,
                                            r_vals=r_vals, power_iters=power_iters,
                                            trace_mode=trace_mode, hutch_s=hutch_s,
                                            lru_rows=rows_lru, max_streams=max_streams,
                                            enable_exact=enable_exact,
                                            enable_spectral=enable_spectral,
                                            exact_threshold=exact_threshold,
                                            cache_dir=cache_dir,
                                            use_cache=use_cache)
            for pos, fit in zip(retry_pos, outs):
                key = keys[pos]
                cache[key] = fit; results[pos] = fit
        if enable_exact and enable_spectral:
            try:
                summarize_trace_comparison([r for r in results if isinstance(r, dict)], logger=logger)
            except Exception:
                logger.debug("compare summary failed", exc_info=True)
        return results, normed

    # generations
    for gen in range(generations):
        t0 = time.time()
        fits, pop = eval_batch(pop)
        fronts = nondominated_sort(fits)
        front0 = fronts[0] if fronts else []
        dists  = crowding_distance(front0, fits)

        # gen summary
        best_points = [(fits[i]["rule_count"], fits[i]["sum_lambda_powers"]) for i in front0] if front0 else []
        best_points.sort(key=lambda x: (x[0], -x[1]))
        best_R = best_points[0][0] if best_points else None
        best_sum = best_points[0][1] if best_points else None
        with open(csv_gen, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([tag, n, k, gen, len(front0), best_R, best_sum, pop_size, device, trace_mode, sym_mode])

        # append this generation to front csv
        _append_front_rows_csv(csv_front, tag, n, k, pop_bits=pop, fits=fits, front0_idx=front0)

        # heartbeat
        if progress_every and (gen % progress_every == 0):
            with open(heartbeat, "w", encoding="utf-8") as hb:
                hb.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} GEN={gen}\n")
            print(f"[GA] GEN {gen:02d} | sym={sym_mode} | front0={len(front0):3d} | best=({best_R},{best_sum:.3e}) | pop={pop_size} | dt={time.time()-t0:.2f}s", flush=True)
        else:
            logger.info(f"[GEN {gen:02d}] sym={sym_mode} front0={len(front0)} best=({best_R},{best_sum:.3e})")

        # selection -> offspring
        new_pop: List[np.ndarray] = []
        if front0:
            elite_sorted = sorted(front0, key=lambda i: dists.get(i, 0.0), reverse=True)
            for i in elite_sorted[:elite_keep]: new_pop.append(pop[i].copy())

        layer_of = {}
        for rank, fr in enumerate(fronts):
            for idx in fr: layer_of[idx] = rank
        d_map = {}
        for fr in fronts: d_map.update(crowding_distance(fr, fits))

        def tournament():
            a, b = random.sample(range(len(pop)), 2)
            la, lb = layer_of.get(a, 10**9), layer_of.get(b, 10**9)
            if la < lb: return pop[a]
            if lb < la: return pop[b]
            return pop[a] if d_map.get(a,0.0) >= d_map.get(b,0.0) else pop[b]

        while len(new_pop) < pop_size:
            if random.random() < p_cx and len(new_pop)+1 < pop_size:
                p1, p2 = tournament(), tournament()
                c1, c2 = crossover_aligned(p1, p2, k)
                c1 = canonical_bits(mutate(c1, p_mut, k), k)
                c2 = canonical_bits(mutate(c2, p_mut, k), k)
                new_pop += [c1, c2]
            else:
                p = tournament()
                c = canonical_bits(mutate(p, p_mut, k), k)
                new_pop.append(c)
        pop = new_pop[:pop_size]

    # final eval
    fits, pop = eval_batch(pop)
    fronts = nondominated_sort(fits)
    pareto_idx = fronts[0] if fronts else []
    pareto = [(pop[i], fits[i]) for i in pareto_idx]
    pareto_sorted = sorted(pareto, key=lambda x: (x[1]["rule_count"], -x[1]["sum_lambda_powers"]))
    logger.info("=== Final Pareto (top10) ===")
    for i, (b, ft) in enumerate(pareto_sorted[:10]):
        logger.info(f"[FINAL {i:02d}] |R|={ft['rule_count']:3d}, rows_m={ft['rows_m']:7d}, λ1≈{ft['lambda_max']:.3e}, tr≈{ft['sum_lambda_powers']:.3e}")
    return pareto_sorted, csv_front, csv_gen
