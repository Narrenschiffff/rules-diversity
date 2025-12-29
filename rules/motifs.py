# -*- coding: utf-8 -*-
"""Motif analysis: knee & MUR finding on Pareto fronts and structural feature extraction."""
from __future__ import annotations
import os, re, math, csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np

# -------- style (soft dependency) --------
try:
    from .viz import apply_style
except Exception:
    def apply_style(style: str = "default"):  # type: ignore
        pass

# -------- I/O utils (your project) --------
from .utils_io import ensure_dir, expand_globs, load_csv_rows, parse_nk_from_filename, to_int

# ==============================================================
# bits -> rule matrix (prefers eval.rule_from_bits; fallback ok)
# ==============================================================
_HAS_EVAL = False
try:
    from .eval import rule_from_bits as _rule_from_bits_eval
    _HAS_EVAL = True
except Exception:
    _HAS_EVAL = False

def _bits_len_for_k(k:int)->int:
    # diag k + undirected upper-tri k*(k-1)/2
    return k + k*(k-1)//2

def _rule_from_bits_fallback(k:int, bits:np.ndarray)->np.ndarray:
    L = _bits_len_for_k(k)
    assert bits.size == L, f"bits length {bits.size} != expected {L} for k={k}"
    R = np.zeros((k,k), dtype=bool)
    p = 0
    # diagonal: self-loops
    for i in range(k):
        R[i,i] = bool(bits[p]); p+=1
    # upper triangle (undirected)
    for i in range(k):
        for j in range(i+1,k):
            b = bool(bits[p]); p+=1
            if b:
                R[i,j] = True; R[j,i] = True
    return R

def rule_from_bits_any(k:int, bits:np.ndarray)->np.ndarray:
    if _HAS_EVAL:
        return _rule_from_bits_eval(k, bits)
    return _rule_from_bits_fallback(k, bits)

# =========================================
# Knee & MUR: index finders (discrete-safe)
# =========================================
def knee_second_diff(xs, ys, logy=True)->Optional[int]:
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if logy:
        ys = np.log(np.maximum(ys, 1e-300))
    if len(xs) < 3:
        return None
    d2 = ys[2:] - 2*ys[1:-1] + ys[:-2]
    idx = int(np.argmax(d2))
    return idx + 1

def knee_lcurve(xs, ys, logxy=True)->Optional[int]:
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if logxy:
        xs = np.log(np.maximum(xs, 1e-9))
        ys = np.log(np.maximum(ys, 1e-300))
    if len(xs) < 3:
        return None
    x0,y0 = xs[0],ys[0]; x1,y1 = xs[-1],ys[-1]
    vx,vy = x1-x0, y1-y0
    vnorm = math.hypot(vx,vy) + 1e-15
    dmax, imax = -1.0, None
    for i in range(len(xs)):
        wx,wy = xs[i]-x0, ys[i]-y0
        area2 = abs(vx*wy - vy*wx)
        d = area2 / vnorm
        if d > dmax:
            dmax, imax = d, i
    return imax

def robust_knee(xs, ys, logy=True)->Optional[int]:
    """Fuse 2nd-diff and L-curve; prefer earlier if tie ±1."""
    i2 = knee_second_diff(xs, ys, logy=logy)
    il = knee_lcurve(xs, ys, logxy=True)
    if i2 is None and il is None:
        return None
    if i2 is None:
        return il
    if il is None:
        return i2
    return i2 if (abs(i2-il) <= 1 and xs[i2] <= xs[il]) else il

def mur_index(xs, ys, logy=True)->Optional[int]:
    """
    边际单位回报（MUR）点：相邻点局部斜率最大的那一段的“右端点”索引。
    - xs: 复杂度（|R| 等）
    - ys: 目标（Y 或 trace/Z）
    - logy=True：用 log(y) 做斜率，等价于“单位复杂度的相对收益”
    返回：索引 j （对应 xs[j], ys[j]）
    """
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 2:
        return None
    Y = np.log(np.maximum(ys, 1e-300)) if logy else ys
    dx = np.maximum(xs[1:] - xs[:-1], 1e-12)
    slope = (Y[1:] - Y[:-1]) / dx
    j = int(np.argmax(slope))
    return j + 1

# =========================================
# Structural feature extractors
# =========================================
import numpy as np
from typing import Tuple, Dict

def _triangle_count(G: np.ndarray) -> int:
    """无向简单图三角形个数。要求 G 对称、对角为 0。"""
    k = G.shape[0]
    tri = 0
    for i in range(k):
        for j in range(i + 1, k):
            if not G[i, j]:
                continue
            # 统计 i,j 的共同邻居（只数 index > j，避免重复）
            for l in range(j + 1, k):
                if G[i, l] and G[j, l]:
                    tri += 1
    return tri

def _c4_count(G: np.ndarray) -> int:
    """
    4-环个数（不要求“诱导”）：#C4 = (1/2) * sum_{i<j} C( common_nbr(i,j), 2 ).
    注意：必须是简单图（对角 0、无多重边），G 为 bool 或 0/1。
    """
    A = G.astype(np.int8).copy()
    # 确保对角为 0（避免把自环当成“共同邻居”生成伪 4 环）
    np.fill_diagonal(A, 0)
    S = A @ A  # S[i,j] = 两步可达条数 = 共同邻居数（在简单图等于共同邻居数）
    k = A.shape[0]
    twice = 0
    for i in range(k):
        for j in range(i + 1, k):
            c = int(S[i, j])
            if c >= 2:
                twice += c * (c - 1) // 2  # 组合数 C(c,2)
    # 每个 4 环会在两对对顶点上各被计一次，所以除以 2
    return int(twice // 2)

def _c5_count(G: np.ndarray) -> int:
    """
    5-环个数（不要求“诱导”）：采用无向简单图的 DFS 回路计数（避免重复，去方向与起点重数）。
    复杂度 O(k * d^4)，k<=~10 时可接受；你的规模（小 k）足够稳。
    计数细节：每个 5 环会被 10 次（5 个起点 × 2 个方向）访问，因此最终除以 10。
    """
    A = G.astype(bool).copy()
    np.fill_diagonal(A, False)
    n = A.shape[0]
    if n < 5:
        return 0

    count = 0
    visited = np.zeros(n, dtype=bool)

    def dfs(start: int, u: int, depth: int):
        nonlocal count
        if depth == 4:
            # 长度 4 的路径，再加一条边回到 start 构成 5 环
            if A[u, start]:
                count += 1
            return
        # 为避免重复，把路径上节点严格递增起点约束：只从 >= start 的节点扩展；
        # 再配合“除以 10”去重（起点与方向），可以稳定计数。
        for v in range(start, n):
            if not A[u, v] or visited[v]:
                continue
            visited[v] = True
            dfs(start, v, depth + 1)
            visited[v] = False

    for s in range(n):
        visited[s] = True
        for v in range(s + 1, n):  # 从 s 的较大编号邻居出发，减少重复
            if not A[s, v]:
                continue
            visited[v] = True
            dfs(s, v, 2)  # 已经用掉 s->v 两个节点，深度=2
            visited[v] = False
        visited[s] = False

    # 每个 5 环被计数 10 次（5 个起点 × 2 个方向）
    return count // 10

def _odd_girth(G: np.ndarray) -> float:
    """启发式：检测是否存在奇环；若有返回 3（最短奇环至少为 3），否则返回 inf。"""
    k = G.shape[0]
    INF = 1e9
    best = INF
    for s in range(k):
        color = {s: 0}
        q = [s]
        while q:
            u = q.pop(0)
            for v in range(k):
                if not G[u, v]:
                    continue
                if v not in color:
                    color[v] = color[u] ^ 1
                    q.append(v)
                elif color[v] == color[u]:
                    best = min(best, 3.0)
    return best if best < INF else float("inf")

def _clustering_coeff(G: np.ndarray) -> float:
    """平均聚类系数（逐点的邻接子图三角率均值）。"""
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

def _kcore_number(G: np.ndarray) -> int:
    """k-core 指数（传统 peeling）。"""
    k = G.shape[0]
    deg = G.sum(axis=1).astype(int).tolist()
    alive = [True] * k
    core = 0
    changed = True
    while changed:
        changed = False
        for v in range(k):
            if not alive[v]:
                continue
            if deg[v] < core:
                alive[v] = False
                changed = True
                for u in range(k):
                    if alive[u] and G[u, v]:
                        deg[u] -= 1
        if not changed:
            cand = min([deg[v] for v in range(k) if alive[v]] + [10**9])
            if cand > core and cand < 10**9:
                core = cand
                changed = True
    return core

def _spectral_feats(G: np.ndarray) -> Tuple[float, float, float, float]:
    """邻接谱：λ1, λ2, gap；拉普拉斯代数连通度（λ2(L)）。"""
    if not G.any():
        return 0.0, 0.0, 0.0, 0.0
    A = G.astype(float)
    # 邻接谱
    w = np.linalg.eigvalsh(A)
    lam1 = float(w[-1])
    lam2 = float(w[-2]) if w.size >= 2 else 0.0
    gap = lam1 - lam2
    # 拉普拉斯谱
    D = np.diag(A.sum(axis=1))
    L = D - A
    wl = np.linalg.eigvalsh(L)
    alg_con = float(sorted(wl)[1]) if wl.size >= 2 else 0.0
    return lam1, lam2, gap, alg_con

def extract_features_from_R(R: np.ndarray) -> Dict[str, float]:
    """把规则矩阵 R（含对角自环信息）转为无向简单图 G，再抽取结构+谱特征。"""
    k = R.shape[0]
    G = R.copy().astype(bool)
    # 把对角（自环）从图里去掉，仅用于结构统计；自环数量单独作为特征。
    np.fill_diagonal(G, False)

    deg = G.sum(axis=1)
    tri = _triangle_count(G)
    c4 = _c4_count(G)
    c5 = _c5_count(G)
    oddg = _odd_girth(G)
    Cbar = _clustering_coeff(G)
    kcore = _kcore_number(G)
    lam1, lam2, gap, alg = _spectral_feats(G)

    feats = dict(
        deg_max=float(deg.max() if deg.size else 0.0),
        deg_mean=float(deg.mean() if deg.size else 0.0),
        deg_std=float(deg.std(ddof=0) if deg.size else 0.0),
        diag_cnt=float(np.trace(R)),  # 自环数（留作单独特征）
        selfloop_rich=float(np.trace(R) >= max(1, k // 2)),
        tri=float(tri),
        c4=float(c4),
        c5=float(c5),
        odd_girth=float(oddg),
        near_bip_chord=float((tri + c5) <= 2),  # 近二分（只看 3/5 奇环的启发式）
        clustering=float(Cbar),
        kcore=float(kcore),
        lambda1=lam1,
        lambda2=lam2,
        gap=gap,
        lap_algebraic=alg,
        star_core=float(deg.max() >= max(k - 1, 2)),  # 是否存在“星核级”中心
    )
    return feats

def edge_criticality_scores(R: np.ndarray):
    """边关键性打分：基于与该边相关的三角、4-环与“二步合流”。”
    """
    k = R.shape[0]
    G = R.copy().astype(bool)
    np.fill_diagonal(G, False)
    feats = []
    for u in range(k):
        for v in range(u + 1, k):
            if not G[u, v]:
                continue
            tri = int(np.logical_and(G[u], G[v]).sum())
            # 通过共同邻居对数近似估 c4 贡献
            c4 = 0
            for x in range(k):
                if x == u or x == v or not G[u, x]:
                    continue
                for y in range(k):
                    if y <= x or y in (u, v):
                        continue
                    if G[v, y] and G[x, y]:
                        c4 += 1
            # 统计 u-v 之间的长度 2 路径条数（“奇环弦机会”的近似）
            oddchord = 0
            for w in range(k):
                if w in (u, v):
                    continue
                if G[u, w] and G[w, v]:
                    oddchord += 1
            score = tri + 0.5 * c4 + oddchord
            feats.append(((u, v), float(score), dict(tri=tri, c4=c4, oddchord=oddchord)))
    feats.sort(key=lambda t: (-t[1], -(t[2]["tri"]), -(t[2]["c4"])))
    return feats


# =========================================
# Row utilities (grouping, metric, buckets)
# =========================================
def _y_metric(row: dict) -> float:
    """
    统一计算 Y：
    - 优先 Z_exact（精确解）
    - 若无，则取 Z_est / sum_lambda_powers / objective 等近似
    """
    for key in ("Z_exact", "Z_est", "objective_penalized", "objective_raw", "sum_lambda_powers", "objective", "y"):
        val = row.get(key, "")
        if val not in ("", None, "nan", "NaN"):
            try:
                return float(val)
            except Exception:
                pass
    return float("nan")

def _extract_spectral_metrics(r: dict, k: int) -> Tuple[float, float, float]:
    """提取或重算 λ1、λ2、gap"""
    import numpy as np
    try:
        l1 = float(r.get("lambda1", np.nan))
        l2 = float(r.get("lambda2", np.nan))
        gap = float(r.get("gap", np.nan))
        if not np.isfinite(gap) and np.isfinite(l1) and np.isfinite(l2):
            gap = l1 - l2
        # 若都缺，则通过 rule_bits 反推谱
        if not np.isfinite(l1) or not np.isfinite(l2):
            bits_s = _normalize_bits_field(r)
            if bits_s:
                bits = np.fromiter((1 if ch == "1" else 0 for ch in bits_s), dtype=np.uint8)
                R = rule_from_bits_any(k, bits)
                from .motifs import extract_features_from_R
                feats = extract_features_from_R(R)
                l1 = feats.get("lambda1", np.nan)
                l2 = feats.get("lambda2", np.nan)
                gap = feats.get("gap", np.nan)
        return l1, l2, gap
    except Exception:
        return (np.nan, np.nan, np.nan)


def _is_front_row(row:dict)->bool:
    # Respect is_front0 if present; otherwise assume already filtered front CSV
    if "is_front0" in row:
        try:
            return int(row["is_front0"]) == 1
        except Exception:
            return str(row["is_front0"]).strip() in ("1","true","True")
    return True

def _normalize_bits_field(r: dict) -> str:
    """统一提取位串字段（兼容 GA 与 stage1 命名）"""
    bits_s = str(
        r.get("rule_bits", "")
        or r.get("bits", "")
        or r.get("rule_bits_canon", "")
        or r.get("rule_bits_raw", "")
    ).strip()
    return bits_s

def _group_by_nks(rows: List[dict]) -> Dict[Tuple[int, int, str], List[dict]]:
    """按 (n, k, source) 分组；source: 'stage1' or 'ga'"""
    by: Dict[Tuple[int, int, str], List[dict]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        n = int(r.get("n", 0) or 0)
        k = int(r.get("k", 0) or 0)

        tag = str(r.get("run_tag", "") or r.get("__file__", "")).lower()
        src = "stage1" if "stage1" in tag else ("ga" if ("ga" in tag or "pareto_front" in tag) else "unknown")

        by.setdefault((n, k, src), []).append(r)
    return by

def _best_bucket_by_rulecount(rs: List[dict]) -> Dict[int,dict]:
    """每个 rule_count 只保留 y 最大的记录"""
    buckets: Dict[int,dict] = {}
    for r in rs:
        rc = to_int(r.get("rule_count"))
        y  = _y_metric(r)
        if rc is None or not np.isfinite(y):
            continue
        cur = buckets.get(rc)
        if (cur is None) or (y > cur["y"]):
            buckets[rc] = dict(row=r, y=float(y))
    return buckets

# =========================================
# CSV writers (examples / summary / global)
# =========================================
def _write_examples(examples: List[dict], ex_path: Path) -> None:
    ex_path.parent.mkdir(parents=True, exist_ok=True)
    if examples:
        keys = sorted(set().union(*[set(r.keys()) for r in examples]))
        with open(ex_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(examples)
    else:
        with open(ex_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["n","k"]) ; w.writeheader()

def _write_summary(sum_path: Path,
                   center_counter,
                   pre_counter,
                   post_counter,
                   examples: List[dict],
                   center_label: str = "knee") -> None:
    total_center = sum(1 for r in examples if isinstance(r.get(f"{center_label}_bits",""), str))
    total_pre    = sum(1 for r in examples if isinstance(r.get("pre_bits",""), str) and r.get("pre_bits","")!="")
    total_post   = sum(1 for r in examples if isinstance(r.get("post_bits",""), str) and r.get("post_bits","")!="")
    features = ["selfloop_rich","star_core","near_bip_chord","tri","c4","c5"]
    summ_rows=[]
    for feat in features:
        cn = center_counter.get(feat,0); pr=pre_counter.get(feat,0); po=post_counter.get(feat,0)
        cn_p = cn / max(1,total_center); pr_p = pr / max(1,total_pre); po_p = po / max(1,total_post)
        or_pre  = ((cn+0.5)/(total_center+1)) / (((pr+0.5)/(total_pre+1)) if total_pre>0 else 1.0)
        or_post = ((cn+0.5)/(total_center+1)) / (((po+0.5)/(total_post+1)) if total_post>0 else 1.0)
        summ_rows.append(dict(feature=feat,
                              knee_count=cn, knee_total=total_center, knee_ratio=cn_p,
                              pre_count=pr, pre_total=total_pre, pre_ratio=pr_p, OR_knee_vs_pre=or_pre,
                              post_count=po, post_total=total_post, post_ratio=po_p, OR_knee_vs_post=or_post))
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    if summ_rows:
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summ_rows[0].keys()))
            w.writeheader(); w.writerows(summ_rows)
    else:
        with open(sum_path, "w", encoding="utf-8") as f:
            f.write("feature,knee_count,knee_total,knee_ratio,pre_count,pre_total,pre_ratio,OR_knee_vs_pre,post_count,post_total,post_ratio,OR_knee_vs_post\n")

def _write_global(glob_path: Path, examples: List[dict], center_label: str = "knee") -> None:
    def agg(vals):
        arr = np.array([v for v in vals if np.isfinite(v)], float)
        if arr.size==0: return dict(med=np.nan, mean=np.nan, q25=np.nan, q75=np.nan)
        return dict(med=float(np.median(arr)), mean=float(arr.mean()),
                    q25=float(np.quantile(arr,0.25)), q75=float(np.quantile(arr,0.75)))
    cont_keys = ["deg_max","diag_cnt","tri","c4","c5","clustering","kcore","gap","lambda1","lap_algebraic"]
    glob = []
    for where in ["pre", center_label, "post"]:
        grp = [r for r in examples if isinstance(r.get(f"{where}_bits",""), str) and r.get(f"{where}_bits","")!=""]
        row = dict(where=where, count=len(grp))
        for key in cont_keys:
            vals = [r.get(f"{where}_{key}", np.nan) for r in grp]
            a = agg(vals)
            for sfx,v in a.items():
                row[f"{key}_{sfx}"] = v
        glob.append(row)
    glob_path.parent.mkdir(parents=True, exist_ok=True)
    if glob:
        with open(glob_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(glob[0].keys()))
            w.writeheader(); w.writerows(glob)
    else:
        with open(glob_path, "w", encoding="utf-8") as f:
            f.write("where,count\n")

# =========================================
# Main: Knee analysis (examples/summary/global + figs)
# =========================================
def analyze_fronts_for_knees(csv_paths: List[str],
                             out_csv_dir: str = "./results/motifs",
                             out_fig_dir: str = "./out_fig",
                             style: str = "default",
                             logy: bool = True):
    """Main entry: group by (n,k,source)，stage1(canon)>GA，统一 Z 定义"""
    apply_style(style)
    out_csv_dir = ensure_dir(out_csv_dir)
    out_fig_dir = ensure_dir(out_fig_dir)

    rows = load_csv_rows(csv_paths)
    if not rows:
        raise FileNotFoundError("[motifs] No rows loaded from CSV inputs.")

    # -------- 优先级：stage1(canon) > GA --------
    by_src = _group_by_nks(rows)
    all_nk = sorted(set((n, k) for (n, k, _) in by_src.keys()))
    by_nk = {}
    for (n, k) in all_nk:
        if (n, k, "stage1") in by_src:
            by_nk[(n, k)] = by_src[(n, k, "stage1")]
        elif (n, k, "ga") in by_src:
            by_nk[(n, k)] = by_src[(n, k, "ga")]
        else:
            by_nk[(n, k)] = by_src.get((n, k, "unknown"), [])

    examples: List[dict] = []
    from collections import Counter
    knee_feat_counter = Counter(); pre_feat_counter = Counter(); post_feat_counter = Counter()

    for (n, k), rs in sorted(by_nk.items()):
        buckets = _best_bucket_by_rulecount(rs)
        if not buckets:
            continue
        xs = sorted(buckets.keys())
        ys = np.array([buckets[x]["y"] for x in xs], float)

        idx_knee = robust_knee(xs, ys, logy=logy)
        if idx_knee is None:
            continue

        idx_pre = idx_knee - 1 if idx_knee - 1 >= 0 else None
        idx_post = idx_knee + 1 if idx_knee + 1 < len(xs) else None

        def row_of(idx): return buckets[xs[idx]]["row"] if idx is not None else None
        rows_sel = dict(pre=row_of(idx_pre), knee=row_of(idx_knee), post=row_of(idx_post))

        record = dict(n=n, k=k)
        for pos in ("pre", "knee", "post"):
            r = rows_sel[pos]
            if r is None:
                for f in ["deg_max","deg_mean","deg_std","diag_cnt","selfloop_rich",
                          "tri","c4","c5","odd_girth","near_bip_chord","clustering",
                          "kcore","lambda1","lambda2","gap","lap_algebraic","star_core",
                          "rule_count","y"]:
                    record[f"{pos}_{f}"] = np.nan
                record[f"{pos}_bits"] = ""
                continue

            bits_s = _normalize_bits_field(r)
            if not bits_s:
                for f in ["deg_max","deg_mean","deg_std","diag_cnt","selfloop_rich",
                          "tri","c4","c5","odd_girth","near_bip_chord","clustering",
                          "kcore","lambda1","lambda2","gap","lap_algebraic","star_core",
                          "rule_count","y"]:
                    record[f"{pos}_{f}"] = np.nan
                record[f"{pos}_bits"] = ""
                continue

            bits = np.fromiter((1 if ch == "1" else 0 for ch in bits_s), dtype=np.uint8)
            R = rule_from_bits_any(k, bits)
            feats = extract_features_from_R(R)
            for f, v in feats.items():
                record[f"{pos}_{f}"] = v
            record[f"{pos}_bits"] = bits_s
            record[f"{pos}_rule_count"] = int(r.get("rule_count", bits.sum()))
            record[f"{pos}_y"] = _y_metric(r)

            # 校正谱指标（λ1,λ2,gap）
            l1, l2, g = _extract_spectral_metrics(r, k)
            record[f"{pos}_lambda1"] = l1
            record[f"{pos}_lambda2"] = l2
            record[f"{pos}_gap"] = g

            if pos == "knee":
                edges = edge_criticality_scores(R)[:min(8, k*(k-1)//2)]
                record["knee_key_edges"] = ";".join([f"{e[0]}|{e[1]:.2f}|tri{e[2]['tri']}" for e in edges])

        # 计算 Δ
        def delta(a, b, name):
            va = record.get(f"{a}_{name}", np.nan)
            vb = record.get(f"{b}_{name}", np.nan)
            if not (np.isfinite(va) and np.isfinite(vb)):
                return np.nan
            return vb - va

        for nm in ["deg_max","diag_cnt","selfloop_rich","tri","c4","c5",
                   "near_bip_chord","clustering","kcore","gap","lap_algebraic","lambda1"]:
            record[f"delta_pre_to_knee_{nm}"] = delta("pre", "knee", nm)
            record[f"delta_knee_to_post_{nm}"] = delta("knee", "post", nm)

        # 统计计数器
        def bump(counter, prefix):
            val = lambda key: record.get(f"{prefix}_{key}", np.nan)
            for key in ["selfloop_rich","star_core","near_bip_chord"]:
                v = val(key)
                if np.isfinite(v) and v >= 0.5:
                    counter[key] += 1
            for key in ["tri","c4","c5"]:
                v = val(key)
                if np.isfinite(v) and v >= 1.0:
                    counter[key] += 1

        bump(knee_feat_counter, "knee")
        if rows_sel["pre"] is not None: bump(pre_feat_counter, "pre")
        if rows_sel["post"] is not None: bump(post_feat_counter, "post")
        examples.append(record)

    # 输出表格
    out_csv_dir = Path(out_csv_dir)
    ex_path = out_csv_dir / "motif_knee_examples.csv"
    _write_examples(examples, ex_path)
    sum_path = out_csv_dir / "motif_knee_summary.csv"
    _write_summary(sum_path, knee_feat_counter, pre_feat_counter, post_feat_counter, examples)
    glob_path = out_csv_dir / "motif_global_report.csv"
    _write_global(glob_path, examples)

    # 可视化部分保持原逻辑
    try:
        import matplotlib.pyplot as plt
        with open(sum_path, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            summ_rows = list(rd)
        labels = [r["feature"] for r in summ_rows] if summ_rows else []
        if labels:
            knee_ratio = [float(r["knee_ratio"]) for r in summ_rows]
            pre_ratio  = [float(r["pre_ratio"]) for r in summ_rows]
            post_ratio = [float(r["post_ratio"]) for r in summ_rows]
            x = np.arange(len(labels))
            plt.figure(figsize=(8.8,5.2))
            width=0.28
            plt.bar(x- width, pre_ratio, width, label="pre")
            plt.bar(x, knee_ratio, width, label="knee")
            plt.bar(x+ width, post_ratio, width, label="post")
            plt.xticks(x, labels, rotation=15)
            plt.ylabel("ratio")
            plt.title("Knee-driving motif prevalence")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(out_fig_dir / "knee_motif_bar.png", dpi=200)
            plt.close()
    except Exception:
        pass

        # === 三点汇总曲线（按每个 n,k 画 pre-knee-post 三点）===
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8.0, 5.2))
        ax = fig.add_subplot(111)

        # 直接复用上文 by_nk（已经是 stage1>GA 合并后的字典）
        for (n, k), rs in sorted(by_nk.items()):
            buckets = _best_bucket_by_rulecount(rs)
            if not buckets:
                continue
            xs = sorted(buckets.keys())
            ys = np.array([buckets[x]["y"] for x in xs], float)

            idx = robust_knee(xs, ys, logy=logy)
            if idx is None:
                continue

            three_x, three_y = [], []
            for j in (idx - 1, idx, idx + 1):
                if 0 <= j < len(xs):
                    three_x.append(xs[j]); three_y.append(ys[j])
            if len(three_x) >= 2:
                ax.plot(three_x, three_y, marker="o", alpha=0.9, label=f"n{n}k{k}")

        ax.set_xlabel("|R|")
        ax.set_ylabel("Y (Z or its estimate)")
        ax.set_yscale("log")
        ax.set_title("Three-point links around knee (stage1>GA)")
        # 图例多时自动分列
        leg = ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(Path(out_fig_dir) / "knee_three_point_links.png", dpi=220)
        plt.close(fig)
    except Exception as e:
        # 不阻塞主流程
        pass

    return str(ex_path), str(sum_path), str(glob_path), []

# =========================================
# Main: MUR analysis (examples/summary/global; figs optional)
# =========================================
def analyze_fronts_for_mur(csv_paths: List[str],
                           out_csv_dir: str,
                           out_fig_dir: str,
                           style: str = "ieee",
                           logy: bool = True):
    """
    与 analyze_fronts_for_knees 同源实现，但把中心点换成 MUR。
    - 数据来源优先级：stage1(canon) > GA；raw 仅备查
    - Y 量纲统一：优先 Z_exact，其次 Z_est / sum_lambda_powers / objective
    - 位串字段兼容：rule_bits / bits / rule_bits_canon / rule_bits_raw
    输出：
      - motif_mur_examples.csv
      - motif_mur_summary.csv
      - motif_mur_global_report.csv
    注意：为了复用下游解释器（不改 CLI），examples 的列名仍用 pre/knee/post，
         其中 knee_* 在此语义为 “mur_*”（中心行）。
    """
    apply_style(style)
    out_csv_dir = ensure_dir(out_csv_dir)
    out_fig_dir = ensure_dir(out_fig_dir)

    rows = load_csv_rows(csv_paths)
    if not rows:
        raise FileNotFoundError("[motifs/MUR] No rows loaded from CSV inputs.")

    # -------- 优先级合并：stage1(canon) > GA --------
    by_src = _group_by_nks(rows)  # (n,k,source) -> rows
    all_nk = sorted(set((n, k) for (n, k, _) in by_src.keys()))
    by_nk: Dict[Tuple[int,int], List[dict]] = {}
    for (n, k) in all_nk:
        if (n, k, "stage1") in by_src:
            by_nk[(n, k)] = by_src[(n, k, "stage1")]
        elif (n, k, "ga") in by_src:
            by_nk[(n, k)] = by_src[(n, k, "ga")]
        else:
            by_nk[(n, k)] = by_src.get((n, k, "unknown"), [])

    from collections import Counter
    examples: List[dict] = []
    center_counter = Counter(); pre_counter = Counter(); post_counter = Counter()

    for (n, k), rs in sorted(by_nk.items()):
        buckets = _best_bucket_by_rulecount(rs)  # 统一用 _y_metric 选每个 |R| 的最佳
        if not buckets:
            continue
        xs = sorted(buckets.keys())
        ys = np.array([buckets[x]["y"] for x in xs], float)

        idx = mur_index(xs, ys, logy=logy)  # 中心点 = MUR 右端点索引
        if idx is None:
            continue

        idx_pre  = idx - 1 if idx - 1 >= 0 else None
        idx_post = idx + 1 if idx + 1 < len(xs) else None

        def _row(i):
            return buckets[xs[i]]["row"] if i is not None else None

        rows_sel = dict(pre=_row(idx_pre), knee=_row(idx), post=_row(idx_post))  # “knee”列此处= MUR

        rec = dict(n=n, k=k)

        for pos in ("pre", "knee", "post"):
            r = rows_sel[pos]
            if r is None:
                for f in ["deg_max","deg_mean","deg_std","diag_cnt","selfloop_rich","tri","c4","c5",
                          "odd_girth","near_bip_chord","clustering","kcore","lambda1","lambda2",
                          "gap","lap_algebraic","star_core","rule_count","y"]:
                    rec[f"{pos}_{f}"] = np.nan
                rec[f"{pos}_bits"] = ""
                continue

            bits_s = _normalize_bits_field(r)
            if not bits_s:
                for f in ["deg_max","deg_mean","deg_std","diag_cnt","selfloop_rich","tri","c4","c5",
                          "odd_girth","near_bip_chord","clustering","kcore","lambda1","lambda2",
                          "gap","lap_algebraic","star_core","rule_count","y"]:
                    rec[f"{pos}_{f}"] = np.nan
                rec[f"{pos}_bits"] = ""
                continue

            # 结构与谱特征
            bits = np.fromiter((1 if ch == "1" else 0 for ch in bits_s), dtype=np.uint8)
            R = rule_from_bits_any(k, bits)
            feats = extract_features_from_R(R)
            for f, v in feats.items():
                rec[f"{pos}_{f}"] = float(v)

            # 统一 Y 与 rule_count、bits 保存
            rec[f"{pos}_rule_count"] = int(r.get("rule_count", bits.sum()))
            rec[f"{pos}_y"] = _y_metric(r)
            rec[f"{pos}_bits"] = bits_s

            # 若表内无谱列，尝试通过位串补齐 λ1/λ2/gap
            l1, l2, g = _extract_spectral_metrics(r, k)
            rec[f"{pos}_lambda1"] = l1
            rec[f"{pos}_lambda2"] = l2
            rec[f"{pos}_gap"] = g

        # Δ（字段名保持与 knee 版一致，便于下游解释器复用）
        def _delta(a, b, name):
            va = rec.get(f"{a}_{name}", np.nan)
            vb = rec.get(f"{b}_{name}", np.nan)
            if not (np.isfinite(va) and np.isfinite(vb)):
                return np.nan
            return vb - va

        for nm in ["deg_max","diag_cnt","selfloop_rich","tri","c4","c5",
                   "near_bip_chord","clustering","kcore","gap","lap_algebraic","lambda1"]:
            rec[f"delta_pre_to_knee_{nm}"]  = _delta("pre", "knee", nm)   # “knee”=MUR
            rec[f"delta_knee_to_post_{nm}"] = _delta("knee", "post", nm)

        # 计数器（中心=“knee（MUR）”）
        def bump(counter, prefix):
            val = lambda key: rec.get(f"{prefix}_{key}", np.nan)
            for key in ["selfloop_rich","star_core","near_bip_chord"]:
                v = val(key)
                if np.isfinite(v) and v >= 0.5:
                    counter[key] += 1
            for key in ["tri","c4","c5"]:
                v = val(key)
                if np.isfinite(v) and v >= 1.0:
                    counter[key] += 1

        bump(center_counter, "knee")
        if rows_sel["pre"]  is not None: bump(pre_counter , "pre")
        if rows_sel["post"] is not None: bump(post_counter, "post")

        examples.append(rec)

    # 写 MUR 三表
    out_csv_dir = Path(out_csv_dir)
    ex_path = out_csv_dir / "motif_mur_examples.csv"
    _write_examples(examples, ex_path)

    sum_path = out_csv_dir / "motif_mur_summary.csv"
    _write_summary(sum_path, center_counter, pre_counter, post_counter, examples, center_label="knee")

    glob_path = out_csv_dir / "motif_mur_global_report.csv"
    _write_global(glob_path, examples, center_label="knee")

    # 更新索引文件（可供上游 Cell 解析）
    try:
        idx_file = out_csv_dir / "motifs_index.txt"
        with open(idx_file, "a", encoding="utf-8") as f:
            f.write(f"\nmur_examples = {ex_path}\n")
            f.write(f"mur_summary  = {sum_path}\n")
            f.write(f"mur_global   = {glob_path}\n")
    except Exception:
        pass

        # === 三点汇总曲线（按每个 n,k 画 pre-MUR-post 三点）===
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8.0, 5.2))
        ax = fig.add_subplot(111)

        # 复用本函数中已构造的 by_nk（stage1>GA 优先）
        for (n, k), rs in sorted(by_nk.items()):
            buckets = _best_bucket_by_rulecount(rs)
            if not buckets:
                continue
            xs = sorted(buckets.keys())
            ys = np.array([buckets[x]["y"] for x in xs], float)

            idx = mur_index(xs, ys, logy=logy)
            if idx is None:
                continue

            three_x, three_y = [], []
            for j in (idx - 1, idx, idx + 1):
                if 0 <= j < len(xs):
                    three_x.append(xs[j]); three_y.append(ys[j])
            if len(three_x) >= 2:
                ax.plot(three_x, three_y, marker="s", alpha=0.9, label=f"n{n}k{k}")

        ax.set_xlabel("|R|")
        ax.set_ylabel("Y (Z or its estimate)")
        ax.set_yscale("log")
        ax.set_title("Three-point links around MUR (stage1>GA)")
        leg = ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(Path(out_fig_dir) / "mur_three_point_links.png", dpi=220)
        plt.close(fig)
    except Exception:
        pass

    return str(ex_path), str(sum_path), str(glob_path), []
