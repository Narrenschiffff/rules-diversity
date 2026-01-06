# -*- coding: utf-8 -*-
"""Motif analysis: knee & MUR finding on Pareto fronts and structural feature extraction."""
from __future__ import annotations
import os, re, math, csv, json, traceback
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
try:
    from .utils_io import ensure_dir, expand_globs, load_csv_rows, parse_nk_from_filename, to_int, write_csv
except ImportError:
    # Minimal fallback
    def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
    
    # === FIX: Robust to_int in Fallback ===
    # Must handle "2.0" -> 2.0 -> 2
    def to_int(v):
        try: 
            if v is None or (isinstance(v, str) and not v.strip()):
                return None
            return int(float(v))
        except: return None
        
    def load_csv_rows(paths):
        out = []
        for p in paths:
            if not os.path.exists(p): continue
            with open(p, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                rows = list(rdr)
                for r in rows: r["__file__"] = p
                out.extend(rows)
        return out
    def write_csv(p, rows, fieldnames):
        if not rows: return
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

# ==============================================================
# bits -> rule matrix
# ==============================================================
_HAS_EVAL = False
try:
    from .eval import rule_from_bits as _rule_from_bits_eval
    _HAS_EVAL = True
except Exception:
    _HAS_EVAL = False

def _bits_len_for_k(k:int)->int:
    return k + k*(k-1)//2

def _rule_from_bits_fallback(k:int, bits:np.ndarray)->np.ndarray:
    L = _bits_len_for_k(k)
    if bits.size != L:
         k_eff = int((math.isqrt(1 + 8 * bits.size) - 1) // 2)
         if k_eff * (k_eff + 1) // 2 == bits.size:
             k = k_eff
             L = bits.size
    
    if bits.size != L: return np.zeros((k,k), dtype=bool)

    R = np.zeros((k,k), dtype=bool)
    p = 0
    for i in range(k):
        R[i,i] = bool(bits[p]); p+=1
    for i in range(k):
        for j in range(i+1,k):
            b = bool(bits[p]); p+=1
            if b: R[i,j] = True; R[j,i] = True
    return R

def rule_from_bits_any(k:int, bits:np.ndarray)->np.ndarray:
    expected = k + k*(k-1)//2
    if bits.size != expected:
        k_eff = int((math.isqrt(1 + 8 * bits.size) - 1) // 2)
        if k_eff * (k_eff + 1) // 2 == bits.size: k = k_eff
            
    if _HAS_EVAL:
        try: return _rule_from_bits_eval(k, bits)
        except AssertionError: return _rule_from_bits_fallback(k, bits)
    return _rule_from_bits_fallback(k, bits)

# =========================================
# Knee & MUR & Optimal Indexers
# =========================================
def knee_second_diff(xs, ys, logy=True)->Optional[int]:
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if logy: ys = np.log(np.maximum(ys, 1e-300))
    if len(xs) < 3: return None
    d2 = ys[2:] - 2*ys[1:-1] + ys[:-2]
    return int(np.argmax(d2)) + 1

def knee_lcurve(xs, ys, logxy=True)->Optional[int]:
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if logxy:
        xs = np.log(np.maximum(xs, 1e-9)); ys = np.log(np.maximum(ys, 1e-300))
    if len(xs) < 3: return None
    x0,y0 = xs[0],ys[0]; x1,y1 = xs[-1],ys[-1]
    vx,vy = x1-x0, y1-y0
    vnorm = math.hypot(vx,vy) + 1e-15
    dmax, imax = -1.0, None
    for i in range(len(xs)):
        wx,wy = xs[i]-x0, ys[i]-y0
        area2 = abs(vx*wy - vy*wx)
        d = area2 / vnorm
        if d > dmax: dmax, imax = d, i
    return imax

def robust_knee(xs, ys, logy=True)->Optional[int]:
    i2 = knee_second_diff(xs, ys, logy=logy)
    il = knee_lcurve(xs, ys, logxy=True)
    if i2 is None: return il
    if il is None: return i2
    return i2 if (abs(i2-il) <= 1 and xs[i2] <= xs[il]) else il

def mur_index(xs, ys, logy=True)->Optional[int]:
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 2: return None
    Y = np.log(np.maximum(ys, 1e-300)) if logy else ys
    dx = np.maximum(xs[1:] - xs[:-1], 1e-12)
    slope = (Y[1:] - Y[:-1]) / dx
    return int(np.argmax(slope)) + 1

def optimal_index(xs, ys, logy: bool = True) -> int:
    """
    Efficiency-based Optimal Point Selection.
    Robust to single-point fronts (returns 0).
    """
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if not m.any(): return 0
    xs = xs[m]; ys = ys[m]
    Y = np.log(np.maximum(ys, 1e-300)) if logy else ys
    
    # Default: Global Max
    try: i_max = int(np.nanargmax(Y))
    except Exception: return 0

    # Candidates
    i_knee = knee_second_diff(xs, ys, logy=logy)
    if i_knee is None: i_knee = knee_lcurve(xs, ys, logxy=logy)
    i_mur = mur_index(xs, ys, logy=logy)

    candidates = []
    for c in (i_knee, i_mur, None if i_mur is None else i_mur + 1):
        if c is not None and 0 <= c < len(xs) and c < i_max:
            candidates.append(c)

    best_idx = i_max
    for c in candidates:
        gain_rem = ys[i_max] - ys[c]
        cost_rem = xs[i_max] - xs[c]
        if cost_rem <= 0: continue
        slope_tail = gain_rem / cost_rem
        slope_base = ys[c] / xs[c] if xs[c] > 0 else 0.0
        # Efficiency Consensus Rule
        if slope_tail < 0.2 * slope_base:
            if xs[c] < xs[best_idx]: best_idx = c
    return int(best_idx)

# =========================================
# Feature extractors
# =========================================
def _triangle_count(G):
    k = G.shape[0]; tri = 0
    for i in range(k):
        for j in range(i+1, k):
            if G[i,j]:
                for l in range(j+1, k):
                    if G[i,l] and G[j,l]: tri += 1
    return tri

def _c4_count(G):
    A = G.astype(np.int8).copy(); np.fill_diagonal(A, 0); S = A @ A; k = A.shape[0]
    twice = 0
    for i in range(k):
        for j in range(i+1, k):
            c = int(S[i,j]); 
            if c >= 2: twice += c*(c-1)//2
    return int(twice//2)

def _c5_count(G):
    A = G.astype(bool).copy(); np.fill_diagonal(A, False); n = A.shape[0]
    if n < 5: return 0
    count = 0; visited = np.zeros(n, dtype=bool)
    def dfs(start, u, depth):
        nonlocal count
        if depth == 4:
            if A[u, start]: count += 1
            return
        for v in range(start, n):
            if A[u, v] and not visited[v]:
                visited[v] = True; dfs(start, v, depth+1); visited[v] = False
    for s in range(n):
        visited[s] = True
        for v in range(s+1, n):
            if A[s, v]:
                visited[v] = True; dfs(s, v, 2); visited[v] = False
        visited[s] = False
    return count // 10

def _odd_girth(G):
    k = G.shape[0]; INF = 1e9; best = INF
    for s in range(k):
        color = {s: 0}; q = [s]
        while q:
            u = q.pop(0)
            for v in range(k):
                if not G[u, v]: continue
                if v not in color:
                    color[v] = color[u] ^ 1; q.append(v)
                elif color[v] == color[u]:
                    best = min(best, 3.0)
    return best if best < INF else float("inf")

def _clustering_coeff(G):
    k = G.shape[0]; cvals = []
    for u in range(k):
        nbr = np.flatnonzero(G[u]); d = len(nbr)
        if d < 2:
            cvals.append(0.0); continue
        sub = G[np.ix_(nbr, nbr)]; e = np.triu(sub, 1).sum()
        cvals.append(2.0 * e / (d * (d - 1)))
    return float(np.mean(cvals)) if cvals else 0.0

def _kcore_number(G):
    k = G.shape[0]; deg = G.sum(axis=1).astype(int).tolist(); alive = [True]*k; core = 0; changed = True
    while changed:
        changed = False
        for v in range(k):
            if not alive[v]: continue
            if deg[v] < core:
                alive[v] = False; changed = True
                for u in range(k):
                    if alive[u] and G[u, v]: deg[u] -= 1
        if not changed:
            cand = min([deg[v] for v in range(k) if alive[v]] + [10**9])
            if cand > core and cand < 10**9: core = cand; changed = True
    return core

def _get_lcc_size(G):
    k = G.shape[0]; visited = np.zeros(k, dtype=bool); max_size = 0
    for i in range(k):
        if not visited[i]:
            q = [i]; visited[i] = True; current_size = 0
            while q:
                u = q.pop(0); current_size += 1
                for v in range(k):
                    if G[u, v] and not visited[v]: visited[v] = True; q.append(v)
            if current_size > max_size: max_size = current_size
    return max_size

def _spectral_feats(G):
    if not G.any(): return 0.0, 0.0, 0.0, 0.0
    A = G.astype(float); w = np.linalg.eigvalsh(A)
    lam1 = float(w[-1]); lam2 = float(w[-2]) if w.size >= 2 else 0.0
    gap = lam1 - lam2
    deg = A.sum(axis=1); D = np.diag(deg); L = D - A; wl = np.linalg.eigvalsh(L)
    alg_con = float(sorted(wl)[1]) if wl.size >= 2 else 0.0
    return lam1, lam2, gap, alg_con

def extract_features_from_R(R):
    k = R.shape[0]; G = R.copy().astype(bool); np.fill_diagonal(G, False)
    deg = G.sum(axis=1); tri = _triangle_count(G); c4 = _c4_count(G); c5 = _c5_count(G)
    oddg = _odd_girth(G); Cbar = _clustering_coeff(G); kcore = _kcore_number(G)
    lam1, lam2, gap, alg = _spectral_feats(G); lcc = _get_lcc_size(G)
    return dict(
        deg_max=float(deg.max() if deg.size else 0.0),
        deg_mean=float(deg.mean() if deg.size else 0.0),
        deg_std=float(deg.std(ddof=0) if deg.size else 0.0),
        diag_cnt=float(np.trace(R)),
        selfloop_rich=float(np.trace(R) >= max(1, k // 2)),
        tri=float(tri), c4=float(c4), c5=float(c5), odd_girth=float(oddg),
        near_bip_chord=float((tri + c5) <= 2), clustering=float(Cbar), kcore=float(kcore),
        lambda1=lam1, lambda2=lam2, gap=gap, lap_algebraic=alg,
        star_core=float(deg.max() >= max(k - 1, 2)), lcc_size=float(lcc),
        lcc_ratio=float(lcc)/k if k>0 else 0.0, is_connected=float(lcc == k)
    )

def edge_criticality_scores(R):
    k = R.shape[0]; G = R.copy().astype(bool); np.fill_diagonal(G, False); feats = []
    for u in range(k):
        for v in range(u + 1, k):
            if not G[u, v]: continue
            tri = int(np.logical_and(G[u], G[v]).sum()); c4 = 0
            for x in range(k):
                if x == u or x == v or not G[u, x]: continue
                for y in range(k):
                    if y <= x or y in (u, v): continue
                    if G[v, y] and G[x, y]: c4 += 1
            oddchord = 0
            for w in range(k):
                if w in (u, v): continue
                if G[u, w] and G[w, v]: oddchord += 1
            score = tri + 0.5 * c4 + oddchord
            feats.append(((u, v), float(score), dict(tri=tri, c4=c4, oddchord=oddchord)))
    feats.sort(key=lambda t: (-t[1], -(t[2]["tri"]), -(t[2]["c4"])))
    return feats

# =========================================
# Utils
# =========================================
def _y_metric(row: dict, logy: bool = True) -> float:
    def _to_float(v):
        try: return float(v)
        except: return float("nan")

    pf = _to_float(row.get("penalty_factor", 1.0))
    if not np.isfinite(pf) or pf <= 0: pf = 1.0

    raw_candidates = ("objective_raw", "sum_lambda_powers_raw", "trace_estimate_raw", "Z_exact", "Z_est", "objective")
    pen_candidates = ("objective_penalized", "sum_lambda_powers_penalized", "sum_lambda_powers", "trace_estimate", "y")

    def _first_valid(keys):
        for key in keys:
            val = row.get(key, "")
            if val in ("", None, "nan", "NaN"): continue
            f = _to_float(val)
            if np.isfinite(f): return f
        return float("nan")

    raw = _first_valid(raw_candidates)
    pen = _first_valid(pen_candidates)

    if (not np.isfinite(pen) or pen <= -1e200) and np.isfinite(raw):
        pen = raw / pf
    if (not np.isfinite(raw) or raw <= -1e200) and np.isfinite(pen):
        raw = pen * pf

    y = pen if (np.isfinite(pen) and pen > -1e200) else raw
    if not np.isfinite(y) or y <= -1e200:
        return float("nan")
    if logy and y <= 0:
        return float("nan")
    return float(y)

def _infer_k_from_bits_len(L):
    delta = 1 + 8 * L; s = math.isqrt(delta)
    if s * s != delta: raise ValueError(f"Bit length {L} invalid.")
    return (s - 1) // 2

def _normalize_bits_field(r, k=None):
    candidates = ["rule_bits", "rule_bits_raw", "bits", "rule_bits_canon"]
    if k is not None:
        expected = k + k*(k-1)//2
        for key in candidates:
            val = str(r.get(key, "")).strip()
            if len(val) == expected: return val
    for key in candidates:
        val = str(r.get(key, "")).strip()
        if val: return val
    return ""

def _extract_spectral_metrics(r, k):
    try:
        l1 = float(r.get("lambda1", np.nan)); l2 = float(r.get("lambda2", np.nan))
        gap = float(r.get("gap", np.nan))
        if not np.isfinite(gap) and np.isfinite(l1) and np.isfinite(l2): gap = l1 - l2
        if not np.isfinite(l1) or not np.isfinite(l2):
            bits_s = _normalize_bits_field(r, None)
            if bits_s:
                try:
                    k_eff = _infer_k_from_bits_len(len(bits_s))
                    bits = np.fromiter((1 if ch == "1" else 0 for ch in bits_s), dtype=np.uint8)
                    R = rule_from_bits_any(k_eff, bits)
                    f = extract_features_from_R(R)
                    l1 = f.get("lambda1", np.nan); l2 = f.get("lambda2", np.nan); gap = f.get("gap", np.nan)
                except: pass
        return l1, l2, gap
    except: return np.nan, np.nan, np.nan

def _is_front_row(row):
    if "is_front0" in row:
        try: return int(row["is_front0"]) == 1
        except: return str(row["is_front0"]).strip() in ("1","true","True")
    return True

def _group_by_nks(rows):
    by = {}
    for r in rows:
        if not isinstance(r, dict): continue
        n = int(r.get("n", 0) or 0); k = int(r.get("k", 0) or 0)
        tag = str(r.get("run_tag", "") or r.get("__file__", "")).lower()
        src = "stage1" if "stage1" in tag else ("ga" if ("ga" in tag or "pareto_front" in tag) else "unknown")
        by.setdefault((n, k, src), []).append(r)
    return by

def _best_bucket_by_rulecount(rs, logy: bool = True):
    buckets = {}
    for r in rs:
        # === FIX 2: Handle rule_count carefully ===
        rc = to_int(r.get("rule_count"))
        y = _y_metric(r, logy=logy)
        if rc is None or not np.isfinite(y): continue
        
        cur = buckets.get(rc)
        if (cur is None) or (y > cur["y"]):
            buckets[rc] = dict(row=r, y=float(y))
    return buckets

def _write_examples(examples, ex_path):
    ex_path.parent.mkdir(parents=True, exist_ok=True)
    if examples:
        keys = sorted(set().union(*[set(r.keys()) for r in examples]))
        write_csv(ex_path, examples, keys)
    else:
        with open(ex_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["n","k"]).writeheader()

def _write_summary(sum_path, center_counter, pre_counter, post_counter, examples, center_label="knee"):
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
        summ_rows.append(dict(feature=feat, **{f"{center_label}_count": cn, f"{center_label}_total": total_center, f"{center_label}_ratio": cn_p, "pre_count": pr, "pre_total": total_pre, "pre_ratio": pr_p, f"OR_{center_label}_vs_pre": or_pre, "post_count": po, "post_total": total_post, "post_ratio": po_p, f"OR_{center_label}_vs_post": or_post}))
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    if summ_rows: write_csv(sum_path, summ_rows, list(summ_rows[0].keys()))
    else:
        with open(sum_path, "w", encoding="utf-8") as f: f.write("feature\n")

def _write_global(glob_path, examples, center_label="knee"):
    def agg(vals):
        arr = np.array([v for v in vals if np.isfinite(v)], float)
        if arr.size==0: return dict(med=np.nan, mean=np.nan, q25=np.nan, q75=np.nan)
        return dict(med=float(np.median(arr)), mean=float(arr.mean()), q25=float(np.quantile(arr,0.25)), q75=float(np.quantile(arr,0.75)))
    cont_keys = ["deg_max","diag_cnt","tri","c4","c5","clustering","kcore","gap","lambda1","lap_algebraic"]
    glob = []
    for where in ["pre", center_label, "post"]:
        grp = [r for r in examples if isinstance(r.get(f"{where}_bits",""), str) and r.get(f"{where}_bits","")!=""]
        row = dict(where=where, count=len(grp))
        for key in cont_keys:
            vals = [r.get(f"{where}_{key}", np.nan) for r in grp]
            a = agg(vals)
            for sfx,v in a.items(): row[f"{key}_{sfx}"] = v
        glob.append(row)
    glob_path.parent.mkdir(parents=True, exist_ok=True)
    if glob: write_csv(glob_path, glob, list(glob[0].keys()))
    else:
        with open(glob_path, "w", encoding="utf-8") as f: f.write("where,count\n")

# =========================================
# Main
# =========================================
def analyze_fronts_for_optimal(csv_paths: List[str],
                               out_csv_dir: str = "./results/motifs",
                               out_fig_dir: str = "./out_fig",
                               style: str = "default",
                               logy: bool = True):
    apply_style(style); out_csv_dir = ensure_dir(out_csv_dir); out_fig_dir = ensure_dir(out_fig_dir)
    fig_paths = []
    rows = load_csv_rows(csv_paths)
    if not rows: raise FileNotFoundError("No rows.")
    by_src = _group_by_nks(rows)
    all_nk = sorted(set((n, k) for (n, k, _) in by_src.keys()))
    by_nk = {}
    for (n, k) in all_nk:
        if (n, k, "stage1") in by_src: by_nk[(n, k)] = by_src[(n, k, "stage1")]
        elif (n, k, "ga") in by_src: by_nk[(n, k)] = by_src[(n, k, "ga")]
        else: by_nk[(n, k)] = by_src.get((n, k, "unknown"), [])

    examples = []
    from collections import Counter
    optimal_cnt = Counter(); pre_cnt = Counter(); post_cnt = Counter()
    center_label = "optimal"

    for (n, k), rs in sorted(by_nk.items()):
        buckets = _best_bucket_by_rulecount(rs, logy=logy)
        if not buckets: continue
        xs = sorted(buckets.keys())
        ys = np.array([buckets[x]["y"] for x in xs], float)
        
        idx_opt = optimal_index(xs, ys, logy=logy)
        idx_pre = idx_opt - 1 if idx_opt - 1 >= 0 else None
        idx_post = idx_opt + 1 if idx_opt + 1 < len(xs) else None
        
        row_of = lambda idx: buckets[xs[idx]]["row"] if idx is not None else None
        rows_sel = dict(pre=row_of(idx_pre), optimal=row_of(idx_opt), post=row_of(idx_post))
        
        record = dict(n=n, k=k)
        for pos in ("pre", center_label, "post"):
            r = rows_sel[pos]
            if r is None:
                record[f"{pos}_bits"] = ""; continue
            
            bits_s = _normalize_bits_field(r, None)
            if not bits_s: record[f"{pos}_bits"] = ""; continue
            
            try:
                eff_k = _infer_k_from_bits_len(len(bits_s))
                bits = np.fromiter((1 if ch == "1" else 0 for ch in bits_s), dtype=np.uint8)
                R = rule_from_bits_any(eff_k, bits)
            except:
                record[f"{pos}_bits"] = ""; continue
            
            feats = extract_features_from_R(R)
            for f, v in feats.items(): record[f"{pos}_{f}"] = v
            record[f"{pos}_bits"] = bits_s
            record[f"{pos}_rule_count"] = int(r.get("rule_count", bits.sum()))
            record[f"{pos}_y"] = _y_metric(r, logy=logy)
            record[f"{pos}_effective_k"] = eff_k
            l1, l2, g = _extract_spectral_metrics(r, eff_k)
            record[f"{pos}_lambda1"] = l1; record[f"{pos}_lambda2"] = l2; record[f"{pos}_gap"] = g
            
            if pos == center_label:
                edges = edge_criticality_scores(R)[:min(8, eff_k*(eff_k-1)//2)]
                record[f"{center_label}_key_edges"] = ";".join([f"{e[0]}|{e[1]:.2f}|tri{e[2]['tri']}" for e in edges])

        def delta(a, b, name):
            va = record.get(f"{a}_{name}", np.nan); vb = record.get(f"{b}_{name}", np.nan)
            return vb - va if (np.isfinite(va) and np.isfinite(vb)) else np.nan

        for nm in ["deg_max","diag_cnt","selfloop_rich","tri","c4","c5","near_bip_chord","clustering","kcore","gap","lap_algebraic","lambda1", "lcc_ratio"]:
            record[f"delta_pre_to_{center_label}_{nm}"] = delta("pre", center_label, nm)
            record[f"delta_{center_label}_to_post_{nm}"] = delta(center_label, "post", nm)

        def bump(counter, prefix):
            val = lambda key: record.get(f"{prefix}_{key}", np.nan)
            for key in ["selfloop_rich","star_core","near_bip_chord"]:
                if (v:=val(key)) >= 0.5: counter[key] += 1
            for key in ["tri","c4","c5"]:
                if (v:=val(key)) >= 1.0: counter[key] += 1

        bump(optimal_cnt, center_label)
        if rows_sel["pre"]: bump(pre_cnt, "pre")
        if rows_sel["post"]: bump(post_cnt, "post")
        examples.append(record)

    out_csv_dir = Path(out_csv_dir)
    ex_path = out_csv_dir / "motif_optimal_examples.csv"
    _write_examples(examples, ex_path)
    sum_path = out_csv_dir / "motif_optimal_summary.csv"
    _write_summary(sum_path, optimal_cnt, pre_cnt, post_cnt, examples, center_label=center_label)
    glob_path = out_csv_dir / "motif_optimal_global_report.csv"
    _write_global(glob_path, examples, center_label=center_label)

    # Bar chart
    try:
        import matplotlib.pyplot as plt
        with open(sum_path, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f); summ_rows = list(rd)
        labels = [r["feature"] for r in summ_rows] if summ_rows else []
        if labels:
            opt_r = [float(r.get(f"{center_label}_ratio", 0.0)) for r in summ_rows]
            pre_r  = [float(r["pre_ratio"]) for r in summ_rows]
            post_r = [float(r["post_ratio"]) for r in summ_rows]
            x = np.arange(len(labels)); width=0.28
            plt.figure(figsize=(8.8,5.2))
            plt.bar(x-width, pre_r, width, label="pre")
            plt.bar(x, opt_r, width, label=center_label)
            plt.bar(x+width, post_r, width, label="post")
            plt.xticks(x, labels, rotation=15); plt.ylabel("Ratio")
            plt.title("Optimal Motif Prevalence")
            plt.legend(frameon=False); plt.tight_layout()
            bp = Path(out_fig_dir) / "optimal_motif_bar.png"
            plt.savefig(bp, dpi=200); plt.close()
            fig_paths.append(str(bp))
    except: pass

    # Three-point links - FIX: Remove silent try-except
    import matplotlib.pyplot as plt
    try:
        fig = plt.figure(figsize=(8.0, 5.2)); ax = fig.add_subplot(111)
        valid_plots = 0
        for (n, k), rs in sorted(by_nk.items()):
            buckets = _best_bucket_by_rulecount(rs, logy=logy)
            if not buckets: continue
            xs = sorted(buckets.keys())
            ys = np.array([buckets[x]["y"] for x in xs], float)
            
            # Use optimal_index
            idx = optimal_index(xs, ys, logy=logy)
            
            three_x, three_y = [], []
            for j in (idx - 1, idx, idx + 1):
                if 0 <= j < len(xs): three_x.append(xs[j]); three_y.append(ys[j])
            
            if len(three_x) >= 1:
                ax.plot(three_x, three_y, marker="D", alpha=0.9, label=f"n{n}k{k}")
                valid_plots += 1

        ax.set_xlabel("|R|"); ax.set_ylabel(r"Objective (log $Z$) / penalty")
        if logy: ax.set_yscale("log")
        ax.set_title("Three-point links around optimal")
        fig.tight_layout()
        otp = Path(out_fig_dir) / "optimal_three_point_links.png"
        fig.savefig(otp, dpi=200); plt.close(fig)
        fig_paths.append(str(otp))
        
        # === DEBUG INFO ===
        print(f"[Motifs] Successfully plotted {valid_plots} nk-series in optimal_three_point_links")
        
    except Exception as e:
        print(f"[Motifs] Plotting Error: {e}")
        traceback.print_exc()

    return str(ex_path), str(sum_path), str(glob_path), fig_paths

if __name__ == "__main__":
    pass
