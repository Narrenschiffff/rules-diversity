# -*- coding: utf-8 -*-
"""
rules/symmetry.py
几何对称 + 状态置换对称（模式层面）统计与可视化

公开 API（供 rd_cli.py 调用）：
- summarize_front_symmetry(front_paths, n, k, geo_ops, state_perm, samples, out_csv_dir,
                           enum_limit=1_000_000, knee_only=False, motifs_examples=None, reuse=False)
    -> str (summary_csv path)

- plot_symmetry_examples(summary_csv, out_dir, style) -> List[str]
- count_before_after(bits, n, k, geo_ops, state_perm, enum_limit=1_000_000, samples=64) -> dict
- render_examples(bits, n, k, geo_ops, state_perm, out_dir, style,
                  enum_limit=200_000, samples=64) -> List[str]
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Iterable
import os, csv, math, hashlib
import numpy as np
import matplotlib.pyplot as plt

from .viz import apply_style
from .eval import rule_from_bits
from .utils_io import ensure_dir, load_csv_rows, write_csv

# ----------------- D4 几何变换与平移 -----------------
def rotate90(grid: np.ndarray) -> np.ndarray:  return np.rot90(grid, k=1)
def rotate180(grid: np.ndarray) -> np.ndarray: return np.rot90(grid, k=2)
def rotate270(grid: np.ndarray) -> np.ndarray: return np.rot90(grid, k=3)
def flip_h(grid: np.ndarray) -> np.ndarray:    return np.flip(grid, axis=1)
def flip_v(grid: np.ndarray) -> np.ndarray:    return np.flip(grid, axis=0)
def flip_d1(grid: np.ndarray) -> np.ndarray:   return np.transpose(grid)
def flip_d2(grid: np.ndarray) -> np.ndarray:   return np.rot90(np.transpose(grid), 2)

def _apply_d4(grid: np.ndarray, op: str) -> np.ndarray:
    op = op.lower()
    if op == "rot0":  return grid
    if op == "rot90": return rotate90(grid)
    if op == "rot180":return rotate180(grid)
    if op == "rot270":return rotate270(grid)
    if op == "refh":  return flip_h(grid)
    if op == "refv":  return flip_v(grid)
    if op == "refd1": return flip_d1(grid)
    if op == "refd2": return flip_d2(grid)
    return grid

def d4_transforms(grid: np.ndarray, ops: Optional[List[str]] = None) -> List[np.ndarray]:
    all_ops = ["rot0","rot90","rot180","rot270","refh","refv","refd1","refd2"]
    ops = ops or all_ops
    uniq = []; seen=set()
    for name in ops:
        a = _apply_d4(grid, name)
        key = a.tobytes()
        if key not in seen:
            seen.add(key); uniq.append(a)
    return uniq

def roll_xy(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(grid, shift=dx, axis=0), shift=dy, axis=1)

def all_torus_translates(grid: np.ndarray) -> Iterable[np.ndarray]:
    n = grid.shape[0]
    for dx in range(n):
        for dy in range(n):
            yield roll_xy(grid, dx, dy)

def relabel_states_minlex(grid: np.ndarray, k: int) -> np.ndarray:
    n = grid.shape[0]
    seen_map = {}; nxt = 0
    out = grid.copy()
    for i in range(n):
        for j in range(n):
            v = int(out[i, j])
            if v not in seen_map:
                seen_map[v] = nxt; nxt += 1
            out[i, j] = seen_map[v]
    return out

def pattern_hash(grid: np.ndarray) -> str:
    return hashlib.sha1(grid.tobytes()).hexdigest()

def canonicalize_pattern(grid: np.ndarray,
                         consider_state_relabel: bool = False,
                         k: Optional[int] = None,
                         geo_ops: Optional[List[str]] = None) -> Tuple[np.ndarray, str]:
    best = None; best_hash = None
    for g1 in d4_transforms(grid, ops=geo_ops):
        for g2 in all_torus_translates(g1):
            g3 = g2
            if consider_state_relabel:
                assert k is not None, "state relabel needs k"
                g3 = relabel_states_minlex(g2, k)
            h = pattern_hash(g3)
            if (best is None) or (h < best_hash):
                best, best_hash = g3.copy(), h
    return best, best_hash

# ----------------- 规则可行模式生成 -----------------
def _is_valid_grid(grid: np.ndarray, R: np.ndarray) -> bool:
    n = grid.shape[0]
    for i in range(n):
        for j in range(n):
            u = grid[i, j]
            if not R[u, grid[i, (j+1)%n]]: return False
            if not R[u, grid[i, (j-1)%n]]: return False
            if not R[u, grid[(i+1)%n, j]]: return False
            if not R[u, grid[(i-1)%n, j]]: return False
    return True

def _enumerate_all_grids(n: int, k: int, R: np.ndarray, limit: int = 1_000_000) -> List[np.ndarray]:
    total = k ** (n*n)
    if total > limit:
        return []
    outs: List[np.ndarray] = []
    for idx in range(total):
        arr = np.zeros((n, n), dtype=np.int32)
        x = idx
        for i in range(n):
            for j in range(n):
                arr[i, j] = x % k; x //= k
        if _is_valid_grid(arr, R):
            outs.append(arr)
    return outs

def _random_feasible_sample(n: int, k: int, R: np.ndarray, samples: int, max_trials: int = 200_000) -> List[np.ndarray]:
    rng = np.random.default_rng(0)
    outs: List[np.ndarray] = []
    trials = 0
    while len(outs) < samples and trials < max_trials:
        trials += 1
        g = rng.integers(0, k, size=(n, n), dtype=np.int32)
        # 简单局部修补
        for _ in range(4*n*n):
            i, j = rng.integers(0, n), rng.integers(0, n)
            ok = [x for x in range(k)
                  if R[x, g[i,(j+1)%n]] and R[x, g[i,(j-1)%n]] and
                     R[x, g[(i+1)%n, j]] and R[x, g[(i-1)%n, j]]]
            if ok:
                g[i,j] = ok[rng.integers(0, len(ok))]
        if _is_valid_grid(g, R):
            outs.append(g.copy())
    return outs

# ----------------- 去重计数 -----------------
def _before_after_counts(grids: List[np.ndarray], k: int,
                         geo_ops: Optional[List[str]], state_perm: bool) -> Dict[str, int]:
    raw = len(grids)
    seen_geo = set(); seen_both = set()
    for g in grids:
        _, h1 = canonicalize_pattern(g, consider_state_relabel=False, k=None, geo_ops=geo_ops)
        seen_geo.add(h1)
        if state_perm:
            _, h2 = canonicalize_pattern(g, consider_state_relabel=True, k=k, geo_ops=geo_ops)
            seen_both.add(h2)
    return {
        "raw_count": raw,
        "geom_dedup": len(seen_geo),
        "geom_perm_dedup": len(seen_both) if state_perm else len(seen_geo),
    }

def count_before_after(bits: np.ndarray, n: int, k: int,
                       geo_ops: List[str], state_perm: bool,
                       enum_limit: int = 1_000_000, samples: int = 64) -> Dict[str, int]:
    R = rule_from_bits(k, bits)
    grids = _enumerate_all_grids(n, k, R, limit=enum_limit)
    if not grids:
        grids = _random_feasible_sample(n, k, R, samples=samples)
    return _before_after_counts(grids, k, geo_ops, state_perm)

# ----------------- 汇总（前沿 or 膝点） -----------------
def _collect_front0_rules(front_paths: List[str]) -> List[dict]:
    rows = []
    for p in front_paths:
        if not os.path.exists(p): 
            continue
        with open(p, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                if r.get("is_front0", "1") in ("1","true","True"):
                    rows.append(r)
    # 按 rule_bits 去重
    uniq = {}
    for r in rows:
        key = r.get("rule_bits", "")
        if key and key not in uniq:
            uniq[key] = r
    return list(uniq.values())

def _collect_knee_rules_from_examples(examples_csv: str, n: int, k: int) -> List[str]:
    """从 motif_knee_examples.csv 提取 (n,k) 对应的 knee_bits（返回 bit 字符串列表）"""
    if not (examples_csv and os.path.exists(examples_csv)):
        return []
    rows = load_csv_rows([examples_csv])
    outs = []
    for r in rows:
        try:
            nn = int(r.get("n", -1)); kk = int(r.get("k", -1))
        except Exception:
            continue
        if nn == n and kk == k:
            bs = str(r.get("knee_bits", "")).strip()
            if bs:
                outs.append(bs)
    # 去重
    seen=set(); uniq=[]
    for s in outs:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

def summarize_front_symmetry(front_paths: List[str], n: int, k: int,
                             geo_ops: List[str], state_perm: bool,
                             samples: int, out_csv_dir: str = "./results/symmetry",
                             enum_limit: int = 1_000_000,
                             knee_only: bool = False,
                             motifs_examples: Optional[str] = None,
                             reuse: bool = False) -> str:
    os.makedirs(out_csv_dir, exist_ok=True)
    out_csv = os.path.join(out_csv_dir, f"symmetry_summary_n{n}_k{k}.csv")
    if reuse and os.path.exists(out_csv):
        print(f"[symmetry] reuse: {out_csv}")
        return out_csv

    # 目标规则集合
    if knee_only:
        knees = _collect_knee_rules_from_examples(motifs_examples or "", n, k)
        if knees:
            rows = [{"rule_bits": s, "rule_count": str(s.count('1'))} for s in knees]
        else:
            print(f"[symmetry] --knee-only 打开，但未在 {motifs_examples} 找到 (n={n},k={k}) 的膝点规则；将回退为全前沿。")
            rows = _collect_front0_rules(front_paths)
    else:
        rows = _collect_front0_rules(front_paths)

    # 逐条统计
    out_rows: List[Dict] = []
    for r in rows:
        bits_s = str(r.get("rule_bits", "")).strip()
        if not bits_s:
            continue
        bits = np.fromiter((1 if ch == '1' else 0 for ch in bits_s), dtype=np.uint8)
        stats = count_before_after(bits, n, k,
                                   geo_ops=geo_ops, state_perm=state_perm,
                                   enum_limit=enum_limit, samples=samples)
        out_rows.append(dict(
            n=n, k=k, rule_bits=bits_s, rule_count=int(r.get("rule_count", bits.sum())),
            raw_count=stats["raw_count"],
            geom_dedup=stats["geom_dedup"],
            geom_perm_dedup=stats["geom_perm_dedup"],
        ))

    write_csv(out_csv, out_rows,
              fieldnames=["n","k","rule_bits","rule_count","raw_count","geom_dedup","geom_perm_dedup"])
    return out_csv

# ----------------- 绘图 -----------------
def _mosaic(grids: List[np.ndarray], k: int, cols: int = 8) -> np.ndarray:
    if not grids: return np.zeros((1,1), dtype=int)
    n = grids[0].shape[0]
    cols = max(1, cols)
    rows = math.ceil(len(grids) / cols)
    out = np.zeros((rows*n, cols*n), dtype=int)
    for idx, g in enumerate(grids):
        r = idx // cols; c = idx % cols
        out[r*n:(r+1)*n, c*n:(c+1)*n] = g
    return out

def render_examples(bits: np.ndarray, n: int, k: int,
                    geo_ops: List[str], state_perm: bool,
                    out_dir: str = "./out_fig", style: str = "default",
                    enum_limit: int = 200_000, samples: int = 64) -> List[str]:
    apply_style(style)
    os.makedirs(out_dir, exist_ok=True)
    R = rule_from_bits(k, bits)
    grids = _enumerate_all_grids(n, k, R, limit=enum_limit)
    exact = True
    if not grids:
        grids = _random_feasible_sample(n, k, R, samples=samples)
        exact = False

    reps_geo = []
    reps_both = []
    seen_g = set(); seen_b = set()
    for g in grids:
        _, h1 = canonicalize_pattern(g, consider_state_relabel=False, k=None, geo_ops=geo_ops)
        if h1 not in seen_g:
            seen_g.add(h1); reps_geo.append(g)
        if state_perm:
            _, h2 = canonicalize_pattern(g, consider_state_relabel=True, k=k, geo_ops=geo_ops)
            if h2 not in seen_b:
                seen_b.add(h2); reps_both.append(g)

    paths = []
    for title, mosa, fname in [
        (f"Raw {'(exact)' if exact else '(sampled)'}", _mosaic(grids[:min(64,len(grids))], k, cols=8), "symmetry_raw.png"),
        ("Geom representatives", _mosaic(reps_geo[:min(64,len(reps_geo))], k, cols=8), "symmetry_geom.png"),
        ("Geom+Perm representatives" if state_perm else "Geom representatives (perm off)",
         _mosaic(reps_both[:min(64,len(reps_both))] if state_perm else reps_geo, k, cols=8),
         "symmetry_geom_perm.png"),
    ]:
        plt.figure()
        plt.imshow(mosa, interpolation="nearest")
        plt.title(title); plt.axis("off")
        p = os.path.join(out_dir, fname)
        plt.tight_layout(); plt.savefig(p, dpi=180); plt.close()
        paths.append(p)
    return paths

def plot_symmetry_examples(summary_csv: str, out_dir: str = "./out_fig", style: str = "default") -> List[str]:
    apply_style(style)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    if os.path.exists(summary_csv):
        with open(summary_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        print("[symmetry] empty summary.")
        return []

    xs = list(range(len(rows)))
    raw = np.array([int(r["raw_count"]) for r in rows], float)
    geo = np.array([int(r["geom_dedup"]) for r in rows], float)
    both = np.array([int(r["geom_perm_dedup"]) for r in rows], float)

    geo_ratio = np.divide(geo, raw, out=np.zeros_like(geo), where=raw>0)
    both_ratio = np.divide(both, raw, out=np.zeros_like(both), where=raw>0)

    # 条形图
    plt.figure()
    width = 0.38
    xx = np.array(xs, float)
    plt.bar(xx - width/2, geo_ratio, width=width, label="geom/raw")
    plt.bar(xx + width/2, both_ratio, width=width, label="geom+perm/raw")
    plt.xticks(xs, [str(i) for i in xs]); plt.ylim(0, 1.05)
    plt.ylabel("dedup ratio"); plt.title("Symmetry dedup ratios per rule (front0 / knee)")
    plt.legend(frameon=False); plt.tight_layout()
    p1 = os.path.join(out_dir, "symmetry_ratios.png"); plt.savefig(p1, dpi=200); plt.close()

    # 文本表
    K = min(12, len(rows))
    header = ["idx", "|R|", "raw", "geom", "geom+perm"]
    data = []
    for i, r in enumerate(rows[:K]):
        data.append([i, int(r.get("rule_count", 0)), int(r["raw_count"]),
                     int(r["geom_dedup"]), int(r["geom_perm_dedup"])])
    fig, ax = plt.subplots(figsize=(9, 0.4*K + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=data, colLabels=header, loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    plt.title("Symmetry summary (subset)"); 
    p2 = os.path.join(out_dir, "symmetry_table.png"); plt.savefig(p2, dpi=200); plt.close()

    return [p1, p2]
