# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, csv, math
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------- 样式 ----------------
_STYLES = {
    "default": {"figure.dpi":120,"savefig.dpi":170,"font.size":10,"axes.titlesize":11,"axes.labelsize":10,
                "legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,"axes.spines.top":False,
                "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"lines.linewidth":1.3,
                "lines.markersize":4.5,"legend.frameon":False},
    "ieee":    {"figure.dpi":120,"savefig.dpi":200,"font.size":9,"axes.titlesize":10,"axes.labelsize":9,
                "legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,"axes.spines.top":False,
                "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"lines.linewidth":1.2,
                "lines.markersize":4.0,"legend.frameon":False},
    "acm":     {"figure.dpi":120,"savefig.dpi":200,"font.size":10,"axes.titlesize":12,"axes.labelsize":10,
                "legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,"axes.spines.top":False,
                "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"lines.linewidth":1.4,
                "lines.markersize":5.0,"legend.frameon":False},
    "nature":  {"figure.dpi":120,"savefig.dpi":200,"font.size":11,"axes.titlesize":13,"axes.labelsize":11,
                "legend.fontsize":10,"xtick.labelsize":10,"ytick.labelsize":10,"axes.spines.top":False,
                "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"lines.linewidth":1.2,
                "lines.markersize":4.8,"legend.frameon":False},
}
def apply_style(style:str="default"): mpl.rcParams.update(_STYLES.get(style,_STYLES["default"]))
apply_style("default")

# ---------------- 工具 ----------------
def _unique_path(path:str)->str:
    if not os.path.exists(path): return path
    b,e = os.path.splitext(path); i=1
    while True:
        cand=f"{b}_{i}{e}"
        if not os.path.exists(cand): return cand
        i+=1

_TAG_TS = re.compile(r"(?:[_-]\d{9,})$")
def _shorten_tag(tag:str, max_len:int=24)->str:
    t = _TAG_TS.sub("", tag or "")
    return t if len(t)<=max_len else (t[:max_len-3]+"...")

def _jitter(xs: np.ndarray, scale: float = 0.12) -> np.ndarray:
    if xs.size == 0: return xs
    rng = np.random.default_rng(0)
    return xs + rng.normal(0.0, scale, size=xs.shape)

def _load_runs(csvs:Iterable[str])->Dict[str,List[dict]]:
    """
    将一组 CSV 读成 {run_tag: [row,...]}
    要求 CSV 至少包含：rule_count，且 Y 轴度量使用 sum_lambda_powers 或 Z_exact。
    """
    runs={}
    for p in csvs:
        if not os.path.exists(p): 
            continue
        with open(p,"r",encoding="utf-8") as f:
            rdr = list(csv.DictReader(f))
        if not rdr:
            continue
        tag = os.path.basename(p)
        for r in rdr:
            r["_file"] = p
        runs.setdefault(tag,[]).extend(rdr)
    return runs

def _pick_bits(row:dict)->str:
    return row.get("rule_bits","") or row.get("rule_bits_canon","") or row.get("rule_bits_raw","")

def _y_metric(row: dict) -> float:
    """
    统一取 Y 值：
      - GA: sum_lambda_powers
      - stage1: Z_exact
    """
    v = row.get("sum_lambda_powers", "")
    if v != "":
        try: return float(v)
        except: pass
    v = row.get("Z_exact", "")
    if v != "":
        try: return float(v)
        except: pass
    return float("nan")

def _bounds(row: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    统一上下界：
      - GA: 直接取 lower_bound/upper_bound
      - stage1: 返回 (None, None)，下游会用桶内 min/max 代替
    """
    lo=row.get("lower_bound",""); hi=row.get("upper_bound","")
    try: lo = float(lo) if lo!="" else None
    except: lo=None
    try: hi = float(hi) if hi!="" else None
    except: hi=None
    return lo,hi

def _gap_12(row:dict)->float:
    try:
        lam1 = float(row.get("lambda_max","nan"))
    except:
        lam1 = float("nan")
    lam2 = float("nan")
    s = row.get("lambda_top2","")
    if s and s.strip().startswith("("):
        parts = s.strip("() ").split(",")
        if len(parts)>1:
            try: lam2 = float(parts[1])
            except: pass
    return lam1 - lam2

def _select_front0(rows:List[dict])->List[dict]:
    if not rows: return rows
    if ("is_front0" in rows[0]) and any(str(r.get("is_front0","0"))=="1" for r in rows):
        return [r for r in rows if str(r.get("is_front0","0"))=="1"]
    return rows

# ---------------- 膝点 ----------------
def _knee_second(xs, ys, logy=True):
    xs, ys = np.asarray(xs,float), np.asarray(ys,float)
    if logy: ys = np.log(np.maximum(ys,1e-300))
    if len(xs)<3: return None
    d2 = ys[2:] - 2*ys[1:-1] + ys[:-2]
    idx = int(np.argmax(d2)); return idx+1

def _knee_l(xs, ys, logxy=True):
    xs, ys = np.asarray(xs,float), np.asarray(ys,float)
    if logxy: xs=np.log(np.maximum(xs,1e-9)); ys=np.log(np.maximum(ys,1e-300))
    if len(xs)<3: return None
    x0,y0 = xs[0],ys[0]; vx,vy = xs[-1]-x0, ys[-1]-y0
    vnorm = math.hypot(vx,vy)+1e-15
    imax, dmax = None, -1.0
    for i in range(len(xs)):
        wx,wy = xs[i]-x0, ys[i]-y0
        d = abs(vx*wy - vy*wx)/vnorm
        if d>dmax: dmax, imax = d, i
    return imax

# ---------------- (A) 混合三图（原 viz-all 逻辑，聚合所有 CSV） ----------------
def plot_pareto_from_csv(csv_path_fronts,
                         out_dir:str="./out_fig",
                         y_log:bool=False,
                         style:str="default"):
    apply_style(style); os.makedirs(out_dir, exist_ok=True)
    runs = _load_runs(csv_path_fronts)

    # (1) scatter
    fig, ax = plt.subplots()
    anyp = False
    for tag, rows in runs.items():
        rows = _select_front0(rows)
        xs, ys = [], []
        for r in rows:
            try:
                xs.append(int(r["rule_count"]))
                ys.append(_y_metric(r))
            except: pass
        xs, ys = np.asarray(xs,float), np.asarray(ys,float)
        m = np.isfinite(ys) & (ys>0 if y_log else np.isfinite(ys))
        xs, ys = xs[m], ys[m]
        if xs.size==0: continue
        ax.scatter(_jitter(xs, 0.15), ys, label=_shorten_tag(tag), alpha=0.9); anyp=True
    if y_log: ax.set_yscale("log")
    ax.set_xlabel("|R|"); ax.set_ylabel(r"$\widehat{\mathrm{trace}}(T^n)$ / $Z_{\mathrm{exact}}$")
    ax.set_title("Pareto Frontier (scatter)")
    if anyp: ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    p_sc = _unique_path(os.path.join(out_dir, f"pareto_scatter{'_log' if y_log else ''}.png"))
    fig.savefig(p_sc); plt.close(fig)

    # (2) growth with band & knees
    fig, ax = plt.subplots(); anyp=False
    for tag, rows in runs.items():
        rows = _select_front0(rows)
        bucket={}
        mins, maxs = {}, {}
        for r in rows:
            try:
                rc = int(r["rule_count"])
                est = _y_metric(r)
                lo, hi = _bounds(r)
                cur = bucket.get(rc)
                if (cur is None) or (est>cur["est"]):
                    bucket[rc]={"est":est,"lo":lo,"hi":hi}
                if np.isfinite(est):
                    mins[rc] = min(mins.get(rc, +np.inf), est)
                    maxs[rc] = max(maxs.get(rc, -np.inf), est)
            except: pass
        if not bucket: continue
        xs = np.array(sorted(bucket.keys()), float)
        ests = np.array([bucket[int(x)]["est"] for x in xs], float)
        los  = np.array([bucket[int(x)]["lo"] if bucket[int(x)]["lo"] is not None else mins.get(int(x), np.nan) for x in xs], float)
        his  = np.array([bucket[int(x)]["hi"] if bucket[int(x)]["hi"] is not None else maxs.get(int(x), np.nan) for x in xs], float)
        m = np.isfinite(ests) & (ests>0 if y_log else np.isfinite(ests))
        xs, ests, los, his = xs[m], ests[m], los[m], his[m]
        if xs.size==0: continue
        ax.plot(xs, ests, marker="o", label=_shorten_tag(tag), alpha=0.95); anyp=True
        vb = np.isfinite(los) & np.isfinite(his) & (his>=los)
        if vb.any(): ax.fill_between(xs[vb], los[vb], his[vb], alpha=0.15)
        i2 = _knee_second(xs, ests, logy=y_log)
        il = _knee_l(xs, ests, logxy=True)
        if i2 is not None: ax.scatter([xs[i2]], [ests[i2]], s=70, marker="D", label=f"{_shorten_tag(tag)}: knee-2Δ |R|={int(xs[i2])}")
        if il is not None: ax.scatter([xs[il]], [ests[il]], s=70, marker="s", label=f"{_shorten_tag(tag)}: knee-L |R|={int(xs[il])}")
        if i2 is not None and il is not None and abs(int(xs[i2])-int(xs[il]))<=1:
            idx = i2 if xs[i2]<=xs[il] else il
            ax.scatter([xs[idx]], [ests[idx]], s=110, marker="*", label=f"{_shorten_tag(tag)}: robust-knee |R|={int(xs[idx])}")
    if y_log: ax.set_yscale("log")
    ax.set_xlabel("|R|"); ax.set_ylabel(r"Best $\widehat{\mathrm{trace}}(T^n)$ / $Z_{\mathrm{exact}}$")
    ax.set_title("Growth Curve on Pareto Frontier")
    if anyp: ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    p_gr = _unique_path(os.path.join(out_dir, f"growth_curve{'_log' if y_log else ''}_with_band.png"))
    fig.savefig(p_gr); plt.close(fig)

    return p_sc, p_gr

def plot_knee_and_gap(csv_path_fronts,
                      out_dir:str="./out_fig",
                      y_log:bool=False,
                      style:str="default"):
    apply_style(style); os.makedirs(out_dir, exist_ok=True)
    runs = _load_runs(csv_path_fronts)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    painted_trace=False; painted_gap=False

    for tag, rows in runs.items():
        rows = _select_front0(rows)
        bucket={}
        for r in rows:
            try:
                rc = int(r["rule_count"])
                est = _y_metric(r)
                lam1 = float(r.get("lambda_max","nan"))
                lam2 = float("nan")
                s = r.get("lambda_top2","")
                if s and s.strip().startswith("("):
                    parts = s.strip("() ").split(",")
                    if len(parts)>1:
                        try: lam2=float(parts[1])
                        except: lam2=float("nan")
                cur=bucket.get(rc)
                if (cur is None) or (est>cur["est"]):
                    bucket[rc]={"est":est,"lam1":lam1,"lam2":lam2}
            except: pass
        if not bucket: continue
        xs = np.array(sorted(bucket.keys()), float)
        ys = np.array([bucket[int(x)]["est"] for x in xs], float)
        lam1s = np.array([bucket[int(x)]["lam1"] for x in xs], float)
        lam2s = np.array([bucket[int(x)]["lam2"] for x in xs], float)
        gaps = lam1s - lam2s

        m = np.isfinite(ys) & (ys>0 if y_log else np.isfinite(ys))
        xs, ys, gaps = xs[m], ys[m], gaps[m]
        if xs.size==0: continue

        ax1.plot(xs, ys, marker="o", label=_shorten_tag(tag) if not painted_trace else None, alpha=0.9)
        painted_trace=True or painted_trace
        if y_log: ax1.set_yscale("log")

        if np.isfinite(gaps).any():
            ax2.plot(xs, gaps, marker="^", linestyle="--",
                     label="spectral gap" if not painted_gap else None, alpha=0.8, color="tab:orange")
            painted_gap=True or painted_gap

        i2 = _knee_second(xs, ys, logy=y_log)
        il = _knee_l(xs, ys, logxy=True)
        if i2 is not None: ax1.scatter([xs[i2]],[ys[i2]],s=70,marker="D",label="knee-2Δ")
        if il is not None: ax1.scatter([xs[il]],[ys[il]],s=70,marker="s",label="knee-L")
        if i2 is not None and il is not None and abs(int(xs[i2])-int(xs[il]))<=1:
            idx = i2 if xs[i2]<=xs[il] else il
            ax1.scatter([xs[idx]],[ys[idx]],s=110,marker="*",label="robust-knee")

    ax1.set_xlabel("|R|"); ax1.set_ylabel(r"Best $\widehat{\mathrm{trace}}(T^n)$ / $Z_{\mathrm{exact}}$")
    ax2.set_ylabel(r"$\lambda_1 - \lambda_2$")
    ax1.set_title("Knees & Spectral Gap (linked)")
    ax1.legend(loc="upper left", ncol=2)
    if painted_gap: ax2.legend(loc="upper right")
    fig.tight_layout()
    path = _unique_path(os.path.join(out_dir, f"knees_gap_linked{'_log' if y_log else ''}.png"))
    fig.savefig(path); plt.close(fig)
    return path

# ---------------- (B) 指定 (n,k)：raw vs canon 三图 ----------------

_RX_STAGE1 = re.compile(r"stage1_(?:all|pareto)_n(\d+)_k(\d+)")
_RX_GA     = re.compile(r"pareto_front_(?:nk_)?n(\d+)_k(\d+)(?:_|\.csv)")

def _parse_nk_from_filename(fname: str) -> Tuple[Optional[int], Optional[int]]:
    m = _RX_STAGE1.search(fname) or _RX_GA.search(fname)
    if m: return int(m.group(1)), int(m.group(2))
    return None, None

def _file_origin_series(file_basename: str) -> str:
    """
    只根据文件名给出初步系列标记：
      - stage1_pareto_*_raw.csv   -> 'stage1_raw'
      - stage1_pareto_*_canon.csv -> 'stage1_canon'
      - pareto_front_*            -> 'ga_canon'
    其他：None（留给行级兜底）
    """
    name = file_basename.lower()
    if name.startswith("stage1_pareto_") and "_raw" in name:
        return "stage1_raw"
    if name.startswith("stage1_pareto_") and "_canon" in name:
        return "stage1_canon"
    if name.startswith("pareto_front_"):
        return "ga_canon"
    return ""

def _series_for_row(row: dict, file_basename: str) -> str:
    """
    最终系列判定（优先文件名）：
      - 若文件名能判定 -> 返回该系列
      - 否则：有 sum_lambda_powers -> 当作 GA；否则当作 stage1
        * 行含 rule_bits_canon 或 is_canon==1 -> canon；否则 raw
    """
    base_flag = _file_origin_series(file_basename)
    if base_flag:
        return base_flag

    # 行级兜底：看度量列
    is_ga = ("sum_lambda_powers" in row) or ("lower_bound" in row) or ("upper_bound" in row)
    if is_ga:
        return "ga_canon"  # GA 默认 canon（搜索空间已经做了规则折叠）

    # stage1：看是否 canon
    is_canon = False
    if (row.get("rule_bits_canon","") or "").strip() != "":
        is_canon = True
    if str(row.get("is_canonical_rep", row.get("is_canon","0"))).strip() == "1":
        is_canon = True
    return "stage1_canon" if is_canon else "stage1_raw"

def _collect_rows_by_series_for_nk(csv_paths: Iterable[str], n:int, k:int) -> Dict[str, List[dict]]:
    """
    收集并按系列分桶：{'stage1_raw': [...], 'stage1_canon': [...], 'ga_canon': [...]}
    (n,k) 过滤优先按列；列无则按文件名解析。
    """
    out = {"stage1_raw": [], "stage1_canon": [], "ga_canon": []}
    for p in csv_paths:
        if not os.path.exists(p): 
            continue
        with open(p, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        # 过滤 (n,k)
        if "n" in rows[0] and "k" in rows[0]:
            rows = [r for r in rows if str(r.get("n","")).isdigit() and str(r.get("k","")).isdigit()
                    and int(r["n"])==n and int(r["k"])==k]
        else:
            fn_n, fn_k = _parse_nk_from_filename(os.path.basename(p))
            if not (fn_n==n and fn_k==k):
                rows = []

        if not rows:
            continue

        base = os.path.basename(p)
        for r in rows:
            r["_file"] = p
            out[_series_for_row(r, base)].append(r)
    return out

def _bucket_best_and_band(rows: List[dict], use_logy: bool) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    每个 |R| 取最佳 y，并提供 band：
      - GA: 优先用 lower/upper；若缺失，则退化到 min/max
      - stage1: 无上下界 -> 使用 min/max 作为带宽
    """
    bucket: Dict[int, Dict[str, float]] = {}
    mins: Dict[int, float] = {}
    maxs: Dict[int, float] = {}

    for r in rows:
        try:
            rc = int(r["rule_count"])
        except:
            continue
        est = _y_metric(r)
        lo, hi = _bounds(r)
        # 最优值
        cur = bucket.get(rc)
        if (cur is None) or (est > cur["est"]):
            bucket[rc] = {"est": est, "lo": lo, "hi": hi}
        # min/max 备份
        if np.isfinite(est):
            mins[rc] = min(mins.get(rc, +np.inf), est)
            maxs[rc] = max(maxs.get(rc, -np.inf), est)

    if not bucket:
        return (np.array([]),)*4

    xs  = np.array(sorted(bucket.keys()), float)
    est = np.array([bucket[int(x)]["est"] for x in xs], float)
    los = []
    his = []
    for x in xs:
        d = bucket[int(x)]
        lo, hi = d["lo"], d["hi"]
        if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi) and hi>=lo):
            lo = mins.get(int(x), np.nan)
            hi = maxs.get(int(x), np.nan)
        los.append(lo); his.append(hi)
    los = np.array(los,float); his = np.array(his,float)

    m = np.isfinite(est) & (est>0 if use_logy else np.isfinite(est))
    return xs[m], est[m], los[m], his[m]


def plot_three_raw_canon_for_nk(front_paths: List[str],
                                n: int, k: int,
                                out_dir: str = "./out_fig",
                                y_log: bool = False,
                                style: str = "default") -> Tuple[str,str,str]:
    """
    同图展示：stage1_raw、stage1_canon、ga_canon 三条系列 + 膝点 + band
    返回 (scatter.png, growth_with_knees.png, knees_gap_linked.png)
    """
    apply_style(style); os.makedirs(out_dir, exist_ok=True)
    series = _collect_rows_by_series_for_nk(front_paths, n, k)

    order = ["stage1_raw", "stage1_canon", "ga_canon"]
    markers = {"stage1_raw":"s", "stage1_canon":"o", "ga_canon":"^"}
    linest  = {"stage1_raw":"--", "stage1_canon":"-", "ga_canon":"-."}
    jitter  = {"stage1_raw":-0.07, "stage1_canon":+0.07, "ga_canon":+0.00}
    labels  = {"stage1_raw":"stage1_raw", "stage1_canon":"stage1_canon", "ga_canon":"ga_canon"}

    # ---------- (1) scatter ----------
    fig1, ax1 = plt.subplots()
    anyp=False
    for key in order:
        rows = series.get(key, [])
        xs, ys = [], []
        for r in rows:
            try:
                xs.append(int(r["rule_count"]))
                ys.append(_y_metric(r))
            except:
                pass
        xs = np.asarray(xs,float); ys = np.asarray(ys,float)
        m = np.isfinite(ys) & (ys>0 if y_log else np.isfinite(ys))
        xs, ys = xs[m], ys[m]
        if xs.size==0: continue
        ax1.scatter(xs + jitter[key], ys, label=labels[key], alpha=0.9, marker=markers[key])
        anyp=True
    if y_log: ax1.set_yscale("log")
    ax1.set_xlabel("|R|"); ax1.set_ylabel(r"$\widehat{\mathrm{trace}}(T^n)$ / $Z_{\mathrm{exact}}$")
    ax1.set_title(f"(n={n}, k={k}) stage1_raw vs stage1_canon vs ga_canon — Scatter")
    if anyp: ax1.legend(loc="best")
    fig1.tight_layout()
    p_sc = os.path.join(out_dir, f"nk_n{n}_k{k}_scatter{'_log' if y_log else ''}.png")
    fig1.savefig(p_sc, dpi=200); plt.close(fig1)

    # ---------- (2) growth + knees + band ----------
    fig2, ax2 = plt.subplots(); anyp=False
    for key in order:
        rows = series.get(key, [])
        xs, est, lo, hi = _bucket_best_and_band(rows, use_logy=y_log)
        if xs.size==0: continue
        xj = xs + jitter[key]
        ax2.plot(xj, est, marker=markers[key], linestyle=linest[key], alpha=0.95, label=labels[key])
        vb = np.isfinite(lo) & np.isfinite(hi) & (hi>=lo)
        if vb.any(): ax2.fill_between(xj[vb], lo[vb], hi[vb], alpha=0.12, linewidth=0)

        i2 = _knee_second(xs, est, logy=y_log)
        il = _knee_l(xs, est, logxy=True)
        if i2 is not None:
            ax2.scatter([xj[i2]],[est[i2]],s=70,marker="D",label=f"{labels[key]}: knee-2Δ |R|={int(xs[i2])}")
        if il is not None:
            ax2.scatter([xj[il]],[est[il]],s=70,marker="s",label=f"{labels[key]}: knee-L  |R|={int(xs[il])}")
        if (i2 is not None) and (il is not None) and abs(int(xs[i2])-int(xs[il]))<=1:
            idx = i2 if xs[i2]<=xs[il] else il
            ax2.scatter([xj[idx]],[est[idx]],s=110,marker="*",label=f"{labels[key]}: robust-knee |R|={int(xs[idx])}")
        anyp=True
    if y_log: ax2.set_yscale("log")
    ax2.set_xlabel("|R|"); ax2.set_ylabel(r"Best $\widehat{\mathrm{trace}}(T^n)$ / $Z_{\mathrm{exact}}$")
    ax2.set_title(f"(n={n}, k={k}) Growth Curves with Knees & Bands")
    if anyp: ax2.legend(loc="best", ncol=2)
    fig2.tight_layout()
    p_gr = os.path.join(out_dir, f"nk_n{n}_k{k}_growth_knees{'_log' if y_log else ''}.png")
    fig2.savefig(p_gr, dpi=200); plt.close(fig2)

    # ---------- (3) knees & spectral gap (linked) ----------
    def _gap_12(row: dict) -> float:
        lam1 = row.get("lambda_max","")
        lam2 = float("nan")
        try:
            lam1 = float(lam1) if lam1!="" else float("nan")
        except:
            lam1 = float("nan")
        s = row.get("lambda_top2","")
        if s and str(s).strip().startswith("("):
            parts = str(s).strip("() ").split(",")
            if len(parts)>1:
                try: lam2=float(parts[1]); 
                except: lam2=float("nan")
        return lam1 - lam2

    fig3, ax3 = plt.subplots()
    ax4 = ax3.twinx()
    painted_gap=False; painted_trace=False
    for key in order:
        rows = series.get(key, [])
        xs, est, _, _ = _bucket_best_and_band(rows, use_logy=y_log)
        if xs.size==0: continue
        # gaps
        bucket_gap = {}
        for r in rows:
            try:
                rc = int(r["rule_count"])
                y  = _y_metric(r)
                if not np.isfinite(y): 
                    continue
                cur = bucket_gap.get(rc)
                if (cur is None) or (y > cur["y"]):
                    bucket_gap[rc] = {"y": y, "gap": _gap_12(r)}
            except:
                pass
        gxs = sorted(bucket_gap.keys())
        gaps = np.array([bucket_gap[x]["gap"] for x in gxs], float)
        gxs  = np.array(gxs, float)

        xj = xs + jitter[key]
        ax3.plot(xj, est, marker=markers[key], linestyle=linest[key], alpha=0.95,
                 label=(labels[key] if not painted_trace else None))
        painted_trace=True or painted_trace
        if y_log: ax3.set_yscale("log")
        if np.isfinite(gaps).any():
            ax4.plot(gxs, gaps, marker="^", linestyle=":", alpha=0.8,
                     label=("spectral gap" if not painted_gap else None), color="tab:orange")
            painted_gap=True or painted_gap

        i2 = _knee_second(xs, est, logy=y_log)
        il = _knee_l(xs, est, logxy=True)
        if i2 is not None: ax3.scatter([xj[i2]],[est[i2]],s=70,marker="D")
        if il is not None: ax3.scatter([xj[il]],[est[il]],s=70,marker="s")
        if (i2 is not None) and (il is not None) and abs(int(xs[i2])-int(xs[il]))<=1:
            idx = i2 if xs[i2]<=xs[il] else il
            ax3.scatter([xj[idx]],[est[idx]],s=110,marker="*")

    ax3.set_xlabel("|R|"); ax3.set_ylabel(r"Best $\widehat{\mathrm{trace}}(T^n)$ / $Z_{\mathrm{exact}}$")
    ax4.set_ylabel(r"$\lambda_1-\lambda_2$")
    ax3.set_title(f"(n={n}, k={k}) Knees & Spectral Gap — stage1_raw vs stage1_canon vs ga_canon")
    ax3.legend(loc="upper left")
    if painted_gap: ax4.legend(loc="upper right")
    fig3.tight_layout()
    p_kg = os.path.join(out_dir, f"nk_n{n}_k{k}_knees_gap{'_log' if y_log else ''}.png")
    fig3.savefig(p_kg, dpi=200); plt.close(fig3)

    return p_sc, p_gr, p_kg


# ---------------- (C) 条带熵收敛（完整版） ----------------
def plot_entropy_convergence(rule_bits: np.ndarray,
                             k: int,
                             n_min: int = 3,
                             n_max: int = 10,
                             device: str = "cpu",
                             out_dir: str = "./out_fig",
                             style: str = "default",
                             logy: bool = False) -> List[str]:
    """
    计算并绘制条带熵 H_n / n 的收敛曲线（线性尺度 + 对数尺度）。
    需在 rules.eval 中实现 entropy_for_n(rule_bits,k,n,device)。
    """
    apply_style(style); os.makedirs(out_dir, exist_ok=True)
    try:
        from rules.eval import entropy_for_n
    except Exception as e:
        raise RuntimeError("rules.eval.entropy_for_n 未提供，请在 eval.py 中实现") from e

    ns = list(range(int(n_min), int(n_max)+1))
    vals = []
    for n in ns:
        v = float(entropy_for_n(rule_bits=rule_bits, k=int(k), n=int(n), device=device))
        vals.append(v)

    # 线性
    fig, ax = plt.subplots()
    ax.plot(ns, vals, marker="o")
    ax.set_xlabel("n")
    ax.set_ylabel("strip entropy H_n / n")
    ax.set_title("Entropy convergence")
    fig.tight_layout()
    p1 = _unique_path(os.path.join(out_dir, "entropy_curve.png"))
    fig.savefig(p1); plt.close(fig)

    # 对数
    fig, ax = plt.subplots()
    ax.plot(ns, vals, marker="o")
    ax.set_xlabel("n")
    ax.set_ylabel("strip entropy H_n / n")
    ax.set_title("Entropy convergence (log-y)")
    ax.set_yscale("log")
    fig.tight_layout()
    p2 = _unique_path(os.path.join(out_dir, "entropy_curve_log.png"))
    fig.savefig(p2); plt.close(fig)

    return [p1, p2]

# ---------------- 汇总入口 ----------------
def plot_all(front_paths: List[str],
             out_dir:str="./out_fig",
             y_log:bool=False,
             entropy_bits:Optional[np.ndarray]=None,
             entropy_k:Optional[int]=None,
             entropy_n_min:int=3,
             entropy_n_max:int=10,
             device:str="cpu",
             style:str="default")->List[str]:
    """
    聚合所有 CSV（不区分 (n,k)），输出三图（散点、增长+膝点、膝点–谱隙联动）；
    若提供 entropy_* 则加绘两张熵曲线。
    """
    apply_style(style); os.makedirs(out_dir, exist_ok=True)
    sc, gr = plot_pareto_from_csv(front_paths, out_dir=out_dir, y_log=y_log, style=style)
    kg = plot_knee_and_gap(front_paths, out_dir=out_dir, y_log=y_log, style=style)
    outs=[sc, gr, kg]
    if entropy_bits is not None and entropy_k is not None:
        try:
            outs += plot_entropy_convergence(rule_bits=entropy_bits, k=entropy_k,
                                             n_min=entropy_n_min, n_max=entropy_n_max,
                                             device=device, out_dir=out_dir, style=style, logy=y_log)
        except Exception as e:
            print("[viz-all] entropy plot skipped:", e)
    return outs
