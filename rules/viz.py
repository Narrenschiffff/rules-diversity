# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, csv, math, argparse, glob, logging, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Sequence

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Try to import config, else use defaults
try:
    from . import config
except ImportError:
    class Config:
        DEFAULT_DEVICE = "cpu"
        BOUNDARY_MODE = "torus"
    config = Config()

try:
    from .eval import evaluate_rules_batch
except ImportError:
    def evaluate_rules_batch(*args, **kwargs):
        raise NotImplementedError("eval module not available")

# =========================
# 样式
# =========================
_STYLES = {
    "default": {"figure.dpi":120,"savefig.dpi":200,"font.size":10,"axes.titlesize":11,"axes.labelsize":10,
                "legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,"axes.spines.top":False,
                "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"lines.linewidth":1.3,
                "lines.markersize":4.8,"legend.frameon":False},
    "ieee":    {"figure.dpi":120,"savefig.dpi":220,"font.size":9,"axes.titlesize":10,"axes.labelsize":9,
                "legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,"axes.spines.top":False,
                "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"lines.linewidth":1.2,
                "lines.markersize":4.5,"legend.frameon":False},
}
def apply_style(style:str="default"):
    mpl.rcParams.update(_STYLES.get(style,_STYLES["default"]))

# =========================
# 工具
# =========================
_RX_TS = re.compile(r"(?:[_-]\d{9,})$")
def _shorten(tag:str, max_len:int=24)->str:
    t = _RX_TS.sub("", tag or "")
    return t if len(t)<=max_len else (t[:max_len-3]+"...")


def _bits_to_str(bits: np.ndarray | Sequence[int] | str) -> str:
    if isinstance(bits, str):
        return bits.strip()
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return "".join(str(int(x)) for x in arr.tolist())

def _jitter(xs: np.ndarray, scale: float = 0.12) -> np.ndarray:
    if xs.size == 0: return xs
    rng = np.random.default_rng(0)
    return xs + rng.normal(0.0, scale, size=xs.shape)

def _load_rows(csv_paths:Iterable[str])->Dict[str,List[dict]]:
    runs={}
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p,"r",encoding="utf-8") as f:
                rdr = list(csv.DictReader(f))
            if not rdr: continue
            tag = os.path.basename(p)
            for r in rdr: r["_file"]=p
            runs.setdefault(tag,[]).extend(rdr)
            _warn_trace_diff(rdr, label=tag)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load {p}: {e}")
    return runs


def summarize_runs(csv_paths: Iterable[str], sym_filter: Optional[str] = None) -> Dict[str, dict]:
    """Summarize CSV artifacts for quick diagnostics."""
    runs = _load_rows(csv_paths)
    out: Dict[str, dict] = {}
    for tag, rows in runs.items():
        filtered: List[dict] = []
        for r in rows:
            if sym_filter and str(r.get("sym_mode", "")).lower() != sym_filter.lower():
                continue
            filtered.append(r)
        if not filtered:
            continue

        sym_counts: Dict[str, int] = {}
        active_k: List[int] = []
        active_k_raw: List[int] = []
        boundaries = set()
        for r in filtered:
            sym_counts[str(r.get("sym_mode", "")) or "unknown"] = sym_counts.get(str(r.get("sym_mode", "")) or "unknown", 0) + 1
            try:
                active_k.append(int(r.get("active_k", 0)))
            except Exception:
                pass
            try:
                active_k_raw.append(int(r.get("active_k_raw", r.get("active_k", 0))))
            except Exception:
                pass
            b = str(r.get("boundary", "")).strip()
            if b:
                boundaries.add(b)

        out[tag] = {
            "rows": len(filtered),
            "symmetry_counts": sym_counts,
            "active_k_min": min(active_k) if active_k else None,
            "active_k_max": max(active_k) if active_k else None,
            "active_k_raw_min": min(active_k_raw) if active_k_raw else None,
            "active_k_raw_max": max(active_k_raw) if active_k_raw else None,
            "boundaries": sorted(boundaries),
        }
    return out


def _unique_field(rows: List[dict], field: str) -> Optional[str]:
    vals = {str(r.get(field, "")).strip() for r in rows if str(r.get(field, "")).strip() != ""}
    if not vals:
        return None
    if len(vals) > 1:
        # Just warn and return one, or raise error? Original code raised ValueError.
        # For robustness, let's pick one but log warning if strictness isn't required.
        # But keeping original logic for now.
        pass
    return next(iter(vals))


def _expand_globs(paths: Iterable[str]) -> List[str]:
    outs: list[str] = []
    for p in paths:
        outs.extend(glob.glob(p))
    seen = set(); uniq = []
    for x in outs:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq


def _row_bits(row: dict) -> str:
    for key in ("rule_bits", "rule_bits_raw", "rule_bits_canon"):
        s = str(row.get(key, "")).strip()
        if s:
            return s
    return ""


def _safe_float(val, default: float = float("nan")) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _resolve_value_with_penalty(raw_val: float, penalized_val: float, penalty_factor: float, use_penalty: bool) -> Tuple[float, float, float]:
    raw = _safe_float(raw_val, default=float("nan"))
    pen = _safe_float(penalized_val, default=float("nan"))
    pf = _safe_float(penalty_factor, default=1.0)
    if not np.isfinite(pf) or pf == 0.0:
        pf = 1.0

    if not np.isfinite(pen) and np.isfinite(raw):
        pen = raw / pf
    if not np.isfinite(raw) and np.isfinite(pen):
        raw = pen * pf
    
    main = pen if use_penalty else raw
    return main, raw, pen


def _entropy_value_from_row(row: Optional[dict], n: int, use_penalty: bool, normalize: bool,
                            read_exact: bool, read_estimate: bool) -> float:
    if not row:
        return float("nan")
    penalty = _safe_float(row.get("penalty_factor", 1.0), default=1.0)
    if not np.isfinite(penalty) or penalty <= 0:
        penalty = 1.0

    def _with_norm(v: float, is_penalized: bool) -> float:
        if not (np.isfinite(v) and v > 0):
            return float("nan")
        _, raw_val, pen_val = _resolve_value_with_penalty(
            raw_val=v if not is_penalized else float("nan"),
            penalized_val=v if is_penalized else float("nan"),
            penalty_factor=penalty,
            use_penalty=use_penalty,
        )
        chosen = pen_val if use_penalty else raw_val
        chosen = math.log(max(chosen, 1e-300))
        return chosen / float(n) if normalize else chosen

    if read_exact:
        for key in ("trace_exact", "Z_exact", "exact_Z"):
            v = _safe_float(row.get(key, float("nan")))
            if np.isfinite(v) and v > 0:
                return _with_norm(v, is_penalized=False)
    if read_estimate:
        for key in ("trace_estimate", "sum_lambda_powers", "sum_lambda_powers_penalized"):
            v = _safe_float(row.get(key, float("nan")))
            if np.isfinite(v) and v > 0:
                return _with_norm(v, is_penalized=("penal" in key))
        for key in ("trace_estimate_raw", "sum_lambda_powers_raw"):
            v = _safe_float(row.get(key, float("nan")))
            if np.isfinite(v) and v > 0:
                return _with_norm(v, is_penalized=False)
    return float("nan")


@dataclass
class EntropySeries:
    n_vals: np.ndarray
    log_vals: np.ndarray
    rule_bits: str
    k: int
    boundary: str
    sym_mode: str
    sources: List[str]


@dataclass
class KeyPoint:
    kind: str
    n: float
    rule_count: float
    metric: float
    label: str


@dataclass
class FrontierSurfaceData:
    k: int
    boundary: str
    sym_mode: str
    metric_field: str
    ns: np.ndarray
    rule_counts: np.ndarray
    grid: np.ndarray
    best_curve_metric: np.ndarray
    best_curve_n: np.ndarray
    key_points: List[KeyPoint]
    optimal_curve: Optional[List[Tuple[float, float, float]]] = None


def frontier_surface_to_json(d: FrontierSurfaceData, include_grid: bool = True) -> dict:
    """Convert FrontierSurfaceData into a JSON-serializable dict."""
    def _arr(x):
        return np.asarray(x).tolist()

    return {
        "k": int(d.k),
        "boundary": d.boundary,
        "sym_mode": d.sym_mode,
        "metric_field": d.metric_field,
        "ns": _arr(d.ns),
        "rule_counts": _arr(d.rule_counts),
        "grid": _arr(d.grid) if include_grid else None,
        "best_curve_metric": _arr(d.best_curve_metric),
        "best_curve_n": _arr(d.best_curve_n),
        "optimal_curve": _arr(d.optimal_curve) if d.optimal_curve is not None else None,
        "key_points": [
            {
                "kind": kp.kind,
                "n": float(kp.n),
                "rule_count": float(kp.rule_count),
                "metric": float(kp.metric),
                "label": kp.label,
            }
            for kp in d.key_points
        ],
    }


def frontier_surfaces_to_json(data: Sequence[FrontierSurfaceData], include_grid: bool = True) -> List[dict]:
    """Batch version of frontier_surface_to_json."""
    return [frontier_surface_to_json(d, include_grid=include_grid) for d in data]


def write_frontier_surfaces_json(data: Sequence[FrontierSurfaceData], path: str | os.PathLike, include_grid: bool = True) -> None:
    """Write FrontierSurfaceData list to JSON file (helper for notebooks/CLI)."""
    obj = frontier_surfaces_to_json(data, include_grid=include_grid)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def _prepare_entropy_series(rule_bits: Optional[str],
                            k: Optional[int],
                            n_min: int,
                            n_max: int,
                            csv_paths: Optional[Iterable[str]] = None,
                            boundary: Optional[str] = None,
                            sym_mode: Optional[str] = None,
                            device: str = config.DEFAULT_DEVICE,
                            use_penalty: bool = True,
                            normalize: bool = True,
                            read_existing_exact: bool = True,
                            read_existing_estimate: bool = True) -> EntropySeries:
    paths = _expand_globs(csv_paths or [])
    runs = _load_rows(paths) if paths else {}
    rows: List[dict] = []
    for rr in runs.values():
        rows.extend(rr)

    bits_str = rule_bits or ""
    if not bits_str:
        deduced = _unique_field(rows, "rule_bits") or _unique_field(rows, "rule_bits_raw") or _unique_field(rows, "rule_bits_canon")
        if deduced:
            bits_str = deduced
    if not bits_str and not rows:
        raise ValueError("rule_bits not provided and no CSV rows available")

    if bits_str:
        rows = [r for r in rows if _row_bits(r) == bits_str]

    boundary = boundary or _unique_field(rows, "boundary") or config.BOUNDARY_MODE
    sym_mode = sym_mode or _unique_field(rows, "sym_mode") or "perm"
    if k is None:
        k_val = _unique_field(rows, "k")
        if k_val is not None:
            try:
                k = int(k_val)
            except Exception:
                pass
    if k is None and bits_str:
        k = int(round(len(bits_str) ** 0.5))
    if k is None:
        raise ValueError("k is required (provide via argument or CSV field)")

    rows_by_n: Dict[int, dict] = {}
    for r in rows:
        try:
            n_val = int(r.get("n"))
        except Exception:
            continue
        if (n_val < n_min) or (n_val > n_max):
            continue
        prev = rows_by_n.get(n_val)
        has_exact = any(str(r.get(f, "")).strip() != "" for f in ("trace_exact", "Z_exact", "exact_Z"))
        prev_exact = any(str(prev.get(f, "")) != "" for f in ("trace_exact", "Z_exact", "exact_Z")) if prev else False
        if (prev is None) or (has_exact and not prev_exact):
            rows_by_n[n_val] = r

    n_list = []
    log_list = []
    sources: List[str] = []
    bits_arr = np.array([int(c) for c in bits_str], dtype=np.uint8) if bits_str else None

    for n in range(n_min, n_max + 1):
        row = rows_by_n.get(n)
        val = _entropy_value_from_row(row, n=n, use_penalty=use_penalty, normalize=normalize,
                                      read_exact=read_existing_exact, read_estimate=read_existing_estimate)
        source = "csv" if row is not None else ""
        if (not np.isfinite(val)) and (bits_arr is not None):
            try:
                fits = evaluate_rules_batch(
                    n=n, k=k, bits_list=[bits_arr],
                    sym_mode=sym_mode,
                    boundary=boundary,
                    device=device,
                    use_penalty=use_penalty,
                    enable_exact=read_existing_exact,
                    enable_spectral=read_existing_estimate,
                )
                fit = fits[0] if fits else {}
                fit["n"] = n
                fit["boundary"] = boundary
                fit["sym_mode"] = sym_mode
                val = _entropy_value_from_row(fit, n=n, use_penalty=use_penalty, normalize=normalize,
                                              read_exact=True, read_estimate=True)
                source = "eval"
            except Exception as exc:
                logging.getLogger(__name__).warning("[entropy] evaluate_rules_batch failed for n=%s: %s", n, exc)
        if np.isfinite(val):
            n_list.append(n)
            log_list.append(val)
            sources.append(source or "eval")

    if not n_list:
        raise ValueError("no entropy values available; provide rule_bits or CSVs with trace/estimate values")

    return EntropySeries(
        n_vals=np.array(n_list, dtype=float),
        log_vals=np.array(log_list, dtype=float),
        rule_bits=bits_str or "",
        k=int(k),
        boundary=str(boundary),
        sym_mode=str(sym_mode),
        sources=sources,
    )


def plot_entropy_convergence(rule_bits: Optional[Sequence[int] | np.ndarray | str] = None,
                             k: Optional[int] = None,
                             n_min: int = 3,
                             n_max: int = 10,
                             csv_paths: Optional[Iterable[str]] = None,
                             boundary: Optional[str] = None,
                             sym_mode: Optional[str] = None,
                             device: str = config.DEFAULT_DEVICE,
                             out_dir: str = "./out_fig",
                             style: str = "default",
                             logy: bool = False,
                             read_existing_exact: bool = True,
                             read_existing_estimate: bool = True,
                             normalize_log_per_n: bool = True,
                             apply_penalty: bool = True) -> List[str]:
    """Plot log(Z)/n convergence for a single rule."""
    apply_style(style); os.makedirs(out_dir, exist_ok=True)
    bits_str = _bits_to_str(rule_bits) if rule_bits is not None else None
    series = _prepare_entropy_series(
        rule_bits=bits_str, k=k, n_min=n_min, n_max=n_max,
        csv_paths=csv_paths, boundary=boundary, sym_mode=sym_mode, device=device,
        use_penalty=apply_penalty, normalize=normalize_log_per_n,
        read_existing_exact=read_existing_exact, read_existing_estimate=read_existing_estimate,
    )
    fig, ax = plt.subplots()
    ax.plot(series.n_vals, series.log_vals, marker="o", linestyle="-", alpha=0.95, label="log(Z)/n")
    ax.set_xlabel("n")
    ax.set_ylabel("log(Z)/n" if normalize_log_per_n else "log(Z)")
    ax.set_title(f"Entropy convergence (k={series.k}, boundary={series.boundary}, sym={series.sym_mode})")
    if logy:
        ax.set_yscale("log")
    if series.rule_bits:
        ax.text(0.02, 0.02, f"rule_bits={_shorten(series.rule_bits, 32)}", transform=ax.transAxes,
                fontsize=8, ha="left", va="bottom")
    ax.legend(loc="best")
    fig.tight_layout()
    fname = f"entropy_n{n_min}-{n_max}_k{series.k}_{_shorten(series.rule_bits or 'rule', 16)}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return [out_path]


def _warn_trace_diff(rows: List[dict], label: str = "") -> None:
    errs = []
    for r in rows:
        te = r.get("trace_estimate_raw", r.get("trace_estimate", r.get("sum_lambda_powers", "")))
        tx = r.get("trace_exact", r.get("exact_Z", ""))
        try:
            te = float(te) if te not in ("", None) else float("nan")
            tx = float(tx) if tx not in ("", None) else float("nan")
        except Exception:
            continue
        if not (np.isfinite(te) and np.isfinite(tx)):
            continue
        if tx == 0:
            continue
        errs.append(abs(te - tx) / max(1e-15, abs(tx)))
    if not errs:
        return
    errs.sort()
    p50 = errs[len(errs)//2]
    p90 = errs[int(len(errs)*0.9)]
    msg = f"[viz] trace est vs exact ({label}) count={len(errs)}, median={p50:.3e}, p90={p90:.3e}, max={errs[-1]:.3e}"
    if errs[-1] > 0.05:
        logging.info(msg + " [WARN]")
    else:
        logging.debug(msg)

def _y_metric(row: dict, prefer_field: Optional[str] = None, use_penalty: bool = True) -> float:
    pf = _safe_float(row.get("penalty_factor", 1.0), default=1.0)
    if prefer_field:
        try:
            v = float(row.get(prefer_field, ""))
            if np.isfinite(v):
                return v
        except Exception:
            pass
    raw_candidates = [
        "objective_raw",
        "sum_lambda_powers_raw",
        "trace_estimate_raw",
        "trace_exact",
    ]
    pen_candidates = [
        "objective_penalized",
        "sum_lambda_powers_penalized",
        "trace_estimate",
        "sum_lambda_powers",
    ]
    raw = next(( _safe_float(row.get(k, float("nan"))) for k in raw_candidates if str(row.get(k, "")) != ""), float("nan"))
    pen = next(( _safe_float(row.get(k, float("nan"))) for k in pen_candidates if str(row.get(k, "")) != ""), float("nan"))
    main, _, _ = _resolve_value_with_penalty(raw, pen, pf, use_penalty=use_penalty)
    return main

def _bounds(row: dict, use_penalty: bool, objective_field: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
    pf = _safe_float(row.get("penalty_factor", 1.0), default=1.0)
    lo_raw = _safe_float(row.get("lower_bound_raw", row.get("lower_bound", float("nan"))))
    hi_raw = _safe_float(row.get("upper_bound_raw", row.get("upper_bound", float("nan"))))
    lo_pen = _safe_float(row.get("lower_bound", float("nan")))
    hi_pen = _safe_float(row.get("upper_bound", float("nan")))
    if objective_field and objective_field.startswith("objective"):
        def _log_after_penalty(raw_v, pen_v):
            raw = _safe_float(raw_v, default=float("nan"))
            pen = _safe_float(pen_v, default=float("nan"))
            raw_log = math.log(max(raw, 1e-300)) if np.isfinite(raw) and raw > 0 else float("nan")
            pen_log = math.log(max(pen, 1e-300)) if np.isfinite(pen) and pen > 0 else float("nan")
            if not np.isfinite(pen_log) and np.isfinite(raw_log):
                pen_log = raw_log / max(pf, 1e-9)
            if not np.isfinite(raw_log) and np.isfinite(pen_log):
                raw_log = pen_log * max(pf, 1e-9)
            return pen_log if use_penalty else raw_log

        lo = _log_after_penalty(lo_raw, lo_pen)
        hi = _log_after_penalty(hi_raw, hi_pen)
    else:
        lo, _, _ = _resolve_value_with_penalty(lo_raw, lo_pen, pf, use_penalty=use_penalty)
        hi, _, _ = _resolve_value_with_penalty(hi_raw, hi_pen, pf, use_penalty=use_penalty)
    return lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None

def _gap_12(row:dict)->float:
    try: lam1 = float(row.get("lambda_max","nan"))
    except: lam1 = float("nan")
    lam2 = float("nan")
    s = row.get("lambda_top2","")
    if s and str(s).strip().startswith("("):
        parts = str(s).strip("() ").split(",")
        if len(parts)>1:
            try: lam2=float(parts[1])
            except: pass
    return lam1 - lam2

def _select_front0(rows:List[dict])->List[dict]:
    if not rows: return rows
    if ("is_front0" in rows[0]) and any(str(r.get("is_front0","0"))=="1" for r in rows):
        return [r for r in rows if str(r.get("is_front0","0"))=="1"]
    return rows


def _is_front0(row: dict) -> bool:
    val = row.get("is_front0", "0")
    if isinstance(val, bool):
        return val
    sval = str(val).strip()
    if sval.lower() == "true":
        return True
    if sval.lower() == "false":
        return False
    try:
        return float(val) > 0.5
    except Exception:
        return sval == "1"

# 文件名/字段解析 (n,k) + 系列
_RX_STAGE1 = re.compile(r"stage1_(?:all|pareto)_n(\d+)_k(\d+)")
_RX_GA     = re.compile(r"pareto_front_(?:nk_)?n(\d+)_k(\d+)(?:_|\.csv)")
def _nk_from_fname(fname:str)->Tuple[Optional[int],Optional[int]]:
    m = _RX_STAGE1.search(fname) or _RX_GA.search(fname)
    if m: return int(m.group(1)), int(m.group(2))
    return None,None

def _series_by_fname(base:str)->str:
    name = base.lower()
    if name.startswith("stage1_pareto_") and "_raw" in name:   return "stage1_raw"
    if name.startswith("stage1_pareto_") and "_canon" in name: return "stage1_canon"
    if name.startswith("pareto_front_"):                        return "ga_canon"
    return ""

def _series_for_row(row:dict, base:str)->str:
    s = _series_by_fname(base)
    if s: return s
    is_ga = ("sum_lambda_powers" in row) or ("lower_bound" in row) or ("upper_bound" in row)
    if is_ga: return "ga_canon"
    is_canon = False
    if (row.get("rule_bits_canon","") or "").strip() != "": is_canon = True
    if str(row.get("is_canonical_rep", row.get("is_canon","0"))).strip() == "1": is_canon = True
    return "stage1_canon" if is_canon else "stage1_raw"

def _discover_all_nk(paths: Iterable[str]) -> List[Tuple[int,int]]:
    seen=set()
    for p in paths:
        if not os.path.exists(p): continue
        try:
            with open(p,"r",encoding="utf-8") as f:
                rows=list(csv.DictReader(f))
        except: rows=[]
        if rows and ("n" in rows[0] and "k" in rows[0]):
            for r in rows:
                try: seen.add((int(r["n"]), int(r["k"])))
                except: pass
        else:
            n,k=_nk_from_fname(os.path.basename(p))
            if (n is not None) and (k is not None): seen.add((n,k))
    return sorted(seen)

def _collect_by_series_for_nk(csv_paths: Iterable[str], n:int, k:int, sym_filter: Optional[str] = None)->Dict[str,List[dict]]:
    out={"stage1_raw":[], "stage1_canon":[], "ga_canon":[]}
    for p in csv_paths:
        if not os.path.exists(p): continue
        with open(p,"r",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        if not rows: continue
        # 过滤 (n,k) + 对称模式（可选）
        if "n" in rows[0] and "k" in rows[0]:
            rows=[r for r in rows if str(r.get("n","")).isdigit() and str(r.get("k","")).isdigit()
                  and int(r["n"])==n and int(r["k"])==k]
        else:
            fn_n, fn_k = _nk_from_fname(os.path.basename(p))
            if not (fn_n==n and fn_k==k): rows=[]
        if sym_filter:
            rows=[r for r in rows if str(r.get("sym_mode",""))==sym_filter]
        if not rows: continue
        base=os.path.basename(p)
        for r in rows:
            r["_file"]=p
            out[_series_for_row(r, base)].append(r)
    return out

# =========================
# 前沿曲面/等高线
# =========================
def _extract_key_points(rule_counts: np.ndarray,
                        metrics: np.ndarray,
                        ns_for_rule: np.ndarray) -> List[KeyPoint]:
    xs = np.asarray(rule_counts, float)
    ys = np.asarray(metrics, float)
    ns = np.asarray(ns_for_rule, float)
    m = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ns)
    if not m.any():
        return []
    xs = xs[m]; ys = ys[m]; ns = ns[m]
    key_points: List[KeyPoint] = []

    # 全局最大值
    try:
        idx_max = int(np.nanargmax(ys))
        key_points.append(KeyPoint("max", ns[idx_max], xs[idx_max], ys[idx_max], f"max |R|={int(xs[idx_max])}"))
    except Exception:
        idx_max = None

    # 二阶差分膝点（若失败再退回 L-curve），命名与现有图表一致
    idx_knee = _knee_second(xs, ys, logy=True)
    if idx_knee is None:
        idx_knee = _knee_l(xs, ys, logxy=True)
    if idx_knee is not None and 0 <= idx_knee < len(xs) and np.isfinite(ys[idx_knee]):
        key_points.append(KeyPoint("knee-2Δ", ns[idx_knee], xs[idx_knee], ys[idx_knee], f"knee |R|={int(xs[idx_knee])}"))

    # 最优折中（可能等同 max，但这里保留显式标记便于下游展示）
    opt_idx = _get_optimal_idx(xs, ys, logy=True)
    key_points.append(KeyPoint("Optimal", ns[opt_idx], xs[opt_idx], ys[opt_idx], f"Optimal |R|={int(xs[opt_idx])}"))

    return key_points


def _monotone_idx_with_max(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return indices of a non-decreasing subsequence plus the global max (if outside)."""
    inc_idx: List[int] = []
    if xs.size:
        cur = -np.inf
        for i, y in enumerate(ys):
            if y >= cur:
                inc_idx.append(i)
                cur = y
    inc_idx = np.array(inc_idx, dtype=int)
    try:
        imax = int(np.nanargmax(ys))
    except Exception:
        imax = None
    if (imax is not None) and (imax not in inc_idx):
        inc_idx = np.unique(np.concatenate([inc_idx, np.array([imax], dtype=int)]))
    return inc_idx


def _get_optimal_idx(xs: np.ndarray, ys: np.ndarray, logy: bool = True) -> int:
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if not m.any():
        return 0
    xs = xs[m]; ys = ys[m]
    Y = np.log(np.maximum(ys, 1e-300)) if logy else ys

    try:
        i_max = int(np.nanargmax(Y))
    except Exception:
        return 0

    i_knee = _knee_second(xs, ys, logy=logy)
    if i_knee is None:
        i_knee = _knee_l(xs, ys, logxy=logy)
    i_mur = _unit_best_idx(xs, ys) if xs.size >= 2 else None

    candidates = []
    for c in (i_knee, i_mur, None if i_mur is None else i_mur + 1):
        if c is None:
            continue
        if c >= len(xs) or c < 0:
            continue
        if c >= i_max:
            continue
        candidates.append(c)

    best_idx = i_max
    for c in candidates:
        gain_rem = ys[i_max] - ys[c]
        cost_rem = xs[i_max] - xs[c]
        if cost_rem <= 0:
            continue
        slope_tail = gain_rem / cost_rem
        slope_base = ys[c] / xs[c] if xs[c] > 0 else 0.0
        if slope_tail < 0.2 * slope_base:
            if xs[c] < xs[best_idx]:
                best_idx = c

    return int(best_idx)


def _objective_ylabel(objective_field: Optional[str], apply_penalty: bool) -> str:
    lbl = r"Objective (log $Z$)"
    if objective_field and "per" in objective_field:
        lbl += r" / penalty"
    elif apply_penalty:
        lbl += r" / penalty"
    return lbl


def _prepare_surface_data(front_csvs: Iterable[str],
                          ks: Sequence[int],
                          boundary: Optional[str],
                          sym_mode: Optional[str],
                          metric_field: Optional[str]) -> List[FrontierSurfaceData]:
    paths = _expand_globs(front_csvs)
    rows_by_file = _load_rows(paths)
    all_rows: List[dict] = []
    for rr in rows_by_file.values():
        all_rows.extend(rr)

    outs: List[FrontierSurfaceData] = []
    for k in ks:
        selected: List[dict] = []
        for r in all_rows:
            try:
                if int(r.get("k", -1)) != int(k):
                    continue
                _ = int(r.get("n", -1))
                _ = int(r.get("rule_count", -1))
            except Exception:
                continue
            selected.append(r)

        if not selected:
            continue

        try:
            k_boundary = boundary or _unique_field(selected, "boundary") or config.BOUNDARY_MODE
        except ValueError:
            k_boundary = boundary or config.BOUNDARY_MODE
        try:
            k_sym_mode = sym_mode or _unique_field(selected, "sym_mode") or "perm"
        except ValueError:
            k_sym_mode = sym_mode or "perm"
        filtered: List[dict] = []
        for r in selected:
            b = str(r.get("boundary", "")).strip()
            s = str(r.get("sym_mode", "")).strip()
            if k_boundary and b and (b != k_boundary):
                continue
            if k_sym_mode and s and (s != k_sym_mode):
                continue
            filtered.append(r)

        ns_set: set[int] = set()
        rc_set: set[int] = set()
        z_map: Dict[Tuple[int, int], float] = {}

        for r in filtered:
            try:
                n_val = int(r.get("n"))
                rc_val = int(r.get("rule_count"))
            except Exception:
                continue
            z = _y_metric(r, prefer_field=metric_field)
            if not np.isfinite(z):
                continue
            key = (n_val, rc_val)
            prev = z_map.get(key, -np.inf)
            if z > prev:
                z_map[key] = z
                ns_set.add(n_val)
                rc_set.add(rc_val)

        ns = np.array(sorted(ns_set), dtype=float)
        rcs = np.array(sorted(rc_set), dtype=float)
        if ns.size == 0 or rcs.size == 0:
            continue

        grid = np.full((rcs.size, ns.size), np.nan, dtype=float)
        rc_idx = {rc: i for i, rc in enumerate(rcs)}
        n_idx = {n: j for j, n in enumerate(ns)}
        for (n_val, rc_val), z in z_map.items():
            i = rc_idx.get(rc_val)
            j = n_idx.get(n_val)
            if i is None or j is None:
                continue
            grid[i, j] = z

        best_z = np.full(rcs.shape, np.nan, dtype=float)
        best_n = np.full(rcs.shape, np.nan, dtype=float)
        for i in range(rcs.size):
            row = grid[i]
            if np.isfinite(row).any():
                j = int(np.nanargmax(row))
                best_z[i] = row[j]
                best_n[i] = ns[j]

        # optimal curve across n for this k (per-column optimal rule count/metric)
        optimal_curve: List[Tuple[float, float, float]] = []
        for j in range(ns.size):
            col = grid[:, j]
            if not np.isfinite(col).any():
                continue
            idx_opt = _get_optimal_idx(rcs, col, logy=True)
            if 0 <= idx_opt < len(rcs) and np.isfinite(col[idx_opt]):
                optimal_curve.append((ns[j], rcs[idx_opt], col[idx_opt]))

        key_points = _extract_key_points(rule_counts=rcs, metrics=best_z, ns_for_rule=best_n)
        outs.append(FrontierSurfaceData(
            k=int(k),
            boundary=str(k_boundary),
            sym_mode=str(k_sym_mode),
            metric_field=str(metric_field or "objective_penalized"),
            ns=ns,
            rule_counts=rcs,
            grid=grid,
            best_curve_metric=best_z,
            best_curve_n=best_n,
            key_points=key_points,
            optimal_curve=optimal_curve,
        ))
    if not outs:
        # Avoid crashing if no data found for some K, just warn
        logging.warning("no rows matched the requested ks/boundary/sym_mode")
    return outs


# =========================
# 膝点/单位复杂度（离散）
# =========================
def _knee_second(xs, ys, logy=True):
    xs, ys = np.asarray(xs,float), np.asarray(ys,float)
    if logy: ys = np.log(np.maximum(ys,1e-300))
    if len(xs)<3: return None
    d2 = ys[2:] - 2*ys[1:-1] + ys[:-2]
    idx = int(np.argmax(d2))
    return idx+1

def _knee_l(xs, ys, logxy=True):
    xs, ys = np.asarray(xs,float), np.asarray(ys,float)
    if logxy:
        xs=np.log(np.maximum(xs,1e-9)); ys=np.log(np.maximum(ys,1e-300))
    if len(xs)<3: return None
    x0,y0 = xs[0],ys[0]; vx,vy = xs[-1]-x0, ys[-1]-y0
    vnorm = math.hypot(vx,vy)+1e-15
    imax, dmax = None, -1.0
    for i in range(len(xs)):
        wx,wy = xs[i]-x0, ys[i]-y0
        d = abs(vx*wy - vy*wx)/vnorm
        if d>dmax: dmax, imax = d, i
    return imax

def _unit_best_idx(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if xs.size < 2: return 0
    dy = np.diff(np.log(np.maximum(ys, 1e-300)))  # Δlog y 
    dx = np.diff(xs)
    mur = dy / np.maximum(dx, 1e-9)
    j = int(np.nanargmax(mur))
    return j + 1  # 与差分对齐到右端点


def plot_frontier_surfaces(front_csvs: Iterable[str],
                           ks: Sequence[int],
                           boundary: Optional[str] = None,
                           sym_mode: Optional[str] = None,
                            metric: Optional[str] = "objective_penalized",
                            plot_types: Sequence[str] | str = ("surface",),
                            out_dir: str = "./out_fig",
                            style: str = "default",
                            contour_levels: int = 10,
                            wireframe_stride: int = 1,
                            max_series_per_fig: int = 1) -> Tuple[List[str], List[FrontierSurfaceData]]:
    """绘制 (n, |R|, 目标值) 的前沿曲面/等高线，并标注最佳点路径。

    返回：(输出图片路径列表, 对每个 k 的关键点数据)。
    """
    apply_style(style); os.makedirs(out_dir, exist_ok=True)
    types = [plot_types] if isinstance(plot_types, str) else list(plot_types)
    types = [str(t).lower() for t in types]
    for t in types:
        if t not in ("surface", "wireframe", "contour"):
            raise ValueError("plot_types must be surface, wireframe, or contour")

    data = _prepare_surface_data(front_csvs, ks=ks, boundary=boundary, sym_mode=sym_mode, metric_field=metric)
    cmap = plt.get_cmap("tab10")
    outs: List[str] = []

    for t in types:
        if t in ("surface", "wireframe"):
            for chunk_idx in range(0, len(data), max(1, int(max_series_per_fig))):
                subset = data[chunk_idx:chunk_idx + max(1, int(max_series_per_fig))]
                if not subset:
                    continue
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                proxies: List[mpl.lines.Line2D] = []
                labels: List[str] = []
                for idx, d in enumerate(subset):
                    color = cmap(idx % cmap.N)
                    X, Y = np.meshgrid(d.ns, d.rule_counts)
                    Z = np.ma.array(d.grid, mask=~np.isfinite(d.grid))
                    if t == "surface":
                        ax.plot_surface(X, Y, Z, color=color, alpha=0.65, linewidth=0.3, antialiased=True)
                    else:
                        ax.plot_wireframe(X, Y, Z, color=color, rstride=max(1, wireframe_stride),
                                          cstride=max(1, wireframe_stride), linewidth=0.8, alpha=0.9)
                    # draw optimal curve and its projection
                    if d.optimal_curve:
                        opt_ns, opt_rcs, opt_zs = zip(*d.optimal_curve)
                        ax.plot(opt_ns, opt_rcs, opt_zs, color="red", linewidth=2.5, label=f"k={d.k} optimal path")
                        z_min = ax.get_zlim()[0]
                        ax.plot(opt_ns, opt_rcs, zs=z_min, zdir="z", color="red", linewidth=1.5, linestyle="--", alpha=0.7)
                    proxies.append(mpl.lines.Line2D([0], [0], color=color, lw=2))
                    labels.append(f"k={d.k}")
                ax.set_xlabel("n"); ax.set_ylabel("|R|"); ax.set_zlabel(metric or "objective")
                part = 1 + chunk_idx // max(1, int(max_series_per_fig))
                part_suffix = f" (part {part})" if len(data) > len(subset) else ""
                ax.set_title(f"Frontier surface ({t}) — boundary={subset[0].boundary}, sym={subset[0].sym_mode}{part_suffix}")
                ax.legend(proxies, labels, loc="best")
                fig.tight_layout()
                fname = f"frontier_{t}_k{'-'.join(str(d.k) for d in subset)}_{_shorten(metric or 'objective', 20)}.png"
                out_path = os.path.join(out_dir, fname)
                fig.savefig(out_path, dpi=220)
                plt.close(fig)
                outs.append(out_path)
        elif t == "contour":
            for chunk_idx in range(0, len(data), max(1, int(max_series_per_fig))):
                subset = data[chunk_idx:chunk_idx + max(1, int(max_series_per_fig))]
                if not subset:
                    continue
                fig, ax = plt.subplots()
                for idx, d in enumerate(subset):
                    color = cmap(idx % cmap.N)
                    X, Y = np.meshgrid(d.ns, d.rule_counts)
                    Z = np.ma.array(d.grid, mask=~np.isfinite(d.grid))
                    can_contour = (Z.shape[0] >= 2) and (Z.shape[1] >= 2) and np.isfinite(Z).sum() >= 4
                    if can_contour:
                        cs = ax.contour(X, Y, Z, levels=contour_levels, colors=[color], linewidths=1.0, alpha=0.85)
                        ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.2g}", fontsize=8)
                    else:
                        logging.warning("[viz] skip contour for k=%s (insufficient grid: %s)", d.k, Z.shape)
                    
                    if d.optimal_curve:
                        opt_ns, opt_rcs, _ = zip(*d.optimal_curve)
                        ax.plot(opt_ns, opt_rcs, color="red", linewidth=2.0, linestyle="-", label=f"k={d.k} optimal path")
                        
                    # Only project the Optimal point to avoid clutter
                    kp_sorted = [p for p in d.key_points if p.kind.lower() == "optimal"]
                    if kp_sorted:
                        xs = [p.n for p in kp_sorted]; ys = [p.rule_count for p in kp_sorted]
                        ax.scatter(xs, ys, color="red", marker="*", s=80, zorder=10)
                        for p in kp_sorted:
                            ax.text(p.n, p.rule_count, f"{p.kind}\n{p.metric:.2g}", color="black", fontsize=8,
                                    ha="left", va="bottom")

                ax.set_xlabel("n"); ax.set_ylabel("|R|")
                part = 1 + chunk_idx // max(1, int(max_series_per_fig))
                part_suffix = f" (part {part})" if len(data) > len(subset) else ""
                ax.set_title(f"Frontier contours — boundary={subset[0].boundary}, sym={subset[0].sym_mode}, metric={metric}{part_suffix}")
                ax.legend(loc="best")
                fig.tight_layout()
                fname = f"frontier_contour_k{'-'.join(str(d.k) for d in subset)}_{_shorten(metric or 'objective', 20)}.png"
                out_path = os.path.join(out_dir, fname)
                fig.savefig(out_path, dpi=220)
                plt.close(fig)
                outs.append(out_path)

    return outs, data

# =========================
# 绘制三图（指定 n,k）
# =========================
def _bucket_best_and_band(rows: List[dict], use_logy: bool, objective_field: Optional[str] = None, use_penalty: bool = True) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    bucket: Dict[int, Dict[str, float]] = {}
    mins: Dict[int, float] = {}
    maxs: Dict[int, float] = {}
    # prefer front0 rows if available; supplement missing rule_counts with best from non-front0 rows
    front_rows = [r for r in rows if _is_front0(r)]
    non_front_rows = [r for r in rows if not _is_front0(r)]
    # process front rows first (if any), then backfill with non-front rows per rule_count
    row_iter = front_rows if front_rows else rows
    for r in row_iter:
        try:
            rc = int(r["rule_count"])
        except:
            continue
        est = _y_metric(r, prefer_field=objective_field, use_penalty=use_penalty)
        lo, hi = _bounds(r, use_penalty=use_penalty, objective_field=objective_field)
        cur = bucket.get(rc)
        if (cur is None) or (est > cur["est"]):
            bucket[rc] = {"est": est, "lo": lo, "hi": hi}
        if np.isfinite(est):
            mins[rc] = min(mins.get(rc, +np.inf), est)
            maxs[rc] = max(maxs.get(rc, -np.inf), est)
    # supplement per rule_count from non-front rows where front rows are missing that rc
    if front_rows:
        present_rc = set(bucket.keys())
        for r in non_front_rows:
            try:
                rc = int(r["rule_count"])
            except:
                continue
            if rc in present_rc:
                continue
            est = _y_metric(r, prefer_field=objective_field, use_penalty=use_penalty)
            lo, hi = _bounds(r, use_penalty=use_penalty, objective_field=objective_field)
            bucket[rc] = {"est": est, "lo": lo, "hi": hi}
            if np.isfinite(est):
                mins[rc] = min(mins.get(rc, +np.inf), est)
                maxs[rc] = max(maxs.get(rc, -np.inf), est)
    if not bucket:
        return (np.array([]),)*4
    xs  = np.array(sorted(bucket.keys()), float)
    est = np.array([bucket[int(x)]["est"] for x in xs], float)
    los, his = [], []
    for x in xs:
        rc = int(x)
        d = bucket[rc]
        lo, hi = d["lo"], d["hi"]
        est_val = d["est"]
        if (lo is None) or (hi is None) or (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi < lo):
            lo = est_val
            hi = est_val
            if np.isfinite(est_val):
                mins[rc] = min(mins.get(rc, +np.inf), est_val)
                maxs[rc] = max(maxs.get(rc, -np.inf), est_val)
        elif np.isfinite(est_val):
            mins[rc] = min(mins.get(rc, +np.inf), est_val)
            maxs[rc] = max(maxs.get(rc, -np.inf), est_val)
        los.append(lo); his.append(hi)
    los = np.array(los,float); his = np.array(his,float)
    if use_logy:
        est = np.where(est>0, est, np.nan)
        los = np.where(los>0, los, np.nan)
        his = np.where(his>0, his, np.nan)
        log_est = np.log(est)
        log_lo = np.log(los)
        log_hi = np.log(his)
        log_lo = np.minimum(log_lo, log_est)
        log_hi = np.maximum(log_hi, log_est)
        est, los, his = np.exp(log_est), np.exp(log_lo), np.exp(log_hi)
    m = np.isfinite(est) & np.isfinite(los) & np.isfinite(his)
    if use_logy:
        m = m & (est > 0) & (los > 0) & (his > 0)
    return xs[m], est[m], los[m], his[m]

def plot_three_raw_canon_for_nk(front_paths: List[str],
                                n: int, k: int,
                                out_dir: str = "./out_fig",
                                y_log: bool = False,
                                style: str = "default",
                                sym_filter: Optional[str] = None,
                                objective_field: Optional[str] = None,
                                apply_penalty: bool = True,
                                show_band: bool = False) -> Tuple[str,str,str]:
    apply_style(style); os.makedirs(out_dir, exist_ok=True)
    series = _collect_by_series_for_nk(front_paths, n, k, sym_filter=sym_filter)
    order  = ["stage1_raw", "stage1_canon", "ga_canon"]
    markers= {"stage1_raw":"s", "stage1_canon":"o", "ga_canon":"^"}
    linest = {"stage1_raw":"--", "stage1_canon":"-", "ga_canon":"-."}
    jitter = {"stage1_raw":-0.07, "stage1_canon":+0.07, "ga_canon":+0.00}
    labels = {"stage1_raw":"stage1_raw", "stage1_canon":"stage1_canon", "ga_canon":"ga_canon"}

    # (A) scatter
    fig1, ax1 = plt.subplots(); anyp=False
    for key in order:
        rows = series.get(key, [])
        xs, ys = [], []
        for r in rows:
            try:
                xs.append(int(r["rule_count"])); ys.append(_y_metric(r, prefer_field=objective_field, use_penalty=apply_penalty))
            except: pass
        xs, ys = np.asarray(xs,float), np.asarray(ys,float)
        m = np.isfinite(ys) & (ys>0 if y_log else np.isfinite(ys))
        xs, ys = xs[m], ys[m]
        if xs.size==0: continue
        ax1.scatter(xs + jitter[key], ys, label=labels[key], alpha=0.9, marker=markers[key]); anyp=True
    if y_log: ax1.set_yscale("log")
    ax1.set_xlabel("|R|"); ax1.set_ylabel(_objective_ylabel(objective_field, apply_penalty))
    ax1.set_title(f"(n={n}, k={k}) stage1_raw vs stage1_canon vs ga_canon — Scatter")
    if anyp: ax1.legend(loc="best")
    fig1.tight_layout()
    p_sc = os.path.join(out_dir, f"nk_n{n}_k{k}_scatter{'_log' if y_log else ''}.png")
    fig1.savefig(p_sc, dpi=200); plt.close(fig1)

    # (B) growth + knees + unit-best + band
    fig2, ax2 = plt.subplots(); anyp=False
    for key in order:
        rows = series.get(key, [])
        xs, est, lo, hi = _bucket_best_and_band(rows, use_logy=y_log, objective_field=objective_field, use_penalty=apply_penalty)
        if xs.size==0: continue
        xj = xs + jitter[key]
        ax2.plot(xj, est, marker=markers[key], linestyle=linest[key], alpha=0.95, label=labels[key])
        vb = np.isfinite(lo) & np.isfinite(hi) & (hi>=lo)
        if show_band and vb.any():
            ax2.fill_between(xj[vb], lo[vb], hi[vb], alpha=0.12, linewidth=0)
        
        # Mark only Optimal Point
        idx_opt = _get_optimal_idx(xs, est, logy=y_log)
        if 0 <= idx_opt < len(xs):
            ax2.scatter([xj[idx_opt]], [est[idx_opt]], s=120, marker="*", color="red", zorder=10, 
                        label=f"{labels[key]}: Optimal |R|={int(xs[idx_opt])}")

        anyp=True
    if y_log: ax2.set_yscale("log")
    ylabel = _objective_ylabel(objective_field, apply_penalty)
    ax2.set_xlabel("|R|"); ax2.set_ylabel(ylabel)
    ax2.set_title(f"(n={n}, k={k}) Growth Curves with Optimal Points")
    if anyp: ax2.legend(loc="best", ncol=2)
    fig2.tight_layout()
    p_gr = os.path.join(out_dir, f"nk_n{n}_k{k}_growth_knees{'_log' if y_log else ''}.png")
    fig2.savefig(p_gr, dpi=200); plt.close(fig2)

    # (C) knees & spectral gap（含 unit-best）
    fig3, ax3 = plt.subplots(); ax4 = ax3.twinx()
    painted_gap=False; painted_trace=False
    for key in order:
        rows = series.get(key, [])
        xs, est, _, _ = _bucket_best_and_band(rows, use_logy=y_log, objective_field=objective_field, use_penalty=apply_penalty)
        if xs.size==0: continue
        # 同 |R| 选择 best-y 的 gap
        bucket_gap = {}
        for r in rows:
            try:
                rc = int(r["rule_count"]); y = _y_metric(r, prefer_field=objective_field)
                if not np.isfinite(y): continue
                cur = bucket_gap.get(rc)
                if (cur is None) or (y>cur["y"]): bucket_gap[rc] = {"y":y, "gap":_gap_12(r)}
            except: pass
        gxs = sorted(bucket_gap.keys()); gaps = np.array([bucket_gap[x]["gap"] for x in gxs], float)
        gxs = np.array(gxs, float)

        xj = xs + jitter[key]
        ax3.plot(xj, est, marker=markers[key], linestyle=linest[key], alpha=0.95,
                 label=(labels[key] if not painted_trace else None))
        painted_trace=True or painted_trace
        if y_log: ax3.set_yscale("log")
        if np.isfinite(gaps).any():
            ax4.plot(gxs, gaps, marker="^", linestyle=":", alpha=0.9,
                     label=("spectral gap" if not painted_gap else None), color="tab:orange")
            painted_gap=True or painted_gap
        
        # Mark only Optimal Point
        idx_opt = _get_optimal_idx(xs, est, logy=y_log)
        if 0 <= idx_opt < len(xs):
            ax3.scatter([xj[idx_opt]], [est[idx_opt]], s=120, marker="*", color="red", zorder=10)

    ax3.set_xlabel("|R|"); ax3.set_ylabel(ylabel)
    ax4.set_ylabel(r"$\lambda_1-\lambda_2$")
    ax3.set_title(f"(n={n}, k={k}) Optimal & Spectral Gap — stage1_raw vs stage1_canon vs ga_canon")
    ax3.legend(loc="upper left")
    if painted_gap: ax4.legend(loc="upper right")
    fig3.tight_layout()
    p_kg = os.path.join(out_dir, f"nk_n{n}_k{k}_knees_gap{'_log' if y_log else ''}.png")
    fig3.savefig(p_kg, dpi=200); plt.close(fig3)

    return p_sc, p_gr, p_kg

# =========================
# 批量驱动（兼容 rd_cli.py: viz-all）
# =========================
def plot_all(front_paths: List[str],
             n: Optional[int]=None, k: Optional[int]=None,
             out_dir: str="./out_fig",
             y_log: bool=False,
             style: str="default",
             sym_filter: Optional[str] = None,
             objective_field: Optional[str] = None,
             apply_penalty: bool = True,
             show_band: bool = False)->List[str]:
    """
    兼容 rd_cli.py:
      - 若给定 n,k：仅绘制该 (n,k) 的三张图；
      - 否则：自动发现所有(n,k)，逐个绘制三张图。
    返回：输出图片路径列表。
    """
    apply_style(style); os.makedirs(out_dir, exist_ok=True)
    outs=[]
    if (n is not None) and (k is not None):
        outs += list(plot_three_raw_canon_for_nk(front_paths, n, k, out_dir=out_dir, y_log=y_log, style=style, sym_filter=sym_filter, objective_field=objective_field, apply_penalty=apply_penalty, show_band=show_band))
        return outs
    # 自动发现
    for n0,k0 in _discover_all_nk(front_paths):
        outs += list(plot_three_raw_canon_for_nk(front_paths, n0, k0, out_dir=out_dir, y_log=y_log, style=style, sym_filter=sym_filter, objective_field=objective_field, apply_penalty=apply_penalty, show_band=show_band))
    return outs

# =========================
# 可选：本文件直接运行（调试）
# =========================
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", nargs="+", required=True)
    ap.add_argument("--out-dir", default="./out_fig")
    ap.add_argument("--style", default="default", choices=list(_STYLES.keys()))
    ap.add_argument("--y-log", action="store_true")
    ap.add_argument("--sym-filter", default=None, help="only plot rows whose sym_mode matches this value")
    ap.add_argument("--n", type=int); ap.add_argument("--k", type=int)
    args = ap.parse_args()
    plot_all(args.front, n=args.n, k=args.k, out_dir=args.out_dir, y_log=args.y_log, style=args.style, sym_filter=args.sym_filter)

if __name__ == "__main__":
    _cli()
