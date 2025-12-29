# -*- coding: utf-8 -*-
"""Unified CLI for rules-diversity (final, refactored).
Commands: stage1, ga, viz, entropy, viz-all, motifs, motifs-explain, archetypes, symmetry.
"""
from __future__ import annotations
import sys, argparse, logging, glob, os, numpy as np
import inspect
import csv
from pathlib import Path
from typing import List, Optional

# Ensure in-repo execution works without PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rules.bootstrap import ensure_repo_on_path

ROOT = ensure_repo_on_path()

from rules import config as _config

# optional modules (标记可用性；真正使用时在各自 cmd_* 内延迟导入)
try:
    from rules import structures as _structures
    _HAS_STRUCTURES = True
except Exception:
    _HAS_STRUCTURES = False
try:
    from rules import symmetry as _sym
    _HAS_SYMMETRY = True
except Exception:
    _HAS_SYMMETRY = False


# 轻量 I/O 工具（与 utils_io 对齐；若你已有实现，也可直接 from rules.utils_io import ...）
def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def expand_globs(paths: List[str]) -> List[str]:
    outs: list[str] = []
    for p in paths:
        outs.extend(glob.glob(p))
    # 去重且保持顺序
    seen = set(); uniq = []
    for x in outs:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def _setup_logging(verbosity: int = 1):
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

# --- add: global flag pre-parser so --verbosity can appear anywhere ---
def _preparse_global_flags(argv: list[str]) -> int:
    """Extract --verbosity/-v from argv regardless of position; remove it from argv and return level."""
    v = None
    i = 1
    # work on a copy; we'll mutate argv in place (sys.argv)
    while i < len(argv):
        tok = argv[i]
        if tok in ("--verbosity", "-v"):
            # value can be next token or --verbosity=2
            if "=" in tok:
                try:
                    v = int(tok.split("=", 1)[1])
                except Exception:
                    v = 1
                # remove this token only
                argv.pop(i)
                continue
            # consume next token as value
            if i + 1 < len(argv):
                try:
                    v = int(argv[i + 1])
                    # remove flag + value
                    argv.pop(i)      # remove flag
                    argv.pop(i)      # remove value at same index
                    continue
                except Exception:
                    # malformed value; drop flag only, fall back to INFO
                    argv.pop(i)
                    v = 1
                    continue
            else:
                # flag at end with no value -> default INFO
                argv.pop(i)
                v = 1
                continue
        i += 1
    # normalize
    if v is None:
        v = 1
    return max(0, min(2, v))


# ---------------- commands ----------------
def cmd_stage1(args):
    # 延迟导入，避免其它子命令受依赖影响
    from rules.stage1_exact import scan_all_rules_exact
    n, k = int(args.n), int(args.k)
    out_csv = ensure_dir(args.out_csv)
    logging.info(f"[stage1] exhaustive scan n={n}, k={k}, boundary={args.boundary}, canonical={not args.no_canon}, archetypes={not args.no_archetypes}")
    try:
        all_csv, pareto_csv = scan_all_rules_exact(
            n=n, k=k,
            out_csv_dir=str(out_csv),
            canonical=(not args.no_canon),
            mark_archetypes=(not args.no_archetypes),
            save_rows_m=True,
            boundary=args.boundary,
            run_tag=f"stage1_n{n}_k{k}",
        )
    except Exception:
        logging.exception("[stage1] failed"); sys.exit(2)
    print("  all   CSV:", all_csv)
    print("  pareto CSV:", pareto_csv)

def cmd_ga(args):
    from rules.ga import GAConfig, ga_search_with_batch
    from rules.eval import evaluate_rules_batch, summarize_trace_comparison
    n, k = int(args.n), int(args.k)
    os.makedirs(args.out_csv, exist_ok=True)

    sym_modes = [s.strip() for s in str(getattr(args, "sym", "perm") or "perm").split(",") if s.strip()]
    if not sym_modes:
        sym_modes = ["perm"]

    if args.reuse:
        existed = expand_globs([os.path.join(args.out_csv, f"pareto_front_n{n}_k{k}_*.csv")])
        if existed:
            print("[GA] reuse on; found existing:"); [print("  ",p) for p in existed]; return

    conf = GAConfig(
        pop_size=args.pop_size, generations=args.generations,
        p_mut=args.p_mut, p_cx=args.p_cx, elite_keep=args.elite_keep,
        device=args.device if args.device else None,
        use_lanczos=not args.no_lanczos,
        r_vals=args.r_vals, power_iters=args.power_iters,
        trace_mode=args.trace_mode, hutch_s=args.hutch_s,
        lru_rows_capacity=args.lru_rows_capacity, batch_streams=args.batch_streams,
        progress_every=args.progress_every, fast_eval=args.fast_eval,
        seed_from_stage1=args.seed_from_stage1, max_stage1_seeds=args.max_stage1_seeds,
        sym_mode=sym_modes[0],
        enable_exact=not args.no_exact,
        enable_spectral=not args.no_spectral,
        exact_threshold=args.exact_threshold,
        boundary=args.boundary,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    pareto, csv_front, csv_gen = ga_search_with_batch(n, k, conf, out_csv_dir=args.out_csv)
    print("[GA] Front CSV:", csv_front)
    print("[GA] Gen   CSV:", csv_gen)

    # 追加对照：在相同个体上测试其他对称模式
    if len(sym_modes) > 1 and pareto:
        bits_batch = [b for b, _ in pareto]
        for mode in sym_modes[1:]:
            try:
                reports = evaluate_rules_batch(
                    n=n, k=k, bits_list=bits_batch,
                    sym_mode=mode,
                    boundary=args.boundary,
                    device=args.device if args.device else conf.device,
                    use_lanczos=not args.no_lanczos,
                    r_vals=args.r_vals, power_iters=args.power_iters,
                    trace_mode=args.trace_mode, hutch_s=args.hutch_s,
                    lru_rows=None, max_streams=args.batch_streams,
                    enable_exact=not args.no_exact,
                    enable_spectral=not args.no_spectral,
                    exact_threshold=args.exact_threshold,
                    cache_dir=args.cache_dir,
                    use_cache=not args.no_cache,
                )
                summarize_trace_comparison(reports, logger=logging.getLogger(__name__))
                best = max((float(r.get("sum_lambda_powers", -1e300)) for r in reports), default=float("nan"))
                mmin = min((int(r.get("rows_m", 0)) for r in reports), default=0)
                mmax = max((int(r.get("rows_m", 0)) for r in reports), default=0)
                print(f"[GA][sym={mode}] sum_lambda_powers_best={best:.3e}, rows_m_range=[{mmin},{mmax}], k_sym={reports[0].get('k_sym', k) if reports else k}")
            except Exception:
                logging.exception("[GA] symmetry compare failed for mode=%s", mode)


def cmd_entropy(args):
    from rules.viz import plot_entropy_convergence, apply_style
    import numpy as np
    apply_style(args.style)
    bits = np.array([1 if ch == '1' else 0 for ch in args.bits.strip()], dtype=np.uint8)
    try:
        outs = plot_entropy_convergence(
            rule_bits=bits, k=args.k,
            n_min=args.n_min, n_max=args.n_max,
            device=args.device, out_dir=args.out_dir, style=args.style, logy=args.logy
        )
        for p in outs:
            print("[entropy] saved:", os.path.abspath(p))
    except Exception:
        logging.exception("[entropy] failed"); sys.exit(1)

def cmd_viz_all(args):
    from rules.viz import apply_style, plot_all, plot_three_raw_canon_for_nk
    apply_style(args.style)
    front_paths = args.front or []
    if (args.n is not None) and (args.k is not None):
        # 针对单一 (n,k)：raw vs canon 三图
        try:
            p_sc, p_gr, p_kg = plot_three_raw_canon_for_nk(
                front_paths=front_paths, n=int(args.n), k=int(args.k),
                out_dir=args.out_dir, y_log=bool(args.y_log), style=args.style
            )
            print("[viz-all(nk)] saved:", p_sc, p_gr, p_kg)
        except Exception:
            import sys, logging; logging.exception("[viz-all(nk)] failed"); sys.exit(1)
    else:
        # 聚合所有 CSV：混合三图 + 可选熵曲线
        try:
            outs = plot_all(front_paths=front_paths, out_dir=args.out_dir,
                            y_log=bool(args.y_log), style=args.style,
                            entropy_bits=None if args.entropy_bits is None else
                                np.array([1 if ch=='1' else 0 for ch in str(args.entropy_bits).strip()], dtype=np.uint8),
                            entropy_k=args.entropy_k,
                            entropy_n_min=args.n_min, entropy_n_max=args.n_max,
                            device=args.device)
            print("[viz-all] saved:", *outs, sep="\n  - ")
        except Exception:
            import sys, logging; logging.exception("[viz-all] failed"); sys.exit(1)

def cmd_motifs(args):
    from rules.motifs import analyze_fronts_for_knees, analyze_fronts_for_mur
    paths = expand_globs(args.front)
    if not paths:
        print("[motifs] no front CSV matched"); sys.exit(1)

    ensure_dir(args.out_csv); ensure_dir(args.out_dir)

    # 1) 先导出膝点三表（examples/summary/global）+ 基础图件（和你现有一致）
    try:
        ex_csv_knee, sum_csv_knee, glob_csv_knee, figs_knee = analyze_fronts_for_knees(
            csv_paths=paths,
            out_csv_dir=args.out_csv,
            out_fig_dir=args.out_dir,
            style=args.style,
            logy=args.y_log
        )
    except Exception:
        logging.exception("[motifs] knee export failed")
        sys.exit(1)

    # 2) 再导出 MUR 三表（与膝点同目录，文件名以 motif_mur_* 开头；不覆盖膝点）
    try:
        ex_csv_mur, sum_csv_mur, glob_csv_mur, figs_mur = analyze_fronts_for_mur(
            csv_paths=paths,
            out_csv_dir=args.out_csv,
            out_fig_dir=args.out_dir,
            style=args.style,
            logy=args.y_log
        )
    except Exception:
        # MUR 不是强制的；如果失败，给出告警但不中断膝点的流程
        logging.exception("[motifs] MUR export failed")
        ex_csv_mur = sum_csv_mur = glob_csv_mur = None
        figs_mur = []

    # 3) 写入索引：同时记录膝点与 MUR 的路径，方便 Cell 自动解析
    idx_path = Path(args.out_csv) / "motifs_index.txt"
    with open(idx_path, "w", encoding="utf-8") as f:
        # 膝点
        f.write(f"examples={os.path.abspath(ex_csv_knee)}\n")
        f.write(f"summary={os.path.abspath(sum_csv_knee)}\n")
        f.write(f"global={os.path.abspath(glob_csv_knee)}\n")
        for i, p in enumerate(figs_knee or []):
            f.write(f"fig{i+1}={os.path.abspath(p)}\n")

        # MUR（如存在）
        if ex_csv_mur:
            f.write(f"mur_examples={os.path.abspath(ex_csv_mur)}\n")
        if sum_csv_mur:
            f.write(f"mur_summary={os.path.abspath(sum_csv_mur)}\n")
        if glob_csv_mur:
            f.write(f"mur_global={os.path.abspath(glob_csv_mur)}\n")
        for i, p in enumerate(figs_mur or []):
            f.write(f"mur_fig{i+1}={os.path.abspath(p)}\n")

    # 4) 控制台输出
    print("[motifs] (knee) examples:", os.path.abspath(ex_csv_knee))
    print("[motifs] (knee) summary :", os.path.abspath(sum_csv_knee))
    print("[motifs] (knee) global  :", os.path.abspath(glob_csv_knee))
    if figs_knee:
        print("[motifs] (knee) figs:")
        [print("  ", os.path.abspath(p)) for p in figs_knee]
    if ex_csv_mur:
        print("[motifs] (mur)  examples:", os.path.abspath(ex_csv_mur))
        print("[motifs] (mur)  summary :", os.path.abspath(sum_csv_mur))
        print("[motifs] (mur)  global  :", os.path.abspath(glob_csv_mur))
        if figs_mur:
            print("[motifs] (mur)  figs:")
            [print("  ", os.path.abspath(p)) for p in figs_mur]
    print("[motifs] index  :", os.path.abspath(idx_path))


def cmd_motifs_explain(args):
    from rules.motifs_explainer import main as _main  # 若在包内，按实际导入修正；或直接复用 run()
    # 直接通过命令行转发给 motifs_explainer.py
    # 兼容：这里我们显式拼装 argv 调用
    import sys
    argv = [
        "--examples", args.examples,
        "--out-csv",  args.out_csv,
        "--out-dir",  args.out_dir,
        "--style",    args.style or "ieee",
        "--topN",     str(args.topN),
        "--tree-depth", str(args.tree_depth),
        "--tree-min-leaf", str(args.tree_min_leaf),
        "--seed",     str(args.seed),
    ]
    if args.include_growth:
        argv.append("--include-growth")
    # 直接调用模块 main
    sys.argv = ["motifs_explainer.py"] + argv
    _main()


def cmd_archetypes(args):
    from rules.viz import apply_style
    if not globals().get("_HAS_STRUCTURES", False):
        print("[archetypes] rules.structures not found"); return
    apply_style(args.style)
    n, k = int(args.n), int(args.k)
    ensure_dir(args.out_csv)
    types = [t.strip() for t in args.types.split(",")] if args.types else ["star","cycle","bip","shortloop"]
    try:
        result = _structures.scan_archetypes(n=n, k=k, types=types, top_m=args.top_m, out_csv_dir=args.out_csv, seed=args.seed)
    except TypeError:
        result = _structures.scan_archetypes(n, k, types, args.top_m, args.out_csv, args.seed)
    except Exception:
        logging.exception("[archetypes] scan failed"); sys.exit(1)
    try:
        if hasattr(_structures, "plot_archetypes"):
            csvs = []
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, (list, tuple)): csvs.extend(v)
                    elif isinstance(v, str): csvs.append(v)
            elif isinstance(result, (list, tuple)):
                csvs = list(result)
            figs = _structures.plot_archetypes(csv_paths=csvs, out_dir=args.out_dir, style=args.style)
            if figs:
                print("[archetypes] figs:"); [print("  ", p) for p in figs]
    except Exception:
        logging.exception("[archetypes] plot failed")
    print("[archetypes] CSV dir:", os.path.abspath(args.out_csv))


def cmd_symmetry(args):
    from rules.viz import apply_style
    from rules.utils_io import ensure_dir, expand_globs
    from rules.symmetry import summarize_front_symmetry, plot_symmetry_examples

    apply_style(args.style)
    ensure_dir(args.out_csv); ensure_dir(args.out_dir)

    front_paths = expand_globs(args.front) if args.front else []
    geo_ops = [g.strip() for g in (args.geo or "rot,ref,trans").split(",")]

    summary_csv = summarize_front_symmetry(
        front_paths=front_paths,
        n=int(args.n), k=int(args.k),
        geo_ops=geo_ops,
        state_perm=bool(args.state_perm),
        samples=int(args.samples),
        out_csv_dir=str(args.out_csv),
        enum_limit=int(args.enum_limit),
        knee_only=bool(args.knee_only),
        motifs_examples=str(args.motifs_examples) if args.motifs_examples else None,
        reuse=bool(args.reuse)
    )
    print("[symmetry] summary:", summary_csv)

    try:
        figs = plot_symmetry_examples(summary_csv=summary_csv,
                                      out_dir=str(args.out_dir),
                                      style=args.style)
        if figs:
            print("[symmetry] figs:"); [print("  ", p) for p in figs]
    except Exception:
        import logging; logging.exception("[symmetry] plot failed")
    print("[symmetry] done.")


def main():
    # 允许 --verbosity/-v 出现在任意位置（包括子命令之后）
    # 我们先把它从 sys.argv 里取出并设置日志，再交给 argparse 解析剩余参数
    verbosity = _preparse_global_flags(sys.argv)
    _setup_logging(verbosity)

    parser = argparse.ArgumentParser(description="rules-diversity unified CLI",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 注意：这里不再声明全局 --verbosity；已由预解析负责
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # --- stage1 ---
    sp = subparsers.add_parser("stage1", help="小规模精确计数（穷举、前沿、原型标注）")
    sp.add_argument("--n", type=int, required=True)
    sp.add_argument("--k", type=int, required=True)
    sp.add_argument("--boundary", default=_config.BOUNDARY_MODE, choices=["torus", "open"],
                    help="网格边界条件：torus=平铺环面，open=有界不回绕")
    sp.add_argument("--out-csv", default="results/out_csv")
    sp.add_argument("--no-canon", action="store_true")
    sp.add_argument("--no-archetypes", action="store_true")
    sp.add_argument("--style", default="default", choices=["default","ieee","acm","nature"])
    sp.set_defaults(func=cmd_stage1)

    # --- ga ---
    sp = subparsers.add_parser("ga", help="运行 NSGA-II 进行规则搜索");
    sp.add_argument("--n", type=int, required=True); sp.add_argument("--k", type=int, required=True)
    sp.add_argument("--out-csv", default="results/out_csv"); sp.add_argument("--reuse", action="store_true")
    sp.add_argument("--device", default=None, choices=["cpu","cuda",None])
    sp.add_argument("--pop-size", type=int, default=24); sp.add_argument("--generations", type=int, default=10)
    sp.add_argument("--p-mut", type=float, default=0.08); sp.add_argument("--p-cx", type=float, default=0.85)
    sp.add_argument("--elite-keep", type=int, default=6)
    sp.add_argument("--boundary", default=_config.BOUNDARY_MODE, choices=["torus", "open"],
                    help="网格边界条件；torus=环面，open=开放边界（与谱/精确评估共用缓存隔离）")
    sp.add_argument("--no-lanczos", action="store_true")
    sp.add_argument("--r-vals", type=int, default=3); sp.add_argument("--power-iters", type=int, default=50)
    sp.add_argument("--trace-mode", default="hutchpp", choices=["hutchpp","hutch","lanczos_sum"])
    sp.add_argument("--hutch-s", type=int, default=24)
    sp.add_argument("--lru-rows-capacity", type=int, default=128); sp.add_argument("--batch-streams", type=int, default=2)
    sp.add_argument("--no-exact", action="store_true", help="禁用精确计数/对照（默认开启，受阈值限制）")
    sp.add_argument("--no-spectral", action="store_true", help="禁用谱估计（仅依赖精确计数）")
    sp.add_argument("--exact-threshold", default=_config.EXACT_THRESHOLD,
                    help="精确计数阈值：如 nk<=12 或 rows<=500000")
    sp.add_argument("--cache-dir", default=str(_config.EVAL_CACHE_DIR), help="评估结果缓存目录")
    sp.add_argument("--no-cache", action="store_true", help="禁用评估缓存（始终重新计算）")
    # 新增
    sp.add_argument("--progress-every", type=int, default=2)
    sp.add_argument("--fast-eval", action="store_true")
    sp.add_argument("--seed-from-stage1", action="store_true")
    sp.add_argument("--max-stage1-seeds", type=int, default=256)
    sp.add_argument("--sym", default="perm", help="对称模式: none|perm|perm+swap，可逗号分隔对照输出")
    sp.set_defaults(func=cmd_ga)

    # --- entropy ---
    sp = subparsers.add_parser("entropy", help="指定规则的条带熵收敛曲线")
    sp.add_argument("--bits", required=True)
    sp.add_argument("--k", type=int, required=True)
    sp.add_argument("--n-min", type=int, default=3)
    sp.add_argument("--n-max", type=int, default=10)
    sp.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    sp.add_argument("--out-dir", default="out_fig")
    sp.add_argument("--style", default="default", choices=["default","ieee","acm","nature"])
    sp.add_argument("--logy", action="store_true")
    sp.set_defaults(func=cmd_entropy)

    # --- viz-all ---
    sp = subparsers.add_parser("viz-all", help="可视化：不带 (n,k) 输出混合三图；带 --n --k 则输出 raw/canon 对比三图")
    sp.add_argument("--front", nargs="+", required=True)
    sp.add_argument("--out-dir", default="out_fig")
    sp.add_argument("--y-log", action="store_true")
    sp.add_argument("--style", default="default", choices=["default","ieee","acm","nature"])
    # 针对 (n,k) 的 raw/canon 对比
    sp.add_argument("--n", type=int, default=None)
    sp.add_argument("--k", type=int, default=None)
    # 可选熵曲线（只在不指定 (n,k) 时才会尝试）
    sp.add_argument("--entropy-bits", default=None)
    sp.add_argument("--entropy-k", type=int, default=None)
    sp.add_argument("--n-min", type=int, default=3)
    sp.add_argument("--n-max", type=int, default=10)
    sp.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    sp.set_defaults(func=cmd_viz_all)

    # --- motifs ---
    sp = subparsers.add_parser("motifs", help="膝点关键结构识别与跨(n,k)归纳")
    sp.add_argument("--front", nargs="+", required=True,
                    help="前沿CSV：stage1_pareto_* 与/或 pareto_front_*（支持通配）")
    sp.add_argument("--out-csv", default="results/motifs")
    sp.add_argument("--out-dir", default="out_fig")
    sp.add_argument("--style", default="default", choices=["default","ieee","acm","nature"])
    sp.add_argument("--y-log", action="store_true")
    sp.set_defaults(func=cmd_motifs)

    # --- motifs-explain ---
    sp = subparsers.add_parser("motifs-explain", help="解释膝点结构（Δ特征）并出图，不重复生成 viz 的 nk 增长曲线。")
    sp.add_argument("--examples", required=True, help="motif_knee_examples.csv")
    sp.add_argument("--out-csv",  required=True)
    sp.add_argument("--out-dir",  required=True)
    sp.add_argument("--style", default="ieee")
    sp.add_argument("--topN", type=int, default=20)
    sp.add_argument("--tree-depth", type=int, default=3)
    sp.add_argument("--tree-min-leaf", type=int, default=8)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--include-growth", action="store_true", help="通常关闭；viz 已生成 nk 增长曲线。")
    sp.set_defaults(func=cmd_motifs_explain)


    # --- archetypes ---
    sp = subparsers.add_parser("archetypes", help="结构原型扫描与可视化（需 rules.structures）")
    sp.add_argument("--n", type=int, required=True)
    sp.add_argument("--k", type=int, required=True)
    sp.add_argument("--types", default="star,cycle,bip,shortloop")
    sp.add_argument("--top-m", type=int, default=8)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out-csv", default="results/archetypes")
    sp.add_argument("--out-dir", default="out_fig")
    sp.add_argument("--style", default="default", choices=["default","ieee","acm","nature"])
    sp.set_defaults(func=cmd_archetypes)

    # --- symmetry ---
    sp = subparsers.add_parser("symmetry", help="对称性“前-后”计数与示例（需 rules.symmetry）")
    sp.add_argument("--front", nargs="+", help="若缺省则走 --bits 单例分析（此处我们不再支持单例，保留接口兼容）")
    sp.add_argument("--n", type=int, required=True)
    sp.add_argument("--k", type=int, required=True)
    sp.add_argument("--geo", default="rot,ref,trans")
    sp.add_argument("--state-perm", action="store_true")
    sp.add_argument("--samples", type=int, default=6)
    sp.add_argument("--enum-limit", type=int, default=1000000)
    sp.add_argument("--knee-only", action="store_true")
    sp.add_argument("--motifs-examples", default=None, help="motif_knee_examples.csv 路径")
    sp.add_argument("--reuse", action="store_true")
    sp.add_argument("--out-csv", default="results/symmetry")
    sp.add_argument("--out-dir", default="out_fig")
    sp.add_argument("--style", default="default", choices=["default","ieee","acm","nature"])
    sp.set_defaults(func=cmd_symmetry)

    args = parser.parse_args()
    # 已在最前面设置好日志，这里直接执行
    args.func(args)

if __name__ == "__main__":
    main()
