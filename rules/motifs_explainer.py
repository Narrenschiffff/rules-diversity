# -*- coding: utf-8 -*-
"""
motifs_explainer.py — 膝点结构驱动的解释与可视化（仅做“结构/谱”分析，不重复生成 nk 增长曲线）

输入（来自 rd_cli.py motifs 的产物）：
  - examples: motif_knee_examples.csv
可选：
  - summary : motif_knee_summary.csv
  - global  : motif_global_report.csv

输出（到 --out-csv / --out-dir）：
  CSV：motif_delta_dataset.csv / motif_feature_importance_*.csv / motif_perm_importance.csv /
       motif_lr_coeffs.csv / spectrum_structure_joint.csv / integrity_report.json
  图： knee_motif_bar.png / knee_motif_heatmap.png / motif_importance_* 成套（含/不含 gap）

要点：
- 统一字段：arch_has_quad→c4，arch_has_pent→c5（避免“缺列”）
- 若 summary 缺某些结构行，用 examples 现算 pre/knee/post 占比补齐（never drop）
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ---------------------------
# 常量
# ---------------------------
FEAT_DELTA_PREFIX = "delta_pre_to_knee_"
MOTIF_CANON = ["selfloop_rich","star_core","near_bip_chord","tri","c4","c5"]
SPECTRUM_COLUMNS = ["lambda1","lambda2","lap_algebraic","gap"]  # 仅用于联合表展示顺序

# ---------------------------
# 工具
# ---------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_csv_maybe(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def _normalize_motif_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把多口径字段统一到内部命名；缺列补 0（不会在聚合时消失）"""
    if df is None:
        return df
    mapping = {
        "arch_has_tri":"tri",
        "arch_has_quad":"c4",
        "arch_has_pent":"c5",
        "arch_star_core":"star_core",
        "arch_near_bipartite_chord":"near_bip_chord",
        "arch_selfloop_rich":"selfloop_rich",
        # 自映射（部分表本身已是规范名）
        "tri":"tri","c4":"c4","c5":"c5",
        "star_core":"star_core","near_bip_chord":"near_bip_chord","selfloop_rich":"selfloop_rich",
    }
    for src, dst in mapping.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    for key in MOTIF_CANON:
        if key not in df.columns:
            df[key] = 0
        df[key] = df[key].fillna(0).astype(int)
    return df

def _normalize_summary_table(df_sum: pd.DataFrame, df_ex: pd.DataFrame | None) -> Tuple[pd.DataFrame, Dict]:
    """
    1) 把 summary['feature'] 从 arch_* 映射成规范名
    2) 若缺某些结构行，用 examples 现算 pre/knee/post 占比补齐
    """
    info = {"mapped": {}, "filled_from_examples": []}
    if df_sum is None:
        return None, info

    # 1) 规范化 feature 名
    repl = {
        "arch_has_tri":"tri",
        "arch_has_quad":"c4",
        "arch_has_pent":"c5",
        "arch_star_core":"star_core",
        "arch_near_bipartite_chord":"near_bip_chord",
        "arch_selfloop_rich":"selfloop_rich",
    }
    if "feature" in df_sum.columns:
        df_sum["feature"] = df_sum["feature"].replace(repl)
        info["mapped"] = repl

    # 2) 用 examples 兜底补齐缺行
    present = set(df_sum["feature"].unique()) if "feature" in df_sum.columns else set()
    need = [f for f in MOTIF_CANON if f not in present]
    if need and df_ex is not None:
        # examples 表里是 knee_* / pre_* / post_* 的 0/1 标注
        add_rows = []
        for f in need:
            pre_col  = f"pre_{f}"
            knee_col = f"knee_{f}"
            post_col = f"post_{f}"
            if all(c in df_ex.columns for c in [pre_col,knee_col,post_col]):
                r = {
                    "feature": f,
                    "pre_total":  len(df_ex),
                    "knee_total": len(df_ex),
                    "post_total": len(df_ex),
                    "pre_count":  int((df_ex[pre_col]  > 0).sum()),
                    "knee_count": int((df_ex[knee_col] > 0).sum()),
                    "post_count": int((df_ex[post_col] > 0).sum()),
                }
                r["pre_ratio"]  = r["pre_count"]/max(1,r["pre_total"])
                r["knee_ratio"] = r["knee_count"]/max(1,r["knee_total"])
                r["post_ratio"] = r["post_count"]/max(1,r["post_total"])
                add_rows.append(r)
                info["filled_from_examples"].append(f)
        if add_rows:
            df_sum = pd.concat([df_sum, pd.DataFrame(add_rows)], ignore_index=True)

    # 最终只保留我们需要的列（缺的以 0/NaN 处理后填 0）
    keep = ["feature","pre_ratio","knee_ratio","post_ratio","pre_count","knee_count","post_count",
            "pre_total","knee_total","post_total","OR_knee_vs_pre","OR_knee_vs_post"]
    for c in keep:
        if c not in df_sum.columns:
            df_sum[c] = np.nan
    return df_sum, info

def _auto_target_from_examples(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """优先用显式标签；否则用 δ(gap)>0 或 δ(lambda1)>0 的逻辑或"""
    for col in ["knee_label","target","label","is_knee_positive"]:
        if col in df.columns:
            y = df[col].astype(int).clip(0,1)
            return y, col
    cand = []
    for c in [f"{FEAT_DELTA_PREFIX}gap", f"{FEAT_DELTA_PREFIX}lambda1"]:
        if c in df.columns:
            cand.append((df[c] > 0).astype(int))
    if cand:
        y = pd.concat(cand, axis=1).max(axis=1)
        return y, "auto_gap_or_lambda1_pos"
    return pd.Series(np.ones(len(df), dtype=int), index=df.index), "all_pos_fallback"

def _select_delta_features(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith(FEAT_DELTA_PREFIX)]
    good = []
    for c in cols:
        v = df[c].values
        if np.nanmax(v) != np.nanmin(v):
            good.append(c)
    return good

def _strip_prefix(cols: List[str]) -> List[str]:
    return [c.replace(FEAT_DELTA_PREFIX, "") for c in cols]

# ---------------------------
# 可视化
# ---------------------------
def _bar_pre_knee_post(df: pd.DataFrame, out_png: Path):
    desired = MOTIF_CANON
    mat = df.set_index("feature")[["pre_ratio","knee_ratio","post_ratio"]].copy()
    present = [f for f in desired if f in mat.index]
    missing = [f for f in desired if f not in mat.index]
    if not present:
        raise RuntimeError("summary 中没有任何可用的结构特征（pre/knee/post 比例均缺失）。")

    plot_df = mat.loc[present].fillna(0.0)
    labels = plot_df.index.tolist()
    X = np.arange(len(labels)); width = 0.25

    fig, ax = plt.subplots(figsize=(9.6,6.2))
    ax.bar(X - width, plot_df["pre_ratio"].values,  width, label="pre")
    ax.bar(X,         plot_df["knee_ratio"].values, width, label="knee")
    ax.bar(X + width, plot_df["post_ratio"].values, width, label="post")
    ax.set_xticks(X, labels, rotation=20)
    ax.set_ylabel("ratio")
    title = "Knee-driving motif prevalence (pre / knee / post)"
    if missing:
        title += f"  — missing: {', '.join(missing)}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def _heatmap_pre_knee_post(df: pd.DataFrame, out_png: Path):
    desired = MOTIF_CANON
    mat = df.set_index("feature")[["pre_ratio","knee_ratio","post_ratio"]].copy()
    present = [f for f in desired if f in mat.index]
    missing = [f for f in desired if f not in mat.index]
    if not present:
        raise RuntimeError("summary 中没有可绘制到热力图的结构特征。")

    mat = mat.loc[present].fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.8,5.3))
    im = ax.imshow(mat.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(3), ["pre","knee","post"])
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
    title = "Motif prevalence heatmap"
    if missing:
        title += f"  — missing: {', '.join(missing)}"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

# ---------------------------
# 学习器
# ---------------------------
def _fit_logreg_importance(X, y, names, seed, out_csv, out_png):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed))])
    pipe.fit(X, y)
    coefs = np.abs(pipe.named_steps["clf"].coef_.reshape(-1))
    df = pd.DataFrame({"feature": names, "abs_coef": coefs}).sort_values("abs_coef", ascending=False)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.2,5.2))
    ax.barh(df["feature"], df["abs_coef"])
    ax.set_xlabel("|coef|"); ax.set_title("Logistic Regression feature influence (|coef|)")
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

def _fit_tree_importance(X, y, names, depth, min_leaf, seed, out_csv, out_png):
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed)
    tree.fit(X, y)
    imp = tree.feature_importances_
    df = pd.DataFrame({"feature": names, "gini_importance": imp}).sort_values("gini_importance", ascending=False)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.2,5.2))
    ax.barh(df["feature"], df["gini_importance"])
    ax.set_xlabel("Gini importance"); ax.set_title("Decision Tree importance")
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

def _cv_l1_logreg_coeffs(X, y, names, seed, out_csv, out_png):
    # 把 n_splits 限制在 [3, min(5, 正负样本最小计数)]
    pos = int(y.sum()); neg = int(len(y) - pos)
    n_splits = max(3, min(5, pos, neg)) if min(pos,neg) >= 2 else 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    for (tr, te) in skf.split(X, y):
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000, random_state=seed)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
        clf.fit(Xtr, y[tr]); rows.append(np.abs(clf.coef_.reshape(-1)))
    M = np.vstack(rows); mean, std = M.mean(axis=0), M.std(axis=0)

    df = pd.DataFrame({"feature": names, "abs_coef_mean": mean, "abs_coef_std": std}).sort_values("abs_coef_mean", ascending=False)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.5,5.4))
    ax.barh(df["feature"], df["abs_coef_mean"], xerr=df["abs_coef_std"], alpha=0.85, capsize=3)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("L1-LogReg coefficient (mean ± std across folds)")
    ax.set_title("Knee vs Non-knee Δ: logistic coefficients")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

def _perm_importance_logreg(X, y, names, seed, out_csv, out_png):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed))])
    pipe.fit(X, y)
    r = permutation_importance(pipe, X, y, n_repeats=50, random_state=seed, scoring="roc_auc")
    df = pd.DataFrame({"feature": names, "perm_mean": r.importances_mean, "perm_std": r.importances_std})\
           .sort_values("perm_mean", ascending=False)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.5,5.0))
    ax.barh(df["feature"], df["perm_mean"], xerr=df["perm_std"], alpha=0.9, capsize=3)
    ax.set_xlabel("Permutation importance (mean ± std, roc_auc)")
    ax.set_title("Permutation importance of Δ-features (logistic)")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

# ---------------------------
# 谱-结构联合表
# ---------------------------
def _make_joint_table(ex: pd.DataFrame, out_csv: Path, topN: int):
    keep_cols = [c for c in ex.columns if c.startswith("knee_")] + ["n","k"]
    knee = ex[keep_cols].copy()
    knee.columns = [c.replace("knee_", "") for c in knee.columns]
    order = ["n","k","rule_count","y","lambda1","lambda2","gap","lap_algebraic",
             "deg_max","deg_mean","deg_std","diag_cnt","kcore","clustering",
             "tri","c4","c5","near_bip_chord","selfloop_rich","star_core","odd_girth","key_edges","bits"]
    cols = [c for c in order if c in knee.columns]
    if "y" in knee.columns:
        knee = knee.sort_values("y", ascending=False).head(topN)
    knee[cols].to_csv(out_csv, index=False)

# ---------------------------
# 主流程
# ---------------------------
def run(examples_csv: Path, out_csv_dir: Path, out_fig_dir: Path,
        style: str = "ieee", topN: int = 20, tree_depth: int = 3, tree_min_leaf: int = 8, seed: int = 0,
        include_growth: bool = False):
    _ensure_dir(out_csv_dir); _ensure_dir(out_fig_dir)

    df_ex = pd.read_csv(examples_csv)
    # 规范化 examples（保证 pre_/knee_/post_ 的结构列存在）
    df_ex = _normalize_motif_columns(df_ex)

    df_sum = _read_csv_maybe(out_csv_dir / "motif_knee_summary.csv")
    if df_sum is None:
        df_sum = _read_csv_maybe((out_csv_dir.parent / "motifs") / "motif_knee_summary.csv")
    if df_sum is not None:
        df_sum, sum_info = _normalize_summary_table(df_sum.copy(), df_ex)
    else:
        sum_info = {"mapped":{}, "filled_from_examples":[]}

    # -------- Slide 20：条形图与热力图（永不丢行；缺列会在标题注明 missing）--------
    if df_sum is not None:
        _bar_pre_knee_post(df_sum, out_fig_dir / "knee_motif_bar.png")
        _heatmap_pre_knee_post(df_sum, out_fig_dir / "knee_motif_heatmap.png")

    # -------- Δ 特征与标签 --------
    y, y_name = _auto_target_from_examples(df_ex)
    feat_cols = _select_delta_features(df_ex)
    X = df_ex[feat_cols].fillna(0.0).values
    names = _strip_prefix(feat_cols)

    # 完整性报告
    zero_vars = [c for c in feat_cols if df_ex[c].nunique(dropna=False) <= 1]
    integ = {
        "N_total": int(len(y)),
        "pos(1)": int(y.sum()),
        "neg(0)": int((1 - y).sum()),
        "label_used": y_name,
        "delta_features": names,
        "zero_variance_features": _strip_prefix(zero_vars),
        "summary_missing_features": [f for f in MOTIF_CANON
                                     if df_sum is None or f not in set(df_sum["feature"].unique())],
        "summary_filled_from_examples": sum_info.get("filled_from_examples", []),
        "summary_mapped": sum_info.get("mapped", {}),
    }
    (out_csv_dir / "integrity_report.json").write_text(json.dumps(integ, ensure_ascii=False, indent=2), encoding="utf-8")

    # 备份 Δ 数据（便于手工分析）
    delta_keep = [c for c in df_ex.columns if c.startswith(FEAT_DELTA_PREFIX)] + ["n","k","knee_y","pre_y","post_y"]
    try:
        df_ex[delta_keep].to_csv(out_csv_dir / "motif_delta_dataset.csv", index=False)
    except Exception:
        pass

    # -------- 含 gap 的解释 --------
    _fit_logreg_importance(X, y, names, seed,
                           out_csv_dir / "motif_feature_importance_logreg.csv",
                           out_fig_dir / "motif_importance_logreg.png")
    _fit_tree_importance(X, y, names, tree_depth, tree_min_leaf, seed,
                         out_csv_dir / "motif_feature_importance_tree.csv",
                         out_fig_dir / "motif_importance_tree.png")
    _cv_l1_logreg_coeffs(X, y, names, seed,
                         out_csv_dir / "motif_lr_coeffs.csv",
                         out_fig_dir / "motif_lr_coeffs.png")
    _perm_importance_logreg(X, y, names, seed,
                            out_csv_dir / "motif_perm_importance.csv",
                            out_fig_dir / "motif_perm_importance.png")

    # -------- 去 gap 的鲁棒性版本 --------
    if "gap" in names:
        idx = [i for i, n in enumerate(names) if n != "gap"]
        if idx:
            X2 = X[:, idx]; names2 = [names[i] for i in idx]
            _fit_logreg_importance(X2, y, names2, seed,
                                   out_csv_dir / "motif_feature_importance_logreg_nogap.csv",
                                   out_fig_dir / "motif_importance_logreg_nogap.png")
            _fit_tree_importance(X2, y, names2, tree_depth, tree_min_leaf, seed,
                                 out_csv_dir / "motif_feature_importance_tree_nogap.csv",
                                 out_fig_dir / "motif_importance_tree_nogap.png")
            _cv_l1_logreg_coeffs(X2, y, names2, seed,
                                 out_csv_dir / "motif_lr_coeffs_nogap.csv",
                                 out_fig_dir / "motif_lr_coeffs_nogap.png")
            _perm_importance_logreg(X2, y, names2, seed,
                                    out_csv_dir / "motif_perm_importance_nogap.csv",
                                    out_fig_dir / "motif_perm_importance_nogap.png")

    # -------- 谱-结构联合表（仅 knee 行）--------
    _make_joint_table(df_ex, out_csv_dir / "spectrum_structure_joint.csv", topN=topN)

    return {
        "dataset_csv":        str(out_csv_dir / "motif_delta_dataset.csv"),
        "logreg_csv":         str(out_csv_dir / "motif_feature_importance_logreg.csv"),
        "tree_csv":           str(out_csv_dir / "motif_feature_importance_tree.csv"),
        "perm_csv":           str(out_csv_dir / "motif_perm_importance.csv"),
        "l1_coeffs":          str(out_csv_dir / "motif_lr_coeffs.csv"),
        "integrity_report":   str(out_csv_dir / "integrity_report.json"),
        "joint_table":        str(out_csv_dir / "spectrum_structure_joint.csv"),
        "png_bar":            str(out_fig_dir / "knee_motif_bar.png"),
        "png_heatmap":        str(out_fig_dir / "knee_motif_heatmap.png"),
        "png_logreg":         str(out_fig_dir / "motif_importance_logreg.png"),
        "png_tree":           str(out_fig_dir / "motif_importance_tree.png"),
        "png_l1":             str(out_fig_dir / "motif_lr_coeffs.png"),
        "png_perm":           str(out_fig_dir / "motif_perm_importance.png"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True, type=Path)
    ap.add_argument("--out-csv",  required=True, type=Path)
    ap.add_argument("--out-dir",  required=True, type=Path)
    ap.add_argument("--style", default="ieee")
    ap.add_argument("--topN", type=int, default=20)
    ap.add_argument("--tree-depth", type=int, default=3)
    ap.add_argument("--tree-min-leaf", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-growth", action="store_true",
                    help="保持为 False：nk 增长曲线已由 viz 生成，避免重复。")
    args = ap.parse_args()

    _ensure_dir(args.out_csv); _ensure_dir(args.out_dir)

    out = run(args.examples, args.out_csv, args.out_dir,
              style=args.style, topN=args.topN, tree_depth=args.tree_depth,
              tree_min_leaf=args.tree_min_leaf, seed=args.seed,
              include_growth=args.include_growth)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
