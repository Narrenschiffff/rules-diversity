# -*- coding: utf-8 -*-
"""
motifs_explainer.py — 结构Δ→谱Δ→增长 的解释与可视化（稳健均值 + 统一口径）
支持 KNEE / MUR / OPTIMAL 分析。

集成优化版 (v2.2):
1. 修复 NameError: '_layered_bar_heat_generic' not defined.
2. 恢复并优化缺失的 '_boxplot_deltas_generic' 和 '_scatter_gap_vs_structure_generic' 可视化。
3. 保持 v2.1 的所有修复 (joint table, seaborn warnings)。
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ================= 基本常量/开关 =================
DELTA_PREFIXES = (
    "delta_pre_to_knee_", "delta_knee_to_post_",
    "delta_pre_to_mur_",  "delta_mur_to_post_",
    "delta_pre_to_optimal_", "delta_optimal_to_post_"
)

COUNT_FEATURES  = {"tri", "c4", "c5"}
INDIC_FEATURES  = {"selfloop_rich", "star_core", "near_bip_chord"}
AUTO_BOOL_SUFFIX = {} 

TRIM_PCT = float(os.environ.get("RD_TRIM_PCT", "0.01"))
OBJ_LABEL = r"Objective (log $Z$) / penalty"
RC_LABEL  = r"|R|"

# ================= 基础工具 =================
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _read_csv_maybe(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None

def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return set(cols).issubset(df.columns)

def _winsorized_mean(x: pd.Series, trim_pct: float = TRIM_PCT) -> float:
    v = pd.to_numeric(x, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty: return np.nan
    p = max(0.0, min(0.2, float(trim_pct)))
    if p > 0.0:
        lo, hi = np.nanpercentile(v, [100*p, 100*(1-p)])
        v = v.clip(lo, hi)
    return float(v.mean())

# ================= 凹凸性打标 =================
def _label_by_concavity(df: pd.DataFrame,
                        pre_col="pre_y", knee_col="knee_y", post_col="post_y",
                        x_pre="pre_rule_count", x_knee="knee_rule_count", x_post="post_rule_count",
                        eps: float = 1e-12, q_d2: float = 0.1, slope_drop_ratio: float = 0.01):
    if not _has_cols(df, [pre_col, knee_col, post_col]):
        na = pd.Series([np.nan]*len(df), index=df.index)
        return na, "concavity_na", na, na, na, na

    Yp = np.log(pd.to_numeric(df[pre_col], errors="coerce").clip(lower=eps))
    Yk = np.log(pd.to_numeric(df[knee_col], errors="coerce").clip(lower=eps))
    Yo = np.log(pd.to_numeric(df[post_col], errors="coerce").clip(lower=eps))

    if _has_cols(df, [x_pre, x_knee, x_post]):
        Xp = pd.to_numeric(df[x_pre], errors="coerce")
        Xk = pd.to_numeric(df[x_knee], errors="coerce")
        Xo = pd.to_numeric(df[x_post], errors="coerce")
        dL = (Xk - Xp).replace(0, 1.0).fillna(1.0).abs()
        dR = (Xo - Xk).replace(0, 1.0).fillna(1.0).abs()
    else:
        dL = pd.Series(1.0, index=df.index)
        dR = pd.Series(1.0, index=df.index)

    slope_L = (Yk - Yp) / dL
    slope_R = (Yo - Yk) / dR
    d2 = slope_R - slope_L
    ratio = (slope_R / slope_L.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    valid = d2.replace([np.inf, -np.inf], np.nan).dropna()
    tau = max(np.nanquantile(np.abs(valid), q_d2) if len(valid) else 0.0, 1e-6)
    y = ((d2 <= -tau) & (ratio <= (1.0 - slope_drop_ratio))).astype(int)
    y = y.fillna(0).astype(int)
    return y, "concavity(d2,ratio)", slope_L, slope_R, d2, ratio

def _forced_target_concavity(df: pd.DataFrame, prefix: str):
    pre_col, mid_col, post_col = "pre_y", f"{prefix}_y", "post_y"
    x_pre, x_mid, x_post = "pre_rule_count", f"{prefix}_rule_count", "post_rule_count"
    if not _has_cols(df, [pre_col, mid_col, post_col]):
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index), "concavity_na_no_cols"
    y, name, *_ = _label_by_concavity(df, pre_col, mid_col, post_col, x_pre, x_mid, x_post)
    if y.notna().any():
        return y, name
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index), "concavity_na_all_zero"

def _augment_concavity_columns_generic(df_ex: pd.DataFrame, prefix: str) -> pd.DataFrame:
    pre_col, mid_col, post_col = "pre_y", f"{prefix}_y", "post_y"
    x_pre, x_mid, x_post = "pre_rule_count", f"{prefix}_rule_count", "post_rule_count"
    y, _, slope_L, slope_R, d2, ratio = _label_by_concavity(df_ex, pre_col, mid_col, post_col, x_pre, x_mid, x_post)
    df = df_ex.copy()
    df[f"{prefix}_slope_L"] = slope_L
    df[f"{prefix}_slope_R"] = slope_R
    df[f"{prefix}_d2"] = d2
    df[f"{prefix}_slope_ratio"] = ratio
    sign = np.sign(d2.replace([np.inf, -np.inf], np.nan)).fillna(0.0)
    df[f"{prefix}_concavity_class"] = sign.apply(lambda v: "up" if v < 0 else ("down" if v > 0 else "flat"))
    return df

# ================= Δ 特征收集 =================
def _select_delta_features(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if any(c.startswith(p) for p in DELTA_PREFIXES)]
    good = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").values
        if np.nanmax(v) != np.nanmin(v):
            good.append(c)
    return good

def _strip_prefix(c: str) -> str:
    for p in DELTA_PREFIXES:
        if c.startswith(p):
            return c.replace(p, "")
    return c

def _strip_prefix_list(cols: List[str]) -> List[str]:
    return [_strip_prefix(c) for c in cols]

# ================= 输出命名 & 目录 =================
def _subdir_name(prefix: str) -> str: return prefix.upper()
def _csv_path(root: Path, prefix: str, filename: str) -> Path: return _ensure_dir(root / _subdir_name(prefix)) / filename
def _fig_path(root: Path, prefix: str, filename: str) -> Path: return _ensure_dir(root / _subdir_name(prefix)) / filename
def _title_head(prefix: str) -> str: return f"[{prefix.upper()}]"
def _center_label(prefix: str) -> str: return prefix.upper()

# ================= 占比统计 =================
def _discover_auto_bool_features(df: pd.DataFrame, prefix: str) -> set:
    found = set()
    for suf in AUTO_BOOL_SUFFIX:
        for stage in ("pre", prefix, "post"):
            if f"{stage}_{suf}" in df.columns: found.add(suf)
    return found

def _presence_one_stage(df: pd.DataFrame, stage: str, bool_feats: set) -> pd.DataFrame:
    rows = []
    all_feats = sorted((COUNT_FEATURES | INDIC_FEATURES | bool_feats))
    for feat in all_feats:
        col = f"{stage}_{feat}"
        if col not in df.columns: continue
        v = pd.to_numeric(df[col], errors="coerce")
        pres = (v > 0).astype(float) if feat in COUNT_FEATURES else (v >= 1).astype(float)
        if pres.sum() == 0: continue
        rows.append((feat, float(np.nanmean(pres))))
    return pd.DataFrame(rows, columns=["feature", f"{stage}_ratio"])

def _presence_from_examples(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    bool_feats = _discover_auto_bool_features(df, prefix)
    pre  = _presence_one_stage(df, "pre",  bool_feats)
    mid  = _presence_one_stage(df, prefix, bool_feats)
    post = _presence_one_stage(df, "post", bool_feats)
    mid_ratio_col = f"{prefix}_ratio"
    out = pd.DataFrame({"feature": list(set(pre.feature) | set(mid.feature) | set(post.feature))})
    out = out.merge(pre,  on="feature", how="left").merge(mid,  on="feature", how="left").merge(post, on="feature", how="left")
    mask = out[["pre_ratio", mid_ratio_col, "post_ratio"]].notna().any(axis=1)
    out = out.loc[mask].fillna(0.0)
    order = ["selfloop_rich","star_core","near_bip_chord","tri","c4","c5"]
    out["__ord__"] = out["feature"].apply(lambda x: order.index(x) if x in order else 10_000)
    out = out.sort_values(["__ord__","feature"]).drop(columns="__ord__").reset_index(drop=True)
    return out

def _bar_pre_center_post(prefix: str, df_summary: pd.DataFrame, out_png: Path):
    mid_col = f"{prefix}_ratio"
    if mid_col not in df_summary.columns: return
    labels = df_summary["feature"].tolist()
    X = np.arange(len(labels)); width = 0.25
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.bar(X - width, df_summary["pre_ratio"],  width, label="pre")
    ax.bar(X,         df_summary[mid_col], width, label=_center_label(prefix))
    ax.bar(X + width, df_summary["post_ratio"], width, label="post")
    ax.set_xticks(X, labels, rotation=20)
    ax.set_ylabel("Presence Ratio")
    ax.set_title(f"{_title_head(prefix)} Motif Prevalence")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def _heatmap_pre_center_post(prefix: str, df_summary: pd.DataFrame, out_png: Path):
    mid_col = f"{prefix}_ratio"
    if mid_col not in df_summary.columns: return
    Z = df_summary.set_index("feature")[["pre_ratio", mid_col, "post_ratio"]].values
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    sns.heatmap(Z, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0, 
                xticklabels=["pre", _center_label(prefix), "post"], 
                yticklabels=df_summary["feature"], ax=ax)
    ax.set_title(f"{_title_head(prefix)} Motif Prevalence (Heatmap)")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

# ================= 优化：谱-拓扑连通性演化 =================
def _spectral_connectivity_plot(df_ex: pd.DataFrame, prefix: str, out_png: Path):
    stages = ["pre", prefix, "post"]
    metrics = ["lap_algebraic", "lcc_ratio"]
    data = {m: [] for m in metrics}
    
    valid_cnt = 0
    for idx, row in df_ex.iterrows():
        temp_row = {m: [] for m in metrics}
        for s in stages:
            for m in metrics:
                col = f"{s}_{m}"
                val = pd.to_numeric(row.get(col), errors="coerce")
                if pd.isna(val):
                    val = 0.0 # 若缺失默认为0
                temp_row[m].append(val)
        
        if pd.notna(row.get(f"{prefix}_lap_algebraic")):
             for m in metrics:
                 data[m].append(temp_row[m])
             valid_cnt += 1

    if valid_cnt < 3: return

    means = {m: np.mean(data[m], axis=0) for m in metrics}
    stds = {m: np.std(data[m], axis=0) / np.sqrt(valid_cnt) for m in metrics}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(3)
    
    # Plot Algebraic Connectivity (Left Axis)
    ax.errorbar(x, means["lap_algebraic"], yerr=stds["lap_algebraic"], fmt='-o', 
                color='tab:blue', label="Algebraic Connectivity ($\\lambda_2(L)$)", capsize=5)
    ax.set_ylabel("Alg. Connectivity ($\\lambda_2$)")
    ax.set_ylim(bottom=0)
    
    # Plot LCC Ratio (Right Axis)
    ax2 = ax.twinx()
    ax2.errorbar(x, means["lcc_ratio"], yerr=stds["lcc_ratio"], fmt='--s', 
                 color='tab:orange', label="LCC Size Ratio", capsize=5)
    ax2.set_ylabel("LCC Ratio (Size/N)")
    ax2.set_ylim(0, 1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(["Pre", _center_label(prefix), "Post"])
    ax.set_title(f"{_title_head(prefix)} Topological Connectivity Evolution")
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=False)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# ================= 优化：结构-功能相关性热图 =================
def _correlation_heatmap_delta(df_ex: pd.DataFrame, prefix: str, out_png: Path):
    struct_feats = ["tri", "star_core", "selfloop_rich", "near_bip_chord", "c4", "c5"]
    func_feats = ["lambda1", "gap", "y", "lap_algebraic"]
    
    data = {}
    p1 = f"delta_pre_to_{prefix}_"
    for sf in struct_feats:
        col = p1 + sf
        if col in df_ex.columns:
            data[f"Δ{sf}"] = pd.to_numeric(df_ex[col], errors="coerce")
            
    for ff in func_feats:
        col = p1 + ff
        if col in df_ex.columns:
            data[f"Δ{ff}"] = pd.to_numeric(df_ex[col], errors="coerce")
        elif _has_cols(df_ex, [f"pre_{ff}", f"{prefix}_{ff}"]):
             data[f"Δ{ff}"] = pd.to_numeric(df_ex[f"{prefix}_{ff}"], errors="coerce") - pd.to_numeric(df_ex[f"pre_{ff}"], errors="coerce")

    df_corr = pd.DataFrame(data).dropna()
    if df_corr.shape[1] < 2 or df_corr.shape[0] < 5: return

    corr_mat = df_corr.corr(method="spearman")
    s_cols = [c for c in corr_mat.columns if c.startswith("Δ") and c[1:] in struct_feats]
    f_cols = [c for c in corr_mat.columns if c.startswith("Δ") and c[1:] in func_feats]
    
    if not s_cols or not f_cols: sub_mat = corr_mat
    else: sub_mat = corr_mat.loc[s_cols, f_cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sub_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"{_title_head(prefix)} Structure-Function $\Delta$ Correlation (Pre->{_center_label(prefix)})")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

# ================= 优化：效率轨迹图 =================
def _efficiency_trajectory_plot(df_ex: pd.DataFrame, prefix: str, out_png: Path):
    stages = ["pre", prefix, "post"]
    obj_data = {s: [] for s in stages}
    
    valid_rows = 0
    for idx, row in df_ex.iterrows():
        row_vals = []
        for s in stages:
            y = pd.to_numeric(row.get(f"{s}_y"), errors="coerce")
            row_vals.append(y)
        
        if pd.notna(row_vals[1]) and row_vals[1] != 0:
            norm_vals = [v / row_vals[1] for v in row_vals]
            for i, s in enumerate(stages):
                obj_data[s].append(norm_vals[i])
            valid_rows += 1
            
    if valid_rows < 5: return

    means = [np.nanmean(obj_data[s]) for s in stages]
    stds = [np.nanstd(obj_data[s]) for s in stages]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(3)
    ax.errorbar(x, means, yerr=stds, fmt='-o', capsize=5, color='tab:green', lw=2, label="Relative Objective")
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Pre", _center_label(prefix), "Post"])
    ax.set_ylabel(f"Relative {OBJ_LABEL}") 
    ax.set_title(f"{_title_head(prefix)} Objective Function Trajectory")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

# ================= ML Feature Importance =================
def _fit_logreg_importance(prefix, X, y, names, seed, out_csv, out_png):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed))])
    pipe.fit(X, y)
    coefs = np.abs(pipe.named_steps["clf"].coef_.reshape(-1))
    df = pd.DataFrame({"feature": names, "abs_coef": coefs}).sort_values("abs_coef", ascending=False)
    df.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    sns.barplot(x="abs_coef", y="feature", hue="feature", data=df.head(15), ax=ax, palette="viridis", legend=False)
    ax.set_xlabel("|coef|")
    ax.set_title(f"{_title_head(prefix)} Logistic Regression Feature Influence (Top 15)")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

def _fit_tree_importance(prefix, X, y, names, depth, min_leaf, seed, out_csv, out_png):
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed)
    tree.fit(X, y)
    imp = tree.feature_importances_
    df = pd.DataFrame({"feature": names, "gini_importance": imp}).sort_values("gini_importance", ascending=False)
    df.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    sns.barplot(x="gini_importance", y="feature", hue="feature", data=df.head(15), ax=ax, palette="magma", legend=False)
    ax.set_xlabel("Gini importance")
    ax.set_title(f"{_title_head(prefix)} Decision Tree Importance (Top 15)")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

# ================= 通用事件曲线 =================
def _event_series_dual_axis(df_ex: pd.DataFrame, prefix: str, out_fig_dir: Path, cls: str, out_csv_dir: Path):
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns: return
    sub = df_ex[df_ex[cls_col] == cls]
    mid_y, mid_lam, mid_gap = f"{prefix}_y", f"{prefix}_lambda1", f"{prefix}_gap"
    need = ["pre_y", mid_y, "post_y", "pre_lambda1", mid_lam, "post_lambda1", "pre_gap", mid_gap, "post_gap"]
    if sub.empty or not _has_cols(sub, need): return

    def _rel3(a, b, c): # 归一化辅助
        with np.errstate(divide="ignore", invalid="ignore"): return a/a, b/a, c/a

    Y1, Y2, Y3 = _rel3(pd.to_numeric(sub["pre_y"], errors="coerce"), pd.to_numeric(sub[mid_y], errors="coerce"), pd.to_numeric(sub["post_y"], errors="coerce"))
    L1, L2, L3 = _rel3(pd.to_numeric(sub["pre_lambda1"], errors="coerce"), pd.to_numeric(sub[mid_lam], errors="coerce"), pd.to_numeric(sub["post_lambda1"], errors="coerce"))
    G1, G2, G3 = _rel3(pd.to_numeric(sub["pre_gap"], errors="coerce"), pd.to_numeric(sub[mid_gap], errors="coerce"), pd.to_numeric(sub["post_gap"], errors="coerce"))

    def rmean(s): return _winsorized_mean(pd.Series(s))
    stats = pd.DataFrame({
        "metric": ["lambda1_rel_pre", "gap_rel_pre", "Y_rel_pre"],
        "pre":  [rmean(L1), rmean(G1), rmean(Y1)],
        prefix: [rmean(L2), rmean(G2), rmean(Y2)],
        "post": [rmean(L3), rmean(G3), rmean(Y3)],
        "N":    [len(sub)]*3
    })
    stats.to_csv(out_csv_dir / f"event_series_stats_concavity_{cls}.csv", index=False)

    X = np.array([0, 1, 2]); labs = ["pre", _center_label(prefix), "post"]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(X, stats.loc[0, ["pre", prefix, "post"]], marker="o", color="tab:blue", label="lambda1 (rel)")
    ax1.plot(X, stats.loc[1, ["pre", prefix, "post"]], marker="^", color="tab:orange", label="gap (rel)")
    ax1.set_ylabel("Spectral Metrics (Relative)")
    ax1.set_xticks(X, labs)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(X, stats.loc[2, ["pre", prefix, "post"]], marker="s", linestyle="--", color="gray", label=OBJ_LABEL + " (rel)")
    ax2.set_ylabel(f"{OBJ_LABEL} (Relative)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, frameon=False, loc="upper left")
    title_tag = "Up-concave (Accelerating)" if cls=="up" else ("Down-convex (Saturating)" if cls=="down" else "Flat")
    ax1.set_title(f"[{_center_label(prefix)}: {title_tag}] Event Study")
    fig.tight_layout(); fig.savefig(out_fig_dir / f"event_series_concavity_{cls}.png", dpi=160); plt.close(fig)

# ================= 缺失函数补全 =================
def _layered_bar_heat_generic(prefix: str, df_ex: pd.DataFrame, out_fig_dir: Path, cls: str):
    """分层（按凹凸性）绘制 Motifs 占比的柱状图和热图"""
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns: return
    sub = df_ex[df_ex[cls_col] == cls]
    if sub.empty: return
    
    M = _presence_from_examples(sub, prefix)
    if M.empty: return
    
    _bar_pre_center_post(prefix, M, out_fig_dir / f"{prefix}_motif_bar_concavity_{cls}.png")
    _heatmap_pre_center_post(prefix, M, out_fig_dir / f"{prefix}_motif_heatmap_concavity_{cls}.png")

def _boxplot_deltas_generic(df_ex: pd.DataFrame, prefix: str, out_fig_dir: Path, cls: str):
    """分层绘制 Δ 特征的箱线图"""
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns: return
    sub = df_ex[df_ex[cls_col] == cls]
    if sub.empty: return

    def dcol(part, feat):
        if part == "pre_to_mid": return f"delta_pre_to_{prefix}_{feat}"
        if part == "mid_to_post": return f"delta_{prefix}_to_post_{feat}"
        return ""

    # Spectrum Boxplot
    spec_vars = [
        (dcol("pre_to_mid","lambda1"), f"Δλ1(pre→{prefix})"),
        (dcol("mid_to_post","lambda1"), f"Δλ1({prefix}→post)"),
        (dcol("pre_to_mid","gap"), f"Δgap(pre→{prefix})"),
        (dcol("mid_to_post","gap"), f"Δgap({prefix}→post)"),
        (dcol("pre_to_mid","lap_algebraic"), f"ΔλL(pre→{prefix})"),
    ]
    data, labels = [], []
    for c, l in spec_vars:
        if c in sub.columns:
            data.append(pd.to_numeric(sub[c], errors="coerce").values)
            labels.append(l)
    if data:
        fig, ax = plt.subplots(figsize=(9.6, 5.6))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"[{_center_label(prefix)}: {'Up' if cls=='up' else 'Down'}] spectral deltas")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_fig_dir / f"{prefix}_delta_box_spectrum_concavity_{cls}.png", dpi=160)
        plt.close(fig)

    # Motif Boxplot
    motif_vars = []
    for feat, nick in [("tri","tri"), ("star_core","star"), ("selfloop_rich","selfloop"), ("near_bip_chord","near-bip")]:
        motif_vars.append((dcol("pre_to_mid", feat), f"Δ{nick}(pre→{prefix})"))
        motif_vars.append((dcol("mid_to_post", feat), f"Δ{nick}({prefix}→post)"))

    data, labels = [], []
    for c, l in motif_vars:
        if c in sub.columns:
            data.append(pd.to_numeric(sub[c], errors="coerce").values)
            labels.append(l)
    if data:
        fig, ax = plt.subplots(figsize=(10.0, 5.8))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"[{_center_label(prefix)}: {'Up' if cls=='up' else 'Down'}] motif deltas")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_fig_dir / f"{prefix}_delta_box_motif_concavity_{cls}.png", dpi=160)
        plt.close(fig)

def _scatter_gap_vs_structure_generic(df_ex: pd.DataFrame, prefix: str, out_fig_dir: Path, cls: str):
    """绘制 Gap 变化与结构变化的散点图"""
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns: return
    sub = df_ex[df_ex[cls_col] == cls]
    
    y_col = f"delta_{prefix}_to_post_gap"
    if sub.empty or y_col not in sub.columns: return
    Y = pd.to_numeric(sub[y_col], errors="coerce")
    
    pairs = [
        (f"delta_pre_to_{prefix}_tri", f"Δtri(pre→{prefix})", "dtri"),
        (f"delta_pre_to_{prefix}_star_core", f"Δstar(pre→{prefix})", "dstar"),
        (f"delta_pre_to_{prefix}_selfloop_rich", f"Δselfloop(pre→{prefix})", "dselfloop"),
    ]
    for col, lab, tag in pairs:
        if col not in sub.columns: continue
        X = pd.to_numeric(sub[col], errors="coerce")
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        ax.scatter(X, Y, alpha=0.7)
        try:
            mask = np.isfinite(X.values) & np.isfinite(Y.values)
            if mask.sum() >= 2:
                a, b = np.polyfit(X.values[mask], Y.values[mask], 1)
                xg = np.linspace(np.nanmin(X), np.nanmax(X), 100)
                ax.plot(xg, a*xg + b, lw=1.5)
        except Exception:
            pass
        ax.set_xlabel(lab); ax.set_ylabel(f"Δgap({prefix}→post)")
        ax.set_title(f"[{_center_label(prefix)}: {'Up' if cls=='up' else 'Down'}] Δgap vs {lab}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_fig_dir / f"{prefix}_scatter_dgap_vs_{tag}_concavity_{cls}.png", dpi=160)
        plt.close(fig)

# ================= 联合表生成 =================
def _make_joint_table_generic(ex: pd.DataFrame, prefix: str, out_csv: Path, topN: int):
    """生成谱学-结构联合对照表，用于人工查阅"""
    keep = [c for c in ex.columns if c.startswith(f"{prefix}_")] + ["n", "k"]
    sub = ex[keep].copy()
    if sub.empty: return
    sub.columns = [c.replace(f"{prefix}_", "") for c in sub.columns]
    
    cols = ["n","k","rule_count","y","lambda1","lambda2","gap","lap_algebraic",
            "lcc_ratio", "is_connected",
            "deg_max","deg_mean","deg_std","diag_cnt","kcore","clustering",
            "tri","c4","c5","near_bip_chord","selfloop_rich","star_core",
            "odd_girth","key_edges","bits"]
    cols = [c for c in cols if c in sub.columns]
    
    if "y" in sub.columns:
        sub = sub.sort_values("y", ascending=False).head(topN)
    
    sub[cols].to_csv(out_csv, index=False)

# ================= 自动解读报告 =================
def _write_interpretation_report_generic(out_csv_dir: Path, df_ex: pd.DataFrame, prefix: str):
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns: return
    rep = {"N_total": int(len(df_ex))}
    for cls in ["up", "down"]:
        sub = df_ex[df_ex[cls_col] == cls]
        if sub.empty: continue
        sec = {"N": int(len(sub))}
        for m in ["lambda1", "gap", "y", "lap_algebraic", "lcc_ratio"]:
            pre, mid, po = f"pre_{m}", f"{prefix}_{m}", f"post_{m}"
            if _has_cols(sub, [pre, mid, po]):
                sec[f"winsor_mean_{m}"] = {"pre": _winsorized_mean(sub[pre]), prefix: _winsorized_mean(sub[mid]), "post": _winsorized_mean(sub[po])}
        rep[cls] = sec

    md = [f"# {prefix.upper()} Analysis Report", f"- Total Samples: {rep.get('N_total', 0)}", "", "## Concavity Segments"]
    for cls in ["up", "down"]:
        if cls not in rep: continue
        tag = "Accelerating Growth (Up-concave)" if cls=='up' else "Saturation (Down-convex)"
        md.append(f"### {tag}"); sec = rep[cls]; md.append(f"- Count: {sec['N']}")
        for m in ["lambda1","gap","y", "lap_algebraic", "lcc_ratio"]:
            if f"winsor_mean_{m}" in sec:
                r=sec[f"winsor_mean_{m}"]
                md.append(f"- **{m}**: {r['pre']:.2f} -> {r[prefix]:.2f} -> {r['post']:.2f}")
    (out_csv_dir/f"interpretation_{prefix}.md").write_text("\n".join(md), encoding="utf-8")

# ================= 单前缀执行 =================
def _run_one_prefix(prefix: str, df_ex: pd.DataFrame,
                    out_csv_root: Path, out_fig_root: Path,
                    topN: int, tree_depth: int, tree_min_leaf: int, seed: int,
                    df_sum: Optional[pd.DataFrame]=None, df_global: Optional[pd.DataFrame]=None) -> Dict[str,str]:

    df_ex = _augment_concavity_columns_generic(df_ex, prefix)
    df_ex.to_csv(_csv_path(out_csv_root, prefix, f"motif_{prefix}_examples_labeled.csv"), index=False)

    overall_presence = _presence_from_examples(df_ex, prefix)
    if not overall_presence.empty:
        _bar_pre_center_post(prefix, overall_presence, _fig_path(out_fig_root, prefix, "motif_bar.png"))
        _heatmap_pre_center_post(prefix, overall_presence, _fig_path(out_fig_root, prefix, "motif_heatmap.png"))

    _correlation_heatmap_delta(df_ex, prefix, _fig_path(out_fig_root, prefix, "delta_correlation.png"))
    _efficiency_trajectory_plot(df_ex, prefix, _fig_path(out_fig_root, prefix, "objective_trajectory.png"))
    _spectral_connectivity_plot(df_ex, prefix, _fig_path(out_fig_root, prefix, "connectivity_evolution.png"))

    y, y_name = _forced_target_concavity(df_ex, prefix)
    feat_cols = _select_delta_features(df_ex)
    X = df_ex[feat_cols].fillna(0.0).values if feat_cols else np.zeros((len(df_ex), 0))
    names = _strip_prefix_list(feat_cols)

    integ = {"prefix": prefix, "N_total": int(len(y)), "pos(1)": int(y.sum()), "neg(0)": int((1-y).sum()), "label_used": y_name}
    _csv_path(out_csv_root, prefix, "integrity_report.json").write_text(json.dumps(integ, ensure_ascii=False, indent=2), "utf-8")

    if feat_cols and len(y) > 5 and y.nunique() > 1:
        _fit_logreg_importance(prefix, X, y, names, seed, _csv_path(out_csv_root, prefix, "motif_feat_imp_logreg.csv"), _fig_path(out_fig_root, prefix, "motif_imp_logreg.png"))
        _fit_tree_importance(prefix, X, y, names, tree_depth, tree_min_leaf, seed, _csv_path(out_csv_root, prefix, "motif_feat_imp_tree.csv"), _fig_path(out_fig_root, prefix, "motif_imp_tree.png"))

    if f"{prefix}_concavity_class" in df_ex.columns:
        _make_joint_table_generic(df_ex, prefix, _csv_path(out_csv_root, prefix, "spectrum_structure_joint.csv"), topN=topN)
        for cls in ["up", "down"]:
            subdir_fig, subdir_csv = _ensure_dir(out_fig_root / _subdir_name(prefix)), _ensure_dir(out_csv_root / _subdir_name(prefix))
            _event_series_dual_axis(df_ex, prefix, subdir_fig, cls, subdir_csv)
            _layered_bar_heat_generic(prefix, df_ex, subdir_fig, cls)
            _boxplot_deltas_generic(df_ex, prefix, subdir_fig, cls)
            _scatter_gap_vs_structure_generic(df_ex, prefix, subdir_fig, cls)
        
        _write_interpretation_report_generic(_ensure_dir(out_csv_root / _subdir_name(prefix)), df_ex, prefix)

    return {"dataset_csv": str(_csv_path(out_csv_root, prefix, "motif_delta_dataset.csv"))}

# ================= 顶层：run =================
def run(examples_csv: Path, out_csv_dir: Path, out_fig_dir: Path,
        style: str="ieee", topN: int=20, tree_depth: int=3, tree_min_leaf: int=8,
        seed: int=0, include_growth: bool=False) -> Dict[str, Dict[str,str]]:
    _ensure_dir(out_csv_dir); _ensure_dir(out_fig_dir)
    name_lower = examples_csv.name.lower()
    main_prefix = "optimal" if "optimal" in name_lower else ("mur" if "mur" in name_lower else "knee")
    
    def _load(p, f): 
        df = _read_csv_maybe(f)
        return (p, df, None, None) if df is not None and not df.empty else None

    prefixes = []
    first = _load(main_prefix, examples_csv)
    if first: prefixes.append(first)

    results = {}
    for (prefix, df_ex, df_sum, df_global) in prefixes:
        try:
            res = _run_one_prefix(prefix, df_ex, out_csv_root=out_csv_dir, out_fig_root=out_fig_dir,
                                  topN=topN, tree_depth=tree_depth, tree_min_leaf=tree_min_leaf, seed=seed)
            results[prefix] = res
        except Exception as e:
            print(f"[Explainer] Error {prefix}: {e}")
            import traceback; traceback.print_exc()
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True, type=Path)
    ap.add_argument("--out-csv",  required=True, type=Path)
    ap.add_argument("--out-dir",  required=True, type=Path)
    ap.add_argument("--style", default="ieee")
    args = ap.parse_args()
    run(args.examples, args.out_csv, args.out_dir, style=args.style)

if __name__ == "__main__":
    main()