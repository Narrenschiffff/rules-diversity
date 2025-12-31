# -*- coding: utf-8 -*-
"""
motifs_explainer.py — 结构Δ→谱Δ→增长 的解释与可视化（稳健均值 + 统一口径）
支持 KNEE / MUR / OPTIMAL 分析。

CLI/调用保持不变：
rd_cli.py motifs-explain --examples ... --out-csv ... --out-dir ... [其他参数]
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ================= 基本常量/开关 =================
# 支持的前缀组合 (pre->X, X->post)
# 注意：_select_delta_features 会根据这些前缀来筛选特征列
DELTA_PREFIXES = (
    "delta_pre_to_knee_", "delta_knee_to_post_",
    "delta_pre_to_mur_",  "delta_mur_to_post_",
    "delta_pre_to_optimal_", "delta_optimal_to_post_"
)

# 精确 presence 口径：计数类>0，指示类>=1
COUNT_FEATURES  = {"tri", "c4", "c5"}
INDIC_FEATURES  = {"selfloop_rich", "star_core", "near_bip_chord"}
# 允许自动发现额外的“0/1 指示列”
AUTO_BOOL_SUFFIX = {}   # 可按需扩展

# 稳健均值的裁剪比例（双侧各裁 pct），可用环境变量覆盖：RD_TRIM_PCT=0.1 表示 10%
TRIM_PCT = float(os.environ.get("RD_TRIM_PCT", "0.01"))

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
    """对传入序列做 winsorize（按分位数截断到 [p, 1-p]），再取均值。"""
    v = pd.to_numeric(x, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return np.nan
    p = max(0.0, min(0.2, float(trim_pct)))  # 安全约束
    if p > 0.0:
        lo, hi = np.nanpercentile(v, [100*p, 100*(1-p)])
        v = v.clip(lo, hi)
    return float(v.mean())

# ================= 凹凸性打标（强制） =================
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
    # 根据 prefix 动态构造列名
    pre_col = "pre_y"
    mid_col = f"{prefix}_y"
    post_col = "post_y"
    x_pre = "pre_rule_count"
    x_mid = f"{prefix}_rule_count"
    x_post = "post_rule_count"

    # 如果列不存在，尝试回退到 defaults 或者报错？这里简单处理
    if not _has_cols(df, [pre_col, mid_col, post_col]):
        # 如果没有y值列，可能无法计算凹凸性，返回全0
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index), "concavity_na_no_cols"

    y, name, *_ = _label_by_concavity(df, pre_col, mid_col, post_col, x_pre, x_mid, x_post)
    if y.notna().any():
        return y, name
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index), "concavity_na_all_zero"

# ================= Δ 特征收集 =================
def _select_delta_features(df: pd.DataFrame) -> List[str]:
    # 只要列名以 DELTA_PREFIXES 里的任何一个开头，就选入
    cols = [c for c in df.columns if any(c.startswith(p) for p in DELTA_PREFIXES)]
    good = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").values
        # 简单过滤全 NaN 或全常数的列
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
def _subdir_name(prefix: str) -> str:
    return prefix.upper()

def _csv_path(root: Path, prefix: str, filename: str) -> Path:
    return _ensure_dir(root / _subdir_name(prefix)) / filename

def _fig_path(root: Path, prefix: str, filename: str) -> Path:
    return _ensure_dir(root / _subdir_name(prefix)) / filename

def _title_head(prefix: str) -> str:
    return f"[{prefix.upper()}]"

def _center_label(prefix: str) -> str:
    return prefix.upper()

# ================= 占比统计（统一口径：只由 examples 计算） =================
def _discover_auto_bool_features(df: pd.DataFrame, prefix: str) -> set:
    """自动发现 0/1 指示类特征（若存在）。"""
    found = set()
    for suf in AUTO_BOOL_SUFFIX:
        for stage in ("pre", prefix, "post"):
            col = f"{stage}_{suf}"
            if col in df.columns:
                found.add(suf)
    return found

def _presence_one_stage(df: pd.DataFrame, stage: str, bool_feats: set) -> pd.DataFrame:
    """返回一列：feature, {stage}_ratio。只纳入至少出现一次的特征。"""
    rows = []
    # 计数/指示的核心集合
    all_feats = sorted((COUNT_FEATURES | INDIC_FEATURES | bool_feats))
    for feat in all_feats:
        col = f"{stage}_{feat}"
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        if feat in COUNT_FEATURES:
            pres = (v > 0).astype(float)
        else:
            pres = (v >= 1).astype(float)
        if pres.sum() == 0:
            continue
        rows.append((feat, float(np.nanmean(pres))))
    return pd.DataFrame(rows, columns=["feature", f"{stage}_ratio"])

def _presence_from_examples(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """整体（不分型）presence 表：feature, pre_ratio, {prefix}_ratio, post_ratio"""
    bool_feats = _discover_auto_bool_features(df, prefix)
    pre  = _presence_one_stage(df, "pre",  bool_feats)
    mid  = _presence_one_stage(df, prefix, bool_feats)
    post = _presence_one_stage(df, "post", bool_feats)
    
    mid_ratio_col = f"{prefix}_ratio"
    
    # 联合并只保留至少一列非空
    out = pd.DataFrame({"feature": list(set(pre.feature) | set(mid.feature) | set(post.feature))})
    out = out.merge(pre,  on="feature", how="left")
    out = out.merge(mid,  on="feature", how="left")
    out = out.merge(post, on="feature", how="left")
    mask = out[["pre_ratio", mid_ratio_col, "post_ratio"]].notna().any(axis=1)
    out = out.loc[mask].fillna(0.0)
    # 统一的特征顺序（和 motifs.py/_write_summary 保持一致）
    order = ["selfloop_rich","star_core","near_bip_chord","tri","c4","c5"]
    # 落在 order 中的按该顺序，不在的排在后面并按名称排序
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
    ax.set_ylabel("presence ratio")
    ax.set_title(f"{_title_head(prefix)} Motif prevalence: pre / {_center_label(prefix)} / post")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def _heatmap_pre_center_post(prefix: str, df_summary: pd.DataFrame, out_png: Path):
    mid_col = f"{prefix}_ratio"
    if mid_col not in df_summary.columns: return

    Z = df_summary.set_index("feature")[["pre_ratio", mid_col, "post_ratio"]].values
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    im = ax.imshow(Z, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(3), ["pre", _center_label(prefix), "post"])
    ax.set_yticks(range(len(df_summary)), df_summary["feature"])
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"{_title_head(prefix)} Motif prevalence heatmap (presence)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# ================= 学习器 =================
def _fit_logreg_importance(prefix, X, y, names, seed, out_csv, out_png):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed)),
    ])
    pipe.fit(X, y)
    coefs = np.abs(pipe.named_steps["clf"].coef_.reshape(-1))
    df = pd.DataFrame({"feature": names, "abs_coef": coefs}).sort_values("abs_coef", ascending=False)
    df.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.barh(df["feature"], df["abs_coef"])
    ax.set_xlabel("|coef|")
    ax.set_title(f"{_title_head(prefix)} Logistic Regression feature influence")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

def _fit_tree_importance(prefix, X, y, names, depth, min_leaf, seed, out_csv, out_png):
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed)
    tree.fit(X, y)
    imp = tree.feature_importances_
    df = pd.DataFrame({"feature": names, "gini_importance": imp}).sort_values("gini_importance", ascending=False)
    df.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.barh(df["feature"], df["gini_importance"])
    ax.set_xlabel("Gini importance")
    ax.set_title(f"{_title_head(prefix)} Decision Tree importance")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

def _cv_l1_logreg_coeffs(prefix, X, y, names, seed, out_csv, out_png):
    n_pos = int(np.sum(y == 1)); n_neg = int(np.sum(y == 0))
    k = max(2, min(5, n_pos if n_pos > 0 else 2, n_neg if n_neg > 0 else 2))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scaler = StandardScaler(); rows = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=4000, random_state=seed)
        Xtr = scaler.fit_transform(X[tr])
        clf.fit(Xtr, y[tr])
        rows.append(np.abs(clf.coef_.reshape(-1)))
    M = np.vstack(rows); mean, std = M.mean(0), M.std(0)
    df = pd.DataFrame({"feature": names, "abs_coef_mean": mean, "abs_coef_std": std}) \
            .sort_values("abs_coef_mean", ascending=False)
    df.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.barh(df["feature"], df["abs_coef_mean"], xerr=df["abs_coef_std"], alpha=0.85, capsize=3)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("L1 coef (mean±std)")
    ax.set_title(f"{_title_head(prefix)} Δ-feature coefficients (CV L1)")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

def _perm_importance_logreg(prefix, X, y, names, seed, out_csv, out_png):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed)),
    ])
    pipe.fit(X, y)
    r = permutation_importance(pipe, X, y, n_repeats=50, random_state=seed, scoring="roc_auc")
    mean, std = r.importances_mean, r.importances_std
    df = pd.DataFrame({"feature": names, "perm_mean": mean, "perm_std": std}).sort_values("perm_mean", ascending=False)
    df.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.barh(df["feature"], df["perm_mean"], xerr=df["perm_std"], alpha=0.9, capsize=3)
    ax.set_xlabel("Permutation importance (roc_auc)")
    ax.set_title(f"{_title_head(prefix)} Permutation importance")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return df

# ================= KNEE/OPTIMAL 分型 & 事件曲线（稳健均值） =================
def _augment_concavity_columns_generic(df_ex: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    计算基于 prefix 点的凹凸性。
    Input columns: pre_y, {prefix}_y, post_y, pre_rule_count...
    Output columns: {prefix}_slope_L, {prefix}_concavity_class, etc.
    """
    pre_col = "pre_y"
    mid_col = f"{prefix}_y"
    post_col = "post_y"
    x_pre = "pre_rule_count"
    x_mid = f"{prefix}_rule_count"
    x_post = "post_rule_count"

    y, _, slope_L, slope_R, d2, ratio = _label_by_concavity(
        df_ex, pre_col, mid_col, post_col, x_pre, x_mid, x_post
    )
    df = df_ex.copy()
    df[f"{prefix}_slope_L"] = slope_L
    df[f"{prefix}_slope_R"] = slope_R
    df[f"{prefix}_d2"] = d2
    df[f"{prefix}_slope_ratio"] = ratio
    sign = np.sign(d2.replace([np.inf, -np.inf], np.nan)).fillna(0.0)
    df[f"{prefix}_concavity_class"] = sign.apply(lambda v: "up" if v < 0 else ("down" if v > 0 else "flat"))
    return df

def _rel3(a, b, c):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / a, b / a, c / a

def _event_series_dual_axis(df_ex: pd.DataFrame, prefix: str, out_fig_dir: Path, cls: str, out_csv_dir: Path):
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns:
        return
    sub = df_ex[df_ex[cls_col] == cls]

    mid_y = f"{prefix}_y"
    mid_lam = f"{prefix}_lambda1"
    mid_gap = f"{prefix}_gap"

    need = ["pre_y", mid_y, "post_y", "pre_lambda1", mid_lam, "post_lambda1",
            "pre_gap", mid_gap, "post_gap"]
    if sub.empty or not _has_cols(sub, need):
        return

    # 个体相对 pre 归一后，再按“稳健均值”聚合
    Y1, Y2, Y3 = _rel3(pd.to_numeric(sub["pre_y"], errors="coerce"),
                       pd.to_numeric(sub[mid_y], errors="coerce"),
                       pd.to_numeric(sub["post_y"], errors="coerce"))
    L1, L2, L3 = _rel3(pd.to_numeric(sub["pre_lambda1"], errors="coerce"),
                       pd.to_numeric(sub[mid_lam], errors="coerce"),
                       pd.to_numeric(sub["post_lambda1"], errors="coerce"))
    G1, G2, G3 = _rel3(pd.to_numeric(sub["pre_gap"], errors="coerce"),
                       pd.to_numeric(sub[mid_gap], errors="coerce"),
                       pd.to_numeric(sub["post_gap"], errors="coerce"))

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
    fig, ax1 = plt.subplots(figsize=(9.6, 5.6))
    ax1.plot(X, stats.loc[0, ["pre", prefix, "post"]], marker="o", label="lambda1 (rel, left)")
    ax1.plot(X, stats.loc[1, ["pre", prefix, "post"]], marker="o", label="gap (rel, left)")
    ax1.set_ylabel("relative λ1 / gap (left axis)")
    ax1.set_xticks(X, labs)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(X, stats.loc[2, ["pre", prefix, "post"]], marker="s", linestyle="--", label="Y (rel, right)")
    ax2.set_yscale("log")
    ax2.set_ylabel("relative Y (right axis)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, frameon=False, loc="best")

    title_tag = "Up-concave" if cls=="up" else ("Down-convex" if cls=="down" else "Flat")
    ax1.set_title(f"[{_center_label(prefix)}: {title_tag}] event study (winsorized mean, relative)")
    fig.tight_layout()
    fig.savefig(out_fig_dir / f"event_series_concavity_{cls}.png", dpi=160)
    plt.close(fig)

# ================= 分层占比 =================
def _layered_bar_heat_generic(prefix: str, df_ex: pd.DataFrame, out_fig_dir: Path, cls: str):
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns: return
    
    sub = df_ex[df_ex[cls_col] == cls]
    if sub.empty:
        return
    M = _presence_from_examples(sub, prefix)
    if M.empty:
        return
    _bar_pre_center_post(prefix, M, out_fig_dir / f"{prefix}_motif_bar_concavity_{cls}.png")
    _heatmap_pre_center_post(prefix, M, out_fig_dir / f"{prefix}_motif_heatmap_concavity_{cls}.png")

# ================= Δ 箱线 & 散点 =================
def _boxplot_deltas_generic(df_ex: pd.DataFrame, prefix: str, out_fig_dir: Path, cls: str):
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns: return
    sub = df_ex[df_ex[cls_col] == cls]
    if sub.empty: return

    # 动态构造列名
    def dcol(part, feat):
        if part == "pre_to_mid": return f"delta_pre_to_{prefix}_{feat}"
        if part == "mid_to_post": return f"delta_{prefix}_to_post_{feat}"
        return ""

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
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns: return
    sub = df_ex[df_ex[cls_col] == cls]
    
    y_col = f"delta_{prefix}_to_post_gap"
    if sub.empty or y_col not in sub.columns:
        return
    Y = pd.to_numeric(sub[y_col], errors="coerce")
    
    pairs = [
        (f"delta_pre_to_{prefix}_tri", f"Δtri(pre→{prefix})", "dtri"),
        (f"delta_pre_to_{prefix}_star_core", f"Δstar(pre→{prefix})", "dstar"),
        (f"delta_pre_to_{prefix}_selfloop_rich", f"Δselfloop(pre→{prefix})", "dselfloop"),
    ]
    for col, lab, tag in pairs:
        if col not in sub.columns:
            continue
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

# ================= 谱-结构联合表 =================
def _make_joint_table_generic(ex: pd.DataFrame, prefix: str, out_csv: Path, topN: int):
    # 保留 {prefix}_xxx 列
    keep = [c for c in ex.columns if c.startswith(f"{prefix}_")] + ["n", "k"]
    sub = ex[keep].copy()
    if sub.empty: return
    sub.columns = [c.replace(f"{prefix}_", "") for c in sub.columns]
    
    cols = ["n","k","rule_count","y","lambda1","lambda2","gap","lap_algebraic",
            "deg_max","deg_mean","deg_std","diag_cnt","kcore","clustering",
            "tri","c4","c5","near_bip_chord","selfloop_rich","star_core",
            "odd_girth","key_edges","bits"]
    cols = [c for c in cols if c in sub.columns]
    if "y" in sub.columns:
        sub = sub.sort_values("y", ascending=False).head(topN)
    sub[cols].to_csv(out_csv, index=False)

# ================= 自动“解读报告” =================
def _write_interpretation_report_generic(out_csv_dir: Path, df_ex: pd.DataFrame, prefix: str):
    cls_col = f"{prefix}_concavity_class"
    if cls_col not in df_ex.columns: return
    
    rep = {"N_total": int(len(df_ex))}
    for cls in ["up", "down"]:
        sub = df_ex[df_ex[cls_col] == cls]
        if sub.empty:
            continue
        sec = {"N": int(len(sub))}
        for m in ["lambda1", "gap", "y"]:
            pre=f"pre_{m}"; mid=f"{prefix}_{m}"; po=f"post_{m}"
            if _has_cols(sub, [pre, mid, po]):
                sec[f"winsor_mean_{m}"] = {
                    "pre":  _winsorized_mean(sub[pre]),
                    prefix: _winsorized_mean(sub[mid]),
                    "post": _winsorized_mean(sub[po]),
                }
        rep[cls] = sec

    md = [f"# {prefix.upper()} 结构解释（自动生成，稳健均值）",
          f"- 总样本：{rep.get('N_total', 0)}",
          "", "## 分型综述"]
    for cls in ["up", "down"]:
        if cls not in rep:
            md.append(f"- {cls}: 无样本"); continue
        md.append(f"### {('上凹 Up-concave' if cls=='up' else '下凸 Down-convex')}")
        sec = rep[cls]; md.append(f"- 样本数：{sec['N']}")
        for m in ["lambda1","gap","y"]:
            key=f"winsor_mean_{m}"
            if key in sec:
                r=sec[key]
                md.append(f"- {m}（winsorized mean）: pre={r['pre']:.4g}, {prefix}={r[prefix]:.4g}, post={r['post']:.4g}")
        md.append(f"- 读法：左段（pre→{prefix.upper()}）与右段（{prefix.upper()}→post）的变化趋势。")
    (out_csv_dir/f"interpretation_{prefix}.md").write_text("\n".join(md), encoding="utf-8")

# ================= 单前缀执行 =================
def _run_one_prefix(prefix: str, df_ex: pd.DataFrame,
                    out_csv_root: Path, out_fig_root: Path,
                    topN: int, tree_depth: int, tree_min_leaf: int, seed: int,
                    df_sum: Optional[pd.DataFrame]=None, df_global: Optional[pd.DataFrame]=None) -> Dict[str,str]:

    # 1. 凹凸性打标 & 保存
    # 对所有类型（knee/mur/optimal）都计算凹凸性，便于后续分析
    df_ex = _augment_concavity_columns_generic(df_ex, prefix)
    df_ex.to_csv(_csv_path(out_csv_root, prefix, f"motif_{prefix}_examples_labeled.csv"), index=False)

    # 2. 整体占比
    overall_presence = _presence_from_examples(df_ex, prefix)
    if not overall_presence.empty:
        _bar_pre_center_post(prefix, overall_presence, _fig_path(out_fig_root, prefix, "motif_bar.png"))
        _heatmap_pre_center_post(prefix, overall_presence, _fig_path(out_fig_root, prefix, "motif_heatmap.png"))

    # 3. Δ 特征 & 标签 (强制基于凹凸性作为目标 y)
    if df_ex is None or df_ex.empty:
        raise RuntimeError(f"[{prefix}] examples DataFrame 为空。")
    
    y, y_name = _forced_target_concavity(df_ex, prefix)
    feat_cols = _select_delta_features(df_ex)
    if not feat_cols:
        # 如果没有 Δ 特征，可能无法做机器学习分析，但这在 optimal 分析中也可能发生
        # 尝试继续，或者生成空报告
        pass
    
    X = df_ex[feat_cols].fillna(0.0).values if feat_cols else np.zeros((len(df_ex), 0))
    names = _strip_prefix_list(feat_cols)

    # 完整性报告
    zero_vars = [c for c in feat_cols if df_ex[c].nunique(dropna=False) <= 1]
    integ = {
        "prefix": prefix, "N_total": int(len(y)), "pos(1)": int(y.sum()),
        "neg(0)": int((1-y).sum()), "label_used": y_name,
        "delta_features": names, "zero_variance_features": _strip_prefix_list(zero_vars)
    }
    _csv_path(out_csv_root, prefix, "integrity_report.json").write_text(
        json.dumps(integ, ensure_ascii=False, indent=2), "utf-8"
    )

    # 4. 机器学习解释 (仅当特征存在且样本足够时)
    if feat_cols and len(y) > 5 and y.nunique() > 1:
        _fit_logreg_importance(prefix, X, y, names, seed,
            _csv_path(out_csv_root, prefix, "motif_feature_importance_logreg.csv"),
            _fig_path(out_fig_root, prefix, "motif_importance_logreg.png"))
        _fit_tree_importance(prefix, X, y, names, tree_depth, tree_min_leaf, seed,
            _csv_path(out_csv_root, prefix, "motif_feature_importance_tree.csv"),
            _fig_path(out_fig_root, prefix, "motif_importance_tree.png"))
        _cv_l1_logreg_coeffs(prefix, X, y, names, seed,
            _csv_path(out_csv_root, prefix, "motif_lr_coeffs.csv"),
            _fig_path(out_fig_root, prefix, "motif_lr_coeffs.png"))
        _perm_importance_logreg(prefix, X, y, names, seed,
            _csv_path(out_csv_root, prefix, "motif_perm_importance.csv"),
            _fig_path(out_fig_root, prefix, "motif_perm_importance.png"))

        # 去 gap 消融
        if any(n.endswith("gap") for n in names) or ("gap" in names):
            idx = [i for i, n in enumerate(names) if not n.endswith("gap")]
            if idx:
                X2 = X[:, idx]; names2 = [names[i] for i in idx]
                _fit_logreg_importance(prefix, X2, y, names2, seed,
                    _csv_path(out_csv_root, prefix, "motif_feature_importance_logreg_nogap.csv"),
                    _fig_path(out_fig_root, prefix, "motif_importance_logreg_nogap.png"))
                _fit_tree_importance(prefix, X2, y, names2, tree_depth, tree_min_leaf, seed,
                    _csv_path(out_csv_root, prefix, "motif_feature_importance_tree_nogap.csv"),
                    _fig_path(out_fig_root, prefix, "motif_importance_tree_nogap.png"))
                _cv_l1_logreg_coeffs(prefix, X2, y, names2, seed,
                    _csv_path(out_csv_root, prefix, "motif_lr_coeffs_nogap.csv"),
                    _fig_path(out_fig_root, prefix, "motif_lr_coeffs_nogap.png"))
                _perm_importance_logreg(prefix, X2, y, names2, seed,
                    _csv_path(out_csv_root, prefix, "motif_perm_importance_nogap.csv"),
                    _fig_path(out_fig_root, prefix, "motif_perm_importance_nogap.png"))

    # 5. 联合表 + 分型图 + 事件诊断 + 报告 (所有类型都做)
    # 只要支持 concave class
    if f"{prefix}_concavity_class" in df_ex.columns:
        _make_joint_table_generic(df_ex, prefix, _csv_path(out_csv_root, prefix, "spectrum_structure_joint.csv"), topN=topN)
        for cls in ["up", "down"]:
            subdir_fig = _ensure_dir(out_fig_root / _subdir_name(prefix))
            subdir_csv = _ensure_dir(out_csv_root / _subdir_name(prefix))
            
            _event_series_dual_axis(df_ex, prefix, subdir_fig, cls, subdir_csv)
            _layered_bar_heat_generic(prefix, df_ex, subdir_fig, cls)
            _boxplot_deltas_generic(df_ex, prefix, subdir_fig, cls)
            _scatter_gap_vs_structure_generic(df_ex, prefix, subdir_fig, cls)

        sanity = {
            "N_total": int(len(df_ex)),
            "N_up": int((df_ex.get(f"{prefix}_concavity_class") == "up").sum()),
            "N_down": int((df_ex.get(f"{prefix}_concavity_class") == "down").sum()),
        }
        _csv_path(out_csv_root, prefix, "sanity_report.json").write_text(
            json.dumps(sanity, ensure_ascii=False, indent=2), "utf-8"
        )
        _write_interpretation_report_generic(_ensure_dir(out_csv_root / _subdir_name(prefix)), df_ex, prefix)

    return {
        "dataset_csv":  str(_csv_path(out_csv_root, prefix, "motif_delta_dataset.csv")),
        "integrity":    str(_csv_path(out_csv_root, prefix, "integrity_report.json")),
        "bar_png":      str(_fig_path(out_fig_root, prefix, "motif_bar.png")),
    }

# ================= 顶层：run =================
def run(examples_csv: Path, out_csv_dir: Path, out_fig_dir: Path,
        style: str="ieee", topN: int=20, tree_depth: int=3, tree_min_leaf: int=8,
        seed: int=0, include_growth: bool=False) -> Dict[str, Dict[str,str]]:

    _ensure_dir(out_csv_dir); _ensure_dir(out_fig_dir)

    # 自动推断主要前缀
    name_lower = examples_csv.name.lower()
    if "optimal" in name_lower:
        main_prefix = "optimal"
    elif "knee" in name_lower:
        main_prefix = "knee"
    elif "mur" in name_lower:
        main_prefix = "mur"
    else:
        # 默认回退
        main_prefix = "knee"

    def _load_triplet(prefix: str, ex_path: Path):
        df_ex = _read_csv_maybe(ex_path)
        if df_ex is None or df_ex.empty:
            raise FileNotFoundError(f"[{prefix}] examples 缺失或为空：{ex_path}")
        sum_path = out_csv_dir / f"motif_{prefix}_summary.csv"
        glob_path = out_csv_dir / f"motif_{prefix}_global_report.csv"
        return prefix, df_ex, _read_csv_maybe(sum_path), _read_csv_maybe(glob_path)

    prefixes = []
    try:
        prefixes.append(_load_triplet(main_prefix, examples_csv))
    except Exception as e:
        print(f"[ERROR] Failed to load main prefix {main_prefix}: {e}")

    # 尝试加载同一目录下的其他类型文件 (optional)
    # 例如，如果当前跑 optimal，也顺便看看有没有 knee/mur
    for other in ["knee", "mur", "optimal"]:
        if other == main_prefix: continue
        other_path = examples_csv.parent / f"motif_{other}_examples.csv"
        if other_path.exists():
            try:
                prefixes.append(_load_triplet(other, other_path))
            except:
                pass

    results = {}
    for (prefix, df_ex, df_sum, df_global) in prefixes:
        try:
            print(f"[Explainer] Running for prefix: {prefix} ...")
            res = _run_one_prefix(prefix, df_ex, out_csv_root=out_csv_dir, out_fig_root=out_fig_dir,
                                  topN=topN, tree_depth=tree_depth, tree_min_leaf=tree_min_leaf,
                                  seed=seed, df_sum=df_sum, df_global=df_global)
            results[prefix] = res
        except Exception as e:
            print(f"[Explainer] Error running {prefix}: {e}")
            import traceback
            traceback.print_exc()

    return results

# ================= CLI =================
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
    ap.add_argument("--include-growth", action="store_true")
    args = ap.parse_args()

    _ensure_dir(args.out_csv); _ensure_dir(args.out_dir)
    out = run(args.examples, args.out_csv, args.out_dir, args.style,
              args.topN, args.tree_depth, args.tree_min_leaf, args.seed, args.include_growth)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
