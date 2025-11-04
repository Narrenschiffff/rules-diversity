# -*- coding: utf-8 -*-
"""
motifs_explainer.py — 膝点(KNEE) & MUR：结构Δ→谱Δ→增长 的解释与可视化（稳健均值 + 统一口径）

变更要点（相对你提供的修订版）：
- 事件曲线改用稳健均值（winsorized mean，默认双侧各裁5%），避免极端值把均值拽偏；
- 所有占比（整体与分型）统一从 examples 精确计算（count>0 / indicator>=1），不再用 summary 的估计值；
- 修复 sort_values 位置参数错误；
- 自动解读报告中的聚合也改为稳健均值。

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
DELTA_PREFIXES = ("delta_pre_to_knee_", "delta_knee_to_post_")
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

def _forced_target_concavity(df: pd.DataFrame):
    y, name, *_ = _label_by_concavity(df)
    if y.notna().any():
        return y, name
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index), "concavity_na_all_zero"

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
def _subdir_name(prefix: str) -> str:
    return "KNEE" if prefix == "knee" else "MUR"

def _csv_path(root: Path, prefix: str, filename: str) -> Path:
    return _ensure_dir(root / _subdir_name(prefix)) / filename

def _fig_path(root: Path, prefix: str, filename: str) -> Path:
    return _ensure_dir(root / _subdir_name(prefix)) / filename

def _title_head(prefix: str) -> str:
    return "[KNEE]" if prefix == "knee" else "[MUR]"

def _center_label(prefix: str) -> str:
    return "KNEE" if prefix == "knee" else "MUR"

# ================= 占比统计（统一口径：只由 examples 计算） =================
def _discover_auto_bool_features(df: pd.DataFrame) -> set:
    """自动发现 0/1 指示类特征（若存在）。"""
    found = set()
    for suf in AUTO_BOOL_SUFFIX:
        for stage in ("pre", "knee", "post"):
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

def _presence_from_examples(df: pd.DataFrame) -> pd.DataFrame:
    """整体（不分型）presence 表：feature, pre_ratio, knee_ratio, post_ratio"""
    bool_feats = _discover_auto_bool_features(df)
    pre  = _presence_one_stage(df, "pre",  bool_feats)
    knee = _presence_one_stage(df, "knee", bool_feats)
    post = _presence_one_stage(df, "post", bool_feats)
    # 联合并只保留至少一列非空
    out = pd.DataFrame({"feature": list(set(pre.feature) | set(knee.feature) | set(post.feature))})
    out = out.merge(pre,  on="feature", how="left")
    out = out.merge(knee, on="feature", how="left")
    out = out.merge(post, on="feature", how="left")
    mask = out[["pre_ratio", "knee_ratio", "post_ratio"]].notna().any(axis=1)
    out = out.loc[mask].fillna(0.0)
    # 统一的特征顺序（和 motifs.py/_write_summary 保持一致）
    order = ["selfloop_rich","star_core","near_bip_chord","tri","c4","c5"]
    # 落在 order 中的按该顺序，不在的排在后面并按名称排序
    out["__ord__"] = out["feature"].apply(lambda x: order.index(x) if x in order else 10_000)
    out = out.sort_values(["__ord__","feature"]).drop(columns="__ord__").reset_index(drop=True)
    return out

def _bar_pre_center_post(prefix: str, df_summary: pd.DataFrame, out_png: Path):
    labels = df_summary["feature"].tolist()
    X = np.arange(len(labels)); width = 0.25
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.bar(X - width, df_summary["pre_ratio"],  width, label="pre")
    ax.bar(X,         df_summary["knee_ratio"], width, label=_center_label(prefix))
    ax.bar(X + width, df_summary["post_ratio"], width, label="post")
    ax.set_xticks(X, labels, rotation=20)
    ax.set_ylabel("presence ratio")
    ax.set_title(f"{_title_head(prefix)} Motif prevalence: pre / {_center_label(prefix)} / post")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def _heatmap_pre_center_post(prefix: str, df_summary: pd.DataFrame, out_png: Path):
    Z = df_summary.set_index("feature")[["pre_ratio", "knee_ratio", "post_ratio"]].values
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

# ================= KNEE 分型 & 事件曲线（稳健均值） =================
def _augment_concavity_columns_for_knee(df_ex: pd.DataFrame) -> pd.DataFrame:
    y, _, slope_L, slope_R, d2, ratio = _label_by_concavity(df_ex)
    df = df_ex.copy()
    df["knee_slope_L"] = slope_L
    df["knee_slope_R"] = slope_R
    df["knee_d2"] = d2
    df["knee_slope_ratio"] = ratio
    sign = np.sign(d2.replace([np.inf, -np.inf], np.nan)).fillna(0.0)
    df["knee_concavity_class"] = sign.apply(lambda v: "up" if v < 0 else ("down" if v > 0 else "flat"))
    return df

def _rel3(a, b, c):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / a, b / a, c / a

def _event_series_dual_axis(df_knee: pd.DataFrame, out_fig_dir: Path, cls: str, out_csv_dir: Path):
    sub = df_knee[df_knee["knee_concavity_class"] == cls]
    need = ["pre_y", "knee_y", "post_y", "pre_lambda1", "knee_lambda1", "post_lambda1",
            "pre_gap", "knee_gap", "post_gap"]
    if sub.empty or not _has_cols(sub, need):
        return

    # 个体相对 pre 归一后，再按“稳健均值”聚合
    Y1, Y2, Y3 = _rel3(pd.to_numeric(sub["pre_y"], errors="coerce"),
                       pd.to_numeric(sub["knee_y"], errors="coerce"),
                       pd.to_numeric(sub["post_y"], errors="coerce"))
    L1, L2, L3 = _rel3(pd.to_numeric(sub["pre_lambda1"], errors="coerce"),
                       pd.to_numeric(sub["knee_lambda1"], errors="coerce"),
                       pd.to_numeric(sub["post_lambda1"], errors="coerce"))
    G1, G2, G3 = _rel3(pd.to_numeric(sub["pre_gap"], errors="coerce"),
                       pd.to_numeric(sub["knee_gap"], errors="coerce"),
                       pd.to_numeric(sub["post_gap"], errors="coerce"))

    def rmean(s): return _winsorized_mean(pd.Series(s))

    stats = pd.DataFrame({
        "metric": ["lambda1_rel_pre", "gap_rel_pre", "Y_rel_pre"],
        "pre":  [rmean(L1), rmean(G1), rmean(Y1)],
        "knee": [rmean(L2), rmean(G2), rmean(Y2)],
        "post": [rmean(L3), rmean(G3), rmean(Y3)],
        "N":    [len(sub)]*3
    })
    stats.to_csv(out_csv_dir / f"event_series_stats_concavity_{cls}.csv", index=False)

    # === 诊断：单调性 + “后段增量 > 前段增量” ===
    def _check_monotonic(a, b, c):
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        mono = (a < b) & (b < c)
        back_gt_front = ((c - b) > (b - a))
        return ok, mono, back_gt_front
    okY, monoY, bgtfY = _check_monotonic(pd.to_numeric(sub["pre_y"], errors="coerce"),
                                         pd.to_numeric(sub["knee_y"], errors="coerce"),
                                         pd.to_numeric(sub["post_y"], errors="coerce"))
    okG, monoG, bgtfG = _check_monotonic(pd.to_numeric(sub["pre_gap"], errors="coerce"),
                                         pd.to_numeric(sub["knee_gap"], errors="coerce"),
                                         pd.to_numeric(sub["post_gap"], errors="coerce"))
    okL, monoL, bgtfL = _check_monotonic(pd.to_numeric(sub["pre_lambda1"], errors="coerce"),
                                         pd.to_numeric(sub["knee_lambda1"], errors="coerce"),
                                         pd.to_numeric(sub["post_lambda1"], errors="coerce"))
    rows_chk = [
        {"metric":"Y",       "N":int(okY.sum()), "monotonic_inc":int((okY & monoY).sum()), "post_minus_knee_gt_knee_minus_pre":int((okY & bgtfY).sum())},
        {"metric":"gap",     "N":int(okG.sum()), "monotonic_inc":int((okG & monoG).sum()), "post_minus_knee_gt_knee_minus_pre":int((okG & bgtfG).sum())},
        {"metric":"lambda1", "N":int(okL.sum()), "monotonic_inc":int((okL & monoL).sum()), "post_minus_knee_gt_knee_minus_pre":int((okL & bgtfL).sum())},
    ]
    pd.DataFrame(rows_chk).to_csv(out_csv_dir / f"event_series_checks_concavity_{cls}.csv", index=False)

    X = np.array([0, 1, 2]); labs = ["pre", "KNEE", "post"]
    fig, ax1 = plt.subplots(figsize=(9.6, 5.6))
    ax1.plot(X, stats.loc[0, ["pre","knee","post"]], marker="o", label="lambda1 (rel, left)")
    ax1.plot(X, stats.loc[1, ["pre","knee","post"]], marker="o", label="gap (rel, left)")
    ax1.set_ylabel("relative λ1 / gap (left axis)")
    ax1.set_xticks(X, labs)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(X, stats.loc[2, ["pre","knee","post"]], marker="s", linestyle="--", label="Y (rel, right)")
    ax2.set_yscale("log")  # 右轴用对数尺度，避免“knee 看起来不动”的错觉
    ax2.set_ylabel("relative Y (right axis)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, frameon=False, loc="best")

    title_tag = "Up-concave" if cls=="up" else ("Down-convex" if cls=="down" else "Flat")
    ax1.set_title(f"[KNEE: {title_tag}] event study (winsorized mean, relative)")
    fig.tight_layout()
    fig.savefig(out_fig_dir / f"event_series_concavity_{cls}.png", dpi=160)
    plt.close(fig)

# ================= 分层占比（仍然是 examples 精确口径） =================
def _layered_bar_heat(prefix: str, df_knee: pd.DataFrame, out_fig_dir: Path, cls: str):
    sub = df_knee[df_knee["knee_concavity_class"] == cls]
    if sub.empty:
        return
    M = _presence_from_examples(sub)
    if M.empty:
        return
    _bar_pre_center_post(prefix, M, out_fig_dir / f"knee_motif_bar_concavity_{cls}.png")
    _heatmap_pre_center_post(prefix, M, out_fig_dir / f"knee_motif_heatmap_concavity_{cls}.png")

# ================= Δ 箱线 & 散点 =================
def _boxplot_deltas(df_knee: pd.DataFrame, out_fig_dir: Path, cls: str):
    sub = df_knee[df_knee["knee_concavity_class"] == cls]
    if sub.empty:
        return
    spec_vars = [
        ("delta_pre_to_knee_lambda1", "Δλ1(pre→knee)"),
        ("delta_knee_to_post_lambda1", "Δλ1(knee→post)"),
        ("delta_pre_to_knee_gap", "Δgap(pre→knee)"),
        ("delta_knee_to_post_gap", "Δgap(knee→post)"),
        ("delta_pre_to_knee_lap_algebraic", "ΔλL(pre→knee)"),
    ]
    data, labels = [], []
    for c, l in spec_vars:
        if c in sub.columns:
            data.append(pd.to_numeric(sub[c], errors="coerce").values)
            labels.append(l)
    if data:
        fig, ax = plt.subplots(figsize=(9.6, 5.6))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"[KNEE: {'Up' if cls=='up' else 'Down'}] spectral deltas")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_fig_dir / f"knee_delta_box_spectrum_concavity_{cls}.png", dpi=160)
        plt.close(fig)

    motif_vars = [
        ("delta_pre_to_knee_tri", "Δtri(pre→knee)"),
        ("delta_knee_to_post_tri", "Δtri(knee→post)"),
        ("delta_pre_to_knee_star_core", "Δstar(pre→knee)"),
        ("delta_knee_to_post_star_core", "Δstar(knee→post)"),
        ("delta_pre_to_knee_selfloop_rich", "Δselfloop(pre→knee)"),
        ("delta_knee_to_post_selfloop_rich", "Δselfloop(knee→post)"),
        ("delta_pre_to_knee_near_bip_chord", "Δnear-bip(pre→knee)"),
        ("delta_knee_to_post_near_bip_chord", "Δnear-bip(knee→post)"),
    ]
    data, labels = [], []
    for c, l in motif_vars:
        if c in sub.columns:
            data.append(pd.to_numeric(sub[c], errors="coerce").values)
            labels.append(l)
    if data:
        fig, ax = plt.subplots(figsize=(10.0, 5.8))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"[KNEE: {'Up' if cls=='up' else 'Down'}] motif deltas")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_fig_dir / f"knee_delta_box_motif_concavity_{cls}.png", dpi=160)
        plt.close(fig)

def _scatter_gap_vs_structure(df_knee: pd.DataFrame, out_fig_dir: Path, cls: str):
    sub = df_knee[df_knee["knee_concavity_class"] == cls]
    if sub.empty or "delta_knee_to_post_gap" not in sub.columns:
        return
    Y = pd.to_numeric(sub["delta_knee_to_post_gap"], errors="coerce")
    pairs = [
        ("delta_pre_to_knee_tri", "Δtri(pre→knee)", "dtri"),
        ("delta_pre_to_knee_star_core", "Δstar(pre→knee)", "dstar"),
        ("delta_pre_to_knee_selfloop_rich", "Δselfloop(pre→knee)", "dselfloop"),
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
        ax.set_xlabel(lab); ax.set_ylabel("Δgap(knee→post)")
        ax.set_title(f"[KNEE: {'Up' if cls=='up' else 'Down'}] Δgap vs {lab}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_fig_dir / f"knee_scatter_dgap_vs_{tag}_concavity_{cls}.png", dpi=160)
        plt.close(fig)

# ================= 谱-结构联合表（仅 KNEE） =================
def _make_joint_table(ex: pd.DataFrame, out_csv: Path, topN: int):
    keep = [c for c in ex.columns if c.startswith("knee_")] + ["n", "k"]
    knee = ex[keep].copy()
    if knee.empty:
        return
    knee.columns = [c.replace("knee_", "") for c in knee.columns]
    cols = ["n","k","rule_count","y","lambda1","lambda2","gap","lap_algebraic",
            "deg_max","deg_mean","deg_std","diag_cnt","kcore","clustering",
            "tri","c4","c5","near_bip_chord","selfloop_rich","star_core",
            "odd_girth","key_edges","bits"]
    cols = [c for c in cols if c in knee.columns]
    if "y" in knee.columns:
        knee = knee.sort_values("y", ascending=False).head(topN)
    knee[cols].to_csv(out_csv, index=False)

# ================= 自动“解读报告”（仅 KNEE，稳健均值） =================
def _write_interpretation_report_knee(out_csv_dir: Path, df_knee: pd.DataFrame):
    rep = {"N_total": int(len(df_knee))}
    for cls in ["up", "down"]:
        sub = df_knee[df_knee["knee_concavity_class"] == cls]
        if sub.empty:
            continue
        sec = {"N": int(len(sub))}
        for m in ["lambda1", "gap", "y"]:
            pre=f"pre_{m}"; kn=f"knee_{m}"; po=f"post_{m}"
            if _has_cols(sub, [pre, kn, po]):
                sec[f"winsor_mean_{m}"] = {
                    "pre":  _winsorized_mean(sub[pre]),
                    "knee": _winsorized_mean(sub[kn]),
                    "post": _winsorized_mean(sub[po]),
                }
        rep[cls] = sec

    md = ["# 膝点结构解释（自动生成，稳健均值）",
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
                r=sec[key]; md.append(f"- {m}（winsorized mean）: pre={r['pre']:.4g}, knee={r['knee']:.4g}, post={r['post']:.4g}")
        md.append("- 读法：左段（pre→KNEE）反映连通扩张；右段（KNEE→post）常伴随模块化抬头（gap 上升）。")
    (out_csv_dir/"interpretation_knee.md").write_text("\n".join(md), encoding="utf-8")

# ================= 单前缀执行 =================
def _run_one_prefix(prefix: str, df_ex: pd.DataFrame,
                    out_csv_root: Path, out_fig_root: Path,
                    topN: int, tree_depth: int, tree_min_leaf: int, seed: int,
                    df_sum: Optional[pd.DataFrame]=None, df_global: Optional[pd.DataFrame]=None) -> Dict[str,str]:

    # 先保存带分型的 examples（knee）
    if prefix == "knee":
        df_ex = _augment_concavity_columns_for_knee(df_ex)
        df_ex.to_csv(_csv_path(out_csv_root, "knee", "motif_knee_examples_labeled.csv"), index=False)

    # === 统一口径：整体占比也从 examples 精确计算 ===
    overall_presence = _presence_from_examples(df_ex)
    if not overall_presence.empty:
        _bar_pre_center_post(prefix, overall_presence, _fig_path(out_fig_root, prefix, "motif_bar.png"))
        _heatmap_pre_center_post(prefix, overall_presence, _fig_path(out_fig_root, prefix, "motif_heatmap.png"))

    # === Δ 特征 & 标签 ===
    if df_ex is None or df_ex.empty:
        raise RuntimeError(f"[{prefix}] examples DataFrame 为空。")
    y, y_name = _forced_target_concavity(df_ex)
    feat_cols = _select_delta_features(df_ex)
    if not feat_cols:
        raise RuntimeError(f"[{prefix}] 找不到可用的 Δ 特征列。")
    X = df_ex[feat_cols].fillna(0.0).values
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

    # 四类解释（含 gap）
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

    # 仅 KNEE：联合表 + 分型图 + 事件诊断 + 报告
    if prefix == "knee":
        _make_joint_table(df_ex, _csv_path(out_csv_root, "knee", "spectrum_structure_joint.csv"), topN=topN)
        for cls in ["up", "down"]:
            _event_series_dual_axis(df_ex, _ensure_dir(out_fig_root/"KNEE"), cls, _ensure_dir(out_csv_root/"KNEE"))
            _layered_bar_heat("knee", df_ex, _ensure_dir(out_fig_root/"KNEE"), cls)
            _boxplot_deltas(df_ex, _ensure_dir(out_fig_root/"KNEE"), cls)
            _scatter_gap_vs_structure(df_ex, _ensure_dir(out_fig_root/"KNEE"), cls)
        # 总览
        sanity = {
            "N_total": int(len(df_ex)),
            "N_up": int((df_ex.get("knee_concavity_class") == "up").sum()),
            "N_down": int((df_ex.get("knee_concavity_class") == "down").sum()),
        }
        _csv_path(out_csv_root, "knee", "sanity_report.json").write_text(
            json.dumps(sanity, ensure_ascii=False, indent=2), "utf-8"
        )
        _write_interpretation_report_knee(_ensure_dir(out_csv_root/"KNEE"), df_ex)

    return {
        "dataset_csv":  str(_csv_path(out_csv_root, prefix, "motif_delta_dataset.csv")),
        "logreg_csv":   str(_csv_path(out_csv_root, prefix, "motif_feature_importance_logreg.csv")),
        "tree_csv":     str(_csv_path(out_csv_root, prefix, "motif_feature_importance_tree.csv")),
        "perm_csv":     str(_csv_path(out_csv_root, prefix, "motif_perm_importance.csv")),
        "l1_coeffs":    str(_csv_path(out_csv_root, prefix, "motif_lr_coeffs.csv")),
        "integrity":    str(_csv_path(out_csv_root, prefix, "integrity_report.json")),
        "bar_png":      str(_fig_path(out_fig_root, prefix, "motif_bar.png")),
        "heat_png":     str(_fig_path(out_fig_root, prefix, "motif_heatmap.png")),
        "logreg_png":   str(_fig_path(out_fig_root, prefix, "motif_importance_logreg.png")),
        "tree_png":     str(_fig_path(out_fig_root, prefix, "motif_importance_tree.png")),
        "l1_png":       str(_fig_path(out_fig_root, prefix, "motif_lr_coeffs.png")),
        "perm_png":     str(_fig_path(out_fig_root, prefix, "motif_perm_importance.png")),
    }

# ================= 顶层：knee + mur（若存在） =================
def run(examples_csv: Path, out_csv_dir: Path, out_fig_dir: Path,
        style: str="ieee", topN: int=20, tree_depth: int=3, tree_min_leaf: int=8,
        seed: int=0, include_growth: bool=False) -> Dict[str, Dict[str,str]]:

    _ensure_dir(out_csv_dir); _ensure_dir(out_fig_dir)

    # 备份 Δ 数据（便于溯源）
    try:
        df_raw = pd.read_csv(examples_csv)
        cols = [c for c in df_raw.columns if any(c.startswith(p) for p in DELTA_PREFIXES)] + \
               [c for c in ["k","n","pre_y","knee_y","post_y",
                            "pre_rule_count","knee_rule_count","post_rule_count",
                            "pre_lambda1","knee_lambda1","post_lambda1",
                            "pre_gap","knee_gap","post_gap"] if c in df_raw.columns]
        if cols:
            (_csv_path(out_csv_dir, "knee" if "knee" in examples_csv.name.lower() else "mur",
                       "motif_delta_dataset.csv")).write_text(
                df_raw[cols].to_csv(index=False), encoding="utf-8"
            )
    except Exception:
        pass

    def _load_triplet(prefix: str, ex_path: Path):
        df_ex = _read_csv_maybe(ex_path)
        if df_ex is None or df_ex.empty:
            raise FileNotFoundError(f"[{prefix}] examples 缺失或为空：{ex_path}")
        sum_path = out_csv_dir / f"motif_{prefix}_summary.csv"     # 不再直接使用，只做备查
        glob_path = out_csv_dir / f"motif_{prefix}_global_report.csv"
        return prefix, df_ex, _read_csv_maybe(sum_path), _read_csv_maybe(glob_path)

    main_prefix = "mur" if ("mur" in examples_csv.name.lower()) else "knee"
    prefixes = [_load_triplet(main_prefix, examples_csv)]
    other_path = out_csv_dir / ("motif_mur_examples.csv" if main_prefix=="knee" else "motif_knee_examples.csv")
    if other_path.exists():
        try:
            other_prefix = "mur" if main_prefix=="knee" else "knee"
            prefixes.append(_load_triplet(other_prefix, other_path))
        except Exception as e:
            print("[warn] second prefix skipped:", e)

    results = {}
    for (prefix, df_ex, df_sum, df_global) in prefixes:
        res = _run_one_prefix(prefix, df_ex, out_csv_root=out_csv_dir, out_fig_root=out_fig_dir,
                              topN=topN, tree_depth=tree_depth, tree_min_leaf=tree_min_leaf,
                              seed=seed, df_sum=df_sum, df_global=df_global)
        results[prefix] = res
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
