# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .viz import apply_style
except Exception:
    def apply_style(style: str = "default"):  # type: ignore
        pass

# ========== 1) 读取工具（优先用 utils_io；若无则本地兜底） ==========
try:
    from .utils_io import read_csv_with_synonyms as _read_csv_with_synonyms  # type: ignore
    HAVE_EXT_READ = True
except Exception:
    HAVE_EXT_READ = False

def _norm_col(s: str) -> str:
    """列名归一化：小写 + 去空白 + 非字母数字改下划线 + 连续下划线压缩"""
    s2 = re.sub(r"[^0-9a-zA-Z]+", "_", str(s).strip().lower())
    s2 = re.sub(r"_+", "_", s2).strip("_")
    return s2

def _read_csv_with_synonyms_fallback(
    path: str | Path,
    synonyms: Dict[str, Iterable[str]],
    *,
    must_have_any: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    """本地兜底版本：宽松匹配列名并返回 {标准名 -> 实际列名} 映射。"""
    path = Path(path)
    df = pd.read_csv(path)
    norm_to_real: Dict[str, str] = {}
    for c in df.columns:
        nc = _norm_col(c)
        if nc not in norm_to_real:
            norm_to_real[nc] = c

    used_map: Dict[str, str] = {}
    for std, cands in synonyms.items():
        for c in cands:
            nc = _norm_col(c)
            if nc in norm_to_real:
                used_map[std] = norm_to_real[nc]
                break

    if must_have_any:
        if not any(std in used_map for std in must_have_any):
            raise KeyError(
                f"[motifs_explainer] {path} 缺少必需列之一：{list(must_have_any)}；实际列：{list(df.columns)}"
            )
    return df, used_map

def read_csv_with_synonyms(
    path: str | Path,
    synonyms: Dict[str, Iterable[str]],
    *,
    must_have_any: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    if HAVE_EXT_READ:
        return _read_csv_with_synonyms(path, synonyms, must_have_any=must_have_any)
    return _read_csv_with_synonyms_fallback(path, synonyms, must_have_any=must_have_any)

# ========== 2) 导出符号 ==========
__all__ = ["explain_motif_deltas"]

# ========== 3) 内部工具 ==========
def _any_finite(vals) -> bool:
    arr = np.asarray(list(vals), dtype=float)
    return np.isfinite(arr).any()

def _build_dataset_from_examples(examples_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """从 motifs 产出的 examples CSV 构造二分类数据集：y=1(前→膝), y=0(膝→后)。"""
    df = pd.read_csv(examples_csv)
    pos_cols = [c for c in df.columns if str(c).startswith("delta_pre_to_knee_")]
    neg_cols = [c for c in df.columns if str(c).startswith("delta_knee_to_post_")]
    if not pos_cols:
        raise FileNotFoundError("未找到 delta_pre_to_knee_* 列，请先运行 rd_cli.py motifs。")
    if not neg_cols:
        raise FileNotFoundError("未找到 delta_knee_to_post_* 列，请先运行 rd_cli.py motifs。")

    base_all = sorted(
        set(re.sub(r"^delta_pre_to_knee_", "", c) for c in pos_cols) |
        set(re.sub(r"^delta_knee_to_post_", "", c) for c in neg_cols)
    )
    rows = []
    for _, r in df.iterrows():
        nk = f"n{int(r.get('n', -1))}_k{int(r.get('k', -1))}"
        pos_feat = {f"delta_pre_to_knee_{b}": r.get(f"delta_pre_to_knee_{b}", np.nan) for b in base_all}
        if _any_finite(pos_feat.values()):
            rows.append({"y":1, "__nk": nk, "__tag":"pos", **pos_feat})
        neg_feat = {f"delta_knee_to_post_{b}": r.get(f"delta_knee_to_post_{b}", np.nan) for b in base_all}
        if _any_finite(neg_feat.values()):
            rows.append({"y":0, "__nk": nk, "__tag":"neg", **neg_feat})
    Xy = pd.DataFrame(rows)
    if Xy.empty:
        raise RuntimeError("没有可用样本：examples.csv 中 Δ 列全为空。")

    Z = pd.DataFrame({"y": Xy["y"], "__nk": Xy["__nk"], "__tag": Xy["__tag"]})
    feat_cols: list[str] = []
    for b in base_all:
        col_pos = f"delta_pre_to_knee_{b}"
        col_neg = f"delta_knee_to_post_{b}"
        vp = Xy[col_pos] if col_pos in Xy.columns else np.nan
        vn = Xy[col_neg] if col_neg in Xy.columns else np.nan
        vv = np.where(Xy["y"]==1, vp, vn)
        Z[col_pos] = vv
        feat_cols.append(col_pos)
    nonempty = Z[feat_cols].apply(lambda row: np.isfinite(row.values).any(), axis=1)
    Z = Z[nonempty].copy()
    for c in feat_cols:
        Z[c] = pd.to_numeric(Z[c], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return Z, feat_cols

# ========== 4) 主函数 ==========
def explain_motif_deltas(
    examples_csv: str | Path,
    out_csv_dir: str | Path = "./results/motifs",
    out_fig_dir: str | Path = "./results/figs/motifs",
    style: str = "default",
    topN: int = 20,
    tree_max_depth: int = 3,
    tree_min_samples_leaf: int = 8,
    random_state: int = 0,
) -> Dict[str, Path]:
    """
    解释“膝点前后”的 Δ 特征：
    - 若已存在 *importance*.csv / *coeffs*.csv，则优先直接读取并可视化；
    - 否则从 examples_csv 构建数据集并训练生成；
    - 最终合并为 markdown 摘要与图表。
    """
    apply_style(style)
    out_csv_dir = Path(out_csv_dir); out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir = Path(out_fig_dir); out_fig_dir.mkdir(parents=True, exist_ok=True)
    examples_csv = Path(examples_csv)

    # 始终构造并落地标准数据集（便于后续复现/排错）
    Z, feat_cols = _build_dataset_from_examples(examples_csv)
    ds_path = out_csv_dir / "motif_delta_dataset.csv"
    Z.to_csv(ds_path, index=False)
    paths: Dict[str, Path] = {"dataset_csv": ds_path}

    # ---------- 4.1 优先尝试“直接读取已生成 CSV” ----------
    csv_lr = out_csv_dir / "motif_feature_importance_logreg.csv"
    csv_tr = out_csv_dir / "motif_feature_importance_tree.csv"
    csv_perm = out_csv_dir / "motif_perm_importance.csv"
    csv_l1 = out_csv_dir / "motif_lr_coeffs.csv"
    txt_rules = out_csv_dir / "motif_tree_rules.txt"

    have_all_existing = csv_lr.exists() and csv_tr.exists() and csv_perm.exists() and csv_l1.exists()

    def _load_existing():
        # LogReg
        lr_syn = {
            "feature":   ["feature","name","term","variable"],
            "coef":      ["coef","coefficient","beta","weight"],
            "coef_abs":  ["coef_abs","abs_coef","abs_coefficient","abs_beta","abs_weight"],
        }
        df_lr, m_lr = read_csv_with_synonyms(csv_lr, lr_syn, must_have_any=["feature"])
        if "coef_abs" not in m_lr and "coef" in m_lr:
            # 若无绝对值列，按 coef 取绝对值构造
            df_lr["coef_abs"] = df_lr[m_lr["coef"]].abs()
            m_lr["coef_abs"] = "coef_abs"
        # Tree
        tr_syn = {
            "feature":    ["feature","name","term","variable"],
            "importance": ["importance","gain","weight","impurity_decrease"],
        }
        df_tr, m_tr = read_csv_with_synonyms(csv_tr, tr_syn, must_have_any=["feature"])
        # Permutation
        p_syn = {
            "feature":               ["feature","name","term","variable"],
            "perm_importance_mean":  ["perm_importance_mean","importance_mean","importance","mean","score"],
            "perm_importance_std":   ["perm_importance_std","importance_std","std","stddev","stdev"],
        }
        df_p, m_p = read_csv_with_synonyms(csv_perm, p_syn, must_have_any=["feature"])
        # L1 stability
        l1_syn = {
            "feature":   ["feature","name","term","variable"],
            "coef_mean": ["coef_mean","mean_coef","coef","weight","avg_coef"],
            "coef_std":  ["coef_std","std_coef","coef_se","std","stdev"],
            "stability": ["stability","freq","support","repeat_rate","presence","nonzero_rate"],
            "nonzero_in_folds": ["nonzero_in_folds","nonzero","count_nonzero","freq"],
        }
        df_l1, m_l1 = read_csv_with_synonyms(csv_l1, l1_syn, must_have_any=["feature"])

        # 统一返回标准列名的 DataFrame（不破坏原始文件）
        lr_out = pd.DataFrame({
            "feature": df_lr[m_lr["feature"]],
            "coef_abs": df_lr[m_lr.get("coef_abs", m_lr.get("coef", m_lr["feature"]))].astype(float)
        })
        tr_out = pd.DataFrame({
            "feature": df_tr[m_tr["feature"]],
            "importance": df_tr[m_tr.get("importance", m_tr["feature"])].astype(float)
        })
        perm_out = pd.DataFrame({
            "feature": df_p[m_p["feature"]],
            "perm_importance_mean": df_p[m_p.get("perm_importance_mean", m_p["feature"])].astype(float),
            "perm_importance_std":  df_p[m_p.get("perm_importance_std",  m_p.get("perm_importance_mean", m_p["feature"]))].astype(float)
        })
        l1_out = pd.DataFrame({
            "feature": df_l1[m_l1["feature"]],
            "coef_mean": df_l1[m_l1.get("coef_mean", m_l1["feature"])].astype(float),
            "coef_std":  df_l1[m_l1.get("coef_std",  m_l1.get("coef_mean", m_l1["feature"]))].astype(float),
            "stability": df_l1[m_l1.get("stability", m_l1["feature"])].astype(float),
        })
        return lr_out, tr_out, perm_out, l1_out

    # ---------- 4.2 否则训练生成 ----------
    def _train_and_save():
        # Optional sklearn
        try:
            from sklearn.model_selection import train_test_split, StratifiedKFold
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier, export_text
            from sklearn.metrics import roc_auc_score
            from sklearn.pipeline import Pipeline
            from sklearn.inspection import permutation_importance
            SK_OK = True
        except Exception:
            SK_OK = False

        if not SK_OK:
            print("[motifs-explain] 未发现 scikit-learn；仅导出 motif_delta_dataset.csv。")
            return None

        X = Z[feat_cols].values.astype(float)
        y = Z["y"].values.astype(int)

        # 训练/验证切分
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, y, test_size=0.25, random_state=random_state, stratify=y
        )

        # L2-LogReg
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty="l2", solver="liblinear",
                                random_state=random_state, max_iter=600)
        lr.fit(Xtr, ytr)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yte, lr.predict_proba(Xte)[:,1])

        imp_lr = pd.DataFrame({
            "feature": feat_cols,
            "coef": lr.coef_.ravel(),
            "coef_abs": np.abs(lr.coef_.ravel()),
        }).sort_values("coef_abs", ascending=False)
        imp_lr.to_csv(csv_lr, index=False)

        # Tree
        from sklearn.tree import DecisionTreeClassifier, export_text
        tree = DecisionTreeClassifier(max_depth=tree_max_depth,
                                      min_samples_leaf=tree_min_samples_leaf,
                                      class_weight="balanced",
                                      random_state=random_state)
        tree.fit(Xtr, ytr)
        auc_t = roc_auc_score(yte, tree.predict_proba(Xte)[:,1])
        imp_tr = pd.DataFrame({
            "feature": feat_cols,
            "importance": tree.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp_tr.to_csv(csv_tr, index=False)
        f_names = [c.replace("delta_pre_to_knee_","") for c in feat_cols] if feat_cols else None
        txt = export_text(tree, feature_names=f_names) if f_names else export_text(tree)
        txt_rules.write_text(txt, encoding="utf-8")

        # L1-LR 稳定性 (5 折)
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        def fit_lr_l1(X, y, C=0.5, seed=0):
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", LogisticRegression(penalty="l1", solver="liblinear",
                                          C=C, class_weight="balanced",
                                          random_state=seed, max_iter=2000))
            ])
            pipe.fit(X, y)
            return pipe.named_steps["lr"].coef_[0]

        coefs = []; nonzero = np.zeros(len(feat_cols), dtype=int)
        for fold, (tr, te) in enumerate(kfold.split(X, y), 1):
            coef = fit_lr_l1(X[tr], y[tr], C=0.5, seed=fold)
            coefs.append(coef); nonzero += (np.abs(coef) > 1e-12).astype(int)
        coef_avg, coef_std = np.mean(coefs, axis=0), np.std(coefs, axis=0)
        coef_df = pd.DataFrame({
            "feature": feat_cols,
            "coef_mean": coef_avg,
            "coef_std": coef_std,
            "nonzero_in_folds": nonzero,
            "stability": nonzero / kfold.get_n_splits()
        }).sort_values(["stability","coef_mean"], ascending=[False, False])
        coef_df.to_csv(csv_l1, index=False)

        # Permutation importance（在全量上）
        from sklearn.inspection import permutation_importance
        from sklearn.pipeline import Pipeline
        lr_full = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(penalty="l1", solver="liblinear",
                                      C=0.5, class_weight="balanced",
                                      random_state=42, max_iter=2000))
        ])
        lr_full.fit(X, y)
        perm = permutation_importance(lr_full, X, y, n_repeats=5,
                                      random_state=random_state, scoring="f1")
        perm_df = pd.DataFrame({
            "feature": [c.replace("delta_pre_to_knee_","") for c in feat_cols],
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std":  perm.importances_std
        }).sort_values("perm_importance_mean", ascending=False)
        perm_df.to_csv(csv_perm, index=False)

        return True  # 已训练并写出

    if not have_all_existing:
        ok = _train_and_save()
        if ok is None:
            # sklearn 不可用且结果文件又不存在：仅返回数据集路径
            return paths

    # 读“现成或刚写出的”CSV，统一标准列再绘图/汇总
    lr_df, tr_df, p_df, l1_df = _load_existing()

    # ---------- 4.3 画图（topN） ----------
    topN = max(0, min(int(topN), len(lr_df)))
    if topN > 0:
        # LogReg
        plt.figure(figsize=(7.2, max(3, 0.35*topN)))
        tmp = lr_df.sort_values("coef_abs", ascending=False).head(topN).iloc[::-1]
        plt.barh(tmp["feature"].str.replace("delta_pre_to_knee_",""), tmp["coef_abs"])
        plt.xlabel("|coefficient|"); plt.title("Logistic Regression feature influence (|coef|)")
        plt.tight_layout(); png_lr = out_fig_dir / "motif_importance_logreg.png"
        plt.savefig(png_lr, dpi=220); plt.close(); paths["logreg_png"] = png_lr

        # Tree
        plt.figure(figsize=(7.2, max(3, 0.35*topN)))
        tmp2 = tr_df.sort_values("importance", ascending=False).head(topN).iloc[::-1]
        plt.barh(tmp2["feature"].str.replace("delta_pre_to_knee_",""), tmp2["importance"])
        plt.xlabel("Gini importance"); plt.title("Decision Tree importance")
        plt.tight_layout(); png_tr = out_fig_dir / "motif_importance_tree.png"
        plt.savefig(png_tr, dpi=220); plt.close(); paths["tree_png"] = png_tr

        # L1-LR coeffs
        topN2 = min(18, len(l1_df))
        if topN2 > 0:
            plot_df = l1_df.reindex(l1_df["coef_mean"].abs().sort_values(ascending=False).index)[:topN2]
            plt.figure(figsize=(8, 5))
            plt.barh(range(topN2), plot_df["coef_mean"].values, xerr=plot_df["coef_std"].values, alpha=0.85)
            plt.yticks(range(topN2), [s.replace("delta_pre_to_knee_","") for s in plot_df["feature"].values])
            plt.axvline(0, color="k", linewidth=1); plt.xlabel("L1-LogReg coefficient (mean ± std across folds)")
            plt.title("Knee vs Non-knee Δ: logistic coefficients (top features)")
            plt.tight_layout(); png_l1 = out_fig_dir / "motif_lr_coeffs.png"
            plt.savefig(png_l1, dpi=220); plt.close(); paths["lr_l1_png"] = png_l1

        # Permutation
        plt.figure(figsize=(7.2, 4.8))
        pp = p_df.sort_values("perm_importance_mean", ascending=False).head(18)
        plt.barh(pp["feature"], pp["perm_importance_mean"], xerr=pp["perm_importance_std"], alpha=0.85)
        plt.gca().invert_yaxis(); plt.xlabel("Permutation importance (mean ± std, F1)")
        plt.title("Permutation importance of Δ-features (logistic)")
        plt.tight_layout(); png_perm = out_fig_dir / "motif_perm_importance.png"
        plt.savefig(png_perm, dpi=220); plt.close(); paths["perm_png"] = png_perm

    # ---------- 4.4 Markdown 摘要 ----------
    top_lr   = set(lr_df.sort_values("coef_abs", ascending=False).head(10)["feature"].str.replace("delta_pre_to_knee_",""))
    top_tree = set(tr_df.sort_values("importance", ascending=False).head(10)["feature"].str.replace("delta_pre_to_knee_",""))
    top_perm = set(p_df.sort_values("perm_importance_mean", ascending=False).head(10)["feature"])
    union = sorted(top_lr | top_tree | top_perm)

    coef_map = {str(row["feature"]).replace("delta_pre_to_knee_",""): (float(row.get("coef_mean", np.nan)), float(row.get("stability", np.nan)))
                for _, row in l1_df.iterrows()}
    tree_map = {str(row["feature"]).replace("delta_pre_to_knee_",""): float(row.get("importance", np.nan))
                for _, row in tr_df.iterrows()}
    perm_map = {str(row["feature"]): (float(row.get("perm_importance_mean", np.nan)), float(row.get("perm_importance_std", np.nan)))
                for _, row in p_df.iterrows()}

    def sign_arrow(x: float) -> str:
        try:
            return "↑(+)" if x > 0 else "↓(-)"
        except Exception:
            return ""

    explanations = {
        "tri": "三角增加 → 局部闭合度上升，通路指数率加快，易触发膝点",
        "c5": "五环出现或增多 → 最短奇环变短，组合通路数骤增",
        "c4": "四环增多 → 局部格栅化，通路冗余提升（适度）",
        "deg_max": "最大度跨阈（近星核）→ 单色“万能配对”形成",
        "selfloop_rich": "自环富集跃迁 → 同色堆叠自由度骤升",
        "gap": "谱隙变大 → 主导与次主导模式分离，生成率更集中",
        "lambda1": "主特征值上升 → 组合通路指数率上升",
        "clustering": "平均聚类系数上升 → 闭合度与三角密度上升",
        "kcore": "核数上升 → 稠密核心形成",
        "near_bip_chord": "近二分+少量弦 → 以小 |R| 获得较高可达性",
    }

    md_lines = [
        "# 膝点判别的关键结构变化（Δ 特征）摘要",
        f"- 数据来源：{Path(examples_csv).name}",
        f"- 样本量：{len(Z)}（正类={int(Z['y'].sum())} / 负类={int((1-Z['y']).sum())}）",
        "- 方法：L2-LogReg、浅层决策树、置换重要性 + L1-LogReg(5折) 稳定性。",
        "",
        "## 交叉验证最重要的 Δ 特征（合并 Top-10）",
        "",
        "| Δ-特征 | L1-系数(均值) | 稳定性 | 树-重要性 | 置换重要性 | 解释 |",
        "|---|---:|---:|---:|---:|---|"
    ]
    def score_key(f: str):
        c_mean, stab = coef_map.get(f, (0.0, 0.0))
        t_imp = tree_map.get(f, 0.0)
        p_mean, _ = perm_map.get(f, (0.0, 0.0))
        return -(abs(c_mean) + 0.5*t_imp + 0.5*p_mean)

    for feat in sorted(union, key=score_key):
        c_mean, stab = coef_map.get(feat, (0.0, 0.0))
        t_imp = tree_map.get(feat, 0.0)
        p_mean, p_std = perm_map.get(feat, (0.0, 0.0))
        md_lines.append(
            f"| Δ{feat} | {c_mean:+.3f} | {stab:.2f} | {t_imp:.3f} | {p_mean:.3f} | {sign_arrow(c_mean)} {explanations.get(feat,'')} |"
        )

    md_path = out_csv_dir / "motif_knee_topline.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    paths.update({
        "logreg_csv": csv_lr,
        "tree_csv":   csv_tr,
        "perm_csv":   csv_perm,
        "lr_l1_coeffs_csv": csv_l1,
        "tree_rules_txt": txt_rules,
        "summary_md": md_path,
    })
    return paths
