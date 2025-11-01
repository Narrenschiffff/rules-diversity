# 规则–模式多目标搜索与膝点分析

> 在 \(n\times n\) 环面格上，以尽量小的规则复杂度 \(|R|\) 生成尽量多的合法模式 \(Z_n\)，并在双目标平面上捕捉帕累托前沿与膝点代表。
>
> 核心方法：Stage-1 穷举基线 + NSGA-II 进化搜索；评估链路由 Hutch++ 迹估计、Krylov/Lanczos 谱分析与对称性折叠支撑。

![Quicklook](docs/_assets/quicklook.png)

---

## 目录

1. [问题与目标](#问题与目标)
2. [图同态视角与数学基础](#图同态视角与数学基础)
3. [转移算子与迹计数](#转移算子与迹计数)
4. [谱界、随机迹估计与可扩展性](#谱界随机迹估计与可扩展性)
5. [两阶段算法框架](#两阶段算法框架)
6. [对称性折叠与结构分析](#对称性折叠与结构分析)
7. [膝点判别与结构画像](#膝点判别与结构画像)
8. [项目结构与核心模块](#项目结构与核心模块)
9. [命令行使用指南](#命令行使用指南)
10. [Python 批量评估示例](#python-批量评估示例)
11. [输出目录约定](#输出目录约定)

---

## 问题与目标

- **空间与状态**：在环面格 \(T_n = C_n \times C_n\) 上，每个格点取 \([k]\) 中的一种状态。
- **规则矩阵** \(R \in \{0,1\}^{k\times k}\)：元素 \(R_{ab}=1\) 表示状态 \(a\) 与 \(b\) 可以在格点相邻；自环允许状态自适配。
- **目标函数**：
  - 复杂度 \(f_1(R)=|R|=\sum_{a,b}R_{ab}\)（越小越好）。
  - 合法模式规模 \(f_2(R)=Z_n(R)\) 或其估计 \(\widehat{\mathrm{trace}}(T_R^n)\)（越大越好）。
- **研究问题**：在 \((|R|, Z_n)\) 平面上刻画帕累托前沿，识别稳定的“膝点”作为复杂度与收益的最佳折中。

---

## 图同态视角与数学基础

1. **状态图构造**：由规则矩阵生成状态图 \(G_R\)，顶点为状态集合 \([k]\)，若 \(R_{ab}=1\) 则 \((a,b)\in E(G_R)\)。
2. **命题**：合法模式数量等价于图同态数：
   \[
   Z_n(R) = \#\mathcal{X}_n(R) = \mathrm{Hom}(T_n, G_R).
   \]
3. **证明要点**：每个合法赋值 \(f\colon V(T_n)\to [k]\) 保持邻接关系，因此是从 \(T_n\) 到 \(G_R\) 的图同态；反之亦然。
4. **直观理解**：在图同态框架下，规则优化即是约束状态图结构以放大小尺度图 \(T_n\) 的同态数。

---

## 转移算子与迹计数

- **环状合法行集合** \(\mathcal{L}\)：所有长度为 \(n\) 的环状行向量，逐列满足 \(R\)。记 \(m=|\mathcal{L}|\)。
- **转移算子** \(T\in \{0,1\}^{m\times m}\)：
  \[
  T_{ab}=1 \iff \forall j \in \{1,\dots,n\},\; R_{a_j b_j}=1.
  \]
- **矩阵幂的含义**：
  - \((T^2)_{ac}=\sum_b T_{ab}T_{bc}\) 计数行 \(a\) 堆叠到 \(b\) 再到 \(c\) 的方案。
  - \((T^n)_{aa}\) 对应从行 \(a\) 出发、堆叠 \(n\) 行并闭合的合法序列。
- **模式与迹**：
  \[
  Z_n(R) = \sum_{a\in\mathcal{L}} (T^n)_{aa} = \mathrm{trace}(T^n).
  \]
- **示例（\(n=2, k=2\)）**：
  \[
  R = \begin{pmatrix}1&1\\1&0\end{pmatrix},\; \mathcal{L}=\{\texttt{00},\texttt{01},\texttt{10}\},\; T = \begin{pmatrix}1&1&1\\1&0&1\\1&1&0\end{pmatrix},\; \mathrm{trace}(T^2)=7.
  \]

---

## 谱界、随机迹估计与可扩展性

- **Perron–Frobenius**：\(T\ge 0\Rightarrow \lambda_{\max} = \rho(T)\)，从而 \(Z_n \ge \lambda_{\max}^n\)。
- **上界**：朴素估计 \(Z_n \le m \cdot \lambda_{\max}^n\)，配合 Gershgorin 或最大行和可得 \(\lambda_{\max}\) 的易计算界。
- **随机化迹估计**：
  - Hutchinson：随机向量 \(z\in\{\pm1\}^m\) 使 \(\mathbb{E}[z^\top A z]=\mathrm{trace}(A)\)。
  - Hutch++：显式提取主子空间，再对剩余部分做 Hutchinson，方差显著降低。
- **Krylov / Lanczos**：仅依赖矩阵向量积即可提取领先特征值与幂次估计，适合大规模 \(m\)。

---

## 两阶段算法框架

### Stage-1：小规模精确基线

- 穷举所有规则，生成 **raw** 与 **canon** 两种口径的帕累托前沿。
- 额外标注 archetypes，便于后续种子初始化与结构对比。
- 输出完整 CSV（全体、前沿）作为真值参考。

### GA：NSGA-II 多目标进化

- **目标**：最小化 \(|R|\)，最大化 \(\widehat{\mathrm{trace}}(T^n)\)。
- **核心机制**：
  - 父本 canonical 对齐后再交叉，保持块结构。
  - 评估缓存：规则 canonical 哈希 + 行态 LRU，避免重复计算。
  - 自适应评估：前沿邻域提升样本数与迭代次数，远离前沿则降低开销。
- **经验超参**：种群 24–48，迭代 30–80，Hutch++ 样本 8–24，power iters 20–50。

---

## 对称性折叠与结构分析

- **状态重标（规则置换）**：\(\pi \in S_k\) 诱导 \(R' = P_\pi^\top R P_\pi\)。`rules.symcanon.canonical_bits` 负责寻找最小字典序代表。
- **模式几何**：平移 × 旋转 × 镜像（\(D_4\)）及其与状态置换的组合。
- **统计口径**：
  - `raw_count`：不去重。
  - `geom_dedup`：仅几何对称折叠。
  - `geom_perm_dedup`：几何 + 状态重标。
- **影响**：减少等价个体占用档案，强化 GA 多样性；在结果解释中突出“本质不同”的模式。

---

## 膝点判别与结构画像

- **二阶差分法**：\(\Delta^2 y(r)=y(r+1)-2y(r)+y(r-1)\)，峰值对应局部最大曲率。
- **L-curve 垂距法**：取前沿两端点连线，寻找垂直距离最大点。
- **稳健策略**：两法结果一致或相差不超过 1 的邻域时视为膝点区间。
- **结构特征**：分析度分布、自环占比、二分性、motif 统计（tri/c4/c5）、k-core 等，理解膝点前后结构变化。
- **谱隙观察**：经验上膝点附近常伴随 \(\lambda_1-\lambda_2\) 的显著变化，提示主导模式增长机制的转折。

---

## 项目结构与核心模块

```
rules-diversity/
├─ rules/
│  ├─ stage1_exact.py      # Stage-1 穷举、前沿筛选、原型标注
│  ├─ ga.py                # NSGA-II 配置、canonical 对齐、批量评估
│  ├─ eval.py              # make_rule_matrix / bits 编码 / evaluate_rules_batch
│  ├─ ops.py               # 行枚举、转移算子 matvec、Hutch++ / Lanczos 管线
│  ├─ spectrum.py          # 谱分析与上界、特征值估计
│  ├─ symcanon.py          # 状态置换 canonical、父本对齐工具
│  ├─ symmetry.py          # 模式几何/重标统计与示例导出
│  ├─ motifs.py            # 膝点 motif 挖掘与跨 (n,k) 归纳
│  ├─ motifs_explainer.py  # 膝点结构特征的解释性分析
│  ├─ viz.py               # 前沿/谱/熵可视化工具
│  ├─ structures.py        # 结构原型扫描与绘制（可选依赖）
│  ├─ utils_io.py          # CSV/目录工具，与 CLI 保持一致
│  └─ logging_setup.py     # 统一日志格式
├─ scripts/rd_cli.py       # 统一命令行入口（本 README 的所有示例）
├─ notebooks/results/      # 默认输出：out_csv / figs / symmetry / motifs / archetypes
├─ requirements.txt        # 运行依赖
└─ pyproject.toml          # 包配置
```

---

## 命令行使用指南

所有命令均通过 `python -u scripts/rd_cli.py <subcommand> [...]` 调用，可用 `-v/--verbosity {0,1,2}` 控制日志级别。

### 1. Stage-1：小规模精确扫描

```bash
python -u scripts/rd_cli.py stage1 \
  --n 4 --k 4 \
  --out-csv notebooks/results/out_csv \
  --verbosity 2
```

- 生成全量规则 CSV 与帕累托前沿 CSV。
- `--no-canon` 关闭 canonical 去重；`--no-archetypes` 跳过原型标注。

### 2. GA：NSGA-II 多目标进化

```bash
python -u scripts/rd_cli.py ga \
  --n 4 --k 4 --device cuda \
  --pop-size 48 --generations 60 \
  --p-mut 0.08 --p-cx 0.85 --elite-keep 8 \
  --r-vals 2 --power-iters 40 \
  --trace-mode hutchpp --hutch-s 24 \
  --seed-from-stage1 --max-stage1-seeds 400 \
  --progress-every 8 --verbosity 2 \
  --out-csv notebooks/results/out_csv
```

- `--reuse` 若检测到现有前沿 CSV 则跳过运行。
- `--fast-eval` 使用轻量评估；`--no-lanczos` 仅用 Hutch++ 估迹。

### 3. viz-all：前沿与谱图可视化

- **单一 (n,k) 对比图**：

  ```bash
  python -u scripts/rd_cli.py viz-all \
    --front notebooks/results/out_csv/stage1_pareto_n4_k4_raw.csv \
           notebooks/results/out_csv/stage1_pareto_n4_k4_canon.csv \
           notebooks/results/out_csv/pareto_front_n4_k4_ga.csv \
    --n 4 --k 4 --y-log --style ieee \
    --out-dir notebooks/results/figs
  ```

- **跨 (n,k) 聚合图**：不提供 `--n/--k` 时自动混合所有前沿 CSV，可额外指定熵曲线参数。

### 4. entropy：条带熵收敛

```bash
python -u scripts/rd_cli.py entropy \
  --bits 1110010... --k 3 \
  --n-min 3 --n-max 10 \
  --device cpu --out-dir notebooks/results/figs --logy
```

- 输入规则比特串，输出熵随 \(n\) 的收敛曲线。

### 5. motifs：膝点关键结构

```bash
python -u scripts/rd_cli.py motifs \
  --front notebooks/results/out_csv/stage1_pareto_n4_k4_canon.csv \
          notebooks/results/out_csv/pareto_front_n4_k4_ga.csv \
  --out-csv notebooks/results/out_csv/motifs \
  --out-dir notebooks/results/figs --style ieee --y-log
```

- 识别膝点两侧 motif 的出现频次，导出示例与汇总 CSV/图表。

### 6. motifs-explain：特征解释

```bash
python -u scripts/rd_cli.py motifs-explain \
  --examples notebooks/results/out_csv/motifs/motif_knee_examples.csv \
  --out-csv notebooks/results/out_csv/motifs \
  --out-dir notebooks/results/figs --topN 20 --tree-depth 3
```

- 使用逻辑回归、决策树等方法解释膝点 motif 贡献。

### 7. archetypes：结构原型扫描（可选）

```bash
python -u scripts/rd_cli.py archetypes \
  --n 4 --k 4 --types star,cycle,bip,shortloop \
  --top-m 8 --seed 0 \
  --out-csv notebooks/results/out_csv/archetypes \
  --out-dir notebooks/results/figs
```

- 需安装 `rules.structures` 依赖；自动绘制代表结构。

### 8. symmetry：对称性摘要

```bash
python -u scripts/rd_cli.py symmetry \
  --front notebooks/results/out_csv/stage1_pareto_n4_k4_canon.csv \
  --n 4 --k 4 --geo rot,ref,trans --state-perm --samples 6 \
  --out-csv notebooks/results/symmetry \
  --out-dir notebooks/results/figs --style ieee
```

- 输出对称性统计 CSV，并可生成代表模式图片。

---

## Python 批量评估示例

```python
import numpy as np

from rules.eval import (
    make_rule_matrix,
    bits_from_rule,
    evaluate_rules_batch,
)

# 以两个简单规则为例：先构造邻接矩阵，再编码为上三角比特串
def build_demo_rules(k: int) -> list[np.ndarray]:
    rule_a = make_rule_matrix(k, allowed_pairs=[(0, 1), (1, 2)], allow_self_loops=True)
    rule_b = make_rule_matrix(k, allowed_pairs=[(0, 2)], allow_self_loops=False)
    return [bits_from_rule(rule_a), bits_from_rule(rule_b)]

bits_list = build_demo_rules(k=3)

results = evaluate_rules_batch(
    n=4,
    k=3,
    bits_list=bits_list,
    device="cpu",          # 无 GPU 可直接使用 CPU
    use_lanczos=True,
    r_vals=3,
    power_iters=50,
    trace_mode="hutchpp",
    hutch_s=24,
)

for idx, stats in enumerate(results):
    print(f"Rule #{idx}: λ₁={stats['lambda_max']:.4f}, lower={stats['lower_bound']:.4f}")
```

---

## 输出目录约定

| 目录 | 内容 |
|------|------|
| `notebooks/results/out_csv/` | Stage-1、GA 前沿 CSV、motif 汇总、谱统计等数值结果 |
| `notebooks/results/figs/` | 前沿对比、熵曲线、motif/symmetry 可视化图像 |
| `notebooks/results/symmetry/` | 对称性统计表及索引 |
| `notebooks/results/out_csv/motifs/` | 膝点 motif 示例与全局统计 |
| `notebooks/results/out_csv/archetypes/` | 原型规则列表与结构特征 |

使用不同命令时可通过 `--out-csv`、`--out-dir` 参数覆盖默认位置，以便与实验管理系统集成。
