# 规则-图案多目标搜索与膝点分析

本仓库围绕如下核心问题：在 $n\times n$ 的环面网格上，给定 $k$ 种状态与一组对称的邻接规则 $R$，如何在**使用最少规则**的前提下**最大化可行图案数量**，并识别多目标帕累托前沿上的**结构转折点（knee point）**。研究框架来自组合优化、图同态理论与谱方法，同时与《系统科学概论》系列讲义（`lecture1-Intro.pdf`、`lecture2-GoL和蚂蚁.pdf`、`lecture3-多主体.pdf`）中的“局部规则驱动的涌现”主题保持一致。

> **快速导览**
> - 若想直接跑一遍“精确 + 谱估计 + 对称处理 + 边界开关 + 缓存”统一流水线，请跳转到 [§7.1 示例 1](#示例-1精确计数) 与 [§7.5 示例 5](#示例-5统一流水线-CLI精确--谱估计--缓存)。
> - 如果要改代码或做二次开发，可以从 [§3 仓库结构总览](#3-仓库结构总览) 与 [§4 算法阶段与实现细节](#4-算法阶段与实现细节) 定位模块，再结合 [§5 数据分析流程](#5-数据分析流程) 了解输出结果的字段含义。
> - 需要自定义规则/配置时，优先使用 `scripts/run_pipeline.py` 提供的配置文件 + CLI 覆盖模式；命令行参数详见 [§7 示例](#7-实验复现与快速上手)。

---

## 1. 背景与研究目标
- **科学动机**：借鉴 Conway 的 Game of Life，从局部相互作用出发探索宏观多样性；识别“少量规则即可产生大量模式”的机制。  
- **优化目标**：同时最小化规则图 $R$ 中的边数 $|R|$，最大化环面图 $G_{n\times n}$ 到 $R$ 的图同态数量，即图案计数 $\mathrm{Pattern}(R)$。  
- **复杂系统视角**：帕累托前沿的形态、谱半径的跃迁、以及规则结构（星状、近二部图、自环集）对应了系统的自组织阶段与临界转折。

---

## 2. 数学建模与核心公式

- **规则表示**： $R$ 以 $k\times k$ 对称邻接矩阵编码，编码长度

  $L = k + \frac{k(k-1)}{2}$

  由对角自环位与上三角边位组成。位串与矩阵互转在 `rules.eval.bits_from_rule` / `rule_from_bits` 中实现。

- **图案计数**：

  $\mathrm{Pattern}(R) = \left|\mathrm{Hom}\bigl(G_{n\times n} \, R\bigr)\right| = \mathrm{Tr}\left(T_R^{n}\right)$

- **转移算子**：

  $T_R = \sum_{i=1}^{n} P_i  R  P_i^{\top}$

  其中 $P_i$ 为列偏移置换矩阵，`rules.ops.TransferOp` 通过稀疏隐式乘法在 CPU/GPU 上实现。

- **谱指标与估计**：`rules.spectrum` 提供幂迭代与 Lanczos 顶特征值估计；`estimate_trace_T_power_hutch` 与 `estimate_trace_T_power_hutchpp` 实现 Hutchinson / Hutch++ 随机迹估计：

  $\mathrm{Tr}\left(T_R^{n}\right) \approx \frac{1}{s}\sum_{j=1}^{s} z_j^{\top} T_R^{n} z_j$

  其中 $z_j$ 为 Rademacher 或高斯向量。

- **上/下界**：`rules.eval._upper_bound_raw` 结合 Gershgorin、最大行和与谱信息；下界来自最大特征值 $\lambda_1$ 及其 $n$ 次幂。

- **对称化**：`rules.eval.canonical_bits` 对位串进行置换同构规范化，$k\le 8$ 时枚举全排列，大规模时采用度数与自环启发式。
- **谱-结构剖析**：`evaluate_rules_batch` 在构造规则图后会额外输出度矩阵、无向/归一化拉普拉斯、邻接谱间隙、代数连通度与平均聚类系数，便于后续结构对齐与解释；在 `perm+swap` 模式下，这些结构特征基于“仅 perm”规范化后的规则图计算，以保持与未合并数据的可比性（行枚举仍使用合并后的图以保证速度）。

### 功能开关总览（CLI 与 API 通用）
| 需求 | CLI 入口 | 配置字段 | 说明 |
| --- | --- | --- | --- |
| 置换对称 / 交换对称 | `--sym-mode {none,perm,perm+swap}` | `sym_mode` | `perm` 走规则位串规范化，`perm+swap` 额外合并邻接等价节点；`none` 不做对称处理。 |
| 边界模式 | `--boundary {torus,open}` | `boundary` | `torus` 为循环边界；`open` 为开放边界（行枚举不回绕，支持谱估计与精确计数，规模大时行数爆炸仍需谨慎）。 |
| 精确计算开关 | `--exact / --no-exact` | `use_exact` | 若开启，且行数未超过 `exact_rows_cap`，调用行枚举 + 转移矩阵精确计数。 |
| 谱估计开关 | `--spectral / --no-spectral` | `use_spectral` | 启用/关闭 Hutch/Hutch++ 迹估计与 Lanczos 顶值估计。 |
| 阈值联动 | `--exact-rows-cap` | `exact_rows_cap` | 行数超过阈值即跳过精确计算，以谱估计为主。 |
| 缓存 | `--cache-dir / --no-cache` | `cache_dir` | 规则位串 + 有效状态数 + 边界 + 对称模式 + `n` 共同构成缓存键；命中后直接复用计算结果。若早期缓存缺少 archetype 字段，可用 `--refresh-cache` 触发重算并补写。 |
| rule_count 覆盖 | `--strict-rulecount-cover / --no-strict-rulecount-cover` | `strict_rulecount_cover` | 默认开启：每代前沿输出会为每个 `|R|≤L`（可用 `rulecount_cover_max` 上限截断）补齐至少一条 `is_front0=1` 记录；关闭可提高搜索效率但可能出现缺失桶。 |
| 设备选择 | `--device cpu|cuda` | `device` | `rules.eval.TransferOp` 支持 CPU/GPU，默认自动检测。 |

> 若使用 Python API，可直接向 `evaluate_rules_batch` 传入上述同名参数；`scripts/run_pipeline.py` 将 CLI/配置映射到同一接口。

---

## 3. 仓库结构总览
| 路径 | 功能概述 |
| --- | --- |
| `rules/stage1_exact.py` | 小规模精确计算：行枚举 (`enumerate_ring_rows`)、转移矩阵构造 (`build_row_compat_matrix`)、精确迹计算 (`exact_trace_by_transfer`)。 |
| `rules/eval.py` | 批量评估入口 `evaluate_rules_batch`，包含行枚举加速 (`enumerate_ring_rows_fast`)、`TransferOp`、Hutch/Hutch++、Lanczos、上下界与精确兜底。 |
| `rules/ga.py` | 简化 NSGA-II 流程：初始化、交叉、变异、不可行修复、非支配排序、拥挤距离、`evaluate_rules_batch` 集成、CSV 输出。 |
| `rules/config.py` | `EvalConfig`、`GAConfig` 等配置数据类，集中管理设备、采样数、迭代次数、缓存与并行策略。 |
| `rules/spectrum.py` | 谱计算工具：幂迭代、Lanczos 顶值、迹估计。 |
| `rules/viz.py` | 数据可视化：帕累托散点、增长曲线、上下界叠加、二阶差分与 L-curve 膝点检测 (`detect_knee`)，支持按 `sym_mode` 过滤与运行摘要 (`summarize_runs`)。 |
| `rules/utils_io.py` | CSV 读写与实验记录辅助。 |
| `rules/logging_setup.py` | 统一日志格式与第三方库降噪。 |
| `scripts/run_stage1.py` | 阶段一 CLI：手动配置 $(n,k)$ 与规则，输出精确计数。 |
| `scripts/run_ga.py` | NSGA-II CLI：批量跑 $(n,k)$ 组合，输出 CSV/图像并调用 `plot_pareto_from_csv`。 |

---

## 4. 算法阶段与实现细节

### 阶段一：精确枚举验证（`n\le5`, `k\le4`）
1. `make_rule_matrix` 构造对称规则矩阵。  
2. `enumerate_ring_rows` 或 `enumerate_ring_rows_fast` 生成环列合法行。  
3. `build_row_compat_matrix` 构建转移矩阵 $T$，`exact_trace_by_transfer` 计算 $\mathrm{Tr}(T^{n})$。  
4. `scripts/run_stage1.py` 演示完整流程并打印精确图案数。

### 阶段二：谱估计与批量评估
- `evaluate_rules_batch` 接收规则位串列表，自动：
  - 规范化、行缓存（`RowsCacheLRU`）。
  - 构造 `TransferOp` 并选择 CPU/GPU。
  - 执行 Lanczos/幂迭代、Hutch/Hutch++、上下界与惩罚项。
  - 计算规则图的度序列与拉普拉斯（含归一化形式）、邻接谱间隙（`adj_lambda1/2`）、代数连通度（`laplacian_alg_conn`）、平均聚类系数等结构谱指标，并返回结构识别结果（若启用 `rules.structures`）。
  - 小规模时调用阶段一获得 `exact_trace`/`exact_rows_m`。
- 输出字段包含 `rule_count`、`sum_lambda_powers`（惩罚后的估计模式数）、`lower_bound`、`upper_bound`、`rows_m`、`active_k` 等，便于多目标排序。

### 阶段三：多目标遗传搜索
- `rules.ga` 中的 NSGA-II 结合：  
  - 位串初始化倾向稀疏结构并执行可行修复 `_ensure_minimal_feasible`。  
  - `canonical_bits` 保持等价规则唯一性。  
  - 非支配排序与拥挤距离（`nondominated_sort`、`crowding_distance`）。  
  - 逐代调用 `evaluate_rules_batch`，记录 CSV 并可选绘图。  
  - CLI `scripts/run_ga.py` 支持 `--nk`、`--gens`、`--pop`、`--trace`、`--device`、`--out-csv`、`--out-fig` 等参数。

### 阶段四：数据分析与膝点识别
- `rules.viz.plot_pareto_from_csv` 汇总多次实验，生成：
  1) 帕累托散点（横轴 $|R|$，纵轴估计的模式数）。
  2) 增长曲线：累积最佳点随规则数变化，并叠加上下界及 knee 标记。
- `rules.viz.plot_all(front_paths, sym_filter="perm")`：在包含多种对称模式的 CSV 中，仅保留指定 `sym_mode` 进行绘制。
- `rules.viz.summarize_runs(csvs, sym_filter=None)`：快速统计每个 CSV 的对称模式计数、`active_k`/`active_k_raw` 范围及边界模式，便于排查数据质量。
- `detect_knee` 结合二阶差分与 L-curve 曲率，输出稳定的膝点候选，并由 `_annotate_knee` 在图中标注。

---

## 5. 数据分析流程
1. **CSV 记录**：`rules/utils_io` 提供追加/初始化工具，GA 搜索自动写入 `out_csv/pareto_front_*.csv`。
2. **指标解析**：重点指标包括 `sum_lambda_powers`（经惩罚的模式估计）、`lambda_max`、`lower_bound_raw`/`upper_bound_raw`、`active_k`。
3. **膝点诊断**：
   - **二阶差分**：检测增量收益骤降的位置。
   - **L-curve 曲率**：在 log–log 空间拟合曲线并计算曲率最大点。
   - **投票融合**：`detect_knee` 内置二阶差分与 L-curve 两种候选，可在增长曲线中同时标注，辅助判断膝点置信度。
   - **对称模式过滤**：若同一目录下存在 `sym_mode` 不同的运行，绘制/汇总时建议传入 `sym_filter` 聚焦某一模式，避免不同有效状态数混杂造成的解读偏差。
4. **结构对照**：结合 `canonical_bits` 输出的标准位串，可在后续分析中关联特定 motif。

### 输出文件与字段对照
- **流水线 / GA CSV**
  - `rule_bits_raw` / `rule_bits_canon`：原始与对称化后的位串。
  - `rows_m` / `active_k`：合法行数量与有效状态数（对称模式下会发生折算）。
  - `exact_Z` / `exact_note`：精确计数结果及跳过原因。
  - `sum_lambda_powers` / `lambda_max` / `adj_spectral_gap`：谱估计核心指标。
  - `lower_bound` / `upper_bound`：估计上下界；`upper_bound_note` 说明使用的上界类型。
  - `cache_key`：包含规则、边界、对称、规模的缓存键，便于快速定位缓存命中情况。
  - `rule_count_sym`：在 `perm+swap` 合并后的 |R|，并行保留 `rule_count`（合并前）以保证与仅 `perm` 路径的桶划分一致；GA front CSV 现保证每个 `rule_count` 至少有一条 `is_front0=1` 记录，避免按 |R| 聚合时出现空桶。
- **JSONL 摘要**（流水线）
  - 行内包含与 CSV 相同的字段，便于流式处理。
  - 若启用 exact/spectral 双开关，JSONL 中会包含两种模式的对照（精确值为 baseline）。
- **日志**
  - 统一通过 `rules.logging_setup.setup_logging` 控制，CLI 默认 INFO 级别。
  - 运行中会输出对称模式、缓存命中率、行枚举规模、估计采样数、边界模式等关键信息。

### 运行上下文与续算（GA / pipeline）
- `rules.runtime.RunContext` 统一封装日志（stdout + 文件）、心跳与 checkpoint：每次运行会在对应 `run_dir` 下生成 `{run_tag}.log`、`heartbeat.txt` 与 `checkpoint.json`（内含 RNG 状态）。
- `scripts/run_pipeline.py` 默认检测 `summary.jsonl/csv` 并跳过已写入的规则（可用 `--no-resume` 禁用）；同一 `run_tag` 会在 JSONL 中即时追加，再在结束时重写 CSV/JSONL 以去重。
- `rules.ga.ga_search_with_batch` 现在在 CSV 中记录 `generation` 列，并写入 checkpoint。重复使用相同 `run_tag` 时会从最新一代继续（保留上一代种群）；若要重新开始，请更换 `run_tag` 或清理输出目录。
- 缓存/输出目录复用：`cache_dir`（光谱缓存）与 `run_dir`（日志+checkpoint）均可在配置/CLI 中显式设置，避免跨实验混淆的同时减少重复计算。
- 查看进度：运行时可直接 `tail -f run_dir/{run_tag}.log` 或检查 `heartbeat.txt`；中断后再次运行相同配置即可续写，无需手动拼接 CSV/JSONL。

### 可视化与数据诊断示例
```bash
# 仅绘制 perm 对称模式的 GA 与阶段一结果，并输出三张图至 out_fig/
python - <<'PY'
from pathlib import Path
from rules import viz

csvs = sorted(str(p) for p in Path('out_csv').glob('pareto_front_*.csv'))
viz.plot_all(csvs, n=2, k=2, out_dir='out_fig', sym_filter='perm')
print(viz.summarize_runs(csvs, sym_filter='perm'))
PY
```
#### open / torus 混合增长曲线示例
在 open 边界下仅写入精确计数（禁用谱估计），但 CSV 字段与 torus 结果兼容，可直接用 `viz.plot_all` 生成同一张增长曲线：
```python
from pathlib import Path
from rules import viz

torus = sorted(str(p) for p in Path("results/torus_csv").glob("*.csv"))
open_ = sorted(str(p) for p in Path("results/open_csv").glob("*.csv"))
fronts = torus + open_
viz.plot_all(fronts, n=4, k=3, out_dir="results/fig_open_torus", sym_filter="perm")
```
> open CSV 仅包含精确值，但列名与 torus 对齐；`plot_all` 会自动选择 `trace_estimate/sum_lambda_powers` 或 `exact_Z` 作为纵轴，因此混合输入也能绘制增长曲线与膝点。

---

## 6. 结构洞察摘要
| 结构类型 | 规则描述 | 膝点附近的作用 |
| --- | --- | --- |
| 星核结构 | 单节点连接多数其他节点 | 快速扩散，提高最大特征值 $\lambda_1$ |
| 近二部 + 弦边 | 偶环为主，少量跨部弦边 | 两步可达性最大化，膝点前沿常见 |
| 自环富集 | 大量对角自环 | 提升局部复用率，增加图案多样性 |
| 三角与小团 | 局部闭合路径 | 稳定主导模式，但过多会导致收益递减 |

- **凹凸性解读**：前沿上升段呈“向上凹”提示结构尚在扩展，出现“向下凹”时意味着进入收益递减区，即膝点附近。  
- **谱-结构协同**：经验发现 `clustering`、`k-core`、$\lambda_1$ 与谱间隙 $\lambda_1-\lambda_2$ 与膝点形成高度相关。

### 谱指标解读与使用
- **度矩阵 $D$**：对角为节点度数（自环仅计入对角），决定拉普拉斯的尺度。
- **无向拉普拉斯 $L=D-A$**：其第二小特征值 $\lambda_2(L)$（代数连通度）越大，说明图越稳健、分量越少。
- **归一化拉普拉斯 $\mathcal{L}=I-D^{-1/2} A D^{-1/2}$**：抑制度差的影响；$\lambda_2(\mathcal{L})$ 接近 0 表示近似分离、接近 2 表示存在强二分结构。
- **邻接谱间隙与顶特征值**：`adj_lambda1/2` 与 `adj_spectral_gap` 捕捉扩散与主导模式强度，常与膝点跃迁相关。
- **聚类系数**：平均局部三角密度，区分树状/低闭合（近二部）与高闭合（团状）的规则图。
- **结构识别**：若 `rules/structures` 可用，`archetype_hits` 与 `archetype_tags` 会标记星核、自环富集、近二分等模式，可与上述谱指标交叉验证；`archetype_*_merged` 则对应 perm+swap 合并后统计。

---

## 7. 实验复现与快速上手

### 调用方式索引
- **最小样例/教学演示**：`scripts/run_stage1.py`（或安装后 `rules-stage1`），走精确枚举路径，便于理解规则-图案映射。
- **批量评估 / 研究复现**：`scripts/run_pipeline.py`（或安装后 `rules-pipeline`），统一支持对称性、边界、精确/谱估计双开关与缓存，是推荐入口，默认按 `run_tag` 自动续算（如需从头运行可加 `--no-resume`，或指定 `--seed` 固定 RNG）。
- **多目标遗传搜索**：`scripts/run_ga.py`（或安装后 `rules-run-ga`），适合同时探索多个 $(n,k)$ 组合的帕累托前沿。
- **可视化与 knee 点分析**：`rules.viz.plot_pareto_from_csv` / `detect_knee`，可在 GA 输出或自定义 CSV 上复用。
- **API 复用**：直接调用 `evaluate_rules_batch`、`make_rule_matrix`、`bits_from_rule` 等函数，灵活嵌入其他优化器或实验框架。

### 环境安装
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 下为 .venv\Scripts\activate
pip install -e .
```
或使用需求文件：
```bash
pip install -r requirements.txt
```
> **GPU 提示**：`torch>=2.1` 可选择 CUDA 版本；若未安装 GPU 版，评估会自动回退到 CPU。

### 示例 1：精确计数
```bash
python scripts/run_stage1.py
```
根据需要编辑脚本内的 `(n, k)` 与允许边集合 `pairs`。

### 示例 2：批量评估 API（与当前函数签名一致）
```python
import numpy as np
from rules.eval import evaluate_rules_batch, bits_from_rule, make_rule_matrix

n, k = 5, 4
pairs = [(0, 1), (1, 2), (2, 3)]
R = make_rule_matrix(k, pairs, allow_self_loops=True)
bits = bits_from_rule(R)
reports = evaluate_rules_batch(
    n=n,
    k=k,
    bits_list=[bits],  # 传入规则位串列表
    device="cpu",
    use_lanczos=True,
    trace_mode="hutchpp",
    hutch_s=24,
)
print(reports[0]["sum_lambda_powers"], reports[0]["upper_bound"])
```
当满足 `n<=4`、`k<=3` 且行数不超过 `20_000` 时，函数会自动调用阶段一获得精确计数（`_maybe_exact_trace`）。

### 示例 3：NSGA-II 多目标搜索
```bash
python scripts/run_ga.py --nk "(6,3);(6,4)" --gens 12 --pop 32 --trace hutchpp --device cuda \
    --out-csv ./out_csv --out-fig ./out_fig
```
脚本会在 `out_csv/` 写入每代帕累托点，并在 `out_fig/` 输出散点 + 增长曲线，日志默认保存在控制台。
> `--boundary open` 也支持谱估计，但行枚举随规模增长极快，建议在小型 $(n,k)$ 上启用或结合缓存使用。输出 CSV 会同时写出主 `archetype_*`（perm 统计）与合并后 `archetype_*_merged`（perm+swap 合并结构）两列。

### 示例 4：多平台命令行
```bash
# Bash
python scripts/run_ga.py --nk "(8,4)" --pop 200 --gens 100 --trace hutchpp --out-csv ./out_csv --out-fig ./out_fig
python - <<'PY'
from pathlib import Path
from rules.viz import plot_pareto_from_csv

out_dir = Path("./out_fig")
out_dir.mkdir(exist_ok=True, parents=True)
csv_paths = sorted(str(p) for p in Path("./out_csv").glob("pareto_front_*.csv"))
if csv_paths:
    plot_pareto_from_csv(csv_paths, out_dir=str(out_dir), y_log=True)
PY
```
```powershell
# PowerShell
python .\scripts\run_ga.py -nk "(8,4)" -pop 200 -gens 100 -trace hutchpp -out-csv .\out_csv -out-fig .\out_fig
python -c "from pathlib import Path; from rules.viz import plot_pareto_from_csv; out_dir = Path('.\\out_fig'); out_dir.mkdir(exist_ok=True, parents=True); csv_paths = sorted(str(p) for p in Path('.\\out_csv').glob('pareto_front_*.csv')); csv_paths and plot_pareto_from_csv(csv_paths, out_dir=str(out_dir), y_log=True)"
```
> 若使用自定义分析脚本，请先调用 `rules.logging_setup.setup_logging()` 统一日志格式；上例演示如何跨平台批量汇总 GA 输出。

### 示例 5：统一流水线 CLI（精确 + 谱估计 + 缓存）
新脚本 `scripts/run_pipeline.py` 封装了精确计数、谱估计、对称规范化、边界模式与缓存，支持 JSON/YAML 配置与进度条，并默认接入 `rules.logging_setup` 统一日志输出：

```yaml
# config.yml
n: 5
k: 4
boundary: torus          # 也支持 open（开放边界，谱/精确都可用；规模大时行数爆炸需谨慎）
sym_mode: perm
use_exact: true          # rows_m 超过 exact_rows_cap 会自动跳过
use_spectral: true
rules:
  - name: cycle
    pairs: [[0,1],[1,2],[2,3],[3,0]]
    allow_self_loops: true
  - name: custom_bits
    bits: "111100100"  # 长度需匹配 k 的上三角编码
```

运行命令（CLI 参数可覆盖配置文件）：
```bash
python scripts/run_pipeline.py --config config.yml --run-tag demo --out-dir ./pipeline_out \
    --trace-mode hutchpp --lanczos-r 4 --heartbeat 5
```
输出目录：`./pipeline_out/demo/`
- `summary.csv` / `summary.jsonl`：包含规则位串（raw/canon）、精确计数 `exact_Z`、行数 `rows_m`、谱估计核心字段 (`sum_lambda_powers`、`lambda_max`、上下界等)、缓存键 `cache_key`、日志备注。
- `cache/`：按规则 + 边界生成的精确计数 JSON 缓存，便于二次复用；谱估计结果复用 `rules.eval` 内置缓存（`RULES_EVAL_CACHE` 或 `--cache-dir`）。
- 进度条：默认使用 `tqdm`，如未安装则每隔 `--heartbeat` 秒打印一次进度日志。

### CLI 快速参考（安装后）
通过 `pip install -e .` 安装后会暴露以下命令行工具（对应 `pyproject.toml` 的 `project.scripts`）：
- `rules-run-ga` / `rules-stage1` → 统一 CLI `scripts/rd_cli.py`，包含穷举 `stage1`、GA 搜索、可视化、motif 分析、对称性扫描等子命令。
- `rules-pipeline` → 统一流水线 `scripts/run_pipeline.py`，支持配置文件 + CLI 覆盖。

示例：
```bash
# 小规模快速验证：仅走精确路径，结果写入 ./pipeline_out/unit
rules-pipeline --n 2 --k 2 --rule-bits 111 --no-spectral --run-tag unit \
  --out-dir ./pipeline_out --cache-dir ./pipeline_out/cache
```

### 测试与质量检查
- 运行全量测试：`pytest`
- 新增了流水线 CLI 的单测，覆盖规则解析（位串 / 边集合）与“仅精确计算”分支的落盘行为，便于回归验证工程化接口。

---

## 8. 配置要点与实验记录
- `EvalConfig`：  
  - `device`、`use_lanczos`、`trace_mode` 控制谱估计策略。  
  - `hutch_s_base`、`estimate_second_moment`、`upper_bounds` 管理采样自适应与上界策略。  
  - `parallel_backend`、`max_streams`、`block_size_i/j` 管控 CPU/GPU 资源利用。  
- `GAConfig`：
  - `pop_size`、`generations`、`p_cx`、`p_mut` 调整遗传算法搜索强度。
  - `lru_rows_capacity`、`batch_streams` 优化批量评估吞吐。
- **评估缓存**：默认写入 `~/.cache/rules-diversity/eval`（可通过环境变量 `RULES_EVAL_CACHE` 或 CLI `--cache-dir` 覆盖，`--no-cache` 禁用）。
  - **键设计**：对称化规则位串 + 有效状态数 `active_k` + 边界模式 + 对称选项 + 棋盘规模 `n`，避免跨设置污染。
  - **命中策略**：`evaluate_rules_batch` 在执行谱估计/精确计数前优先查找缓存命中，直接返回已有的转移算子谱估计或精确值。
  - **失效条件**：元数据记录来源与版本号（`RULES_EVAL_CACHE_VERSION`）；修改版本或删除目录即可强制重新计算。
- **记录模板**：建议为每个 $(n,k)$ 保存：
  1) 帕累托点集合 $(|R|,\, \widehat{\mathrm{Pattern}})$；
  2) 典型规则位串 + 结构解读；
  3) 运行时间、硬件、采样参数、估计误差或上下界；  
  4) 可视化图像（散点、增长曲线、典型图案示例）。

---

## 9. 复杂系统课程对照
| 讲义 | 核心主题 | 在本项目中的映射 |
| --- | --- | --- |
| Lecture 1 Intro | 涌现、混沌、自相似 | 规则-图案映射演示局部规则如何产生宏观复杂度；帕累托膝点对应相变。 |
| Lecture 2 GoL 和蚂蚁 | 局部规则 → 集体智能 | 网格-规则模型推广 Game of Life，探索“最小规则”带来的复杂行为。 |
| Lecture 3 多主体 | 自底向上的协同、同步 | 遗传算法视作智能体群体，协同搜索高多样性规则；谱指标对应协调程度。 |

---

## 10. 规模扩展与资源评估
- **时间复杂度**：约为 $O(\text{pop}\times \text{gen}\times C_{\text{TransferOp}})$，其中 `TransferOp` 对一次乘法的成本近似 $O(mn)$，行数 $m$ 随规则疏密指数级变化。  
- **并行策略**：  
  - GPU 上使用多 CUDA stream (`max_streams`) 管理批处理；  
  - CPU 模式下可启用 `parallel_backend="mp"` 多进程行枚举；  
  - `RowsCacheLRU` 在规则重复评估时显著降低枚举开销。  
- **推荐规模**：  
  - 精确枚举： $n\le5$, $k\le4$。  
  - 谱估计： $n=6\sim8$, $k=4\sim6$（单 GPU/CPU）。  
  - GA 搜索：`pop=200`, `gens=100` 可完成 $n=8, k=4$ 级别实验。

---

## 11. 理论展望与未来工作
1. **学习辅助搜索**：用 CSV 训练代理模型，指导候选规则生成。  
2. **强化学习 / 元启发式**：在 GA 框架上叠加策略梯度或自适应算子选择。  
3. **更大规模**：扩展 `enumerate_ring_rows_fast` 到 64-bit 掩码，支持 $k>16$；或探索块分解与张量化。  
4. **随机/自适应规则**：引入时间演化或概率性规则，研究图案多样性随动态变化的演化。  
5. **跨领域比较**：将高多样性规则与生物网络、语言网络进行 motif 对照，验证谱指标的普适性。

---

## 12. 参考文献
- Hutchinson, M. F. (1989). *A stochastic estimator of the trace of the influence matrix*. Communications in Statistics.  
- Avron, H., & Toledo, S. (2011). *Randomized algorithms for estimating the trace of an implicit SPSD matrix*. JACM.  
- Deb, K. et al. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*. IEEE TEC.  
- Gershgorin, S. (1931). *Über die Abgrenzung der Eigenwerte einer Matrix*.  
- 图同态理论与谱图理论经典教材。  
- ecsLab 《系统科学概论》 Lecture 1–3 教学资料。
