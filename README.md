# 规则-图案多目标搜索与膝点分析

本仓库围绕如下核心问题：在 $n\times n$ 的环面网格上，给定 $k$ 种状态与一组对称的邻接规则 $R$，如何在**使用最少规则**的前提下**最大化可行图案数量**，并识别多目标帕累托前沿上的**结构转折点（knee point）**。研究框架来自组合优化、图同态理论与谱方法，同时与《系统科学概论》系列讲义（`lecture1-Intro.pdf`、`lecture2-GoL和蚂蚁.pdf`、`lecture3-多主体.pdf`）中的“局部规则驱动的涌现”主题保持一致。

---

## 1. 背景与研究目标
- **科学动机**：借鉴 Conway 的 Game of Life，从局部相互作用出发探索宏观多样性；识别“少量规则即可产生大量模式”的机制。
- **优化目标**：同时最小化规则图 $R$ 中的边数 $|R|$，最大化环面图 $G_{n\times n}$ 到 $R$ 的图同态数量，即图案计数 $\mathrm{Pattern}(R)$。
- **复杂系统视角**：帕累托前沿的形态、谱半径的跃迁、以及规则结构（星状、近二部图、自环集）对应了系统的自组织阶段与临界转折。

---

## 2. 数学建模与核心公式
- **规则表示**：$R$ 以 $k\times k$ 对称邻接矩阵编码，编码长度 $L = k + \frac{k(k-1)}{2}$，由对角自环位与上三角边位组成。位串与矩阵互转在 `rules.eval.bits_from_rule` / `rule_from_bits` 中实现。
- **图案计数**：
  $$
  \mathrm{Pattern}(R) = |\mathrm{Hom}(G_{n\times n}, R)| = \operatorname{trace}(T_R^n)
  $$
  其中 $T_R$ 为按环列枚举得到的转移算子。
- **转移算子**：
  $$
  T_R = \sum_{i=1}^{n} P_i R P_i^\top,
  $$
  $P_i$ 为列偏移置换矩阵，`rules.ops.TransferOp` 通过稀疏隐式乘法在 CPU/GPU 上实现。
- **谱指标与估计**：`rules.spectrum` 提供幂迭代与 Lanczos 顶特征值估计；`estimate_trace_T_power_hutch` 与 `estimate_trace_T_power_hutchpp` 实现 Hutchinson / Hutch++ 随机迹估计：
  $$
  \operatorname{Tr}(T_R^n) \approx \frac{1}{s} \sum_{j=1}^s z_j^\top T_R^n z_j,
  $$
  其中 $z_j$ 为 Rademacher 或高斯向量。
- **上/下界**：`rules.eval._upper_bound_raw` 结合 Gershgorin、最大行和与谱信息；下界来自最大特征值 $\lambda_1$ 及其 $n$ 次幂。
- **对称化**：`rules.eval.canonical_bits` 对位串进行置换同构规范化，$k\le8$ 时枚举全排列，大规模时采用度数与自环启发式。

---

## 3. 仓库结构总览
| 路径 | 功能概述 |
| --- | --- |
| `rules/stage1_exact.py` | 小规模精确计算：行枚举 (`enumerate_ring_rows`)、转移矩阵构造 (`build_row_compat_matrix`)、精确迹计算 (`exact_trace_by_transfer`)。 |
| `rules/eval.py` | 批量评估入口 `evaluate_rules_batch`，包含行枚举加速 (`enumerate_ring_rows_fast`)、`TransferOp`、Hutch/Hutch++、Lanczos、上下界与精确兜底。 |
| `rules/ga.py` | 简化 NSGA-II 流程：初始化、交叉、变异、不可行修复、非支配排序、拥挤距离、`evaluate_rules_batch` 集成、CSV 输出。 |
| `rules/config.py` | `EvalConfig`、`GAConfig` 等配置数据类，集中管理设备、采样数、迭代次数、缓存与并行策略。 |
| `rules/spectrum.py` | 谱计算工具：幂迭代、Lanczos 顶值、迹估计。 |
| `rules/viz.py` | 数据可视化：帕累托散点、增长曲线、上下界叠加、二阶差分与 L-curve 膝点检测 (`detect_knee`)。 |
| `rules/utils_io.py` | CSV 读写与实验记录辅助。 |
| `rules/logging_setup.py` | 统一日志格式与第三方库降噪。 |
| `scripts/run_stage1.py` | 阶段一 CLI：手动配置 $(n,k)$ 与规则，输出精确计数。 |
| `scripts/run_ga.py` | NSGA-II CLI：批量跑 $(n,k)$ 组合，输出 CSV/图像并调用 `plot_pareto_from_csv`。 |

---

## 4. 算法阶段与实现细节
### 阶段一：精确枚举验证（`n\le5`, `k\le4`）
1. `make_rule_matrix` 构造对称规则矩阵。
2. `enumerate_ring_rows` 或 `enumerate_ring_rows_fast` 生成环列合法行。
3. `build_row_compat_matrix` 构建转移矩阵 $T$，`exact_trace_by_transfer` 计算 $\sum{trace}(T^n)$。
4. `scripts/run_stage1.py` 演示完整流程并打印精确图案数。

### 阶段二：谱估计与批量评估
- `evaluate_rules_batch` 接收规则位串列表，自动：
  - 规范化、行缓存（`RowsCacheLRU`）。
  - 构造 `TransferOp` 并选择 CPU/GPU。
  - 执行 Lanczos/幂迭代、Hutch/Hutch++、上下界与惩罚项。
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
  1. 帕累托散点（横轴 $|R|$，纵轴估计的模式数）。
  2. 增长曲线：累积最佳点随规则数变化，并叠加上下界及 knee 标记。
- `detect_knee` 结合二阶差分与 L-curve 曲率，输出稳定的膝点候选，并由 `_annotate_knee` 在图中标注。

---

## 5. 数据分析流程
1. **CSV 记录**：`rules/utils_io` 提供追加/初始化工具，GA 搜索自动写入 `out_csv/pareto_front_*.csv`。
2. **指标解析**：重点指标包括 `sum_lambda_powers`（经惩罚的模式估计）、`lambda_max`、`lower_bound_raw`/`upper_bound_raw`、`active_k`。
3. **膝点诊断**：
   - **二阶差分**：检测增量收益骤降的位置。
   - **L-curve 曲率**：在 log-log 空间拟合曲线并计算曲率最大点。
   - **投票融合**：`detect_knee` 内置二阶差分与 L-curve 两种候选，可在增长曲线中同时标注，辅助判断膝点置信度。
4. **结构对照**：结合 `canonical_bits` 输出的标准位串，可在后续分析中关联特定 motif。

---

## 6. 结构洞察摘要
| 结构类型 | 规则描述 | 膝点附近的作用 |
| --- | --- | --- |
| 星核结构 | 单节点连接多数其他节点 | 快速扩散，提高最大特征值 $\lambda_1$ |
| 近二部 + 弦边 | 偶环为主，少量跨部弦边 | 两步可达性最大化，膝点前沿常见 |
| 自环富集 | 大量对角自环 | 提升局部复用率，增加图案多样性 |
| 三角与小团 | 局部闭合路径 | 稳定主导模式，但过多会导致收益递减 |

- **凹凸性解读**：前沿上升段呈“向上凹”提示结构尚在扩展，出现“向下凹”时意味着进入收益递减区，即膝点附近。
- **谱-结构协同**：经验发现 `clustering`、`k-core`、`\lambda_1` 与谱间隙 `\lambda_1-\lambda_2` 与膝点形成高度相关。

---

## 7. 实验复现与快速上手
### 环境安装
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 下为 .venv\\Scripts\\activate
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
当满足 `n\le4`、`k\le3` 且行数不超过 `20_000` 时，函数会自动调用阶段一的精确计数进行交叉验证（`_maybe_exact_trace`）。

### 示例 3：NSGA-II 多目标搜索
```bash
python scripts/run_ga.py --nk "(6,3);(6,4)" --gens 12 --pop 32 --trace hutchpp --device cuda \
    --out-csv ./out_csv --out-fig ./out_fig
```
脚本会在 `out_csv/` 写入每代帕累托点，并在 `out_fig/` 输出散点 + 增长曲线，日志默认保存在控制台。

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

---

## 8. 配置要点与实验记录
- `EvalConfig`：
  - `device`、`use_lanczos`、`trace_mode` 控制谱估计策略。
  - `hutch_s_base`、`estimate_second_moment`、`upper_bounds` 管理采样自适应与上界策略。
  - `parallel_backend`、`max_streams`、`block_size_i/j` 管控 CPU/GPU 资源利用。
- `GAConfig`：
  - `pop_size`、`generations`、`p_cx`、`p_mut` 调整遗传算法搜索强度。
  - `lru_rows_capacity`、`batch_streams` 优化批量评估吞吐。
- **记录模板**：建议为每个 $(n,k)$ 保存：
  1. 帕累托点集合 `(|R|, \widehat{\mathrm{Pattern}})`。
  2. 典型规则位串 + 结构解读。
  3. 运行时间、硬件、采样参数、估计误差或上下界。
  4. 可视化图像（散点、增长曲线、典型图案示例）。

---

## 9. 复杂系统课程对照
| 讲义 | 核心主题 | 在本项目中的映射 |
| --- | --- | --- |
| Lecture 1 Intro | 涌现、混沌、自相似 | 规则-图案映射演示局部规则如何产生宏观复杂度；帕累托膝点对应相变。 |
| Lecture 2 GoL 和蚂蚁 | 局部规则 → 集体智能 | 网格-规则模型推广 Game of Life，探索“最小规则”带来的复杂行为。 |
| Lecture 3 多主体 | 自底向上的协同、同步 | 遗传算法视作智能体群体，协同搜索高多样性规则；谱指标对应协调程度。 |

---

## 10. 规模扩展与资源评估
- **时间复杂度**：约为 $O(\text{pop}\times\text{gen}\times C_{\text{TransferOp}})$，其中 `TransferOp` 对一次乘法的成本近似 $O(m n)$，行数 $m$ 随规则疏密指数级变化。
- **并行策略**：
  - GPU 上使用多 CUDA stream (`max_streams`) 管理批处理；
  - CPU 模式下可启用 `parallel_backend="mp"` 进行多进程行枚举；
  - `RowsCacheLRU` 在规则重复评估时显著降低枚举开销。
- **推荐规模**：
  - 精确枚举：$n\le5$, $k\le4$。
  - 谱估计：$n=6\sim8$, $k=4\sim6$（单 GPU/CPU）。
  - GA 搜索：`pop=200`, `gens=100` 在单 GPU 上可在数小时内完成 $n=8, k=4$ 实验。

---

## 11. 理论展望与未来工作
1. **学习辅助搜索**：使用已采集的 CSV 训练代理模型，指导候选规则的生成。
2. **强化学习 / 元启发式**：在 GA 框架上叠加策略梯度或自适应算子选择。
3. **更大规模**：扩展 `enumerate_ring_rows_fast` 到 64 bit 掩码，支持 $k>16$；或探索块分解与张量化。
4. **随机/自适应规则**：引入时间演化或概率性规则，研究图案多样性随动态变化的演化。
5. **跨领域比较**：将高多样性规则与生物网络、语言网络进行 motif 对照，验证谱指标的普适性。

---

## 12. 参考文献
- Hutchinson, M. F. (1989). *A stochastic estimator of the trace of the influence matrix*. Communications in Statistics.
- Avron, H., & Toledo, S. (2011). *Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix*. Journal of the ACM.
- Goldberg, D. E., & Deb, K. (2002). *A comparative analysis of selection schemes used in genetic algorithms* (NSGA-II).
- Gershgorin, S. (1931). *Über die Abgrenzung der Eigenwerte einer Matrix*.
- 图同态理论与谱图理论经典教材。
- ecsLab 《系统科学概论》 Lecture 1–3 教学资料。

