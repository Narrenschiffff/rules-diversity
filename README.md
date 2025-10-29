# rules-diversity

探索在环面网格上使用**最少规则**实现**最大模式多样性**的研究型代码库。本项目围绕 \(n\times n\) 网格、\(k\) 种状态以及对称规则矩阵 \(R\) 展开，重点支持以下目标：

- **多目标优化**：在最小化 \(|R|\) 的同时最大化可行模式数量（等价于图同态数量或 \(\mathrm{trace}(T^n)\)）。
- **跨尺度求解**：覆盖从小规模（精确计数）到中等规模（谱方法估计）再到大规模（启发式搜索）的完整流程。
- **可扩展实验框架**：集成 GPU/CPU 并行、智能搜索和可视化，以探索帕累托前沿及其结构规律。

## 研究问题概述
- **状态空间**：\(k\) 种状态、对称规则矩阵 \(R\in \{0,1\}^{k\times k}\) 表示允许的相邻状态组合。
- **模式空间**：在 \(n\times n\) 环面网格（周期边界）上构造图同态 \(G_{n,n} \to R\)。
- **优化目标**：在减少规则的同时最大化可行模式数量，形成 \((|R|, \text{pattern\_count})\) 的帕累托前沿。
- **挑战**：规则空间规模 \(2^{k(k-1)/2}\)，模式空间规模 \(k^{n^2}\)，需要结合组合计数、谱分析与启发式搜索。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `rules/stage1_exact.py` | 小规模精确计数：DFS 枚举合法行 + 转移矩阵乘幂求 \(\mathrm{trace}(T^n)\)。【F:rules/stage1_exact.py†L1-L50】 |
| `rules/eval.py` | 中等规模批量估计：行枚举、隐式转移算子 `TransferOp`、特征值估计（Power/Lanczos/Hutch++）及上下界。兼容 GPU。【F:rules/eval.py†L1-L170】 |
| `rules/ga.py` | 简化版 NSGA-II：规则位串编码、可行性修复、非支配排序、拥挤距离、批量评估与日志记录。【F:rules/ga.py†L1-L170】 |
| `rules/config.py` | `EvalConfig` / `GAConfig` 参数与默认值，包含 GPU 控制、采样次数、并行配置等。【F:rules/config.py†L1-L52】 |
| `rules/viz.py` | 帕累托前沿与增长曲线可视化，支持 knee detection（second diff / L-curve）。【F:rules/viz.py†L1-L120】 |
| `scripts/run_stage1.py` | Stage-1 演示脚本：示例规则的精确计数并写入日志。【F:scripts/run_stage1.py†L1-L22】 |
| `scripts/run_ga.py` | Stage-2 + GA 入口：批量运行 NSGA-II，输出 CSV 与可视化图像。【F:scripts/run_ga.py†L1-L44】 |

## 安装

项目基于 Python 3.9+，推荐使用 Conda 或 venv 创建隔离环境。

```bash
# 克隆并进入仓库
git clone <your_repo_url>
cd rules-diversity

# 安装依赖（可选 `-e` 便于本地开发）
pip install -e .
```

若使用 GPU，请确保已安装匹配版本的 PyTorch（`torch>=2.1`）。`EvalConfig.device` 会自动检测可用设备，亦可在命令行通过 `--device cpu` 强制运行在 CPU。

## 快速上手

### 1. 小规模精确计数（Stage 1）

```bash
python scripts/run_stage1.py
```

默认示例为 \(n=4, k=2\) 且禁止同色相邻，会在日志中输出 `trace(T^n)` 精确值。可根据需要修改脚本内的 `pairs` 或封装成 CLI。

### 2. 单批次规则评估（Stage 2 核心 API）

`rules.eval.evaluate_rules_batch` 接收一组规则位串，返回谱估计、上下界与可行性信息，适用于独立实验或与其它搜索器集成：

```python
import numpy as np
from rules.eval import evaluate_rules_batch, bits_from_rule, make_rule_matrix
from rules.config import EvalConfig

k, n = 4, 5
pairs = [(0,1), (1,2), (2,3)]
R = make_rule_matrix(k, pairs, allow_self_loops=True)
bits = bits_from_rule(R)[None, :]
res = evaluate_rules_batch(n=n, k=k, bits_batch=bits, config=EvalConfig(device="cpu"))
print(res[0]["sum_lambda_powers"], res[0]["upper_bound"])
```

该接口会在 `EvalConfig.enable_exact_crosscheck=True` 且状态空间较小时自动回退到 Stage 1，以验证估计质量。【F:rules/eval.py†L1-L170】【F:rules/config.py†L1-L52】

### 3. NSGA-II 规则搜索

```bash
python scripts/run_ga.py --nk "(6,3);(6,4)" --gens 12 --pop 32 --trace hutchpp
```

- 通过 `--nk` 传入一个或多个 `(n,k)` 组合；
- `--trace` 选择迹估计模式（`hutchpp`/`hutch`/`lanczos_sum`/`lam_only`）。

脚本会在 `./out_csv/` 保存逐代与前沿数据，并调用可视化工具生成帕累托散点与增长曲线。【F:scripts/run_ga.py†L1-L44】【F:rules/viz.py†L70-L120】

### 4. 可视化输出

GA 脚本默认调用 `plot_pareto_from_csv`，也可单独运行以叠加多个实验结果：

```python
from rules.viz import plot_pareto_from_csv
figs = plot_pareto_from_csv([
    "./out_csv/front_n6_k3.csv",
    "./out_csv/front_n6_k4.csv",
], out_dir="./out_fig", y_log=True)
print(figs)
```

该函数会生成两张图：Front-0 散点图与带上下界阴影的增长曲线，并在关键位置标注 knee 点。【F:rules/viz.py†L70-L120】

## 配置要点

`rules/config.py` 定义了计算的关键超参数，可在脚本或自定义实验中覆盖：

- **谱估计**：`power_iters`, `r_vals`, `trace_mode` 控制 Power/Lanczos/Hutch++ 的迭代次数与样本数。
- **方差控制**：`hutch_s_base` 与自适应阈值 (`small_m_threshold` 等) 决定 Hutch++ 采样规模。
- **上下界**：`upper_bounds` 可选择 `row_sum`、`gershgorin`、`power_mean` 等估计上界方式。
- **资源管理**：`device`、`block_size_i/j`、`max_streams` 控制 GPU/CPU 并行；`parallel_backend="mp"` 可在 CPU 上启用多进程批量评估。
- **GA 超参**：`GAConfig` 提供种群规模、代数、交叉/变异率、批量流数等设置，并在内部做合法性修复（至少一条边）。【F:rules/config.py†L1-L52】【F:rules/ga.py†L66-L120】

## 阶段化研究计划

1. **基础验证（阶段 1）**：\(n=2\sim4, k=2\sim3\)；使用 `stage1_exact` 精确计数验证理论预期并建立基准。
2. **算法开发（阶段 2）**：\(n=5\sim6, k=3\sim4\)；实现谱估计、NSGA-II 搜索，并以 CSV/图像记录帕累托前沿。
3. **规模扩展（阶段 3）**：\(n=7\sim10, k=4\sim6\)；利用 GPU 批量评估、探索大规模规则集结构，必要时调节 `EvalConfig` 的并行参数。
4. **理论分析（阶段 4）**：结合实验数据识别最优规则的结构特征，分析增长律、相变点以及与图谱理论的联系。

## 结果交付模板

针对每个 \((n,k)\) 建议整理如下内容：

- **结果摘要**：帕累托前沿点集 \{(|R|, pattern\_count)\}；具有代表性的规则结构（例如星型、完全图、二部图等）。
- **技术细节**：所用算法、关键超参、运行时间与硬件资源；若启用谱估计请提供误差或上下界信息。
- **可视化**：散点图、增长曲线及 knee 标注；必要时展示典型模式示例与规则矩阵。

## 复现与日志

- 统一使用 `rules.logging_setup.setup_logging()` 输出带时间戳的日志，便于对齐不同阶段的结果。
- `scripts/run_ga.py` 内部固定随机种子（0），保证同配置下结果可复现；可在脚本中自行扩展为实验网格。
- 建议将 `./out_csv` 与 `./out_fig` 纳入版本控制忽略列表，并按 `(n,k)`/日期组织实验档案。

## 下一步方向

- **机器学习辅助**：利用搜索历史训练评分模型，指导规则空间采样。
- **GPU/多机扩展**：`TransferOp` 已支持流批次，可进一步扩展到分布式框架以处理更大 \(n,k\)。
- **理论挖掘**：对最优规则的谱半径、度分布等指标做系统分析，寻找可推广的数学规律。

欢迎基于本框架继续深入探索最小规则下的最大模式多样性问题。
