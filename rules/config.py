from dataclasses import dataclass
from typing import Optional, Literal

UpperBoundStrategy = Literal[
    "none",         # 不计算上界（仅返回估计）
    "row_sum",      # m * (max_row_sum)^n
    "gershgorin",   # 与 row_sum 等价；保留用于学理与日志
    "power_mean"    # sqrt(m) * trace(T^{2n})^{1/2}
]

@dataclass
class EvalConfig:
    device: str = "cuda"
    use_lanczos: bool = True
    r_vals: int = 3
    power_iters: int = 60
    trace_mode: Literal["hutchpp","hutch","lanczos_sum","lam_only"] = "hutchpp"
    hutch_s_base: int = 24
    penalty_alpha: float = 1.5
    block_size_i: Optional[int] = None
    block_size_j: Optional[int] = None
    lru_rows_capacity: int = 128
    max_streams: int = 2
    enable_exact_crosscheck: bool = True

    # —— 新增：上界策略（可多选并输出） ——
    upper_bounds: tuple[UpperBoundStrategy, ...] = ("row_sum","power_mean")

    # —— 新增：二阶矩估计开关（给 power_mean 用） ——
    estimate_second_moment: bool = True
    hutch_s_base_2nd: int = 24   # 估计 trace(T^{2n}) 的采样基数

    # —— 并行外层（仅 CPU 时生效；CUDA 建议关闭） ——
    parallel_backend: Literal["none","mp"] = "none"
    num_workers: int = 0  # 0/1=不并行；>1 启用多进程

    # 自适应采样阈值
    small_m_threshold: int = 2_000
    large_m_threshold: int = 50_000
    min_s: int = 12
    max_s: int = 64

@dataclass
class GAConfig:
    pop_size: int = 24
    generations: int = 10
    p_mut: float = 0.08
    p_cx: float = 0.85
    elite_keep: int = 6
    device: Optional[str] = None         # None -> 自动探测
    # 下列字段会透传给 EvalConfig（若不指定则用 EvalConfig 默认）
    use_lanczos: Optional[bool] = None
    r_vals: Optional[int] = None
    power_iters: Optional[int] = None
    trace_mode: Optional[str] = None
    hutch_s_base: Optional[int] = None
    lru_rows_capacity: int = 128
    batch_streams: int = 2
