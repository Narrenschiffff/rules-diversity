# -*- coding: utf-8 -*-
"""
rules/config.py
全局轻量配置：默认风格、设备、输出目录等。
"""

from __future__ import annotations
from pathlib import Path
import os

from typing import Optional

# 视觉风格（viz.apply_style 支持的枚举）
DEFAULT_STYLE = os.getenv("RULES_STYLE", "ieee")

# 设备（eval.TransferOp 会再次判断）
DEFAULT_DEVICE = os.getenv("RULES_DEVICE", "cuda")

# 边界模式（torus|open）
BOUNDARY_MODE = os.getenv("RULES_BOUNDARY", "torus")

# 结果根目录（示例；各 CLI 可覆盖）
RESULTS_ROOT = Path(os.getenv("RULES_RESULTS_ROOT", "./notebooks/results")).resolve()
OUT_CSV_DEFAULT = RESULTS_ROOT / "out_csv"
OUT_FIG_DEFAULT = RESULTS_ROOT / "figs"

# 评估缓存目录（json 级别，可自定义）
EVAL_CACHE_DIR = Path(os.getenv("RULES_EVAL_CACHE", "~/.cache/rules-diversity/eval")).expanduser().resolve()
EVAL_CACHE_VERSION = os.getenv("RULES_EVAL_CACHE_VERSION", "v1")

# LRU 行缓存容量（eval.RowsCacheLRU）
ROWS_LRU_CAPACITY = int(os.getenv("RULES_ROWS_LRU", "128"))

# 计算相关默认值（可被 CLI 覆盖）
LANCZOS_R = int(os.getenv("RULES_LANCZOS_R", "3"))
POWER_ITERS = int(os.getenv("RULES_POWER_ITERS", "50"))
HUTCH_S = int(os.getenv("RULES_HUTCH_S", "24"))
TRACE_MODE = os.getenv("RULES_TRACE_MODE", "hutchpp")  # hutch|hutchpp|lanczos_sum

# 精确/谱估计开关与阈值
ENABLE_EXACT = os.getenv("RULES_ENABLE_EXACT", "1") != "0"
ENABLE_SPECTRAL = os.getenv("RULES_ENABLE_SPECTRAL", "1") != "0"
EXACT_THRESHOLD = os.getenv("RULES_EXACT_THRESHOLD", "nk<=12")

# 目标函数 / 惩罚：logZ（raw）与 logZ/(penalty_factor)（penalized）可切换；可通过 CLI 或 GAConfig 覆盖
OBJECTIVE_MODE = os.getenv("RULES_OBJECTIVE_MODE", "logZ_per_penalty")
OBJECTIVE_USE_PENALTY = os.getenv("RULES_OBJECTIVE_USE_PENALTY", "1") != "0"
# 惩罚模式：n、n*rule_count、n*rows_m
PENALTY_MODE = os.getenv("RULES_PENALTY_MODE", "n_times_rows_m")

# 对称性分析默认
SYM_GEO_OPS = os.getenv("RULES_SYM_GEO_OPS", "rot,ref,trans")
SYM_ENUM_LIMIT = int(os.getenv("RULES_SYM_ENUM_LIMIT", "1000000"))
SYM_SAMPLES = int(os.getenv("RULES_SYM_SAMPLES", "6"))

# -------------------------
# 参数规范化 / 校验
# -------------------------

_BOUNDARY_CHOICES = {"torus", "open"}
_SYM_CHOICES = {"perm", "perm+swap", "none"}
_OBJECTIVE_FIELD_CHOICES = {"objective_penalized", "objective_raw"}
_PENALTY_MODE_CHOICES = {"n", "n_times_rule_count", "n_times_rows_m"}


def normalize_boundary(boundary: Optional[str]) -> str:
    b = (boundary or BOUNDARY_MODE).strip().lower()
    if b not in _BOUNDARY_CHOICES:
        raise ValueError(f"boundary must be one of {sorted(_BOUNDARY_CHOICES)}, got '{boundary}'")
    return b


def normalize_sym_mode(sym_mode: Optional[str]) -> str:
    s = (sym_mode or "perm").strip().lower().replace("permswap", "perm+swap")
    if s == "swap":
        s = "perm+swap"
    if s not in _SYM_CHOICES:
        raise ValueError(f"sym_mode must be one of {sorted(_SYM_CHOICES)}, got '{sym_mode}'")
    return s


def normalize_penalty_mode(mode: Optional[str]) -> str:
    m = (mode or PENALTY_MODE).strip().lower().replace("n*r", "n_times_rows_m")
    if m in ("n_times_r", "n_times_rows", "n_times_rows_m"):
        m = "n_times_rows_m"
    elif m in ("n_times_rc", "n_times_rulecount"):
        m = "n_times_rule_count"
    elif m in ("n_only", "n"):
        m = "n"
    if m not in _PENALTY_MODE_CHOICES:
        raise ValueError(f"penalty_mode must be one of {sorted(_PENALTY_MODE_CHOICES)}, got '{mode}'")
    return m


def normalize_objective_mode(mode: Optional[str]) -> str:
    m = (mode or "logZ_per_penalty").strip().lower()
    if m in ("logz", "log_z", "log", "raw", "no_penalty", "unpenalized", "nop"):
        return "logZ"
    if m in ("logz_per_nr", "logz_per_n", "logz_per_penalty", "logz_norm", "logz_per_nk"):
        return "logZ_per_penalty"
    return "logZ_per_penalty"


def resolve_objective(mode: Optional[str], use_penalty: Optional[bool], prefer_penalized_field: bool = True):
    mode_norm = normalize_objective_mode(mode)
    penalty = OBJECTIVE_USE_PENALTY if use_penalty is None else bool(use_penalty)
    if prefer_penalized_field and penalty:
        field = "objective_penalized"
    else:
        field = "objective_raw"
    if field not in _OBJECTIVE_FIELD_CHOICES:
        raise ValueError(f"objective_field must be one of {sorted(_OBJECTIVE_FIELD_CHOICES)}, got '{field}'")
    return {
        "objective_mode": mode_norm,
        "objective_use_penalty": penalty,
        "objective_field": field,
    }


def penalty_factor_from_shape(n: int, rows_m: int, rule_count: Optional[int] = None, mode: Optional[str] = None) -> float:
    """
    统一的惩罚因子定义。
    支持：
      - n                     -> penalty = n
      - n_times_rule_count    -> penalty = n * rule_count
      - n_times_rows_m        -> penalty = n * rows_m
    """
    pmode = normalize_penalty_mode(mode)
    n_val = max(1.0, float(n))
    rows_val = max(1.0, float(rows_m))
    if pmode == "n":
        return n_val
    if pmode == "n_times_rule_count":
        rc = max(1.0, float(rule_count) if rule_count is not None else 1.0)
        return n_val * rc
    return n_val * rows_val


__all__ = [
    "DEFAULT_STYLE",
    "DEFAULT_DEVICE",
    "BOUNDARY_MODE",
    "RESULTS_ROOT",
    "OUT_CSV_DEFAULT",
    "OUT_FIG_DEFAULT",
    "EVAL_CACHE_DIR",
    "EVAL_CACHE_VERSION",
    "ROWS_LRU_CAPACITY",
    "LANCZOS_R",
    "POWER_ITERS",
    "HUTCH_S",
    "TRACE_MODE",
    "ENABLE_EXACT",
    "ENABLE_SPECTRAL",
    "EXACT_THRESHOLD",
    "OBJECTIVE_MODE",
    "OBJECTIVE_USE_PENALTY",
    "PENALTY_MODE",
    "SYM_GEO_OPS",
    "SYM_ENUM_LIMIT",
    "SYM_SAMPLES",
    "normalize_boundary",
    "normalize_sym_mode",
    "normalize_penalty_mode",
    "normalize_objective_mode",
    "resolve_objective",
    "penalty_factor_from_shape",
]
