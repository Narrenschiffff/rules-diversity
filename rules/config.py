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
# 目标函数：logZ（默认）、logZ/(n*r)、no_penalty（不施加惩罚因子）；可通过 CLI 或 GAConfig 覆盖
OBJECTIVE_MODE = os.getenv("RULES_OBJECTIVE_MODE", "logZ")
OBJECTIVE_USE_PENALTY = os.getenv("RULES_OBJECTIVE_USE_PENALTY", "1") != "0"

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


def normalize_objective_mode(mode: Optional[str]) -> str:
    m = (mode or "logZ").strip().lower()
    if m in ("logz", "log_z", "log", "logz_penalized"):
        return "logZ"
    if m in ("logz_per_nr", "logz_per_mr", "logz_norm", "logz_per_nk"):
        return "logZ_per_nr"
    if m in ("no_penalty", "raw", "unpenalized", "nop"):
        return "no_penalty"
    return "logZ"


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
    "SYM_GEO_OPS",
    "SYM_ENUM_LIMIT",
    "SYM_SAMPLES",
    "normalize_boundary",
    "normalize_sym_mode",
    "normalize_objective_mode",
    "resolve_objective",
]
