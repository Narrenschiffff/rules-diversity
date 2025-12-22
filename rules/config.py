# -*- coding: utf-8 -*-
"""
rules/config.py
全局轻量配置：默认风格、设备、输出目录等。
"""

from __future__ import annotations
from pathlib import Path
import os

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

# 对称性分析默认
SYM_GEO_OPS = os.getenv("RULES_SYM_GEO_OPS", "rot,ref,trans")
SYM_ENUM_LIMIT = int(os.getenv("RULES_SYM_ENUM_LIMIT", "1000000"))
SYM_SAMPLES = int(os.getenv("RULES_SYM_SAMPLES", "6"))
