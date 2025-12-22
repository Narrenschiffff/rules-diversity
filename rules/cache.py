# -*- coding: utf-8 -*-
"""
轻量文件缓存：用于复用 evaluate_rules_batch 的谱估计/精确结果。

缓存键包含：对称化后的规则位串、有效状态数（active_k）、边界模式、对称选项、棋盘规模 n。
所有记录带有来源与版本元数据，便于失效与调试。
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

from . import config

EVAL_CACHE_VERSION = config.EVAL_CACHE_VERSION


def ensure_eval_cache_dir(cache_dir: Optional[str | Path] = None) -> Path:
    path = Path(cache_dir) if cache_dir is not None else config.EVAL_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_eval_cache_key(bits_sym: np.ndarray, active_k: int, boundary: str, sym_mode: str, n: int) -> str:
    h = hashlib.sha256()
    h.update(sym_mode.encode("utf-8"))
    h.update(b"|")
    h.update(boundary.encode("utf-8"))
    h.update(b"|")
    h.update(str(active_k).encode("ascii"))
    h.update(b"|")
    h.update(str(n).encode("ascii"))
    h.update(b"|")
    h.update(bits_sym.tobytes())
    return h.hexdigest()


def _to_serializable(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def load_eval_cache(cache_dir: Path, key: str) -> Optional[Dict]:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta", {})
        if meta.get("version") != EVAL_CACHE_VERSION:
            return None
        return data.get("result")
    except Exception:
        return None


def dump_eval_cache(cache_dir: Path, key: str, result: Dict, meta_extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "meta": {
            "source": "evaluate_rules_batch",
            "version": EVAL_CACHE_VERSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
        "result": {k: _to_serializable(v) for k, v in result.items()},
    }
    if meta_extra:
        payload["meta"].update(meta_extra)
    path = cache_dir / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)

