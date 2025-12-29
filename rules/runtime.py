# -*- coding: utf-8 -*-
"""
统一运行上下文：日志（stdout + 文件）、心跳、缓存目录、随机种子与 checkpoint。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import config

try:  # optional torch seeding
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _encode_state(state: dict) -> str:
    import pickle

    return base64.b64encode(pickle.dumps(state)).decode("ascii")


def _decode_state(data: str) -> dict:
    import pickle

    return pickle.loads(base64.b64decode(data.encode("ascii")))


@dataclass
class Checkpoint:
    path: Path
    payload: Dict[str, Any]
    rng_state: Optional[dict]


class RunContext:
    """
    提供统一的运行上下文，封装：
    - 日志（stdout + 文件，单一格式）
    - 心跳文件（周期性覆盖）
    - checkpoint 读写（含 RNG 状态）
    - 缓存目录管理
    - 随机种子统一设置
    """

    def __init__(
        self,
        run_tag: str,
        run_dir: str | Path,
        log_name: str = "rules",
        log_level: str = "INFO",
        cache_dir: str | Path | None = None,
        seed: Optional[int] = None,
    ):
        self.run_tag = run_tag
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_name = log_name
        self.log_level = log_level
        self.log_path = self.run_dir / f"{self.run_tag}.log"
        self.heartbeat_path = self.run_dir / "heartbeat.txt"
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else config.EVAL_CACHE_DIR
        self.cache_dir = self.cache_dir.expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._logging_configured = False
        if seed is not None:
            self.set_seed(seed)

    # --------------------- logging ---------------------
    def configure_logging(self) -> logging.Logger:
        """
        配置 stdout + 文件双通道日志；重复调用时自动去重 handler。
        """

        root = logging.getLogger()
        lvl = getattr(logging, self.log_level.upper(), logging.INFO)
        root.setLevel(lvl)

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        def _has_handler(cls, target):
            for h in root.handlers:
                if isinstance(h, cls):
                    if cls is logging.FileHandler:
                        if getattr(h, "baseFilename", None) == os.path.abspath(target):
                            return True
                    else:
                        return True
            return False

        if not _has_handler(logging.StreamHandler, None):
            sh = logging.StreamHandler()
            sh.setLevel(lvl)
            sh.setFormatter(formatter)
            root.addHandler(sh)

        log_path_str = str(self.log_path)
        if not _has_handler(logging.FileHandler, log_path_str):
            fh = logging.FileHandler(log_path_str, mode="a", encoding="utf-8")
            fh.setLevel(lvl)
            fh.setFormatter(formatter)
            root.addHandler(fh)

        self._logging_configured = True
        return logging.getLogger(self.log_name)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        if not self._logging_configured:
            self.configure_logging()
        return logging.getLogger(name or self.log_name)

    # --------------------- seed & RNG ---------------------
    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            try:
                torch.manual_seed(seed)
            except Exception:
                pass

    def capture_rng_state(self) -> dict:
        state = {"random": random.getstate(), "numpy": np.random.get_state()}
        if torch is not None:
            try:
                state["torch"] = torch.random.get_rng_state()
            except Exception:
                state["torch"] = None
        return state

    def restore_rng_state(self, state: dict) -> None:
        if not state:
            return
        try:
            random.setstate(state.get("random"))  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            np.random.set_state(state.get("numpy"))  # type: ignore[arg-type]
        except Exception:
            pass
        if torch is not None and state.get("torch") is not None:
            try:
                torch.random.set_rng_state(state["torch"])
            except Exception:
                pass

    # --------------------- heartbeat ---------------------
    def heartbeat(self, note: str = "") -> Path:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        payload = f"{ts}Z {note}".strip()
        with open(self.heartbeat_path, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        return self.heartbeat_path

    # --------------------- checkpoint ---------------------
    def save_checkpoint(self, payload: Dict[str, Any], rng_state: Optional[dict] = None) -> Path:
        data = {
            "meta": {
                "run_tag": self.run_tag,
                "saved_at": datetime.utcnow().isoformat() + "Z",
            },
            "payload": payload,
        }
        if rng_state is None:
            rng_state = self.capture_rng_state()
        data["rng_state"] = _encode_state(rng_state)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.checkpoint_path)
        return self.checkpoint_path

    def load_checkpoint(self) -> Optional[Checkpoint]:
        if not self.checkpoint_path.exists():
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            payload = data.get("payload") or {}
            rng_blob = data.get("rng_state")
            rng_state = _decode_state(rng_blob) if rng_blob else None
            return Checkpoint(path=self.checkpoint_path, payload=payload, rng_state=rng_state)
        except Exception:
            return None


__all__ = ["RunContext", "Checkpoint"]
