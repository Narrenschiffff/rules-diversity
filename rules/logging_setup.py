# -*- coding: utf-8 -*-
"""
rules/logging_setup.py
基础日志配置：在 CLI 入口处调用 setup_logging(level="INFO")
"""

import logging, sys

def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )
