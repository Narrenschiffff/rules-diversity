# -*- coding: utf-8 -*-
import logging
from typing import Optional

def setup_logging(level: int = logging.INFO,
                  fmt: str = "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                  datefmt: str = "%H:%M:%S",
                  disable_third_party_noise: bool = True):
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    if disable_third_party_noise:
        # 常见第三方库降噪，如 matplotlib/numba/urllib3 等
        for noisy in ["matplotlib", "numexpr", "PIL", "urllib3", "numba", "PIL.PngImagePlugin"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)
