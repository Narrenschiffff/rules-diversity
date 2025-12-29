# -*- coding: utf-8 -*-
"""Bootstrap helpers for CLI scripts and notebooks."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


def _detect_repo_root() -> Path:
    """
    Detect repository root based on this file's location.

    Falls back to the parent of the ``rules`` package directory when no marker
    files (``pyproject.toml`` / ``.git``) are found.
    """

    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return here.parents[1]


def ensure_repo_on_path() -> Path:
    """
    Ensure the repository root is present on ``sys.path`` and validate that the
    ``rules`` package is importable.

    Returns
    -------
    Path
        The detected repository root path (inserted at the front of
        ``sys.path`` if absent).

    Raises
    ------
    ModuleNotFoundError
        If ``rules`` cannot be imported even after the repo root is added,
        with a helpful hint to avoid redundant ``PYTHONPATH`` tweaks.
    """

    repo_root = _detect_repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    try:
        import_module("rules")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise ModuleNotFoundError(
            "Failed to import 'rules' even after adding repo root to sys.path. "
            "Remove conflicting PYTHONPATH entries or install via 'pip install -e .'."
        ) from exc

    return repo_root


__all__ = ["ensure_repo_on_path"]
