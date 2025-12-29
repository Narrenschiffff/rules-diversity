# -*- coding: utf-8 -*-
import importlib.util
import sys
import types
from pathlib import Path


def _load_bootstrap_module():
    """Load rules.bootstrap directly from file without relying on sys.path."""
    repo_root = Path(__file__).resolve().parents[1]
    bootstrap_path = repo_root / "rules" / "bootstrap.py"

    pkg = types.ModuleType("rules")
    pkg.__path__ = [str(bootstrap_path.parent)]
    sys.modules["rules"] = pkg

    spec = importlib.util.spec_from_file_location(
        "rules.bootstrap", bootstrap_path, submodule_search_locations=[str(bootstrap_path.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["rules.bootstrap"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ensure_repo_on_path_idempotent(monkeypatch):
    # Simulate a clean sys.path without repo root and without pre-imported rules.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    sys.path = [p for p in sys.path if "rules-diversity" not in p]
    sys.modules.pop("rules", None)
    sys.modules.pop("rules.bootstrap", None)

    mod = _load_bootstrap_module()

    before = list(sys.path)
    root = mod.ensure_repo_on_path()
    after = list(sys.path)

    assert str(root) == str(Path(__file__).resolve().parents[1])
    assert before != after
    assert after[0] == str(root)

    # idempotent: second call should not add duplicates
    again = mod.ensure_repo_on_path()
    assert again == root
    assert list(sys.path) == after
