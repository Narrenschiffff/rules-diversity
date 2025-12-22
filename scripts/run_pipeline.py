"""
Pipeline CLI that stitches together exact counting, spectral estimation,
symmetry/canonicalization, boundary selection, and cache-aware output.

Features
- Accept JSON/YAML config files plus CLI overrides.
- Reuse rules.logging_setup for consistent logs.
- Progress reporting via tqdm when available (fallback: heartbeat logs).
- Persist results to CSV + JSONL with cache keys for reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from rules import config as rules_config
from rules import logging_setup
from rules.cache import make_eval_cache_key
from rules.eval import (
    bits_from_rule,
    canonical_bits,
    evaluate_rules_batch,
    make_rule_matrix,
    rule_from_bits,
)
from rules.stage1_exact import count_patterns_transfer_matrix, enumerate_ring_rows
from rules.utils_io import ensure_dir, make_run_tag, slugify

try:  # optional YAML
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:  # optional progress bar
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


LOGGER = logging.getLogger("run_pipeline")
DEFAULT_OUT_ROOT = rules_config.RESULTS_ROOT / "pipeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bits_from_any(rule_cfg: Dict[str, Any], k: int) -> np.ndarray:
    """Parse bits from config entry. Supports `bits` string/list or `pairs` edges.

    Raises
    ------
    ValueError
        If the spec cannot be resolved into a bit array.
    """
    if "bits" in rule_cfg and rule_cfg["bits"] is not None:
        raw = rule_cfg["bits"]
        if isinstance(raw, str):
            raw = raw.strip().replace(" ", "")
            bits = np.array([1 if ch == "1" else 0 for ch in raw], dtype=np.uint8)
        elif isinstance(raw, (list, tuple, np.ndarray)):
            bits = np.array(list(raw), dtype=np.uint8)
        else:
            raise ValueError("bits must be str or list")
        expected = k + (k * (k - 1)) // 2
        if bits.size != expected:
            raise ValueError(f"bits length={bits.size} mismatch expected={expected} for k={k}")
        return bits
    if "pairs" in rule_cfg and rule_cfg["pairs"]:
        pairs = [(int(a), int(b)) for a, b in rule_cfg["pairs"]]
        allow_self = rule_cfg.get("allow_self_loops")
        R = make_rule_matrix(k=k, allowed_pairs=pairs, allow_self_loops=allow_self)
        return bits_from_rule(R)
    raise ValueError("rule spec must contain either 'bits' or 'pairs'")


def _bits_to_str(bits: np.ndarray) -> str:
    return "".join("1" if int(b) else "0" for b in bits.tolist())


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed; cannot parse YAML config")
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text or "{}")
    # fallback: try YAML then JSON
    if yaml is not None:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            pass
    return json.loads(text or "{}")


def _heartbeat_logger(iteration: int, total: int, last: float, interval: float) -> float:
    now = time.time()
    if now - last >= interval:
        LOGGER.info("progress %s/%s (%.1f%%)", iteration, total, 100 * iteration / max(1, total))
        return now
    return last


def _exact_cache_key(bits: np.ndarray, n: int, k: int, boundary: str) -> str:
    h = hashlib.sha256()
    h.update(str(n).encode("ascii"))
    h.update(str(k).encode("ascii"))
    h.update(boundary.encode("utf-8"))
    h.update(bits.tobytes())
    return h.hexdigest()


def _run_exact(bits: np.ndarray, n: int, k: int, boundary: str, cache_dir: Path, rows_cap: int) -> Dict[str, Any]:
    boundary = (boundary or "torus").lower()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _exact_cache_key(bits, n, k, boundary)
    cache_path = cache_dir / f"exact_{key}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["exact_cache"] = True
            return data
        except Exception:
            pass

    R = rule_from_bits(k, bits)
    rows = enumerate_ring_rows(n=n, k=k, R=R, boundary=boundary)
    rows_m = len(rows)
    active_k = len({int(v) for row in rows for v in row}) if rows else 0
    if rows_cap > 0 and rows_m > rows_cap:
        return {
            "exact_cache": False,
            "exact_Z": None,
            "rows_m_exact": rows_m,
            "active_k_exact": active_k,
            "exact_note": f"skip exact: rows_m={rows_m} exceeds cap={rows_cap}",
            "exact_boundary": boundary,
        }
    Z_exact = count_patterns_transfer_matrix(n=n, k=k, R=R, boundary=boundary)
    payload = {
        "exact_cache": False,
        "exact_Z": int(Z_exact),
        "rows_m_exact": rows_m,
        "active_k_exact": active_k,
        "exact_note": "",
        "exact_boundary": boundary,
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        LOGGER.warning("failed to write exact cache to %s", cache_path, exc_info=True)
    return payload


def _prepare_rules(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[int, int, List[Dict[str, Any]]]:
    base_n = args.n or cfg.get("n") or cfg.get("N")
    base_k = args.k or cfg.get("k") or cfg.get("K")
    if base_n is None or base_k is None:
        raise ValueError("`n` and `k` must be provided via CLI or config")
    try:
        base_n = int(base_n)
        base_k = int(base_k)
    except Exception as exc:  # pragma: no cover - validated via argparse
        raise ValueError("n and k must be integers") from exc

    rules_cfg: List[Dict[str, Any]] = []
    if args.rule_bits:
        for b in args.rule_bits:
            rules_cfg.append({"bits": b, "name": slugify(b)})
    rules_cfg.extend(cfg.get("rules") or [])
    if not rules_cfg:
        raise ValueError("no rules provided; use --rule-bits or config rules[]")
    return base_n, base_k, rules_cfg


def run_pipeline(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg = _load_config(args.config)
    logging_setup.setup_logging(level=args.log_level or cfg.get("log_level", "INFO"))

    n, k, rule_specs = _prepare_rules(args, cfg)
    boundary = (args.boundary or cfg.get("boundary") or rules_config.BOUNDARY_MODE).lower()
    sym_mode = args.sym_mode or cfg.get("sym_mode") or "perm"
    device = args.device or cfg.get("device") or rules_config.DEFAULT_DEVICE
    trace_mode = args.trace_mode or cfg.get("trace_mode") or rules_config.TRACE_MODE
    hutch_s = args.hutch_s or cfg.get("hutch_s") or rules_config.HUTCH_S
    power_iters = args.power_iters or cfg.get("power_iters") or rules_config.POWER_ITERS
    lanczos_r = args.lanczos_r or cfg.get("lanczos_r") or rules_config.LANCZOS_R
    use_exact = not args.no_exact if args.no_exact is not None else cfg.get("use_exact", rules_config.ENABLE_EXACT)
    use_spectral = not args.no_spectral if args.no_spectral is not None else cfg.get("use_spectral", rules_config.ENABLE_SPECTRAL)
    use_cache = not args.no_cache if args.no_cache is not None else cfg.get("use_cache", True)
    rows_cap = args.exact_rows_cap if args.exact_rows_cap is not None else cfg.get("exact_rows_cap", 200_000)
    out_dir = Path(args.out_dir or cfg.get("out_dir") or DEFAULT_OUT_ROOT)
    cache_dir = Path(args.cache_dir or cfg.get("cache_dir") or rules_config.EVAL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag or cfg.get("run_tag") or make_run_tag(n, k)
    run_dir = ensure_dir(out_dir / slugify(run_tag))
    LOGGER.info("pipeline start | n=%s k=%s boundary=%s rules=%s", n, k, boundary, len(rule_specs))

    results: List[Dict[str, Any]] = []
    iterator: Iterable[Dict[str, Any]]
    if tqdm is not None:
        iterator = tqdm(rule_specs, desc="rules", unit="rule")
    else:
        iterator = rule_specs
    last_log = time.time()
    heartbeat = float(args.heartbeat or cfg.get("heartbeat", 10.0))

    for idx, spec in enumerate(iterator, start=1):
        last_log = _heartbeat_logger(idx, len(rule_specs), last_log, heartbeat) if tqdm is None else last_log
        label = spec.get("name") or spec.get("label") or f"rule_{idx}"
        try:
            bits_raw = _bits_from_any(spec, k)
        except Exception as exc:
            LOGGER.exception("failed to parse rule %s", label)
            raise exc
        bits_canon = canonical_bits(bits_raw, k) if spec.get("canonicalize", True) else bits_raw
        base_row: Dict[str, Any] = {
            "run_tag": run_tag,
            "name": label,
            "n": n,
            "k": k,
            "boundary": boundary,
            "sym_mode": sym_mode,
            "rule_bits_raw": _bits_to_str(bits_raw),
            "rule_bits_canon": _bits_to_str(bits_canon),
        }

        # exact path
        exact_info: Dict[str, Any] = {}
        if use_exact:
            try:
                exact_info = _run_exact(bits_canon, n, k, boundary, cache_dir=run_dir / "cache", rows_cap=rows_cap)
            except Exception:
                LOGGER.exception("exact counting failed for %s", label)
                exact_info = {"exact_note": "exact failed"}
        else:
            exact_info["exact_note"] = "exact disabled"

        # spectral path
        spectral_info: Dict[str, Any] = {}
        if use_spectral:
            if boundary != "torus":
                spectral_info["spectral_note"] = "spectral evaluator supports torus only"
            else:
                try:
                    reports = evaluate_rules_batch(
                        n=n,
                        k=k,
                        bits_list=[bits_canon],
                        sym_mode=sym_mode,
                        boundary=boundary,
                        device=device,
                        use_lanczos=True,
                        r_vals=lanczos_r,
                        power_iters=power_iters,
                        trace_mode=trace_mode,
                        hutch_s=hutch_s,
                        lru_rows=None,
                        max_streams=args.max_streams or cfg.get("max_streams", 2),
                        enable_exact=use_exact,
                        enable_spectral=True,
                        exact_threshold=cfg.get("exact_threshold", rules_config.EXACT_THRESHOLD),
                        cache_dir=str(cache_dir),
                        use_cache=use_cache,
                    )
                    spectral_info = reports[0] if reports else {}
                    if spectral_info:
                        spectral_info["cache_key"] = make_eval_cache_key(
                            bits_canon,
                            int(spectral_info.get("active_k", spectral_info.get("k_sym", k))),
                            boundary,
                            sym_mode,
                            n,
                        )
                except Exception:
                    LOGGER.exception("spectral evaluation failed for %s", label)
                    spectral_info["spectral_note"] = "spectral failed"
        else:
            spectral_info["spectral_note"] = "spectral disabled"

        row = {**base_row, **exact_info, **spectral_info}
        results.append(row)

    # persist outputs
    csv_path = run_dir / "summary.csv"
    jsonl_path = run_dir / "summary.jsonl"
    if args.save_csv or cfg.get("save_csv", True):
        _write_csv(csv_path, results)
    if args.save_jsonl or cfg.get("save_jsonl", True):
        _write_jsonl(jsonl_path, results)

    LOGGER.info("pipeline finished -> %s entries (csv=%s, jsonl=%s)", len(results), csv_path, jsonl_path)
    return results


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys: List[str] = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rules-diversity unified pipeline")
    parser.add_argument("--config", type=str, help="JSON/YAML config path", default=None)
    parser.add_argument("--n", type=int, help="grid size", default=None)
    parser.add_argument("--k", type=int, help="state count", default=None)
    parser.add_argument("--boundary", type=str, help="boundary (torus|open)", default=None)
    parser.add_argument("--sym-mode", type=str, dest="sym_mode", help="rule symmetry mode", default=None)
    parser.add_argument("--device", type=str, help="torch device for spectral path", default=None)
    parser.add_argument("--trace-mode", type=str, dest="trace_mode", help="trace estimator (hutch|hutchpp|lanczos_sum)", default=None)
    parser.add_argument("--hutch-s", type=int, dest="hutch_s", help="samples for Hutch/Hutch++", default=None)
    parser.add_argument("--power-iters", type=int, dest="power_iters", help="power iterations for eigen", default=None)
    parser.add_argument("--lanczos-r", type=int, dest="lanczos_r", help="Lanczos r (top eigenvalues)", default=None)
    parser.add_argument("--max-streams", type=int, dest="max_streams", help="max CUDA streams (spectral)", default=None)
    parser.add_argument("--run-tag", type=str, help="run tag for output dir", default=None)
    parser.add_argument("--out-dir", type=str, help="output root directory", default=None)
    parser.add_argument("--cache-dir", type=str, help="shared eval cache directory", default=None)
    parser.add_argument("--heartbeat", type=float, help="seconds between progress heartbeats", default=None)
    parser.add_argument("--log-level", type=str, help="logging level (INFO/DEBUG)", default=None)
    parser.add_argument("--rule-bits", action="append", help="rule bits (0/1) string; can be repeated", default=None)
    parser.add_argument("--no-exact", action="store_true", help="disable exact counting", default=None)
    parser.add_argument("--no-spectral", action="store_true", help="disable spectral eval", default=None)
    parser.add_argument("--no-cache", action="store_true", help="disable spectral cache reuse", default=None)
    parser.add_argument("--exact-rows-cap", type=int, dest="exact_rows_cap", default=None, help="skip exact when rows exceed cap")
    parser.add_argument("--save-csv", action="store_true", help="force write CSV summary", default=None)
    parser.add_argument("--save-jsonl", action="store_true", help="force write JSONL summary", default=None)
    return parser


def main(argv: Optional[Iterable[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_pipeline(args)
    except Exception as exc:  # pragma: no cover - CLI guard
        LOGGER.error("pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
