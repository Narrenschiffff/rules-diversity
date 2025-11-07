# -*- coding: utf-8 -*-
"""
rules/utils_io.py
通用 I/O 与 CLI 工具：目录创建、CSV 装载、通配展开、(n,k) 解析、子进程执行等。
"""

from __future__ import annotations
import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple
import os, re, csv, subprocess, shlex

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def expand_globs(paths: Optional[Iterable[str]]) -> List[str]:
    import glob
    out: List[str] = []
    if not paths:
        return out
    for p in paths:
        ps = glob.glob(str(p))
        if ps:
            out.extend(ps)
        else:
            # 也允许原样加入，后续存在性再判断
            out.append(p)
    # 去重并保持相对顺序
    seen=set(); uniq=[]
    for x in out:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

def load_csv_rows(csv_paths: Iterable[str]) -> List[dict]:
    rows: List[dict] = []
    for p in expand_globs(csv_paths):
        if not os.path.exists(p): 
            continue
        with open(p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                row["__file__"] = p
                rows.append(row)
    return rows

def write_csv(path: str | Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> str:
    path = str(path)
    if not rows:
        # 如果给了表头，写 header；否则写空文件
        with open(path, "w", newline="", encoding="utf-8") as f:
            if fieldnames:
                w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        return path
    if fieldnames is None:
        keys = set()
        for r in rows:
            keys |= set(r.keys())
        fieldnames = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    return path

def parse_nk_from_filename(fn: str) -> Optional[Tuple[int,int]]:
    # 支持两种命名：stage1_pareto_n{n}_k{k}_...  /  pareto_front_n{n}_k{k}[_...].csv
    rx_stage1 = re.compile(r"stage1_pareto_n(\d+)_k(\d+)")
    rx_ga     = re.compile(r"pareto_front_(?:nk_)?n(\d+)_k(\d+)(?:_|\.csv)")
    fn = os.path.basename(fn)
    m = rx_stage1.search(fn) or rx_ga.search(fn)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def to_int(x) -> Optional[int]:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        return int(float(x))
    except Exception:
        return None

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_{2,}", "_", s).strip("_")
    return s or "run"

def safe_run(cmd: list[str] | str, cwd: Optional[str | Path] = None, env: Optional[dict] = None, verbosity: int = 1):
    """
    统一的子进程执行器：
    - 返回 (retcode, (stdout, stderr))
    - stdout/stderr 以 utf-8 解码
    """
    if isinstance(cmd, list):
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
    else:
        cmd_str = cmd
    if verbosity >= 2:
        print("[run]", cmd_str)
    proc = subprocess.Popen(
        cmd_str, cwd=str(cwd) if cwd else None, env=env,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_b, err_b = proc.communicate()
    out = out_b.decode("utf-8", errors="replace")
    err = err_b.decode("utf-8", errors="replace")
    if verbosity >= 2:
        if out.strip():
            print("┈ stdout ┈\n" + out)
        if err.strip():
            print("┈ stderr ┈\n" + err)
    return proc.returncode, (out, err)

def make_run_tag(n: int, k: int, add_timestamp: bool = True, suffix: str = "") -> str:
    """
    run_tag 生成：
      - 本地复用时：add_timestamp=False（更短；便于图例整洁）
      - 新运行：add_timestamp=True
    """
    base = f"n{n}_k{k}"
    if add_timestamp:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        tag = f"{base}_{ts}"
    else:
        tag = f"{base}"
    if suffix:
        tag += f"_{suffix}"
    return tag
