# -*- coding: utf-8 -*-
import csv
from typing import List, Dict, Iterable

def init_csv_front(path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "run_tag","n","k","rule_bits","rule_count","rows_m",
            "lambda_max","sum_lambda_powers","is_front0",
            "active_k","lower_bound","upper_bound","lower_bound_raw","upper_bound_raw"
        ])

def init_csv_gen(path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "run_tag","n","k","generation","front0_size","best_sample_R","best_sample_sumlam",
            "pop_size","device","trace_mode"
        ])

def append_front_rows_csv(path: str, tag: str, n: int, k: int,
                          pop_bits: List, fits: List[Dict], front0_idx: List[int]):
    front0_set = set(front0_idx)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i, (bits, fit) in enumerate(zip(pop_bits, fits)):
            writer.writerow([
                tag, n, k,
                "".join(map(str, bits.tolist())),
                int(fit.get("rule_count", 0)),
                int(fit.get("rows_m", 0)),
                f"{float(fit.get('lambda_max', 0.0)):.6e}",
                f"{float(fit.get('sum_lambda_powers', -1e300)):.6e}",
                1 if i in front0_set else 0,
                int(fit.get("active_k", 0)),
                f"{float(fit.get('lower_bound', 0.0)):.6e}",
                f"{float(fit.get('upper_bound', 0.0)):.6e}",
                f"{float(fit.get('lower_bound_raw', 0.0)):.6e}",
                f"{float(fit.get('upper_bound_raw', 0.0)):.6e}",
            ])
