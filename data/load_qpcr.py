# src/load_qpcr.py
from __future__ import annotations
import csv
from datetime import datetime
import numpy as np

def load_qpcr_csv(path: str, *, date_format: str = "%Y-%m-%d"):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = datetime.strptime(r["date"].strip(), date_format).date()
            v = float(r["bcr_abl_is_percent"])
            rows.append((d, v))
    rows.sort(key=lambda x: x[0])
    return rows

def to_days_and_log10_is(rows, *, t0_date=None, floor=1e-6):
    # floor handles "undetectable" or very small values; tune to assay LOD if known
    dates = [d for d, _ in rows]
    vals = np.array([v for _, v in rows], dtype=float)

    if t0_date is None:
        t0_date = dates[0]

    days = np.array([(d - t0_date).days for d in dates], dtype=float)
    vals_safe = np.maximum(vals, floor)
    log10_is = np.log10(vals_safe)
    return days, log10_is, t0_date
