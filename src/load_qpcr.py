# src/load_qpcr.py
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np


# Common date formats encountered in exports (DK + ISO + timestamps)
COMMON_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d/%m/%Y %H:%M",
)


def _norm_header(s: str) -> str:
    """Normalize header names for matching."""
    s = (s or "").strip().lower()
    s = s.replace("%", "percent")
    s = re.sub(r"[\s\-_]+", "", s)
    return s


def _guess_delimiter(sample_text: str) -> str:
    """
    Guess delimiter based on first chunk.
    DK Excel is often ';' while many exports are ','.
    """
    # crude but effective: count separators in the header line
    lines = [ln for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        return ","
    header = lines[0]
    return ";" if header.count(";") >= header.count(",") else ","


def _parse_date(s: str, formats: Iterable[str]) -> datetime.date:
    s = (s or "").strip()
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Could not parse date '{s}'. Tried: {', '.join(formats)}")


@dataclass(frozen=True)
class QpcrTableSpec:
    date_col: str
    value_col: str
    delimiter: str


def _pick_columns(fieldnames: List[str]) -> Tuple[str, str]:
    """
    Guess date/value columns from CSV headers.

    Date candidates: date, dato, prøvedato, sampledate...
    Value candidates: bcr-abl, is, international scale, percent etc.
    """
    if not fieldnames:
        raise ValueError("CSV has no header columns (fieldnames empty).")

    norm_map = {name: _norm_header(name) for name in fieldnames}
    norms = list(norm_map.values())

    # Date column heuristics (ordered by preference)
    date_patterns = (
        "date",
        "dato",
        "sampledate",
        "proevedato",
        "prøvedato",
        "testdate",
        "measurementdate",
        "måledato",
    )

    # Value column heuristics
    value_patterns = (
        "bcrabl",
        "bcrabl1",
        "bcrablis",
        "internationalscale",
        "ispercent",
        "is",
        "bcrablpercent",
        "bcrabl1percent",
        "result",
        "value",
    )

    # Find best date col
    date_col = None
    for pat in date_patterns:
        for orig, n in norm_map.items():
            if pat in n:
                date_col = orig
                break
        if date_col:
            break

    # If still none, fallback: first column that looks like a date label
    if date_col is None:
        # common: first column is date
        date_col = fieldnames[0]

    # Find best value col (prefer those containing bcr + is)
    value_col = None

    # 1) strongest: contains both 'bcr' and 'is' (in some form)
    for orig, n in norm_map.items():
        if ("bcr" in n and "is" in n) or ("bcrabl" in n and "internationalscale" in n):
            value_col = orig
            break

    # 2) any value_patterns
    if value_col is None:
        for pat in value_patterns:
            for orig, n in norm_map.items():
                if pat in n:
                    value_col = orig
                    break
            if value_col:
                break

    # 3) fallback: second column if exists
    if value_col is None:
        if len(fieldnames) >= 2:
            value_col = fieldnames[1]
        else:
            raise ValueError("Could not infer value column: CSV only has one column.")

    return date_col, value_col


def load_qpcr_csv(
    path: str | Path,
    *,
    date_formats: Optional[Iterable[str]] = None,
    delimiter: Optional[str] = None,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
) -> List[Tuple[datetime.date, float]]:
    """
    Loads qPCR measurements from CSV.

    - Auto-detects delimiter if not provided.
    - Auto-detects date/value columns if not provided.
    - Accepts comma decimals (0,006) and dot decimals.
    - Provides helpful error messages with line numbers.
    """
    if date_formats is None:
        date_formats = COMMON_DATE_FORMATS

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Read a small sample to guess delimiter robustly
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    if delimiter is None:
        delimiter = _guess_delimiter(sample)

    rows: List[Tuple[datetime.date, float]] = []

    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row (fieldnames is None).")

        # Auto-pick columns if not provided
        if date_column is None or value_column is None:
            guessed_date, guessed_val = _pick_columns(reader.fieldnames)
            date_column = date_column or guessed_date
            value_column = value_column or guessed_val

        for line_no, r in enumerate(reader, start=2):  # header is line 1
            if not r:
                continue

            raw_date = (r.get(date_column) or "").strip()
            raw_val = (r.get(value_column) or "").strip()



            # Skip empty lines
            if raw_date == "" and raw_val == "":
                continue

            if raw_date == "":
                # If a value exists but date missing, treat as error (bad row)
                raise ValueError(
                    f"Missing date on CSV line {line_no}. "
                    f"Expected column '{date_column}'. "
                    f"Available columns: {reader.fieldnames}"
                )

            try:
                d = _parse_date(raw_date, date_formats)
            except ValueError as e:
                raise ValueError(
                    f"{e} (CSV line {line_no}, column '{date_column}')."
                ) from None

            # allow comma decimals
            raw_val_norm = raw_val.replace(",", ".")
            try:
                v = float(raw_val_norm)
            except ValueError:
                raise ValueError(
                    f"Could not parse value '{raw_val}' as float "
                    f"(CSV line {line_no}, column '{value_column}')."
                ) from None

            rows.append((d, v))

    rows.sort(key=lambda x: x[0])
    return rows


def to_days_and_log10_is(
    rows: List[Tuple[datetime.date, float]],
    *,
    t0_date=None,
    floor: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, datetime.date]:
    """
    Convert (date, IS%) rows to:
      - days since t0_date
      - log10(IS%) with a floor for undetectable/near-zero values
    """
    if not rows:
        raise ValueError(
            "No qPCR rows loaded. "
            "This usually means: wrong delimiter, wrong column guessing, or the CSV has no data rows."
        )

    dates = [d for d, _ in rows]
    vals = np.array([v for _, v in rows], dtype=float)

    if t0_date is None:
        t0_date = dates[0]

    days = np.array([(d - t0_date).days for d in dates], dtype=float)
    vals_safe = np.maximum(vals, float(floor))
    log10_is = np.log10(vals_safe)
    return days, log10_is, t0_date
