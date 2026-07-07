#!/usr/bin/env python3
"""Emit the init-month matrix for the HRRR backfill as JSON (for GitHub Actions).

Prints a JSON list of "YYYY-MM" strings from --start to --end inclusive. The archive
begins 2018-07; default range is 2019-01 .. the month before today so every month is
complete. GitHub matrix caps at 256 entries — ~90 months fits comfortably.

    python scripts/gen_month_matrix.py --start 2019-01 --end 2025-12
"""
from __future__ import annotations

import argparse
import json
from datetime import date


def months(start: str, end: str) -> list[str]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default=None, help="default: month before current")
    args = ap.parse_args()
    if args.end is None:
        today = date.today()
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        args.end = f"{y:04d}-{m:02d}"
    print(json.dumps(months(args.start, args.end)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
