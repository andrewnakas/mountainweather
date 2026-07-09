#!/usr/bin/env python3
"""Build the training table by joining HRRR shards + obs + terrain.

Pulls the HRRR predictor shards and station catalogue from HF, collects the matching
observations for the covered period, joins everything via features.build, and writes the
training table (locally, and optionally back to HF). This is the bridge between the
backfill (M3) and training (M4).

    python scripts/build_training_table.py            # full (CI)
    python scripts/build_training_table.py --months 2024-01,2024-02   # subset
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtnwx.config import data_dir  # noqa: E402
from mtnwx.data.collect_obs import collect  # noqa: E402
from mtnwx.features.build import build_training_table, feature_columns  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default=None, help="Comma-separated YYYY-MM subset")
    ap.add_argument("--local-shards", default=None, help="Dir of hrrr_*.parquet (skip HF)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dd = data_dir()

    # 1. Station catalogue (with terrain) + HRRR shards, from HF unless local.
    if args.local_shards:
        shard_dir = Path(args.local_shards)
        stations = pd.read_parquet(dd / "stations_terrain.parquet")
    else:
        from mtnwx.data.hub_io import download_dataset_snapshot

        st_dir = download_dataset_snapshot("stations")
        stfiles = glob.glob(f"{st_dir}/*terrain*.parquet") or glob.glob(f"{st_dir}/*.parquet")
        stations = pd.read_parquet(stfiles[0])
        shard_dir = Path(download_dataset_snapshot("training"))

    shards = sorted(glob.glob(str(shard_dir / "**" / "hrrr_*.parquet"), recursive=True))
    if args.months:
        want = set(args.months.split(","))
        shards = [s for s in shards if any(m in s for m in want)]
    if not shards:
        print("ERROR: no HRRR shards found")
        return 1

    # Determine the full valid-time window from shard names (YYYY-MM) without loading
    # them — 84 shards x ~4.8M rows won't fit in RAM at once.
    months = sorted(s.split("hrrr_")[-1].replace(".parquet", "") for s in shards)
    start = pd.Timestamp(months[0] + "-01").date()
    end = (pd.Timestamp(months[-1] + "-01") + pd.offsets.MonthEnd(1)).date()

    # 1. Collect observations ONCE for the whole window (hourly; the join key).
    print(f"Collecting obs {start} .. {end} for {len(stations)} stations...")
    obs = collect(stations, start, end)
    print(f"  {len(obs)} QC'd obs rows")
    if obs.empty:
        print("ERROR: no observations collected")
        return 1

    # 2. Stream shards one at a time: join each to obs (inner join keeps only rows with
    # a matching observation), accumulate the much-smaller joined result. This keeps
    # peak memory to one shard + obs, not all 84 shards.
    out = Path(args.out) if args.out else dd / "training_table.parquet"
    parts: list[pd.DataFrame] = []
    total = 0
    for i, s in enumerate(shards, 1):
        hrrr = pd.read_parquet(s)
        hrrr["valid_time"] = pd.to_datetime(hrrr["init_time"]) + pd.to_timedelta(
            hrrr["lead_hour"], unit="h"
        )
        joined = build_training_table(hrrr, obs, stations)
        if not joined.empty:
            parts.append(joined)
            total += len(joined)
        del hrrr, joined
        if i % 12 == 0 or i == len(shards):
            print(f"  joined {i}/{len(shards)} shards, {total} training rows so far")

    if not parts:
        print("ERROR: no rows survived the obs join")
        return 1
    table = pd.concat(parts, ignore_index=True)
    table.to_parquet(out, index=False)
    print(f"Wrote {len(table)} training rows ({len(feature_columns(table))} features) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
