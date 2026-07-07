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

    shards = glob.glob(str(shard_dir / "**" / "hrrr_*.parquet"), recursive=True)
    if args.months:
        want = set(args.months.split(","))
        shards = [s for s in shards if any(m in s for m in want)]
    if not shards:
        print("ERROR: no HRRR shards found")
        return 1
    print(f"Loading {len(shards)} HRRR shards...")
    hrrr = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    hrrr["valid_time"] = pd.to_datetime(hrrr["init_time"]) + pd.to_timedelta(hrrr["lead_hour"], unit="h")
    print(f"  {len(hrrr)} predictor rows")

    # 2. Observations over the covered valid-time window.
    vt = pd.to_datetime(hrrr["valid_time"])
    start, end = vt.min().date(), vt.max().date()
    print(f"Collecting obs {start} .. {end} for {len(stations)} stations...")
    obs = collect(stations, start, end)
    print(f"  {len(obs)} QC'd obs rows")

    # 3. Join.
    table = build_training_table(hrrr, obs, stations)
    out = Path(args.out) if args.out else dd / "training_table.parquet"
    table.to_parquet(out, index=False)
    print(f"Wrote {len(table)} training rows ({len(feature_columns(table))} features) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
