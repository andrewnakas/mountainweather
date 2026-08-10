#!/usr/bin/env python3
"""Build ONE training-table part (one GFS-shard month) and upload it to HF (resumable).

At 26k stations the monolithic table build downloads ~7 GB of shards + 1.9 GB obs and
runs >48 min — longer than a fair-use runner survives, so it never finishes. This does the
join for a SINGLE GFS month: download that one shard, join against the cached obs parts +
ECMWF, upload the part, skip-if-exists. Each job is short (~5-10 min) and survivable; the
training job then just loads the finished table_parts/ (no joining).

    MTNWX_REGION=global python scripts/build_table_shard.py --month 2024-06

Parts land at table_parts/<window+hash>/part-<month>.parquet in the training dataset.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtnwx.config import data_dir, load_configs  # noqa: E402
from build_training_table_global import (  # noqa: E402
    OBS_JOIN_COLS, build_global_table, load_ecmwf, prepare_obs, table_parts_prefix,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="GFS shard month YYYY-MM")
    args = ap.parse_args()
    hub = load_configs()["hub"]
    dd = data_dir()
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)

    # Global catalogue (sorted for a stable hash matching the obs parts).
    try:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_terrain_global.parquet", repo_type="dataset"))
    except Exception:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_global.parquet", repo_type="dataset"))
    stations = stations.sort_values("station_id").reset_index(drop=True)

    prefix = table_parts_prefix(stations)
    target = f"{prefix}/part-{args.month}.parquet"
    files = set(api.list_repo_files(hub["datasets"]["training"], repo_type="dataset"))
    if target in files:
        print(f"table part already exists, skipping: {target}", flush=True)
        return 0

    gfs_name = f"gfs_global/gfs_{args.month}.parquet"
    if gfs_name not in files:
        print(f"no GFS shard for {args.month} ({gfs_name}) — nothing to build")
        return 0

    # Load obs from the batch parts (join cols only).
    from build_training_table_global import assemble_obs_from_parts
    obs = assemble_obs_from_parts(hub, stations)
    if obs is None:
        print("ERROR: obs parts not found — run collect_obs_global.yml first")
        return 1
    obs_slim = prepare_obs(obs)
    del obs

    # ECMWF (all months; additive). load_ecmwf streams+deletes each shard.
    ecmwf_names = sorted(f for f in files if f.startswith("ecmwf_global/") and f.endswith(".parquet"))
    ecmwf_slim = load_ecmwf(hub["datasets"]["training"], ecmwf_names)

    # Join this one GFS month.
    p = hf_hub_download(hub["datasets"]["training"], gfs_name, repo_type="dataset")
    gfs = pd.read_parquet(p)
    joined = build_global_table(gfs, obs_slim, stations, ecmwf_slim)
    if joined.empty:
        print(f"WARN: no rows survived the obs join for {args.month}")
        return 0
    for c in joined.select_dtypes("float64").columns:
        joined[c] = joined[c].astype("float32")
    out = dd / f"table_part_{args.month}.parquet"
    joined.to_parquet(out, index=False, compression="zstd")
    print(f"  {len(joined)} rows for {args.month}", flush=True)

    from mtnwx.data.hub_io import upload_file
    upload_file(out, target, hub["datasets"]["training"], repo_type="dataset")
    print(f"uploaded {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
