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
    ap.add_argument("--month", help="Single GFS shard month YYYY-MM")
    ap.add_argument("--months", help="Comma-separated months (loads obs+ECMWF once, loops)")
    ap.add_argument("--ecmwf-slim-only", action="store_true",
                    help="Just build+cache the ECMWF slim frame (short job, better survival), then exit")
    args = ap.parse_args()
    hub = load_configs()["hub"]
    dd = data_dir()
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)

    # Cache-only mode: build the ECMWF slim frame and upload it, nothing else. Once cached,
    # every table-part dispatch skips the ~4.6 GB ECMWF re-download and finishes fast — so
    # banking this one cache (in a short, survivable job) unblocks the rest.
    if args.ecmwf_slim_only:
        files = set(api.list_repo_files(hub["datasets"]["training"], repo_type="dataset"))
        ecmwf_names = sorted(f for f in files if f.startswith("ecmwf_global/") and f.endswith(".parquet"))
        load_ecmwf(hub["datasets"]["training"], ecmwf_names, hub=hub)
        print("ECMWF slim cache built (or already present)", flush=True)
        return 0

    months = []
    if args.months:
        months = [m.strip() for m in args.months.split(",") if m.strip()]
    elif args.month:
        months = [args.month]
    else:
        print("ERROR: pass --month or --months")
        return 1

    # Global catalogue (sorted for a stable hash matching the obs parts).
    try:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_terrain_global.parquet", repo_type="dataset"))
    except Exception:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_global.parquet", repo_type="dataset"))
    stations = stations.sort_values("station_id").reset_index(drop=True)

    prefix = table_parts_prefix(stations)
    files = set(api.list_repo_files(hub["datasets"]["training"], repo_type="dataset"))

    # Which months still need building (skip-if-exists), and have a GFS shard?
    todo = []
    for m in months:
        if f"{prefix}/part-{m}.parquet" in files:
            print(f"skip {m}: table part exists", flush=True)
            continue
        if f"gfs_global/gfs_{m}.parquet" not in files:
            print(f"skip {m}: no GFS shard", flush=True)
            continue
        todo.append(m)
    if not todo:
        print("nothing to build (all present or no shard)", flush=True)
        return 0

    # Load obs + ECMWF ONCE, reuse across all months in this job.
    from build_training_table_global import assemble_obs_from_parts
    obs = assemble_obs_from_parts(hub, stations)
    if obs is None:
        print("ERROR: obs parts not found — run collect_obs_global.yml first")
        return 1
    obs_slim = prepare_obs(obs)
    del obs
    ecmwf_names = sorted(f for f in files if f.startswith("ecmwf_global/") and f.endswith(".parquet"))
    ecmwf_slim = load_ecmwf(hub["datasets"]["training"], ecmwf_names, hub=hub)

    from mtnwx.data.hub_io import upload_file
    for m in todo:
        target = f"{prefix}/part-{m}.parquet"
        p = hf_hub_download(hub["datasets"]["training"], f"gfs_global/gfs_{m}.parquet", repo_type="dataset")
        gfs = pd.read_parquet(p)
        joined = build_global_table(gfs, obs_slim, stations, ecmwf_slim)
        os.remove(p) if os.path.exists(p) else None
        if joined.empty:
            print(f"WARN: no rows survived join for {m}", flush=True)
            del gfs, joined
            continue
        for c in joined.select_dtypes("float64").columns:
            joined[c] = joined[c].astype("float32")
        out = dd / f"table_part_{m}.parquet"
        joined.to_parquet(out, index=False, compression="zstd")
        n = len(joined)
        del gfs, joined
        upload_file(out, target, hub["datasets"]["training"], repo_type="dataset")
        os.remove(out) if os.path.exists(out) else None
        print(f"uploaded {target} ({n} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
