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

from mtnwx.config import data_dir, load_configs  # noqa: E402
from mtnwx.data.collect_obs import collect  # noqa: E402
from mtnwx.features.build import build_training_table, feature_columns  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default=None, help="Comma-separated YYYY-MM subset")
    ap.add_argument("--local-shards", default=None, help="Dir of hrrr_*.parquet (skip HF)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-obs-cache", action="store_true", help="Don't read/write the HF obs cache")
    ap.add_argument("--no-nbm-features", action="store_true", help="Don't merge NBM predictor columns")
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

    # 1. Observations for the whole window (hourly; the join key). Collecting the full
    # station set x 7 years is slow, so cache the result on the HF verify dataset — keyed
    # by the window AND the station set (count + a short id hash) so adding stations
    # (e.g. ASOS) invalidates the cache and re-collects. Disable with --no-obs-cache.
    import hashlib

    sid_hash = hashlib.md5(
        "|".join(sorted(stations["station_id"].astype(str))).encode()
    ).hexdigest()[:8]
    obs_key = f"obs/obs_{start}_{end}_n{len(stations)}_{sid_hash}.parquet"
    obs = None
    if not args.no_obs_cache and not args.local_shards:
        try:
            from huggingface_hub import hf_hub_download

            repo = load_configs()["hub"]["datasets"]["verify"]
            p = hf_hub_download(repo, obs_key, repo_type="dataset")
            obs = pd.read_parquet(p)
            print(f"  loaded cached obs from HF ({len(obs)} rows): {obs_key}")
        except Exception:  # noqa: BLE001 — cache miss is normal
            obs = None

    if obs is None:
        print(f"Collecting obs {start} .. {end} for {len(stations)} stations...")
        obs = collect(stations, start, end)
        print(f"  {len(obs)} QC'd obs rows")
        if obs.empty:
            print("ERROR: no observations collected")
            return 1
        if not args.no_obs_cache and not args.local_shards:
            try:
                from mtnwx.data.hub_io import upload_file

                obs.to_parquet(dd / "obs_cache.parquet", index=False)
                upload_file(dd / "obs_cache.parquet", obs_key,
                            load_configs()["hub"]["datasets"]["verify"], repo_type="dataset")
                print(f"  cached obs to HF: {obs_key}")
            except Exception as e:  # noqa: BLE001 — caching is best-effort
                print(f"  WARN: could not cache obs ({e})")

    # 1b. NBM predictors (optional): if a matching NBM cache exists on HF, load it and
    # merge nbm_* columns into each joined shard as PREDICTORS — this lets the model
    # improve on NBM's blend instead of losing to it on wind/RH.
    nbm = None
    if not args.no_nbm_features and not args.local_shards:
        try:
            from huggingface_hub import hf_hub_download

            nbm_key = f"nbm/nbm_{start}_{end}_n{len(stations)}_{sid_hash}.parquet"
            p = hf_hub_download(load_configs()["hub"]["datasets"]["verify"], nbm_key, repo_type="dataset")
            nbm = pd.read_parquet(p)
            nbm["valid_time"] = pd.to_datetime(nbm["valid_time"])
            print(f"  loaded NBM predictors ({len(nbm)} rows): {nbm_key}")
        except Exception as e:  # noqa: BLE001 — NBM features are optional
            print(f"  NBM predictor cache not found ({type(e).__name__}); training without NBM features")
            nbm = None

    # 2. Stream shards one at a time, writing each joined shard to its OWN parquet file
    # in a directory. Never hold more than one joined shard in memory — concatenating
    # all 84 joined shards blows a 16 GB runner (OOM / exit 143). Training reads the
    # directory lazily. `out` is a directory of part-*.parquet files.
    out = Path(args.out) if args.out else dd / "training_table"
    if out.suffix == ".parquet":  # tolerate a .parquet arg -> use as a dir name
        out = out.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part-*.parquet"):
        old.unlink()

    total = 0
    written = 0
    ncols = 0
    for i, s in enumerate(shards, 1):
        hrrr = pd.read_parquet(s)
        hrrr["valid_time"] = pd.to_datetime(hrrr["init_time"]) + pd.to_timedelta(
            hrrr["lead_hour"], unit="h"
        )
        joined = build_training_table(hrrr, obs, stations)
        if not joined.empty:
            if nbm is not None:
                joined = joined.merge(nbm, on=["station_id", "valid_time"], how="left")
                # Explicit availability flag: NBM only covers ~300/978 stations, so give
                # the trees a clean 0/1 split instead of inferring from NaN columns.
                joined["nbm_available"] = joined["nbm_air_temp_c"].notna().astype("float32")
            # Downcast floats to save disk + training memory.
            for c in joined.select_dtypes("float64").columns:
                joined[c] = joined[c].astype("float32")
            joined.to_parquet(out / f"part-{i:03d}.parquet", index=False)
            total += len(joined)
            written += 1
            ncols = len(feature_columns(joined))
        del hrrr, joined
        if i % 12 == 0 or i == len(shards):
            print(f"  joined {i}/{len(shards)} shards, {total} training rows across {written} parts")

    if written == 0:
        print("ERROR: no rows survived the obs join")
        return 1
    print(f"Wrote {total} training rows ({ncols} features) across {written} parts -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
