#!/usr/bin/env python3
"""Build the GLOBAL training table — GFS as the base model (not HRRR).

The western-US model uses HRRR (3km, US-only) as its base forecast. Globally there is no
HRRR, so the global model uses GFS (0.25°, worldwide) as the base: each training row is a
GFS forecast at a station/init/lead, joined to terrain, time encodings, and the observed
value at valid_time. GFS's own forecast of each target is both a predictor and the
baseline the post-processor must beat.

Reads gfs_global/ shards + the global ASOS obs, streams per-shard to a directory of parts
(memory-safe, like the US builder). Run with MTNWX_REGION=global so the global catalogue
and bounds are used.

    MTNWX_REGION=global python scripts/build_training_table_global.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtnwx.config import data_dir, load_configs  # noqa: E402
from mtnwx.data.collect_obs import collect  # noqa: E402
from mtnwx.features.build import TERRAIN_FEATURES, add_time_features  # noqa: E402

# GFS target spec: target obs column -> GFS field that forecasts it (the baseline).
GLOBAL_TARGETS = {
    "air_temp_c": "gfs_temperature_2m",           # GFS temp in C
    "relative_humidity_pct": "gfs_relative_humidity_2m",
    "wind_speed_ms": "gfs_wind_speed_10m",         # derived below
    "precip_1h_mm": "gfs_precip_mm",               # derived (rate->mm) below
}


def add_gfs_derived(gfs: pd.DataFrame) -> pd.DataFrame:
    """Derive wind speed from GFS u/v and precip mm/h; add valid_time."""
    out = gfs.copy()
    if {"gfs_wind_u_10m", "gfs_wind_v_10m"}.issubset(out.columns):
        out["gfs_wind_speed_10m"] = np.hypot(out["gfs_wind_u_10m"], out["gfs_wind_v_10m"]).astype("float32")
        out["gfs_wind_dir_10m"] = (
            (np.degrees(np.arctan2(-out["gfs_wind_u_10m"], -out["gfs_wind_v_10m"])) + 360.0) % 360.0
        ).astype("float32")
    if "gfs_precipitation_surface" in out.columns:
        # GFS precip is a rate (kg m-2 s-1 == mm s-1) -> mm/hour.
        out["gfs_precip_mm"] = (out["gfs_precipitation_surface"] * 3600.0).clip(lower=0).astype("float32")
    out["valid_time"] = pd.to_datetime(out["init_time"]) + pd.to_timedelta(out["lead_hour"], unit="h")
    return out


def _rh_from_temp_dewpoint(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from temperature and dewpoint (both °C), Magnus formula."""
    a, b = 17.625, 243.04
    with np.errstate(over="ignore", invalid="ignore"):
        rh = 100.0 * np.exp((a * td_c) / (b + td_c) - (a * t_c) / (b + t_c))
    return np.clip(rh, 0.0, 100.0).astype("float32")


# ECMWF (AIFS) predictor columns to derive/keep for the join, prefixed ecmwf_.
ECMWF_KEEP = [
    "ecmwf_temperature_2m", "ecmwf_relative_humidity_2m", "ecmwf_wind_speed_10m",
    "ecmwf_wind_dir_10m", "ecmwf_precip_mm", "ecmwf_pressure_surface",
    "ecmwf_total_cloud_cover_atmosphere",
]


def add_ecmwf_derived(ec: pd.DataFrame) -> pd.DataFrame:
    """Derive AIFS wind speed/dir, RH (from dewpoint), precip mm/h; add valid_time.

    AIFS carries 2 m dewpoint (not RH) and a precip *rate* (kg m-2 s-1, same units as
    GFS), so RH is Magnus-derived and precip is rate×3600, exactly like GFS."""
    out = ec.copy()
    if {"ecmwf_wind_u_10m", "ecmwf_wind_v_10m"}.issubset(out.columns):
        out["ecmwf_wind_speed_10m"] = np.hypot(out["ecmwf_wind_u_10m"], out["ecmwf_wind_v_10m"]).astype("float32")
        out["ecmwf_wind_dir_10m"] = (
            (np.degrees(np.arctan2(-out["ecmwf_wind_u_10m"], -out["ecmwf_wind_v_10m"])) + 360.0) % 360.0
        ).astype("float32")
    if {"ecmwf_temperature_2m", "ecmwf_dew_point_temperature_2m"}.issubset(out.columns):
        out["ecmwf_relative_humidity_2m"] = _rh_from_temp_dewpoint(
            out["ecmwf_temperature_2m"].to_numpy(), out["ecmwf_dew_point_temperature_2m"].to_numpy()
        )
    if "ecmwf_precipitation_surface" in out.columns:
        out["ecmwf_precip_mm"] = (out["ecmwf_precipitation_surface"] * 3600.0).clip(lower=0).astype("float32")
    out["valid_time"] = pd.to_datetime(out["init_time"]) + pd.to_timedelta(out["lead_hour"], unit="h")
    keep = ["station_id", "valid_time"] + [c for c in ECMWF_KEEP if c in out.columns]
    return out[keep]


def load_ecmwf(shard_dir: str) -> pd.DataFrame | None:
    """Load + derive all ECMWF-global shards into one slim (station_id, valid_time, ecmwf_*)
    frame for the per-shard merge. Returns None if no ECMWF shards are present yet (the
    global model then trains GFS-only, unchanged)."""
    ec_shards = sorted(glob.glob(f"{shard_dir}/ecmwf_global/ecmwf_*.parquet"))
    if not ec_shards:
        print("no ECMWF-global shards yet — training GFS-only (ECMWF is additive)")
        return None
    frames = [add_ecmwf_derived(pd.read_parquet(s)) for s in ec_shards]
    ec = pd.concat(frames, ignore_index=True)
    ec["valid_time"] = pd.to_datetime(ec["valid_time"])
    # A station/valid_time can recur across overlapping inits; keep the latest-init row
    # (already ordered by shard/init) — drop dup keys to keep the merge one-to-one.
    ec = ec.drop_duplicates(["station_id", "valid_time"], keep="last")
    print(f"ECMWF: {len(ec)} rows from {len(ec_shards)} shards, cols {[c for c in ec.columns if c.startswith('ecmwf_')]}")
    return ec


def add_terrain(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    # Merge elevation + terrain features only. lat/lon are added by add_time_features
    # (it needs them for solar elevation) — merging them here too would collide into
    # lat_x/lat_y and drop the `lat` column (KeyError downstream).
    keep = ["station_id", "elevation_m"] + [c for c in stations.columns if c in TERRAIN_FEATURES]
    st = stations[list(dict.fromkeys(keep))].copy()
    return df.merge(st, on="station_id", how="left")


OBS_JOIN_COLS = ["station_id", "valid_time", "air_temp_c", "relative_humidity_pct",
                 "wind_speed_ms", "wind_gust_ms", "dewpoint_c", "precip_1h_mm"]


def prepare_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Trim obs to the join columns once, datetime-parsed, so the per-shard merge
    never copies the full 50M-row frame (that per-shard copy OOM-killed the runner)."""
    keep = [c for c in OBS_JOIN_COLS if c in obs.columns]
    o = obs[keep].copy()
    o["valid_time"] = pd.to_datetime(o["valid_time"])
    return o


def build_global_table(gfs: pd.DataFrame, obs_slim: pd.DataFrame, stations: pd.DataFrame,
                       ecmwf_slim: pd.DataFrame | None = None) -> pd.DataFrame:
    df = add_gfs_derived(gfs)
    df = add_time_features(df, stations)   # adds lat/lon (+ solar) from stations
    df = add_terrain(df, stations)         # adds elevation + terrain (no lat/lon dup)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    if ecmwf_slim is not None:
        # LEFT join: GFS rows at non-6h leads get NaN ECMWF features (AIFS is 6-hourly);
        # LightGBM handles the missings. ECMWF adds model diversity as extra predictors.
        df = df.merge(ecmwf_slim, on=["station_id", "valid_time"], how="left")
    # obs_slim is already trimmed + datetime-parsed (see prepare_obs) — no copy here.
    return df.merge(obs_slim, on=["station_id", "valid_time"], how="inner")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-obs-cache", action="store_true")
    args = ap.parse_args()
    dd = data_dir()
    hub = load_configs()["hub"]

    # Global catalogue + GFS-global shards from HF.
    from huggingface_hub import hf_hub_download, snapshot_download

    st_name = "stations_terrain_global.parquet"
    try:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], st_name, repo_type="dataset"))
    except Exception:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_global.parquet", repo_type="dataset"))
    print(f"global stations: {len(stations)}")

    # Download ONLY the global GFS + ECMWF shards. The training repo also holds ~10 GB of
    # US HRRR/GFS/ASOS shards; a full snapshot_download pulls all 16.6 GB and fills the
    # GitHub runner's ~14 GB disk, killing the job at ~2-4 min ("operation canceled").
    # allow_patterns keeps the download to the global shards we actually join.
    shard_dir = snapshot_download(
        hub["datasets"]["training"], repo_type="dataset",
        allow_patterns=["gfs_global/*.parquet", "ecmwf_global/*.parquet"],
    )
    shards = sorted(glob.glob(f"{shard_dir}/gfs_global/gfs_*.parquet"))
    if not shards:
        print("ERROR: no gfs_global shards found — run extract_gfs_global.yml first")
        return 1
    months = sorted(os.path.basename(s).replace("gfs_", "").replace(".parquet", "") for s in shards)
    shard_start = pd.Timestamp(months[0] + "-01").date()
    shard_end = (pd.Timestamp(months[-1] + "-01") + pd.offsets.MonthEnd(1)).date()

    # Global obs cache: keyed on the FULL MVP window (2024-01..2025-12) + station-set,
    # NOT the shard window. The shard window drifts as the backfill adds months, so a
    # shard-derived key changes every run and misses the cache — triggering a slow
    # (~3 min) ASOS re-collection that keeps getting killed by runner reclaims. Pinning
    # the key to the full window means the cache always hits once obs are collected once;
    # we collect the whole window and filter to the shards we actually have below.
    start = pd.Timestamp("2024-01-01").date()
    end = pd.Timestamp("2025-12-31").date()
    print(f"shard window {shard_start}..{shard_end}; obs cache window {start}..{end}")
    sid_hash = hashlib.md5("|".join(sorted(stations["station_id"].astype(str))).encode()).hexdigest()[:8]
    obs_key = f"obs/obsG_{start}_{end}_n{len(stations)}_{sid_hash}.parquet"
    obs = None
    if not args.no_obs_cache:
        try:
            # Read ONLY the join columns from the 491 MB parquet — loading all columns
            # of the ~50M-row frame is what pushed the 7 GB runner into OOM (exit 143).
            obs_path = hf_hub_download(hub["datasets"]["verify"], obs_key, repo_type="dataset")
            obs = pd.read_parquet(obs_path, columns=[c for c in OBS_JOIN_COLS])
            print(f"loaded cached global obs ({len(obs)} rows, {len(obs.columns)} cols)")
        except Exception:
            obs = None
    if obs is None:
        print(f"collecting global obs {start}..{end} for {len(stations)} stations...")
        obs = collect(stations, start, end)
        for c in obs.select_dtypes("float64").columns:
            obs[c] = obs[c].astype("float32")
        if not args.no_obs_cache:
            try:
                from mtnwx.data.hub_io import upload_file
                obs.to_parquet(dd / "obsG.parquet", index=False, compression="zstd")
                upload_file(dd / "obsG.parquet", obs_key, hub["datasets"]["verify"], repo_type="dataset")
            except Exception as e:  # noqa: BLE001
                print(f"WARN: obs cache upload failed ({e})")
    print(f"  {len(obs)} global obs rows")

    # Prep obs ONCE (trim + datetime) so the per-shard merge never copies it.
    obs_slim = prepare_obs(obs)
    del obs

    # Load ECMWF (AIFS) predictors once, if present. Additive: absent => GFS-only.
    ecmwf_slim = load_ecmwf(shard_dir)

    out = Path(args.out) if args.out else dd / "training_table_global"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part-*.parquet"):
        old.unlink()
    total = written = 0
    for i, s in enumerate(shards, 1):
        gfs = pd.read_parquet(s)
        joined = build_global_table(gfs, obs_slim, stations, ecmwf_slim)
        if not joined.empty:
            for c in joined.select_dtypes("float64").columns:
                joined[c] = joined[c].astype("float32")
            joined.to_parquet(out / f"part-{i:03d}.parquet", index=False)
            total += len(joined); written += 1
        del gfs, joined
        if i % 6 == 0 or i == len(shards):
            print(f"  joined {i}/{len(shards)} shards, {total} rows")
    if written == 0:
        print("ERROR: no rows survived the obs join")
        return 1
    print(f"Wrote {total} global training rows ({written} parts) -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
