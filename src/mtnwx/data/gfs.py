"""Extract GFS forecast predictors at station points (the second model, for diversity).

mtnwx beats NBM decisively on temperature and gust but trails on wind-speed/RH/precip —
because NBM is a multi-model *blend* and a single-model (HRRR) post-processor can't match
that on the blend-favored variables. Adding a second, independent model as predictors
gives the post-processor the diversity NBM has.

GFS (NOAA global, 0.25 deg, icechunk archive on dynamical.org from 2021-05) is a good
complement to HRRR: a completely different model (global vs regional 3km), with the
longest archive overlap of the dynamical.org options. Accessed via ``dynamical_catalog``.
Its grid is regular lat/lon, so point extraction is a direct ``.sel(method="nearest")`` —
simpler than HRRR's Lambert KD-tree.

Output mirrors the HRRR shards: gfs/gfs_<YYYY-MM>.parquet on the training dataset, columns
prefixed ``gfs_`` so build_training_table merges them as predictors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mtnwx.config import data_dir

# GFS variables to carry as predictors (prefixed gfs_ in the output).
GFS_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_u_10m",
    "wind_v_10m",
    "precipitation_surface",
    "pressure_surface",
    "total_cloud_cover_atmosphere",
]


def open_archive():
    import dynamical_catalog as dc

    return dc.open("noaa-gfs-forecast")


def month_init_times(ds, month: str) -> pd.DatetimeIndex:
    all_inits = pd.to_datetime(ds.init_time.values)
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthBegin(1)
    return all_inits[(all_inits >= start) & (all_inits < end)]


def _extract_one_init(ds, init, fields, lat_da, lon_da, *, retries: int = 4):
    """Long DataFrame for one GFS init: rows = station x lead, cols = gfs_<field>."""
    import time

    last: Exception | None = None
    for attempt in range(retries):
        try:
            sub = (
                ds[fields]
                .sel(init_time=init)
                .sel(latitude=lat_da, longitude=lon_da, method="nearest")
                .compute()
            )
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt + 1)
    else:
        raise RuntimeError(f"GFS extract failed for init {init}") from last

    lead_h = (ds.lead_time.values / np.timedelta64(1, "h")).astype("int32")
    n_lead, n_st = sub[fields[0]].shape
    out = pd.DataFrame(
        {
            "init_time": np.repeat(np.datetime64(init), n_lead * n_st),
            "lead_hour": np.tile(np.repeat(lead_h, n_st), 1),
            "station_ix": np.tile(np.arange(n_st), n_lead),
        }
    )
    for f in fields:
        out[f"gfs_{f}"] = sub[f].values.reshape(-1).astype("float32")
    return out


def extract_month(month: str, stations: pd.DataFrame) -> pd.DataFrame:
    """Extract GFS predictors for every station over one init-month (leads to 48h)."""
    import xarray as xr

    ds = open_archive()
    lat_da = xr.DataArray(stations["lat"].to_numpy(), dims="station")
    lon_da = xr.DataArray(stations["lon"].to_numpy(), dims="station")

    inits = month_init_times(ds, month)
    if len(inits) == 0:
        print(f"No GFS inits in {month}")
        return pd.DataFrame()

    # Only keep leads 0-48h to match the HRRR window (GFS goes to 384h).
    lead_h_all = (ds.lead_time.values / np.timedelta64(1, "h")).astype("int32")
    keep = lead_h_all <= 48
    ds = ds.isel(lead_time=np.flatnonzero(keep))

    print(f"{month}: {len(inits)} GFS inits x {len(stations)} stations x {len(GFS_VARS)} fields")
    frames = []
    for k, it in enumerate(inits, 1):
        frames.append(_extract_one_init(ds, it, GFS_VARS, lat_da, lon_da))
        if k % 20 == 0 or k == len(inits):
            print(f"  extracted {k}/{len(inits)} inits", flush=True)

    allf = pd.concat(frames, ignore_index=True)
    allf["station_id"] = stations["station_id"].to_numpy()[allf["station_ix"].to_numpy()]
    allf = allf.drop(columns="station_ix")
    return allf


def main(args: argparse.Namespace) -> int:
    stations_path = (
        Path(args.stations) if args.stations else data_dir() / "stations_terrain.parquet"
    )
    if not stations_path.exists():
        alt = data_dir() / "stations.parquet"
        stations_path = alt if alt.exists() else stations_path
    if not stations_path.exists():
        print(f"ERROR: no station catalogue at {stations_path}")
        return 1
    stations = pd.read_parquet(stations_path)

    df = extract_month(args.month, stations)
    if df.empty:
        print("No GFS data extracted.")
        return 0
    out = Path(args.out) if args.out else data_dir() / f"gfs_{args.month}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} GFS predictor rows -> {out}")
    return 0
