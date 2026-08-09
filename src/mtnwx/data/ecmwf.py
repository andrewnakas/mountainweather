"""Extract ECMWF AIFS forecast predictors at station points (the 3rd blend member).

The global model post-processes GFS; adding a second *independent* global model as
predictors gives it the model diversity that a single base can't provide (the same
reasoning that motivated GFS-alongside-HRRR for the US model). ECMWF's **AIFS Single**
is the natural choice: a state-of-the-art AI forecast, global 0.25°, on dynamical.org
from 2024-04-01, a completely different model family from GFS's physics core.

Accessed via ``dynamical_catalog`` (dataset id ``ecmwf-aifs-single-forecast``). Grid is
regular lat/lon, so point extraction is a direct ``.sel(method="nearest")`` like GFS.

Notes vs GFS:
  - AIFS is **6-hourly** (init every 6h, leads every 6h) — the predictor exists only at
    6-hour lead steps; the join to hourly GFS rows is a left merge, so non-6h leads get
    NaN ECMWF features (LightGBM handles missing values natively).
  - AIFS has **no gust** product and **no 2 m RH** — it carries 2 m dewpoint instead, from
    which RH is derived in build_training_table_global.
Output: ecmwf_global/ecmwf_<YYYY-MM>.parquet on the training dataset, columns prefixed
``ecmwf_`` so the global builder merges them as predictors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mtnwx.config import data_dir

# AIFS Single fields to carry as predictors (prefixed ecmwf_ in the output). No gust; no
# 2 m RH (dewpoint carried instead, RH derived downstream).
ECMWF_VARS = [
    "temperature_2m",
    "dew_point_temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
    "precipitation_surface",
    "pressure_surface",
    "total_cloud_cover_atmosphere",
]

DATASET_ID = "ecmwf-aifs-single-forecast"
MAX_LEAD_H = 48  # match the GFS/HRRR training window


def open_archive():
    import dynamical_catalog as dc

    return dc.open(DATASET_ID)


def month_init_times(ds, month: str) -> pd.DatetimeIndex:
    all_inits = pd.to_datetime(ds.init_time.values)
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthBegin(1)
    return all_inits[(all_inits >= start) & (all_inits < end)]


def _extract_one_init(ds, init, fields, lat_da, lon_da, *, retries: int = 4):
    """Long DataFrame for one AIFS init: rows = station x lead, cols = ecmwf_<field>."""
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
        raise RuntimeError(f"AIFS extract failed for init {init}") from last

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
        out[f"ecmwf_{f}"] = sub[f].values.reshape(-1).astype("float32")
    return out


def extract_month(month: str, stations: pd.DataFrame) -> pd.DataFrame:
    """Extract AIFS predictors for every station over one init-month (leads to 48h)."""
    import xarray as xr

    ds = open_archive()
    lat_da = xr.DataArray(stations["lat"].to_numpy(), dims="station")
    lon_da = xr.DataArray(stations["lon"].to_numpy(), dims="station")

    inits = month_init_times(ds, month)
    if len(inits) == 0:
        # AIFS archive starts 2024-04 — earlier months legitimately have no inits.
        print(f"No AIFS inits in {month} (archive starts 2024-04)")
        return pd.DataFrame()

    lead_h_all = (ds.lead_time.values / np.timedelta64(1, "h")).astype("int32")
    keep = lead_h_all <= MAX_LEAD_H
    ds = ds.isel(lead_time=np.flatnonzero(keep))

    print(f"{month}: {len(inits)} AIFS inits x {len(stations)} stations x {len(ECMWF_VARS)} fields")
    frames = []
    for k, it in enumerate(inits, 1):
        frames.append(_extract_one_init(ds, it, ECMWF_VARS, lat_da, lon_da))
        if k % 20 == 0 or k == len(inits):
            print(f"  extracted {k}/{len(inits)} inits", flush=True)

    allf = pd.concat(frames, ignore_index=True)
    allf["station_id"] = stations["station_id"].to_numpy()[allf["station_ix"].to_numpy()]
    allf = allf.drop(columns="station_ix")
    return allf


def main(args: argparse.Namespace) -> int:
    stations_path = (
        Path(args.stations) if args.stations else data_dir() / "stations_terrain_global.parquet"
    )
    if not stations_path.exists():
        alt = data_dir() / "stations_global.parquet"
        stations_path = alt if alt.exists() else stations_path
    if not stations_path.exists():
        print(f"ERROR: no station catalogue at {stations_path}")
        return 1
    stations = pd.read_parquet(stations_path)

    df = extract_month(args.month, stations)
    if df.empty:
        print("No AIFS data extracted.")
        return 0
    out = Path(args.out) if args.out else data_dir() / f"ecmwf_{args.month}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} AIFS predictor rows -> {out}")
    return 0
