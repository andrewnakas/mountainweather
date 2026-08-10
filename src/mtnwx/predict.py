"""Generate live forecasts from the latest HRRR cycle using the trained models.

Operational path (runs hourly in GitHub Actions):
  1. Open the dynamical.org HRRR archive and take the most recent init cycle.
  2. Extract predictor fields at every station (same code as the backfill).
  3. Build the feature table (terrain + time + derived predictors) — no obs join, since
     we're forecasting the future.
  4. For each target, load the LightGBM quantile boosters and predict the point (q0.50)
     and quantile band at every station and lead hour.
  5. Emit compact JSON/GeoJSON for the website: per-station hourly forecast with bands.

Models are loaded from the local models dir (populated from HF in CI). The same
build_* feature functions as training guarantee train/serve feature parity.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from mtnwx.config import data_dir, load_configs
from mtnwx.data import hrrr
from mtnwx.features.build import (
    add_derived_predictors,
    add_terrain_features,
    add_time_features,
)

TARGET_UNITS = {
    "air_temp_c": "C",
    "wind_speed_ms": "m/s",
    "wind_gust_ms": "m/s",
    "relative_humidity_pct": "%",
    "precip_1h_mm": "mm/h",
}


def load_models(models_dir: Path):
    """Load {target: {quantile: booster}} and metadata from a models dir."""
    import lightgbm as lgb

    meta = json.loads((models_dir / "metadata.json").read_text())
    models: dict[str, dict[float, object]] = {}
    for target in meta["targets"]:
        p = models_dir / f"{target}.pkl"
        if not p.exists():
            continue
        blob = pickle.loads(p.read_bytes())
        models[target] = {float(q): lgb.Booster(model_str=s) for q, s in blob.items()}
    return models, meta


def latest_init(ds) -> pd.Timestamp:
    return pd.to_datetime(ds.init_time.values).max()


def build_forecast_features(stations: pd.DataFrame, init: pd.Timestamp) -> pd.DataFrame:
    """Extract predictors at the given init and assemble the (obs-free) feature table."""
    import xarray as xr

    ds = hrrr.open_archive()
    cfg = load_configs()
    fields = cfg["predictors"]["hrrr_fields"]
    yi, xi, ok, dist = hrrr.build_grid_index(ds, stations, data_dir() / "hrrr_grid_index.json")
    st = stations.loc[ok].reset_index(drop=True)
    y_da = xr.DataArray(yi[ok], dims="station")
    x_da = xr.DataArray(xi[ok], dims="station")

    hx = hrrr._extract_one_init(ds, init, fields, y_da, x_da)
    hx["station_id"] = st["station_id"].to_numpy()[hx["station_ix"].to_numpy()]
    hx = hx.drop(columns="station_ix")

    df = add_derived_predictors(hx)
    df = add_time_features(df, st)
    df = add_terrain_features(df, st)
    return df


def build_forecast_features_gfs(stations: pd.DataFrame, init: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """GLOBAL live path: build the (obs-free) feature table from the latest GFS cycle,
    optionally augmented with ECMWF AIFS, using the SAME derived-feature functions as the
    global training table so train/serve features match. Returns (df, init)."""
    import sys as _sys

    # Reuse the global training-table feature builders (single source of truth).
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))
    from build_training_table_global import add_gfs_derived  # noqa: E402
    from mtnwx.data import gfs

    import xarray as xr

    ds = gfs.open_archive()
    if init is None:
        init = pd.to_datetime(ds.init_time.values).max()
    lat_da = xr.DataArray(stations["lat"].to_numpy(), dims="station")
    lon_da = xr.DataArray(stations["lon"].to_numpy(), dims="station")
    lead_h_all = (ds.lead_time.values / np.timedelta64(1, "h")).astype("int32")
    ds = ds.isel(lead_time=np.flatnonzero(lead_h_all <= 48))

    gx = gfs._extract_one_init(ds, init, gfs.GFS_VARS, lat_da, lon_da)
    gx["station_id"] = stations["station_id"].to_numpy()[gx["station_ix"].to_numpy()]
    gx = gx.drop(columns="station_ix")

    df = add_gfs_derived(gx)                 # gfs_wind_speed/dir, gfs_precip_mm, valid_time
    df = add_time_features(df, stations)     # lat/lon + solar + time encodings
    df = add_terrain_features(df, stations)  # elevation + terrain
    # Rename gfs_wind_dir_10m -> wind_dir_10m so the viewer's arrow layer finds it.
    if "gfs_wind_dir_10m" in df.columns and "wind_dir_10m" not in df.columns:
        df["wind_dir_10m"] = df["gfs_wind_dir_10m"]
    return df, init


def predict_table(df: pd.DataFrame, models: dict, meta: dict) -> pd.DataFrame:
    """Predict point + quantiles for every target; return a long forecast frame."""
    feats = meta["features"]
    # Ensure every training feature is present (fill missing with NaN -> LGBM handles).
    for f in feats:
        if f not in df.columns:
            df[f] = np.nan
    X = df[feats].astype("float32")

    keep = ["station_id", "init_time", "lead_hour", "valid_time"]
    # Carry wind direction through (an input feature, not a corrected target) so the
    # viewer can draw the wind field's direction.
    if "wind_dir_10m" in df.columns:
        keep.append("wind_dir_10m")
    out = df[keep].copy()
    for target, boosters in models.items():
        for q, booster in boosters.items():
            out[f"{target}_q{int(q * 100):02d}"] = booster.predict(X).astype("float32")
        # Point forecast = median quantile.
        if 0.5 in boosters:
            out[f"{target}"] = out[f"{target}_q50"]
    return out


def to_station_json(forecast: pd.DataFrame, stations: pd.DataFrame, models: dict, init: pd.Timestamp,
                    generated_from: str = "HRRR via dynamical.org, post-processed by mtnwx",
                    lean: bool = False) -> dict:
    """Per-station hourly forecast JSON.

    ``lean=True`` emits a viewer-optimized payload: point values only (no q10/q90 bands or
    per-var units — the map never reads them), and a single top-level ``valid_times`` shared
    by all features (they're identical hourly sequences). This shrinks the file ~5-6x, which
    at 26k stations is the difference between a 166 MB unusable download and a ~30 MB one."""
    meta_st = stations.set_index("station_id")
    features = []
    shared_times = None
    for sid, g in forecast.groupby("station_id"):
        g = g.sort_values("lead_hour")
        s = meta_st.loc[sid] if sid in meta_st.index else None
        if lean and shared_times is None:
            shared_times = [t.isoformat() for t in g["valid_time"]]
        series = {}
        for target in models:
            entry = {"point": [_r(v) for v in g.get(target, pd.Series([np.nan] * len(g)))]}
            if not lean:
                if f"{target}_q10" in g and f"{target}_q90" in g:
                    entry["q10"] = [_r(v) for v in g[f"{target}_q10"]]
                    entry["q90"] = [_r(v) for v in g[f"{target}_q90"]]
                entry["units"] = TARGET_UNITS.get(target, "")
            series[target] = entry
        # Wind direction (deg from N) — needed to draw the wind field/arrows.
        if "wind_dir_10m" in g.columns:
            wd = {"point": [_r(v) for v in g["wind_dir_10m"]]}
            if not lean:
                wd["units"] = "deg"
            series["wind_dir_deg"] = wd
        props = {
            "station_id": sid,
            "name": (s["name"] if s is not None and "name" in s else sid),
            "elevation_m": (round(float(s["elevation_m"]), 1) if s is not None and pd.notna(s["elevation_m"]) else None),
            "forecast": series,
        }
        if not lean:  # in lean mode valid_times is shared at the top level
            props["valid_times"] = [t.isoformat() for t in g["valid_time"]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [round(float(s["lon"]), 4), round(float(s["lat"]), 4)] if s is not None else [None, None]},
            "properties": props,
        })
    out = {
        "type": "FeatureCollection",
        "model": "mtnwx",
        "init_time": init.isoformat(),
        "generated_from": generated_from,
        "features": features,
    }
    if lean and shared_times is not None:
        out["valid_times"] = shared_times  # shared by all features (identical hourly seq)
    return out


def _r(v):
    return None if pd.isna(v) else round(float(v), 2)


def main(args: argparse.Namespace) -> int:
    base = getattr(args, "base", "hrrr")
    default_stations = "stations_terrain_global.parquet" if base == "gfs" else "stations_terrain.parquet"
    stations_path = Path(args.stations) if args.stations else data_dir() / default_stations
    if not stations_path.exists():
        alt = "stations_global.parquet" if base == "gfs" else "stations.parquet"
        stations_path = data_dir() / alt
    stations = pd.read_parquet(stations_path)
    models_dir = Path(args.models) if args.models else data_dir() / "models"
    models, meta = load_models(models_dir)
    if not models:
        print(f"ERROR: no models in {models_dir}")
        return 1

    init_arg = pd.Timestamp(args.init) if args.init else None
    lean = base == "gfs"
    if base == "gfs":
        # GLOBAL path: latest GFS cycle (+ ECMWF where present), global models/stations.
        # Thin to a ~1° spatial grid: the full 26k stations make the viewer's IDW field +
        # particles lag badly and blow the JSON to 166 MB. One station per ~1° cell keeps a
        # smooth global field (~8.9k stations, ~12 MB) — the map interpolates either way, and
        # the full 26k is still used for training/verification, just not the live map.
        stations = _thin_spatial(stations, cell_deg=1.0)
        print(f"Forecasting (GLOBAL/GFS base) for {len(stations)} thinned stations")
        feats, init = build_forecast_features_gfs(stations, init_arg)
        gen_from = "GFS (+ECMWF AIFS) via dynamical.org, post-processed by mtnwx (global)"
    else:
        ds = hrrr.open_archive()
        init = init_arg if init_arg is not None else latest_init(ds)
        print(f"Forecasting from HRRR init {init} for {len(stations)} stations")
        feats = build_forecast_features(stations, init)
        gen_from = "HRRR via dynamical.org, post-processed by mtnwx"

    fc = predict_table(feats, models, meta)
    payload = to_station_json(fc, stations, models, init, generated_from=gen_from, lean=lean)

    out = Path(args.out) if args.out else Path("site") / "forecast.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    sz = out.stat().st_size / 1e6
    print(f"Wrote forecast for {len(payload['features'])} stations -> {out} ({sz:.1f} MB)")
    return 0


def _thin_spatial(stations: pd.DataFrame, cell_deg: float = 0.4) -> pd.DataFrame:
    """Keep at most one station per (lat,lon) grid cell — spatial thinning that preserves
    global coverage while cutting dense clusters. Prefers the lowest-elevation (airport-
    like) station per cell for stable obs, deterministic (sorted)."""
    df = stations.copy()
    df["_cy"] = (df["lat"] / cell_deg).round().astype(int)
    df["_cx"] = (df["lon"] / cell_deg).round().astype(int)
    df = df.sort_values(["_cy", "_cx", "elevation_m", "station_id"])
    df = df.drop_duplicates(["_cy", "_cx"], keep="first")
    return df.drop(columns=["_cy", "_cx"]).reset_index(drop=True)
