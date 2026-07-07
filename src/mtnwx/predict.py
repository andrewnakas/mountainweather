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


def predict_table(df: pd.DataFrame, models: dict, meta: dict) -> pd.DataFrame:
    """Predict point + quantiles for every target; return a long forecast frame."""
    feats = meta["features"]
    # Ensure every training feature is present (fill missing with NaN -> LGBM handles).
    for f in feats:
        if f not in df.columns:
            df[f] = np.nan
    X = df[feats].astype("float32")

    out = df[["station_id", "init_time", "lead_hour", "valid_time"]].copy()
    for target, boosters in models.items():
        for q, booster in boosters.items():
            out[f"{target}_q{int(q * 100):02d}"] = booster.predict(X).astype("float32")
        # Point forecast = median quantile.
        if 0.5 in boosters:
            out[f"{target}"] = out[f"{target}_q50"]
    return out


def to_station_json(forecast: pd.DataFrame, stations: pd.DataFrame, models: dict, init: pd.Timestamp) -> dict:
    """Compact JSON: per-station hourly forecast with point + q10/q90 band."""
    meta_st = stations.set_index("station_id")
    features = []
    for sid, g in forecast.groupby("station_id"):
        g = g.sort_values("lead_hour")
        s = meta_st.loc[sid] if sid in meta_st.index else None
        series = {}
        for target in models:
            entry = {
                "point": [_r(v) for v in g.get(target, pd.Series([np.nan] * len(g)))],
            }
            if f"{target}_q10" in g and f"{target}_q90" in g:
                entry["q10"] = [_r(v) for v in g[f"{target}_q10"]]
                entry["q90"] = [_r(v) for v in g[f"{target}_q90"]]
            entry["units"] = TARGET_UNITS.get(target, "")
            series[target] = entry
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(s["lon"]), float(s["lat"])] if s is not None else [None, None],
                },
                "properties": {
                    "station_id": sid,
                    "name": (s["name"] if s is not None and "name" in s else sid),
                    "elevation_m": (float(s["elevation_m"]) if s is not None else None),
                    "valid_times": [t.isoformat() for t in g["valid_time"]],
                    "forecast": series,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "model": "mtnwx",
        "init_time": init.isoformat(),
        "generated_from": "HRRR via dynamical.org, post-processed by mtnwx",
        "features": features,
    }


def _r(v):
    return None if pd.isna(v) else round(float(v), 2)


def main(args: argparse.Namespace) -> int:
    stations_path = Path(args.stations) if args.stations else data_dir() / "stations_terrain.parquet"
    if not stations_path.exists():
        stations_path = data_dir() / "stations.parquet"
    stations = pd.read_parquet(stations_path)
    models_dir = Path(args.models) if args.models else data_dir() / "models"
    models, meta = load_models(models_dir)
    if not models:
        print(f"ERROR: no models in {models_dir}")
        return 1

    ds = hrrr.open_archive()
    init = pd.Timestamp(args.init) if args.init else latest_init(ds)
    print(f"Forecasting from HRRR init {init} for {len(stations)} stations")

    feats = build_forecast_features(stations, init)
    fc = predict_table(feats, models, meta)
    payload = to_station_json(fc, stations, models, init)

    out = Path(args.out) if args.out else Path("site") / "forecast.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"Wrote forecast for {len(payload['features'])} stations -> {out}")
    return 0
