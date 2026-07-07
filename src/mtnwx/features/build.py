"""Assemble the training table: HRRR predictors + terrain + time encodings + obs target.

One training row = one (station, init_time, lead_hour) forecast, joined to:
  - the HRRR predictor fields extracted at that station/init/lead,
  - static terrain features for the station (elevation delta, slope, TPI, ...),
  - cyclical time encodings (hour-of-day, day-of-year) and solar elevation,
  - the *observed* value at valid_time = init_time + lead_hour  (the target),
  - the raw HRRR forecast of the target itself (so the model learns a correction).

The join key on the obs side is (station_id, valid_time). Rows without a matching QC'd
observation are dropped — you can't train or verify without ground truth.

Derived predictors added here (not stored at extraction to keep that lossless):
  - wind_speed_10m / wind_dir from u/v,
  - elevation_delta_m = station elevation - HRRR grid elevation (THE key feature),
  - neighborhood stats are left to a later pass (extraction stores nearest cell only
    in v1; the 3x3 window is a v2 enhancement).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Target variable -> the observation column and the raw-HRRR field that forecasts it.
# The raw-HRRR field is both a predictor and the baseline we must beat.
TARGET_SPEC = {
    "air_temp_c": {"hrrr_field": "temperature_2m"},
    "relative_humidity_pct": {"hrrr_field": "relative_humidity_2m"},
    "wind_speed_ms": {"hrrr_field": "wind_speed_10m"},   # derived below
    "wind_gust_ms": {"hrrr_field": "wind_gust_surface"},
}


def add_derived_predictors(hrrr: pd.DataFrame) -> pd.DataFrame:
    """Add wind speed/direction from components and valid_time."""
    out = hrrr.copy()
    if {"wind_u_10m", "wind_v_10m"}.issubset(out.columns):
        out["wind_speed_10m"] = np.hypot(out["wind_u_10m"], out["wind_v_10m"]).astype("float32")
        out["wind_dir_10m"] = (
            (np.degrees(np.arctan2(-out["wind_u_10m"], -out["wind_v_10m"])) + 360.0) % 360.0
        ).astype("float32")
    if {"wind_u_80m", "wind_v_80m"}.issubset(out.columns):
        out["wind_speed_80m"] = np.hypot(out["wind_u_80m"], out["wind_v_80m"]).astype("float32")
    out["valid_time"] = pd.to_datetime(out["init_time"]) + pd.to_timedelta(out["lead_hour"], unit="h")
    return out


def add_time_features(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour/day encodings, lead hour, and solar elevation."""
    out = df.copy()
    vt = pd.to_datetime(out["valid_time"])
    hour = vt.dt.hour + vt.dt.minute / 60.0
    doy = vt.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype("float32")
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    out["lead_hour"] = out["lead_hour"].astype("float32")

    # Solar elevation angle (deg) — drives the diurnal temperature bias in valleys.
    latlon = stations.set_index("station_id")[["lat", "lon"]]
    out = out.merge(latlon, left_on="station_id", right_index=True, how="left")
    out["solar_elev_deg"] = _solar_elevation(vt.values, out["lat"].to_numpy(), out["lon"].to_numpy())
    return out


def _solar_elevation(times: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Approximate solar elevation angle in degrees (NOAA low-precision formula)."""
    t = pd.to_datetime(times)
    doy = t.dayofyear.to_numpy()
    hour_utc = (t.hour + t.minute / 60.0).to_numpy()
    decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365.0)
    # Solar hour angle from UTC hour and longitude.
    solar_time = hour_utc + lon / 15.0
    H = np.radians(15.0 * (solar_time - 12.0))
    latr = np.radians(lat)
    sin_elev = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(H)
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1))).astype("float32")


TERRAIN_FEATURES = [
    "elevation_delta_m", "dem_elevation_m", "local_relief_m",
    "slope_deg", "aspect_deg", "tpi_500m", "tpi_2km", "tpi_10km",
]


def add_terrain_features(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Join static terrain and compute the elevation delta (station - HRRR grid).

    ``elevation_delta_m`` is the single most important mountain predictor: HRRR's 3 km
    orography smooths terrain, so a station well above/below its grid cell has a
    predictable temperature/wind bias the model can learn out."""
    keep = ["station_id", "elevation_m"] + [c for c in stations.columns if c in TERRAIN_FEATURES]
    st = stations[list(dict.fromkeys(keep))].copy()
    out = df.merge(st, on="station_id", how="left")
    # Prefer the grid elevation extracted with the HRRR data; fall back to DEM at grid.
    if "grid_elevation_m" in out.columns:
        grid_elev = out["grid_elevation_m"]
    else:
        grid_elev = out.get("dem_elevation_m")
    if grid_elev is not None and "elevation_m" in out.columns:
        out["elevation_delta_m"] = (out["elevation_m"] - grid_elev).astype("float32")
    return out


def build_training_table(
    hrrr: pd.DataFrame, obs: pd.DataFrame, stations: pd.DataFrame
) -> pd.DataFrame:
    """Join predictors + terrain + time + observations into one training table."""
    df = add_derived_predictors(hrrr)
    df = add_time_features(df, stations)
    df = add_terrain_features(df, stations)

    obs_use = obs.copy()
    obs_use["valid_time"] = pd.to_datetime(obs_use["valid_time"])
    df["valid_time"] = pd.to_datetime(df["valid_time"])

    merged = df.merge(
        obs_use[["station_id", "valid_time", "air_temp_c", "relative_humidity_pct",
                 "wind_speed_ms", "wind_gust_ms", "dewpoint_c", "precip_1h_mm"]],
        on=["station_id", "valid_time"],
        how="inner",
    )
    return merged


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything numeric except identifiers and targets."""
    exclude = {
        "init_time", "valid_time", "station_id", "lat", "lon",
        "air_temp_c", "relative_humidity_pct", "wind_speed_ms", "wind_gust_ms",
        "dewpoint_c", "precip_1h_mm",
    }
    cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols
