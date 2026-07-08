"""NBM benchmark forecasts at station points (the target we must beat).

The National Blend of Models is NOAA's calibrated, bias-corrected operational guidance —
the strongest public benchmark for point forecasts. We compare our post-processor against
NBM at held-out mountain stations.

Access: Open-Meteo exposes NBM via the ``ncep_nbm_conus`` model with hourly output and a
historical archive (the forecast as it was issued on past dates), which lets us line NBM
up against our forecasts at matched valid times. This avoids wrangling NBM GRIB2 from AWS
for the verification-only use case. (The raw-HRRR and persistence baselines come for free
from our own extraction + obs, so they live in verify.py, not here.)

We pull the *archived* hourly NBM temperature/wind for the verification window at each
station and normalize to the same schema as our forecasts for a like-for-like MAE/CRPS
comparison. Note Open-Meteo NBM gives a deterministic hourly value (no native quantiles),
so NBM enters the CRPS comparison as a degenerate (point) forecast — standard practice.
"""
from __future__ import annotations

import json
import time
from datetime import date
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m")

# Open-Meteo var -> our obs/target column name.
_RENAME = {
    "temperature_2m": "nbm_air_temp_c",
    "relative_humidity_2m": "nbm_relative_humidity_pct",
    "wind_speed_10m": "nbm_wind_speed_ms",
    "wind_gusts_10m": "nbm_wind_gust_ms",
}


def _http_json(url: str, *, timeout: int = 60, retries: int = 3) -> dict:
    req = Request(url, headers={"User-Agent": "mtnwx/0.1"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt + 1)
    assert last is not None
    raise last


def fetch_nbm_hourly(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Archived hourly NBM forecast at (lat, lon) over [start, end].

    Uses the Open-Meteo archive endpoint with the NBM model. Wind speed/gust are
    requested in m/s to match our schema. Returns columns: valid_time + nbm_* fields.
    """
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "models": "ncep_nbm_conus",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    payload = _http_json(f"{ARCHIVE_URL}?" + urlencode(params))
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=["valid_time", *_RENAME.values()])

    out = pd.DataFrame({"valid_time": pd.to_datetime(hourly["time"])})
    for src, dst in _RENAME.items():
        vals = hourly.get(src)
        out[dst] = pd.Series(vals, dtype="float64") if vals is not None else np.nan
    return out


def fetch_nbm_for_stations(stations: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """NBM hourly forecasts for every station over the verification window."""
    frames = []
    n = len(stations)
    for i, (_, s) in enumerate(stations.iterrows(), 1):
        try:
            df = fetch_nbm_hourly(float(s["lat"]), float(s["lon"]), start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: NBM fetch failed for {s['station_id']}: {exc}")
            continue
        if not df.empty:
            df["station_id"] = s["station_id"]
            frames.append(df)
        if i % 25 == 0 or i == n:
            print(f"  NBM: {i}/{n} stations")
        time.sleep(0.2)  # be gentle with the free API
    if not frames:
        return pd.DataFrame(columns=["station_id", "valid_time", *_RENAME.values()])
    return pd.concat(frames, ignore_index=True)
