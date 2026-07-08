"""Hourly surface observations — the ground truth the post-processor is trained and
verified against.

The HRRR post-processor predicts hourly, so obs must be hourly too. Three sources,
each fetched at native hourly resolution and normalized to a common schema:

  columns: station_id, valid_time (UTC, hourly), air_temp_c, dewpoint_c,
           relative_humidity_pct, wind_speed_ms, wind_gust_ms, wind_dir_deg,
           precip_1h_mm, source

  - **SNOTEL** (NRCS AWDB, HOURLY duration): TOBS (air temp °F), PREC (accumulated
    precip in) differenced to hourly increments. No wind (SNOTEL has none).
  - **ASOS/RAWS** (Iowa Environmental Mesonet bulk archive): temp/dewpoint/wind/gust
    + hourly precip — the wind ground truth SNOTEL lacks.
  - **Synoptic Data** (timeseries API, optional token): fills remaining mesonets.

All fetchers are resumable via a per-station on-disk cache (parquet), so the M3-scale
backfill only ever pulls new hours. QC lives in ``qc()``.
"""
from __future__ import annotations

import io
import json
import time
from datetime import date, datetime, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
SYNOPTIC_TS = "https://api.synopticdata.com/v2/stations/timeseries"

F_TO_C = lambda f: (f - 32.0) * 5.0 / 9.0  # noqa: E731
IN_TO_MM = 25.4
MPH_TO_MS = 0.44704
KT_TO_MS = 0.514444

OBS_COLUMNS = [
    "station_id",
    "valid_time",
    "air_temp_c",
    "dewpoint_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_gust_ms",
    "wind_dir_deg",
    "precip_1h_mm",
    "source",
]


def _http(url: str, *, timeout: int = 90, retries: int = 3) -> bytes:
    req = Request(url, headers={"User-Agent": "mtnwx/0.1"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt + 1)
    assert last is not None
    raise last


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OBS_COLUMNS)


# --------------------------------------------------------------------------- SNOTEL


def fetch_snotel_hourly(triplet: str, start: date, end: date) -> pd.DataFrame:
    """Hourly SNOTEL obs for one station triplet.

    Elements: TOBS (observed air temp °F) -> air_temp_c; PREC (season-accumulated
    precip, in) -> hourly increment precip_1h_mm (clamped at 0 to drop the sensor's
    end-of-season resets). SNOTEL has no wind or dewpoint.
    """
    params = {
        "stationTriplets": triplet,
        "elements": "TOBS,PREC",
        "duration": "HOURLY",
        "beginDate": start.isoformat(),
        "endDate": end.isoformat(),
    }
    raw = _http(f"{AWDB}/data?" + urlencode(params))
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        return _empty()

    series: dict[str, dict[str, float]] = {}
    for el in payload[0].get("data", []):
        code = el.get("stationElement", {}).get("elementCode")
        for v in el.get("values", []):
            ts = v.get("date")
            val = v.get("value")
            if ts is None or val is None:
                continue
            series.setdefault(ts, {})[code] = float(val)

    if not series:
        return _empty()
    df = pd.DataFrame.from_dict(series, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)

    out = pd.DataFrame(index=df.index)
    out["air_temp_c"] = F_TO_C(df["TOBS"]) if "TOBS" in df else np.nan
    # PREC is cumulative for the water year; hourly increment = positive diff.
    if "PREC" in df:
        inc = df["PREC"].diff() * IN_TO_MM
        out["precip_1h_mm"] = inc.clip(lower=0.0)
    else:
        out["precip_1h_mm"] = np.nan

    out = out.reset_index(names="valid_time")
    out["station_id"] = triplet
    out["source"] = "SNOTEL"
    for c in OBS_COLUMNS:
        if c not in out:
            out[c] = np.nan
    return out[OBS_COLUMNS]


# --------------------------------------------------------------------------- IEM ASOS/RAWS


def fetch_iem_hourly(iem_id: str, network: str, start: date, end: date) -> pd.DataFrame:
    """Hourly ASOS/AWOS/RAWS obs from the Iowa Environmental Mesonet bulk archive.

    ``iem_id`` is the 3-4 char station id (no leading K); ``network`` is an IEM
    network code (e.g. 'CO_ASOS', 'AZ_RWIS'). Returns temp/dewpoint/RH/wind/gust and
    hourly precip.
    """
    params = {
        "station": iem_id,
        "data": "tmpf,dwpf,relh,sknt,gust,p01i",
        "network": network,
        "tz": "UTC",
        "format": "onlycomma",
        "missing": "empty",
        "sts": start.isoformat(),
        "ets": (end + timedelta(days=1)).isoformat(),
    }
    raw = _http(f"{IEM}?" + urlencode(params))
    text = raw.decode("utf-8", errors="replace")
    if not text.strip() or "station,valid" not in text:
        return _empty()
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return _empty()

    out = pd.DataFrame()
    out["valid_time"] = pd.to_datetime(df["valid"], utc=True).dt.tz_localize(None)
    out["air_temp_c"] = F_TO_C(pd.to_numeric(df.get("tmpf"), errors="coerce"))
    out["dewpoint_c"] = F_TO_C(pd.to_numeric(df.get("dwpf"), errors="coerce"))
    out["relative_humidity_pct"] = pd.to_numeric(df.get("relh"), errors="coerce")
    out["wind_speed_ms"] = pd.to_numeric(df.get("sknt"), errors="coerce") * KT_TO_MS
    out["wind_gust_ms"] = pd.to_numeric(df.get("gust"), errors="coerce") * KT_TO_MS
    out["precip_1h_mm"] = pd.to_numeric(df.get("p01i"), errors="coerce") * IN_TO_MM
    # Resample sub-hourly METARs to the top of each hour (mean temp, max gust/precip).
    out = out.set_index("valid_time")
    agg = {
        "air_temp_c": "mean", "dewpoint_c": "mean", "relative_humidity_pct": "mean",
        "wind_speed_ms": "mean", "wind_gust_ms": "max", "precip_1h_mm": "max",
    }
    out = out.resample("1h").agg(agg).dropna(how="all").reset_index()
    out["station_id"] = f"IEM:{iem_id}"
    out["source"] = "IEM"
    out["wind_dir_deg"] = np.nan
    for c in OBS_COLUMNS:
        if c not in out:
            out[c] = np.nan
    return out[OBS_COLUMNS]


# --------------------------------------------------------------------------- Synoptic


def fetch_synoptic_hourly(stid: str, token: str, start: date, end: date) -> pd.DataFrame:
    """Hourly obs from the Synoptic timeseries API for stations not covered above."""
    params = {
        "token": token,
        "stid": stid,
        "start": start.strftime("%Y%m%d0000"),
        "end": end.strftime("%Y%m%d2359"),
        "vars": "air_temp,dew_point_temperature,relative_humidity,wind_speed,wind_gust,wind_direction,precip_accum_one_hour",
        "units": "metric,speed|ms",
        "obtimezone": "utc",
        "output": "json",
    }
    raw = _http(f"{SYNOPTIC_TS}?" + urlencode(params))
    payload = json.loads(raw.decode("utf-8"))
    stations = payload.get("STATION", []) if isinstance(payload, dict) else []
    if not stations:
        return _empty()
    obs = stations[0].get("OBSERVATIONS", {})
    times = obs.get("date_time", [])
    if not times:
        return _empty()

    def col(key: str) -> pd.Series:
        # Synoptic suffixes variables with _set_1 etc.; pick the first matching set.
        for k, v in obs.items():
            if k.startswith(key) and k != "date_time":
                return pd.Series(v, dtype="float64")
        return pd.Series([np.nan] * len(times))

    out = pd.DataFrame()
    out["valid_time"] = pd.to_datetime(times, utc=True).tz_localize(None)
    out["air_temp_c"] = col("air_temp")
    out["dewpoint_c"] = col("dew_point_temperature")
    out["relative_humidity_pct"] = col("relative_humidity")
    out["wind_speed_ms"] = col("wind_speed")
    out["wind_gust_ms"] = col("wind_gust")
    out["wind_dir_deg"] = col("wind_direction")
    out["precip_1h_mm"] = col("precip_accum_one_hour")
    out = out.set_index("valid_time").resample("1h").mean().dropna(how="all").reset_index()
    out["station_id"] = f"SYN:{stid}"
    out["source"] = "SYNOPTIC"
    for c in OBS_COLUMNS:
        if c not in out:
            out[c] = np.nan
    return out[OBS_COLUMNS]


# --------------------------------------------------------------------------- QC

# Physical plausibility bounds (mountain-appropriate). Values outside -> NaN.
QC_BOUNDS = {
    "air_temp_c": (-60.0, 55.0),
    "dewpoint_c": (-70.0, 40.0),
    "relative_humidity_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, 110.0),
    "wind_gust_ms": (0.0, 130.0),
    "wind_dir_deg": (0.0, 360.0),
    "precip_1h_mm": (0.0, 250.0),
}

# Max believable hour-to-hour step (spike / stuck-sensor screen).
QC_MAX_STEP = {
    "air_temp_c": 20.0,
    "dewpoint_c": 20.0,
    "wind_speed_ms": 60.0,
}


def qc(df: pd.DataFrame, *, persistence_hours: int = 24) -> pd.DataFrame:
    """Quality-control an obs frame in place-safe fashion (returns a new frame).

    Applies range checks, hour-to-hour step limits, and a flatline/stuck-sensor
    screen (a value repeated unchanged for ``persistence_hours`` is nulled — a
    common failure mode of unheated mountain sensors). Dewpoint is also forced
    <= air temp. Rows with all measurements nulled are dropped.
    """
    if df.empty:
        return df
    out = df.copy().sort_values(["station_id", "valid_time"]).reset_index(drop=True)

    for col, (lo, hi) in QC_BOUNDS.items():
        if col in out:
            out.loc[(out[col] < lo) | (out[col] > hi), col] = np.nan

    # Dewpoint cannot exceed air temp (allow tiny sensor slop).
    if "dewpoint_c" in out and "air_temp_c" in out:
        bad = out["dewpoint_c"] > out["air_temp_c"] + 1.0
        out.loc[bad, "dewpoint_c"] = np.nan

    # Per-station step and flatline screens. Operate on positional slices so the
    # grouping column (station_id) is never consumed by groupby-apply.
    for _, idx in out.groupby("station_id").groups.items():
        g = out.loc[idx]
        for col, step in QC_MAX_STEP.items():
            if col in g:
                jump = g[col].diff().abs() > step
                out.loc[idx[jump.values], col] = np.nan
        # Flatline: a run of identical values >= persistence_hours -> null the run.
        for col in ("air_temp_c", "wind_speed_ms", "relative_humidity_pct"):
            if col in g and g[col].notna().sum() > persistence_hours:
                same = g[col].eq(g[col].shift())
                run = same.groupby((~same).cumsum()).cumcount() + 1
                out.loc[idx[(run >= persistence_hours).values], col] = np.nan

    meas = [c for c in OBS_COLUMNS if c not in ("station_id", "valid_time", "source")]
    out = out.dropna(subset=meas, how="all").reset_index(drop=True)
    return out


def normalize_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Snap valid_time to the top of the hour and ensure column order/dtypes."""
    if df.empty:
        return df
    out = df.copy()
    out["valid_time"] = pd.to_datetime(out["valid_time"]).dt.floor("h")
    out = out.drop_duplicates(["station_id", "valid_time"]).reset_index(drop=True)
    return out[OBS_COLUMNS]
