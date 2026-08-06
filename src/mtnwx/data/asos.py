"""ASOS/AWOS station obs + selection from the dynamical.org ASOS-parquet dataset.

A clean, hourly, year-partitioned GeoParquet of global airport observations (1940–present)
served at ``data.source.coop/dynamical/asos-parquet``, queryable with DuckDB. It gives us
two things SNOTEL can't:

  1. **Wind & gust ground truth** — SNOTEL has no anemometer; ASOS airports do. This is
     what makes the wind-speed and gust models trainable and verifiable.
  2. A second, independent network for the benchmark — beating NBM at airports *and*
     snow-telemetry sites is a broader claim than SNOTEL alone.

Columns (already metric): tmpc, dwpc, relh, sknt (kt), gust (kt), p01m (mm), drct,
latitude, longitude, elevation (m), state. We normalize to the same schema as
``obs.py`` so ASOS rows drop straight into the training/verification pipeline.

Selection reuses the region's mountain filter (elevation + relief): many western-US
airports sit in mountain valleys and on plateaus and are exactly the complex-terrain
points we want.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from mtnwx.data.obs import KT_TO_MS, OBS_COLUMNS

ASOS_BASE = "https://data.source.coop/dynamical/asos-parquet"


def _con():
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con


def _year_urls(start: date, end: date) -> list[str]:
    return [f"{ASOS_BASE}/year={y}/data.parquet" for y in range(start.year, end.year + 1)]


def select_stations(bounds: dict, start_year: int = 2019) -> pd.DataFrame:
    """Distinct ASOS stations inside the region box, with lat/lon/elevation/state.

    Reads one recent year (station set is stable) and filters by the region bbox
    server-side. Elevation/relief mountain filtering is applied later in stations.py /
    terrain.py, same as SNOTEL — here we just enumerate in-region airports."""
    url = f"{ASOS_BASE}/year={start_year}/data.parquet"
    con = _con()
    q = f"""
        SELECT station,
               any_value(latitude)  AS lat,
               any_value(longitude) AS lon,
               any_value(elevation) AS elevation_m,
               any_value(state)     AS state,
               any_value(name)      AS name
        FROM read_parquet('{url}')
        WHERE longitude BETWEEN {bounds['lon_min']} AND {bounds['lon_max']}
          AND latitude  BETWEEN {bounds['lat_min']} AND {bounds['lat_max']}
        GROUP BY station
    """
    df = con.execute(q).fetchdf()
    df["station_id"] = "ASOS:" + df["station"].astype(str)
    df["network"] = "ASOS"
    return df[["station_id", "name", "network", "state", "lat", "lon", "elevation_m"]]


def fetch_asos_hourly(station_ids: list[str], start: date, end: date) -> pd.DataFrame:
    """Hourly ASOS obs for the given stations over [start, end], normalized to OBS_COLUMNS.

    ``station_ids`` are our ``ASOS:<id>`` ids; we strip the prefix for the query. METAR
    reports are sub-hourly, so we aggregate to the top of each hour (mean temp/dewpoint/
    RH/wind, max gust/precip) to match HRRR's hourly leads."""
    raw_ids = [s.split("ASOS:", 1)[-1] for s in station_ids]
    if not raw_ids:
        return pd.DataFrame(columns=OBS_COLUMNS)
    in_list = ",".join("'" + i.replace("'", "") + "'" for i in raw_ids)
    urls = _year_urls(start, end)
    url_list = "[" + ",".join(f"'{u}'" for u in urls) + "]"
    con = _con()
    q = f"""
        SELECT station,
               valid,
               tmpc, dwpc, relh, sknt, gust, drct, p01m
        FROM read_parquet({url_list}, hive_partitioning=true, union_by_name=true)
        WHERE station IN ({in_list})
          AND valid >= TIMESTAMP '{start.isoformat()} 00:00:00'
          AND valid <  TIMESTAMP '{end.isoformat()} 23:59:59'
    """
    raw = con.execute(q).fetchdf()
    if raw.empty:
        return pd.DataFrame(columns=OBS_COLUMNS)

    out = pd.DataFrame()
    # `valid` is tz-aware (station local); convert to UTC naive to match our schema.
    vt = pd.to_datetime(raw["valid"], utc=True).dt.tz_localize(None)
    out["valid_time"] = vt
    out["station_id"] = "ASOS:" + raw["station"].astype(str)
    out["air_temp_c"] = pd.to_numeric(raw["tmpc"], errors="coerce")
    out["dewpoint_c"] = pd.to_numeric(raw["dwpc"], errors="coerce")
    out["relative_humidity_pct"] = pd.to_numeric(raw["relh"], errors="coerce")
    out["wind_speed_ms"] = pd.to_numeric(raw["sknt"], errors="coerce") * KT_TO_MS
    out["wind_gust_ms"] = pd.to_numeric(raw["gust"], errors="coerce") * KT_TO_MS
    out["wind_dir_deg"] = pd.to_numeric(raw["drct"], errors="coerce")
    out["precip_1h_mm"] = pd.to_numeric(raw["p01m"], errors="coerce")
    out["source"] = "ASOS"

    # Aggregate sub-hourly METARs to the top of each hour. Floor-then-groupby (NOT
    # groupby.resample): resample fills every empty hour across each station's full
    # multi-year span, exploding to billions of rows and hanging. Grouping on the
    # floored hour only emits hours that actually have data — vectorized and fast.
    out["valid_time"] = out["valid_time"].dt.floor("h")
    agg = {
        "air_temp_c": "mean", "dewpoint_c": "mean", "relative_humidity_pct": "mean",
        "wind_speed_ms": "mean", "wind_gust_ms": "max", "wind_dir_deg": "mean",
        "precip_1h_mm": "max",
    }
    out = out.groupby(["station_id", "valid_time"], as_index=False).agg(agg)
    for c in OBS_COLUMNS:
        if c not in out:
            out[c] = np.nan
    out["source"] = "ASOS"
    return out[OBS_COLUMNS]
