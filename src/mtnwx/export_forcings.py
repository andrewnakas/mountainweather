"""Export bias-corrected hourly forcings at SNOTEL points for SnowWatch (M8).

The sibling SnowWatch project drives SNOW-17 and its ML members with raw Open-Meteo/NBM
weather. mtnwx produces *bias-corrected* hourly temperature, precip, and wind at exactly
the SNOTEL points SnowWatch cares about — better forcings should yield better snowpack.

This writes a compact per-station CSV/parquet in a schema SnowWatch can ingest as a new
forcing source or ensemble member:

    triplet, valid_time, tmean_c, precip_mm, wind_speed_ms

Only SNOTEL stations (triplets like ``NNN:ST:SNTL``) are exported. The values are the
mtnwx point (q0.50) forecasts from the latest cycle. SnowWatch consumes the published
file from the mtnwx-models HF repo; nothing is imported across project boundaries.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mtnwx.config import data_dir


def is_snotel(station_id: str) -> bool:
    return station_id.endswith(":SNTL") or station_id.endswith(":SNOTEL")


def build_forcings(forecast_json: dict) -> pd.DataFrame:
    """Flatten the forecast GeoJSON to SnowWatch's forcing schema (SNOTEL only)."""
    rows = []
    for f in forecast_json.get("features", []):
        sid = f["properties"]["station_id"]
        if not is_snotel(sid):
            continue
        times = f["properties"].get("valid_times", [])
        fc = f["properties"].get("forecast", {})
        temp = (fc.get("air_temp_c") or {}).get("point", [])
        wind = (fc.get("wind_speed_ms") or {}).get("point", [])
        # Precip isn't a phase-1 target; emit NaN until the precip model ships (M7).
        for i, t in enumerate(times):
            rows.append(
                {
                    "triplet": sid,
                    "valid_time": t,
                    "tmean_c": temp[i] if i < len(temp) else None,
                    "precip_mm": None,
                    "wind_speed_ms": wind[i] if i < len(wind) else None,
                }
            )
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> int:
    import json

    fpath = Path(args.forecast) if args.forecast else Path("site") / "forecast.json"
    if not fpath.exists():
        print(f"ERROR: forecast not found at {fpath}; run `mtnwx predict` first")
        return 1
    payload = json.loads(fpath.read_text())
    df = build_forcings(payload)
    out = Path(args.out) if args.out else data_dir() / "snowwatch_forcings.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} forcing rows for {df['triplet'].nunique()} SNOTEL stations -> {out}")
    return 0
