"""Mountain station catalogue.

Builds the set of stations `mtnwx` trains and forecasts on. Sources:

  - **SNOTEL** (NRCS AWDB REST API) — the backbone: ~900 high-elevation snow-telemetry
    sites across the western US, all in complex terrain. No API key needed.
  - **Synoptic Data** (optional, needs ``SYNOPTIC_API_TOKEN``) — RAWS, mountain
    ASOS/AWOS, and ski-area mesonets, for wind/temp coverage SNOTEL lacks.

Stations are filtered to "mountain" by elevation and local relief (see
``configs/region.yaml``). Relief is computed later against the DEM in
``terrain.py``; here we apply the elevation floor and the region bounding box, and
carry a ``needs_relief_check`` flag. Output is a parquet catalogue consumed by every
downstream stage.

The AWDB fetch pattern is adapted from the sibling SnowWatch project
(``app/snotel.py``), which hardened it against AWDB's intermittent SSL timeouts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from mtnwx.config import data_dir, load_configs

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
SYNOPTIC_META = "https://api.synopticdata.com/v2/stations/metadata"

FT_TO_M = 0.3048


def _http_json(url: str, *, timeout: int = 60, retries: int = 3) -> object:
    """GET JSON with bounded exponential-backoff retries.

    AWDB intermittently SSL-handshake-times-out under load; a couple of short
    retries turn transient failures into successes (learned in SnowWatch CI)."""
    req = Request(url, headers={"User-Agent": "mtnwx/0.1"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — retry any transport error
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt + 1)
    assert last is not None
    raise last


def fetch_snotel_stations() -> list[dict]:
    """Active SNOTEL (SNTL network) sites with lat/lon/elevation."""
    url = (
        f"{AWDB}/stations?networkCds=SNTL"
        "&returnForecastPointMetadata=false"
        "&returnReservoirMetadata=false"
        "&returnStationElements=false"
    )
    payload = _http_json(url)
    today = date.today().isoformat()
    out: list[dict] = []
    for s in payload if isinstance(payload, list) else []:
        if s.get("networkCode") != "SNTL":
            continue
        lat, lon = s.get("latitude"), s.get("longitude")
        if lat is None or lon is None:
            continue
        end_date = (s.get("endDate") or "")[:10]
        # AWDB marks active sites with a far-future end date (~2100-01-01).
        if end_date and end_date < today:
            continue
        elev_ft = s.get("elevation")
        out.append(
            {
                "station_id": (s.get("stationTriplet") or s.get("triplet") or "").strip(),
                "name": s.get("name"),
                "network": "SNOTEL",
                "state": s.get("stateCode"),
                "lat": float(lat),
                "lon": float(lon),
                "elevation_m": float(elev_ft) * FT_TO_M if elev_ft is not None else None,
            }
        )
    return out


def fetch_synoptic_stations(bounds: dict, token: str) -> list[dict]:
    """Mountain-relevant Synoptic networks (RAWS, ASOS/AWOS, mesonets) in-bounds.

    Uses the metadata endpoint with a bounding box. Networks are filtered to the
    ones that add value over SNOTEL: RAWS (fire-weather, ridge/slope wind), and
    ASOS/AWOS (quality-controlled temp/wind at airports near ranges)."""
    bbox = f"{bounds['lon_min']},{bounds['lat_min']},{bounds['lon_max']},{bounds['lat_max']}"
    url = (
        f"{SYNOPTIC_META}?token={token}&bbox={bbox}"
        "&status=active&sensorvars=1&complete=1"
    )
    payload = _http_json(url)
    stations = payload.get("STATION", []) if isinstance(payload, dict) else []
    out: list[dict] = []
    for s in stations:
        try:
            lat, lon = float(s["LATITUDE"]), float(s["LONGITUDE"])
        except (KeyError, TypeError, ValueError):
            continue
        elev_ft = s.get("ELEVATION")
        try:
            elev_m = float(elev_ft) * FT_TO_M if elev_ft is not None else None
        except (TypeError, ValueError):
            elev_m = None
        out.append(
            {
                "station_id": f"SYN:{s.get('STID')}",
                "name": s.get("NAME"),
                "network": (s.get("MNET_SHORTNAME") or s.get("GACC") or "SYNOPTIC"),
                "state": s.get("STATE"),
                "lat": lat,
                "lon": lon,
                "elevation_m": elev_m,
            }
        )
    return out


def apply_mountain_filter(df: pd.DataFrame, region: dict) -> pd.DataFrame:
    """Keep stations inside the region box and above the elevation floor.

    Local-relief filtering is deferred to terrain.py (needs the DEM); rows are
    flagged ``needs_relief_check=True`` so that stage can prune high-but-flat sites."""
    b = region["bounds"]
    f = region["station_filter"]
    in_box = (
        df["lon"].between(b["lon_min"], b["lon_max"])
        & df["lat"].between(b["lat_min"], b["lat_max"])
    )
    high_enough = df["elevation_m"].fillna(-1) >= f["min_elevation_m"]
    out = df[in_box & high_enough].copy()
    out["needs_relief_check"] = True
    return out.reset_index(drop=True)


def build_catalogue(limit: int | None = None) -> pd.DataFrame:
    """Assemble, dedupe, and filter the mountain station catalogue."""
    cfg = load_configs()
    region = cfg["region"]

    rows = fetch_snotel_stations()

    token = os.environ.get("SYNOPTIC_API_TOKEN")
    if token:
        try:
            rows += fetch_synoptic_stations(region["bounds"], token)
        except Exception as exc:  # noqa: BLE001 — Synoptic is optional
            print(f"WARN: Synoptic fetch failed ({exc}); continuing with SNOTEL only")
    else:
        print("NOTE: SYNOPTIC_API_TOKEN not set; SNOTEL-only catalogue")

    df = pd.DataFrame(rows).dropna(subset=["lat", "lon"])
    # Dedupe by rounded location (some networks double-list co-located sensors).
    df["_k"] = list(zip(df["lat"].round(4), df["lon"].round(4)))
    df = df.drop_duplicates("_k").drop(columns="_k")

    df = apply_mountain_filter(df, region)
    df = df.sort_values(["state", "name"]).reset_index(drop=True)
    if limit:
        df = df.head(limit).reset_index(drop=True)
    return df


def main(args: argparse.Namespace) -> int:
    df = build_catalogue(limit=getattr(args, "limit", None))
    out = Path(args.out) if getattr(args, "out", None) else data_dir() / "stations.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    by_net = df["network"].value_counts().to_dict()
    print(f"Wrote {len(df)} mountain stations -> {out}")
    print(f"By network: {by_net}")
    return 0
