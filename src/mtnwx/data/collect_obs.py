"""Collect hourly observations for the whole station catalogue over a date range.

Routes each station to the right fetcher by network, runs QC, and writes a single
tidy parquet (partitionable by month for the M3-scale backfill). This is the M1
deliverable that produces the training/verification ground truth.

Routing:
  - SNOTEL triplets            -> AWDB hourly (temp + precip)
  - IEM:<id> (with iem_network) -> IEM bulk archive (temp/dewpoint/wind/gust/precip)
  - SYN:<stid>                 -> Synoptic timeseries (needs SYNOPTIC_API_TOKEN)

Fetch failures for a single station are logged and skipped, never fatal — a mountain
network of ~1000 sensors always has a few offline.
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from mtnwx.config import data_dir
from mtnwx.data import obs


def _fetch_one(row: pd.Series, start: date, end: date, token: str | None) -> pd.DataFrame:
    sid = str(row["station_id"])
    try:
        if sid.startswith("SYN:"):
            if not token:
                return obs._empty()
            return obs.fetch_synoptic_hourly(sid[4:], token, start, end)
        if sid.startswith("IEM:"):
            net = row.get("iem_network")
            if not net:
                return obs._empty()
            return obs.fetch_iem_hourly(sid[4:], str(net), start, end)
        if sid.startswith("MS:"):
            from mtnwx.data.meteostat import fetch_meteostat_hourly
            return fetch_meteostat_hourly(sid[3:], start, end)
        # Default: a bare SNOTEL triplet like "308:AZ:SNTL".
        if sid.endswith(":SNTL") or sid.count(":") == 2:
            return obs.fetch_snotel_hourly(sid, start, end)
    except Exception as exc:  # noqa: BLE001 — never let one station kill the run
        print(f"WARN: obs fetch failed for {sid}: {exc}")
    return obs._empty()


def collect(
    stations: pd.DataFrame, start: date, end: date, *, token: str | None = None,
    workers: int = 8,
) -> pd.DataFrame:
    """Fetch + QC hourly obs for every station in ``stations`` over [start, end].

    Fetches are network-bound and the obs providers (AWDB/IEM/Synoptic) tolerate
    concurrent requests, so stations are pulled with a bounded thread pool — serial
    collection of 788 stations over 7 years was the slowest stage of the full run."""
    from concurrent.futures import ThreadPoolExecutor

    frames: list[pd.DataFrame] = []

    # ASOS obs come from one bulk DuckDB query for all ASOS stations at once (not
    # per-station), so pull them together rather than through the per-station path.
    is_asos = stations["station_id"].astype(str).str.startswith("ASOS:")
    asos_ids = stations.loc[is_asos, "station_id"].astype(str).tolist()
    if asos_ids:
        # Fetch ASOS year-by-year: one 7-year DuckDB scan is opaque and slow to start;
        # per-year scans make progress visible and bound memory. Flush so the CI log
        # shows where we are (a silent multi-minute stall previously read as a hang).
        from datetime import date as _date

        from mtnwx.data.asos import fetch_asos_hourly

        print(f"  ASOS: fetching {len(asos_ids)} stations, {start.year}-{end.year}...", flush=True)
        got = 0
        for yr in range(start.year, end.year + 1):
            ys = max(start, _date(yr, 1, 1))
            ye = min(end, _date(yr, 12, 31))
            try:
                adf = fetch_asos_hourly(asos_ids, ys, ye)
                if not adf.empty:
                    frames.append(obs.normalize_hourly(adf))
                    got += len(adf)
                print(f"    ASOS {yr}: {len(adf)} rows (total {got})", flush=True)
            except Exception as exc:  # noqa: BLE001 — ASOS is additive
                print(f"    WARN: ASOS {yr} failed ({exc})", flush=True)

    other = stations.loc[~is_asos]
    rows = [row for _, row in other.iterrows()]
    n = len(rows)
    done = 0
    if n:
        print(f"  SNOTEL/other: fetching {n} stations ({workers} workers)...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for df in ex.map(lambda r: _fetch_one(r, start, end, token), rows):
                done += 1
                if not df.empty:
                    frames.append(obs.normalize_hourly(df))
                if done % 100 == 0 or done == n:
                    got = sum(len(f) for f in frames)
                    print(f"  [{done}/{n}] stations processed, {got} obs rows so far", flush=True)
    if not frames:
        return obs._empty()
    # Downcast each frame to float32 BEFORE concat so peak memory (concat + the QC
    # sort/groupby that follows) stays roughly halved — 56.9M rows x float64 OOM'd a
    # 16 GB runner right after collection.
    for f in frames:
        for c in f.select_dtypes("float64").columns:
            f[c] = f[c].astype("float32")
    allobs = pd.concat(frames, ignore_index=True)
    frames.clear()
    return obs.qc(allobs, copy=False)


def main(args: argparse.Namespace) -> int:
    stations_path = Path(args.stations) if args.stations else data_dir() / "stations.parquet"
    if not stations_path.exists():
        print(f"ERROR: station catalogue not found at {stations_path}; run `mtnwx stations` first")
        return 1
    stations = pd.read_parquet(stations_path)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    token = os.environ.get("SYNOPTIC_API_TOKEN")

    print(f"Collecting hourly obs for {len(stations)} stations, {start} .. {end}")
    df = collect(stations, start, end, token=token)

    out = Path(args.out) if args.out else data_dir() / f"obs_{start}_{end}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} QC'd obs rows -> {out}")
    if not df.empty:
        print("By source:", df["source"].value_counts().to_dict())
    return 0
