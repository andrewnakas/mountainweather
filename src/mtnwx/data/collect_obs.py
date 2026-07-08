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
        # Default: a bare SNOTEL triplet like "308:AZ:SNTL".
        if sid.endswith(":SNTL") or sid.count(":") == 2:
            return obs.fetch_snotel_hourly(sid, start, end)
    except Exception as exc:  # noqa: BLE001 — never let one station kill the run
        print(f"WARN: obs fetch failed for {sid}: {exc}")
    return obs._empty()


def collect(
    stations: pd.DataFrame, start: date, end: date, *, token: str | None = None
) -> pd.DataFrame:
    """Fetch + QC hourly obs for every station in ``stations`` over [start, end]."""
    frames: list[pd.DataFrame] = []
    n = len(stations)
    for i, (_, row) in enumerate(stations.iterrows(), 1):
        df = _fetch_one(row, start, end, token)
        if not df.empty:
            frames.append(obs.normalize_hourly(df))
        if i % 50 == 0 or i == n:
            got = sum(len(f) for f in frames)
            print(f"  [{i}/{n}] stations processed, {got} obs rows so far")
    if not frames:
        return obs._empty()
    allobs = pd.concat(frames, ignore_index=True)
    return obs.qc(allobs)


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
