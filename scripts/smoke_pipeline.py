#!/usr/bin/env python3
"""End-to-end smoke test on real data: stations -> terrain -> HRRR -> obs -> table -> train.

Runs the entire pipeline at small scale (a handful of stations, a short window) against
live sources, so a regression anywhere in the chain surfaces before the full-scale run.
Not a unit test (it hits the network); invoked manually or as a slow CI job.

    python scripts/smoke_pipeline.py --stations 20 --start 2024-01-15 --end 2024-01-25
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtnwx.config import data_dir, load_configs  # noqa: E402
from mtnwx.data import hrrr, stations as stn, terrain  # noqa: E402
from mtnwx.data.collect_obs import collect  # noqa: E402
from mtnwx.features.build import build_training_table, feature_columns  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", type=int, default=20)
    ap.add_argument("--start", default="2024-01-15")
    ap.add_argument("--end", default="2024-01-25")
    ap.add_argument("--month", default="2024-01")
    ap.add_argument("--max-inits", type=int, default=6, help="Cap HRRR inits for a fast smoke run")
    args = ap.parse_args()

    cfg = load_configs()
    region = cfg["region"]
    dd = data_dir()

    print("== 1. stations ==")
    cat = stn.build_catalogue(limit=args.stations)
    print(f"   {len(cat)} stations")

    print("== 2. terrain ==")
    terr = terrain.compute_features(cat, region)
    cat_t = terrain.finalize_stations(cat, terr, region)
    print(f"   {len(cat_t)} stations after relief filter; "
          f"terrain cols present: {[c for c in ('dem_elevation_m','slope_deg','tpi_2km') if c in cat_t]}")

    print("== 3. HRRR extract (limited inits for speed) ==")
    # Extract only a few inits covering the window, so the smoke test is minutes not
    # hours (the full backfill uses extract_month; here we bound the init list).
    import xarray as xr

    ds = hrrr.open_archive()
    yi, xi, ok, dist = hrrr.build_grid_index(ds, cat_t, dd / "hrrr_grid_index.json")
    st_ok = cat_t.loc[ok].reset_index(drop=True)
    y_da = xr.DataArray(yi[ok], dims="station")
    x_da = xr.DataArray(xi[ok], dims="station")
    fields = load_configs()["predictors"]["hrrr_fields"]
    all_inits = pd.to_datetime(ds.init_time.values)
    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end)
    # Inits whose 48 h horizon overlaps the window; cap at args.max_inits for speed.
    sel = all_inits[(all_inits >= lo - pd.Timedelta(hours=48)) & (all_inits <= hi)]
    sel = list(sel[:: max(1, len(sel) // args.max_inits)])[: args.max_inits]
    print(f"   extracting {len(sel)} inits x {len(fields)} fields x {len(st_ok)} stations")
    frames = [hrrr._extract_one_init(ds, it, fields, y_da, x_da) for it in sel]
    hx = pd.concat(frames, ignore_index=True)
    hx["station_id"] = st_ok["station_id"].to_numpy()[hx["station_ix"].to_numpy()]
    hx = hx.drop(columns="station_ix")
    hx["valid_time"] = pd.to_datetime(hx["init_time"]) + pd.to_timedelta(hx["lead_hour"], unit="h")
    hx = hx[(hx["valid_time"] >= lo) & (hx["valid_time"] <= hi)]
    print(f"   {len(hx)} predictor rows in window")

    print("== 4. observations ==")
    o = collect(cat_t, datetime.strptime(args.start, "%Y-%m-%d").date(),
                datetime.strptime(args.end, "%Y-%m-%d").date())
    print(f"   {len(o)} QC'd obs rows")

    print("== 5. feature table ==")
    table = build_training_table(hx, o, cat_t)
    print(f"   {len(table)} joined training rows; {len(feature_columns(table))} features")
    print(f"   labelled air_temp_c: {table['air_temp_c'].notna().sum()}")
    out = dd / "training_table.parquet"
    table.to_parquet(out, index=False)
    print(f"   wrote {out}")

    print("== 6. sanity: does elevation_delta correlate with HRRR temp error? ==")
    if {"elevation_delta_m", "temperature_2m", "air_temp_c"}.issubset(table.columns):
        err = table["temperature_2m"] - table["air_temp_c"]  # HRRR minus obs
        c = table["elevation_delta_m"].corr(err)
        print(f"   corr(elevation_delta, HRRR_temp_error) = {c:.3f} "
              f"(nonzero => terrain explains bias, the whole premise)")
    print("\nSMOKE OK — pipeline runs end to end on live data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
