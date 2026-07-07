#!/usr/bin/env python3
"""Validate the real train + verify code on a synthetic table with a planted bias.

We construct a training table with the exact schema build_training_table produces, but
with observations generated from HRRR by a *known* terrain-dependent bias:

    obs_temp = hrrr_temp - lapse * elevation_delta + diurnal + noise

A working post-processor must LEARN this correction and beat raw HRRR at held-out
stations. If mtnwx's MAE isn't well below raw HRRR's here, the training/verify code is
broken. This exercises LightGBM training, the no-leakage split, and the full verify +
report path — everything the slow network smoke test would, minus the extraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtnwx.config import data_dir  # noqa: E402


def build_synthetic(n_stations=40, n_days=45, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    st_ids = [f"{i:03d}:CO:SNTL" for i in range(n_stations)]
    # Each station has an elevation delta (station vs HRRR grid) that biases HRRR.
    elev = rng.uniform(1500, 3600, n_stations)
    delta = rng.uniform(-400, 400, n_stations)  # station minus grid elevation
    slope = rng.uniform(0, 30, n_stations)
    tpi = rng.uniform(-100, 100, n_stations)

    inits = pd.date_range("2024-01-01", periods=n_days * 4, freq="6h")
    leads = np.arange(1, 25)
    rows = []
    for si, sid in enumerate(st_ids):
        for init in inits:
            for lh in leads:
                vt = init + pd.Timedelta(hours=int(lh))
                hour = vt.hour
                # Raw HRRR "forecast" temp: seasonal + diurnal, unaware of the delta.
                hrrr_t = 2.0 + 5 * np.sin(2 * np.pi * (vt.dayofyear) / 365) \
                    + 6 * np.sin(2 * np.pi * (hour - 15) / 24)
                # TRUE obs = HRRR minus lapse over the delta + a slope/tpi effect + noise.
                obs_t = (
                    hrrr_t
                    - 0.0065 * delta[si]           # the elevation bias to learn
                    - 0.02 * slope[si]             # cold-air pooling proxy
                    + 0.01 * tpi[si]
                    + rng.normal(0, 1.0)
                )
                rows.append(
                    {
                        "station_id": sid,
                        "init_time": init,
                        "lead_hour": float(lh),
                        "valid_time": vt,
                        "temperature_2m": np.float32(hrrr_t),
                        "grid_elevation_m": np.float32(elev[si] - delta[si]),
                        "elevation_m": np.float32(elev[si]),
                        "elevation_delta_m": np.float32(delta[si]),
                        "dem_elevation_m": np.float32(elev[si]),
                        "slope_deg": np.float32(slope[si]),
                        "aspect_deg": np.float32(rng.uniform(0, 360)),
                        "tpi_500m": np.float32(tpi[si] * 0.5),
                        "tpi_2km": np.float32(tpi[si]),
                        "tpi_10km": np.float32(tpi[si] * 1.5),
                        "hour_sin": np.float32(np.sin(2 * np.pi * hour / 24)),
                        "hour_cos": np.float32(np.cos(2 * np.pi * hour / 24)),
                        "doy_sin": np.float32(np.sin(2 * np.pi * vt.dayofyear / 365)),
                        "doy_cos": np.float32(np.cos(2 * np.pi * vt.dayofyear / 365)),
                        "solar_elev_deg": np.float32(max(-10, 40 * np.sin(2 * np.pi * (hour - 6) / 24))),
                        "lat": np.float32(39 + si * 0.02),
                        "lon": np.float32(-106 - si * 0.02),
                        "air_temp_c": np.float32(obs_t),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    print("Building synthetic training table with planted elevation bias...")
    df = build_synthetic()
    out = data_dir() / "training_table.parquet"
    df.to_parquet(out, index=False)
    print(f"  {len(df)} rows, {df['station_id'].nunique()} stations -> {out}")
    print("Now run: python -m mtnwx.cli train --targets air_temp_c")
    print("Then:    python -m mtnwx.cli verify --no-nbm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
