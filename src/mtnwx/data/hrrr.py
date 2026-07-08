"""Extract HRRR forecast predictors at station points from the dynamical.org Zarr.

This is the heavy lift: for each HRRR init cycle (00/06/12/18 UTC) and every lead hour
0-48, pull the model's forecast at each mountain station. Output is one tidy parquet
row per (station, init_time, lead_hour) with all predictor fields — the model inputs
the post-processor learns to correct.

Design (from throughput profiling against the archive):
  - The archive is chunked ``1 init x 49 leads x 265 y x 300 x``. Pointwise station
    indexing (``isel`` with per-station y/x) reads only the chunks covering the
    stations — far cheaper than reading the whole region.
  - Each init is an independent HTTP-bound read. Profiling from a single client showed
    dynamical.org effectively serializes concurrent reads from one IP (no threading
    speedup), so in-job concurrency is kept low; the real parallelism is the GitHub
    Actions matrix, where each job is a separate client. GitHub runners sit close to
    the data (AWS-adjacent) and read much faster than a home connection.
  - One job == one init-month, matching the GitHub Actions matrix backfill. Output is
    written per month and the manifest records completion, so the backfill is resumable.

The KD-tree station->grid index is cached to disk (keyed by the station-id list) so
repeated runs don't rebuild it. Instantaneous fields are taken as-is; precip and
snowfall are stored as their native per-lead values (accumulation windows are built at
feature time, not here, to keep extraction lossless).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mtnwx.config import data_dir, load_configs

EMAIL = "nakas@tree60weather.com"


def zarr_url() -> str:
    cfg = load_configs()
    base = cfg["predictors"]["zarr_url"]
    return f"{base}?email={EMAIL}"


def open_archive():
    import xarray as xr

    return xr.open_zarr(zarr_url(), decode_timedelta=True)


def build_grid_index(ds, stations: pd.DataFrame, cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest (y, x) grid index per station, plus grid lat/lon, cached to disk.

    Returns (y_idx, x_idx, in_domain_mask, grid_dist_deg). Stations farther than
    ~5 km from any grid cell (i.e. outside the HRRR CONUS domain) are masked out."""
    ids = list(stations["station_id"])
    if cache_path.exists():
        try:
            c = json.loads(cache_path.read_text())
            if c.get("ids") == ids:
                return (
                    np.array(c["y"]), np.array(c["x"]),
                    np.array(c["ok"], dtype=bool), np.array(c["dist"]),
                )
        except Exception:  # noqa: BLE001 — rebuild on any cache problem
            pass

    from scipy.spatial import cKDTree

    lat2d = ds.latitude.values
    lon2d = ds.longitude.values
    tree = cKDTree(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
    pts = np.column_stack([stations["lat"].to_numpy(), stations["lon"].to_numpy()])
    dist, flat = tree.query(pts)
    y_idx, x_idx = np.unravel_index(flat, lat2d.shape)
    ok = dist < 0.05  # ~4-5 km in degrees at these latitudes
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "ids": ids, "y": y_idx.tolist(), "x": x_idx.tolist(),
                "ok": ok.tolist(), "dist": dist.tolist(),
            }
        )
    )
    return y_idx, x_idx, ok, dist


def month_init_times(ds, month: str) -> pd.DatetimeIndex:
    """The archive's init times that fall in ``month`` (YYYY-MM)."""
    all_inits = pd.to_datetime(ds.init_time.values)
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthBegin(1)
    return all_inits[(all_inits >= start) & (all_inits < end)]


def _extract_one_init(ds, init, fields, y_da, x_da):
    """Return a long DataFrame for one init: rows = station x lead, cols = fields."""

    sub = ds[fields].sel(init_time=init).isel(y=y_da, x=x_da).compute()
    lead_h = (ds.lead_time.values / np.timedelta64(1, "h")).astype("int32")
    n_lead, n_st = sub[fields[0]].shape
    # Build the long frame column by column (float32 to keep memory down).
    base = {
        "init_time": np.repeat(np.datetime64(init), n_lead * n_st),
        "lead_hour": np.tile(np.repeat(lead_h, n_st), 1),
        "station_ix": np.tile(np.arange(n_st), n_lead),
    }
    out = pd.DataFrame(base)
    for f in fields:
        out[f] = sub[f].values.reshape(-1).astype("float32")
    return out


def extract_month(
    month: str, stations: pd.DataFrame, *, workers: int = 12
) -> pd.DataFrame:
    """Extract all predictor fields for every station over one init-month."""
    import xarray as xr

    cfg = load_configs()
    fields = cfg["predictors"]["hrrr_fields"]
    ds = open_archive()

    idx_cache = data_dir() / "hrrr_grid_index.json"
    y_idx, x_idx, ok, dist = build_grid_index(ds, stations, idx_cache)
    st = stations.loc[ok].reset_index(drop=True)
    y_da = xr.DataArray(y_idx[ok], dims="station")
    x_da = xr.DataArray(x_idx[ok], dims="station")

    inits = month_init_times(ds, month)
    if len(inits) == 0:
        print(f"No archive inits in {month}")
        return pd.DataFrame()

    print(f"{month}: {len(inits)} inits x {len(st)} stations x {len(fields)} fields")

    def job(it):
        return _extract_one_init(ds, it, fields, y_da, x_da)

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, df in enumerate(ex.map(job, list(inits)), 1):
            frames.append(df)
            if i % 20 == 0 or i == len(inits):
                print(f"  extracted {i}/{len(inits)} inits")

    allf = pd.concat(frames, ignore_index=True)
    # Map station_ix back to station_id + carry grid metadata.
    allf["station_id"] = st["station_id"].to_numpy()[allf["station_ix"].to_numpy()]
    allf["grid_dist_deg"] = dist[ok][allf["station_ix"].to_numpy()].astype("float32")
    allf = allf.drop(columns="station_ix")
    return allf


def main(args: argparse.Namespace) -> int:
    stations_path = (
        Path(args.stations)
        if args.stations
        else data_dir() / "stations_terrain.parquet"
    )
    if not stations_path.exists():
        # Fall back to the pre-terrain catalogue so extraction can run before M2.
        alt = data_dir() / "stations.parquet"
        if alt.exists():
            stations_path = alt
        else:
            print(f"ERROR: no station catalogue found (looked for {stations_path}, {alt})")
            return 1
    stations = pd.read_parquet(stations_path)

    df = extract_month(args.month, stations, workers=args.workers)
    if df.empty:
        print("No data extracted; nothing written.")
        return 0

    out = Path(args.out) if args.out else data_dir() / f"hrrr_{args.month}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} predictor rows -> {out}")
    return 0
