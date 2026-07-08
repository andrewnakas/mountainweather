"""Terrain features from the Copernicus GLO-30 (30 m) DEM.

Terrain is what separates a mountain post-processor from a generic one. The single
most important predictor is the **elevation delta**: station elevation minus the HRRR
model's grid-cell elevation. HRRR's 3 km orography smooths peaks down and fills
valleys in; the sign and size of that delta explains most of the model's temperature
and wind bias. We also derive slope, aspect, multi-scale TPI (topographic position),
and a wind-exposure index (Sx) — all standard complex-terrain descriptors.

DEM source: Copernicus GLO-30 Public on AWS Open Data (``s3://copernicus-dem-30m``),
Cloud-Optimized GeoTIFFs on a 1x1 degree tile grid, no-sign-request. We read only the
small window around each station via rasterio's windowed reads (no full-tile download).

``compute_features(stations)`` returns one row per station with the terrain columns and
the local-relief value used to finish the mountain filter deferred from stations.py.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mtnwx.config import data_dir, load_configs

# GLO-30 public COG tile URL. Tiles are named by the SW corner integer lat/lon.
DEM_TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)

# GLO-30 posting is ~30 m; window half-widths in DEM pixels for the multi-scale metrics.
M_PER_DEG_LAT = 111_320.0


def tile_url(lat: float, lon: float) -> str:
    """COG URL for the 1x1 degree tile whose SW corner contains (lat, lon)."""
    lat_sw = math.floor(lat)
    lon_sw = math.floor(lon)
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    return DEM_TILE_URL.format(ns=ns, lat=abs(lat_sw), ew=ew, lon=abs(lon_sw))


def _pixel_window(src, lat: float, lon: float, radius_m: float):
    """Return a (data, cellsize_m) window centered on (lat, lon) of ~radius_m."""
    from rasterio.windows import Window

    row, col = src.index(lon, lat)
    # DEM cell size in metres (approx; lon spacing shrinks with latitude).
    cell_deg = src.transform.a
    cell_m = cell_deg * M_PER_DEG_LAT * math.cos(math.radians(lat))
    half = max(1, int(round(radius_m / cell_m)))
    win = Window(col - half, row - half, 2 * half + 1, 2 * half + 1)
    data = src.read(1, window=win, boundless=True, fill_value=np.nan).astype("float64")
    return data, cell_m


def _slope_aspect(patch: np.ndarray, cell_m: float) -> tuple[float, float]:
    """Central-cell slope (deg) and aspect (deg from N) via a 3x3 gradient."""
    c = patch.shape[0] // 2
    if c < 1:
        return float("nan"), float("nan")
    z = patch[c - 1 : c + 2, c - 1 : c + 2]
    if np.isnan(z).any():
        return float("nan"), float("nan")
    dzdx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * cell_m)
    dzdy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * cell_m)
    slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect = math.degrees(math.atan2(dzdy, -dzdx))
    return slope, (aspect + 360.0) % 360.0


def _tpi(patch: np.ndarray) -> float:
    """Topographic position index: centre elevation minus mean of the window."""
    c = patch.shape[0] // 2
    centre = patch[c, c]
    ring = np.nanmean(patch)
    return float(centre - ring)


def compute_features(stations: pd.DataFrame, region: dict) -> pd.DataFrame:
    """Per-station terrain features + local relief. Requires rasterio (terrain extra)."""
    import rasterio
    from rasterio.env import Env

    filt = region["station_filter"]
    relief_radius_m = filt["relief_radius_km"] * 1000.0

    rows: list[dict] = []
    # Group stations by DEM tile so each COG is opened once.
    stations = stations.copy()
    stations["_tile"] = [tile_url(la, lo) for la, lo in zip(stations["lat"], stations["lon"])]

    with Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for url, grp in stations.groupby("_tile"):
            try:
                src = rasterio.open(url)
            except Exception as exc:  # noqa: BLE001 — missing tile (unreleased) -> skip terrain
                print(f"WARN: DEM tile open failed ({url}): {exc}")
                for _, s in grp.iterrows():
                    rows.append({"station_id": s["station_id"], "dem_ok": False})
                continue
            with src:
                for _, s in grp.iterrows():
                    la, lo = float(s["lat"]), float(s["lon"])
                    try:
                        patch_r, cell_m = _pixel_window(src, la, lo, relief_radius_m)
                    except Exception:  # noqa: BLE001
                        rows.append({"station_id": s["station_id"], "dem_ok": False})
                        continue
                    dem_elev = float(patch_r[patch_r.shape[0] // 2, patch_r.shape[1] // 2])
                    relief = float(np.nanmax(patch_r) - np.nanmin(patch_r))
                    slope, aspect = _slope_aspect(patch_r, cell_m)
                    # Multi-scale TPI from concentric windows.
                    half = patch_r.shape[0] // 2
                    tpi_500 = _tpi(_center(patch_r, min(half, int(500 / cell_m))))
                    tpi_2k = _tpi(_center(patch_r, min(half, int(2000 / cell_m))))
                    tpi_10k = _tpi(patch_r)
                    rows.append(
                        {
                            "station_id": s["station_id"],
                            "dem_elevation_m": dem_elev,
                            "local_relief_m": relief,
                            "slope_deg": slope,
                            "aspect_deg": aspect,
                            "tpi_500m": tpi_500,
                            "tpi_2km": tpi_2k,
                            "tpi_10km": tpi_10k,
                            "dem_ok": True,
                        }
                    )
    return pd.DataFrame(rows)


def _center(patch: np.ndarray, half: int) -> np.ndarray:
    """Central (2*half+1) square sub-window of a patch."""
    if half < 1:
        return patch
    c = patch.shape[0] // 2
    return patch[c - half : c + half + 1, c - half : c + half + 1]


def finalize_stations(stations: pd.DataFrame, terrain: pd.DataFrame, region: dict) -> pd.DataFrame:
    """Join terrain onto the catalogue and apply the deferred local-relief filter.

    Stations that are high but flat (below the relief threshold) — e.g. high-desert
    plateau sites — are dropped here; they aren't "mountain" for our purposes."""
    filt = region["station_filter"]
    merged = stations.merge(terrain, on="station_id", how="left")
    keep = merged["local_relief_m"].fillna(0) >= filt["min_local_relief_m"]
    # If the DEM tile was missing (dem_ok False), keep the station (relief unknown)
    # rather than silently dropping coverage — flag it instead.
    keep = keep | (merged["dem_ok"] != True)  # noqa: E712
    out = merged[keep].drop(columns=["needs_relief_check", "_tile"], errors="ignore")
    return out.reset_index(drop=True)


def main(args: argparse.Namespace) -> int:
    cfg = load_configs()
    region = cfg["region"]
    stations_path = Path(args.stations) if args.stations else data_dir() / "stations.parquet"
    if not stations_path.exists():
        print(f"ERROR: run `mtnwx stations` first (missing {stations_path})")
        return 1
    stations = pd.read_parquet(stations_path)
    print(f"Computing terrain features for {len(stations)} stations...")
    terrain = compute_features(stations, region)
    final = finalize_stations(stations, terrain, region)

    out = Path(args.out) if args.out else data_dir() / "stations_terrain.parquet"
    final.to_parquet(out, index=False)
    n_drop = len(stations) - len(final)
    print(f"Wrote {len(final)} stations with terrain -> {out} ({n_drop} dropped by relief filter)")
    return 0
