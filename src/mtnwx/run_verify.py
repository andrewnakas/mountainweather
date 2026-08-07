"""Verification orchestrator: score the trained models on the held-out set + report.

Ties together train.make_splits (to recover the exact held-out rows), the trained
quantile models, the NBM benchmark, and verify.score_frame, then writes the skill
report. This is the CLI entry point behind ``mtnwx verify``.

It expects the training table (with obs joined) and the trained models. It regenerates
mtnwx's own predictions on the held-out rows, pulls NBM for the held-out window/stations,
and produces the scorecard the site publishes.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from mtnwx.config import data_dir
from mtnwx.features.build import TARGET_SPEC
from mtnwx.report import write_report
from mtnwx.train import make_splits
from mtnwx.verify import score_frame

# Raw-GFS baseline columns in the GLOBAL training table (see build_training_table_global).
# GFS has no gust product, so wind_gust_ms has no raw-GFS baseline (falls to persistence).
GFS_BASE_FIELD = {
    "air_temp_c": "gfs_temperature_2m",
    "relative_humidity_pct": "gfs_relative_humidity_2m",
    "wind_speed_ms": "gfs_wind_speed_10m",
    "precip_1h_mm": "gfs_precip_mm",
}


def _load_models(models_dir: Path):
    import json

    meta = json.loads((models_dir / "metadata.json").read_text())
    models = {}
    for target in meta["targets"]:
        p = models_dir / f"{target}.pkl"
        if p.exists():
            blob = pickle.loads(p.read_bytes())
            models[target] = {float(q): lgb.Booster(model_str=s) for q, s in blob.items()}
    return models, meta


def main(args: argparse.Namespace) -> int:
    # Default NBM sample cap when the attribute isn't provided (programmatic calls).
    if not hasattr(args, "nbm_stations"):
        args.nbm_stations = 60
    from mtnwx.train import _load_table

    table_path = Path(args.table) if args.table else data_dir() / "training_table"
    models_dir = Path(args.models) if args.models else data_dir() / "models"
    # Verify only needs enough held-out rows for stable MAE/CRPS estimates — far fewer
    # than training. Cap load (default 3M) so scoring 28 quantile models stays in memory.
    max_rows = getattr(args, "verify_rows", getattr(args, "max_rows", 3_000_000))
    df = _load_table(table_path, max_rows=max_rows)
    if df is None:
        print(f"ERROR: training table not found at {table_path}")
        return 1
    models, meta = _load_models(models_dir)
    feats = meta["features"]

    _, test_mask, _ = make_splits(df)
    test = df.loc[test_mask].reset_index(drop=True)
    print(f"Verifying on {len(test)} held-out rows")

    # If NBM columns are already in the table (merged as predictors at build time), use
    # them directly — no re-fetch needed, and it's the full held-out set, not a sample.
    have_nbm_cols = any(c.startswith("nbm_") for c in test.columns)
    if have_nbm_cols:
        print("Using in-table NBM columns for the benchmark (no re-fetch).")

    # Otherwise attach NBM benchmark for a sample of held-out stations.
    if not args.no_nbm and not have_nbm_cols:
        from mtnwx.data.nbm import fetch_nbm_for_stations

        vt = pd.to_datetime(test["valid_time"])
        st = test[["station_id"]].drop_duplicates().merge(
            df[["station_id", "lat", "lon"]].drop_duplicates() if "lat" in df else
            pd.read_parquet(data_dir() / "stations_terrain.parquet")[["station_id", "lat", "lon"]],
            on="station_id", how="left",
        ).dropna(subset=["lat", "lon"])
        # NBM comes from the Open-Meteo archive API (per-station, rate-limited). A
        # random sample of held-out stations gives a statistically sound benchmark MAE
        # without querying hundreds of stations over 7 years (which is slow / hits
        # rate limits). Cap configurable via --nbm-stations.
        if args.nbm_stations and len(st) > args.nbm_stations:
            st = st.sample(n=args.nbm_stations, random_state=17).reset_index(drop=True)
            print(f"NBM benchmark sampled to {len(st)} held-out stations")
        try:
            nbm = fetch_nbm_for_stations(st, vt.min().date(), vt.max().date())
            nbm["valid_time"] = pd.to_datetime(nbm["valid_time"])
            test = test.merge(nbm, on=["station_id", "valid_time"], how="left")
        except Exception as exc:  # noqa: BLE001 — NBM is a nice-to-have, not fatal
            print(f"WARN: NBM benchmark unavailable ({exc}); scoring without it")

    nbm_map = {
        "air_temp_c": "nbm_air_temp_c",
        "relative_humidity_pct": "nbm_relative_humidity_pct",
        "wind_speed_ms": "nbm_wind_speed_ms",
        "wind_gust_ms": "nbm_wind_gust_ms",
        "precip_1h_mm": "nbm_precip_1h_mm",
    }

    import os
    is_global = os.environ.get("MTNWX_REGION") == "global"
    all_metrics = []
    X = test[feats].astype("float32")
    for target, boosters in models.items():
        if target not in test.columns:
            continue
        point = boosters[0.5].predict(X) if 0.5 in boosters else None
        quantiles = {q: b.predict(X) for q, b in boosters.items()}
        # Region-aware raw-NWP baseline. The US model's base is HRRR; the global model's
        # base is GFS, whose forecast columns are gfs_* in the global training table.
        # Picking the wrong field name yields an all-NaN baseline (empty comparison),
        # which is exactly why the first global scorecard had no skill_vs_benchmark rows.
        if is_global:
            base_field = GFS_BASE_FIELD.get(target, "")
            base_label = "raw_gfs"
        else:
            base_field = TARGET_SPEC.get(target, {}).get("hrrr_field", "")
            base_label = "raw_hrrr"
        m = score_frame(
            test, target, base_field, point, quantiles,
            nbm_col=nbm_map.get(target), base_label=base_label,
        )
        all_metrics.append(m)

    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    out_dir = Path(args.out) if args.out else data_dir() / "verify"
    write_report(metrics, out_dir, meta)
    print(f"Wrote skill report -> {out_dir}")
    if not metrics.empty:
        from mtnwx.report import skill_vs_benchmark, headline
        print("\nHeadline skill vs benchmarks:")
        print(skill_vs_benchmark(headline(metrics)).to_string(index=False))
    return 0
