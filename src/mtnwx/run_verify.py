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
    table_path = Path(args.table) if args.table else data_dir() / "training_table.parquet"
    models_dir = Path(args.models) if args.models else data_dir() / "models"
    df = pd.read_parquet(table_path)
    models, meta = _load_models(models_dir)
    feats = meta["features"]

    _, test_mask, _ = make_splits(df)
    test = df.loc[test_mask].reset_index(drop=True)
    print(f"Verifying on {len(test)} held-out rows")

    # Optionally attach NBM benchmark for the held-out window/stations.
    if not args.no_nbm:
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
    }

    all_metrics = []
    X = test[feats].astype("float32")
    for target, boosters in models.items():
        if target not in test.columns:
            continue
        point = boosters[0.5].predict(X) if 0.5 in boosters else None
        quantiles = {q: b.predict(X) for q, b in boosters.items()}
        hrrr_field = TARGET_SPEC.get(target, {}).get("hrrr_field", "")
        m = score_frame(
            test, target, hrrr_field, point, quantiles,
            nbm_col=nbm_map.get(target),
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
