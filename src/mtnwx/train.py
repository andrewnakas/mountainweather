"""Train the LightGBM quantile post-processors.

For each phase-1 target (temperature, wind speed, gust, RH) we train one LightGBM model
per quantile level (configs/variables.yaml), with lead_hour as a feature. Predicting a
spread of quantiles gives calibrated probabilistic output; the q0.50 model is the point
forecast.

Honest evaluation is the whole point — the model must generalize to *unseen mountains in
unseen weather*, not memorize stations. So the split holds out both:
  - the most recent N months (temporal), and
  - a fraction of stations chosen spatially (never seen in training).

Artifacts (one booster per target x quantile) plus feature list and metadata are written
locally and, in CI, pushed to the HF models repo. Verification against NBM/HRRR is a
separate stage (verify.py) run on the held-out set.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from mtnwx.config import data_dir, load_configs
from mtnwx.features.build import feature_columns

PHASE1_TARGETS = ["air_temp_c", "wind_speed_ms", "wind_gust_ms", "relative_humidity_pct"]


def make_splits(
    df: pd.DataFrame, *, holdout_months: int = 12, holdout_station_frac: float = 0.2, seed: int = 17
):
    """Return boolean masks (train, test). Test = recent months OR held-out stations.

    Using OR (not AND) for the test set means we measure generalization to unseen time
    *and* unseen space; the train set is strictly the complement so there is no leakage.

    The temporal cutoff is capped so it never swallows more than ~30% of the actual time
    span — otherwise a dataset shorter than ``holdout_months`` (e.g. a smoke run) would
    put every row in the test set and leave nothing to train on."""
    vt = pd.to_datetime(df["valid_time"])
    span_days = (vt.max() - vt.min()).days or 1
    holdout_days = min(holdout_months * 30, int(span_days * 0.3))
    cutoff = vt.max() - pd.Timedelta(days=holdout_days)
    recent = vt > cutoff

    stations = df["station_id"].unique()
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(len(stations) * holdout_station_frac))
    held_stations = set(rng.choice(stations, size=n_hold, replace=False))
    held_station_mask = df["station_id"].isin(held_stations)

    test = recent | held_station_mask
    train = ~test
    return train.to_numpy(), test.to_numpy(), sorted(held_stations)


def train_quantile_models(
    df: pd.DataFrame, target: str, feat_cols: list[str], quantiles: list[float], params: dict
):
    """Train one LightGBM booster per quantile for ``target``. Returns {q: booster}."""
    import lightgbm as lgb

    sub = df.dropna(subset=[target]).reset_index(drop=True)
    train_mask, test_mask, _ = make_splits(sub)
    X = sub[feat_cols].astype("float32")
    y = sub[target].astype("float32")
    Xtr, ytr = X[train_mask], y[train_mask]
    Xval, yval = X[test_mask], y[test_mask]
    if len(Xtr) == 0 or len(Xval) == 0:
        raise ValueError(
            f"empty split for {target}: train={len(Xtr)} val={len(Xval)} "
            f"(dataset span may be too short for the holdout config)"
        )

    boosters: dict[float, object] = {}
    for q in quantiles:
        p = dict(params)
        p.update(objective="quantile", alpha=q, metric="quantile")
        es = p.pop("early_stopping_rounds", 100)
        n_est = p.pop("n_estimators", 1500)
        dtr = lgb.Dataset(Xtr, label=ytr)
        dval = lgb.Dataset(Xval, label=yval, reference=dtr)
        booster = lgb.train(
            p, dtr, num_boost_round=n_est, valid_sets=[dval],
            callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)],
        )
        boosters[q] = booster
    return boosters


def main(args: argparse.Namespace) -> int:
    cfg = load_configs()
    quantiles = cfg["variables"]["quantiles"]
    params = dict(cfg["model"]["lgbm"])

    table_path = Path(args.table) if args.table else data_dir() / "training_table.parquet"
    if not table_path.exists():
        print(f"ERROR: training table not found at {table_path} (build it first)")
        return 1
    df = pd.read_parquet(table_path)
    feat_cols = feature_columns(df)
    print(f"Training on {len(df)} rows, {len(feat_cols)} features")
    print("Features:", feat_cols)

    targets = args.targets.split(",") if args.targets else PHASE1_TARGETS
    out_dir = Path(args.out) if args.out else data_dir() / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, test_mask, held_stations = make_splits(df)
    meta = {
        "features": feat_cols,
        "quantiles": quantiles,
        "targets": targets,
        "held_out_stations": held_stations,
        "n_train_rows": int((~test_mask).sum()),
        "n_test_rows": int(test_mask.sum()),
    }

    for target in targets:
        if target not in df.columns:
            print(f"  skip {target}: not in table")
            continue
        n = df[target].notna().sum()
        if n < 1000:
            print(f"  skip {target}: only {n} labelled rows")
            continue
        print(f"  training {target} ({n} labelled rows) x {len(quantiles)} quantiles...")
        boosters = train_quantile_models(df, target, feat_cols, quantiles, params)
        with open(out_dir / f"{target}.pkl", "wb") as fh:
            pickle.dump({q: b.model_to_string() for q, b in boosters.items()}, fh)
        print(f"    saved {out_dir / f'{target}.pkl'}")

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote models + metadata -> {out_dir}")
    return 0
