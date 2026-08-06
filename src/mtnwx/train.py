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

PHASE1_TARGETS = [
    "air_temp_c", "wind_speed_ms", "wind_gust_ms", "relative_humidity_pct", "precip_1h_mm",
]


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

    import gc

    label = df[target].to_numpy("float32")
    keep = ~np.isnan(label)
    train_mask, test_mask, _ = make_splits(df)
    tr = train_mask & keep
    va = test_mask & keep
    if tr.sum() == 0 or va.sum() == 0:
        raise ValueError(
            f"empty split for {target}: train={int(tr.sum())} val={int(va.sum())} "
            f"(dataset span may be too short for the holdout config)"
        )

    # Build the train/val Datasets ONCE and reuse across quantiles — only the objective
    # (alpha) changes per quantile, and binning the features 7x was needless memory + time.
    Xtr = df.loc[tr, feat_cols].to_numpy("float32")
    Xval = df.loc[va, feat_cols].to_numpy("float32")
    dtr = lgb.Dataset(Xtr, label=label[tr], free_raw_data=True)
    dval = lgb.Dataset(Xval, label=label[va], reference=dtr, free_raw_data=True)
    dtr.construct()
    dval.construct()
    del Xtr, Xval
    gc.collect()

    boosters: dict[float, object] = {}
    for q in quantiles:
        p = dict(params)
        p.update(objective="quantile", alpha=q, metric="quantile")
        es = p.pop("early_stopping_rounds", 100)
        n_est = p.pop("n_estimators", 1500)
        boosters[q] = lgb.train(
            p, dtr, num_boost_round=n_est, valid_sets=[dval],
            callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)],
        )
    del dtr, dval
    gc.collect()
    return boosters


def _load_table(path: Path, *, max_rows: int = 40_000_000) -> pd.DataFrame | None:
    """Load the training table from a single parquet OR a directory of part files.

    The full 7-year joined table is far larger than RAM, so we cap total rows: read
    parts one at a time and, once the running total would exceed ``max_rows``,
    proportionally subsample each further part. LightGBM converges fine on tens of
    millions of rows — no need to hold hundreds of millions."""
    if path.is_dir():
        parts = sorted(path.glob("part-*.parquet"))
    elif path.suffix == ".parquet" and path.exists():
        parts = [path]
    else:
        # Tolerate a directory passed without existing suffix, or a legacy file.
        alt = path.with_suffix("")
        if alt.is_dir():
            parts = sorted(alt.glob("part-*.parquet"))
        elif Path(str(path) + ".parquet").exists():
            parts = [Path(str(path) + ".parquet")]
        else:
            return None
    if not parts:
        return None

    # Row count per part (cheap metadata read) to set a global sampling fraction.
    import pyarrow.parquet as pq

    counts = [pq.ParquetFile(p).metadata.num_rows for p in parts]
    total = sum(counts)
    frac = min(1.0, max_rows / total) if total else 1.0
    if frac < 1.0:
        print(f"Table has {total} rows; subsampling to ~{max_rows} (frac={frac:.3f})")

    frames = []
    rng = np.random.default_rng(17)
    for p, c in zip(parts, counts):
        d = pd.read_parquet(p)
        if frac < 1.0 and len(d):
            d = d.iloc[rng.random(len(d)) < frac]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main(args: argparse.Namespace) -> int:
    cfg = load_configs()
    quantiles = cfg["variables"]["quantiles"]
    params = dict(cfg["model"]["lgbm"])

    table_path = Path(args.table) if args.table else data_dir() / "training_table"
    # 380M joined rows won't fit; ~8M is ample for LightGBM and leaves headroom for the
    # per-quantile Datasets on a 16 GB runner.
    df = _load_table(table_path, max_rows=getattr(args, "max_rows", 8_000_000))
    if df is None:
        print(f"ERROR: training table not found at {table_path} (build it first)")
        return 1
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
