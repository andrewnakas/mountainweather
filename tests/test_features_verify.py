"""Feature-join, split, and metric unit tests on synthetic data (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtnwx.features.build import (
    add_derived_predictors,
    build_training_table,
    feature_columns,
)
from mtnwx.train import make_splits
from mtnwx.verify import (
    crps_from_quantiles,
    lapse_corrected_temp,
    mae,
    paired_bootstrap_mae_diff,
)


def _synthetic_hrrr(stations, init, leads):
    rows = []
    for _, s in stations.iterrows():
        for lh in leads:
            rows.append(
                {
                    "station_id": s["station_id"],
                    "init_time": init,
                    "lead_hour": lh,
                    "temperature_2m": 5.0 - 0.1 * lh,
                    "wind_u_10m": 3.0,
                    "wind_v_10m": 4.0,
                    "relative_humidity_2m": 50.0,
                    "wind_gust_surface": 12.0,
                    "grid_elevation_m": s["elevation_m"] - 150.0,  # HRRR grid 150 m low
                }
            )
    return pd.DataFrame(rows)


def _stations():
    return pd.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "lat": [40.0, 41.0, 39.0],
            "lon": [-106.0, -107.0, -105.0],
            "elevation_m": [3000.0, 2500.0, 3200.0],
            "dem_elevation_m": [3000.0, 2500.0, 3200.0],
            "slope_deg": [10.0, 5.0, 15.0],
            "tpi_2km": [20.0, -10.0, 40.0],
        }
    )


def test_derived_wind_and_valid_time():
    st = _stations()
    hrrr = _synthetic_hrrr(st, pd.Timestamp("2024-01-15"), [1, 2, 3])
    out = add_derived_predictors(hrrr)
    # sqrt(3^2 + 4^2) = 5
    assert np.allclose(out["wind_speed_10m"], 5.0)
    assert (out["valid_time"] == pd.to_datetime(out["init_time"]) + pd.to_timedelta(out["lead_hour"], "h")).all()


def test_build_table_joins_obs_and_elevation_delta():
    st = _stations()
    init = pd.Timestamp("2024-01-15")
    leads = list(range(1, 6))
    hrrr = _synthetic_hrrr(st, init, leads)
    # Obs at matching valid times.
    obs_rows = []
    for _, s in st.iterrows():
        for lh in leads:
            obs_rows.append(
                {
                    "station_id": s["station_id"],
                    "valid_time": init + pd.Timedelta(hours=lh),
                    "air_temp_c": 3.0 - 0.1 * lh,
                    "relative_humidity_pct": 55.0,
                    "wind_speed_ms": 6.0,
                    "wind_gust_ms": 11.0,
                    "dewpoint_c": -2.0,
                    "precip_1h_mm": 0.0,
                }
            )
    obs = pd.DataFrame(obs_rows)
    table = build_training_table(hrrr, obs, st)
    assert len(table) == len(st) * len(leads)
    # elevation_delta_m = station elev - grid elev = +150 everywhere
    assert np.allclose(table["elevation_delta_m"], 150.0)
    assert "air_temp_c" in table.columns
    feats = feature_columns(table)
    assert "elevation_delta_m" in feats and "slope_deg" in feats
    # Targets must NOT be features.
    assert "air_temp_c" not in feats


def test_make_splits_no_leakage():
    n = 2000
    df = pd.DataFrame(
        {
            "station_id": np.random.default_rng(0).choice(list("ABCDEFGHIJ"), n),
            "valid_time": pd.date_range("2023-01-01", periods=n, freq="6h"),
        }
    )
    tr, te, held = make_splits(df, holdout_months=6, holdout_station_frac=0.2)
    # Disjoint and covering.
    assert not (tr & te).any()
    assert (tr | te).all()
    # Held-out stations never appear in train.
    assert not df.loc[tr, "station_id"].isin(held).any()


def test_make_splits_short_span_leaves_training_data():
    # A dataset shorter than holdout_months must still yield a non-empty train set
    # (regression: a 45-day span with holdout_months=12 previously put every row in test).
    n = 4000
    df = pd.DataFrame(
        {
            "station_id": np.random.default_rng(1).choice(list("ABCDEFGHIJKLMNOP"), n),
            "valid_time": pd.date_range("2024-01-01", periods=n, freq="15min"),  # ~42 days
        }
    )
    tr, te, _ = make_splits(df, holdout_months=12, holdout_station_frac=0.2)
    assert tr.sum() > 0, "train set must be non-empty for short spans"
    assert te.sum() > 0, "test set must be non-empty"
    assert not (tr & te).any()


def test_lapse_correction_direction():
    # Station 150 m above grid -> lapse subtracts ~1 C from raw HRRR temp.
    df = pd.DataFrame({"temperature_2m": [5.0], "elevation_delta_m": [150.0]})
    corrected = lapse_corrected_temp(df)
    assert corrected[0] < 5.0
    assert abs(corrected[0] - (5.0 - 6.5 / 1000 * 150)) < 1e-6


def test_crps_and_bootstrap():
    obs = np.array([0.0, 1.0, 2.0, 3.0, 4.0] * 20, dtype=float)
    # Perfect quantiles centered on obs -> low CRPS.
    q = {0.25: obs - 0.5, 0.5: obs, 0.75: obs + 0.5}
    good = crps_from_quantiles(q, obs)
    bad = crps_from_quantiles({0.25: obs + 4, 0.5: obs + 5, 0.75: obs + 6}, obs)
    assert good < bad

    # mtnwx (a) is closer to obs than benchmark (b) -> positive mean diff.
    a = obs + 0.1
    b = obs + 2.0
    mean_d, lo, hi = paired_bootstrap_mae_diff(a, b, obs, n=200)
    assert mean_d > 0 and lo > 0  # CI excludes zero -> real improvement
    assert mae(a, obs) < mae(b, obs)
