"""QC unit tests — synthetic data with planted defects, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtnwx.data.obs import OBS_COLUMNS, normalize_hourly, qc


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in OBS_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[OBS_COLUMNS]


def test_qc_preserves_station_id_and_schema():
    df = _frame(
        [
            {"station_id": "A", "valid_time": pd.Timestamp("2024-01-01T00:00"),
             "air_temp_c": 1.0, "source": "SNOTEL"},
        ]
    )
    out = qc(df)
    assert "station_id" in out.columns
    assert list(out.columns) == OBS_COLUMNS


def test_qc_range_bounds_null_out_of_range():
    # Each row carries an extra valid measurement so it survives the all-NaN drop
    # and we can assert on the specific field that was nulled.
    df = _frame(
        [
            {"station_id": "A", "valid_time": pd.Timestamp("2024-01-01T00:00"),
             "air_temp_c": 200.0, "relative_humidity_pct": 50.0, "source": "IEM"},  # impossible temp
            {"station_id": "A", "valid_time": pd.Timestamp("2024-01-01T01:00"),
             "air_temp_c": 5.0, "wind_speed_ms": -3.0, "source": "IEM"},            # negative wind
        ]
    )
    out = qc(df).set_index("valid_time")
    assert len(out) == 2
    # The impossible 200C temp is nulled; the valid 5C temp survives.
    assert pd.isna(out.loc[pd.Timestamp("2024-01-01T00:00"), "air_temp_c"])
    assert out.loc[pd.Timestamp("2024-01-01T01:00"), "air_temp_c"] == 5.0
    # The negative wind is nulled.
    assert pd.isna(out.loc[pd.Timestamp("2024-01-01T01:00"), "wind_speed_ms"])


def test_qc_dewpoint_not_above_temp():
    df = _frame(
        [
            {"station_id": "A", "valid_time": pd.Timestamp("2024-01-01T00:00"),
             "air_temp_c": 0.0, "dewpoint_c": 10.0, "source": "IEM"},
        ]
    )
    out = qc(df)
    assert out["dewpoint_c"].isna().all()


def test_qc_flatline_nulls_stuck_sensor():
    # 30 h of identical temperature -> the stuck run past the persistence window is
    # nulled. Those rows carry a (varying) RH so they survive the all-NaN drop and
    # the nulled temps remain visible for the assertion.
    times = pd.date_range("2024-01-01", periods=30, freq="h")
    df = _frame(
        [
            {"station_id": "A", "valid_time": t, "air_temp_c": 3.3,
             "relative_humidity_pct": 40.0 + i, "source": "SNOTEL"}
            for i, t in enumerate(times)
        ]
    )
    out = qc(df, persistence_hours=24)
    assert len(out) == 30                              # RH keeps rows alive
    # First 23 identical temps survive; hours 24..30 are nulled as a stuck run.
    assert out["air_temp_c"].isna().sum() >= 5


def test_qc_step_limit_nulls_spike():
    df = _frame(
        [
            {"station_id": "A", "valid_time": pd.Timestamp("2024-01-01T00:00"),
             "air_temp_c": 2.0, "source": "IEM"},
            {"station_id": "A", "valid_time": pd.Timestamp("2024-01-01T01:00"),
             "air_temp_c": 40.0, "source": "IEM"},   # +38C spike > 20C step limit
        ]
    )
    out = qc(df)
    assert out.loc[out["valid_time"] == pd.Timestamp("2024-01-01T01:00"),
                   "air_temp_c"].isna().all()


def test_normalize_floors_to_hour_and_dedupes():
    df = _frame(
        [
            {"station_id": "A", "valid_time": pd.Timestamp("2024-01-01T00:37"),
             "air_temp_c": 1.0, "source": "IEM"},
            {"station_id": "A", "valid_time": pd.Timestamp("2024-01-01T00:52"),
             "air_temp_c": 2.0, "source": "IEM"},   # same hour -> deduped
        ]
    )
    out = normalize_hourly(df)
    assert len(out) == 1
    assert (out["valid_time"] == pd.Timestamp("2024-01-01T00:00")).all()
