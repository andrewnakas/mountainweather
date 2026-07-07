"""Predict → GeoJSON assembly test with synthetic boosters (no network, no lightgbm)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtnwx.export_forcings import build_forcings
from mtnwx.predict import predict_table, to_station_json


class _FakeBooster:
    def __init__(self, offset):
        self.offset = offset

    def predict(self, X):
        return np.full(len(X), self.offset, dtype="float32")


def _fc_frame():
    models = {"air_temp_c": {0.1: _FakeBooster(-2), 0.5: _FakeBooster(0), 0.9: _FakeBooster(2)}}
    meta = {"features": ["lead_hour", "elevation_delta_m"]}
    df = pd.DataFrame(
        {
            "station_id": ["308:CO:SNTL"] * 3,
            "init_time": [pd.Timestamp("2024-01-15")] * 3,
            "lead_hour": [1.0, 2.0, 3.0],
            "valid_time": pd.to_datetime(
                ["2024-01-15 01:00", "2024-01-15 02:00", "2024-01-15 03:00"]
            ),
            "elevation_delta_m": [100.0, 100.0, 100.0],
        }
    )
    return predict_table(df, models, meta), models


def test_predict_point_equals_median():
    fc, _ = _fc_frame()
    assert "air_temp_c" in fc.columns and "air_temp_c_q50" in fc.columns
    assert (fc["air_temp_c"] == fc["air_temp_c_q50"]).all()


def test_geojson_structure():
    fc, models = _fc_frame()
    stations = pd.DataFrame(
        {
            "station_id": ["308:CO:SNTL"],
            "name": ["Test Peak"],
            "lat": [40.0],
            "lon": [-106.0],
            "elevation_m": [3000.0],
        }
    )
    payload = to_station_json(fc, stations, models, pd.Timestamp("2024-01-15"))
    assert payload["type"] == "FeatureCollection"
    f0 = payload["features"][0]
    assert f0["geometry"]["coordinates"] == [-106.0, 40.0]
    at = f0["properties"]["forecast"]["air_temp_c"]
    assert at["point"] == [0.0, 0.0, 0.0]
    assert at["q10"] == [-2.0, -2.0, -2.0] and at["q90"] == [2.0, 2.0, 2.0]
    assert len(f0["properties"]["valid_times"]) == 3


def test_forcings_export_snotel_only():
    fc, models = _fc_frame()
    stations = pd.DataFrame(
        {"station_id": ["308:CO:SNTL"], "name": ["P"], "lat": [40.0], "lon": [-106.0],
         "elevation_m": [3000.0]}
    )
    payload = to_station_json(fc, stations, models, pd.Timestamp("2024-01-15"))
    forc = build_forcings(payload)
    assert len(forc) == 3
    assert (forc["triplet"] == "308:CO:SNTL").all()
    assert set(forc.columns) >= {"triplet", "valid_time", "tmean_c", "wind_speed_ms"}
