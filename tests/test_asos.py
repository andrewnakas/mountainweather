"""ASOS module tests — schema conformance without network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtnwx.data.asos import ASOS_BASE, _year_urls, fetch_asos_hourly
from mtnwx.data.obs import OBS_COLUMNS, qc


def test_year_urls_span():
    from datetime import date

    urls = _year_urls(date(2019, 3, 1), date(2021, 6, 1))
    assert len(urls) == 3
    assert urls[0] == f"{ASOS_BASE}/year=2019/data.parquet"
    assert urls[-1] == f"{ASOS_BASE}/year=2021/data.parquet"


def test_fetch_empty_station_list_returns_schema():
    from datetime import date

    out = fetch_asos_hourly([], date(2024, 1, 1), date(2024, 1, 2))
    assert list(out.columns) == OBS_COLUMNS
    assert out.empty


def test_asos_rows_survive_qc():
    # A synthetic ASOS-shaped frame (already normalized to OBS_COLUMNS) must pass QC,
    # including the wind/gust columns SNOTEL never provides.
    times = pd.date_range("2024-01-15", periods=5, freq="h")
    df = pd.DataFrame(
        {
            "station_id": ["ASOS:0CO"] * 5,
            "valid_time": times,
            "air_temp_c": [-28.0, -27.3, -26.7, -26.0, -25.0],
            "dewpoint_c": [-30.0, -29.0, -28.5, -28.0, -27.0],
            "relative_humidity_pct": [82.0, 80.0, 78.0, 79.0, 81.0],
            "wind_speed_ms": [18.5, 17.1, 15.8, 16.2, 14.0],
            "wind_gust_ms": [22.6, 24.2, 23.1, 21.0, 19.5],
            "wind_dir_deg": [280.0, 275.0, 290.0, 285.0, 270.0],
            "precip_1h_mm": [0.0, 0.2, 0.0, 0.0, 0.1],
            "source": ["ASOS"] * 5,
        }
    )[OBS_COLUMNS]
    out = qc(df)
    assert len(out) == 5
    assert out["wind_speed_ms"].notna().all()
    assert out["wind_gust_ms"].notna().all()
    # Dewpoint <= temp preserved (no nulling).
    assert out["dewpoint_c"].notna().all()
    assert np.isclose(out["wind_speed_ms"].iloc[0], 18.5)
