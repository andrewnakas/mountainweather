"""Config loading + station-filter unit tests (no network required)."""
from __future__ import annotations

import pandas as pd

from mtnwx.config import load_configs
from mtnwx.data.stations import apply_mountain_filter


def test_configs_load():
    cfg = load_configs()
    assert set(cfg) >= {"region", "variables", "predictors", "model", "hub"}
    assert cfg["region"]["name"] == "western_conus"
    # Every phase-1 target must name a quantile objective.
    for name, spec in cfg["variables"]["targets"].items():
        assert "phase" in spec, name
        assert spec["lgbm_objective"] == "quantile", name


def test_predictor_fields_nonempty():
    cfg = load_configs()
    fields = cfg["predictors"]["hrrr_fields"]
    assert len(fields) >= 15
    # Elevation delta relies on grid_elevation_m being carried through.
    assert "grid_elevation_m" in cfg["predictors"]["grid_metadata"]


def test_mountain_filter_bounds_and_elevation():
    region = load_configs()["region"]
    df = pd.DataFrame(
        [
            # In-box, high enough -> kept
            {"lat": 40.0, "lon": -106.0, "elevation_m": 3000, "name": "peak", "network": "SNOTEL"},
            # In-box, too low -> dropped
            {"lat": 40.0, "lon": -106.0, "elevation_m": 500, "name": "valley", "network": "SNOTEL"},
            # Out of box (east of Rockies front), high -> dropped
            {"lat": 40.0, "lon": -90.0, "elevation_m": 3000, "name": "plains", "network": "SNOTEL"},
        ]
    )
    out = apply_mountain_filter(df, region)
    assert list(out["name"]) == ["peak"]
    assert out["needs_relief_check"].all()
