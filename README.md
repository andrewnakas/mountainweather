# mountainweather (`mtnwx`)

A high-resolution, short-range weather model built to be **the best available for mountain
climates in North America** — and to prove it against the operational benchmarks.

Raw numerical weather prediction models, even the 3 km HRRR, carry large systematic errors in
complex terrain: smoothed elevations, missed valley cold pools, unresolved ridge-top wind
acceleration, mislocated snow levels. `mtnwx` corrects those errors with machine-learned
**post-processing**: gradient-boosted models trained on years of *(forecast, terrain,
observation)* triplets at mountain stations. The target it must beat is NOAA's calibrated
**National Blend of Models (NBM)** — the strongest operational guidance — measured at held-out
mountain stations.

## What it does

- **Inputs**: HRRR 3 km forecasts (from the [dynamical.org](https://dynamical.org) analysis-ready
  Zarr archive), high-resolution terrain (Copernicus GLO-30 DEM), and surface observations from
  SNOTEL, RAWS, ASOS/AWOS and mesonets.
- **Outputs**: bias-corrected hourly forecasts of 2 m temperature, 10 m wind speed & gusts,
  humidity, and (phase 2) precipitation & snow — as point forecasts with calibrated quantile
  bands, out to 48 h.
- **Delivery**: static JSON/GeoJSON published to GitHub Pages and mirrored to a Hugging Face
  Space, consumed by [tree60weather.com](https://tree60weather.com).

## Design at a glance

```
[one-time backfill]                                    [hourly operations]
HRRR Zarr (dynamical.org) ─┐                           latest HRRR cycle (AWS)
Synoptic/SNOTEL/IEM/MRMS ──┼→ training parquet ──→ train ──→ models │
DEM terrain features ──────┘   (HF Datasets)    (LightGBM,  (HF Hub)─┴→ predict → JSON/GeoJSON
                                                 GH Actions)              → GitHub Pages + HF Space
NBM archive (AWS) ────→ verification vs baselines → skill report          → tree60weather.com
```

All data and model artifacts live on Hugging Face; all compute runs in GitHub Actions
(LightGBM is CPU-light — no GPU required).

## Status

Early development. See `PLAN.md` for milestones and `configs/` for the region, variable, and
station definitions. This project shares station tooling and snow-QC logic with the sibling
[SnowWatch](../SnowWatch) SNOTEL snow-depth predictor.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,terrain,dev]"
pytest                                   # unit tests
python -m mtnwx.cli --help               # CLI entry points
```

## License

MIT for code. Data products inherit their sources' licenses (HRRR/NBM: public domain;
dynamical.org archives: CC BY 4.0).
