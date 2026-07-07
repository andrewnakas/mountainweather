# mtnwx implementation plan

Short-range, high-resolution mountain weather post-processor for North America. Beats
the National Blend of Models (NBM) at held-out mountain stations by ML-correcting HRRR.

## Milestones

- **M0 — Scaffold** ✅ package, configs, CLI, CI, HF setup script.
- **M1 — Stations + obs** ✅ SNOTEL (AWDB) + Synoptic catalogue, mountain filter; hourly
  obs from SNOTEL/IEM/Synoptic; QC (range/step/flatline). CLI `stations`, `obs`.
- **M2 — Terrain** ✅ Copernicus GLO-30 windowed reads; elevation, dem_elevation,
  relief, slope, aspect, multi-scale TPI; relief filter finalizes the catalogue. `terrain`.
- **M3 — HRRR backfill** ✅ pointwise KD-tree extraction from the dynamical.org zarr v3
  archive (leads 0–48); resumable per-month matrix workflow → HF. `extract`.
- **M4 — Train v1** ✅ LightGBM quantile models for temp/wind/gust/RH; no-leakage splits;
  artifacts → HF. `train`, `build_training_table.py`, `train.yml`.
- **M5 — Verify** ✅ NBM (Open-Meteo) + raw-HRRR + lapse + persistence baselines;
  MAE/RMSE/CRPS/coverage by lead+elevation; paired bootstrap; skill report. `verify`.
- **M6 — Operations** ✅ `predict` (latest cycle → forecast GeoJSON); Leaflet site;
  hourly `forecast.yml` → Pages; HF Space demo (`site/space/`).
- **M8 — SnowWatch loop** ✅ (partial) `export_forcings.py` emits bias-corrected SNOTEL
  forcings for SnowWatch to ingest.
- **M7 — Precip/snow** — *next*: two-stage models; SnowWatch `targets.py` QC + NOHRSC/MRMS.
- **M9 — Iterate** — full-scale backfill run, tune features/models, optional RRFS/gridded.

**Status:** the full pipeline is code-complete and unit-tested; the remaining work is
running the full-scale backfill in Actions (needs `HF_TOKEN` + a public repo) and the
phase-2 precip/snow models.

## Data sources

| Source | Use | Access |
|---|---|---|
| dynamical.org HRRR Zarr | training predictors + live forecast | `https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr` (CC BY 4.0) |
| NRCS AWDB | SNOTEL obs (temp/precip/SWE/depth) | REST, no key |
| Iowa Environmental Mesonet | ASOS/RAWS bulk history | free bulk archive |
| Synoptic Data | mesonet obs + station metadata | free tier, `SYNOPTIC_API_TOKEN` |
| Copernicus GLO-30 DEM | terrain features | AWS open data |
| NBM (AWS `noaa-nbm-grib2-pds`) | benchmark comparator | free |
| MRMS / NOHRSC | precip/snow truth cross-check | dynamical.org / NOHRSC |

## Secrets (GitHub Actions repository secrets)

- `HF_TOKEN` — Hugging Face write token (datasets + model + Space).
- `SYNOPTIC_API_TOKEN` — Synoptic Data free-tier token.

## Verification acceptance test

The M5 skill report is the primary acceptance test: our MAE/CRPS must beat NBM, raw
HRRR, and persistence at held-out mountain stations across held-out months, reported by
lead time and elevation band.

## Sibling project

Shares station tooling and snow-QC with `../SnowWatch` (SNOTEL snow-depth predictor).
We copy-adapt its code (no live import) and can feed it bias-corrected forcings (M8).
