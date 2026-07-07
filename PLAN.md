# mtnwx implementation plan

Short-range, high-resolution mountain weather post-processor for North America. Beats
the National Blend of Models (NBM) at held-out mountain stations by ML-correcting HRRR.

## Milestones

- **M0 — Scaffold** ✅ package, configs, CLI, CI, HF setup script.
- **M1 — Stations + obs** — station catalogue (SNOTEL + Synoptic), obs downloaders
  (NRCS/IEM/Synoptic), QC → parquet.
- **M2 — Terrain** — Copernicus GLO-30 DEM; per-station elevation, elevation-delta vs
  HRRR grid, slope, aspect, multi-scale TPI, Sx wind exposure; prunes flat sites.
- **M3 — HRRR backfill** — extract HRRR-Zarr predictors at all stations, leads 1–48,
  ~6 years, as a resumable GitHub Actions matrix → parquet on HF.
- **M4 — Train v1** — LightGBM quantile models for temp/wind/gust/RH → HF Hub.
- **M5 — Verify** — extract NBM; MAE/RMSE/CRPS vs NBM/HRRR/persistence at held-out
  stations *and* months, by lead/elevation/season; skill report. **The benchmark moment.**
- **M6 — Operations** — hourly forecast workflow, Pages JSON, demo map, tree60weather.com.
- **M7 — Precip/snow** — two-stage models; SnowWatch `targets.py` QC + NOHRSC/MRMS truth.
- **M8 — SnowWatch loop** — publish bias-corrected forcings at SNOTEL points.
- **M9 — Iterate** — close verification gaps; optional RRFS predictors, gridded output.

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
