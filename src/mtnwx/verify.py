"""Verification: does mtnwx beat NBM, raw HRRR, and persistence at held-out mountains?

This is the acceptance test. On the held-out set (recent months AND unseen stations) we
score four forecasts against the QC'd observations:

  - **mtnwx**       — our LightGBM quantile post-processor (point = q0.50).
  - **raw HRRR**    — the model's own forecast of the target (the thing we correct).
  - **HRRR+lapse**  — raw HRRR temperature corrected by a standard lapse rate applied to
                      the elevation delta (a cheap physical baseline for temperature).
  - **NBM**         — NOAA's calibrated blend (data/nbm.py), the strongest benchmark.
  - **persistence** — last observed value carried forward by the lead time.

Metrics per (variable, forecast): MAE, RMSE, bias, and — for mtnwx's quantiles — CRPS and
coverage of the central 80% interval. Everything is also broken out by lead time and
elevation band so we can see *where* we win. A paired bootstrap gives confidence that the
MAE difference vs NBM is real, not noise.

Output: a metrics table (parquet + JSON) and a human-readable skill report the M6 site
renders. Success = lower MAE and CRPS than NBM and raw HRRR across 1-24 h leads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Standard environmental lapse rate (deg C per metre) for the physical temp baseline.
LAPSE_RATE_C_PER_M = 6.5 / 1000.0

ELEV_BANDS = [(0, 1500), (1500, 2200), (2200, 2800), (2800, 3500), (3500, 9000)]


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(obs)
    return float(np.mean(np.abs(pred[m] - obs[m]))) if m.any() else float("nan")


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(obs)
    return float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2))) if m.any() else float("nan")


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(obs)
    return float(np.mean(pred[m] - obs[m])) if m.any() else float("nan")


def crps_from_quantiles(quantile_preds: dict[float, np.ndarray], obs: np.ndarray) -> float:
    """Approximate CRPS as the mean pinball loss across quantile levels.

    For a set of predicted quantiles, the average pinball (quantile) loss is a proper
    scoring rule that equals CRPS in the limit of dense quantiles — the standard way to
    score quantile forecasts."""
    qs = sorted(quantile_preds)
    total = np.zeros_like(obs, dtype="float64")
    count = 0
    for q in qs:
        pred = quantile_preds[q]
        m = np.isfinite(pred) & np.isfinite(obs)
        if not m.any():
            continue
        diff = obs - pred
        loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
        total_masked = np.where(m, loss, 0.0)
        total += total_masked
        count += 1
    if count == 0:
        return float("nan")
    valid = np.isfinite(obs)
    return float(np.mean((total / count)[valid])) if valid.any() else float("nan")


def interval_coverage(lo: np.ndarray, hi: np.ndarray, obs: np.ndarray) -> float:
    """Fraction of observations inside [lo, hi] — should approach the nominal level."""
    m = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(obs)
    if not m.any():
        return float("nan")
    return float(np.mean((obs[m] >= lo[m]) & (obs[m] <= hi[m])))


def persistence_forecast(df: pd.DataFrame, target: str, obs_col: str) -> np.ndarray:
    """Persistence: the observed value at init_time, carried to valid_time.

    Requires the obs at each init hour; joined upstream as ``{obs_col}_at_init``. Where
    that is missing, persistence is NaN (excluded from its own metric)."""
    col = f"{obs_col}_at_init"
    return df[col].to_numpy() if col in df.columns else np.full(len(df), np.nan)


def lapse_corrected_temp(df: pd.DataFrame) -> np.ndarray:
    """Raw HRRR 2 m temp corrected by lapse rate over the elevation delta.

    If the station sits ``delta`` metres above its HRRR grid cell, subtract
    lapse_rate*delta from the model temperature — the textbook physical correction we
    must beat with ML."""
    if not {"temperature_2m", "elevation_delta_m"}.issubset(df.columns):
        return np.full(len(df), np.nan)
    return (df["temperature_2m"] - LAPSE_RATE_C_PER_M * df["elevation_delta_m"]).to_numpy()


def elevation_band(elev: float) -> str:
    for lo, hi in ELEV_BANDS:
        if lo <= elev < hi:
            return f"{lo}-{hi}m"
    return "unknown"


def score_frame(
    df: pd.DataFrame,
    target: str,
    hrrr_field: str,
    mtnwx_point: np.ndarray,
    mtnwx_quantiles: dict[float, np.ndarray] | None = None,
    nbm_col: str | None = None,
) -> pd.DataFrame:
    """Score every forecast against ``target`` obs; return long metrics by lead+band.

    ``df`` must contain the target obs column, the raw HRRR field, elevation_delta_m,
    lead_hour, and (optionally) the NBM column. Returns one row per
    (forecast, lead_group, elevation_band) with mae/rmse/bias (+ crps for mtnwx)."""
    obs = df[target].to_numpy()
    forecasts: dict[str, np.ndarray] = {
        "mtnwx": mtnwx_point,
        "raw_hrrr": df[hrrr_field].to_numpy() if hrrr_field in df.columns else np.full(len(df), np.nan),
        "persistence": persistence_forecast(df, target, target),
    }
    if target == "air_temp_c":
        forecasts["hrrr_lapse"] = lapse_corrected_temp(df)
    if nbm_col and nbm_col in df.columns:
        forecasts["nbm"] = df[nbm_col].to_numpy()

    lead = df["lead_hour"].to_numpy()
    lead_group = np.select(
        [lead <= 6, lead <= 12, lead <= 24, lead <= 48],
        ["01-06h", "07-12h", "13-24h", "25-48h"],
        default="48h+",
    )
    band = (
        df["elevation_m"].map(elevation_band).to_numpy()
        if "elevation_m" in df.columns
        else np.full(len(df), "all")
    )

    rows = []
    for fname, pred in forecasts.items():
        for lg in np.unique(lead_group):
            for bd in np.unique(band):
                m = (lead_group == lg) & (band == bd)
                if m.sum() < 30:
                    continue
                row = {
                    "target": target, "forecast": fname, "lead_group": lg,
                    "elevation_band": bd, "n": int(m.sum()),
                    "mae": mae(pred[m], obs[m]), "rmse": rmse(pred[m], obs[m]),
                    "bias": bias(pred[m], obs[m]),
                }
                if fname == "mtnwx" and mtnwx_quantiles is not None:
                    qm = {q: v[m] for q, v in mtnwx_quantiles.items()}
                    row["crps"] = crps_from_quantiles(qm, obs[m])
                    if 0.1 in mtnwx_quantiles and 0.9 in mtnwx_quantiles:
                        row["coverage_80"] = interval_coverage(qm[0.1], qm[0.9], obs[m])
                rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_mae_diff(
    pred_a: np.ndarray, pred_b: np.ndarray, obs: np.ndarray, *, n: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap the MAE(b) - MAE(a) difference. Returns (mean_diff, lo95, hi95).

    Positive => forecast A (mtnwx) has lower error than B (the benchmark). CI excluding
    zero => the improvement is statistically real, not sampling noise."""
    m = np.isfinite(pred_a) & np.isfinite(pred_b) & np.isfinite(obs)
    a, b, o = pred_a[m], pred_b[m], obs[m]
    if len(o) < 50:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    idx = np.arange(len(o))
    for i in range(n):
        s = rng.choice(idx, size=len(o), replace=True)
        diffs[i] = np.mean(np.abs(b[s] - o[s])) - np.mean(np.abs(a[s] - o[s]))
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
