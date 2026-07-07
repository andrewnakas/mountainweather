"""Render the verification metrics into a human-readable skill report.

Takes the long metrics table from verify.py and produces:
  - a markdown/HTML summary (headline: our MAE/CRPS vs NBM and raw HRRR),
  - the per-lead, per-elevation-band breakout tables,
  - the bootstrap significance of the improvement over NBM.

The site (M6) embeds the HTML; the JSON is the machine-readable scorecard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def headline(metrics: pd.DataFrame) -> pd.DataFrame:
    """Overall MAE per (target, forecast), pooled across leads and bands.

    Weighted by sample count so it's a true pooled MAE, not a mean of MAEs."""
    m = metrics.dropna(subset=["mae"]).copy()
    m["mae_weighted"] = m["mae"] * m["n"]
    grp = m.groupby(["target", "forecast"]).agg(
        mae=("mae_weighted", "sum"), n=("n", "sum")
    )
    grp["mae"] = grp["mae"] / grp["n"]
    return grp.reset_index()[["target", "forecast", "mae", "n"]]


def skill_vs_benchmark(headline_df: pd.DataFrame, benchmark: str = "nbm") -> pd.DataFrame:
    """Percent MAE reduction of mtnwx vs each benchmark, per target."""
    rows = []
    for target, g in headline_df.groupby("target"):
        g = g.set_index("forecast")["mae"]
        if "mtnwx" not in g:
            continue
        ours = g["mtnwx"]
        for bench in ("nbm", "raw_hrrr", "hrrr_lapse", "persistence"):
            if bench in g and g[bench] > 0:
                rows.append(
                    {
                        "target": target,
                        "benchmark": bench,
                        "mtnwx_mae": round(ours, 3),
                        "benchmark_mae": round(g[bench], 3),
                        "pct_improvement": round(100 * (g[bench] - ours) / g[bench], 1),
                    }
                )
    return pd.DataFrame(rows)


def render_markdown(metrics: pd.DataFrame, meta: dict | None = None) -> str:
    """Full markdown skill report."""
    hd = headline(metrics)
    skill = skill_vs_benchmark(hd)
    lines = ["# mtnwx skill report", ""]
    if meta:
        lines += [
            f"- Train rows: {meta.get('n_train_rows', '?')}  |  Test rows: {meta.get('n_test_rows', '?')}",
            f"- Held-out stations: {len(meta.get('held_out_stations', []))}",
            "",
        ]
    lines += ["## Headline: MAE vs benchmarks (held-out set)", ""]
    if not skill.empty:
        lines.append("| Target | Benchmark | mtnwx MAE | Benchmark MAE | Improvement |")
        lines.append("|---|---|---|---|---|")
        for _, r in skill.iterrows():
            sign = "✅" if r["pct_improvement"] > 0 else "⚠️"
            lines.append(
                f"| {r['target']} | {r['benchmark']} | {r['mtnwx_mae']} | "
                f"{r['benchmark_mae']} | {sign} {r['pct_improvement']}% |"
            )
    lines += ["", "## MAE by lead time and elevation band", ""]
    piv = metrics.pivot_table(
        index=["target", "lead_group", "elevation_band"],
        columns="forecast", values="mae",
    ).round(2)
    lines.append("```")
    lines.append(piv.to_string())
    lines.append("```")
    return "\n".join(lines)


def write_report(metrics: pd.DataFrame, out_dir: Path, meta: dict | None = None) -> None:
    """Write scorecard.json + skill_report.md to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hd = headline(metrics)
    skill = skill_vs_benchmark(hd)
    scorecard = {
        "headline": hd.to_dict(orient="records"),
        "skill_vs_benchmark": skill.to_dict(orient="records"),
        "meta": meta or {},
    }
    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2, default=str))
    (out_dir / "skill_report.md").write_text(render_markdown(metrics, meta))
    metrics.to_parquet(out_dir / "metrics.parquet", index=False)
