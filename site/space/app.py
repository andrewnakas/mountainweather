"""Hugging Face Space: mtnwx forecast demo.

A lightweight Gradio app that reads the same forecast.json the GitHub Pages site uses
(published to the mtnwx-models HF repo each run) and lets a user pick a station and
variable to see the corrected mountain forecast with its uncertainty band. Keeps the
whole data + model + demo story on Hugging Face alongside the user's other datasets.

Deployed by pushing this dir to the mtnwx-demo Space. Requires: gradio, huggingface_hub.
"""
from __future__ import annotations

import json
import os

import gradio as gr
from huggingface_hub import hf_hub_download

MODELS_REPO = os.environ.get("MTNWX_MODELS_REPO", "tree60weather/mtnwx-models")


def _load_forecast() -> dict:
    try:
        path = hf_hub_download(
            repo_id=MODELS_REPO, repo_type="model", filename="forecast/forecast.json"
        )
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return {"features": [], "error": str(exc)}


DATA = _load_forecast()
STATIONS = {
    f["properties"].get("name") or f["properties"]["station_id"]: f
    for f in DATA.get("features", [])
}
VAR_LABELS = {
    "air_temp_c": "Temperature (°C)",
    "wind_speed_ms": "Wind speed (m/s)",
    "wind_gust_ms": "Wind gust (m/s)",
    "relative_humidity_pct": "Humidity (%)",
}


def show(station: str, variable: str):
    f = STATIONS.get(station)
    if not f:
        return "No forecast loaded."
    fc = (f["properties"].get("forecast") or {}).get(variable)
    times = f["properties"].get("valid_times", [])
    if not fc:
        return f"No {variable} forecast for {station}."
    lines = [f"### {station} — {VAR_LABELS.get(variable, variable)}", "", "| Valid (UTC) | Forecast | q10–q90 |", "|---|---|---|"]
    for i, t in enumerate(times[:48]):
        pt = fc.get("point", [None])[i] if i < len(fc.get("point", [])) else None
        lo = fc.get("q10", [None])[i] if i < len(fc.get("q10", [])) else None
        hi = fc.get("q90", [None])[i] if i < len(fc.get("q90", [])) else None
        band = f"{lo} – {hi}" if lo is not None and hi is not None else "–"
        lines.append(f"| {t[5:16].replace('T', ' ')} | {pt if pt is not None else '–'} | {band} |")
    return "\n".join(lines)


with gr.Blocks(title="mtnwx — mountain weather") as demo:
    gr.Markdown("# 🏔 mtnwx\nHRRR post-processed for complex terrain. "
                f"Init: {DATA.get('init_time', 'n/a')}")
    with gr.Row():
        station = gr.Dropdown(sorted(STATIONS), label="Station")
        variable = gr.Dropdown(list(VAR_LABELS), value="air_temp_c", label="Variable")
    out = gr.Markdown()
    station.change(show, [station, variable], out)
    variable.change(show, [station, variable], out)

if __name__ == "__main__":
    demo.launch()
