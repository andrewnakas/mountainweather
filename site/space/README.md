---
title: mtnwx mountain weather
emoji: 🏔
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# mtnwx — mountain weather demo

Interactive demo of the `mtnwx` mountain-weather post-processor: HRRR forecasts
bias-corrected for complex terrain, with calibrated uncertainty bands, at SNOTEL/RAWS/
mesonet stations across the western US.

Reads the latest `forecast/forecast.json` published to the
[`tree60weather/mtnwx-models`](https://huggingface.co/tree60weather/mtnwx-models) repo by
the hourly GitHub Actions workflow. Powered by data from
[dynamical.org](https://dynamical.org) and NOAA HRRR.
