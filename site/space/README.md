---
title: mtnwx mountain weather
emoji: 🏔
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
---

# mtnwx — mountain weather demo

Interactive demo of the `mtnwx` mountain-weather post-processor: HRRR forecasts
bias-corrected for complex terrain, with calibrated uncertainty bands, at SNOTEL/RAWS/
mesonet stations across the western US.

Static Space — a self-contained Leaflet map that reads the latest `forecast.json`
published by the hourly GitHub Actions workflow. Powered by data from
[dynamical.org](https://dynamical.org) and NOAA HRRR. Source:
[github.com/andrewnakas/mountainweather](https://github.com/andrewnakas/mountainweather).
