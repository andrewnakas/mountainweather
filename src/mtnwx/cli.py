"""mtnwx command-line interface.

Thin dispatcher over the milestone modules. Each subcommand imports its module
lazily so that, e.g., running ``mtnwx stations`` doesn't require lightgbm to be
installed. Subcommands are wired up as their milestones land.
"""
from __future__ import annotations

import argparse
import sys


def _cmd_config(args: argparse.Namespace) -> int:
    """Print the resolved config (smoke check that YAML loads)."""
    import json

    from mtnwx.config import load_configs

    cfg = load_configs()
    print(json.dumps(cfg, indent=2, default=str))
    return 0


def _cmd_stations(args: argparse.Namespace) -> int:
    from mtnwx.data import stations

    return stations.main(args)


def _cmd_obs(args: argparse.Namespace) -> int:
    from mtnwx.data import collect_obs

    return collect_obs.main(args)


def _cmd_terrain(args: argparse.Namespace) -> int:
    from mtnwx.data import terrain

    return terrain.main(args)


def _cmd_extract(args: argparse.Namespace) -> int:
    from mtnwx.data import hrrr

    return hrrr.main(args)


def _cmd_train(args: argparse.Namespace) -> int:
    from mtnwx import train

    return train.main(args)


def _cmd_verify(args: argparse.Namespace) -> int:
    from mtnwx import run_verify

    return run_verify.main(args)


def _cmd_predict(args: argparse.Namespace) -> int:
    from mtnwx import predict

    return predict.main(args)


def _cmd_export_forcings(args: argparse.Namespace) -> int:
    from mtnwx import export_forcings

    return export_forcings.main(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mtnwx", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    p_config = sub.add_parser("config", help="Print resolved configuration")
    p_config.set_defaults(func=_cmd_config)

    p_stations = sub.add_parser("stations", help="Build the mountain station catalogue")
    p_stations.add_argument("--limit", type=int, default=None, help="Cap station count (smoke test)")
    p_stations.add_argument("--out", type=str, default=None, help="Output parquet path")
    p_stations.set_defaults(func=_cmd_stations)

    p_obs = sub.add_parser("obs", help="Collect + QC hourly observations for the catalogue")
    p_obs.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC)")
    p_obs.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC, inclusive)")
    p_obs.add_argument("--stations", default=None, help="Station parquet (default: data/stations.parquet)")
    p_obs.add_argument("--out", default=None, help="Output parquet path")
    p_obs.set_defaults(func=_cmd_obs)

    p_terrain = sub.add_parser("terrain", help="Compute per-station DEM terrain features")
    p_terrain.add_argument("--stations", default=None, help="Station parquet (default: data/stations.parquet)")
    p_terrain.add_argument("--out", default=None, help="Output parquet path")
    p_terrain.set_defaults(func=_cmd_terrain)

    p_extract = sub.add_parser("extract", help="Extract HRRR predictors at stations (one month chunk)")
    p_extract.add_argument("--month", required=True, help="Init-time month YYYY-MM")
    p_extract.add_argument("--stations", default=None, help="Stations+terrain parquet")
    p_extract.add_argument("--out", default=None, help="Output parquet path")
    p_extract.add_argument("--workers", type=int, default=12, help="Concurrent init fetches")
    p_extract.set_defaults(func=_cmd_extract)

    p_train = sub.add_parser("train", help="Train LightGBM quantile post-processors")
    p_train.add_argument("--table", default=None, help="Training table parquet")
    p_train.add_argument("--targets", default=None, help="Comma-separated target list")
    p_train.add_argument("--out", default=None, help="Model output dir")
    p_train.set_defaults(func=_cmd_train)

    p_verify = sub.add_parser("verify", help="Score models vs NBM/HRRR/persistence + report")
    p_verify.add_argument("--table", default=None, help="Training table parquet")
    p_verify.add_argument("--models", default=None, help="Models dir")
    p_verify.add_argument("--out", default=None, help="Report output dir")
    p_verify.add_argument("--no-nbm", action="store_true", help="Skip the NBM benchmark fetch")
    p_verify.set_defaults(func=_cmd_verify)

    p_predict = sub.add_parser("predict", help="Forecast from the latest HRRR cycle")
    p_predict.add_argument("--stations", default=None, help="Stations+terrain parquet")
    p_predict.add_argument("--models", default=None, help="Models dir")
    p_predict.add_argument("--init", default=None, help="HRRR init time (default: latest)")
    p_predict.add_argument("--out", default=None, help="Output JSON path")
    p_predict.set_defaults(func=_cmd_predict)

    p_forcings = sub.add_parser("export-forcings", help="Export SNOTEL forcings for SnowWatch")
    p_forcings.add_argument("--forecast", default=None, help="forecast.json path")
    p_forcings.add_argument("--out", default=None, help="Output parquet path")
    p_forcings.set_defaults(func=_cmd_export_forcings)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
