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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mtnwx", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    p_config = sub.add_parser("config", help="Print resolved configuration")
    p_config.set_defaults(func=_cmd_config)

    p_stations = sub.add_parser("stations", help="Build the mountain station catalogue")
    p_stations.add_argument("--limit", type=int, default=None, help="Cap station count (smoke test)")
    p_stations.add_argument("--out", type=str, default=None, help="Output parquet path")
    p_stations.set_defaults(func=_cmd_stations)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
