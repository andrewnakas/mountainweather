#!/usr/bin/env python3
"""Collect global obs for ONE batch of stations and upload the part to HF (resumable).

Inline obs collection for all ~26k global stations takes >90 min and gets reclaimed by
the runner throttle mid-collection — losing everything (no incremental cache). This splits
the catalogue into N batches; each batch (~2k stations, ~8 min) collects its obs and
uploads a part, with skip-if-exists, so the whole thing is resumable and each job is short
enough to survive. The builder then loads all parts as the obs cache.

    MTNWX_REGION=global python scripts/collect_obs_batch.py --batch 3 --nbatches 12

Parts land at obs_parts/obsG_{start}_{end}_n{N}_{hash}/part-{batch:02d}-of-{nbatches:02d}.parquet
in the verify dataset — the same window+hash namespace the builder derives, so it finds them.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtnwx.config import data_dir, load_configs  # noqa: E402
from mtnwx.data.collect_obs import collect  # noqa: E402

# The full MVP obs window (matches the builder's pinned cache window).
START = pd.Timestamp("2024-01-01").date()
END = pd.Timestamp("2025-12-31").date()


def part_key(stations: pd.DataFrame, batch: int, nbatches: int) -> tuple[str, str]:
    sid_hash = hashlib.md5("|".join(sorted(stations["station_id"].astype(str))).encode()).hexdigest()[:8]
    ns = len(stations)
    folder = f"obs_parts/obsG_{START}_{END}_n{ns}_{sid_hash}"
    return folder, f"{folder}/part-{batch:02d}-of-{nbatches:02d}.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True, help="0-indexed batch number")
    ap.add_argument("--nbatches", type=int, required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    hub = load_configs()["hub"]
    from huggingface_hub import HfApi, hf_hub_download

    # Load the FULL global catalogue (same file the builder uses), then slice this batch.
    st_name = "stations_terrain_global.parquet"
    try:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], st_name, repo_type="dataset"))
    except Exception:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_global.parquet", repo_type="dataset"))
    stations = stations.sort_values("station_id").reset_index(drop=True)
    print(f"full global catalogue: {len(stations)} stations", flush=True)

    _, key = part_key(stations, args.batch, args.nbatches)

    # Skip if this part is already on HF.
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    try:
        existing = set(api.list_repo_files(hub["datasets"]["verify"], repo_type="dataset"))
    except Exception:
        existing = set()
    if key in existing:
        print(f"part already exists, skipping: {key}", flush=True)
        return 0

    # Deterministic contiguous slice for this batch (np.array_split preserves the whole set).
    idx_batches = np.array_split(np.arange(len(stations)), args.nbatches)
    my_idx = idx_batches[args.batch]
    batch_st = stations.iloc[my_idx].reset_index(drop=True)
    print(f"batch {args.batch}/{args.nbatches}: {len(batch_st)} stations, collecting {START}..{END}", flush=True)

    obs = collect(batch_st, START, END, workers=args.workers)
    for c in obs.select_dtypes("float64").columns:
        obs[c] = obs[c].astype("float32")
    print(f"  collected {len(obs)} obs rows for batch {args.batch}", flush=True)

    dd = data_dir()
    out = dd / f"obs_part_{args.batch:02d}.parquet"
    obs.to_parquet(out, index=False, compression="zstd")
    from mtnwx.data.hub_io import upload_file
    upload_file(out, key, hub["datasets"]["verify"], repo_type="dataset")
    print(f"uploaded {key}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
