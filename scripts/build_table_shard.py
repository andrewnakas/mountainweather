#!/usr/bin/env python3
"""Build ONE training-table part (one GFS-shard month) and upload it to HF (resumable).

At 26k stations the monolithic table build downloads ~7 GB of shards + 1.9 GB obs and
runs >48 min — longer than a fair-use runner survives, so it never finishes. This does the
join for a SINGLE GFS month: download that one shard, join against the cached obs parts +
ECMWF, upload the part, skip-if-exists. Each job is short (~5-10 min) and survivable; the
training job then just loads the finished table_parts/ (no joining).

    MTNWX_REGION=global python scripts/build_table_shard.py --month 2024-06

Parts land at table_parts/<window+hash>/part-<month>.parquet in the training dataset.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtnwx.config import data_dir, load_configs  # noqa: E402
from build_training_table_global import (  # noqa: E402
    OBS_JOIN_COLS, build_global_table, load_ecmwf, prepare_obs, table_parts_prefix,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="Single GFS shard month YYYY-MM")
    ap.add_argument("--months", help="Comma-separated months (loads obs+ECMWF once, loops)")
    ap.add_argument("--ecmwf-slim-only", action="store_true",
                    help="Just build+cache the ECMWF slim frame (short job, better survival), then exit")
    ap.add_argument("--obs-merge-only", action="store_true",
                    help="Merge the 12 obs batch parts into ONE obs cache file (so the table build "
                         "loads obs in a single fast read instead of 12 part downloads), then exit")
    args = ap.parse_args()
    hub = load_configs()["hub"]
    dd = data_dir()
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)

    # Merge obs parts -> single obs cache file. The builder checks the single-file obs_key
    # cache; assembling it once turns the table build's obs load from 12 downloads (~1.9 GB,
    # too slow to reach month 1 before reclaim) into one read.
    if args.obs_merge_only:
        import hashlib
        try:
            stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_terrain_global.parquet", repo_type="dataset"))
        except Exception:
            stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_global.parquet", repo_type="dataset"))
        stations = stations.sort_values("station_id").reset_index(drop=True)
        sidh = hashlib.md5("|".join(sorted(stations["station_id"].astype(str))).encode()).hexdigest()[:8]
        obs_key = f"obs/obsG_2024-01-01_2025-12-31_n{len(stations)}_{sidh}.parquet"
        files = set(api.list_repo_files(hub["datasets"]["verify"], repo_type="dataset"))
        if obs_key in files:
            print(f"single-file obs cache already exists: {obs_key}", flush=True)
            return 0
        # CHECKPOINTED merge of the 12 obs parts (each ~160 MB) — the runner is reclaimed
        # ~2min into downloading all of them, so fold + checkpoint every 4 parts and resume.
        from build_training_table_global import obs_parts_prefix, OBS_JOIN_COLS
        from mtnwx.data.hub_io import upload_file
        prefix = obs_parts_prefix(stations)
        vfiles = api.list_repo_files(hub["datasets"]["verify"], repo_type="dataset")
        part_names = sorted(f for f in vfiles if f.startswith(prefix + "/") and f.endswith(".parquet"))
        if not part_names:
            print("ERROR: obs parts not found"); return 1
        ck_key = obs_key.replace(".parquet", "_ckpt.parquet")
        done_key = obs_key.replace(".parquet", "_ckpt_done.txt")
        obs, folded = None, set()
        try:
            obs = pd.read_parquet(hf_hub_download(hub["datasets"]["verify"], ck_key, repo_type="dataset"))
            folded = set(open(hf_hub_download(hub["datasets"]["verify"], done_key, repo_type="dataset")).read().split())
            print(f"resumed obs merge: {len(folded)} parts, {len(obs)} rows", flush=True)
        except Exception:
            obs, folded = None, set()
        todo = [pn for pn in part_names if pn not in folded]
        for i, pn in enumerate(todo, 1):
            pp = hf_hub_download(hub["datasets"]["verify"], pn, repo_type="dataset")
            part = pd.read_parquet(pp, columns=[c for c in OBS_JOIN_COLS])
            os.remove(pp) if os.path.exists(pp) else None
            obs = part if obs is None else pd.concat([obs, part], ignore_index=True)
            del part; folded.add(pn)
            if i % 4 == 0 or i == len(todo):
                cf = dd / "obs_ckpt.parquet"; obs.to_parquet(cf, index=False, compression="zstd")
                upload_file(cf, ck_key, hub["datasets"]["verify"], repo_type="dataset")
                (dd / "obs_done.txt").write_text("\n".join(sorted(folded)))
                upload_file(dd / "obs_done.txt", done_key, hub["datasets"]["verify"], repo_type="dataset")
                print(f"obs checkpoint: {len(folded)}/{len(part_names)} ({len(obs)} rows)", flush=True)
        outp = dd / "obsG_merged.parquet"
        obs.to_parquet(outp, index=False, compression="zstd")
        upload_file(outp, obs_key, hub["datasets"]["verify"], repo_type="dataset")
        print(f"merged obs cache built: {obs_key} ({len(obs)} rows)", flush=True)
        return 0

    # Cache-only mode: build the ECMWF slim frame CHUNKED + resumable. Downloading all
    # ~4.6 GB of ECMWF shards in one job takes >5 min and the runner is reclaimed ~3 min
    # into it (persistent throttle). Instead derive ONE shard's slim part per iteration,
    # upload it (skip-if-exists), and once all parts exist, merge+dedup into the final
    # slim cache. Each per-shard step downloads ~220 MB (<1 min) — survivable.
    if args.ecmwf_slim_only:
        from build_training_table_global import add_ecmwf_derived, _ecmwf_slim_key
        files = set(api.list_repo_files(hub["datasets"]["training"], repo_type="dataset"))
        ecmwf_names = sorted(f for f in files if f.startswith("ecmwf_global/") and f.endswith(".parquet"))
        slim_key = _ecmwf_slim_key(ecmwf_names)
        if slim_key in files:
            print(f"ECMWF slim already cached: {slim_key}", flush=True)
            return 0
        from mtnwx.data.hub_io import upload_file
        # per-shard slim parts land under a sibling folder keyed by the same shard set
        part_dir = slim_key.replace(".parquet", "_parts")
        for name in ecmwf_names:
            m = os.path.basename(name).replace("ecmwf_", "").replace(".parquet", "")
            ppart = f"{part_dir}/{m}.parquet"
            if ppart in files:
                print(f"skip slim part {m} (exists)", flush=True)
                continue
            p = hf_hub_download(hub["datasets"]["training"], name, repo_type="dataset")
            slim = add_ecmwf_derived(pd.read_parquet(p))
            os.remove(p) if os.path.exists(p) else None
            # Dedup within the shard-month up front (6-hourly inits × leads create many
            # duplicate station×valid_time rows) so each part — and thus the merge
            # download — is a fraction of the raw shard size.
            slim["valid_time"] = pd.to_datetime(slim["valid_time"])
            slim = slim.drop_duplicates(["station_id", "valid_time"], keep="last")
            outp = dd / f"ecmwf_slimpart_{m}.parquet"
            slim.to_parquet(outp, index=False, compression="zstd")
            upload_file(outp, ppart, hub["datasets"]["training"], repo_type="dataset")
            os.remove(outp) if os.path.exists(outp) else None
            print(f"uploaded slim part {m} ({len(slim)} rows)", flush=True)
        # All per-shard parts present? merge+dedup -> final slim cache.
        files = set(api.list_repo_files(hub["datasets"]["training"], repo_type="dataset"))
        part_names = sorted(f for f in files if f.startswith(part_dir + "/") and f.endswith(".parquet"))
        if len(part_names) < len(ecmwf_names):
            print(f"slim parts {len(part_names)}/{len(ecmwf_names)} — re-dispatch to finish", flush=True)
            return 0
        # CHECKPOINTED merge: fold parts into a running deduped frame, and every few parts
        # save the running frame + the list of folded parts back to HF. On reclaim, resume
        # from the checkpoint instead of restarting — so the merge survives the throttle
        # across dispatches. Each part is ~400 MB (<1 min), checkpoint is the deduped frame.
        ckpt_key = slim_key.replace(".parquet", "_ckpt.parquet")
        done_key = slim_key.replace(".parquet", "_ckpt_done.txt")
        ec, folded = None, set()
        try:
            cp = hf_hub_download(hub["datasets"]["training"], ckpt_key, repo_type="dataset")
            ec = pd.read_parquet(cp)
            ec["valid_time"] = pd.to_datetime(ec["valid_time"])
            dp = hf_hub_download(hub["datasets"]["training"], done_key, repo_type="dataset")
            folded = set(open(dp).read().split())
            print(f"resumed merge checkpoint: {len(folded)} parts folded, {len(ec)} rows", flush=True)
        except Exception:
            ec, folded = None, set()
        todo_parts = [pn for pn in part_names if pn not in folded]
        for i, pn in enumerate(todo_parts, 1):
            pp = hf_hub_download(hub["datasets"]["training"], pn, repo_type="dataset")
            part = pd.read_parquet(pp)
            os.remove(pp) if os.path.exists(pp) else None
            part["valid_time"] = pd.to_datetime(part["valid_time"])
            ec = part if ec is None else pd.concat([ec, part], ignore_index=True)
            del part
            ec = ec.drop_duplicates(["station_id", "valid_time"], keep="last")
            folded.add(pn)
            if i % 3 == 0 or i == len(todo_parts):  # checkpoint every 3 parts
                cf = dd / "ecmwf_ckpt.parquet"; ec.to_parquet(cf, index=False, compression="zstd")
                upload_file(cf, ckpt_key, hub["datasets"]["training"], repo_type="dataset")
                (dd / "ckpt_done.txt").write_text("\n".join(sorted(folded)))
                upload_file(dd / "ckpt_done.txt", done_key, hub["datasets"]["training"], repo_type="dataset")
                print(f"checkpoint: {len(folded)}/{len(part_names)} folded ({len(ec)} rows)", flush=True)
        outp = dd / "ecmwf_slim.parquet"
        ec.to_parquet(outp, index=False, compression="zstd")
        upload_file(outp, slim_key, hub["datasets"]["training"], repo_type="dataset")
        print(f"ECMWF slim cache built: {slim_key} ({len(ec)} rows from {len(part_names)} parts)", flush=True)
        return 0

    months = []
    if args.months:
        months = [m.strip() for m in args.months.split(",") if m.strip()]
    elif args.month:
        months = [args.month]
    else:
        print("ERROR: pass --month or --months")
        return 1

    # Global catalogue (sorted for a stable hash matching the obs parts).
    try:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_terrain_global.parquet", repo_type="dataset"))
    except Exception:
        stations = pd.read_parquet(hf_hub_download(hub["datasets"]["stations"], "stations_global.parquet", repo_type="dataset"))
    stations = stations.sort_values("station_id").reset_index(drop=True)

    prefix = table_parts_prefix(stations)
    files = set(api.list_repo_files(hub["datasets"]["training"], repo_type="dataset"))

    # Which months still need building (skip-if-exists), and have a GFS shard?
    todo = []
    for m in months:
        if f"{prefix}/part-{m}.parquet" in files:
            print(f"skip {m}: table part exists", flush=True)
            continue
        if f"gfs_global/gfs_{m}.parquet" not in files:
            print(f"skip {m}: no GFS shard", flush=True)
            continue
        todo.append(m)
    if not todo:
        print("nothing to build (all present or no shard)", flush=True)
        return 0

    # Load obs + ECMWF ONCE, reuse across all months in this job.
    from build_training_table_global import assemble_obs_from_parts
    obs = assemble_obs_from_parts(hub, stations)
    if obs is None:
        print("ERROR: obs parts not found — run collect_obs_global.yml first")
        return 1
    obs_slim = prepare_obs(obs)
    del obs
    ecmwf_names = sorted(f for f in files if f.startswith("ecmwf_global/") and f.endswith(".parquet"))
    ecmwf_slim = load_ecmwf(hub["datasets"]["training"], ecmwf_names, hub=hub)

    from mtnwx.data.hub_io import upload_file
    for m in todo:
        target = f"{prefix}/part-{m}.parquet"
        p = hf_hub_download(hub["datasets"]["training"], f"gfs_global/gfs_{m}.parquet", repo_type="dataset")
        gfs = pd.read_parquet(p)
        joined = build_global_table(gfs, obs_slim, stations, ecmwf_slim)
        os.remove(p) if os.path.exists(p) else None
        if joined.empty:
            print(f"WARN: no rows survived join for {m}", flush=True)
            del gfs, joined
            continue
        for c in joined.select_dtypes("float64").columns:
            joined[c] = joined[c].astype("float32")
        out = dd / f"table_part_{m}.parquet"
        joined.to_parquet(out, index=False, compression="zstd")
        n = len(joined)
        del gfs, joined
        upload_file(out, target, hub["datasets"]["training"], repo_type="dataset")
        os.remove(out) if os.path.exists(out) else None
        print(f"uploaded {target} ({n} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
