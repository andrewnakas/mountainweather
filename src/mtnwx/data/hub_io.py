"""Push/pull parquet shards and model artifacts to the Hugging Face Hub.

All large artifacts live on HF (not git). These helpers wrap huggingface_hub so the
extraction/training workflows can upload a shard or download the training set with one
call. Requires ``HF_TOKEN`` in the environment and the [train] extra installed.
"""
from __future__ import annotations

import os
from pathlib import Path

from mtnwx.config import load_configs


def _api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    return HfApi(token=token), token


def upload_file(local_path: str | Path, path_in_repo: str, repo_id: str, repo_type: str = "dataset") -> str:
    """Upload one file to an HF repo, creating the repo if needed."""
    api, token = _api()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )
    return f"{repo_id}/{path_in_repo}"


def upload_shard(local_path: str | Path, path_in_repo: str, which: str = "training") -> str:
    """Upload a parquet shard to the configured dataset repo (training/verify/stations)."""
    repo_id = load_configs()["hub"]["datasets"][which]
    return upload_file(local_path, path_in_repo, repo_id, repo_type="dataset")


def download_dataset_snapshot(which: str = "training", local_dir: str | Path | None = None) -> str:
    """Download an entire dataset repo (all shards) for training."""
    from huggingface_hub import snapshot_download

    repo_id = load_configs()["hub"]["datasets"][which]
    _, token = _api()
    return snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir) if local_dir else None, token=token
    )
