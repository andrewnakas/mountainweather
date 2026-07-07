#!/usr/bin/env python3
"""Create the Hugging Face repos this project publishes to.

Run once after setting HF_TOKEN and the real namespace in configs/hub.yaml:

    HF_TOKEN=hf_xxx python scripts/hf_setup.py [--namespace NAME]

Creates the training/stations/verify datasets, the models repo, and the demo Space.
Idempotent: existing repos are left as-is (exist_ok=True). The namespace defaults to
configs/hub.yaml's ``hf_namespace`` but can be overridden for a different account.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtnwx.config import load_configs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default=None, help="Override HF namespace (user/org)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN in the environment")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install 'huggingface-hub' (part of the [train] extra)")
        return 1

    hub = load_configs()["hub"]
    ns = args.namespace or hub["hf_namespace"]
    api = HfApi(token=token)

    def _rename(repo_id: str) -> str:
        # Swap the placeholder namespace for the real one.
        return f"{ns}/{repo_id.split('/', 1)[1]}" if "/" in repo_id else f"{ns}/{repo_id}"

    plan = [
        ("dataset", _rename(hub["datasets"]["training"])),
        ("dataset", _rename(hub["datasets"]["stations"])),
        ("dataset", _rename(hub["datasets"]["verify"])),
        ("model", _rename(hub["models"]["repo"])),
        ("space", _rename(hub["space"]["demo"])),
    ]
    for repo_type, repo_id in plan:
        kwargs = {"repo_id": repo_id, "repo_type": repo_type, "exist_ok": True}
        if repo_type == "space":
            kwargs["space_sdk"] = "gradio"
        url = api.create_repo(**kwargs)
        print(f"  ✓ {repo_type:8s} {repo_id}  ->  {url}")

    print(f"\nDone. Namespace: {ns}")
    print("Update configs/hub.yaml `hf_namespace` if you overrode it here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
