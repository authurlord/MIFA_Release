"""Run provenance: resolve config, stamp env/git/dataset hash, write metrics JSON."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def backend_banner():
    """Return (and the caller may print) the active KNN backend for transparency."""
    try:
        from mifa.knn import active_backend
        return active_backend()
    except Exception:
        return "cpu"


def write_report(out_dir, name, payload):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / name
    payload = {
        **payload,
        "_provenance": {
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "knn_backend": backend_banner(),
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return str(path)
