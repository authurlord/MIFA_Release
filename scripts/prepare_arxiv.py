#!/usr/bin/env python3
"""Download and verify the public arxiv-for-fanns files used by MIFA.

This downloader is a fallback for environments where a proxy strips metadata
needed by ``hf_hub_download``.  It resumes partial files, validates both the
advertised byte size and the Hugging Face LFS SHA-256, and atomically publishes
only verified files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url


DEFAULT_REPO = "SPCL/arxiv-for-fanns-medium"
DEFAULT_FILES = (
    "database_attributes.jsonl",
    "r_query_attributes.jsonl",
    "query_vectors.fvecs",
    "database_vectors.fvecs",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(repo: str, metadata, output: Path) -> dict:
    destination = output / metadata.path
    expected_size = int(metadata.size)
    expected_sha = metadata.lfs.sha256 if metadata.lfs else None
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size == expected_size:
        digest = sha256_file(destination)
        if expected_sha is None or digest == expected_sha:
            print(f"skip file={metadata.path} bytes={expected_size} sha256={digest}", flush=True)
            return {"file": metadata.path, "size": expected_size, "sha256": digest}

    part = destination.with_name(destination.name + ".part")
    url = hf_hub_url(repo, metadata.path, repo_type="dataset")
    for attempt in range(1, 6):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if 0 < offset < expected_size else {}
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 600),
            ) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                if not append:
                    offset = 0
                print(
                    f"download file={metadata.path} attempt={attempt} "
                    f"offset={offset} expected={expected_size}",
                    flush=True,
                )
                with part.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if part.stat().st_size != expected_size:
                if part.stat().st_size > expected_size:
                    part.unlink()
                raise RuntimeError(
                    f"size mismatch: {part.stat().st_size if part.exists() else 0} "
                    f"!= {expected_size}"
                )
            digest = sha256_file(part)
            if expected_sha is not None and digest != expected_sha:
                part.unlink()
                raise RuntimeError(f"SHA-256 mismatch: {digest} != {expected_sha}")
            part.replace(destination)
            print(f"done file={metadata.path} bytes={expected_size} sha256={digest}", flush=True)
            return {"file": metadata.path, "size": expected_size, "sha256": digest}
        except (requests.RequestException, RuntimeError) as exc:
            print(
                f"retry file={metadata.path} attempt={attempt} error={type(exc).__name__}",
                flush=True,
            )
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/arxiv-for-fanns-medium"),
    )
    args = parser.parse_args()

    api = HfApi()
    paths = api.get_paths_info(
        args.repo,
        list(DEFAULT_FILES),
        repo_type="dataset",
        expand=True,
    )
    by_name = {path.path: path for path in paths}
    missing = sorted(set(DEFAULT_FILES) - set(by_name))
    if missing:
        raise RuntimeError(f"files missing from {args.repo}: {missing}")

    manifest = [download_one(args.repo, by_name[name], args.output) for name in DEFAULT_FILES]
    manifest_path = args.output / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps({"repo": args.repo, "files": manifest}, indent=2) + "\n"
    )
    print(f"verified files={len(manifest)} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
