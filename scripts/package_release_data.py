#!/usr/bin/env python3
"""Build data ZIPs outside Git and print decimal-GB/binary-GiB sizes.

Large NumPy/FVEC payloads are stored without a second compression pass; Parquet
and JSON metadata are already compact.  No checksum gate is part of this
packager.  Upload verification is by remote file name and byte size.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parents[1]

PAPER_READY_FILES = {
    "wands": [
        "data/processed/wands/table.parquet",
        "data/processed/wands/text.parquet",
        "data/processed/wands/queries.parquet",
        "data/processed/wands/meta.json",
        "data/processed/final_main_assets/wands/emb_bgem3_1024.npy",
        "data/processed/final_main_assets/wands/query_emb_bgem3_1024.npy",
    ],
    "books_v2": [
        "data/processed/books_v2/table.parquet",
        "data/processed/books_v2/text.parquet",
        "data/processed/books_v2/meta.json",
        "data/processed/books_v2/emb_bgem3.npy",
    ],
    "movielens": [
        "data/processed/movielens/table.parquet",
        "data/processed/movielens/text.parquet",
        "data/processed/movielens/meta.json",
        "data/processed/movielens/emb_bgem3.npy",
    ],
    "yelp": [
        "data/processed/yelp/table.parquet",
        "data/processed/yelp/text.parquet",
        "data/processed/yelp/meta.json",
        "data/processed/yelp/emb_bgem3.npy",
    ],
    "qdrant_hm": [
        "data/processed/qdrant_hm/table.parquet",
        "data/processed/qdrant_hm/texts.json",
        "data/processed/qdrant_hm/meta.json",
        "data/processed/qdrant_hm/queries.parquet",
        "data/processed/final_main_assets/qdrant_hm/emb_bgem3_1024.npy",
        "data/processed/final_main_assets/qdrant_hm/query_emb_bgem3_1024.npy",
    ],
    "amazon_multi": [
        "data/processed/amazon_multi/table.parquet",
        "data/processed/amazon_multi/text.parquet",
        "data/processed/amazon_multi/meta.json",
        "data/processed/amazon_multi/emb_bgem3.npy",
    ],
    "appreviews": [
        "data/processed/appreviews/table.parquet",
        "data/processed/appreviews/text.parquet",
        "data/processed/appreviews/meta.json",
        "data/processed/appreviews/queries.parquet",
        "data/processed/appreviews/emb_jina_v5_mrl_d1024.npy",
        "data/processed/appreviews/query_emb_jina_v5_mrl_d1024.npy",
    ],
    "amazon_c4": [
        "data/processed/amazon_c4/table.parquet",
        "data/processed/amazon_c4/text.parquet",
        "data/processed/amazon_c4/meta.json",
        "data/processed/amazon_c4/queries.parquet",
        "data/processed/amazon_c4/emb_bgem3.npy",
        "data/processed/amazon_c4/query_emb_bgem3.npy",
    ],
}

LOCAL_SOURCE_PATHS = [
    "data/raw/wands/dataset",
    "data/raw/appreviews_raw.parquet",
    "data/raw/hm_ecommerce_products_enriched.parquet",
    "data/raw/arxiv-for-fanns-medium",
]


def iter_path(path: Path):
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        if any(part in {".cache", ".git"} for part in item.parts):
            continue
        if item.name.endswith(".partial"):
            continue
        yield item


def archive_files(path: Path, source_root: Path, files: list[Path], label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    entries = []
    with ZipFile(path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for file_path in files:
            relative = file_path.relative_to(source_root)
            archive.write(file_path, relative.as_posix())
            size = file_path.stat().st_size
            total += size
            entries.append({"path": relative.as_posix(), "bytes": size})
        archive.writestr(
            "RELEASE_DATA_CONTENTS.json",
            json.dumps({
                "schema_version": 1,
                "label": label,
                "files": entries,
                "uncompressed_bytes": total,
                "arxiv_large_included": False,
            }, indent=2) + "\n")
    return {
        "package": path.name,
        "kind": label,
        "files": len(files),
        "uncompressed_bytes": total,
        "archive_bytes": path.stat().st_size,
    }


def complete_files(source_root: Path) -> list[Path]:
    requested = [item for files in PAPER_READY_FILES.values() for item in files]
    for relative in LOCAL_SOURCE_PATHS:
        requested.append(relative)
    output = []
    for relative in requested:
        path = source_root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        output.extend(iter_path(path))
    unique = {path.resolve(): path for path in output}
    return sorted(unique.values())


def injected_files(release_root: Path, mechanism: str) -> list[Path]:
    path = release_root / "data" / "injected" / mechanism
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = list(iter_path(path))
    if not files:
        raise RuntimeError(f"no injected data under {path}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--injected", default="",
                        help="comma-separated mcar,hce,qae")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    rows = []
    if args.complete:
        rows.append(archive_files(
            output_dir / "MIFA_NON_ARXIV_LARGE_COMPLETE_AND_SOURCE_DATA.zip",
            source_root, complete_files(source_root),
            "non_arxiv_large_complete_and_available_source_data"))
    for mechanism in [x.strip() for x in args.injected.split(",") if x.strip()]:
        if mechanism not in {"mcar", "hce", "qae"}:
            parser.error(f"unknown mechanism: {mechanism}")
        rows.append(archive_files(
            output_dir / f"MIFA_INJECTED_{mechanism.upper()}_RHO020.zip",
            ROOT, injected_files(ROOT, mechanism),
            f"paper_aligned_{mechanism}_rho020_injected_contexts"))
    for row in rows:
        row["archive_GB"] = row["archive_bytes"] / 1_000_000_000
        row["archive_GiB"] = row["archive_bytes"] / (1024 ** 3)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "DATA_PACKAGE_SIZES.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "DATA_PACKAGE_SIZES.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["package"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
