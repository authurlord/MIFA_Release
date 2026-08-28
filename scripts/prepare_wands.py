#!/usr/bin/env python3
"""Prepare the official WANDS release for the paper-v2 native-query track.

This stage is deliberately embedding-agnostic: it writes row-aligned product
text/scalars plus the 480 native queries and 233k relevance labels.  Run
``scripts/embed_dataset.py`` afterwards to materialize the frozen BGE-M3 axis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/wands/dataset")
    parser.add_argument("--out", default="data/processed/wands")
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    products = pd.read_csv(raw / "product.csv", sep="\t")
    queries = pd.read_csv(raw / "query.csv", sep="\t")
    labels = pd.read_csv(raw / "label.csv", sep="\t")

    hierarchy = products["category hierarchy"].fillna("").astype(str).str.split("/")
    products["category"] = hierarchy.str[0].replace("", pd.NA)
    products["subcategory"] = hierarchy.str[1].replace("", pd.NA)
    text_columns = ["product_name", "product_description", "product_features"]
    products["text"] = products[text_columns].fillna("").astype(str).agg(". ".join, axis=1)
    scalar_columns = [
        "product_class", "category", "subcategory",
        "rating_count", "average_rating", "review_count",
    ]
    table = products[["product_id", *scalar_columns]].copy()
    for column in ("rating_count", "average_rating", "review_count"):
        table[column] = pd.to_numeric(table[column], errors="coerce")

    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "table.parquet", index=False)
    products[["product_id", "text"]].to_parquet(out / "text.parquet", index=False)
    queries.to_parquet(out / "queries.parquet", index=False)
    labels.to_parquet(out / "labels.parquet", index=False)

    controlled = table[scalar_columns].notna().all(axis=1)
    meta = {
        "dataset": "WANDS",
        "source": "https://github.com/wayfair/WANDS",
        "source_commit": "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5",
        "raw_sha256": {
            path.name: _sha256(path) for path in sorted(raw.glob("*.csv"))
        },
        "products": int(len(products)),
        "controlled_complete_products": int(controlled.sum()),
        "native_queries": int(len(queries)),
        "relevance_labels": int(len(labels)),
        "query_relevant_attrs": scalar_columns,
        "dtypes": {
            "product_class": "cat", "category": "cat", "subcategory": "cat",
            "rating_count": "num", "average_rating": "num", "review_count": "num",
        },
        "tracks": {
            "controlled_hide": "complete scalar rows only",
            "natural_missing": "all rows; no oracle F1 for originally absent cells",
            "native_query": "queries.parquet + labels.parquet",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(out), **meta}, indent=2))


if __name__ == "__main__":
    main()
