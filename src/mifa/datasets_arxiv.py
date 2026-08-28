"""Loader for arxiv-for-fanns (Tier-0 real dataset).

Downloads (cached) the HF files, reads embeddings from .fvecs, flattens the
JSONL attributes into a scalar table, and returns a DatasetSpec. Also exposes
the native query embeddings and range/EM filters for workload construction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mifa.datasets import DatasetSpec
from mifa.io_fvecs import read_fvecs


def _hf(repo, fn):
    default_root = Path(__file__).resolve().parents[2] / "data" / "raw" / repo.rsplit("/", 1)[-1]
    local_root = Path(os.environ.get("MIFA_ARXIV_DIR", default_root))
    local_file = local_root / fn
    if local_file.is_file():
        return str(local_file)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset")


def _read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _flatten_attrs(records):
    """Flatten the 11 raw arXiv attributes into scalar columns."""
    out = {
        "main_category": [],
        "update_date": [],
        "number_of_authors": [],
        "number_of_versions": [],
        "number_of_main_categories": [],
        "number_of_sub_categories": [],
        "has_comments": [],
        "license": [],
    }
    for r in records:
        mc = r.get("main_categories") or [None]
        out["main_category"].append(mc[0] if mc else None)
        out["update_date"].append(r.get("update_date"))
        out["number_of_authors"].append(r.get("number_of_authors"))
        out["number_of_versions"].append(r.get("number_of_versions"))
        out["number_of_main_categories"].append(r.get("number_of_main_categories"))
        out["number_of_sub_categories"].append(r.get("number_of_sub_categories"))
        out["has_comments"].append(bool(r.get("has_comments")))
        out["license"].append(r.get("license"))
    table = pd.DataFrame(out)
    dtypes = {
        "main_category": "cat", "update_date": "num", "number_of_authors": "num",
        "number_of_versions": "num", "number_of_main_categories": "num",
        "number_of_sub_categories": "num", "has_comments": "cat", "license": "cat",
    }
    return table, dtypes


def load_arxiv(dcfg, seed=0) -> DatasetSpec:
    repo = dcfg.get("hf_repo", "SPCL/arxiv-for-fanns-medium")
    vecs = read_fvecs(_hf(repo, "database_vectors.fvecs"))
    records = _read_jsonl(_hf(repo, "database_attributes.jsonl"))
    assert len(records) == vecs.shape[0], (len(records), vecs.shape)
    table, dtypes = _flatten_attrs(records)

    qra = dcfg.get("query_relevant_attrs") or [
        "main_category", "update_date", "number_of_authors",
        "number_of_versions", "has_comments",
    ]
    spec = DatasetSpec(
        name="arxiv", A_e=vecs.astype(np.float32), table=table, dtypes=dtypes,
        metric=dcfg.get("metric", "cosine"), query_relevant_attrs=qra,
        meta={"hf_repo": repo, "native_d": int(vecs.shape[1])},
    )
    return spec


def load_arxiv_queries(repo="SPCL/arxiv-for-fanns-medium", metric="cosine"):
    """Native query embeddings (L2-normalized for cosine) + range filters."""
    q = read_fvecs(_hf(repo, "query_vectors.fvecs")).astype(np.float32)
    if metric == "cosine":
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q = q / norms
    r_filters = _read_jsonl(_hf(repo, "r_query_attributes.jsonl"))  # update_date range
    return q, r_filters
