"""Loader for prebuilt unique-row scale contexts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from mifa.datasets import DatasetSpec


def load_unique_scale_context(path: str | Path, dtypes: dict) -> tuple[dict, dict]:
    source = Path(path).resolve()
    manifest_path = source if source.is_file() else source / "manifest.json"
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    embeddings = np.load(root / manifest.get(
        "embeddings_file", "embeddings_d128.npy"), mmap_mode="r")
    complete = pd.read_pickle(root / manifest.get(
        "complete_table_file", "complete_table.pkl"))
    observed = pd.read_pickle(root / manifest.get(
        "observed_table_file", "observed_table.pkl"))
    with (root / manifest.get("queries_file", "queries.pkl")).open("rb") as stream:
        queries = pickle.load(stream)
    truth_ids = np.load(root / manifest.get("truth_ids_file", "truth_ids.npy"))
    affected = np.load(root / manifest.get(
        "masked_answer_affected_file", "masked_answer_affected.npy"))
    mask = np.load(root / manifest.get("mask_file", "mask.npy"))
    source_row_ids = np.load(root / manifest.get(
        "source_row_ids_file", "source_row_ids.npy"))
    expected_n = int(manifest["n"])
    expected_dimension = int(manifest["dimension"])
    if embeddings.shape != (expected_n, expected_dimension):
        raise RuntimeError("unique scale embedding shape does not match manifest")
    if len(complete) != expected_n or len(observed) != expected_n:
        raise RuntimeError("unique scale table length does not match manifest")
    if list(complete.columns) != list(observed.columns):
        raise RuntimeError("unique scale complete/observed schemas differ")
    if len(queries) != int(manifest["query_count"]):
        raise RuntimeError("unique scale query count does not match manifest")
    if len(truth_ids) != len(queries) or len(affected) != len(queries):
        raise RuntimeError("unique scale truth arrays are not query aligned")
    if len(source_row_ids) != expected_n or len(np.unique(source_row_ids)) != expected_n:
        raise RuntimeError("unique scale source row IDs are not unique and aligned")
    if mask.shape[0] != expected_n:
        raise RuntimeError("unique scale missingness mask is not row aligned")
    required = set(dtypes)
    if not required.issubset(complete.columns):
        raise RuntimeError("unique scale context lacks required scalar columns")
    base = DatasetSpec(
        name=f"arxiv_large_unique_n{expected_n}",
        A_e=embeddings,
        table=complete,
        dtypes=dict(dtypes),
        metric="cosine",
        query_relevant_attrs=list(dtypes),
        meta={"unique_scale_manifest": manifest},
    )
    context = {
        "role": "test",
        "base": base,
        "observed": observed,
        "queries": queries,
        "embeddings": embeddings,
        "truth_ids": np.ascontiguousarray(truth_ids, dtype=np.int64),
        "masked_answer_affected": np.ascontiguousarray(
            affected, dtype=np.bool_),
        "source_row_ids": np.ascontiguousarray(
            source_row_ids, dtype=np.int64),
        "mask": np.ascontiguousarray(mask, dtype=np.bool_),
        "scale_unique_manifest": manifest,
        "scale_unique_context_root": str(root),
    }
    receipt = {
        "kind": manifest["kind"],
        "prebuilt_unique_context": True,
        "context_root": str(root),
        "manifest": str(manifest_path),
        "n": expected_n,
        "dimension": expected_dimension,
        "handler": manifest.get("handler", "mifa_native_online_eligibility"),
        "rho": manifest.get("rho"),
        "missingness_kind": manifest.get("missingness_kind"),
        "repetition": False,
        "jitter": False,
    }
    return context, receipt
