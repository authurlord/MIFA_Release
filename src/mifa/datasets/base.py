"""Unified filtered-dataset adapter protocol and base classes for MIFA."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetProvenance:
    dataset_id: str
    source_revision: str
    source_url_or_repo: str
    license: str
    raw_sha256: str
    processed_sha256: str
    n_rows_raw: int
    n_rows_controlled_complete: int
    embedding_dim: int
    n_queries_available: int
    native_query: bool
    native_filter: bool
    constructed_filter: bool
    designated_attributes: list[str]
    selection_rule: str = "schema_and_historical_workload_only"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class FilteredDatasetAdapter(Protocol):
    """Unified protocol for filtered vector search datasets with missing scalar attributes."""

    def dataset_id(self) -> str:
        ...

    def designated_attributes(self) -> list[str]:
        ...

    def attribute_types(self) -> dict[str, str]:
        ...

    def load_complete_table(self) -> pd.DataFrame:
        ...

    def load_embeddings(self) -> np.ndarray:
        ...

    def load_queries(self) -> pd.DataFrame:
        ...

    def load_query_embeddings(self) -> np.ndarray:
        ...

    def build_query_roles(
        self,
        n_hist: int = 200,
        n_val: int = 500,
        n_cal: int = 300,
        n_test: int = 1000,
        seed: int = 42,
    ) -> dict[str, list[int]]:
        ...

    def evaluate_exact_truth(
        self,
        query_idx: int,
        k: int = 10,
        distance_metric: str = "euclidean",
    ) -> list[int]:
        ...

    def provenance_manifest(self) -> dict:
        ...
