"""Qdrant H&M Dataset Adapter (Categorical: department_name + product_type_name)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class QdrantHMAdapter:
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            root = Path(__file__).resolve().parents[3]
            data_dir = root / "data/processed/qdrant_hm"
        self.data_dir = Path(data_dir)
        self.meta = json.loads((self.data_dir / "meta.json").read_text(encoding="utf-8")) if (self.data_dir / "meta.json").exists() else {}
        self._table = None
        self._queries = None
        self._embeddings = None
        self._query_embeddings = None

    def dataset_id(self) -> str:
        return "qdrant_hm"

    def designated_attributes(self) -> list[str]:
        return ["department_name", "product_type_name"]

    def attribute_types(self) -> dict[str, str]:
        return {
            "department_name": "categorical",
            "product_type_name": "categorical",
            "product_group_name": "categorical",
            "colour_group_name": "categorical",
            "section_name": "categorical",
            "garment_group_name": "categorical",
        }

    def load_complete_table(self) -> pd.DataFrame:
        if self._table is None:
            self._table = pd.read_parquet(self.data_dir / "table.parquet")
        return self._table

    def load_embeddings(self, embedding_type: str = "bgem3") -> np.ndarray:
        if self._embeddings is None:
            npy_path = self.data_dir / f"emb_{embedding_type}.npy"
            if not npy_path.exists():
                npy_path = self.data_dir / "emb_dense.npy"
            self._embeddings = np.load(npy_path)
        return self._embeddings

    def load_query_embeddings(self, embedding_type: str = "bgem3") -> np.ndarray:
        if self._query_embeddings is None:
            npy_path = self.data_dir / f"query_emb_{embedding_type}.npy"
            if not npy_path.exists():
                npy_path = self.data_dir / "query_emb_dense.npy"
            self._query_embeddings = np.load(npy_path)
        return self._query_embeddings

    def load_queries(self) -> pd.DataFrame:
        if self._queries is None:
            self._queries = pd.read_parquet(self.data_dir / "queries.parquet")
        return self._queries

    def evaluate_ground_truth_filter(self, row: dict, query: dict) -> bool:
        if "department_name" in query and query["department_name"] is not None:
            if str(row.get("department_name", "")).lower() != str(query["department_name"]).lower():
                return False
        if "product_type_name" in query and query["product_type_name"] is not None:
            if str(row.get("product_type_name", "")).lower() != str(query["product_type_name"]).lower():
                return False
        return True

    def build_query_roles(self, past_ratio: float = 0.50, val_ratio: float = 0.20, test_ratio: float = 0.30, seed: int = 42) -> dict[str, list[int]]:
        n_queries = len(self.load_queries())
        indices = np.arange(n_queries)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)

        n_past = int(past_ratio * n_queries)
        n_val = int(val_ratio * n_queries)

        past_list = shuffled[:n_past].tolist()
        val_list = shuffled[n_past:n_past+n_val].tolist()
        test_list = shuffled[n_past+n_val:].tolist()

        return {
            "historical": past_list,
            "validation": val_list,
            "test": test_list,
            "past": past_list,
            "val": val_list,
        }

    get_query_role_split = build_query_roles
