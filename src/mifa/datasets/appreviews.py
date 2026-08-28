"""AppReviews Dataset Adapter (Numerical: review_length + star_rating)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class AppReviewsAdapter:
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            root = Path(__file__).resolve().parents[3]
            data_dir = root / "data/processed/appreviews"
        self.data_dir = Path(data_dir)
        self.meta = json.loads((self.data_dir / "meta.json").read_text(encoding="utf-8")) if (self.data_dir / "meta.json").exists() else {}
        self._table = None
        self._queries = None
        self._embeddings = None
        self._query_embeddings = None

    def dataset_id(self) -> str:
        return "appreviews"

    def designated_attributes(self) -> list[str]:
        return ["review_length", "star_rating"]

    def attribute_types(self) -> dict[str, str]:
        return {
            "review_length": "numerical",
            "star_rating": "numerical",
        }

    def load_complete_table(self) -> pd.DataFrame:
        if self._table is None:
            self._table = pd.read_parquet(self.data_dir / "table.parquet")
        return self._table

    def load_embeddings(self, embedding_type: str = "bgem3") -> np.ndarray:
        if self._embeddings is None:
            npy_path = self.data_dir / f"emb_{embedding_type}.npy"
            if not npy_path.exists():
                npy_path = self.data_dir / "emb_minilm.npy"
            self._embeddings = np.load(npy_path)
        return self._embeddings

    def load_query_embeddings(self, embedding_type: str = "bgem3") -> np.ndarray:
        if self._query_embeddings is None:
            npy_path = self.data_dir / f"query_emb_{embedding_type}.npy"
            if not npy_path.exists():
                npy_path = self.data_dir / "query_emb_minilm.npy"
            self._query_embeddings = np.load(npy_path)
        return self._query_embeddings

    def load_queries(self) -> pd.DataFrame:
        if self._queries is None:
            self._queries = pd.read_parquet(self.data_dir / "queries.parquet")
        return self._queries

    def evaluate_ground_truth_filter(self, row: dict, query: dict) -> bool:
        if "review_length_min" in query and query["review_length_min"] is not None:
            val = float(row.get("review_length", -1.0))
            if np.isnan(val) or val < float(query["review_length_min"]):
                return False
        if "review_length_max" in query and query["review_length_max"] is not None:
            val = float(row.get("review_length", -1.0))
            if np.isnan(val) or val > float(query["review_length_max"]):
                return False
        if "star_rating_min" in query and query["star_rating_min"] is not None:
            val = float(row.get("star_rating", -1.0))
            if np.isnan(val) or val < float(query["star_rating_min"]):
                return False
        if "star_rating_max" in query and query["star_rating_max"] is not None:
            val = float(row.get("star_rating", -1.0))
            if np.isnan(val) or val > float(query["star_rating_max"]):
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
