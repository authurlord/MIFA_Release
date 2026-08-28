"""ANN backends: ExactBackend (brute force) + HNSW (hnswlib, faiss fallback).

All backends operate in L2 space. For cosine datasets, callers pass
spec.normalized_embeddings() so that L2 ranking == cosine ranking.
"""
from __future__ import annotations

import numpy as np


class ExactBackend:
    """Brute-force L2 KNN. Ground-truth backend."""

    def __init__(self):
        self.X = None

    def build(self, embeddings):
        self.X = np.ascontiguousarray(embeddings, dtype=np.float32)
        return self

    def search(self, q, k, subset=None):
        """Return (ids, dists) of k nearest among `subset` (or all)."""
        X = self.X if subset is None else self.X[subset]
        diff = X - q[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        kk = min(k, len(d2))
        part = np.argpartition(d2, kk - 1)[:kk]
        order = part[np.argsort(d2[part])]
        ids = order if subset is None else np.asarray(subset)[order]
        return np.asarray(ids), d2[order]


class HNSWBackend:
    def __init__(self, M=16, efConstruction=200, efSearch=64, seed=0):
        self.M = M
        self.efC = efConstruction
        self.efS = efSearch
        self.seed = seed
        self.index = None
        self.n = 0

    def build(self, embeddings):
        import hnswlib

        X = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.n, d = X.shape
        self.index = hnswlib.Index(space="l2", dim=d)
        self.index.init_index(
            max_elements=self.n, ef_construction=self.efC, M=self.M, random_seed=self.seed
        )
        self.index.add_items(X, np.arange(self.n))
        self.index.set_ef(self.efS)
        return self

    def set_ef(self, ef):
        self.efS = ef
        self.index.set_ef(ef)

    def search(self, q, k, subset=None):
        """Unfiltered ANN search. `subset` ignored (HNSW is global); callers that
        need filtering use it for seeding only."""
        q = np.ascontiguousarray(q[None, :], dtype=np.float32)
        kk = min(k, self.n)
        labels, dists = self.index.knn_query(q, k=kk)
        return labels[0], dists[0]


def make_backend(kind, ann_cfg, seed=0):
    if kind in ("exact", "brute"):
        return ExactBackend()
    if kind == "hnsw":
        return HNSWBackend(
            M=ann_cfg.get("M", 16),
            efConstruction=ann_cfg.get("efConstruction", 200),
            efSearch=ann_cfg.get("efSearch", 64),
            seed=seed,
        )
    raise ValueError(kind)
