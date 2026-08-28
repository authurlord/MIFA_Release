"""Coarse PCA cell index for MIFA's naive coarse inference (Phase 2a).

This is the *placeholder* for the paper's adaptive sparse grid G_e (Phase 2d).
For Phase 2a we only need a fixed embedding-space partition that yields, per cell,
the empirical distribution of each scalar attribute -> used to estimate p_phi(t)
for tuples whose predicate attribute is missing.

Partition = PCA(r) projection + MiniBatchKMeans(C) cells. KMeans cells capture
embedding locality far better than coarse axis-aligned bins (which explode in
count), while remaining a fixed, non-adaptive partition (no split-utility yet).
"""
from __future__ import annotations

import numpy as np


class CoarseCellIndex:
    def __init__(self, pca_dim=16, n_cells=512, seed=0):
        self.pca_dim = pca_dim
        self.n_cells = n_cells
        self.seed = seed
        self.cell_of = None      # (n,) cell id per tuple
        self.proj = None         # PCA projection matrix (d, r)
        self.mean_ = None
        self.pca_ = None
        self.kmeans_ = None

    def build(self, embeddings):
        from sklearn.decomposition import PCA
        from sklearn.cluster import MiniBatchKMeans

        X = np.ascontiguousarray(embeddings, dtype=np.float32)
        r = min(self.pca_dim, X.shape[1])
        pca = PCA(n_components=r, random_state=self.seed)
        Z = pca.fit_transform(X)
        self.pca_ = pca
        self.proj = pca.components_.T
        self.mean_ = pca.mean_
        C = min(self.n_cells, X.shape[0])
        km = MiniBatchKMeans(n_clusters=C, random_state=self.seed,
                             n_init=3, batch_size=4096)
        self.cell_of = km.fit_predict(Z).astype(np.int64)
        self.kmeans_ = km
        self._C = C
        return self

    def route(self, embeddings):
        """Route held-out embeddings through the training-fitted PCA/KMeans cells."""
        if self.pca_ is None or self.kmeans_ is None:
            raise RuntimeError("CoarseCellIndex must be built before route()")
        X = np.ascontiguousarray(embeddings, dtype=np.float32)
        return self.kmeans_.predict(self.pca_.transform(X)).astype(np.int64)

    def cell_atom_prob(self, atom_sat_observed, observed):
        """Per-cell P(atom satisfied | observed values in cell).

        atom_sat_observed: bool (n,) atom satisfied, valid where observed.
        observed: bool (n,) value is observed (not masked).
        Returns p_cell (C,) and a per-tuple lookup p_tuple (n,) = p_cell[cell].
        Cells with no observed values fall back to the global observed rate.
        """
        cell = self.cell_of
        w_obs = observed.astype(np.float64)
        w_sat = (atom_sat_observed & observed).astype(np.float64)
        denom = np.bincount(cell, weights=w_obs, minlength=self._C)
        numer = np.bincount(cell, weights=w_sat, minlength=self._C)
        global_rate = (w_sat.sum() / w_obs.sum()) if w_obs.sum() > 0 else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            p_cell = np.where(denom > 0, numer / np.maximum(denom, 1e-9), global_rate)
        return p_cell, p_cell[cell]
