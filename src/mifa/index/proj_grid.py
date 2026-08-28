"""Projected sparse grid G_e with SAFE lower bounds (Phase 2b).

Unlike the KMeans CoarseCellIndex (which gives no admissible distance bound), this
builds an axis-aligned grid over a PCA projection whose components are ORTHONORMAL.
Projection onto an orthonormal subspace is non-expansive:

    ||P(x) - P(y)||_2  <=  ||x - y||_2

so the distance from a (projected) query to a cell's minimum bounding rectangle is
a valid LOWER BOUND on the true L2 distance to any tuple in that cell. Cells whose
LB exceeds the current top-k boundary tau_B can be pruned safely.

Using r_grid leading dims (a subset) only loosens the bound (still admissible),
trading pruning power for fewer/denser cells.
"""
from __future__ import annotations

import numpy as np


class ProjectedGrid:
    def __init__(self, pca_dim=32, grid_dims=8, bins=4, seed=0):
        self.pca_dim = pca_dim        # PCA dims fit (for projecting queries)
        self.grid_dims = grid_dims    # leading dims actually binned into the grid
        self.bins = bins
        self.seed = seed

    def build(self, embeddings):
        from sklearn.decomposition import PCA

        X = np.ascontiguousarray(embeddings, dtype=np.float32)
        r = min(self.pca_dim, X.shape[1])
        pca = PCA(n_components=r, random_state=self.seed)
        Z = pca.fit_transform(X).astype(np.float32)     # (n, r) projected coords
        # PCA components_ rows are orthonormal -> projection is non-expansive.
        self.mean_ = pca.mean_.astype(np.float32)
        self.components_ = pca.components_.astype(np.float32)   # (r, d)
        self.Z = Z
        g = min(self.grid_dims, r)
        self.g = g
        Zg = Z[:, :g]

        # equal-width bins per grid dim over the observed projected range
        self.lo_ = Zg.min(axis=0)
        self.hi_ = Zg.max(axis=0)
        span = np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)
        self.width_ = span / self.bins
        bin_idx = np.floor((Zg - self.lo_) / self.width_).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, self.bins - 1)            # (n, g)

        # sparse occupied cells: map bin-tuple -> contiguous cell id
        keys = np.zeros(len(Zg), dtype=np.int64)
        mult = 1
        for d in range(g):
            keys += bin_idx[:, d] * mult
            mult *= self.bins
        order = np.argsort(keys, kind="stable")
        keys_sorted = keys[order]
        uniq, starts = np.unique(keys_sorted, return_index=True)
        self.n_cells = len(uniq)
        self.cell_members = [order[starts[i]:starts[i + 1]]
                             for i in range(len(uniq) - 1)]
        self.cell_members.append(order[starts[-1]:])
        # per-cell MBR in the g projected grid dims (tighter than bin edges)
        self.cell_lo = np.empty((self.n_cells, g), dtype=np.float32)
        self.cell_hi = np.empty((self.n_cells, g), dtype=np.float32)
        for i, mem in enumerate(self.cell_members):
            zg = Zg[mem]
            self.cell_lo[i] = zg.min(axis=0)
            self.cell_hi[i] = zg.max(axis=0)
        self.cell_size = np.array([len(m) for m in self.cell_members], dtype=np.int64)
        return self

    def project(self, q):
        """Project a query embedding into the g grid dims."""
        qz = (np.asarray(q, dtype=np.float32) - self.mean_) @ self.components_[: self.g].T
        return qz

    def cell_lower_bounds_sq(self, q_grid):
        """Squared LB distance from projected query to each cell's MBR (vectorized).

        These are lower bounds on the projected distance, hence on the full L2
        distance, so pruning cells with LB > tau_B is safe."""
        below = self.cell_lo - q_grid[None, :]      # >0 if q below box on that dim
        above = q_grid[None, :] - self.cell_hi      # >0 if q above box
        excess = np.maximum(np.maximum(below, above), 0.0)
        return np.einsum("ij,ij->i", excess, excess)
