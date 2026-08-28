"""Paper-aligned projected binary partition tree.

The tree topology is fitted from the declared topology fold.  Temporary
validation trees use masked validation embeddings, while the final deployment
tree uses the searchable snapshot embeddings.  Historical support tuples are
subsequently routed through the frozen split tests.  The same root-to-leaf
paths are used as nested inference contexts and as the search hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False)
def _all_mbr_lower_bounds_sq(mbr_lo, mbr_hi, active, projected_query):
    output = np.full(len(active), np.inf, dtype=np.float64)
    for node_id in range(len(active)):
        if not active[node_id]:
            continue
        total = 0.0
        for dim in range(len(projected_query)):
            value = projected_query[dim]
            if value < mbr_lo[node_id, dim]:
                difference = float(mbr_lo[node_id, dim] - value)
                total += difference * difference
            elif value > mbr_hi[node_id, dim]:
                difference = float(value - mbr_hi[node_id, dim])
                total += difference * difference
        output[node_id] = total
    return output


@njit(cache=True, parallel=True, fastmath=False)
def _quantized_projection_point_lower_bounds_sq(
        codes, scales, quantization_error_norms, residual_norms,
        projected_query, query_residual_norm):
    """Safe per-point bounds from a compact orthogonal projection.

    For each stored point ``x``, the decoded int8 projection ``x_hat`` has a
    measured error norm ``e``.  Reverse triangle inequality gives
    ``||Pq-Px|| >= max(0, ||Pq-x_hat||-e)``.  The norm difference in the
    orthogonal residual subspace is independently safe and can be added in
    squared-distance space.
    """
    output = np.empty(codes.shape[0], dtype=np.float64)
    for row in prange(codes.shape[0]):
        decoded_distance_sq = 0.0
        for dim in range(codes.shape[1]):
            decoded = float(codes[row, dim]) * float(scales[dim])
            difference = float(projected_query[dim]) - decoded
            decoded_distance_sq += difference * difference
        projected_lower = max(
            np.sqrt(decoded_distance_sq) -
            float(quantization_error_norms[row]), 0.0)
        residual_difference = (
            float(query_residual_norm) - float(residual_norms[row]))
        # A conservative numeric slack prevents float32 quantization metadata
        # from rounding the mathematical lower bound upward.
        output[row] = max(
            projected_lower * projected_lower +
            residual_difference * residual_difference - 1e-5, 0.0)
    return output


@njit(cache=True, fastmath=False)
def _tree_bounds_from_point_bounds(
        point_bounds, point_leaf_nodes, left, right, active):
    bounds = np.full(len(active), np.inf, dtype=np.float64)
    for row in range(len(point_bounds)):
        node_id = int(point_leaf_nodes[row])
        if point_bounds[row] < bounds[node_id]:
            bounds[node_id] = point_bounds[row]
    # Nodes are allocated before their descendants, so reverse node order is
    # a valid bottom-up reduction.
    for node_id in range(len(active) - 1, -1, -1):
        if not active[node_id] or left[node_id] < 0:
            continue
        value = np.inf
        left_id = int(left[node_id])
        right_id = int(right[node_id])
        if left_id >= 0 and bounds[left_id] < value:
            value = bounds[left_id]
        if right_id >= 0 and bounds[right_id] < value:
            value = bounds[right_id]
        bounds[node_id] = value
    return bounds


@dataclass
class ProjectedTreeNode:
    node_id: int
    parent: int | None
    depth: int
    left: int | None = None
    right: int | None = None
    split_dim: int | None = None
    split_value: float | None = None
    split_tie_id: int | None = None
    leaf_id: int | None = None
    leaf_lo: int = -1
    leaf_hi: int = -1
    tuple_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    mbr_lo: np.ndarray | None = None
    mbr_hi: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def active(self) -> bool:
        return self.mbr_lo is not None


class ProjectedPartitionTree:
    """Balanced PCA partition tree with safe projected MBR lower bounds."""

    def __init__(self, projection_dim=16, leaf_capacity=128, seed=0):
        self.projection_dim = int(projection_dim)
        self.leaf_capacity = int(leaf_capacity)
        self.seed = int(seed)
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.leaf_capacity <= 0:
            raise ValueError("leaf_capacity must be positive")
        self.nodes: list[ProjectedTreeNode] = []
        self.root_id: int | None = None
        self.leaf_node_ids: list[int] = []
        self.training_leaf_ids: np.ndarray | None = None
        self.search_tuple_ids: np.ndarray | None = None
        self.search_projected: np.ndarray | None = None
        self._pca = None
        self._projection_components = None
        self._projection_mean = None
        self._mbr_lo_matrix = None
        self._mbr_hi_matrix = None
        self._full_mbr_lo_matrix = None
        self._full_mbr_hi_matrix = None
        self._bound_projection_dim = None
        self._bound_projection_components = None
        self._bound_projection_mean = None
        self._bound_mbr_lo_matrix = None
        self._bound_mbr_hi_matrix = None
        self._quantized_bound_codes = None
        self._quantized_bound_scales = None
        self._quantized_bound_error_norms = None
        self._bound_residual_norms = None
        self._point_leaf_node_ids = None
        self._node_left = None
        self._node_right = None
        self._mbr_active = None
        self.topology_source = "unspecified_embeddings"

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_node_ids)

    @property
    def projection_matrix(self):
        if self._pca is None:
            raise RuntimeError("tree structure has not been fitted")
        return self._pca.components_.T

    @property
    def projection_mean(self):
        if self._pca is None:
            raise RuntimeError("tree structure has not been fitted")
        return self._pca.mean_

    def project(self, embeddings):
        if self._pca is None:
            raise RuntimeError("tree structure has not been fitted")
        values = np.ascontiguousarray(embeddings, dtype=np.float32)
        one = values.ndim == 1
        if one:
            values = values[None, :]
        projected = np.asarray(
            (values - self._projection_mean) @ self._projection_components,
            dtype=np.float32)
        return projected[0] if one else projected

    def fit_structure(self, training_embeddings, tuple_ids=None, *,
                      topology_source="unspecified_embeddings"):
        """Fit PCA and deterministic median split tests from one declared fold."""
        from sklearn.decomposition import PCA

        values = np.ascontiguousarray(training_embeddings, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("training_embeddings must be a non-empty 2-D matrix")
        ids = (np.arange(len(values), dtype=np.int64) if tuple_ids is None else
               np.ascontiguousarray(tuple_ids, dtype=np.int64))
        if ids.shape != (len(values),) or len(np.unique(ids)) != len(ids):
            raise ValueError("tuple_ids must be unique and aligned with training rows")
        self.topology_source = str(topology_source)

        components = min(self.projection_dim, values.shape[1], len(values))
        # Exact LAPACK SVD is appropriate for the 4K--7K construction folds,
        # but becomes the dominant (and unnecessary) cost for Yelp/ArXiv.
        # Randomized SVD still yields one fixed orthonormal PCA projection;
        # deterministic ``random_state`` keeps the frozen split replay stable.
        solver = "randomized" if len(values) >= 10_000 else "full"
        self._pca = PCA(
            n_components=components, svd_solver=solver,
            random_state=self.seed)
        # Use the same transform path here and during later routing.  Some BLAS
        # implementations round ``fit_transform`` and ``transform`` differently
        # at an exact split pivot, which would violate frozen split replay.
        self._pca.fit(values)
        self._projection_mean = np.ascontiguousarray(
            self._pca.mean_, dtype=np.float32)
        self._projection_components = np.ascontiguousarray(
            self._pca.components_.T, dtype=np.float32)
        # Topology keeps the validation-selected low-dimensional PCA.  A
        # separate, wider orthonormal projection is used only for geometric
        # lower bounds; it does not alter splits or inference contexts.
        self._bound_projection_dim = min(128, values.shape[1], len(values))
        if self._bound_projection_dim < values.shape[1]:
            self._bound_pca = PCA(
                n_components=self._bound_projection_dim,
                svd_solver="randomized", random_state=self.seed + 1701,
                iterated_power=2)
            self._bound_pca.fit(values)
            self._bound_projection_mean = np.ascontiguousarray(
                self._bound_pca.mean_, dtype=np.float32)
            self._bound_projection_components = np.ascontiguousarray(
                self._bound_pca.components_.T, dtype=np.float32)
        else:
            self._bound_pca = None
            self._bound_projection_mean = None
            self._bound_projection_components = None
        projected = self.project(values)
        self.nodes = []
        self.leaf_node_ids = []
        self.training_leaf_ids = np.full(len(values), -1, dtype=np.int64)

        def build(rows, parent, depth):
            node_id = len(self.nodes)
            node = ProjectedTreeNode(node_id=node_id, parent=parent, depth=depth)
            self.nodes.append(node)
            if len(rows) <= self.leaf_capacity:
                leaf_id = len(self.leaf_node_ids)
                node.leaf_id = leaf_id
                node.leaf_lo = leaf_id
                node.leaf_hi = leaf_id
                self.leaf_node_ids.append(node_id)
                self.training_leaf_ids[rows] = leaf_id
                return node_id

            variances = projected[rows].var(axis=0)
            split_dim = int(np.argmax(variances))
            order = np.lexsort((ids[rows], projected[rows, split_dim]))
            ordered = rows[order]
            middle = len(ordered) // 2
            left_rows = ordered[:middle]
            right_rows = ordered[middle:]
            pivot_row = int(left_rows[-1])
            node.split_dim = split_dim
            node.split_value = float(projected[pivot_row, split_dim])
            node.split_tie_id = int(ids[pivot_row])
            node.left = build(left_rows, node_id, depth + 1)
            node.right = build(right_rows, node_id, depth + 1)
            node.leaf_lo = self.nodes[node.left].leaf_lo
            node.leaf_hi = self.nodes[node.right].leaf_hi
            return node_id

        self.root_id = build(np.arange(len(values), dtype=np.int64), None, 0)
        if (self.training_leaf_ids < 0).any():
            raise AssertionError("some historical tuples were not assigned to a leaf")
        return self

    def project_bound(self, embeddings):
        values = np.ascontiguousarray(embeddings, dtype=np.float32)
        one = values.ndim == 1
        if one:
            values = values[None, :]
        if self._bound_projection_components is None:
            projected = values
        else:
            projected = np.asarray(
                (values - self._bound_projection_mean) @
                self._bound_projection_components, dtype=np.float32)
        return projected[0] if one else projected

    @staticmethod
    def _go_left(node, coordinate, tuple_id):
        if coordinate < node.split_value:
            return True
        if coordinate > node.split_value:
            return False
        return int(tuple_id) <= int(node.split_tie_id)

    def route_projected_one(self, projected, tuple_id):
        if self.root_id is None:
            raise RuntimeError("tree structure has not been fitted")
        node_id = self.root_id
        path = [node_id]
        while not self.nodes[node_id].is_leaf:
            node = self.nodes[node_id]
            coordinate = float(projected[node.split_dim])
            node_id = node.left if self._go_left(
                node, coordinate, tuple_id) else node.right
            path.append(node_id)
        return self.nodes[node_id].leaf_id, tuple(path)

    def route(self, embeddings, tuple_ids=None, *, return_paths=False):
        projected = self.project(embeddings)
        if projected.ndim == 1:
            projected = projected[None, :]
        ids = (np.arange(len(projected), dtype=np.int64) if tuple_ids is None else
               np.ascontiguousarray(tuple_ids, dtype=np.int64))
        if ids.shape != (len(projected),):
            raise ValueError("tuple_ids must align with embeddings")
        leaves = np.empty(len(projected), dtype=np.int64)
        paths = []
        for row, (point, tuple_id) in enumerate(zip(projected, ids)):
            leaf, path = self.route_projected_one(point, int(tuple_id))
            leaves[row] = leaf
            if return_paths:
                paths.append(path)
        return (leaves, paths) if return_paths else leaves

    def materialize(self, searchable_embeddings, tuple_ids=None):
        """Populate leaf IDs and node MBRs from the current searchable snapshot."""
        values = np.ascontiguousarray(searchable_embeddings, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("searchable_embeddings must be a 2-D matrix")
        ids = (np.arange(len(values), dtype=np.int64) if tuple_ids is None else
               np.ascontiguousarray(tuple_ids, dtype=np.int64))
        if ids.shape != (len(values),) or len(np.unique(ids)) != len(ids):
            raise ValueError("tuple_ids must be unique and aligned with searchable rows")
        projected = self.project(values)
        positions_by_leaf = [[] for _ in range(self.n_leaves)]
        for position, (point, tuple_id) in enumerate(zip(projected, ids)):
            leaf_id, _ = self.route_projected_one(point, int(tuple_id))
            positions_by_leaf[leaf_id].append(position)

        self.search_tuple_ids = ids
        self.search_projected = projected
        for node in self.nodes:
            node.tuple_ids = np.empty(0, dtype=np.int64)
            node.mbr_lo = None
            node.mbr_hi = None
        for leaf_id, node_id in enumerate(self.leaf_node_ids):
            node = self.nodes[node_id]
            positions = np.asarray(positions_by_leaf[leaf_id], dtype=np.int64)
            if len(positions):
                node.tuple_ids = np.sort(ids[positions])
                points = projected[positions]
                node.mbr_lo = points.min(axis=0)
                node.mbr_hi = points.max(axis=0)

        for node in reversed(self.nodes):
            if node.is_leaf:
                continue
            children = [self.nodes[node.left], self.nodes[node.right]]
            active = [child for child in children if child.active]
            if active:
                node.mbr_lo = np.minimum.reduce([child.mbr_lo for child in active])
                node.mbr_hi = np.maximum.reduce([child.mbr_hi for child in active])
        projection_width = int(self._pca.n_components_)
        self._mbr_lo_matrix = np.zeros(
            (len(self.nodes), projection_width), dtype=np.float32)
        self._mbr_hi_matrix = np.zeros_like(self._mbr_lo_matrix)
        self._mbr_active = np.zeros(len(self.nodes), dtype=np.bool_)
        for node in self.nodes:
            if node.active:
                self._mbr_lo_matrix[node.node_id] = node.mbr_lo
                self._mbr_hi_matrix[node.node_id] = node.mbr_hi
                self._mbr_active[node.node_id] = True
        # A second MBR in the original orthonormal embedding coordinates is
        # also a safe Euclidean lower bound.  Taking the maximum of this bound
        # and the PCA-projected bound tightens high-dimensional frontiers
        # without changing topology or search semantics.  Storage scales with
        # tree nodes (roughly two per leaf), not with query count.
        self._full_mbr_lo_matrix = np.zeros(
            (len(self.nodes), values.shape[1]), dtype=np.float32)
        self._full_mbr_hi_matrix = np.zeros_like(
            self._full_mbr_lo_matrix)
        bound_projected = self.project_bound(values)
        self._bound_mbr_lo_matrix = np.zeros(
            (len(self.nodes), bound_projected.shape[1]), dtype=np.float32)
        self._bound_mbr_hi_matrix = np.zeros_like(
            self._bound_mbr_lo_matrix)
        for leaf_id, node_id in enumerate(self.leaf_node_ids):
            positions = np.asarray(positions_by_leaf[leaf_id], dtype=np.int64)
            if len(positions):
                points = values[positions]
                self._full_mbr_lo_matrix[node_id] = points.min(axis=0)
                self._full_mbr_hi_matrix[node_id] = points.max(axis=0)
                bound_points = bound_projected[positions]
                self._bound_mbr_lo_matrix[node_id] = bound_points.min(axis=0)
                self._bound_mbr_hi_matrix[node_id] = bound_points.max(axis=0)
        # Compact point bounds make the tree frontier useful in high
        # dimensions without computing full exact distances.  They are used
        # only to tighten physical traversal; topology and inference contexts
        # remain unchanged.
        maximum = np.max(np.abs(bound_projected), axis=0)
        scales = np.where(maximum > 0.0, maximum / 127.0, 1.0).astype(
            np.float32)
        codes = np.rint(bound_projected / scales[None, :])
        codes = np.clip(codes, -127, 127).astype(np.int8)
        decoded = codes.astype(np.float32) * scales[None, :]
        quantization_error_norms = np.sqrt(np.maximum(
            np.einsum("ij,ij->i", bound_projected - decoded,
                      bound_projected - decoded), 0.0)).astype(np.float32)
        if self._bound_projection_components is None:
            centered_sqnorm = np.einsum("ij,ij->i", values, values)
            projected_sqnorm = centered_sqnorm
        else:
            mean = self._bound_projection_mean
            centered_sqnorm = (
                np.einsum("ij,ij->i", values, values) -
                2.0 * (values @ mean) + float(mean @ mean))
            projected_sqnorm = np.einsum(
                "ij,ij->i", bound_projected, bound_projected)
        residual_norms = np.sqrt(np.maximum(
            centered_sqnorm - projected_sqnorm, 0.0)).astype(np.float32)
        point_leaf_node_ids = np.full(len(values), -1, dtype=np.int64)
        for leaf_id, node_id in enumerate(self.leaf_node_ids):
            positions = np.asarray(positions_by_leaf[leaf_id], dtype=np.int64)
            point_leaf_node_ids[positions] = int(node_id)
        if np.any(point_leaf_node_ids < 0):
            raise AssertionError("point bounds do not cover searchable rows")
        self._quantized_bound_codes = np.ascontiguousarray(codes)
        self._quantized_bound_scales = np.ascontiguousarray(scales)
        self._quantized_bound_error_norms = np.ascontiguousarray(
            quantization_error_norms)
        self._bound_residual_norms = np.ascontiguousarray(residual_norms)
        self._point_leaf_node_ids = np.ascontiguousarray(
            point_leaf_node_ids)
        self._node_left = np.asarray([
            -1 if node.left is None else int(node.left)
            for node in self.nodes], dtype=np.int64)
        self._node_right = np.asarray([
            -1 if node.right is None else int(node.right)
            for node in self.nodes], dtype=np.int64)
        for node in reversed(self.nodes):
            if node.is_leaf or not node.active:
                continue
            child_ids = [
                child_id for child_id in (node.left, node.right)
                if self.nodes[child_id].active]
            self._full_mbr_lo_matrix[node.node_id] = np.minimum.reduce([
                self._full_mbr_lo_matrix[child_id]
                for child_id in child_ids])
            self._full_mbr_hi_matrix[node.node_id] = np.maximum.reduce([
                self._full_mbr_hi_matrix[child_id]
                for child_id in child_ids])
            self._bound_mbr_lo_matrix[node.node_id] = np.minimum.reduce([
                self._bound_mbr_lo_matrix[child_id]
                for child_id in child_ids])
            self._bound_mbr_hi_matrix[node.node_id] = np.maximum.reduce([
                self._bound_mbr_hi_matrix[child_id]
                for child_id in child_ids])
        # Compile the serving kernel while the tree is constructed, outside
        # online query timing.
        _all_mbr_lower_bounds_sq(
            self._mbr_lo_matrix, self._mbr_hi_matrix, self._mbr_active,
            np.zeros(projection_width, dtype=np.float32))
        _all_mbr_lower_bounds_sq(
            self._full_mbr_lo_matrix, self._full_mbr_hi_matrix,
            self._mbr_active,
            np.zeros(values.shape[1], dtype=np.float32))
        _all_mbr_lower_bounds_sq(
            self._bound_mbr_lo_matrix, self._bound_mbr_hi_matrix,
            self._mbr_active,
            np.zeros(bound_projected.shape[1], dtype=np.float32))
        _quantized_projection_point_lower_bounds_sq(
            self._quantized_bound_codes, self._quantized_bound_scales,
            self._quantized_bound_error_norms, self._bound_residual_norms,
            np.zeros(bound_projected.shape[1], dtype=np.float32), 0.0)
        _tree_bounds_from_point_bounds(
            np.zeros(len(values), dtype=np.float64),
            self._point_leaf_node_ids, self._node_left, self._node_right,
            self._mbr_active)
        return self

    def ancestor_path(self, embedding, tuple_id=0, *, fine_to_broad=True):
        projected = self.project(embedding)
        _, root_to_leaf = self.route_projected_one(projected, int(tuple_id))
        return tuple(reversed(root_to_leaf)) if fine_to_broad else root_to_leaf

    def node_lower_bound_sq_projected(self, projected_query, node_id):
        node = self.nodes[int(node_id)]
        if not node.active:
            return float("inf")
        query = np.asarray(projected_query, dtype=np.float32)
        below = np.maximum(node.mbr_lo - query, 0.0)
        above = np.maximum(query - node.mbr_hi, 0.0)
        excess = below + above
        return float(excess @ excess)

    def all_node_lower_bounds_sq_projected(self, projected_query):
        if self._mbr_lo_matrix is None:
            raise RuntimeError("searchable MBRs have not been materialized")
        query = np.ascontiguousarray(projected_query, dtype=np.float32)
        return _all_mbr_lower_bounds_sq(
            self._mbr_lo_matrix, self._mbr_hi_matrix,
            self._mbr_active, query)

    def all_node_lower_bounds_sq(self, query_embedding):
        """Return the maximum of two independently safe MBR bounds."""
        if self._full_mbr_lo_matrix is None:
            raise RuntimeError("searchable full-space MBRs are unavailable")
        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != \
                self._full_mbr_lo_matrix.shape[1]:
            raise ValueError("query dimensions do not match full-space MBRs")
        projected = self.all_node_lower_bounds_sq_projected(
            self.project(query))
        full = _all_mbr_lower_bounds_sq(
            self._full_mbr_lo_matrix, self._full_mbr_hi_matrix,
            self._mbr_active, query)
        bound_query = np.ascontiguousarray(
            self.project_bound(query), dtype=np.float32)
        wide = _all_mbr_lower_bounds_sq(
            self._bound_mbr_lo_matrix, self._bound_mbr_hi_matrix,
            self._mbr_active, bound_query)
        # Float32 projections can round an otherwise exact boundary upward by
        # a few ulps.  A tiny conservative slack keeps every combined bound
        # below the mathematical lower bound used by the theorem.
        wide = np.maximum(wide - 1e-6, 0.0)
        full = np.maximum(full - 1e-6, 0.0)
        return np.maximum(projected, np.maximum(wide, full))

    def all_node_quantized_point_lower_bounds_sq(self, query_embedding):
        """Return tighter safe subtree bounds for positive-epsilon search."""
        if self._quantized_bound_codes is None:
            raise RuntimeError("quantized point bounds are unavailable")
        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        bound_query = np.ascontiguousarray(
            self.project_bound(query), dtype=np.float32)
        if self._bound_projection_components is None:
            query_residual_norm = 0.0
        else:
            centered = query - self._bound_projection_mean
            residual_sq = max(
                float(centered @ centered) - float(bound_query @ bound_query),
                0.0)
            query_residual_norm = float(np.sqrt(residual_sq))
        point_bounds = _quantized_projection_point_lower_bounds_sq(
            self._quantized_bound_codes, self._quantized_bound_scales,
            self._quantized_bound_error_norms, self._bound_residual_norms,
            bound_query, query_residual_norm)
        return _tree_bounds_from_point_bounds(
            point_bounds, self._point_leaf_node_ids,
            self._node_left, self._node_right, self._mbr_active)

    def node_lower_bound_sq(self, query_embedding, node_id):
        return self.node_lower_bound_sq_projected(
            self.project(query_embedding), node_id)

    def active_roots(self):
        if self.root_id is None or not self.nodes[self.root_id].active:
            return ()
        return (self.root_id,)

    def manifest(self):
        return {
            "kind": "projected_binary_partition_tree",
            "projection_dim": int(self._pca.n_components_) if self._pca else None,
            "pca_solver": (str(self._pca.svd_solver)
                           if self._pca is not None else None),
            "leaf_capacity": self.leaf_capacity,
            "nodes": len(self.nodes),
            "leaves": self.n_leaves,
            "active_search_rows": (0 if self.search_tuple_ids is None else
                                   int(len(self.search_tuple_ids))),
            "topology_source": self.topology_source,
            "mbr_source": "searchable_snapshot",
            "distance_lower_bound": (
                "max(topology_projected_mbr,wide_orthonormal_mbr,"
                "full_space_mbr)"),
            "wide_bound_projection_dim": self._bound_projection_dim,
            "wide_bound_mbr_bytes": (
                0 if self._bound_mbr_lo_matrix is None else int(
                    self._bound_mbr_lo_matrix.nbytes +
                    self._bound_mbr_hi_matrix.nbytes)),
            "full_space_mbr_bytes": (
                0 if self._full_mbr_lo_matrix is None else int(
                    self._full_mbr_lo_matrix.nbytes +
                    self._full_mbr_hi_matrix.nbytes)),
            "quantized_point_bound_bytes": (
                0 if self._quantized_bound_codes is None else int(
                    self._quantized_bound_codes.nbytes +
                    self._quantized_bound_scales.nbytes +
                    self._quantized_bound_error_norms.nbytes +
                    self._bound_residual_norms.nbytes)),
        }
