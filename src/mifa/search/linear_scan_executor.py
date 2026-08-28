"""Exact linear serving path for the Exp-3 structure comparison.

The executor intentionally has no projected-tree traversal.  It may use an
ANN seed to establish an early top-k boundary, but the distance row and final
ordering remain exact and stable by ``(distance, tuple_id)``.
"""
from __future__ import annotations

import numpy as np


class LinearScanExecutor:
    """Distance-order linear executor with optional ANN boundary seeding."""

    def __init__(self, embeddings, inference, *, ann_backend=None,
                 seed_size=0, distance_provider=None):
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.inference = inference
        self.ann_backend = ann_backend
        self.seed_size = int(seed_size)
        self.distance_provider = distance_provider
        self.fused_cpu = False
        self.leaf_frontier = False
        self.allow_distance_order_specialization = False
        self.sqnorm = np.einsum(
            "ij,ij->i", self.embeddings, self.embeddings).astype(np.float32)
        if self.seed_size < 0:
            raise ValueError("seed_size cannot be negative")

    @staticmethod
    def _counter_delta(after, before, key):
        return int(after.get(key, 0)) - int(before.get(key, 0))

    def _prepare_distances(self, query, precomputed_distances):
        if precomputed_distances is not None:
            distances = np.ascontiguousarray(
                precomputed_distances, dtype=np.float32)
            if distances.shape != (len(self.embeddings),):
                raise ValueError(
                    "precomputed_distances must contain one value per row")
            return distances
        if self.distance_provider is not None:
            return np.ascontiguousarray(
                self.distance_provider.prepare(query), dtype=np.float32)
        differences = self.embeddings - query[None, :]
        return np.einsum(
            "ij,ij->i", differences, differences).astype(np.float32)

    def search(self, query_embedding, k, predicate, epsilon=0.0,
               precomputed_distances=None, **_ignored):
        if float(epsilon) != 0.0:
            raise ValueError("linear scan executor requires epsilon=0")
        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query.shape != (self.embeddings.shape[1],):
            raise ValueError("query dimension does not match embeddings")
        k = min(int(k), len(self.embeddings))
        if k <= 0:
            raise ValueError("k must be positive")
        if hasattr(self.inference, "begin_query"):
            self.inference.begin_query(predicate)
        before = dict(getattr(self.inference, "counters", {}))
        distances = self._prepare_distances(query, precomputed_distances)

        seen = np.zeros(len(self.embeddings), dtype=np.bool_)
        eligible = {}
        visited = hard_decisions = resolution_requests = 0

        def consider(tuple_id):
            nonlocal visited, hard_decisions, resolution_requests
            tuple_id = int(tuple_id)
            if seen[tuple_id]:
                return
            seen[tuple_id] = True
            visited += 1
            hard_decisions += 1
            state, _ = self.inference.hard_decision(tuple_id, predicate)
            accepted = state == "pass"
            if state == "unresolved":
                resolution_requests += 1
                accepted = self.inference.evaluate(
                    tuple_id, predicate, assume_unresolved=True).eligible
            if accepted:
                eligible[tuple_id] = float(distances[tuple_id])

        seed_ids = np.empty(0, dtype=np.int64)
        if self.ann_backend is not None and self.seed_size:
            seed_ids, _ = self.ann_backend.search(
                query, min(self.seed_size, len(self.embeddings)))
            for tuple_id in seed_ids:
                consider(tuple_id)

        tuple_ids = np.arange(len(self.embeddings), dtype=np.int64)
        order = np.lexsort((tuple_ids, distances))
        for tuple_id in order:
            if len(eligible) >= k:
                top = sorted(
                    ((distance, row) for row, distance in eligible.items()))[:k]
                worst_distance, worst_id = top[-1]
                current = (float(distances[int(tuple_id)]), int(tuple_id))
                if current > (worst_distance, worst_id):
                    break
            consider(tuple_id)

        ranked = sorted(
            ((distance, row) for row, distance in eligible.items()))[:k]
        ids = np.asarray([row for _, row in ranked], dtype=np.int64)
        returned_distances = np.asarray(
            [distance for distance, _ in ranked], dtype=np.float64)
        after = dict(getattr(self.inference, "counters", {}))
        cache_hits = self._counter_delta(
            after, before, "inference_cache_hits")
        cache_misses = self._counter_delta(
            after, before, "inference_cache_misses")
        return ids, returned_distances, {
            "ann_seed_candidates": int(len(seed_ids)),
            "ann_seed_precomputed_reuse": int(len(seed_ids) > 0),
            "node_queue_pops": 0,
            "tuple_queue_pops": int(visited),
            "expanded_nodes": 0,
            "expanded_leaves": 0,
            "node_bound_evaluations": 0,
            "pruned_nodes": 0,
            "visited_tuples": int(visited),
            "tuple_queue_insertions": int(visited),
            "exact_distance_evaluations": int(len(self.embeddings)),
            "physical_distance_evaluations": int(len(self.embeddings)),
            "hard_decisions": int(hard_decisions),
            "lazy_inferences": int(resolution_requests),
            "statistical_resolution_requests": int(resolution_requests),
            "statistical_resolution_computations": int(cache_misses),
            "statistical_resolution_cache_hits": int(cache_hits),
            "statistical_resolution_cache_misses": int(cache_misses),
            "statistical_cache_hits": int(cache_hits),
            "inferences_skipped_by_boundary": int(
                len(self.embeddings) - np.count_nonzero(seen)),
            "candidate_rows_not_eligibility_checked": int(
                len(self.embeddings) - np.count_nonzero(seen)),
            "registered_tuples": int(np.count_nonzero(seen)),
            "buffer_update_attempts": int(len(eligible)),
            "max_queue_size": 0,
        }
