"""Single paper-aligned MIFA serving object.

This module is intentionally separate from the historical surrogate harness.
Formal experiments can only claim paper-defined MIFA after using this object (or
an implementation with the same manifest) end to end.
"""
from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from mifa.index.ann_backend import ExactBackend, HNSWBackend
from mifa.index.projected_tree import ProjectedPartitionTree
from mifa.inference.paper_policy import IndexedJointInference
from mifa.inference.support_index import SharedSupportIndex
from mifa.search.unified_executor import UnifiedBestFirstExecutor


class PaperMIFA:
    METHOD_ID = "mifa-paper-4297018"

    def __init__(self, training_embeddings, training_table,
                 searchable_embeddings, searchable_table, dtypes, policy, theta,
                 *, projection_dim=16, leaf_capacity=128, ann_backend=None,
                 ann_seed_size=64, deterministic_constraints=(), seed=0,
                 fused_cpu=True, leaf_frontier=False,
                 distance_provider=None,
                 allow_distance_order_specialization=False,
                 topology_embeddings=None,
                 external_ann_build_seconds=0.0,
                 support_mode="joint",
                 template_selection_mode="indexed",
                 support_retrieval_mode="indexed",
                 context_mode="hierarchical",
                 approximate_tail_budget=1,
                 search_projection_dim=None,
                 search_leaf_capacity=None):
        construction_start = time.perf_counter()
        phase_seconds = {}
        start = time.perf_counter()
        self.training_embeddings = np.ascontiguousarray(
            training_embeddings, dtype=np.float32)
        self.searchable_embeddings = np.ascontiguousarray(
            searchable_embeddings, dtype=np.float32)
        if topology_embeddings is not None:
            supplied_topology = np.ascontiguousarray(
                topology_embeddings, dtype=np.float32)
            if supplied_topology.shape != self.training_embeddings.shape or \
                    not np.array_equal(
                        supplied_topology, self.training_embeddings):
                raise ValueError(
                    "topology_embeddings must equal training_embeddings; "
                    "validation/searchable embeddings cannot fit topology")
        phase_seconds["input_normalization"] = time.perf_counter() - start
        self.tree = ProjectedPartitionTree(
            projection_dim=projection_dim, leaf_capacity=leaf_capacity, seed=seed)
        # Projections and split tests are learned from D_tr only.  Searchable
        # tuples are materialized into that frozen hierarchy below.
        start = time.perf_counter()
        self.tree.fit_structure(
            self.training_embeddings,
            topology_source="training_embeddings")
        phase_seconds["tree_fit_structure"] = time.perf_counter() - start
        start = time.perf_counter()
        self.tree.materialize(self.searchable_embeddings)
        phase_seconds["tree_materialize"] = time.perf_counter() - start
        # The validation-selected hierarchy remains the sole source of
        # inference contexts.  Approximate serving may optionally use a finer
        # physical partition tree for geometric traversal.  That tree is fit
        # on the same D_tr fold and never contributes support evidence,
        # policies, thresholds, or eligibility decisions.
        if search_projection_dim is None and search_leaf_capacity is None:
            self.search_tree = self.tree
        else:
            search_projection_dim = int(
                projection_dim if search_projection_dim is None else
                search_projection_dim)
            search_leaf_capacity = int(
                leaf_capacity if search_leaf_capacity is None else
                search_leaf_capacity)
            self.search_tree = ProjectedPartitionTree(
                projection_dim=search_projection_dim,
                leaf_capacity=search_leaf_capacity, seed=seed)
            start = time.perf_counter()
            self.search_tree.fit_structure(
                self.training_embeddings,
                topology_source="training_embeddings_search_only")
            phase_seconds["search_tree_fit_structure"] = \
                time.perf_counter() - start
            start = time.perf_counter()
            self.search_tree.materialize(self.searchable_embeddings)
            phase_seconds["search_tree_materialize"] = \
                time.perf_counter() - start
        start = time.perf_counter()
        training_leaf_ids = self.tree.route(self.training_embeddings)
        phase_seconds["training_route"] = time.perf_counter() - start
        start = time.perf_counter()
        self.support_index = SharedSupportIndex(
            training_table, dtypes, training_leaf_ids, self.tree)
        phase_seconds["support_index"] = time.perf_counter() - start
        start = time.perf_counter()
        self.inference = IndexedJointInference(
            searchable_table, self.searchable_embeddings, self.tree,
            self.support_index, policy, theta, deterministic_constraints,
            support_mode=support_mode,
            template_selection_mode=template_selection_mode,
            support_retrieval_mode=support_retrieval_mode,
            context_mode=context_mode)
        phase_seconds["inference"] = time.perf_counter() - start
        external_ann_build_seconds = float(external_ann_build_seconds)
        if not np.isfinite(external_ann_build_seconds) or \
                external_ann_build_seconds < 0.0:
            raise ValueError(
                "external_ann_build_seconds must be finite and non-negative")
        if ann_backend is None:
            start = time.perf_counter()
            ann_backend = HNSWBackend(seed=seed).build(self.searchable_embeddings)
            phase_seconds["ann_backend"] = time.perf_counter() - start
            ann_build_source = "constructed_inside_paper_mifa"
        else:
            phase_seconds["ann_backend"] = external_ann_build_seconds
            ann_build_source = "caller_supplied_backend"
        self.ann_backend = ann_backend
        start = time.perf_counter()
        self.executor = UnifiedBestFirstExecutor(
            self.searchable_embeddings, self.search_tree, self.inference,
            ann_backend=self.ann_backend, seed_size=ann_seed_size,
            fused_cpu=fused_cpu, leaf_frontier=leaf_frontier,
            distance_provider=distance_provider,
            allow_distance_order_specialization=(
                allow_distance_order_specialization),
            approximate_tail_budget=approximate_tail_budget)
        phase_seconds["executor"] = time.perf_counter() - start
        internal_wall_seconds = time.perf_counter() - construction_start
        self.construction_timing = {
            "scope": (
                "deployment snapshot construction; offline design selection "
                "and validation/calibration are reported by the runner"),
            "ann_build_source": ann_build_source,
            "internal_wall_seconds": float(internal_wall_seconds),
            "external_ann_build_seconds": float(external_ann_build_seconds),
            "total_seconds": float(
                internal_wall_seconds + external_ann_build_seconds),
            "phase_seconds": {
                key: float(value) for key, value in phase_seconds.items()
            },
        }

    def rebuild_searchable_snapshot(
            self, searchable_embeddings, searchable_table, *,
            ann_backend=None, distance_provider=None,
            external_ann_build_seconds=0.0):
        """Build one searchable snapshot while reusing the frozen design.

        The projected topology, historical support index, deployment policy,
        threshold, and deterministic constraints are immutable design state.
        A restart or a base--delta segment only rematerializes searchable MBRs,
        tuple-side inference state, the ANN backend, and the executor.  This is
        the Exp-4 contract: newly arrived tuples are searchable without being
        added to historical evidence before an explicit refresh.
        """
        started = time.perf_counter()
        phase_seconds = {}
        phase_start = time.perf_counter()
        embeddings = np.ascontiguousarray(
            searchable_embeddings, dtype=np.float32)
        table = searchable_table.reset_index(drop=True).copy()
        if embeddings.ndim != 2 or embeddings.shape[1] != \
                self.training_embeddings.shape[1]:
            raise ValueError(
                "searchable snapshot must be a 2-D matrix with the frozen "
                "embedding dimension")
        if len(table) != len(embeddings) or len(embeddings) == 0:
            raise ValueError(
                "searchable table/embeddings must align and be non-empty")
        phase_seconds["input_normalization"] = \
            time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        tree = copy.deepcopy(self.tree)
        tree.materialize(embeddings)
        phase_seconds["tree_materialize"] = time.perf_counter() - phase_start

        if self.search_tree is self.tree:
            search_tree = tree
        else:
            phase_start = time.perf_counter()
            search_tree = copy.deepcopy(self.search_tree)
            search_tree.materialize(embeddings)
            phase_seconds["search_tree_materialize"] = \
                time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        inference = IndexedJointInference(
            table, embeddings, tree, self.support_index,
            self.inference.policy, self.inference.theta,
            self.inference.deterministic_constraints,
            support_mode=self.inference.support_mode,
            template_selection_mode=self.inference.template_selection_mode,
            support_retrieval_mode=self.inference.support_retrieval_mode,
            context_mode=self.inference.context_mode)
        phase_seconds["inference"] = time.perf_counter() - phase_start

        external_ann_build_seconds = float(external_ann_build_seconds)
        if not np.isfinite(external_ann_build_seconds) or \
                external_ann_build_seconds < 0.0:
            raise ValueError(
                "external_ann_build_seconds must be finite and non-negative")
        if ann_backend is None:
            phase_start = time.perf_counter()
            if isinstance(self.ann_backend, ExactBackend):
                ann_backend = ExactBackend().build(embeddings)
            elif isinstance(self.ann_backend, HNSWBackend):
                ann_backend = HNSWBackend(
                    M=self.ann_backend.M,
                    efConstruction=self.ann_backend.efC,
                    efSearch=self.ann_backend.efS,
                    seed=self.ann_backend.seed).build(embeddings)
            else:
                raise TypeError(
                    "restart requires an explicit backend for an unknown "
                    "ANN implementation")
            phase_seconds["ann_backend"] = \
                time.perf_counter() - phase_start
            ann_build_source = "constructed_inside_frozen_snapshot"
        else:
            phase_seconds["ann_backend"] = external_ann_build_seconds
            ann_build_source = "caller_supplied_backend"

        phase_start = time.perf_counter()
        executor = UnifiedBestFirstExecutor(
            embeddings, search_tree, inference, ann_backend=ann_backend,
            seed_size=self.executor.seed_size,
            fused_cpu=self.executor.fused_cpu,
            leaf_frontier=self.executor.leaf_frontier,
            distance_provider=distance_provider,
            allow_distance_order_specialization=(
                self.executor.allow_distance_order_specialization),
            approximate_tail_budget=(
                self.executor.approximate_tail_budget))
        phase_seconds["executor"] = time.perf_counter() - phase_start

        rebuilt = self.__class__.__new__(self.__class__)
        rebuilt.training_embeddings = self.training_embeddings
        rebuilt.searchable_embeddings = embeddings
        rebuilt.tree = tree
        rebuilt.search_tree = search_tree
        rebuilt.support_index = self.support_index
        rebuilt.inference = inference
        rebuilt.ann_backend = ann_backend
        rebuilt.executor = executor
        internal_wall_seconds = time.perf_counter() - started
        rebuilt.construction_timing = {
            "scope": (
                "searchable snapshot maintenance with frozen topology, "
                "historical support, policy, and theta"),
            "ann_build_source": ann_build_source,
            "internal_wall_seconds": float(internal_wall_seconds),
            "external_ann_build_seconds": float(external_ann_build_seconds),
            "total_seconds": float(
                internal_wall_seconds + external_ann_build_seconds),
            "phase_seconds": {
                key: float(value) for key, value in phase_seconds.items()
            },
            "frozen_components": {
                "projected_topology": True,
                "historical_support_index": True,
                "deployment_policy": True,
                "theta": True,
                "deterministic_constraints": True,
            },
        }
        return rebuilt

    def search(self, query_embedding, k, predicate, epsilon=0.0):
        return self.executor.search(query_embedding, k, predicate, epsilon)

    def materialize_eligibility(self, predicate):
        """Correctness oracle and fixed-eligibility isolation only."""
        return self.inference.materialize_eligibility(predicate)

    def search_batch(self, query_embeddings, ks, predicates, epsilon=0.0,
                     *, batch_size=512):
        """Prepare CUDA distances in one batch, then replay faithful schedulers."""
        provider = self.executor.distance_provider
        if provider is None or not hasattr(provider, "prepare_batch"):
            raise RuntimeError("a batch-capable distance provider is required")
        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        ks = list(ks) if not np.isscalar(ks) else [int(ks)] * len(queries)
        predicates = list(predicates)
        if len(ks) != len(queries) or len(predicates) != len(queries):
            raise ValueError("queries, ks, and predicates must align")
        distance_rows = provider.prepare_batch(queries, batch_size=batch_size)
        return [
            self.executor.search(
                query, int(k), predicate, epsilon,
                precomputed_distances=distance_rows[row])
            for row, (query, k, predicate) in enumerate(
                zip(queries, ks, predicates))
        ]

    def search_candidate_stream_batch(
            self, query_embeddings, ks, predicates, candidate_provider, *,
            initial_page_size=4096, max_page_size=65536,
            max_candidate_events=None, fallback_to_reference=False):
        """Run fail-closed exact candidate stream; fallback is diagnostic."""
        return self.executor.search_candidate_stream_batch(
            query_embeddings, ks, predicates, candidate_provider,
            initial_page_size=initial_page_size,
            max_page_size=max_page_size,
            max_candidate_events=max_candidate_events,
            fallback_to_reference=fallback_to_reference)

    def implementation_manifest(self):
        fitted = self.inference.policy.fit_manifest
        distance_provider = self.executor.distance_provider
        return {
            "method_id": self.METHOD_ID,
            "features": {
                "projected_hierarchy": True,
                "ancestor_backoff": True,
                "deterministic_constraint_domains": True,
                "deterministic_constraints_active": bool(
                    self.inference.deterministic_constraints),
                "scalar_backoff": True,
                "ordered_templates": True,
                "joint_estimator": True,
                "support_mode": self.inference.support_mode,
                "template_selection_mode": (
                    self.inference.template_selection_mode),
                "support_retrieval_mode": (
                    self.inference.support_retrieval_mode),
                "context_mode": self.inference.context_mode,
                "recursive_shrinkage": True,
                "per_template_score_maps": True,
                "indexed_support_retrieval": True,
                "prebuilt_low_cardinality_body_masks": True,
                "ann_seed": self.ann_backend is not None,
                "unified_node_tuple_queue": not (
                    self.executor.allow_distance_order_specialization),
                "lazy_statistical_inference": True,
                "exact_termination": True,
                "membership_protected_epsilon_termination": True,
                "approximate_tail_budget": int(
                    self.executor.approximate_tail_budget),
                "fused_cpu_executor": self.executor.fused_cpu,
                "compressed_leaf_frontier": self.executor.leaf_frontier,
                "native_unified_leaf_scheduler": True,
                "exact_distance_order_specialization": bool(
                    self.executor.allow_distance_order_specialization),
                "gpu_exact_candidate_stream_available": True,
                "candidate_stream_fail_closed_by_default": True,
                "immutable_inference_result_cache": True,
                "dense_predicate_eligibility_cache": True,
                "base_delta": False,
                "joint_structure_policy_selection": bool(
                    fitted.get("joint_design_selection", False)),
                "validation_exact_executor_replay_timecost": bool(
                    fitted.get("paper_exact_executor_replay_timecost", False)),
                "cross_fitted_score_maps": bool(fitted.get("cross_fitted", False)),
            },
            "tree": self.tree.manifest(),
            "physical_search_tree": self.search_tree.manifest(),
            "physical_search_tree_separate": bool(
                self.search_tree is not self.tree),
            "executor_contract": {
                "formal_online_path": (
                    "distance_order_ablation" if
                    self.executor.allow_distance_order_specialization else
                    "unified_tree_ann"),
                "distance_order_scope": "explicit_ablation_only",
                "candidate_stream_reference_fallback_default": False,
                "positive_epsilon_contract": (
                    "standard_(1+epsilon)_distance_bound_plus_exact_"
                    "top_(k-tail_budget)_membership_prefix"),
                "approximate_tail_budget": int(
                    self.executor.approximate_tail_budget),
            },
            "support_index": self.support_index.manifest(),
            "serving_cache": self.inference.serving_cache_manifest(),
            "distance_provider": (
                distance_provider.manifest()
                if distance_provider is not None and
                hasattr(distance_provider, "manifest") else None),
            "construction_timing": self.construction_timing,
        }


class PaperMIFABaseDelta:
    """Versioned immutable-base / append-only-delta PaperMIFA serving object.

    Appends rebuild only the cumulative delta segment.  The base object and all
    historical design state remain unchanged.  Searches run the paper unified
    tree/ANN executor on both segments and perform a deterministic exact merge
    by ``(squared_distance, global_id)``.
    """

    METHOD_ID = f"{PaperMIFA.METHOD_ID}-base-delta"

    def __init__(self, base_model, base_global_ids, *,
                 ann_backend_factory=None, distance_provider_factory=None,
                 parallel_segment_search=False,
                 candidate_stream_provider_factory=None,
                 base_candidate_stream_provider=None,
                 candidate_stream_initial_page_size=4096,
                 candidate_stream_max_page_size=65536):
        if not isinstance(base_model, PaperMIFA):
            raise TypeError("base_model must be PaperMIFA")
        base_ids = np.ascontiguousarray(base_global_ids, dtype=np.int64)
        if base_ids.shape != (len(base_model.searchable_embeddings),) or \
                len(np.unique(base_ids)) != len(base_ids):
            raise ValueError("base_global_ids must be unique and row-aligned")
        self.base_model = base_model
        self.base_global_ids = base_ids
        self.ann_backend_factory = ann_backend_factory
        self.distance_provider_factory = distance_provider_factory
        self.candidate_stream_provider_factory = \
            candidate_stream_provider_factory
        self.candidate_stream_initial_page_size = int(
            candidate_stream_initial_page_size)
        self.candidate_stream_max_page_size = int(
            candidate_stream_max_page_size)
        if self.candidate_stream_initial_page_size < 1 or \
                self.candidate_stream_max_page_size < \
                self.candidate_stream_initial_page_size:
            raise ValueError("candidate-stream page sizes are invalid")
        self.base_candidate_stream_provider = (
            base_candidate_stream_provider
            if base_candidate_stream_provider is not None else
            (None if candidate_stream_provider_factory is None else
             candidate_stream_provider_factory(
                 base_model.searchable_embeddings)))
        self.delta_candidate_stream_provider = None
        self.parallel_segment_search = bool(parallel_segment_search)
        self._segment_pool = (
            ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="mifa-base-delta")
            if self.parallel_segment_search else None)
        self.delta_model = None
        self.delta_global_ids = np.empty(0, dtype=np.int64)
        self._delta_embeddings = np.empty(
            (0, base_model.searchable_embeddings.shape[1]), dtype=np.float32)
        self._delta_table = pd.DataFrame(
            columns=base_model.inference.target_table.columns)
        self.version = 0
        self.append_receipts = []
        self._base_identity = id(base_model)
        self._policy_identity = id(base_model.inference.policy)
        self._support_identity = id(base_model.support_index)
        self._theta = float(base_model.inference.theta)

    @property
    def searchable_embeddings(self):
        return np.concatenate(
            (self.base_model.searchable_embeddings, self._delta_embeddings),
            axis=0)

    def append(self, embeddings, table, global_ids):
        """Append one arrival batch and publish a new delta version."""
        values = np.ascontiguousarray(embeddings, dtype=np.float32)
        ids = np.ascontiguousarray(global_ids, dtype=np.int64)
        batch_table = table.reset_index(drop=True).copy()
        if values.ndim != 2 or values.shape[1] != \
                self.base_model.searchable_embeddings.shape[1]:
            raise ValueError("delta embeddings have the wrong shape")
        if len(values) == 0 or len(batch_table) != len(values) or \
                ids.shape != (len(values),):
            raise ValueError("delta embeddings/table/global_ids must align")
        if list(batch_table.columns) != list(self._delta_table.columns):
            raise ValueError("delta table columns drifted from the base snapshot")
        existing = np.concatenate(
            (self.base_global_ids, self.delta_global_ids))
        if len(np.unique(ids)) != len(ids) or np.intersect1d(
                existing, ids, assume_unique=False).size:
            raise ValueError("delta global IDs must be new and unique")

        started = time.perf_counter()
        self._delta_embeddings = np.concatenate(
            (self._delta_embeddings, values), axis=0)
        self._delta_table = pd.concat(
            (self._delta_table, batch_table), ignore_index=True)
        self.delta_global_ids = np.concatenate(
            (self.delta_global_ids, ids))
        backend = (None if self.ann_backend_factory is None else
                   self.ann_backend_factory(self._delta_embeddings))
        provider = (None if self.distance_provider_factory is None else
                    self.distance_provider_factory(self._delta_embeddings))
        backend_seconds = 0.0
        if backend is not None:
            backend_started = time.perf_counter()
            if getattr(backend, "X", None) is None and \
                    getattr(backend, "index", None) is None:
                backend.build(self._delta_embeddings)
            backend_seconds = time.perf_counter() - backend_started
        self.delta_model = self.base_model.rebuild_searchable_snapshot(
            self._delta_embeddings, self._delta_table,
            ann_backend=backend, distance_provider=provider,
            external_ann_build_seconds=backend_seconds)
        self.delta_candidate_stream_provider = (
            None if self.candidate_stream_provider_factory is None else
            self.candidate_stream_provider_factory(self._delta_embeddings))
        self.version += 1
        receipt = {
            "version": int(self.version),
            "batch_rows": int(len(values)),
            "delta_rows": int(len(self._delta_embeddings)),
            "maintenance_seconds": float(time.perf_counter() - started),
            "base_rows": int(len(self.base_global_ids)),
            "base_object_unchanged": id(self.base_model) == self._base_identity,
            "policy_frozen": id(self.delta_model.inference.policy) ==
            self._policy_identity,
            "historical_support_frozen": id(self.delta_model.support_index) ==
            self._support_identity,
            "theta_frozen": float(self.delta_model.inference.theta) == self._theta,
        }
        if not all(receipt[key] for key in (
                "base_object_unchanged", "policy_frozen",
                "historical_support_frozen", "theta_frozen")):
            raise AssertionError("base--delta frozen-state contract changed")
        self.append_receipts.append(receipt)
        return dict(receipt)

    def _result(self, model, global_ids, query, k, predicate, epsilon,
                candidate_stream_provider=None):
        if candidate_stream_provider is None:
            local_ids, distances, counters = model.search(
                query, k, predicate, epsilon)
        else:
            local_ids, distances, counters = \
                model.search_candidate_stream_batch(
                    np.asarray(query, dtype=np.float32)[None, :],
                    [int(k)], [predicate], candidate_stream_provider,
                    initial_page_size=self.candidate_stream_initial_page_size,
                    max_page_size=self.candidate_stream_max_page_size,
                    fallback_to_reference=False)[0]
        local_ids = np.asarray(local_ids, dtype=np.int64)
        return (global_ids[local_ids],
                np.asarray(distances, dtype=np.float64), counters)

    def search(self, query_embedding, k, predicate, epsilon=0.0):
        if self.delta_model is None:
            base_ids, base_distances, base_counters = self._result(
                self.base_model, self.base_global_ids,
                query_embedding, k, predicate, epsilon,
                self.base_candidate_stream_provider)
            return base_ids, base_distances, {
                "base": base_counters, "delta": None,
                "merge_candidates": int(len(base_ids)),
                "version": int(self.version)}
        if self._segment_pool is None:
            base_ids, base_distances, base_counters = self._result(
                self.base_model, self.base_global_ids,
                query_embedding, k, predicate, epsilon,
                self.base_candidate_stream_provider)
            delta_ids, delta_distances, delta_counters = self._result(
                self.delta_model, self.delta_global_ids,
                query_embedding, k, predicate, epsilon,
                self.delta_candidate_stream_provider)
        else:
            base_future = self._segment_pool.submit(
                self._result, self.base_model, self.base_global_ids,
                query_embedding, k, predicate, epsilon,
                self.base_candidate_stream_provider)
            delta_ids, delta_distances, delta_counters = self._result(
                self.delta_model, self.delta_global_ids,
                query_embedding, k, predicate, epsilon,
                self.delta_candidate_stream_provider)
            base_ids, base_distances, base_counters = base_future.result()
        ids = np.concatenate((base_ids, delta_ids))
        distances = np.concatenate((base_distances, delta_distances))
        order = np.lexsort((ids, distances))[:int(k)]
        return ids[order], distances[order], {
            "base": base_counters, "delta": delta_counters,
            "merge_candidates": int(len(ids)),
            "version": int(self.version)}

    def implementation_manifest(self):
        return {
            "method_id": self.METHOD_ID,
            "features": {
                "base_delta": True,
                "immutable_base": True,
                "append_only_delta": True,
                "versioned_publication": True,
                "frozen_policy": True,
                "frozen_theta": True,
                "frozen_historical_support_index": True,
                "new_tuples_enter_historical_evidence_before_refresh": False,
                "segment_executor": "unified_tree_ann",
                "parallel_segment_search": bool(
                    self.parallel_segment_search),
                "candidate_stream": bool(
                    self.candidate_stream_provider_factory is not None),
                "candidate_stream_initial_page_size": int(
                    self.candidate_stream_initial_page_size),
                "candidate_stream_max_page_size": int(
                    self.candidate_stream_max_page_size),
                "exact_global_merge_key": "(squared_distance,global_id)",
            },
            "version": int(self.version),
            "base_rows": int(len(self.base_global_ids)),
            "delta_rows": int(len(self.delta_global_ids)),
            "append_receipts": list(self.append_receipts),
        }
