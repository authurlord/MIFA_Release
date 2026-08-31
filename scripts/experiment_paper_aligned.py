#!/usr/bin/env python3
"""Static paper-aligned MIFA experiment entry point.

``experiment_v2.py`` is retained as a frozen two-level surrogate and as a
source of dataset/workload/scoring utilities.  The MIFA path in this file uses
the paper-defined hierarchy, ordered target-exact templates, indexed joint
support, cross-fitted isotonic maps, validation-only joint design selection,
grouped calibration, and the unified lazy node--tuple executor end to end.

The entry point records the selected paper path and its measured outputs.  It
does not make experiment execution conditional on result thresholds.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import experiment_v2 as legacy
from mifa.datasets import load_dataset
from mifa.index.ann_backend import ExactBackend, HNSWBackend
from mifa.inference.offline_design import (
    OfflineDesignSelector,
    StructuralConfig,
    build_policy_candidate_family,
    calibrate_threshold_grouped,
    discover_scalar_body_candidates,
    refit_deployment_policy,
)
from mifa.inference.fused_online import PaperGroupedEligibilityCache
from mifa.inference.fitted_imputers import MeanModeImputer, MixedIterativeImputer
from mifa.inference.paper_policy import ValidationInstance
from mifa.inference.query_cls import QueryConditionedClassifier
from mifa.knn import active_backend, query_dists_batch
from mifa.knn_numba import topk_masked_batch_numba
from mifa.missingness import (
    HISTORICAL_EXPOSURE_NESTED_PROTOCOL,
    _historical_exposure_scores,
    historical_candidate_exposure_nested_audit,
    historical_candidate_exposure_priority,
    make_missing,
    training_missingness_kind,
)
from mifa.paper_mifa import PaperMIFA
from mifa.search.cpu_distance import NumpyResidentDistanceProvider
from mifa.search.cpu_candidate_stream import (
    NumpyExactCandidateStreamProvider,
)
from mifa.search.gpu_distance import TorchResidentDistanceProvider
from mifa.search.gpu_candidate_stream import (
    TorchExactCandidateStreamProvider,
    TorchQueryShardedExactCandidateStreamProvider,
    TorchReplicatedExactCandidateStreamProvider,
    TorchShardedExactCandidateStreamProvider,
)
from mifa.search.tree_probe_candidate_stream import (
    FaissIVFPartitionIndex,
    NumpyTreeProbeCandidateStreamProvider,
    TorchTreeProbeCandidateStreamProvider,
)
from mifa.search.fused_exact import (
    CpuSelectedGroupedTopK, TorchResidentMaskedTopK,
    TorchResidentSelectedTopK, cpu_blocked_grouped_masked_topk,
)
from mifa.search.unified_executor import UnifiedBestFirstExecutor
from mifa.predicates import TRUE, UNKNOWN
from mifa.protocol import make_four_way_split, subset_spec
from mifa.scale_cohort import (
    expand_repeated_jitter_embeddings,
    project_normalize,
)
from mifa.workload_v2 import (
    selectivity_matched_arity_summary,
    workload_counts,
    workload_digest,
)


HISTORICAL_EXPOSURE_KINDS = {
    "historical_candidate_exposure",
    "historical_candidate_exposure_nested",
}


def _csv_ints(value):
    return tuple(int(item) for item in value.split(",") if item.strip())


def _csv_floats(value):
    return tuple(float(item) for item in value.split(",") if item.strip())


def _sha_json(value):
    payload = json.dumps(
        value, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _digest_array(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _padded_id_matrix(rows, k):
    """Return fixed-width top-k IDs, preserving short-result semantics."""
    width = int(k)
    if width < 0:
        raise ValueError("top-k width must be nonnegative")
    output = np.full((len(rows), width), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        values = np.asarray(row, dtype=np.int64).reshape(-1)
        if len(values) > width:
            raise ValueError(
                f"returned ID row {index} exceeds top-k width {width}")
        output[index, :len(values)] = values
    return np.ascontiguousarray(output)


class _DiagnosticProjectedLeafSeedBackend:
    """Bound-ordered projected-leaf ANN seed with exposed distance work.

    This diagnostic-only backend probes the leaf with the smallest admissible
    MBR bound (and further leaves only when needed), then ranks that compact
    candidate set by exact distance.  It is deliberately not a reported ANN
    baseline; its purpose is to retain the deployed ANN-seeding mechanism in
    the disable-one study without hiding distance computations inside a native
    library or forcing an all-N exact seed scan.
    """

    def __init__(self, embeddings, tree):
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.tree = tree
        self.last_distance_evaluations = 0
        self.last_candidate_rows = 0
        self.last_candidate_ids = np.empty(0, dtype=np.int64)
        self.last_candidate_distances = np.empty(0, dtype=np.float32)

    def search(self, query, k, subset=None):
        if subset is not None:
            raise ValueError("diagnostic projected-leaf seed is global")
        query = np.ascontiguousarray(query, dtype=np.float32)
        projected = self.tree.project(query)
        bounds = self.tree.all_node_lower_bounds_sq_projected(projected)
        leaf_ids = np.asarray([
            int(node_id) for node_id in self.tree.leaf_node_ids
            if self.tree.nodes[int(node_id)].active
        ], dtype=np.int64)
        order = np.lexsort((leaf_ids, bounds[leaf_ids]))
        candidates = []
        for slot in order:
            node = self.tree.nodes[int(leaf_ids[int(slot)])]
            candidates.extend(int(row) for row in node.tuple_ids)
            if len(candidates) >= int(k):
                break
        candidate_ids = np.asarray(candidates, dtype=np.int64)
        differences = self.embeddings[candidate_ids] - query[None, :]
        distances = np.einsum(
            "ij,ij->i", differences, differences).astype(np.float32)
        self.last_distance_evaluations = int(len(candidate_ids))
        self.last_candidate_rows = int(len(candidate_ids))
        self.last_candidate_ids = candidate_ids.copy()
        self.last_candidate_distances = distances.copy()
        count = min(int(k), len(candidate_ids))
        selected = np.lexsort((candidate_ids, distances))[:count]
        return candidate_ids[selected], distances[selected]


def _full_unresolved_count(inference, predicate):
    """Count ImputationFirst's eager unresolved rows outside timed search."""
    plan = (inference.compile_hard_predicate(predicate)
            if hasattr(inference, "compile_hard_predicate") else None)
    if plan is None:
        return int(sum(
            inference.hard_decision(row, predicate)[0] == "unresolved"
            for row in range(len(inference.target_table))))
    n_rows = plan["missing"].shape[1]
    possible = np.ones(n_rows, dtype=np.bool_)
    has_missing = np.zeros(n_rows, dtype=np.bool_)
    for attr, op, lower, upper in zip(
            plan["attrs"], plan["ops"], plan["lower"], plan["upper"]):
        missing = plan["missing"][int(attr)]
        values = plan["values"][int(attr)]
        if int(op) == 0:
            satisfied = values <= lower
        elif int(op) == 1:
            satisfied = values >= lower
        elif int(op) == 2:
            satisfied = (values >= lower) & (values <= upper)
        else:
            satisfied = values == lower
        possible &= missing | satisfied
        has_missing |= missing
    return int(np.count_nonzero(possible & has_missing))


def _run_disable_one_diagnostic(
        model, queries, *, query_rows=None, query_count=0,
        reference_rows=None, workload_name="unknown", variant_keys=None,
        persist_per_query=False):
    """Measure a frozen pruning/work-avoidance diagnostic.

    ``variant_keys`` is an opt-in closeout selector.  The historical default
    remains the five-row diagnostic; a caller can request only the declared
    cells without executing the other ablations.  ``full`` must remain first
    because ordered-ID agreement is measured against that result.
    """
    query_count = min(int(query_count), len(queries))
    if query_count <= 0:
        return None
    selected = list(queries[:query_count])
    rows = (np.arange(query_count, dtype=np.int64) if query_rows is None else
            np.asarray(query_rows, dtype=np.int64)[:query_count])
    seed_backend = _DiagnosticProjectedLeafSeedBackend(
        model.searchable_embeddings, model.tree)
    executor = UnifiedBestFirstExecutor(
        model.searchable_embeddings, model.tree, model.inference,
        ann_backend=seed_backend, seed_size=max(
            int(model.executor.seed_size), 64),
        fused_cpu=False, leaf_frontier=False, distance_provider=None,
        allow_distance_order_specialization=False)
    available_variants = {
        "full": ("Full", {}),
        "ann_seed": ("w/o ANN seed", {"disable_ann_seed": True}),
        "node_bound": ("w/o node-bound pruning", {
            "disable_node_bound_pruning": True}),
        "tuple_boundary": ("w/o tuple-boundary check", {
            "disable_tuple_boundary": True}),
        "query_cache": ("w/o query-local cache", {
            "disable_query_cache": True}),
    }
    if variant_keys is None:
        variant_keys = (
            "full", "ann_seed", "node_bound", "tuple_boundary",
            "query_cache")
    else:
        variant_keys = tuple(str(value) for value in variant_keys)
    unknown_variants = set(variant_keys) - set(available_variants)
    if unknown_variants:
        raise ValueError(
            f"unknown pruning diagnostic variants: {sorted(unknown_variants)}")
    if not variant_keys or variant_keys[0] != "full":
        raise ValueError("pruning diagnostic variants must start with full")
    variants = tuple(available_variants[key] for key in variant_keys)
    eager_unresolved = [
        _full_unresolved_count(model.inference, query["phi"])
        for query in selected]
    full_ids = None
    output = []
    traces = []
    for variant, options in variants:
        model.inference.clear_all_serving_caches()
        latencies = []
        ids_by_query = []
        counters_by_query = []
        for query_index, query in enumerate(selected):
            started = time.perf_counter()
            ids, _, counters = executor.search(
                query["q_e"], int(query["k"]), query["phi"], epsilon=0.0,
                diagnostic_options=options)
            latencies.append((time.perf_counter() - started) * 1000.0)
            ids_by_query.append(np.asarray(ids, dtype=np.int64))
            counters_by_query.append(counters)
            for event_index, event in enumerate(
                    counters.get("tau_boundary_trace", ())):
                traces.append({
                    "variant": variant,
                    "query_index": int(query_index),
                    "query_row": int(rows[query_index]),
                    "event_index": int(event_index),
                    "visited_tuples": int(event["visited_tuples"]),
                    "visited_fraction": float(
                        event["visited_tuples"] /
                        len(model.searchable_embeddings)),
                    "physical_distance_evaluations": int(
                        event["physical_distance_evaluations"]),
                    "tau_b_sq": float(event["tau_b_sq"]),
                    "tau_b": float(np.sqrt(event["tau_b_sq"])),
                })
        if full_ids is None:
            full_ids = ids_by_query
        exact_matches = sum(
            np.array_equal(left, right)
            for left, right in zip(ids_by_query, full_ids))
        if reference_rows is not None:
            source_matches = sum(np.array_equal(
                ids_by_query[index], np.asarray(
                    reference_rows[index]["returned_ids"], dtype=np.int64))
                for index in range(query_count))
        else:
            source_matches = "N/A_NO_SEPARATE_DEPLOYED_REFERENCE"
        values = np.asarray(latencies, dtype=np.float64)

        per_query = []
        if persist_per_query:
            for query_index, counters in enumerate(counters_by_query):
                fixed_agreement = bool(np.array_equal(
                    ids_by_query[query_index], full_ids[query_index]))
                deployed_agreement = (
                    bool(np.array_equal(
                        ids_by_query[query_index], np.asarray(
                            reference_rows[query_index]["returned_ids"],
                            dtype=np.int64)))
                    if reference_rows is not None else None)
                per_query.append({
                    "query_index": int(query_index),
                    "query_row": int(rows[query_index]),
                    "searchable_tuples": int(
                        len(model.searchable_embeddings)),
                    "mbr_pruned_rows": int(counters.get(
                        "mbr_pruned_subtree_rows", 0)),
                    "frontier_terminated_rows": int(counters.get(
                        "frontier_terminated_node_rows", 0)),
                    "expanded_leaves": int(counters.get(
                        "expanded_leaves", 0)),
                    "total_leaves": int(model.tree.n_leaves),
                    "unique_exact_distance_evaluations": int(counters.get(
                        "unique_physical_distance_evaluations", 0)),
                    "tuple_boundary_skips": int(counters.get(
                        "boundary_skips_before_eligibility", 0)),
                    "statistical_resolutions": int(counters.get(
                        "lazy_inferences", 0)),
                    "latency_ms": float(values[query_index]),
                    "ordered_result_id_agreement": fixed_agreement,
                    "deployed_result_id_agreement": deployed_agreement,
                })

        def mean_counter(name):
            return float(np.mean([
                int(item.get(name, 0)) for item in counters_by_query]))

        max_queue = max(int(item["max_queue_size"])
                        for item in counters_by_query)
        explicit_host_payload = int(
            len(model.searchable_embeddings) * np.dtype(np.bool_).itemsize +
            len(model.tree.nodes) * np.dtype(np.float64).itemsize +
            model.tree.n_leaves * np.dtype(np.int64).itemsize +
            max_queue * 24 + int(selected[0]["k"]) * 16)
        output.append({
            "variant": variant,
            "options": options,
            "queries": int(query_count),
            "query_rows": [int(value) for value in rows],
            "fixed_policy_exact_matches": int(exact_matches),
            "fixed_policy_id_agreement": float(exact_matches / query_count),
            "deployed_reference_exact_matches": source_matches,
            "returned_ids_sha256": _sha_json([
                item.tolist() for item in ids_by_query]),
            "latency_ms": {
                "mean": float(values.mean()),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "per_query": [float(value) for value in values],
            },
            "total_leaves": int(model.tree.n_leaves),
            "per_query": per_query,
            "counters_per_query": {
                name: mean_counter(name) for name in (
                    "visited_tuples", "physical_distance_evaluations",
                    "unique_physical_distance_evaluations",
                    "hard_decisions", "lazy_inferences", "expanded_leaves",
                    "mbr_pruned_subtree_rows",
                    "frontier_terminated_node_rows",
                    "boundary_skips_before_eligibility",
                    "queued_unresolved_resolutions_avoided",
                    "support_requests", "cache_hits", "cache_misses",
                    "ann_seed_physical_distance_evaluations",
                    "on_demand_physical_distance_evaluations")
            },
            "imputation_first_full_unresolved_resolutions_per_query": float(
                np.mean(eager_unresolved)),
            "d2h_bytes_per_query": 0,
            "explicit_host_state_bytes_per_query": explicit_host_payload,
            "explicit_host_state_scope": (
                "exact numeric payload for visited bitmap, node bounds, leaf "
                "counts, logical frontier records, and top-k buffers; Python "
                "allocator/object overhead excluded"),
            "timer_scope": (
                "projected-leaf ANN seeding when enabled + root-tree MBR "
                "frontier + on-demand exact distances + fixed-policy "
                "eligibility resolution + final exact top-k"),
        })
    if any(row["fixed_policy_id_agreement"] != 1.0 for row in output):
        raise AssertionError("disable-one diagnostic changed fixed-policy IDs")
    if reference_rows is not None and any(
            row["deployed_reference_exact_matches"] != query_count
            for row in output):
        raise AssertionError(
            "root-tree diagnostic disagrees with deployed fixed-policy IDs")
    return {
        "status": "complete",
        "workload": str(workload_name),
        "N": int(len(model.searchable_embeddings)),
        "dimension": int(model.searchable_embeddings.shape[1]),
        "k": int(selected[0]["k"]),
        "queries": int(query_count),
        "variants": output,
        "tau_boundary_trace": traces,
        "ann_seed": {
            "kind": "diagnostic_projected_leaf_bound_order",
            "distance_work_exposed": True,
            "reported_as_filtered_ann_baseline": False,
        },
        "root_tree_reference_executor": True,
        "on_demand_distances": True,
        "exact_id_audit": "PASS",
    }


def _exact_distance_order_reference(
        executor, query, k, predicate, precomputed_distances):
    """Return the fixed-policy exact oracle without a full eligibility scan.

    Once all exact distances are available, consuming tuples in deterministic
    ``(distance, id)`` order and stopping at the kth predicted-eligible tuple
    is exactly equivalent to ranking a fully materialized eligibility vector.
    This audit helper invokes the executor's already-tested specialization
    directly; it never changes the deployed executor's formal-path flag.
    """
    if not executor.fused_cpu or not hasattr(
            executor.inference, "compile_hard_predicate"):
        raise RuntimeError("exact distance-order audit needs fused hard predicates")
    executor.inference.begin_query(predicate)
    plan = executor.inference.compile_hard_predicate(predicate)
    if plan is None:
        raise RuntimeError("predicate cannot use the exact distance-order audit")
    distances = np.ascontiguousarray(precomputed_distances, dtype=np.float32)
    if distances.shape != (len(executor.embeddings),):
        raise ValueError("audit distance row must align with searchable rows")
    return executor._search_fused_distance_order(
        int(k), predicate, plan, distances)


def _source_provenance():
    paths = (
        "scripts/experiment_paper_aligned.py",
        "src/mifa/protocol.py",
        "src/mifa/missingness.py",
        "src/mifa/workload_v2.py",
        "src/mifa/index/projected_tree.py",
        "src/mifa/inference/support_index.py",
        "src/mifa/inference/paper_policy.py",
        "src/mifa/inference/offline_design.py",
        "src/mifa/search/unified_executor.py",
        "src/mifa/search/cpu_distance.py",
        "src/mifa/search/gpu_distance.py",
        "src/mifa/search/gpu_candidate_stream.py",
        "src/mifa/scale_cohort.py",
        "src/mifa/paper_mifa.py",
    )
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        dirty = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = None
    return {
        "files": list(paths),
        "content_hashes_recorded": False,
        "git_dirty_for_manifest": dirty,
        "source_commit": _git_head(),
    }


def _git_head():
    deployed = os.environ.get("MIFA_SOURCE_COMMIT")
    if deployed:
        return deployed
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _progress(stage, **details):
    print(json.dumps({"stage": stage, **details}, default=str),
          file=sys.stderr, flush=True)


def _normalize_queries(vectors, metric):
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if metric != "cosine":
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Preserve an already normalized embedding byte-for-byte.  Besides avoiding
    # redundant data movement, this keeps workload digests compatible with the
    # frozen baseline reports generated from the same vLLM/BGE assets.
    if np.all(np.isfinite(norms)) and np.all(np.abs(norms - 1.0) <= 1e-6):
        return vectors
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _historical_query_bundle(query_spec, native_filters, attrs,
                             signature_mix):
    """Return frozen embedding-only history and its predicate-attribute use."""
    if query_spec is None or query_spec.n == 0:
        raise ValueError(
            "historical candidate exposure needs a non-empty historical role")
    attr_position = {attr: index for index, attr in enumerate(attrs)}
    descriptors = []
    if native_filters is not None:
        if len(native_filters) != query_spec.n:
            raise ValueError("historical native filters are not row-aligned")
        for item in native_filters:
            family = str(item.get("family", "r")).lower()
            if family == "explicit":
                used = [str(atom["attr"]) for atom in item.get("atoms", ())]
            elif family == "r":
                used = ["update_date"]
            elif family == "em":
                used = ["number_of_sub_categories"]
            else:
                raise ValueError(
                    f"unsupported historical native filter family: {family}")
            unknown = sorted(set(used) - set(attr_position))
            if unknown:
                raise ValueError(
                    f"historical filters contain unknown attrs: {unknown}")
            descriptors.append({
                "attributes": [attr_position[attr] for attr in used],
            })
    else:
        schedule = []
        for item in signature_mix or ():
            chosen = tuple(str(attr) for attr in (
                item["attrs"] if isinstance(item, dict) else item))
            weight = int(round(
                float(item.get("weight", 1.0))
                if isinstance(item, dict) else 1.0))
            schedule.extend([chosen] * max(1, weight))
        if not schedule:
            raise ValueError(
                "historical candidate exposure needs predeclared signatures "
                "when native filters are unavailable")
        for row in range(query_spec.n):
            chosen = schedule[row % len(schedule)]
            unknown = sorted(set(chosen) - set(attr_position))
            if unknown:
                raise ValueError(
                    f"historical signatures contain unknown attrs: {unknown}")
            descriptors.append({
                "attributes": [attr_position[attr] for attr in chosen],
            })
    return query_spec.normalized_embeddings(), descriptors


def _historical_candidate_ids(query_vectors, embeddings, top_l, *,
                              device, batch_size):
    """Compute exact embedding-only historical top-L candidate sets offline."""
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    width = min(int(top_l), len(embeddings))
    if width < 1:
        raise ValueError("historical top-L requires a non-empty relation")
    try:
        provider = TorchResidentDistanceProvider(embeddings, device=device)
    except (RuntimeError, ImportError):
        provider = NumpyResidentDistanceProvider(embeddings)
    output = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        distances = provider.prepare_batch(
            queries[start:end], batch_size=batch_size)
        for row in distances:
            ids = np.argpartition(row, width - 1)[:width]
            order = np.lexsort((ids, row[ids]))
            output.append(ids[order].astype(np.int64).tolist())
    return output


def _direct_answer_mask_metrics(mask, attrs, queries, truth_ids):
    """Measure answer exposure directly from the frozen truth and mask.

    This evaluator-owned diagnostic is deliberately independent of Drop (or
    any other method).  A query is affected when at least one of its complete-
    data answer tuples has a masked cell on an attribute referenced by that
    query.  The slot rate uses valid complete-answer slots as its denominator.
    """
    attr_positions = {str(attr): index for index, attr in enumerate(attrs)}
    affected_queries = np.zeros(len(queries), dtype=np.bool_)
    affected_slots = 0
    answer_slots = 0
    for query_index, (query, answer_ids) in enumerate(zip(queries, truth_ids)):
        referenced = tuple(dict.fromkeys(
            str(atom.attr) for atom in query["phi"].atoms))
        unknown = sorted(set(referenced) - set(attr_positions))
        if unknown:
            raise ValueError(
                f"query references attributes outside the mask schema: {unknown}")
        columns = [attr_positions[attr] for attr in referenced]
        for row in np.asarray(answer_ids, dtype=np.int64):
            if row < 0:
                continue
            answer_slots += 1
            slot_affected = bool(columns and mask[int(row), columns].any())
            if slot_affected:
                affected_slots += 1
                affected_queries[query_index] = True
    return {
        "affected_queries": affected_queries,
        "affected_query_count": int(affected_queries.sum()),
        "affected_query_rate": float(affected_queries.mean()),
        "affected_answer_slots": int(affected_slots),
        "answer_slots": int(answer_slots),
        "masked_answer_slot_rate": (
            float(affected_slots / answer_slots) if answer_slots else 0.0),
    }


def _predicate_diagnostics(model, context):
    """Evaluate test predicate decisions without changing the timed counters.

    The online executor resolves only tuples reached before its top-k stopping
    boundary.  This diagnostic separately materializes the fixed deployed
    policy over every test tuple/query pair so false admission, false
    rejection, support coverage, and abstention have a stable denominator.
    Hidden values are used only by this evaluator-owned test measurement.
    """
    inference = model.inference
    saved_counters = dict(inference.counters)
    tp = fp = fn = tn = 0
    unresolved_pairs = resolved_pairs = abstained_pairs = 0
    strata_counts = {
        axis: {} for axis in
        ("arity", "predicate_type", "selectivity_bucket")}
    try:
        for query, truth in zip(
                context["queries"], context["truth_elig"]):
            states = query["phi"].eval_table(context["observed"])
            hard_pass = np.asarray(states == TRUE, dtype=np.bool_)
            unresolved = np.flatnonzero(states == UNKNOWN)
            predicted = hard_pass.copy()
            if len(unresolved):
                inference.begin_query(query["phi"])
                codes, _batch = inference.evaluate_candidate_batch(
                    unresolved, query["phi"])
                accepted = codes == 1
                abstained = codes == 3
                predicted[unresolved[accepted]] = True
                unresolved_pairs += int(len(unresolved))
                abstained_pairs += int(np.count_nonzero(abstained))
                resolved_pairs += int(np.count_nonzero(~abstained))
            truth = np.asarray(truth, dtype=np.bool_)
            query_counts = {
                "tp": int(np.count_nonzero(predicted & truth)),
                "fp": int(np.count_nonzero(predicted & ~truth)),
                "fn": int(np.count_nonzero(~predicted & truth)),
                "tn": int(np.count_nonzero(~predicted & ~truth)),
                "unresolved": int(len(unresolved)),
                "resolved": int(np.count_nonzero(~abstained))
                if len(unresolved) else 0,
                "abstained": int(np.count_nonzero(abstained))
                if len(unresolved) else 0,
                "queries": 1,
            }
            tp += query_counts["tp"]
            fp += query_counts["fp"]
            fn += query_counts["fn"]
            tn += query_counts["tn"]
            atom_kinds = {
                "categorical" if atom.op == "eq" else "numerical"
                for atom in query["phi"].atoms}
            values = {
                "arity": str(len(query["phi"].atoms)),
                "predicate_type": (
                    next(iter(atom_kinds)) if len(atom_kinds) == 1
                    else "mixed"),
                "selectivity_bucket": str(query.get(
                    "selectivity_bucket", "unknown")),
            }
            for axis, value in values.items():
                target = strata_counts[axis].setdefault(value, {
                    key: 0 for key in query_counts})
                for key, count in query_counts.items():
                    target[key] += count
    finally:
        inference.counters.clear()
        inference.counters.update(saved_counters)
    def summarize(counts):
        local_tp = counts["tp"]
        local_fp = counts["fp"]
        local_fn = counts["fn"]
        local_tn = counts["tn"]
        local_unresolved = counts["unresolved"]
        return {
            "queries": int(counts["queries"]),
            "pairs": int(local_tp + local_fp + local_fn + local_tn),
            "support_coverage": (
                float(counts["resolved"] / local_unresolved)
                if local_unresolved else 1.0),
            "abstention_rate": (
                float(counts["abstained"] / local_unresolved)
                if local_unresolved else 0.0),
            "predicate_precision": (
                float(local_tp / (local_tp + local_fp))
                if local_tp + local_fp else 1.0),
            "predicate_recall": (
                float(local_tp / (local_tp + local_fn))
                if local_tp + local_fn else 1.0),
            "false_admission_rate": (
                float(local_fp / (local_fp + local_tn))
                if local_fp + local_tn else 0.0),
            "false_rejection_rate": (
                float(local_fn / (local_fn + local_tp))
                if local_fn + local_tp else 0.0),
        }

    return {
        "scope": "complete_test_tuple_query_pairs",
        "pairs": int(tp + fp + fn + tn),
        "unresolved_pairs": int(unresolved_pairs),
        "resolved_pairs": int(resolved_pairs),
        "abstained_pairs": int(abstained_pairs),
        "support_coverage": (
            float(resolved_pairs / unresolved_pairs)
            if unresolved_pairs else 1.0),
        "abstention_rate": (
            float(abstained_pairs / unresolved_pairs)
            if unresolved_pairs else 0.0),
        "predicate_precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "predicate_recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "false_admission_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_rejection_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "strata": {
            axis: {value: summarize(counts)
                   for value, counts in sorted(values.items())}
            for axis, values in strata_counts.items()
        },
    }


def _prepare_eval(partition, *, role, n_queries, k, rho, mask_seed,
                  workload_seed, attrs, predicate_reference,
                  missingness_kind, block_size, query_spec=None,
                  block_attrs=(), attribute_weights=None,
                  answer_impact_fraction=1.0,
                  answer_cells_per_tuple=1,
                  native_filters=None, signature_mix=None,
                  matched_arity_signatures=(),
                  matched_arity_count_per_signature=False,
                  canonical_bucket_weights=(0.5, 0.5, 0.0),
                  historical_query_vectors=None,
                  historical_query_descriptors=None,
                  historical_top_l=100, historical_beta_exp=0.50,
                  historical_device="cuda:0", historical_batch_size=16):
    """Build an evaluator-only context without using the surrogate cell index."""
    workload_query_count = int(n_queries)
    if matched_arity_count_per_signature:
        if not matched_arity_signatures:
            raise ValueError(
                "per-signature query counts require matched arity signatures")
        workload_query_count *= len(matched_arity_signatures)
    if query_spec is None:
        query_spec, base, query_rows, search_rows = legacy._holdout_query_sources(
            partition, n_queries, workload_seed + 1009, role)
    else:
        if query_spec.n < n_queries and not query_spec.meta.get("allow_source_reuse"):
            raise ValueError(
                f"{role}: frozen query role has {query_spec.n}, expected {n_queries}")
        base = partition
        query_rows = np.asarray(query_spec.meta["source_ids"], dtype=np.int64)
        search_rows = np.arange(base.n, dtype=np.int64)
    base.query_relevant_attrs = list(attrs)
    queries = legacy._make_workload(
        query_spec, base, workload_query_count, k, seed=workload_seed,
        attrs=attrs,
        predicate_reference=predicate_reference, native_filters=native_filters,
        signature_mix=signature_mix,
        canonical_bucket_weights=canonical_bucket_weights,
        matched_arity_signatures=matched_arity_signatures)
    query_vectors = _normalize_queries(
        np.asarray([query["q_e"] for query in queries], dtype=np.float32),
        base.metric)
    for query, vector in zip(queries, query_vectors):
        query["q_e"] = vector
    embeddings = np.ascontiguousarray(base.normalized_embeddings(), dtype=np.float32)
    d2 = query_dists_batch(
        query_vectors, embeddings, np.einsum("ij,ij->i", embeddings, embeddings))
    truth_eligibility = np.asarray([
        query["phi"].eval_table(base.table) == TRUE for query in queries
    ], dtype=np.bool_)
    truth_ids = topk_masked_batch_numba(d2, truth_eligibility, k)[0]
    historical_candidate_ids = None
    historical_priority = None
    nested_exposure_protocol = None
    if missingness_kind in HISTORICAL_EXPOSURE_KINDS:
        historical_candidate_ids = _historical_candidate_ids(
            historical_query_vectors, embeddings, historical_top_l,
            device=historical_device, batch_size=historical_batch_size)
    if missingness_kind == "historical_candidate_exposure_nested":
        designated_indices = [attrs.index(attr) for attr in block_attrs]
        historical_priority = historical_candidate_exposure_priority(
            base.n, len(attrs), designated_indices,
            historical_query_descriptors, historical_candidate_ids,
            beta_exp=historical_beta_exp,
            rng=np.random.default_rng(mask_seed))
        audit_rhos = tuple(sorted({0.1, 0.2, 0.3, float(rho)}))
        nested_exposure_protocol = \
            historical_candidate_exposure_nested_audit(
                historical_priority, base.n, len(attrs), rhos=audit_rhos)
        nested_exposure_protocol.update({
            "priority_seed": int(mask_seed),
            "actual_rho": float(rho),
        })
    observed, mask, _ = make_missing(
        base, rho, missingness_kind, seed=mask_seed, block_size=block_size,
        block_attrs=block_attrs, usage_freq=attribute_weights,
        queries=queries, truth_ids=truth_ids,
        answer_impact_fraction=answer_impact_fraction,
        answer_cells_per_tuple=answer_cells_per_tuple,
        historical_queries=historical_query_descriptors,
        historical_candidate_ids=historical_candidate_ids,
        beta_exp=historical_beta_exp,
        historical_priority=historical_priority)
    historical_exposure_diagnostics = None
    if missingness_kind in HISTORICAL_EXPOSURE_KINDS:
        designated_attr_names = block_attrs or tuple(attrs)
        designated_indices = [
            attrs.index(attr) for attr in designated_attr_names]
        _, exposure_scores = _historical_exposure_scores(
            base.n, len(attrs), designated_indices,
            historical_query_descriptors, historical_candidate_ids)
        masked_scores = np.concatenate([
            exposure_scores[index][mask[:, index]]
            for index in designated_indices
            if np.any(mask[:, index])
        ])
        historical_exposure_diagnostics = {
            "masked_assignments": int(len(masked_scores)),
            "positive_exposure_mask_fraction": float(
                np.mean(masked_scores > 0.0)) if len(masked_scores) else 0.0,
            "mean_exposure_score_masked": float(
                np.mean(masked_scores)) if len(masked_scores) else 0.0,
        }
    if nested_exposure_protocol is not None:
        actual_digest = _digest_array(np.packbits(mask))
        expected = next(
            item["mask_digest"]
            for item in nested_exposure_protocol["rho_levels"]
            if np.isclose(item["rho"], rho, rtol=0.0, atol=1e-12))
        if actual_digest != expected:
            raise RuntimeError(
                "nested exposure mask does not match its frozen priority "
                "audit")
        nested_exposure_protocol["actual_mask_digest"] = actual_digest
        nested_exposure_protocol["actual_mask_matches_level"] = True
    drop_eligibility = np.asarray([
        query["phi"].eval_table(observed) == TRUE for query in queries
    ], dtype=np.bool_)
    drop_ids = topk_masked_batch_numba(d2, drop_eligibility, k)[0]
    masked_answer_affected = np.asarray([
        set(returned[returned >= 0]) != set(truth[truth >= 0])
        for returned, truth in zip(drop_ids, truth_ids)
    ], dtype=np.bool_)
    direct_metrics = _direct_answer_mask_metrics(
        mask, attrs, queries, truth_ids)
    if historical_exposure_diagnostics is not None:
        historical_exposure_diagnostics["affected_query_rate"] = float(
            direct_metrics["affected_query_rate"])
    return {
        "role": role,
        "base": base,
        "observed": observed,
        "mask": mask,
        "queries": queries,
        "embeddings": embeddings,
        "D2": d2,
        "truth_elig": truth_eligibility,
        "truth_ids": truth_ids,
        "drop_ids": drop_ids,
        "masked_answer_affected": masked_answer_affected,
        "affected_query_direct": direct_metrics["affected_queries"],
        "affected_query_count_direct": direct_metrics[
            "affected_query_count"],
        "affected_query_rate_direct": direct_metrics[
            "affected_query_rate"],
        "affected_answer_slots_direct": direct_metrics[
            "affected_answer_slots"],
        "answer_slots_direct": direct_metrics["answer_slots"],
        "masked_answer_slot_rate_direct": direct_metrics[
            "masked_answer_slot_rate"],
        "query_rows": query_rows,
        "search_rows": search_rows,
        "query_source_corpus": query_spec.meta.get(
            "source_corpus", query_spec.name),
        "mask_digest": _digest_array(np.packbits(mask)),
        "workload_digest": workload_digest(queries),
        "workload_counts": workload_counts(queries),
        "matched_arity": selectivity_matched_arity_summary(queries),
        "incomplete_tuple_rate": float(mask.any(axis=1).mean()),
        "masked_cell_rate": float(mask.mean()),
        "nested_exposure_protocol": nested_exposure_protocol,
        "historical_exposure_diagnostics": historical_exposure_diagnostics,
    }


def _validation_instances(context, per_query, n_folds):
    """Sample labeled UNKNOWN pairs; repeated target tuples stay in one fold."""
    query_groups = sorted({int(query["src"]) for query in context["queries"]})
    rng = np.random.default_rng(1729)
    instances = []
    target_sets = set()
    selected_by_query = []
    for query_index, query in enumerate(context["queries"]):
        states = query["phi"].eval_table(context["observed"])
        unresolved = np.flatnonzero(states == UNKNOWN)
        # Section 3.2 fixes pseudo-missing instances before validation labels
        # are examined.  Sampling a label-balanced subset would change the
        # workload class prior and bias both Brier selection and isotonic maps.
        # Draw uniformly from unresolved instances; consult hidden outcomes
        # only after the rows have been fixed.
        rng.shuffle(unresolved)
        chosen = unresolved[:int(per_query)]
        for row in chosen:
            residual = tuple(sorted(
                atom.attr for atom in query["phi"].atoms
                if pd.isna(context["observed"][atom.attr].iat[int(row)])))
            if not residual:
                continue
            target_sets.add(residual)
            instances.append(ValidationInstance(
                int(row), query["phi"],
                int(context["truth_elig"][query_index, int(row)]),
                int(row) % int(n_folds)))
        selected_by_query.append(int(len(chosen)))
    if not instances:
        raise ValueError("validation workload produced no UNKNOWN tuple-predicate pairs")
    manifest = {
        "instances": int(len(instances)),
        "query_source_groups": int(len(query_groups)),
        "folds_requested": int(n_folds),
        "folds_realized": sorted({int(item.fold) for item in instances}),
        "per_query_min": int(min(selected_by_query, default=0)),
        "per_query_max": int(max(selected_by_query, default=0)),
        "target_sets": [list(item) for item in sorted(target_sets)],
        "labels_digest": _digest_array(
            np.asarray([item.label for item in instances], dtype=np.int8)),
        "instance_sampling": "uniform_unresolved_without_label_stratification",
        "selection_labels_used": False,
        "selection_source": "validation_only",
        "fold_unit": "target_tuple_id",
        "entity_group_fold_available": False,
    }
    return instances, sorted(target_sets), manifest


def _ann_backend(kind, embeddings, seed):
    if kind == "exact":
        return ExactBackend().build(embeddings)
    if kind == "hnsw":
        return HNSWBackend(seed=seed).build(embeddings)
    raise ValueError(f"unknown ANN backend: {kind}")


def _books_derived_scale_snapshot(
        embeddings, table, target_n, *, dimension, seed, noise_std,
        jitter_reference_embeddings=None):
    """Deterministically expand one real searchable snapshot for scale timing.

    The expanded relation is a systems benchmark, not a new effectiveness
    dataset. Rows retain their real Books attributes while fixed seeded iid
    jitter prevents a pathological all-duplicate vector index. The exact same
    helper is used by the external-baseline transport.
    """
    expanded, _, source_rows = expand_repeated_jitter_embeddings(
        embeddings, target_n, dimension, noise_std, seed,
        jitter_reference_values=jitter_reference_embeddings,
    )
    expanded_table = table.iloc[source_rows].reset_index(drop=True)
    return expanded, expanded_table, source_rows


def _prepare_corpus_sharded_distances(
        provider_slices, queries, *, batch_size):
    """Prepare complete rows with contiguous corpus shards on multiple GPUs."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    provider_slices = tuple(provider_slices)
    if not provider_slices:
        raise ValueError("at least one GPU distance provider is required")
    if len(provider_slices) == 1:
        provider, start, end = provider_slices[0]
        if int(start) != 0 or int(end) != provider.base_rows:
            raise ValueError("the single-GPU provider must cover the full corpus")
        return provider.prepare_batch(
            queries, batch_size=int(batch_size))
    output = np.empty(
        (len(queries), max(int(end) for _, _, end in provider_slices)),
        dtype=np.float32)

    def prepare(item):
        provider, start, end = item
        return start, end, provider.prepare_batch(
            queries, batch_size=int(batch_size))

    with ThreadPoolExecutor(max_workers=len(provider_slices)) as pool:
        for start, end, values in pool.map(prepare, provider_slices):
            output[:, int(start):int(end)] = values
    return output


def _prepare_query_sharded_distances(
        replicated_providers, queries, *, batch_size):
    """Prepare complete rows with a replicated corpus and query sharding."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    providers = tuple(item[0] for item in replicated_providers)
    if not providers:
        raise ValueError("at least one replicated GPU provider is required")
    if any(provider.base_rows != providers[0].base_rows
           for provider in providers):
        raise ValueError("query-sharded providers must replicate one corpus")
    output = np.empty(
        (len(queries), providers[0].base_rows), dtype=np.float32)
    row_groups = [
        rows for rows in np.array_split(
            np.arange(len(queries), dtype=np.int64), len(providers))
        if len(rows)
    ]

    def prepare(item):
        provider, rows = item
        return rows, provider.prepare_batch(
            queries[rows], batch_size=int(batch_size))

    with ThreadPoolExecutor(max_workers=len(row_groups)) as pool:
        for rows, values in pool.map(prepare, zip(providers, row_groups)):
            output[rows] = values
    return output


def _measure_candidate_response_batch1(
        model, queries, ks, predicates, provider, reference_ids, *,
        query_count, repeats, initial_page_size, max_page_size,
        force_generic_cpu=False, recompute_single_shape_reference=True,
        strict_ids=True, exact_prewarm_strategy="group_first"):
    """Measure the optimized, exact, one-request response path.

    This shared implementation is used by both the controlled 1M benchmark
    and the real-workload retiming closeout so their P50/P95 values have the
    same physical scope and aggregation rule.
    """
    response_count = min(int(query_count), len(queries))
    repeats = int(repeats)
    if response_count <= 0 or repeats <= 0:
        raise ValueError("response query count/repeats must be positive")
    queries = np.ascontiguousarray(queries[:response_count], dtype=np.float32)
    ks = list(ks[:response_count])
    predicates = list(predicates[:response_count])
    references = [
        np.asarray(row, dtype=np.int64)
        for row in reference_ids[:response_count]
    ]
    provider_manifest = (
        provider.manifest() if hasattr(provider, "manifest") else {})
    provider_device = str(provider_manifest.get("device", "unknown"))
    if force_generic_cpu and provider_device != "cpu":
        raise ValueError("generic CPU response mode requires a CPU provider")
    if force_generic_cpu:
        provider.batch_one_topk_enabled = False
    if exact_prewarm_strategy not in {"group_first", "posting_first"}:
        raise ValueError(
            f"unknown exact prewarm strategy: {exact_prewarm_strategy}")
    model.executor.candidate_prefetch_strategy = exact_prewarm_strategy
    if (provider_device == "cpu" and recompute_single_shape_reference and
            hasattr(provider, "distance_provider")):
        # BLAS may choose a different accumulation/blocking schedule for a
        # QxN GEMM than for the 1xN request measured below.  Near an exact
        # float32 distance tie this can change stable cutoff membership by one
        # ULP.  Audit a single-request timer against a single-request exact
        # reference with the same distance-provider shape, then clear every
        # mutable cache before the measured cold pass.
        references = []
        for query_index in range(response_count):
            one_distances = provider.distance_provider.prepare_batch(
                queries[query_index:query_index + 1])
            one_output = model.executor.search_precomputed_batch(
                queries[query_index:query_index + 1],
                [ks[query_index]], [predicates[query_index]],
                one_distances)
            references.append(np.asarray(
                one_output[0][0], dtype=np.int64))

    # Eligibility materialization belongs to serving construction, not online
    # query latency.  The CPU provider has a second compact-plan prewarm hook,
    # but both CPU and GPU providers must share the same inference prewarm.
    # Previously the GPU branch cleared every inference cache here, so its
    # first measured pass compiled one exact posting per query while the GPU
    # sat idle.  Besides inflating latency, a 1,000-query ArXiv cohort could
    # spend tens of minutes in Python before issuing any device work.
    prewarm_started = time.perf_counter()
    prewarm_plans = []
    for predicate in predicates:
        plan = model.inference.compile_hard_predicate(predicate)
        if plan is None:
            raise RuntimeError(
                "exact cohort prewarm requires compiled predicate support")
        posting = None
        prefetched = None
        if exact_prewarm_strategy == "posting_first" and hasattr(
                model.inference,
                "compile_candidate_eligibility_posting"):
            posting = model.inference.compile_candidate_eligibility_posting(
                predicate, materialize=True)
        if posting is None and hasattr(
                model.inference, "compile_candidate_group_prefetch"):
            prefetched = model.inference.compile_candidate_group_prefetch(
                predicate)
        if (posting is None and prefetched is None and hasattr(
                model.inference,
                "compile_candidate_eligibility_posting")):
            posting = model.inference.compile_candidate_eligibility_posting(
                predicate, materialize=True)
        if posting is not None:
            plan = {
                **plan,
                "precomputed_eligible_ids": posting,
            }
        elif prefetched is not None:
            plan = {
                **plan,
                "statistical_group_ids": prefetched["group_ids"],
                "statistical_group_decisions": prefetched["decisions"],
            }
        prewarm_plans.append(plan)
    provider_prewarm = (
        provider.prewarm_eligibility(prewarm_plans)
        if hasattr(provider, "prewarm_eligibility") else None)
    prewarm_manifest = {
        "excluded_from_online_latency": True,
        "strategy": exact_prewarm_strategy,
        "seconds": float(time.perf_counter() - prewarm_started),
        "provider": provider_prewarm,
        "inference": model.inference.serving_cache_manifest(),
    }
    cold_inference_manifest = model.inference.serving_cache_manifest()
    cold_executor_manifest = (
        model.executor.clear_candidate_stream_plan_cache()
        if hasattr(model.executor, "clear_candidate_stream_plan_cache")
        else None)
    # CPU compact providers retain their prewarmed plan arrays.  GPU
    # providers have no such arrays, so only their transient online state is
    # cleared before the measured cold pass.
    cold_provider_manifest = (
        provider.manifest()
        if hasattr(provider, "prewarm_eligibility") else
        provider.clear_online_caches()
        if hasattr(provider, "clear_online_caches") else None)
    cold_runs = np.zeros(response_count, dtype=np.float64)
    cold_logical_ann_seed = np.zeros(response_count, dtype=np.float64)
    cold_logical_direct = np.zeros(response_count, dtype=np.float64)
    cold_logical_cache = np.zeros(response_count, dtype=np.float64)
    cold_logical_exact_distance = np.zeros(response_count, dtype=np.float64)
    cold_logical_support_inference = np.zeros(
        response_count, dtype=np.float64)
    cold_ids = []
    for query_index in range(response_count):
        started = time.perf_counter()
        cold_output = model.search_candidate_stream_batch(
            queries[query_index:query_index + 1],
            [ks[query_index]], [predicates[query_index]], provider,
            initial_page_size=initial_page_size,
            max_page_size=max_page_size,
            fallback_to_reference=False)
        cold_runs[query_index] = (
            time.perf_counter() - started) * 1000.0
        cold_ids.append(np.asarray(cold_output[0][0], dtype=np.int64))
        cold_counters = cold_output[0][2]
        cold_logical_ann_seed[query_index] = float(cold_counters.get(
            "candidate_stream_logical_ann_seeding_ms", 0.0))
        cold_logical_direct[query_index] = float(cold_counters.get(
            "candidate_stream_logical_direct_checks_ms", 0.0))
        cold_logical_cache[query_index] = float(cold_counters.get(
            "candidate_stream_logical_cache_lookup_ms", 0.0))
        cold_logical_exact_distance[query_index] = float(cold_counters.get(
            "candidate_stream_logical_exact_distance_ms", 0.0))
        cold_logical_support_inference[query_index] = float(
            cold_counters.get(
                "candidate_stream_logical_support_inference_ms",
                cold_counters.get(
                    "candidate_stream_statistical_inference_ms", 0.0)))
    cold_mismatches = [
        index for index, (reference, observed) in enumerate(
            zip(references, cold_ids))
        if not np.array_equal(reference, observed)]

    response_runs = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_d2h = np.zeros(
        (repeats, response_count), dtype=np.int64)
    response_host_state = np.zeros(
        (repeats, response_count), dtype=np.int64)
    response_candidate_events = np.zeros(
        (repeats, response_count), dtype=np.int64)
    response_inference_events = np.zeros(
        (repeats, response_count), dtype=np.int64)
    response_provider_wall = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_device = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_device_d2h_wait = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_cpu_consume = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_distance_stage = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_predicate_stage = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_select_stage = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_distance_rows = np.zeros(
        (repeats, response_count), dtype=np.int64)
    response_eligible_rows = np.zeros(
        (repeats, response_count), dtype=np.int64)
    response_logical_ann_seed = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_logical_direct = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_logical_cache = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_logical_exact_distance = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_logical_support_inference = np.zeros(
        (repeats, response_count), dtype=np.float64)
    response_ids = None
    response_peak_gpu = 0
    response_execution_config = None
    for repeat in range(repeats):
        ids_this_repeat = []
        for query_index in range(response_count):
            started = time.perf_counter()
            output = model.search_candidate_stream_batch(
                queries[query_index:query_index + 1],
                [ks[query_index]], [predicates[query_index]], provider,
                initial_page_size=initial_page_size,
                max_page_size=max_page_size,
                fallback_to_reference=False)
            response_runs[repeat, query_index] = (
                time.perf_counter() - started) * 1000.0
            ids_this_repeat.append(np.asarray(
                output[0][0], dtype=np.int64))
            counters = output[0][2]
            if response_execution_config is None:
                response_execution_config = {
                    "numba_partitions": int(counters.get(
                        "candidate_stream_batch1_numba_partitions", 0)),
                    "refine_width": int(counters.get(
                        "candidate_stream_batch1_refine_width", 0)),
                    "projection_dim": int(counters.get(
                        "candidate_stream_batch1_projection_dim", 0)),
                    "quantized_screen": bool(counters.get(
                        "candidate_stream_batch1_quantized_screen", 0)),
                    "quantized_dtype": str(counters.get(
                        "candidate_stream_batch1_quantized_dtype", "N/A")),
                    "eligibility_cache_bytes_at_first_measurement": int(
                        counters.get(
                            "candidate_stream_batch1_eligibility_cache_bytes",
                            0)),
                    "eligibility_cache_entries_at_first_measurement": int(
                        counters.get(
                            "candidate_stream_batch1_eligibility_cache_entries",
                            0)),
                    "eligibility_cache_hits_at_first_measurement": int(
                        counters.get(
                            "candidate_stream_batch1_eligibility_cache_hits",
                            0)),
                    "eligibility_cache_misses_at_first_measurement": int(
                        counters.get(
                            "candidate_stream_batch1_eligibility_cache_misses",
                            0)),
                    "eligibility_cache_hit_rate_at_first_measurement": float(
                        counters.get(
                            "candidate_stream_batch1_eligibility_cache_hit_rate",
                            0.0)),
                    "eligibility_cache_max_entries": int(counters.get(
                        "candidate_stream_batch1_eligibility_cache_max_entries",
                        0)),
                    "eligibility_cache_max_bytes": int(counters.get(
                        "candidate_stream_batch1_eligibility_cache_max_bytes",
                        0)),
                }
            response_d2h[repeat, query_index] = int(counters.get(
                "candidate_stream_batch_compact_d2h_bytes", 0))
            response_host_state[repeat, query_index] = int(counters.get(
                "candidate_stream_peak_compact_host_page_bytes_batch", 0))
            response_candidate_events[repeat, query_index] = int(counters.get(
                "candidate_stream_candidate_events", 0))
            response_inference_events[repeat, query_index] = int(counters.get(
                "candidate_stream_inference_events", 0))
            response_provider_wall[repeat, query_index] = float(counters.get(
                "candidate_stream_batch1_fused_wall_ms", 0.0))
            response_device[repeat, query_index] = float(counters.get(
                "candidate_stream_batch1_fused_device_ms", 0.0))
            response_device_d2h_wait[repeat, query_index] = float(counters.get(
                "candidate_stream_batch1_device_d2h_wait_ms", 0.0))
            response_cpu_consume[repeat, query_index] = float(counters.get(
                "candidate_stream_cpu_consume_ms", 0.0))
            response_distance_stage[repeat, query_index] = float(
                counters.get(
                    "candidate_stream_batch_distance_compute_ms", 0.0))
            response_predicate_stage[repeat, query_index] = float(
                counters.get(
                    "candidate_stream_batch_hard_predicate_ms", 0.0))
            response_select_stage[repeat, query_index] = float(
                counters.get(
                    "candidate_stream_batch_candidate_select_ms", 0.0))
            response_distance_rows[repeat, query_index] = int(counters.get(
                "physical_distance_evaluations", 0))
            response_eligible_rows[repeat, query_index] = int(counters.get(
                "candidate_stream_batch1_eligible_rows", 0))
            response_logical_ann_seed[repeat, query_index] = float(
                counters.get(
                    "candidate_stream_logical_ann_seeding_ms", 0.0))
            response_logical_direct[repeat, query_index] = float(
                counters.get(
                    "candidate_stream_logical_direct_checks_ms", 0.0))
            response_logical_cache[repeat, query_index] = float(
                counters.get(
                    "candidate_stream_logical_cache_lookup_ms", 0.0))
            response_logical_exact_distance[repeat, query_index] = float(
                counters.get(
                    "candidate_stream_logical_exact_distance_ms", 0.0))
            response_logical_support_inference[repeat, query_index] = float(
                counters.get(
                    "candidate_stream_logical_support_inference_ms",
                    counters.get(
                        "candidate_stream_statistical_inference_ms", 0.0)))
            response_peak_gpu = max(
                response_peak_gpu, int(counters.get(
                    "candidate_stream_peak_gpu_allocated_bytes", 0)))
        if response_ids is None:
            response_ids = ids_this_repeat
        elif any(not np.array_equal(left, right) for left, right in zip(
                response_ids, ids_this_repeat)):
            raise AssertionError(
                f"{provider_device} batch-1 response timing changed result IDs")
    steady_exact = [
        np.array_equal(reference, observed)
        for reference, observed in zip(references, response_ids)]
    set_recalls = []
    set_f1s = []
    for reference, observed in zip(references, response_ids):
        reference_set = set(map(int, reference))
        observed_set = set(map(int, observed))
        overlap = len(reference_set & observed_set)
        recall = overlap / max(1, len(reference_set))
        precision = overlap / max(1, len(observed_set))
        set_recalls.append(recall)
        set_f1s.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0)

    response_values = np.median(response_runs, axis=0)
    logical_ann = np.median(response_logical_ann_seed, axis=0)
    logical_direct = np.median(response_logical_direct, axis=0)
    logical_cache = np.median(response_logical_cache, axis=0)
    logical_exact_distance = np.median(
        response_logical_exact_distance, axis=0)
    logical_support_inference = np.median(
        response_logical_support_inference, axis=0)
    logical_node_queue = np.maximum(
        response_values - logical_ann - logical_direct - logical_cache -
        logical_exact_distance - logical_support_inference,
        np.float64(0.0))
    logical_component_means = {
        "candidate_seeding_ms": float(logical_ann.mean()),
        "node_queue_operations_ms": float(logical_node_queue.mean()),
        "exact_distance_evaluation_ms": float(
            logical_exact_distance.mean()),
        "direct_deterministic_checks_ms": float(logical_direct.mean()),
        "support_retrieval_joint_inference_ms": float(
            logical_support_inference.mean()),
        "cache_lookup_ms": float(logical_cache.mean()),
    }
    logical_accounted_total = float(sum(logical_component_means.values()))
    cold_logical_node_queue = np.maximum(
        cold_runs - cold_logical_ann_seed - cold_logical_direct -
        cold_logical_cache - cold_logical_exact_distance -
        cold_logical_support_inference, np.float64(0.0))
    cold_logical_component_means = {
        "candidate_seeding_ms": float(cold_logical_ann_seed.mean()),
        "node_queue_operations_ms": float(cold_logical_node_queue.mean()),
        "exact_distance_evaluation_ms": float(
            cold_logical_exact_distance.mean()),
        "direct_deterministic_checks_ms": float(
            cold_logical_direct.mean()),
        "support_retrieval_joint_inference_ms": float(
            cold_logical_support_inference.mean()),
        "cache_lookup_ms": float(cold_logical_cache.mean()),
    }
    cold_logical_accounted_total = float(sum(
        cold_logical_component_means.values()))
    timing_scope = (
        "one-query CPU eligibility-first exact candidate generation + "
        "CPU exact fixed-policy consumption/final top-k"
        if provider_device == "cpu" else
        "one-query GPU candidate generation + compact D2H + "
        "CPU exact fixed-policy consumption/final top-k")
    steady_inference_cache_manifest = \
        model.inference.serving_cache_manifest()
    return {
        "status": "steady_single_request_latency",
        "platform": provider_device,
        "query_count": int(response_count),
        "repeats": int(repeats),
        "cache_regime": (
            "one empty-online-cache fixed-order cohort pass, then repeated "
            "steady passes"),
        "timing_scope": timing_scope,
        "timing_scope_contract": timing_scope,
        "numeric_reference_scope": (
            "single-request FP32 exact distance/reference with the same 1xN "
            "provider shape" if (
                provider_device == "cpu" and
                recompute_single_shape_reference) else
            "frozen backend exact reference supplied by the matrix runner"),
        "strict_ordered_id_gate": False,
        "result_set_recall_vs_exact_mean": float(np.mean(set_recalls)),
        "result_set_recall_vs_exact_p05": float(
            np.percentile(set_recalls, 5)),
        "result_set_f1_vs_exact_mean": float(np.mean(set_f1s)),
        "result_set_f1_delta_vs_exact": float(np.mean(set_f1s) - 1.0),
        "execution_config": response_execution_config,
        "offline_serving_prewarm": prewarm_manifest,
        "steady_inference_cache_manifest":
            steady_inference_cache_manifest,
        "latency_ms_mean": float(response_values.mean()),
        "latency_ms_p50": float(np.percentile(response_values, 50)),
        "latency_ms_p95": float(np.percentile(response_values, 95)),
        "qps_from_mean": float(1000.0 / response_values.mean()),
        "logical_runtime_breakdown": {
            "scope": (
                "mean of per-query steady medians; six non-overlapping "
                "logical E2-1 components"),
            "components": logical_component_means,
            "total_ms": float(response_values.mean()),
            "accounted_total_ms": logical_accounted_total,
            "accounting_error_ms": float(
                response_values.mean() - logical_accounted_total),
            "node_queue_scope": (
                "executor/page orchestration and final top-k response "
                "consumption; the optimized path performs zero tree/tuple "
                "queue pops"),
            "exact_distance_scope": (
                "stable top-k selection after complete-dimensional FP32 "
                "posting distance evaluation"),
            "candidate_seeding_scope": (
                "complete-dimensional FP32 eligible-posting SGEMV (or the "
                "explicit resident full-SGEMV layout above its crossover); "
                "this CPU path does not invoke an external ANN index"),
        },
        "cold_cohort": {
            "scope": (
                "one unmeasured validation pass after the explicitly "
                "reported offline cohort prewarm"
                if prewarm_manifest is not None else
                "one fixed-order pass after clearing inference, compiled-"
                "plan, and provider eligibility caches; immutable index, "
                "policy, quantized corpus, and JIT code remain deployed"),
            "inference_cache_manifest_before": cold_inference_manifest,
            "executor_cache_manifest_before": cold_executor_manifest,
            "provider_cache_manifest_before": cold_provider_manifest,
            "latency_ms_mean": float(cold_runs.mean()),
            "latency_ms_p50": float(np.percentile(cold_runs, 50)),
            "latency_ms_p95": float(np.percentile(cold_runs, 95)),
            "qps_from_mean": float(1000.0 / cold_runs.mean()),
            "components": cold_logical_component_means,
            "total_ms": float(cold_runs.mean()),
            "accounted_total_ms": cold_logical_accounted_total,
            "accounting_error_ms": float(
                cold_runs.mean() - cold_logical_accounted_total),
            "per_query_ms": cold_runs.tolist(),
            "ordered_exact_queries": int(
                response_count - len(cold_mismatches)),
            "ordered_exact": bool(not cold_mismatches),
        },
        "per_query_median_ms": response_values.tolist(),
        "per_query_repeat_ms": response_runs.tolist(),
        "compact_d2h_bytes_per_query_mean": float(
            np.median(response_d2h, axis=0).mean()),
        "compact_d2h_bytes_per_query": np.median(
            response_d2h, axis=0).astype(np.int64).tolist(),
        "explicit_host_state_bytes_per_query_mean": float(
            np.median(response_host_state, axis=0).mean()),
        "explicit_host_state_bytes_per_query": np.median(
            response_host_state, axis=0).astype(np.int64).tolist(),
        "candidate_events_per_query_mean": float(
            np.median(response_candidate_events, axis=0).mean()),
        "statistical_inference_events_per_query_mean": float(
            np.median(response_inference_events, axis=0).mean()),
        "provider_fused_wall_ms_per_query_mean": float(
            np.median(response_provider_wall, axis=0).mean()),
        "provider_fused_device_ms_per_query_mean": float(
            np.median(response_device, axis=0).mean()),
        "provider_device_d2h_wait_ms_per_query_mean": float(
            np.median(response_device_d2h_wait, axis=0).mean()),
        "cpu_consume_ms_per_query_mean": float(
            np.median(response_cpu_consume, axis=0).mean()),
        "distance_stage_ms_per_query_mean": float(
            np.median(response_distance_stage, axis=0).mean()),
        "predicate_stage_ms_per_query_mean": float(
            np.median(response_predicate_stage, axis=0).mean()),
        "candidate_select_ms_per_query_mean": float(
            np.median(response_select_stage, axis=0).mean()),
        "distance_stage_ms_per_query": np.median(
            response_distance_stage, axis=0).tolist(),
        "predicate_stage_ms_per_query": np.median(
            response_predicate_stage, axis=0).tolist(),
        "candidate_select_ms_per_query": np.median(
            response_select_stage, axis=0).tolist(),
        "distance_rows_per_query_mean": float(
            np.median(response_distance_rows, axis=0).mean()),
        "eligible_rows_per_query_mean": float(
            np.median(response_eligible_rows, axis=0).mean()),
        "distance_rows_per_query": np.median(
            response_distance_rows, axis=0).astype(np.int64).tolist(),
        "eligible_rows_per_query": np.median(
            response_eligible_rows, axis=0).astype(np.int64).tolist(),
        "ordered_exact_queries": int(sum(steady_exact)),
        "ordered_exact": bool(all(steady_exact)),
        "peak_gpu_allocated_bytes": int(response_peak_gpu),
    }


def _measure_replicated_candidate_throughput(
        model, queries, ks, predicates, provider, reference_ids, *,
        query_count, repeats, initial_page_size, max_page_size):
    """Measure aggregate QPS from concurrent full-corpus GPU replicas.

    Batch-one P50/P95 remains an unloaded per-request measurement.  This
    companion pass assigns disjoint query streams to full-corpus replicas and
    starts them at one barrier, so two-GPU throughput is observed rather than
    inferred from a serial round-robin loop.
    """
    if not isinstance(provider, TorchReplicatedExactCandidateStreamProvider):
        raise TypeError("aggregate replica throughput requires replicas")
    response_count = min(int(query_count), len(queries))
    repeats = int(repeats)
    if response_count <= 0 or repeats <= 0:
        raise ValueError("throughput query count/repeats must be positive")
    replicas = tuple(provider.providers)
    groups = [
        np.arange(index, response_count, len(replicas), dtype=np.int64)
        for index in range(len(replicas))]
    groups = [group for group in groups if len(group)]
    replicas = replicas[:len(groups)]
    queries = np.ascontiguousarray(
        queries[:response_count], dtype=np.float32)
    ks = list(ks[:response_count])
    predicates = list(predicates[:response_count])
    references = [
        np.asarray(row, dtype=np.int64)
        for row in reference_ids[:response_count]]

    # Populate immutable support/predicate and device-local eligibility caches
    # on the replica that owns each measured query stream.
    for replica, group in zip(replicas, groups):
        for query_index in group:
            output = model.search_candidate_stream_batch(
                queries[query_index:query_index + 1],
                [ks[query_index]], [predicates[query_index]], replica,
                initial_page_size=initial_page_size,
                max_page_size=max_page_size,
                fallback_to_reference=False)
            observed = np.asarray(output[0][0], dtype=np.int64)
            if not np.array_equal(references[query_index], observed):
                raise AssertionError(
                    "replica throughput warm-up disagrees with reference "
                    f"at query {int(query_index)}")

    repeat_rows = []
    all_latencies = []
    for repeat in range(repeats):
        barrier = threading.Barrier(len(groups))

        def run_stream(item):
            replica, group = item
            barrier.wait()
            stream_started = time.perf_counter()
            local_latencies = []
            local_ids = []
            for query_index in group:
                started = time.perf_counter()
                output = model.search_candidate_stream_batch(
                    queries[query_index:query_index + 1],
                    [ks[query_index]], [predicates[query_index]], replica,
                    initial_page_size=initial_page_size,
                    max_page_size=max_page_size,
                    fallback_to_reference=False)
                local_latencies.append(
                    (time.perf_counter() - started) * 1000.0)
                local_ids.append((int(query_index), np.asarray(
                    output[0][0], dtype=np.int64)))
            return {
                "wall_ms": (time.perf_counter() - stream_started) * 1000.0,
                "latency_ms": local_latencies,
                "ids": local_ids,
                "queries": int(len(group)),
                "device": str(replica.device),
            }

        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            streams = list(pool.map(
                run_stream, zip(replicas, groups)))
        for stream in streams:
            for query_index, observed in stream.pop("ids"):
                if not np.array_equal(references[query_index], observed):
                    raise AssertionError(
                        "replica throughput result disagrees with reference "
                        f"at query {query_index}")
            all_latencies.extend(stream["latency_ms"])
        wall_ms = max(float(stream["wall_ms"]) for stream in streams)
        repeat_rows.append({
            "repeat": int(repeat),
            "aggregate_wall_ms": wall_ms,
            "aggregate_qps": float(response_count * 1000.0 / wall_ms),
            "streams": streams,
        })
    aggregate_values = np.asarray(
        [row["aggregate_qps"] for row in repeat_rows], dtype=np.float64)
    latency_values = np.asarray(all_latencies, dtype=np.float64)
    return {
        "status": "measured_concurrent_replica_throughput",
        "query_count": int(response_count),
        "repeats": int(repeats),
        "replica_count": int(len(replicas)),
        "dispatch": "disjoint_query_stream_per_gpu_replica",
        "process_model": "one_process_one_worker_thread_per_gpu",
        "aggregate_qps_median": float(np.median(aggregate_values)),
        "aggregate_qps_values": aggregate_values.tolist(),
        "concurrent_latency_ms_p50": float(np.percentile(
            latency_values, 50)),
        "concurrent_latency_ms_p95": float(np.percentile(
            latency_values, 95)),
        "repeat_rows": repeat_rows,
        "ordered_exact": True,
    }


def _measure_candidate_saturated_throughput(
        model, queries, ks, predicates, provider, *, query_count, repeats,
        initial_page_size, max_page_size):
    """Measure full-cohort saturated QPS separately from batch-one latency."""
    response_count = min(int(query_count), len(queries))
    repeats = int(repeats)
    if response_count <= 0 or repeats <= 0:
        raise ValueError("throughput query count/repeats must be positive")
    batch_queries = np.ascontiguousarray(
        queries[:response_count], dtype=np.float32)
    batch_ks = list(ks[:response_count])
    batch_predicates = list(predicates[:response_count])

    # One untimed batch populates immutable serving caches.  The online
    # latency columns continue to come exclusively from the batch-one pass.
    model.search_candidate_stream_batch(
        batch_queries, batch_ks, batch_predicates, provider,
        initial_page_size=initial_page_size,
        max_page_size=max_page_size,
        fallback_to_reference=False)
    timing_ms = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.search_candidate_stream_batch(
            batch_queries, batch_ks, batch_predicates, provider,
            initial_page_size=initial_page_size,
            max_page_size=max_page_size,
            fallback_to_reference=False)
        timing_ms.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(timing_ms, dtype=np.float64)
    qps_values = response_count * 1000.0 / values
    return {
        "status": "measured_saturated_full_cohort_throughput",
        "query_count": int(response_count),
        "repeats": int(repeats),
        "batch_size": int(response_count),
        "timing_scope": (
            "whole exact candidate-stream cohort batch including compact "
            "D2H, fixed-policy consumption, and final top-k"),
        "wall_ms_values": values.tolist(),
        "aggregate_qps_values": qps_values.tolist(),
        "aggregate_qps_median": float(np.median(qps_values)),
    }


def _wait_for_external_replica_barrier(phase):
    """Synchronize independently launched one-GPU serving processes.

    The optional barrier is excluded from every reported timer.  It is used
    only by the two-process throughput harness so both isolated Python/CUDA
    runtimes enter their steady candidate-stream repeats together.
    """
    barrier_dir = os.environ.get("MIFA_SCALE_REPLICA_BARRIER_DIR")
    if not barrier_dir:
        return None
    role = os.environ.get("MIFA_SCALE_REPLICA_ROLE")
    participants = int(os.environ.get("MIFA_SCALE_REPLICA_PARTICIPANTS", "2"))
    timeout_seconds = float(os.environ.get(
        "MIFA_SCALE_REPLICA_BARRIER_TIMEOUT_SECONDS", "300"))
    if not role or participants < 2 or timeout_seconds <= 0.0:
        raise ValueError("invalid external replica barrier configuration")
    root = Path(barrier_dir)
    root.mkdir(parents=True, exist_ok=True)
    ready = root / f"{phase}.{role}.ready"
    ready.write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    pattern = f"{phase}.*.ready"
    while len(tuple(root.glob(pattern))) < participants:
        if time.perf_counter() - started > timeout_seconds:
            raise TimeoutError(
                f"external replica barrier timed out: {phase}/{role}")
        time.sleep(0.01)
    return {
        "phase": phase,
        "role": role,
        "participants": participants,
        "wait_ms_excluded_from_timing": float(
            (time.perf_counter() - started) * 1000.0),
    }


def _scale_searchable_handler(
        context, dtypes, *, handler, repaired_table, repair_receipt):
    """Install an optional frozen pre-imputation before scale expansion.

    HoloClean may leave cells unresolved.  Those residual missing values stay
    missing and are handled by MIFA's native online policy; the receipt makes
    this sequential semantics explicit instead of silently treating a partial
    repair as a complete relation.
    """
    if handler == "native":
        if repaired_table or repair_receipt:
            raise ValueError(
                "scale repair paths require --scale-searchable-handler "
                "holoclean")
        return context, {
            "kind": "mifa_native_online_eligibility",
            "pre_imputation": False,
        }
    if handler != "holoclean":
        raise ValueError(f"unknown scale searchable handler: {handler}")
    if not repaired_table or not repair_receipt:
        raise ValueError(
            "HoloClean scale serving requires repaired table and receipt")
    table_path = (ROOT / repaired_table).resolve()
    receipt_path = (ROOT / repair_receipt).resolve()
    try:
        table_relative = table_path.relative_to(ROOT)
        receipt_relative = receipt_path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("scale repair artifacts must be inside the repo") from error
    repaired = pd.read_pickle(table_path)
    observed = context["observed"]
    if (len(repaired) != len(observed) or
            list(repaired.columns) != list(observed.columns)):
        raise RuntimeError("HoloClean scale table is not row/column aligned")
    for column, kind in dtypes.items():
        if kind == "num" and not pd.api.types.is_numeric_dtype(
                repaired[column]):
            repaired[column] = pd.to_numeric(repaired[column], errors="raise")
    receipt = json.loads(receipt_path.read_text())
    actual_sha256 = hashlib.sha256(table_path.read_bytes()).hexdigest()
    expected_sha256 = receipt.get("transport", {}).get(
        "repaired_table_sha256")
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise RuntimeError("HoloClean scale table SHA-256 drifted")
    current_mask = _context_manifest(context)["mask_digest"]
    receipt_mask = receipt.get("test", {}).get("mask_digest")
    if current_mask != receipt_mask:
        raise RuntimeError("HoloClean/current scale mask digest mismatch")
    observed_missing = observed.isna()
    repaired_missing = repaired.isna()
    filled = int((observed_missing & ~repaired_missing).to_numpy().sum())
    remaining = int(repaired_missing.to_numpy().sum())
    updated = dict(context)
    updated["observed"] = repaired
    return updated, {
        "kind": "holoclean_pre_imputation_then_mifa_residual_inference",
        "pre_imputation": True,
        "materialized_table": str(table_relative),
        "materialized_table_sha256": actual_sha256,
        "repair_receipt": str(receipt_relative),
        "repair_receipt_sha256": hashlib.sha256(
            receipt_path.read_bytes()).hexdigest(),
        "mask_digest": current_mask,
        "input_missing_cells": int(observed_missing.to_numpy().sum()),
        "filled_cells": filled,
        "remaining_missing_cells": remaining,
        "residual_policy": "MIFA native online eligibility inference",
        "repair_fit_materialization_seconds": receipt.get(
            "repair_resources", {}).get("fit_materialization_seconds"),
    }


def _run_candidate_efficiency_matrix(
        model, searchable_context, reference_ids, *, query_count,
        gpu_devices, repeats, initial_page_size=4096,
        max_page_size=65536, response_strata=(),
        response_generic_cpu=False, response_only=False,
        response_recompute_cpu_reference=True,
        platforms=("cpu", "1gpu", "2gpu"),
        cpu_projection_dim=16, cpu_refine_width=64,
        cpu_approximate_shortlist=False,
        exact_prewarm_strategy="group_first",
        cpu_fused_exact_min_work=0,
        gpu_resident_exact_postings=False,
        response_saturated_throughput=False):
    """Measure one candidate consumer on CPU, one GPU, and two GPUs."""
    devices = tuple(item.strip() for item in gpu_devices if item.strip())
    platforms = tuple(str(item).strip().lower() for item in platforms
                      if str(item).strip())
    unknown_platforms = set(platforms) - {"cpu", "1gpu", "2gpu"}
    if unknown_platforms:
        raise ValueError(
            f"unknown candidate efficiency platforms: "
            f"{sorted(unknown_platforms)}")
    if not platforms:
        raise ValueError("at least one candidate efficiency platform required")
    query_count = min(int(query_count), len(searchable_context["queries"]))
    repeats = int(repeats)
    if query_count <= 0 or repeats <= 0:
        raise ValueError("candidate efficiency query count/repeats must be positive")
    cohort = searchable_context["queries"][:query_count]
    queries = np.ascontiguousarray(
        [query["q_e"] for query in cohort], dtype=np.float32)
    ks = [int(query["k"]) for query in cohort]
    predicates = [query["phi"] for query in cohort]
    expected = []
    for row in reference_ids[:query_count]:
        values = np.asarray(row, dtype=np.int64)
        # ``_run_search`` pads fewer-than-k answers with -1 for scoring;
        # candidate streams return the true variable-length answer.
        expected.append(np.ascontiguousarray(values[values >= 0]))
    embeddings = searchable_context["embeddings"]
    full_distance_bytes = int(
        query_count * len(embeddings) * np.dtype(np.float32).itemsize)

    def build_cpu():
        return NumpyExactCandidateStreamProvider(
            embeddings,
            batch_one_cache_max_entries=max(64, query_count),
            batch_one_projection_dim=min(
                int(cpu_projection_dim), int(embeddings.shape[1])),
            batch_one_refine_width=max(int(cpu_refine_width), max(ks)),
            batch_one_quantized=bool(cpu_approximate_shortlist),
            batch_one_approximate_shortlist=bool(
                cpu_approximate_shortlist),
            batch_one_full_scan_switch_fraction=0.08,
            batch_one_fused_exact_min_work=int(
                cpu_fused_exact_min_work))

    configurations = []
    if "cpu" in platforms:
        configurations.append(("cpu", 0, ["cpu"], build_cpu))
    if "1gpu" in platforms and devices:
        configurations.append(("gpu", 1, [devices[0]], lambda: (
            TorchExactCandidateStreamProvider(
                embeddings, device=devices[0],
                resident_exact_postings=bool(
                    gpu_resident_exact_postings)))))
    if "2gpu" in platforms and len(devices) > 1:
        def build_multi():
            if response_only and len(embeddings) <= 262144:
                def build_replica(index):
                    return TorchExactCandidateStreamProvider(
                        embeddings, device=devices[index],
                        resident_exact_postings=bool(
                            gpu_resident_exact_postings))

                with ThreadPoolExecutor(max_workers=2) as pool:
                    replicas = list(pool.map(build_replica, range(2)))
                return TorchReplicatedExactCandidateStreamProvider(replicas)
            boundaries = np.linspace(
                0, len(embeddings), 3, dtype=np.int64)

            def build_shard(index):
                start = int(boundaries[index])
                end = int(boundaries[index + 1])
                return TorchExactCandidateStreamProvider(
                    embeddings[start:end], device=devices[index],
                    global_offset=start,
                    resident_exact_postings=bool(
                        gpu_resident_exact_postings))

            with ThreadPoolExecutor(max_workers=2) as pool:
                shards = list(pool.map(build_shard, range(2)))
            return TorchShardedExactCandidateStreamProvider(shards)

        configurations.append(("gpu", 2, list(devices[:2]), build_multi))

    rows = []
    for backend, gpu_count, row_devices, builder in configurations:
        gc.collect()
        if gpu_count:
            import torch
            torch.cuda.empty_cache()
        build_started = time.perf_counter()
        provider = builder()
        build_seconds = time.perf_counter() - build_started
        provider_manifest = provider.manifest()

        if response_only:
            response_batch1 = _measure_candidate_response_batch1(
                model, queries, ks, predicates, provider, expected,
                query_count=query_count, repeats=repeats,
                initial_page_size=initial_page_size,
                max_page_size=max_page_size,
                force_generic_cpu=bool(
                    gpu_count == 0 and response_generic_cpu),
                recompute_single_shape_reference=bool(
                    response_recompute_cpu_reference),
                strict_ids=not bool(
                    gpu_count == 0 and cpu_approximate_shortlist),
                exact_prewarm_strategy=exact_prewarm_strategy)
            response_replica_throughput = None
            if isinstance(
                    provider, TorchReplicatedExactCandidateStreamProvider):
                response_replica_throughput = \
                    _measure_replicated_candidate_throughput(
                        model, queries, ks, predicates, provider, expected,
                        query_count=query_count, repeats=repeats,
                        initial_page_size=initial_page_size,
                        max_page_size=max_page_size)
            response_batch_throughput = None
            if bool(response_saturated_throughput) and gpu_count == 1:
                response_batch_throughput = \
                    _measure_candidate_saturated_throughput(
                        model, queries, ks, predicates, provider,
                        query_count=query_count, repeats=repeats,
                        initial_page_size=initial_page_size,
                        max_page_size=max_page_size)
            response_batch1_strata = {}
            for axis in response_strata:
                groups = {}
                for index, query in enumerate(cohort):
                    if axis == "arity":
                        value = str(len(query["phi"].atoms))
                    elif axis == "predicate_type":
                        kinds = {
                            "categorical" if atom.op == "eq" else
                            "numerical" for atom in query["phi"].atoms}
                        value = (next(iter(kinds)) if len(kinds) == 1
                                 else "mixed")
                    elif axis == "k":
                        value = str(query["k"])
                    else:
                        raise ValueError(
                            f"unsupported candidate response stratum: {axis}")
                    groups.setdefault(value, []).append(index)
                response_batch1_strata[axis] = {}
                for value, indices in sorted(groups.items()):
                    selected = np.asarray(indices, dtype=np.int64)
                    response_batch1_strata[axis][value] = \
                        _measure_candidate_response_batch1(
                            model, queries[selected],
                            [ks[index] for index in indices],
                            [predicates[index] for index in indices], provider,
                            [expected[index] for index in indices],
                            query_count=len(indices), repeats=repeats,
                            initial_page_size=initial_page_size,
                            max_page_size=max_page_size,
                            force_generic_cpu=bool(
                                gpu_count == 0 and response_generic_cpu),
                            recompute_single_shape_reference=bool(
                                response_recompute_cpu_reference),
                            strict_ids=not bool(
                                gpu_count == 0 and
                                cpu_approximate_shortlist),
                            exact_prewarm_strategy=exact_prewarm_strategy)
            steady_total_ms = float(
                response_batch1["latency_ms_mean"] * query_count)
            rows.append({
                "backend": backend,
                "gpu_count": int(gpu_count),
                "devices": row_devices,
                "provider": provider_manifest,
                "provider_build_seconds": float(build_seconds),
                "response_only": True,
                "exact_prewarm_strategy": exact_prewarm_strategy,
                "cpu_fused_exact_min_work": int(
                    cpu_fused_exact_min_work),
                "steady_total_ms": steady_total_ms,
                "steady_ms_per_query": float(
                    response_batch1["latency_ms_mean"]),
                "steady_qps": float(
                    response_batch_throughput["aggregate_qps_median"]
                    if response_batch_throughput is not None else
                    response_replica_throughput["aggregate_qps_median"]
                    if response_replica_throughput is not None else
                    response_batch1["qps_from_mean"]),
                "ordered_exact_queries": int(
                    response_batch1["ordered_exact_queries"]),
                "ordered_exact": bool(response_batch1["ordered_exact"]),
                "cold_ordered_exact_queries": int(
                    response_batch1["cold_cohort"][
                        "ordered_exact_queries"]),
                "response_batch1": response_batch1,
                "response_replica_throughput":
                response_replica_throughput,
                "response_batch_throughput": response_batch_throughput,
                "response_batch1_strata": response_batch1_strata,
                "fallback_queries": 0,
            })
            provider = None
            gc.collect()
            if gpu_count:
                torch.cuda.empty_cache()
            continue

        backend_expected = expected
        if gpu_count == 0:
            # CPU and CUDA SGEMM may differ by a few float32 ULPs because of
            # FMA/accumulation order.  Audit each physical distance backend
            # against the unchanged leaf reference over that backend's exact
            # rows, and report cross-backend agreement separately.
            cpu_reference_distances = \
                provider.distance_provider.prepare_batch(queries)
            cpu_reference_outputs = model.executor.search_precomputed_batch(
                queries, ks, predicates, cpu_reference_distances)
            backend_expected = [
                np.asarray(row[0], dtype=np.int64)
                for row in cpu_reference_outputs]
            cpu_reference_distances = None
            cpu_reference_outputs = None
            gc.collect()

        cold_cache_manifest = model.inference.clear_all_serving_caches()
        cold_started = time.perf_counter()
        cold_outputs = model.search_candidate_stream_batch(
            queries, ks, predicates, provider,
            initial_page_size=initial_page_size,
            max_page_size=max_page_size,
            fallback_to_reference=False)
        cold_ms = (time.perf_counter() - cold_started) * 1000.0
        cold_ids = [np.asarray(row[0], dtype=np.int64)
                    for row in cold_outputs]
        timings = []
        stages = []
        outputs = None
        for _ in range(repeats):
            started = time.perf_counter()
            outputs = model.search_candidate_stream_batch(
                queries, ks, predicates, provider,
                initial_page_size=initial_page_size,
                max_page_size=max_page_size,
                fallback_to_reference=False)
            timings.append((time.perf_counter() - started) * 1000.0)
            counters = outputs[0][2]
            stages.append({
                "distance_compute_ms": float(counters[
                    "candidate_stream_batch_distance_compute_ms"]),
                "hard_predicate_ms": float(counters[
                    "candidate_stream_batch_hard_predicate_ms"]),
                "candidate_select_ms": float(counters[
                    "candidate_stream_batch_candidate_select_ms"]),
                "compact_d2h_ms": float(counters[
                    "candidate_stream_batch_compact_d2h_ms"]),
                "cpu_merge_ms": float(counters[
                    "candidate_stream_batch_cpu_merge_ms"]),
                "cpu_consume_ms": float(sum(
                    row[2]["candidate_stream_cpu_consume_ms"]
                    for row in outputs)),
                "statistical_inference_ms": float(sum(
                    row[2]["candidate_stream_statistical_inference_ms"]
                    for row in outputs)),
            })
        median_ms = float(np.median(timings))
        median_index = int(np.argsort(timings)[len(timings) // 2])
        median_stages = stages[median_index]
        output_ids = [np.asarray(row[0], dtype=np.int64)
                      for row in outputs]
        exact = [bool(np.array_equal(left, right))
                 for left, right in zip(backend_expected, output_ids)]
        cold_exact = [bool(np.array_equal(left, right))
                      for left, right in zip(backend_expected, cold_ids)]
        cross_backend_agreement = [
            bool(np.array_equal(left, right))
            for left, right in zip(expected, output_ids)]
        if not all(exact) or not all(cold_exact):
            raise AssertionError(
                f"{backend}/{gpu_count} candidate efficiency IDs disagree")
        counters = [row[2] for row in outputs]
        compact_d2h_bytes = int(counters[0][
            "candidate_stream_batch_compact_d2h_bytes"])
        fallback_queries = int(sum(
            row["candidate_stream_fallback"] for row in counters))
        cpu_ms = float(
            median_stages["cpu_merge_ms"] +
            median_stages["cpu_consume_ms"])
        response_batch1 = None
        response_batch1_strata = {}
        if gpu_count in (0, 1):
            response_batch1 = _measure_candidate_response_batch1(
                model, queries, ks, predicates, provider,
                backend_expected, query_count=query_count, repeats=repeats,
                initial_page_size=initial_page_size,
                max_page_size=max_page_size,
                force_generic_cpu=bool(
                    gpu_count == 0 and response_generic_cpu))
            for axis in response_strata:
                groups = {}
                for index, query in enumerate(cohort):
                    if axis == "arity":
                        value = str(len(query["phi"].atoms))
                    elif axis == "predicate_type":
                        kinds = {
                            "categorical" if atom.op == "eq" else
                            "numerical" for atom in query["phi"].atoms}
                        value = (next(iter(kinds)) if len(kinds) == 1
                                 else "mixed")
                    elif axis == "k":
                        value = str(query["k"])
                    else:
                        raise ValueError(
                            f"unsupported candidate response stratum: {axis}")
                    groups.setdefault(value, []).append(index)
                response_batch1_strata[axis] = {}
                for value, indices in sorted(groups.items()):
                    selected = np.asarray(indices, dtype=np.int64)
                    response_batch1_strata[axis][value] = \
                        _measure_candidate_response_batch1(
                            model, queries[selected],
                            [ks[index] for index in indices],
                            [predicates[index] for index in indices], provider,
                            [backend_expected[index] for index in indices],
                            query_count=len(indices), repeats=repeats,
                            initial_page_size=initial_page_size,
                            max_page_size=max_page_size,
                            force_generic_cpu=bool(
                                gpu_count == 0 and response_generic_cpu))
        rows.append({
            "backend": backend,
            "gpu_count": int(gpu_count),
            "devices": row_devices,
            "provider": provider_manifest,
            "provider_build_seconds": float(build_seconds),
            "cold_cache_manifest_before": cold_cache_manifest,
            "cache_regime": "cold-empty then repeated-predicate steady",
            "cold_total_ms": float(cold_ms),
            "steady_total_ms_repeats": [float(value) for value in timings],
            "steady_total_ms": median_ms,
            "steady_ms_per_query": float(median_ms / query_count),
            "steady_qps": float(1000.0 * query_count / median_ms),
            "ordered_exact_queries": int(sum(exact)),
            "ordered_exact": bool(all(exact)),
            "cold_ordered_exact_queries": int(sum(cold_exact)),
            "frozen_cuda_reference_agreement_queries": int(sum(
                cross_backend_agreement)),
            "frozen_cuda_reference_agreement": bool(all(
                cross_backend_agreement)),
            "numeric_audit_note": (
                "CPU is audited against CPU-SGEMM leaf reference; GPU rows "
                "are audited against the frozen CUDA-SGEMM leaf reference."),
            "output_digest": _sha_json([row.tolist() for row in output_ids]),
            "median_repeat_stages": median_stages,
            "cpu_merge_plus_inference_fraction": float(cpu_ms / median_ms),
            "candidate_events_mean": float(np.mean([
                row["candidate_stream_candidate_events"]
                for row in counters])),
            "inference_events_mean": float(np.mean([
                row["candidate_stream_inference_events"]
                for row in counters])),
            "compact_d2h_bytes": compact_d2h_bytes,
            "full_distance_reference_bytes": full_distance_bytes,
            "d2h_reduction_x": (
                float(full_distance_bytes / compact_d2h_bytes)
                if compact_d2h_bytes else None),
            "peak_provider_host_bytes": int(counters[0].get(
                "candidate_stream_peak_provider_host_bytes", 0)),
            "peak_compact_host_page_bytes": int(counters[0][
                "candidate_stream_peak_compact_host_page_bytes_batch"]),
            "peak_gpu_allocated_bytes": int(counters[0][
                "candidate_stream_peak_gpu_allocated_bytes"]),
            "response_batch1": response_batch1,
            "response_batch1_strata": response_batch1_strata,
            "fallback_queries": fallback_queries,
        })
        provider = None
        cold_outputs = None
        outputs = None
        gc.collect()
        if gpu_count:
            torch.cuda.empty_cache()

    cpu_ms = rows[0]["steady_total_ms"]
    for row in rows:
        row["speedup_vs_candidate_cpu_x"] = float(
            cpu_ms / row["steady_total_ms"])
    one_gpu = next((row for row in rows if row["gpu_count"] == 1), None)
    two_gpu = next((row for row in rows if row["gpu_count"] == 2), None)
    return {
        "status": (
            "candidate_stream_efficiency_matrix_with_explicit_"
            "approximate_cpu_shortlist"
            if cpu_approximate_shortlist else
            "exact_candidate_stream_efficiency_matrix"),
        "query_count": int(query_count),
        "corpus_rows": int(len(embeddings)),
        "dimension": int(embeddings.shape[1]),
        "page_policy": {
            "initial": int(initial_page_size), "growth": "double",
            "maximum": int(max_page_size),
            "selection_source": "global_fixed_not_dataset_tuned",
        },
        "rows": rows,
        "one_to_two_gpu_speedup_x": (
            float(one_gpu["steady_total_ms"] / two_gpu["steady_total_ms"])
            if one_gpu is not None and two_gpu is not None else None),
        "all_ordered_exact": bool(all(
            row["ordered_exact"] for row in rows)),
        "all_zero_fallbacks": bool(all(
            row["fallback_queries"] == 0 for row in rows)),
    }


def _run_scale_benchmark(
        *, training_embeddings, training_table, searchable_context, dtypes,
        policy, theta, structural_config, seed, target_n, query_count,
        gpu_devices, distance_batch_size, repeats, noise_std,
        dimension, expansion_seed, query_rows=None,
        jitter_reference_embeddings=None,
        executor_backend="precomputed_leaf_reference",
        verify_exact_outputs=False, record_digests=False,
        initial_page_size=4096, max_page_size=65536,
        candidate_multi_gpu_mode="corpus",
        candidate_score_precision="fp32_exact", tree_probe_counts=(),
        tree_probe_result_width=64, ivf_probe_counts=(), ivf_nlist=1024,
        ivf_training_rows=262144, ivf_iterations=10,
        ivf_build_threads=24, pruning_diagnostic_queries=0,
        quality_reference=False, quality_query_count=0,
        quality_distance_reference="served_scheduler",
        response_query_count=0, batch_sizes=(), cpu_batch1_query_count=0):
    """Benchmark current PaperMIFA at scale, including 1->2 GPU sharding."""
    devices = tuple(item.strip() for item in gpu_devices if item.strip())
    if len(set(devices)) != len(devices):
        raise ValueError("scale GPU devices must be unique")
    available_query_rows = (
        np.arange(len(searchable_context["queries"]), dtype=np.int64)
        if query_rows is None else np.asarray(query_rows, dtype=np.int64)
    )
    if (available_query_rows.ndim != 1 or
            np.any(available_query_rows < 0) or
            np.any(available_query_rows >= len(searchable_context["queries"]))):
        raise ValueError("scale query rows do not align with the test cohort")
    query_count = min(int(query_count), len(available_query_rows))
    if query_count <= 0:
        raise ValueError("scale query count must be positive")
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("scale repeats must be positive")
    if executor_backend not in {
            "precomputed_leaf_reference", "gpu_candidate_stream"}:
        raise ValueError("unknown scale executor backend")
    if executor_backend == "gpu_candidate_stream" and not devices:
        raise ValueError("the GPU candidate stream requires a GPU device")
    if candidate_multi_gpu_mode not in {"corpus", "query"}:
        raise ValueError("unknown candidate multi-GPU mode")
    if candidate_score_precision not in {"fp32_exact", "fp16_refine"}:
        raise ValueError("unknown candidate score precision")
    quality_query_count = int(quality_query_count)
    if quality_distance_reference not in {
            "served_scheduler", "cpu_exact"}:
        raise ValueError("unknown scale quality distance reference")
    response_query_count = int(response_query_count)
    cpu_batch1_query_count = int(cpu_batch1_query_count)
    batch_sizes = tuple(int(value) for value in batch_sizes)
    if (quality_query_count < 0 or response_query_count < 0 or
            cpu_batch1_query_count < 0):
        raise ValueError("scale quality/response query counts cannot be negative")
    if any(value < 1 for value in batch_sizes):
        raise ValueError("scale batch sizes must be positive")
    if batch_sizes and executor_backend != "gpu_candidate_stream":
        raise ValueError("scale batch sizes require the GPU candidate stream")

    build_start = time.perf_counter()
    embeddings, table, source_rows = _books_derived_scale_snapshot(
        searchable_context["embeddings"], searchable_context["observed"],
        target_n, dimension=dimension, seed=expansion_seed,
        noise_std=noise_std,
        jitter_reference_embeddings=jitter_reference_embeddings)
    complete_table = None
    if quality_reference:
        complete_source = searchable_context["base"].table
        if len(complete_source) != len(searchable_context["observed"]):
            raise RuntimeError("complete and observed scale sources are not aligned")
        complete_table = complete_source.iloc[source_rows].reset_index(drop=True)
    expansion_seconds = time.perf_counter() - build_start
    projected_training = project_normalize(training_embeddings, dimension)
    ann_start = time.perf_counter()
    exact_backend = ExactBackend().build(embeddings)
    ann_setup_seconds = time.perf_counter() - ann_start
    model = PaperMIFA(
        projected_training, training_table, embeddings, table, dtypes,
        policy, theta=theta,
        projection_dim=structural_config.projection_dim,
        leaf_capacity=structural_config.leaf_capacity,
        ann_backend=exact_backend, ann_seed_size=64, seed=seed,
        fused_cpu=True, leaf_frontier=True, distance_provider=None,
        allow_distance_order_specialization=False,
        external_ann_build_seconds=ann_setup_seconds)
    model_build_seconds = time.perf_counter() - build_start

    def unique_array_bytes(values):
        seen = set()
        total = 0
        for value in values:
            if isinstance(value, np.ndarray) and id(value) not in seen:
                seen.add(id(value))
                total += int(value.nbytes)
        return int(total)

    tree_arrays = [
        value for value in model.tree.__dict__.values()
        if isinstance(value, np.ndarray)
    ]
    if model.tree._pca is not None:
        tree_arrays.extend(
            value for value in model.tree._pca.__dict__.values()
            if isinstance(value, np.ndarray)
        )
    for node in model.tree.nodes:
        tree_arrays.extend(
            value for value in node.__dict__.values()
            if isinstance(value, np.ndarray)
        )
    tree_resident_bytes = unique_array_bytes(tree_arrays)
    support_resident_bytes = int(model.support_index.resident_bytes)
    process_ready_rss_bytes = int(
        int(Path("/proc/self/statm").read_text().split()[1])
        * int(os.sysconf("SC_PAGE_SIZE")))
    scale_memory = {
        "raw_vector_bytes": int(embeddings.nbytes),
        "ann_index_bytes": 0,
        "ann_index_scope": (
            "ExactBackend/candidate provider has no separate persisted ANN "
            "graph; the resident raw vectors are reported separately"
        ),
        "projected_tree_resident_bytes": int(tree_resident_bytes),
        "shared_support_index_resident_bytes": int(support_resident_bytes),
        "mifa_auxiliary_index_bytes": int(
            tree_resident_bytes + support_resident_bytes),
        "mifa_auxiliary_scope": (
            "unique resident projected-tree/PCA arrays plus the shared "
            "historical support index; excludes raw vectors and query caches"
        ),
        "model_ready_process_rss_bytes": process_ready_rss_bytes,
    }

    provider_configurations = []
    begin = time.perf_counter()
    cpu_provider = NumpyResidentDistanceProvider(embeddings)
    provider_configurations.append({
        "backend": "cpu",
        "gpu_count": 0,
        "devices": ["cpu"],
        "providers": [(cpu_provider, 0, len(embeddings))],
        "provider_build_seconds": [float(time.perf_counter() - begin)],
        "parallelism": "single_cpu",
    })
    full_provider = None
    if devices:
        begin = time.perf_counter()
        full_provider = TorchResidentDistanceProvider(
            embeddings, device=devices[0])
        provider_configurations.append({
            "backend": "gpu",
            "gpu_count": 1,
            "devices": [devices[0]],
            "providers": [(full_provider, 0, len(embeddings))],
            "provider_build_seconds": [float(time.perf_counter() - begin)],
            "parallelism": "single_gpu",
        })
    if len(devices) > 1:
        boundaries = np.linspace(
            0, len(embeddings), len(devices) + 1, dtype=np.int64)
        shard_providers = []
        shard_build_seconds = []
        for device_index, device in enumerate(devices):
            if candidate_multi_gpu_mode == "query":
                start, end = 0, len(embeddings)
            else:
                start = int(boundaries[device_index])
                end = int(boundaries[device_index + 1])
            begin = time.perf_counter()
            provider = TorchResidentDistanceProvider(
                embeddings[start:end], device=device)
            shard_build_seconds.append(time.perf_counter() - begin)
            shard_providers.append((provider, start, end))
        provider_configurations.append({
            "backend": "gpu",
            "gpu_count": len(devices),
            "devices": list(devices),
            "providers": shard_providers,
            "provider_build_seconds": [
                float(value) for value in shard_build_seconds],
            "parallelism": (
                "replicated_corpus_query_sharded"
                if candidate_multi_gpu_mode == "query" else
                "contiguous_corpus_sharded"),
        })

    selected_query_rows = np.ascontiguousarray(
        available_query_rows[:query_count], dtype=np.int64)
    cohort = [
        searchable_context["queries"][int(row)]
        for row in selected_query_rows
    ]
    queries = np.ascontiguousarray(
        project_normalize(np.ascontiguousarray(
            [query["q_e"] for query in cohort], dtype=np.float32), dimension),
        dtype=np.float32)
    ks = [int(query["k"]) for query in cohort]
    predicates = [query["phi"] for query in cohort]
    pruning_diagnostic = None
    if int(pruning_diagnostic_queries) > 0:
        diagnostic_queries = [
            {**query, "q_e": queries[index], "k": ks[index]}
            for index, query in enumerate(cohort)
        ]
        pruning_diagnostic = _run_disable_one_diagnostic(
            model, diagnostic_queries,
            query_rows=selected_query_rows,
            query_count=pruning_diagnostic_queries,
            workload_name="Books-derived 1M/128d mixed")
    measurement_rows = []
    scheduler_distances = None
    gpu_distance_digests = []
    gpu_fixed_policy_output_digests = []
    canonical_distance_configuration = (
        next((configuration for configuration in provider_configurations
              if configuration["backend"] == "gpu" and
              configuration["gpu_count"] == 1),
             provider_configurations[-1]))
    for configuration in provider_configurations:
        device_count = int(configuration["gpu_count"])
        active = configuration["providers"]
        prepare_distances = (
            _prepare_query_sharded_distances
            if configuration["parallelism"] ==
            "replicated_corpus_query_sharded" else
            _prepare_corpus_sharded_distances)
        # Warm the selected provider outside reported steady-state repeats.
        warm = prepare_distances(
            active, queries[:min(2, query_count)],
            batch_size=distance_batch_size)
        del warm
        timings = []
        final_distances = None
        for _ in range(repeats):
            begin = time.perf_counter()
            final_distances = prepare_distances(
                active, queries, batch_size=distance_batch_size)
            timings.append((time.perf_counter() - begin) * 1000.0)
        digest = (
            _digest_array(final_distances) if record_digests else None)
        if configuration["backend"] == "gpu" and record_digests:
            gpu_distance_digests.append(digest)
        fixed_output_digest = None
        if configuration["backend"] == "gpu":
            # GPU partitioning may change low bits of otherwise equivalent
            # FP32 distances.  Raw-byte equality is diagnostic only; the
            # semantic gate is the ordered fixed-policy ID result.
            model.inference.clear_all_serving_caches()
            fixed_outputs = model.executor.search_precomputed_batch(
                queries, ks, predicates, final_distances)
            fixed_ids = [
                np.asarray(row[0], dtype=np.int64) for row in fixed_outputs]
            fixed_output_digest = _sha_json([
                row.tolist() for row in fixed_ids])
            gpu_fixed_policy_output_digests.append(fixed_output_digest)
            fixed_outputs = None
            fixed_ids = None
        median_ms = float(np.median(timings))
        measurement_rows.append({
            "backend": configuration["backend"],
            "gpu_count": int(device_count),
            "devices": list(configuration["devices"]),
            "parallelism": configuration["parallelism"],
            "corpus_rows_per_gpu": [
                int(end - start) for _, start, end in active],
            "provider_build_seconds": list(
                configuration["provider_build_seconds"]),
            "distance_total_ms_repeats": [float(value) for value in timings],
            "distance_total_ms_median": median_ms,
            "distance_ms_per_query": median_ms / query_count,
            "distance_qps": 1000.0 * query_count / median_ms,
            "distance_digest": digest,
            "fixed_policy_output_digest": (
                fixed_output_digest if fixed_output_digest is not None else
                "N/A_CPU_DISTANCE_NUMERIC_PATH"),
        })
        if configuration is canonical_distance_configuration:
            scheduler_distances = final_distances
        else:
            del final_distances
    gpu_distance_digest_match = bool(
        not gpu_distance_digests or len(set(gpu_distance_digests)) == 1)
    gpu_fixed_policy_ordered_id_agreement = bool(
        not gpu_fixed_policy_output_digests or
        len(set(gpu_fixed_policy_output_digests)) == 1)
    if not gpu_fixed_policy_ordered_id_agreement:
        raise AssertionError(
            "single- and multi-GPU distance slabs changed fixed-policy IDs")
    if scheduler_distances is None:
        raise AssertionError("canonical scale distance slab was not retained")
    for row in measurement_rows:
        if row["backend"] == "gpu":
            row["fixed_policy_ordered_id_agreement_with_single_gpu"] = bool(
                row["fixed_policy_output_digest"] ==
                gpu_fixed_policy_output_digests[0])
        else:
            row["fixed_policy_ordered_id_agreement_with_single_gpu"] = (
                "N/A_CPU_DISTANCE_NUMERIC_PATH")

    scheduler_cold_cache_manifest = \
        model.inference.clear_all_serving_caches()
    begin = time.perf_counter()
    cold_outputs = model.executor.search_precomputed_batch(
        queries, ks, predicates, scheduler_distances)
    scheduler_cold_ms = (time.perf_counter() - begin) * 1000.0
    cold_ids = [np.asarray(row[0], dtype=np.int64) for row in cold_outputs]
    scheduler_steady_repeats = []
    steady_ids = None
    for _ in range(repeats):
        begin = time.perf_counter()
        steady_outputs = model.executor.search_precomputed_batch(
            queries, ks, predicates, scheduler_distances)
        scheduler_steady_repeats.append(
            (time.perf_counter() - begin) * 1000.0)
        current_ids = [
            np.asarray(row[0], dtype=np.int64) for row in steady_outputs]
        if any(not np.array_equal(left, right)
               for left, right in zip(cold_ids, current_ids)):
            raise AssertionError(
                "cold and steady scale scheduler outputs disagree")
        steady_ids = current_ids
    scheduler_steady_ms = float(np.median(scheduler_steady_repeats))

    scale_quality = None
    if quality_reference:
        quality_count = min(
            quality_query_count if quality_query_count else query_count,
            query_count,
        )
        truth_eligibility = np.ascontiguousarray([
            predicate.eval_table(complete_table) == TRUE
            for predicate in predicates[:quality_count]
        ], dtype=np.bool_)
        quality_distances = (
            np.ascontiguousarray(scheduler_distances[:quality_count])
            if quality_distance_reference == "served_scheduler" else
            cpu_provider.prepare_batch(
                queries[:quality_count], batch_size=distance_batch_size)
        )
        truth_ids = topk_masked_batch_numba(
            quality_distances,
            truth_eligibility,
            int(ks[0]),
        )[0]
        returned_ids = _padded_id_matrix(
            steady_ids[:quality_count], int(ks[0]))
        per_query_quality = []
        for query_index, (returned, truth) in enumerate(
                zip(returned_ids, truth_ids)):
            returned_set = set(int(value) for value in returned if value >= 0)
            truth_set = set(int(value) for value in truth if value >= 0)
            overlap = len(returned_set & truth_set)
            precision = overlap / len(returned_set) if returned_set else 0.0
            recall = overlap / len(truth_set) if truth_set else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            per_query_quality.append({
                "query_index": int(query_index),
                "query_row": int(selected_query_rows[query_index]),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            })
        scale_quality = {
            "status": "controlled_scale_same_dimension_complete_truth",
            "query_count": int(quality_count),
            "truth_scope": (
                "same-dimensional complete metadata and exact full-corpus "
                "distance top-k on the deterministic Books-derived scale corpus"
            ),
            "selection_uses_scale_truth": False,
            "distance_reference": quality_distance_reference,
            "precision": float(np.mean([
                row["precision"] for row in per_query_quality])),
            "recall": float(np.mean([
                row["recall"] for row in per_query_quality])),
            "f1": float(np.mean([row["f1"] for row in per_query_quality])),
            "same_dimension_truth_ids": truth_ids.tolist(),
            "mifa_returned_ids": returned_ids.tolist(),
            "same_dimension_truth_ids_sha256": _digest_array(truth_ids),
            "mifa_returned_ids_sha256": _digest_array(returned_ids),
            "complete_eligibility_sha256": _digest_array(
                np.packbits(truth_eligibility)),
            "per_query": per_query_quality,
        }
        truth_eligibility = None
        quality_distances = None

    # Keep single-query latency separate from the multi-query throughput path.
    # Every timed query includes its own CPU distance preparation plus the
    # unchanged unified-tree scheduler; no batched timing is relabeled latency.
    model.inference.clear_all_serving_caches()

    cpu_batch1_count = min(
        cpu_batch1_query_count if cpu_batch1_query_count else query_count,
        query_count,
    )

    def run_cpu_batch1():
        run_rows = []
        run_ids = []
        for query_index, (query, k, predicate) in enumerate(zip(
                queries[:cpu_batch1_count], ks[:cpu_batch1_count],
                predicates[:cpu_batch1_count])):
            begin = time.perf_counter()
            distance_row = cpu_provider.prepare(query)
            distance_ms = (time.perf_counter() - begin) * 1000.0
            begin = time.perf_counter()
            ids, _, counters = model.executor.search(
                query, k, predicate, epsilon=0.0,
                precomputed_distances=distance_row)
            scheduler_ms = (time.perf_counter() - begin) * 1000.0
            run_ids.append(np.asarray(ids, dtype=np.int64))
            run_rows.append({
                "query_index": int(query_index),
                "query_row": int(selected_query_rows[query_index]),
                "distance_ms": float(distance_ms),
                "scheduler_ms": float(scheduler_ms),
                "total_ms": float(distance_ms + scheduler_ms),
                "physical_distance_evaluations": int(counters.get(
                    "physical_distance_evaluations", 0)),
                "visited_tuples": int(counters.get("visited_tuples", 0)),
                "hard_decisions": int(counters.get("hard_decisions", 0)),
                "lazy_inferences": int(counters.get("lazy_inferences", 0)),
            })
        return run_ids, run_rows

    batch1_cold_ids, batch1_cold_rows = run_cpu_batch1()
    batch1_repeat_rows = []
    for _ in range(repeats):
        batch1_ids, batch1_rows = run_cpu_batch1()
        if any(not np.array_equal(left, right) for left, right in zip(
                batch1_cold_ids, batch1_ids)):
            raise AssertionError(
                "cold and steady CPU batch-1 outputs disagree")
        batch1_repeat_rows.append(batch1_rows)
    native_batch_agreement = all(
        np.array_equal(left, right) for left, right in zip(
            steady_ids[:cpu_batch1_count], batch1_cold_ids)
    ) if not devices else None
    if native_batch_agreement is False:
        raise AssertionError(
            "CPU batch-1 and native batch unified-tree outputs disagree")
    batch1_exact = []
    batch1_audit_counters = []
    if verify_exact_outputs:
        model.inference.clear_all_serving_caches()
        for query_index, (query, k, predicate) in enumerate(zip(
                queries[:cpu_batch1_count], ks[:cpu_batch1_count],
                predicates[:cpu_batch1_count])):
            distance_row = cpu_provider.prepare(query)
            oracle, _, audit_counters = _exact_distance_order_reference(
                model.executor, query, k, predicate, distance_row)
            batch1_exact.append(bool(np.array_equal(
                batch1_cold_ids[query_index], oracle)))
            batch1_audit_counters.append(audit_counters)
        if not all(batch1_exact):
            raise AssertionError(
                "CPU batch-1 failed its exact fixed-policy check")
    batch1_per_query = []
    for query_index in range(cpu_batch1_count):
        distance_runs = [
            rows[query_index]["distance_ms"] for rows in batch1_repeat_rows]
        scheduler_runs = [
            rows[query_index]["scheduler_ms"] for rows in batch1_repeat_rows]
        total_runs = [
            rows[query_index]["total_ms"] for rows in batch1_repeat_rows]
        batch1_per_query.append({
            **batch1_repeat_rows[-1][query_index],
            "distance_ms_repeats": distance_runs,
            "scheduler_ms_repeats": scheduler_runs,
            "total_ms_repeats": total_runs,
            "distance_ms": float(np.median(distance_runs)),
            "scheduler_ms": float(np.median(scheduler_runs)),
            "total_ms": float(np.median(total_runs)),
        })
    batch1_values = np.asarray([
        row["total_ms"] for row in batch1_per_query], dtype=np.float64)
    cpu_batch1 = {
        "timing_scope": (
            "one query: CPU exact-distance preparation plus formal "
            "unified-tree scheduler"),
        "executor": "unified_tree_ann",
        "query_threads": 1,
        "query_count": int(cpu_batch1_count),
        "query_prefix_selection": "fixed selected scale cohort prefix",
        "cache_regime": "one cold pass excluded then repeated-predicate steady",
        "repeats": int(repeats),
        "cold_per_query": batch1_cold_rows,
        "latency_ms_mean": float(batch1_values.mean()),
        "latency_ms_p50": float(np.quantile(batch1_values, 0.50)),
        "latency_ms_p95": float(np.quantile(batch1_values, 0.95)),
        "qps_from_mean": float(1000.0 / batch1_values.mean()),
        "per_query": batch1_per_query,
        "ordered_exact_queries": int(sum(batch1_exact)),
        "ordered_exact": (
            bool(all(batch1_exact)) if verify_exact_outputs else None),
        "exact_reference": (
            "exact_distance_order_fixed_policy"
            if verify_exact_outputs else None),
        "exact_reference_visited_tuples_mean": (
            float(np.mean([
                row["visited_tuples"] for row in batch1_audit_counters]))
            if verify_exact_outputs else None),
        "native_batch_agreement": native_batch_agreement,
    }

    exact_reference_comparisons = []
    if verify_exact_outputs:
        model.inference.clear_all_serving_caches()
        for index in range(query_count):
            oracle, _, audit_counters = _exact_distance_order_reference(
                model.executor, queries[index], ks[index], predicates[index],
                scheduler_distances[index])
            passed = bool(np.array_equal(steady_ids[index], oracle))
            exact_reference_comparisons.append({
                "query_index": int(index),
                "eligible_rows": None,
                "reference": "exact_distance_order_fixed_policy",
                "reference_visited_tuples": int(
                    audit_counters["visited_tuples"]),
                "passed": passed,
            })
        if not all(row["passed"] for row in exact_reference_comparisons):
            raise AssertionError(
                "scale scheduler failed the exact fixed-policy check")
    reference_host_state_bytes = int(max(
        row[2].get("batch_explicit_host_state_bytes_total", 0)
        for row in steady_outputs))

    candidate_stream_cpu = None
    candidate_stream = None
    candidate_stream_multi_gpu = None
    candidate_response_batch1 = None
    candidate_response_cpu_batch1 = None
    candidate_batch_scaling = []
    tree_probe_measurements = []
    if executor_backend == "gpu_candidate_stream":
        # The paired reference has already supplied IDs, timing, explicit
        # state bytes, and oracle checks.  Release its QxN host slab and CUDA
        # providers before measuring the candidate path so they cannot distort
        # host-state or allocator telemetry.
        scheduler_distances = None
        final_distances = None
        cold_outputs = None
        steady_outputs = None
        for configuration in provider_configurations:
            configuration["providers"].clear()
        provider_configurations.clear()
        full_provider = None
        cpu_provider = None
        shard_providers = []
        active = None
        provider = None
        configuration = None
        model.inference.clear_all_serving_caches()
        gc.collect()
        full_distance_bytes = int(
            query_count * len(embeddings) * np.dtype(np.float32).itemsize)

        cpu_candidate_provider_start = time.perf_counter()
        cpu_candidate_provider = NumpyExactCandidateStreamProvider(embeddings)
        cpu_candidate_provider_build_seconds = (
            time.perf_counter() - cpu_candidate_provider_start)
        cpu_leaf_distances = \
            cpu_candidate_provider.distance_provider.prepare_batch(queries)
        cpu_leaf_outputs = model.executor.search_precomputed_batch(
            queries, ks, predicates, cpu_leaf_distances)
        cpu_leaf_ids = [
            np.asarray(row[0], dtype=np.int64) for row in cpu_leaf_outputs]
        cpu_leaf_distances = None
        cpu_leaf_outputs = None
        gc.collect()
        cpu_candidate_cold_cache_manifest = \
            model.inference.clear_all_serving_caches()
        cpu_candidate_start = time.perf_counter()
        cold_cpu_candidate_outputs = model.search_candidate_stream_batch(
            queries, ks, predicates, cpu_candidate_provider,
            initial_page_size=initial_page_size,
            max_page_size=max_page_size,
            fallback_to_reference=False)
        cpu_candidate_cold_ms = (
            time.perf_counter() - cpu_candidate_start) * 1000.0
        cold_cpu_candidate_ids = [
            np.asarray(row[0], dtype=np.int64)
            for row in cold_cpu_candidate_outputs]
        cpu_candidate_repeats = []
        cpu_candidate_stage_repeats = []
        cpu_candidate_outputs = None
        for _ in range(repeats):
            cpu_candidate_start = time.perf_counter()
            cpu_candidate_outputs = model.search_candidate_stream_batch(
                queries, ks, predicates, cpu_candidate_provider,
                initial_page_size=initial_page_size,
                max_page_size=max_page_size,
                fallback_to_reference=False)
            cpu_candidate_repeats.append(
                (time.perf_counter() - cpu_candidate_start) * 1000.0)
            cpu_batch_counters = cpu_candidate_outputs[0][2]
            cpu_candidate_stage_repeats.append({
                "distance_compute_ms": float(cpu_batch_counters[
                    "candidate_stream_batch_distance_compute_ms"]),
                "hard_predicate_ms": float(cpu_batch_counters[
                    "candidate_stream_batch_hard_predicate_ms"]),
                "candidate_select_ms": float(cpu_batch_counters[
                    "candidate_stream_batch_candidate_select_ms"]),
                "compact_d2h_ms": float(cpu_batch_counters[
                    "candidate_stream_batch_compact_d2h_ms"]),
                "cpu_consume_ms": float(sum(
                    row[2]["candidate_stream_cpu_consume_ms"]
                    for row in cpu_candidate_outputs)),
                "statistical_inference_ms": float(sum(
                    row[2]["candidate_stream_statistical_inference_ms"]
                    for row in cpu_candidate_outputs)),
            })
        cpu_candidate_ms = float(np.median(cpu_candidate_repeats))
        cpu_candidate_ids = [
            np.asarray(row[0], dtype=np.int64)
            for row in cpu_candidate_outputs]
        cpu_candidate_exact_rows = [
            bool(np.array_equal(reference_ids, current_ids))
            for reference_ids, current_ids in zip(
                cpu_leaf_ids, cpu_candidate_ids)]
        cold_cpu_candidate_exact_rows = [
            bool(np.array_equal(reference_ids, current_ids))
            for reference_ids, current_ids in zip(
                cpu_leaf_ids, cold_cpu_candidate_ids)]
        cpu_cuda_agreement_rows = [
            bool(np.array_equal(reference_ids, current_ids))
            for reference_ids, current_ids in zip(
                steady_ids, cpu_candidate_ids)]
        if (not all(cpu_candidate_exact_rows) or
                not all(cold_cpu_candidate_exact_rows)):
            raise AssertionError(
                "CPU candidate stream disagrees with exact leaf reference")
        cpu_candidate_counters = [row[2] for row in cpu_candidate_outputs]
        cpu_candidate_median_index = int(np.argsort(cpu_candidate_repeats)[
            len(cpu_candidate_repeats) // 2])
        cpu_candidate_stages = cpu_candidate_stage_repeats[
            cpu_candidate_median_index]
        cpu_candidate_events = np.asarray([
            row["candidate_stream_candidate_events"]
            for row in cpu_candidate_counters], dtype=np.int64)
        cpu_candidate_inference_events = np.asarray([
            row["candidate_stream_inference_events"]
            for row in cpu_candidate_counters], dtype=np.int64)
        cpu_candidate_fallbacks = int(sum(
            row["candidate_stream_fallback"]
            for row in cpu_candidate_counters))
        candidate_stream_cpu = {
            "status": "phase_a_full_qn_cpu_exact_compact_host_pages",
            "executor_backend": "cpu_candidate_stream",
            "provider_build_seconds": float(
                cpu_candidate_provider_build_seconds),
            "cold_cache_manifest_before":
                cpu_candidate_cold_cache_manifest,
            "cache_regime": "cold-empty then repeated-predicate steady",
            "cold_total_ms": float(cpu_candidate_cold_ms),
            "steady_total_ms_repeats": [
                float(value) for value in cpu_candidate_repeats],
            "steady_total_ms": float(cpu_candidate_ms),
            "steady_ms_per_query": float(cpu_candidate_ms / query_count),
            "steady_qps": float(1000.0 * query_count / cpu_candidate_ms),
            "median_repeat_stages": cpu_candidate_stages,
            "ordered_exact_queries": int(sum(cpu_candidate_exact_rows)),
            "ordered_exact": bool(all(cpu_candidate_exact_rows)),
            "cold_ordered_exact_queries": int(sum(
                cold_cpu_candidate_exact_rows)),
            "frozen_cuda_reference_agreement_queries": int(sum(
                cpu_cuda_agreement_rows)),
            "frozen_cuda_reference_agreement": bool(all(
                cpu_cuda_agreement_rows)),
            "numeric_audit_note": (
                "CPU candidate IDs are exact against the unchanged leaf "
                "reference over CPU-SGEMM rows; CUDA agreement is reported "
                "separately because float32 accumulation order may differ."),
            "output_digest": _sha_json([
                row.tolist() for row in cpu_candidate_ids]),
            "candidate_events_mean": float(cpu_candidate_events.mean()),
            "inference_events_mean": float(
                cpu_candidate_inference_events.mean()),
            "data_movement": {
                "full_distance_reference_bytes": full_distance_bytes,
                "compact_d2h_bytes": 0,
                "reduction_x": None,
                "reason": "candidate pages are CPU-resident",
            },
            "host_state": {
                "reference_explicit_bytes": reference_host_state_bytes,
                "candidate_peak_provider_bytes": int(
                    cpu_candidate_counters[0][
                        "candidate_stream_peak_provider_host_bytes"]),
                "candidate_peak_compact_page_bytes": int(
                    cpu_candidate_counters[0][
                        "candidate_stream_peak_compact_host_page_bytes_batch"]),
                "asymptotic": "full_distance_and_state_O(QN)_no_leaf_heaps",
            },
            "peak_gpu_allocated_bytes": 0,
            "cpu_processing_fraction": 1.0,
            "fallback_queries": cpu_candidate_fallbacks,
        }
        response_count = min(int(response_query_count), query_count)
        if response_count > 0:
            candidate_response_cpu_batch1 = \
                _measure_candidate_response_batch1(
                    model, queries, ks, predicates,
                    cpu_candidate_provider, cpu_leaf_ids,
                    query_count=response_count, repeats=repeats,
                    initial_page_size=initial_page_size,
                    max_page_size=max_page_size)
        cpu_candidate_provider = None
        cold_cpu_candidate_outputs = None
        cpu_candidate_outputs = None
        gc.collect()

        import torch
        torch.cuda.empty_cache()
        candidate_provider_start = time.perf_counter()
        candidate_provider = TorchExactCandidateStreamProvider(
            embeddings, device=devices[0],
            score_precision=candidate_score_precision)
        candidate_provider_build_seconds = (
            time.perf_counter() - candidate_provider_start)

        # Cold is reported but excluded from the steady median.  The fixed
        # page policy is global for the complete run; it is not selected from
        # this dataset or query set.
        candidate_cold_cache_manifest = \
            model.inference.clear_all_serving_caches()
        candidate_start = time.perf_counter()
        cold_candidate_outputs = model.search_candidate_stream_batch(
            queries, ks, predicates, candidate_provider,
            initial_page_size=initial_page_size,
            max_page_size=max_page_size,
            fallback_to_reference=False)
        candidate_cold_ms = (time.perf_counter() - candidate_start) * 1000.0
        cold_candidate_ids = [
            np.asarray(row[0], dtype=np.int64)
            for row in cold_candidate_outputs]
        external_replica_barrier = _wait_for_external_replica_barrier(
            "candidate_stream_steady")
        candidate_repeats = []
        candidate_stage_repeats = []
        candidate_outputs = None
        for _ in range(repeats):
            candidate_start = time.perf_counter()
            candidate_outputs = model.search_candidate_stream_batch(
                queries, ks, predicates, candidate_provider,
                initial_page_size=initial_page_size,
                max_page_size=max_page_size,
                fallback_to_reference=False)
            candidate_repeats.append(
                (time.perf_counter() - candidate_start) * 1000.0)
            batch_counters = candidate_outputs[0][2]
            candidate_stage_repeats.append({
                "distance_compute_ms": float(batch_counters[
                    "candidate_stream_batch_distance_compute_ms"]),
                "hard_predicate_ms": float(batch_counters[
                    "candidate_stream_batch_hard_predicate_ms"]),
                "candidate_select_ms": float(batch_counters[
                    "candidate_stream_batch_candidate_select_ms"]),
                "compact_d2h_ms": float(batch_counters[
                    "candidate_stream_batch_compact_d2h_ms"]),
                "cpu_consume_ms": float(sum(
                    row[2]["candidate_stream_cpu_consume_ms"]
                    for row in candidate_outputs)),
                "statistical_inference_ms": float(sum(
                    row[2]["candidate_stream_statistical_inference_ms"]
                    for row in candidate_outputs)),
            })
        candidate_ms = float(np.median(candidate_repeats))
        candidate_ids = [
            np.asarray(row[0], dtype=np.int64)
            for row in candidate_outputs]
        candidate_exact_rows = [
            bool(np.array_equal(reference_ids, current_ids))
            for reference_ids, current_ids in zip(steady_ids, candidate_ids)]
        cold_exact_rows = [
            bool(np.array_equal(reference_ids, current_ids))
            for reference_ids, current_ids in zip(
                steady_ids, cold_candidate_ids)]
        if not all(candidate_exact_rows) or not all(cold_exact_rows):
            raise AssertionError(
                "candidate stream disagrees with exact leaf reference")
        final_counters = [row[2] for row in candidate_outputs]
        batch_compact_bytes = int(final_counters[0][
            "candidate_stream_batch_compact_d2h_bytes"])
        candidates_transferred = np.asarray([
            row["candidate_stream_candidates_transferred"]
            for row in final_counters], dtype=np.int64)
        candidate_events = np.asarray([
            row["candidate_stream_candidate_events"]
            for row in final_counters], dtype=np.int64)
        unresolved_events = np.asarray([
            row["candidate_stream_unresolved_events"]
            for row in final_counters], dtype=np.int64)
        inference_events = np.asarray([
            row["candidate_stream_inference_events"]
            for row in final_counters], dtype=np.int64)
        overclassified_unresolved = np.asarray([
            row["candidate_stream_overclassified_unresolved"]
            for row in final_counters], dtype=np.int64)
        representative_evaluations = np.asarray([
            row["lazy_inferences"]
            for row in final_counters], dtype=np.int64)
        statistical_cache_hits = np.asarray([
            row["statistical_cache_hits"]
            for row in final_counters], dtype=np.int64)
        page_rounds = np.asarray([
            row["candidate_stream_page_rounds"]
            for row in final_counters], dtype=np.int64)
        fallback_count = int(sum(
            row["candidate_stream_fallback"] for row in final_counters))
        median_repeat_index = int(np.argsort(candidate_repeats)[
            len(candidate_repeats) // 2])
        median_stages = candidate_stage_repeats[median_repeat_index]
        compact_host_page_bytes = int(final_counters[0][
            "candidate_stream_peak_compact_host_page_bytes_batch"])
        cpu_processing_fraction = (
            median_stages["cpu_consume_ms"] / candidate_ms)
        cpu_measurement = next(
            row for row in measurement_rows if row["backend"] == "cpu")
        cpu_reference_total_ms = float(
            cpu_measurement["distance_total_ms_median"] +
            scheduler_steady_ms)
        candidate_stream = {
            "status": (
                "phase_a_fp16_shortlist_fp32_rerank_compact_host_pages"
                if candidate_score_precision == "fp16_refine" else
                "phase_a_full_qn_gpu_exact_compact_host_pages"),
            "executor_backend": "gpu_candidate_stream",
            "score_precision": candidate_score_precision,
            "device": devices[0],
            "provider_build_seconds": float(
                candidate_provider_build_seconds),
            "cold_cache_manifest_before": candidate_cold_cache_manifest,
            "cache_regime": "cold-empty then repeated-predicate steady",
            "external_replica_barrier": external_replica_barrier,
            "page_policy": {
                "initial": int(initial_page_size),
                "growth": "double",
                "maximum": int(max_page_size),
                "selection_source": "global_fixed_not_dataset_tuned",
            },
            "cold_total_ms": float(candidate_cold_ms),
            "steady_total_ms_repeats": [
                float(value) for value in candidate_repeats],
            "steady_total_ms": float(candidate_ms),
            "steady_ms_per_query": float(candidate_ms / query_count),
            "steady_qps": float(1000.0 * query_count / candidate_ms),
            "median_repeat_stages": median_stages,
            "ordered_exact_queries": int(sum(candidate_exact_rows)),
            "ordered_exact": bool(all(candidate_exact_rows)),
            "cold_ordered_exact_queries": int(sum(cold_exact_rows)),
            "output_digest": _sha_json([
                row.tolist() for row in candidate_ids]),
            "candidate_events": {
                "mean": float(candidate_events.mean()),
                "maximum": int(candidate_events.max(initial=0)),
                "unresolved_mean": float(unresolved_events.mean()),
                "inference_events_mean": float(inference_events.mean()),
                "overclassified_unresolved_mean": float(
                    overclassified_unresolved.mean()),
                "representative_evaluations_mean": float(
                    representative_evaluations.mean()),
                "statistical_group_cache_hits_mean": float(
                    statistical_cache_hits.mean()),
                "transferred_mean": float(candidates_transferred.mean()),
                "transferred_maximum": int(
                    candidates_transferred.max(initial=0)),
                "page_round_histogram": {
                    str(int(value)): int(np.count_nonzero(page_rounds == value))
                    for value in np.unique(page_rounds)
                },
            },
            "data_movement": {
                "full_distance_reference_bytes": full_distance_bytes,
                "compact_d2h_bytes": batch_compact_bytes,
                "reduction_x": float(
                    full_distance_bytes / max(batch_compact_bytes, 1)),
                "fallback_full_distance_bytes": int(final_counters[0][
                    "candidate_stream_batch_full_fallback_d2h_bytes"]),
            },
            "host_state": {
                "reference_explicit_bytes": reference_host_state_bytes,
                "candidate_peak_compact_page_bytes": compact_host_page_bytes,
                "explicit_reduction_x": float(
                    reference_host_state_bytes /
                    max(compact_host_page_bytes, 1)),
                "asymptotic": "reference_O(QN)_candidate_O(QM)",
            },
            "peak_gpu_allocated_bytes": int(final_counters[0][
                "candidate_stream_peak_gpu_allocated_bytes"]),
            "cpu_processing_fraction": float(cpu_processing_fraction),
            "cpu_reference_total_ms": cpu_reference_total_ms,
            "cpu_to_candidate_speedup_x": float(
                cpu_reference_total_ms / candidate_ms),
            "candidate_cpu_to_one_gpu_speedup_x": float(
                cpu_candidate_ms / candidate_ms),
            "fallback_queries": fallback_count,
        }

        if response_query_count:
            response_count = min(response_query_count, query_count)
            model.inference.clear_all_serving_caches()
            for query_index in range(response_count):
                model.search_candidate_stream_batch(
                    queries[query_index:query_index + 1],
                    [ks[query_index]], [predicates[query_index]],
                    candidate_provider,
                    initial_page_size=initial_page_size,
                    max_page_size=max_page_size,
                    fallback_to_reference=False)
            response_runs = np.zeros(
                (repeats, response_count), dtype=np.float64)
            response_d2h = np.zeros(
                (repeats, response_count), dtype=np.int64)
            response_host_state = np.zeros(
                (repeats, response_count), dtype=np.int64)
            response_candidate_events = np.zeros(
                (repeats, response_count), dtype=np.int64)
            response_inference_events = np.zeros(
                (repeats, response_count), dtype=np.int64)
            response_provider_wall = np.zeros(
                (repeats, response_count), dtype=np.float64)
            response_device = np.zeros(
                (repeats, response_count), dtype=np.float64)
            response_device_d2h_wait = np.zeros(
                (repeats, response_count), dtype=np.float64)
            response_cpu_consume = np.zeros(
                (repeats, response_count), dtype=np.float64)
            response_distance_rows = np.zeros(
                (repeats, response_count), dtype=np.int64)
            response_eligible_rows = np.zeros(
                (repeats, response_count), dtype=np.int64)
            response_ids = None
            response_peak_gpu = 0
            for repeat in range(repeats):
                ids_this_repeat = []
                for query_index in range(response_count):
                    started = time.perf_counter()
                    output = model.search_candidate_stream_batch(
                        queries[query_index:query_index + 1],
                        [ks[query_index]], [predicates[query_index]],
                        candidate_provider,
                        initial_page_size=initial_page_size,
                        max_page_size=max_page_size,
                        fallback_to_reference=False)
                    response_runs[repeat, query_index] = (
                        time.perf_counter() - started) * 1000.0
                    ids_this_repeat.append(np.asarray(
                        output[0][0], dtype=np.int64))
                    response_counters = output[0][2]
                    response_d2h[repeat, query_index] = int(
                        response_counters.get(
                            "candidate_stream_batch_compact_d2h_bytes", 0))
                    response_host_state[repeat, query_index] = int(
                        response_counters.get(
                            "candidate_stream_peak_compact_host_page_bytes_batch",
                            0))
                    response_candidate_events[repeat, query_index] = int(
                        response_counters.get(
                            "candidate_stream_candidate_events", 0))
                    response_inference_events[repeat, query_index] = int(
                        response_counters.get(
                            "candidate_stream_inference_events", 0))
                    response_provider_wall[repeat, query_index] = float(
                        response_counters.get(
                            "candidate_stream_batch1_fused_wall_ms", 0.0))
                    response_device[repeat, query_index] = float(
                        response_counters.get(
                            "candidate_stream_batch1_fused_device_ms", 0.0))
                    response_device_d2h_wait[repeat, query_index] = float(
                        response_counters.get(
                            "candidate_stream_batch1_device_d2h_wait_ms", 0.0))
                    response_cpu_consume[repeat, query_index] = float(
                        response_counters.get(
                            "candidate_stream_cpu_consume_ms", 0.0))
                    response_distance_rows[repeat, query_index] = int(
                        response_counters.get(
                            "physical_distance_evaluations", 0))
                    response_eligible_rows[repeat, query_index] = int(
                        response_counters.get(
                            "candidate_stream_batch1_eligible_rows", 0))
                    response_peak_gpu = max(
                        response_peak_gpu,
                        int(response_counters.get(
                            "candidate_stream_peak_gpu_allocated_bytes", 0)))
                if response_ids is None:
                    response_ids = ids_this_repeat
                elif any(not np.array_equal(left, right) for left, right in zip(
                        response_ids, ids_this_repeat)):
                    raise AssertionError(
                        "GPU batch-1 response timing changed result IDs")
            if any(not np.array_equal(reference, observed)
                   for reference, observed in zip(
                       candidate_ids[:response_count], response_ids)):
                raise AssertionError(
                    "GPU batch-1 response IDs disagree with full-batch IDs")
            response_values = np.median(response_runs, axis=0)
            candidate_response_batch1 = {
                "status": "steady_single_request_latency",
                "query_count": int(response_count),
                "repeats": int(repeats),
                "cache_regime": (
                    "one fixed-order warm pass excluded, then repeated steady"
                ),
                "timing_scope": (
                    "one-query GPU candidate generation + compact D2H + "
                    "CPU exact fixed-policy consumption/final top-k"
                ),
                "latency_ms_mean": float(response_values.mean()),
                "latency_ms_p50": float(np.percentile(response_values, 50)),
                "latency_ms_p95": float(np.percentile(response_values, 95)),
                "qps_from_mean": float(1000.0 / response_values.mean()),
                "per_query_median_ms": response_values.tolist(),
                "per_query_repeat_ms": response_runs.tolist(),
                "compact_d2h_bytes_per_query_mean": float(
                    np.median(response_d2h, axis=0).mean()),
                "compact_d2h_bytes_per_query": np.median(
                    response_d2h, axis=0).astype(np.int64).tolist(),
                "explicit_host_state_bytes_per_query_mean": float(
                    np.median(response_host_state, axis=0).mean()),
                "explicit_host_state_bytes_per_query": np.median(
                    response_host_state, axis=0).astype(np.int64).tolist(),
                "candidate_events_per_query_mean": float(
                    np.median(response_candidate_events, axis=0).mean()),
                "statistical_inference_events_per_query_mean": float(
                    np.median(response_inference_events, axis=0).mean()),
                "provider_fused_wall_ms_per_query_mean": float(
                    np.median(response_provider_wall, axis=0).mean()),
                "provider_fused_device_ms_per_query_mean": float(
                    np.median(response_device, axis=0).mean()),
                "provider_device_d2h_wait_ms_per_query_mean": float(
                    np.median(response_device_d2h_wait, axis=0).mean()),
                "cpu_consume_ms_per_query_mean": float(
                    np.median(response_cpu_consume, axis=0).mean()),
                "distance_rows_per_query_mean": float(
                    np.median(response_distance_rows, axis=0).mean()),
                "eligible_rows_per_query_mean": float(
                    np.median(response_eligible_rows, axis=0).mean()),
                "ordered_exact_queries": int(response_count),
                "ordered_exact": True,
                "peak_gpu_allocated_bytes": int(response_peak_gpu),
            }

        for batch_size in batch_sizes:
            if batch_size > query_count:
                raise ValueError(
                    f"scale batch size {batch_size} exceeds query count "
                    f"{query_count}")
            model.inference.clear_all_serving_caches()
            cold = model.search_candidate_stream_batch(
                queries[:batch_size], ks[:batch_size],
                predicates[:batch_size], candidate_provider,
                initial_page_size=initial_page_size,
                max_page_size=max_page_size,
                fallback_to_reference=False)
            timing_runs = []
            output = None
            for _ in range(repeats):
                started = time.perf_counter()
                output = model.search_candidate_stream_batch(
                    queries[:batch_size], ks[:batch_size],
                    predicates[:batch_size], candidate_provider,
                    initial_page_size=initial_page_size,
                    max_page_size=max_page_size,
                    fallback_to_reference=False)
                timing_runs.append((time.perf_counter() - started) * 1000.0)
            cold_ids_for_batch = [
                np.asarray(row[0], dtype=np.int64) for row in cold]
            output_ids_for_batch = [
                np.asarray(row[0], dtype=np.int64) for row in output]
            exact = all(
                np.array_equal(reference, observed)
                for reference, observed in zip(
                    candidate_ids[:batch_size], output_ids_for_batch)
            ) and all(
                np.array_equal(reference, observed)
                for reference, observed in zip(
                    candidate_ids[:batch_size], cold_ids_for_batch)
            )
            if not exact:
                raise AssertionError(
                    f"GPU batch-size {batch_size} IDs disagree")
            median_total_ms = float(np.median(timing_runs))
            batch_counters = output[0][2]
            candidate_batch_scaling.append({
                "status": "saturated_throughput",
                "batch_size": int(batch_size),
                "repeats": int(repeats),
                "cache_regime": (
                    "one cold batch excluded, then repeated-predicate steady"
                ),
                "timing_scope": (
                    "whole GPU candidate-stream batch including compact D2H, "
                    "CPU fixed-policy consumption, and final top-k"
                ),
                "steady_total_ms_repeats": [
                    float(value) for value in timing_runs],
                "steady_total_ms": median_total_ms,
                "steady_ms_per_query": float(
                    median_total_ms / batch_size),
                "steady_qps": float(
                    1000.0 * batch_size / median_total_ms),
                "ordered_exact_queries": int(batch_size),
                "ordered_exact": True,
                "compact_d2h_bytes": int(batch_counters.get(
                    "candidate_stream_batch_compact_d2h_bytes", 0)),
                "explicit_host_state_bytes": int(batch_counters.get(
                    "candidate_stream_peak_compact_host_page_bytes_batch", 0)),
                "candidate_events_per_query_mean": float(np.mean([
                    row[2].get("candidate_stream_candidate_events", 0)
                    for row in output])),
                "statistical_inference_events_per_query_mean": float(np.mean([
                    row[2].get("candidate_stream_inference_events", 0)
                    for row in output])),
                "peak_gpu_allocated_bytes": int(batch_counters.get(
                    "candidate_stream_peak_gpu_allocated_bytes", 0)),
            })
        if len(devices) > 1:
            # Measure multi-GPU from a clean allocator state.  The exact
            # single-GPU IDs and timing are already immutable host values.
            candidate_provider = None
            cold_candidate_outputs = None
            candidate_outputs = None
            gc.collect()
            torch.cuda.empty_cache()
            shard_boundaries = np.linspace(
                0, len(embeddings), len(devices) + 1,
                dtype=np.int64)

            def build_candidate_shard(device_index):
                if candidate_multi_gpu_mode == "query":
                    return TorchExactCandidateStreamProvider(
                        embeddings, device=devices[device_index],
                        score_precision=candidate_score_precision)
                start = int(shard_boundaries[device_index])
                end = int(shard_boundaries[device_index + 1])
                return TorchExactCandidateStreamProvider(
                    embeddings[start:end], device=devices[device_index],
                    global_offset=start,
                    score_precision=candidate_score_precision)

            multi_provider_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                multi_shards = list(pool.map(
                    build_candidate_shard, range(len(devices))))
            multi_provider = (
                TorchQueryShardedExactCandidateStreamProvider(multi_shards)
                if candidate_multi_gpu_mode == "query" else
                TorchShardedExactCandidateStreamProvider(multi_shards))
            multi_provider_build_seconds = (
                time.perf_counter() - multi_provider_start)

            multi_cold_cache_manifest = \
                model.inference.clear_all_serving_caches()
            multi_start = time.perf_counter()
            cold_multi_outputs = model.search_candidate_stream_batch(
                queries, ks, predicates, multi_provider,
                initial_page_size=initial_page_size,
                max_page_size=max_page_size,
                fallback_to_reference=False)
            multi_cold_ms = (time.perf_counter() - multi_start) * 1000.0
            cold_multi_ids = [
                np.asarray(row[0], dtype=np.int64)
                for row in cold_multi_outputs]
            multi_repeats = []
            multi_stage_repeats = []
            multi_outputs = None
            for _ in range(repeats):
                multi_start = time.perf_counter()
                multi_outputs = model.search_candidate_stream_batch(
                    queries, ks, predicates, multi_provider,
                    initial_page_size=initial_page_size,
                    max_page_size=max_page_size,
                    fallback_to_reference=False)
                multi_repeats.append(
                    (time.perf_counter() - multi_start) * 1000.0)
                batch_counters = multi_outputs[0][2]
                multi_stage_repeats.append({
                    "distance_compute_ms": float(batch_counters[
                        "candidate_stream_batch_distance_compute_ms"]),
                    "hard_predicate_ms": float(batch_counters[
                        "candidate_stream_batch_hard_predicate_ms"]),
                    "candidate_select_ms": float(batch_counters[
                        "candidate_stream_batch_candidate_select_ms"]),
                    "compact_d2h_ms": float(batch_counters[
                        "candidate_stream_batch_compact_d2h_ms"]),
                    "cpu_merge_ms": float(batch_counters[
                        "candidate_stream_batch_cpu_merge_ms"]),
                    "cpu_consume_ms": float(sum(
                        row[2]["candidate_stream_cpu_consume_ms"]
                        for row in multi_outputs)),
                    "statistical_inference_ms": float(sum(
                        row[2]["candidate_stream_statistical_inference_ms"]
                        for row in multi_outputs)),
                })
            multi_ms = float(np.median(multi_repeats))
            multi_ids = [
                np.asarray(row[0], dtype=np.int64)
                for row in multi_outputs]
            multi_exact_rows = [
                bool(np.array_equal(single_ids, current_ids))
                for single_ids, current_ids in zip(candidate_ids, multi_ids)]
            cold_multi_exact_rows = [
                bool(np.array_equal(single_ids, current_ids))
                for single_ids, current_ids in zip(
                    candidate_ids, cold_multi_ids)]
            if not all(multi_exact_rows) or not all(cold_multi_exact_rows):
                raise AssertionError(
                    "multi-GPU candidate stream disagrees with single GPU")
            multi_counters = [row[2] for row in multi_outputs]
            multi_median_index = int(np.argsort(multi_repeats)[
                len(multi_repeats) // 2])
            multi_stages = multi_stage_repeats[multi_median_index]
            multi_compact_bytes = int(multi_counters[0][
                "candidate_stream_batch_compact_d2h_bytes"])
            multi_fallbacks = int(sum(
                row["candidate_stream_fallback"]
                for row in multi_counters))
            multi_cpu_ms = float(
                multi_stages["cpu_merge_ms"] +
                multi_stages["cpu_consume_ms"])
            multi_cpu_fraction = multi_cpu_ms / multi_ms
            multi_host_bytes = int(multi_counters[0][
                "candidate_stream_peak_compact_host_page_bytes_batch"])
            multi_candidate_events = np.asarray([
                row["candidate_stream_candidate_events"]
                for row in multi_counters], dtype=np.int64)
            multi_inference_events = np.asarray([
                row["candidate_stream_inference_events"]
                for row in multi_counters], dtype=np.int64)
            candidate_stream_multi_gpu = {
                "status": (
                    "phase_a_replicated_corpus_query_sharded"
                    if candidate_multi_gpu_mode == "query" else
                    "phase_a_corpus_sharded_compact_host_merge"),
                "executor_backend": "gpu_candidate_stream",
                "score_precision": candidate_score_precision,
                "gpu_count": int(len(devices)),
                "devices": list(devices),
                "corpus_rows_per_gpu": [
                    (int(len(embeddings))
                     if candidate_multi_gpu_mode == "query" else
                     int(shard_boundaries[index + 1] -
                         shard_boundaries[index]))
                    for index in range(len(devices))],
                "query_rows_per_gpu": [
                    (int(len(values))
                     if candidate_multi_gpu_mode == "query" else
                     int(query_count))
                    for values in np.array_split(
                        np.arange(query_count), len(devices))],
                "provider_build_seconds": float(
                    multi_provider_build_seconds),
                "cold_cache_manifest_before": multi_cold_cache_manifest,
                "cache_regime":
                    "cold-empty then repeated-predicate steady",
                "cold_total_ms": float(multi_cold_ms),
                "steady_total_ms_repeats": [
                    float(value) for value in multi_repeats],
                "steady_total_ms": float(multi_ms),
                "steady_ms_per_query": float(multi_ms / query_count),
                "steady_qps": float(1000.0 * query_count / multi_ms),
                "median_repeat_stages": multi_stages,
                "ordered_exact_queries": int(sum(multi_exact_rows)),
                "ordered_exact": bool(all(multi_exact_rows)),
                "cold_ordered_exact_queries": int(
                    sum(cold_multi_exact_rows)),
                "output_digest": _sha_json([
                    row.tolist() for row in multi_ids]),
                "candidate_events_mean": float(
                    multi_candidate_events.mean()),
                "inference_events_mean": float(
                    multi_inference_events.mean()),
                "data_movement": {
                    "full_distance_reference_bytes": full_distance_bytes,
                    "compact_d2h_bytes": multi_compact_bytes,
                    "reduction_x": float(
                        full_distance_bytes /
                        max(multi_compact_bytes, 1)),
                    "fallback_full_distance_bytes": int(
                        multi_counters[0][
                            "candidate_stream_batch_full_fallback_d2h_bytes"]),
                },
                "host_state": {
                    "reference_explicit_bytes": reference_host_state_bytes,
                    "candidate_peak_compact_bytes": multi_host_bytes,
                    "explicit_reduction_x": float(
                        reference_host_state_bytes /
                        max(multi_host_bytes, 1)),
                    "asymptotic": "reference_O(QN)_candidate_O(QM)",
                },
                "peak_gpu_allocated_bytes_sum": int(multi_counters[0][
                    "candidate_stream_peak_gpu_allocated_bytes"]),
                "cpu_merge_plus_inference_fraction": float(
                    multi_cpu_fraction),
                "single_to_multi_gpu_speedup_x": float(
                    candidate_ms / multi_ms),
                "fallback_queries": multi_fallbacks,
            }
            if hasattr(multi_provider, "shutdown"):
                multi_provider.shutdown(wait=True)
            multi_provider = None
            multi_replicas = []
            cold_multi_outputs = None
            multi_outputs = None
            gc.collect()
            torch.cuda.empty_cache()

        def fixed_policy_recall(reference_rows, output_rows):
            recalls = []
            complete = 0
            for reference, output in zip(reference_rows, output_rows):
                reference = np.asarray(reference, dtype=np.int64)
                output = np.asarray(output, dtype=np.int64)
                denominator = max(len(reference), 1)
                recalls.append(
                    len(np.intersect1d(reference, output, assume_unique=True)) /
                    denominator)
                complete += int(len(output) >= len(reference))
            return float(np.mean(recalls)), int(complete)

        def measure_tree_probe(provider, *, backend, gpu_count, devices_used,
                               probe_count, build_seconds, partition_kind,
                               partition_count, ann_index_build_seconds=0.0,
                               ann_index_manifest=None):
            model.inference.clear_all_serving_caches()
            cold_started = time.perf_counter()
            cold = model.search_candidate_stream_batch(
                queries, ks, predicates, provider,
                initial_page_size=min(tree_probe_result_width, max(ks)),
                max_page_size=tree_probe_result_width,
                fallback_to_reference=False)
            cold_ms = (time.perf_counter() - cold_started) * 1000.0
            timings = []
            stage_rows = []
            output = None
            for _ in range(repeats):
                started = time.perf_counter()
                output = model.search_candidate_stream_batch(
                    queries, ks, predicates, provider,
                    initial_page_size=min(
                        tree_probe_result_width, max(ks)),
                    max_page_size=tree_probe_result_width,
                    fallback_to_reference=False)
                timings.append((time.perf_counter() - started) * 1000.0)
                counters = output[0][2]
                stage_rows.append({
                    "tree_probe_plan_ms": float(counters[
                        "candidate_stream_batch_tree_probe_plan_ms"]),
                    "fused_distance_predicate_ms": float(counters[
                        "candidate_stream_batch_distance_compute_ms"]),
                    "candidate_select_ms": float(counters[
                        "candidate_stream_batch_candidate_select_ms"]),
                    "compact_d2h_ms": float(counters[
                        "candidate_stream_batch_compact_d2h_ms"]),
                    "cpu_reassembly_ms": float(counters[
                        "candidate_stream_batch_cpu_merge_ms"]),
                    "cpu_consume_ms": float(sum(
                        row[2]["candidate_stream_cpu_consume_ms"]
                        for row in output)),
                })
            median_ms = float(np.median(timings))
            median_index = int(np.argsort(timings)[len(timings) // 2])
            output_ids = [
                np.asarray(row[0], dtype=np.int64) for row in output]
            cold_ids = [np.asarray(row[0], dtype=np.int64) for row in cold]
            recall, complete = fixed_policy_recall(
                cpu_leaf_ids, output_ids)
            cold_recall, _ = fixed_policy_recall(cpu_leaf_ids, cold_ids)
            counters = [row[2] for row in output]
            return {
                "status": "approximate_ann_partition_probe",
                "backend": str(backend),
                "gpu_count": int(gpu_count),
                "devices": list(devices_used),
                "probe_count": int(probe_count),
                "partition_kind": str(partition_kind),
                "active_partition_count": int(partition_count),
                "provider_build_seconds": float(build_seconds),
                "ann_index_build_seconds": float(ann_index_build_seconds),
                "ann_index_manifest": ann_index_manifest,
                "cold_total_ms": float(cold_ms),
                "cold_fixed_policy_recall_at_k": float(cold_recall),
                "steady_total_ms_repeats": [
                    float(value) for value in timings],
                "steady_total_ms": float(median_ms),
                "steady_ms_per_query": float(median_ms / query_count),
                "steady_qps": float(1000.0 * query_count / median_ms),
                "median_repeat_stages": stage_rows[median_index],
                "fixed_policy_recall_at_k": float(recall),
                "queries_returning_reference_k": int(complete),
                "query_count": int(query_count),
                "physical_candidate_rows_mean": float(np.mean([
                    row["physical_distance_evaluations"]
                    for row in counters])),
                "candidate_events_mean": float(np.mean([
                    row["candidate_stream_candidate_events"]
                    for row in counters])),
                "fallback_queries": int(sum(
                    row["candidate_stream_fallback"]
                    for row in counters)),
                "exact_full_scan": False,
                "output_digest": (
                    _sha_json([row.tolist() for row in output_ids])
                    if record_digests else None),
            }

        for probe_count in tuple(int(value) for value in tree_probe_counts):
            if probe_count < 1 or probe_count > model.tree.n_leaves:
                raise ValueError(
                    "tree probe counts must fit the active paper tree")
            build_started = time.perf_counter()
            cpu_probe = NumpyTreeProbeCandidateStreamProvider(
                embeddings, model.tree, probe_count=probe_count,
                result_width=tree_probe_result_width)
            build_seconds = time.perf_counter() - build_started
            tree_probe_measurements.append(measure_tree_probe(
                cpu_probe, backend="cpu", gpu_count=0,
                devices_used=["cpu"], probe_count=probe_count,
                build_seconds=build_seconds,
                partition_kind="paper_projected_tree",
                partition_count=model.tree.n_leaves))
            cpu_probe = None
            gc.collect()

            if devices:
                torch.cuda.empty_cache()
                build_started = time.perf_counter()
                gpu_probe = TorchTreeProbeCandidateStreamProvider(
                    embeddings, model.tree, device=devices[0],
                    probe_count=probe_count,
                    result_width=tree_probe_result_width)
                build_seconds = time.perf_counter() - build_started
                tree_probe_measurements.append(measure_tree_probe(
                    gpu_probe, backend="gpu", gpu_count=1,
                    devices_used=[devices[0]], probe_count=probe_count,
                    build_seconds=build_seconds,
                    partition_kind="paper_projected_tree",
                    partition_count=model.tree.n_leaves))
                gpu_probe = None
                gc.collect()
                torch.cuda.empty_cache()

            if len(devices) > 1:
                build_started = time.perf_counter()

                def build_tree_probe(device):
                    return TorchTreeProbeCandidateStreamProvider(
                        embeddings, model.tree, device=device,
                        probe_count=probe_count,
                        result_width=tree_probe_result_width)

                with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                    replicas = list(pool.map(
                        build_tree_probe, devices))
                multi_probe = TorchQueryShardedExactCandidateStreamProvider(
                    replicas)
                build_seconds = time.perf_counter() - build_started
                tree_probe_measurements.append(measure_tree_probe(
                    multi_probe, backend="gpu", gpu_count=len(devices),
                    devices_used=devices, probe_count=probe_count,
                    build_seconds=build_seconds,
                    partition_kind="paper_projected_tree",
                    partition_count=model.tree.n_leaves))
                multi_probe = None
                replicas = []
                gc.collect()
                torch.cuda.empty_cache()

        ivf_probe_counts = tuple(int(value) for value in ivf_probe_counts)
        if ivf_probe_counts:
            ivf_build_started = time.perf_counter()
            ivf_partitions = FaissIVFPartitionIndex(
                embeddings, nlist=ivf_nlist,
                training_rows=ivf_training_rows,
                iterations=ivf_iterations, seed=seed,
                threads=ivf_build_threads)
            ivf_build_seconds = time.perf_counter() - ivf_build_started
            ivf_manifest = ivf_partitions.manifest()
            for probe_count in ivf_probe_counts:
                if probe_count < 1 or probe_count > ivf_partitions.nlist:
                    raise ValueError(
                        "IVF probe counts must fit the constructed nlist")
                build_started = time.perf_counter()
                cpu_probe = NumpyTreeProbeCandidateStreamProvider(
                    embeddings, model.tree, probe_count=probe_count,
                    result_width=tree_probe_result_width,
                    partition_index=ivf_partitions)
                build_seconds = time.perf_counter() - build_started
                tree_probe_measurements.append(measure_tree_probe(
                    cpu_probe, backend="cpu", gpu_count=0,
                    devices_used=["cpu"], probe_count=probe_count,
                    build_seconds=build_seconds,
                    partition_kind=ivf_partitions.kind,
                    partition_count=ivf_partitions.nlist,
                    ann_index_build_seconds=ivf_build_seconds,
                    ann_index_manifest=ivf_manifest))
                cpu_probe = None
                gc.collect()

                if devices:
                    torch.cuda.empty_cache()
                    build_started = time.perf_counter()
                    gpu_probe = TorchTreeProbeCandidateStreamProvider(
                        embeddings, model.tree, device=devices[0],
                        probe_count=probe_count,
                        result_width=tree_probe_result_width,
                        partition_index=ivf_partitions)
                    build_seconds = time.perf_counter() - build_started
                    tree_probe_measurements.append(measure_tree_probe(
                        gpu_probe, backend="gpu", gpu_count=1,
                        devices_used=[devices[0]],
                        probe_count=probe_count,
                        build_seconds=build_seconds,
                        partition_kind=ivf_partitions.kind,
                        partition_count=ivf_partitions.nlist,
                        ann_index_build_seconds=ivf_build_seconds,
                        ann_index_manifest=ivf_manifest))
                    gpu_probe = None
                    gc.collect()
                    torch.cuda.empty_cache()

                if len(devices) > 1:
                    build_started = time.perf_counter()

                    def build_ivf_probe(device):
                        return TorchTreeProbeCandidateStreamProvider(
                            embeddings, model.tree, device=device,
                            probe_count=probe_count,
                            result_width=tree_probe_result_width,
                            partition_index=ivf_partitions)

                    with ThreadPoolExecutor(
                            max_workers=len(devices)) as pool:
                        replicas = list(pool.map(
                            build_ivf_probe, devices))
                    multi_probe = \
                        TorchQueryShardedExactCandidateStreamProvider(
                            replicas)
                    build_seconds = time.perf_counter() - build_started
                    tree_probe_measurements.append(measure_tree_probe(
                        multi_probe, backend="gpu",
                        gpu_count=len(devices), devices_used=devices,
                        probe_count=probe_count,
                        build_seconds=build_seconds,
                        partition_kind=ivf_partitions.kind,
                        partition_count=ivf_partitions.nlist,
                        ann_index_build_seconds=ivf_build_seconds,
                        ann_index_manifest=ivf_manifest))
                    multi_probe = None
                    replicas = []
                    gc.collect()
                    torch.cuda.empty_cache()

    for row in measurement_rows:
        row["scheduler_steady_total_ms"] = float(scheduler_steady_ms)
        row["scheduler_steady_ms_per_query"] = (
            float(scheduler_steady_ms) / query_count)
        row["stage_composed_e2e_total_ms"] = (
            row["distance_total_ms_median"] + scheduler_steady_ms)
        row["stage_composed_e2e_ms_per_query"] = (
            row["stage_composed_e2e_total_ms"] / query_count)
        row["stage_composed_e2e_qps"] = (
            1000.0 * query_count / row["stage_composed_e2e_total_ms"])
    cpu = next(row for row in measurement_rows if row["backend"] == "cpu")
    one = next((
        row for row in measurement_rows
        if row["backend"] == "gpu" and row["gpu_count"] == 1
    ), None)
    many = measurement_rows[-1] if one is not None else None
    return {
        "status": (
            "controlled-scale-with-same-dimensional-quality-reference"
            if quality_reference else
            "systems-scale-result-not-effectiveness"
        ),
        "corpus": {
            "kind": "books-derived-seeded-repeat-jitter",
            "target_rows": int(len(embeddings)),
            "real_source_rows": int(len(searchable_context["embeddings"])),
            "dimension": int(embeddings.shape[1]),
            "source_row_digest": (
                _digest_array(source_rows) if record_digests else None),
            "source_rows_sha256": (
                _digest_array(source_rows) if record_digests else None),
            "embedding_digest": (
                _digest_array(embeddings) if record_digests else None),
            "embeddings_sha256": (
                _digest_array(embeddings) if record_digests else None),
            "noise_std": float(noise_std),
            "seed": int(expansion_seed),
            "dimension_nested_jitter": bool(
                jitter_reference_embeddings is not None),
            "jitter_reference_dimension": (
                int(jitter_reference_embeddings.shape[1])
                if jitter_reference_embeddings is not None else
                int(embeddings.shape[1])
            ),
            "effectiveness_eligible": bool(quality_reference),
            "effectiveness_scope": (
                "controlled deterministic scale corpus only"
                if quality_reference else "N/A_SYSTEMS_ONLY"
            ),
        },
        "path": {
            "method": PaperMIFA.METHOD_ID,
            "executor": "native fused unified leaf-tree",
            "ann_seed": "ExactBackend reusing precomputed CPU/GPU distances",
            "multi_gpu_parallelism": (
                "replicated corpus with query sharding across GPUs"
                if candidate_multi_gpu_mode == "query" else
                "contiguous corpus sharding across GPUs"),
            "scheduler_location": "CPU",
            "distance_location": "CPU-resident or GPU-resident base with host distance rows",
            "pipeline_overlap": False,
            "selected_executor_backend": executor_backend,
        },
        "queries": {
            "count": int(query_count),
            "row_ids": selected_query_rows.tolist(),
            "row_ids_sha256": (
                _digest_array(selected_query_rows) if record_digests else None),
            "embeddings_sha256": (
                _digest_array(queries) if record_digests else None),
            "workload_digest": (
                workload_digest(cohort) if record_digests else None),
            "k_values": sorted(set(ks)),
        },
        "construction": {
            "expansion_seconds": float(expansion_seconds),
            "model_total_seconds_including_expansion": float(model_build_seconds),
            "paper_mifa": model.construction_timing,
            "memory": scale_memory,
            "provider_build_scope": (
                "reported per measurement row; excluded from online timing"),
        },
        "measurements": measurement_rows,
        "strong_scaling": (None if one is None else {
            "from_gpus": int(one["gpu_count"]),
            "to_gpus": int(many["gpu_count"]),
            "distance_speedup_x": (
                one["distance_total_ms_median"] /
                many["distance_total_ms_median"]),
            "stage_composed_e2e_speedup_x": (
                one["stage_composed_e2e_total_ms"] /
                many["stage_composed_e2e_total_ms"]),
            "scheduler_bottleneck_fraction_two_gpu": (
                scheduler_steady_ms /
                many["stage_composed_e2e_total_ms"]),
            "cpu_to_one_gpu_distance_speedup_x": (
                cpu["distance_total_ms_median"] /
                one["distance_total_ms_median"]),
            "cpu_to_one_gpu_stage_composed_e2e_speedup_x": (
                cpu["stage_composed_e2e_total_ms"] /
                one["stage_composed_e2e_total_ms"]),
        }),
        "distance_consistency": {
            "raw_gpu_distance_digest_match": gpu_distance_digest_match,
            "raw_digest_scope": (
                "diagnostic only; GPU partitioning may differ in low FP32 "
                "bits while preserving ordered fixed-policy IDs"),
            "fixed_policy_ordered_id_agreement": (
                gpu_fixed_policy_ordered_id_agreement),
            "fixed_policy_output_digests": gpu_fixed_policy_output_digests,
            "semantic_gate": "ordered fixed-policy IDs, not raw FP32 bytes",
        },
        "scheduler": {
            "cold_cache_manifest_before": scheduler_cold_cache_manifest,
            "cache_regime": "cold-empty then repeated-predicate steady",
            "cold_total_ms": float(scheduler_cold_ms),
            "steady_total_ms_repeats": [
                float(value) for value in scheduler_steady_repeats],
            "steady_total_ms": float(scheduler_steady_ms),
            "steady_aggregation": "median",
            "output_digest": (
                _sha_json([row.tolist() for row in steady_ids])
                if record_digests else None),
        },
        "cpu_batch1": cpu_batch1,
        "candidate_stream_cpu": candidate_stream_cpu,
        "candidate_stream": candidate_stream,
        "candidate_stream_multi_gpu": candidate_stream_multi_gpu,
        "candidate_response_cpu_batch1": candidate_response_cpu_batch1,
        "candidate_response_batch1": candidate_response_batch1,
        "candidate_batch_scaling": candidate_batch_scaling,
        "quality_reference": scale_quality,
        "tree_probe_measurements": tree_probe_measurements,
        "exact_reference_comparisons": exact_reference_comparisons,
        "pruning_diagnostic": pruning_diagnostic,
    }


def _run_search(model, context, epsilon, progress_callback=None,
                distance_query_batch_size=1, exact_audit_queries=-1,
                retain_prepared_distances=False):
    rows = []
    counter_totals = {}
    latency_ms = []
    exact_matches = []
    exact_mismatches = []
    execution_rows = []
    seen_predicates = set()
    total_queries = len(context["queries"])
    prepared_distances = [None] * total_queries
    preparation_share_ms = np.zeros(total_queries, dtype=np.float64)
    provider = model.executor.distance_provider
    distance_query_batch_size = int(distance_query_batch_size)
    if (provider is not None and distance_query_batch_size > 1 and
            hasattr(provider, "prepare_batch")):
        query_matrix = np.ascontiguousarray(
            [query["q_e"] for query in context["queries"]], dtype=np.float32)
        for batch_start in range(0, total_queries, distance_query_batch_size):
            batch_end = min(
                batch_start + distance_query_batch_size, total_queries)
            start = time.perf_counter()
            batch_distances = provider.prepare_batch(
                query_matrix[batch_start:batch_end],
                batch_size=distance_query_batch_size)
            batch_ms = (time.perf_counter() - start) * 1000.0
            share = batch_ms / (batch_end - batch_start)
            for local, query_row in enumerate(range(batch_start, batch_end)):
                prepared_distances[query_row] = batch_distances[local]
                preparation_share_ms[query_row] = share
    elif provider is not None:
        # Keep the exact distance row used by the scheduler available to the
        # fixed-policy oracle.  Preparation remains inside online latency; it
        # is merely timed separately from the CPU scheduler here.
        for query_row, query in enumerate(context["queries"]):
            start = time.perf_counter()
            prepared_distances[query_row] = provider.prepare(query["q_e"])
            preparation_share_ms[query_row] = (
                time.perf_counter() - start) * 1000.0
    native_batch_outputs = None
    scheduler_share_ms = np.zeros(total_queries, dtype=np.float64)
    native_scheduler_batches = 0
    if (distance_query_batch_size > 1 and
            all(row is not None for row in prepared_distances) and
            hasattr(model.executor, "search_precomputed_batch") and
            model.executor.fused_cpu and model.executor.leaf_frontier and
            not model.executor.allow_distance_order_specialization):
        native_batch_outputs = [None] * total_queries
        for batch_start in range(0, total_queries, distance_query_batch_size):
            batch_end = min(
                batch_start + distance_query_batch_size, total_queries)
            batch_queries = context["queries"][batch_start:batch_end]
            start = time.perf_counter()
            outputs = model.executor.search_precomputed_batch(
                np.ascontiguousarray(
                    [query["q_e"] for query in batch_queries],
                    dtype=np.float32),
                [int(query["k"]) for query in batch_queries],
                [query["phi"] for query in batch_queries],
                np.ascontiguousarray(
                    prepared_distances[batch_start:batch_end],
                    dtype=np.float32))
            batch_ms = (time.perf_counter() - start) * 1000.0
            share = batch_ms / (batch_end - batch_start)
            native_scheduler_batches += 1
            for local, output in enumerate(outputs):
                query_row = batch_start + local
                native_batch_outputs[query_row] = output
                scheduler_share_ms[query_row] = share
    for query_index, query in enumerate(context["queries"], start=1):
        predicate_key = model.inference._predicate_key(query["phi"].atoms)
        predicate_cache_status = (
            "steady" if predicate_key in seen_predicates else "cold")
        seen_predicates.add(predicate_key)
        if native_batch_outputs is not None:
            ids, _, counters = native_batch_outputs[query_index - 1]
            scheduler_ms = float(scheduler_share_ms[query_index - 1])
        else:
            start = time.perf_counter()
            ids, _, counters = model.executor.search(
                query["q_e"], query["k"], query["phi"], epsilon=epsilon,
                precomputed_distances=prepared_distances[query_index - 1])
            scheduler_ms = (time.perf_counter() - start) * 1000.0
        latency_ms.append(
            scheduler_ms + preparation_share_ms[query_index - 1])
        execution_rows.append({
            "query_index": int(query_index - 1),
            "signature": "&".join(sorted(
                str(atom.attr) for atom in query["phi"].atoms)),
            "arity": int(query["arity"]),
            "predicate_type": str(query["predicate_type"]),
            "predicate_cache_status": predicate_cache_status,
            "selectivity_bucket": int(query["selectivity_bucket"]),
            "masked_answer_affected": bool(
                context["masked_answer_affected"][query_index - 1]),
            "latency_ms": float(latency_ms[-1]),
            "scheduler_ms": float(scheduler_ms),
            "distance_preparation_ms": float(
                preparation_share_ms[query_index - 1]),
            "exact_distance_evaluations": int(
                counters.get("exact_distance_evaluations", 0)),
            "physical_distance_evaluations": int(
                counters.get("physical_distance_evaluations", 0)),
            "lazy_inferences": int(counters.get("lazy_inferences", 0)),
            "statistical_resolution_requests": int(counters.get(
                "statistical_resolution_requests",
                counters.get("lazy_inferences", 0) +
                counters.get("statistical_cache_hits", 0))),
            "statistical_resolution_computations": int(counters.get(
                "statistical_resolution_computations",
                counters.get("lazy_inferences", 0))),
            "statistical_resolution_cache_hits": int(counters.get(
                "statistical_resolution_cache_hits",
                counters.get("statistical_cache_hits", 0))),
            "statistical_resolution_cache_misses": int(counters.get(
                "statistical_resolution_cache_misses",
                counters.get("lazy_inferences", 0))),
            "candidate_rows_not_eligibility_checked": int(counters.get(
                "candidate_rows_not_eligibility_checked",
                max(len(model.searchable_embeddings) -
                    int(counters.get("hard_decisions", 0)), 0))),
            "queued_unresolved_resolutions_avoided": int(counters.get(
                "queued_unresolved_resolutions_avoided", 0)),
            "pruned_nodes": int(counters.get("pruned_nodes", 0)),
        })
        padded = np.full(query["k"], -1, dtype=np.int64)
        padded[:len(ids)] = ids
        execution_rows[-1]["returned_ids"] = padded.tolist()
        rows.append(padded)
        for key, value in counters.items():
            counter_totals[key] = counter_totals.get(key, 0) + int(value)

        compare_exact_this_query = (
            epsilon == 0.0 and
            int(exact_audit_queries) != 0 and
            (int(exact_audit_queries) < 0 or
             query_index <= int(exact_audit_queries)))
        if compare_exact_this_query:
            eligible = model.materialize_eligibility(query["phi"])
            candidates = np.flatnonzero(eligible)
            vector = np.asarray(query["q_e"], dtype=np.float32)
            # Compare under the same finite-precision
            # squared-L2 implementation it served.  A broadcast-subtract
            # oracle can round algebraically identical distances differently
            # and report a false ordering mismatch for an unchanged top-k set.
            if prepared_distances[query_index - 1] is not None:
                d2 = prepared_distances[query_index - 1]
            else:
                d2 = np.maximum(
                    model.executor.sqnorm -
                    2.0 * (model.searchable_embeddings @ vector) +
                    float(vector @ vector), 0.0)
            oracle = np.asarray(sorted(
                candidates, key=lambda row: (float(d2[row]), int(row)))[:query["k"]],
                dtype=np.int64)
            oracle_padded = np.full(query["k"], -1, dtype=np.int64)
            oracle_padded[:len(oracle)] = oracle
            exact_match = bool(np.array_equal(padded, oracle_padded))
            exact_matches.append(exact_match)
            if not exact_match:
                direct_d2 = np.einsum(
                    "ij,ij->i", model.searchable_embeddings - vector,
                    model.searchable_embeddings - vector)
                algebraic_d2 = np.maximum(
                    model.executor.sqnorm -
                    2.0 * (model.searchable_embeddings @ vector) +
                    float(vector @ vector), 0.0)
                differing = sorted({
                    int(row)
                    for left, right in zip(padded, oracle_padded)
                    if left != right
                    for row in (left, right)
                })
                detail = {
                    "query_index": int(query_index - 1),
                    "executor_ids": padded.tolist(),
                    "oracle_ids": oracle_padded.tolist(),
                    "differing_ids": [int(row) for row in differing if row >= 0],
                    "direct_d2": {
                        str(int(row)): float(direct_d2[row])
                        for row in differing if row >= 0},
                    "algebraic_d2": {
                        str(int(row)): float(algebraic_d2[row])
                        for row in differing if row >= 0},
                }
                exact_mismatches.append(detail)
                _progress("exact_mismatch", **detail)
        if progress_callback is not None and (
                query_index == total_queries or query_index % 10 == 0):
            progress_callback(
                "test_query_completed", completed=query_index,
                total=total_queries,
                online_latency_ms=float(latency_ms[-1]),
                exact_match=(exact_matches[-1]
                if compare_exact_this_query else None))
    latency = np.asarray(latency_ms, dtype=np.float64)
    def aggregate_execution(selected):
        values = np.asarray(
            [execution_rows[index]["latency_ms"] for index in selected],
            dtype=np.float64)
        return {
            "n_queries": int(len(selected)),
            "latency_ms_mean": float(values.mean()),
            "latency_ms_p50": float(np.quantile(values, 0.50)),
            "latency_ms_p95": float(np.quantile(values, 0.95)),
            "scheduler_ms_mean": float(np.mean([
                execution_rows[index]["scheduler_ms"]
                for index in selected])),
            "distance_preparation_ms_mean": float(np.mean([
                execution_rows[index]["distance_preparation_ms"]
                for index in selected])),
            "physical_distance_evaluations_mean": float(np.mean([
                execution_rows[index]["physical_distance_evaluations"]
                for index in selected])),
            "lazy_inferences_mean": float(np.mean([
                execution_rows[index]["lazy_inferences"]
                for index in selected])),
        }

    execution_strata = {}
    for key in ("signature", "arity", "predicate_type",
                "predicate_cache_status",
                "selectivity_bucket", "masked_answer_affected"):
        execution_strata[key] = {}
        for value in sorted({str(row[key]) for row in execution_rows}):
            selected = [
                index for index, row in enumerate(execution_rows)
                if str(row[key]) == value]
            execution_strata[key][value] = aggregate_execution(selected)
    execution = {
        "queries": int(len(rows)),
        "latency_ms_mean": float(latency.mean()),
        "latency_ms_p50": float(np.quantile(latency, 0.50)),
        "latency_ms_p95": float(np.quantile(latency, 0.95)),
        "counters": counter_totals,
        "executor_exact_matches_fixed_policy": (
            bool(all(exact_matches)) if epsilon == 0.0 else None),
        "executor_exact_audited_queries": int(len(exact_matches)),
        "executor_exact_mismatches": exact_mismatches,
        "distance_query_batch_size": int(distance_query_batch_size),
        "native_query_batch_scheduler": bool(
            native_batch_outputs is not None),
        "native_query_scheduler_batches": int(native_scheduler_batches),
        "scheduler_timing_scope": (
            "native batch wall time amortized across each query batch"
            if native_batch_outputs is not None else
            "per-query scheduler wall time"),
        "distance_preparation_ms_total": float(preparation_share_ms.sum()),
        "cache_protocol": {
            "scope": "immutable deployment snapshot",
            "key": "exact predicate plus selected-template/location/body signature",
            "cold_definition": "first online query for an exact predicate",
            "steady_definition": "subsequent online queries for that exact predicate",
            "cold": execution_strata["predicate_cache_status"].get("cold"),
            "steady": execution_strata["predicate_cache_status"].get("steady"),
        },
        "strata": execution_strata,
        "per_query": execution_rows,
    }
    if retain_prepared_distances:
        # Internal hand-off for execution-order ablations.  The caller must
        # remove these NumPy arrays before JSON serialization.  Reusing the
        # exact rows served by the deployed executor avoids manufacturing a
        # semantic mismatch when a one-query GEMM and a batched GEMM round a
        # boundary distance differently.
        execution["_prepared_distances"] = prepared_distances
        execution["_distance_preparation_share_ms"] = preparation_share_ms
    return np.asarray(rows), execution


def _drop_ids(context):
    if "drop_ids" in context:
        return context["drop_ids"]
    eligible = np.asarray([
        query["phi"].eval_table(context["observed"]) == TRUE
        for query in context["queries"]
    ], dtype=np.bool_)
    return topk_masked_batch_numba(
        context["D2"], eligible, context["queries"][0]["k"])[0]


def _score_completed_table(context, completed):
    """Evaluate one imputation baseline on the already-frozen test input."""
    eligible = np.asarray([
        query["phi"].eval_table(completed) == TRUE
        for query in context["queries"]
    ], dtype=np.bool_)
    ids = topk_masked_batch_numba(
        context["D2"], eligible, context["queries"][0]["k"])[0]
    score = legacy._score_ids(
        ids, context["truth_ids"], context["queries"],
        context.get("headline_admitted"),
        context.get("masked_answer_affected"))
    direct = legacy._score_ids(
        ids, context["truth_ids"], context["queries"],
        masked_answer_affected=context.get("affected_query_direct"))
    if "masked_answer" in direct:
        score["affected_query_direct"] = direct["masked_answer"]
    return score


def _run_aligned_baselines(
        requested, *, spec, parts, train_observed, train_mask,
        query_specs, native_filters, predicate_reference, signature_mix,
        canonical_bucket_weights, role_counts, calibration, test, seed):
    """Run baselines against the exact PaperMIFA table/query/mask objects.

    This deliberately lives in the paper runner: passing the in-memory
    contexts removes the possibility of reconstructing a subtly different
    mask or query role in a second campaign entry point.
    """
    methods = tuple(dict.fromkeys(
        item.strip().lower() for item in requested.split(",")
        if item.strip()))
    allowed = {"mean_mode", "mice", "missforest", "query_cls"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"unknown aligned baselines: {sorted(unknown)}")
    results = {}
    manifest = {
        "same_in_memory_test_context": True,
        "methods": list(methods),
        "implementations": {},
    }
    for name in methods:
        if name == "query_cls":
            continue
        started = time.perf_counter()
        if name == "mean_mode":
            imputer = MeanModeImputer(spec.dtypes).fit(train_observed)
            implementation = "training-fitted mean/mode"
        else:
            imputer = MixedIterativeImputer(
                spec.dtypes, kind=name, seed=seed, max_iter=5,
                n_estimators=50, n_jobs=8).fit(train_observed)
            implementation = (
                "sklearn IterativeImputer/BayesianRidge"
                if name == "mice" else
                "sklearn IterativeImputer/RandomForestRegressor")
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        completed = imputer.transform(test["observed"])
        results[name] = _score_completed_table(test, completed)
        results[name]["fit_seconds"] = float(fit_seconds)
        results[name]["materialization_seconds"] = float(
            time.perf_counter() - started)
        manifest["implementations"][name] = implementation
        _progress(
            "aligned_baseline_completed", method=name,
            f1=results[name]["macro"]["f1"], fit_seconds=fit_seconds)

    if "query_cls" in methods:
        started = time.perf_counter()
        query_spec = query_specs.get("query_cls_training")
        if query_spec is None:
            raise ValueError(
                "aligned QueryCls requires a frozen query_cls_training role")
        query_count = int(role_counts["query_cls_training"])
        training_queries = legacy._make_workload(
            query_spec, parts["train"], query_count,
            test["queries"][0]["k"], seed=seed * 100 + 72,
            attrs=list(spec.query_relevant_attrs),
            predicate_reference=predicate_reference,
            native_filters=native_filters.get("query_cls_training"),
            signature_mix=signature_mix,
            canonical_bucket_weights=canonical_bucket_weights)
        projector = legacy.CoarseCellIndex(
            pca_dim=min(16, parts["train"].d), n_cells=256,
            seed=seed).build(parts["train"].normalized_embeddings()).pca_
        classifier = QueryConditionedClassifier(
            spec.dtypes, projector, seed=seed).fit(
                parts["train"].table, train_observed,
                parts["train"].normalized_embeddings(), training_queries)
        theta, curve = legacy._calibrate_query_cls(calibration, classifier)
        fit_seconds = time.perf_counter() - started
        probabilities = classifier.probabilities(
            test["observed"], test["embeddings"], test["queries"])
        eligible = np.asarray(
            classifier.eligible(probabilities, theta), dtype=np.bool_)
        ids = topk_masked_batch_numba(
            test["D2"], eligible, test["queries"][0]["k"])[0]
        results["query_cls"] = legacy._score_ids(
            ids, test["truth_ids"], test["queries"],
            test.get("headline_admitted"),
            test.get("masked_answer_affected"))
        direct = legacy._score_ids(
            ids, test["truth_ids"], test["queries"],
            masked_answer_affected=test.get("affected_query_direct"))
        if "masked_answer" in direct:
            results["query_cls"]["affected_query_direct"] = \
                direct["masked_answer"]
        results["query_cls"]["fit_seconds"] = float(fit_seconds)
        results["query_cls"]["theta"] = float(theta)
        manifest["implementations"]["query_cls"] = (
            "mifa.inference.QueryConditionedClassifier")
        manifest["query_cls"] = {
            **classifier.manifest,
            "theta": float(theta),
            "calibration_curve": curve,
            "train_mask_is_paper_train_mask": bool(
                train_mask is not None),
        }
        _progress(
            "aligned_baseline_completed", method="query_cls",
            f1=results["query_cls"]["macro"]["f1"],
            fit_seconds=fit_seconds)
    return results, manifest


def _executor_ablation(
        model, context, reference_rows, n_queries, *,
        prepared_distances=None, distance_preparation_ms=None,
        query_workers=1):
    """Compare the three E2-2 execution orders on one fixed decision policy.

    ``imputation_first`` eagerly resolves every tuple before it computes query
    distances.  ``distance_first`` computes the complete distance order and
    resolves only the prefix needed for top-k.  ``reference_mifa`` uses the
    same unfused executor object, inference cache, distance row, and exact tie
    rule, but interleaves distance and eligibility work through the tree
    frontier.  The deployed ``mifa`` row separately exposes the physical
    compact/fused-kernel gain.
    """
    n_queries = min(int(n_queries), len(context["queries"]))
    if n_queries <= 0:
        return None
    provider = model.executor.distance_provider
    if provider is None:
        provider = NumpyResidentDistanceProvider(model.searchable_embeddings)
    query_workers = min(max(1, int(query_workers)), n_queries)
    if query_workers > 1 and prepared_distances is None:
        raise ValueError(
            "parallel executor ablation requires deployed distance rows")
    methods = {
        "imputation_first": [],
        "distance_first": [],
        "reference_mifa": [],
    }
    work = {name: {
                "evaluated_tuples": 0,
                "lazy_inferences": 0,
                "resolution_requests": 0,
                "resolution_computations": 0,
                "resolution_cache_hits": 0,
                "resolution_cache_misses": 0,
                "candidate_rows_not_eligibility_checked": 0,
                "queued_unresolved_resolutions_avoided": 0,
                "exact_matches": 0,
                "set_recall": [],
                "mismatches": [],
            }
            for name in methods}

    if prepared_distances is not None:
        if len(prepared_distances) < n_queries:
            raise ValueError("prepared distance rows do not cover ablation")
        if any(row is None for row in prepared_distances[:n_queries]):
            raise ValueError("prepared distance rows contain missing entries")
    if distance_preparation_ms is not None:
        distance_preparation_ms = np.asarray(
            distance_preparation_ms, dtype=np.float64)
        if len(distance_preparation_ms) < n_queries:
            raise ValueError("distance timing rows do not cover ablation")

    def finish(name, query_index, latency_ms, returned, reference, evaluated,
               inferred, counters):
        methods[name].append(float(latency_ms))
        work[name]["evaluated_tuples"] += int(evaluated)
        work[name]["lazy_inferences"] += int(inferred)
        requests = int(counters.get(
            "statistical_resolution_requests", inferred))
        cache_hits = int(counters.get(
            "statistical_resolution_cache_hits", 0))
        if cache_hits < 0 or cache_hits > requests:
            raise AssertionError(
                f"invalid resolution cache accounting for {name}")
        computations = int(counters.get(
            "statistical_resolution_computations",
            requests - cache_hits))
        if computations < 0 or computations > requests:
            raise AssertionError(
                f"invalid resolution computation count for {name}")
        work[name]["resolution_requests"] += requests
        work[name]["resolution_computations"] += computations
        work[name]["resolution_cache_hits"] += cache_hits
        work[name]["resolution_cache_misses"] += int(counters.get(
            "statistical_resolution_cache_misses", computations))
        work[name]["candidate_rows_not_eligibility_checked"] += int(
            counters.get("candidate_rows_not_eligibility_checked", 0))
        work[name]["queued_unresolved_resolutions_avoided"] += int(
            counters.get("queued_unresolved_resolutions_avoided", 0))
        exact = bool(np.array_equal(returned, reference))
        work[name]["exact_matches"] += int(exact)
        if not exact and len(work[name]["mismatches"]) < 20:
            differing = sorted(set(returned.tolist()) ^ set(reference.tolist()))
            work[name]["mismatches"].append({
                "query_index": int(query_index),
                "returned_ids": returned.tolist(),
                "reference_ids": reference.tolist(),
                "differing_ids": [int(value) for value in differing],
            })
        if len(reference) == 0:
            # The complete-data top-k semantics return the true available
            # cardinality.  Two empty result sets are therefore an exact
            # semantic match, not zero recall.
            set_recall = float(len(returned) == 0)
        else:
            set_recall = len(set(returned) & set(reference)) / len(reference)
        work[name]["set_recall"].append(set_recall)

    reference_executor = None
    if hasattr(model, "tree"):
        reference_executor = UnifiedBestFirstExecutor(
            model.searchable_embeddings, model.tree, model.inference,
            ann_backend=None, seed_size=0, fused_cpu=False,
            leaf_frontier=False, distance_provider=provider,
            allow_distance_order_specialization=False)
    worker_state = {
        "model": model,
        "context": context,
        "reference_rows": reference_rows,
        "prepared_distances": prepared_distances,
        "distance_preparation_ms": distance_preparation_ms,
        "reference_executor": reference_executor,
    }
    previous_worker_state = _EXECUTOR_ABLATION_WORKER_STATE
    try:
        _set_executor_ablation_worker_state(worker_state)
        if query_workers == 1:
            query_results = [
                _executor_ablation_query(query_index)
                for query_index in range(n_queries)
            ]
        else:
            if "fork" not in multiprocessing.get_all_start_methods():
                raise RuntimeError(
                    "parallel executor ablation requires POSIX fork")
            context_fork = multiprocessing.get_context("fork")
            with context_fork.Pool(processes=query_workers) as pool:
                query_results = list(pool.imap(
                    _executor_ablation_query, range(n_queries), chunksize=1))
    finally:
        _set_executor_ablation_worker_state(previous_worker_state)

    for query_index, result in enumerate(query_results):
        reference = result["reference"]
        for name in (
                "imputation_first", "distance_first", "reference_mifa"):
            row = result[name]
            finish(
                name, query_index, row["latency_ms"], row["returned"],
                reference, row["evaluated"], row["inferred"],
                row.get("counters", {}))

    output = {
        "queries": n_queries,
        "query_workers": query_workers,
        "query_parallelism": (
            "isolated_process_per_query_pool"
            if query_workers > 1 else "serial_queries"),
        "fixed_policy": True,
        "distance_preparation_included": True,
        "distance_rows_shared_with_deployed_mifa": bool(
            prepared_distances is not None),
        "timing_composition": (
            "strategy wall time plus deployed batch distance-preparation "
            "share per query" if prepared_distances is not None else
            "strategy wall time including per-query distance preparation"),
        "experiment_id": "E2-2",
        "variants": [
            "mifa", "reference_mifa", "imputation_first", "distance_first"],
        "inference_first_alias_allowed": False,
        "reference_physical_executor": (
            "UnifiedBestFirstExecutor" if reference_executor is not None
            else "lightweight_test_fallback"),
        "reference_cache_protocol": {
            "scope": "one strategy and one query",
            "regime": "cold reset before each timed strategy query",
            "reset_outside_timed_region": True,
        },
        "methods": {},
    }
    imputation_requests = work["imputation_first"]["resolution_requests"]
    imputation_computations = work[
        "imputation_first"]["resolution_computations"]
    for name, latencies in methods.items():
        values = np.asarray(latencies, dtype=np.float64)
        avoided_requests = imputation_requests - work[name][
            "resolution_requests"]
        avoided_computations = imputation_computations - work[name][
            "resolution_computations"]
        if avoided_requests < 0 or avoided_computations < 0:
            raise AssertionError(
                f"{name} performed more eligibility work than the eager "
                "imputation-first oracle")
        output["methods"][name] = {
            "latency_ms_mean": float(values.mean()),
            "latency_ms_p50": float(np.quantile(values, 0.50)),
            "latency_ms_p95": float(np.quantile(values, 0.95)),
            "evaluated_tuples_mean": (
                work[name]["evaluated_tuples"] / n_queries),
            "lazy_inferences_mean": work[name]["lazy_inferences"] / n_queries,
            "resolution_requests_mean": (
                work[name]["resolution_requests"] / n_queries),
            "statistical_resolutions_mean": (
                work[name]["resolution_computations"] / n_queries),
            "resolution_cache_hits_mean": (
                work[name]["resolution_cache_hits"] / n_queries),
            "resolution_cache_misses_mean": (
                work[name]["resolution_cache_misses"] / n_queries),
            "resolution_requests_avoided_vs_imputation_first_mean": (
                avoided_requests / n_queries),
            "statistical_resolutions_avoided_vs_imputation_first_mean": (
                avoided_computations / n_queries),
            "candidate_rows_not_eligibility_checked_mean": (
                work[name]["candidate_rows_not_eligibility_checked"] /
                n_queries),
            "queued_unresolved_resolutions_avoided_mean": (
                work[name]["queued_unresolved_resolutions_avoided"] /
                n_queries),
            "exact_matches": int(work[name]["exact_matches"]),
            "set_recall_mean": float(np.mean(work[name]["set_recall"])),
            "semantic_equivalence": bool(
                work[name]["exact_matches"] == n_queries and
                np.allclose(work[name]["set_recall"], 1.0)),
            "mismatches": work[name]["mismatches"],
        }
    mifa_latency = np.asarray([
        row["latency_ms"] for row in reference_rows[:n_queries]],
        dtype=np.float64)
    output["methods"]["mifa"] = {
        "latency_ms_mean": float(mifa_latency.mean()),
        "latency_ms_p50": float(np.quantile(mifa_latency, 0.50)),
        "latency_ms_p95": float(np.quantile(mifa_latency, 0.95)),
        "evaluated_tuples_mean": float(np.mean([
            row["physical_distance_evaluations"]
            for row in reference_rows[:n_queries]])),
        "lazy_inferences_mean": float(np.mean([
            row["lazy_inferences"] for row in reference_rows[:n_queries]])),
        "resolution_requests_mean": float(np.mean([
            row.get(
                "statistical_resolution_requests",
                row["lazy_inferences"] +
                row.get("statistical_resolution_cache_hits", 0))
            for row in reference_rows[:n_queries]])),
        "statistical_resolutions_mean": float(np.mean([
            row.get("statistical_resolution_computations",
                    row["lazy_inferences"])
            for row in reference_rows[:n_queries]])),
        "resolution_cache_hits_mean": float(np.mean([
            row.get("statistical_resolution_cache_hits", 0)
            for row in reference_rows[:n_queries]])),
        "resolution_cache_misses_mean": float(np.mean([
            row.get("statistical_resolution_cache_misses",
                    row.get("statistical_resolution_computations",
                            row["lazy_inferences"]))
            for row in reference_rows[:n_queries]])),
        "candidate_rows_not_eligibility_checked_mean": float(np.mean([
            row.get("candidate_rows_not_eligibility_checked", 0)
            for row in reference_rows[:n_queries]])),
        "queued_unresolved_resolutions_avoided_mean": float(np.mean([
            row.get("queued_unresolved_resolutions_avoided", 0)
            for row in reference_rows[:n_queries]])),
        "exact_matches": n_queries,
        "set_recall_mean": 1.0,
        "semantic_equivalence": True,
    }
    deployed_requests = output["methods"]["mifa"][
        "resolution_requests_mean"]
    deployed_computations = output["methods"]["mifa"][
        "statistical_resolutions_mean"]
    eager_requests_mean = imputation_requests / n_queries
    eager_computations_mean = imputation_computations / n_queries
    deployed_avoided_requests = eager_requests_mean - deployed_requests
    deployed_avoided_computations = (
        eager_computations_mean - deployed_computations)
    if (deployed_avoided_requests < 0 or
            deployed_avoided_computations < 0):
        raise AssertionError(
            "deployed MIFA performed more eligibility work than the eager "
            "imputation-first oracle")
    output["methods"]["mifa"][
        "resolution_requests_avoided_vs_imputation_first_mean"] = (
            deployed_avoided_requests)
    output["methods"]["mifa"][
        "statistical_resolutions_avoided_vs_imputation_first_mean"] = (
            deployed_avoided_computations)
    reference_method = output["methods"]["reference_mifa"]
    distance_method = output["methods"]["distance_first"]
    reference_request_reduction = (
        reference_method[
            "resolution_requests_avoided_vs_imputation_first_mean"] /
        eager_requests_mean if eager_requests_mean > 0 else 0.0)
    reference_computation_reduction = (
        reference_method[
            "statistical_resolutions_avoided_vs_imputation_first_mean"] /
        eager_computations_mean if eager_computations_mean > 0 else 0.0)
    output["mechanism_isolation"] = {
        "reference_variants_share_physical_executor": bool(
            reference_executor is not None),
        "reference_variants_share_cache_protocol": True,
        "reference_variants_share_precomputed_distance_rows": bool(
            prepared_distances is not None),
        "reference_variants_tie_rule": "ascending (distance, tuple_id)",
        "reference_mifa_p95_speedup_vs_distance_first_x": float(
            distance_method["latency_ms_p95"] /
            reference_method["latency_ms_p95"]),
        "reference_mifa_resolution_request_reduction_vs_imputation_first": (
            float(reference_request_reduction)),
        "reference_mifa_statistical_computation_reduction_vs_imputation_first": (
            float(reference_computation_reduction)),
        "compact_fused_p95_speedup_vs_reference_mifa_x": float(
            reference_method["latency_ms_p95"] /
            output["methods"]["mifa"]["latency_ms_p95"]),
        "compact_fused_computation_reduction_vs_reference_mifa": float(
            ((reference_method["statistical_resolutions_mean"] -
              output["methods"]["mifa"]["statistical_resolutions_mean"]) /
             reference_method["statistical_resolutions_mean"])
            if reference_method["statistical_resolutions_mean"] > 0
            else 0.0),
        "attribution": {
            "reference_mifa_vs_distance_first": (
                "execution timing under one unfused physical executor"),
            "reference_mifa_vs_imputation_first": (
                "eligibility requests and actual computations avoided under "
                "one unfused physical executor"),
            "compact_fused_vs_reference_mifa": (
                "physical kernel and deployment-cache gain; not credited to "
                "execution-order isolation"),
        },
    }
    return output


_EXECUTOR_ABLATION_WORKER_STATE = None


def _set_executor_ablation_worker_state(state):
    global _EXECUTOR_ABLATION_WORKER_STATE
    _EXECUTOR_ABLATION_WORKER_STATE = state


def _executor_ablation_query(query_index):
    """Run both fixed-policy strategy orders for one independent query."""
    state = _EXECUTOR_ABLATION_WORKER_STATE
    if state is None:
        raise RuntimeError("executor ablation worker state is not initialized")
    model = state["model"]
    context = state["context"]
    reference_rows = state["reference_rows"]
    prepared_distances = state["prepared_distances"]
    distance_preparation_ms = state["distance_preparation_ms"]
    reference_executor = state["reference_executor"]

    query_index = int(query_index)
    query = context["queries"][query_index]
    vector = np.ascontiguousarray(query["q_e"], dtype=np.float32)
    predicate = query["phi"]
    k = int(query["k"])
    reference = np.asarray(
        reference_rows[query_index]["returned_ids"], dtype=np.int64)
    reference = reference[reference >= 0]
    provider = model.executor.distance_provider
    if provider is None:
        provider = NumpyResidentDistanceProvider(model.searchable_embeddings)
    shared_distances = (
        None if prepared_distances is None else np.ascontiguousarray(
            prepared_distances[query_index], dtype=np.float32))
    preparation_ms = (
        float(distance_preparation_ms[query_index])
        if distance_preparation_ms is not None else 0.0)

    if reference_executor is not None:
        strategy_rows = {}
        for name, strategy in (
                ("imputation_first", "imputation_first"),
                ("distance_first", "distance_first"),
                ("reference_mifa", None)):
            # Each strategy/query starts from the same cold serving state.
            # Resetting outside the timer avoids both cross-strategy warm-up
            # bias and charging cache administration to one execution order.
            if hasattr(model.inference, "clear_all_serving_caches"):
                model.inference.clear_all_serving_caches()
            started = time.perf_counter()
            returned, _, counters = reference_executor.search(
                vector, k, predicate, epsilon=0.0,
                precomputed_distances=shared_distances,
                reference_strategy=strategy)
            latency_ms = (
                (time.perf_counter() - started) * 1000.0 + preparation_ms)
            strategy_rows[name] = {
                "latency_ms": latency_ms,
                "returned": np.asarray(returned, dtype=np.int64),
                "evaluated": int(counters.get(
                    "visited_tuples",
                    counters.get("exact_distance_evaluations", 0))),
                "inferred": int(counters.get(
                    "statistical_resolution_requests",
                    counters.get("lazy_inferences", 0))),
                "counters": counters,
            }
        return {"reference": reference, **strategy_rows}

    # Lightweight unit-test doubles do not carry a materialized tree.  This
    # fallback preserves the public report schema; formal experiment reports
    # always take the single-executor path above.
    started = time.perf_counter()
    model.inference.begin_query()
    eligible = np.zeros(len(model.searchable_embeddings), dtype=np.bool_)
    full_inferred = 0
    for tuple_id in range(len(eligible)):
        decision, _ = model.inference.hard_decision(tuple_id, predicate)
        if decision == "pass":
            eligible[tuple_id] = True
        elif decision == "unresolved":
            full_inferred += 1
            eligible[tuple_id] = model.inference.evaluate(
                tuple_id, predicate, record=False,
                assume_unresolved=True).eligible
    distances = (shared_distances if shared_distances is not None else
                 provider.prepare(vector))
    candidate_ids = np.flatnonzero(eligible)
    order = np.lexsort((candidate_ids, distances[candidate_ids]))
    imputation_returned = candidate_ids[order[:k]]
    imputation_latency = (
        (time.perf_counter() - started) * 1000.0 + preparation_ms)

    started = time.perf_counter()
    distances = (shared_distances if shared_distances is not None else
                 provider.prepare(vector))
    ranked = np.lexsort((
        np.arange(len(distances), dtype=np.int64), distances))
    model.inference.begin_query()
    chosen = []
    inferred = 0
    evaluated = 0
    for tuple_id in ranked:
        evaluated += 1
        decision, _ = model.inference.hard_decision(int(tuple_id), predicate)
        if decision == "pass":
            chosen.append(int(tuple_id))
        elif decision == "unresolved":
            inferred += 1
            if model.inference.evaluate(
                    int(tuple_id), predicate, record=False,
                    assume_unresolved=True).eligible:
                chosen.append(int(tuple_id))
        if len(chosen) == k:
            break
    distance_returned = np.asarray(chosen, dtype=np.int64)
    distance_latency = (
        (time.perf_counter() - started) * 1000.0 + preparation_ms)
    reference_returned = distance_returned.copy()
    reference_latency = distance_latency
    reference_evaluated = evaluated
    reference_inferred = inferred
    imputation_counters = {
        "statistical_resolution_requests": int(full_inferred),
        "statistical_resolution_computations": int(full_inferred),
        "statistical_resolution_cache_hits": 0,
        "candidate_rows_not_eligibility_checked": 0,
    }
    distance_counters = {
        "statistical_resolution_requests": int(inferred),
        "statistical_resolution_computations": int(inferred),
        "statistical_resolution_cache_hits": 0,
        "candidate_rows_not_eligibility_checked": int(
            len(model.searchable_embeddings) - evaluated),
    }
    return {
        "reference": reference,
        "imputation_first": {
            "latency_ms": imputation_latency,
            "returned": imputation_returned,
            "evaluated": len(distances),
            "inferred": full_inferred,
            "counters": imputation_counters,
        },
        "distance_first": {
            "latency_ms": distance_latency,
            "returned": distance_returned,
            "evaluated": evaluated,
            "inferred": inferred,
            "counters": distance_counters,
        },
        "reference_mifa": {
            "latency_ms": reference_latency,
            "returned": np.asarray(reference_returned, dtype=np.int64),
            "evaluated": reference_evaluated,
            "inferred": reference_inferred,
            "counters": dict(distance_counters),
        },
    }


def _support_efficiency_benchmark(model, context, max_requests, repeats=5):
    """Compare E2-10 on a trace and its full reachable list universe."""
    max_requests = int(max_requests)
    if max_requests <= 0:
        return None
    inference = model.inference
    index = inference.support_index
    requests = []
    universe_requests = {}
    satisfied_cache = {}

    def residual_key(residual):
        return tuple((atom.attr, atom.op, repr(atom.a),
                      repr(getattr(atom, "b", None)))
                     for atom in residual)

    def materialized_key(request):
        template = request["template"]
        tuple_id = request["tuple_id"]
        body_key = []
        for body in template.body:
            level_slot = int(request["levels"][body.attr])
            value = inference._target_columns[body.attr][tuple_id]
            body_key.append((body.attr, level_slot, repr(value)))
        return (
            str(template.template_id), tuple(template.targets),
            tuple(body_key), int(request["node_id"]),
        )

    for query_index, query in enumerate(context["queries"]):
        ranked = np.lexsort((
            np.arange(len(context["D2"][query_index]), dtype=np.int64),
            context["D2"][query_index]))
        for tuple_id in ranked:
            state, residual = inference.hard_decision(
                int(tuple_id), query["phi"])
            if state != "unresolved":
                continue
            template = inference.policy.select_indexed(
                (atom.attr for atom in residual),
                inference._target_columns, int(tuple_id))
            if template is None:
                continue
            predicate_key = residual_key(residual)
            satisfied = satisfied_cache.get(predicate_key)
            if satisfied is None:
                satisfied = index.predicate_satisfaction_mask(residual)
                satisfied_cache[predicate_key] = satisfied
            group = (query_index, int(tuple_id), predicate_key)
            for node_id, levels in inference._contexts(template, int(tuple_id)):
                core_request = {
                    "group": group,
                    "template": template,
                    "tuple_id": int(tuple_id),
                    "node_id": int(node_id),
                    "levels": dict(levels),
                    "predicate_key": predicate_key,
                }
                key = materialized_key(core_request)
                universe_requests.setdefault(key, core_request)
                if len(requests) < max_requests:
                    requests.append({
                        **core_request,
                        "satisfied": satisfied,
                    })
    if not requests:
        raise RuntimeError("E2-10 found no real unresolved support requests")
    if len(requests) < max_requests:
        raise RuntimeError(
            f"E2-10 found only {len(requests)} requests; required "
            f"{max_requests}")

    def scan_ids(request):
        template = request["template"]
        tuple_id = request["tuple_id"]
        keep = np.ones(len(index.table), dtype=np.bool_)
        for attr in template.targets:
            keep &= index.availability_mask[attr]
        for body in template.body:
            level_slot = int(request["levels"][body.attr])
            level = body.levels[level_slot]
            value = inference._target_columns[body.attr][tuple_id]
            keep &= index.availability_mask[body.attr]
            if level.kind == "eq":
                keep &= index.columns[body.attr] == value
            elif level.kind == "radius":
                values = np.asarray(index.columns[body.attr], dtype=np.float64)
                keep &= values >= float(value) - float(level.radius)
                keep &= values <= float(value) + float(level.radius)
            elif level.kind != "all":
                raise ValueError(f"unsupported scalar level {level.kind}")
        node = index.tree.nodes[request["node_id"]]
        keep &= index.leaf_ids >= int(node.leaf_lo)
        keep &= index.leaf_ids <= int(node.leaf_hi)
        key = materialized_key(request)
        return np.flatnonzero(keep).astype(np.int64), key

    materialize_started = time.perf_counter()
    materialized = {}
    request_keys = []
    for key, request in universe_requests.items():
        ids, key = scan_ids(request)
        materialized[key] = ids
    for request in requests:
        key = materialized_key(request)
        if key not in materialized:
            raise AssertionError(
                "serving request escaped the materialized reachable universe")
        request_keys.append(key)
    materialize_ms = (time.perf_counter() - materialize_started) * 1000.0

    def run_indexed():
        output = []
        group_masks = {}
        packed = {}
        for request in requests:
            group_cache = group_masks.setdefault(request["group"], {})
            predicate_key = request["predicate_key"]
            satisfied_packed = packed.get(predicate_key)
            if satisfied_packed is None:
                satisfied_packed = index.pack_mask(request["satisfied"])
                packed[predicate_key] = satisfied_packed
            output.append(index.count_joint_indexed(
                request["template"], inference._target_columns,
                request["tuple_id"], request["satisfied"],
                node_id=request["node_id"],
                level_by_attr=request["levels"], mask_cache=group_cache,
                cache_body=True, record_stats=False,
                satisfied_packed=satisfied_packed))
        return output

    def run_scan():
        output = []
        for request in requests:
            ids, _ = scan_ids(request)
            output.append((
                int(len(ids)),
                int(np.count_nonzero(request["satisfied"][ids]))))
        return output

    def run_materialized():
        output = []
        for request, key in zip(requests, request_keys):
            ids = materialized[key]
            output.append((
                int(len(ids)),
                int(np.count_nonzero(request["satisfied"][ids]))))
        return output

    reference = run_scan()
    indexed = run_indexed()
    prebuilt = run_materialized()
    if indexed != reference or prebuilt != reference:
        raise AssertionError("E2-10 support variants returned different counts")
    variants = {}
    for name, function in (
            ("I_mv_shared_index", run_indexed),
            ("scan_D_tr", run_scan),
            ("materialized_template_node_lists", run_materialized)):
        samples = []
        for _ in range(max(int(repeats), 1)):
            started = time.perf_counter()
            result = function()
            samples.append((time.perf_counter() - started) * 1000.0)
            if result != reference:
                raise AssertionError(f"E2-10 {name} replay drifted")
        variants[name] = {
            "total_lookup_ms_median": float(np.median(samples)),
            "lookup_us_per_request": float(
                1000.0 * np.median(samples) / len(requests)),
            "timing_repeats": int(len(samples)),
            "counts_match": True,
        }
    variants["I_mv_shared_index"].update({
        "resident_bytes": int(index.resident_bytes),
        "offline_build_ms": float(
            model.construction_timing["phase_seconds"]["support_index"] * 1000.0
        ),
        "offline_build_scope": (
            "measured SharedSupportIndex construction in the same formal process"
        ),
    })
    variants["scan_D_tr"].update({
        "resident_bytes": 0,
        "working_mask_bytes": int(len(index.table)),
        "offline_build_ms": 0.0,
    })
    variants["materialized_template_node_lists"].update({
        "resident_bytes": int(
            sys.getsizeof(materialized) + sum(
                sys.getsizeof(key) + sys.getsizeof(ids)
                for key, ids in materialized.items())),
        "numeric_payload_bytes": int(sum(
            ids.nbytes for ids in materialized.values())),
        "offline_build_ms": float(materialize_ms),
        "materialized_lists": int(len(materialized)),
        "materialization_scope": (
            "all unique deployment-reachable template-body-level-node lists "
            "over the complete frozen query trace; online timing replays the "
            "first max_requests requests only"),
    })
    return {
        "experiment_id": "E2-10",
        "requests": int(len(requests)),
        "complete_trace_reachable_lists": int(len(universe_requests)),
        "complete_trace_queries": int(len(context["queries"])),
        "unique_predicates": int(len(satisfied_cache)),
        "support_rows": int(len(index.table)),
        "same_support_counts": True,
        "variants": variants,
    }


def _grouped_exact_benchmark(model, context, reference_ids, backends,
                             gpu_device, repeats):
    """Measure semantic-preserving cross-query predicate deduplication."""
    backends = tuple(item.strip() for item in backends.split(",")
                     if item.strip())
    if not backends:
        return None
    unknown = set(backends) - {"cpu", "torch"}
    if unknown:
        raise ValueError(f"unknown grouped exact backends: {sorted(unknown)}")
    repeats = max(int(repeats), 1)
    predicates = [query["phi"] for query in context["queries"]]
    queries = np.ascontiguousarray(
        [query["q_e"] for query in context["queries"]], dtype=np.float32)
    k = int(context["queries"][0]["k"])
    base = model.searchable_embeddings
    base_sqnorm = np.einsum("ij,ij->i", base, base)

    inference_samples = []
    cache = None
    bank = None
    group_ids = None
    for _ in range(repeats):
        cache = PaperGroupedEligibilityCache(model.inference)
        started = time.perf_counter()
        bank, group_ids = cache.eligible_grouped(predicates)
        inference_samples.append((time.perf_counter() - started) * 1000.0)
    lookup_samples = []
    for _ in range(max(repeats, 5)):
        started = time.perf_counter()
        lookup_ids = PaperGroupedEligibilityCache.group_ids(predicates)
        lookup_samples.append((time.perf_counter() - started) * 1000.0)
    if not np.array_equal(group_ids, lookup_ids):
        raise AssertionError("cold and steady grouped predicate IDs disagree")

    output = {
        "queries": int(len(queries)),
        "unique_predicates": int(len(bank)),
        "predicate_compression_ratio": float(len(queries) / len(bank)),
        "eligibility_bank_bytes": int(bank.nbytes),
        "candidate_rows": int(bank.sum()),
        "cold_inference_ms_median": float(np.median(inference_samples)),
        "steady_predicate_lookup_ms_median": float(np.median(lookup_samples)),
        "cache_stats": cache.stats.as_dict(),
        "backends": {},
    }

    strategies = []
    for backend in backends:
        strategies.extend((f"{backend}_masked", f"{backend}_selected"))
    for strategy in strategies:
        started = time.perf_counter()
        if strategy == "cpu_selected":
            searcher = CpuSelectedGroupedTopK(base, base_sqnorm)
        elif strategy == "torch_selected":
            searcher = TorchResidentSelectedTopK(
                base, base_sqnorm, device=gpu_device)
        elif strategy == "torch_masked":
            searcher = TorchResidentMaskedTopK(
                base, base_sqnorm, device=gpu_device)
        else:
            searcher = None
        construction_ms = (time.perf_counter() - started) * 1000.0
        install_samples = []
        if searcher is not None:
            for _ in range(repeats):
                started = time.perf_counter()
                searcher.install_eligibility(bank)
                install_samples.append(
                    (time.perf_counter() - started) * 1000.0)
        else:
            install_samples = [0.0]

        def search(current_queries, current_groups):
            if strategy == "cpu_masked":
                return cpu_blocked_grouped_masked_topk(
                    current_queries, base, base_sqnorm, bank, current_groups,
                    k, query_batch=512, row_block=8192)[0]
            result = searcher.search(current_queries, current_groups, k)
            return result[0] if isinstance(result, tuple) else result
        search(queries[:1], group_ids[:1])
        search_samples = []
        returned = None
        for _ in range(max(repeats, 3)):
            started = time.perf_counter()
            returned = search(queries, group_ids)
            search_samples.append((time.perf_counter() - started) * 1000.0)
        returned = np.asarray(returned, dtype=np.int64)
        exact_rows = np.all(returned == reference_ids, axis=1)
        set_recalls = []
        for expected, actual in zip(reference_ids, returned):
            expected_set = {int(value) for value in expected if value >= 0}
            actual_set = {int(value) for value in actual if value >= 0}
            set_recalls.append(
                1.0 if not expected_set else
                len(expected_set & actual_set) / len(expected_set))
        install_ms = float(np.median(install_samples))
        search_ms = float(np.median(search_samples))
        cold_total_ms = (
            output["cold_inference_ms_median"] + install_ms + search_ms)
        steady_total_ms = (
            output["steady_predicate_lookup_ms_median"] + search_ms)
        resident_bytes = (
            int(base.nbytes + base_sqnorm.nbytes + bank.nbytes)
            if searcher is None else int(searcher.resident_bytes))
        output["backends"][strategy] = {
            "implementation": {
                "cpu_masked": "cpu_blocked_blas_numba_grouped_bank",
                "cpu_selected": "cpu_selected_candidate_blas_numba",
                "torch_masked": "torch_resident_grouped_bitmap_gemm",
                "torch_selected": "torch_resident_selected_candidate_gemm",
            }[strategy],
            "construction_ms": float(construction_ms),
            "eligibility_install_ms_median": install_ms,
            "search_ms_median": search_ms,
            "cold_end_to_end_ms": float(cold_total_ms),
            "cold_latency_ms_per_query": float(cold_total_ms / len(queries)),
            "cold_qps": float(len(queries) * 1000.0 / cold_total_ms),
            "steady_end_to_end_ms": float(steady_total_ms),
            "steady_latency_ms_per_query": float(
                steady_total_ms / len(queries)),
            "steady_qps": float(len(queries) * 1000.0 / steady_total_ms),
            "ordered_exact_queries": int(exact_rows.sum()),
            "ordered_exact": bool(exact_rows.all()),
            "set_recall_mean": float(np.mean(set_recalls)),
            "resident_bytes": resident_bytes,
        }
    return output


def _context_manifest(context):
    manifest = {
        "base_n": int(context["base"].n),
        "query_n": int(len(context["queries"])),
        "mask_digest": context["mask_digest"],
        "workload_digest": context["workload_digest"],
        "workload_counts": context["workload_counts"],
        "incomplete_tuple_rate": context["incomplete_tuple_rate"],
        "masked_cell_rate": context["masked_cell_rate"],
        "query_source_corpus": context["query_source_corpus"],
        "masked_answer_affected_queries": int(
            context["masked_answer_affected"].sum()),
        "masked_answer_affected_digest": _digest_array(
            np.packbits(context["masked_answer_affected"])),
        "truth_ids_digest": _digest_array(
            np.asarray(context["truth_ids"], dtype=np.int64)),
        "affected_query_count_direct": int(
            context["affected_query_count_direct"]),
        "affected_query_rate_direct": float(
            context["affected_query_rate_direct"]),
        "affected_query_direct_digest": _digest_array(
            np.packbits(context["affected_query_direct"])),
        "affected_answer_slots_direct": int(
            context["affected_answer_slots_direct"]),
        "answer_slots_direct": int(context["answer_slots_direct"]),
        "masked_answer_slot_rate_direct": float(
            context["masked_answer_slot_rate_direct"]),
        "query_source_rows_digest": _digest_array(
            np.asarray(context["query_rows"], dtype=np.int64)),
    }
    if "headline_admission" in context:
        manifest["headline_admission"] = context["headline_admission"]
    if context.get("matched_arity") is not None:
        manifest["matched_arity"] = context["matched_arity"]
    if context.get("nested_exposure_protocol") is not None:
        manifest["nested_exposure_protocol"] = context[
            "nested_exposure_protocol"]
    if context.get("historical_exposure_diagnostics") is not None:
        manifest["historical_exposure_diagnostics"] = context[
            "historical_exposure_diagnostics"]
    return manifest


def _frozen_replay_context_audit(
        frozen_replay, current_dataset, split_digest, train_mask_digest,
        role_manifests, ignored_roles=()):
    """Audit a replay context, including legacy reports without truth digests.

    Early frozen reports did not persist ``truth_ids_digest``.  For those
    reports the exact truth is still uniquely determined by the complete
    dataset content, split, and workload, all of which are content-digest
    gated here.  No other missing context field is accepted.
    """
    expected_dataset = frozen_replay.get("dataset") or {}
    dataset_fields = ("name", "n", "d", "hash", "attrs", "embedding_tag")
    missing_dataset = [
        field for field in dataset_fields
        if field not in expected_dataset or field not in current_dataset
    ]
    if missing_dataset:
        raise RuntimeError(
            "frozen design replay lacks dataset identity fields: "
            f"{missing_dataset}")
    checks = {
        f"dataset.{field}": current_dataset[field] == expected_dataset[field]
        for field in dataset_fields
    }
    checks.update({
        "split_digest": split_digest == frozen_replay["split"]["digest"],
        "train_mask_digest": (
            train_mask_digest == frozen_replay["train_mask"]["digest"]),
    })
    required_fields = (
        "base_n", "query_n", "mask_digest", "workload_digest",
        "masked_answer_affected_digest",
    )
    derived_truth = {}
    ignored_roles = set(ignored_roles)
    for role, current_manifest in role_manifests.items():
        if role in ignored_roles:
            continue
        expected_manifest = frozen_replay["partitions"][role]
        missing = [
            field for field in required_fields
            if field not in expected_manifest or field not in current_manifest
        ]
        if missing:
            raise RuntimeError(
                f"frozen design replay {role} lacks context fields: {missing}")
        for field in required_fields:
            checks[f"{role}.{field}"] = (
                current_manifest[field] == expected_manifest[field])
        if "truth_ids_digest" in expected_manifest:
            checks[f"{role}.truth_ids_digest"] = (
                current_manifest["truth_ids_digest"] ==
                expected_manifest["truth_ids_digest"])
        else:
            derived_truth[role] = {
                "status": (
                    "source_field_absent_exact_identity_reconstructed_from_"
                    "matching_dataset_split_workload"),
                "current_truth_ids_digest": current_manifest[
                    "truth_ids_digest"],
                "derivation_basis": [
                    "dataset.hash", "split_digest",
                    f"{role}.workload_digest",
                ],
            }
    return checks, derived_truth


def _canonical_json_sha256(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deployment_policy_artifact(policy) -> dict:
    """Serialize the exact deployment semantics needed before theta."""
    templates = []
    for template in policy.templates:
        templates.append({
            "template_id": template.template_id,
            "targets": list(template.targets),
            "body": [{
                "attr": body.attr,
                "levels": [{"kind": level.kind, "radius": level.radius}
                           for level in body.levels],
            } for body in template.body],
            "location_aware": bool(template.location_aware),
            "min_support": int(template.min_support),
            "kappa": float(template.kappa),
            "priority": int(template.priority),
            "scalar_relax_order": list(template.scalar_relax_order),
        })
    score_maps = {
        template.template_id: {
            "thresholds": policy.score_maps[
                template.template_id].thresholds.tolist(),
            "values": policy.score_maps[template.template_id].values.tolist(),
        }
        for template in policy.templates
    }
    payload = {
        "schema_version": 1,
        "inference_contract": {
            "support_mode": policy.fit_manifest.get(
                "support_mode", "joint"),
            "template_selection_mode": "indexed",
            "support_retrieval_mode": "indexed",
        },
        "templates": templates,
        "score_maps": score_maps,
    }
    return {**payload, "sha256": _canonical_json_sha256(payload)}


def _calibration_cache_key(
        *, dataset_manifest, split_digest, train_mask_digest,
        validation_manifest, calibration_manifest, design,
        deployment_policy_sha256, experiment_arguments, theta_grid,
        calibration_group_ids, eligibility_truth_masks,
        query_source_manifest) -> dict:
    """Bind cached theta to every input that can change calibration."""
    frozen_arguments = {
        key: experiment_arguments.get(key)
        for key in (
            "seed", "rho", "missingness_kind", "historical_beta_exp",
            "historical_topk_multiplier", "block_size", "block_attrs",
            "attribute_weights", "k", "workload_profile",
            "resolved_result_k",
            "canonical_signature_attrs", "matched_arity_signatures",
            "workload_reference", "historical_evidence_fraction",
            "calibration_loss_tolerance",
            "design_objective_tolerance",
        )
    }
    truth_matrix = np.ascontiguousarray(
        np.stack(eligibility_truth_masks), dtype=np.bool_)
    payload = {
        "schema_version": 1,
        "algorithm": "grouped_theta_calibration_near_optimal_v2",
        "dataset": dataset_manifest,
        "split_digest": split_digest,
        "train_mask_digest": train_mask_digest,
        "validation": validation_manifest,
        "calibration": calibration_manifest,
        "design": {
            "structure": design.structural_config.key,
            "policy_candidate": design.candidate_id,
        },
        "deployment_policy_sha256": deployment_policy_sha256,
        "experiment_arguments": frozen_arguments,
        "theta_grid_digest": _digest_array(np.asarray(
            theta_grid, dtype=np.float64)),
        "calibration_group_ids_digest": _digest_array(np.asarray(
            calibration_group_ids, dtype=np.int64)),
        "eligibility_truth_masks_digest": _digest_array(
            np.packbits(truth_matrix)),
        "query_source_manifest_sha256": _canonical_json_sha256(
            query_source_manifest),
    }
    return {"payload": payload, "sha256": _canonical_json_sha256(payload)}


def _frozen_calibration_cache_hit(
        frozen_report, current_key, current_policy_artifact):
    """Return a validated frozen curve/theta, or a machine-readable miss."""
    frozen_cache = frozen_report.get("calibration_cache") or {}
    frozen_policy = frozen_report.get("deployment_policy_artifact") or {}
    required_cache_fields = {
        "key_sha256", "deployment_policy_sha256", "selected_theta",
        "calibration_curve_sha256",
    }
    checks = {
        "cache_schema": frozen_cache.get("schema_version") == 1,
        "cache_payload_complete": required_cache_fields.issubset(
            frozen_cache),
        "cache_key": frozen_cache.get("key_sha256") == current_key["sha256"],
        "deployment_policy": frozen_policy.get("sha256") ==
        current_policy_artifact.get("sha256"),
        "cache_policy": frozen_cache.get("deployment_policy_sha256") ==
        current_policy_artifact.get("sha256"),
        "selected_theta": frozen_cache.get("selected_theta") ==
        frozen_report.get("selected_theta"),
        "curve_digest": frozen_cache.get("calibration_curve_sha256") ==
        _canonical_json_sha256(frozen_report.get("calibration_curve")),
    }
    if not all(checks.values()):
        return None, {"status": "miss", "checks": checks}
    return {
        "theta": float(frozen_cache["selected_theta"]),
        "curve": frozen_report["calibration_curve"],
    }, {"status": "hit", "checks": checks}


def _fresh_final_holdout_audit(frozen_replay, current_test_rows,
                               reserved_queries):
    """Prove that a freshly revealed test cohort excludes selection rows."""
    try:
        previous_rows = [
            int(row["source_row"])
            for row in frozen_replay["results"]["mifa_paper_static"][
                "per_query"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "fresh-final replay source lacks persisted selection source rows"
        ) from exc
    current_rows = [int(value) for value in current_test_rows]
    previous_unique = set(previous_rows)
    current_unique = set(current_rows)
    overlap = sorted(previous_unique & current_unique)
    if overlap:
        raise RuntimeError(
            "fresh-final query cohort overlaps the selection cohort: "
            f"{overlap[:10]}")
    if int(reserved_queries) < len(previous_unique):
        raise RuntimeError(
            "query reservation is smaller than the revealed selection cohort")
    return {
        "status": "complete_disjoint_fresh_final_holdout",
        "selection_queries": int(len(previous_rows)),
        "selection_unique_source_rows": int(len(previous_unique)),
        "reserved_queries": int(reserved_queries),
        "final_queries": int(len(current_rows)),
        "final_unique_source_rows": int(len(current_unique)),
        "source_row_overlap": 0,
        "selection_source_rows_digest": _digest_array(
            np.asarray(previous_rows, dtype=np.int64)),
        "final_source_rows_digest": _digest_array(
            np.asarray(current_rows, dtype=np.int64)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--paper-config", default="configs/paper_main_v2.yaml")
    parser.add_argument("--emb", default="")
    parser.add_argument(
        "--external-query-embeddings", default="",
        help="explicit query-vector asset for the selected embedding tag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--injection-seed", type=int, default=None,
        help=("optional missingness-only seed; dataset split, query sources, "
              "workload, deployment, and ANN seeds remain controlled by "
              "--seed"))
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument(
        "--missingness-kind", default="tuple_mcar",
        choices=("tuple_mcar", "tuple_mcar_nested",
                 "single_attribute_mcar_nested", "tuple_attr_skewed",
                 "tuple_predicate_mcar", "multi_attribute_mcar",
                 "tuple_query_answer_mnar",
                 "fixed_tuple_block", "joint_tuple_block",
                 "tuple_value_dependent",
                 "historical_candidate_exposure",
                 "historical_candidate_exposure_nested"))
    parser.add_argument(
        "--historical-beta-exp", type=float, default=0.50,
        help="fraction of each designated-attribute budget guided by exposure")
    parser.add_argument(
        "--historical-topk-multiplier", type=int, default=10,
        help="historical candidate width L as a multiple of result k")
    parser.add_argument(
        "--historical-distance-batch-size", type=int, default=16,
        help="offline exact historical-query retrieval batch size")
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument(
        "--block-attrs", default="",
        help=("comma-separated workload-correlated attributes masked together; "
              "empty keeps random tuple blocks"))
    parser.add_argument(
        "--answer-impact-fraction", type=float, default=1.0,
        help=("fraction of the tuple missingness budget assigned to frozen "
              "query-answer cells for tuple_query_answer_mnar"))
    parser.add_argument(
        "--answer-cells-per-tuple", type=int, default=1,
        help=("number of predicate cells hidden on each selected tuple for "
              "tuple_query_answer_mnar; default 1 preserves the main setting"))
    parser.add_argument(
        "--attribute-weights", default="",
        help=("workload-frequency weights as attr=weight comma pairs; "
              "required for tuple_attr_skewed unless declared by the query source"))
    parser.add_argument("--n-validation-queries", type=int, default=100)
    parser.add_argument("--n-calibration-queries", type=int, default=50)
    parser.add_argument("--n-test-queries", type=int, default=100)
    parser.add_argument(
        "--n-query-reservation", type=int, default=0,
        help=("reserve this many external query rows after calibration and "
              "before test; used to reveal a disjoint fresh-final cohort"))
    parser.add_argument(
        "--evaluation-query-role",
        choices=("selection_reservation", "test"), default="test",
        help=("evaluate the reserved selection cohort or the blind final "
              "cohort while allocating both roles identically"))
    parser.add_argument(
        "--k", type=int, default=0,
        help="override the paper default top-k for the E1-6/E2-7 sweep")
    parser.add_argument(
        "--historical-evidence-fraction", type=float, default=1.0,
        help=("nested fraction of D_tr exposed to fitting for E1-10; "
              "validation/calibration/test and their workload stay fixed"))
    parser.add_argument(
        "--workload-profile", choices=("stratified", "canonical"),
        default="stratified")
    parser.add_argument(
        "--canonical-signature-attrs", default="",
        help=("comma-separated, predeclared canonical signature override for "
              "controlled predicate-type/arity experiments; constants remain "
              "training-derived"))
    parser.add_argument(
        "--canonical-signature-groups", default="",
        help=("semicolon-separated canonical signatures for a controlled "
              "multi-stratum workload; each group is a comma-separated "
              "attribute list"))
    parser.add_argument(
        "--matched-arity-signatures", default="",
        help=("semicolon-separated, predeclared arity-1/2/3 signatures; "
              "queries are source-paired and matched on complete-data "
              "selectivity using training-instantiated predicates"))
    parser.add_argument(
        "--matched-arity-count-per-signature", action="store_true",
        help=("interpret each role's query count as the number of paired "
              "source queries per arity, preserving the frozen source cohort"))
    parser.add_argument(
        "--workload-reference", choices=(
            "masked_train", "complete_train", "frozen_hce_beta100"),
        default="masked_train",
        help=("evaluator-only table used to freeze predicate values and "
              "selectivity; complete_train is for paired rho sweeps and is "
              "never exposed to inference; frozen_hce_beta100 reconstructs "
              "the existing HCE-1 training mask so Table 3 changes only the "
              "tuple-selection mechanism"))
    parser.add_argument("--validation-instances-per-query", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--projection-dims", default="8,16")
    parser.add_argument("--leaf-capacities", default="64,128")
    parser.add_argument("--support-grid", default="10,30,50")
    parser.add_argument("--shrinkage-grid", default="0,10,30")
    parser.add_argument(
        "--policy-families", default="global,location,scalar,full",
        help="comma-separated validation candidate families")
    parser.add_argument(
        "--deployed-policy-candidate", default="",
        help=("frozen public deployment candidate; when set, evaluate only "
              "this validation-selected candidate"))
    parser.add_argument(
        "--deployed-theta", type=float, default=None,
        help=("frozen calibration-selected threshold from the public "
              "deployment configuration"))
    parser.add_argument("--lambda-time", type=float, default=0.0)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--min-template-coverage", type=float, default=0.80)
    parser.add_argument("--min-shape-coverage", type=float, default=0.80)
    parser.add_argument("--min-map-instances", type=int, default=2)
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument(
        "--design-objective-tolerance", type=float, default=0.001,
        help=("validation objective band for preferring a simpler hierarchy "
              "over a negligible Brier improvement"))
    parser.add_argument(
        "--calibration-loss-tolerance", type=float, default=0.0,
        help=("calibration-only top-k loss band used to select the highest "
              "eligibility-F1 threshold; zero preserves exact optimum"))
    parser.add_argument(
        "--aligned-baselines", default="",
        help=("comma-separated baselines evaluated on the exact in-memory "
              "PaperMIFA contexts: mean_mode,mice,missforest,query_cls"))
    parser.add_argument("--ann-backend", choices=("exact", "hnsw"), default="exact")
    parser.add_argument("--ann-seed-size", type=int, default=64)
    parser.add_argument(
        "--leaf-frontier", action="store_true",
        help="use the exact sorted-leaf/tuple merge executor")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--distance-backend",
        choices=("cpu_fused", "cpu_resident", "torch_resident"),
        default="cpu_fused")
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--distance-query-batch-size", type=int, default=1)
    parser.add_argument(
        "--executor-path", choices=("unified_tree", "distance_order"),
        default="unified_tree",
        help=("explicit exact executor path; the paper path is unified_tree, "
              "while distance_order is a separately labeled ablation"))
    parser.add_argument(
        "--exact-audit-queries", type=int, default=-1,
        help="exact-oracle prefix size; -1 audits every query")
    parser.add_argument(
        "--persist-test-truth-ids", action="store_true",
        help=("persist evaluator-owned complete-data top-k IDs for audited "
              "cross-dimension recall; never used by method selection"))
    parser.add_argument(
        "--executor-ablation-queries", type=int, default=0,
        help="fixed-policy FullInference/DistanceFirst/ANNOrder prefix")
    parser.add_argument(
        "--executor-ablation-workers", type=int, default=1,
        help=("parallelize independent ablation queries with forked CPU "
              "workers; each query still times both strategy orders locally"))
    parser.add_argument(
        "--frozen-design-replay-report", default="",
        help=("terminal report whose validation-selected structure/policy is "
              "replayed as a single audited candidate; context and theta "
              "digests must reproduce exactly"))
    parser.add_argument(
        "--fresh-final-replay", action="store_true",
        help=("with --frozen-design-replay-report, preserve the training, "
              "validation, calibration, design, and theta contexts while "
              "allowing a disjoint reserved test-query cohort"))
    parser.add_argument(
        "--support-efficiency-requests", type=int, default=0,
        help="run E2-10 on this many real support requests")
    parser.add_argument(
        "--pruning-diagnostic-queries", type=int, default=0,
        help=("run the five-row disable-one root-tree diagnostic on this "
              "many current test queries"))
    parser.add_argument(
        "--pruning-diagnostic-variants", default="",
        help=("optional comma-separated diagnostic cell selector: full,"
              "ann_seed,node_bound,tuple_boundary,query_cache; full must be "
              "first"))
    parser.add_argument(
        "--pruning-diagnostic-persist-per-query", action="store_true",
        help="persist latency, pruning counters, and ordered-ID agreement")
    parser.add_argument(
        "--skip-predicate-diagnostics", action="store_true",
        help=("omit the evaluator-only all-query/all-tuple predicate matrix; "
              "used by isolated kernel retiming where those diagnostic "
              "metrics are outside the experiment contract"))
    parser.add_argument(
        "--test-theta-diagnostic", action="store_true",
        help=("development-only test-optimal threshold sweep for measuring "
              "selection regret; never used by the reported method"))
    parser.add_argument(
        "--grouped-exact-backends", default="",
        help="optional semantic-preserving predicate cache: cpu,torch")
    parser.add_argument("--grouped-benchmark-repeats", type=int, default=3)
    parser.add_argument(
        "--candidate-efficiency-matrix", action="store_true",
        help="measure the exact candidate executor on CPU, 1 GPU, and 2 GPUs")
    parser.add_argument(
        "--candidate-efficiency-query-count", type=int, default=100)
    parser.add_argument(
        "--candidate-efficiency-repeats", type=int, default=5)
    parser.add_argument(
        "--candidate-page-initial", type=int, default=4096,
        help="initial compact candidate page size shared by CPU and GPU")
    parser.add_argument(
        "--candidate-page-maximum", type=int, default=65536,
        help="maximum compact candidate page size shared by CPU and GPU")
    parser.add_argument(
        "--candidate-efficiency-gpu-devices", default="",
        help="comma-separated CUDA devices; the first two are measured")
    parser.add_argument(
        "--candidate-efficiency-platforms", default="cpu,1gpu,2gpu",
        help=("comma-separated isolated matrix segments: "
              "cpu,1gpu,2gpu"))
    parser.add_argument(
        "--candidate-efficiency-cpu-projection-dim", type=int, default=16)
    parser.add_argument(
        "--candidate-efficiency-cpu-refine-width", type=int, default=64)
    parser.add_argument(
        "--candidate-efficiency-cpu-fused-exact-min-work",
        type=int, default=0,
        help=("explicit per-dataset work threshold for the exact fused "
              "eligibility/distance/top-k kernel; zero preserves the legacy "
              "CPU path"))
    parser.add_argument(
        "--candidate-efficiency-cpu-approximate-shortlist",
        action="store_true",
        help=("explicit latency-first int8 projection shortlist followed by "
              "full-dimensional FP32 refinement; reports result-set delta "
              "and does not claim exact ordered IDs"))
    parser.add_argument(
        "--candidate-efficiency-cpu-only", action="store_true",
        help="measure only the optimized CPU candidate/response path")
    parser.add_argument(
        "--candidate-efficiency-response-generic-cpu", action="store_true",
        help=("use the common exact CPU candidate stream for batch-one "
              "response timing, disabling dataset-specific eligibility-first "
              "shortcuts so cross-workload breakdowns share one path"))
    parser.add_argument(
        "--candidate-efficiency-response-use-frozen-reference",
        action="store_true",
        help=("compare batch-one CPU responses directly with frozen exact "
              "ordered IDs instead of regenerating an equivalent 1xN "
              "reference row per query"))
    parser.add_argument(
        "--candidate-efficiency-response-only", action="store_true",
        help=("measure exact single-request response latency without the "
              "separate full-cohort batch candidate-throughput diagnostic"))
    parser.add_argument(
        "--candidate-efficiency-response-saturated-throughput",
        action="store_true",
        help=("in addition to exact single-request P50/P95, measure QPS as "
              "a saturated full-cohort exact batch; this keeps latency and "
              "throughput scopes explicit"))
    parser.add_argument(
        "--candidate-efficiency-exact-prewarm-strategy",
        choices=("group_first", "posting_first"), default="group_first",
        help=("versioned exact serving-cache strategy; group_first preserves "
              "the legacy default, while posting_first is selected only by "
              "an explicit per-dataset kernel manifest"))
    parser.add_argument(
        "--candidate-efficiency-gpu-resident-exact-postings",
        action="store_true",
        help=("keep explicitly prewarmed exact eligible-row postings on GPU "
              "and reuse them in the batch-one response kernel; disabled by "
              "default and intended for dataset-specific serving manifests"))
    parser.add_argument(
        "--candidate-efficiency-response-strata", default="",
        help="comma-separated response strata: predicate_type,arity,k")
    parser.add_argument(
        "--scale-searchable-n", type=int, default=0,
        help=("optional systems-only expanded searchable corpus; generated "
              "deterministically from the real searchable snapshot"))
    parser.add_argument("--scale-query-count", type=int, default=16)
    parser.add_argument(
        "--scale-pruning-diagnostic-queries", type=int, default=0,
        help=("run the disable-one root-tree/on-demand diagnostic on this "
              "many queries of the expanded scale snapshot"))
    parser.add_argument(
        "--scale-quality-reference", action="store_true",
        help=("compute controlled-scale F1 against same-dimensional complete "
              "metadata/full-corpus exact truth; never used for selection"))
    parser.add_argument(
        "--scale-quality-query-count", type=int, default=0,
        help=("fixed query prefix used by the controlled-scale quality audit; "
              "zero uses the full selected scale cohort"))
    parser.add_argument(
        "--scale-quality-distance-reference",
        choices=("served_scheduler", "cpu_exact"),
        default="served_scheduler",
        help=("distance implementation used only to construct controlled-"
              "scale complete-metadata truth; never used for selection"))
    parser.add_argument(
        "--scale-response-query-count", type=int, default=0,
        help=("fixed query prefix used for true GPU batch-1 response-latency "
              "P50/P95; zero disables the extra measurement"))
    parser.add_argument(
        "--scale-cpu-batch1-query-count", type=int, default=0,
        help=("fixed query prefix used for CPU single-query latency; zero "
              "uses the full selected scale cohort"))
    parser.add_argument(
        "--scale-batch-sizes", default="",
        help=("comma-separated GPU candidate-stream batch sizes measured in "
              "the saturated-throughput regime"))
    parser.add_argument(
        "--scale-query-repetitions", type=int, default=1,
        help=("repeat the fixed query row sequence to measure saturated "
              "batch throughput; one preserves the unique-query run"))
    parser.add_argument("--scale-dimension", type=int, default=128)
    parser.add_argument("--scale-expansion-seed", type=int, default=0x78C38D41)
    parser.add_argument(
        "--scale-cohort-artifact", default="",
        help=("optional frozen external-baseline cohort receipt supplying "
              "the exact query row IDs"))
    parser.add_argument(
        "--scale-cohort-track", default="numerical",
        help="track name inside --scale-cohort-artifact")
    parser.add_argument(
        "--scale-cohort-role", choices=("test",),
        default="test")
    parser.add_argument(
        "--scale-searchable-handler",
        choices=("native", "holoclean"), default="native",
        help=("optional frozen pre-imputation applied only to the expanded "
              "searchable snapshot; residual missing cells remain under "
              "MIFA's native online policy"))
    parser.add_argument(
        "--scale-repaired-test-table", default="",
        help="row-aligned HoloClean test materialization")
    parser.add_argument(
        "--scale-repair-test-receipt", default="",
        help="official HoloClean receipt for the scale test materialization")
    parser.add_argument(
        "--scale-cpu-only", action="store_true",
        help="run the scale benchmark without initializing CUDA")
    parser.add_argument(
        "--scale-gpu-devices", default="",
        help="comma-separated CUDA devices for 1-to-N query sharding")
    parser.add_argument("--scale-distance-batch-size", type=int, default=8)
    parser.add_argument("--scale-repeats", type=int, default=3)
    parser.add_argument("--scale-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--scale-jitter-reference-embeddings", default="",
        help=("optional row-aligned highest-dimensional embedding file; its "
              "test-partition prefixes and full-width seeded jitter define "
              "one dimension-nested MRL scale corpus"))
    parser.add_argument(
        "--scale-verify-exact-outputs", action="store_true",
        help=("opt in to the additional exact-reference replay; disabled "
              "for normal latency measurements"))
    parser.add_argument(
        "--scale-record-digests", action="store_true",
        help=("opt in to content digests for scale arrays and outputs; "
              "disabled for normal latency measurements"))
    parser.add_argument(
        "--scale-executor-backend",
        choices=("precomputed_leaf_reference", "gpu_candidate_stream"),
        default="precomputed_leaf_reference",
        help=("opt-in Phase-A exact candidate stream; the unchanged leaf "
              "reference remains the default and exact oracle"))
    parser.add_argument(
        "--scale-candidate-multi-gpu-mode",
        choices=("corpus", "query"), default="corpus",
        help=("candidate-stream multi-GPU scheduling: shard the corpus or "
              "replicate it and shard the query batch"))
    parser.add_argument(
        "--scale-candidate-score-precision",
        choices=("fp32_exact", "fp16_refine"), default="fp32_exact",
        help=("candidate distance/ranking kernel: exact FP32 or FP16 "
              "shortlist followed by FP32 candidate reranking"))
    parser.add_argument(
        "--scale-tree-probe-counts", default="",
        help=("comma-separated globally fixed projected-leaf probe counts; "
              "empty disables the approximate ANN/tree measurement"))
    parser.add_argument(
        "--scale-tree-probe-result-width", type=int, default=64)
    parser.add_argument(
        "--scale-ivf-probe-counts", default="",
        help=("comma-separated globally fixed IVF nprobe values; empty "
              "disables the searchable-snapshot ANN partition index"))
    parser.add_argument("--scale-ivf-nlist", type=int, default=1024)
    parser.add_argument(
        "--scale-ivf-training-rows", type=int, default=262144)
    parser.add_argument("--scale-ivf-iterations", type=int, default=10)
    parser.add_argument("--scale-ivf-build-threads", type=int, default=24)
    parser.add_argument("--tag", default="audit")
    args = parser.parse_args()
    if args.calibration_loss_tolerance < 0:
        parser.error("--calibration-loss-tolerance must be non-negative")
    if args.design_objective_tolerance < 0:
        parser.error("--design-objective-tolerance must be non-negative")
    frozen_replay = None
    frozen_replay_path = None
    frozen_replay_manifest = None
    if args.frozen_design_replay_report:
        frozen_replay_path = (ROOT / args.frozen_design_replay_report).resolve()
        if not frozen_replay_path.is_file():
            raise ValueError("frozen design replay report does not exist")
        frozen_replay = json.loads(frozen_replay_path.read_text())
        if not frozen_replay.get("selected_design"):
            raise ValueError("frozen design replay report lacks selected design")
        try:
            frozen_replay_label = str(frozen_replay_path.relative_to(ROOT))
        except ValueError:
            # A closeout branch may replay an immutable result from a sibling
            # checkout.  Preserve that resolved source without copying it.
            frozen_replay_label = str(frozen_replay_path)
        frozen_replay_manifest = {
            "status": "validation_selected_design_replay_requested",
            "source_report": frozen_replay_label,
            "source_report_sha256": hashlib.sha256(
                frozen_replay_path.read_bytes()).hexdigest(),
            "context_checks": {},
            "selected_design_match": False,
            "selected_theta_match": False,
            "test_metrics_used_for_replay_selection": False,
        }
    block_attrs = tuple(
        item.strip() for item in args.block_attrs.split(",") if item.strip())
    if block_attrs and args.missingness_kind not in {
            "single_attribute_mcar_nested", "tuple_predicate_mcar",
            "multi_attribute_mcar",
            "tuple_query_answer_mnar",
            "fixed_tuple_block", "joint_tuple_block",
            *HISTORICAL_EXPOSURE_KINDS}:
        raise ValueError(
            "--block-attrs requires single_attribute_mcar_nested, "
            "tuple_predicate_mcar, "
            "multi_attribute_mcar, "
            "tuple_query_answer_mnar, "
            "fixed_tuple_block, or joint_tuple_block missingness")
    if (block_attrs and args.missingness_kind in {
            "fixed_tuple_block", "joint_tuple_block"} and
            len(block_attrs) != args.block_size):
        raise ValueError("--block-attrs count must equal --block-size")
    if args.missingness_kind == "tuple_predicate_mcar" and not block_attrs:
        raise ValueError(
            "tuple_predicate_mcar requires --block-attrs")
    if args.missingness_kind == "multi_attribute_mcar" and not block_attrs:
        raise ValueError(
            "multi_attribute_mcar requires --block-attrs")
    if (args.missingness_kind in HISTORICAL_EXPOSURE_KINDS and
            not block_attrs):
        raise ValueError(
            f"{args.missingness_kind} requires --block-attrs")
    if not 0.0 <= args.historical_beta_exp <= 1.0:
        raise ValueError("--historical-beta-exp must be in [0, 1]")
    if args.historical_topk_multiplier < 1:
        raise ValueError("--historical-topk-multiplier must be positive")
    if args.historical_distance_batch_size < 1:
        raise ValueError("--historical-distance-batch-size must be positive")
    if (args.missingness_kind == "single_attribute_mcar_nested" and
            len(block_attrs) != 1):
        raise ValueError(
            "single_attribute_mcar_nested requires one --block-attrs value")
    if not 0.0 <= args.answer_impact_fraction <= 1.0:
        raise ValueError("--answer-impact-fraction must be in [0, 1]")
    if args.answer_cells_per_tuple < 1:
        raise ValueError("--answer-cells-per-tuple must be positive")
    if args.k < 0:
        raise ValueError("--k must be zero (paper default) or positive")
    if args.n_query_reservation < 0:
        raise ValueError("--n-query-reservation cannot be negative")
    if args.fresh_final_replay and not args.frozen_design_replay_report:
        raise ValueError(
            "--fresh-final-replay requires --frozen-design-replay-report")
    if args.fresh_final_replay and args.n_query_reservation < 1:
        raise ValueError(
            "--fresh-final-replay requires a positive query reservation")
    if (args.evaluation_query_role == "selection_reservation" and
            args.n_query_reservation < 1):
        raise ValueError(
            "selection_reservation evaluation requires a positive query "
            "reservation")
    if (args.fresh_final_replay and
            args.evaluation_query_role != "test"):
        raise ValueError("fresh-final replay must evaluate the test role")
    if not 0.0 < args.historical_evidence_fraction <= 1.0:
        raise ValueError("--historical-evidence-fraction must be in (0, 1]")
    if (args.historical_evidence_fraction < 1.0 and
            args.workload_reference != "complete_train"):
        raise ValueError(
            "evidence-size sweeps require --workload-reference complete_train "
            "to keep validation/calibration/test predicates fixed")
    if args.executor_path == "distance_order" and args.epsilon != 0.0:
        raise ValueError("distance_order executor path requires epsilon=0")
    if (args.executor_path == "distance_order" and
            args.distance_backend == "cpu_fused"):
        raise ValueError(
            "distance_order executor path requires a resident distance row")
    if args.candidate_page_initial < 1:
        raise ValueError("--candidate-page-initial must be positive")
    if (args.pruning_diagnostic_queries < 0 or
            args.scale_pruning_diagnostic_queries < 0):
        raise ValueError("pruning diagnostic query counts cannot be negative")
    pruning_variant_keys = tuple(
        value.strip() for value in args.pruning_diagnostic_variants.split(",")
        if value.strip())
    allowed_pruning_variants = {
        "full", "ann_seed", "node_bound", "tuple_boundary", "query_cache"}
    if set(pruning_variant_keys) - allowed_pruning_variants:
        raise ValueError("unknown --pruning-diagnostic-variants value")
    if pruning_variant_keys and pruning_variant_keys[0] != "full":
        raise ValueError("--pruning-diagnostic-variants must start with full")
    if args.executor_ablation_workers < 1:
        raise ValueError("executor ablation workers must be positive")
    if (args.scale_quality_query_count < 0 or
            args.scale_response_query_count < 0 or
            args.scale_cpu_batch1_query_count < 0):
        raise ValueError("scale quality/response query counts cannot be negative")
    if args.candidate_page_maximum < args.candidate_page_initial:
        raise ValueError(
            "--candidate-page-maximum must be at least the initial size")

    injection_seed = (
        args.seed if args.injection_seed is None else args.injection_seed)

    paper_path = ROOT / args.paper_config
    dataset_path = ROOT / args.dataset_config
    paper_config = yaml.safe_load(paper_path.read_text())
    dataset_config = yaml.safe_load(dataset_path.read_text())
    if args.emb:
        dataset_config["dataset"]["emb_file"] = f"emb_{args.emb}.npy"
    dataset_config["seed"] = args.seed
    spec = load_dataset(dataset_config)
    if args.external_query_embeddings:
        source_cfg = paper_config.get("query_sources", {}).get(spec.name)
        if source_cfg is None or source_cfg.get("kind") != "external_embeddings":
            raise ValueError(
                "explicit external query vectors require an external source")
        source_cfg.setdefault("embeddings_by_tag", {})[args.emb] = \
            args.external_query_embeddings
    source_cfg = paper_config.get("query_sources", {}).get(spec.name, {})
    attrs = list(source_cfg.get("predicate_attrs", spec.query_relevant_attrs))
    unknown = set(attrs) - set(spec.dtypes)
    if unknown:
        raise ValueError(f"unknown predicate attrs: {sorted(unknown)}")
    spec.query_relevant_attrs = attrs
    attribute_weights, attribute_weights_manifest = \
        legacy._resolve_attribute_weights(
            args.missingness_kind, args.attribute_weights,
            source_cfg.get("attribute_missingness_weights"), attrs)
    signature_mix = None
    if args.workload_profile == "canonical":
        signature_mix = source_cfg.get("canonical_signatures")
        if not signature_mix:
            raise ValueError(
                f"{spec.name}: canonical workload has no predeclared signatures")
        signature_override = tuple(
            item.strip() for item in args.canonical_signature_attrs.split(",")
            if item.strip())
        signature_groups = tuple(
            tuple(attr.strip() for attr in group.split(",") if attr.strip())
            for group in args.canonical_signature_groups.split(";")
            if group.strip())
        if signature_override and signature_groups:
            raise ValueError(
                "canonical signature attrs and groups are mutually exclusive")
        if signature_override:
            unknown_signature_attrs = set(signature_override) - set(attrs)
            if unknown_signature_attrs:
                raise ValueError(
                    "canonical signature override contains undeclared attrs: "
                    f"{sorted(unknown_signature_attrs)}")
            signature_mix = [{"attrs": list(signature_override), "weight": 1}]
        elif signature_groups:
            unknown_signature_attrs = set().union(
                *(set(group) for group in signature_groups)) - set(attrs)
            if unknown_signature_attrs:
                raise ValueError(
                    "canonical signature groups contain undeclared attrs: "
                    f"{sorted(unknown_signature_attrs)}")
            signature_mix = [
                {"attrs": list(group), "weight": 1}
                for group in signature_groups]
    elif args.canonical_signature_attrs or args.canonical_signature_groups:
        raise ValueError(
            "canonical signature overrides require --workload-profile canonical")
    matched_arity_signatures = tuple(
        tuple(attr.strip() for attr in group.split(",") if attr.strip())
        for group in args.matched_arity_signatures.split(";") if group.strip())
    if matched_arity_signatures:
        if args.workload_profile != "canonical":
            raise ValueError(
                "--matched-arity-signatures requires --workload-profile canonical")
        if args.canonical_signature_attrs or args.canonical_signature_groups:
            raise ValueError(
                "matched arity and canonical signature overrides are "
                "mutually exclusive")
        signature_mix = [
            {"attrs": list(signature), "weight": 1}
            for signature in matched_arity_signatures]
    canonical_bucket_weights = tuple(source_cfg.get(
        "canonical_selectivity_weights",
        paper_config["workload"].get(
            "canonical_selectivity_weights", (0.5, 0.5, 0.0))))

    # Reserve the exact query-source role used by the faithful baseline even
    # though this paper-only runner does not fit QueryCls.  This is essential
    # for heldout-content sources, where allocated query rows are removed from
    # the searchable relation before the four-way database split.
    role_counts = {
        "query_cls_training": legacy._query_cls_training_reservation(
            paper_config, args.n_validation_queries),
        "validation": args.n_validation_queries,
        "calibration": args.n_calibration_queries,
    }
    if args.n_query_reservation:
        role_counts["selection_reservation"] = args.n_query_reservation
    role_counts["test"] = args.n_test_queries
    spec, query_specs, native_filters, query_source_manifest = \
        legacy._prepare_query_sources(
            spec, paper_config, role_counts=role_counts,
            seed=int(paper_config["split"]["seed"]) + args.seed * 100003,
            embedding_tag=args.emb or None)
    spec.query_relevant_attrs = attrs
    dataset_manifest = {
        "name": spec.name,
        "n": int(spec.n),
        "d": int(spec.d),
        "hash": spec.hash(),
        "attrs": attrs,
        "embedding_tag": args.emb or "dataset_default",
    }
    split_cfg = paper_config["split"]
    fraction_cfg = split_cfg.get("fractions", {})
    fractions = tuple(float(fraction_cfg.get(name, default)) for name, default in (
        ("train", 0.50), ("validation", 0.20),
        ("calibration", 0.10), ("test", 0.20)))
    split = make_four_way_split(
        spec.n, seed=int(split_cfg["seed"]), fractions=fractions)
    parts = {name: subset_spec(spec, rows, name=f"{spec.name}_{name}")
             for name, rows in split.partitions().items()}
    for part in parts.values():
        part.query_relevant_attrs = attrs
    historical_query_vectors = None
    historical_query_descriptors = None
    result_k = (int(args.k) if args.k else
                int(paper_config["workload"]["k_default"]))
    historical_top_l = int(args.historical_topk_multiplier * result_k)
    needs_historical_reference = (
        args.missingness_kind in HISTORICAL_EXPOSURE_KINDS
        or args.workload_reference == "frozen_hce_beta100")
    if needs_historical_reference:
        historical_query_vectors, historical_query_descriptors = \
            _historical_query_bundle(
                query_specs.get("query_cls_training"),
                native_filters.get("query_cls_training"), attrs,
                signature_mix)
        _progress(
            "historical_queries_frozen",
            queries=len(historical_query_descriptors),
            top_l=historical_top_l,
            beta_exp=args.historical_beta_exp,
            designated_attrs=block_attrs)
    full_training_part = parts["train"]
    evidence_order = np.random.default_rng(
        0x78C38DA5 + args.seed).permutation(full_training_part.n)
    evidence_count = max(1, int(round(
        args.historical_evidence_fraction * full_training_part.n)))
    evidence_rows = np.sort(evidence_order[:evidence_count])
    if evidence_count != full_training_part.n:
        parts["train"] = subset_spec(
            full_training_part, evidence_rows,
            name=f"{spec.name}_train_evidence_{evidence_count}")
        parts["train"].query_relevant_attrs = attrs
    train_missingness_kind = training_missingness_kind(
        args.missingness_kind, args.answer_cells_per_tuple)
    train_block_attrs = (
        block_attrs or tuple(attrs)
        if args.missingness_kind == "tuple_query_answer_mnar"
        else block_attrs)
    train_historical_candidates_all = None
    if needs_historical_reference:
        train_historical_candidates_all = _historical_candidate_ids(
            historical_query_vectors,
            parts["train"].normalized_embeddings(), historical_top_l,
            device=args.gpu_device,
            batch_size=args.historical_distance_batch_size)
    train_historical_candidates = (
        train_historical_candidates_all
        if args.missingness_kind in HISTORICAL_EXPOSURE_KINDS else None)
    train_historical_priority = None
    train_nested_exposure_protocol = None
    if args.missingness_kind == "historical_candidate_exposure_nested":
        train_designated_indices = [
            attrs.index(attr) for attr in train_block_attrs]
        train_historical_priority = historical_candidate_exposure_priority(
            parts["train"].n, len(attrs), train_designated_indices,
            historical_query_descriptors, train_historical_candidates,
            beta_exp=args.historical_beta_exp,
            rng=np.random.default_rng(injection_seed * 100 + 1))
        train_audit_rhos = tuple(sorted({
            0.1, 0.2, 0.3, float(args.rho)}))
        train_nested_exposure_protocol = \
            historical_candidate_exposure_nested_audit(
                train_historical_priority, parts["train"].n, len(attrs),
                rhos=train_audit_rhos)
        train_nested_exposure_protocol.update({
            "priority_seed": int(injection_seed * 100 + 1),
            "actual_rho": float(args.rho),
        })
    train_observed, train_mask, _ = make_missing(
        parts["train"], args.rho, train_missingness_kind,
        seed=injection_seed * 100 + 1, block_size=args.block_size,
        block_attrs=train_block_attrs, usage_freq=attribute_weights,
        answer_cells_per_tuple=args.answer_cells_per_tuple,
        historical_queries=historical_query_descriptors,
        historical_candidate_ids=train_historical_candidates,
        beta_exp=args.historical_beta_exp,
        historical_priority=train_historical_priority)
    if train_nested_exposure_protocol is not None:
        train_mask_digest = _digest_array(np.packbits(train_mask))
        expected_train_mask_digest = next(
            item["mask_digest"]
            for item in train_nested_exposure_protocol["rho_levels"]
            if np.isclose(
                item["rho"], args.rho, rtol=0.0, atol=1e-12))
        if train_mask_digest != expected_train_mask_digest:
            raise RuntimeError(
                "training nested exposure mask does not match its frozen "
                "priority audit")
        train_nested_exposure_protocol.update({
            "actual_mask_digest": train_mask_digest,
            "actual_mask_matches_level": True,
        })
    predicate_reference_source = (
        full_training_part if args.workload_reference == "complete_train"
        else parts["train"])
    predicate_reference = subset_spec(
        predicate_reference_source,
        np.arange(predicate_reference_source.n),
        name=f"{spec.name}_training_predicate_reference")
    # Workload construction is evaluator-owned.  In a rho sweep it may use
    # complete D_tr to keep predicates fixed; the inference path still receives
    # only ``train_observed`` below.
    if args.workload_reference == "complete_train":
        predicate_reference.table = full_training_part.table.copy()
        predicate_reference_manifest = {
            "kind": "complete_train", "mask_digest": None,
            "evaluator_only": True,
        }
    elif args.workload_reference == "frozen_hce_beta100":
        frozen_reference, frozen_reference_mask, _ = make_missing(
            parts["train"], args.rho, "historical_candidate_exposure",
            seed=injection_seed * 100 + 1, block_size=args.block_size,
            block_attrs=block_attrs, usage_freq=attribute_weights,
            historical_queries=historical_query_descriptors,
            historical_candidate_ids=train_historical_candidates_all,
            beta_exp=1.0)
        predicate_reference.table = frozen_reference
        predicate_reference_manifest = {
            "kind": "frozen_hce_beta100",
            "mask_digest": _digest_array(np.packbits(frozen_reference_mask)),
            "beta_exp": 1.0,
            "evaluator_only": True,
        }
    else:
        predicate_reference.table = train_observed.copy()
        predicate_reference_manifest = {
            "kind": "masked_train",
            "mask_digest": _digest_array(np.packbits(train_mask)),
            "evaluator_only": True,
        }
    _progress("training_ready", n=parts["train"].n, attrs=attrs)

    def prepare(role, count, mask_offset, workload_offset, *, query_role=None):
        query_role = role if query_role is None else query_role
        return _prepare_eval(
            parts[role], role=role, n_queries=count,
            k=result_k, rho=args.rho,
            mask_seed=injection_seed * 100 + mask_offset,
            workload_seed=args.seed * 100 + workload_offset, attrs=attrs,
            predicate_reference=predicate_reference,
            missingness_kind=args.missingness_kind, block_size=args.block_size,
            block_attrs=block_attrs, attribute_weights=attribute_weights,
            answer_impact_fraction=args.answer_impact_fraction,
            answer_cells_per_tuple=args.answer_cells_per_tuple,
            query_spec=query_specs.get(query_role),
            native_filters=native_filters.get(query_role),
            signature_mix=signature_mix,
            matched_arity_signatures=matched_arity_signatures,
            matched_arity_count_per_signature=(
                args.matched_arity_count_per_signature),
            canonical_bucket_weights=canonical_bucket_weights,
            historical_query_vectors=historical_query_vectors,
            historical_query_descriptors=historical_query_descriptors,
            historical_top_l=historical_top_l,
            historical_beta_exp=args.historical_beta_exp,
            historical_device=args.gpu_device,
            historical_batch_size=args.historical_distance_batch_size)

    validation = prepare("validation", args.n_validation_queries, 2, 12)
    calibration = prepare("calibration", args.n_calibration_queries, 3, 13)
    test = prepare(
        "test", args.n_test_queries, 4, 14,
        query_role=args.evaluation_query_role)
    if os.environ.get("MIFA_EXPORT_INJECTED_ONLY") == "1":
        export_root = os.environ.get("MIFA_EXPORT_INJECTED_ROOT")
        if not export_root:
            raise ValueError(
                "MIFA_EXPORT_INJECTED_ONLY requires "
                "MIFA_EXPORT_INJECTED_ROOT")
        print(json.dumps({
            "status": "injected_data_exported",
            "dataset": spec.name,
            "mechanism": args.missingness_kind,
            "rho": float(args.rho),
            "output_root": export_root,
            "partitions": {
                "train": int(parts["train"].n),
                "validation": int(parts["validation"].n),
                "calibration": int(parts["calibration"].n),
                "test": int(parts["test"].n),
            },
        }))
        return
    fresh_final_protocol = None
    if frozen_replay is not None:
        role_manifests = {
            current["role"]: _context_manifest(current)
            for current in (validation, calibration, test)
        }
        checks, derived_truth = _frozen_replay_context_audit(
            frozen_replay, dataset_manifest, split.digest(),
            _digest_array(np.packbits(train_mask)), role_manifests,
            ignored_roles=("test",) if args.fresh_final_replay else ())
        if not all(checks.values()):
            failed = sorted(key for key, value in checks.items() if not value)
            raise RuntimeError(
                f"frozen design replay context mismatch: {failed}")
        frozen_replay_manifest["context_checks"] = checks
        frozen_replay_manifest["derived_truth_identity"] = derived_truth
        if args.fresh_final_replay:
            fresh_final_protocol = _fresh_final_holdout_audit(
                frozen_replay, test["query_rows"],
                args.n_query_reservation)
            frozen_replay_manifest["fresh_final_holdout"] = \
                fresh_final_protocol
    # Structured headline protocols predeclare a support-rich common scope
    # using masked D_tr only.  Reuse the exact admission routine used by the
    # frozen baselines so method and baseline rows are compared on identical
    # queries; the full unconditioned macro remains in every report.
    headline_cfg = paper_config.get("position", {}).get(
        "headline_admission")
    if headline_cfg:
        min_joint_support = int(
            headline_cfg["min_training_joint_support"])
        for context in (validation, calibration, test):
            legacy._attach_headline_admission(
                context, train_observed,
                min_joint_support=min_joint_support)
    _progress("workloads_ready", validation=len(validation["queries"]),
              calibration=len(calibration["queries"]), test=len(test["queries"]))

    instances, target_sets, instance_manifest = _validation_instances(
        validation, args.validation_instances_per_query, args.folds)
    body_candidates = discover_scalar_body_candidates(
        parts["train"], train_observed, target_sets,
        min_support=min(_csv_ints(args.support_grid)))
    policy_candidates = build_policy_candidate_family(
        parts["train"], train_observed, target_sets, body_candidates,
        support_values=_csv_ints(args.support_grid),
        shrinkage_values=_csv_floats(args.shrinkage_grid))
    allowed_policy_families = {
        item.strip() for item in args.policy_families.split(",")
        if item.strip()}
    unknown_policy_families = allowed_policy_families - {
        "global", "location", "scalar", "full"}
    if unknown_policy_families:
        raise ValueError(
            f"unknown policy families: {sorted(unknown_policy_families)}")
    policy_candidates = [
        candidate for candidate in policy_candidates
        if candidate.candidate_id.split("_", 1)[0]
        in allowed_policy_families]
    if args.deployed_policy_candidate:
        policy_candidates = [
            candidate for candidate in policy_candidates
            if candidate.candidate_id == args.deployed_policy_candidate]
    if not policy_candidates:
        raise ValueError("policy-family filter removed every candidate")
    structural_configs = [
        StructuralConfig(projection_dim, leaf_capacity)
        for projection_dim in _csv_ints(args.projection_dims)
        for leaf_capacity in _csv_ints(args.leaf_capacities)
    ]
    selector = OfflineDesignSelector(
        parts["train"].normalized_embeddings(), train_observed, spec.dtypes,
        validation["embeddings"], validation["observed"], instances,
        lambda_time=args.lambda_time, min_coverage=args.min_coverage,
        min_template_coverage=args.min_template_coverage,
        min_shape_coverage=args.min_shape_coverage,
        min_map_instances=args.min_map_instances,
        seed=args.seed, progress_callback=_progress,
        candidate_workers=args.candidate_workers,
        objective_tolerance=args.design_objective_tolerance)
    design = selector.select(structural_configs, policy_candidates)
    if frozen_replay is not None:
        expected_design = frozen_replay["selected_design"]
        design_match = bool(
            design.structural_config.key == expected_design["structure"] and
            design.candidate_id == expected_design["policy_candidate"])
        if not design_match:
            raise RuntimeError(
                "single-candidate replay did not reproduce frozen design")
        frozen_replay_manifest["selected_design_match"] = True
    _progress("design_selected", structure=design.structural_config.key,
              policy=design.candidate_id,
              templates=len(design.policy.templates))

    deployment_policy, final_refit = refit_deployment_policy(
        parts["train"].normalized_embeddings(), train_observed, spec.dtypes,
        validation["embeddings"], validation["observed"], instances,
        design.structural_config, design.candidate_policy,
        min_map_instances=args.min_map_instances,
        min_template_coverage=args.min_template_coverage,
        min_shape_coverage=args.min_shape_coverage,
        seed=args.seed)
    deployment_policy.fit_manifest.update({
        key: design.policy.fit_manifest[key]
        for key in (
            "selection_objective", "lambda_time",
            "paper_exact_executor_replay_timecost", "selected_candidate")
        if key in design.policy.fit_manifest
    })
    _progress("deployment_policy_refitted",
              templates=len(deployment_policy.templates),
              structure_source="training_embeddings")

    theta_grid_cfg = paper_config.get("inference", {}).get("theta_grid", {})
    theta_grid = np.arange(
        float(theta_grid_cfg.get("start", 0.0)),
        float(theta_grid_cfg.get("stop", 1.0)) + 1e-12,
        float(theta_grid_cfg.get("step", 0.01)))
    group_ids = [int(query["src"]) for query in calibration["queries"]]
    calibration_eligibility_truth = [
        query["phi"].eval_table(calibration["base"].table) == TRUE
        for query in calibration["queries"]
    ]
    deployment_policy_artifact = _deployment_policy_artifact(
        deployment_policy)
    cache_arguments = {**vars(args), "resolved_result_k": int(result_k)}
    calibration_cache_key = _calibration_cache_key(
        dataset_manifest=dataset_manifest,
        split_digest=split.digest(),
        train_mask_digest=_digest_array(np.packbits(train_mask)),
        validation_manifest=_context_manifest(validation),
        calibration_manifest=_context_manifest(calibration),
        design=design,
        deployment_policy_sha256=deployment_policy_artifact["sha256"],
        experiment_arguments=cache_arguments,
        theta_grid=theta_grid,
        calibration_group_ids=group_ids,
        eligibility_truth_masks=calibration_eligibility_truth,
        query_source_manifest=query_source_manifest)
    cache_lookup_started = time.perf_counter()
    frozen_calibration = None
    calibration_cache_lookup = {
        "status": "not_requested",
        "checks": {"fresh_final_cache_requested": False},
    }
    if frozen_replay is not None and args.fresh_final_replay:
        frozen_calibration, calibration_cache_lookup = \
            _frozen_calibration_cache_hit(
                frozen_replay, calibration_cache_key,
                deployment_policy_artifact)
    cache_lookup_seconds = time.perf_counter() - cache_lookup_started
    if frozen_calibration is not None:
        theta = float(frozen_calibration["theta"])
        calibration_curve = frozen_calibration["curve"]
        calibration_timing = {
            "offline_calibration_materialization_seconds": 0.0,
            "offline_theta_sweep_seconds": 0.0,
            "offline_calibration_total_seconds": float(
                cache_lookup_seconds),
            "excluded_from_online_e2e": True,
            "online_timer_fields_unchanged": True,
            "materialization_breakdown_seconds": {
                "frozen_sha_cache_validation": float(
                    cache_lookup_seconds),
            },
            "materialization_path": "frozen_deployment_theta_sha_cache_hit",
            "cache_hit": True,
            "state_counts": {
                "hard_pass": 0, "hard_fail": 0,
                "statistical": 0, "abstain": 0,
            },
            "state_fractions": {
                "hard_pass": 0.0, "hard_fail": 0.0,
                "statistical": 0.0, "abstain": 0.0,
            },
            "offline_materialized_cache_resident_bytes": 0,
            "offline_prepared_query_array_bytes": 0,
            "offline_owned_array_bytes": 0,
        }
    else:
        start = time.perf_counter()
        calibration_ann = _ann_backend(
            "exact", calibration["embeddings"], args.seed)
        calibration_ann_build_seconds = time.perf_counter() - start
        calibration_model = PaperMIFA(
            parts["train"].normalized_embeddings(), train_observed,
            calibration["embeddings"], calibration["observed"], spec.dtypes,
            deployment_policy, theta=0.5,
            projection_dim=design.structural_config.projection_dim,
            leaf_capacity=design.structural_config.leaf_capacity,
            ann_backend=calibration_ann,
            ann_seed_size=0, seed=args.seed,
            external_ann_build_seconds=calibration_ann_build_seconds)
        theta, calibration_curve, calibration_timing = \
            calibrate_threshold_grouped(
                calibration_model.inference, calibration["queries"],
                calibration["truth_ids"], group_ids, theta_grid,
                progress_callback=_progress,
                eligibility_truth_masks=calibration_eligibility_truth,
                loss_tolerance=args.calibration_loss_tolerance,
                return_timing=True)
        calibration_timing["cache_hit"] = False
        calibration_timing["cache_lookup_seconds"] = float(
            cache_lookup_seconds)
    if args.deployed_theta is not None:
        calibrated_theta = float(theta)
        theta = float(args.deployed_theta)
        _progress(
            "deployed_theta_loaded", theta=theta,
            calibration_recomputed_theta=calibrated_theta)
    calibration_cache = {
        "schema_version": 1,
        "status": (
            "reused_frozen_deployment_policy_and_theta"
            if frozen_calibration is not None else
            "computed_after_safe_cache_miss"
            if calibration_cache_lookup["status"] == "miss" else
            "computed_for_selection"),
        "key_sha256": calibration_cache_key["sha256"],
        "key_payload": calibration_cache_key["payload"],
        "deployment_policy_sha256": deployment_policy_artifact["sha256"],
        "selected_theta": float(theta),
        "calibration_curve_sha256": _canonical_json_sha256(
            calibration_curve),
        "lookup": calibration_cache_lookup,
        "source_report": (
            frozen_replay_manifest["source_report"]
            if frozen_calibration is not None else
            "N/A_SELECTION_COMPUTATION"),
        "source_report_sha256": (
            frozen_replay_manifest["source_report_sha256"]
            if frozen_calibration is not None else
            "N/A_SELECTION_COMPUTATION"),
        "cache_key_includes_resolved_k": True,
        "cache_key_includes_beta_exp": True,
        "excluded_from_online_e2e": True,
    }
    if frozen_replay is not None:
        theta_match = bool(np.isclose(
            float(theta), float(frozen_replay["selected_theta"]),
            rtol=0.0, atol=1e-12))
        if not theta_match:
            raise RuntimeError(
                "frozen design replay did not reproduce calibration theta")
        frozen_replay_manifest["selected_theta_match"] = True
        frozen_replay_manifest["status"] = (
            "complete_training_validation_calibration_design_theta_replay_"
            "with_disjoint_fresh_final"
            if args.fresh_final_replay else
            "complete_exact_context_design_theta_replay")
    _progress("theta_calibrated", theta=theta, groups=len(set(group_ids)))

    distance_provider = None
    if args.distance_backend == "cpu_resident":
        distance_provider = NumpyResidentDistanceProvider(test["embeddings"])
    elif args.distance_backend == "torch_resident":
        distance_provider = TorchResidentDistanceProvider(
            test["embeddings"], device=args.gpu_device)
    start = time.perf_counter()
    test_ann = _ann_backend(
        args.ann_backend, test["embeddings"], args.seed)
    test_ann_build_seconds = time.perf_counter() - start
    test_model = PaperMIFA(
        parts["train"].normalized_embeddings(), train_observed,
        test["embeddings"], test["observed"], spec.dtypes,
        deployment_policy, theta=theta,
        projection_dim=design.structural_config.projection_dim,
        leaf_capacity=design.structural_config.leaf_capacity,
        ann_backend=test_ann,
        ann_seed_size=args.ann_seed_size, seed=args.seed,
        leaf_frontier=args.leaf_frontier,
        distance_provider=distance_provider,
        allow_distance_order_specialization=(
            args.executor_path == "distance_order"),
        external_ann_build_seconds=test_ann_build_seconds)
    candidate_cohort_cache = None
    if args.candidate_efficiency_matrix:
        candidate_cohort_cache = \
            test_model.inference.configure_candidate_cohort_cache(
                max(2048, int(args.candidate_efficiency_query_count)))
    test_theta_diagnostic = None
    if args.test_theta_diagnostic:
        test_group_ids = [int(query["src"]) for query in test["queries"]]
        test_optimal_theta, test_theta_curve = calibrate_threshold_grouped(
            test_model.inference, test["queries"], test["truth_ids"],
            test_group_ids, theta_grid, progress_callback=_progress)
        selected_row = min(
            test_theta_curve, key=lambda row: abs(row["theta"] - theta))
        optimal_row = min(
            test_theta_curve, key=lambda row: (
                row["group_macro_loss"], row["theta"]))
        test_theta_diagnostic = {
            "status": "diagnostic_only_test_hidden_values_used",
            "eligible_for_method_selection": False,
            "calibration_selected_theta": float(theta),
            "test_optimal_theta": float(test_optimal_theta),
            "calibration_selected_test_group_macro_f1": float(
                selected_row["group_macro_f1"]),
            "test_optimal_group_macro_f1": float(
                optimal_row["group_macro_f1"]),
            "selection_regret_f1": float(
                optimal_row["group_macro_f1"] -
                selected_row["group_macro_f1"]),
            "curve": test_theta_curve,
        }
        _progress(
            "test_theta_diagnostic_complete",
            calibration_theta=theta, test_optimal_theta=test_optimal_theta,
            selection_regret=test_theta_diagnostic["selection_regret_f1"])
    retain_ablation_distances = int(args.executor_ablation_queries) > 0
    mifa_ids, execution = _run_search(
        test_model, test, args.epsilon, progress_callback=_progress,
        distance_query_batch_size=args.distance_query_batch_size,
        exact_audit_queries=args.exact_audit_queries,
        retain_prepared_distances=retain_ablation_distances)
    ablation_distances = execution.pop("_prepared_distances", None)
    ablation_distance_ms = execution.pop(
        "_distance_preparation_share_ms", None)
    if args.epsilon == 0.0 and not execution["executor_exact_matches_fixed_policy"]:
        raise AssertionError("unified executor disagrees with fixed-policy exact oracle")
    predicate_diagnostics = (
        None if args.skip_predicate_diagnostics else
        _predicate_diagnostics(test_model, test))
    drop_ids = _drop_ids(test)
    results = {
        "drop": legacy._score_ids(
            drop_ids, test["truth_ids"], test["queries"],
            test.get("headline_admitted"),
            test.get("masked_answer_affected")),
        "mifa_paper_static": legacy._score_ids(
            mifa_ids, test["truth_ids"], test["queries"],
            test.get("headline_admitted"),
            test.get("masked_answer_affected")),
    }
    for name, ids in (("drop", drop_ids), ("mifa_paper_static", mifa_ids)):
        direct = legacy._score_ids(
            ids, test["truth_ids"], test["queries"],
            masked_answer_affected=test.get("affected_query_direct"))
        if "masked_answer" in direct:
            results[name]["affected_query_direct"] = direct["masked_answer"]
    aligned_baseline_manifest = None
    if args.aligned_baselines:
        aligned_results, aligned_baseline_manifest = _run_aligned_baselines(
            args.aligned_baselines, spec=spec, parts=parts,
            train_observed=train_observed, train_mask=train_mask,
            query_specs=query_specs, native_filters=native_filters,
            predicate_reference=predicate_reference,
            signature_mix=signature_mix,
            canonical_bucket_weights=canonical_bucket_weights,
            role_counts=role_counts, calibration=calibration, test=test,
            seed=args.seed)
        results.update(aligned_results)
    implementation = test_model.implementation_manifest()
    executor_ablation = _executor_ablation(
        test_model, test, execution.get("per_query", []),
        args.executor_ablation_queries,
        prepared_distances=ablation_distances,
        distance_preparation_ms=ablation_distance_ms,
        query_workers=args.executor_ablation_workers)
    del ablation_distances, ablation_distance_ms
    support_efficiency = _support_efficiency_benchmark(
        test_model, test, args.support_efficiency_requests)
    main_inference_counters = dict(test_model.inference.counters)
    pruning_diagnostic = _run_disable_one_diagnostic(
        test_model, test["queries"],
        query_count=args.pruning_diagnostic_queries,
        reference_rows=execution.get("per_query", []),
        workload_name=f"{spec.name} current paper-default",
        variant_keys=pruning_variant_keys or None,
        persist_per_query=args.pruning_diagnostic_persist_per_query)
    # The diagnostic is a separate, opt-in measurement and must not change
    # the main execution's persisted inference counters.
    test_model.inference.counters.clear()
    test_model.inference.counters.update(main_inference_counters)
    grouped_exact = _grouped_exact_benchmark(
        test_model, test, mifa_ids, args.grouped_exact_backends,
        args.gpu_device, args.grouped_benchmark_repeats)
    candidate_efficiency = None
    if args.candidate_efficiency_matrix:
        efficiency_devices = (() if args.candidate_efficiency_cpu_only else
            tuple(
                item.strip() for item in (
                    args.candidate_efficiency_gpu_devices or
                    args.scale_gpu_devices or args.gpu_device).split(",")
                if item.strip()))
        response_strata = tuple(
            item.strip() for item in
            args.candidate_efficiency_response_strata.split(",")
            if item.strip())
        _progress(
            "candidate_efficiency_started",
            query_count=args.candidate_efficiency_query_count,
            devices=efficiency_devices)
        candidate_efficiency = _run_candidate_efficiency_matrix(
            test_model, test, mifa_ids,
            query_count=args.candidate_efficiency_query_count,
            gpu_devices=efficiency_devices,
            repeats=args.candidate_efficiency_repeats,
            initial_page_size=args.candidate_page_initial,
            max_page_size=args.candidate_page_maximum,
            response_strata=response_strata,
            response_generic_cpu=(
                args.candidate_efficiency_response_generic_cpu),
            response_only=args.candidate_efficiency_response_only,
            response_recompute_cpu_reference=not (
                args.candidate_efficiency_response_use_frozen_reference),
            platforms=tuple(
                item.strip() for item in
                args.candidate_efficiency_platforms.split(",")
                if item.strip()),
            cpu_projection_dim=(
                args.candidate_efficiency_cpu_projection_dim),
            cpu_refine_width=args.candidate_efficiency_cpu_refine_width,
            cpu_approximate_shortlist=(
                args.candidate_efficiency_cpu_approximate_shortlist),
            exact_prewarm_strategy=(
                args.candidate_efficiency_exact_prewarm_strategy),
            cpu_fused_exact_min_work=(
                args.candidate_efficiency_cpu_fused_exact_min_work),
            gpu_resident_exact_postings=(
                args.candidate_efficiency_gpu_resident_exact_postings),
            response_saturated_throughput=(
                args.candidate_efficiency_response_saturated_throughput))
        _progress(
            "candidate_efficiency_completed",
            all_ordered_exact=candidate_efficiency["all_ordered_exact"],
            all_zero_fallbacks=candidate_efficiency["all_zero_fallbacks"])
    scale_benchmark = None
    if args.scale_searchable_n:
        scale_query_rows = None
        scale_cohort_receipt = None
        scale_jitter_reference = None
        if args.scale_jitter_reference_embeddings:
            reference_path = (
                ROOT / args.scale_jitter_reference_embeddings).resolve()
            if not reference_path.is_file():
                raise ValueError(
                    "scale jitter reference embedding file does not exist")
            reference_all = np.load(reference_path, mmap_mode="r")
            if reference_all.ndim != 2 or len(reference_all) != spec.n:
                raise ValueError(
                    "scale jitter reference must align with the complete "
                    "post-query-source dataset")
            if reference_all.shape[1] < args.scale_dimension:
                raise ValueError(
                    "scale jitter reference does not cover the requested "
                    "dimension")
            scale_jitter_reference = np.ascontiguousarray(
                reference_all[split.test], dtype=np.float32)
        if args.scale_cohort_artifact:
            scale_cohort_path = ROOT / args.scale_cohort_artifact
            scale_cohort_payload = json.loads(scale_cohort_path.read_text())
            try:
                if "query_rows" in scale_cohort_payload:
                    scale_query_rows = scale_cohort_payload["query_rows"][
                        args.scale_cohort_role]
                    cohort_row_source = "scale_result.query_rows"
                else:
                    scale_query_rows = scale_cohort_payload["tracks"][
                        args.scale_cohort_track]["full_k_query_rows"][
                            args.scale_cohort_role]
                    cohort_row_source = "track.full_k_query_rows"
            except KeyError as exc:
                raise ValueError(
                    "scale cohort artifact lacks the requested track/role"
                ) from exc
            scale_cohort_receipt = {
                "path": str(scale_cohort_path.relative_to(ROOT)),
                "sha256": (
                    hashlib.sha256(
                        scale_cohort_path.read_bytes()).hexdigest()
                    if args.scale_record_digests else None),
                "track": args.scale_cohort_track,
                "role": args.scale_cohort_role,
                "row_source": cohort_row_source,
            }
        scale_devices = tuple(
            item.strip() for item in
            ("" if args.scale_cpu_only else (
                args.scale_gpu_devices or args.gpu_device)).split(",")
            if item.strip())
        if args.scale_query_repetitions < 1:
            raise ValueError("--scale-query-repetitions must be positive")
        if (scale_query_rows is not None and
                args.scale_query_repetitions > 1):
            scale_query_rows = np.tile(
                np.asarray(scale_query_rows, dtype=np.int64),
                args.scale_query_repetitions)
        if scale_cohort_receipt is not None:
            scale_cohort_receipt["query_repetitions"] = int(
                args.scale_query_repetitions)
        scale_searchable_context, scale_handler_receipt = \
            _scale_searchable_handler(
                test, spec.dtypes,
                handler=args.scale_searchable_handler,
                repaired_table=args.scale_repaired_test_table,
                repair_receipt=args.scale_repair_test_receipt)
        _progress(
            "scale_benchmark_started", target_n=args.scale_searchable_n,
            query_count=args.scale_query_count, devices=scale_devices,
            searchable_handler=args.scale_searchable_handler)
        scale_benchmark = _run_scale_benchmark(
            training_embeddings=parts["train"].normalized_embeddings(),
            training_table=train_observed,
            searchable_context=scale_searchable_context,
            dtypes=spec.dtypes, policy=deployment_policy, theta=theta,
            structural_config=design.structural_config, seed=args.seed,
            target_n=args.scale_searchable_n,
            query_count=args.scale_query_count,
            gpu_devices=scale_devices,
            distance_batch_size=args.scale_distance_batch_size,
            repeats=args.scale_repeats, noise_std=args.scale_noise_std,
            dimension=args.scale_dimension,
            expansion_seed=args.scale_expansion_seed,
            query_rows=scale_query_rows,
            jitter_reference_embeddings=scale_jitter_reference,
            executor_backend=args.scale_executor_backend,
            verify_exact_outputs=args.scale_verify_exact_outputs,
            record_digests=args.scale_record_digests,
            initial_page_size=args.candidate_page_initial,
            max_page_size=args.candidate_page_maximum,
            candidate_multi_gpu_mode=
            args.scale_candidate_multi_gpu_mode,
            candidate_score_precision=
            args.scale_candidate_score_precision,
            tree_probe_counts=_csv_ints(args.scale_tree_probe_counts),
            tree_probe_result_width=args.scale_tree_probe_result_width,
            ivf_probe_counts=_csv_ints(args.scale_ivf_probe_counts),
            ivf_nlist=args.scale_ivf_nlist,
            ivf_training_rows=args.scale_ivf_training_rows,
            ivf_iterations=args.scale_ivf_iterations,
            ivf_build_threads=args.scale_ivf_build_threads,
            pruning_diagnostic_queries=
            args.scale_pruning_diagnostic_queries,
            quality_reference=args.scale_quality_reference,
            quality_query_count=args.scale_quality_query_count,
            quality_distance_reference=
            args.scale_quality_distance_reference,
            response_query_count=args.scale_response_query_count,
            batch_sizes=_csv_ints(args.scale_batch_sizes),
            cpu_batch1_query_count=args.scale_cpu_batch1_query_count)
        scale_benchmark["cohort_receipt"] = scale_cohort_receipt
        scale_benchmark["searchable_handler"] = scale_handler_receipt
        _progress(
            "scale_benchmark_completed",
            target_n=scale_benchmark["corpus"]["target_rows"],
            exact_reference_queries=len(
                scale_benchmark["exact_reference_comparisons"]),
            strong_scaling=scale_benchmark["strong_scaling"])
    runtime_mechanism_coverage = {
        "ancestor_backoff_exercised": bool(
            test_model.inference.counters["ancestor_context_requests"]),
        "scalar_backoff_exercised": bool(
            test_model.inference.counters["scalar_relaxation_requests"]),
        "statistical_accept_exercised": bool(
            test_model.inference.counters["statistical_accept"]),
        "statistical_reject_exercised": bool(
            test_model.inference.counters["statistical_reject"]),
        "lazy_boundary_skip_exercised": bool(
            execution["counters"].get("inferences_skipped_by_boundary", 0)),
    }
    required_static = (
        "projected_hierarchy", "ancestor_backoff", "scalar_backoff",
        "deterministic_constraint_domains",
        "ordered_templates", "joint_estimator", "recursive_shrinkage",
        "per_template_score_maps", "indexed_support_retrieval", "ann_seed",
        "unified_node_tuple_queue", "lazy_statistical_inference",
        "exact_termination", "joint_structure_policy_selection",
        "cross_fitted_score_maps",
    )
    implementation_coverage_complete = all(
        implementation["features"].get(feature, False)
        for feature in required_static)
    payload = {
        "status": "complete",
        "paper_commit": "42970186db5a54c4918971c815c1ec70b150b6da",
        "experiment_arguments": vars(args),
        "workload_profile": args.workload_profile,
        "workload_reference": predicate_reference_manifest,
        "dataset": dataset_manifest,
        "protocol": paper_config.get("protocol", {}),
        "seed": int(args.seed),
        "injection_seed": int(injection_seed),
        "rho_tuple": float(args.rho),
        "historical_evidence": {
            "fraction": float(args.historical_evidence_fraction),
            "rows": int(evidence_count),
            "full_training_rows": int(full_training_part.n),
            "nested_order_seed": int(0x78C38DA5 + args.seed),
            "rows_digest": _digest_array(evidence_rows),
            "validation_calibration_test_fixed": True,
        },
        "missingness": {
            "kind": args.missingness_kind,
            "training_kind": train_missingness_kind,
            "historical_candidate_exposure": ({
                "beta_exp": float(args.historical_beta_exp),
                "top_l": int(historical_top_l),
                "top_l_definition": (
                    f"{int(args.historical_topk_multiplier)} * k"),
                "historical_queries": int(
                    len(historical_query_descriptors)),
                "predicate_attribute_identity": "frozen_historical_queries",
                "candidate_retrieval": "exact_embedding_only",
                "nested_protocol": (
                    HISTORICAL_EXPOSURE_NESTED_PROTOCOL
                    if args.missingness_kind ==
                    "historical_candidate_exposure_nested" else None),
            } if args.missingness_kind in
                 HISTORICAL_EXPOSURE_KINDS else None),
            "answer_impact_fraction": (
                float(args.answer_impact_fraction)
                if args.missingness_kind == "tuple_query_answer_mnar"
                else None),
            "answer_cells_per_tuple": (
                int(args.answer_cells_per_tuple)
                if args.missingness_kind == "tuple_query_answer_mnar"
                else None),
            "block_size": (int(args.block_size)
                           if args.missingness_kind in {
                               "fixed_tuple_block", "joint_tuple_block"}
                           else int(args.answer_cells_per_tuple)
                           if args.missingness_kind == "tuple_query_answer_mnar"
                           else 1),
            "block_attrs": list(block_attrs),
            "attribute_weights": attribute_weights_manifest,
        },
        "split": {**split.as_dict(include_indices=False), "digest": split.digest()},
        "query_source": query_source_manifest,
        "train_mask": {
            "digest": _digest_array(np.packbits(train_mask)),
            "incomplete_tuple_rate": float(train_mask.any(axis=1).mean()),
            "masked_cell_rate": float(train_mask.mean()),
            "nested_exposure_protocol": train_nested_exposure_protocol,
        },
        "partitions": {context["role"]: _context_manifest(context)
                       for context in (validation, calibration, test)},
        "validation_instances": instance_manifest,
        "training_only_body_candidates": {
            "|".join(targets): [list(body) for body in bodies]
            for targets, bodies in body_candidates.items()},
        "selected_design": {
            "structure": design.structural_config.key,
            "policy_candidate": design.candidate_id,
            "temporary_validation_templates": [
                template.template_id for template in design.policy.templates],
            "templates": [
                template.template_id for template in deployment_policy.templates],
            "validation_policy_fit_manifest": design.policy.fit_manifest,
            "policy_fit_manifest": deployment_policy.fit_manifest,
            "final_refit_structures": final_refit,
        },
        "validation_candidates": design.rows,
        "selected_theta": float(theta),
        "deployment_policy_artifact": deployment_policy_artifact,
        "calibration_cache": calibration_cache,
        "offline_calibration_materialization_seconds": float(
            calibration_timing[
                "offline_calibration_materialization_seconds"]),
        "offline_theta_sweep_seconds": float(
            calibration_timing["offline_theta_sweep_seconds"]),
        "offline_calibration_total_seconds": float(
            calibration_timing["offline_calibration_total_seconds"]),
        "offline_calibration_timing": calibration_timing,
        "frozen_design_replay": frozen_replay_manifest,
        "fresh_final_protocol": fresh_final_protocol,
        "calibration_curve": calibration_curve,
        "test_theta_diagnostic": test_theta_diagnostic,
        "implementation": implementation,
        "paper_core_static_semantic_coverage": {
            "required": list(required_static),
            "complete": bool(implementation_coverage_complete),
        },
        "execution": execution,
        "predicate_diagnostics": predicate_diagnostics,
        "executor_ablation": executor_ablation,
        "support_efficiency": support_efficiency,
        "pruning_diagnostic": pruning_diagnostic,
        "grouped_exact": grouped_exact,
        "candidate_efficiency": candidate_efficiency,
        "candidate_cohort_cache": candidate_cohort_cache,
        "scale_benchmark": scale_benchmark,
        "inference_counters": dict(test_model.inference.counters),
        "runtime_mechanism_coverage": runtime_mechanism_coverage,
        "results": results,
        "aligned_baseline_manifest": aligned_baseline_manifest,
        "pilot_delta_vs_drop": (
            results["mifa_paper_static"]["macro"]["f1"] -
            results["drop"]["macro"]["f1"]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads_env": os.environ.get(
                "OPENBLAS_NUM_THREADS"),
            "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
            "omp_proc_bind_env": os.environ.get("OMP_PROC_BIND"),
            "omp_places_env": os.environ.get("OMP_PLACES"),
            "scale_host_placement": os.environ.get(
                "MIFA_SCALE_HOST_PLACEMENT", "default_first_touch"),
            "distance_backend": active_backend(),
            "paper_executor_distance_backend": args.distance_backend,
            "gpu_device": (args.gpu_device if distance_provider is not None else None),
            "ann_backend": args.ann_backend,
            "git_head": _git_head(),
            "source_provenance": _source_provenance(),
            "paper_config_path": str(paper_path.relative_to(ROOT)),
            "dataset_config_path": str(dataset_path.relative_to(ROOT)),
            "config_content_hashes_recorded": False,
        },
    }
    if args.persist_test_truth_ids:
        payload["test_truth_ids"] = np.asarray(
            test["truth_ids"], dtype=np.int64).tolist()
        payload["test_query_source_rows"] = np.asarray(
            test["query_rows"], dtype=np.int64).tolist()
    if test.get("matched_arity") is not None:
        payload["matched_arity_design"] = test["matched_arity"]
    out = ROOT / "reports" / (
        f"paper_aligned_static_{spec.name}_seed{args.seed}_{args.tag}.json")
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({
        "report": str(out),
        "implementation_coverage_complete": implementation_coverage_complete,
        "exact_executor": execution["executor_exact_matches_fixed_policy"],
        "mifa_f1": results["mifa_paper_static"]["macro"]["f1"],
        "drop_f1": results["drop"]["macro"]["f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
