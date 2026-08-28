"""CPU-resident exact candidate pages for the paper executor.

This provider is the CPU counterpart of the Phase-A CUDA candidate stream.
It keeps complete exact distances and hard-predicate states on the host, but
does not allocate the legacy leaf scheduler's ``registered`` and heap slabs.
Only the exact ``(distance, row_id)`` prefix requested by the shared compact
consumer is ranked and materialized.

The implementation intentionally uses the same NumPy SGEMM distance formula
as :class:`NumpyResidentDistanceProvider`.  It changes only the physical
executor; predicate, inference, policy, and threshold semantics are shared
with the frozen paper path.
"""
from __future__ import annotations

from collections import OrderedDict
from time import perf_counter

import numpy as np
from numba import get_num_threads, njit, prange

from .cpu_distance import NumpyResidentDistanceProvider
from .gpu_candidate_stream import (
    CandidatePage,
    CandidateStreamHandle,
    HARD_FAIL,
    HARD_PASS,
    UNRESOLVED,
)


@njit(cache=True, parallel=True, fastmath=False)
def _hard_states_batch_numba(
        values, missing, attrs, ops, lower, upper, atom_counts,
        statistical_group_ids, statistical_decisions,
        use_statistical_groups):
    """Fuse every atom and query into one parallel hard-state pass."""
    query_count = len(atom_counts)
    base_rows = values.shape[1]
    states = np.empty((query_count, base_rows), dtype=np.int8)
    for query_index in prange(query_count):
        atom_count = int(atom_counts[query_index])
        for row in range(base_rows):
            hard_fail = False
            unresolved = False
            for atom_index in range(atom_count):
                attr = int(attrs[query_index, atom_index])
                if missing[attr, row]:
                    unresolved = True
                    continue
                value = values[attr, row]
                op = int(ops[query_index, atom_index])
                if op == 0:
                    satisfied = value <= lower[query_index, atom_index]
                elif op == 1:
                    satisfied = value >= lower[query_index, atom_index]
                elif op == 2:
                    satisfied = (
                        value >= lower[query_index, atom_index] and
                        value <= upper[query_index, atom_index])
                else:
                    satisfied = value == lower[query_index, atom_index]
                if not satisfied:
                    hard_fail = True
                    break
            if hard_fail:
                states[query_index, row] = HARD_FAIL
            elif unresolved:
                if use_statistical_groups:
                    group_id = int(statistical_group_ids[row])
                    if (group_id >= 0 and
                            statistical_decisions[
                                query_index, group_id] == 1):
                        states[query_index, row] = HARD_PASS
                    else:
                        states[query_index, row] = HARD_FAIL
                else:
                    states[query_index, row] = UNRESOLVED
            else:
                states[query_index, row] = HARD_PASS
    return states


@njit(cache=True, parallel=True, fastmath=False)
def _candidate_pages_batch_numba(
        scores, states, global_ids, active, starts, targets, width):
    """Create exact deterministic candidate pages in parallel by query."""
    output_ids = np.full((len(active), width), -1, dtype=np.int64)
    output_distances = np.full(
        (len(active), width), np.inf, dtype=np.float32)
    output_states = np.zeros((len(active), width), dtype=np.int8)
    distance_bits = scores.view(np.uint32)
    base_rows = scores.shape[1]
    maximum_key = np.uint64(0xFFFFFFFFFFFFFFFF)
    for output_index in prange(len(active)):
        query_index = int(active[output_index])
        start = int(starts[output_index])
        target = int(targets[output_index])
        count = target - start
        keys = np.empty(base_rows, dtype=np.uint64)
        for row in range(base_rows):
            if states[query_index, row] == HARD_FAIL:
                keys[row] = maximum_key
            else:
                keys[row] = (
                    (np.uint64(distance_bits[query_index, row]) <<
                     np.uint64(32)) |
                    global_ids[row])
        if target < base_rows:
            prefix = np.argpartition(keys, target - 1)[:target]
        else:
            prefix = np.arange(base_rows, dtype=np.int64)
        order = np.argsort(keys[prefix])
        ranked = prefix[order]
        for slot in range(count):
            row = int(ranked[start + slot])
            output_ids[output_index, slot] = row
            output_distances[output_index, slot] = scores[query_index, row]
            output_states[output_index, slot] = states[query_index, row]
    return output_ids, output_distances, output_states


@njit(cache=True, parallel=True, fastmath=False)
def _fused_hard_topk_batch_numba(
        scores, values, missing, attrs, ops, lower, upper, atom_counts,
        statistical_group_ids, statistical_decisions, width):
    """Fuse exact hard/statistical classification with compact top-k."""
    query_count = scores.shape[0]
    base_rows = scores.shape[1]
    output_ids = np.full((query_count, width), -1, dtype=np.int64)
    output_distances = np.full(
        (query_count, width), np.inf, dtype=np.float32)
    output_counts = np.zeros(query_count, dtype=np.int64)
    for query_index in prange(query_count):
        count = 0
        for row in range(base_rows):
            hard_fail = False
            unresolved = False
            for atom_index in range(int(atom_counts[query_index])):
                attr = int(attrs[query_index, atom_index])
                if missing[attr, row]:
                    unresolved = True
                    continue
                value = values[attr, row]
                op = int(ops[query_index, atom_index])
                if op == 0:
                    satisfied = value <= lower[query_index, atom_index]
                elif op == 1:
                    satisfied = value >= lower[query_index, atom_index]
                elif op == 2:
                    satisfied = (
                        value >= lower[query_index, atom_index] and
                        value <= upper[query_index, atom_index])
                else:
                    satisfied = value == lower[query_index, atom_index]
                if not satisfied:
                    hard_fail = True
                    break
            if hard_fail:
                continue
            if unresolved:
                group = int(statistical_group_ids[row])
                if group < 0 or statistical_decisions[
                        query_index, group] != 1:
                    continue
            distance = scores[query_index, row]
            if count == width:
                worst_distance = output_distances[query_index, width - 1]
                worst_id = output_ids[query_index, width - 1]
                if distance > worst_distance or \
                        (distance == worst_distance and row >= worst_id):
                    continue
                position = width - 1
            else:
                position = count
                count += 1
            while position > 0:
                previous_distance = output_distances[
                    query_index, position - 1]
                previous_id = output_ids[query_index, position - 1]
                if previous_distance < distance or \
                        (previous_distance == distance and previous_id <= row):
                    break
                output_distances[query_index, position] = previous_distance
                output_ids[query_index, position] = previous_id
                position -= 1
            output_distances[query_index, position] = distance
            output_ids[query_index, position] = row
        output_counts[query_index] = count
    return output_ids, output_distances, output_counts


@njit(cache=True, parallel=True, fastmath=True)
def _eligible_shortlist_numba(
        base, base_sqnorm, query, eligible_ids, width, partitions):
    """Fuse eligible-row distance evaluation with parallel local top-k.

    The vector loop is deliberately compiled with reassociation/FMA enabled
    so LLVM can emit the AVX-512 reduction used by the latency path.  These
    scores select only a conservative shortlist.  The provider recomputes
    that shortlist with the unchanged NumPy FP32 distance formula before it
    returns any row, so the public ordering remains the exact stable
    ``(float32 distance, row_id)`` ordering.
    """
    partitions = max(1, int(partitions))
    width = max(1, int(width))
    partial_distances = np.full(
        (partitions, width), np.float32(np.inf))
    partial_ids = np.full((partitions, width), np.int64(-1))
    partial_counts = np.zeros(partitions, dtype=np.int64)
    query_sqnorm = np.float32(0.0)
    for dimension in range(query.shape[0]):
        query_sqnorm = np.float32(
            query_sqnorm + np.float32(
                query[dimension] * query[dimension]))

    for partition in prange(partitions):
        start = len(eligible_ids) * partition // partitions
        end = len(eligible_ids) * (partition + 1) // partitions
        count = 0
        for position in range(start, end):
            row = int(eligible_ids[position])
            dot = np.float32(0.0)
            for dimension in range(query.shape[0]):
                dot = np.float32(
                    dot + np.float32(
                        base[row, dimension] * query[dimension]))
            distance = np.float32(
                base_sqnorm[row] - np.float32(2.0) * dot + query_sqnorm)
            if distance < 0.0:
                distance = np.float32(0.0)
            if count == width:
                if (distance > partial_distances[partition, width - 1] or
                        (distance == partial_distances[
                            partition, width - 1] and
                         row >= partial_ids[partition, width - 1])):
                    continue
                slot = width - 1
            else:
                slot = count
                count += 1
            while slot > 0:
                previous_distance = partial_distances[
                    partition, slot - 1]
                previous_id = partial_ids[partition, slot - 1]
                if (previous_distance < distance or
                        (previous_distance == distance and
                         previous_id <= row)):
                    break
                partial_distances[partition, slot] = previous_distance
                partial_ids[partition, slot] = previous_id
                slot -= 1
            partial_distances[partition, slot] = distance
            partial_ids[partition, slot] = row
        partial_counts[partition] = count

    output_distances = np.full(width, np.float32(np.inf))
    output_ids = np.full(width, np.int64(-1))
    output_count = 0
    for partition in range(partitions):
        for position in range(int(partial_counts[partition])):
            distance = partial_distances[partition, position]
            row = partial_ids[partition, position]
            if output_count == width:
                if (distance > output_distances[width - 1] or
                        (distance == output_distances[width - 1] and
                         row >= output_ids[width - 1])):
                    continue
                slot = width - 1
            else:
                slot = output_count
                output_count += 1
            while slot > 0:
                previous_distance = output_distances[slot - 1]
                previous_id = output_ids[slot - 1]
                if (previous_distance < distance or
                        (previous_distance == distance and
                         previous_id <= row)):
                    break
                output_distances[slot] = previous_distance
                output_ids[slot] = previous_id
                slot -= 1
            output_distances[slot] = distance
            output_ids[slot] = row
    return output_ids, output_count


@njit(cache=True, parallel=True, fastmath=True)
def _eligible_exhaustive_topk_numba(
        base, base_sqnorm, query, eligible_ids, width, partitions):
    """Score every eligible full-dimensional row without a gather copy.

    Each worker keeps a small stable local prefix.  The returned prefix is
    rescored by the caller with the resident NumPy FP32 formula before it is
    published.  Unlike the projected/quantized path, every eligible row is
    evaluated in the complete embedding space.
    """
    partitions = max(1, int(partitions))
    width = max(1, int(width))
    partial_distances = np.full(
        (partitions, width), np.float32(np.inf))
    partial_ids = np.full((partitions, width), np.int64(-1))
    partial_counts = np.zeros(partitions, dtype=np.int64)
    query_sqnorm = np.float32(0.0)
    for dimension in range(query.shape[0]):
        query_sqnorm = np.float32(
            query_sqnorm + np.float32(
                query[dimension] * query[dimension]))

    for partition in prange(partitions):
        start = len(eligible_ids) * partition // partitions
        end = len(eligible_ids) * (partition + 1) // partitions
        count = 0
        for position in range(start, end):
            row = int(eligible_ids[position])
            dot = np.float32(0.0)
            for dimension in range(query.shape[0]):
                dot = np.float32(
                    dot + np.float32(
                        base[row, dimension] * query[dimension]))
            distance = np.float32(
                base_sqnorm[row] - np.float32(2.0) * dot + query_sqnorm)
            if distance < 0.0:
                distance = np.float32(0.0)
            if count == width:
                if (distance > partial_distances[partition, width - 1] or
                        (distance == partial_distances[
                            partition, width - 1] and
                         row >= partial_ids[partition, width - 1])):
                    continue
                slot = width - 1
            else:
                slot = count
                count += 1
            while slot > 0:
                previous_distance = partial_distances[partition, slot - 1]
                previous_id = partial_ids[partition, slot - 1]
                if (previous_distance < distance or
                        (previous_distance == distance and
                         previous_id <= row)):
                    break
                partial_distances[partition, slot] = previous_distance
                partial_ids[partition, slot] = previous_id
                slot -= 1
            partial_distances[partition, slot] = distance
            partial_ids[partition, slot] = row
        partial_counts[partition] = count

    output_distances = np.full(width, np.float32(np.inf))
    output_ids = np.full(width, np.int64(-1))
    output_count = 0
    for partition in range(partitions):
        for position in range(int(partial_counts[partition])):
            distance = partial_distances[partition, position]
            row = partial_ids[partition, position]
            if output_count == width:
                if (distance > output_distances[width - 1] or
                        (distance == output_distances[width - 1] and
                         row >= output_ids[width - 1])):
                    continue
                slot = width - 1
            else:
                slot = output_count
                output_count += 1
            while slot > 0:
                previous_distance = output_distances[slot - 1]
                previous_id = output_ids[slot - 1]
                if (previous_distance < distance or
                        (previous_distance == distance and
                         previous_id <= row)):
                    break
                output_distances[slot] = previous_distance
                output_ids[slot] = previous_id
                slot -= 1
            output_distances[slot] = distance
            output_ids[slot] = row
    return output_ids, output_count


@njit(cache=True, parallel=True, fastmath=True)
def _eligible_quantized_shortlist_numba(
        quantized_base, base_scales, base_sqnorm, quantized_query,
        query_scale, query_sqnorm, eligible_ids, width, partitions):
    """Screen eligible rows with AVX-512-friendly signed-byte dot products.

    Quantized scores never leave this shortlist kernel.  The selected rows
    are rescored from the original resident FP32 vectors before publication,
    matching the GPU provider's low-precision-shortlist/FP32-refine design.
    """
    partitions = max(1, int(partitions))
    width = max(1, int(width))
    partial_distances = np.full(
        (partitions, width), np.float32(np.inf))
    partial_ids = np.full((partitions, width), np.int64(-1))
    partial_counts = np.zeros(partitions, dtype=np.int64)
    for partition in prange(partitions):
        start = len(eligible_ids) * partition // partitions
        end = len(eligible_ids) * (partition + 1) // partitions
        count = 0
        for position in range(start, end):
            row = int(eligible_ids[position])
            # A signed-byte dot over the supported embedding widths is far
            # below int32 overflow.  Keeping the accumulator at int32 avoids
            # widening every operand to int64 and lets LLVM vectorize the
            # fused screening loop with the native byte-to-dword path.
            integer_dot = np.int32(0)
            for dimension in range(quantized_query.shape[0]):
                integer_dot = np.int32(
                    integer_dot +
                    np.int32(quantized_base[row, dimension]) *
                    np.int32(quantized_query[dimension]))
            dot = (
                np.float32(integer_dot) * base_scales[row] * query_scale)
            distance = np.float32(
                base_sqnorm[row] - np.float32(2.0) * dot + query_sqnorm)
            if distance < 0.0:
                distance = np.float32(0.0)
            if count == width:
                if (distance > partial_distances[partition, width - 1] or
                        (distance == partial_distances[
                            partition, width - 1] and
                         row >= partial_ids[partition, width - 1])):
                    continue
                slot = width - 1
            else:
                slot = count
                count += 1
            while slot > 0:
                previous_distance = partial_distances[
                    partition, slot - 1]
                previous_id = partial_ids[partition, slot - 1]
                if (previous_distance < distance or
                        (previous_distance == distance and
                         previous_id <= row)):
                    break
                partial_distances[partition, slot] = previous_distance
                partial_ids[partition, slot] = previous_id
                slot -= 1
            partial_distances[partition, slot] = distance
            partial_ids[partition, slot] = row
        partial_counts[partition] = count

    output_distances = np.full(width, np.float32(np.inf))
    output_ids = np.full(width, np.int64(-1))
    output_count = 0
    for partition in range(partitions):
        for position in range(int(partial_counts[partition])):
            distance = partial_distances[partition, position]
            row = partial_ids[partition, position]
            if output_count == width:
                if (distance > output_distances[width - 1] or
                        (distance == output_distances[width - 1] and
                         row >= output_ids[width - 1])):
                    continue
                slot = width - 1
            else:
                slot = output_count
                output_count += 1
            while slot > 0:
                previous_distance = output_distances[slot - 1]
                previous_id = output_ids[slot - 1]
                if (previous_distance < distance or
                        (previous_distance == distance and
                         previous_id <= row)):
                    break
                output_distances[slot] = previous_distance
                output_ids[slot] = previous_id
                slot -= 1
            output_distances[slot] = distance
            output_ids[slot] = row
    return output_ids, output_count


class NumpyExactCandidateStreamProvider:
    """Produce exact candidate pages from CPU-resident full distances."""

    def __init__(self, base, base_sqnorm=None, *,
                 batch_one_cache_max_bytes=256 * 1024 * 1024,
                 batch_one_cache_max_entries=64,
                 batch_one_refine_width=64,
                 batch_one_projection_dim=None,
                 batch_one_quantized=True,
                 batch_one_approximate_shortlist=False,
                 batch_one_full_scan_switch_fraction=1.0,
                 batch_one_fused_exact_min_work=0):
        base = np.ascontiguousarray(base, dtype=np.float32)
        if base.ndim != 2:
            raise ValueError("base must be a 2-D matrix")
        if len(base) - 1 > np.iinfo(np.uint32).max:
            raise ValueError("exact candidate keys require row IDs in uint32")
        self.distance_provider = NumpyResidentDistanceProvider(
            base, base_sqnorm=base_sqnorm)
        self.base_rows = int(self.distance_provider.base_rows)
        self.dim = int(self.distance_provider.dim)
        self.global_ids = np.arange(self.base_rows, dtype=np.uint64)
        self.compact_topk_width = 10
        self.batch_one_topk_enabled = True
        self.batch_one_cache_max_bytes = int(batch_one_cache_max_bytes)
        self.batch_one_cache_max_entries = int(batch_one_cache_max_entries)
        self.batch_one_refine_width = int(batch_one_refine_width)
        self.batch_one_projection_dim = min(
            self.dim if batch_one_projection_dim is None else
            int(batch_one_projection_dim), self.dim)
        self.batch_one_quantized = bool(batch_one_quantized)
        self.batch_one_approximate_shortlist = bool(
            batch_one_approximate_shortlist)
        self.batch_one_full_scan_switch_fraction = float(
            batch_one_full_scan_switch_fraction)
        self.batch_one_fused_exact_min_work = int(
            batch_one_fused_exact_min_work)
        if self.batch_one_cache_max_bytes < 0:
            raise ValueError("batch-one cache byte limit cannot be negative")
        if self.batch_one_cache_max_entries < 0:
            raise ValueError("batch-one cache entry limit cannot be negative")
        if self.batch_one_refine_width < 10:
            raise ValueError("batch-one refine width must be at least 10")
        if self.batch_one_projection_dim < 1:
            raise ValueError("batch-one projection dimension must be positive")
        if not 0.0 <= self.batch_one_full_scan_switch_fraction <= 1.0:
            raise ValueError(
                "batch-one full-scan switch fraction must be in [0, 1]")
        if self.batch_one_fused_exact_min_work < 0:
            raise ValueError("batch-one fused exact work threshold is negative")
        if self.batch_one_projection_dim < self.dim:
            projected_base = np.ascontiguousarray(
                self.distance_provider.base[
                    :, :self.batch_one_projection_dim], dtype=np.float32)
            projected_norm = np.sqrt(np.einsum(
                "ij,ij->i", projected_base, projected_base))
            projected_norm = np.maximum(
                projected_norm, np.finfo(np.float32).tiny)
            projected_base /= projected_norm[:, None]
            self._batch_one_projection_base = projected_base
            self._batch_one_projection_sqnorm = np.einsum(
                "ij,ij->i", projected_base, projected_base).astype(
                    np.float32, copy=False)
        else:
            self._batch_one_projection_base = self.distance_provider.base
            self._batch_one_projection_sqnorm = \
                self.distance_provider.base_sqnorm
        self._batch_one_quantized_base = None
        self._batch_one_quantized_scales = None
        if self.batch_one_quantized:
            self._batch_one_quantized_base = np.empty(
                (self.base_rows, self.batch_one_projection_dim),
                dtype=np.int8)
            self._batch_one_quantized_scales = np.empty(
                self.base_rows, dtype=np.float32)
            block_rows = 65536
            for start in range(0, self.base_rows, block_rows):
                end = min(start + block_rows, self.base_rows)
                block = self._batch_one_projection_base[start:end]
                scales = np.max(np.abs(block), axis=1).astype(
                    np.float32, copy=False) / np.float32(127.0)
                scales = np.maximum(scales, np.finfo(np.float32).tiny)
                self._batch_one_quantized_scales[start:end] = scales
                quantized = np.rint(block / scales[:, None])
                np.clip(quantized, -127.0, 127.0, out=quantized)
                self._batch_one_quantized_base[start:end] = \
                    quantized.astype(np.int8)
        self._batch_one_eligibility_cache = OrderedDict()
        self._batch_one_eligibility_cache_bytes = 0
        self._batch_one_cache_hits = 0
        self._batch_one_cache_misses = 0
        # Compile both fused kernels outside measured batches.
        _hard_states_batch_numba(
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.bool_),
            np.zeros((1, 1), dtype=np.int32),
            np.zeros((1, 1), dtype=np.int8),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.ones(1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.zeros((1, 1), dtype=np.int8), False)
        _candidate_pages_batch_numba(
            np.zeros((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.int8),
            np.zeros(1, dtype=np.uint64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.ones(1, dtype=np.int64), 1)
        _fused_hard_topk_batch_numba(
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.bool_),
            np.zeros((1, 1), dtype=np.int32),
            np.zeros((1, 1), dtype=np.int8),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.ones(1, dtype=np.int32), np.zeros(1, dtype=np.int32),
            np.ones((1, 1), dtype=np.int8), 1)
        _eligible_shortlist_numba(
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.int64), 1, 1)
        _eligible_quantized_shortlist_numba(
            np.zeros((1, 1), dtype=np.int8),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.int8), np.float32(1.0),
            np.float32(0.0), np.zeros(1, dtype=np.int64), 1, 1)

    def clear_online_caches(self):
        """Clear mutable request caches without rebuilding the provider.

        The quantized corpus representation and compiled Numba kernels are
        deployment artifacts.  The eligibility postings below are populated
        by online predicates, so a cold-cache timing must reset them together
        with the inference-side serving caches.
        """
        entries = len(self._batch_one_eligibility_cache)
        bytes_ = int(self._batch_one_eligibility_cache_bytes)
        self._batch_one_eligibility_cache.clear()
        self._batch_one_eligibility_cache_bytes = 0
        self._batch_one_cache_hits = 0
        self._batch_one_cache_misses = 0
        return {
            "cleared_entries": int(entries),
            "cleared_bytes": bytes_,
            "all_online_caches_empty": True,
        }

    @staticmethod
    def _plan_source(plan):
        references = (
            plan["values"], plan["missing"], plan["attrs"], plan["ops"],
            plan["lower"], plan["upper"],
            plan.get("statistical_group_ids"),
            plan.get("statistical_group_decisions"),
            plan.get("precomputed_eligible_ids"),
        )
        source = tuple(id(value) for value in references)
        return source, references

    def _batch_one_eligible_ids(self, plan):
        source, references = self._plan_source(plan)
        cached = self._batch_one_eligibility_cache.pop(source, None)
        if cached is not None:
            self._batch_one_eligibility_cache[source] = cached
            self._batch_one_cache_hits += 1
            return cached[0], True

        self._batch_one_cache_misses += 1
        precomputed = plan.get("precomputed_eligible_ids")
        if precomputed is not None:
            eligible_ids = np.ascontiguousarray(
                precomputed, dtype=np.uint32)
            self._remember_eligible_ids(source, references, eligible_ids)
            return eligible_ids, False
        (group_ids, decisions, statistical_group_prefetch) = \
            self._statistical_group_data([plan], self.base_rows)
        if not statistical_group_prefetch:
            return None, False
        states, _ = self._hard_states([plan], self.base_rows)
        # Corpus row IDs are already constrained to uint32 by the provider.
        # Keeping cached postings at that width doubles the number of cohort
        # queries that fit under the same explicit byte budget; IDs are
        # widened only in the public candidate page.
        eligible_ids = np.ascontiguousarray(
            np.flatnonzero(states[0] != int(HARD_FAIL)), dtype=np.uint32)
        self._remember_eligible_ids(source, references, eligible_ids)
        return eligible_ids, False

    def _remember_eligible_ids(self, source, references, eligible_ids):
        """Insert one immutable posting under the configured LRU budget."""
        if (self.batch_one_cache_max_entries > 0 and
                eligible_ids.nbytes <= self.batch_one_cache_max_bytes):
            while self._batch_one_eligibility_cache and (
                    len(self._batch_one_eligibility_cache) >=
                    self.batch_one_cache_max_entries or
                    self._batch_one_eligibility_cache_bytes +
                    eligible_ids.nbytes > self.batch_one_cache_max_bytes):
                _, evicted = self._batch_one_eligibility_cache.popitem(
                    last=False)
                self._batch_one_eligibility_cache_bytes -= int(
                    evicted[0].nbytes)
            self._batch_one_eligibility_cache[source] = (
                eligible_ids, references)
            self._batch_one_eligibility_cache_bytes += int(
                eligible_ids.nbytes)

    def prewarm_eligibility(self, plans):
        """Materialize a fixed cohort's postings outside online timing."""
        plans = list(plans)
        if not plans:
            return {
                "plans": 0, "materialized": 0, "cache_entries": int(
                    len(self._batch_one_eligibility_cache)),
                "cache_bytes": int(self._batch_one_eligibility_cache_bytes),
            }
        grouped = OrderedDict()
        for plan in plans:
            key = (
                id(plan.get("statistical_group_ids")),
                len(plan.get("statistical_group_decisions", ())))
            grouped.setdefault(key, []).append(plan)
        materialized = 0
        skipped_unaligned = 0
        for current in grouped.values():
            missing = []
            scheduled_sources = set()
            for plan in current:
                source, references = self._plan_source(plan)
                if (source not in self._batch_one_eligibility_cache and
                        source not in scheduled_sources):
                    precomputed = plan.get("precomputed_eligible_ids")
                    if precomputed is not None:
                        eligible_ids = np.ascontiguousarray(
                            precomputed, dtype=np.uint32)
                        self._remember_eligible_ids(
                            source, references, eligible_ids)
                        materialized += 1
                    else:
                        missing.append((plan, source, references))
                    scheduled_sources.add(source)
            if not missing:
                continue
            states, statistical = self._hard_states(
                [item[0] for item in missing], self.base_rows)
            if not statistical:
                # Some exact predicates cannot share one immutable
                # statistical-group layout.  They remain valid and are
                # materialized by the unchanged per-request exact path.
                skipped_unaligned += len(missing)
                continue
            for index, (_, source, references) in enumerate(missing):
                eligible_ids = np.ascontiguousarray(
                    np.flatnonzero(states[index] != int(HARD_FAIL)),
                    dtype=np.uint32)
                self._remember_eligible_ids(
                    source, references, eligible_ids)
                materialized += 1
        return {
            "plans": int(len(plans)),
            "materialized": int(materialized),
            "skipped_unaligned": int(skipped_unaligned),
            "cache_entries": int(len(self._batch_one_eligibility_cache)),
            "cache_bytes": int(self._batch_one_eligibility_cache_bytes),
            "cache_max_entries": int(self.batch_one_cache_max_entries),
            "cache_max_bytes": int(self.batch_one_cache_max_bytes),
        }

    def begin_batch_topk(self, queries, plans, *, topk):
        """Return an exact compact page after eligibility-first CPU ranking.

        This batch-one specialization mirrors the CUDA response path while
        remaining entirely CPU resident.  Immutable statistical decisions
        are compiled into a bounded row-ID posting cache, after which one
        complete-dimensional SGEMV ranks the eligible posting exactly.
        """
        if not self.batch_one_topk_enabled:
            return None
        plans = list(plans)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        topk = int(topk)
        if len(queries) != 1 or len(plans) != 1 or topk <= 0:
            return None
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width does not match resident base")
        if not np.isfinite(queries).all():
            raise ValueError("candidate queries must be finite")
        plan = plans[0]
        has_statistical_groups = (
            "statistical_group_ids" in plan and
            "statistical_group_decisions" in plan)
        if (not has_statistical_groups and
                "precomputed_eligible_ids" not in plan):
            return None

        wall_started = perf_counter()
        eligible_ids, eligibility_cache_hit = \
            self._batch_one_eligible_ids(plan)
        if eligible_ids is None:
            return None
        cache_ready_ms = (perf_counter() - wall_started) * 1000.0
        selected_count = min(topk, len(eligible_ids))
        partitions = min(
            max(1, int(get_num_threads())), max(1, len(eligible_ids)))

        query_sqnorm = np.float32(queries[0] @ queries[0])
        full_scan_threshold = max(
            selected_count,
            int(np.ceil(
                self.base_rows * self.batch_one_full_scan_switch_fraction)))
        use_full_scan = False
        exhaustive_screen = False
        distance_rows_evaluated = len(eligible_ids)
        kernel_started = perf_counter()
        if self.batch_one_approximate_shortlist and len(eligible_ids):
            refine_width = min(
                max(selected_count, self.batch_one_refine_width),
                len(eligible_ids))
            if self.batch_one_projection_dim < self.dim:
                ranking_query = np.ascontiguousarray(
                    queries[0, :self.batch_one_projection_dim],
                    dtype=np.float32)
                ranking_norm = np.linalg.norm(ranking_query)
                if ranking_norm > 0.0:
                    ranking_query /= np.float32(ranking_norm)
            else:
                ranking_query = queries[0]
            if self.batch_one_quantized:
                query_scale = np.float32(
                    np.max(np.abs(ranking_query)) / np.float32(127.0))
                query_scale = max(
                    query_scale, np.finfo(np.float32).tiny)
                quantized_query = np.rint(ranking_query / query_scale)
                np.clip(quantized_query, -127.0, 127.0,
                        out=quantized_query)
                quantized_query = np.ascontiguousarray(
                    quantized_query, dtype=np.int8)
                ranking_query_sqnorm = np.float32(
                    ranking_query @ ranking_query)
                shortlist_ids, shortlist_count = \
                    _eligible_quantized_shortlist_numba(
                        self._batch_one_quantized_base,
                        self._batch_one_quantized_scales,
                        self._batch_one_projection_sqnorm,
                        quantized_query, query_scale,
                        ranking_query_sqnorm, eligible_ids,
                        refine_width, partitions)
            else:
                shortlist_ids, shortlist_count = _eligible_shortlist_numba(
                    self._batch_one_projection_base,
                    self._batch_one_projection_sqnorm,
                    ranking_query, eligible_ids, refine_width, partitions)
            ranking_ids = np.ascontiguousarray(
                shortlist_ids[:shortlist_count], dtype=np.uint32)
            execution_mode = "approximate_shortlist_fp32_refine"
        else:
            exhaustive_screen = (
                len(eligible_ids) * self.dim >=
                self.batch_one_fused_exact_min_work and
                selected_count < len(eligible_ids))
            if exhaustive_screen:
                refine_width = min(
                    max(selected_count, self.batch_one_refine_width),
                    len(eligible_ids))
                shortlist_ids, shortlist_count = \
                    _eligible_exhaustive_topk_numba(
                        self.distance_provider.base,
                        self.distance_provider.base_sqnorm,
                        queries[0], eligible_ids, refine_width, partitions)
                ranking_ids = np.ascontiguousarray(
                    shortlist_ids[:shortlist_count], dtype=np.uint32)
                execution_mode = \
                    "eligible_exhaustive_fp32_screen_fp32_refine"
            else:
                refine_width = len(eligible_ids)
                use_full_scan = len(eligible_ids) > full_scan_threshold
                ranking_ids = eligible_ids
                execution_mode = (
                    "resident_full_sgemv_then_posting_exact_topk"
                    if use_full_scan else
                    "eligible_gather_sgemv_exact_topk")
        kernel_ms = (perf_counter() - kernel_started) * 1000.0

        refine_started = perf_counter()
        if len(ranking_ids):
            if use_full_scan:
                complete_distances = self.distance_provider.prepare(
                    queries[0])
                exact_distances = np.ascontiguousarray(
                    complete_distances[ranking_ids], dtype=np.float32)
            else:
                ranking_vectors = np.ascontiguousarray(
                    self.distance_provider.base[ranking_ids],
                    dtype=np.float32)
                exact_distances = np.empty(
                    len(ranking_ids), dtype=np.float32)
                np.matmul(
                    ranking_vectors, queries[0], out=exact_distances)
                exact_distances *= np.float32(-2.0)
                exact_distances += self.distance_provider.base_sqnorm[
                    ranking_ids]
                exact_distances += query_sqnorm
                np.maximum(
                    exact_distances, np.float32(0.0),
                    out=exact_distances)
            exact_keys = (
                exact_distances.view(np.uint32).astype(np.uint64) <<
                np.uint64(32)) | ranking_ids.astype(np.uint64)
            if selected_count < len(exact_keys):
                prefix = np.argpartition(
                    exact_keys, selected_count - 1)[:selected_count]
                order = prefix[np.argsort(exact_keys[prefix])]
            else:
                order = np.argsort(exact_keys)
            selected_ids = np.ascontiguousarray(
                ranking_ids[order][None, :], dtype=np.int64)
            selected_distances = np.ascontiguousarray(
                exact_distances[order][None, :], dtype=np.float32)
        else:
            selected_ids = np.empty((1, 0), dtype=np.int64)
            selected_distances = np.empty((1, 0), dtype=np.float32)
        refine_ms = (perf_counter() - refine_started) * 1000.0

        selected_states = np.full(
            selected_ids.shape, int(HARD_PASS), dtype=np.int8)
        page = CandidatePage(
            query_indices=np.asarray([0], dtype=np.int64),
            ids=selected_ids,
            distances=selected_distances,
            states=selected_states,
            counts=np.asarray([selected_count], dtype=np.int64),
            exhausted=np.asarray([True], dtype=np.bool_),
        )
        compact_bytes = int(
            selected_ids.nbytes + selected_distances.nbytes +
            selected_states.nbytes)
        cache_requests = self._batch_one_cache_hits + \
            self._batch_one_cache_misses
        telemetry = {
            "distance_compute_ms_total": float(kernel_ms),
            "predicate_column_upload_ms_total": 0.0,
            "hard_predicate_ms_total": float(cache_ready_ms),
            "candidate_select_ms_total": float(refine_ms),
            "compact_d2h_ms_total": 0.0,
            "compact_d2h_bytes_total": 0,
            "full_distance_fallback_bytes_total": 0,
            "page_rounds": 1,
            "phase": (
                "batch1_cpu_eligibility_first_approximate_shortlist_"
                "fp32_refine" if self.batch_one_approximate_shortlist else
                "batch1_cpu_eligibility_first_exact_topk"),
            "statistical_group_prefetch": True,
            "hard_state_rank_kernel": execution_mode,
            "batch1_filtered_distance_path": True,
            "batch1_distance_rows_evaluated": int(
                self.base_rows if use_full_scan else
                distance_rows_evaluated),
            "batch1_eligible_rows": int(len(eligible_ids)),
            "batch1_numba_partitions": int(partitions),
            "batch1_refine_width": int(refine_width),
            "batch1_projection_dim": int(self.batch_one_projection_dim),
            "batch1_quantized_screen": bool(
                self.batch_one_approximate_shortlist and
                self.batch_one_quantized),
            "batch1_quantized_dtype": (
                "int8" if self.batch_one_approximate_shortlist and
                self.batch_one_quantized else "N/A"),
            "batch1_execution_mode": execution_mode,
            "batch1_exhaustive_full_dimension_screen": bool(
                exhaustive_screen),
            "batch1_full_scan_switch_rows": int(full_scan_threshold),
            "batch1_kernel_ms": float(kernel_ms),
            "batch1_refine_ms": float(refine_ms),
            "batch1_eligibility_cache_ms": float(cache_ready_ms),
            "batch1_eligibility_cache_hit": bool(eligibility_cache_hit),
            "batch1_eligibility_cache_hits": int(
                self._batch_one_cache_hits),
            "batch1_eligibility_cache_misses": int(
                self._batch_one_cache_misses),
            "batch1_eligibility_cache_hit_rate": float(
                self._batch_one_cache_hits / cache_requests
                if cache_requests else 0.0),
            "batch1_eligibility_cache_entries": int(
                len(self._batch_one_eligibility_cache)),
            "batch1_eligibility_cache_bytes": int(
                self._batch_one_eligibility_cache_bytes),
            "batch1_eligibility_cache_max_entries": int(
                self.batch_one_cache_max_entries),
            "batch1_eligibility_cache_max_bytes": int(
                self.batch_one_cache_max_bytes),
            "batch1_fused_wall_ms": float(
                (perf_counter() - wall_started) * 1000.0),
            "batch1_fused_device_ms_total": 0.0,
            "batch1_fused_device_and_d2h_wait_ms_total": 0.0,
            "peak_gpu_allocated_bytes": 0,
            "resident_working_host_bytes": int(
                eligible_ids.nbytes + compact_bytes),
            "peak_provider_host_bytes": int(
                self.resident_bytes + compact_bytes),
            "candidate_stream_bytes_per_event": 13,
        }
        return CandidateStreamHandle(
            scores=None, query_sqnorm=None, states=None, ranked_keys=None,
            available_counts=np.asarray(
                [len(eligible_ids)], dtype=np.int64),
            emitted_counts=np.zeros(1, dtype=np.int64), query_count=1,
            telemetry=telemetry, prefetched_page=page)

    @staticmethod
    def _statistical_group_data(plans, base_rows):
        statistical_group_ids = np.zeros(base_rows, dtype=np.int32)
        statistical_decisions = np.zeros(
            (len(plans), 1), dtype=np.int8)
        prefetched_plans = [
            plan for plan in plans
            if "statistical_group_ids" in plan and
            "statistical_group_decisions" in plan]
        if not prefetched_plans:
            return statistical_group_ids, statistical_decisions, False
        first_ids = prefetched_plans[0]["statistical_group_ids"]
        group_count = len(
            prefetched_plans[0]["statistical_group_decisions"])
        aligned = (
            len(first_ids) == base_rows and group_count > 0 and
            all(
                (plan["statistical_group_ids"] is first_ids and
                 len(plan["statistical_group_decisions"]) == group_count)
                if "statistical_group_ids" in plan else
                not bool(plan.get("has_unresolved_attributes", True))
                for plan in plans))
        if not aligned:
            return statistical_group_ids, statistical_decisions, False
        statistical_group_ids = np.ascontiguousarray(
            first_ids, dtype=np.int32)
        statistical_decisions = np.ascontiguousarray([
            plan["statistical_group_decisions"]
            if "statistical_group_decisions" in plan else
            np.zeros(group_count, dtype=np.int8)
            for plan in plans], dtype=np.int8)
        return statistical_group_ids, statistical_decisions, True

    @staticmethod
    def _hard_states(plans, base_rows):
        first = plans[0]
        values = np.ascontiguousarray(first["values"], dtype=np.float64)
        missing = np.ascontiguousarray(first["missing"], dtype=np.bool_)
        if (values.ndim != 2 or missing.shape != values.shape or
                values.shape[1] != base_rows):
            raise ValueError(
                "compiled predicate columns do not align with resident base")
        max_atoms = max(len(plan["attrs"]) for plan in plans)
        attrs = np.zeros((len(plans), max_atoms), dtype=np.int32)
        ops = np.zeros((len(plans), max_atoms), dtype=np.int8)
        lower = np.zeros((len(plans), max_atoms), dtype=np.float64)
        upper = np.zeros((len(plans), max_atoms), dtype=np.float64)
        counts = np.empty(len(plans), dtype=np.int32)
        for query_index, plan in enumerate(plans):
            if plan["values"] is not first["values"] or \
                    plan["missing"] is not first["missing"]:
                raise ValueError(
                    "one candidate batch must share immutable predicate data")
            count = len(plan["attrs"])
            counts[query_index] = count
            attrs[query_index, :count] = np.asarray(
                plan["attrs"], dtype=np.int32)
            ops[query_index, :count] = np.asarray(
                plan["ops"], dtype=np.int8)
            lower[query_index, :count] = np.asarray(
                plan["lower"], dtype=np.float64)
            upper[query_index, :count] = np.asarray(
                plan["upper"], dtype=np.float64)
        (statistical_group_ids, statistical_decisions,
         use_statistical_groups) = \
            NumpyExactCandidateStreamProvider._statistical_group_data(
                plans, base_rows)
        states = _hard_states_batch_numba(
            values, missing, attrs, ops, lower, upper, counts,
            statistical_group_ids, statistical_decisions,
            use_statistical_groups)
        return states, use_statistical_groups

    def begin_batch(self, queries, plans):
        plans = list(plans)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width does not match resident base")
        if len(plans) != len(queries):
            raise ValueError("queries and compiled predicates must align")
        if not len(queries):
            raise ValueError("candidate stream requires a non-empty batch")

        started = perf_counter()
        scores = self.distance_provider.prepare_batch(queries)
        distance_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        (group_ids, decisions, statistical_group_prefetch) = \
            self._statistical_group_data(plans, self.base_rows)
        states = None
        compact_ids = None
        compact_distances = None
        if statistical_group_prefetch:
            first = plans[0]
            max_atoms = max(len(plan["attrs"]) for plan in plans)
            attrs = np.zeros((len(plans), max_atoms), dtype=np.int32)
            ops = np.zeros((len(plans), max_atoms), dtype=np.int8)
            lower = np.zeros((len(plans), max_atoms), dtype=np.float64)
            upper = np.zeros((len(plans), max_atoms), dtype=np.float64)
            atom_counts = np.empty(len(plans), dtype=np.int32)
            for query_index, plan in enumerate(plans):
                count = len(plan["attrs"])
                atom_counts[query_index] = count
                attrs[query_index, :count] = plan["attrs"]
                ops[query_index, :count] = plan["ops"]
                lower[query_index, :count] = plan["lower"]
                upper[query_index, :count] = plan["upper"]
            compact_ids, compact_distances, available_counts = \
                _fused_hard_topk_batch_numba(
                    scores, np.ascontiguousarray(
                        first["values"], dtype=np.float64),
                    np.ascontiguousarray(
                        first["missing"], dtype=np.bool_),
                    attrs, ops, lower, upper, atom_counts,
                    group_ids, decisions, self.compact_topk_width)
            states = None
        else:
            states, _ = self._hard_states(plans, self.base_rows)
            available_counts = np.ascontiguousarray(
                np.count_nonzero(states != int(HARD_FAIL), axis=1),
                dtype=np.int64)
        hard_ms = (perf_counter() - started) * 1000.0
        resident_working_host_bytes = int(
            scores.nbytes +
            (0 if states is None else states.nbytes) +
            (0 if compact_ids is None else compact_ids.nbytes) +
            (0 if compact_distances is None else compact_distances.nbytes) +
            available_counts.nbytes)
        telemetry = {
            "distance_compute_ms_total": float(distance_ms),
            "predicate_column_upload_ms_total": 0.0,
            "hard_predicate_ms_total": float(hard_ms),
            "candidate_select_ms_total": 0.0,
            "compact_d2h_ms_total": 0.0,
            "compact_d2h_bytes_total": 0,
            "full_distance_fallback_bytes_total": 0,
            "page_rounds": 0,
            "phase": "full_qn_cpu_compact_host_pages",
            "peak_gpu_allocated_bytes": 0,
            "resident_working_host_bytes": resident_working_host_bytes,
            "peak_provider_host_bytes": resident_working_host_bytes,
            "candidate_stream_bytes_per_event": 0,
            "statistical_group_prefetch": bool(
                statistical_group_prefetch),
        }
        handle = CandidateStreamHandle(
            scores=scores, query_sqnorm=None,
            states=states, ranked_keys=None,
            available_counts=available_counts,
            emitted_counts=np.zeros(len(queries), dtype=np.int64),
            query_count=len(queries), telemetry=telemetry)
        if compact_ids is not None:
            handle.compact_ids = compact_ids
            handle.compact_distances = compact_distances
        return handle


    def next_page(self, handle, active_mask, *, page_size):
        if handle.closed:
            raise RuntimeError("candidate stream handle is closed")
        page_size = int(page_size)
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        active_mask = np.asarray(active_mask, dtype=np.bool_)
        if active_mask.shape != (handle.query_count,):
            raise ValueError("active mask does not align with candidate batch")
        active = np.flatnonzero(
            active_mask &
            (handle.emitted_counts < handle.available_counts))
        if not len(active):
            empty = np.empty((0, 0), dtype=np.int64)
            return CandidatePage(
                query_indices=np.empty(0, dtype=np.int64), ids=empty,
                distances=np.empty((0, 0), dtype=np.float32),
                states=np.empty((0, 0), dtype=np.int8),
                counts=np.empty(0, dtype=np.int64),
                exhausted=np.empty(0, dtype=np.bool_))

        starts = handle.emitted_counts[active].copy()
        targets = np.minimum(
            starts + page_size, handle.available_counts[active])
        counts = np.ascontiguousarray(targets - starts, dtype=np.int64)
        width = int(counts.max(initial=0))
        if hasattr(handle, "compact_ids"):
            if int(targets.max(initial=0)) > handle.compact_ids.shape[1]:
                raise RuntimeError(
                    "requested CPU page exceeds fused compact top-k")
            output_ids = np.full(
                (len(active), width), -1, dtype=np.int64)
            output_distances = np.full(
                (len(active), width), np.inf, dtype=np.float32)
            output_states = np.zeros(
                (len(active), width), dtype=np.int8)
            for output_index, query_index in enumerate(active):
                start = int(starts[output_index])
                target = int(targets[output_index])
                count = target - start
                output_ids[output_index, :count] = \
                    handle.compact_ids[query_index, start:target]
                output_distances[output_index, :count] = \
                    handle.compact_distances[query_index, start:target]
                output_states[output_index, :count] = int(HARD_PASS)
            handle.emitted_counts[active] = targets
            handle.telemetry["page_rounds"] += 1
            return CandidatePage(
                query_indices=np.ascontiguousarray(active, dtype=np.int64),
                ids=output_ids, distances=output_distances,
                states=output_states, counts=counts,
                exhausted=np.ascontiguousarray(
                    targets >= handle.available_counts[active],
                    dtype=np.bool_))
        started = perf_counter()
        output_ids, output_distances, output_states = \
            _candidate_pages_batch_numba(
                handle.scores, handle.states, self.global_ids,
                np.ascontiguousarray(active, dtype=np.int64), starts,
                targets, width)
        select_ms = (perf_counter() - started) * 1000.0

        handle.emitted_counts[active] = targets
        exhausted = np.ascontiguousarray(
            targets >= handle.available_counts[active], dtype=np.bool_)
        handle.telemetry["candidate_select_ms_total"] += float(select_ms)
        handle.telemetry["page_rounds"] += 1
        compact_bytes = int(
            output_ids.nbytes + output_distances.nbytes +
            output_states.nbytes)
        handle.telemetry["peak_provider_host_bytes"] = max(
            int(handle.telemetry["peak_provider_host_bytes"]),
            int(handle.telemetry["resident_working_host_bytes"] +
                len(active) * self.base_rows * np.dtype(np.uint64).itemsize +
                compact_bytes))
        return CandidatePage(
            query_indices=np.ascontiguousarray(active, dtype=np.int64),
            ids=output_ids, distances=output_distances,
            states=output_states, counts=counts, exhausted=exhausted)

    def full_distance_rows(self, handle, query_indices):
        if handle.closed:
            raise RuntimeError("candidate stream handle is closed")
        query_indices = np.ascontiguousarray(query_indices, dtype=np.int64)
        return np.ascontiguousarray(handle.scores[query_indices])

    def close(self, handle):
        handle.closed = True
        handle.scores = None
        handle.states = None
        handle.ranked_keys = None
        if hasattr(handle, "compact_ids"):
            handle.compact_ids = None
            handle.compact_distances = None

    @property
    def resident_bytes(self):
        return int(
            self.distance_provider.resident_bytes + self.global_ids.nbytes +
            self._batch_one_eligibility_cache_bytes +
            (self._batch_one_projection_base.nbytes +
             self._batch_one_projection_sqnorm.nbytes
             if self.batch_one_projection_dim < self.dim else 0) +
            (self._batch_one_quantized_base.nbytes +
             self._batch_one_quantized_scales.nbytes
             if self.batch_one_quantized else 0))

    def manifest(self):
        return {
            "kind": (
                "numpy_approximate_shortlist_fp32_refine_candidate_stream"
                if self.batch_one_approximate_shortlist else
                "numpy_exact_candidate_stream"),
            "phase": (
                "approximate_shortlist_fp32_refine"
                if self.batch_one_approximate_shortlist else
                "full_qn_cpu_compact_host_pages"),
            "device": "cpu",
            "base_rows": self.base_rows,
            "dimension": self.dim,
            "resident_bytes": self.resident_bytes,
            "rank_order": (
                "int8_projected_shortlist_then_exact_fp32_refine"
                if self.batch_one_approximate_shortlist else
                "exact_(float32_squared_l2,global_id)"),
            "hard_predicate_dtype": "float64",
            "normal_path_full_distance_d2h": False,
            "normal_path_dense_leaf_scheduler_state": False,
            "batch_one_path": "eligibility_first_numba_local_topk",
            "batch_one_refine_width": self.batch_one_refine_width,
            "batch_one_projection_dim": self.batch_one_projection_dim,
            "batch_one_projection_bytes": int(
                self._batch_one_projection_base.nbytes +
                self._batch_one_projection_sqnorm.nbytes
                if self.batch_one_projection_dim < self.dim else 0),
            "batch_one_quantized_screen": self.batch_one_quantized,
            "batch_one_approximate_shortlist": bool(
                self.batch_one_approximate_shortlist),
            "batch_one_fused_exact_min_work": int(
                self.batch_one_fused_exact_min_work),
            "batch_one_quantized_bytes": int(
                self._batch_one_quantized_base.nbytes +
                self._batch_one_quantized_scales.nbytes
                if self.batch_one_quantized else 0),
            "batch_one_cache_entries": int(
                len(self._batch_one_eligibility_cache)),
            "batch_one_cache_bytes": int(
                self._batch_one_eligibility_cache_bytes),
            "batch_one_cache_max_bytes": self.batch_one_cache_max_bytes,
        }
