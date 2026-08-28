"""GPU-resident exact candidate pages for the paper executor.

Phase A deliberately keeps the complete ``Q x N`` exact-distance matrix on
CUDA.  Unlike :mod:`mifa.search.gpu_distance`, it never materializes that
matrix on the host during the normal path.  Deterministic hard failures are
discarded on device and only compact pages ordered by ``(distance, row_id)``
cross into the CPU statistical-inference path.

This is an exact distance-order specialization, not an ANN or tree-pruning
kernel.  Chunked distance production is a later phase after the current
data-movement and host-state measurements are recorded.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit


HARD_FAIL = np.int8(0)
HARD_PASS = np.int8(1)
UNRESOLVED = np.int8(2)


@dataclass(frozen=True)
class CandidatePage:
    """One compact host page for the active query rows."""

    query_indices: np.ndarray
    ids: np.ndarray
    distances: np.ndarray
    states: np.ndarray
    counts: np.ndarray
    exhausted: np.ndarray


@dataclass
class CandidateStreamHandle:
    """Mutable state owned by one resident candidate-stream batch."""

    scores: object
    query_sqnorm: object
    states: object
    ranked_keys: object
    available_counts: np.ndarray
    emitted_counts: np.ndarray
    query_count: int
    telemetry: dict
    query_vectors: object = None
    prefetched_page: CandidatePage | None = None
    closed: bool = False


@dataclass(frozen=True)
class BatchOneEligibilityKeys:
    """Cached eligibility used by the low-launch batch-one exact path."""

    eligibility: object
    eligible_ids: object
    plan_references: tuple = ()


@dataclass
class ShardedCandidateStreamHandle:
    """Host merge state for exact candidate streams from corpus shards."""

    shard_handles: list
    pending: list
    available_counts: np.ndarray
    emitted_counts: np.ndarray
    query_count: int
    telemetry: dict
    closed: bool = False


@dataclass
class QueryShardedCandidateStreamHandle:
    """Batch state for replicated-corpus, disjoint-query GPU shards."""

    shard_handles: list
    query_indices: list
    available_counts: np.ndarray
    emitted_counts: np.ndarray
    query_count: int
    telemetry: dict
    closed: bool = False


@dataclass
class ReplicatedCandidateStreamHandle:
    """One batch-one request delegated to a resident GPU replica."""

    provider: object
    inner: object

    @property
    def telemetry(self):
        return self.inner.telemetry

    @property
    def available_counts(self):
        return self.inner.available_counts

    @property
    def emitted_counts(self):
        return self.inner.emitted_counts

    @property
    def query_count(self):
        return self.inner.query_count

    @property
    def prefetched_page(self):
        return self.inner.prefetched_page

    @prefetched_page.setter
    def prefetched_page(self, value):
        self.inner.prefetched_page = value

    @property
    def closed(self):
        return self.inner.closed


@njit(cache=True, fastmath=False)
def _merge_two_sorted_candidate_arrays(
        left_ids, left_distances, left_states,
        right_ids, right_distances, right_states, limit):
    """Consume an exact prefix from two ``(distance,id)`` streams."""
    limit = min(int(limit), len(left_ids) + len(right_ids))
    output_ids = np.empty(limit, dtype=np.int64)
    output_distances = np.empty(limit, dtype=np.float32)
    output_states = np.empty(limit, dtype=np.int8)
    left = 0
    right = 0
    count = 0
    while count < limit:
        take_left = right >= len(right_ids)
        if left < len(left_ids) and right < len(right_ids):
            left_distance = left_distances[left]
            right_distance = right_distances[right]
            take_left = (
                left_distance < right_distance or
                (left_distance == right_distance and
                 left_ids[left] < right_ids[right]))
        if take_left:
            output_ids[count] = left_ids[left]
            output_distances[count] = left_distances[left]
            output_states[count] = left_states[left]
            left += 1
        else:
            output_ids[count] = right_ids[right]
            output_distances[count] = right_distances[right]
            output_states[count] = right_states[right]
            right += 1
        count += 1
    return (output_ids, output_distances, output_states,
            int(left), int(right))


class TorchExactCandidateStreamProvider:
    """Produce exact, deterministically ranked candidate pages on CUDA."""

    def __init__(self, base, base_sqnorm=None, *, device="cuda:0",
                 global_offset=0, score_precision="fp32_exact",
                 resident_exact_postings=False):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for TorchExactCandidateStreamProvider")
        base = np.ascontiguousarray(base, dtype=np.float32)
        if base.ndim != 2:
            raise ValueError("base must be a 2-D matrix")
        if not np.isfinite(base).all():
            raise ValueError("base vectors must be finite")
        if base_sqnorm is None:
            base_sqnorm = np.einsum("ij,ij->i", base, base)
        base_sqnorm = np.ascontiguousarray(base_sqnorm, dtype=np.float32)
        if base_sqnorm.shape != (len(base),):
            raise ValueError("base_sqnorm must have one value per base row")
        if not np.isfinite(base_sqnorm).all() or np.any(base_sqnorm < 0):
            raise ValueError("base squared norms must be finite and non-negative")
        global_offset = int(global_offset)
        global_last = global_offset + max(len(base) - 1, 0)
        if global_offset < 0 or global_last > np.iinfo(np.uint32).max:
            raise ValueError(
                "exact candidate keys require global row IDs in uint32")

        self.torch = torch
        self.device = torch.device(device)
        score_precision = str(score_precision)
        if score_precision not in {"fp32_exact", "fp16_refine"}:
            raise ValueError("unknown candidate score precision")
        self.score_precision = score_precision
        self.resident_exact_postings = bool(resident_exact_postings)
        self.base_rows, self.dim = map(int, base.shape)
        self.global_offset = global_offset
        base_device = torch.from_numpy(base).to(self.device)
        self.base_t = base_device.T.contiguous()
        self.base_half = (
            base_device.to(dtype=torch.float16)
            if self.score_precision == "fp16_refine" else None)
        self.base_t_half = (
            self.base_half.T
            if self.score_precision == "fp16_refine" else None)
        self.base_sqnorm = torch.from_numpy(base_sqnorm).to(self.device)
        self.base_sqnorm_half = (
            self.base_sqnorm.to(dtype=torch.float16)
            if self.score_precision == "fp16_refine" else None)
        self.global_ids = torch.arange(
            global_offset, global_offset + self.base_rows,
            dtype=torch.int64, device=self.device)
        self._column_source = None
        self._value_matrix = None
        self._missing_matrix = None
        self._statistical_group_source = None
        self._statistical_group_ids = None
        self._compiled_batch_plan_source = None
        self._compiled_batch_plan_device = None
        self._eligibility_plan_source = None
        self._eligibility_matrix = None
        self._batch_plan_cache = OrderedDict()
        self._eligibility_matrix_cache = OrderedDict()
        self._resident_exact_posting_cache = OrderedDict()
        self._batch_one_cache_max_entries = 32
        self._batch1_timing_events = tuple(
            torch.cuda.Event(enable_timing=True) for _ in range(2))
        self._value_columns = {}
        self._missing_columns = {}

        # Keep library/context initialization outside measured online batches.
        warm_query = torch.zeros(
            (1, self.dim), dtype=torch.float32, device=self.device)
        warm_scores = torch.addmm(
            self.base_sqnorm[None, :], warm_query, self.base_t,
            beta=1.0, alpha=-2.0)
        warm_scores.clamp_min_(0.0)
        torch.cuda.synchronize(self.device)
        del warm_scores

    def _resident_exact_posting(self, plan):
        """Return one precomputed exact posting as resident local row IDs."""
        if not self.resident_exact_postings:
            return None
        posting_source = plan.get("precomputed_eligible_ids")
        if posting_source is None:
            return None
        posting = np.asarray(posting_source, dtype=np.uint32)
        if posting.ndim != 1:
            raise ValueError(
                "precomputed exact eligibility posting must be one-dimensional")
        shard_start = self.global_offset
        shard_end = shard_start + self.base_rows
        source = (id(posting_source), len(posting), shard_start, shard_end)
        cached = self._resident_exact_posting_cache.get(source)
        if cached is not None:
            return cached
        local = posting[
            (posting >= shard_start) & (posting < shard_end)
        ].astype(np.int64, copy=False)
        if shard_start:
            local = np.ascontiguousarray(local - shard_start, dtype=np.int64)
        else:
            local = np.ascontiguousarray(local, dtype=np.int64)
        cached = BatchOneEligibilityKeys(
            eligibility=None,
            eligible_ids=self.torch.from_numpy(local).to(self.device),
            plan_references=(posting_source,),
        )
        self._resident_exact_posting_cache[source] = cached
        return cached

    def prewarm_eligibility(self, plans):
        """Keep exact cohort postings on device outside online timing."""
        if not self.resident_exact_postings:
            return None
        for plan in plans:
            self._resident_exact_posting(plan)
        self._synchronize()
        resident_bytes = int(sum(
            item.eligible_ids.numel() * item.eligible_ids.element_size()
            for item in self._resident_exact_posting_cache.values()))
        return {
            "kind": "resident_exact_eligibility_postings",
            "entries": int(len(self._resident_exact_posting_cache)),
            "cohort_plans": int(len(plans)),
            "resident_bytes": resident_bytes,
            "online_predicate_materialization": False,
        }

    def _synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def _compute_scores(self, queries):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width does not match resident base")
        query_device = self.torch.from_numpy(queries).to(self.device)
        bias = self.base_sqnorm[None, :].expand(len(queries), -1)
        if self.score_precision == "fp16_refine":
            scores = self.torch.addmm(
                self.base_sqnorm_half[None, :],
                query_device.to(dtype=self.torch.float16), self.base_t_half,
                beta=1.0, alpha=-2.0)
        else:
            scores = self.torch.addmm(
                bias, query_device, self.base_t, beta=1.0, alpha=-2.0)
        # The query norm is constant within a row, so retain the ranking
        # surrogate on device and add the norm only to the compact selected
        # values.  This removes a complete QxN add/clamp pass without changing
        # mathematical L2 order.  Boundary ties after float32 norm addition
        # are detected in ``next_page`` and replayed through the exact packed
        # key path for that query only.
        query_sqnorm = np.einsum("ij,ij->i", queries, queries)
        query_sqnorm_device = self.torch.from_numpy(np.ascontiguousarray(
            query_sqnorm)).to(self.device)
        return scores, query_sqnorm_device, query_device

    def _compute_scores_for_rows(self, queries, row_ids):
        """Compute the exact ranking surrogate for one eligible row set."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        query_device = self.torch.from_numpy(queries).to(self.device)
        if self.score_precision == "fp16_refine":
            from .triton_candidate import eligible_row_scores

            scores = eligible_row_scores(
                self.base_half, query_device, row_ids,
                self.base_sqnorm_half)
        else:
            selected_base_t = self.base_t.index_select(1, row_ids)
            selected_sqnorm = self.base_sqnorm.index_select(0, row_ids)
            scores = self.torch.addmm(
                selected_sqnorm[None, :], query_device, selected_base_t,
                beta=1.0, alpha=-2.0)
        query_sqnorm = np.einsum("ij,ij->i", queries, queries)
        query_sqnorm_device = self.torch.from_numpy(np.ascontiguousarray(
            query_sqnorm)).to(self.device)
        return scores, query_sqnorm_device, query_device

    def _ensure_predicate_columns(self, plans):
        first = plans[0]
        values = first["values"]
        missing = first["missing"]
        source = (id(values), id(missing), tuple(values.shape),
                  tuple(missing.shape))
        if self._column_source != source:
            self._value_columns.clear()
            self._missing_columns.clear()
            self._value_matrix = None
            self._missing_matrix = None
            self._compiled_batch_plan_source = None
            self._compiled_batch_plan_device = None
            self._eligibility_plan_source = None
            self._eligibility_matrix = None
            self._batch_plan_cache.clear()
            self._eligibility_matrix_cache.clear()
            self._column_source = source
        shard_end = self.global_offset + self.base_rows
        if (values.ndim != 2 or missing.shape != values.shape or
                values.shape[1] < shard_end):
            raise ValueError(
                "compiled predicate columns do not align with resident base")
        for plan in plans:
            if plan["values"] is not values or plan["missing"] is not missing:
                raise ValueError(
                    "one candidate batch must share immutable predicate data")
        if self._value_matrix is None:
            self._value_matrix = self.torch.from_numpy(
                np.ascontiguousarray(
                    values[:, self.global_offset:shard_end],
                    dtype=np.float64)).to(self.device)
            self._missing_matrix = self.torch.from_numpy(
                np.ascontiguousarray(
                    missing[:, self.global_offset:shard_end],
                    dtype=np.bool_)).to(self.device)
        for plan in plans:
            for attr in np.asarray(plan["attrs"], dtype=np.int32):
                attr = int(attr)
                if attr not in self._value_columns:
                    self._value_columns[attr] = self._value_matrix[attr]
                    self._missing_columns[attr] = self._missing_matrix[attr]

    def _fused_hard_states_and_keys(
            self, plans, scores, *, defer_batch_one_count=False):
        from .triton_candidate import (
            eligibility_block_topk,
            fused_hard_states_and_selection_scores,
            hard_eligibility_mask,
        )

        def to_device(values):
            return self.torch.from_numpy(
                np.ascontiguousarray(values)).to(self.device)

        plan_references = tuple(
            (plan["attrs"], plan["ops"], plan["lower"], plan["upper"],
             plan.get("statistical_group_ids"),
             plan.get("statistical_group_decisions"))
            for plan in plans)
        source = tuple(
            tuple(id(value) for value in references)
            for references in plan_references)
        cached_device_entry = self._batch_plan_cache.pop(source, None)
        if cached_device_entry is not None:
            self._batch_plan_cache[source] = cached_device_entry
            cached_device_plan, _host_references = cached_device_entry
            (device_attrs, device_ops, device_lower, device_upper,
             device_counts, statistical_group_ids,
             statistical_decisions, statistical_group_count,
             use_statistical_groups) = cached_device_plan
        else:
            query_count = len(plans)
            max_atoms = max(
                1, max(len(plan["attrs"]) for plan in plans))
            attrs = np.zeros((query_count, max_atoms), dtype=np.int32)
            ops = np.full((query_count, max_atoms), -1, dtype=np.int8)
            lower = np.zeros((query_count, max_atoms), dtype=np.float64)
            upper = np.zeros((query_count, max_atoms), dtype=np.float64)
            counts = np.empty(query_count, dtype=np.int32)
            for query_index, plan in enumerate(plans):
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
            statistical = self._statistical_group_data(plans)
            if statistical is None:
                statistical_group_ids = self.torch.empty(
                    1, dtype=self.torch.int32, device=self.device)
                statistical_decisions = self.torch.empty(
                    (1, 1), dtype=self.torch.int8, device=self.device)
                statistical_group_count = 1
                use_statistical_groups = False
            else:
                (statistical_group_ids, statistical_decisions,
                 statistical_group_count) = statistical
                use_statistical_groups = True
            device_attrs = to_device(attrs)
            device_ops = to_device(ops)
            device_lower = to_device(lower)
            device_upper = to_device(upper)
            device_counts = to_device(counts)
            self._compiled_batch_plan_source = source
            self._compiled_batch_plan_device = (
                device_attrs, device_ops, device_lower, device_upper,
                device_counts, statistical_group_ids,
                statistical_decisions, statistical_group_count,
                use_statistical_groups)
            self._batch_plan_cache[source] = (
                self._compiled_batch_plan_device, plan_references)
            while len(self._batch_plan_cache) > \
                    self._batch_one_cache_max_entries:
                self._batch_plan_cache.popitem(last=False)
        if use_statistical_groups:
            query_count = len(plans)
            cached_eligibility = None
            if query_count == 1:
                cached_eligibility = self._eligibility_matrix_cache.pop(
                    source, None)
            if cached_eligibility is not None:
                self._eligibility_matrix = cached_eligibility.eligibility
                self._eligibility_matrix_cache[source] = cached_eligibility
            elif self._eligibility_plan_source != source:
                self._eligibility_matrix = hard_eligibility_mask(
                    self._value_matrix, self._missing_matrix,
                    device_attrs, device_ops, device_lower, device_upper,
                    device_counts, statistical_group_ids,
                    statistical_decisions, statistical_group_count)
                if query_count == 1:
                    cached_eligibility = BatchOneEligibilityKeys(
                        eligibility=self._eligibility_matrix,
                        eligible_ids=self.torch.nonzero(
                            self._eligibility_matrix[0], as_tuple=False
                        ).flatten(),
                        plan_references=plan_references,
                    )
                    self._eligibility_matrix_cache[source] = \
                        cached_eligibility
                    while len(self._eligibility_matrix_cache) > \
                            self._batch_one_cache_max_entries:
                        self._eligibility_matrix_cache.popitem(last=False)
            self._eligibility_plan_source = source
            # A full-row torch top-k is substantially faster than producing
            # and merging hundreds of per-block shortlists for one request.
            # Keep the exact eligibility bitmap on device and let
            # ``next_page`` use the optimized batch-one reduction.  Larger
            # batches retain the block-top-k path, which has better memory
            # scaling and throughput there.
            if int(scores.shape[0]) == 1:
                if cached_eligibility is None:
                    cached_eligibility = BatchOneEligibilityKeys(
                        eligibility=self._eligibility_matrix,
                        eligible_ids=self.torch.nonzero(
                            self._eligibility_matrix[0], as_tuple=False
                        ).flatten(),
                        plan_references=plan_references,
                    )
                available = (
                    self.torch.empty(
                        0, dtype=self.torch.int64, device=self.device)
                    if defer_batch_one_count else
                    self.torch.full(
                        (1,), int(cached_eligibility.eligible_ids.numel()),
                        dtype=self.torch.int64, device=self.device)
                )
                return (
                    None,
                    cached_eligibility,
                    available,
                    True,
                )
            block_scores, block_ids, available = eligibility_block_topk(
                scores, self._eligibility_matrix, self.global_offset,
                block_size=8192, block_topk=10)
            return (None, (block_scores, block_ids), available, True)
        return (*fused_hard_states_and_selection_scores(
            scores, self._value_matrix, self._missing_matrix,
            device_attrs, device_ops, device_lower, device_upper,
            device_counts, self.global_offset, statistical_group_ids,
            statistical_decisions, statistical_group_count, False), False)

    def _statistical_group_data(self, plans):
        prefetched_plans = [
            plan for plan in plans
            if "statistical_group_ids" in plan and
            "statistical_group_decisions" in plan]
        if not prefetched_plans:
            return None
        first_ids = prefetched_plans[0]["statistical_group_ids"]
        group_count = len(
            prefetched_plans[0]["statistical_group_decisions"])
        if group_count < 1:
            return None
        if any(
                (
                    plan["statistical_group_ids"] is not first_ids or
                    len(plan["statistical_group_decisions"]) != group_count
                ) if "statistical_group_ids" in plan else
                bool(plan.get("has_unresolved_attributes", True))
                for plan in plans):
            return None
        shard_end = self.global_offset + self.base_rows
        source = (id(first_ids), len(first_ids), self.global_offset, shard_end)
        if self._statistical_group_source != source:
            self._statistical_group_ids = self.torch.from_numpy(
                np.ascontiguousarray(
                    first_ids[self.global_offset:shard_end],
                    dtype=np.int32)).to(self.device)
            self._statistical_group_source = source
        decisions = self.torch.from_numpy(np.ascontiguousarray([
            plan["statistical_group_decisions"]
            if "statistical_group_decisions" in plan else
            np.zeros(group_count, dtype=np.int8)
            for plan in plans
        ], dtype=np.int8)).to(self.device)
        return self._statistical_group_ids, decisions, group_count

    def _hard_states(self, plans):
        torch = self.torch
        query_count = len(plans)
        hard_fail = torch.zeros(
            (query_count, self.base_rows), dtype=torch.bool,
            device=self.device)
        unresolved = torch.zeros_like(hard_fail)
        grouped_atoms = {}
        for query_index, plan in enumerate(plans):
            attrs = np.asarray(plan["attrs"], dtype=np.int32)
            ops = np.asarray(plan["ops"], dtype=np.int8)
            lower = np.asarray(plan["lower"], dtype=np.float64)
            upper = np.asarray(plan["upper"], dtype=np.float64)
            for atom_index, attr_value in enumerate(attrs):
                attr = int(attr_value)
                op = int(ops[atom_index])
                grouped_atoms.setdefault((attr, op), []).append((
                    int(query_index), float(lower[atom_index]),
                    float(upper[atom_index])))

        # One predicate normally has at most one atom for an (attribute, op)
        # pair.  Split the uncommon duplicate case into conflict-free rounds
        # so advanced-index writes remain exact instead of dropping an atom.
        for (attr, op), pending_atoms in grouped_atoms.items():
            pending_atoms = list(pending_atoms)
            while pending_atoms:
                round_atoms = []
                remaining = []
                seen_queries = set()
                for atom in pending_atoms:
                    if atom[0] in seen_queries:
                        remaining.append(atom)
                    else:
                        round_atoms.append(atom)
                        seen_queries.add(atom[0])
                pending_atoms = remaining
                query_indices = torch.as_tensor(
                    [atom[0] for atom in round_atoms], dtype=torch.int64,
                    device=self.device)
                lower_values = torch.as_tensor(
                    [atom[1] for atom in round_atoms], dtype=torch.float64,
                    device=self.device)[:, None]
                upper_values = torch.as_tensor(
                    [atom[2] for atom in round_atoms], dtype=torch.float64,
                    device=self.device)[:, None]
                values = self._value_columns[attr][None, :]
                missing = self._missing_columns[attr][None, :]
                if op == 0:
                    satisfied = values <= lower_values
                elif op == 1:
                    satisfied = values >= lower_values
                elif op == 2:
                    satisfied = ((values >= lower_values) &
                                 (values <= upper_values))
                elif op == 3:
                    satisfied = values == lower_values
                else:
                    raise ValueError("unknown compiled hard-predicate opcode")
                hard_fail[query_indices] |= (~missing) & (~satisfied)
                unresolved[query_indices] |= missing
        states = torch.full(
            (query_count, self.base_rows), int(HARD_PASS),
            dtype=torch.int8, device=self.device)
        states[unresolved & (~hard_fail)] = int(UNRESOLVED)
        states[hard_fail] = int(HARD_FAIL)
        return states

    def begin_batch(self, queries, plans):
        """Compute full exact distances and hard states, retained on device."""
        plans = list(plans)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if len(plans) != len(queries):
            raise ValueError("queries and compiled predicates must align")
        if not len(queries):
            raise ValueError("candidate stream requires a non-empty batch")
        if not np.isfinite(queries).all():
            raise ValueError("candidate queries must be finite")

        self.torch.cuda.reset_peak_memory_stats(self.device)
        self._synchronize()
        started = perf_counter()
        scores, query_sqnorm, query_vectors = self._compute_scores(queries)
        self._synchronize()
        distance_ms = (perf_counter() - started) * 1000.0

        started = perf_counter()
        self._ensure_predicate_columns(plans)
        self._synchronize()
        column_upload_ms = (perf_counter() - started) * 1000.0

        started = perf_counter()
        (states, selection_scores, available_device,
         statistical_group_prefetch) = \
            self._fused_hard_states_and_keys(plans, scores)
        available_counts = np.ascontiguousarray(
            available_device.cpu().numpy(), dtype=np.int64)
        hard_ms = (perf_counter() - started) * 1000.0

        telemetry = {
            "distance_compute_ms_total": float(distance_ms),
            "predicate_column_upload_ms_total": float(column_upload_ms),
            "hard_predicate_ms_total": float(hard_ms),
            "candidate_select_ms_total": 0.0,
            "boundary_tie_repair_queries_total": 0,
            "compact_d2h_ms_total": 0.0,
            # ``available_counts`` is the only non-page D2H on the normal
            # path; include it in the complete byte-reduction measurement.
            "compact_d2h_bytes_total": int(available_counts.nbytes),
            "full_distance_fallback_bytes_total": 0,
            "page_rounds": 0,
            "phase": "full_qn_gpu_compact_host_pages",
            "statistical_group_prefetch": bool(
                statistical_group_prefetch),
            "block_topk_width": int(
                10 if isinstance(selection_scores, tuple) else 0),
            "hard_state_rank_kernel": (
                "torch_batch1_full_mask_topk" if isinstance(
                    selection_scores, BatchOneEligibilityKeys) else
                "triton_fused_block_topk" if
                isinstance(selection_scores, tuple) else
                "triton_fused"),
            "score_precision": self.score_precision,
            "shortlist_refine_width": int(
                64 if self.score_precision == "fp16_refine" else 0),
        }
        telemetry["peak_gpu_allocated_bytes"] = int(
            self.torch.cuda.max_memory_allocated(self.device))
        return CandidateStreamHandle(
            scores=scores, query_sqnorm=query_sqnorm,
            states=states, ranked_keys=selection_scores,
            available_counts=available_counts,
            emitted_counts=np.zeros(len(queries), dtype=np.int64),
            query_count=len(queries), telemetry=telemetry,
            query_vectors=query_vectors)

    def begin_batch_topk(self, queries, plans, *, topk):
        """Fuse the complete batch-one exact path behind one GPU barrier.

        This specialization is valid only when immutable statistical-group
        decisions resolve every missing predicate value before ranking.  In
        that case the executor needs exactly the first ``topk`` eligible rows,
        so materializing and later merging block-local candidate pages adds
        launch/synchronization overhead without changing semantics.
        """
        plans = list(plans)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        topk = int(topk)
        if len(queries) != 1 or len(plans) != 1 or topk <= 0:
            return None
        plan = plans[0]
        if (
            "statistical_group_ids" not in plan
            or "statistical_group_decisions" not in plan
            or len(plan["statistical_group_decisions"]) < 1
        ):
            return self._begin_batch_one_unresolved_prefetch(
                queries, plans, topk=topk)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width does not match resident base")
        if not np.isfinite(queries).all():
            raise ValueError("candidate queries must be finite")

        resident_ranked_keys = self._resident_exact_posting(plan)

        torch = self.torch
        topk = min(topk, self.base_rows)
        # The preceding response's compact host copy is a completion barrier;
        # avoid a second device-wide synchronization and per-request allocator
        # reset on the latency-critical path.  Process-level peak VRAM remains
        # available through ``max_memory_allocated``.
        wall_started = perf_counter()
        events = self._batch1_timing_events
        events[0].record()
        if resident_ranked_keys is not None:
            states = None
            ranked_keys = resident_ranked_keys
            statistical_group_prefetch = True
        else:
            self._ensure_predicate_columns(plans)
            (states, ranked_keys, _available_device,
             statistical_group_prefetch) = \
                self._fused_hard_states_and_keys(
                    plans,
                    torch.empty(
                        (1, 0), dtype=torch.float32, device=self.device),
                    defer_batch_one_count=True)
        if (
            not statistical_group_prefetch
            or not isinstance(ranked_keys, BatchOneEligibilityKeys)
            or states is not None
        ):
            self._synchronize()
            return None

        eligible_ids = ranked_keys.eligible_ids
        available_count = int(eligible_ids.numel())
        selected_count = min(topk, available_count)
        # A fixed architecture threshold chooses between two exact physical
        # paths.  Selective predicates avoid the full-corpus GEMV and rank
        # only inferred-eligible rows; broad predicates retain the contiguous
        # full-corpus GEMV.  No labels, distances, or result IDs enter this
        # decision.
        filtered_distance_path = available_count * 2 <= self.base_rows
        if filtered_distance_path:
            scores, query_sqnorm, query_vectors = \
                self._compute_scores_for_rows(queries, eligible_ids)
            selection_scores = scores
        else:
            scores, query_sqnorm, query_vectors = self._compute_scores(queries)
            selection_scores = scores.index_select(1, eligible_ids)
        if self.score_precision == "fp16_refine" and selected_count:
            refine_width = min(
                max(selected_count + 1, 64), available_count)
            approximate_values, approximate_ids = torch.topk(
                selection_scores, refine_width, dim=1,
                largest=False, sorted=False)
            approximate_ids = eligible_ids[approximate_ids]
            candidate_vectors = self.base_t.T[approximate_ids]
            exact_values = (
                self.base_sqnorm[approximate_ids] -
                2.0 * torch.bmm(
                    candidate_vectors, query_vectors[:, :, None]
                ).squeeze(2) +
                query_sqnorm[:, None]).clamp_min_(0.0)
            valid_candidates = torch.isfinite(approximate_values)
            exact_keys_unmasked = (
                exact_values.view(torch.int32).to(torch.int64) << 32
            ) | (approximate_ids + self.global_offset)
            exact_keys = torch.where(
                valid_candidates, exact_keys_unmasked,
                torch.full_like(
                    exact_keys_unmasked, torch.iinfo(torch.int64).max))
            selected_keys, selected_slots = torch.topk(
                exact_keys, selected_count, dim=1,
                largest=False, sorted=True)
            selected_local_ids = approximate_ids.gather(1, selected_slots)
            selected_distances = exact_values.gather(1, selected_slots)
            selected_distances = torch.where(
                selected_keys < torch.iinfo(torch.int64).max,
                selected_distances,
                torch.full_like(selected_distances, float("inf")))
        elif selected_count:
            surrogate_values, selected_local_ids = torch.topk(
                selection_scores, selected_count, dim=1,
                largest=False, sorted=False)
            selected_local_ids = eligible_ids[selected_local_ids]
            selected_distances = (
                surrogate_values + query_sqnorm[:, None]).clamp_min_(0.0)
            safe_ids = torch.where(
                torch.isfinite(surrogate_values),
                selected_local_ids + self.global_offset,
                torch.full_like(selected_local_ids, 0xFFFFFFFF))
            exact_keys = (
                selected_distances.view(torch.int32).to(torch.int64) << 32
            ) | safe_ids
            _, selected_slots = torch.sort(exact_keys, dim=1)
            selected_local_ids = selected_local_ids.gather(
                1, selected_slots)
            selected_distances = selected_distances.gather(
                1, selected_slots)
        else:
            selected_local_ids = torch.empty(
                (1, 0), dtype=torch.int64, device=self.device)
            selected_distances = torch.empty(
                (1, 0), dtype=torch.float32, device=self.device)
        events[1].record()
        copy_started = perf_counter()
        host_ids = np.ascontiguousarray(
            selected_local_ids.cpu().numpy(), dtype=np.int64)
        host_distances = np.ascontiguousarray(
            selected_distances.cpu().numpy(), dtype=np.float32)
        copy_wait_ms = (perf_counter() - copy_started) * 1000.0
        valid = np.isfinite(host_distances)
        count = int(np.count_nonzero(valid[0]))
        host_ids = np.where(
            valid, host_ids + self.global_offset, -1).astype(
                np.int64, copy=False)
        host_distances = np.where(
            valid, host_distances, np.inf).astype(np.float32, copy=False)
        host_states = np.where(valid, HARD_PASS, HARD_FAIL).astype(
            np.int8, copy=False)
        page = CandidatePage(
            query_indices=np.asarray([0], dtype=np.int64),
            ids=host_ids,
            distances=host_distances,
            states=host_states,
            counts=np.asarray([count], dtype=np.int64),
            exhausted=np.asarray([True], dtype=np.bool_),
        )
        fused_device_ms = float(events[0].elapsed_time(events[1]))
        transferred_bytes = int(
            host_ids.nbytes + host_distances.nbytes)
        telemetry = {
            # Fine-grained CUDA event collection costs enough to distort a
            # one-millisecond request.  The regular batched path retains the
            # complete phase breakdown; this latency-specialized path records
            # one aggregate device interval and the exact host wait instead.
            "distance_compute_ms_total": 0.0,
            "predicate_column_upload_ms_total": 0.0,
            "hard_predicate_ms_total": 0.0,
            "candidate_select_ms_total": 0.0,
            "boundary_tie_repair_queries_total": 0,
            "compact_d2h_ms_total": 0.0,
            "batch1_fused_device_ms_total": fused_device_ms,
            "batch1_fused_device_and_d2h_wait_ms_total": float(copy_wait_ms),
            "timing_detail": "aggregate_batch1",
            "compact_d2h_bytes_total": transferred_bytes,
            "full_distance_fallback_bytes_total": 0,
            "page_rounds": 1,
            "phase": "batch1_fused_exact_topk_compact_host_page",
            "statistical_group_prefetch": True,
            "block_topk_width": 0,
            "hard_state_rank_kernel": (
                "torch_batch1_resident_exact_posting_topk" if
                resident_ranked_keys is not None else
                "triton_batch1_eligible_row_topk" if
                filtered_distance_path else
                "torch_batch1_full_distance_topk"),
            "batch1_resident_exact_posting": bool(
                resident_ranked_keys is not None),
            "batch1_filtered_distance_path": bool(filtered_distance_path),
            "batch1_distance_rows_evaluated": int(
                available_count if filtered_distance_path else self.base_rows),
            "batch1_eligible_rows": int(available_count),
            "score_precision": self.score_precision,
            "shortlist_refine_width": int(
                64 if self.score_precision == "fp16_refine" else 0),
            "peak_gpu_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)),
            "candidate_stream_bytes_per_event": 13,
            "batch1_fused_wall_ms": float(
                (perf_counter() - wall_started) * 1000.0),
        }
        return CandidateStreamHandle(
            scores=scores,
            query_sqnorm=query_sqnorm,
            states=None,
            ranked_keys=ranked_keys,
            available_counts=np.asarray([count], dtype=np.int64),
            emitted_counts=np.zeros(1, dtype=np.int64),
            query_count=1,
            telemetry=telemetry,
            query_vectors=query_vectors,
            prefetched_page=page,
        )

    def _begin_batch_one_unresolved_prefetch(self, queries, plans, *, topk):
        """Fuse the common one-query unresolved path into one compact page.

        The generic implementation synchronizes after distance generation,
        predicate classification, candidate selection, and compact D2H.  It
        then enters the multi-query Python page scheduler even when the first
        page contains every candidate the executor consumes.  For one query,
        retain the same exact FP32 scores and hard states, but prefetch a
        small, deterministically packed ``(distance, global_id)`` prefix in
        one CUDA stream.  If inference rejects the whole prefix, the retained
        full tensors continue through the unchanged generic page path.
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        plans = list(plans)
        topk = int(topk)
        if len(queries) != 1 or len(plans) != 1 or topk <= 0:
            return None
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width does not match resident base")
        if not np.isfinite(queries).all():
            raise ValueError("candidate queries must be finite")

        torch = self.torch
        # Small/medium corpora use a compact prefix large enough for the
        # observed on-demand inference frontier.  Larger corpora retain a
        # wider prefix to amortize any second page without transferring a
        # dense candidate row.
        architecture_width = (
            64 if self.base_rows <= 2048 else
            1024 if self.base_rows <= 8192 else
            512 if self.base_rows <= 65536 else 1024)
        prefetch_width = min(
            self.base_rows, max(topk, architecture_width))
        wall_started = perf_counter()
        events = self._batch1_timing_events
        events[0].record()
        scores, query_sqnorm, query_vectors = self._compute_scores(queries)
        self._ensure_predicate_columns(plans)
        (states, selection_scores, available_device,
         statistical_group_prefetch) = \
            self._fused_hard_states_and_keys(plans, scores)
        if (
            statistical_group_prefetch
            or states is None
            or not isinstance(selection_scores, torch.Tensor)
        ):
            self._synchronize()
            return None

        true_values = (
            selection_scores + query_sqnorm[:, None]).clamp_min_(0.0)
        valid_scores = torch.isfinite(selection_scores)
        safe_ids = torch.where(
            valid_scores,
            self.global_ids[None, :],
            torch.zeros_like(self.global_ids[None, :]))
        packed_keys = (
            true_values.view(torch.int32).to(torch.int64) << 32
        ) | safe_ids
        packed_keys = torch.where(
            valid_scores, packed_keys,
            torch.full_like(packed_keys, torch.iinfo(torch.int64).max))
        selected_keys, selected_local_ids = torch.topk(
            packed_keys, prefetch_width, dim=1,
            largest=False, sorted=True)
        selected_distances = true_values.gather(1, selected_local_ids)
        selected_states = states.gather(1, selected_local_ids)
        selected_valid = selected_keys < torch.iinfo(torch.int64).max
        events[1].record()

        copy_started = perf_counter()
        host_ids = np.ascontiguousarray(
            selected_local_ids.cpu().numpy(), dtype=np.int64)
        host_distances = np.ascontiguousarray(
            selected_distances.cpu().numpy(), dtype=np.float32)
        host_states = np.ascontiguousarray(
            selected_states.cpu().numpy(), dtype=np.int8)
        host_valid = np.ascontiguousarray(
            selected_valid.cpu().numpy(), dtype=np.bool_)
        available_count = int(available_device.cpu().numpy()[0])
        copy_wait_ms = (perf_counter() - copy_started) * 1000.0
        count = min(prefetch_width, available_count)
        host_ids = np.where(
            host_valid, host_ids + self.global_offset, -1).astype(
                np.int64, copy=False)
        host_distances = np.where(
            host_valid, host_distances, np.inf).astype(
                np.float32, copy=False)
        host_states = np.where(
            host_valid, host_states, HARD_FAIL).astype(np.int8, copy=False)
        page = CandidatePage(
            query_indices=np.asarray([0], dtype=np.int64),
            ids=host_ids,
            distances=host_distances,
            states=host_states,
            counts=np.asarray([count], dtype=np.int64),
            exhausted=np.asarray(
                [count >= available_count], dtype=np.bool_),
        )
        fused_device_ms = float(events[0].elapsed_time(events[1]))
        transferred_bytes = int(
            host_ids.nbytes + host_distances.nbytes + host_states.nbytes)
        telemetry = {
            "distance_compute_ms_total": 0.0,
            "predicate_column_upload_ms_total": 0.0,
            "hard_predicate_ms_total": 0.0,
            "candidate_select_ms_total": 0.0,
            "boundary_tie_repair_queries_total": 0,
            "compact_d2h_ms_total": 0.0,
            "batch1_fused_device_ms_total": fused_device_ms,
            "batch1_fused_device_and_d2h_wait_ms_total": float(copy_wait_ms),
            "timing_detail": "aggregate_batch1_unresolved_prefetch",
            "compact_d2h_bytes_total": transferred_bytes,
            "full_distance_fallback_bytes_total": 0,
            "page_rounds": 1,
            "phase": "batch1_fused_unresolved_exact_prefetch",
            "statistical_group_prefetch": False,
            "block_topk_width": 0,
            "hard_state_rank_kernel": "torch_batch1_packed_exact_prefetch",
            "batch1_filtered_distance_path": False,
            "batch1_distance_rows_evaluated": int(self.base_rows),
            "batch1_eligible_rows": int(available_count),
            "batch1_prefetch_width": int(prefetch_width),
            "score_precision": self.score_precision,
            "shortlist_refine_width": 0,
            "peak_gpu_allocated_bytes": int(
                torch.cuda.max_memory_allocated(self.device)),
            "candidate_stream_bytes_per_event": 13,
            "batch1_fused_wall_ms": float(
                (perf_counter() - wall_started) * 1000.0),
        }
        return CandidateStreamHandle(
            scores=scores,
            query_sqnorm=query_sqnorm,
            states=states,
            ranked_keys=selection_scores,
            available_counts=np.asarray(
                [available_count], dtype=np.int64),
            emitted_counts=np.zeros(1, dtype=np.int64),
            query_count=1,
            telemetry=telemetry,
            query_vectors=query_vectors,
            prefetched_page=page,
        )

    def next_page(self, handle, active_mask, *, page_size):
        """Return the next exact page for each active, non-exhausted query."""
        if handle.closed:
            raise RuntimeError("candidate stream handle is closed")
        page_size = int(page_size)
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        active_mask = np.asarray(active_mask, dtype=np.bool_)
        if active_mask.shape != (handle.query_count,):
            raise ValueError("active mask does not align with candidate batch")
        if handle.prefetched_page is not None:
            page = handle.prefetched_page
            handle.prefetched_page = None
            if not bool(active_mask[0]) or int(page.counts[0]) == 0:
                empty = np.empty((0, 0), dtype=np.int64)
                return CandidatePage(
                    query_indices=np.empty(0, dtype=np.int64),
                    ids=empty,
                    distances=np.empty((0, 0), dtype=np.float32),
                    states=np.empty((0, 0), dtype=np.int8),
                    counts=np.empty(0, dtype=np.int64),
                    exhausted=np.empty(0, dtype=np.bool_))
            handle.emitted_counts[0] = int(page.counts[0])
            return page
        active = np.flatnonzero(
            active_mask &
            (handle.emitted_counts < handle.available_counts))
        if not len(active):
            empty = np.empty((0, 0), dtype=np.int64)
            return CandidatePage(
                query_indices=np.empty(0, dtype=np.int64),
                ids=empty,
                distances=np.empty((0, 0), dtype=np.float32),
                states=np.empty((0, 0), dtype=np.int8),
                counts=np.empty(0, dtype=np.int64),
                exhausted=np.empty(0, dtype=np.bool_))

        starts = handle.emitted_counts[active].copy()
        targets = np.minimum(
            starts + page_size, handle.available_counts[active])
        counts = np.ascontiguousarray(targets - starts, dtype=np.int64)
        width = int(counts.max(initial=0))
        max_target = int(targets.max(initial=0))

        started = perf_counter()
        active_device = self.torch.as_tensor(
            active, dtype=self.torch.int64, device=self.device)
        all_queries_active = (
            len(active) == handle.query_count and
            np.array_equal(active, np.arange(handle.query_count)))
        starts_device = self.torch.as_tensor(
            starts, dtype=self.torch.int64, device=self.device)
        targets_device = self.torch.as_tensor(
            targets, dtype=self.torch.int64, device=self.device)
        if isinstance(handle.ranked_keys, BatchOneEligibilityKeys):
            if len(active) != 1:
                raise RuntimeError(
                    "batch-one eligibility path received multiple queries")
            active_scores = handle.scores.index_select(0, active_device)
            eligible_ids = handle.ranked_keys.eligible_ids
            active_selection_scores = active_scores.index_select(
                1, eligible_ids)
            if handle.telemetry["score_precision"] == "fp16_refine":
                refine_width = min(
                    max(max_target + 1, int(handle.telemetry[
                        "shortlist_refine_width"])),
                    int(eligible_ids.numel()))
                approximate_values, approximate_ids = self.torch.topk(
                    active_selection_scores, refine_width, dim=1,
                    largest=False, sorted=False)
                approximate_ids = eligible_ids[approximate_ids]
                candidate_vectors = self.base_t.T[approximate_ids]
                active_queries = handle.query_vectors.index_select(
                    0, active_device)
                exact_true_values = (
                    self.base_sqnorm[approximate_ids] -
                    2.0 * self.torch.sum(
                        candidate_vectors * active_queries[:, None, :],
                        dim=2) +
                    handle.query_sqnorm.index_select(
                        0, active_device)[:, None]).clamp_min_(0.0)
                valid_candidates = self.torch.isfinite(approximate_values)
                safe_ids = self.torch.where(
                    valid_candidates,
                    approximate_ids + self.global_offset,
                    self.torch.full_like(approximate_ids, 0xFFFFFFFF))
                exact_true_values = self.torch.where(
                    valid_candidates, exact_true_values,
                    self.torch.full_like(exact_true_values, float("inf")))
                compact_keys = (
                    exact_true_values.view(self.torch.int32).to(
                        self.torch.int64) << 32) | safe_ids
                _, selected_slots = self.torch.topk(
                    compact_keys, max_target, dim=1,
                    largest=False, sorted=True)
                ranked_local_ids = approximate_ids.gather(
                    1, selected_slots)
                tie_repairs = 0
            else:
                topk_width = min(
                    max_target + 1, int(eligible_ids.numel()))
                float_values, float_ids = self.torch.topk(
                    active_selection_scores, topk_width, dim=1,
                    largest=False, sorted=True)
                compact_true_values = (
                    float_values +
                    handle.query_sqnorm.index_select(
                        0, active_device)[:, None]).clamp_min_(0.0)
                compact_keys = (
                    compact_true_values.view(self.torch.int32).to(
                        self.torch.int64) << 32) | (
                            float_ids + self.global_offset)
                _, compact_order = self.torch.sort(compact_keys, dim=1)
                ranked_local_ids = float_ids.gather(
                    1, compact_order)[:, :max_target]
                tie_repairs = 0
        elif isinstance(handle.ranked_keys, tuple):
            block_scores, block_ids = handle.ranked_keys
            active_block_scores = (
                block_scores if all_queries_active else
                block_scores.index_select(0, active_device))
            active_block_ids = (
                block_ids if all_queries_active else
                block_ids.index_select(0, active_device))
            block_topk = int(handle.telemetry["block_topk_width"])
            if max_target > block_topk:
                raise RuntimeError(
                    "requested candidate page exceeds fused block top-k")
            if handle.telemetry["score_precision"] == "fp16_refine":
                refine_width = min(
                    int(handle.telemetry["shortlist_refine_width"]),
                    int(active_block_scores.shape[1]))
                _, approximate_slots = self.torch.topk(
                    active_block_scores, refine_width, dim=1,
                    largest=False, sorted=False)
                approximate_ids = active_block_ids.gather(
                    1, approximate_slots)
                valid_ids = approximate_ids >= self.global_offset
                local_ids = (approximate_ids - self.global_offset).clamp_min(0)
                candidate_vectors = self.base_t.T[local_ids]
                active_queries = handle.query_vectors.index_select(
                    0, active_device)
                exact_surrogate = (
                    self.base_sqnorm[local_ids] - 2.0 * self.torch.sum(
                        candidate_vectors * active_queries[:, None, :],
                        dim=2))
                exact_true_values = (
                    exact_surrogate + handle.query_sqnorm.index_select(
                        0, active_device)[:, None]).clamp_min_(0.0)
                safe_ids = self.torch.where(
                    valid_ids, approximate_ids,
                    self.torch.full_like(approximate_ids, 0xFFFFFFFF))
                compact_keys = (
                    exact_true_values.view(self.torch.int32).to(
                        self.torch.int64) << 32) | safe_ids
                _, selected_slots = self.torch.topk(
                    compact_keys, max_target, dim=1,
                    largest=False, sorted=True)
                ranked_local_ids = local_ids.gather(1, selected_slots)
            else:
                compact_true_values = (
                    active_block_scores +
                    handle.query_sqnorm.index_select(
                        0, active_device)[:, None]).clamp_min_(0.0)
                valid_block_ids = active_block_ids >= self.global_offset
                safe_ids = self.torch.where(
                    valid_block_ids, active_block_ids,
                    self.torch.full_like(active_block_ids, 0xFFFFFFFF))
                compact_keys = (
                    compact_true_values.view(self.torch.int32).to(
                        self.torch.int64) << 32) | safe_ids
                _, selected_slots = self.torch.topk(
                    compact_keys, max_target, dim=1,
                    largest=False, sorted=True)
                ranked_local_ids = (
                    active_block_ids.gather(1, selected_slots) -
                    self.global_offset)
            tie_repairs = 0
        else:
            active_selection_scores = (
                handle.ranked_keys if all_queries_active else
                handle.ranked_keys.index_select(0, active_device))
            if handle.telemetry["score_precision"] == "fp16_refine":
                refine_width = min(
                    max(max_target + 1, int(handle.telemetry[
                        "shortlist_refine_width"])),
                    self.base_rows)
                approximate_values, approximate_ids = self.torch.topk(
                    active_selection_scores, refine_width, dim=1,
                    largest=False, sorted=False)
                candidate_vectors = self.base_t.T[approximate_ids]
                active_queries = handle.query_vectors.index_select(
                    0, active_device)
                exact_true_values = (
                    self.base_sqnorm[approximate_ids] -
                    2.0 * self.torch.sum(
                        candidate_vectors * active_queries[:, None, :],
                        dim=2) +
                    handle.query_sqnorm.index_select(
                        0, active_device)[:, None]).clamp_min_(0.0)
                valid = self.torch.isfinite(approximate_values)
                safe_ids = self.torch.where(
                    valid, approximate_ids + self.global_offset,
                    self.torch.full_like(approximate_ids, 0xFFFFFFFF))
                exact_true_values = self.torch.where(
                    valid, exact_true_values,
                    self.torch.full_like(exact_true_values, float("inf")))
                compact_keys = (
                    exact_true_values.view(self.torch.int32).to(
                        self.torch.int64) << 32) | safe_ids
                _, selected_slots = self.torch.topk(
                    compact_keys, max_target, dim=1,
                    largest=False, sorted=True)
                ranked_local_ids = approximate_ids.gather(
                    1, selected_slots)
                tie_repairs = 0
            else:
                topk_width = min(max_target + 1, self.base_rows)
                float_values, float_ids = self.torch.topk(
                    active_selection_scores, topk_width, dim=1,
                    largest=False, sorted=True)
                compact_true_values = (
                    float_values +
                    handle.query_sqnorm.index_select(
                        0, active_device)[:, None]).clamp_min_(0.0)
                compact_keys = (
                    compact_true_values.view(self.torch.int32).to(
                        self.torch.int64) << 32) | (
                            float_ids + self.global_offset)
                _, compact_order = self.torch.sort(compact_keys, dim=1)
                ranked_local_ids = float_ids.gather(
                    1, compact_order)[:, :max_target]
                target_slots = targets_device - 1
                ranked_true_values = compact_true_values.gather(
                    1, compact_order)
                cutoff = ranked_true_values.gather(
                    1, target_slots[:, None]).squeeze(1)
                has_next = targets_device < self.base_rows
                next_slots = targets_device.clamp_max(topk_width - 1)
                next_values = ranked_true_values.gather(
                    1, next_slots[:, None]).squeeze(1)
                ambiguous = has_next & (cutoff == next_values)

                # Float32 top-k is substantially faster than int64 top-k.  Its
                # membership is exact whenever the requested boundary is not
                # tied.  Sort the compact selection by the packed exact key so
                # every interior tie follows global row ID.  Only a
                # boundary-tied row is replayed with the full packed-key path.
                tie_repairs = int(
                    self.torch.count_nonzero(ambiguous).item())
                if tie_repairs:
                    ambiguous_rows = self.torch.nonzero(
                        ambiguous, as_tuple=False).flatten()
                    ambiguous_scores = (
                        active_selection_scores.index_select(
                            0, ambiguous_rows) +
                        handle.query_sqnorm.index_select(
                            0, active_device.index_select(
                                0, ambiguous_rows))[:, None]).clamp_min_(0.0)
                    full_keys = (
                        ambiguous_scores.view(self.torch.int32).to(
                            self.torch.int64) << 32) | self.global_ids[None, :]
                    _, exact_ids = self.torch.topk(
                        full_keys, max_target, dim=1,
                        largest=False, sorted=True)
                    ranked_local_ids[ambiguous_rows] = exact_ids
        offsets = starts_device[:, None] + self.torch.arange(
            width, dtype=self.torch.int64, device=self.device)[None, :]
        valid = offsets < targets_device[:, None]
        selected = ranked_local_ids.gather(
            1, offsets.clamp_max(max_target - 1))
        flat_offsets = active_device[:, None] * self.base_rows + selected
        if handle.telemetry["score_precision"] == "fp16_refine":
            selected_vectors = self.base_t.T[selected]
            selected_scores = (
                self.base_sqnorm[selected] - 2.0 * self.torch.sum(
                    selected_vectors * handle.query_vectors.index_select(
                        0, active_device)[:, None, :], dim=2) +
                handle.query_sqnorm.index_select(
                    0, active_device)[:, None]).clamp_min_(0.0)
        else:
            selected_scores = (
                handle.scores.reshape(-1)[flat_offsets] +
                handle.query_sqnorm.index_select(
                    0, active_device)[:, None]).clamp_min_(0.0)
        selected_states = (
            self.torch.ones_like(selected, dtype=self.torch.int8)
            if handle.states is None else
            handle.states.reshape(-1)[flat_offsets])
        page_ids = self.torch.where(
            valid, selected + self.global_offset,
            self.torch.full_like(selected, -1))
        page_distances = self.torch.where(
            valid, selected_scores,
            self.torch.full_like(selected_scores, float("inf")))
        page_states = self.torch.where(
            valid, selected_states,
            self.torch.zeros_like(selected_states))
        self._synchronize()
        select_ms = (perf_counter() - started) * 1000.0

        started = perf_counter()
        host_ids = np.ascontiguousarray(
            page_ids.cpu().numpy(), dtype=np.int64)
        host_distances = np.ascontiguousarray(
            page_distances.cpu().numpy(), dtype=np.float32)
        host_states = np.ascontiguousarray(
            page_states.cpu().numpy(), dtype=np.int8)
        d2h_ms = (perf_counter() - started) * 1000.0
        transferred_bytes = int(
            host_ids.nbytes + host_distances.nbytes + host_states.nbytes)

        handle.emitted_counts[active] = targets
        exhausted = np.ascontiguousarray(
            targets >= handle.available_counts[active], dtype=np.bool_)
        handle.telemetry["candidate_select_ms_total"] += float(select_ms)
        handle.telemetry["boundary_tie_repair_queries_total"] += tie_repairs
        handle.telemetry["compact_d2h_ms_total"] += float(d2h_ms)
        handle.telemetry["compact_d2h_bytes_total"] += transferred_bytes
        handle.telemetry["page_rounds"] += 1
        handle.telemetry["peak_gpu_allocated_bytes"] = max(
            int(handle.telemetry["peak_gpu_allocated_bytes"]),
            int(self.torch.cuda.max_memory_allocated(self.device)))
        return CandidatePage(
            query_indices=np.ascontiguousarray(active, dtype=np.int64),
            ids=host_ids, distances=host_distances, states=host_states,
            counts=counts, exhausted=exhausted)

    def full_distance_rows(self, handle, query_indices):
        """Materialize selected complete rows only for explicit fallback."""
        if handle.closed:
            raise RuntimeError("candidate stream handle is closed")
        query_indices = np.ascontiguousarray(query_indices, dtype=np.int64)
        query_device = self.torch.as_tensor(
            query_indices, dtype=self.torch.int64, device=self.device)
        rows_device = (
            handle.scores.index_select(0, query_device) +
            handle.query_sqnorm.index_select(
                0, query_device)[:, None]).clamp_min_(0.0)
        rows = np.ascontiguousarray(
            rows_device.cpu().numpy(), dtype=np.float32)
        self._synchronize()
        handle.telemetry["full_distance_fallback_bytes_total"] += int(
            rows.nbytes)
        return rows

    def close(self, handle):
        handle.closed = True
        handle.scores = None
        handle.query_sqnorm = None
        handle.query_vectors = None
        handle.states = None
        handle.ranked_keys = None
        handle.prefetched_page = None

    @property
    def resident_bytes(self):
        return int(
            self.base_t.numel() * self.base_t.element_size() +
             (0 if self.base_t_half is None else
              self.base_t_half.numel() * self.base_t_half.element_size()) +
            self.base_sqnorm.numel() * self.base_sqnorm.element_size() +
            (0 if self.base_sqnorm_half is None else
             self.base_sqnorm_half.numel() *
             self.base_sqnorm_half.element_size()) +
            self.global_ids.numel() * self.global_ids.element_size())

    def manifest(self):
        return {
            "kind": "torch_exact_candidate_stream",
            "phase": "full_qn_gpu_compact_host_pages",
            "device": str(self.device),
            "base_rows": self.base_rows,
            "dimension": self.dim,
            "global_offset": self.global_offset,
            "resident_bytes": self.resident_bytes,
            "rank_order": (
                "fp16_local_shortlist_then_exact_fp32_(distance,global_id)"
                if self.score_precision == "fp16_refine" else
                "exact_(float32_squared_l2,global_id)"),
            "score_precision": self.score_precision,
            "batch_one_eligibility_cache_max_entries": int(
                self._batch_one_cache_max_entries),
            "batch_one_eligibility_cache_entries": int(
                len(self._eligibility_matrix_cache)),
            "resident_exact_postings": bool(
                self.resident_exact_postings),
            "resident_exact_posting_entries": int(
                len(self._resident_exact_posting_cache)),
            "resident_exact_posting_bytes": int(sum(
                item.eligible_ids.numel() * item.eligible_ids.element_size()
                for item in self._resident_exact_posting_cache.values())),
            "hard_predicate_dtype": "float64",
            "normal_path_full_distance_d2h": False,
        }


class TorchReplicatedExactCandidateStreamProvider:
    """Round-robin batch-one latency across full-corpus GPU replicas.

    Small and medium corpora do not have enough device work to amortize a
    corpus-sharded request and host k-way merge.  A two-GPU serving process
    instead keeps one exact copy per device and sends each independent
    request to one replica.  Per-request latency follows the optimized
    single-GPU kernel while both devices contribute to concurrent QPS.
    """

    def __init__(self, providers):
        self.providers = tuple(providers)
        if len(self.providers) < 2:
            raise ValueError("replicated serving requires multiple providers")
        if len({str(provider.device) for provider in self.providers}) != \
                len(self.providers):
            raise ValueError("replicas require distinct CUDA devices")
        if len({provider.base_rows for provider in self.providers}) != 1 or \
                len({provider.dim for provider in self.providers}) != 1 or \
                any(provider.global_offset != 0 for provider in self.providers):
            raise ValueError("replicas must contain the same full corpus")
        self.base_rows = int(self.providers[0].base_rows)
        self.dim = int(self.providers[0].dim)
        self._next_replica = 0

    def _take_provider(self):
        provider = self.providers[self._next_replica]
        self._next_replica = (self._next_replica + 1) % len(self.providers)
        return provider

    def begin_batch_topk(self, queries, plans, *, topk):
        if len(queries) != 1:
            return None
        provider = self._take_provider()
        inner = provider.begin_batch_topk(queries, plans, topk=topk)
        if inner is None:
            inner = provider.begin_batch(queries, plans)
        inner.telemetry["phase"] = \
            "multi_gpu_replicated_batch1_round_robin"
        inner.telemetry["replica_count"] = int(len(self.providers))
        inner.telemetry["selected_replica_device"] = str(provider.device)
        return ReplicatedCandidateStreamHandle(provider, inner)

    def begin_batch(self, queries, plans):
        if len(queries) != 1:
            raise ValueError(
                "replicated latency provider accepts one request per call")
        provider = self._take_provider()
        inner = provider.begin_batch(queries, plans)
        inner.telemetry["phase"] = \
            "multi_gpu_replicated_batch1_round_robin"
        inner.telemetry["replica_count"] = int(len(self.providers))
        inner.telemetry["selected_replica_device"] = str(provider.device)
        return ReplicatedCandidateStreamHandle(provider, inner)

    def next_page(self, handle, active_mask, *, page_size):
        return handle.provider.next_page(
            handle.inner, active_mask, page_size=page_size)

    def full_distance_rows(self, handle, query_indices):
        return handle.provider.full_distance_rows(
            handle.inner, query_indices)

    def close(self, handle):
        handle.provider.close(handle.inner)

    def clear_online_caches(self):
        manifests = []
        for provider in self.providers:
            if hasattr(provider, "clear_online_caches"):
                manifests.append(provider.clear_online_caches())
        return {
            "replica_manifests": manifests,
            "cleared_entries": int(sum(
                item.get("cleared_entries", 0) for item in manifests)),
            "cleared_bytes": int(sum(
                item.get("cleared_bytes", 0) for item in manifests)),
            "all_online_caches_empty": bool(all(
                item.get("all_online_caches_empty", False)
                for item in manifests)),
        }

    def manifest(self):
        return {
            "kind": "torch_replicated_exact_candidate_stream",
            "phase": "multi_gpu_replicated_batch1_round_robin",
            "base_rows": self.base_rows,
            "dimension": self.dim,
            "devices": [str(provider.device) for provider in self.providers],
            "replicas": [provider.manifest() for provider in self.providers],
            "resident_bytes": int(sum(
                provider.resident_bytes for provider in self.providers)),
            "rank_order": "exact_(float32_squared_l2,global_id)",
            "normal_path_full_distance_d2h": False,
            "normal_path_candidate_kway_merge": False,
            "dispatch": "round_robin_independent_requests",
        }


class TorchShardedExactCandidateStreamProvider:
    """Merge exact compact pages from two or more CUDA corpus shards."""

    def __init__(self, providers):
        self.providers = tuple(providers)
        if len(self.providers) < 2:
            raise ValueError("sharded candidate stream requires multiple providers")
        offsets = [provider.global_offset for provider in self.providers]
        ends = [
            provider.global_offset + provider.base_rows
            for provider in self.providers]
        if offsets[0] != 0 or any(
                left != right for left, right in zip(ends[:-1], offsets[1:])):
            raise ValueError("candidate corpus shards must be contiguous")
        if len({str(provider.device) for provider in self.providers}) != \
                len(self.providers):
            raise ValueError("candidate corpus shards require distinct devices")
        dimensions = {provider.dim for provider in self.providers}
        if len(dimensions) != 1:
            raise ValueError("candidate corpus shards must share a dimension")
        self.base_rows = int(ends[-1])
        self.dim = int(self.providers[0].dim)
        # Keep one worker bound to each device for the lifetime of the
        # provider.  Creating two fresh pools in every begin/page round cost
        # more than the CUDA work for compact top-k serving.
        self._worker_pool = ThreadPoolExecutor(
            max_workers=len(self.providers),
            thread_name_prefix="mifa-corpus-shard")
        # Compile the only hot merge used by the target two-GPU machine.
        _merge_two_sorted_candidate_arrays(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int8), 0)

    @staticmethod
    def _empty_pending():
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int8))

    @staticmethod
    def _append_pending(current, ids, distances, states):
        if not len(ids):
            return current
        if not len(current[0]):
            return (
                np.ascontiguousarray(ids, dtype=np.int64),
                np.ascontiguousarray(distances, dtype=np.float32),
                np.ascontiguousarray(states, dtype=np.int8))
        return (
            np.concatenate((current[0], ids)),
            np.concatenate((current[1], distances)),
            np.concatenate((current[2], states)))

    def begin_batch(self, queries, plans):
        started = perf_counter()
        shard_handles = list(self._worker_pool.map(
            lambda provider: provider.begin_batch(queries, plans),
            self.providers))
        begin_wall_ms = (perf_counter() - started) * 1000.0
        query_count = shard_handles[0].query_count
        if any(handle.query_count != query_count for handle in shard_handles):
            raise RuntimeError("candidate shard query counts disagree")
        available_counts = np.sum(np.stack([
            handle.available_counts for handle in shard_handles]), axis=0)
        telemetry = {
            "distance_compute_ms_total": float(max(
                handle.telemetry["distance_compute_ms_total"]
                for handle in shard_handles)),
            "predicate_column_upload_ms_total": float(max(
                handle.telemetry["predicate_column_upload_ms_total"]
                for handle in shard_handles)),
            "hard_predicate_ms_total": float(max(
                handle.telemetry["hard_predicate_ms_total"]
                for handle in shard_handles)),
            "candidate_select_ms_total": 0.0,
            "compact_d2h_ms_total": 0.0,
            "compact_d2h_bytes_total": int(sum(
                handle.telemetry["compact_d2h_bytes_total"]
                for handle in shard_handles)),
            "full_distance_fallback_bytes_total": 0,
            "page_rounds": 0,
            "phase": "multi_gpu_corpus_sharded_compact_host_merge",
            "statistical_group_prefetch": bool(all(
                handle.telemetry.get("statistical_group_prefetch", False)
                for handle in shard_handles)),
            "multi_gpu_begin_wall_ms_total": float(begin_wall_ms),
            "multi_gpu_cpu_merge_ms_total": 0.0,
            "multi_gpu_peak_pending_host_bytes": 0,
            "peak_gpu_allocated_bytes": int(sum(
                handle.telemetry["peak_gpu_allocated_bytes"]
                for handle in shard_handles)),
            "peak_gpu_allocated_bytes_per_device": [int(
                handle.telemetry["peak_gpu_allocated_bytes"])
                for handle in shard_handles],
        }
        pending = [[
            self._empty_pending() for _ in self.providers]
            for _ in range(query_count)]
        return ShardedCandidateStreamHandle(
            shard_handles=shard_handles, pending=pending,
            available_counts=np.ascontiguousarray(
                available_counts, dtype=np.int64),
            emitted_counts=np.zeros(query_count, dtype=np.int64),
            query_count=query_count, telemetry=telemetry)

    def next_page(self, handle, active_mask, *, page_size):
        if handle.closed:
            raise RuntimeError("candidate stream handle is closed")
        active_mask = np.asarray(active_mask, dtype=np.bool_)
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

        before_select = [float(shard.telemetry[
            "candidate_select_ms_total"]) for shard in handle.shard_handles]
        before_d2h = [float(shard.telemetry[
            "compact_d2h_ms_total"]) for shard in handle.shard_handles]
        before_bytes = [int(shard.telemetry[
            "compact_d2h_bytes_total"]) for shard in handle.shard_handles]
        pages = list(self._worker_pool.map(
            lambda item: item[0].next_page(
                item[1], active_mask, page_size=page_size),
            zip(self.providers, handle.shard_handles)))
        for shard_index, page in enumerate(pages):
            for page_row, query_index_value in enumerate(page.query_indices):
                query_index = int(query_index_value)
                count = int(page.counts[page_row])
                handle.pending[query_index][shard_index] = \
                    self._append_pending(
                        handle.pending[query_index][shard_index],
                        page.ids[page_row, :count],
                        page.distances[page_row, :count],
                        page.states[page_row, :count])

        pending_input_bytes = int(sum(
            stream[0].nbytes + stream[1].nbytes + stream[2].nbytes
            for query_streams in handle.pending
            for stream in query_streams))
        handle.telemetry["multi_gpu_peak_pending_host_bytes"] = max(
            int(handle.telemetry["multi_gpu_peak_pending_host_bytes"]),
            pending_input_bytes)

        merge_started = perf_counter()
        merged = []
        exhausted_rows = []
        for query_index in active:
            streams = handle.pending[int(query_index)]
            if len(streams) == 2:
                (ids, distances, states,
                 consumed_left, consumed_right) = \
                    _merge_two_sorted_candidate_arrays(
                        *streams[0], *streams[1], int(page_size))
                consumed = (consumed_left, consumed_right)
            else:
                all_ids = np.concatenate([stream[0] for stream in streams])
                all_distances = np.concatenate([
                    stream[1] for stream in streams])
                all_states = np.concatenate([
                    stream[2] for stream in streams])
                order = np.lexsort((all_ids, all_distances))
                chosen = order[:int(page_size)]
                ids = all_ids[chosen]
                distances = all_distances[chosen]
                states = all_states[chosen]
                consumed = tuple(int(np.count_nonzero(
                    np.isin(stream[0], ids))) for stream in streams)
            for shard_index, count in enumerate(consumed):
                stream = streams[shard_index]
                streams[shard_index] = (
                    stream[0][count:], stream[1][count:], stream[2][count:])
            handle.emitted_counts[int(query_index)] += len(ids)
            shard_exhausted = [
                shard.emitted_counts[int(query_index)] >=
                shard.available_counts[int(query_index)]
                for shard in handle.shard_handles]
            globally_exhausted = bool(
                all(shard_exhausted) and
                all(not len(stream[0]) for stream in streams))
            merged.append((ids, distances, states))
            exhausted_rows.append(globally_exhausted)
        merge_ms = (perf_counter() - merge_started) * 1000.0

        width = max((len(row[0]) for row in merged), default=0)
        output_ids = np.full((len(active), width), -1, dtype=np.int64)
        output_distances = np.full(
            (len(active), width), np.inf, dtype=np.float32)
        output_states = np.zeros((len(active), width), dtype=np.int8)
        counts = np.empty(len(active), dtype=np.int64)
        for row_index, (ids, distances, states) in enumerate(merged):
            counts[row_index] = len(ids)
            output_ids[row_index, :len(ids)] = ids
            output_distances[row_index, :len(ids)] = distances
            output_states[row_index, :len(ids)] = states
        pending_output_bytes = int(sum(
            stream[0].nbytes + stream[1].nbytes + stream[2].nbytes
            for query_streams in handle.pending
            for stream in query_streams))
        handle.telemetry["multi_gpu_peak_pending_host_bytes"] = max(
            int(handle.telemetry["multi_gpu_peak_pending_host_bytes"]),
            int(pending_output_bytes + output_ids.nbytes +
                output_distances.nbytes + output_states.nbytes))

        select_deltas = [
            float(shard.telemetry["candidate_select_ms_total"]) - before
            for shard, before in zip(handle.shard_handles, before_select)]
        d2h_deltas = [
            float(shard.telemetry["compact_d2h_ms_total"]) - before
            for shard, before in zip(handle.shard_handles, before_d2h)]
        byte_deltas = [
            int(shard.telemetry["compact_d2h_bytes_total"]) - before
            for shard, before in zip(handle.shard_handles, before_bytes)]
        handle.telemetry["candidate_select_ms_total"] += max(
            select_deltas, default=0.0)
        handle.telemetry["compact_d2h_ms_total"] += max(
            d2h_deltas, default=0.0)
        handle.telemetry["compact_d2h_bytes_total"] += sum(byte_deltas)
        handle.telemetry["multi_gpu_cpu_merge_ms_total"] += float(merge_ms)
        handle.telemetry["page_rounds"] += 1
        handle.telemetry["peak_gpu_allocated_bytes"] = int(sum(
            shard.telemetry["peak_gpu_allocated_bytes"]
            for shard in handle.shard_handles))
        handle.telemetry["peak_gpu_allocated_bytes_per_device"] = [int(
            shard.telemetry["peak_gpu_allocated_bytes"])
            for shard in handle.shard_handles]
        return CandidatePage(
            query_indices=np.ascontiguousarray(active, dtype=np.int64),
            ids=output_ids, distances=output_distances,
            states=output_states, counts=counts,
            exhausted=np.asarray(exhausted_rows, dtype=np.bool_))

    def full_distance_rows(self, handle, query_indices):
        rows = list(self._worker_pool.map(
            lambda item: item[0].full_distance_rows(
                item[1], query_indices),
            zip(self.providers, handle.shard_handles)))
        output = np.ascontiguousarray(np.concatenate(rows, axis=1))
        handle.telemetry["full_distance_fallback_bytes_total"] += int(
            output.nbytes)
        return output

    def close(self, handle):
        if handle.closed:
            return
        for provider, shard in zip(self.providers, handle.shard_handles):
            provider.close(shard)
        handle.closed = True
        handle.pending.clear()

    def shutdown(self, *, wait=True):
        pool = getattr(self, "_worker_pool", None)
        if pool is not None:
            self._worker_pool = None
            pool.shutdown(wait=bool(wait), cancel_futures=True)

    def __del__(self):
        try:
            self.shutdown(wait=False)
        except Exception:
            pass

    def manifest(self):
        return {
            "kind": "torch_sharded_exact_candidate_stream",
            "phase": "multi_gpu_compact_host_merge",
            "base_rows": self.base_rows,
            "dimension": self.dim,
            "devices": [str(provider.device) for provider in self.providers],
            "shards": [provider.manifest() for provider in self.providers],
            "rank_order": "exact_(float32_squared_l2,global_id)",
            "normal_path_full_distance_d2h": False,
        }


class TorchQueryShardedExactCandidateStreamProvider:
    """Serve disjoint query rows concurrently from replicated CUDA corpora.

    Corpus sharding is useful when one corpus cannot fit on one device, but
    for a 1M x 128 corpus it duplicates top-M selection and requires a host
    k-way merge.  This specialization instead keeps one complete resident
    corpus per GPU and assigns each query to exactly one device.  Candidate
    ordering remains local exact ``(distance,id)`` order, so the CPU consumer
    only reassembles query rows and never merges candidate streams.
    """

    def __init__(self, providers):
        self.providers = tuple(providers)
        if len(self.providers) < 2:
            raise ValueError("query sharding requires multiple providers")
        if len({str(provider.device) for provider in self.providers}) != \
                len(self.providers):
            raise ValueError("query shards require distinct CUDA devices")
        dimensions = {provider.dim for provider in self.providers}
        base_rows = {provider.base_rows for provider in self.providers}
        offsets = {provider.global_offset for provider in self.providers}
        if len(dimensions) != 1 or len(base_rows) != 1 or offsets != {0}:
            raise ValueError(
                "query shards require identical full corpora with offset zero")
        self.base_rows = int(self.providers[0].base_rows)
        self.dim = int(self.providers[0].dim)
        self._worker_pool = ThreadPoolExecutor(
            max_workers=len(self.providers),
            thread_name_prefix="mifa-query-shard")
        self._query_index_cache = {}

    def begin_batch(self, queries, plans):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        plans = list(plans)
        if len(queries) != len(plans):
            raise ValueError("queries and compiled predicates must align")
        if len(queries) < len(self.providers):
            raise ValueError(
                "query count must be at least the number of query shards")
        query_indices = self._query_index_cache.get(len(queries))
        if query_indices is None:
            query_indices = tuple(
                np.ascontiguousarray(values, dtype=np.int64)
                for values in np.array_split(
                    np.arange(len(queries), dtype=np.int64),
                    len(self.providers)))
            self._query_index_cache[len(queries)] = query_indices

        def begin(item):
            provider, indices = item
            return provider.begin_batch(
                queries[indices], [plans[int(index)] for index in indices])

        started = perf_counter()
        shard_handles = list(self._worker_pool.map(
            begin, zip(self.providers, query_indices)))
        begin_wall_ms = (perf_counter() - started) * 1000.0
        available_counts = np.empty(len(queries), dtype=np.int64)
        for indices, shard in zip(query_indices, shard_handles):
            available_counts[indices] = shard.available_counts
        telemetry = {
            "distance_compute_ms_total": float(max(
                shard.telemetry["distance_compute_ms_total"]
                for shard in shard_handles)),
            "predicate_column_upload_ms_total": float(max(
                shard.telemetry["predicate_column_upload_ms_total"]
                for shard in shard_handles)),
            "hard_predicate_ms_total": float(max(
                shard.telemetry["hard_predicate_ms_total"]
                for shard in shard_handles)),
            "candidate_select_ms_total": 0.0,
            "compact_d2h_ms_total": 0.0,
            "compact_d2h_bytes_total": int(sum(
                shard.telemetry["compact_d2h_bytes_total"]
                for shard in shard_handles)),
            "full_distance_fallback_bytes_total": 0,
            "page_rounds": 0,
            "phase": "multi_gpu_replicated_corpus_query_sharded",
            "statistical_group_prefetch": bool(all(
                shard.telemetry.get("statistical_group_prefetch", False)
                for shard in shard_handles)),
            "multi_gpu_begin_wall_ms_total": float(begin_wall_ms),
            "multi_gpu_cpu_merge_ms_total": 0.0,
            "multi_gpu_peak_pending_host_bytes": 0,
            "peak_gpu_allocated_bytes": int(sum(
                shard.telemetry["peak_gpu_allocated_bytes"]
                for shard in shard_handles)),
            "peak_gpu_allocated_bytes_per_device": [int(
                shard.telemetry["peak_gpu_allocated_bytes"])
                for shard in shard_handles],
            "tree_probe_plan_ms_total": float(max(
                shard.telemetry.get("tree_probe_plan_ms_total", 0.0)
                for shard in shard_handles)),
            "tree_probe_candidate_rows_total": int(sum(
                shard.telemetry.get(
                    "tree_probe_candidate_rows_total", 0)
                for shard in shard_handles)),
            "tree_probe_count": int(max(
                shard.telemetry.get("tree_probe_count", 0)
                for shard in shard_handles)),
        }
        return QueryShardedCandidateStreamHandle(
            shard_handles=shard_handles, query_indices=query_indices,
            available_counts=available_counts,
            emitted_counts=np.zeros(len(queries), dtype=np.int64),
            query_count=len(queries), telemetry=telemetry)

    def next_page(self, handle, active_mask, *, page_size):
        if handle.closed:
            raise RuntimeError("candidate stream handle is closed")
        active_mask = np.asarray(active_mask, dtype=np.bool_)
        if active_mask.shape != (handle.query_count,):
            raise ValueError("active mask does not align with candidate batch")
        before_select = [float(shard.telemetry[
            "candidate_select_ms_total"]) for shard in handle.shard_handles]
        before_d2h = [float(shard.telemetry[
            "compact_d2h_ms_total"]) for shard in handle.shard_handles]
        before_bytes = [int(shard.telemetry[
            "compact_d2h_bytes_total"]) for shard in handle.shard_handles]

        def fetch(item):
            provider, shard, indices = item
            return provider.next_page(
                shard, active_mask[indices], page_size=page_size)

        pages = list(self._worker_pool.map(fetch, zip(
            self.providers, handle.shard_handles,
            handle.query_indices)))

        reassembly_started = perf_counter()
        row_counts = [len(page.query_indices) for page in pages]
        output_rows = int(sum(row_counts))
        width = max((page.ids.shape[1] for page in pages), default=0)
        output_ids = np.full((output_rows, width), -1, dtype=np.int64)
        output_distances = np.full(
            (output_rows, width), np.inf, dtype=np.float32)
        output_states = np.zeros((output_rows, width), dtype=np.int8)
        output_queries = np.empty(output_rows, dtype=np.int64)
        counts = np.empty(output_rows, dtype=np.int64)
        exhausted = np.empty(output_rows, dtype=np.bool_)
        cursor = 0
        for indices, page, count in zip(
                handle.query_indices, pages, row_counts):
            end = cursor + count
            page_width = int(page.ids.shape[1])
            output_queries[cursor:end] = indices[page.query_indices]
            output_ids[cursor:end, :page_width] = page.ids
            output_distances[cursor:end, :page_width] = page.distances
            output_states[cursor:end, :page_width] = page.states
            counts[cursor:end] = page.counts
            exhausted[cursor:end] = page.exhausted
            cursor = end
        for indices, shard in zip(
                handle.query_indices, handle.shard_handles):
            handle.emitted_counts[indices] = shard.emitted_counts
        reassembly_ms = (perf_counter() - reassembly_started) * 1000.0

        select_deltas = [
            float(shard.telemetry["candidate_select_ms_total"]) - before
            for shard, before in zip(handle.shard_handles, before_select)]
        d2h_deltas = [
            float(shard.telemetry["compact_d2h_ms_total"]) - before
            for shard, before in zip(handle.shard_handles, before_d2h)]
        byte_deltas = [
            int(shard.telemetry["compact_d2h_bytes_total"]) - before
            for shard, before in zip(handle.shard_handles, before_bytes)]
        handle.telemetry["candidate_select_ms_total"] += max(
            select_deltas, default=0.0)
        handle.telemetry["compact_d2h_ms_total"] += max(
            d2h_deltas, default=0.0)
        handle.telemetry["compact_d2h_bytes_total"] += sum(byte_deltas)
        handle.telemetry["multi_gpu_cpu_merge_ms_total"] += float(
            reassembly_ms)
        handle.telemetry["page_rounds"] += 1
        compact_bytes = int(
            output_ids.nbytes + output_distances.nbytes +
            output_states.nbytes)
        handle.telemetry["multi_gpu_peak_pending_host_bytes"] = max(
            int(handle.telemetry["multi_gpu_peak_pending_host_bytes"]),
            compact_bytes)
        handle.telemetry["peak_gpu_allocated_bytes"] = int(sum(
            shard.telemetry["peak_gpu_allocated_bytes"]
            for shard in handle.shard_handles))
        handle.telemetry["peak_gpu_allocated_bytes_per_device"] = [int(
            shard.telemetry["peak_gpu_allocated_bytes"])
            for shard in handle.shard_handles]
        return CandidatePage(
            query_indices=output_queries, ids=output_ids,
            distances=output_distances, states=output_states,
            counts=counts, exhausted=exhausted)

    def full_distance_rows(self, handle, query_indices):
        if handle.closed:
            raise RuntimeError("candidate stream handle is closed")
        query_indices = np.ascontiguousarray(query_indices, dtype=np.int64)
        output = np.empty(
            (len(query_indices), self.base_rows), dtype=np.float32)
        for provider, shard, assigned in zip(
                self.providers, handle.shard_handles, handle.query_indices):
            positions = np.flatnonzero(np.isin(query_indices, assigned))
            if not len(positions):
                continue
            local_indices = np.searchsorted(
                assigned, query_indices[positions]).astype(np.int64)
            output[positions] = provider.full_distance_rows(
                shard, local_indices)
        handle.telemetry["full_distance_fallback_bytes_total"] += int(
            output.nbytes)
        return output

    def close(self, handle):
        if handle.closed:
            return
        for provider, shard in zip(self.providers, handle.shard_handles):
            provider.close(shard)
        handle.closed = True
        handle.shard_handles.clear()

    def shutdown(self, *, wait=True):
        pool = getattr(self, "_worker_pool", None)
        if pool is not None:
            self._worker_pool = None
            pool.shutdown(wait=bool(wait), cancel_futures=True)

    def __del__(self):
        try:
            self.shutdown(wait=False)
        except Exception:
            pass

    def manifest(self):
        return {
            "kind": "torch_query_sharded_exact_candidate_stream",
            "phase": "multi_gpu_replicated_corpus_query_sharded",
            "base_rows": self.base_rows,
            "dimension": self.dim,
            "devices": [str(provider.device) for provider in self.providers],
            "replicas": [provider.manifest() for provider in self.providers],
            "rank_order": "exact_(float32_squared_l2,global_id)",
            "normal_path_full_distance_d2h": False,
            "normal_path_candidate_kway_merge": False,
        }
