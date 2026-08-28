"""Unified node--tuple best-first execution for paper-aligned MIFA."""
from __future__ import annotations

import heapq
import threading
from time import perf_counter

import numpy as np
from numba import njit, prange

from mifa.index.ann_backend import ExactBackend


@njit(cache=True, fastmath=False)
def _rank_can_affect(buffer_ids, buffer_distances, count, k, distance, tuple_id):
    if count < k:
        return True
    worst_distance = buffer_distances[k - 1]
    worst_id = buffer_ids[k - 1]
    return (distance < worst_distance or
            (distance == worst_distance and tuple_id < worst_id))


@njit(cache=True, fastmath=False)
def _rank_insert(buffer_ids, buffer_distances, count, k, distance, tuple_id):
    if count == k and not _rank_can_affect(
            buffer_ids, buffer_distances, count, k, distance, tuple_id):
        return count
    position = count if count < k else k - 1
    if count < k:
        count += 1
    while position > 0:
        previous_distance = buffer_distances[position - 1]
        previous_id = buffer_ids[position - 1]
        if (previous_distance < distance or
                (previous_distance == distance and previous_id <= tuple_id)):
            break
        buffer_distances[position] = previous_distance
        buffer_ids[position] = previous_id
        position -= 1
    buffer_distances[position] = distance
    buffer_ids[position] = tuple_id
    return count


@njit(cache=True, fastmath=False)
def _event_precedes(left_distance, left_id, right_distance, right_id):
    return (left_distance < right_distance or
            (left_distance == right_distance and left_id < right_id))


@njit(cache=True, fastmath=False)
def _event_heap_push(heap_distances, heap_ids, size, distance, tuple_id):
    position = int(size)
    size += 1
    while position > 0:
        parent = (position - 1) // 2
        if not _event_precedes(
                distance, tuple_id,
                heap_distances[parent], heap_ids[parent]):
            break
        heap_distances[position] = heap_distances[parent]
        heap_ids[position] = heap_ids[parent]
        position = parent
    heap_distances[position] = distance
    heap_ids[position] = tuple_id
    return size


@njit(cache=True, fastmath=False)
def _event_heap_pop(heap_distances, heap_ids, size):
    distance = heap_distances[0]
    tuple_id = heap_ids[0]
    size -= 1
    if size > 0:
        tail_distance = heap_distances[size]
        tail_id = heap_ids[size]
        position = 0
        while True:
            left = 2 * position + 1
            if left >= size:
                break
            right = left + 1
            child = left
            if (right < size and _event_precedes(
                    heap_distances[right], heap_ids[right],
                    heap_distances[left], heap_ids[left])):
                child = right
            if not _event_precedes(
                    heap_distances[child], heap_ids[child],
                    tail_distance, tail_id):
                break
            heap_distances[position] = heap_distances[child]
            heap_ids[position] = heap_ids[child]
            position = child
        heap_distances[position] = tail_distance
        heap_ids[position] = tail_id
    return distance, tuple_id, size


@njit(cache=True, fastmath=False)
def _direct_state(values, missing, atom_attrs, atom_ops, atom_lower,
                  atom_upper, tuple_id):
    unresolved = False
    for atom_index in range(len(atom_attrs)):
        attr = atom_attrs[atom_index]
        if missing[attr, tuple_id]:
            unresolved = True
            continue
        value = values[attr, tuple_id]
        op = atom_ops[atom_index]
        if op == 0:
            satisfied = value <= atom_lower[atom_index]
        elif op == 1:
            satisfied = value >= atom_lower[atom_index]
        elif op == 2:
            satisfied = (value >= atom_lower[atom_index] and
                         value <= atom_upper[atom_index])
        else:
            satisfied = value == atom_lower[atom_index]
        if not satisfied:
            return 0  # hard fail dominates missing atoms
    return 2 if unresolved else 1  # unresolved / hard pass


@njit(cache=True, fastmath=False)
def _distance_sq_f32(embeddings, sqnorm, query, query_sqnorm, tuple_id):
    # Numba dispatches the contiguous 1-D dot to BLAS, matching the scalar
    # reference path's float32 reduction while avoiding a Python call.
    dot = np.dot(embeddings[tuple_id], query)
    value = float(sqnorm[tuple_id]) - 2.0 * float(dot) + query_sqnorm
    return value if value > 0.0 else 0.0


@njit(cache=True, fastmath=False)
def _fused_seed_candidates(
        embeddings, sqnorm, query, query_sqnorm, tuple_ids, registered,
        values, missing, atom_attrs, atom_ops, atom_lower, atom_upper,
        buffer_ids, buffer_distances, count, k, prepared_distances,
        use_prepared_distances):
    hard_decisions = 0
    exact_distances = 0
    updates = 0
    for position in range(len(tuple_ids)):
        tuple_id = int(tuple_ids[position])
        if registered[tuple_id]:
            continue
        hard_decisions += 1
        state = _direct_state(
            values, missing, atom_attrs, atom_ops, atom_lower, atom_upper,
            tuple_id)
        if state == 2:
            # Paper semantics: unresolved ANN seeds remain unregistered and
            # are encountered later through complete tree coverage.
            continue
        registered[tuple_id] = True
        if state == 1:
            distance = (float(prepared_distances[tuple_id])
                        if use_prepared_distances else
                        _distance_sq_f32(
                            embeddings, sqnorm, query, query_sqnorm, tuple_id))
            exact_distances += 1
            updates += 1
            count = _rank_insert(
                buffer_ids, buffer_distances, count, k, distance, tuple_id)
    return count, hard_decisions, exact_distances, updates


@njit(cache=True, fastmath=False)
def _fused_leaf_candidates(
        embeddings, sqnorm, query, query_sqnorm, tuple_ids, registered,
        values, missing, atom_attrs, atom_ops, atom_lower, atom_upper,
        buffer_ids, buffer_distances, count, k, prepared_distances,
        use_prepared_distances, statistical_cache, use_statistical_cache):
    unresolved_ids = np.empty(len(tuple_ids), dtype=np.int64)
    unresolved_distances = np.empty(len(tuple_ids), dtype=np.float64)
    unresolved_count = 0
    visited = 0
    exact_distances = 0
    hard_decisions = 0
    boundary_skips = 0
    updates = 0
    cached_accepts = 0
    cached_rejects = 0
    cached_abstains = 0
    for position in range(len(tuple_ids)):
        tuple_id = int(tuple_ids[position])
        if registered[tuple_id]:
            continue
        visited += 1
        distance = (float(prepared_distances[tuple_id])
                    if use_prepared_distances else
                    _distance_sq_f32(
                        embeddings, sqnorm, query, query_sqnorm, tuple_id))
        exact_distances += 1
        if not _rank_can_affect(
                buffer_ids, buffer_distances, count, k, distance, tuple_id):
            registered[tuple_id] = True
            boundary_skips += 1
            continue
        hard_decisions += 1
        state = _direct_state(
            values, missing, atom_attrs, atom_ops, atom_lower, atom_upper,
            tuple_id)
        if state == 1:
            updates += 1
            count = _rank_insert(
                buffer_ids, buffer_distances, count, k, distance, tuple_id)
        elif state == 2:
            cached = (-1 if not use_statistical_cache else
                      int(statistical_cache[tuple_id]))
            if cached == 1:
                cached_accepts += 1
                updates += 1
                count = _rank_insert(
                    buffer_ids, buffer_distances, count, k,
                    distance, tuple_id)
            elif cached == 2:
                cached_rejects += 1
            elif cached == 3:
                cached_abstains += 1
            else:
                unresolved_ids[unresolved_count] = tuple_id
                unresolved_distances[unresolved_count] = distance
                unresolved_count += 1
        registered[tuple_id] = True
    return (count, unresolved_ids[:unresolved_count],
            unresolved_distances[:unresolved_count], visited,
            exact_distances, hard_decisions, boundary_skips, updates,
            cached_accepts, cached_rejects, cached_abstains)


# Mutable native frontier state layout.  Keeping the scalars in one int64
# vector lets Numba resume the exact heap traversal without rebuilding any
# query-sized structure after a Python statistical-inference call.
_LEAF_POSITION = 0
_LEAF_HEAP_SIZE = 1
_LEAF_BUFFER_COUNT = 2
_LEAF_NODE_POPS = 3
_LEAF_VISITED = 4
_LEAF_EXACT = 5
_LEAF_HARD = 6
_LEAF_BOUNDARY_SKIPS = 7
_LEAF_REGISTERED = 8
_LEAF_UPDATES = 9
_LEAF_MAX_QUEUE = 10
_LEAF_TUPLE_INSERTIONS = 11
_LEAF_TUPLE_POPS = 12
_LEAF_CACHE_ACCEPTS = 13
_LEAF_CACHE_REJECTS = 14
_LEAF_CACHE_ABSTAINS = 15
_LEAF_STATE_SIZE = 16


@njit(cache=True, fastmath=False)
def _fused_leaf_frontier_resume(
        leaf_order, leaf_lowers, leaf_offsets, leaf_tuple_ids,
        values, missing, atom_attrs, atom_ops, atom_lower, atom_upper,
        prepared_distances, statistical_cache, k,
        buffer_ids, buffer_distances, registered,
        heap_distances, heap_ids, frontier_state):
    """Resume the exact leaf/tuple frontier until inference or completion.

    The unresolved tuple remains at the heap root while Python fills its dense
    statistical cache entry.  A negative return value means the queue has
    completed; otherwise the returned tuple is the sole required inference.
    """
    leaf_position = int(frontier_state[_LEAF_POSITION])
    heap_size = int(frontier_state[_LEAF_HEAP_SIZE])
    buffer_count = int(frontier_state[_LEAF_BUFFER_COUNT])

    while leaf_position < len(leaf_order) or heap_size > 0:
        boundary = (np.inf if buffer_count < k else
                    buffer_distances[k - 1])
        leaf_frontier = (leaf_lowers[leaf_position]
                         if leaf_position < len(leaf_order) else np.inf)
        tuple_frontier = (heap_distances[0] if heap_size > 0 else np.inf)
        frontier = min(leaf_frontier, tuple_frontier)
        if buffer_count == k and frontier > boundary:
            break

        # Nodes precede tuple events with equal keys.
        if leaf_frontier <= tuple_frontier:
            leaf_slot = int(leaf_order[leaf_position])
            leaf_position += 1
            frontier_state[_LEAF_NODE_POPS] += 1
            start = int(leaf_offsets[leaf_slot])
            stop = int(leaf_offsets[leaf_slot + 1])
            for offset in range(start, stop):
                tuple_id = int(leaf_tuple_ids[offset])
                if registered[tuple_id]:
                    continue
                registered[tuple_id] = True
                frontier_state[_LEAF_REGISTERED] += 1
                frontier_state[_LEAF_VISITED] += 1
                distance = float(prepared_distances[tuple_id])
                frontier_state[_LEAF_EXACT] += 1
                if not _rank_can_affect(
                        buffer_ids, buffer_distances, buffer_count, k,
                        distance, tuple_id):
                    frontier_state[_LEAF_BOUNDARY_SKIPS] += 1
                    continue
                frontier_state[_LEAF_HARD] += 1
                state = _direct_state(
                    values, missing, atom_attrs, atom_ops,
                    atom_lower, atom_upper, tuple_id)
                if state == 1:
                    frontier_state[_LEAF_UPDATES] += 1
                    buffer_count = _rank_insert(
                        buffer_ids, buffer_distances, buffer_count, k,
                        distance, tuple_id)
                elif state == 2:
                    heap_size = _event_heap_push(
                        heap_distances, heap_ids, heap_size,
                        distance, tuple_id)
                    frontier_state[_LEAF_TUPLE_INSERTIONS] += 1
        else:
            # Do not pop an uncached tuple.  It stays as the deterministic
            # frontier event and is processed exactly once after cache fill.
            distance = float(heap_distances[0])
            tuple_id = int(heap_ids[0])
            if _rank_can_affect(
                    buffer_ids, buffer_distances, buffer_count, k,
                    distance, tuple_id):
                cached = int(statistical_cache[tuple_id])
                if cached < 0:
                    frontier_state[_LEAF_POSITION] = leaf_position
                    frontier_state[_LEAF_HEAP_SIZE] = heap_size
                    frontier_state[_LEAF_BUFFER_COUNT] = buffer_count
                    logical_queue_size = (
                        len(leaf_order) - leaf_position + heap_size)
                    if logical_queue_size > frontier_state[_LEAF_MAX_QUEUE]:
                        frontier_state[_LEAF_MAX_QUEUE] = logical_queue_size
                    return tuple_id
            distance, tuple_id, heap_size = _event_heap_pop(
                heap_distances, heap_ids, heap_size)
            frontier_state[_LEAF_TUPLE_POPS] += 1
            if not _rank_can_affect(
                    buffer_ids, buffer_distances, buffer_count, k,
                    distance, tuple_id):
                frontier_state[_LEAF_BOUNDARY_SKIPS] += 1
                continue
            cached = int(statistical_cache[tuple_id])
            if cached == 1:
                frontier_state[_LEAF_CACHE_ACCEPTS] += 1
                frontier_state[_LEAF_UPDATES] += 1
                buffer_count = _rank_insert(
                    buffer_ids, buffer_distances, buffer_count, k,
                    distance, tuple_id)
            elif cached == 2:
                frontier_state[_LEAF_CACHE_REJECTS] += 1
            else:
                frontier_state[_LEAF_CACHE_ABSTAINS] += 1
        logical_queue_size = len(leaf_order) - leaf_position + heap_size
        if logical_queue_size > frontier_state[_LEAF_MAX_QUEUE]:
            frontier_state[_LEAF_MAX_QUEUE] = logical_queue_size

    frontier_state[_LEAF_POSITION] = leaf_position
    frontier_state[_LEAF_HEAP_SIZE] = heap_size
    frontier_state[_LEAF_BUFFER_COUNT] = buffer_count
    return -1


@njit(cache=True, fastmath=False, parallel=True)
def _leaf_lower_bounds_batch(mbr_lo, mbr_hi, projected_queries, output):
    """Compute the scalar executor's float64 MBR bounds in parallel."""
    for query_index in prange(len(projected_queries)):
        for leaf_index in range(len(mbr_lo)):
            total = 0.0
            for dim in range(projected_queries.shape[1]):
                value = projected_queries[query_index, dim]
                if value < mbr_lo[leaf_index, dim]:
                    difference = float(mbr_lo[leaf_index, dim] - value)
                    total += difference * difference
                elif value > mbr_hi[leaf_index, dim]:
                    difference = float(value - mbr_hi[leaf_index, dim])
                    total += difference * difference
            output[query_index, leaf_index] = total


@njit(cache=True, fastmath=False)
def _precomputed_leaf_min_bounds(
        prepared_distances, leaf_offsets, leaf_tuple_ids, output):
    """Bound leaves in the exact finite-precision distance domain served.

    A projected MBR is a lower bound on mathematical squared L2, but it need
    not be a lower bound on a precomputed float32 row after cancellation and
    zero clamping.  Taking each leaf's minimum from that already-materialized
    row preserves exact ``(distance, tuple_id)`` semantics, including ties.
    """
    for leaf_index in range(len(output)):
        lower = np.inf
        start = int(leaf_offsets[leaf_index])
        stop = int(leaf_offsets[leaf_index + 1])
        for offset in range(start, stop):
            distance = float(prepared_distances[int(leaf_tuple_ids[offset])])
            if distance < lower:
                lower = distance
        output[leaf_index] = lower


@njit(cache=True, fastmath=False, parallel=True)
def _precomputed_leaf_min_bounds_batch(
        prepared_distances, leaf_offsets, leaf_tuple_ids, output):
    """Parallel batch form of :func:`_precomputed_leaf_min_bounds`."""
    for query_index in prange(len(prepared_distances)):
        for leaf_index in range(output.shape[1]):
            lower = np.inf
            start = int(leaf_offsets[leaf_index])
            stop = int(leaf_offsets[leaf_index + 1])
            for offset in range(start, stop):
                distance = float(prepared_distances[
                    query_index, int(leaf_tuple_ids[offset])])
                if distance < lower:
                    lower = distance
            output[query_index, leaf_index] = lower


@njit(cache=True, fastmath=False, parallel=True)
def _fused_leaf_batch_seed(
        seed_ids, prepared_distances, values, missing,
        atom_attrs, atom_ops, atom_lower, atom_upper, atom_counts, k,
        buffer_ids, buffer_distances, registered, frontier_states,
        leaf_count):
    """Initialize independent exact frontiers across a native query batch."""
    for query_index in prange(len(seed_ids)):
        count = 0
        hard = 0
        exact = 0
        updates = 0
        registered_count = 0
        atom_count = int(atom_counts[query_index])
        for position in range(seed_ids.shape[1]):
            tuple_id = int(seed_ids[query_index, position])
            if tuple_id < 0 or registered[query_index, tuple_id]:
                continue
            hard += 1
            state = _direct_state(
                values, missing,
                atom_attrs[query_index, :atom_count],
                atom_ops[query_index, :atom_count],
                atom_lower[query_index, :atom_count],
                atom_upper[query_index, :atom_count], tuple_id)
            if state == 2:
                continue
            registered[query_index, tuple_id] = True
            registered_count += 1
            if state == 1:
                exact += 1
                updates += 1
                count = _rank_insert(
                    buffer_ids[query_index], buffer_distances[query_index],
                    count, k,
                    float(prepared_distances[query_index, tuple_id]),
                    tuple_id)
        frontier_states[query_index, _LEAF_BUFFER_COUNT] = count
        frontier_states[query_index, _LEAF_HARD] = hard
        frontier_states[query_index, _LEAF_EXACT] = exact
        frontier_states[query_index, _LEAF_UPDATES] = updates
        frontier_states[query_index, _LEAF_REGISTERED] = registered_count
        frontier_states[query_index, _LEAF_MAX_QUEUE] = leaf_count


@njit(cache=True, fastmath=False, parallel=True)
def _fused_leaf_frontier_resume_batch(
        active, needed, leaf_orders, ordered_lowers,
        leaf_offsets, leaf_tuple_ids, values, missing,
        atom_attrs, atom_ops, atom_lower, atom_upper, atom_counts,
        prepared_distances, statistical_caches, cache_indices, k,
        buffer_ids, buffer_distances, registered,
        heap_distances, heap_ids, frontier_states):
    """Advance every active query independently on Numba's worker pool."""
    for query_index in prange(len(active)):
        if not active[query_index]:
            needed[query_index] = -1
            continue
        atom_count = int(atom_counts[query_index])
        needed[query_index] = _fused_leaf_frontier_resume(
            leaf_orders[query_index], ordered_lowers[query_index],
            leaf_offsets, leaf_tuple_ids,
            values, missing,
            atom_attrs[query_index, :atom_count],
            atom_ops[query_index, :atom_count],
            atom_lower[query_index, :atom_count],
            atom_upper[query_index, :atom_count],
            prepared_distances[query_index],
            statistical_caches[cache_indices[query_index]], k,
            buffer_ids[query_index], buffer_distances[query_index],
            registered[query_index], heap_distances[query_index],
            heap_ids[query_index], frontier_states[query_index])


@njit(cache=True, fastmath=False)
def _fused_ranked_events(
        ranked_ids, start, max_events, values, missing, atom_attrs, atom_ops,
        atom_lower, atom_upper, statistical_cache, use_statistical_cache):
    """Compact the next hard-pass/unresolved events from an exact ranking.

    Hard failures never need to cross the Python boundary.  The kernel may
    scan an arbitrarily long run of failures, but returns only a small fixed
    batch of candidates whose eligibility can still affect the answer.  This
    preserves the paper executor's distance order while removing one Python
    call per distance-competitive tuple.
    """
    event_ids = np.empty(max_events, dtype=np.int64)
    event_states = np.empty(max_events, dtype=np.int8)
    event_count = 0
    position = int(start)
    visited = 0
    cached_accepts = 0
    cached_rejects = 0
    cached_abstains = 0
    while position < len(ranked_ids) and event_count < max_events:
        tuple_id = int(ranked_ids[position])
        position += 1
        visited += 1
        state = _direct_state(
            values, missing, atom_attrs, atom_ops, atom_lower, atom_upper,
            tuple_id)
        if state == 0:
            continue
        if state == 2 and use_statistical_cache:
            cached = int(statistical_cache[tuple_id])
            if cached == 2:
                cached_rejects += 1
                continue
            if cached == 1:
                cached_accepts += 1
                state = 1
            elif cached == 3:
                cached_abstains += 1
                continue
        event_ids[event_count] = tuple_id
        event_states[event_count] = state
        event_count += 1
    return (event_ids[:event_count], event_states[:event_count], position,
            visited, cached_accepts, cached_rejects, cached_abstains)


class UnifiedBestFirstExecutor:
    """Exact/epsilon search over a complete active-tree coverage."""

    NODE = 0
    TUPLE = 1

    def __init__(self, embeddings, tree, inference, ann_backend=None, seed_size=64,
                 fused_cpu=True, leaf_frontier=False, distance_provider=None,
                 allow_distance_order_specialization=False,
                 approximate_tail_budget=1):
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.sqnorm = np.einsum("ij,ij->i", self.embeddings, self.embeddings)
        self.tree = tree
        self.inference = inference
        self.ann_backend = ann_backend
        self.seed_size = int(seed_size)
        self.fused_cpu = bool(fused_cpu)
        self.leaf_frontier = bool(leaf_frontier)
        self.distance_provider = distance_provider
        self.allow_distance_order_specialization = bool(
            allow_distance_order_specialization)
        if isinstance(approximate_tail_budget, bool) or int(
                approximate_tail_budget) != approximate_tail_budget or int(
                    approximate_tail_budget) < 0:
            raise ValueError(
                "approximate_tail_budget must be a non-negative integer")
        # Positive-epsilon execution may approximate only this many tail
        # positions.  For the paper's k=10 setting, the default therefore
        # proves the first nine result members exact before terminating.  The
        # ordinary (1+epsilon) distance test still applies to the final slot.
        # epsilon=0 never consults this setting and retains the exact path.
        self.approximate_tail_budget = int(approximate_tail_budget)
        # Scale/serving batches commonly pair many vectors with one immutable
        # predicate list.  Keep the compiled batch schedule with the executor
        # so repeated-predicate steady timing does not rebuild 100 small Python
        # plan dictionaries before every CUDA launch.
        self._candidate_batch_predicate_source = None
        self._candidate_batch_plans = None
        # Replica serving compiles request-local plans from multiple workers.
        # Serialize only shared plan/prefetch-cache access; CUDA candidate
        # generation remains concurrent across devices.
        self._candidate_batch_plan_lock = threading.Lock()
        if tree.search_tuple_ids is None:
            raise ValueError("tree must be materialized over the searchable snapshot")
        expected = np.arange(len(self.embeddings), dtype=np.int64)
        if not np.array_equal(tree.search_tuple_ids, expected):
            raise ValueError("executor currently requires row IDs 0..N-1")
        self._tuple_ids_int64 = expected
        self._tuple_ids_uint64 = expected.astype(np.uint64)
        self._leaf_node_ids = np.asarray([
            int(node_id) for node_id in self.tree.leaf_node_ids
            if self.tree.nodes[int(node_id)].active], dtype=np.int64)
        self._leaf_mbr_lo = np.ascontiguousarray([
            self.tree.nodes[int(node_id)].mbr_lo
            for node_id in self._leaf_node_ids], dtype=np.float32)
        self._leaf_mbr_hi = np.ascontiguousarray([
            self.tree.nodes[int(node_id)].mbr_hi
            for node_id in self._leaf_node_ids], dtype=np.float32)
        leaf_offsets = [0]
        leaf_tuple_ids = []
        for node_id in self._leaf_node_ids:
            rows = np.asarray(
                self.tree.nodes[int(node_id)].tuple_ids, dtype=np.int64)
            leaf_tuple_ids.extend(int(row) for row in rows)
            leaf_offsets.append(len(leaf_tuple_ids))
        self._leaf_offsets = np.asarray(leaf_offsets, dtype=np.int64)
        self._leaf_tuple_ids = np.asarray(leaf_tuple_ids, dtype=np.int64)
        self._searchable_leaf_ids = np.full(
            len(self.embeddings), -1, dtype=np.int64)
        for node_id in self._leaf_node_ids:
            node = self.tree.nodes[int(node_id)]
            self._searchable_leaf_ids[
                np.asarray(node.tuple_ids, dtype=np.int64)] = int(node.leaf_id)
        if np.any(self._searchable_leaf_ids < 0):
            raise AssertionError("materialized tree leaves do not cover N")
        # Searchable-subtree mass is diagnostic instrumentation only.  It is
        # computed once from the materialized snapshot and never participates
        # in pruning decisions.
        self._subtree_searchable_rows = np.zeros(
            len(self.tree.nodes), dtype=np.int64)
        for node in reversed(self.tree.nodes):
            if not node.active:
                continue
            if node.is_leaf:
                rows = len(node.tuple_ids)
            else:
                rows = sum(
                    int(self._subtree_searchable_rows[child_id])
                    for child_id in (node.left, node.right)
                    if child_id is not None)
            self._subtree_searchable_rows[node.node_id] = int(rows)
        if self.tree.root_id is not None and int(
                self._subtree_searchable_rows[self.tree.root_id]
        ) != len(self.embeddings):
            raise AssertionError(
                "materialized tree subtree cardinalities do not cover N")
        if self.fused_cpu and hasattr(self.inference, "compile_hard_predicate"):
            self._warm_fused_cpu()

    def _warm_fused_cpu(self):
        """Compile serving kernels during construction, outside query timing."""
        empty = np.empty(0, dtype=np.int64)
        registered = np.zeros(len(self.embeddings), dtype=np.bool_)
        values = np.empty((1, len(self.embeddings)), dtype=np.float64)
        missing = np.zeros_like(values, dtype=np.bool_)
        attrs = np.asarray([0], dtype=np.int32)
        ops = np.asarray([3], dtype=np.int8)
        limits = np.asarray([0.0], dtype=np.float64)
        ids = np.full(1, -1, dtype=np.int64)
        distances = np.full(1, np.inf, dtype=np.float64)
        query = np.zeros(self.embeddings.shape[1], dtype=np.float32)
        prepared = np.zeros(len(self.embeddings), dtype=np.float32)
        _fused_seed_candidates(
            self.embeddings, self.sqnorm, query, 0.0, empty, registered,
            values, missing, attrs, ops, limits, limits,
            ids, distances, 0, 1, prepared, False)
        _fused_leaf_candidates(
            self.embeddings, self.sqnorm, query, 0.0, empty, registered,
            values, missing, attrs, ops, limits, limits,
            ids, distances, 0, 1, prepared, False,
            np.empty(0, dtype=np.int8), False)
        _fused_ranked_events(
            np.asarray([0], dtype=np.int64), 0, 1,
            values, missing, attrs, ops, limits, limits,
            np.full(len(self.embeddings), -1, dtype=np.int8), True)
        resume_ids = np.full(1, -1, dtype=np.int64)
        resume_distances = np.full(1, np.inf, dtype=np.float64)
        resume_state = np.zeros(_LEAF_STATE_SIZE, dtype=np.int64)
        _fused_leaf_frontier_resume(
            empty, np.empty(0, dtype=np.float64),
            np.asarray([0], dtype=np.int64), empty,
            values, missing, attrs, ops, limits, limits,
            prepared, np.full(len(self.embeddings), -1, dtype=np.int8), 1,
            resume_ids, resume_distances, registered,
            np.empty(len(self.embeddings), dtype=np.float64),
            np.empty(len(self.embeddings), dtype=np.int64), resume_state)
        batch_buffers = np.full((1, 1), -1, dtype=np.int64)
        batch_buffer_distances = np.full((1, 1), np.inf, dtype=np.float64)
        batch_registered = np.zeros(
            (1, len(self.embeddings)), dtype=np.bool_)
        batch_states = np.zeros((1, _LEAF_STATE_SIZE), dtype=np.int64)
        batch_attrs = attrs[None, :]
        batch_ops = ops[None, :]
        batch_limits = limits[None, :]
        batch_prepared = prepared[None, :]
        _leaf_lower_bounds_batch(
            np.empty((0, 1), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((1, 1), dtype=np.float32),
            np.empty((1, 0), dtype=np.float64))
        _precomputed_leaf_min_bounds(
            prepared, np.asarray([0], dtype=np.int64), empty,
            np.empty(0, dtype=np.float64))
        _precomputed_leaf_min_bounds_batch(
            batch_prepared, np.asarray([0], dtype=np.int64), empty,
            np.empty((1, 0), dtype=np.float64))
        _fused_leaf_batch_seed(
            np.empty((1, 0), dtype=np.int64), batch_prepared,
            values, missing, batch_attrs, batch_ops,
            batch_limits, batch_limits, np.asarray([1], dtype=np.int64), 1,
            batch_buffers, batch_buffer_distances, batch_registered,
            batch_states, 0)
        _fused_leaf_frontier_resume_batch(
            np.asarray([True]), np.empty(1, dtype=np.int64),
            np.empty((1, 0), dtype=np.int64),
            np.empty((1, 0), dtype=np.float64),
            np.asarray([0], dtype=np.int64), empty,
            values, missing, batch_attrs, batch_ops,
            batch_limits, batch_limits, np.asarray([1], dtype=np.int64),
            batch_prepared,
            np.full((1, len(self.embeddings)), -1, dtype=np.int8),
            np.asarray([0], dtype=np.int64), 1,
            batch_buffers, batch_buffer_distances, batch_registered,
            np.empty((1, len(self.embeddings)), dtype=np.float64),
            np.empty((1, len(self.embeddings)), dtype=np.int64),
            batch_states)

    def clear_candidate_stream_plan_cache(self):
        """Reset request-derived compiled plans for a cold-cache timing."""
        with self._candidate_batch_plan_lock:
            had_plan = self._candidate_batch_predicate_source is not None
            self._candidate_batch_predicate_source = None
            self._candidate_batch_plans = None
        return {"cleared_compiled_batch_plan": bool(had_plan)}

    @staticmethod
    def _buffer_boundary(buffer, k):
        return float("inf") if len(buffer) < k else float(-buffer[0][0])

    def _protected_result_count(self, k):
        """Number of result positions that positive epsilon keeps exact."""
        return max(int(k) - min(self.approximate_tail_budget, int(k)), 0)

    @staticmethod
    def _heap_rank_boundary(buffer, protected_count):
        """Distance of the protected prefix's last item in a heap buffer."""
        if int(protected_count) <= 0:
            return float("-inf")
        ranked = sorted(
            ((-float(distance), -int(neg_id))
             for distance, neg_id, _ in buffer),
            key=lambda item: (item[0], item[1]))
        return float(ranked[int(protected_count) - 1][0])

    @staticmethod
    def _record_approximate_stop(counters, *, frontier, boundary,
                                 protected_boundary, protected_count):
        counters["approximate_stop_triggered"] = 1
        counters["approximate_stop_frontier_sq"] = float(frontier)
        counters["approximate_stop_kth_boundary_sq"] = float(boundary)
        counters["approximate_stop_protected_boundary_sq"] = float(
            protected_boundary)
        counters["approximate_protected_results"] = int(protected_count)

    @staticmethod
    def _can_affect(buffer, k, distance, tuple_id):
        if len(buffer) < k:
            return True
        worst_distance = -buffer[0][0]
        worst_id = -buffer[0][1]
        return (distance, tuple_id) < (worst_distance, worst_id)

    @staticmethod
    def _update_buffer(buffer, k, distance, tuple_id):
        item = (-float(distance), -int(tuple_id), int(tuple_id))
        if len(buffer) < k:
            heapq.heappush(buffer, item)
        elif (distance, tuple_id) < (-buffer[0][0], -buffer[0][1]):
            heapq.heapreplace(buffer, item)

    def _distance_sq(self, query, query_sqnorm, tuple_id):
        value = (self.sqnorm[tuple_id] -
                 2.0 * float(self.embeddings[tuple_id] @ query) + query_sqnorm)
        return max(float(value), 0.0)

    def _precomputed_subtree_lower_bounds(self, distances):
        """Return bounds in the same numeric domain as a frozen distance row.

        Projected MBR bounds are valid for distances computed from the same
        query/vector arithmetic.  They are not safe when a caller supplies a
        separately prepared float32 row: cancellation can round several exact
        distances to zero while the projected bound remains positive.  An
        exact subtree minimum keeps pruning and the deterministic tie rule in
        one domain without recomputing any vector distance.
        """
        bounds = np.full(len(self.tree.nodes), np.inf, dtype=np.float64)
        leaf_bounds = np.empty(len(self._leaf_node_ids), dtype=np.float64)
        _precomputed_leaf_min_bounds(
            distances, self._leaf_offsets, self._leaf_tuple_ids, leaf_bounds)
        bounds[self._leaf_node_ids] = leaf_bounds
        for node in reversed(self.tree.nodes):
            if not node.active:
                continue
            node_id = int(node.node_id)
            if node.is_leaf:
                # Active materialized leaves are initialized by the native
                # row scan above.  Keeping that O(N) pass in Numba avoids a
                # Python reduction per leaf on larger snapshots.
                if np.isnan(bounds[node_id]):
                    bounds[node_id] = -np.inf
            else:
                children = [
                    int(child_id) for child_id in (node.left, node.right)
                    if child_id is not None and
                    self.tree.nodes[int(child_id)].active
                ]
                if children:
                    bounds[node_id] = min(bounds[child_id]
                                          for child_id in children)
        return bounds

    def _search_reference_order(
            self, query, k, predicate, strategy, precomputed_distances,
            use_precomputed_distances, inference_before):
        """Run a fixed-policy oracle order through one physical executor.

        Both oracle strategies use the same hard decision, statistical
        inference, cache, distance row, and ``(distance, id)`` tie rule.  Only
        the point at which eligibility is requested differs.
        """
        if strategy not in {"imputation_first", "distance_first"}:
            raise ValueError(f"unknown reference execution strategy: {strategy}")
        n_rows = len(self.embeddings)

        def prepare_distances():
            if use_precomputed_distances:
                return precomputed_distances
            if self.distance_provider is not None:
                return np.ascontiguousarray(
                    self.distance_provider.prepare(query), dtype=np.float32)
            differences = self.embeddings - query[None, :]
            return np.einsum(
                "ij,ij->i", differences, differences).astype(np.float32)

        distances = None
        ranked = None
        chosen = []
        visited = 0
        resolution_requests = 0

        if strategy == "imputation_first":
            eligible = np.zeros(n_rows, dtype=np.bool_)
            for tuple_id in self._tuple_ids_int64:
                tuple_id = int(tuple_id)
                visited += 1
                state, _ = self.inference.hard_decision(tuple_id, predicate)
                if state == "pass":
                    eligible[tuple_id] = True
                elif state == "unresolved":
                    resolution_requests += 1
                    eligible[tuple_id] = self.inference.evaluate(
                        tuple_id, predicate,
                        assume_unresolved=True).eligible
            distances = prepare_distances()
            ranked = np.lexsort((self._tuple_ids_int64, distances))
            chosen = [
                int(tuple_id) for tuple_id in ranked
                if eligible[int(tuple_id)]
            ][:k]
        else:
            distances = prepare_distances()
            ranked = np.lexsort((self._tuple_ids_int64, distances))
            for tuple_id in ranked:
                tuple_id = int(tuple_id)
                visited += 1
                state, _ = self.inference.hard_decision(tuple_id, predicate)
                eligible = state == "pass"
                if state == "unresolved":
                    resolution_requests += 1
                    eligible = self.inference.evaluate(
                        tuple_id, predicate,
                        assume_unresolved=True).eligible
                if eligible:
                    chosen.append(tuple_id)
                    if len(chosen) == k:
                        break

        inference_after = getattr(self.inference, "counters", {})
        cache_hits = int(
            inference_after.get("inference_cache_hits", 0) -
            inference_before.get("inference_cache_hits", 0))
        cache_misses = int(
            inference_after.get("inference_cache_misses", 0) -
            inference_before.get("inference_cache_misses", 0))
        if (cache_hits < 0 or cache_misses < 0 or
                cache_hits > resolution_requests):
            raise AssertionError(
                "reference resolution cache accounting is inconsistent")
        resolution_computations = int(resolution_requests) - int(cache_hits)
        ids = np.asarray(chosen, dtype=np.int64)
        returned_distances = np.asarray(
            distances[ids], dtype=np.float64)
        counters = {
            "ann_seed_candidates": 0,
            "node_queue_pops": 0,
            "tuple_queue_pops": int(resolution_requests),
            "expanded_nodes": 0,
            "expanded_leaves": 0,
            "node_bound_evaluations": 0,
            "pruned_nodes": 0,
            "visited_tuples": int(visited),
            "tuple_queue_insertions": int(resolution_requests),
            "exact_distance_evaluations": int(n_rows),
            "hard_decisions": int(visited),
            "lazy_inferences": int(resolution_requests),
            "inferences_skipped_by_boundary": 0,
            "registered_tuples": int(visited),
            "buffer_update_attempts": int(len(ids)),
            "max_queue_size": 0,
            "ann_seed_precomputed_reuse": 0,
            "precomputed_distance_rows": (
                int(n_rows) if use_precomputed_distances else 0),
            "physical_distance_evaluations": int(n_rows),
            "unique_physical_distance_evaluations": int(n_rows),
            "statistical_resolution_requests": int(resolution_requests),
            "statistical_resolution_computations": int(
                resolution_computations),
            "statistical_resolution_cache_hits": int(cache_hits),
            "statistical_resolution_cache_misses": int(cache_misses),
            "candidate_rows_not_eligibility_checked": int(n_rows - visited),
            "queued_unresolved_resolutions_avoided": 0,
        }
        return ids, returned_distances, counters

    def _ann_seed_ids(self, query, k, precomputed_distances,
                      use_precomputed_distances):
        """Reuse a resident exact-distance row for ExactBackend seeding."""
        if (use_precomputed_distances and
                isinstance(self.ann_backend, ExactBackend)):
            count = min(int(k), len(precomputed_distances))
            if count <= 0:
                return np.empty(0, dtype=np.int64), True
            candidates = np.argpartition(
                precomputed_distances, count - 1)[:count]
            order = np.lexsort((
                candidates, precomputed_distances[candidates]))
            ordered = np.ascontiguousarray(
                candidates[order], dtype=np.int64)
            self._last_ann_seed_distances = np.ascontiguousarray(
                precomputed_distances[ordered], dtype=np.float32)
            return ordered, True
        seed_ids, seed_distances = self.ann_backend.search(
            query, min(int(k), len(self.embeddings)))
        self._last_ann_seed_distances = np.ascontiguousarray(
            seed_distances, dtype=np.float32)
        return np.ascontiguousarray(seed_ids, dtype=np.int64), False

    @staticmethod
    def _cache_code(result):
        if result.eligible:
            return 1
        return 2 if result.state == "statistical" else 3

    def search_precomputed_batch(
            self, query_embeddings, ks, predicates, precomputed_distances):
        """Serve one exact unified-tree cohort on the native worker pool.

        Statistical decisions remain in the indexed Python policy.  Native
        frontier transitions run in parallel and synchronize only when a
        query discovers one previously unseen immutable inference result.
        Duplicate requests share the exact-predicate dense cache.
        """
        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        distances = np.ascontiguousarray(
            precomputed_distances, dtype=np.float32)
        predicates = list(predicates)
        ks = ([int(ks)] * len(queries) if np.isscalar(ks)
              else [int(value) for value in ks])
        query_count = len(queries)
        n_rows = len(self.embeddings)
        if queries.ndim != 2 or queries.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query dimensions do not match embeddings")
        if distances.shape != (query_count, n_rows):
            raise ValueError("precomputed distance rows do not align")
        if len(predicates) != query_count or len(ks) != query_count:
            raise ValueError("queries, ks, and predicates must align")
        if not query_count:
            return []
        k = min(ks[0], n_rows)
        if k <= 0 or any(min(value, n_rows) != k for value in ks):
            raise ValueError("native query batches require one positive k")
        if not (self.fused_cpu and self.leaf_frontier and
                isinstance(self.ann_backend, ExactBackend)):
            raise RuntimeError(
                "native batch requires the exact unified leaf-tree path")

        plans = []
        cache_arrays = []
        cache_plan_indices = []
        cache_by_key = {}
        cache_indices = np.empty(query_count, dtype=np.int64)
        for query_index, predicate in enumerate(predicates):
            plan = self.inference.compile_hard_predicate(predicate)
            if plan is None:
                raise RuntimeError(
                    "native batch requires a compiled hard predicate")
            plans.append(plan)
            self.inference.begin_query(predicate)
            dense_cache = self.inference.statistical_eligibility_cache()
            if dense_cache is None:
                raise RuntimeError(
                    "native batch requires the immutable predicate cache")
            key = self.inference._predicate_key(predicate.atoms)
            cache_slot = cache_by_key.get(key)
            if cache_slot is None:
                cache_slot = len(cache_arrays)
                cache_by_key[key] = cache_slot
                cache_arrays.append(dense_cache)
                cache_plan_indices.append(query_index)
            cache_indices[query_index] = cache_slot
        statistical_caches = np.ascontiguousarray(
            np.stack(cache_arrays), dtype=np.int8)

        max_atoms = max(len(plan["attrs"]) for plan in plans)
        atom_attrs = np.zeros((query_count, max_atoms), dtype=np.int32)
        atom_ops = np.zeros((query_count, max_atoms), dtype=np.int8)
        atom_lower = np.zeros((query_count, max_atoms), dtype=np.float64)
        atom_upper = np.zeros((query_count, max_atoms), dtype=np.float64)
        atom_counts = np.empty(query_count, dtype=np.int64)
        for query_index, plan in enumerate(plans):
            count = len(plan["attrs"])
            atom_counts[query_index] = count
            atom_attrs[query_index, :count] = plan["attrs"]
            atom_ops[query_index, :count] = plan["ops"]
            atom_lower[query_index, :count] = plan["lower"]
            atom_upper[query_index, :count] = plan["upper"]
        values = plans[0]["values"]
        missing = plans[0]["missing"]
        predicate_residual_codes = np.zeros(
            (len(cache_arrays), n_rows), dtype=np.int64)
        for cache_slot, query_index in enumerate(cache_plan_indices):
            plan = plans[query_index]
            hard_fail = np.zeros(n_rows, dtype=np.bool_)
            codes = predicate_residual_codes[cache_slot]
            for atom_index in range(len(plan["attrs"])):
                attr = int(plan["attrs"][atom_index])
                item_missing = plan["missing"][attr]
                item_values = plan["values"][attr]
                op = int(plan["ops"][atom_index])
                if op == 0:
                    satisfied = item_values <= plan["lower"][atom_index]
                elif op == 1:
                    satisfied = item_values >= plan["lower"][atom_index]
                elif op == 2:
                    satisfied = (
                        (item_values >= plan["lower"][atom_index]) &
                        (item_values <= plan["upper"][atom_index]))
                else:
                    satisfied = item_values == plan["lower"][atom_index]
                hard_fail |= (~item_missing) & (~satisfied)
                codes[item_missing] |= (
                    np.int64(1) << np.int64(atom_index))
            codes[hard_fail] = -1

        leaf_count = len(self._leaf_node_ids)
        leaf_lowers = np.empty(
            (query_count, leaf_count), dtype=np.float64)
        _precomputed_leaf_min_bounds_batch(
            distances, self._leaf_offsets, self._leaf_tuple_ids,
            leaf_lowers)
        leaf_orders = np.empty(
            (query_count, leaf_count), dtype=np.int64)
        for query_index in range(query_count):
            leaf_orders[query_index] = np.lexsort((
                self._leaf_node_ids, leaf_lowers[query_index]))
        ordered_lowers = np.ascontiguousarray(
            np.take_along_axis(leaf_lowers, leaf_orders, axis=1),
            dtype=np.float64)

        seed_count = min(self.seed_size, n_rows)
        if seed_count:
            candidates = np.argpartition(
                distances, seed_count - 1, axis=1)[:, :seed_count]
            candidate_distances = np.take_along_axis(
                distances, candidates, axis=1)
            seed_orders = np.empty_like(candidates)
            for query_index in range(query_count):
                seed_orders[query_index] = np.lexsort((
                    candidates[query_index],
                    candidate_distances[query_index]))
            seed_ids = np.ascontiguousarray(
                np.take_along_axis(candidates, seed_orders, axis=1),
                dtype=np.int64)
        else:
            seed_ids = np.empty((query_count, 0), dtype=np.int64)

        buffer_ids = np.full((query_count, k), -1, dtype=np.int64)
        buffer_distances = np.full(
            (query_count, k), np.inf, dtype=np.float64)
        registered = np.zeros((query_count, n_rows), dtype=np.bool_)
        heap_distances = np.empty(
            (query_count, n_rows), dtype=np.float64)
        heap_ids = np.empty((query_count, n_rows), dtype=np.int64)
        frontier_states = np.zeros(
            (query_count, _LEAF_STATE_SIZE), dtype=np.int64)
        # Reference-path accounting only.  Keep this explicit rather than
        # relying on process RSS, which is distorted by NumPy/CUDA allocator
        # retention and NUMA placement.  These arrays are the live, dense host
        # state owned by this batch invocation; the input distance matrix is
        # included because GPU providers materialize it on host before entry.
        explicit_host_state_bytes = int(sum(array.nbytes for array in (
            distances, statistical_caches, predicate_residual_codes,
            leaf_lowers, leaf_orders, ordered_lowers, seed_ids,
            buffer_ids, buffer_distances, registered, heap_distances,
            heap_ids, frontier_states,
        )))
        precomputed_distance_host_bytes = int(distances.nbytes)
        predicate_residual_code_bytes = int(
            predicate_residual_codes.nbytes)
        _fused_leaf_batch_seed(
            seed_ids, distances, values, missing,
            atom_attrs, atom_ops, atom_lower, atom_upper, atom_counts, k,
            buffer_ids, buffer_distances, registered, frontier_states,
            leaf_count)

        active = np.ones(query_count, dtype=np.bool_)
        needed = np.full(query_count, -1, dtype=np.int64)
        resume_counts = np.zeros(query_count, dtype=np.int64)
        lazy_inferences = np.zeros(query_count, dtype=np.int64)
        new_accepts = np.zeros(query_count, dtype=np.int64)
        new_rejects = np.zeros(query_count, dtype=np.int64)
        new_abstains = np.zeros(query_count, dtype=np.int64)
        propagated_rows = np.zeros(query_count, dtype=np.int64)
        rounds = 0
        while np.any(active):
            rounds += 1
            was_active = active.copy()
            resume_counts[was_active] += 1
            _fused_leaf_frontier_resume_batch(
                active, needed, leaf_orders, ordered_lowers,
                self._leaf_offsets, self._leaf_tuple_ids,
                values, missing,
                atom_attrs, atom_ops, atom_lower, atom_upper, atom_counts,
                distances, statistical_caches, cache_indices, k,
                buffer_ids, buffer_distances, registered,
                heap_distances, heap_ids, frontier_states)
            active[was_active & (needed < 0)] = False
            for query_index in np.flatnonzero(was_active & (needed >= 0)):
                tuple_id = int(needed[query_index])
                cache_slot = int(cache_indices[query_index])
                if statistical_caches[cache_slot, tuple_id] >= 0:
                    continue
                evaluated = self.inference.evaluate(
                    tuple_id, predicates[query_index],
                    assume_unresolved=True)
                code = self._cache_code(evaluated)
                # Keep the native batch cache exact at tuple granularity.
                # The nominal cross-tuple equivalence propagation did not
                # preserve the unfused oracle on three repeated AppReviews
                # queries under the location-aware policy.  Until that wider
                # equivalence relation is separately certified, cache only
                # the evaluated immutable tuple.  Repeated predicates still
                # reuse this exact tuple decision.
                equivalent_rows = np.asarray([tuple_id], dtype=np.int64)
                previously_empty = (
                    statistical_caches[cache_slot, equivalent_rows] < 0)
                statistical_caches[
                    cache_slot, equivalent_rows[previously_empty]] = code
                propagated_rows[query_index] += max(
                    int(np.count_nonzero(previously_empty)) - 1, 0)
                lazy_inferences[query_index] += 1
                if code == 1:
                    new_accepts[query_index] += 1
                elif code == 2:
                    new_rejects[query_index] += 1
                else:
                    new_abstains[query_index] += 1
            if rounds > n_rows + 1:
                raise RuntimeError(
                    "native query batch did not converge after cache fill")

        for cache_slot, dense_cache in enumerate(cache_arrays):
            dense_cache[:] = statistical_caches[cache_slot]

        outputs = []
        for query_index in range(query_count):
            state = frontier_states[query_index]
            count = int(state[_LEAF_BUFFER_COUNT])
            cache_accepts = max(
                int(state[_LEAF_CACHE_ACCEPTS]) -
                int(new_accepts[query_index]), 0)
            cache_rejects = max(
                int(state[_LEAF_CACHE_REJECTS]) -
                int(new_rejects[query_index]), 0)
            cache_abstains = max(
                int(state[_LEAF_CACHE_ABSTAINS]) -
                int(new_abstains[query_index]), 0)
            counters = {
                "ann_seed_candidates": int(seed_count),
                "ann_seed_precomputed_reuse": 1,
                "node_queue_pops": int(state[_LEAF_NODE_POPS]),
                "tuple_queue_pops": int(state[_LEAF_TUPLE_POPS]),
                "expanded_nodes": 0,
                "expanded_leaves": int(state[_LEAF_NODE_POPS]),
                "node_bound_evaluations": int(leaf_count),
                "pruned_nodes": 0,
                "visited_tuples": int(state[_LEAF_VISITED]),
                "tuple_queue_insertions": int(
                    state[_LEAF_TUPLE_INSERTIONS]),
                "exact_distance_evaluations": int(state[_LEAF_EXACT]),
                "hard_decisions": int(state[_LEAF_HARD]),
                "lazy_inferences": int(lazy_inferences[query_index]),
                "inferences_skipped_by_boundary": int(
                    state[_LEAF_BOUNDARY_SKIPS]),
                "registered_tuples": int(state[_LEAF_REGISTERED]),
                "buffer_update_attempts": int(state[_LEAF_UPDATES]),
                "max_queue_size": int(state[_LEAF_MAX_QUEUE]),
                "statistical_cache_hits": int(
                    cache_accepts + cache_rejects + cache_abstains),
                "statistical_cache_accepts": int(cache_accepts),
                "statistical_cache_rejects": int(cache_rejects),
                "statistical_cache_abstains": int(cache_abstains),
                "precomputed_distance_rows": int(n_rows),
                "physical_distance_evaluations": int(n_rows),
                "fused_leaf_frontier_native": 1,
                "native_scheduler_replays": 0,
                "native_scheduler_resumes": int(
                    resume_counts[query_index]),
                "native_query_batch_scheduler": 1,
                "predicate_cache_propagated_rows": int(
                    propagated_rows[query_index]),
                "precomputed_distance_host_bytes_per_query": int(
                    precomputed_distance_host_bytes // query_count),
                "batch_explicit_host_state_bytes_total": int(
                    explicit_host_state_bytes),
                "batch_explicit_host_state_bytes_per_query": int(
                    explicit_host_state_bytes // query_count),
                "predicate_residual_code_bytes_total": int(
                    predicate_residual_code_bytes),
                "precomputed_leaf_min_bound_rows": int(n_rows),
            }
            if hasattr(self.inference, "record_executor_cache_hits"):
                self.inference.record_executor_cache_hits(
                    cache_accepts, cache_rejects, cache_abstains)
            outputs.append((
                buffer_ids[query_index, :count].copy(),
                buffer_distances[query_index, :count].copy(), counters))
        return outputs

    def search_candidate_stream_batch(
            self, query_embeddings, ks, predicates, candidate_provider, *,
            initial_page_size=4096, max_page_size=65536,
            max_candidate_events=None, fallback_to_reference=False):
        """Serve exact GPU-resident distance-order candidate pages.

        This method is an opt-in Phase-A specialization.  It preserves the
        frozen predicate/inference semantics while replacing the complete
        distance D2H and dense leaf scheduler with compact exact pages.  The
        Existing ``search`` remains the correctness reference.  Serving is
        fail-closed by default: reference replay is available only when an
        explicit diagnostic/ablation call sets ``fallback_to_reference=True``.

        Candidate ordering is supplied by ``candidate_provider`` and must be
        exact ``(distance, row_id)`` order after deterministic hard failures
        are removed.  Only unresolved candidates enter indexed inference.
        """
        logical_call_started = perf_counter()
        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        predicates = list(predicates)
        ks = ([int(ks)] * len(queries) if np.isscalar(ks)
              else [int(value) for value in ks])
        query_count = len(queries)
        n_rows = len(self.embeddings)
        if queries.ndim != 2 or queries.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query dimensions do not match embeddings")
        if len(predicates) != query_count or len(ks) != query_count:
            raise ValueError("queries, ks, and predicates must align")
        if not query_count:
            return []
        if any(value <= 0 for value in ks):
            raise ValueError("k must be positive")
        ks = [min(value, n_rows) for value in ks]
        initial_page_size = int(initial_page_size)
        max_page_size = int(max_page_size)
        if initial_page_size <= 0 or max_page_size < initial_page_size:
            raise ValueError("candidate page sizes must be positive and ordered")
        if max_candidate_events is None:
            max_candidate_events = n_rows
        max_candidate_events = int(max_candidate_events)
        if max_candidate_events <= 0:
            raise ValueError("max_candidate_events must be positive")

        def reference_batch(reason):
            if not fallback_to_reference:
                raise RuntimeError(
                    f"candidate stream cannot serve batch: {reason}")
            reference = []
            for query, k, predicate in zip(queries, ks, predicates):
                ids, distances, counters = self.search(
                    query, k, predicate, epsilon=0.0)
                counters = dict(counters)
                counters.update({
                    "gpu_candidate_stream": 0,
                    "candidate_stream_fallback": 1,
                    "candidate_stream_fallback_reason": str(reason),
                    "candidate_stream_reference_fallback_enabled": 1,
                })
                reference.append((ids, distances, counters))
            return reference

        if not self.fused_cpu or not hasattr(
                self.inference, "compile_hard_predicate"):
            return reference_batch("hard_predicate_compiler_unavailable")
        predicate_source = tuple(predicates)
        with self._candidate_batch_plan_lock:
            same_predicates = (
                self._candidate_batch_predicate_source is not None and
                len(self._candidate_batch_predicate_source) == len(
                    predicate_source) and
                all(left is right for left, right in zip(
                    self._candidate_batch_predicate_source,
                    predicate_source)))
            if same_predicates:
                plans = self._candidate_batch_plans
                logical_direct_checks_ms = 0.0
                logical_support_inference_ms = 0.0
            else:
                hard_plan_started = perf_counter()
                plans = [
                    self.inference.compile_hard_predicate(predicate)
                    for predicate in predicates]
                logical_direct_checks_ms = (
                    perf_counter() - hard_plan_started) * 1000.0
                if any(plan is None for plan in plans):
                    return reference_batch(
                        "unsupported_compiled_predicate")
                support_inference_started = perf_counter()
                for plan_index, predicate in enumerate(predicates):
                    posting = None
                    prefetched = None
                    strategy = getattr(
                        self, "candidate_prefetch_strategy", "group_first")
                    if strategy == "posting_first" and hasattr(
                            self.inference,
                            "compile_candidate_eligibility_posting"):
                        posting = self.inference.\
                            compile_candidate_eligibility_posting(
                                predicate, materialize=False)
                    if posting is None and hasattr(
                            self.inference,
                            "compile_candidate_group_prefetch"):
                        prefetched = self.inference.\
                            compile_candidate_group_prefetch(predicate)
                    if (posting is None and prefetched is None and hasattr(
                            self.inference,
                            "compile_candidate_eligibility_posting")):
                        posting = self.inference.\
                            compile_candidate_eligibility_posting(
                                predicate, materialize=False)
                    if posting is not None:
                        plans[plan_index] = {
                            **plans[plan_index],
                            "precomputed_eligible_ids": posting,
                        }
                    elif prefetched is not None:
                        plans[plan_index] = {
                            **plans[plan_index],
                            "statistical_group_ids": prefetched["group_ids"],
                            "statistical_group_decisions":
                            prefetched["decisions"],
                        }
                logical_support_inference_ms = (
                    perf_counter() - support_inference_started) * 1000.0
                self._candidate_batch_predicate_source = predicate_source
                self._candidate_batch_plans = plans

        # Candidate-local caches replace the native batch's dense QxN cache.
        # Equal exact predicates share only decisions actually encountered in
        # compact pages.  IndexedJointInference retains its own bounded,
        # immutable serving-result cache for reusable statistical signatures.
        compact_cache_by_key = {}
        compact_caches = []
        for predicate in predicates:
            key = self.inference._predicate_key(predicate.atoms)
            compact_caches.append(compact_cache_by_key.setdefault(key, {}))
        logical_executor_setup_ms = max(
            (perf_counter() - logical_call_started) * 1000.0 -
            logical_direct_checks_ms - logical_support_inference_ms,
            0.0)

        handle = None
        try:
            try:
                handle = None
                if query_count == 1 and hasattr(
                        candidate_provider, "begin_batch_topk"):
                    handle = candidate_provider.begin_batch_topk(
                        queries, plans, topk=max(ks))
                if handle is None:
                    handle = candidate_provider.begin_batch(queries, plans)
            except (RuntimeError, ValueError) as error:
                return reference_batch(
                    f"provider_begin:{type(error).__name__}:{error}")

            # The CUDA provider has already produced the complete exact
            # batch-one page when immutable statistical groups resolve every
            # predicate state.  Consume that page directly instead of
            # allocating the generic multi-query scheduler's arrays, Python
            # lists, caches, and page loop.  Distance ranking is still
            # executed for every request; only fixed executor bookkeeping is
            # specialized away.
            prefetched_page = getattr(handle, "prefetched_page", None)
            if (
                query_count == 1
                and prefetched_page is not None
                and handle.telemetry.get("statistical_group_prefetch", False)
            ):
                consume_started = perf_counter()
                count = int(prefetched_page.counts[0])
                take = min(count, ks[0])
                ids = np.ascontiguousarray(
                    prefetched_page.ids[0, :take], dtype=np.int64)
                distances = np.ascontiguousarray(
                    prefetched_page.distances[0, :take], dtype=np.float64)
                cpu_consume = (perf_counter() - consume_started) * 1000.0
                telemetry = handle.telemetry
                physical_rows = int(telemetry.get(
                    "batch1_distance_rows_evaluated", n_rows))
                compact_page_bytes = int(
                    prefetched_page.ids.nbytes +
                    prefetched_page.distances.nbytes +
                    prefetched_page.states.nbytes)
                counters = {
                    "ann_seed_candidates": 0,
                    "ann_seed_precomputed_reuse": 0,
                    "node_queue_pops": 0,
                    "tuple_queue_pops": 0,
                    "expanded_nodes": 0,
                    "expanded_leaves": 0,
                    "node_bound_evaluations": 0,
                    "pruned_nodes": 0,
                    "visited_tuples": int(take),
                    "tuple_queue_insertions": 0,
                    "exact_distance_evaluations": physical_rows,
                    "hard_decisions": int(n_rows),
                    "lazy_inferences": 0,
                    "inferences_skipped_by_boundary": int(n_rows - take),
                    "registered_tuples": int(take),
                    "buffer_update_attempts": int(take),
                    "max_queue_size": 0,
                    "precomputed_distance_rows": 0,
                    "physical_distance_evaluations": physical_rows,
                    "distance_order_specialization": 1,
                    "approximate_tree_probe": 0,
                    "statistical_cache_hits": 0,
                    "statistical_cache_accepts": 0,
                    "statistical_cache_rejects": 0,
                    "statistical_cache_abstains": 0,
                    "gpu_candidate_stream": 1,
                    "candidate_stream_phase_a": 1,
                    "candidate_stream_reference_fallback_enabled": int(
                        fallback_to_reference),
                    "candidate_stream_fallback": 0,
                    "candidate_stream_fallback_reason": "",
                    "candidate_stream_page_rounds": int(count > 0),
                    "candidate_stream_inference_event_block": 4096,
                    "candidate_stream_candidates_transferred": int(count),
                    "candidate_stream_candidate_events": int(take),
                    "candidate_stream_unresolved_events": 0,
                    "candidate_stream_inference_events": 0,
                    "candidate_stream_overclassified_unresolved": 0,
                    "candidate_stream_statistical_inference_ms": 0.0,
                    "candidate_stream_cpu_consume_ms": float(cpu_consume),
                    "candidate_stream_compact_d2h_bytes": int(13 * count),
                    "precomputed_distance_host_bytes_per_query": 0,
                    "candidate_stream_peak_compact_host_page_bytes_batch":
                        compact_page_bytes,
                    "candidate_stream_batch_distance_compute_ms": float(
                        telemetry["distance_compute_ms_total"]),
                    "candidate_stream_batch_hard_predicate_ms": float(
                        telemetry["hard_predicate_ms_total"]),
                    "candidate_stream_batch_candidate_select_ms": float(
                        telemetry["candidate_select_ms_total"]),
                    "candidate_stream_batch_compact_d2h_ms": float(
                        telemetry["compact_d2h_ms_total"]),
                    "candidate_stream_batch_compact_d2h_bytes": int(
                        telemetry["compact_d2h_bytes_total"]),
                    "candidate_stream_batch_cpu_merge_ms": 0.0,
                    "candidate_stream_multi_gpu": 0,
                    "candidate_stream_batch_full_fallback_d2h_bytes": int(
                        telemetry["full_distance_fallback_bytes_total"]),
                    "candidate_stream_peak_gpu_allocated_bytes": int(
                        telemetry["peak_gpu_allocated_bytes"]),
                    "candidate_stream_peak_provider_host_bytes": int(
                        telemetry.get("peak_provider_host_bytes", 0)),
                    "candidate_stream_batch_tree_probe_plan_ms": 0.0,
                    "candidate_stream_batch1_fused_wall_ms": float(
                        telemetry.get("batch1_fused_wall_ms", 0.0)),
                    "candidate_stream_batch1_fused_device_ms": float(
                        telemetry.get("batch1_fused_device_ms_total", 0.0)),
                    "candidate_stream_batch1_device_d2h_wait_ms": float(
                        telemetry.get(
                            "batch1_fused_device_and_d2h_wait_ms_total", 0.0)),
                    "candidate_stream_tree_probe_count": 0,
                    "candidate_stream_tree_probe_rows_total": 0,
                    "candidate_stream_batch1_direct_consume": 1,
                    "candidate_stream_batch1_filtered_distance": int(
                        telemetry.get("batch1_filtered_distance_path", False)),
                    "candidate_stream_batch1_eligible_rows": int(
                        telemetry.get("batch1_eligible_rows", n_rows)),
                    "candidate_stream_batch1_numba_partitions": int(
                        telemetry.get("batch1_numba_partitions", 0)),
                    "candidate_stream_batch1_refine_width": int(
                        telemetry.get("batch1_refine_width", 0)),
                    "candidate_stream_batch1_projection_dim": int(
                        telemetry.get("batch1_projection_dim", 0)),
                    "candidate_stream_batch1_quantized_screen": int(
                        telemetry.get("batch1_quantized_screen", False)),
                    "candidate_stream_batch1_quantized_dtype": str(
                        telemetry.get("batch1_quantized_dtype", "N/A")),
                    "candidate_stream_batch1_eligibility_cache_bytes": int(
                        telemetry.get(
                            "batch1_eligibility_cache_bytes", 0)),
                    "candidate_stream_batch1_eligibility_cache_entries": int(
                        telemetry.get(
                            "batch1_eligibility_cache_entries", 0)),
                    "candidate_stream_batch1_eligibility_cache_hits": int(
                        telemetry.get("batch1_eligibility_cache_hits", 0)),
                    "candidate_stream_batch1_eligibility_cache_misses": int(
                        telemetry.get("batch1_eligibility_cache_misses", 0)),
                    "candidate_stream_batch1_eligibility_cache_hit_rate": float(
                        telemetry.get(
                            "batch1_eligibility_cache_hit_rate", 0.0)),
                    "candidate_stream_batch1_eligibility_cache_max_entries": int(
                        telemetry.get(
                            "batch1_eligibility_cache_max_entries", 0)),
                    "candidate_stream_batch1_eligibility_cache_max_bytes": int(
                        telemetry.get(
                            "batch1_eligibility_cache_max_bytes", 0)),
                    # Logical E2-1 CPU breakdown.  The controlled CPU path
                    # uses a fused int8 eligible-row shortlist rather than an
                    # external ANN call.  Report that real candidate-seeding
                    # work explicitly, then keep FP32 exact refinement as the
                    # distance component.  A cache miss materializes the
                    # eligibility posting through the direct-check kernel;
                    # a hit is charged to cache lookup.
                    "candidate_stream_logical_ann_seeding_ms": float(
                        telemetry.get("batch1_kernel_ms", 0.0)),
                    "candidate_stream_logical_direct_checks_ms": float(
                        logical_direct_checks_ms +
                        (telemetry.get("batch1_eligibility_cache_ms", 0.0)
                         if not telemetry.get(
                             "batch1_eligibility_cache_hit", False)
                         else 0.0)),
                    "candidate_stream_logical_cache_lookup_ms": float(
                        telemetry.get("batch1_eligibility_cache_ms", 0.0)
                        if telemetry.get(
                            "batch1_eligibility_cache_hit", False)
                        else 0.0),
                    "candidate_stream_logical_exact_distance_ms": float(
                        telemetry.get("batch1_refine_ms", 0.0)),
                    "candidate_stream_logical_support_inference_ms": float(
                        logical_support_inference_ms),
                    "candidate_stream_logical_executor_setup_ms": float(
                        logical_executor_setup_ms),
                }
                if hasattr(self.inference, "record_executor_cache_hits"):
                    self.inference.record_executor_cache_hits(0, 0, 0)
                return [(ids, distances, counters)]

            output_ids = [[] for _ in range(query_count)]
            output_distances = [[] for _ in range(query_count)]
            active = np.asarray([
                handle.available_counts[index] > 0 and ks[index] > 0
                for index in range(query_count)], dtype=np.bool_)
            page_rounds = np.zeros(query_count, dtype=np.int64)
            transferred_candidates = np.zeros(query_count, dtype=np.int64)
            candidate_events = np.zeros(query_count, dtype=np.int64)
            unresolved_events = np.zeros(query_count, dtype=np.int64)
            candidate_inference_events = np.zeros(
                query_count, dtype=np.int64)
            overclassified_unresolved = np.zeros(
                query_count, dtype=np.int64)
            lazy_inferences = np.zeros(query_count, dtype=np.int64)
            cache_accepts = np.zeros(query_count, dtype=np.int64)
            cache_rejects = np.zeros(query_count, dtype=np.int64)
            cache_abstains = np.zeros(query_count, dtype=np.int64)
            inference_support_ms = np.zeros(query_count, dtype=np.float64)
            inference_cache_ms = np.zeros(query_count, dtype=np.float64)
            fallback_queries = np.zeros(query_count, dtype=np.bool_)
            inference_ms = np.zeros(query_count, dtype=np.float64)
            cpu_consume_ms = np.zeros(query_count, dtype=np.float64)
            peak_compact_host_page_bytes = 0
            page_size = initial_page_size
            if handle.telemetry.get("statistical_group_prefetch", False):
                page_size = max(ks)

            while np.any(active):
                page = candidate_provider.next_page(
                    handle, active, page_size=page_size)
                if not len(page.query_indices):
                    break
                peak_compact_host_page_bytes = max(
                    peak_compact_host_page_bytes,
                    int(page.ids.nbytes + page.distances.nbytes +
                        page.states.nbytes))
                for page_row, query_index_value in enumerate(
                        page.query_indices):
                    query_index = int(query_index_value)
                    page_rounds[query_index] += 1
                    count = int(page.counts[page_row])
                    transferred_candidates[query_index] += count
                    consume_started = perf_counter()
                    if handle.telemetry.get(
                            "statistical_group_prefetch", False):
                        # Every unresolved row has already been classified by
                        # its immutable inference-equivalence group in the
                        # provider kernel.  The first compact page is therefore
                        # the final exact eligible prefix; avoid rebuilding the
                        # generic event/inference machinery query by query.
                        take = min(count, ks[query_index])
                        if take:
                            output_ids[query_index].extend(
                                page.ids[page_row, :take].astype(
                                    np.int64, copy=False).tolist())
                            output_distances[query_index].extend(
                                page.distances[page_row, :take].astype(
                                    np.float64, copy=False).tolist())
                        candidate_events[query_index] += take
                        active[query_index] = False
                        cpu_consume_ms[query_index] += (
                            perf_counter() - consume_started) * 1000.0
                        continue
                    position = 0
                    # Candidate pages already bound host work by M.  Resolve
                    # their sparse serving groups in fixed 4096-event blocks to
                    # amortize Python/NumPy transitions while keeping possible
                    # over-classification below the 4096-event first page.
                    # This is one global executor constant, never selected by
                    # dataset or test labels.
                    event_block = 4096
                    while position < count and active[query_index]:
                        remaining = (
                            max_candidate_events -
                            candidate_events[query_index])
                        if remaining <= 0:
                            fallback_queries[query_index] = True
                            active[query_index] = False
                            break
                        end = min(
                            count, position + event_block,
                            position + int(remaining))
                        chunk_ids = page.ids[page_row, position:end]
                        chunk_distances = page.distances[
                            page_row, position:end]
                        chunk_states = page.states[page_row, position:end]
                        needed = (
                            ks[query_index] -
                            len(output_ids[query_index]))
                        # The kth deterministic hard pass is an exact upper
                        # bound on the prefix that can affect this query: any
                        # unresolved pass before it can only terminate sooner.
                        # Cropping here avoids statistical inference for a
                        # suffix that provably cannot enter top-k.
                        hard_pass_positions = np.flatnonzero(
                            chunk_states == 1)
                        if len(hard_pass_positions) >= needed:
                            useful = int(
                                hard_pass_positions[needed - 1]) + 1
                            end = position + useful
                            chunk_ids = chunk_ids[:useful]
                            chunk_distances = chunk_distances[:useful]
                            chunk_states = chunk_states[:useful]
                        if np.any((chunk_states != 1) & (chunk_states != 2)):
                            raise RuntimeError(
                                "candidate page contains a hard-fail event")
                        chunk_codes = np.zeros(
                            len(chunk_ids), dtype=np.int8)
                        unresolved_positions = np.flatnonzero(
                            chunk_states == 2)
                        candidate_inference_events[query_index] += int(
                            len(unresolved_positions))
                        if len(unresolved_positions):
                            if hasattr(
                                    self.inference,
                                    "evaluate_candidate_batch"):
                                missing_positions = unresolved_positions
                                missing_ids = chunk_ids[missing_positions]
                                inference_started = perf_counter()
                                codes, batch_stats = \
                                    self.inference.evaluate_candidate_batch(
                                        missing_ids,
                                        predicates[query_index])
                                inference_ms[query_index] += (
                                    perf_counter() - inference_started) * 1000.0
                                inference_support_ms[query_index] += float(
                                    batch_stats.get(
                                        "support_inference_ms", 0.0))
                                inference_cache_ms[query_index] += float(
                                    batch_stats.get("cache_lookup_ms", 0.0))
                                lazy_inferences[query_index] += int(
                                    batch_stats[
                                        "representative_evaluations"])
                                cache_accepts[query_index] += int(
                                    batch_stats["cache_accepts"])
                                cache_rejects[query_index] += int(
                                    batch_stats["cache_rejects"])
                                cache_abstains[query_index] += int(
                                    batch_stats["cache_abstains"])
                                chunk_codes[missing_positions] = codes
                            else:
                                compact_cache = compact_caches[query_index]
                                for local_position in unresolved_positions:
                                    tuple_id = int(chunk_ids[local_position])
                                    code = compact_cache.get(tuple_id)
                                    if code is None:
                                        inference_started = perf_counter()
                                        evaluated = self.inference.evaluate(
                                            tuple_id,
                                            predicates[query_index],
                                            assume_unresolved=True)
                                        inference_ms[query_index] += (
                                            perf_counter() -
                                            inference_started) * 1000.0
                                        lazy_inferences[query_index] += 1
                                        code = self._cache_code(evaluated)
                                        compact_cache[tuple_id] = code
                                    elif code == 1:
                                        cache_accepts[query_index] += 1
                                    elif code == 2:
                                        cache_rejects[query_index] += 1
                                    else:
                                        cache_abstains[query_index] += 1
                                    chunk_codes[local_position] = int(code)
                        eligible_positions = np.flatnonzero(
                            (chunk_states == 1) |
                            ((chunk_states == 2) & (chunk_codes == 1)))
                        chosen = eligible_positions[:needed]
                        output_ids[query_index].extend(
                            chunk_ids[chosen].astype(
                                np.int64, copy=False).tolist())
                        output_distances[query_index].extend(
                            chunk_distances[chosen].astype(
                                np.float64, copy=False).tolist())
                        if len(eligible_positions) >= needed:
                            stop = int(chosen[-1]) + 1
                            candidate_events[query_index] += stop
                            processed_unresolved = int(np.count_nonzero(
                                chunk_states[:stop] == 2))
                            unresolved_events[query_index] += \
                                processed_unresolved
                            overclassified_unresolved[query_index] += int(
                                len(unresolved_positions) -
                                processed_unresolved)
                            active[query_index] = False
                            break
                        candidate_events[query_index] += len(chunk_ids)
                        unresolved_events[query_index] += int(
                            len(unresolved_positions))
                        position = end
                    cpu_consume_ms[query_index] += (
                        perf_counter() - consume_started) * 1000.0
                    if bool(page.exhausted[page_row]):
                        active[query_index] = False
                    elif (active[query_index] and
                          candidate_events[query_index] >=
                          max_candidate_events):
                        fallback_queries[query_index] = True
                        active[query_index] = False
                page_size = min(page_size * 2, max_page_size)

            # A cap is a safety valve, never an approximation.  Any query that
            # hits it is replayed through the unchanged exact reference path.
            fallback_indices = np.flatnonzero(fallback_queries)
            fallback_outputs = {}
            if len(fallback_indices):
                if not fallback_to_reference:
                    raise RuntimeError(
                        "candidate stream reached max_candidate_events")
                full_rows = candidate_provider.full_distance_rows(
                    handle, fallback_indices)
                for row_index, query_index_value in enumerate(
                        fallback_indices):
                    query_index = int(query_index_value)
                    fallback_outputs[query_index] = self.search(
                        queries[query_index], ks[query_index],
                        predicates[query_index], epsilon=0.0,
                        precomputed_distances=full_rows[row_index])

            telemetry = handle.telemetry
            tree_probe = bool(telemetry.get(
                "tree_probe_candidate_rows_total", 0))
            physical_rows_per_query = int(np.ceil(
                telemetry.get(
                    "tree_probe_candidate_rows_total",
                    n_rows * query_count) / max(query_count, 1)))
            outputs = []
            for query_index in range(query_count):
                if query_index in fallback_outputs:
                    ids, distances, counters = fallback_outputs[query_index]
                    counters = dict(counters)
                else:
                    ids = np.ascontiguousarray(
                        output_ids[query_index], dtype=np.int64)
                    distances = np.ascontiguousarray(
                        output_distances[query_index], dtype=np.float64)
                    counters = {
                        "ann_seed_candidates": 0,
                        "ann_seed_precomputed_reuse": 0,
                        "node_queue_pops": 0,
                        "tuple_queue_pops": int(
                            unresolved_events[query_index]),
                        "expanded_nodes": 0,
                        "expanded_leaves": 0,
                        "node_bound_evaluations": 0,
                        "pruned_nodes": 0,
                        "visited_tuples": int(
                            candidate_events[query_index]),
                        "tuple_queue_insertions": int(
                            unresolved_events[query_index]),
                        "exact_distance_evaluations": int(
                            physical_rows_per_query),
                        "hard_decisions": int(physical_rows_per_query),
                        "lazy_inferences": int(
                            lazy_inferences[query_index]),
                        "inferences_skipped_by_boundary": int(
                            n_rows - candidate_events[query_index]),
                        "registered_tuples": int(
                            candidate_events[query_index]),
                        "buffer_update_attempts": int(len(ids)),
                        "max_queue_size": 0,
                        "precomputed_distance_rows": 0,
                        "physical_distance_evaluations": int(
                            physical_rows_per_query),
                        "distance_order_specialization": int(
                            not tree_probe),
                        "approximate_tree_probe": int(tree_probe),
                        "statistical_cache_hits": int(
                            cache_accepts[query_index] +
                            cache_rejects[query_index] +
                            cache_abstains[query_index]),
                        "statistical_cache_accepts": int(
                            cache_accepts[query_index]),
                        "statistical_cache_rejects": int(
                            cache_rejects[query_index]),
                        "statistical_cache_abstains": int(
                            cache_abstains[query_index]),
                    }
                counters.update({
                    "gpu_candidate_stream": 1,
                    "candidate_stream_phase_a": 1,
                    "candidate_stream_reference_fallback_enabled": int(
                        fallback_to_reference),
                    "candidate_stream_fallback": int(
                        fallback_queries[query_index]),
                    "candidate_stream_fallback_reason": (
                        "max_candidate_events" if
                        fallback_queries[query_index] else ""),
                    "candidate_stream_page_rounds": int(
                        page_rounds[query_index]),
                    "candidate_stream_inference_event_block": 4096,
                    "candidate_stream_candidates_transferred": int(
                        transferred_candidates[query_index]),
                    "candidate_stream_candidate_events": int(
                        candidate_events[query_index]),
                    "candidate_stream_unresolved_events": int(
                        unresolved_events[query_index]),
                    "candidate_stream_inference_events": int(
                        candidate_inference_events[query_index]),
                    "candidate_stream_overclassified_unresolved": int(
                        overclassified_unresolved[query_index]),
                    "candidate_stream_statistical_inference_ms": float(
                        inference_ms[query_index]),
                    "candidate_stream_cpu_consume_ms": float(
                        cpu_consume_ms[query_index]),
                    "candidate_stream_compact_d2h_bytes": int(
                        telemetry.get("candidate_stream_bytes_per_event", 13) *
                        transferred_candidates[query_index]),
                    "precomputed_distance_host_bytes_per_query": 0,
                    "candidate_stream_peak_compact_host_page_bytes_batch": int(
                        max(peak_compact_host_page_bytes, telemetry.get(
                            "multi_gpu_peak_pending_host_bytes", 0))),
                    "candidate_stream_batch_distance_compute_ms": float(
                        telemetry["distance_compute_ms_total"]),
                    "candidate_stream_batch_hard_predicate_ms": float(
                        telemetry["hard_predicate_ms_total"]),
                    "candidate_stream_batch_candidate_select_ms": float(
                        telemetry["candidate_select_ms_total"]),
                    "candidate_stream_batch_compact_d2h_ms": float(
                        telemetry["compact_d2h_ms_total"]),
                    "candidate_stream_batch_compact_d2h_bytes": int(
                        telemetry["compact_d2h_bytes_total"]),
                    "candidate_stream_batch_cpu_merge_ms": float(
                        telemetry.get("multi_gpu_cpu_merge_ms_total", 0.0)),
                    "candidate_stream_multi_gpu": int(
                        "multi_gpu_cpu_merge_ms_total" in telemetry),
                    "candidate_stream_batch_full_fallback_d2h_bytes": int(
                        telemetry["full_distance_fallback_bytes_total"]),
                    "candidate_stream_peak_gpu_allocated_bytes": int(
                        telemetry["peak_gpu_allocated_bytes"]),
                    "candidate_stream_peak_provider_host_bytes": int(
                        telemetry.get("peak_provider_host_bytes", 0)),
                    "candidate_stream_batch_tree_probe_plan_ms": float(
                        telemetry.get("tree_probe_plan_ms_total", 0.0)),
                    "candidate_stream_batch1_fused_wall_ms": float(
                        telemetry.get("batch1_fused_wall_ms", 0.0)),
                    "candidate_stream_batch1_fused_device_ms": float(
                        telemetry.get("batch1_fused_device_ms_total", 0.0)),
                    "candidate_stream_batch1_device_d2h_wait_ms": float(
                        telemetry.get(
                            "batch1_fused_device_and_d2h_wait_ms_total", 0.0)),
                    "candidate_stream_tree_probe_count": int(
                        telemetry.get("tree_probe_count", 0)),
                    "candidate_stream_tree_probe_rows_total": int(
                        telemetry.get(
                            "tree_probe_candidate_rows_total", 0)),
                    # Generic candidate streams do not have the single-
                    # attribute eligibility-first shortcut.  Map their actual
                    # physical stages onto the same E2-1 logical components.
                    # Batch-one response calls have query_count=1; division
                    # keeps this correct for diagnostic multi-query callers.
                    "candidate_stream_logical_ann_seeding_ms": float(
                        telemetry.get("candidate_select_ms_total", 0.0) /
                        max(query_count, 1)),
                    "candidate_stream_logical_direct_checks_ms": float(
                        logical_direct_checks_ms / max(query_count, 1) +
                        telemetry.get("hard_predicate_ms_total", 0.0) /
                        max(query_count, 1)),
                    "candidate_stream_logical_cache_lookup_ms": float(
                        inference_cache_ms[query_index]),
                    "candidate_stream_logical_exact_distance_ms": float(
                        telemetry.get("distance_compute_ms_total", 0.0) /
                        max(query_count, 1)),
                    "candidate_stream_logical_support_inference_ms": float(
                        logical_support_inference_ms / max(query_count, 1) +
                        inference_support_ms[query_index]),
                })
                if hasattr(self.inference, "record_executor_cache_hits"):
                    self.inference.record_executor_cache_hits(
                        cache_accepts[query_index],
                        cache_rejects[query_index],
                        cache_abstains[query_index])
                outputs.append((ids, distances, counters))
            return outputs
        finally:
            if handle is not None:
                candidate_provider.close(handle)

    def search(self, query_embedding, k, predicate, epsilon=0.0,
               precomputed_distances=None, diagnostic_options=None,
               reference_strategy=None):
        diagnostic = {
            "disable_ann_seed": False,
            "disable_node_bound_pruning": False,
            "disable_tuple_boundary": False,
            "disable_query_cache": False,
        }
        if diagnostic_options is not None:
            unknown = set(diagnostic_options) - set(diagnostic)
            if unknown:
                raise ValueError(
                    f"unknown diagnostic executor options: {sorted(unknown)}")
            diagnostic.update({
                key: bool(value) for key, value in diagnostic_options.items()
            })
        diagnostic_active = bool(any(diagnostic.values()) or
                                 diagnostic_options is not None)
        if reference_strategy is not None and diagnostic_active:
            raise ValueError(
                "reference execution strategies do not accept diagnostics")
        if diagnostic_active and float(epsilon) != 0.0:
            raise ValueError("disable-one diagnostics require exact epsilon=0")
        cache_enabled = not diagnostic["disable_query_cache"]
        inference_before = {
            key: int(self.inference.counters.get(key, 0))
            for key in (
                "support_requests", "inference_cache_hits",
                "inference_cache_misses")
        } if hasattr(self.inference, "counters") else {}
        if hasattr(self.inference, "begin_query"):
            if diagnostic_active:
                self.inference.begin_query(
                    predicate, cache=cache_enabled)
            else:
                self.inference.begin_query(predicate)
        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query.shape != (self.embeddings.shape[1],):
            raise ValueError("query dimension does not match the searchable embeddings")
        k = int(k)
        epsilon = float(epsilon)
        if k <= 0 or epsilon < 0:
            raise ValueError("k must be positive and epsilon non-negative")
        k = min(k, len(self.embeddings))
        if (reference_strategy is not None and
                precomputed_distances is None):
            return self._search_reference_order(
                query, k, predicate, str(reference_strategy),
                np.empty(0, dtype=np.float32), False, inference_before)
        if precomputed_distances is None and self.distance_provider is not None:
            precomputed_distances = self.distance_provider.prepare(query)
        use_precomputed_distances = precomputed_distances is not None
        if use_precomputed_distances:
            precomputed_distances = np.ascontiguousarray(
                precomputed_distances, dtype=np.float32)
            if precomputed_distances.shape != (len(self.embeddings),):
                raise ValueError(
                    "precomputed_distances must contain one value per row")
        else:
            precomputed_distances = np.empty(0, dtype=np.float32)
        if reference_strategy is not None:
            return self._search_reference_order(
                query, k, predicate, str(reference_strategy),
                precomputed_distances, use_precomputed_distances,
                inference_before)
        plan = (self.inference.compile_hard_predicate(predicate)
                if self.fused_cpu and not diagnostic_active and
                hasattr(self.inference, "compile_hard_predicate") else None)
        if plan is not None:
            if (self.allow_distance_order_specialization and
                    use_precomputed_distances and epsilon == 0.0):
                return self._search_fused_distance_order(
                    k, predicate, plan, precomputed_distances)
            return self._search_fused_cpu(
                query, k, predicate, epsilon, plan,
                precomputed_distances, use_precomputed_distances)
        query_sqnorm = float(query @ query)
        if use_precomputed_distances:
            node_lower_bounds = self._precomputed_subtree_lower_bounds(
                precomputed_distances)
        else:
            node_lower_bounds = (
                self.tree.all_node_quantized_point_lower_bounds_sq(query)
                if epsilon > 0.0 and hasattr(
                    self.tree,
                    "all_node_quantized_point_lower_bounds_sq") else
                self.tree.all_node_lower_bounds_sq(query))
        buffer = []
        registered = set()
        queue = []
        counters = {
            "ann_seed_candidates": 0, "node_queue_pops": 0,
            "tuple_queue_pops": 0, "expanded_nodes": 0, "expanded_leaves": 0,
            "node_bound_evaluations": 0, "pruned_nodes": 0,
            "visited_tuples": 0, "tuple_queue_insertions": 0,
            "exact_distance_evaluations": 0, "hard_decisions": 0,
            "lazy_inferences": 0, "inferences_skipped_by_boundary": 0,
            "registered_tuples": 0, "buffer_update_attempts": 0,
            "max_queue_size": 0,
            "ann_seed_precomputed_reuse": 0,
            "statistical_cache_hits": 0,
            "statistical_cache_accepts": 0,
            "statistical_cache_rejects": 0,
            "statistical_cache_abstains": 0,
            "mbr_pruned_subtree_rows": 0,
            "frontier_terminated_node_rows": 0,
            "child_nodes_pruned": 0,
            "boundary_skips_before_eligibility": 0,
            "queued_unresolved_resolutions_avoided": 0,
            "ann_seed_physical_distance_evaluations": 0,
            "on_demand_physical_distance_evaluations": 0,
            "unique_physical_distance_evaluations": 0,
            "statistical_resolution_requests": 0,
            "statistical_resolution_computations": 0,
            "statistical_resolution_cache_hits": 0,
            "statistical_resolution_cache_misses": 0,
            "candidate_rows_not_eligibility_checked": 0,
            "approximate_tail_budget": int(self.approximate_tail_budget),
            "approximate_protected_results": 0,
            "approximate_stop_triggered": 0,
            "approximate_stop_deferred_by_membership_guard": 0,
            "approximate_warm_start_resolutions": 0,
        }
        diagnostic_visited = (
            np.zeros(len(self.embeddings), dtype=np.bool_)
            if diagnostic_active else None)
        diagnostic_visited_leaf_counts = np.zeros(
            self.tree.n_leaves, dtype=np.int64)
        diagnostic_visited_count = 0
        diagnostic_tau_trace = []

        def mark_diagnostic_visited(tuple_id):
            nonlocal diagnostic_visited_count
            if diagnostic_visited is None or diagnostic_visited[tuple_id]:
                return
            diagnostic_visited[tuple_id] = True
            diagnostic_visited_count += 1
            diagnostic_visited_leaf_counts[
                self._searchable_leaf_ids[tuple_id]] += 1

        def unvisited_subtree_rows(node_id):
            node = self.tree.nodes[int(node_id)]
            already_visited = int(diagnostic_visited_leaf_counts[
                int(node.leaf_lo):int(node.leaf_hi) + 1].sum())
            return max(
                int(self._subtree_searchable_rows[int(node_id)]) -
                already_visited, 0)

        def record_diagnostic_tau():
            if not diagnostic_active or len(buffer) < k:
                return
            boundary_value = self._buffer_boundary(buffer, k)
            if (diagnostic_tau_trace and
                    diagnostic_tau_trace[-1]["tau_b_sq"] == boundary_value):
                return
            diagnostic_tau_trace.append({
                "visited_tuples": int(diagnostic_visited_count),
                "physical_distance_evaluations": int(
                    counters["ann_seed_physical_distance_evaluations"] +
                    counters["on_demand_physical_distance_evaluations"]),
                "tau_b_sq": float(boundary_value),
            })

        diagnostic_seed_distances = {}
        diagnostic_distance_seen = (
            np.zeros(len(self.embeddings), dtype=np.bool_)
            if diagnostic_active and not use_precomputed_distances else None)

        def record_diagnostic_distance(tuple_id, counter_name):
            if diagnostic_distance_seen is None:
                return
            tuple_id = int(tuple_id)
            if diagnostic_distance_seen[tuple_id]:
                return
            diagnostic_distance_seen[tuple_id] = True
            counters[counter_name] += 1

        approximate_seed_unresolved = []
        if (self.ann_backend is not None and self.seed_size > 0 and
                not diagnostic["disable_ann_seed"]):
            seed_ids, reused = self._ann_seed_ids(
                query, self.seed_size, precomputed_distances,
                use_precomputed_distances)
            counters["ann_seed_precomputed_reuse"] += int(reused)
            if diagnostic_active and not use_precomputed_distances:
                exposed_ids = getattr(
                    self.ann_backend, "last_candidate_ids", None)
                exposed_distances = getattr(
                    self.ann_backend, "last_candidate_distances", None)
                if (exposed_ids is not None and exposed_distances is not None
                        and len(exposed_ids) == len(exposed_distances)):
                    exposed_ids = np.asarray(exposed_ids, dtype=np.int64)
                    exposed_distances = np.asarray(
                        exposed_distances, dtype=np.float32)
                    diagnostic_seed_distances = {
                        int(tuple_id): float(distance)
                        for tuple_id, distance in zip(
                            exposed_ids, exposed_distances)
                    }
                    for tuple_id in exposed_ids:
                        record_diagnostic_distance(
                            int(tuple_id),
                            "ann_seed_physical_distance_evaluations")
                else:
                    diagnostic_seed_distances = {
                        int(tuple_id): float(distance)
                        for tuple_id, distance in zip(
                            np.asarray(seed_ids, dtype=np.int64),
                            self._last_ann_seed_distances)
                    }
                    if isinstance(self.ann_backend, ExactBackend):
                        diagnostic_distance_seen[:] = True
                        counters[
                            "ann_seed_physical_distance_evaluations"
                        ] = len(self.embeddings)
                    else:
                        for tuple_id in seed_ids:
                            record_diagnostic_distance(
                                int(tuple_id),
                                "ann_seed_physical_distance_evaluations")
            for seed_position, tuple_id in enumerate(
                    np.asarray(seed_ids, dtype=np.int64)):
                mark_diagnostic_visited(int(tuple_id))
                counters["ann_seed_candidates"] += 1
                state, _ = self.inference.hard_decision(int(tuple_id), predicate)
                counters["hard_decisions"] += 1
                if state == "unresolved":
                    # Positive-epsilon execution needs a useful eligible
                    # upper bound before a weak high-dimensional tree
                    # frontier can terminate.  Resolve only the nearest ANN
                    # seed events needed to fill top-k below.  These are
                    # exact distances for concrete tuple IDs, so they only
                    # initialize an upper bound; the theorem-backed frontier
                    # stopping test remains unchanged.
                    if epsilon > 0.0:
                        if use_precomputed_distances:
                            seed_distance = float(
                                precomputed_distances[int(tuple_id)])
                        else:
                            seed_distance = self._distance_sq(
                                query, query_sqnorm, int(tuple_id))
                            record_diagnostic_distance(
                                int(tuple_id),
                                "on_demand_physical_distance_evaluations")
                        counters["exact_distance_evaluations"] += 1
                        approximate_seed_unresolved.append(
                            (seed_distance, int(tuple_id)))
                    continue
                registered.add(int(tuple_id))
                if state == "pass":
                    if use_precomputed_distances:
                        distance = float(precomputed_distances[int(tuple_id)])
                    elif diagnostic_active:
                        distance = float(
                            self._last_ann_seed_distances[seed_position])
                    else:
                        distance = self._distance_sq(
                            query, query_sqnorm, int(tuple_id))
                        record_diagnostic_distance(
                            int(tuple_id),
                            "on_demand_physical_distance_evaluations")
                    counters["exact_distance_evaluations"] += 1
                    counters["buffer_update_attempts"] += 1
                    self._update_buffer(buffer, k, distance, int(tuple_id))
                    record_diagnostic_tau()

        if epsilon > 0.0 and len(buffer) < k:
            for distance, tuple_id in sorted(approximate_seed_unresolved):
                if len(buffer) >= k:
                    break
                result = self.inference.evaluate(
                    tuple_id, predicate, assume_unresolved=True,
                    cache=cache_enabled)
                counters["lazy_inferences"] += 1
                counters["approximate_warm_start_resolutions"] += 1
                registered.add(tuple_id)
                if result.eligible:
                    counters["buffer_update_attempts"] += 1
                    self._update_buffer(
                        buffer, k, float(distance), tuple_id)
                    record_diagnostic_tau()

        initial_nodes = (self.tree.leaf_node_ids if self.leaf_frontier else
                         self.tree.active_roots())
        for node_id in initial_nodes:
            if not self.tree.nodes[int(node_id)].active:
                continue
            lower = float(node_lower_bounds[node_id])
            counters["node_bound_evaluations"] += 1
            heapq.heappush(queue, (lower, self.NODE, int(node_id), "node"))
        counters["max_queue_size"] = max(counters["max_queue_size"], len(queue))

        while queue:
            boundary = self._buffer_boundary(buffer, k)
            frontier = float(queue[0][0])
            if len(buffer) == k:
                if (epsilon == 0.0 and frontier > boundary and
                        not diagnostic["disable_node_bound_pruning"]):
                    counters["frontier_terminated_node_rows"] += int(sum(
                        unvisited_subtree_rows(int(object_id))
                        for _, kind_value, object_id, _ in queue
                        if kind_value == self.NODE))
                    queued = [
                        (float(key_value), int(object_id))
                        for key_value, kind_value, object_id, _ in queue
                        if kind_value == self.TUPLE]
                    if diagnostic["disable_tuple_boundary"]:
                        for _, tuple_id in queued:
                            self.inference.evaluate(
                                tuple_id, predicate,
                                assume_unresolved=True,
                                cache=cache_enabled)
                            counters["lazy_inferences"] += 1
                    else:
                        counters[
                            "queued_unresolved_resolutions_avoided"
                        ] += len(queued)
                    break
                if (epsilon > 0.0 and
                        boundary <= (1.0 + epsilon) ** 2 * frontier):
                    protected_count = self._protected_result_count(k)
                    protected_boundary = self._heap_rank_boundary(
                        buffer, protected_count)
                    if frontier > protected_boundary:
                        self._record_approximate_stop(
                            counters, frontier=frontier, boundary=boundary,
                            protected_boundary=protected_boundary,
                            protected_count=protected_count)
                        break
                    counters[
                        "approximate_stop_deferred_by_membership_guard"] += 1
            key, _, object_id, kind = heapq.heappop(queue)
            if kind == "node":
                counters["node_queue_pops"] += 1
                node = self.tree.nodes[object_id]
                if node.is_leaf:
                    counters["expanded_leaves"] += 1
                    warm_unresolved = []
                    for tuple_id in node.tuple_ids:
                        tuple_id = int(tuple_id)
                        if tuple_id in registered:
                            continue
                        mark_diagnostic_visited(tuple_id)
                        counters["visited_tuples"] += 1
                        if use_precomputed_distances:
                            distance = float(precomputed_distances[tuple_id])
                        elif tuple_id in diagnostic_seed_distances:
                            distance = diagnostic_seed_distances[tuple_id]
                        else:
                            distance = self._distance_sq(
                                query, query_sqnorm, tuple_id)
                            record_diagnostic_distance(
                                tuple_id,
                                "on_demand_physical_distance_evaluations")
                        counters["exact_distance_evaluations"] += 1
                        can_affect = self._can_affect(
                            buffer, k, distance, tuple_id)
                        if (not can_affect and
                                not diagnostic["disable_tuple_boundary"]):
                            registered.add(tuple_id)
                            counters["inferences_skipped_by_boundary"] += 1
                            counters[
                                "boundary_skips_before_eligibility"] += 1
                            continue
                        state, _ = self.inference.hard_decision(tuple_id, predicate)
                        counters["hard_decisions"] += 1
                        if state == "pass" and can_affect:
                            counters["buffer_update_attempts"] += 1
                            self._update_buffer(buffer, k, distance, tuple_id)
                            record_diagnostic_tau()
                        elif state == "unresolved":
                            if can_affect:
                                if epsilon > 0.0 and len(buffer) < k:
                                    warm_unresolved.append(
                                        (float(distance), tuple_id))
                                else:
                                    heapq.heappush(
                                        queue, (distance, self.TUPLE,
                                                tuple_id, "tuple"))
                                    counters["tuple_queue_insertions"] += 1
                            else:
                                self.inference.evaluate(
                                    tuple_id, predicate,
                                    assume_unresolved=True,
                                    cache=cache_enabled)
                                counters["lazy_inferences"] += 1
                        registered.add(tuple_id)
                    # Positive epsilon needs a useful distance boundary before
                    # it can terminate.  Resolve the nearest deferred tuples
                    # from the just-expanded leaf only until k candidates are
                    # available; all remaining events rejoin the unified heap.
                    # This changes scheduling, not the distance certificate.
                    for distance, tuple_id in sorted(warm_unresolved):
                        if (len(buffer) < k and self._can_affect(
                                buffer, k, distance, tuple_id)):
                            result = self.inference.evaluate(
                                tuple_id, predicate,
                                assume_unresolved=True,
                                cache=cache_enabled)
                            counters["lazy_inferences"] += 1
                            counters[
                                "approximate_warm_start_resolutions"] += 1
                            if result.eligible:
                                counters["buffer_update_attempts"] += 1
                                self._update_buffer(
                                    buffer, k, distance, tuple_id)
                                record_diagnostic_tau()
                        elif self._can_affect(
                                buffer, k, distance, tuple_id):
                            heapq.heappush(
                                queue, (distance, self.TUPLE, tuple_id,
                                        "tuple"))
                            counters["tuple_queue_insertions"] += 1
                        else:
                            counters["inferences_skipped_by_boundary"] += 1
                            counters[
                                "queued_unresolved_resolutions_avoided"] += 1
                else:
                    counters["expanded_nodes"] += 1
                    boundary = self._buffer_boundary(buffer, k)
                    for child_id in (node.left, node.right):
                        child = self.tree.nodes[child_id]
                        if not child.active:
                            continue
                        lower = float(node_lower_bounds[child_id])
                        counters["node_bound_evaluations"] += 1
                        if (len(buffer) == k and lower > boundary and
                                not diagnostic[
                                    "disable_node_bound_pruning"]):
                            counters["pruned_nodes"] += 1
                            counters["child_nodes_pruned"] += 1
                            counters["mbr_pruned_subtree_rows"] += int(
                                unvisited_subtree_rows(child_id))
                            continue
                        heapq.heappush(
                            queue, (lower, self.NODE, int(child_id), "node"))
            else:
                counters["tuple_queue_pops"] += 1
                tuple_id = int(object_id)
                if not self._can_affect(buffer, k, float(key), tuple_id):
                    if diagnostic["disable_tuple_boundary"]:
                        self.inference.evaluate(
                            tuple_id, predicate, assume_unresolved=True,
                            cache=cache_enabled)
                        counters["lazy_inferences"] += 1
                    else:
                        counters["inferences_skipped_by_boundary"] += 1
                        counters[
                            "queued_unresolved_resolutions_avoided"] += 1
                    continue
                result = self.inference.evaluate(
                    tuple_id, predicate, assume_unresolved=True,
                    cache=cache_enabled)
                counters["lazy_inferences"] += 1
                if result.eligible:
                    counters["buffer_update_attempts"] += 1
                    self._update_buffer(buffer, k, float(key), tuple_id)
                    record_diagnostic_tau()
            counters["max_queue_size"] = max(
                counters["max_queue_size"], len(queue))

        counters["registered_tuples"] = len(registered)
        if diagnostic_visited is not None:
            counters["visited_tuples"] = int(
                np.count_nonzero(diagnostic_visited))
        counters["precomputed_distance_rows"] = (
            len(self.embeddings) if use_precomputed_distances else 0)
        unique_physical_distances = (
            len(self.embeddings) if use_precomputed_distances else
            int(np.count_nonzero(diagnostic_distance_seen))
            if diagnostic_distance_seen is not None else
            counters["exact_distance_evaluations"])
        counters["unique_physical_distance_evaluations"] = int(
            unique_physical_distances)
        counters["physical_distance_evaluations"] = int(
            unique_physical_distances)
        inference_after = getattr(self.inference, "counters", {})
        observed_hits = int(
            inference_after.get("inference_cache_hits", 0) -
            inference_before.get("inference_cache_hits", 0))
        observed_misses = int(
            inference_after.get("inference_cache_misses", 0) -
            inference_before.get("inference_cache_misses", 0))
        if (observed_hits < 0 or observed_misses < 0 or
                observed_hits > counters["lazy_inferences"]):
            raise AssertionError(
                "resolution cache accounting is inconsistent")
        counters["statistical_resolution_requests"] = int(
            counters["lazy_inferences"])
        counters["statistical_resolution_computations"] = (
            int(counters["lazy_inferences"]) - int(observed_hits))
        counters["statistical_resolution_cache_hits"] = int(observed_hits)
        counters["statistical_resolution_cache_misses"] = int(
            observed_misses)
        counters["candidate_rows_not_eligibility_checked"] = max(
            int(len(self.embeddings) - counters["hard_decisions"]), 0)
        if diagnostic_active and hasattr(self.inference, "counters"):
            inference_after = self.inference.counters
            counters["support_requests"] = int(
                inference_after.get("support_requests", 0) -
                inference_before.get("support_requests", 0))
            counters["cache_hits"] = int(
                inference_after.get("inference_cache_hits", 0) -
                inference_before.get("inference_cache_hits", 0))
            counters["cache_misses"] = (
                counters["lazy_inferences"]
                if diagnostic["disable_query_cache"] else observed_misses)
            counters["diagnostic_options"] = dict(diagnostic)
            counters["diagnostic_root_tree_on_demand"] = bool(
                not self.leaf_frontier and not use_precomputed_distances)
            counters["tau_boundary_trace"] = diagnostic_tau_trace
            accounted_rows = (
                counters["visited_tuples"] +
                counters["mbr_pruned_subtree_rows"] +
                counters["frontier_terminated_node_rows"])
            if accounted_rows > len(self.embeddings):
                raise AssertionError(
                    "diagnostic row-avoidance counters overlap")
        output = sorted(
            ((-distance, -neg_id, tuple_id)
             for distance, neg_id, tuple_id in buffer),
            key=lambda item: (item[0], item[1]))
        ids = np.asarray([item[2] for item in output], dtype=np.int64)
        distances = np.asarray([item[0] for item in output], dtype=np.float64)
        return ids, distances, counters

    @staticmethod
    def _array_update(buffer_ids, buffer_distances, count, k, distance, tuple_id):
        return int(_rank_insert(
            buffer_ids, buffer_distances, int(count), int(k),
            float(distance), int(tuple_id)))

    def _fused_approximate_warm_start(
            self, unresolved_ids, unresolved_distances, buffer_ids,
            buffer_distances, buffer_count, k, predicate, counters,
            statistical_cache, use_statistical_cache):
        """Resolve only enough nearest leaf-local events to seed top-k."""
        if int(buffer_count) >= int(k) or not len(unresolved_ids):
            return (int(buffer_count), unresolved_ids,
                    unresolved_distances)
        order = np.lexsort((unresolved_ids, unresolved_distances))
        remaining_ids = []
        remaining_distances = []
        for position in order:
            tuple_id = int(unresolved_ids[int(position)])
            distance = float(unresolved_distances[int(position)])
            can_affect = _rank_can_affect(
                buffer_ids, buffer_distances, int(buffer_count), int(k),
                distance, tuple_id)
            if int(buffer_count) < int(k) and can_affect:
                result = self.inference.evaluate(
                    tuple_id, predicate, assume_unresolved=True)
                counters["lazy_inferences"] += 1
                counters["approximate_warm_start_resolutions"] += 1
                if use_statistical_cache:
                    statistical_cache[tuple_id] = self._cache_code(result)
                if result.eligible:
                    counters["buffer_update_attempts"] += 1
                    buffer_count = self._array_update(
                        buffer_ids, buffer_distances, buffer_count, k,
                        distance, tuple_id)
            elif can_affect:
                remaining_ids.append(tuple_id)
                remaining_distances.append(distance)
            else:
                counters["inferences_skipped_by_boundary"] += 1
        return (
            int(buffer_count),
            np.asarray(remaining_ids, dtype=np.int64),
            np.asarray(remaining_distances, dtype=np.float64),
        )

    def _search_fused_cpu(self, query, k, predicate, epsilon, plan,
                          precomputed_distances, use_precomputed_distances):
        """Paper-equivalent executor with fused native leaf processing."""
        query_sqnorm = float(query @ query)
        node_lower_bounds = (
            self.tree.all_node_quantized_point_lower_bounds_sq(query)
            if epsilon > 0.0 and hasattr(
                self.tree, "all_node_quantized_point_lower_bounds_sq") else
            self.tree.all_node_lower_bounds_sq(query))
        buffer_ids = np.full(k, -1, dtype=np.int64)
        buffer_distances = np.full(k, np.inf, dtype=np.float64)
        buffer_count = 0
        registered = np.zeros(len(self.embeddings), dtype=np.bool_)
        queue = []
        counters = {
            "ann_seed_candidates": 0, "node_queue_pops": 0,
            "tuple_queue_pops": 0, "expanded_nodes": 0, "expanded_leaves": 0,
            "node_bound_evaluations": 0, "pruned_nodes": 0,
            "visited_tuples": 0, "tuple_queue_insertions": 0,
            "exact_distance_evaluations": 0, "hard_decisions": 0,
            "lazy_inferences": 0, "inferences_skipped_by_boundary": 0,
            "registered_tuples": 0, "buffer_update_attempts": 0,
            "max_queue_size": 0,
            "ann_seed_precomputed_reuse": 0,
            "statistical_cache_hits": 0,
            "statistical_cache_accepts": 0,
            "statistical_cache_rejects": 0,
            "statistical_cache_abstains": 0,
            "approximate_tail_budget": int(self.approximate_tail_budget),
            "approximate_protected_results": 0,
            "approximate_stop_triggered": 0,
            "approximate_stop_deferred_by_membership_guard": 0,
            "approximate_warm_start_resolutions": 0,
        }

        seed_ids = np.empty(0, dtype=np.int64)
        if self.ann_backend is not None and self.seed_size > 0:
            seed_ids, reused = self._ann_seed_ids(
                query, self.seed_size, precomputed_distances,
                use_precomputed_distances)
            counters["ann_seed_precomputed_reuse"] += int(reused)
            counters["ann_seed_candidates"] += int(len(seed_ids))

        statistical_cache = (
            self.inference.statistical_eligibility_cache()
            if hasattr(self.inference, "statistical_eligibility_cache")
            else None)
        use_statistical_cache = statistical_cache is not None
        if not use_statistical_cache:
            statistical_cache = np.empty(0, dtype=np.int8)

        if (self.leaf_frontier and use_precomputed_distances and
                epsilon == 0.0 and use_statistical_cache):
            return self._search_fused_leaf_native(
                k, predicate, plan, precomputed_distances,
                node_lower_bounds, seed_ids, counters, statistical_cache)

        if len(seed_ids):
            (buffer_count, hard, distances, updates) = _fused_seed_candidates(
                self.embeddings, self.sqnorm, query, query_sqnorm,
                seed_ids, registered, plan["values"], plan["missing"],
                plan["attrs"], plan["ops"], plan["lower"], plan["upper"],
                buffer_ids, buffer_distances, buffer_count, k,
                precomputed_distances, use_precomputed_distances)
            counters["hard_decisions"] += int(hard)
            counters["exact_distance_evaluations"] += int(distances)
            counters["buffer_update_attempts"] += int(updates)

        if self.leaf_frontier:
            return self._search_fused_leaf_merge(
                query, query_sqnorm, k, predicate, epsilon, plan,
                precomputed_distances, use_precomputed_distances,
                node_lower_bounds, buffer_ids, buffer_distances,
                buffer_count, registered, counters,
                statistical_cache, use_statistical_cache)

        initial_nodes = self.tree.active_roots()
        for node_id in initial_nodes:
            if not self.tree.nodes[int(node_id)].active:
                continue
            lower = float(node_lower_bounds[node_id])
            counters["node_bound_evaluations"] += 1
            heapq.heappush(queue, (lower, self.NODE, int(node_id)))
        counters["max_queue_size"] = len(queue)

        while queue:
            boundary = (float("inf") if buffer_count < k else
                        float(buffer_distances[k - 1]))
            frontier = float(queue[0][0])
            if buffer_count == k:
                if epsilon == 0.0 and frontier > boundary:
                    break
                if (epsilon > 0.0 and
                        boundary <= (1.0 + epsilon) ** 2 * frontier):
                    protected_count = self._protected_result_count(k)
                    protected_boundary = (
                        float("-inf") if protected_count == 0 else
                        float(buffer_distances[protected_count - 1]))
                    if frontier > protected_boundary:
                        self._record_approximate_stop(
                            counters, frontier=frontier, boundary=boundary,
                            protected_boundary=protected_boundary,
                            protected_count=protected_count)
                        break
                    counters[
                        "approximate_stop_deferred_by_membership_guard"] += 1
            key, kind, object_id = heapq.heappop(queue)
            if kind == self.NODE:
                counters["node_queue_pops"] += 1
                node = self.tree.nodes[object_id]
                if node.is_leaf:
                    counters["expanded_leaves"] += 1
                    result = _fused_leaf_candidates(
                        self.embeddings, self.sqnorm, query, query_sqnorm,
                        node.tuple_ids, registered,
                        plan["values"], plan["missing"], plan["attrs"],
                        plan["ops"], plan["lower"], plan["upper"],
                        buffer_ids, buffer_distances, buffer_count, k,
                        precomputed_distances, use_precomputed_distances,
                        statistical_cache, use_statistical_cache)
                    (buffer_count, unresolved_ids, unresolved_distances,
                     visited, exact, hard, skipped, updates,
                     cached_accepts, cached_rejects, cached_abstains) = result
                    counters["visited_tuples"] += int(visited)
                    counters["exact_distance_evaluations"] += int(exact)
                    counters["hard_decisions"] += int(hard)
                    counters["inferences_skipped_by_boundary"] += int(skipped)
                    counters["buffer_update_attempts"] += int(updates)
                    counters["statistical_cache_accepts"] += int(
                        cached_accepts)
                    counters["statistical_cache_rejects"] += int(
                        cached_rejects)
                    counters["statistical_cache_abstains"] += int(
                        cached_abstains)
                    if epsilon > 0.0:
                        (buffer_count, unresolved_ids,
                         unresolved_distances) = \
                            self._fused_approximate_warm_start(
                                unresolved_ids, unresolved_distances,
                                buffer_ids, buffer_distances, buffer_count, k,
                                predicate, counters, statistical_cache,
                                use_statistical_cache)
                    counters["tuple_queue_insertions"] += int(len(unresolved_ids))
                    for tuple_id, distance in zip(
                            unresolved_ids, unresolved_distances):
                        heapq.heappush(
                            queue, (float(distance), self.TUPLE, int(tuple_id)))
                else:
                    counters["expanded_nodes"] += 1
                    boundary = (float("inf") if buffer_count < k else
                                float(buffer_distances[k - 1]))
                    for child_id in (node.left, node.right):
                        child = self.tree.nodes[child_id]
                        if not child.active:
                            continue
                        lower = float(node_lower_bounds[child_id])
                        counters["node_bound_evaluations"] += 1
                        if buffer_count == k and lower > boundary:
                            counters["pruned_nodes"] += 1
                            continue
                        heapq.heappush(
                            queue, (lower, self.NODE, int(child_id)))
            else:
                counters["tuple_queue_pops"] += 1
                tuple_id = int(object_id)
                if not _rank_can_affect(
                        buffer_ids, buffer_distances, buffer_count, k,
                        float(key), tuple_id):
                    counters["inferences_skipped_by_boundary"] += 1
                    continue
                result = self.inference.evaluate(
                    tuple_id, predicate, assume_unresolved=True)
                counters["lazy_inferences"] += 1
                if use_statistical_cache:
                    statistical_cache[tuple_id] = self._cache_code(result)
                if result.eligible:
                    counters["buffer_update_attempts"] += 1
                    buffer_count = self._array_update(
                        buffer_ids, buffer_distances, buffer_count, k,
                        float(key), tuple_id)
            if len(queue) > counters["max_queue_size"]:
                counters["max_queue_size"] = len(queue)

        counters["registered_tuples"] = int(registered.sum())
        counters["statistical_cache_hits"] = int(
            counters["statistical_cache_accepts"] +
            counters["statistical_cache_rejects"] +
            counters["statistical_cache_abstains"])
        if hasattr(self.inference, "record_executor_cache_hits"):
            self.inference.record_executor_cache_hits(
                counters["statistical_cache_accepts"],
                counters["statistical_cache_rejects"],
                counters["statistical_cache_abstains"])
        counters["precomputed_distance_rows"] = (
            len(self.embeddings) if use_precomputed_distances else 0)
        counters["physical_distance_evaluations"] = (
            len(self.embeddings) if use_precomputed_distances else
            counters["exact_distance_evaluations"])
        return (buffer_ids[:buffer_count].copy(),
                buffer_distances[:buffer_count].copy(), counters)

    def _search_fused_leaf_native(
            self, k, predicate, plan, precomputed_distances,
            _node_lower_bounds, seed_ids, counters, statistical_cache):
        """Stateful native exact leaf frontier with cached inference events."""
        # Use the same finite-precision distance row that defines ranking.
        # Projected MBR bounds can be slightly positive when a float32
        # algebraic distance cancels to zero, which would incorrectly prune
        # later leaves containing lower-ID ties.
        leaf_lowers = np.empty(len(self._leaf_node_ids), dtype=np.float64)
        _precomputed_leaf_min_bounds(
            precomputed_distances, self._leaf_offsets,
            self._leaf_tuple_ids, leaf_lowers)
        leaf_order = np.lexsort((self._leaf_node_ids, leaf_lowers))
        ordered_lowers = np.ascontiguousarray(
            leaf_lowers[leaf_order], dtype=np.float64)
        leaf_order = np.ascontiguousarray(leaf_order, dtype=np.int64)
        lazy_inferences = 0
        new_accepts = 0
        new_rejects = 0
        new_abstains = 0
        n_rows = len(self.embeddings)
        buffer_ids = np.full(k, -1, dtype=np.int64)
        buffer_distances = np.full(k, np.inf, dtype=np.float64)
        registered = np.zeros(n_rows, dtype=np.bool_)
        heap_distances = np.empty(n_rows, dtype=np.float64)
        heap_ids = np.empty(n_rows, dtype=np.int64)
        frontier_state = np.zeros(_LEAF_STATE_SIZE, dtype=np.int64)
        buffer_count, seed_hard, seed_exact, seed_updates = (
            _fused_seed_candidates(
                self.embeddings, self.sqnorm,
                np.empty(0, dtype=np.float32), 0.0,
                seed_ids, registered, plan["values"], plan["missing"],
                plan["attrs"], plan["ops"], plan["lower"], plan["upper"],
                buffer_ids, buffer_distances, 0, k,
                precomputed_distances, True))
        frontier_state[_LEAF_BUFFER_COUNT] = int(buffer_count)
        frontier_state[_LEAF_HARD] = int(seed_hard)
        frontier_state[_LEAF_EXACT] = int(seed_exact)
        frontier_state[_LEAF_UPDATES] = int(seed_updates)
        frontier_state[_LEAF_REGISTERED] = int(np.count_nonzero(registered))
        frontier_state[_LEAF_MAX_QUEUE] = int(len(leaf_order))
        resumes = 0
        while True:
            resumes += 1
            need_inference = _fused_leaf_frontier_resume(
                leaf_order, ordered_lowers,
                self._leaf_offsets, self._leaf_tuple_ids,
                plan["values"], plan["missing"], plan["attrs"],
                plan["ops"], plan["lower"], plan["upper"],
                precomputed_distances, statistical_cache, k,
                buffer_ids, buffer_distances, registered,
                heap_distances, heap_ids, frontier_state)
            if int(need_inference) < 0:
                break
            tuple_id = int(need_inference)
            evaluated = self.inference.evaluate(
                tuple_id, predicate, assume_unresolved=True)
            code = self._cache_code(evaluated)
            statistical_cache[tuple_id] = code
            lazy_inferences += 1
            if code == 1:
                new_accepts += 1
            elif code == 2:
                new_rejects += 1
            else:
                new_abstains += 1
            if lazy_inferences > len(self.embeddings):
                raise RuntimeError(
                    "native leaf scheduler did not converge after cache fill")

        buffer_count = int(frontier_state[_LEAF_BUFFER_COUNT])
        node_pops = int(frontier_state[_LEAF_NODE_POPS])
        visited = int(frontier_state[_LEAF_VISITED])
        exact = int(frontier_state[_LEAF_EXACT])
        hard = int(frontier_state[_LEAF_HARD])
        skipped = int(frontier_state[_LEAF_BOUNDARY_SKIPS])
        registered_count = int(frontier_state[_LEAF_REGISTERED])
        updates = int(frontier_state[_LEAF_UPDATES])
        max_queue = int(frontier_state[_LEAF_MAX_QUEUE])
        tuple_insertions = int(frontier_state[_LEAF_TUPLE_INSERTIONS])
        tuple_pops = int(frontier_state[_LEAF_TUPLE_POPS])
        cached_accepts = int(frontier_state[_LEAF_CACHE_ACCEPTS])
        cached_rejects = int(frontier_state[_LEAF_CACHE_REJECTS])
        cached_abstains = int(frontier_state[_LEAF_CACHE_ABSTAINS])
        cache_accepts = max(int(cached_accepts) - new_accepts, 0)
        cache_rejects = max(int(cached_rejects) - new_rejects, 0)
        cache_abstains = max(int(cached_abstains) - new_abstains, 0)
        counters.update({
            "node_queue_pops": int(node_pops),
            "tuple_queue_pops": int(tuple_pops),
            "expanded_nodes": 0,
            "expanded_leaves": int(node_pops),
            "node_bound_evaluations": int(len(self._leaf_node_ids)),
            "pruned_nodes": 0,
            "visited_tuples": int(visited),
            "tuple_queue_insertions": int(tuple_insertions),
            "exact_distance_evaluations": int(exact),
            "hard_decisions": int(hard),
            "lazy_inferences": int(lazy_inferences),
            "inferences_skipped_by_boundary": int(skipped),
            "registered_tuples": int(registered_count),
            "buffer_update_attempts": int(updates),
            "max_queue_size": int(max_queue),
            "statistical_cache_hits": int(
                cache_accepts + cache_rejects + cache_abstains),
            "statistical_cache_accepts": int(cache_accepts),
            "statistical_cache_rejects": int(cache_rejects),
            "statistical_cache_abstains": int(cache_abstains),
            "precomputed_distance_rows": int(len(self.embeddings)),
            "physical_distance_evaluations": int(len(self.embeddings)),
            "fused_leaf_frontier_native": 1,
            "native_scheduler_replays": 0,
            "native_scheduler_resumes": int(resumes),
            "precomputed_leaf_min_bound_rows": int(len(self.embeddings)),
        })
        if hasattr(self.inference, "record_executor_cache_hits"):
            self.inference.record_executor_cache_hits(
                cache_accepts, cache_rejects, cache_abstains)
        return (buffer_ids[:buffer_count].copy(),
                buffer_distances[:buffer_count].copy(), counters)

    def _search_fused_distance_order(
            self, k, predicate, plan, precomputed_distances):
        """Exact specialization when a complete distance row already exists.

        A complete exact distance row dominates every projected-tree lower
        bound.  Sorting it by the paper's deterministic ``(distance, id)``
        order and stopping after the kth predicted-eligible tuple is therefore
        exactly equivalent to draining the unified node--tuple frontier.  The
        direct predicate scan is compacted in Numba; only unresolved tuples
        cross into the indexed statistical inference path.
        """
        # Squared-L2 providers return finite, non-negative float32 values.  In
        # that domain the IEEE-754 bit pattern is already numerically ordered,
        # so a unique uint64 ``(distance_bits, tuple_id)`` key replaces the
        # considerably slower two-column ``lexsort`` while preserving the
        # paper's exact deterministic tie rule.  The generic public-executor
        # fallback retains lexsort semantics for caller-supplied exceptional
        # values (negative values, signed negative zero, infinities, or NaNs).
        if (len(self.embeddings) <= np.iinfo(np.uint32).max and
                np.isfinite(precomputed_distances).all() and
                not np.signbit(precomputed_distances).any()):
            distance_bits = precomputed_distances.view(np.uint32).astype(
                np.uint64)
            ranked_keys = ((distance_bits << np.uint64(32)) |
                           self._tuple_ids_uint64)
            ranked_ids = np.argsort(ranked_keys)
        else:
            ranked_ids = np.lexsort(
                (self._tuple_ids_int64, precomputed_distances))
        output_ids = np.empty(k, dtype=np.int64)
        output_distances = np.empty(k, dtype=np.float64)
        output_count = 0
        position = 0
        visited = 0
        lazy_inferences = 0
        # One native transition usually supplies the complete top-k while
        # bounding over-classification when predicates are highly selective.
        event_batch = max(16, min(128, 2 * int(k)))
        max_event_batch = 0
        statistical_cache = (
            self.inference.statistical_eligibility_cache()
            if hasattr(self.inference, "statistical_eligibility_cache")
            else None)
        use_statistical_cache = statistical_cache is not None
        if not use_statistical_cache:
            statistical_cache = np.empty(0, dtype=np.int8)
        cached_accepts = 0
        cached_rejects = 0
        cached_abstains = 0
        while position < len(ranked_ids) and output_count < k:
            (events, states, position, scanned,
             batch_cached_accepts, batch_cached_rejects,
             batch_cached_abstains) = \
                _fused_ranked_events(
                ranked_ids, position, event_batch,
                plan["values"], plan["missing"], plan["attrs"],
                plan["ops"], plan["lower"], plan["upper"],
                statistical_cache, use_statistical_cache)
            visited += int(scanned)
            cached_accepts += int(batch_cached_accepts)
            cached_rejects += int(batch_cached_rejects)
            cached_abstains += int(batch_cached_abstains)
            max_event_batch = max(max_event_batch, int(len(events)))
            for tuple_id, state in zip(events, states):
                tuple_id = int(tuple_id)
                eligible = int(state) == 1
                if int(state) == 2:
                    lazy_inferences += 1
                    result = self.inference.evaluate(
                        tuple_id, predicate,
                        assume_unresolved=True)
                    eligible = result.eligible
                    if use_statistical_cache:
                        if eligible:
                            statistical_cache[tuple_id] = 1
                        elif result.state == "statistical":
                            statistical_cache[tuple_id] = 2
                        else:
                            statistical_cache[tuple_id] = 3
                if eligible:
                    output_ids[output_count] = tuple_id
                    output_distances[output_count] = float(
                        precomputed_distances[tuple_id])
                    output_count += 1
                    if output_count == k:
                        break

        if hasattr(self.inference, "record_executor_cache_hits"):
            self.inference.record_executor_cache_hits(
                cached_accepts, cached_rejects, cached_abstains)

        counters = {
            "ann_seed_candidates": 0,
            "node_queue_pops": 0,
            "tuple_queue_pops": int(lazy_inferences),
            "expanded_nodes": 0,
            "expanded_leaves": 0,
            "node_bound_evaluations": 0,
            "pruned_nodes": 0,
            "visited_tuples": int(visited),
            "tuple_queue_insertions": int(lazy_inferences),
            "exact_distance_evaluations": int(visited),
            "hard_decisions": int(visited),
            "lazy_inferences": int(lazy_inferences),
            "inferences_skipped_by_boundary": int(
                len(self.embeddings) - visited),
            "registered_tuples": int(visited),
            "buffer_update_attempts": int(output_count),
            "max_queue_size": int(max_event_batch),
            "precomputed_distance_rows": int(len(self.embeddings)),
            "physical_distance_evaluations": int(len(self.embeddings)),
            "distance_order_specialization": 1,
            "statistical_cache_hits": int(
                cached_accepts + cached_rejects + cached_abstains),
            "statistical_cache_accepts": int(cached_accepts),
            "statistical_cache_rejects": int(cached_rejects),
            "statistical_cache_abstains": int(cached_abstains),
        }
        return (output_ids[:output_count].copy(),
                output_distances[:output_count].copy(), counters)

    def _search_fused_leaf_merge(
            self, query, query_sqnorm, k, predicate, epsilon, plan,
            precomputed_distances, use_precomputed_distances,
            node_lower_bounds, buffer_ids, buffer_distances, buffer_count,
            registered, counters, statistical_cache,
            use_statistical_cache):
        """Merge a sorted leaf frontier with lazy tuples without node heaps.

        Every searchable row belongs to exactly one active leaf and each leaf
        retains its safe MBR lower bound.  Sorting those bounds once is
        therefore the same best-first frontier as inserting all leaf nodes in
        the unified heap; unresolved tuples are still interleaved by exact
        distance and nodes still win equal-key ties.
        """
        leaf_ids = np.asarray([
            int(node_id) for node_id in self.tree.leaf_node_ids
            if self.tree.nodes[int(node_id)].active], dtype=np.int64)
        leaf_lowers = np.asarray(
            node_lower_bounds[leaf_ids], dtype=np.float64)
        order = np.lexsort((leaf_ids, leaf_lowers))
        leaf_ids = leaf_ids[order]
        leaf_lowers = leaf_lowers[order]
        counters["node_bound_evaluations"] += int(len(leaf_ids))
        tuple_queue = []
        leaf_position = 0
        counters["max_queue_size"] = max(
            counters["max_queue_size"], int(len(leaf_ids)))

        while leaf_position < len(leaf_ids) or tuple_queue:
            boundary = (float("inf") if buffer_count < k else
                        float(buffer_distances[k - 1]))
            leaf_frontier = (float(leaf_lowers[leaf_position])
                             if leaf_position < len(leaf_ids) else
                             float("inf"))
            tuple_frontier = (float(tuple_queue[0][0]) if tuple_queue else
                              float("inf"))
            frontier = min(leaf_frontier, tuple_frontier)
            if buffer_count == k:
                if epsilon == 0.0 and frontier > boundary:
                    break
                if epsilon > 0.0 and boundary <= (
                        1.0 + epsilon) ** 2 * frontier:
                    protected_count = self._protected_result_count(k)
                    protected_boundary = (
                        float("-inf") if protected_count == 0 else
                        float(buffer_distances[protected_count - 1]))
                    if frontier > protected_boundary:
                        self._record_approximate_stop(
                            counters, frontier=frontier, boundary=boundary,
                            protected_boundary=protected_boundary,
                            protected_count=protected_count)
                        break
                    counters[
                        "approximate_stop_deferred_by_membership_guard"] += 1

            # A node precedes a tuple when their queue keys are identical.
            if leaf_frontier <= tuple_frontier:
                node_id = int(leaf_ids[leaf_position])
                leaf_position += 1
                counters["node_queue_pops"] += 1
                counters["expanded_leaves"] += 1
                node = self.tree.nodes[node_id]
                result = _fused_leaf_candidates(
                    self.embeddings, self.sqnorm, query, query_sqnorm,
                    node.tuple_ids, registered,
                    plan["values"], plan["missing"], plan["attrs"],
                    plan["ops"], plan["lower"], plan["upper"],
                    buffer_ids, buffer_distances, buffer_count, k,
                    precomputed_distances, use_precomputed_distances,
                    statistical_cache, use_statistical_cache)
                (buffer_count, unresolved_ids, unresolved_distances,
                 visited, exact, hard, skipped, updates,
                 cached_accepts, cached_rejects, cached_abstains) = result
                counters["visited_tuples"] += int(visited)
                counters["exact_distance_evaluations"] += int(exact)
                counters["hard_decisions"] += int(hard)
                counters["inferences_skipped_by_boundary"] += int(skipped)
                counters["buffer_update_attempts"] += int(updates)
                counters["statistical_cache_accepts"] += int(cached_accepts)
                counters["statistical_cache_rejects"] += int(cached_rejects)
                counters["statistical_cache_abstains"] += int(cached_abstains)
                if epsilon > 0.0:
                    (buffer_count, unresolved_ids,
                     unresolved_distances) = \
                        self._fused_approximate_warm_start(
                            unresolved_ids, unresolved_distances,
                            buffer_ids, buffer_distances, buffer_count, k,
                            predicate, counters, statistical_cache,
                            use_statistical_cache)
                counters["tuple_queue_insertions"] += int(len(unresolved_ids))
                for tuple_id, distance in zip(
                        unresolved_ids, unresolved_distances):
                    heapq.heappush(
                        tuple_queue, (float(distance), int(tuple_id)))
            else:
                key, tuple_id = heapq.heappop(tuple_queue)
                counters["tuple_queue_pops"] += 1
                if not _rank_can_affect(
                        buffer_ids, buffer_distances, buffer_count, k,
                        float(key), int(tuple_id)):
                    counters["inferences_skipped_by_boundary"] += 1
                    continue
                result = self.inference.evaluate(
                    int(tuple_id), predicate, assume_unresolved=True)
                counters["lazy_inferences"] += 1
                if use_statistical_cache:
                    statistical_cache[int(tuple_id)] = self._cache_code(result)
                if result.eligible:
                    counters["buffer_update_attempts"] += 1
                    buffer_count = self._array_update(
                        buffer_ids, buffer_distances, buffer_count, k,
                        float(key), int(tuple_id))
            logical_queue_size = (
                len(leaf_ids) - leaf_position + len(tuple_queue))
            if logical_queue_size > counters["max_queue_size"]:
                counters["max_queue_size"] = logical_queue_size

        counters["registered_tuples"] = int(registered.sum())
        counters["statistical_cache_hits"] = int(
            counters["statistical_cache_accepts"] +
            counters["statistical_cache_rejects"] +
            counters["statistical_cache_abstains"])
        if hasattr(self.inference, "record_executor_cache_hits"):
            self.inference.record_executor_cache_hits(
                counters["statistical_cache_accepts"],
                counters["statistical_cache_rejects"],
                counters["statistical_cache_abstains"])
        counters["precomputed_distance_rows"] = (
            len(self.embeddings) if use_precomputed_distances else 0)
        counters["physical_distance_evaluations"] = (
            len(self.embeddings) if use_precomputed_distances else
            counters["exact_distance_evaluations"])
        return (buffer_ids[:buffer_count].copy(),
                buffer_distances[:buffer_count].copy(), counters)
