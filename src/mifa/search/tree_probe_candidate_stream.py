"""Projected-tree compact candidate streams for CPU and CUDA.

This is the physical ANN/tree serving specialization.  It can reuse the
paper's projected leaves or a searchable-snapshot-only physical partition,
ranks leaves by centroid, and probes one globally configured number of leaves.
Predicate semantics and statistical group decisions are identical to the
exact candidate executor; only physical candidate generation is approximate
and is therefore reported with fixed-policy recall instead of being labelled
exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit, prange

from .gpu_candidate_stream import CandidatePage, HARD_PASS


@dataclass
class TreeProbeCandidateStreamHandle:
    ids: np.ndarray
    distances: np.ndarray
    available_counts: np.ndarray
    emitted_counts: np.ndarray
    query_count: int
    telemetry: dict
    closed: bool = False


def _compile_plans(plans, base_rows):
    plans = list(plans)
    first = plans[0]
    values = np.ascontiguousarray(first["values"], dtype=np.float64)
    missing = np.ascontiguousarray(first["missing"], dtype=np.bool_)
    if values.ndim != 2 or missing.shape != values.shape or \
            values.shape[1] != base_rows:
        raise ValueError("compiled predicate columns do not align with base")
    max_atoms = max(1, max(len(plan["attrs"]) for plan in plans))
    attrs = np.zeros((len(plans), max_atoms), dtype=np.int32)
    ops = np.full((len(plans), max_atoms), -1, dtype=np.int8)
    lower = np.zeros((len(plans), max_atoms), dtype=np.float64)
    upper = np.zeros((len(plans), max_atoms), dtype=np.float64)
    counts = np.empty(len(plans), dtype=np.int32)
    for query_index, plan in enumerate(plans):
        if plan["values"] is not first["values"] or \
                plan["missing"] is not first["missing"]:
            raise ValueError("one probe batch must share predicate data")
        count = len(plan["attrs"])
        counts[query_index] = count
        attrs[query_index, :count] = plan["attrs"]
        ops[query_index, :count] = plan["ops"]
        lower[query_index, :count] = plan["lower"]
        upper[query_index, :count] = plan["upper"]

    prefetched = [
        plan for plan in plans
        if "statistical_group_ids" in plan and
        "statistical_group_decisions" in plan]
    if not prefetched:
        if not all(
                not bool(plan.get("has_unresolved_attributes", True))
                for plan in plans):
            raise ValueError(
                "tree probe requires immutable statistical-group prefetch")
        # A hard-only single-query batch has no statistical groups to
        # prefetch.  Supply one unreachable neutral group so it can use the
        # same fused kernel as a mixed batch; every candidate is decided by
        # the hard predicate before this code is consulted.
        group_ids = np.zeros(base_rows, dtype=np.int32)
        decisions = np.zeros((len(plans), 1), dtype=np.int8)
        return {
            "values": values,
            "missing": missing,
            "attrs": attrs,
            "ops": ops,
            "lower": lower,
            "upper": upper,
            "counts": counts,
            "group_ids": group_ids,
            "decisions": decisions,
            "group_count": 1,
        }
    group_ids = prefetched[0]["statistical_group_ids"]
    group_count = len(prefetched[0]["statistical_group_decisions"])
    aligned = (
        len(group_ids) == base_rows and group_count > 0 and
        all(
            (plan["statistical_group_ids"] is group_ids and
             len(plan["statistical_group_decisions"]) == group_count)
            if "statistical_group_ids" in plan else
            not bool(plan.get("has_unresolved_attributes", True))
            for plan in plans))
    if not aligned:
        raise ValueError("tree-probe statistical groups are not aligned")
    decisions = np.ascontiguousarray([
        plan["statistical_group_decisions"]
        if "statistical_group_decisions" in plan else
        np.zeros(group_count, dtype=np.int8)
        for plan in plans], dtype=np.int8)
    return {
        "values": values,
        "missing": missing,
        "attrs": attrs,
        "ops": ops,
        "lower": lower,
        "upper": upper,
        "counts": counts,
        "group_ids": np.ascontiguousarray(group_ids, dtype=np.int32),
        "decisions": decisions,
        "group_count": int(group_count),
    }


class _ProjectedLeafProbe:
    def __init__(self, tree, base_rows, probe_count, *, partition_index=None):
        self.tree = tree
        self.probe_count = int(probe_count)
        if self.probe_count < 1:
            raise ValueError("probe_count must be positive")
        if partition_index is None:
            leaf_nodes = [
                tree.nodes[int(node_id)] for node_id in tree.leaf_node_ids
                if tree.nodes[int(node_id)].active]
            if not leaf_nodes:
                raise ValueError("tree has no active searchable leaves")
            self.leaf_node_ids = np.asarray(
                [node.node_id for node in leaf_nodes], dtype=np.int64)
            self.leaf_rows = tuple(
                np.ascontiguousarray(node.tuple_ids, dtype=np.int64)
                for node in leaf_nodes)
            projected = np.ascontiguousarray(
                tree.search_projected, dtype=np.float32)
            self.centroids = np.ascontiguousarray([
                projected[rows].mean(axis=0, dtype=np.float64)
                for rows in self.leaf_rows], dtype=np.float32)
            self._project_queries = tree.project
            self.partition_kind = "paper_projected_tree"
        else:
            if int(partition_index.base_rows) != int(base_rows):
                raise ValueError("ANN partitions must align with base rows")
            self.leaf_node_ids = np.arange(
                len(partition_index.partition_rows), dtype=np.int64)
            self.leaf_rows = tuple(
                np.ascontiguousarray(rows, dtype=np.int64)
                for rows in partition_index.partition_rows)
            self.centroids = np.ascontiguousarray(
                partition_index.centroids, dtype=np.float32)
            self._project_queries = partition_index.project
            self.partition_kind = str(partition_index.kind)
        if self.probe_count > len(self.leaf_rows):
            raise ValueError("probe_count exceeds active partition count")
        flattened = np.concatenate(self.leaf_rows)
        if len(flattened) != base_rows or \
                not np.array_equal(np.sort(flattened), np.arange(base_rows)):
            raise ValueError("active leaves must partition every base row")

    def candidates(self, queries):
        started = perf_counter()
        projected_queries = np.ascontiguousarray(
            self._project_queries(queries), dtype=np.float32)
        differences = (
            projected_queries[:, None, :] - self.centroids[None, :, :])
        centroid_distances = np.einsum(
            "qld,qld->ql", differences, differences)
        if self.probe_count < len(self.leaf_rows):
            selected = np.argpartition(
                centroid_distances, self.probe_count - 1, axis=1
            )[:, :self.probe_count]
        else:
            selected = np.broadcast_to(
                np.arange(len(self.leaf_rows), dtype=np.int64),
                centroid_distances.shape).copy()
        for query_index in range(len(selected)):
            row = selected[query_index]
            order = np.lexsort((
                self.leaf_node_ids[row], centroid_distances[query_index, row]))
            selected[query_index] = row[order]
        counts = np.asarray([
            sum(len(self.leaf_rows[int(leaf)]) for leaf in row)
            for row in selected], dtype=np.int64)
        width = int(counts.max(initial=0))
        candidates = np.full((len(selected), width), -1, dtype=np.int64)
        for query_index, leaves in enumerate(selected):
            position = 0
            for leaf in leaves:
                rows = self.leaf_rows[int(leaf)]
                candidates[query_index, position:position + len(rows)] = rows
                position += len(rows)
        elapsed_ms = (perf_counter() - started) * 1000.0
        return candidates, counts, selected, elapsed_ms


class FaissIVFPartitionIndex:
    """Offline searchable-snapshot IVF partitions shared by CPU and CUDA."""

    kind = "faiss_kmeans_ivf_partitions"

    def __init__(self, base, *, nlist=1024, training_rows=262144,
                 iterations=10, seed=0, threads=24):
        import faiss

        base = np.ascontiguousarray(base, dtype=np.float32)
        if base.ndim != 2 or not len(base):
            raise ValueError("IVF construction needs a non-empty base matrix")
        self.base_rows, self.dimension = map(int, base.shape)
        self.nlist = min(int(nlist), self.base_rows)
        if self.nlist < 1:
            raise ValueError("nlist must be positive")
        training_rows = min(max(int(training_rows), self.nlist), self.base_rows)
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("IVF k-means iterations must be positive")
        faiss.omp_set_num_threads(max(1, int(threads)))
        if training_rows == self.base_rows:
            training = base
        else:
            rng = np.random.default_rng(int(seed))
            selected = np.sort(rng.choice(
                self.base_rows, size=training_rows, replace=False))
            training = np.ascontiguousarray(base[selected])
        kmeans = faiss.Kmeans(
            self.dimension, self.nlist, niter=iterations,
            verbose=False, seed=int(seed), gpu=False,
            min_points_per_centroid=1,
            max_points_per_centroid=max(
                256, int(np.ceil(training_rows / self.nlist))))
        kmeans.train(training)
        self.centroids = np.ascontiguousarray(
            kmeans.centroids, dtype=np.float32)
        self.quantizer = faiss.IndexFlatL2(self.dimension)
        self.quantizer.add(self.centroids)
        labels = np.empty(self.base_rows, dtype=np.int64)
        assignment_batch = 65536
        for start in range(0, self.base_rows, assignment_batch):
            end = min(start + assignment_batch, self.base_rows)
            _, current = self.quantizer.search(base[start:end], 1)
            labels[start:end] = current[:, 0]
        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=self.nlist)
        offsets = np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.cumsum(counts, dtype=np.int64)))
        self.partition_rows = tuple(
            np.ascontiguousarray(order[offsets[index]:offsets[index + 1]],
                                 dtype=np.int64)
            for index in range(self.nlist))
        if any(not len(rows) for rows in self.partition_rows):
            raise RuntimeError("IVF construction produced an empty partition")

    @staticmethod
    def project(queries):
        return np.ascontiguousarray(queries, dtype=np.float32)

    def manifest(self):
        sizes = np.asarray(
            [len(rows) for rows in self.partition_rows], dtype=np.int64)
        return {
            "kind": self.kind,
            "base_rows": self.base_rows,
            "dimension": self.dimension,
            "nlist": self.nlist,
            "partition_rows_min": int(sizes.min()),
            "partition_rows_mean": float(sizes.mean()),
            "partition_rows_max": int(sizes.max()),
        }


@njit(cache=True, parallel=True, fastmath=False)
def _cpu_fused_tree_probe_topk(
        base, queries, candidate_ids, candidate_counts,
        values, missing, attrs, ops, lower, upper, atom_counts,
        statistical_group_ids, statistical_decisions, result_width):
    query_count = len(queries)
    output_ids = np.full(
        (query_count, result_width), -1, dtype=np.int64)
    output_distances = np.full(
        (query_count, result_width), np.inf, dtype=np.float32)
    output_counts = np.zeros(query_count, dtype=np.int64)
    for query_index in prange(query_count):
        count = 0
        for position in range(int(candidate_counts[query_index])):
            tuple_id = int(candidate_ids[query_index, position])
            failed = False
            unresolved = False
            for atom_index in range(int(atom_counts[query_index])):
                attr = int(attrs[query_index, atom_index])
                if missing[attr, tuple_id]:
                    unresolved = True
                    continue
                value = values[attr, tuple_id]
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
                    failed = True
                    break
            if failed:
                continue
            if unresolved:
                group = int(statistical_group_ids[tuple_id])
                if group < 0 or statistical_decisions[
                        query_index, group] != 1:
                    continue
            distance = np.float32(0.0)
            for dimension in range(base.shape[1]):
                difference = np.float32(
                    base[tuple_id, dimension] -
                    queries[query_index, dimension])
                distance = np.float32(
                    distance + np.float32(difference * difference))
            if count == result_width:
                worst_distance = output_distances[query_index, count - 1]
                worst_id = output_ids[query_index, count - 1]
                if distance > worst_distance or \
                        (distance == worst_distance and tuple_id >= worst_id):
                    continue
                insert = count - 1
            else:
                insert = count
                count += 1
            while insert > 0:
                previous_distance = output_distances[
                    query_index, insert - 1]
                previous_id = output_ids[query_index, insert - 1]
                if previous_distance < distance or \
                        (previous_distance == distance and
                         previous_id <= tuple_id):
                    break
                output_distances[query_index, insert] = previous_distance
                output_ids[query_index, insert] = previous_id
                insert -= 1
            output_distances[query_index, insert] = distance
            output_ids[query_index, insert] = tuple_id
        output_counts[query_index] = count
    return output_ids, output_distances, output_counts


class _TreeProbePageMixin:
    def next_page(self, handle, active_mask, *, page_size):
        if handle.closed:
            raise RuntimeError("candidate stream handle is closed")
        active_mask = np.asarray(active_mask, dtype=np.bool_)
        if active_mask.shape != (handle.query_count,):
            raise ValueError("active mask does not align with probe batch")
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
            starts + int(page_size), handle.available_counts[active])
        counts = np.ascontiguousarray(targets - starts, dtype=np.int64)
        width = int(counts.max(initial=0))
        ids = np.full((len(active), width), -1, dtype=np.int64)
        distances = np.full(
            (len(active), width), np.inf, dtype=np.float32)
        states = np.zeros((len(active), width), dtype=np.int8)
        for output_row, query_index in enumerate(active):
            start = int(starts[output_row])
            target = int(targets[output_row])
            count = target - start
            ids[output_row, :count] = handle.ids[query_index, start:target]
            distances[output_row, :count] = \
                handle.distances[query_index, start:target]
            states[output_row, :count] = int(HARD_PASS)
        handle.emitted_counts[active] = targets
        handle.telemetry["page_rounds"] += 1
        return CandidatePage(
            query_indices=np.ascontiguousarray(active, dtype=np.int64),
            ids=ids, distances=distances, states=states, counts=counts,
            exhausted=np.ascontiguousarray(
                targets >= handle.available_counts[active], dtype=np.bool_))

    def full_distance_rows(self, handle, query_indices):
        raise RuntimeError(
            "approximate tree-probe streams do not materialize full distances")

    def close(self, handle):
        handle.closed = True
        handle.ids = None
        handle.distances = None


class NumpyTreeProbeCandidateStreamProvider(
        _ProjectedLeafProbe, _TreeProbePageMixin):
    """CPU Numba fused projected-leaf candidate stream."""

    def __init__(self, base, tree, *, probe_count=1, result_width=64,
                 partition_index=None):
        self.base = np.ascontiguousarray(base, dtype=np.float32)
        if self.base.ndim != 2:
            raise ValueError("base must be a two-dimensional matrix")
        self.base_rows, self.dim = map(int, self.base.shape)
        self.global_offset = 0
        self.result_width = int(result_width)
        if self.result_width < 1:
            raise ValueError("result_width must be positive")
        super().__init__(
            tree, self.base_rows, probe_count,
            partition_index=partition_index)
        _cpu_fused_tree_probe_topk(
            np.zeros((1, self.dim), dtype=np.float32),
            np.zeros((1, self.dim), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int64), np.ones(1, dtype=np.int64),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.bool_),
            np.zeros((1, 1), dtype=np.int32),
            np.zeros((1, 1), dtype=np.int8),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.ones(1, dtype=np.int32), np.zeros(1, dtype=np.int32),
            np.ones((1, 1), dtype=np.int8), 1)

    def begin_batch(self, queries, plans):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width does not match base")
        plans = list(plans)
        if len(plans) != len(queries) or not len(queries):
            raise ValueError("queries and predicate plans must align")
        compiled = _compile_plans(plans, self.base_rows)
        candidates, candidate_counts, _, plan_ms = self.candidates(queries)
        started = perf_counter()
        ids, distances, available_counts = _cpu_fused_tree_probe_topk(
            self.base, queries, candidates, candidate_counts,
            compiled["values"], compiled["missing"], compiled["attrs"],
            compiled["ops"], compiled["lower"], compiled["upper"],
            compiled["counts"], compiled["group_ids"],
            compiled["decisions"], self.result_width)
        compute_ms = (perf_counter() - started) * 1000.0
        telemetry = {
            "distance_compute_ms_total": float(compute_ms),
            "predicate_column_upload_ms_total": 0.0,
            "hard_predicate_ms_total": 0.0,
            "candidate_select_ms_total": 0.0,
            "compact_d2h_ms_total": 0.0,
            "compact_d2h_bytes_total": 0,
            "full_distance_fallback_bytes_total": 0,
            "page_rounds": 0,
            "phase": "projected_tree_probe_cpu_fused",
            "statistical_group_prefetch": True,
            "tree_probe_plan_ms_total": float(plan_ms),
            "tree_probe_candidate_rows_total": int(candidate_counts.sum()),
            "tree_probe_count": int(self.probe_count),
            "peak_gpu_allocated_bytes": 0,
            "peak_provider_host_bytes": int(
                candidates.nbytes + ids.nbytes + distances.nbytes),
        }
        return TreeProbeCandidateStreamHandle(
            ids=ids, distances=distances,
            available_counts=available_counts,
            emitted_counts=np.zeros(len(queries), dtype=np.int64),
            query_count=len(queries), telemetry=telemetry)

    def manifest(self):
        return {
            "kind": "numpy_projected_tree_probe_candidate_stream",
            "phase": "projected_tree_probe_cpu_fused",
            "base_rows": self.base_rows,
            "dimension": self.dim,
            "probe_count": int(self.probe_count),
            "active_leaves": len(self.leaf_rows),
            "partition_kind": self.partition_kind,
            "result_width": self.result_width,
            "rank_order": "approximate_tree_probe_then_(l2,row_id)",
        }


class TorchTreeProbeCandidateStreamProvider(
        _ProjectedLeafProbe, _TreeProbePageMixin):
    """CUDA Triton fused projected-leaf candidate stream."""

    def __init__(self, base, tree, *, device="cuda:0", probe_count=1,
                 result_width=64, partition_index=None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the tree-probe provider")
        base = np.ascontiguousarray(base, dtype=np.float32)
        if base.ndim != 2:
            raise ValueError("base must be a two-dimensional matrix")
        self.torch = torch
        self.device = torch.device(device)
        self.base_rows, self.dim = map(int, base.shape)
        self.global_offset = 0
        self.result_width = int(result_width)
        if self.result_width < 1:
            raise ValueError("result_width must be positive")
        self.base_device = torch.from_numpy(base).to(self.device)
        self._predicate_source = None
        self._values_device = None
        self._missing_device = None
        self._group_source = None
        self._group_ids_device = None
        super().__init__(
            tree, self.base_rows, probe_count,
            partition_index=partition_index)

    def _synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def _predicate_device_data(self, compiled):
        source = (id(compiled["values"]), id(compiled["missing"]))
        if self._predicate_source != source:
            self._values_device = self.torch.from_numpy(
                compiled["values"]).to(self.device)
            self._missing_device = self.torch.from_numpy(
                compiled["missing"]).to(self.device)
            self._predicate_source = source
        group_source = id(compiled["group_ids"])
        if self._group_source != group_source:
            self._group_ids_device = self.torch.from_numpy(
                compiled["group_ids"]).to(self.device)
            self._group_source = group_source

        def device(values):
            return self.torch.from_numpy(
                np.ascontiguousarray(values)).to(self.device)

        return (
            self._values_device, self._missing_device,
            device(compiled["attrs"]), device(compiled["ops"]),
            device(compiled["lower"]), device(compiled["upper"]),
            device(compiled["counts"]), self._group_ids_device,
            device(compiled["decisions"]))

    def begin_batch(self, queries, plans):
        from .triton_tree_probe import fused_tree_probe_scores

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width does not match base")
        plans = list(plans)
        if len(plans) != len(queries) or not len(queries):
            raise ValueError("queries and predicate plans must align")
        compiled = _compile_plans(plans, self.base_rows)
        candidates, candidate_counts, _, plan_ms = self.candidates(queries)
        self.torch.cuda.reset_peak_memory_stats(self.device)

        self._synchronize()
        upload_started = perf_counter()
        query_device = self.torch.from_numpy(queries).to(self.device)
        candidate_device = self.torch.from_numpy(candidates).to(self.device)
        candidate_count_device = self.torch.from_numpy(
            candidate_counts).to(self.device)
        predicate_device = self._predicate_device_data(compiled)
        self._synchronize()
        upload_ms = (perf_counter() - upload_started) * 1000.0

        compute_started = perf_counter()
        scores, available_device = fused_tree_probe_scores(
            self.base_device, query_device, candidate_device,
            candidate_count_device, *predicate_device,
            compiled["group_count"])
        self._synchronize()
        compute_ms = (perf_counter() - compute_started) * 1000.0

        select_started = perf_counter()
        width = min(self.result_width, scores.shape[1])
        distance_bits = scores.view(self.torch.int32).to(self.torch.int64)
        safe_ids = self.torch.where(
            self.torch.isfinite(scores), candidate_device,
            self.torch.full_like(candidate_device, 0xFFFFFFFF))
        keys = (distance_bits << 32) | safe_ids
        _, selected_slots = self.torch.topk(
            keys, width, dim=1, largest=False, sorted=True)
        selected_ids = candidate_device.gather(1, selected_slots)
        selected_distances = scores.gather(1, selected_slots)
        available_device = available_device.clamp_max(width)
        slots = self.torch.arange(
            width, dtype=self.torch.int32, device=self.device)[None, :]
        valid = slots < available_device[:, None]
        selected_ids = self.torch.where(
            valid, selected_ids, self.torch.full_like(selected_ids, -1))
        selected_distances = self.torch.where(
            valid, selected_distances,
            self.torch.full_like(selected_distances, float("inf")))
        self._synchronize()
        select_ms = (perf_counter() - select_started) * 1000.0

        d2h_started = perf_counter()
        ids = np.ascontiguousarray(selected_ids.cpu().numpy(), dtype=np.int64)
        distances = np.ascontiguousarray(
            selected_distances.cpu().numpy(), dtype=np.float32)
        available_counts = np.ascontiguousarray(
            available_device.cpu().numpy(), dtype=np.int64)
        self._synchronize()
        d2h_ms = (perf_counter() - d2h_started) * 1000.0
        compact_bytes = int(
            ids.nbytes + distances.nbytes + available_counts.nbytes)
        telemetry = {
            "distance_compute_ms_total": float(compute_ms),
            "predicate_column_upload_ms_total": float(upload_ms),
            "hard_predicate_ms_total": 0.0,
            "candidate_select_ms_total": float(select_ms),
            "compact_d2h_ms_total": float(d2h_ms),
            "compact_d2h_bytes_total": compact_bytes,
            "full_distance_fallback_bytes_total": 0,
            "page_rounds": 0,
            "phase": "projected_tree_probe_gpu_fused",
            "statistical_group_prefetch": True,
            "tree_probe_plan_ms_total": float(plan_ms),
            "tree_probe_candidate_rows_total": int(candidate_counts.sum()),
            "tree_probe_count": int(self.probe_count),
            "peak_gpu_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)),
        }
        return TreeProbeCandidateStreamHandle(
            ids=ids, distances=distances,
            available_counts=available_counts,
            emitted_counts=np.zeros(len(queries), dtype=np.int64),
            query_count=len(queries), telemetry=telemetry)

    @property
    def resident_bytes(self):
        return int(self.base_device.numel() * self.base_device.element_size())

    def manifest(self):
        return {
            "kind": "torch_projected_tree_probe_candidate_stream",
            "phase": "projected_tree_probe_gpu_fused",
            "device": str(self.device),
            "base_rows": self.base_rows,
            "dimension": self.dim,
            "probe_count": int(self.probe_count),
            "active_leaves": len(self.leaf_rows),
            "partition_kind": self.partition_kind,
            "result_width": self.result_width,
            "resident_bytes": self.resident_bytes,
            "rank_order": "approximate_tree_probe_then_(l2,row_id)",
            "normal_path_full_distance_d2h": False,
        }
