"""Fused CUDA distance, predicate, and inference-group probe kernel.

The exact candidate executor writes a complete ``Q x N`` score matrix.  The
paper tree already partitions the searchable snapshot, so the approximate
ANN/tree serving path instead supplies a compact, disjoint list of row IDs per
query.  This kernel evaluates only those rows and writes one compact score
matrix suitable for deterministic top-k selection.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tree_probe_distance_predicate_kernel(
        base, queries, candidate_ids, candidate_counts,
        values, missing, plan_attrs, plan_ops, plan_lower, plan_upper,
        atom_counts, statistical_group_ids, statistical_decisions,
        output_scores, available_counts,
        n_rows: tl.constexpr, dimension: tl.constexpr,
        candidate_width: tl.constexpr, max_atoms: tl.constexpr,
        statistical_group_count: tl.constexpr,
        block_rows: tl.constexpr, block_dimension: tl.constexpr):
    query_index = tl.program_id(0)
    block_index = tl.program_id(1)
    candidate_offsets = (
        block_index * block_rows + tl.arange(0, block_rows))
    candidate_count = tl.load(candidate_counts + query_index)
    candidate_valid = candidate_offsets < candidate_count
    candidate_id = tl.load(
        candidate_ids + query_index * candidate_width + candidate_offsets,
        mask=candidate_valid, other=0)

    atom_count = tl.load(atom_counts + query_index)
    hard_fail = ~candidate_valid
    unresolved = tl.zeros((block_rows,), dtype=tl.int1)
    for atom_index in range(max_atoms):
        atom_valid = atom_index < atom_count
        plan_offset = query_index * max_atoms + atom_index
        attr = tl.load(plan_attrs + plan_offset, mask=atom_valid, other=0)
        op = tl.load(plan_ops + plan_offset, mask=atom_valid, other=-1)
        lower = tl.load(
            plan_lower + plan_offset, mask=atom_valid, other=0.0)
        upper = tl.load(
            plan_upper + plan_offset, mask=atom_valid, other=0.0)
        column_offset = attr * n_rows + candidate_id
        value = tl.load(
            values + column_offset,
            mask=candidate_valid & atom_valid, other=0.0)
        is_missing = tl.load(
            missing + column_offset,
            mask=candidate_valid & atom_valid, other=0).to(tl.int1)
        satisfied = (
            ((op == 0) & (value <= lower)) |
            ((op == 1) & (value >= lower)) |
            ((op == 2) & (value >= lower) & (value <= upper)) |
            ((op == 3) & (value == lower)))
        hard_fail |= atom_valid & (~is_missing) & (~satisfied)
        unresolved |= atom_valid & is_missing

    group_id = tl.load(
        statistical_group_ids + candidate_id,
        mask=candidate_valid & unresolved, other=-1)
    valid_group = group_id >= 0
    safe_group = tl.maximum(group_id, 0)
    decision = tl.load(
        statistical_decisions +
        query_index * statistical_group_count + safe_group,
        mask=candidate_valid & unresolved & valid_group, other=3)
    accepted_unresolved = unresolved & valid_group & (decision == 1)
    hard_fail |= unresolved & (~accepted_unresolved)
    eligible = candidate_valid & (~hard_fail)

    dimensions = tl.arange(0, block_dimension)
    dimension_valid = dimensions < dimension
    base_values = tl.load(
        base + candidate_id[:, None] * dimension + dimensions[None, :],
        mask=eligible[:, None] & dimension_valid[None, :], other=0.0)
    query_values = tl.load(
        queries + query_index * dimension + dimensions,
        mask=dimension_valid, other=0.0)
    difference = base_values - query_values[None, :]
    distance = tl.sum(difference * difference, axis=1)
    output_offset = query_index * candidate_width + candidate_offsets
    tl.store(
        output_scores + output_offset,
        tl.where(eligible, distance, float("inf")), mask=candidate_valid)
    block_available = tl.sum(eligible.to(tl.int32), axis=0)
    tl.atomic_add(available_counts + query_index, block_available)


def fused_tree_probe_scores(
        base: torch.Tensor, queries: torch.Tensor,
        candidate_ids: torch.Tensor, candidate_counts: torch.Tensor,
        values: torch.Tensor, missing: torch.Tensor,
        plan_attrs: torch.Tensor, plan_ops: torch.Tensor,
        plan_lower: torch.Tensor, plan_upper: torch.Tensor,
        atom_counts: torch.Tensor, statistical_group_ids: torch.Tensor,
        statistical_decisions: torch.Tensor,
        statistical_group_count: int):
    """Return compact eligible squared-L2 scores and eligible counts."""
    if not base.is_cuda or not queries.is_cuda:
        raise ValueError("tree-probe vectors must be CUDA-resident")
    if base.dtype != torch.float32 or queries.dtype != torch.float32:
        raise ValueError("tree-probe vectors must use float32")
    if base.ndim != 2 or queries.ndim != 2 or \
            base.shape[1] != queries.shape[1]:
        raise ValueError("base/query matrices must align")
    if candidate_ids.dtype != torch.int64 or candidate_ids.ndim != 2:
        raise ValueError("candidate IDs must be a two-dimensional int64 tensor")
    if candidate_counts.shape != (len(queries),):
        raise ValueError("candidate counts must align with queries")
    if values.dtype != torch.float64 or values.ndim != 2:
        raise ValueError("predicate values must be a float64 matrix")
    if missing.dtype != torch.bool or missing.shape != values.shape:
        raise ValueError("predicate missingness must align with values")
    if values.shape[1] != len(base):
        raise ValueError("predicate columns must align with base rows")
    query_count, candidate_width = map(int, candidate_ids.shape)
    if query_count != len(queries) or candidate_width < 1:
        raise ValueError("candidate rows must align with queries")
    if plan_attrs.ndim != 2 or plan_attrs.shape[0] != query_count:
        raise ValueError("predicate plans must align with queries")

    output = torch.full(
        (query_count, candidate_width), float("inf"),
        dtype=torch.float32, device=base.device)
    available = torch.zeros(
        query_count, dtype=torch.int32, device=base.device)
    block_rows = 4
    block_dimension = triton.next_power_of_2(int(base.shape[1]))
    if block_dimension > 256:
        raise ValueError("tree-probe kernel supports dimensions up to 256")
    grid = (query_count, triton.cdiv(candidate_width, block_rows))
    with torch.cuda.device(base.device):
        _tree_probe_distance_predicate_kernel[grid](
            base, queries, candidate_ids, candidate_counts,
            values, missing, plan_attrs, plan_ops, plan_lower, plan_upper,
            atom_counts, statistical_group_ids, statistical_decisions,
            output, available,
            n_rows=int(len(base)), dimension=int(base.shape[1]),
            candidate_width=candidate_width,
            max_atoms=int(plan_attrs.shape[1]),
            statistical_group_count=int(statistical_group_count),
            block_rows=block_rows, block_dimension=block_dimension,
            num_warps=4)
    return output, available
