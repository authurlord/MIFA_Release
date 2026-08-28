"""Fused CUDA hard-predicate classification and deterministic rank keys."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _eligible_row_score_kernel(
        base, query, row_ids, base_sqnorm, output,
        row_count: tl.constexpr, dimension: tl.constexpr,
        block_rows: tl.constexpr, block_dimension: tl.constexpr):
    """Fuse indexed vector reads and one-query distance surrogates."""
    block_index = tl.program_id(0)
    row_offsets = block_index * block_rows + tl.arange(0, block_rows)
    row_valid = row_offsets < row_count
    row_ids_value = tl.load(row_ids + row_offsets, mask=row_valid, other=0)
    dimensions = tl.arange(0, block_dimension)
    dimension_valid = dimensions < dimension
    base_values = tl.load(
        base + row_ids_value[:, None] * dimension + dimensions[None, :],
        mask=row_valid[:, None] & dimension_valid[None, :], other=0.0)
    query_values = tl.load(
        query + dimensions, mask=dimension_valid, other=0.0)
    dot = tl.sum(
        base_values.to(tl.float32) * query_values[None, :].to(tl.float32),
        axis=1)
    norm = tl.load(
        base_sqnorm + row_ids_value, mask=row_valid, other=0.0).to(tl.float32)
    tl.store(output + row_offsets, norm - 2.0 * dot, mask=row_valid)


def eligible_row_scores(
        base: torch.Tensor, query: torch.Tensor,
        row_ids: torch.Tensor, base_sqnorm: torch.Tensor):
    """Return exact-order shortlist surrogates for eligible rows only."""
    if not base.is_cuda or base.dtype != torch.float16 or base.ndim != 2:
        raise ValueError("eligible-row base must be a CUDA float16 matrix")
    if query.shape != (1, base.shape[1]) or query.device != base.device:
        raise ValueError("eligible-row query must align with the base")
    if row_ids.dtype != torch.int64 or row_ids.ndim != 1:
        raise ValueError("eligible row IDs must be one-dimensional int64")
    if base_sqnorm.dtype != torch.float16 or base_sqnorm.shape != (len(base),):
        raise ValueError("eligible-row norms must align with the base")
    row_count = int(row_ids.numel())
    output = torch.empty(
        (1, row_count), dtype=torch.float32, device=base.device)
    if row_count == 0:
        return output
    block_rows = 8
    block_dimension = triton.next_power_of_2(int(base.shape[1]))
    if block_dimension > 256:
        raise ValueError("eligible-row kernel supports dimensions up to 256")
    grid = (triton.cdiv(row_count, block_rows),)
    with torch.cuda.device(base.device):
        _eligible_row_score_kernel[grid](
            base, query.to(dtype=torch.float16), row_ids, base_sqnorm,
            output, row_count=row_count, dimension=int(base.shape[1]),
            block_rows=block_rows, block_dimension=block_dimension,
            num_warps=4)
    return output


@triton.jit
def _hard_state_selection_kernel(
        scores, values, missing, plan_attrs, plan_ops, plan_lower,
        plan_upper, atom_counts, states, selection_scores, available_counts,
        statistical_group_ids, statistical_decisions,
        n_rows: tl.constexpr, max_atoms: tl.constexpr,
        global_offset: tl.constexpr, block_size: tl.constexpr,
        statistical_group_count: tl.constexpr,
        use_statistical_groups: tl.constexpr):
    query_index = tl.program_id(0)
    block_index = tl.program_id(1)
    row_offsets = block_index * block_size + tl.arange(0, block_size)
    row_mask = row_offsets < n_rows
    atom_count = tl.load(atom_counts + query_index)
    hard_fail = tl.zeros((block_size,), dtype=tl.int1)
    unresolved = tl.zeros((block_size,), dtype=tl.int1)

    for atom_index in range(max_atoms):
        atom_valid = atom_index < atom_count
        plan_offset = query_index * max_atoms + atom_index
        attr = tl.load(plan_attrs + plan_offset, mask=atom_valid, other=0)
        op = tl.load(plan_ops + plan_offset, mask=atom_valid, other=-1)
        lower = tl.load(
            plan_lower + plan_offset, mask=atom_valid, other=0.0)
        upper = tl.load(
            plan_upper + plan_offset, mask=atom_valid, other=0.0)
        column_offsets = attr * n_rows + row_offsets
        value = tl.load(
            values + column_offsets, mask=row_mask & atom_valid,
            other=0.0)
        is_missing = tl.load(
            missing + column_offsets, mask=row_mask & atom_valid,
            other=0).to(tl.int1)
        satisfied = (
            ((op == 0) & (value <= lower)) |
            ((op == 1) & (value >= lower)) |
            ((op == 2) & (value >= lower) & (value <= upper)) |
            ((op == 3) & (value == lower)))
        hard_fail |= atom_valid & (~is_missing) & (~satisfied)
        unresolved |= atom_valid & is_missing

    if use_statistical_groups:
        group_id = tl.load(
            statistical_group_ids + row_offsets, mask=row_mask, other=-1)
        valid_group = group_id >= 0
        safe_group = tl.maximum(group_id, 0)
        decision = tl.load(
            statistical_decisions +
            query_index * statistical_group_count + safe_group,
            mask=row_mask & valid_group, other=3)
        statistically_accepted = unresolved & valid_group & (decision == 1)
        hard_fail |= unresolved & (~statistically_accepted)
        unresolved = tl.zeros((block_size,), dtype=tl.int1)

    keep = row_mask & (~hard_fail)
    state = tl.where(hard_fail, 0, tl.where(unresolved, 2, 1))
    score_offsets = query_index * n_rows + row_offsets
    score = tl.load(scores + score_offsets, mask=row_mask, other=0.0)
    if not use_statistical_groups:
        tl.store(states + score_offsets, state, mask=row_mask)
    tl.store(
        selection_scores + score_offsets,
        tl.where(hard_fail, float("inf"), score), mask=row_mask)
    block_available = tl.sum(keep.to(tl.int32), axis=0)
    tl.atomic_add(available_counts + query_index, block_available)


def fused_hard_states_and_selection_scores(
        scores: torch.Tensor, values: torch.Tensor,
        missing: torch.Tensor, plan_attrs: torch.Tensor,
        plan_ops: torch.Tensor, plan_lower: torch.Tensor,
        plan_upper: torch.Tensor, atom_counts: torch.Tensor,
        global_offset: int, statistical_group_ids: torch.Tensor,
        statistical_decisions: torch.Tensor, statistical_group_count: int,
        use_statistical_groups: bool):
    """Return states, masked selection scores, and available counts."""
    if not scores.is_cuda:
        raise ValueError("scores must be CUDA-resident")
    if scores.dtype not in {torch.float16, torch.float32} or scores.ndim != 2:
        raise ValueError(
            "scores must be a two-dimensional float16/float32 tensor")
    query_count, n_rows = map(int, scores.shape)
    if values.dtype != torch.float64 or values.ndim != 2:
        raise ValueError("predicate values must be a float64 matrix")
    if missing.dtype != torch.bool or missing.shape != values.shape:
        raise ValueError("predicate missingness must align with values")
    if values.shape[1] != n_rows:
        raise ValueError("predicate columns must align with scores")
    if plan_attrs.ndim != 2 or plan_attrs.shape[0] != query_count:
        raise ValueError("predicate plans must align with scores")
    max_atoms = int(plan_attrs.shape[1])
    if max_atoms < 1:
        raise ValueError("predicate plans must contain at least one atom")
    states = (
        torch.empty(1, dtype=torch.int8, device=scores.device)
        if use_statistical_groups else
        torch.empty_like(scores, dtype=torch.int8))
    selection_scores = torch.empty_like(scores)
    available_counts = torch.zeros(
        query_count, dtype=torch.int32, device=scores.device)
    block_size = 256
    grid = (query_count, triton.cdiv(n_rows, block_size))
    with torch.cuda.device(scores.device):
        _hard_state_selection_kernel[grid](
            scores, values, missing, plan_attrs, plan_ops, plan_lower,
            plan_upper, atom_counts, states, selection_scores,
            available_counts, statistical_group_ids,
            statistical_decisions,
            n_rows=n_rows, max_atoms=max_atoms,
            global_offset=int(global_offset), block_size=block_size,
            statistical_group_count=int(statistical_group_count),
            use_statistical_groups=bool(use_statistical_groups),
            num_warps=4)
    return (None if use_statistical_groups else states), \
        selection_scores, available_counts


@triton.jit
def _hard_state_block_topk_kernel(
        scores, values, missing, plan_attrs, plan_ops, plan_lower,
        plan_upper, atom_counts, block_scores, block_ids,
        available_counts, statistical_group_ids, statistical_decisions,
        n_rows: tl.constexpr, max_atoms: tl.constexpr,
        block_size: tl.constexpr, block_topk: tl.constexpr,
        block_count: tl.constexpr, statistical_group_count: tl.constexpr,
        global_offset: tl.constexpr):
    query_index = tl.program_id(0)
    block_index = tl.program_id(1)
    row_offsets = block_index * block_size + tl.arange(0, block_size)
    row_valid = row_offsets < n_rows
    atom_count = tl.load(atom_counts + query_index)
    hard_fail = ~row_valid
    unresolved = tl.zeros((block_size,), dtype=tl.int1)
    for atom_index in range(max_atoms):
        atom_valid = atom_index < atom_count
        plan_offset = query_index * max_atoms + atom_index
        attr = tl.load(plan_attrs + plan_offset, mask=atom_valid, other=0)
        op = tl.load(plan_ops + plan_offset, mask=atom_valid, other=-1)
        lower = tl.load(
            plan_lower + plan_offset, mask=atom_valid, other=0.0)
        upper = tl.load(
            plan_upper + plan_offset, mask=atom_valid, other=0.0)
        column_offsets = attr * n_rows + row_offsets
        value = tl.load(
            values + column_offsets, mask=row_valid & atom_valid,
            other=0.0)
        is_missing = tl.load(
            missing + column_offsets, mask=row_valid & atom_valid,
            other=0).to(tl.int1)
        satisfied = (
            ((op == 0) & (value <= lower)) |
            ((op == 1) & (value >= lower)) |
            ((op == 2) & (value >= lower) & (value <= upper)) |
            ((op == 3) & (value == lower)))
        hard_fail |= atom_valid & (~is_missing) & (~satisfied)
        unresolved |= atom_valid & is_missing

    group_id = tl.load(
        statistical_group_ids + row_offsets, mask=row_valid, other=-1)
    valid_group = group_id >= 0
    safe_group = tl.maximum(group_id, 0)
    decision = tl.load(
        statistical_decisions +
        query_index * statistical_group_count + safe_group,
        mask=row_valid & unresolved & valid_group, other=3)
    statistically_accepted = unresolved & valid_group & (decision == 1)
    hard_fail |= unresolved & (~statistically_accepted)
    eligible = row_valid & (~hard_fail)
    score = tl.load(
        scores + query_index * n_rows + row_offsets,
        mask=row_valid, other=float("inf"))
    working = tl.where(eligible, score, float("inf"))
    output_base = (
        (query_index * block_count + block_index) * block_topk)
    for rank in range(block_topk):
        local_position = tl.argmin(
            working, axis=0, tie_break_left=True)
        selected_score = tl.min(working, axis=0)
        selected_id = block_index * block_size + local_position
        finite = selected_score < float("inf")
        tl.store(
            block_scores + output_base + rank, selected_score)
        tl.store(
            block_ids + output_base + rank,
            tl.where(finite, selected_id + global_offset, -1))
        working = tl.where(
            row_offsets == selected_id, float("inf"), working)
    tl.atomic_add(
        available_counts + query_index,
        tl.sum(eligible.to(tl.int32), axis=0))


def fused_hard_block_topk(
        scores: torch.Tensor, values: torch.Tensor,
        missing: torch.Tensor, plan_attrs: torch.Tensor,
        plan_ops: torch.Tensor, plan_lower: torch.Tensor,
        plan_upper: torch.Tensor, atom_counts: torch.Tensor,
        global_offset: int, statistical_group_ids: torch.Tensor,
        statistical_decisions: torch.Tensor, statistical_group_count: int,
        *, block_size: int = 1024, block_topk: int = 16):
    """Fuse hard/statistical classification with per-block top-k reduction."""
    if not scores.is_cuda or scores.dtype != torch.float32 or \
            scores.ndim != 2:
        raise ValueError("scores must be a CUDA float32 matrix")
    query_count, n_rows = map(int, scores.shape)
    block_size = int(block_size)
    block_topk = int(block_topk)
    if block_size < 1 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if block_topk < 1 or block_topk > block_size:
        raise ValueError("block_topk must fit the row block")
    block_count = triton.cdiv(n_rows, block_size)
    block_scores = torch.empty(
        (query_count, block_count * block_topk), dtype=torch.float32,
        device=scores.device)
    block_ids = torch.empty(
        (query_count, block_count * block_topk), dtype=torch.int64,
        device=scores.device)
    available_counts = torch.zeros(
        query_count, dtype=torch.int32, device=scores.device)
    grid = (query_count, block_count)
    with torch.cuda.device(scores.device):
        _hard_state_block_topk_kernel[grid](
            scores, values, missing, plan_attrs, plan_ops, plan_lower,
            plan_upper, atom_counts, block_scores, block_ids,
            available_counts, statistical_group_ids,
            statistical_decisions, n_rows=n_rows,
            max_atoms=int(plan_attrs.shape[1]), block_size=block_size,
            block_topk=block_topk, block_count=block_count,
            statistical_group_count=int(statistical_group_count),
            global_offset=int(global_offset), num_warps=8)
    return block_scores, block_ids, available_counts


@triton.jit
def _hard_eligibility_kernel(
        values, missing, plan_attrs, plan_ops, plan_lower,
        plan_upper, atom_counts, eligibility,
        statistical_group_ids, statistical_decisions,
        n_rows: tl.constexpr, max_atoms: tl.constexpr,
        block_size: tl.constexpr, statistical_group_count: tl.constexpr):
    """Materialize a predicate-only bitmap bank for repeated filters."""
    query_index = tl.program_id(0)
    block_index = tl.program_id(1)
    row_offsets = block_index * block_size + tl.arange(0, block_size)
    row_valid = row_offsets < n_rows
    atom_count = tl.load(atom_counts + query_index)
    hard_fail = ~row_valid
    unresolved = tl.zeros((block_size,), dtype=tl.int1)
    for atom_index in range(max_atoms):
        atom_valid = atom_index < atom_count
        plan_offset = query_index * max_atoms + atom_index
        attr = tl.load(plan_attrs + plan_offset, mask=atom_valid, other=0)
        op = tl.load(plan_ops + plan_offset, mask=atom_valid, other=-1)
        lower = tl.load(
            plan_lower + plan_offset, mask=atom_valid, other=0.0)
        upper = tl.load(
            plan_upper + plan_offset, mask=atom_valid, other=0.0)
        column_offsets = attr * n_rows + row_offsets
        value = tl.load(
            values + column_offsets, mask=row_valid & atom_valid,
            other=0.0)
        is_missing = tl.load(
            missing + column_offsets, mask=row_valid & atom_valid,
            other=0).to(tl.int1)
        satisfied = (
            ((op == 0) & (value <= lower)) |
            ((op == 1) & (value >= lower)) |
            ((op == 2) & (value >= lower) & (value <= upper)) |
            ((op == 3) & (value == lower)))
        hard_fail |= atom_valid & (~is_missing) & (~satisfied)
        unresolved |= atom_valid & is_missing

    group_id = tl.load(
        statistical_group_ids + row_offsets, mask=row_valid, other=-1)
    valid_group = group_id >= 0
    safe_group = tl.maximum(group_id, 0)
    decision = tl.load(
        statistical_decisions +
        query_index * statistical_group_count + safe_group,
        mask=row_valid & unresolved & valid_group, other=3)
    statistically_accepted = unresolved & valid_group & (decision == 1)
    hard_fail |= unresolved & (~statistically_accepted)
    tl.store(
        eligibility + query_index * n_rows + row_offsets,
        row_valid & (~hard_fail), mask=row_valid)


def hard_eligibility_mask(
        values: torch.Tensor, missing: torch.Tensor,
        plan_attrs: torch.Tensor, plan_ops: torch.Tensor,
        plan_lower: torch.Tensor, plan_upper: torch.Tensor,
        atom_counts: torch.Tensor, statistical_group_ids: torch.Tensor,
        statistical_decisions: torch.Tensor,
        statistical_group_count: int):
    """Build exact eligibility once for a repeated-predicate batch."""
    query_count = int(plan_attrs.shape[0])
    n_rows = int(values.shape[1])
    eligibility = torch.empty(
        (query_count, n_rows), dtype=torch.bool, device=values.device)
    block_size = 256
    grid = (query_count, triton.cdiv(n_rows, block_size))
    with torch.cuda.device(values.device):
        _hard_eligibility_kernel[grid](
            values, missing, plan_attrs, plan_ops, plan_lower,
            plan_upper, atom_counts, eligibility,
            statistical_group_ids, statistical_decisions,
            n_rows=n_rows, max_atoms=int(plan_attrs.shape[1]),
            block_size=block_size,
            statistical_group_count=int(statistical_group_count),
            num_warps=4)
    return eligibility


@triton.jit
def _eligibility_block_topk_kernel(
        scores, eligibility, block_scores, block_ids, available_counts,
        n_rows: tl.constexpr, block_size: tl.constexpr,
        block_topk: tl.constexpr, block_count: tl.constexpr,
        global_offset: tl.constexpr):
    """Rank a cached exact eligibility bitmap without re-reading attributes."""
    query_index = tl.program_id(0)
    block_index = tl.program_id(1)
    row_offsets = block_index * block_size + tl.arange(0, block_size)
    row_valid = row_offsets < n_rows
    eligible = tl.load(
        eligibility + query_index * n_rows + row_offsets,
        mask=row_valid, other=0).to(tl.int1)
    score = tl.load(
        scores + query_index * n_rows + row_offsets,
        mask=row_valid, other=float("inf"))
    working = tl.where(row_valid & eligible, score, float("inf"))
    output_base = (
        (query_index * block_count + block_index) * block_topk)
    for rank in range(block_topk):
        local_position = tl.argmin(
            working, axis=0, tie_break_left=True)
        selected_score = tl.min(working, axis=0)
        selected_id = block_index * block_size + local_position
        finite = selected_score < float("inf")
        tl.store(block_scores + output_base + rank, selected_score)
        tl.store(
            block_ids + output_base + rank,
            tl.where(finite, selected_id + global_offset, -1))
        working = tl.where(
            row_offsets == selected_id, float("inf"), working)
    tl.atomic_add(
        available_counts + query_index,
        tl.sum((row_valid & eligible).to(tl.int32), axis=0))


def eligibility_block_topk(
        scores: torch.Tensor, eligibility: torch.Tensor,
        global_offset: int, *, block_size: int = 2048,
        block_topk: int = 10):
    """Top-k per row block using an already materialized exact bitmap."""
    if not scores.is_cuda or scores.dtype not in {torch.float16, torch.float32} or \
            scores.ndim != 2:
        raise ValueError("scores must be a CUDA float16/float32 matrix")
    if eligibility.dtype != torch.bool or eligibility.shape != scores.shape:
        raise ValueError("eligibility must be a bool matrix aligned to scores")
    query_count, n_rows = map(int, scores.shape)
    block_count = triton.cdiv(n_rows, int(block_size))
    block_scores = torch.empty(
        (query_count, block_count * int(block_topk)),
        dtype=scores.dtype, device=scores.device)
    block_ids = torch.empty(
        (query_count, block_count * int(block_topk)),
        dtype=torch.int64, device=scores.device)
    available_counts = torch.zeros(
        query_count, dtype=torch.int32, device=scores.device)
    grid = (query_count, block_count)
    with torch.cuda.device(scores.device):
        _eligibility_block_topk_kernel[grid](
            scores, eligibility, block_scores, block_ids, available_counts,
            n_rows=n_rows, block_size=int(block_size),
            block_topk=int(block_topk), block_count=block_count,
            global_offset=int(global_offset), num_warps=8)
    return block_scores, block_ids, available_counts
