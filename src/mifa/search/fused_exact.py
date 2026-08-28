"""Exact masked search paths that avoid a host-resident full distance matrix.

The effectiveness harness intentionally materializes ``(queries, rows)``
distances because many policies reuse them.  An online deployment has one
eligibility policy and need not pay that data-movement cost.  This module keeps
the two execution-specific alternatives separate from the frozen harness:

* CPU: tile the base matrix, use BLAS for each tile, and merge masked top-k in
  Numba without retaining the full distance matrix.
* CUDA: keep the base and repeated eligibility bitmaps resident, run masked
  exact top-k on device, and copy only ``(queries, k)`` IDs back to the host.

Both paths preserve the squared-L2 ranking used by ``mifa.knn``.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=False)
def _merge_masked_score_block(scores, eligibility, row_offset, ids, values):
    """Merge one ``(queries, block_rows)`` score tile into persistent top-k."""
    n_queries, n_block = scores.shape
    k = ids.shape[1]
    for query in prange(n_queries):
        for local_row in range(n_block):
            row = row_offset + local_row
            if not eligibility[query, row]:
                continue
            value = scores[query, local_row]
            if not np.isfinite(value):
                continue
            if (value > values[query, k - 1] or
                    (value == values[query, k - 1] and
                     ids[query, k - 1] >= 0 and
                     row >= ids[query, k - 1])):
                continue
            position = k - 1
            while position > 0 and (
                    values[query, position - 1] > value or
                    (values[query, position - 1] == value and
                     (ids[query, position - 1] < 0 or
                      ids[query, position - 1] > row))):
                values[query, position] = values[query, position - 1]
                ids[query, position] = ids[query, position - 1]
                position -= 1
            values[query, position] = value
            ids[query, position] = row


@njit(parallel=True, cache=True, fastmath=False)
def _merge_grouped_score_block(
        scores, eligibility_bank, group_ids, row_offset, ids, values):
    """Grouped-bank variant that never expands repeated eligibility rows."""
    n_queries, n_block = scores.shape
    k = ids.shape[1]
    for query in prange(n_queries):
        group = group_ids[query]
        for local_row in range(n_block):
            row = row_offset + local_row
            if not eligibility_bank[group, row]:
                continue
            value = scores[query, local_row]
            if not np.isfinite(value):
                continue
            if (value > values[query, k - 1] or
                    (value == values[query, k - 1] and
                     ids[query, k - 1] >= 0 and
                     row >= ids[query, k - 1])):
                continue
            position = k - 1
            while position > 0 and (
                    values[query, position - 1] > value or
                    (values[query, position - 1] == value and
                     (ids[query, position - 1] < 0 or
                      ids[query, position - 1] > row))):
                values[query, position] = values[query, position - 1]
                ids[query, position] = ids[query, position - 1]
                position -= 1
            values[query, position] = value
            ids[query, position] = row


def cpu_blocked_masked_topk(queries, base, base_sqnorm, eligibility, k, *,
                            query_batch=512, row_block=4096):
    """Exact CPU search with ``O(query_batch * row_block)`` score storage.

    It retains regular BLAS GEMMs but consumes each score tile immediately in
    the Numba insertion kernel.  Returned values are the ranking surrogate
    ``||x||² - 2 q·x``; IDs are identical to squared-L2 ranking.
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    base = np.ascontiguousarray(base, dtype=np.float32)
    base_sqnorm = np.ascontiguousarray(base_sqnorm, dtype=np.float32)
    eligibility = np.ascontiguousarray(eligibility, dtype=np.bool_)
    if queries.ndim != 2 or base.ndim != 2 or queries.shape[1] != base.shape[1]:
        raise ValueError("queries and base must be compatible 2-D matrices")
    if eligibility.shape != (len(queries), len(base)):
        raise ValueError("eligibility must have shape (queries, base rows)")
    if base_sqnorm.shape != (len(base),):
        raise ValueError("base_sqnorm must have one value per base row")
    if k <= 0 or k > len(base):
        raise ValueError("k must be in [1, number of base rows]")
    query_batch = max(1, int(query_batch))
    row_block = max(1, int(row_block))
    all_ids = np.full((len(queries), k), -1, dtype=np.int64)
    all_values = np.full((len(queries), k), np.inf, dtype=np.float32)
    for query_start in range(0, len(queries), query_batch):
        query_end = min(query_start + query_batch, len(queries))
        current_queries = queries[query_start:query_end]
        current_eligibility = eligibility[query_start:query_end]
        current_ids = all_ids[query_start:query_end]
        current_values = all_values[query_start:query_end]
        for row_start in range(0, len(base), row_block):
            row_end = min(row_start + row_block, len(base))
            scores = np.asarray(
                current_queries @ base[row_start:row_end].T,
                dtype=np.float32)
            scores *= np.float32(-2.0)
            scores += base_sqnorm[row_start:row_end][None, :]
            _merge_masked_score_block(
                np.ascontiguousarray(scores), current_eligibility,
                row_start, current_ids, current_values)
    return all_ids, all_values


def cpu_blocked_grouped_masked_topk(
        queries, base, base_sqnorm, eligibility_bank, group_ids, k, *,
        query_batch=512, row_block=4096):
    """Blocked CPU exact search consuming a unique eligibility bitmap bank."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    base = np.ascontiguousarray(base, dtype=np.float32)
    base_sqnorm = np.ascontiguousarray(base_sqnorm, dtype=np.float32)
    eligibility_bank = np.ascontiguousarray(
        eligibility_bank, dtype=np.bool_)
    group_ids = np.ascontiguousarray(group_ids, dtype=np.int64)
    if queries.ndim != 2 or base.ndim != 2 or queries.shape[1] != base.shape[1]:
        raise ValueError("queries and base must be compatible 2-D matrices")
    if eligibility_bank.ndim != 2 or eligibility_bank.shape[1] != len(base):
        raise ValueError("eligibility bank width must match base rows")
    if group_ids.shape != (len(queries),):
        raise ValueError("group_ids must have one value per query")
    if len(group_ids) and (group_ids.min() < 0 or
                           group_ids.max() >= len(eligibility_bank)):
        raise ValueError("group_ids index outside eligibility bank")
    if base_sqnorm.shape != (len(base),):
        raise ValueError("base_sqnorm must have one value per base row")
    if k <= 0 or k > len(base):
        raise ValueError("k must be in [1, number of base rows]")
    query_batch = max(1, int(query_batch))
    row_block = max(1, int(row_block))
    all_ids = np.full((len(queries), k), -1, dtype=np.int64)
    all_values = np.full((len(queries), k), np.inf, dtype=np.float32)
    for query_start in range(0, len(queries), query_batch):
        query_end = min(query_start + query_batch, len(queries))
        current_queries = queries[query_start:query_end]
        current_groups = group_ids[query_start:query_end]
        current_ids = all_ids[query_start:query_end]
        current_values = all_values[query_start:query_end]
        for row_start in range(0, len(base), row_block):
            row_end = min(row_start + row_block, len(base))
            scores = np.asarray(
                current_queries @ base[row_start:row_end].T,
                dtype=np.float32)
            scores *= np.float32(-2.0)
            scores += base_sqnorm[row_start:row_end][None, :]
            _merge_grouped_score_block(
                np.ascontiguousarray(scores), eligibility_bank,
                current_groups, row_start, current_ids, current_values)
    return all_ids, all_values


@njit(cache=True, fastmath=False)
def _lower_bound_int64(values, start, end, target):
    while start < end:
        middle = (start + end) // 2
        if values[middle] < target:
            start = middle + 1
        else:
            end = middle
    return start


@njit(parallel=True, cache=True, fastmath=False)
def _merge_grouped_candidate_score_block(
        scores, candidate_offsets, candidate_ids, group_ids, row_offset,
        ids, values):
    """Merge only CSR-listed eligible rows while retaining one dense GEMM."""
    n_queries, n_block = scores.shape
    k = ids.shape[1]
    row_end = row_offset + n_block
    for query in prange(n_queries):
        group = group_ids[query]
        begin = candidate_offsets[group]
        end = candidate_offsets[group + 1]
        begin = _lower_bound_int64(candidate_ids, begin, end, row_offset)
        end = _lower_bound_int64(candidate_ids, begin, end, row_end)
        for position_in_candidates in range(begin, end):
            row = candidate_ids[position_in_candidates]
            value = scores[query, row - row_offset]
            if not np.isfinite(value):
                continue
            if (value > values[query, k - 1] or
                    (value == values[query, k - 1] and
                     ids[query, k - 1] >= 0 and
                     row >= ids[query, k - 1])):
                continue
            position = k - 1
            while position > 0 and (
                    values[query, position - 1] > value or
                    (values[query, position - 1] == value and
                     (ids[query, position - 1] < 0 or
                      ids[query, position - 1] > row))):
                values[query, position] = values[query, position - 1]
                ids[query, position] = ids[query, position - 1]
                position -= 1
            values[query, position] = value
            ids[query, position] = row


class CpuCandidateMergeTopK:
    """Blocked BLAS exact search with CSR eligibility-aware Numba merging."""

    def __init__(self, base, base_sqnorm=None):
        self.base = np.ascontiguousarray(base, dtype=np.float32)
        if self.base.ndim != 2:
            raise ValueError("base must be a 2-D matrix")
        self.base_rows, self.dim = map(int, self.base.shape)
        if base_sqnorm is None:
            base_sqnorm = np.einsum("ij,ij->i", self.base, self.base)
        self.base_sqnorm = np.ascontiguousarray(
            base_sqnorm, dtype=np.float32)
        if self.base_sqnorm.shape != (self.base_rows,):
            raise ValueError("base_sqnorm must have one value per base row")
        self.candidate_offsets = None
        self.candidate_ids = None

    def install_eligibility(self, unique_eligibility):
        bank = np.ascontiguousarray(unique_eligibility, dtype=np.bool_)
        if bank.ndim != 2 or bank.shape[1] != self.base_rows:
            raise ValueError("eligibility bank width must match the base")
        counts = bank.sum(axis=1, dtype=np.int64)
        self.candidate_offsets = np.empty(len(bank) + 1, dtype=np.int64)
        self.candidate_offsets[0] = 0
        np.cumsum(counts, out=self.candidate_offsets[1:])
        self.candidate_ids = np.empty(
            int(self.candidate_offsets[-1]), dtype=np.int64)
        for group, eligible in enumerate(bank):
            start = int(self.candidate_offsets[group])
            end = int(self.candidate_offsets[group + 1])
            self.candidate_ids[start:end] = np.flatnonzero(eligible)

    def search(self, queries, group_ids, k, *, query_batch=512,
               row_block=4096):
        if self.candidate_ids is None:
            raise RuntimeError("install_eligibility must be called before search")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        group_ids = np.ascontiguousarray(group_ids, dtype=np.int64)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width must match the resident base")
        if group_ids.shape != (len(queries),):
            raise ValueError("group_ids must have one value per query")
        if k <= 0 or k > self.base_rows:
            raise ValueError("k must be in [1, number of base rows]")
        if len(group_ids) and (group_ids.min() < 0 or
                               group_ids.max() + 1 >= len(self.candidate_offsets)):
            raise ValueError("group_ids index outside eligibility bank")
        query_batch = max(1, int(query_batch))
        row_block = max(1, int(row_block))
        all_ids = np.full((len(queries), k), -1, dtype=np.int64)
        all_values = np.full((len(queries), k), np.inf, dtype=np.float32)
        for query_start in range(0, len(queries), query_batch):
            query_end = min(query_start + query_batch, len(queries))
            current_queries = queries[query_start:query_end]
            current_groups = group_ids[query_start:query_end]
            current_ids = all_ids[query_start:query_end]
            current_values = all_values[query_start:query_end]
            for row_start in range(0, self.base_rows, row_block):
                row_end = min(row_start + row_block, self.base_rows)
                scores = np.asarray(
                    current_queries @ self.base[row_start:row_end].T,
                    dtype=np.float32)
                scores *= np.float32(-2.0)
                scores += self.base_sqnorm[row_start:row_end][None, :]
                _merge_grouped_candidate_score_block(
                    np.ascontiguousarray(scores), self.candidate_offsets,
                    self.candidate_ids, current_groups, row_start,
                    current_ids, current_values)
        return all_ids, all_values

    @property
    def candidate_rows(self):
        return 0 if self.candidate_ids is None else int(len(self.candidate_ids))

    @property
    def resident_bytes(self):
        total = self.base.nbytes + self.base_sqnorm.nbytes
        if self.candidate_ids is not None:
            total += self.candidate_ids.nbytes + self.candidate_offsets.nbytes
        return int(total)


def compress_repeated_eligibility(eligibility, keys):
    """Return unique bitmap rows and an integer group ID for every query."""
    eligibility = np.ascontiguousarray(eligibility, dtype=np.bool_)
    keys = list(keys)
    if len(keys) != len(eligibility):
        raise ValueError("keys and eligibility must contain the same rows")
    mapping = {}
    unique_rows = []
    group_ids = np.empty(len(keys), dtype=np.int64)
    for index, key in enumerate(keys):
        group = mapping.get(key)
        if group is None:
            group = len(unique_rows)
            mapping[key] = group
            unique_rows.append(eligibility[index])
        group_ids[index] = group
    return np.ascontiguousarray(unique_rows, dtype=np.bool_), group_ids


@njit(cache=True, fastmath=False)
def _candidate_score_topk(scores, candidate_ids, k):
    """Deterministic top-k for a dense score block over selected row IDs."""
    n_queries, n_candidates = scores.shape
    ids = np.full((n_queries, k), -1, dtype=np.int64)
    values = np.full((n_queries, k), np.inf, dtype=np.float32)
    for query in range(n_queries):
        count = 0
        for local_row in range(n_candidates):
            row = candidate_ids[local_row]
            value = scores[query, local_row]
            if not np.isfinite(value):
                continue
            if count < k:
                position = count
                while position > 0 and (
                        values[query, position - 1] > value or
                        (values[query, position - 1] == value and
                         ids[query, position - 1] > row)):
                    values[query, position] = values[query, position - 1]
                    ids[query, position] = ids[query, position - 1]
                    position -= 1
                values[query, position] = value
                ids[query, position] = row
                count += 1
            elif (value < values[query, k - 1] or
                  (value == values[query, k - 1] and
                   row < ids[query, k - 1])):
                position = k - 1
                while position > 0 and (
                        values[query, position - 1] > value or
                        (values[query, position - 1] == value and
                         ids[query, position - 1] > row)):
                    values[query, position] = values[query, position - 1]
                    ids[query, position] = ids[query, position - 1]
                    position -= 1
                values[query, position] = value
                ids[query, position] = row
    return ids, values


class CpuSelectedGroupedTopK:
    """Exact CPU search over precompiled eligible candidate matrices.

    The grouped bitmap path still scores every base row and discards masked
    values afterwards.  Formal predicates are selective and repeat across many
    vector queries, so compiling each unique bitmap into an ascending candidate
    list removes both those dot products and repeated base-row gathers.
    """

    def __init__(self, base, base_sqnorm=None):
        base = np.ascontiguousarray(base, dtype=np.float32)
        if base.ndim != 2:
            raise ValueError("base must be a 2-D matrix")
        self.base = base
        self.base_rows = int(base.shape[0])
        self.dim = int(base.shape[1])
        if base_sqnorm is None:
            base_sqnorm = np.einsum("ij,ij->i", base, base)
        self.base_sqnorm = np.ascontiguousarray(
            base_sqnorm, dtype=np.float32)
        if self.base_sqnorm.shape != (self.base_rows,):
            raise ValueError("base_sqnorm must have one value per base row")
        self.candidate_ids = None
        self.candidate_bases_t = None
        self.candidate_sqnorms = None

    def install_eligibility(self, unique_eligibility):
        bank = np.ascontiguousarray(unique_eligibility, dtype=np.bool_)
        if bank.ndim != 2 or bank.shape[1] != self.base_rows:
            raise ValueError("eligibility bank width must match the base")
        self.candidate_ids = []
        self.candidate_bases_t = []
        self.candidate_sqnorms = []
        for eligible in bank:
            ids = np.flatnonzero(eligible).astype(np.int64)
            self.candidate_ids.append(ids)
            self.candidate_bases_t.append(
                np.ascontiguousarray(self.base[ids].T, dtype=np.float32))
            self.candidate_sqnorms.append(
                np.ascontiguousarray(self.base_sqnorm[ids], dtype=np.float32))

    def search(self, queries, group_ids, k):
        if self.candidate_ids is None:
            raise RuntimeError("install_eligibility must be called before search")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        group_ids = np.ascontiguousarray(group_ids, dtype=np.int64)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width must match the resident base")
        if group_ids.shape != (len(queries),):
            raise ValueError("group_ids must have one value per query")
        if k <= 0 or k > self.base_rows:
            raise ValueError("k must be in [1, number of base rows]")
        if len(group_ids) and (group_ids.min() < 0 or
                               group_ids.max() >= len(self.candidate_ids)):
            raise ValueError("group_ids index outside eligibility bank")
        output_ids = np.full((len(queries), k), -1, dtype=np.int64)
        output_values = np.full((len(queries), k), np.inf, dtype=np.float32)
        q_sqnorm = np.einsum("ij,ij->i", queries, queries)
        for group in np.unique(group_ids):
            query_rows = np.flatnonzero(group_ids == group)
            candidate_ids = self.candidate_ids[int(group)]
            if not len(candidate_ids):
                continue
            scores = np.asarray(
                self.candidate_sqnorms[int(group)][None, :] -
                np.float32(2.0) *
                (queries[query_rows] @ self.candidate_bases_t[int(group)]) +
                q_sqnorm[query_rows, None], dtype=np.float32)
            np.maximum(scores, np.float32(0.0), out=scores)
            ids, values = _candidate_score_topk(
                np.ascontiguousarray(scores), candidate_ids, int(k))
            output_ids[query_rows] = ids
            output_values[query_rows] = values
        return output_ids, output_values

    @property
    def candidate_rows(self):
        if self.candidate_ids is None:
            return 0
        return int(sum(len(ids) for ids in self.candidate_ids))

    @property
    def resident_bytes(self):
        total = self.base.nbytes + self.base_sqnorm.nbytes
        if self.candidate_ids is not None:
            total += sum(ids.nbytes for ids in self.candidate_ids)
            total += sum(base.nbytes for base in self.candidate_bases_t)
            total += sum(norm.nbytes for norm in self.candidate_sqnorms)
        return int(total)


class TorchResidentMaskedTopK:
    """CUDA exact search with resident base and repeated eligibility bitmaps."""

    def __init__(self, base, base_sqnorm=None, *, device="cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TorchResidentMaskedTopK")
        self.torch = torch
        self.device = torch.device(device)
        base = np.ascontiguousarray(base, dtype=np.float32)
        self.base_rows = int(base.shape[0])
        self.dim = int(base.shape[1])
        if base_sqnorm is None:
            base_sqnorm = np.einsum("ij,ij->i", base, base)
        base_sqnorm = np.ascontiguousarray(base_sqnorm, dtype=np.float32)
        if base_sqnorm.shape != (self.base_rows,):
            raise ValueError("base_sqnorm must have one value per base row")
        base_device = torch.from_numpy(base).to(self.device)
        self.base_t = base_device.T.contiguous()
        # Match ``query_dists_batch`` exactly: its norm is computed on CPU and
        # transferred, so recomputing the reduction on CUDA can move near-ties.
        self.base_sqnorm = torch.from_numpy(base_sqnorm).to(self.device)
        self.eligibility_bank = None
        torch.cuda.synchronize(self.device)

    def install_eligibility(self, unique_eligibility):
        unique_eligibility = np.ascontiguousarray(
            unique_eligibility, dtype=np.bool_)
        if unique_eligibility.ndim != 2 or \
                unique_eligibility.shape[1] != self.base_rows:
            raise ValueError("eligibility bank width must match the base")
        self.eligibility_bank = self.torch.from_numpy(
            unique_eligibility).to(self.device)
        self.torch.cuda.synchronize(self.device)

    def _deterministic_topk(self, scores, k):
        """Fast top-k with stable row-ID tie repair only where necessary."""
        values, ids = self.torch.topk(
            scores, k, dim=1, largest=False, sorted=True)
        if k > 1:
            internal_tie = (values[:, 1:] == values[:, :-1]).any(1)
        else:
            internal_tie = self.torch.zeros(
                len(values), dtype=self.torch.bool, device=self.device)
        boundary = values[:, -1]
        selected_boundary = (values == boundary[:, None]).sum(1)
        all_boundary = (scores == boundary[:, None]).sum(1)
        repair = internal_tie | (all_boundary > selected_boundary)
        if repair.any():
            # Stable sort preserves the original ascending base-row order for
            # equal float32 distances, matching the Numba reference tie rule.
            stable_ids = self.torch.argsort(
                scores[repair], dim=1, stable=True)[:, :k]
            stable_values = self.torch.gather(
                scores[repair], 1, stable_ids)
            ids[repair] = stable_ids
            values[repair] = stable_values
        return values, ids

    def search(self, queries, group_ids, k, *, batch_size=512):
        if self.eligibility_bank is None:
            raise RuntimeError("install_eligibility must be called before search")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        group_ids = np.ascontiguousarray(group_ids, dtype=np.int64)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width must match the resident base")
        if group_ids.shape != (len(queries),):
            raise ValueError("group_ids must have one value per query")
        output = np.full((len(queries), k), -1, dtype=np.int64)
        for start in range(0, len(queries), int(batch_size)):
            end = min(start + int(batch_size), len(queries))
            query_device = self.torch.from_numpy(
                queries[start:end]).to(self.device)
            groups_device = self.torch.from_numpy(
                group_ids[start:end]).to(self.device)
            scores = (self.base_sqnorm[None, :] -
                      2.0 * (query_device @ self.base_t) +
                      (query_device * query_device).sum(1, keepdim=True))
            scores.clamp_min_(0.0)
            scores.masked_fill_(
                ~self.eligibility_bank[groups_device], float("inf"))
            values, ids = self._deterministic_topk(scores, k)
            ids = ids.masked_fill(~self.torch.isfinite(values), -1)
            output[start:end] = ids.cpu().numpy()
        self.torch.cuda.synchronize(self.device)
        return output

    @property
    def resident_bytes(self):
        total = (self.base_t.numel() * self.base_t.element_size() +
                 self.base_sqnorm.numel() * self.base_sqnorm.element_size())
        if self.eligibility_bank is not None:
            total += (self.eligibility_bank.numel() *
                      self.eligibility_bank.element_size())
        return int(total)


class TorchResidentBoundMaskedTopK(TorchResidentMaskedTopK):
    """CUDA resident-mask path with a bound request stream and fused addmm.

    A steady online batch repeatedly maps the same canonical predicates to the
    installed bitmap bank.  Binding that group vector once avoids transferring
    group IDs and gathering a full mask on every search.  Ranking on
    ``||x||^2 - 2 q.x`` also removes the query-norm and clamp kernels; the
    omitted term is constant within every query row.
    """

    def __init__(self, base, base_sqnorm=None, *, device="cuda"):
        super().__init__(base, base_sqnorm, device=device)
        self.bound_group_ids = None
        self.bound_eligibility = None

    def install_eligibility(self, unique_eligibility):
        super().install_eligibility(unique_eligibility)
        self.bound_group_ids = None
        self.bound_eligibility = None

    def bind_groups(self, group_ids):
        if self.eligibility_bank is None:
            raise RuntimeError("install_eligibility must be called before bind_groups")
        group_ids = np.ascontiguousarray(group_ids, dtype=np.int64)
        if len(group_ids) and (group_ids.min() < 0 or
                               group_ids.max() >= len(self.eligibility_bank)):
            raise ValueError("group_ids index outside eligibility bank")
        groups_device = self.torch.from_numpy(group_ids).to(self.device)
        self.bound_eligibility = self.eligibility_bank[
            groups_device].contiguous()
        self.bound_group_ids = group_ids.copy()
        self.torch.cuda.synchronize(self.device)

    def search(self, queries, group_ids, k, *, batch_size=512):
        if self.eligibility_bank is None:
            raise RuntimeError("install_eligibility must be called before search")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        group_ids = np.ascontiguousarray(group_ids, dtype=np.int64)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width must match the resident base")
        if group_ids.shape != (len(queries),):
            raise ValueError("group_ids must have one value per query")
        output = np.full((len(queries), k), -1, dtype=np.int64)
        bound = (self.bound_group_ids is not None and
                 np.array_equal(group_ids, self.bound_group_ids))
        groups_device = None
        if not bound:
            groups_device = self.torch.from_numpy(group_ids).to(self.device)
        for start in range(0, len(queries), int(batch_size)):
            end = min(start + int(batch_size), len(queries))
            query_device = self.torch.from_numpy(
                queries[start:end]).to(self.device)
            bias = self.base_sqnorm[None, :].expand(end - start, -1)
            scores = self.torch.addmm(
                bias, query_device, self.base_t, beta=1.0, alpha=-2.0)
            if bound:
                eligible = self.bound_eligibility[start:end]
            else:
                eligible = self.eligibility_bank[groups_device[start:end]]
            scores.masked_fill_(~eligible, float("inf"))
            values, ids = self._deterministic_topk(scores, k)
            ids = ids.masked_fill(~self.torch.isfinite(values), -1)
            output[start:end] = ids.cpu().numpy()
        self.torch.cuda.synchronize(self.device)
        return output

    @property
    def resident_bytes(self):
        total = super().resident_bytes
        if self.bound_eligibility is not None:
            total += (self.bound_eligibility.numel() *
                      self.bound_eligibility.element_size())
        return int(total)


class TorchResidentSelectedTopK(TorchResidentMaskedTopK):
    """CUDA exact top-k over candidate matrices compiled from bitmap groups.

    Queries and output IDs cross PCIe once per invocation.  Sorting queries by
    group makes every per-group GEMM operate on a contiguous device slice; the
    only remaining trade-off is grouped kernel-launch overhead versus the dot
    products removed by the low-selectivity filters.
    """

    def __init__(self, base, base_sqnorm=None, *, device="cuda"):
        super().__init__(base, base_sqnorm, device=device)
        self.candidate_ids = None
        self.candidate_bases_t = None
        self.candidate_sqnorms = None

    def install_eligibility(self, unique_eligibility):
        unique_eligibility = np.ascontiguousarray(
            unique_eligibility, dtype=np.bool_)
        if unique_eligibility.ndim != 2 or \
                unique_eligibility.shape[1] != self.base_rows:
            raise ValueError("eligibility bank width must match the base")
        self.candidate_ids = []
        self.candidate_bases_t = []
        self.candidate_sqnorms = []
        for eligible in unique_eligibility:
            ids = np.flatnonzero(eligible).astype(np.int64)
            ids_device = self.torch.from_numpy(ids).to(self.device)
            self.candidate_ids.append(ids_device)
            self.candidate_bases_t.append(
                self.base_t[:, ids_device].contiguous())
            self.candidate_sqnorms.append(
                self.base_sqnorm[ids_device].contiguous())
        self.torch.cuda.synchronize(self.device)

    def search(self, queries, group_ids, k, *, batch_size=None):
        del batch_size  # retained for call-site compatibility
        if self.candidate_ids is None:
            raise RuntimeError("install_eligibility must be called before search")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        group_ids = np.ascontiguousarray(group_ids, dtype=np.int64)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width must match the resident base")
        if group_ids.shape != (len(queries),):
            raise ValueError("group_ids must have one value per query")
        if k <= 0 or k > self.base_rows:
            raise ValueError("k must be in [1, number of base rows]")
        if len(group_ids) and (group_ids.min() < 0 or
                               group_ids.max() >= len(self.candidate_ids)):
            raise ValueError("group_ids index outside eligibility bank")

        # Stable sorting preserves query order within each group and requires
        # only one query transfer plus one result transfer per invocation.
        order = np.argsort(group_ids, kind="stable").astype(np.int64)
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order), dtype=np.int64)
        sorted_groups = group_ids[order]
        query_device = self.torch.from_numpy(queries).to(self.device)
        order_device = self.torch.from_numpy(order).to(self.device)
        sorted_queries = query_device[order_device]
        sorted_output = self.torch.full(
            (len(queries), k), -1, dtype=self.torch.int64,
            device=self.device)
        start = 0
        while start < len(sorted_groups):
            group = int(sorted_groups[start])
            end = int(np.searchsorted(sorted_groups, group, side="right"))
            candidate_ids = self.candidate_ids[group]
            if len(candidate_ids):
                current = sorted_queries[start:end]
                scores = (
                    self.candidate_sqnorms[group][None, :] -
                    2.0 * (current @ self.candidate_bases_t[group]) +
                    (current * current).sum(1, keepdim=True))
                scores.clamp_min_(0.0)
                values, local_ids = self._deterministic_topk(
                    scores, min(k, len(candidate_ids)))
                current_ids = candidate_ids[local_ids]
                current_ids = current_ids.masked_fill(
                    ~self.torch.isfinite(values), -1)
                sorted_output[start:end, :current_ids.shape[1]] = current_ids
            start = end
        output = sorted_output.cpu().numpy()[inverse]
        self.torch.cuda.synchronize(self.device)
        return output

    @property
    def candidate_rows(self):
        if self.candidate_ids is None:
            return 0
        return int(sum(ids.numel() for ids in self.candidate_ids))

    @property
    def resident_bytes(self):
        total = super().resident_bytes
        if self.candidate_ids is not None:
            total += sum(ids.numel() * ids.element_size()
                         for ids in self.candidate_ids)
            total += sum(base.numel() * base.element_size()
                         for base in self.candidate_bases_t)
            total += sum(norm.numel() * norm.element_size()
                         for norm in self.candidate_sqnorms)
        return int(total)
