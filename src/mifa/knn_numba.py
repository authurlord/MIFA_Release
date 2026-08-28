"""Numba-fused exact masked top-k kernels.

This is the production form of the insertion kernel first validated in
``scripts/mifa_cpu_bench.py``. It consumes precomputed squared distances, so an
effectiveness experiment can reuse one distance matrix across many eligibility
policies without returning to a Python per-query selection loop.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=False)
def _topk_masked_d2_kernel(d2, eligible, k):
    """Select the k smallest eligible distances in one fused parallel pass."""
    n_queries, n_rows = d2.shape
    ids = np.full((n_queries, k), -1, dtype=np.int64)
    distances = np.full((n_queries, k), np.inf, dtype=np.float32)
    for query in prange(n_queries):
        count = 0
        for row in range(n_rows):
            if not eligible[query, row]:
                continue
            value = d2[query, row]
            if not np.isfinite(value):
                continue
            if count < k:
                pos = count
                while pos > 0 and (
                        distances[query, pos - 1] > value or
                        (distances[query, pos - 1] == value and
                         ids[query, pos - 1] > row)):
                    distances[query, pos] = distances[query, pos - 1]
                    ids[query, pos] = ids[query, pos - 1]
                    pos -= 1
                distances[query, pos] = value
                ids[query, pos] = row
                count += 1
            elif (value < distances[query, k - 1] or
                  (value == distances[query, k - 1] and
                   row < ids[query, k - 1])):
                pos = k - 1
                while pos > 0 and (
                        distances[query, pos - 1] > value or
                        (distances[query, pos - 1] == value and
                         ids[query, pos - 1] > row)):
                    distances[query, pos] = distances[query, pos - 1]
                    ids[query, pos] = ids[query, pos - 1]
                    pos -= 1
                distances[query, pos] = value
                ids[query, pos] = row
    return ids, distances


def topk_masked_batch_numba(d2, eligible, k):
    """Run exact fused masked top-k with deterministic padding and tie order."""
    d2 = np.ascontiguousarray(d2, dtype=np.float32)
    eligible = np.ascontiguousarray(eligible, dtype=np.bool_)
    if d2.ndim != 2 or eligible.ndim != 2 or d2.shape != eligible.shape:
        raise ValueError("d2 and eligible must be same-shaped 2-D arrays")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return (np.empty((d2.shape[0], 0), dtype=np.int64),
                np.empty((d2.shape[0], 0), dtype=np.float32))
    return _topk_masked_d2_kernel(d2, eligible, int(k))

