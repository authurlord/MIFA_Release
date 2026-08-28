"""Shared end-to-end efficiency accounting for paper experiments.

The paper separates online latency from construction/materialization, while
requiring end-to-end comparisons to amortize refresh work over an explicitly
stated number of queries per refresh.  Keeping that arithmetic in one module
prevents a search-only or precomputed-distance number from being mislabeled as
an end-to-end speedup.
"""
from __future__ import annotations

import math


def amortized_latency(
        *, measured_queries: int, predicate_batch_seconds: float,
        search_batch_seconds: float, queries_per_refresh: int,
        fit_seconds: float = 0.0, materialization_seconds: float = 0.0,
        refresh_index_seconds: float = 0.0) -> dict:
    """Return a component-exact amortized latency record.

    ``predicate_batch_seconds`` and ``search_batch_seconds`` must cover the
    same measured online query batch.  Fit, materialization, and index refresh
    are one refresh epoch's costs and are amortized over
    ``queries_per_refresh``.  Initial construction that is not repeated at
    refresh belongs in a separate report field and must not be smuggled into
    one of these arguments.
    """
    measured_queries = int(measured_queries)
    queries_per_refresh = int(queries_per_refresh)
    if measured_queries <= 0:
        raise ValueError("measured_queries must be positive")
    if queries_per_refresh <= 0:
        raise ValueError("queries_per_refresh must be positive")
    components = {
        "predicate_batch_seconds": float(predicate_batch_seconds),
        "search_batch_seconds": float(search_batch_seconds),
        "fit_seconds_per_refresh": float(fit_seconds),
        "materialization_seconds_per_refresh": float(materialization_seconds),
        "index_seconds_per_refresh": float(refresh_index_seconds),
    }
    if any(not math.isfinite(value) or value < 0.0
           for value in components.values()):
        raise ValueError("all timing components must be finite and non-negative")
    predicate_ms = 1000.0 * components["predicate_batch_seconds"] / measured_queries
    search_ms = 1000.0 * components["search_batch_seconds"] / measured_queries
    fit_ms = 1000.0 * components["fit_seconds_per_refresh"] / queries_per_refresh
    materialization_ms = (
        1000.0 * components["materialization_seconds_per_refresh"] /
        queries_per_refresh)
    index_ms = (
        1000.0 * components["index_seconds_per_refresh"] /
        queries_per_refresh)
    online_ms = predicate_ms + search_ms
    refresh_ms = fit_ms + materialization_ms + index_ms
    total_ms = online_ms + refresh_ms
    return {
        "measured_queries": measured_queries,
        "queries_per_refresh": queries_per_refresh,
        "predicate_ms_per_query": predicate_ms,
        "search_ms_per_query": search_ms,
        "online_ms_per_query": online_ms,
        "amortized_fit_ms_per_query": fit_ms,
        "amortized_materialization_ms_per_query": materialization_ms,
        "amortized_index_refresh_ms_per_query": index_ms,
        "amortized_refresh_ms_per_query": refresh_ms,
        "end_to_end_ms_per_query": total_ms,
        "end_to_end_qps": 1000.0 / total_ms if total_ms > 0.0 else math.inf,
        "components_seconds": components,
    }


def amortized_latency_curve(*, queries_per_refresh, **components) -> list[dict]:
    """Evaluate :func:`amortized_latency` over a preregistered cadence grid."""
    cadence = tuple(int(value) for value in queries_per_refresh)
    if not cadence:
        raise ValueError("queries_per_refresh grid must be non-empty")
    if len(set(cadence)) != len(cadence):
        raise ValueError("queries_per_refresh grid must not contain duplicates")
    return [
        amortized_latency(
            queries_per_refresh=value, **components)
        for value in cadence
    ]
