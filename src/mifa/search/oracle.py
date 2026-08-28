"""Exact filtered KNN oracle on D* (the reference answer Ans*(q, D*))."""
from __future__ import annotations

import numpy as np

from mifa.predicates import TRUE


def exact_filtered_knn(spec, q_e, k, predicate, X=None, verdict=None):
    """Brute-force: among tuples whose COMPLETE attrs satisfy phi (TRUE), the k
    closest by embedding distance. This is Ans*(q, D*).

    X and verdict can be precomputed and passed in (the predicate verdict over the
    complete table does not depend on q_e, so callers iterating many queries with
    the same predicate family should cache them)."""
    if X is None:
        X = spec.normalized_embeddings()
    if verdict is None:
        verdict = predicate.eval_table(spec.table)
    elig = np.where(verdict == TRUE)[0]
    if len(elig) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    from mifa.knn import knn_batch
    ids, d2 = knn_batch(q_e[None, :], X[elig], min(k, len(elig)))
    return elig[ids[0]], d2[0]
