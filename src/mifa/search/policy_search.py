"""Baseline policies on the incomplete relation D: Drop and default-fill.

Both reuse the same exact embedding-distance ranking so that quality differences
are attributable purely to the missingness policy, not the ANN backend.
  - Drop:         tuple with ANY UNKNOWN atom -> ineligible (false negatives).
  - default-fill: fill missing so the predicate PASSES (false positives).
"""
from __future__ import annotations

import numpy as np

from mifa.predicates import TRUE, UNKNOWN


def _rank_eligible(spec, q_e, k, eligible_ids):
    if len(eligible_ids) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    X = spec.normalized_embeddings()
    diff = X[eligible_ids] - q_e[None, :]
    d2 = np.einsum("ij,ij->i", diff, diff)
    kk = min(k, len(d2))
    part = np.argpartition(d2, kk - 1)[:kk]
    order = part[np.argsort(d2[part])]
    return np.asarray(eligible_ids)[order], d2[order]


def drop_search(spec, observed_table, q_e, k, predicate):
    verdict = predicate.eval_table(observed_table)
    eligible = np.where(verdict == TRUE)[0]      # UNKNOWN dropped
    return _rank_eligible(spec, q_e, k, eligible)


def default_fill_search(spec, observed_table, q_e, k, predicate):
    """Treat UNKNOWN as satisfying the predicate (optimistic fill)."""
    verdict = predicate.eval_table(observed_table)
    eligible = np.where((verdict == TRUE) | (verdict == UNKNOWN))[0]
    return _rank_eligible(spec, q_e, k, eligible)
