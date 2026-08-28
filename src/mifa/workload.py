"""Query workload generation: q = (q_e, k, phi).

Query embeddings are sampled from existing tuples (held-out points) so that the
embedding is realistic. Predicates are sampled over query-relevant attrs with a
target selectivity band, so Gate 2 has a non-trivial filter.
"""
from __future__ import annotations

import numpy as np

from mifa.predicates import Atom, Predicate


def _atom_for_attr(spec, attr, rng):
    col = spec.table[attr].dropna()
    if spec.dtypes[attr] == "cat":
        vals = col.unique()
        v = rng.choice(vals)
        return Atom(attr=attr, op="eq", a=type(v)(v) if not isinstance(v, str) else v)
    else:
        lo, hi = np.quantile(col, [0.2, 0.8])
        # a range predicate covering ~middle band
        c = float(rng.uniform(lo, hi))
        op = rng.choice(["le", "ge"])
        return Atom(attr=attr, op=op, a=c)


def sample_workload(spec, n_queries, k, predicate_attrs=None, n_atoms=1, seed=0):
    rng = np.random.default_rng(seed + 99991)
    attrs = predicate_attrs or spec.query_relevant_attrs
    n = spec.n
    q_idx = rng.choice(n, size=min(n_queries, n), replace=False)
    X = spec.normalized_embeddings()
    queries = []
    for qi in q_idx:
        q_e = X[qi].copy()
        chosen = rng.choice(attrs, size=min(n_atoms, len(attrs)), replace=False)
        atoms = [_atom_for_attr(spec, a, rng) for a in chosen]
        queries.append({"q_e": q_e, "k": k, "phi": Predicate(atoms), "src": int(qi)})
    return queries
