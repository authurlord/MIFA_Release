"""Certificate-guided filtered ANN search over a projected grid (Phase 2b).

Pipeline per query q=(q_e,k,phi):
  1. HNSW seeds -> exact-score the PASS seeds -> initial top-k buffer + boundary tau_B.
  2. Expand grid cells in increasing lower-bound (LB) order.
  3. PRUNE a cell (and, since sorted, all remaining) once LB > tau_B: no tuple in it
     can enter top-k (LB is admissible via the orthonormal projection).
  4. In a visited cell: fail tuples are dropped; pass tuples are scored and may
     update the buffer; UNCERTAIN tuples are FINE-REFINED only if they could still
     enter top-k (d < tau_B) -- otherwise refinement is skipped.

Fine refinement = per-tuple embedding-kNN over the OBSERVED tuples of the missing
attribute (the expensive inference that the embedding-kNN baseline pays for EVERY
masked cell; MIFA pays it only for boundary-uncertain tuples).

Toggles `prune` and `skip_refine` let the same code produce the no-pruning /
refine-all baseline for the efficiency ablation. Counters record all work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mifa.knn import knn_batch
from mifa.predicates import Predicate, TRUE, FALSE, UNKNOWN


class FineRefiner:
    """MIFA's missing-value inference index I_mv.

    Two modes:
      - "eager" (ablation = first-impute-then-query): in __init__, the embedding-kNN
        of EVERY missing tuple over the observed pool is precomputed for ALL query-
        relevant attrs. Query-time refinement is an O(k) lookup. This pays the same
        heavy offline kNN as the embknn baseline.
      - "lazy" (paper-faithful): __init__ stores only embeddings + per-attr observed
        pool (indices + values). NO kNN precompute. `refine_lazy` computes the
        embedding-kNN ON DEMAND, for exactly the boundary-surviving uncertain tuples
        and ONLY for the attributes referenced by the current predicate. Heavy
        inference work per query is tracked in `self.qcost`.
    """

    def __init__(self, spec, observed_table, knn=10, mode="eager", X=None):
        self.spec = spec
        self.obs = observed_table
        self.knn = knn
        self.mode = mode
        # X override lets callers build I_mv in a reduced embedding space (dim sweep),
        # keeping search and inference self-consistent in the SAME space.
        self.X = spec.normalized_embeddings() if X is None else np.ascontiguousarray(X, np.float32)
        self.qcost = {"refined_tuples": 0, "knn_dotprods": 0}
        # per-attr observed pool (needed by both modes; lazy uses only this)
        self.pool = {}     # attr -> dict(obs_idx, y_obs)
        for a in spec.query_relevant_attrs:
            observed = ~observed_table[a].isna().to_numpy()
            obs_idx = np.where(observed)[0]
            self.pool[a] = {"obs_idx": obs_idx,
                            "y_obs": spec.table[a].to_numpy()[obs_idx]}
        self.attr = {}     # eager only: attr -> dict(pos:(n,), neigh_vals:(m,k))
        if mode == "eager":
            for a in spec.query_relevant_attrs:
                observed = ~observed_table[a].isna().to_numpy()
                miss_idx = np.where(~observed)[0]
                obs_idx = self.pool[a]["obs_idx"]
                pos = np.full(spec.n, -1, dtype=np.int64)
                if len(miss_idx) and len(obs_idx):
                    nn, _ = knn_batch(self.X[miss_idx], self.X[obs_idx], knn)
                    neigh_vals = self.pool[a]["y_obs"][nn]          # (m, k)
                    pos[miss_idx] = np.arange(len(miss_idx))
                else:
                    neigh_vals = np.empty((0, knn), dtype=object)
                self.attr[a] = {"pos": pos, "neigh_vals": neigh_vals}

    def reset_qcost(self):
        self.qcost = {"refined_tuples": 0, "knn_dotprods": 0}

    def dotprod_cost(self, tuple_ids, predicate):
        """Analytic lazy inference cost for refining `tuple_ids` under `predicate`:
        sum over predicate atoms of (#tuples MISSING that attr) * |observed pool|.
        No kNN is run -- this is what lazy mode WOULD pay, computed cheaply so the
        fast eager search can report the lazy cost. Predicate-scoped by construction."""
        tuple_ids = np.asarray(tuple_ids)
        total = 0
        for at in predicate.atoms:
            observed = ~self.obs[at.attr].isna().to_numpy()
            n_miss = int((~observed[tuple_ids]).sum())
            total += n_miss * int(len(self.pool[at.attr]["obs_idx"]))
        return total

    def refine_lazy(self, tuple_ids, predicate: Predicate):
        """On-demand, predicate-scoped p_phi for `tuple_ids` (uncertain survivors).

        For each atom whose attr is MISSING on a tuple, run an embedding-kNN of just
        those tuples against that attr's observed pool, then the per-atom satisfaction
        fraction. Only predicate-referenced attrs are ever inferred. Counts work."""
        tuple_ids = np.asarray(tuple_ids)
        p = np.ones(len(tuple_ids), dtype=np.float64)
        counted = np.zeros(len(tuple_ids), dtype=bool)
        for at in predicate.atoms:
            col = self.obs[at.attr]
            observed = ~col.isna().to_numpy()
            vals = col.to_numpy()
            obs_here = observed[tuple_ids]
            sat_here = Predicate._atom_sat(at, vals[tuple_ids], ~obs_here)
            p = np.where(obs_here & ~sat_here, 0.0, p)
            miss = ~obs_here
            if miss.any():
                pool = self.pool[at.attr]
                obs_idx, y_obs = pool["obs_idx"], pool["y_obs"]
                miss_ids = tuple_ids[miss]
                nn, _ = knn_batch(self.X[miss_ids], self.X[obs_idx], self.knn)
                self.qcost["knn_dotprods"] += int(len(miss_ids) * len(obs_idx))
                counted[miss] = True
                sat = Predicate._atom_sat(at, y_obs, np.zeros(len(y_obs), dtype=bool))
                frac = sat[nn].mean(axis=1)
                p[miss] = p[miss] * frac
        self.qcost["refined_tuples"] += int(counted.sum())
        return p

    def impute_point(self, observed_table):
        """Build a completed table via per-tuple neighbour point estimate (majority
        vote / mean), reusing the offline neighbours. This is the embedding-kNN
        first-impute-then-query reference, with NO extra kNN work."""
        out = observed_table.copy()
        for a, data in self.attr.items():
            pos = data["pos"]; nv = data["neigh_vals"]
            miss_idx = np.where(pos >= 0)[0]
            if len(miss_idx) == 0:
                continue
            rows = pos[miss_idx]
            if out[a].dtype == bool or pd.api.types.is_integer_dtype(self.spec.table[a].dtype):
                out[a] = out[a].astype(object)
            if self.spec.dtypes[a] == "cat":
                # R5: vectorized per-row majority vote (no per-tuple np.unique loop).
                # np.unique gives SORTED uniques + inverse codes, so the bincount-argmax
                # mode breaks ties toward the smallest value -- bit-identical to the old
                # per-row `np.unique(...); vals[argmax(counts)]` path.
                sub = nv[rows]                              # (m, k) neighbour values
                uniq, codes = np.unique(sub.reshape(-1), return_inverse=True)
                codes = codes.reshape(sub.shape)            # (m, k) int codes
                m, U = codes.shape[0], len(uniq)
                offset = codes + np.arange(m)[:, None] * U
                counts = np.bincount(offset.reshape(-1), minlength=m * U).reshape(m, U)
                out.loc[miss_idx, a] = uniq[counts.argmax(1)]
            else:
                out.loc[miss_idx, a] = nv[rows].astype(float).mean(axis=1)
        return out

    def p_phi(self, tuple_ids, predicate: Predicate):
        """p_phi per tuple (conjunction). Observed atoms -> hard 0/1; missing atoms
        -> fraction of the precomputed k neighbours satisfying the atom (lookup)."""
        tuple_ids = np.asarray(tuple_ids)
        p = np.ones(len(tuple_ids), dtype=np.float64)
        for at in predicate.atoms:
            col = self.obs[at.attr]
            observed = ~col.isna().to_numpy()
            vals = col.to_numpy()
            obs_here = observed[tuple_ids]
            sat_here = Predicate._atom_sat(at, vals[tuple_ids], ~obs_here)
            p = np.where(obs_here & ~sat_here, 0.0, p)
            miss = ~obs_here
            if miss.any():
                data = self.attr[at.attr]
                rows = data["pos"][tuple_ids[miss]]            # (n_miss,)
                nv = data["neigh_vals"][rows]                  # (n_miss, k)
                flat = nv.reshape(-1)
                sat = Predicate._atom_sat(at, flat, np.zeros(len(flat), dtype=bool))
                frac = sat.reshape(nv.shape).mean(axis=1)
                p[miss] = p[miss] * frac
        return p


def grid_search(q_e, k, predicate, *, grid, X, base_sqnorm, cert, refiner,
                theta, hnsw=None, ef=64, prune=True, skip_refine=True):
    """Run one query. cert = dict from mifa_engine.certificate (pass/fail/uncertain).
    Returns (topk_ids sorted by distance, counters dict)."""
    n = X.shape[0]
    c = dict(visited_cells=0, pruned_cells=0, cell_lb_evals=0,
             exact_distance_computations=0, uncertain_seen=0,
             uncertain_refined=0, refinement_skipped_by_tau=0, seed_scored=0,
             lazy_knn_dotprods=0)
    pass_mask = cert["pass_mask"]
    fail_mask = cert["fail_mask"]
    unc_mask = cert["uncertain_mask"]

    cand_id, cand_d2 = [], []
    INF = np.inf
    tau2 = INF

    def push(ids, d2):
        cand_id.extend(int(i) for i in ids)
        cand_d2.extend(float(x) for x in d2)

    def tau_update():
        nonlocal tau2
        if len(cand_d2) >= k:
            arr = np.asarray(cand_d2)
            tau2 = float(np.partition(arr, k - 1)[k - 1])

    qq = float(q_e @ q_e)

    # 1. HNSW seeding: score PASS seeds to get an initial boundary.
    if hnsw is not None:
        seeds, _ = hnsw.search(q_e, ef)
        seeds = np.asarray(seeds)
        sp = np.unique(seeds[pass_mask[seeds]])
        if len(sp):
            d2 = base_sqnorm[sp] - 2.0 * (X[sp] @ q_e) + qq
            push(sp, np.maximum(d2, 0.0))
            c["seed_scored"] += len(sp)
            c["exact_distance_computations"] += len(sp)
            tau_update()

    # 2. order cells by admissible LB
    q_grid = grid.project(q_e)
    lb2 = grid.cell_lower_bounds_sq(q_grid)
    c["cell_lb_evals"] += len(lb2)
    order = np.argsort(lb2)

    seed_set = np.asarray(sorted(seen_seeds)) if (seen_seeds := set(cand_id)) else None

    def drop_seeds(ids, d2):
        if seed_set is None or len(ids) == 0:
            return ids, d2
        keep = ~np.isin(ids, seed_set)
        return ids[keep], d2[keep]

    for pos_i, ci in enumerate(order):
        # cells are sorted by ascending LB: once one exceeds tau_B, so do all the
        # rest -> tail-prune (break), not per-cell continue.
        if prune and len(cand_d2) >= k and lb2[ci] > tau2:
            c["pruned_cells"] += float(len(order) - pos_i)
            break
        c["visited_cells"] += 1
        mem = grid.cell_members[ci]
        mem = mem[~fail_mask[mem]]                 # fail tuples need no distance
        if len(mem) == 0:
            continue
        d2 = np.maximum(base_sqnorm[mem] - 2.0 * (X[mem] @ q_e) + qq, 0.0)
        c["exact_distance_computations"] += len(mem)

        pm = pass_mask[mem]
        if pm.any():
            ids, dd = drop_seeds(mem[pm], d2[pm]); push(ids, dd)

        um = unc_mask[mem]
        if um.any():
            u_ids = mem[um]; u_d2 = d2[um]
            c["uncertain_seen"] += len(u_ids)
            if skip_refine and len(cand_d2) >= k:
                survive = u_d2 < tau2
            else:
                survive = np.ones(len(u_ids), dtype=bool)
            c["refinement_skipped_by_tau"] += int((~survive).sum())
            su_ids, su_d2 = u_ids[survive], u_d2[survive]
            if len(su_ids):
                c["uncertain_refined"] += len(su_ids)
                # analytic lazy inference cost (predicate-scoped), regardless of mode
                if hasattr(refiner, "dotprod_cost"):
                    c["lazy_knn_dotprods"] += refiner.dotprod_cost(su_ids, predicate)
                p = (refiner.refine_lazy(su_ids, predicate)
                     if getattr(refiner, "mode", "eager") == "lazy"
                     else refiner.p_phi(su_ids, predicate))
                elig = p >= theta
                ids, dd = drop_seeds(su_ids[elig], su_d2[elig]); push(ids, dd)
        tau_update()

    if not cand_id:
        return np.array([], dtype=int), c
    d2arr = np.asarray(cand_d2); idarr = np.asarray(cand_id)
    kk = min(k, len(idarr))
    top = np.argpartition(d2arr, kk - 1)[:kk]
    top = top[np.argsort(d2arr[top])]
    return idarr[top], c
