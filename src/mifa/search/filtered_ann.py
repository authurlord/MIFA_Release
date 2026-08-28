"""R1 — the SYSTEMS half: a real filtered-ANN harness (not brute-force relabeled).

The quality half (rule_compare.py) ranks by exact brute force to ISOLATE eligibility
quality. This module instead answers the top-k under a real ANN index whose graph is
traversed during search, and measures recall vs QPS — the FANNS-community standard.

Index: a self-contained NSW graph (precomputed k-NN adjacency, one matmul to build),
traversed greedily best-first with a distance-ordered frontier. We own the traversal so
eligibility can be applied DURING search (hnswlib exposes no graph internals). Every
squared-L2 evaluation is counted (`n_dist`), so "exact distance computations" is honest.

Three execution families (all use the SAME graph, compared on equal footing):
  - pre-filter : restrict to the eligible subset, then exact top-k over it (the classic
                 "filter first" system; distance count = |eligible|).
  - post-filter: unfiltered greedy ANN to ef candidates, then keep the eligible ones;
                 enlarge ef until k eligible are found (over-retrieval).
  - in-filter  : greedy traversal over the full graph, eligibility applied at result
                 ACCEPTANCE (ineligible nodes are still traversed for connectivity) —
                 the paper's certificate-guided design (ACORN-style filter-during-search).

Ground truth is always exact filtered top-k on the COMPLETE relation D* (Ans*).
"""
from __future__ import annotations

import heapq
import numpy as np

from mifa.knn import knn_batch


class NSWGraph:
    """Precomputed k-NN proximity graph + greedy best-first search."""

    def __init__(self, M=32, n_entry=8, seed=0):
        self.M = M
        self.n_entry = n_entry
        self.seed = seed

    def build(self, X):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.bsq = np.einsum("ij,ij->i", self.X, self.X)
        # one batched matmul builds the directed k-NN adjacency (exclude self at col 0)
        nn, _ = knn_batch(self.X, self.X, self.M + 1)
        out = nn[:, 1:].astype(np.int32)
        # SYMMETRIZE: a flat k-NN graph has poor navigability (no long-range/back edges);
        # adding reverse edges makes greedy best-first search reach ~HNSW recall. Store as
        # a ragged CSR adjacency (degrees vary after the union).
        rev = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in out[i]:
                rev[int(j)].append(i)
        adj = []
        for i in range(self.n):
            nbrs = np.unique(np.concatenate([out[i], np.asarray(rev[i], dtype=np.int32)])
                             if rev[i] else out[i])
            adj.append(nbrs[nbrs != i].astype(np.int32))
        self.indptr = np.zeros(self.n + 1, dtype=np.int64)
        self.indptr[1:] = np.cumsum([len(a) for a in adj])
        self.indices = (np.concatenate(adj) if self.n else np.array([], np.int32)).astype(np.int32)
        rng = np.random.default_rng(self.seed)
        self.entries = rng.choice(self.n, size=min(self.n_entry, self.n), replace=False)
        return self

    def _nbrs(self, i):
        return self.indices[self.indptr[i]:self.indptr[i + 1]]

    def _d2(self, q, qq, i, ctr):
        ctr[0] += 1
        return float(self.bsq[i] - 2.0 * (self.X[i] @ q) + qq)

    def greedy(self, q, k, ef, accept, ctr, max_visits=None):
        """Best-first NSW search. `accept(i)->bool` gates RESULT membership; traversal
        visits all nodes (connectivity). ef = dynamic candidate-list size. Returns
        (ids, dists) of up to k accepted nodes sorted ascending. Counts distances in ctr.
        """
        qq = float(q @ q)
        visited = set()
        cand = []                      # min-heap (dist, id)
        res = []                       # max-heap (-dist, id) of ACCEPTED nodes, <=ef
        for e in self.entries:
            if e in visited:
                continue
            visited.add(int(e))
            de = self._d2(q, qq, e, ctr)
            heapq.heappush(cand, (de, int(e)))
            if accept(int(e)):
                heapq.heappush(res, (-de, int(e)))
                if len(res) > ef:
                    heapq.heappop(res)
        cap = max_visits if max_visits is not None else self.n
        while cand and len(visited) < cap:
            dc, c = heapq.heappop(cand)
            if res and len(res) >= ef and -res[0][0] < dc:
                break                  # nearest unexpanded candidate worse than worst result
            for nb in self._nbrs(c):
                nb = int(nb)
                if nb in visited:
                    continue
                visited.add(nb)
                dn = self._d2(q, qq, nb, ctr)
                if len(res) < ef or dn < -res[0][0]:
                    heapq.heappush(cand, (dn, nb))
                    if accept(nb):
                        heapq.heappush(res, (-dn, nb))
                        if len(res) > ef:
                            heapq.heappop(res)
        out = sorted((-d, i) for d, i in res)[:k]
        return np.array([i for _, i in out], dtype=np.int64), np.array([d for d, _ in out])


def exact_filtered_topk(graph, q, k, eligible_bool):
    """Ground-truth / pre-filter exact top-k among eligible rows (counts |eligible|)."""
    elig = np.where(eligible_bool)[0]
    if elig.size == 0:
        return np.array([], dtype=np.int64), 0
    qq = float(q @ q)
    d2 = graph.bsq[elig] - 2.0 * (graph.X[elig] @ q) + qq
    kk = min(k, elig.size)
    part = np.argpartition(d2, kk - 1)[:kk]
    order = part[np.argsort(d2[part])]
    return elig[order], int(elig.size)


def prefilter_search(graph, q, k, eligible_bool):
    ids, ndist = exact_filtered_topk(graph, q, k, eligible_bool)
    return ids, {"n_dist": ndist, "family": "pre"}


def postfilter_search(graph, q, k, ef, eligible_bool, max_mult=8):
    """Unfiltered greedy to ef, keep eligible, enlarge ef until k eligible (over-retrieval)."""
    ctr = [0]
    cur = ef
    for _ in range(max_mult):
        ids, _d = graph.greedy(q, max(k, cur), cur, accept=lambda i: True, ctr=ctr)
        keep = ids[eligible_bool[ids]]
        if len(keep) >= k or cur >= graph.n:
            return keep[:k], {"n_dist": ctr[0], "family": "post", "ef_used": cur}
        cur *= 2
    return keep[:k], {"n_dist": ctr[0], "family": "post", "ef_used": cur}


def infilter_search(graph, q, k, ef, accept, max_visits=None):
    """Certificate-/eligibility-guided traversal: accept gates membership, traverse all."""
    ctr = [0]
    ids, _d = graph.greedy(q, k, ef, accept=accept, ctr=ctr, max_visits=max_visits)
    return ids, {"n_dist": ctr[0], "family": "in"}


def infilter_mifa(graph, q, k, ef, pass_mask, fail_mask, unc_mask, refine_fn, theta=0.5,
                  skip_refine=True):
    """The PAPER's online execution (Component 2): NSW best-first traversal with the coarse
    pass/fail/uncertain certificate, where uncertain tuples are FINE-REFINED only when the
    top-k boundary tau_B cannot already rule them out (refinement-skipping, sec-guarantee).

    refine_fn(tuple_id)->p_phi in [0,1]; the tuple is admitted iff p_phi>=theta. Counters
    expose the paper's Q2 claim: uncertain_seen vs refined `u` (u<<n), and n_dist. This is
    the boundary-gated refinement the value-imputation baselines do NOT get to skip.
    """
    import heapq
    qq = float(q @ q)
    c = dict(n_dist=0, uncertain_seen=0, refined=0, refine_skipped=0, refine_calls=0)

    def d2(i):
        c["n_dist"] += 1
        return float(graph.bsq[i] - 2.0 * (graph.X[i] @ q) + qq)

    visited = set()
    cand = []        # min-heap (dist, id) frontier (all non-fail nodes drive traversal)
    res = []         # max-heap (-dist, id) of ADMITTED eligible nodes, <= ef
    tauB = [float("inf")]

    def consider(i, di):
        # decide eligibility per certificate; uncertain -> boundary-gated refine
        if fail_mask[i]:
            return
        if pass_mask[i]:
            heapq.heappush(res, (-di, i))
            if len(res) > ef:
                heapq.heappop(res)
        elif unc_mask[i]:
            c["uncertain_seen"] += 1
            if skip_refine and len(res) >= k and di > tauB[0]:
                c["refine_skipped"] += 1
                return
            c["refined"] += 1
            c["refine_calls"] += 1
            p = refine_fn(i)
            if p >= theta:
                heapq.heappush(res, (-di, i))
                if len(res) > ef:
                    heapq.heappop(res)
        if len(res) >= k:
            tauB[0] = -res[0][0]

    for e in graph.entries:
        e = int(e)
        if e in visited:
            continue
        visited.add(e)
        de = d2(e)
        heapq.heappush(cand, (de, e))
        consider(e, de)
    while cand:
        dc, cc = heapq.heappop(cand)
        if len(res) >= ef and -res[0][0] < dc:
            break
        for nb in graph._nbrs(cc):
            nb = int(nb)
            if nb in visited:
                continue
            visited.add(nb)
            dn = d2(nb)
            if len(res) < ef or dn < -res[0][0]:
                heapq.heappush(cand, (dn, nb))
                consider(nb, dn)
    out = sorted((-d, i) for d, i in res)[:k]
    return np.array([i for _, i in out], dtype=np.int64), c
