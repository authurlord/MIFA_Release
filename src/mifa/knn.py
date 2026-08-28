"""Fast batched k-NN over dense embeddings.

The hot paths in the harness (exact oracle KNN, embedding-kNN imputation) are
dense distance computations. A per-query Python loop with broadcast subtraction
allocates a huge temporary each iteration and runs single-threaded. Instead we
use a single matmul, which BLAS parallelizes across all cores:

    ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q . x

For ranking, ||q||^2 is constant within a query row, so we rank on
    s(q, x) = ||x||^2 - 2 q . x   (smaller = closer).

Backend selection (env MIFA_KNN_BACKEND): "auto" (default) | "cpu" | "torch".
torch uses CUDA if available (good for the 2.7M `-large` split); for medium
(100k) CPU-BLAS over many cores is already fast.
"""
from __future__ import annotations

import os

import numpy as np

_TORCH = None


def _try_torch():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    try:
        import torch
        _TORCH = torch if torch.cuda.is_available() else False
    except Exception:
        _TORCH = False
    return _TORCH


def _backend():
    b = os.environ.get("MIFA_KNN_BACKEND", "auto").lower()
    if b == "cpu":
        return "cpu"
    if b == "torch":
        return "torch" if _try_torch() else "cpu"
    return "torch" if _try_torch() else "cpu"  # auto


def knn_batch(queries, base, k, base_sqnorm=None, batch_size=4096):
    """Return (ids (Q,k), d2 (Q,k)) of k nearest base rows for each query.

    Distances returned are true squared-L2. Ranking uses the cheap surrogate.
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    base = np.ascontiguousarray(base, dtype=np.float32)
    if base_sqnorm is None:
        base_sqnorm = np.einsum("ij,ij->i", base, base)
    k = min(k, base.shape[0])
    backend = _backend()
    if backend == "torch":
        return _knn_torch(queries, base, k, base_sqnorm, batch_size)
    return _knn_cpu(queries, base, k, base_sqnorm, batch_size)


def _topk_ids_cpu(scores, k):
    part = np.argpartition(scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(scores[rows, part], axis=1)
    return part[rows, order]


def _knn_cpu(queries, base, k, base_sqnorm, batch_size):
    Q = queries.shape[0]
    q_sq = np.einsum("ij,ij->i", queries, queries)
    ids = np.empty((Q, k), dtype=np.int64)
    d2 = np.empty((Q, k), dtype=np.float32)
    for s in range(0, Q, batch_size):
        e = min(s + batch_size, Q)
        scores = base_sqnorm[None, :] - 2.0 * (queries[s:e] @ base.T)  # (b, N)
        top = _topk_ids_cpu(scores, k)
        ids[s:e] = top
        rows = np.arange(e - s)[:, None]
        d2[s:e] = np.maximum(scores[rows, top] + q_sq[s:e, None], 0.0)
    return ids, d2


def _knn_torch(queries, base, k, base_sqnorm, batch_size):
    torch = _TORCH
    dev = "cuda"
    Q = queries.shape[0]
    baseT = torch.from_numpy(base).to(dev).T.contiguous()
    bsq = torch.from_numpy(np.asarray(base_sqnorm, dtype=np.float32)).to(dev)
    ids = np.empty((Q, k), dtype=np.int64)
    d2 = np.empty((Q, k), dtype=np.float32)
    for s in range(0, Q, batch_size):
        e = min(s + batch_size, Q)
        qb = torch.from_numpy(queries[s:e]).to(dev)
        scores = bsq[None, :] - 2.0 * (qb @ baseT)          # (b, N)
        vals, top = torch.topk(scores, k, dim=1, largest=False)
        qsq = (qb * qb).sum(1, keepdim=True)
        ids[s:e] = top.cpu().numpy()
        d2[s:e] = torch.clamp(vals + qsq, min=0.0).cpu().numpy()
    return ids, d2


def query_dists(q_e, base, base_sqnorm):
    """Squared-L2 distance from a single query to ALL base rows (one BLAS matvec).

    Avoids materializing a gathered sub-matrix per query; mask afterwards for
    filtered top-k. base_sqnorm precomputed once by the caller."""
    q = np.ascontiguousarray(q_e, dtype=np.float32)
    d2 = base_sqnorm - 2.0 * (base @ q) + float(q @ q)
    return np.maximum(d2, 0.0)


def topk_masked(d2, eligible_bool, k):
    """Top-k smallest distances among rows where eligible_bool is True.
    Returns (ids, dists) sorted ascending."""
    elig = np.where(eligible_bool)[0]
    if elig.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    de = d2[elig]
    kk = min(k, de.size)
    part = np.argpartition(de, kk - 1)[:kk]
    order = part[np.argsort(de[part])]
    return elig[order], de[order]


def query_dists_batch(Q_e, base, base_sqnorm=None, batch_size=512):
    """Squared-L2 distances from EACH of Q queries to ALL base rows: (Q, N) float32.

    One batched matmul (GPU iff backend=torch) instead of a per-query matvec Python
    loop -- this is the R5 perf fix for the quality-isolation harness. base_sqnorm is
    precomputed once by the caller. Memory: Q*N float32; callers cap Q via batch_size.
    """
    Q_e = np.ascontiguousarray(Q_e, dtype=np.float32)
    base = np.ascontiguousarray(base, dtype=np.float32)
    if base_sqnorm is None:
        base_sqnorm = np.einsum("ij,ij->i", base, base)
    Q, N = Q_e.shape[0], base.shape[0]
    out = np.empty((Q, N), dtype=np.float32)
    if _backend() == "torch":
        torch = _TORCH
        dev = "cuda"
        baseT = torch.from_numpy(base).to(dev).T.contiguous()
        bsq = torch.from_numpy(np.asarray(base_sqnorm, dtype=np.float32)).to(dev)
        for s in range(0, Q, batch_size):
            e = min(s + batch_size, Q)
            qb = torch.from_numpy(Q_e[s:e]).to(dev)
            qsq = (qb * qb).sum(1, keepdim=True)
            d = bsq[None, :] - 2.0 * (qb @ baseT) + qsq
            out[s:e] = torch.clamp(d, min=0.0).cpu().numpy()
    else:
        qsq = np.einsum("ij,ij->i", Q_e, Q_e)
        for s in range(0, Q, batch_size):
            e = min(s + batch_size, Q)
            d = base_sqnorm[None, :] - 2.0 * (Q_e[s:e] @ base.T) + qsq[s:e, None]
            out[s:e] = np.maximum(d, 0.0)
    return out


def topk_masked_batch(D2, elig, k, batch_size=256):
    """Top-k smallest distances per row among eligible columns, batched.

    D2: (Q, N) squared distances. elig: (Q, N) bool eligibility mask (per query).
    Returns ids (Q, k) int64 (-1 padded where fewer than k eligible) and dists (Q, k)
    float32 (inf padded), each row sorted ascending. GPU-aware via torch.topk.
    """
    D2 = np.ascontiguousarray(D2, dtype=np.float32)
    Q, N = D2.shape
    kk = min(k, N)
    ids = np.full((Q, k), -1, dtype=np.int64)
    dd = np.full((Q, k), np.inf, dtype=np.float32)
    if _backend() == "torch":
        torch = _TORCH
        dev = "cuda"
        for s in range(0, Q, batch_size):
            e = min(s + batch_size, Q)
            d = torch.from_numpy(D2[s:e]).to(dev)
            m = torch.from_numpy(np.ascontiguousarray(elig[s:e])).to(dev)
            d = torch.where(m, d, torch.full_like(d, float("inf")))
            vals, idx = torch.topk(d, kk, dim=1, largest=False)
            idx = idx.cpu().numpy()
            vals = vals.cpu().numpy()
            idx[~np.isfinite(vals)] = -1          # ineligible padding
            ids[s:e, :kk] = idx
            dd[s:e, :kk] = vals
    else:
        for i in range(Q):
            e_idx = np.where(elig[i])[0]
            if e_idx.size == 0:
                continue
            kq = min(k, e_idx.size)
            de = D2[i, e_idx]
            part = np.argpartition(de, kq - 1)[:kq]
            order = part[np.argsort(de[part])]
            ids[i, :kq] = e_idx[order]
            dd[i, :kq] = de[order]
    return ids, dd


def active_backend():
    return _backend()
