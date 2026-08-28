"""CPU-resident exact-distance providers for the paper executor.

The projected tree can legitimately visit most searchable rows on small or
weakly clustered snapshots.  In that regime, issuing one BLAS dot product per
tuple is dominated by call overhead.  This provider computes the same complete
squared-L2 row with one contiguous SGEMV and leaves every predicate decision,
lazy inference, and termination check in the paper scheduler unchanged.
"""
from __future__ import annotations

import numpy as np


class NumpyResidentDistanceProvider:
    """Compute complete squared-L2 distance rows with a resident CPU matrix."""

    def __init__(self, base, base_sqnorm=None):
        base = np.ascontiguousarray(base, dtype=np.float32)
        if base.ndim != 2:
            raise ValueError("base must be a 2-D matrix")
        if base_sqnorm is None:
            base_sqnorm = np.einsum("ij,ij->i", base, base)
        base_sqnorm = np.ascontiguousarray(base_sqnorm, dtype=np.float32)
        if base_sqnorm.shape != (len(base),):
            raise ValueError("base_sqnorm must have one value per base row")
        self.base = base
        self.base_sqnorm = base_sqnorm
        self.base_rows, self.dim = map(int, base.shape)
        # Warm the BLAS dispatch outside serving/query timing.
        self.prepare(np.zeros(self.dim, dtype=np.float32))

    def prepare(self, query):
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError("query dimension does not match resident base")
        scores = np.asarray(self.base @ query, dtype=np.float32)
        scores *= np.float32(-2.0)
        scores += self.base_sqnorm
        scores += np.float32(query @ query)
        np.maximum(scores, np.float32(0.0), out=scores)
        return np.ascontiguousarray(scores, dtype=np.float32)

    def prepare_batch(self, queries, *, batch_size=512):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width does not match resident base")
        output = np.empty((len(queries), self.base_rows), dtype=np.float32)
        for start in range(0, len(queries), int(batch_size)):
            end = min(start + int(batch_size), len(queries))
            current = queries[start:end]
            # Write SGEMM directly into the persistent output slice.  The old
            # expression allocated a same-sized temporary and copied it into
            # ``output`` for every query batch, needlessly moving the complete
            # distance matrix twice on the CPU path.
            scores = output[start:end]
            np.matmul(current, self.base.T, out=scores)
            scores *= np.float32(-2.0)
            scores += self.base_sqnorm[None, :]
            scores += np.einsum("ij,ij->i", current, current)[:, None]
            np.maximum(scores, np.float32(0.0), out=scores)
        return output

    @property
    def resident_bytes(self):
        return int(self.base.nbytes + self.base_sqnorm.nbytes)

    def manifest(self):
        return {
            "kind": "numpy_resident_full_distance",
            "device": "cpu",
            "base_rows": self.base_rows,
            "dimension": self.dim,
            "resident_bytes": self.resident_bytes,
            "single_query_transfer_included": True,
        }
