"""GPU-resident exact-distance providers for the paper executor.

The dynamic predicate scheduler remains on CPU, while the embedding matrix is
kept resident on CUDA.  Single-query mode includes query upload, GEMV, result
download, and synchronization in the online latency.  ``prepare_batch`` is a
throughput-oriented primitive used by the efficiency harness; it never caches
query results across benchmark repetitions.
"""
from __future__ import annotations

import numpy as np


class TorchResidentDistanceProvider:
    """Compute complete squared-L2 distance rows on a resident CUDA base."""

    def __init__(self, base, base_sqnorm=None, *, device="cuda:0"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TorchResidentDistanceProvider")
        base = np.ascontiguousarray(base, dtype=np.float32)
        if base.ndim != 2:
            raise ValueError("base must be a 2-D matrix")
        if base_sqnorm is None:
            base_sqnorm = np.einsum("ij,ij->i", base, base)
        base_sqnorm = np.ascontiguousarray(base_sqnorm, dtype=np.float32)
        if base_sqnorm.shape != (len(base),):
            raise ValueError("base_sqnorm must have one value per base row")
        self.torch = torch
        self.device = torch.device(device)
        self.base_rows, self.dim = map(int, base.shape)
        base_device = torch.from_numpy(base).to(self.device)
        self.base_t = base_device.T.contiguous()
        self.base_sqnorm = torch.from_numpy(base_sqnorm).to(self.device)
        # Initialize both cuBLAS paths during construction so first-request
        # context/library setup is not mislabeled as steady online latency.
        warm_query = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        _ = torch.addmv(
            self.base_sqnorm, self.base_t.T, warm_query,
            beta=1.0, alpha=-2.0)
        _ = torch.addmm(
            self.base_sqnorm[None, :], warm_query[None, :], self.base_t,
            beta=1.0, alpha=-2.0)
        torch.cuda.synchronize(self.device)

    def prepare(self, query):
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError("query dimension does not match resident base")
        query_device = self.torch.from_numpy(query).to(self.device)
        query_sqnorm = float(query @ query)
        scores = self.torch.addmv(
            self.base_sqnorm, self.base_t.T, query_device,
            beta=1.0, alpha=-2.0)
        scores.add_(query_sqnorm).clamp_min_(0.0)
        return np.ascontiguousarray(scores.cpu().numpy(), dtype=np.float32)

    def prepare_batch(self, queries, *, batch_size=512):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query width does not match resident base")
        output = np.empty((len(queries), self.base_rows), dtype=np.float32)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            count = end - start
            current = queries[start:end]
            # Dispatch one fixed MxKxN GEMM shape for every chunk, including
            # the tail.  A query then follows the same FP32 accumulation path
            # when a replicated-corpus multi-GPU run partitions query rows at
            # a different boundary from the single-GPU run.
            if count < batch_size:
                padded = np.zeros((batch_size, self.dim), dtype=np.float32)
                padded[:count] = current
                current_compute = padded
            else:
                current_compute = current
            query_device = self.torch.from_numpy(current_compute).to(self.device)
            bias = self.base_sqnorm[None, :].expand(len(current_compute), -1)
            scores = self.torch.addmm(
                bias, query_device, self.base_t, beta=1.0, alpha=-2.0)
            query_sqnorm = np.einsum(
                "ij,ij->i", current_compute, current_compute)
            scores.add_(self.torch.from_numpy(
                np.ascontiguousarray(query_sqnorm[:, None])).to(self.device))
            scores.clamp_min_(0.0)
            output[start:end] = scores[:count].cpu().numpy()
        self.torch.cuda.synchronize(self.device)
        return output

    @property
    def resident_bytes(self):
        return int(
            self.base_t.numel() * self.base_t.element_size() +
            self.base_sqnorm.numel() * self.base_sqnorm.element_size())

    def manifest(self):
        return {
            "kind": "torch_resident_full_distance",
            "device": str(self.device),
            "base_rows": self.base_rows,
            "dimension": self.dim,
            "resident_bytes": self.resident_bytes,
            "single_query_transfer_included": True,
            "batch_tail_padded_to_fixed_gemm_shape": True,
        }
