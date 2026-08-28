"""Deterministic systems-only scale cohort construction.

The helper is shared by PaperMIFA and external-baseline transports so a scale
comparison cannot silently use different projections, corpus jitter, or row
replication. It does not define an effectiveness dataset.
"""
from __future__ import annotations

import numpy as np


def project_normalize(values: np.ndarray, dimension: int) -> np.ndarray:
    """Take the fixed MRL prefix and return contiguous unit vectors."""
    values = np.asarray(values)
    dimension = int(dimension)
    if values.ndim != 2:
        raise ValueError("scale vectors must be a matrix")
    if dimension <= 0 or dimension > values.shape[1]:
        raise ValueError("scale dimension must be within the source dimension")
    output = np.ascontiguousarray(values[:, :dimension], dtype=np.float32)
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    output /= norms
    return output


def expand_repeated_jitter_embeddings(
    values: np.ndarray,
    target_n: int,
    dimension: int,
    noise_std: float,
    seed: int,
    *,
    jitter_reference_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeat real rows, add fixed iid jitter, and renormalize.

    Returns ``(expanded, projected_source, source_rows)``. Random draws use a
    fixed block size so byte receipts are stable without allocating a second
    full-size noise matrix.

    ``jitter_reference_values`` is the common highest-dimensional embedding
    matrix used by an MRL sweep.  When supplied, every lower-dimensional run
    draws the same full-width noise vector for a given expanded row and then
    takes its prefix.  Consequently the d-dimensional expanded corpus is the
    normalized prefix of the common reference corpus, rather than a different
    random corpus whose RNG stride happens to depend on d.
    """
    source = project_normalize(values, dimension)
    reference = None
    reference_dimension = int(dimension)
    if jitter_reference_values is not None:
        reference = np.asarray(jitter_reference_values)
        if reference.ndim != 2 or len(reference) != len(source):
            raise ValueError(
                "MRL jitter reference must be a row-aligned embedding matrix")
        if reference.shape[1] < int(dimension):
            raise ValueError(
                "MRL jitter reference dimension must cover the requested prefix")
        reference_dimension = int(reference.shape[1])
        reference = project_normalize(reference, reference_dimension)
    target_n = int(target_n)
    if target_n < len(source):
        raise ValueError("target scale may not shrink the searchable relation")
    noise_std = float(noise_std)
    if noise_std < 0.0 or not np.isfinite(noise_std):
        raise ValueError("scale noise must be finite and non-negative")
    source_rows = np.arange(target_n, dtype=np.int64) % len(source)
    if reference is None:
        expanded = np.ascontiguousarray(source[source_rows], dtype=np.float32)
    else:
        # Keep the common full-dimensional normalization until after jitter.
        # Renormalizing the prefix before adding noise would change the noise
        # scale by dimension and break the nested-corpus identity.
        expanded = np.ascontiguousarray(
            reference[source_rows, :int(dimension)], dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    block = 8192
    for start in range(0, target_n, block):
        end = min(start + block, target_n)
        if noise_std:
            noise = rng.normal(
                0.0, noise_std,
                size=(end - start, reference_dimension),
            ).astype(np.float32)
            expanded[start:end] += noise[:, :int(dimension)]
        norms = np.linalg.norm(expanded[start:end], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        expanded[start:end] /= norms
    return expanded, source, source_rows
