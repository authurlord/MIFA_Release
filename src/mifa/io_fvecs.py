"""Readers for the .fvecs / .ivecs binary formats used by ANN benchmarks.

Layout: a flat sequence of records; each record is
  int32 dim   followed by `dim` float32 (fvecs) or int32 (ivecs) values.
"""
from __future__ import annotations

import numpy as np


def read_fvecs(path, dtype=np.float32):
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=dtype)
    dim = int(raw[0])
    row = dim + 1
    assert raw.size % row == 0, f"{path}: size {raw.size} not divisible by {row}"
    n = raw.size // row
    arr = raw.reshape(n, row)[:, 1:]
    if dtype == np.float32:
        return arr.view(np.float32).astype(np.float32)
    return arr.astype(dtype)


def read_ivecs(path):
    return read_fvecs(path, dtype=np.int32)
