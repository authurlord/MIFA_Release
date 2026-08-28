"""Dataset API plus workload-specific adapters."""

from .core import (
    DatasetSpec,
    build_Dstar,
    load_dataset,
    load_processed,
    make_synthetic,
    make_synthetic_deps,
)

__all__ = [
    "DatasetSpec",
    "build_Dstar",
    "load_dataset",
    "load_processed",
    "make_synthetic",
    "make_synthetic_deps",
]
