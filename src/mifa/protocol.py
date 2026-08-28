"""Leakage-resistant experiment protocol helpers.

The June harness sampled and masked one relation in place.  Paper protocol v2
instead persists a deterministic four-way split before any missingness is
injected.  This module contains only split/manifest mechanics; training,
calibration, and evaluation code must consume the resulting partitions without
crossing their declared roles.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from mifa.datasets import DatasetSpec


SPLIT_NAMES = ("train", "validation", "calibration", "test")
DEFAULT_FRACTIONS = (0.50, 0.20, 0.10, 0.20)
QUERY_ROLE_NAMES = (
    "query_cls_training", "validation", "calibration",
    "selection_reservation", "test",
)


def _validated_fractions(fractions: Sequence[float]) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    if values.shape != (4,):
        raise ValueError("four fractions are required: train/validation/calibration/test")
    if np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("split fractions must be positive and sum to one")
    return values


def _row_counts(n: int, fractions: np.ndarray) -> np.ndarray:
    if n < len(fractions):
        raise ValueError("at least four rows/groups are required for a four-way split")
    raw = n * fractions
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: n - int(counts.sum())]:
        counts[i] += 1
    # Tiny audit fixtures can round the 10% calibration split to zero. Preserve
    # all four roles by borrowing from the largest partition.
    for i in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        if counts[donor] <= 1:
            raise ValueError("cannot create four non-empty partitions")
        counts[donor] -= 1
        counts[i] = 1
    return counts


@dataclass(frozen=True)
class SplitManifest:
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    seed: int
    strategy: str

    def partitions(self) -> Mapping[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name), dtype=np.int64) for name in SPLIT_NAMES}

    def validate(self, n: int) -> None:
        parts = self.partitions()
        joined = np.concatenate(list(parts.values()))
        if any(len(v) == 0 for v in parts.values()):
            raise ValueError("all four partitions must be non-empty")
        if len(joined) != n or len(np.unique(joined)) != n:
            raise ValueError("split partitions must be disjoint and cover every row exactly once")
        if joined.min(initial=0) < 0 or joined.max(initial=-1) >= n:
            raise ValueError("split contains an out-of-range row index")

    def as_dict(self, include_indices: bool = True) -> dict:
        payload = {
            "seed": int(self.seed),
            "strategy": self.strategy,
            "counts": {name: int(len(rows)) for name, rows in self.partitions().items()},
        }
        if include_indices:
            payload["indices"] = {
                name: rows.astype(int).tolist() for name, rows in self.partitions().items()
            }
        return payload

    def digest(self) -> str:
        encoded = json.dumps(self.as_dict(include_indices=True), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _split_ordered_units(units: list[np.ndarray], target_rows: np.ndarray) -> list[np.ndarray]:
    """Cut ordered, indivisible units near cumulative row-count targets."""
    sizes = np.asarray([len(unit) for unit in units], dtype=int)
    cumulative = np.cumsum(sizes)
    boundaries = np.cumsum(target_rows)[:-1]
    cuts: list[int] = []
    lo = 1
    for boundary_number, target in enumerate(boundaries):
        remaining_boundaries = len(boundaries) - boundary_number - 1
        hi = len(units) - remaining_boundaries - 1
        candidates = np.arange(lo, hi + 1)
        cut = int(candidates[np.argmin(np.abs(cumulative[candidates - 1] - target))])
        cuts.append(cut)
        lo = cut + 1
    chunks = np.split(np.arange(len(units)), cuts)
    return [np.concatenate([units[i] for i in chunk]).astype(np.int64) for chunk in chunks]


def make_four_way_split(
    n: int,
    *,
    seed: int,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    group_ids: Sequence[object] | None = None,
    timestamps: Sequence[object] | None = None,
) -> SplitManifest:
    """Create a deterministic random/group or chronological four-way split.

    `group_ids` makes each group indivisible. If `timestamps` is supplied, rows
    (or non-overlapping time groups) are ordered chronologically. A group whose
    time interval overlaps another group is rejected because satisfying both
    strict chronology and group isolation would otherwise be ambiguous.
    """
    fractions_array = _validated_fractions(fractions)
    if n <= 0:
        raise ValueError("n must be positive")
    if group_ids is not None and len(group_ids) != n:
        raise ValueError("group_ids length must equal n")
    if timestamps is not None and len(timestamps) != n:
        raise ValueError("timestamps length must equal n")

    row_ids = np.arange(n, dtype=np.int64)
    target_rows = _row_counts(n, fractions_array)

    if group_ids is None:
        if timestamps is None:
            order = np.random.default_rng(seed).permutation(row_ids)
            strategy = "row_random"
        else:
            parsed_time = pd.to_datetime(np.asarray(timestamps), utc=True, errors="raise")
            order = np.argsort(parsed_time.asi8, kind="stable").astype(np.int64)
            strategy = "chronological"
        cuts = np.cumsum(target_rows)[:-1]
        parts = [part.astype(np.int64) for part in np.split(order, cuts)]
    else:
        groups = np.asarray(group_ids, dtype=object)
        unique_groups = list(pd.unique(groups))
        if len(unique_groups) < 4:
            raise ValueError("at least four distinct groups are required")
        units = [row_ids[groups == group] for group in unique_groups]
        if timestamps is None:
            order = np.random.default_rng(seed).permutation(len(units))
            units = [units[i] for i in order]
            # Assign whole groups to the most under-filled proportional target.
            buckets: list[list[np.ndarray]] = [[] for _ in SPLIT_NAMES]
            counts = np.zeros(4, dtype=int)
            for unit in units:
                relative_deficit = (target_rows - counts) / target_rows
                destination = int(np.argmax(relative_deficit))
                buckets[destination].append(unit)
                counts[destination] += len(unit)
            if any(not bucket for bucket in buckets):
                raise ValueError("group sizes prevent four non-empty partitions")
            parts = [np.concatenate(bucket).astype(np.int64) for bucket in buckets]
            strategy = "group_random"
        else:
            parsed_time = pd.to_datetime(np.asarray(timestamps), utc=True, errors="raise").asi8
            intervals = [(int(parsed_time[u].min()), int(parsed_time[u].max()), u) for u in units]
            intervals.sort(key=lambda item: (item[0], item[1]))
            for left, right in zip(intervals, intervals[1:]):
                if left[1] > right[0]:
                    raise ValueError("group time intervals overlap; strict chronological group split is impossible")
            parts = _split_ordered_units([item[2] for item in intervals], target_rows)
            strategy = "group_chronological"

    manifest = SplitManifest(*parts, seed=int(seed), strategy=strategy)
    manifest.validate(n)
    return manifest


def subset_spec(spec: DatasetSpec, rows: Sequence[int], *, name: str | None = None) -> DatasetSpec:
    """Materialize a row-aligned DatasetSpec partition without sharing mutable frames."""
    idx = np.asarray(rows, dtype=np.int64)
    return DatasetSpec(
        name=name or spec.name,
        A_e=np.ascontiguousarray(spec.A_e[idx]),
        table=spec.table.iloc[idx].reset_index(drop=True).copy(),
        dtypes=dict(spec.dtypes),
        metric=spec.metric,
        query_relevant_attrs=list(spec.query_relevant_attrs),
        meta={**spec.meta, "parent_dataset": spec.name, "partition_rows": int(len(idx))},
    )


def allocate_query_roles(
    n: int,
    role_counts: Mapping[str, int],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Allocate source-disjoint query rows and return the unused rows.

    The role order is fixed so a manifest is reproducible regardless of mapping
    insertion order.  This helper is used both for external query corpora and
    for held-out tuple content; in the latter case ``unused`` is the searchable
    relation from which the four database partitions are subsequently made.
    """
    unknown = set(role_counts) - set(QUERY_ROLE_NAMES)
    if unknown:
        raise ValueError(f"unknown query roles: {sorted(unknown)}")
    counts = {name: int(role_counts.get(name, 0)) for name in QUERY_ROLE_NAMES}
    if any(value < 0 for value in counts.values()):
        raise ValueError("query role counts must be non-negative")
    required = sum(counts.values())
    if required > n:
        raise ValueError(f"query roles require {required} sources but corpus has {n}")
    order = np.random.default_rng(seed).permutation(n).astype(np.int64)
    roles: dict[str, np.ndarray] = {}
    offset = 0
    for name in QUERY_ROLE_NAMES:
        count = counts[name]
        roles[name] = np.sort(order[offset:offset + count])
        offset += count
    return roles, np.sort(order[offset:])
