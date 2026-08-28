"""Maskers: MCAR / attribute-skewed / tuple-block (paper §4.3).

A Masker takes the complete table D* and a seed, and returns:
  - mask:      boolean array (n, m) True where the cell is MASKED (missing)
  - observed:  a copy of the table with masked cells set to NaN/None

Original (unmasked) values are retained by the caller for OFFLINE imputability
scoring only; they are never exposed to online methods.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


HISTORICAL_EXPOSURE_NESTED_PROTOCOL = "exposure_priority_prefix_v1"


@dataclass(frozen=True)
class HistoricalExposurePriority:
    """One frozen row/attribute assignment order for every rho prefix."""

    rows: np.ndarray
    attributes: np.ndarray
    manifest: dict


def _empty_mask(n, m):
    return np.zeros((n, m), dtype=bool)


def mcar_mask(n, m, rho, rng):
    """Legacy cell-wise Bernoulli MCAR used by the June experiments."""
    return rng.random((n, m)) < rho


def tuple_mcar_mask(n, m, rho, rng):
    """Paper-v2 MCAR: select exactly rho*n tuples and hide one cell per tuple."""
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    mask = _empty_mask(n, m)
    selected = rng.choice(n, size=int(round(rho * n)), replace=False)
    columns = rng.integers(0, m, size=len(selected))
    mask[selected, columns] = True
    return mask


def tuple_predicate_mcar_mask(n, m, rho, allowed_indices, rng):
    """Tuple-MCAR restricted to a predeclared predicate-attribute subset.

    Exactly ``rho*n`` tuples are selected and one attribute is hidden on each
    selected tuple.  The hidden attribute is sampled uniformly from
    ``allowed_indices``.  The subset is workload-declared before any method is
    fitted; it must not depend on test labels or method outcomes.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    allowed = np.asarray(allowed_indices, dtype=np.int64)
    if allowed.ndim != 1 or len(allowed) == 0:
        raise ValueError("predicate MCAR requires at least one allowed attribute")
    if len(np.unique(allowed)) != len(allowed):
        raise ValueError("predicate MCAR attributes must be unique")
    if np.any(allowed < 0) or np.any(allowed >= m):
        raise ValueError("predicate MCAR attribute index is out of range")
    mask = _empty_mask(n, m)
    selected = rng.choice(n, size=int(round(rho * n)), replace=False)
    columns = rng.choice(allowed, size=len(selected), replace=True)
    mask[selected, columns] = True
    return mask


def tuple_predicate_block_mask(n, m, rho, allowed_indices, block_size, rng):
    """Tuple missingness restricted to a fixed-width predicate-attribute set."""
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    allowed = np.asarray(allowed_indices, dtype=np.int64)
    width = int(block_size)
    if allowed.ndim != 1 or len(allowed) == 0:
        raise ValueError("predicate block requires allowed attributes")
    if len(np.unique(allowed)) != len(allowed):
        raise ValueError("predicate block attributes must be unique")
    if np.any(allowed < 0) or np.any(allowed >= m):
        raise ValueError("predicate block attribute index is out of range")
    if width < 1 or width > len(allowed):
        raise ValueError("predicate block width exceeds allowed attributes")
    mask = _empty_mask(n, m)
    selected = rng.choice(n, size=int(round(rho * n)), replace=False)
    for row in selected:
        columns = rng.choice(allowed, size=width, replace=False)
        mask[int(row), columns] = True
    return mask


def tuple_query_answer_mnar_mask(n, m, rho, cols, queries, truth_ids, rng,
                                 allowed_indices=None, impact_fraction=1.0,
                                 cells_per_tuple=1):
    """Mask query-answer cells without consulting any evaluated method.

    This is a workload-conditioned MNAR stressor, not MCAR.  The evaluator
    freezes the queries and their complete-data filtered top-k answers first.
    It assigns ``impact_fraction`` of the tuple budget by visiting queries in a
    seeded order and masking one predicate-bearing answer cell per query per
    round.  The remainder is filled by tuple-MCAR over the same declared
    predicate attributes, while the total remains exactly ``round(rho*n)``
    distinct rows.

    The construction is method-independent: no repair, inference, baseline,
    or returned result is inspected.  Its purpose is to measure the regime in
    which missing values actually intersect the retrieval workload, alongside
    (and never in place of) the uniform tuple-MCAR control.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if not 0.0 <= impact_fraction <= 1.0:
        raise ValueError("impact_fraction must be in [0, 1]")
    if len(cols) != m:
        raise ValueError("cols must be aligned with the mask width")
    if len(queries) != len(truth_ids) or not len(queries):
        raise ValueError("queries and truth_ids must be non-empty and aligned")
    if allowed_indices is None:
        allowed = np.arange(m, dtype=np.int64)
    else:
        allowed = np.asarray(allowed_indices, dtype=np.int64)
    if allowed.ndim != 1 or len(allowed) == 0:
        raise ValueError("query-answer MNAR requires at least one allowed attribute")
    if len(np.unique(allowed)) != len(allowed):
        raise ValueError("query-answer MNAR attributes must be unique")
    if np.any(allowed < 0) or np.any(allowed >= m):
        raise ValueError("query-answer MNAR attribute index is out of range")
    cells_per_tuple = int(cells_per_tuple)
    if cells_per_tuple < 1 or cells_per_tuple > len(allowed):
        raise ValueError(
            "query-answer MNAR cells_per_tuple exceeds allowed attributes")

    target_count = int(round(rho * n))
    impact_count = int(round(float(impact_fraction) * target_count))
    mask = _empty_mask(n, m)
    if target_count == 0:
        return mask
    attr_position = {str(attr): index for index, attr in enumerate(cols)}
    allowed_set = set(int(index) for index in allowed)
    candidates_by_query = []
    for query, answer_ids in zip(queries, truth_ids):
        predicate = query["phi"] if isinstance(query, dict) else query.phi
        query_columns = sorted({
            attr_position[atom.attr]
            for atom in predicate.atoms
            if atom.attr in attr_position
            and attr_position[atom.attr] in allowed_set
        })
        rows = np.unique(np.asarray(answer_ids, dtype=np.int64))
        rows = rows[(rows >= 0) & (rows < n)]
        if cells_per_tuple == 1:
            candidates = np.asarray(
                [(int(row), int(column))
                 for row in rows for column in query_columns],
                dtype=np.int64)
            if candidates.size:
                candidates = candidates.reshape(-1, 2)
                candidates = candidates[rng.permutation(len(candidates))]
            else:
                candidates = np.empty((0, 2), dtype=np.int64)
        else:
            candidates = [(int(row), tuple(query_columns)) for row in rows]
            candidates = [candidates[index] for index in
                          rng.permutation(len(candidates))]
        candidates_by_query.append(candidates)

    selected_rows = np.zeros(n, dtype=bool)
    cursors = np.zeros(len(queries), dtype=np.int64)
    query_order = rng.permutation(len(queries))
    selected_count = 0
    while selected_count < impact_count:
        progress = False
        for query_index in query_order:
            candidates = candidates_by_query[int(query_index)]
            cursor = int(cursors[int(query_index)])
            while cursor < len(candidates):
                candidate_row = int(
                    candidates[cursor, 0] if cells_per_tuple == 1
                    else candidates[cursor][0])
                if not selected_rows[candidate_row]:
                    break
                cursor += 1
            cursors[int(query_index)] = cursor
            if cursor >= len(candidates):
                continue
            candidate = candidates[cursor]
            cursors[int(query_index)] += 1
            row = int(candidate[0])
            if cells_per_tuple == 1:
                columns = np.asarray([int(candidate[1])], dtype=np.int64)
            else:
                query_columns = np.asarray(candidate[1], dtype=np.int64)
                if len(query_columns) >= cells_per_tuple:
                    columns = rng.choice(
                        query_columns, size=cells_per_tuple, replace=False)
                else:
                    remainder = np.asarray([
                        column for column in allowed
                        if column not in set(query_columns.tolist())
                    ], dtype=np.int64)
                    extra = rng.choice(
                        remainder,
                        size=cells_per_tuple - len(query_columns),
                        replace=False)
                    columns = np.concatenate([query_columns, extra])
            mask[row, columns] = True
            selected_rows[int(row)] = True
            selected_count += 1
            progress = True
            if selected_count == impact_count:
                break
        if not progress:
            break

    remaining = target_count - selected_count
    if remaining:
        available_rows = np.flatnonzero(~selected_rows)
        fill_rows = rng.choice(available_rows, size=remaining, replace=False)
        if cells_per_tuple == 1:
            fill_columns = rng.choice(allowed, size=remaining, replace=True)
            mask[fill_rows, fill_columns] = True
        else:
            for row in fill_rows:
                columns = rng.choice(
                    allowed, size=cells_per_tuple, replace=False)
                mask[int(row), columns] = True
    return mask


def training_missingness_kind(kind, answer_cells_per_tuple=1):
    """Return the explicitly declared training counterpart of a test stressor."""
    if kind == "tuple_query_answer_mnar":
        # D_tr has no retrieval-answer labels.  Preserve rho and the attribute
        # support, but draw its incomplete rows uniformly.
        return ("tuple_predicate_mcar" if int(answer_cells_per_tuple) == 1
                else "tuple_predicate_block")
    return kind


def tuple_mcar_nested_mask(n, m, rho, rng):
    """Tuple-level MCAR whose masks are nested across ``rho`` values.

    Reinitializing ``rng`` with the same seed produces one fixed tuple
    permutation and one fixed hidden attribute per tuple.  A larger missing
    rate therefore extends, rather than resamples, the mask generated at a
    smaller rate.  This specialization is used only for missing-rate sweeps;
    the historical ``tuple_mcar`` protocol remains byte-for-byte unchanged.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    order = rng.permutation(n)
    columns_by_rank = rng.integers(0, m, size=n)
    selected_count = int(round(rho * n))
    selected = order[:selected_count]
    mask = _empty_mask(n, m)
    mask[selected, columns_by_rank[:selected_count]] = True
    return mask


def single_attribute_mcar_nested_mask(n, m, rho, attribute_index, rng):
    """Query-independent tuple MCAR on one predeclared attribute.

    A seeded permutation fixes the tuple order before any query answers or
    method outputs are inspected.  Increasing ``rho`` therefore extends the
    selected tuple prefix, while every selected tuple masks exactly one cell
    in the same predeclared attribute (``m=1, w=1`` in the paper notation).
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    attribute_index = int(attribute_index)
    if attribute_index < 0 or attribute_index >= m:
        raise ValueError("single MCAR attribute index is out of range")
    selected_count = int(round(rho * n))
    selected = rng.permutation(n)[:selected_count]
    mask = _empty_mask(n, m)
    mask[selected, attribute_index] = True
    return mask


def tuple_attr_skewed_mask(n, m, rho, usage_freq, rng):
    """Tuple-level variant whose single hidden attribute follows usage weights."""
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    usage = np.asarray(usage_freq, dtype=float)
    if usage.shape != (m,) or np.any(usage < 0):
        raise ValueError("usage_freq must contain one non-negative weight per attribute")
    if usage.sum() == 0:
        usage = np.ones(m, dtype=float)
    mask = _empty_mask(n, m)
    selected = rng.choice(n, size=int(round(rho * n)), replace=False)
    columns = rng.choice(m, size=len(selected), replace=True, p=usage / usage.sum())
    mask[selected, columns] = True
    return mask


def attr_skewed_mask(n, m, rho, usage_freq, rng):
    """rho_j proportional to attribute usage frequency, scaled so mean_j rho_j = rho."""
    usage = np.asarray(usage_freq, dtype=float)
    if usage.sum() == 0:
        usage = np.ones(m)
    rho_j = usage / usage.mean() * rho        # mean_j rho_j == rho
    rho_j = np.clip(rho_j, 0.0, 1.0)
    mask = _empty_mask(n, m)
    for j in range(m):
        mask[:, j] = rng.random(n) < rho_j[j]
    return mask


def tuple_block_mask(n, m, rho, rng):
    """Sample fraction p of tuples; mask ceil(0.5m) random attrs each.
    p chosen so expected masked-cell fraction == rho."""
    a = int(np.ceil(0.5 * m))
    p = rho / (a / m)
    p = min(p, 1.0)
    mask = _empty_mask(n, m)
    chosen = rng.random(n) < p
    idx = np.where(chosen)[0]
    for i in idx:
        cols = rng.choice(m, size=a, replace=False)
        mask[i, cols] = True
    return mask


def fixed_tuple_block_mask(n, m, rho, rng, block_size=1,
                           block_indices=None):
    """Hide a fixed-width attribute block on exactly ``rho*n`` tuples.

    This is the controlled masked-attribute-count sweep used by E1-3/E2-4.
    The selected tuple set is identical for a fixed seed and every block
    width; only the declared number of hidden attributes changes.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    block_size = int(block_size)
    if block_size < 1 or block_size > m:
        raise ValueError("block_size must be in [1, m]")
    fixed_block = None
    if block_indices is not None:
        fixed_block = np.asarray(block_indices, dtype=np.int64)
        if fixed_block.ndim != 1 or len(fixed_block) != block_size:
            raise ValueError("fixed block width must equal block_size")
        if len(np.unique(fixed_block)) != len(fixed_block):
            raise ValueError("fixed block attributes must be unique")
        if np.any(fixed_block < 0) or np.any(fixed_block >= m):
            raise ValueError("fixed block attribute index is out of range")
    order = rng.permutation(n)
    selected = order[:int(round(rho * n))]
    if fixed_block is None:
        column_order = rng.permutation(m)
        fixed_block = column_order[:block_size]
    mask = _empty_mask(n, m)
    mask[np.ix_(selected, fixed_block)] = True
    return mask


def joint_tuple_block_mask(n, m, rho, rng, block_size=2,
                           block_indices=None):
    """Select exactly ``rho*n`` tuples and hide a fixed-size attribute block.

    Unlike the legacy ``tuple_block_mask`` (which preserves an expected cell
    rate), this paper-v2 stressor preserves the declared incomplete-tuple rate.
    It therefore isolates genuinely joint residual predicates while keeping
    severity comparable to tuple-MCAR.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    block_size = int(block_size)
    if block_size < 2:
        raise ValueError("joint tuple blocks require block_size >= 2")
    width = min(block_size, m)
    fixed_block = None
    if block_indices is not None:
        fixed_block = np.asarray(block_indices, dtype=np.int64)
        if fixed_block.ndim != 1 or len(fixed_block) < 2:
            raise ValueError("a fixed joint block requires at least two attributes")
        if len(np.unique(fixed_block)) != len(fixed_block):
            raise ValueError("fixed joint block attributes must be unique")
        if np.any(fixed_block < 0) or np.any(fixed_block >= m):
            raise ValueError("fixed joint block attribute index is out of range")
        if len(fixed_block) != width:
            raise ValueError("fixed joint block width must equal block_size")
    mask = _empty_mask(n, m)
    selected = rng.choice(n, size=int(round(rho * n)), replace=False)
    for row in selected:
        columns = (fixed_block if fixed_block is not None else
                   rng.choice(m, size=width, replace=False))
        mask[row, columns] = True
    return mask


def tuple_value_dependent_mask(table, cols, rho, rng):
    """Tuple-level MAR/MNAR stressor driven by complete scalar values.

    Each attribute is converted to a stable percentile score.  Numerical
    attributes use their value rank; categorical attributes use inverse
    empirical frequency.  Exactly ``rho*n`` tuples are sampled without
    replacement with probability proportional to their value-derived score;
    the hidden attribute is sampled from the row's attribute scores.  A small
    fixed floor keeps the pattern value-dependent without collapsing onto a
    few extreme or rare values that a frozen workload may never reference.
    Complete values are used only by the evaluator-owned masker.
    """
    n, m = len(table), len(cols)
    if n <= 0 or m <= 0:
        raise ValueError("table and cols must be non-empty")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    scores = np.zeros((n, m), dtype=np.float64)
    for column_index, column in enumerate(cols):
        series = table[column]
        numeric = np.asarray(
            series.rank(method="average", pct=True, na_option="bottom"),
            dtype=np.float64)
        if not pd.api.types.is_numeric_dtype(series.dtype):
            frequencies = series.value_counts(dropna=False, normalize=True)
            rarity = series.map(lambda value: 1.0 - frequencies.loc[value])
            numeric = np.asarray(rarity, dtype=np.float64)
        scores[:, column_index] = np.nan_to_num(
            numeric, nan=0.0, posinf=1.0, neginf=0.0)
    propensity_floor = 0.05
    row_scores = propensity_floor + scores.mean(axis=1)
    selected = rng.choice(
        n, size=int(round(rho * n)), replace=False,
        p=row_scores / row_scores.sum())
    columns = np.empty(len(selected), dtype=np.int64)
    for slot, row in enumerate(selected):
        attribute_scores = propensity_floor + scores[int(row)]
        columns[slot] = rng.choice(
            m, p=attribute_scores / attribute_scores.sum())
    mask = _empty_mask(n, m)
    mask[selected, columns] = True
    return mask


def apply_mask(table: pd.DataFrame, cols, mask: np.ndarray) -> pd.DataFrame:
    out = table.copy()
    for j, c in enumerate(cols):
        col_mask = mask[:, j]
        if not col_mask.any():
            continue
        # bool / int columns cannot hold NaN -> promote to object so masked cells
        # become a real missing sentinel that 3-valued eval treats as UNKNOWN.
        if out[c].dtype == bool or pd.api.types.is_integer_dtype(out[c].dtype):
            out[c] = out[c].astype(object)
        out.loc[col_mask, c] = np.nan
    return out


def _injected_cache_path(root, spec, kind, rho, seed):
    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(spec.name))
    safe_kind = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(kind))
    return (Path(root) / safe_name /
            f"{safe_kind}_rho{float(rho):.6f}_seed{int(seed)}")


def _load_injected_mask(spec, cols, rho, kind, seed):
    root = os.environ.get("MIFA_INJECTED_DATA_ROOT")
    if not root:
        return None
    path = _injected_cache_path(root, spec, kind, rho, seed) / "mask.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"pre-generated missingness mask is absent: {path}")
    mask = np.load(path, allow_pickle=False).astype(bool, copy=False)
    expected = (spec.n, len(cols))
    if mask.shape != expected:
        raise ValueError(
            f"pre-generated mask {path} has shape {mask.shape}; "
            f"expected {expected}")
    return mask


def _export_injected_snapshot(spec, cols, rho, kind, seed, mask, observed):
    root = os.environ.get("MIFA_EXPORT_INJECTED_ROOT")
    if not root:
        return
    path = _injected_cache_path(root, spec, kind, rho, seed)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "mask.npy", np.asarray(mask, dtype=bool))
    observed.to_parquet(path / "observed_table.parquet", index=False)
    metadata = {
        "schema_version": 1,
        "dataset_partition": str(spec.name),
        "mechanism": os.environ.get("MIFA_INJECTION_LABEL", str(kind)),
        "implementation_kind": str(kind),
        "rho": float(rho),
        "seed": int(seed),
        "rows": int(spec.n),
        "query_relevant_attributes": list(cols),
        "masked_cells": int(np.asarray(mask, dtype=bool).sum()),
        "observed_table": "observed_table.parquet",
        "mask": "mask.npy",
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def make_missing(spec, rho, kind="mcar", usage_freq=None, seed=0, block_size=2,
                 block_attrs=None, queries=None, truth_ids=None,
                 answer_impact_fraction=1.0, answer_cells_per_tuple=1,
                 historical_queries=None, historical_candidate_ids=None,
                 beta_exp=0.50, historical_priority=None):
    """Return (observed_table, mask, cols). Operates on query_relevant_attrs only."""
    cols = spec.query_relevant_attrs
    n, m = spec.n, len(cols)
    cached_mask = _load_injected_mask(spec, cols, rho, kind, seed)
    if cached_mask is not None:
        observed = apply_mask(spec.table, cols, cached_mask)
        return observed, cached_mask, cols
    rng = np.random.default_rng(seed)
    if kind == "mcar":
        mask = mcar_mask(n, m, rho, rng)
    elif kind == "tuple_mcar":
        mask = tuple_mcar_mask(n, m, rho, rng)
    elif kind == "tuple_mcar_nested":
        mask = tuple_mcar_nested_mask(n, m, rho, rng)
    elif kind == "single_attribute_mcar_nested":
        if not block_attrs or len(block_attrs) != 1:
            raise ValueError(
                "single_attribute_mcar_nested requires exactly one "
                "predeclared attribute")
        attribute = str(block_attrs[0])
        if attribute not in cols:
            raise ValueError(
                f"single-attribute MCAR contains unknown attribute: {attribute}")
        mask = single_attribute_mcar_nested_mask(
            n, m, rho, cols.index(attribute), rng)
    elif kind == "tuple_predicate_mcar":
        if not block_attrs:
            raise ValueError(
                "tuple_predicate_mcar requires predeclared predicate attributes")
        unknown = sorted(set(block_attrs) - set(cols))
        if unknown:
            raise ValueError(
                f"predicate MCAR contains unknown attributes: {unknown}")
        allowed_indices = [cols.index(attr) for attr in block_attrs]
        mask = tuple_predicate_mcar_mask(
            n, m, rho, allowed_indices, rng)
    elif kind == "multi_attribute_mcar":
        if not block_attrs:
            raise ValueError(
                "multi_attribute_mcar requires designated attributes")
        unknown = sorted(set(block_attrs) - set(cols))
        if unknown:
            raise ValueError(
                "multi-attribute MCAR contains unknown attributes: "
                f"{unknown}")
        mask = multi_attribute_mcar_mask(
            n, m, rho, [cols.index(attr) for attr in block_attrs], rng)
    elif kind == "tuple_predicate_block":
        if not block_attrs:
            raise ValueError(
                "tuple_predicate_block requires predeclared attributes")
        unknown = sorted(set(block_attrs) - set(cols))
        if unknown:
            raise ValueError(
                f"predicate block contains unknown attributes: {unknown}")
        allowed_indices = [cols.index(attr) for attr in block_attrs]
        mask = tuple_predicate_block_mask(
            n, m, rho, allowed_indices, answer_cells_per_tuple, rng)
    elif kind == "tuple_query_answer_mnar":
        allowed_indices = None
        if block_attrs:
            unknown = sorted(set(block_attrs) - set(cols))
            if unknown:
                raise ValueError(
                    f"query-answer MNAR contains unknown attributes: {unknown}")
            allowed_indices = [cols.index(attr) for attr in block_attrs]
        if queries is None or truth_ids is None:
            raise ValueError(
                "tuple_query_answer_mnar requires frozen queries and truth_ids")
        mask = tuple_query_answer_mnar_mask(
            n, m, rho, cols, queries, truth_ids, rng,
            allowed_indices=allowed_indices,
            impact_fraction=answer_impact_fraction,
            cells_per_tuple=answer_cells_per_tuple)
    elif kind in {
            "historical_candidate_exposure",
            "historical_candidate_exposure_nested"}:
        if not block_attrs:
            raise ValueError(
                "historical candidate exposure requires designated attributes")
        unknown = sorted(set(block_attrs) - set(cols))
        if unknown:
            raise ValueError(
                "historical candidate exposure contains unknown attributes: "
                f"{unknown}")
        if historical_queries is None or historical_candidate_ids is None:
            raise ValueError(
                "historical candidate exposure requires frozen historical "
                "queries and candidate IDs")
        builder = (
            historical_candidate_exposure_nested_mask
            if kind == "historical_candidate_exposure_nested"
            else historical_candidate_exposure_mask)
        kwargs = ({"priority": historical_priority}
                  if kind == "historical_candidate_exposure_nested" else {})
        mask = builder(
            n, m, rho, [cols.index(attr) for attr in block_attrs],
            historical_queries, historical_candidate_ids,
            beta_exp=beta_exp, rng=rng, **kwargs)
    elif kind in {"tuple_attr_skewed", "tuple_attribute_skewed"}:
        if usage_freq is None:
            usage_freq = np.ones(m)
        mask = tuple_attr_skewed_mask(n, m, rho, usage_freq, rng)
    elif kind == "attr_skewed":
        if usage_freq is None:
            usage_freq = np.ones(m)
        mask = attr_skewed_mask(n, m, rho, usage_freq, rng)
    elif kind == "tuple_block":
        mask = tuple_block_mask(n, m, rho, rng)
    elif kind in {"fixed_tuple_block", "joint_tuple_block"}:
        block_indices = None
        if block_attrs:
            unknown = sorted(set(block_attrs) - set(cols))
            if unknown:
                raise ValueError(
                    f"fixed joint block contains unknown attributes: {unknown}")
            block_indices = [cols.index(attr) for attr in block_attrs]
        builder = (fixed_tuple_block_mask if kind == "fixed_tuple_block"
                   else joint_tuple_block_mask)
        mask = builder(n, m, rho, rng, block_size=block_size,
                       block_indices=block_indices)
    elif kind == "tuple_value_dependent":
        mask = tuple_value_dependent_mask(spec.table, cols, rho, rng)
    else:
        raise ValueError(f"unknown missingness kind: {kind}")
    observed = apply_mask(spec.table, cols, mask)
    _export_injected_snapshot(
        spec, cols, rho, kind, seed, mask, observed)
    return observed, mask, cols


def multi_attribute_mcar_mask(n, m, rho, designated_indices, rng):
    """Multi-attribute MCAR: allocates B_s = round(rho*n) budget across designated attributes.

    Attributes share the budget without row overlap (each sampled row has exactly 1 masked cell).
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    desig = [int(idx) for idx in designated_indices]
    if not desig:
        raise ValueError("designated_indices must not be empty")
    k = len(desig)
    total_budget = int(round(rho * n))
    mask = _empty_mask(n, m)
    if total_budget == 0:
        return mask

    # Allocate per-attribute budgets
    base_b = total_budget // k
    rem = total_budget % k
    budgets = [base_b] * k
    if rem:
        for index in range(k - rem, k):
            budgets[index] += 1

    # Sample total_budget distinct rows uniformly
    selected_rows = rng.choice(n, size=total_budget, replace=False)
    offset = 0
    for j, col_idx in enumerate(desig):
        b = budgets[j]
        if b > 0:
            rows_for_col = selected_rows[offset : offset + b]
            mask[rows_for_col, col_idx] = True
            offset += b
    return mask


def historical_candidate_exposure_mask(
    n,
    m,
    rho,
    designated_indices,
    historical_queries,
    historical_candidate_ids,
    beta_exp=0.50,
    rng=None,
):
    """Historical Candidate Exposure Mask (Primary Mechanism).

    Uses historical queries and candidate retrieval sets C_L(q) to guide exposure-based
    missingness without leaking test queries or ground truth.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    desig = [int(idx) for idx in designated_indices]
    if not desig:
        raise ValueError("designated_indices must not be empty")
    if len(set(desig)) != len(desig) or any(idx < 0 or idx >= m for idx in desig):
        raise ValueError("designated_indices must be unique and in range")
    if not 0.0 <= beta_exp <= 1.0:
        raise ValueError("beta_exp must be in [0, 1]")
    if len(historical_queries) != len(historical_candidate_ids):
        raise ValueError(
            "historical queries and candidate lists must be row-aligned")
    k = len(desig)
    total_budget = int(round(rho * n))
    mask = _empty_mask(n, m)
    if total_budget == 0:
        return mask

    base_b = total_budget // k
    rem = total_budget % k
    budgets = [base_b] * k
    if rem:
        for index in range(k - rem, k):
            budgets[index] += 1

    if beta_exp <= 0.0:
        return multi_attribute_mcar_mask(n, m, rho, desig, rng)

    selected_rows_all = set()

    for j, col_idx in enumerate(desig):
        b_j = budgets[j]
        if b_j == 0:
            continue
        b_exp = int(round(beta_exp * b_j))

        # Compute exposure scores from historical candidate retrieval
        scores = np.zeros(n, dtype=np.float64)
        for q_dict, cand_list in zip(historical_queries, historical_candidate_ids):
            # Check if query references this attribute or general workload
            q_attrs = q_dict.get("attributes", []) if isinstance(q_dict, dict) else []
            if col_idx in q_attrs:
                for cand_id in cand_list:
                    if 0 <= cand_id < n:
                        scores[cand_id] += 1.0

        # Mask candidates with score > 0
        candidate_pool = [i for i in np.where(scores > 0)[0] if i not in selected_rows_all]
        sampled_exp = []
        if candidate_pool and b_exp > 0:
            pool_scores = scores[candidate_pool]
            pool_probs = pool_scores / np.sum(pool_scores)
            n_sample = min(b_exp, len(candidate_pool))
            sampled_exp = rng.choice(candidate_pool, size=n_sample, replace=False, p=pool_probs).tolist()

        # Fill remaining budget uniformly from unselected rows
        selected_rows_all.update(sampled_exp)
        remaining_needed = b_j - len(sampled_exp)
        if remaining_needed > 0:
            unselected = [i for i in range(n) if i not in selected_rows_all]
            if len(unselected) < remaining_needed:
                raise RuntimeError(f"Insufficient unselected rows to fulfill budget for col {col_idx}")
            sampled_rem = rng.choice(unselected, size=remaining_needed, replace=False).tolist()
            selected_rows_all.update(sampled_rem)
            col_rows = sampled_exp + sampled_rem
        else:
            col_rows = sampled_exp

        mask[col_rows, col_idx] = True

    return mask


def _array_digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _historical_exposure_scores(
    n, m, designated_indices, historical_queries, historical_candidate_ids
):
    """Return exposure counts aligned with the declared attributes."""
    desig = [int(index) for index in designated_indices]
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not desig:
        raise ValueError("designated_indices must not be empty")
    if (len(set(desig)) != len(desig)
            or any(index < 0 or index >= m for index in desig)):
        raise ValueError(
            "designated_indices must be unique and in range")
    if len(historical_queries) != len(historical_candidate_ids):
        raise ValueError(
            "historical queries and candidate lists must be row-aligned")
    scores = {index: np.zeros(n, dtype=np.float64) for index in desig}
    declared = set(desig)
    for query, candidate_ids in zip(
            historical_queries, historical_candidate_ids):
        attributes = (
            query.get("attributes", ()) if isinstance(query, dict) else ())
        used = declared.intersection(int(index) for index in attributes)
        if not used:
            continue
        candidates = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        candidates = candidates[(candidates >= 0) & (candidates < n)]
        if not len(candidates):
            continue
        for index in used:
            np.add.at(scores[index], candidates, 1.0)
    return desig, scores


def historical_candidate_exposure_priority(
    n,
    m,
    designated_indices,
    historical_queries,
    historical_candidate_ids,
    beta_exp=0.50,
    rng=None,
):
    """Freeze one exposure-weighted assignment order for all rho values.

    This is an explicit, versioned alternative to
    :func:`historical_candidate_exposure_mask`; it does not alter that legacy
    protocol.  Each designated attribute receives one exposure-weighted order
    and one uniform order.  A reverse round-robin attribute schedule preserves
    the legacy per-attribute budget rule at every global prefix: the remainder
    is assigned to the last declared attributes.  Exposure and uniform slots
    are interleaved so the first ``b`` assignments for an attribute contain
    exactly ``round(beta_exp * b)`` exposure-directed slots (unless its
    positive-exposure pool is exhausted).  Rows are assigned globally at most
    once, hence every larger rho is a strict extension whenever its rounded
    budget increases.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if not 0.0 <= beta_exp <= 1.0:
        raise ValueError("beta_exp must be in [0, 1]")
    desig, scores = _historical_exposure_scores(
        n, m, designated_indices, historical_queries,
        historical_candidate_ids)

    exposure_orders = {}
    uniform_orders = {}
    for index in desig:
        positive = np.flatnonzero(scores[index] > 0.0)
        if len(positive):
            # An exponential race is a deterministic weighted-without-
            # replacement priority once the RNG stream is frozen.
            uniforms = np.maximum(
                rng.random(len(positive)), np.finfo(np.float64).tiny)
            keys = -np.log(uniforms) / scores[index][positive]
            order = np.lexsort((positive, keys))
            exposure_orders[index] = np.ascontiguousarray(
                positive[order], dtype=np.int64)
        else:
            exposure_orders[index] = np.empty(0, dtype=np.int64)
        uniform_orders[index] = np.ascontiguousarray(
            rng.permutation(n), dtype=np.int64)

    used = np.zeros(n, dtype=np.bool_)
    exposure_cursors = {index: 0 for index in desig}
    uniform_cursors = {index: 0 for index in desig}
    attribute_counts = {index: 0 for index in desig}
    assigned_rows = np.empty(n, dtype=np.int64)
    assigned_attributes = np.empty(n, dtype=np.int64)
    reverse_schedule = tuple(reversed(desig))

    def take_next(order, cursors, index):
        cursor = cursors[index]
        while cursor < len(order) and used[int(order[cursor])]:
            cursor += 1
        cursors[index] = cursor
        if cursor == len(order):
            return None
        row = int(order[cursor])
        cursors[index] = cursor + 1
        return row

    for rank in range(n):
        index = reverse_schedule[rank % len(reverse_schedule)]
        previous = attribute_counts[index]
        current = previous + 1
        wants_exposure = (
            int(round(beta_exp * current))
            > int(round(beta_exp * previous)))
        row = None
        if wants_exposure:
            row = take_next(
                exposure_orders[index], exposure_cursors, index)
        if row is None:
            row = take_next(uniform_orders[index], uniform_cursors, index)
        if row is None:
            raise RuntimeError(
                "frozen exposure priority exhausted before all rows were "
                "assigned")
        used[row] = True
        attribute_counts[index] = current
        assigned_rows[rank] = row
        assigned_attributes[rank] = index

    if len(np.unique(assigned_rows)) != n:
        raise RuntimeError("frozen exposure priority contains duplicate rows")
    digest = hashlib.sha256()
    digest.update(HISTORICAL_EXPOSURE_NESTED_PROTOCOL.encode("utf-8"))
    digest.update(np.asarray([n, m], dtype=np.int64).tobytes())
    digest.update(np.asarray(desig, dtype=np.int64).tobytes())
    digest.update(np.asarray([beta_exp], dtype=np.float64).tobytes())
    digest.update(assigned_rows.tobytes())
    digest.update(assigned_attributes.tobytes())
    per_attribute = {}
    for index in desig:
        assigned = assigned_rows[assigned_attributes == index]
        per_attribute[str(index)] = {
            "exposure_scores_digest": _array_digest(scores[index]),
            "weighted_priority_digest": _array_digest(
                exposure_orders[index]),
            "uniform_priority_digest": _array_digest(uniform_orders[index]),
            "assigned_priority_digest": _array_digest(assigned),
            "positive_exposure_rows": int(
                np.count_nonzero(scores[index] > 0.0)),
            "assigned_rows": int(len(assigned)),
        }
    manifest = {
        "protocol": HISTORICAL_EXPOSURE_NESTED_PROTOCOL,
        "beta_exp": float(beta_exp),
        "n_rows": int(n),
        "mask_width": int(m),
        "designated_indices": desig,
        "priority_digest": digest.hexdigest(),
        "per_attribute": per_attribute,
        "row_uniqueness": True,
        "attribute_schedule": "reverse_round_robin_remainder_to_last",
    }
    return HistoricalExposurePriority(
        rows=assigned_rows,
        attributes=assigned_attributes,
        manifest=manifest)


def historical_candidate_exposure_nested_mask(
    n,
    m,
    rho,
    designated_indices,
    historical_queries,
    historical_candidate_ids,
    beta_exp=0.50,
    rng=None,
    priority=None,
):
    """Take a rho-sized prefix from one frozen exposure priority."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if priority is None:
        priority = historical_candidate_exposure_priority(
            n, m, designated_indices, historical_queries,
            historical_candidate_ids, beta_exp=beta_exp, rng=rng)
    if not isinstance(priority, HistoricalExposurePriority):
        raise TypeError("priority must be HistoricalExposurePriority")
    if (len(priority.rows) != n or len(priority.attributes) != n
            or priority.manifest.get("n_rows") != n
            or priority.manifest.get("mask_width") != m):
        raise ValueError("priority shape does not match mask shape")
    if not np.isclose(
            float(priority.manifest.get("beta_exp")), float(beta_exp),
            rtol=0.0, atol=0.0):
        raise ValueError("priority beta_exp does not match requested beta_exp")
    selected_count = int(round(rho * n))
    mask = _empty_mask(n, m)
    rows = priority.rows[:selected_count]
    attributes = priority.attributes[:selected_count]
    mask[rows, attributes] = True
    return mask


def historical_candidate_exposure_nested_audit(
    priority, n, m, rhos=(0.1, 0.2, 0.3)
):
    """Return machine-readable mask digests and strict subset checks."""
    values = tuple(float(rho) for rho in rhos)
    if any(not 0.0 <= rho <= 1.0 for rho in values):
        raise ValueError("rho audit values must be in [0, 1]")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("rho audit values must be strictly increasing")
    levels = []
    masks = []
    designated = [int(index) for index in
                  priority.manifest["designated_indices"]]
    for rho in values:
        count = int(round(rho * n))
        mask = _empty_mask(n, m)
        mask[priority.rows[:count], priority.attributes[:count]] = True
        masks.append(mask)
        levels.append({
            "rho": rho,
            "budget": count,
            "masked_cells": int(mask.sum()),
            "mask_digest": _array_digest(np.packbits(mask)),
            "per_attribute_budget": {
                str(index): int(mask[:, index].sum())
                for index in designated},
        })
    subset_checks = []
    for left_index, (left, right) in enumerate(zip(masks, masks[1:])):
        lost = int(np.count_nonzero(left & ~right))
        added = int(np.count_nonzero(right & ~left))
        subset_checks.append({
            "lower_rho": values[left_index],
            "higher_rho": values[left_index + 1],
            "lost_masked_cells": lost,
            "added_masked_cells": added,
            "strict_subset": bool(lost == 0 and added > 0),
        })
    return {
        **priority.manifest,
        "rho_levels": levels,
        "subset_checks": subset_checks,
        "all_strict_subsets": all(
            item["strict_subset"] for item in subset_checks),
    }


def query_answer_oracle_stress_mask(
    n,
    m,
    rho,
    designated_indices,
    evaluation_queries,
    evaluation_answers,
    gamma_ans=0.50,
    rng=None,
):
    """Query-Answer Oracle Stress Mask (Appendix Upper Bound Stress)."""
    if rng is None:
        rng = np.random.default_rng(0)
    total_budget = int(round(rho * n))
    mask = _empty_mask(n, m)
    if total_budget == 0:
        return mask, 0.0

    desig = [int(idx) for idx in designated_indices]
    selected_rows_all = set()
    total_answer_slots = 0
    masked_answer_slots = 0

    for q_dict, ans_list in zip(evaluation_queries, evaluation_answers):
        total_answer_slots += len(ans_list)
        if not ans_list:
            continue
        r_q = int(round(gamma_ans * len(ans_list)))
        if r_q <= 0:
            continue
        avail_ans = [i for i in ans_list if i not in selected_rows_all and 0 <= i < n]
        if avail_ans and len(selected_rows_all) < total_budget:
            n_sel = min(r_q, len(avail_ans), total_budget - len(selected_rows_all))
            sampled_ans = rng.choice(avail_ans, size=n_sel, replace=False).tolist()
            col_choice = rng.choice(desig)
            mask[sampled_ans, col_choice] = True
            selected_rows_all.update(sampled_ans)
            masked_answer_slots += n_sel

    # Fill remaining budget with unselected rows
    if len(selected_rows_all) < total_budget:
        unselected = [i for i in range(n) if i not in selected_rows_all]
        rem = total_budget - len(selected_rows_all)
        sampled_fill = rng.choice(unselected, size=rem, replace=False).tolist()
        col_choices = rng.choice(desig, size=rem, replace=True)
        for r_id, c_id in zip(sampled_fill, col_choices):
            mask[r_id, c_id] = True
        selected_rows_all.update(sampled_fill)

    eta_ans = masked_answer_slots / max(1, total_answer_slots)
    return mask, eta_ans
