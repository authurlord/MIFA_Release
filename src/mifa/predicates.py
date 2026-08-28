"""Predicate language + 3-valued evaluation (paper §2).

Atomic conditions over a scalar attribute:
  - ('le', c)        A <= c
  - ('ge', c)        A >= c
  - ('between', l,u) A in [l,u]
  - ('eq', v)        A = v
  - ('in', V)        A in V   (V a set)

A Predicate is a conjunction of atoms. Evaluation against an observed value is
3-valued: TRUE / FALSE / UNKNOWN. UNKNOWN arises when the value is missing (NaN/None).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRUE, FALSE, UNKNOWN = "true", "false", "unknown"


def _num_cmp(vals, missing, fn):
    """Apply a numeric comparison only to non-missing cells; False where missing."""
    out = np.zeros(len(vals), dtype=bool)
    valid = ~missing
    if valid.any():
        v = np.asarray(vals[valid], dtype=float)
        out[valid] = fn(v)
    return out


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


@dataclass
class Atom:
    attr: str
    op: str
    a: object = None       # c, l, or v
    b: object = None       # u for between

    def eval_value(self, v):
        if _is_missing(v):
            return UNKNOWN
        if self.op == "le":
            return TRUE if v <= self.a else FALSE
        if self.op == "ge":
            return TRUE if v >= self.a else FALSE
        if self.op == "between":
            return TRUE if (self.a <= v <= self.b) else FALSE
        if self.op == "eq":
            return TRUE if v == self.a else FALSE
        if self.op == "in":
            return TRUE if v in self.a else FALSE
        raise ValueError(self.op)


@dataclass
class Predicate:
    atoms: list

    @property
    def attrs(self):
        return [at.attr for at in self.atoms]

    def eval_row(self, row) -> str:
        """3-valued conjunction. FALSE dominates; any UNKNOWN -> UNKNOWN; else TRUE."""
        seen_unknown = False
        for at in self.atoms:
            r = at.eval_value(row.get(at.attr))
            if r == FALSE:
                return FALSE
            if r == UNKNOWN:
                seen_unknown = True
        return UNKNOWN if seen_unknown else TRUE

    def eval_table(self, table: pd.DataFrame) -> np.ndarray:
        """Return array of {true,false,unknown} per row (vectorized over rows).

        3-valued conjunction: FALSE dominates; any UNKNOWN (missing) -> UNKNOWN;
        else TRUE. Computed column-wise with numpy masks instead of a Python loop.
        """
        n = len(table)
        any_false = np.zeros(n, dtype=bool)
        any_unknown = np.zeros(n, dtype=bool)
        for at in self.atoms:
            col = table[at.attr]
            missing = col.isna().to_numpy()
            vals = col.to_numpy()
            sat = self._atom_sat(at, vals, missing)   # bool, only valid where !missing
            any_false |= (~missing) & (~sat)
            any_unknown |= missing
        out = np.full(n, TRUE, dtype=object)
        out[any_unknown] = UNKNOWN
        out[any_false] = FALSE     # FALSE dominates UNKNOWN
        return out

    @staticmethod
    def _atom_sat(at, vals, missing):
        """Vectorized satisfaction of a single atom; result undefined where missing."""
        if at.op == "le":
            return _num_cmp(vals, missing, lambda v: v <= at.a)
        if at.op == "ge":
            return _num_cmp(vals, missing, lambda v: v >= at.a)
        if at.op == "between":
            return _num_cmp(vals, missing, lambda v: (v >= at.a) & (v <= at.b))
        if at.op == "eq":
            out = np.zeros(len(vals), dtype=bool)
            valid = ~missing
            out[valid] = vals[valid] == at.a
            return out
        if at.op == "in":
            S = set(at.a)
            out = np.zeros(len(vals), dtype=bool)
            valid = np.where(~missing)[0]
            for i in valid:
                out[i] = vals[i] in S
            return out
        raise ValueError(at.op)

    def selectivity(self, table: pd.DataFrame) -> float:
        """Fraction of rows that satisfy the predicate on COMPLETE data (TRUE)."""
        r = self.eval_table(table)
        return float(np.mean(r == TRUE))
