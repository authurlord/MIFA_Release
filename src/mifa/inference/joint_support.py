"""Training-only joint residual support for paper protocol v2.

This is the exact inference primitive required by the draft: it counts supports
satisfying the entire residual conjunction.  It provides the body-free
``GlobalJoint`` baseline and a location/scalar-aware policy with nested
body+cell -> body -> global backoff. Product/min marginals are exposed only as
ablations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mifa.inference.joint_support_numba import (
    ABSTAIN, BODY, BODY_CELL, CELL, GLOBAL,
    joint_context_kernel, joint_grouped_context_kernel,
)
from mifa.predicates import FALSE, TRUE, UNKNOWN, Predicate


class JointSupportEngine:
    MODES = {"joint", "product", "min"}

    def __init__(self, support_table, target_observed, *, mode="joint",
                 support_cells=None, target_cells=None, body_by_target=None,
                 dtypes=None, n_bins=8, min_support=30, kappa=20.0):
        if mode not in self.MODES:
            raise ValueError(f"unknown support mode: {mode}")
        if (support_cells is None) != (target_cells is None):
            raise ValueError("support_cells and target_cells must be supplied together")
        self.support = support_table.reset_index(drop=True)
        self.target = target_observed.reset_index(drop=True)
        self.mode = mode
        self.support_cells = None if support_cells is None else np.asarray(support_cells)
        self.target_cells = None if target_cells is None else np.asarray(target_cells)
        self.body_by_target = {
            target: tuple(body) for target, body in (body_by_target or {}).items()
        }
        self.dtypes = dict(dtypes or {})
        self._support_missing = {
            attr: self.support[attr].isna().to_numpy()
            for attr in self.support.columns
        }
        self._target_missing = {
            attr: self.target[attr].isna().to_numpy()
            for attr in self.target.columns
        }
        self._body_code_cache = {}
        self._group_code_cache = {}
        self.n_bins = int(n_bins)
        self.bin_edges = {}
        for attr, dtype in self.dtypes.items():
            if dtype != "num" or attr not in self.support:
                continue
            values = self.support[attr].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            self.bin_edges[attr] = np.unique(
                np.quantile(values, np.linspace(0, 1, self.n_bins + 1))
            )[1:-1] if len(values) else np.array([])
        self.min_support = int(min_support)
        self.kappa = float(kappa)
        if self.min_support <= 0 or self.kappa < 0:
            raise ValueError("min_support must be positive and kappa non-negative")
        if self.support_cells is not None and len(self.support_cells) != len(self.support):
            raise ValueError("support_cells length mismatch")
        if self.target_cells is not None and len(self.target_cells) != len(self.target):
            raise ValueError("target_cells length mismatch")

    @staticmethod
    def _complete_mask(table, atoms):
        attrs = list(dict.fromkeys(atom.attr for atom in atoms))
        return table[attrs].notna().all(axis=1).to_numpy()

    @staticmethod
    def _probability(table, atoms, rows, mode):
        n = int(rows.sum())
        if n == 0:
            return np.nan, 0, 0
        if mode == "joint":
            sat = Predicate(atoms).eval_table(table) == TRUE
            n_plus = int((sat & rows).sum())
            return n_plus / n, n, n_plus
        probs = []
        for atom in atoms:
            atom_rows = rows & table[atom.attr].notna().to_numpy()
            denom = int(atom_rows.sum())
            if denom == 0:
                return np.nan, 0, 0
            sat = Predicate([atom]).eval_table(table) == TRUE
            probs.append(float((sat & atom_rows).sum()) / denom)
        p = float(np.prod(probs) if mode == "product" else np.min(probs))
        return p, n, int(round(p * n))

    def _matching_body_rows(self, atoms, target_row, complete):
        """Training rows matching the frozen scalar body visible on target_row."""
        residual_attrs = {atom.attr for atom in atoms}
        body = []
        for target in residual_attrs:
            for attr in self.body_by_target.get(target, ()):
                if attr not in residual_attrs and attr not in body:
                    body.append(attr)
        rows = complete.copy()
        used = []
        for attr in body:
            if attr not in self.target or self.target[attr].isna().iat[target_row]:
                continue
            target_value = self.target[attr].iat[target_row]
            observed = self.support[attr].notna().to_numpy()
            if self.dtypes.get(attr) == "num":
                edges = self.bin_edges.get(attr, np.array([]))
                target_bin = int(np.digitize([float(target_value)], edges)[0])
                support_values = self.support[attr].to_numpy(dtype=float)
                support_bins = np.digitize(np.nan_to_num(support_values), edges)
                rows &= observed & (support_bins == target_bin)
            else:
                rows &= observed & (self.support[attr].to_numpy() == target_value)
            used.append(attr)
        return rows, tuple(used)

    def _contexts(self, atoms, target_row, complete):
        """Nested fine-to-broad support contexts for one unresolved instance."""
        body_rows, body = self._matching_body_rows(atoms, target_row, complete)
        contexts = []
        if body:
            if self.support_cells is not None:
                contexts.append(("body_cell", body_rows &
                                 (self.support_cells == self.target_cells[target_row])))
            contexts.append(("body", body_rows))
        elif self.support_cells is not None:
            contexts.append(("cell", complete &
                             (self.support_cells == self.target_cells[target_row])))
        contexts.append(("global", complete))
        return contexts

    def _estimate_contexts(self, atoms, target_row, complete):
        estimates = []
        for name, rows in self._contexts(atoms, target_row, complete):
            p, n, nplus = self._probability(self.support, atoms, rows, self.mode)
            estimates.append({"name": name, "p_raw": p, "n": n, "nplus": nplus})
        # Shrink each finer context toward its next broader context. The
        # broadest global estimate is unchanged.
        broader = estimates[-1]["p_raw"]
        estimates[-1]["p"] = broader
        for estimate in reversed(estimates[:-1]):
            if np.isfinite(estimate["p_raw"]):
                estimate["p"] = ((estimate["n"] * estimate["p_raw"] +
                                  self.kappa * broader) /
                                 (estimate["n"] + self.kappa)) \
                    if np.isfinite(broader) else estimate["p_raw"]
                broader = estimate["p"]
            else:
                estimate["p"] = broader
        for estimate in estimates:
            if estimate["n"] >= self.min_support and np.isfinite(estimate["p"]):
                return estimate
        return None

    def certificate(self, predicate):
        """Readable reference implementation used by correctness tests."""
        direct = predicate.eval_table(self.target)
        pass_mask = direct == TRUE
        fail_mask = direct == FALSE
        unresolved = direct == UNKNOWN
        n_target = len(self.target)
        p_phi = np.where(pass_mask, 1.0, 0.0)
        support_n = np.zeros(n_target, dtype=np.int64)
        support_n_plus = np.zeros(n_target, dtype=np.int64)
        local_used = np.zeros(n_target, dtype=bool)
        body_used = np.zeros(n_target, dtype=bool)
        context_name = np.full(n_target, "direct", dtype=object)
        abstain_mask = np.zeros(n_target, dtype=bool)

        # Rows can have different missing subsets under block mechanisms. Group
        # them by residual atom positions so each support count is shared.
        patterns: dict[tuple[int, ...], list[int]] = {}
        for row in np.flatnonzero(unresolved):
            residual = tuple(i for i, atom in enumerate(predicate.atoms)
                             if self.target[atom.attr].isna().iat[row])
            patterns.setdefault(residual, []).append(int(row))

        for pattern, target_rows in patterns.items():
            atoms = [predicate.atoms[i] for i in pattern]
            complete = self._complete_mask(self.support, atoms)
            for row in target_rows:
                estimate = self._estimate_contexts(atoms, row, complete)
                if estimate is None:
                    abstain_mask[row] = True
                    p_phi[row] = 0.0
                    context_name[row] = "abstain"
                else:
                    p_phi[row] = float(np.clip(estimate["p"], 0.0, 1.0))
                    support_n[row] = estimate["n"]
                    support_n_plus[row] = estimate["nplus"]
                    context_name[row] = estimate["name"]
                    local_used[row] = estimate["name"] in {"cell", "body_cell"}
                    body_used[row] = estimate["name"] in {"body", "body_cell"}

        uncertain_mask = unresolved & ~abstain_mask
        states = np.full(n_target, FALSE, dtype=object)
        states[uncertain_mask] = UNKNOWN
        states[pass_mask] = TRUE
        return {
            "p_phi": p_phi,
            "pass_mask": pass_mask,
            "fail_mask": fail_mask,
            "uncertain_mask": uncertain_mask,
            "abstain_mask": abstain_mask,
            "support_n": support_n,
            "support_n_plus": support_n_plus,
            "local_used": local_used,
            "body_used": body_used,
            "context_name": context_name,
            "states": states,
        }

    def _body_codes(self, attr):
        """Aligned integer codes for body matching, cached per engine/attribute."""
        cached = self._body_code_cache.get(attr)
        if cached is not None:
            return cached
        support_missing = self._support_missing[attr]
        target_missing = self._target_missing[attr]
        if self.dtypes.get(attr) == "num":
            edges = self.bin_edges.get(attr, np.array([]))
            support_values = self.support[attr].to_numpy(dtype=float)
            target_values = self.target[attr].to_numpy(dtype=float)
            support_codes = np.digitize(
                np.nan_to_num(support_values), edges).astype(np.int64)
            target_codes = np.digitize(
                np.nan_to_num(target_values), edges).astype(np.int64)
            support_codes[support_missing] = -1
            target_codes[target_missing] = -1
        else:
            combined = pd.concat(
                [self.support[attr], self.target[attr]], ignore_index=True)
            codes, _ = pd.factorize(combined, sort=False)
            support_codes = codes[:len(self.support)].astype(np.int64)
            target_codes = codes[len(self.support):].astype(np.int64)
        cached = (np.ascontiguousarray(support_codes),
                  np.ascontiguousarray(target_codes))
        self._body_code_cache[attr] = cached
        return cached

    def _global_probability(self, complete, joint_satisfied, atom_satisfied):
        n = int(complete.sum())
        if n == 0:
            return np.nan, 0, 0
        if self.mode == "joint":
            nplus = int((joint_satisfied & complete).sum())
            return nplus / n, n, nplus
        probabilities = atom_satisfied[:, complete].mean(axis=1)
        p = float(np.prod(probabilities) if self.mode == "product"
                  else np.min(probabilities))
        return p, n, int(round(p * n))

    @staticmethod
    def _compact_groups(support_values, target_values):
        combined = np.concatenate([support_values, target_values], axis=0)
        _, inverse = np.unique(combined, axis=0, return_inverse=True)
        split = len(support_values)
        return (np.ascontiguousarray(inverse[:split], dtype=np.int64),
                np.ascontiguousarray(inverse[split:], dtype=np.int64),
                int(inverse.max(initial=-1) + 1))

    def _context_group_codes(self, body_attrs):
        """Cache compact body/cell/body-cell ids independent of the predicate."""
        key = tuple(body_attrs)
        cached = self._group_code_cache.get(key)
        if cached is not None:
            return cached
        if body_attrs:
            support_matrix = np.stack(
                [self._body_codes(attr)[0] for attr in body_attrs], axis=1)
            target_matrix = np.stack(
                [self._body_codes(attr)[1] for attr in body_attrs], axis=1)
            support_body, target_body, n_body = self._compact_groups(
                support_matrix, target_matrix)
            target_body_missing = (target_matrix < 0).any(axis=1)
        else:
            support_body = np.zeros(len(self.support), dtype=np.int64)
            target_body = np.zeros(len(self.target), dtype=np.int64)
            n_body = 1
            target_body_missing = np.zeros(len(self.target), dtype=bool)
        if self.support_cells is not None:
            support_cell, target_cell, n_cell = self._compact_groups(
                np.asarray(self.support_cells, dtype=np.int64)[:, None],
                np.asarray(self.target_cells, dtype=np.int64)[:, None])
        else:
            support_cell = np.zeros(len(self.support), dtype=np.int64)
            target_cell = np.zeros(len(self.target), dtype=np.int64)
            n_cell = 1
        if body_attrs and self.support_cells is not None:
            support_pair = np.column_stack([support_body, support_cell])
            target_pair = np.column_stack([target_body, target_cell])
            support_body_cell, target_body_cell, n_body_cell = \
                self._compact_groups(support_pair, target_pair)
        else:
            support_body_cell = np.zeros(len(self.support), dtype=np.int64)
            target_body_cell = np.zeros(len(self.target), dtype=np.int64)
            n_body_cell = 1
        cached = {
            "support_body": support_body, "target_body": target_body,
            "n_body": n_body, "target_body_missing": target_body_missing,
            "support_cell": support_cell, "target_cell": target_cell,
            "n_cell": n_cell,
            "support_body_cell": support_body_cell,
            "target_body_cell": target_body_cell,
            "n_body_cell": n_body_cell,
        }
        self._group_code_cache[key] = cached
        return cached

    def certificate_numba(self, predicate):
        """Protocol-equivalent certificate using fused native context counting."""
        direct = predicate.eval_table(self.target)
        pass_mask = direct == TRUE
        fail_mask = direct == FALSE
        unresolved = direct == UNKNOWN
        n_target = len(self.target)
        p_phi = np.where(pass_mask, 1.0, 0.0)
        support_n = np.zeros(n_target, dtype=np.int64)
        support_n_plus = np.zeros(n_target, dtype=np.int64)
        local_used = np.zeros(n_target, dtype=bool)
        body_used = np.zeros(n_target, dtype=bool)
        context_name = np.full(n_target, "direct", dtype=object)
        abstain_mask = np.zeros(n_target, dtype=bool)

        patterns = {}
        unresolved_rows = np.flatnonzero(unresolved)
        for row in unresolved_rows:
            residual = tuple(
                atom_index for atom_index, atom in enumerate(predicate.atoms)
                if self._target_missing[atom.attr][row])
            patterns.setdefault(residual, []).append(int(row))

        mode = {"joint": 0, "product": 1, "min": 2}[self.mode]
        support_cells = (np.zeros(len(self.support), dtype=np.int64)
                         if self.support_cells is None else
                         np.ascontiguousarray(self.support_cells, dtype=np.int64))
        for pattern, rows_list in patterns.items():
            atoms = [predicate.atoms[index] for index in pattern]
            complete = np.ones(len(self.support), dtype=bool)
            atom_satisfied = np.empty(
                (len(atoms), len(self.support)), dtype=bool)
            for atom_index, atom in enumerate(atoms):
                missing = self._support_missing[atom.attr]
                complete &= ~missing
                atom_satisfied[atom_index] = (
                    Predicate([atom]).eval_table(self.support) == TRUE)
            joint_satisfied = atom_satisfied.all(axis=0)

            residual_attrs = {atom.attr for atom in atoms}
            body_attrs = []
            for target_attr in residual_attrs:
                for attr in self.body_by_target.get(target_attr, ()):
                    if attr not in residual_attrs and attr not in body_attrs:
                        body_attrs.append(attr)
            if body_attrs:
                body_support = np.stack(
                    [self._body_codes(attr)[0] for attr in body_attrs])
                body_target = np.stack(
                    [self._body_codes(attr)[1][rows_list] for attr in body_attrs])
            else:
                body_support = np.empty(
                    (0, len(self.support)), dtype=np.int64)
                body_target = np.empty((0, len(rows_list)), dtype=np.int64)

            grouped = self._context_group_codes(body_attrs)
            rows_array = np.asarray(rows_list, dtype=np.int64)
            # If a target body is missing, reference semantics use the visible
            # subset of body attributes for that row. That rare block-missing
            # case falls back to the scan kernel; the main tuple-MCAR workload
            # keeps evidence/body attributes observed and takes the grouped path.
            grouped_safe = not (
                body_attrs and grouped["target_body_missing"][rows_array].any())
            if grouped_safe:
                values, counts, pluses, contexts = joint_grouped_context_kernel(
                    np.ascontiguousarray(complete),
                    np.ascontiguousarray(joint_satisfied),
                    np.ascontiguousarray(atom_satisfied),
                    grouped["support_body"], grouped["target_body"][rows_array],
                    grouped["n_body"],
                    grouped["support_cell"], grouped["target_cell"][rows_array],
                    grouped["n_cell"],
                    grouped["support_body_cell"],
                    grouped["target_body_cell"][rows_array],
                    grouped["n_body_cell"], bool(body_attrs),
                    self.support_cells is not None, mode,
                    self.min_support, self.kappa)
            else:
                global_p, global_n, global_nplus = self._global_probability(
                    complete, joint_satisfied, atom_satisfied)
                target_cells = (np.zeros(len(rows_list), dtype=np.int64)
                                if self.target_cells is None else
                                np.ascontiguousarray(
                                    self.target_cells[rows_list], dtype=np.int64))
                values, counts, pluses, contexts = joint_context_kernel(
                    np.ascontiguousarray(complete),
                    np.ascontiguousarray(joint_satisfied),
                    np.ascontiguousarray(atom_satisfied),
                    np.ascontiguousarray(body_support),
                    np.ascontiguousarray(body_target),
                    support_cells, target_cells, self.support_cells is not None,
                    mode, self.min_support, self.kappa,
                    global_p, global_n, global_nplus)
            rows = np.asarray(rows_list, dtype=np.int64)
            found = contexts != ABSTAIN
            found_rows = rows[found]
            p_phi[found_rows] = np.clip(values[found], 0.0, 1.0)
            support_n[found_rows] = counts[found]
            support_n_plus[found_rows] = pluses[found]
            abstain_mask[rows[~found]] = True
            context_name[rows[~found]] = "abstain"
            names = np.asarray(["global", "cell", "body", "body_cell"],
                               dtype=object)
            context_name[found_rows] = names[contexts[found]]
            local_used[found_rows] = np.isin(
                contexts[found], np.asarray([CELL, BODY_CELL], dtype=np.int8))
            body_used[found_rows] = np.isin(
                contexts[found], np.asarray([BODY, BODY_CELL], dtype=np.int8))

        uncertain_mask = unresolved & ~abstain_mask
        states = np.full(n_target, FALSE, dtype=object)
        states[uncertain_mask] = UNKNOWN
        states[pass_mask] = TRUE
        return {
            "p_phi": p_phi,
            "pass_mask": pass_mask,
            "fail_mask": fail_mask,
            "uncertain_mask": uncertain_mask,
            "abstain_mask": abstain_mask,
            "support_n": support_n,
            "support_n_plus": support_n_plus,
            "local_used": local_used,
            "body_used": body_used,
            "context_name": context_name,
            "states": states,
        }

    def certificate_batch_numba(self, predicates):
        """Return fused certificates for a query batch."""
        return [self.certificate_numba(predicate) for predicate in predicates]

    @staticmethod
    def eligible(cert, theta):
        return cert["pass_mask"] | (cert["uncertain_mask"] & (cert["p_phi"] >= theta))
