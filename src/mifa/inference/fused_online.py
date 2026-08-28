"""Online grouping and caching for protocol-v2 joint eligibility.

The formal workload pairs many vector queries with the same scalar filter.  A
certificate depends on the observed relation, frozen policy, threshold, and
predicate, but not on the vector query.  Recomputing it for every vector query
therefore repeats identical pandas preparation and identical Numba context
counting.  This module groups those requests by a canonical predicate key and
caches the final eligibility bitmap.

The optimization is semantic preserving: enabled signatures use the production
``certificate_numba`` path, while validation-disabled signatures reduce exactly
to the observed-direct pass mask used by the policy gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from mifa.inference.joint_support_numba import (
    ABSTAIN, joint_context_kernel as _joint_context_kernel,
    joint_grouped_context_kernel as _joint_grouped_context_kernel,
)
from mifa.predicates import Predicate
from mifa.predicates import TRUE


# The production kernels predate concurrent grouped-filter realization and
# retain the Python GIL on entry.  Recompiling the identical Python functions
# with ``nogil=True`` lets independent predicate groups execute concurrently
# without changing their arithmetic or their result ordering.  Keeping these
# aliases in the online-only module also leaves the frozen effectiveness source
# manifest untouched.
joint_context_kernel = njit(
    cache=True, fastmath=False, nogil=True)(_joint_context_kernel.py_func)
joint_grouped_context_kernel = njit(
    cache=True, fastmath=False, nogil=True)(_joint_grouped_context_kernel.py_func)


@njit(cache=True, fastmath=False, nogil=True)
def _paper_direct_states(values, missing, atom_attrs, atom_ops,
                         atom_lower, atom_upper):
    """Vector-free full-row pass of the paper executor's direct semantics."""
    output = np.empty(values.shape[1], dtype=np.int8)
    for tuple_id in range(values.shape[1]):
        unresolved = False
        failed = False
        for atom_index in range(len(atom_attrs)):
            attr = atom_attrs[atom_index]
            if missing[attr, tuple_id]:
                unresolved = True
                continue
            value = values[attr, tuple_id]
            op = atom_ops[atom_index]
            if op == 0:
                satisfied = value <= atom_lower[atom_index]
            elif op == 1:
                satisfied = value >= atom_lower[atom_index]
            elif op == 2:
                satisfied = (value >= atom_lower[atom_index] and
                             value <= atom_upper[atom_index])
            else:
                satisfied = value == atom_lower[atom_index]
            if not satisfied:
                failed = True
                break
        output[tuple_id] = 0 if failed else (2 if unresolved else 1)
    return output


def _freeze(value):
    """Convert atom payloads to a stable, hashable representation."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item))
                            for key, item in value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_freeze(item) for item in value)
    return value


def predicate_key(predicate):
    """Canonical key preserving atom order and all predicate constants."""
    return tuple((str(atom.attr), str(atom.op), _freeze(atom.a), _freeze(atom.b))
                 for atom in predicate.atoms)


def predicate_signature(predicate):
    """Match ``experiment_v2``'s validation-frozen signature gate."""
    return "&".join(sorted(str(atom.attr) for atom in predicate.atoms))


@dataclass
class EligibilityCacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    enabled_misses: int = 0
    direct_only_misses: int = 0

    def as_dict(self):
        return {
            "requests": int(self.requests),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "enabled_misses": int(self.enabled_misses),
            "direct_only_misses": int(self.direct_only_misses),
        }


class FastJointEligibility:
    """Typed-array, eligibility-only realization of ``certificate_numba``.

    The formal engine returns rich diagnostics for policy selection.  Online
    search needs only the final bitmap.  This path retains the exact Numba
    context kernels and backoff equations, but caches per-atom satisfaction
    vectors, vectorizes missing-pattern construction, and omits object-valued
    states and unused diagnostic arrays.
    """

    def __init__(self, engine, policy, theta):
        if engine.mode != "joint":
            raise ValueError("FastJointEligibility currently implements joint mode only")
        self.engine = engine
        self.policy = dict(policy)
        self.theta = float(theta)
        enabled = self.policy.get("enabled_signatures")
        self.enabled_signatures = None if enabled is None else set(enabled)
        self._support_sat_cache = {}
        self._target_sat_cache = {}
        self._support_cells = (
            np.zeros(len(engine.support), dtype=np.int64)
            if engine.support_cells is None else
            np.ascontiguousarray(engine.support_cells, dtype=np.int64))

    def _satisfaction(self, atom, *, target):
        cache = self._target_sat_cache if target else self._support_sat_cache
        key = predicate_key(Predicate([atom]))
        cached = cache.get(key)
        if cached is not None:
            return cached
        table = self.engine.target if target else self.engine.support
        missing_map = (self.engine._target_missing if target
                       else self.engine._support_missing)
        missing = missing_map[atom.attr]
        values = table[atom.attr].to_numpy()
        result = np.ascontiguousarray(
            Predicate._atom_sat(atom, values, missing), dtype=np.bool_)
        result.setflags(write=False)
        cache[key] = result
        return result

    def _direct_masks(self, predicate):
        n_target = len(self.engine.target)
        any_false = np.zeros(n_target, dtype=np.bool_)
        pattern = np.zeros(n_target, dtype=np.uint8)
        for atom_index, atom in enumerate(predicate.atoms):
            missing = self.engine._target_missing[atom.attr]
            satisfied = self._satisfaction(atom, target=True)
            any_false |= (~missing) & (~satisfied)
            pattern |= missing.astype(np.uint8) << atom_index
        pass_mask = (~any_false) & (pattern == 0)
        unresolved = (~any_false) & (pattern != 0)
        return pass_mask, unresolved, pattern

    def _body_attrs(self, atoms):
        residual_attrs = {atom.attr for atom in atoms}
        body_attrs = []
        for target_attr in residual_attrs:
            for attr in self.engine.body_by_target.get(target_attr, ()):
                if attr not in residual_attrs and attr not in body_attrs:
                    body_attrs.append(attr)
        return body_attrs

    def prepare(self, predicates):
        """Populate shared read-only atom/context caches before parallel misses."""
        for predicate in predicates:
            for atom in predicate.atoms:
                self._satisfaction(atom, target=True)
            if (self.enabled_signatures is not None and
                    predicate_signature(predicate) not in self.enabled_signatures):
                # A validation-disabled signature reduces to direct truth.  Do
                # not build support-side satisfaction or context caches that
                # the deployed policy can never consult.
                continue
            for atom in predicate.atoms:
                self._satisfaction(atom, target=False)
            arity = len(predicate.atoms)
            for pattern in range(1, 1 << arity):
                atoms = [atom for atom_index, atom in enumerate(predicate.atoms)
                         if pattern & (1 << atom_index)]
                body_attrs = self._body_attrs(atoms)
                self.engine._context_group_codes(body_attrs)

    def warm_kernels(self, predicates):
        """Compile the online Numba dispatchers without retaining query caches."""
        for predicate in predicates:
            if (self.enabled_signatures is not None and
                    predicate_signature(predicate) not in self.enabled_signatures):
                continue
            self.eligible_one(predicate)
            if (joint_grouped_context_kernel.signatures or
                    joint_context_kernel.signatures):
                break
        # Satisfaction vectors are part of cold predicate realization, not a
        # one-time code-generation cost, so discard them after JIT warm-up.
        self._support_sat_cache.clear()
        self._target_sat_cache.clear()

    def _pattern_values(self, atoms, rows):
        complete = np.ones(len(self.engine.support), dtype=np.bool_)
        atom_satisfied = np.empty(
            (len(atoms), len(self.engine.support)), dtype=np.bool_)
        for atom_index, atom in enumerate(atoms):
            complete &= ~self.engine._support_missing[atom.attr]
            atom_satisfied[atom_index] = self._satisfaction(atom, target=False)
        joint_satisfied = atom_satisfied.all(axis=0)
        body_attrs = self._body_attrs(atoms)
        grouped = self.engine._context_group_codes(body_attrs)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        grouped_safe = not (
            body_attrs and grouped["target_body_missing"][rows].any())
        if grouped_safe:
            return joint_grouped_context_kernel(
                np.ascontiguousarray(complete),
                np.ascontiguousarray(joint_satisfied),
                np.ascontiguousarray(atom_satisfied),
                grouped["support_body"], grouped["target_body"][rows],
                grouped["n_body"],
                grouped["support_cell"], grouped["target_cell"][rows],
                grouped["n_cell"],
                grouped["support_body_cell"],
                grouped["target_body_cell"][rows],
                grouped["n_body_cell"], bool(body_attrs),
                self.engine.support_cells is not None, 0,
                self.engine.min_support, self.engine.kappa)

        if body_attrs:
            body_support = np.stack(
                [self.engine._body_codes(attr)[0] for attr in body_attrs])
            body_target = np.stack(
                [self.engine._body_codes(attr)[1][rows] for attr in body_attrs])
        else:
            body_support = np.empty(
                (0, len(self.engine.support)), dtype=np.int64)
            body_target = np.empty((0, len(rows)), dtype=np.int64)
        global_n = int(complete.sum())
        global_nplus = int((joint_satisfied & complete).sum())
        global_p = (global_nplus / global_n) if global_n else np.nan
        target_cells = (
            np.zeros(len(rows), dtype=np.int64)
            if self.engine.target_cells is None else
            np.ascontiguousarray(self.engine.target_cells[rows], dtype=np.int64))
        return joint_context_kernel(
            np.ascontiguousarray(complete),
            np.ascontiguousarray(joint_satisfied),
            np.ascontiguousarray(atom_satisfied),
            np.ascontiguousarray(body_support),
            np.ascontiguousarray(body_target),
            self._support_cells, target_cells,
            self.engine.support_cells is not None, 0,
            self.engine.min_support, self.engine.kappa,
            global_p, global_n, global_nplus)

    def eligible_one(self, predicate):
        pass_mask, unresolved, patterns = self._direct_masks(predicate)
        eligible = np.ascontiguousarray(pass_mask.copy(), dtype=np.bool_)
        if (self.enabled_signatures is not None and
                predicate_signature(predicate) not in self.enabled_signatures):
            return eligible
        for pattern in np.unique(patterns[unresolved]):
            rows = np.flatnonzero(unresolved & (patterns == pattern))
            atoms = [atom for atom_index, atom in enumerate(predicate.atoms)
                     if int(pattern) & (1 << atom_index)]
            values, _, _, contexts = self._pattern_values(atoms, rows)
            admitted = (contexts != ABSTAIN) & (values >= self.theta)
            eligible[rows[admitted]] = True
        return eligible


class GroupedEligibilityCache:
    """Cache final eligibility bitmaps for a frozen joint-support policy."""

    def __init__(self, engine, policy, theta, *, fast_evaluator=False):
        self.engine = engine
        self.policy = dict(policy)
        self.theta = float(theta)
        enabled = self.policy.get("enabled_signatures")
        self.enabled_signatures = None if enabled is None else set(enabled)
        self._cache = {}
        self.stats = EligibilityCacheStats()
        self.evaluator = (FastJointEligibility(engine, policy, theta)
                          if fast_evaluator else None)

    def clear(self):
        self._cache.clear()
        self.stats = EligibilityCacheStats()

    def _compute(self, predicate):
        signature = predicate_signature(predicate)
        if self.evaluator is not None:
            if (self.enabled_signatures is not None and
                    signature not in self.enabled_signatures):
                self.stats.direct_only_misses += 1
            else:
                self.stats.enabled_misses += 1
            return np.ascontiguousarray(
                self.evaluator.eligible_one(predicate), dtype=np.bool_)
        if (self.enabled_signatures is not None and
                signature not in self.enabled_signatures):
            # The formal gate converts every statistical/abstained row to
            # policy-drop, leaving exactly the rows directly true on D.
            self.stats.direct_only_misses += 1
            return np.ascontiguousarray(
                predicate.eval_table(self.engine.target) == TRUE,
                dtype=np.bool_)
        self.stats.enabled_misses += 1
        certificate = self.engine.certificate_numba(predicate)
        return np.ascontiguousarray(
            self.engine.eligible(certificate, self.theta), dtype=np.bool_)

    def _record_miss_kind(self, predicate):
        signature = predicate_signature(predicate)
        if (self.enabled_signatures is not None and
                signature not in self.enabled_signatures):
            self.stats.direct_only_misses += 1
        else:
            self.stats.enabled_misses += 1

    def _compute_without_stats(self, predicate):
        if self.evaluator is not None:
            return np.ascontiguousarray(
                self.evaluator.eligible_one(predicate), dtype=np.bool_)
        signature = predicate_signature(predicate)
        if (self.enabled_signatures is not None and
                signature not in self.enabled_signatures):
            return np.ascontiguousarray(
                predicate.eval_table(self.engine.target) == TRUE,
                dtype=np.bool_)
        certificate = self.engine.certificate_numba(predicate)
        return np.ascontiguousarray(
            self.engine.eligible(certificate, self.theta), dtype=np.bool_)

    def eligible_one(self, predicate):
        self.stats.requests += 1
        key = predicate_key(predicate)
        cached = self._cache.get(key)
        if cached is not None:
            self.stats.hits += 1
            return cached
        self.stats.misses += 1
        eligible = self._compute(predicate)
        eligible.setflags(write=False)
        self._cache[key] = eligible
        return eligible

    def eligible_grouped(self, predicates, *, workers=1):
        """Return ``(unique bitmap bank, query group IDs)`` without row repeats."""
        predicates = list(predicates)
        if not predicates:
            return (np.empty((0, len(self.engine.target)), dtype=np.bool_),
                    np.empty(0, dtype=np.int64))
        workers = int(workers)
        keys = [predicate_key(predicate) for predicate in predicates]
        unique = {}
        group_ids = np.empty(len(keys), dtype=np.int64)
        for index, (key, predicate) in enumerate(zip(keys, predicates)):
            group = unique.get(key)
            if group is None:
                group = len(unique)
                unique[key] = (group, predicate)
            group_ids[index] = group[0] if isinstance(group, tuple) else group

        if workers <= 1:
            for _, predicate in unique.values():
                self.eligible_one(predicate)
            duplicates = len(predicates) - len(unique)
            self.stats.requests += duplicates
            self.stats.hits += duplicates
            bank = np.ascontiguousarray(
                [self._cache[key] for key in unique], dtype=np.bool_)
            return bank, group_ids

        missing = {}
        for key, (_, predicate) in unique.items():
            if key not in self._cache:
                missing[key] = predicate
        self.stats.requests += len(predicates)
        self.stats.misses += len(missing)
        self.stats.hits += len(predicates) - len(missing)
        for predicate in missing.values():
            self._record_miss_kind(predicate)
        if self.evaluator is not None:
            self.evaluator.prepare(missing.values())
        items = list(missing.items())
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(
                self._compute_without_stats, (predicate for _, predicate in items)))
        for (key, _), eligible in zip(items, values):
            eligible.setflags(write=False)
            self._cache[key] = eligible
        bank = np.ascontiguousarray(
            [self._cache[key] for key in unique], dtype=np.bool_)
        return bank, group_ids

    def eligible_batch(self, predicates, *, workers=1):
        """Return a contiguous ``(queries, rows)`` eligibility matrix."""
        bank, group_ids = self.eligible_grouped(predicates, workers=workers)
        return np.ascontiguousarray(bank[group_ids], dtype=np.bool_)

    def cached_group_ids(self, predicates):
        """Map requests to the installed bitmap bank without copying bitmaps."""
        predicates = list(predicates)
        index = {key: group for group, key in enumerate(self._cache)}
        group_ids = np.empty(len(predicates), dtype=np.int64)
        for row, predicate in enumerate(predicates):
            key = predicate_key(predicate)
            if key not in index:
                raise KeyError("predicate has not been realized in this cache")
            group_ids[row] = index[key]
        self.stats.requests += len(predicates)
        self.stats.hits += len(predicates)
        return group_ids

    @property
    def unique_predicates(self):
        return len(self._cache)


class PaperGroupedEligibilityCache:
    """Deduplicate fixed-policy eligibility for :class:`IndexedJointInference`.

    Paper MIFA's eligibility decision depends on the frozen searchable
    snapshot, policy, threshold, and scalar predicate, but not on the vector
    query.  Production workloads deliberately pair many query vectors with the
    same filter.  This cache materializes each distinct filter exactly once and
    returns a compact bitmap bank plus per-query bank IDs.  It calls the paper
    inference object directly, so hierarchy, ordered templates, score maps,
    and backoff semantics are unchanged.
    """

    def __init__(self, inference):
        self.inference = inference
        self._cache = {}
        self.stats = EligibilityCacheStats()

    def clear(self):
        self._cache.clear()
        self.stats = EligibilityCacheStats()

    @property
    def unique_predicates(self):
        return len(self._cache)

    def eligible_one(self, predicate):
        self.stats.requests += 1
        key = predicate_key(predicate)
        cached = self._cache.get(key)
        if cached is not None:
            self.stats.hits += 1
            return cached
        self.stats.misses += 1
        self.stats.enabled_misses += 1
        # Support-satisfaction and leaf-prefix caches are query local in the
        # paper executor.  Preserve that lifetime when realizing one canonical
        # predicate for a grouped request.
        self.inference.begin_query()
        plan = (self.inference.compile_hard_predicate(predicate)
                if hasattr(self.inference, "compile_hard_predicate") else None)
        if plan is None:
            eligible = np.ascontiguousarray(
                self.inference.materialize_eligibility(predicate),
                dtype=np.bool_)
        else:
            states = _paper_direct_states(
                plan["values"], plan["missing"], plan["attrs"], plan["ops"],
                plan["lower"], plan["upper"])
            eligible = np.ascontiguousarray(states == 1, dtype=np.bool_)
            for tuple_id in np.flatnonzero(states == 2):
                eligible[int(tuple_id)] = self.inference.evaluate(
                    int(tuple_id), predicate, record=False,
                    assume_unresolved=True).eligible
        eligible.setflags(write=False)
        self._cache[key] = eligible
        return eligible

    @staticmethod
    def group_ids(predicates):
        """Return stable local IDs without touching eligibility state."""
        groups = {}
        output = np.empty(len(predicates), dtype=np.int64)
        for index, predicate in enumerate(predicates):
            key = predicate_key(predicate)
            group = groups.get(key)
            if group is None:
                group = len(groups)
                groups[key] = group
            output[index] = group
        return output

    def eligible_grouped(self, predicates):
        predicates = list(predicates)
        if not predicates:
            width = len(self.inference.target_table)
            return (np.empty((0, width), dtype=np.bool_),
                    np.empty(0, dtype=np.int64))
        unique = {}
        group_ids = np.empty(len(predicates), dtype=np.int64)
        for index, predicate in enumerate(predicates):
            key = predicate_key(predicate)
            group = unique.get(key)
            if group is None:
                group = len(unique)
                unique[key] = (group, predicate)
            group_ids[index] = group[0] if isinstance(group, tuple) else group
        for _, predicate in unique.values():
            self.eligible_one(predicate)
        duplicates = len(predicates) - len(unique)
        self.stats.requests += duplicates
        self.stats.hits += duplicates
        bank = np.ascontiguousarray(
            [self._cache[key] for key in unique], dtype=np.bool_)
        return bank, group_ids

    def eligible_batch(self, predicates):
        """Return a contiguous ``(queries, rows)`` eligibility matrix."""
        bank, group_ids = self.eligible_grouped(predicates)
        return np.ascontiguousarray(bank[group_ids], dtype=np.bool_)

    def cached_group_ids(self, predicates):
        """Map requests to the installed bitmap bank without copying bitmaps."""
        predicates = list(predicates)
        index = {key: group for group, key in enumerate(self._cache)}
        group_ids = np.empty(len(predicates), dtype=np.int64)
        for row, predicate in enumerate(predicates):
            key = predicate_key(predicate)
            if key not in index:
                raise KeyError("predicate has not been realized in this cache")
            group_ids[row] = index[key]
        self.stats.requests += len(predicates)
        self.stats.hits += len(predicates)
        return group_ids
