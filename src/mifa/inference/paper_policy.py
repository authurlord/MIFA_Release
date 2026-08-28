"""Paper-aligned templates, monotone score maps, and indexed joint inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from mifa.predicates import FALSE, TRUE, Atom, Predicate


@dataclass(frozen=True)
class CertifiedDomain:
    """A finite or closed numeric domain supplied by an external constraint."""

    values: tuple | None = None
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        finite = self.values is not None
        interval = self.lower is not None or self.upper is not None
        if finite == interval:
            raise ValueError("use exactly one of finite values or a numeric interval")
        if finite:
            object.__setattr__(self, "values", tuple(dict.fromkeys(self.values)))
        elif self.lower is None or self.upper is None or self.lower > self.upper:
            raise ValueError("a numeric interval needs ordered finite endpoints")

    @classmethod
    def finite(cls, values):
        return cls(values=tuple(values))

    @classmethod
    def interval(cls, lower, upper):
        return cls(lower=float(lower), upper=float(upper))

    def intersect(self, other):
        if self.values is not None and other.values is not None:
            allowed = set(other.values)
            return CertifiedDomain.finite(value for value in self.values
                                          if value in allowed)
        if self.values is not None:
            return CertifiedDomain.finite(
                value for value in self.values
                if other.lower <= float(value) <= other.upper)
        if other.values is not None:
            return other.intersect(self)
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)
        if lower > upper:
            return CertifiedDomain.finite(())
        return CertifiedDomain.interval(lower, upper)

    def classify(self, atom):
        """Return TRUE/FALSE only when every/no possible value satisfies atom."""
        if self.values is not None:
            outcomes = [atom.eval_value(value) == TRUE for value in self.values]
            if outcomes and all(outcomes):
                return TRUE
            if not any(outcomes):
                return FALSE
            return "unknown"
        lower, upper = self.lower, self.upper
        if atom.op == "le":
            return TRUE if upper <= atom.a else FALSE if lower > atom.a else "unknown"
        if atom.op == "ge":
            return TRUE if lower >= atom.a else FALSE if upper < atom.a else "unknown"
        if atom.op == "between":
            if lower >= atom.a and upper <= atom.b:
                return TRUE
            return FALSE if upper < atom.a or lower > atom.b else "unknown"
        if atom.op == "eq":
            if lower == upper == atom.a:
                return TRUE
            return FALSE if atom.a < lower or atom.a > upper else "unknown"
        if atom.op == "in":
            possible = [value for value in atom.a if lower <= value <= upper]
            if lower == upper and lower in atom.a:
                return TRUE
            return FALSE if not possible else "unknown"
        raise ValueError(atom.op)


@dataclass(frozen=True)
class MatchLevel:
    """One scalar-body matching level; later levels must be no narrower."""

    kind: str
    radius: float | None = None

    @classmethod
    def equality(cls):
        return cls("eq")

    @classmethod
    def numeric_radius(cls, radius):
        if radius < 0:
            raise ValueError("numeric radius must be non-negative")
        return cls("radius", float(radius))

    @classmethod
    def unrestricted(cls):
        return cls("all")


@dataclass(frozen=True)
class BodyAttribute:
    attr: str
    levels: tuple[MatchLevel, ...]

    def __post_init__(self):
        if not self.levels:
            raise ValueError("a body attribute needs at least one matching level")
        if self.levels[-1].kind != "all":
            raise ValueError("the final body matching level must be unrestricted")
        radii = [level.radius for level in self.levels if level.kind == "radius"]
        if any(right < left for left, right in zip(radii, radii[1:])):
            raise ValueError("numeric radius levels must be nondecreasing")
        if any(level.kind not in {"eq", "radius", "all"} for level in self.levels):
            raise ValueError("unknown body matching level")


@dataclass(frozen=True)
class SupportTemplate:
    template_id: str
    targets: tuple[str, ...]
    body: tuple[BodyAttribute, ...] = ()
    location_aware: bool = True
    min_support: int = 30
    kappa: float = 20.0
    priority: int = 0
    scalar_relax_order: tuple[str, ...] = ()

    def __post_init__(self):
        targets = tuple(sorted(set(self.targets)))
        object.__setattr__(self, "targets", targets)
        if not targets:
            raise ValueError("template target set cannot be empty")
        if self.min_support <= 0 or self.kappa < 0:
            raise ValueError("invalid support/shrinkage parameter")
        body_names = [item.attr for item in self.body]
        if len(set(body_names)) != len(body_names):
            raise ValueError("template body attributes must be unique")
        if set(body_names) & set(targets):
            raise ValueError("template body and target must be disjoint")
        order = self.scalar_relax_order or tuple(body_names)
        if set(order) != set(body_names) or len(order) != len(body_names):
            raise ValueError("scalar_relax_order must list every body attribute once")
        object.__setattr__(self, "scalar_relax_order", tuple(order))


class MonotoneScoreMap:
    """Deterministic least-squares isotonic map using weighted PAVA."""

    def __init__(self, thresholds=None, values=None):
        self.thresholds = (np.asarray([], dtype=np.float64) if thresholds is None
                           else np.asarray(thresholds, dtype=np.float64))
        self.values = (np.asarray([], dtype=np.float64) if values is None
                       else np.asarray(values, dtype=np.float64))

    @classmethod
    def identity(cls):
        return cls([0.0, 1.0], [0.0, 1.0])

    @classmethod
    def fit(cls, raw_scores, labels):
        x = np.asarray(raw_scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) == 0:
            raise ValueError("cannot fit an isotonic map without resolved instances")
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        unique, first, counts = np.unique(x, return_index=True, return_counts=True)
        sums = np.add.reduceat(y, first)
        block_values = list((sums / counts).astype(float))
        block_weights = list(counts.astype(float))
        block_starts = list(range(len(unique)))
        block_ends = list(range(len(unique)))
        index = 0
        while index < len(block_values) - 1:
            if block_values[index] <= block_values[index + 1] + 1e-15:
                index += 1
                continue
            total_weight = block_weights[index] + block_weights[index + 1]
            pooled = ((block_values[index] * block_weights[index] +
                       block_values[index + 1] * block_weights[index + 1]) /
                      total_weight)
            block_values[index] = pooled
            block_weights[index] = total_weight
            block_ends[index] = block_ends[index + 1]
            del block_values[index + 1]
            del block_weights[index + 1]
            del block_starts[index + 1]
            del block_ends[index + 1]
            if index:
                index -= 1
        fitted = np.empty(len(unique), dtype=np.float64)
        for value, start, end in zip(block_values, block_starts, block_ends):
            fitted[start:end + 1] = value
        return cls(unique, np.clip(fitted, 0.0, 1.0))

    def predict(self, scores):
        values = np.asarray(scores, dtype=np.float64)
        if not len(self.thresholds):
            raise RuntimeError("score map has not been fitted")
        result = np.interp(
            values, self.thresholds, self.values,
            left=float(self.values[0]), right=float(self.values[-1]))
        return float(result) if values.ndim == 0 else result

    def manifest(self):
        return {"kind": "least_squares_isotonic_pava",
                "knots": int(len(self.thresholds))}


@dataclass
class DeploymentPolicy:
    templates: tuple[SupportTemplate, ...]
    score_maps: dict[str, MonotoneScoreMap] = field(default_factory=dict)
    fit_manifest: dict = field(default_factory=dict)

    def __post_init__(self):
        self.templates = tuple(sorted(
            self.templates, key=lambda item: (item.priority, item.template_id)))
        identifiers = [item.template_id for item in self.templates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("template IDs must be unique")
        for template in self.templates:
            self.score_maps.setdefault(template.template_id, MonotoneScoreMap.identity())

    def select(self, residual_targets, row):
        target = tuple(sorted(set(residual_targets)))
        fallback = None
        for template in self.templates:
            if template.targets != target:
                continue
            if not template.body and not template.location_aware:
                if fallback is None:
                    fallback = template
                continue
            if all(not _missing(row[item.attr]) for item in template.body):
                return template
        return fallback

    def select_indexed(self, residual_targets, target_columns, tuple_id):
        """Array-backed equivalent of :meth:`select` for online serving."""
        target = tuple(sorted(set(residual_targets)))
        fallback = None
        for template in self.templates:
            if template.targets != target:
                continue
            if not template.body and not template.location_aware:
                if fallback is None:
                    fallback = template
                continue
            if all(not _missing(target_columns[item.attr][int(tuple_id)])
                   for item in template.body):
                return template
        return fallback


@dataclass
class InferenceResult:
    state: str
    eligible: bool
    score: float | None = None
    raw_score: float | None = None
    template_id: str | None = None
    context_node: int | None = None
    context_depth: int | None = None
    support_n: int = 0
    support_n_plus: int = 0
    backoff_steps: int = 0
    ancestor_backoff: bool = False
    scalar_backoff: bool = False


def _missing(value):
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if np.isscalar(missing) else bool(np.asarray(missing).all())
    except (TypeError, ValueError):
        return False


def _freeze_key(value):
    """Convert predicate payloads to a stable cache key."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_key(item))
                            for key, item in value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_key(item) for item in value), key=repr))
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_freeze_key(item) for item in value)
    return value


class IndexedJointInference:
    """Apply one fixed deployment policy on demand to a searchable snapshot."""

    def __init__(self, target_table, target_embeddings, tree, support_index,
                 policy, theta, deterministic_constraints=(), *,
                 support_mode="joint", template_selection_mode="indexed",
                 support_retrieval_mode="indexed",
                 context_mode="hierarchical"):
        self.target_table = target_table.reset_index(drop=True)
        self.target_embeddings = np.ascontiguousarray(
            target_embeddings, dtype=np.float32)
        if len(self.target_table) != len(self.target_embeddings):
            raise ValueError("target table and embeddings must align")
        self.tree = tree
        self.support_index = support_index
        self.policy = policy
        self.theta = float(theta)
        if support_mode not in {"joint", "product", "min"}:
            raise ValueError(f"unknown support mode: {support_mode}")
        if template_selection_mode not in {"indexed", "row_scan"}:
            raise ValueError(
                f"unknown template selection mode: {template_selection_mode}")
        if support_retrieval_mode not in {"indexed", "scan"}:
            raise ValueError(
                f"unknown support retrieval mode: {support_retrieval_mode}")
        if context_mode not in {"hierarchical", "leaf_only"}:
            raise ValueError(f"unknown context mode: {context_mode}")
        self.support_mode = support_mode
        self.template_selection_mode = template_selection_mode
        self.support_retrieval_mode = support_retrieval_mode
        self.context_mode = context_mode
        self.deterministic_constraints = tuple(deterministic_constraints)
        self._target_columns = {
            attr: self.target_table[attr].to_numpy()
            for attr in self.target_table.columns
        }
        # Query-time direct checks are on the critical path and otherwise pay
        # pandas/object dispatch for every distance-competitive tuple.  Keep a
        # compact numeric mirror of the searchable scalar columns so the
        # executor can fuse direct predicate evaluation with leaf distance and
        # top-k maintenance.  Categorical values are dictionary encoded; this
        # representation is serving-only and does not alter support inference.
        self._fast_attr_names = tuple(self.target_table.columns)
        self._fast_attr_index = {
            attr: index for index, attr in enumerate(self._fast_attr_names)
        }
        self._fast_values = np.zeros(
            (len(self._fast_attr_names), len(self.target_table)),
            dtype=np.float64)
        self._fast_missing = np.zeros_like(self._fast_values, dtype=np.bool_)
        self._fast_category_codes = {}
        for attr_index, attr in enumerate(self._fast_attr_names):
            column = self.target_table[attr]
            missing = column.isna().to_numpy(dtype=np.bool_)
            self._fast_missing[attr_index] = missing
            if self.support_index.dtypes.get(attr) == "num":
                values = column.to_numpy(dtype=np.float64, na_value=np.nan)
                self._fast_values[attr_index] = np.nan_to_num(
                    values, nan=0.0)
            else:
                codes, uniques = pd.factorize(
                    column, sort=False, use_na_sentinel=True)
                self._fast_values[attr_index] = codes.astype(np.float64)
                self._fast_category_codes[attr] = {
                    _freeze_key(value): int(code)
                    for code, value in enumerate(uniques)
                }
        self._fast_attrs_with_missing = frozenset(
            attr for attr, attr_index in self._fast_attr_index.items()
            if np.any(self._fast_missing[attr_index]))
        _, target_paths = self.tree.route(
            self.target_embeddings, return_paths=True)
        self._target_ancestor_paths = tuple(
            tuple(reversed(path)) for path in target_paths)
        self._support_satisfaction_cache = {}
        self._support_satisfaction_cache_max_entries = 1024
        self._support_satisfaction_packed_cache = {}
        self._support_leaf_prefix_cache = {}
        self._support_leaf_prefix_cache_max_entries = 8192
        # The deployment snapshot, policy, threshold, support evidence, and
        # searchable scalar values are immutable while serving.  Statistical
        # eligibility therefore depends only on the residual predicate,
        # selected template, target leaf, and scalar body values.  Cache that
        # exact signature across queries so repeated workload predicates do not
        # rebuild identical ancestor/body support counts.
        self._serving_result_cache = {}
        self._serving_result_cache_max_entries = 131072
        # Candidate pages use the same exact equivalence relation already
        # materialized below for reference-batch propagation: residual joint
        # predicate + selected template + location/body group.  Cache only
        # groups actually encountered in compact pages; this remains O(QM),
        # never predicate-by-corpus O(QN).
        self._candidate_group_result_cache = {}
        self._candidate_group_result_cache_entries = 0
        self._candidate_group_result_cache_max_entries = 131072
        self._candidate_predicate_plan_cache = {}
        self._candidate_predicate_plan_cache_max_entries = 256
        self._hard_predicate_plan_cache = {}
        self._candidate_group_prefetch_layouts = {}
        self._candidate_group_prefetch_cache = {}
        # Fixed-cohort CPU serving may precompute an exact eligible posting
        # for predicates with more than one potentially unresolved attribute.
        # Those predicates cannot be represented by the single-target group
        # layout above, but their immutable serving decisions can still be
        # materialized once outside online timing and reused without changing
        # policy, threshold, or answer semantics.
        self._candidate_eligibility_posting_cache = {}
        self._candidate_eligibility_posting_cache_bytes = 0
        self._candidate_eligibility_posting_cache_max_entries = 256
        self._candidate_eligibility_posting_cache_max_bytes = 256 * 1024 * 1024
        self._candidate_eligibility_posting_cache_hits = 0
        self._candidate_eligibility_posting_cache_misses = 0
        self._predicate_eligibility_cache = {}
        self._predicate_eligibility_cache_max_entries = 256
        self._hard_predicate_plan_cache_hits = 0
        self._hard_predicate_plan_cache_misses = 0
        self._candidate_group_prefetch_cache_hits = 0
        self._candidate_group_prefetch_cache_misses = 0
        self._predicate_eligibility_cache_hits = 0
        self._predicate_eligibility_cache_misses = 0
        self._active_predicate_eligibility = None
        self._target_leaf_nodes = np.asarray(
            [int(path[0]) for path in self._target_ancestor_paths],
            dtype=np.int64)
        self._policy_templates_by_id = {
            template.template_id: template
            for template in self.policy.templates
        }
        self._selected_template_ids = {}
        for targets in {template.targets for template in self.policy.templates}:
            selected = np.empty(len(self.target_table), dtype=object)
            for tuple_id in range(len(self.target_table)):
                template = self.policy.select_indexed(
                    targets, self._target_columns, tuple_id)
                selected[tuple_id] = (
                    None if template is None else template.template_id)
            self._selected_template_ids[targets] = selected
        self._template_equivalence_group_ids = {}
        self._template_equivalence_rows = {}
        for template in self.policy.templates:
            group_ids = np.full(len(self.target_table), -1, dtype=np.int64)
            groups = {}
            for tuple_id in range(len(self.target_table)):
                if self._selected_template_ids[
                        template.targets][tuple_id] != template.template_id:
                    continue
                location = (-1 if not template.location_aware else
                            int(self._target_leaf_nodes[tuple_id]))
                body = tuple(
                    _freeze_key(self._target_columns[item.attr][tuple_id])
                    for item in template.body)
                key = (location, body)
                slot = groups.setdefault(key, len(groups))
                group_ids[tuple_id] = slot
            rows = [np.flatnonzero(group_ids == slot).astype(np.int64)
                    for slot in range(len(groups))]
            self._template_equivalence_group_ids[
                template.template_id] = group_ids
            self._template_equivalence_rows[template.template_id] = rows
        # Finite categorical domains admit an exact compact serving table:
        # context/body group x target category.  Building this contingency
        # table from the immutable support relation and frozen policy avoids
        # hundreds of scalar support requests on the first online occurrence
        # of a categorical predicate.  It never uses query/test answers.
        self._categorical_group_prefetch_tables = {}
        for template in self.policy.templates:
            table = self._build_categorical_group_prefetch_table(template)
            if table is not None:
                self._categorical_group_prefetch_tables[
                    template.template_id] = table
        self._prebuilt_body_masks = self.support_index.prebuild_body_masks(
            self.policy.templates, self._target_columns)
        self.counters = {
            "hard_pass": 0, "hard_fail": 0, "unresolved": 0,
            "statistical": 0, "abstain": 0, "support_requests": 0,
            "ancestor_context_requests": 0, "score_map_evaluations": 0,
            "scalar_relaxation_requests": 0,
            "statistical_accept": 0, "statistical_reject": 0,
            "deterministic_atoms": 0,
            "deterministic_pass": 0, "deterministic_fail": 0,
            "inference_cache_hits": 0, "inference_cache_misses": 0,
            "backoff_steps_total": 0, "backoff_steps_max": 0,
            "statistical_with_backoff": 0,
            "statistical_with_ancestor_context": 0,
            "statistical_with_scalar_relaxation": 0,
        }

    def _certified_domain(self, row, attr):
        domains = []
        for constraint in self.deterministic_constraints:
            domain = constraint(row, attr)
            if domain is not None:
                if not isinstance(domain, CertifiedDomain):
                    raise TypeError("deterministic constraints must return CertifiedDomain")
                domains.append(domain)
        if not domains:
            return None
        result = domains[0]
        for domain in domains[1:]:
            result = result.intersect(domain)
        return result

    def hard_decision(self, tuple_id, predicate):
        tuple_id = int(tuple_id)
        row = None
        residual = []
        for atom in predicate.atoms:
            value = self._target_columns[atom.attr][tuple_id]
            if _missing(value):
                if row is None and self.deterministic_constraints:
                    row = self.target_table.iloc[tuple_id]
                domain = self._certified_domain(row, atom.attr)
                if domain is not None:
                    certified = domain.classify(atom)
                    if certified == FALSE:
                        return "fail", ()
                    if certified == TRUE:
                        continue
                residual.append(atom)
            elif atom.eval_value(value) == FALSE:
                return "fail", ()
        if not residual:
            return "pass", ()
        return "unresolved", tuple(residual)

    def _contexts(self, template, tuple_id):
        body_levels = {item.attr: 0 for item in template.body}
        if template.location_aware:
            nodes = self._target_ancestor_paths[int(tuple_id)]
        else:
            nodes = (self.tree.root_id,)
        if self.context_mode == "leaf_only":
            return [(int(nodes[0]), dict(body_levels))]
        contexts = [(int(node), dict(body_levels)) for node in nodes]
        current_node = int(nodes[-1])
        body_by_name = {item.attr: item for item in template.body}
        for attr in template.scalar_relax_order:
            body = body_by_name[attr]
            while body_levels[attr] + 1 < len(body.levels):
                body_levels = dict(body_levels)
                body_levels[attr] += 1
                contexts.append((current_node, body_levels))
        return contexts

    def _build_categorical_group_prefetch_table(self, template):
        """Build exact location-free categorical counts by serving group."""
        if (self.support_mode != "joint" or
                self.support_retrieval_mode != "indexed" or
                template.location_aware or len(template.targets) != 1 or
                self.support_index.dtypes.get(template.targets[0]) == "num"):
            return None
        if any(
                self.support_index.dtypes.get(body.attr) == "num" or
                any(level.kind not in {"eq", "all"}
                    for level in body.levels)
                for body in template.body):
            return None
        group_rows = self._template_equivalence_rows.get(
            template.template_id, ())
        if not group_rows:
            return None
        representatives = np.asarray(
            [int(rows[0]) for rows in group_rows], dtype=np.int64)
        level_configs = []
        levels = {body.attr: 0 for body in template.body}
        level_configs.append(dict(levels))
        body_by_name = {body.attr: body for body in template.body}
        for attr in template.scalar_relax_order:
            body = body_by_name[attr]
            while levels[attr] + 1 < len(body.levels):
                levels = dict(levels)
                levels[attr] += 1
                level_configs.append(dict(levels))

        target_attr = template.targets[0]
        support_target = self.support_index.columns[target_attr]
        support_target_available = self.support_index.availability_mask[
            target_attr]
        category_slots = {}
        support_categories = np.full(len(support_target), -1, dtype=np.int32)
        for row in np.flatnonzero(support_target_available):
            key = _freeze_key(support_target[int(row)])
            slot = category_slots.setdefault(key, len(category_slots))
            support_categories[int(row)] = int(slot)

        group_count = len(group_rows)
        category_count = len(category_slots)
        denominators = np.zeros(
            (len(level_configs), group_count), dtype=np.int32)
        joint = np.zeros(
            (len(level_configs), group_count, category_count),
            dtype=np.int32)
        all_token = ("__MIFA_ALL__",)
        for level_index, level_by_attr in enumerate(level_configs):
            context_slots = {}
            group_contexts = np.empty(group_count, dtype=np.int32)
            for group_index, tuple_id in enumerate(representatives):
                key_items = []
                for body in template.body:
                    level = body.levels[int(level_by_attr[body.attr])]
                    if level.kind == "all":
                        key_items.append(all_token)
                    else:
                        value = self._target_columns[body.attr][tuple_id]
                        key_items.append(
                            ("__MIFA_MISSING__",) if _missing(value)
                            else _freeze_key(value))
                key = tuple(key_items)
                group_contexts[group_index] = context_slots.setdefault(
                    key, len(context_slots))

            counts = np.zeros(
                (len(context_slots), category_count), dtype=np.int32)
            context_rows = []
            category_rows = []
            for row in np.flatnonzero(support_target_available):
                key_items = []
                usable = True
                for body in template.body:
                    level = body.levels[int(level_by_attr[body.attr])]
                    if level.kind == "all":
                        key_items.append(all_token)
                        continue
                    value = self.support_index.columns[body.attr][int(row)]
                    if _missing(value):
                        usable = False
                        break
                    key_items.append(_freeze_key(value))
                if not usable:
                    continue
                context_slot = context_slots.get(tuple(key_items))
                if context_slot is None:
                    continue
                context_rows.append(int(context_slot))
                category_rows.append(int(support_categories[int(row)]))
            if context_rows:
                np.add.at(
                    counts,
                    (np.asarray(context_rows, dtype=np.int32),
                     np.asarray(category_rows, dtype=np.int32)), 1)
            grouped = counts[group_contexts]
            joint[level_index] = grouped
            denominators[level_index] = grouped.sum(
                axis=1, dtype=np.int64).astype(np.int32)
        denominators.setflags(write=False)
        joint.setflags(write=False)
        return {
            "category_slots": category_slots,
            "denominators": denominators,
            "joint": joint,
            "level_configs": tuple(level_configs),
            "resident_bytes": int(denominators.nbytes + joint.nbytes),
        }

    def _compile_categorical_group_prefetch(self, template, atoms):
        """Return all exact group decisions from a deployment contingency."""
        table = self._categorical_group_prefetch_tables.get(
            template.template_id)
        if (table is None or len(atoms) != 1 or atoms[0].op != "eq" or
                atoms[0].attr != template.targets[0]):
            return None
        category_slot = table["category_slots"].get(
            _freeze_key(atoms[0].a))
        denominators = table["denominators"].astype(
            np.float64, copy=False)
        if category_slot is None:
            plus = np.zeros_like(denominators)
        else:
            plus = table["joint"][:, :, int(category_slot)].astype(
                np.float64, copy=False)
        raw = np.full(denominators.shape, np.nan, dtype=np.float64)
        observed = denominators > 0
        raw[observed] = plus[observed] / denominators[observed]
        valid = (denominators >= int(template.min_support)) & np.isfinite(raw)
        has_selected = np.any(valid, axis=0)
        selected = np.argmax(valid, axis=0)
        broad = np.nan_to_num(raw[-1], nan=0.0)
        shrunken = broad.copy()
        for level_index in range(len(raw) - 2, -1, -1):
            use = (has_selected & (selected <= level_index) &
                   np.isfinite(raw[level_index]))
            if not np.any(use):
                continue
            n = denominators[level_index, use]
            shrunken[use] = (
                n * raw[level_index, use] +
                float(template.kappa) * shrunken[use]) / (
                    n + float(template.kappa))
        scores = self.policy.score_maps[template.template_id].predict(
            np.clip(shrunken, 0.0, 1.0))
        decisions = np.full(len(shrunken), 3, dtype=np.int8)
        decisions[has_selected & (scores >= self.theta)] = 1
        decisions[has_selected & (scores < self.theta)] = 2
        return decisions

    @staticmethod
    def _predicate_key(atoms):
        return tuple((str(atom.attr), str(atom.op),
                      _freeze_key(atom.a), _freeze_key(atom.b))
                     for atom in atoms)

    def begin_query(self, predicate=None, *, cache=True):
        """Select the exact-predicate cache for an online query.

        ``cache=False`` is reserved for disable-one diagnostics.  It leaves
        the immutable serving snapshot untouched while preventing this query
        from reading or populating the dense predicate-result cache.  Normal
        serving calls retain the historical default.
        """
        self._active_predicate_eligibility = None
        if (cache and predicate is not None and
                not self.deterministic_constraints):
            key = self._predicate_key(predicate.atoms)
            cached = self._predicate_eligibility_cache.get(key)
            if cached is None:
                self._predicate_eligibility_cache_misses += 1
                if len(self._predicate_eligibility_cache) >= \
                        self._predicate_eligibility_cache_max_entries:
                    self._predicate_eligibility_cache.clear()
                cached = np.full(
                    len(self.target_table), -1, dtype=np.int8)
                self._predicate_eligibility_cache[key] = cached
            else:
                self._predicate_eligibility_cache_hits += 1
            self._active_predicate_eligibility = cached

    def statistical_eligibility_cache(self):
        """Return the active predicate's dense unresolved-result cache."""
        return self._active_predicate_eligibility

    def serving_cache_manifest(self):
        """Describe snapshot-scoped exact caches used by the online executor."""
        counts = {
            "result_cache_entries": int(len(self._serving_result_cache)),
            "candidate_group_result_cache_entries": int(
                self._candidate_group_result_cache_entries),
            "candidate_predicate_plan_cache_entries": int(
                len(self._candidate_predicate_plan_cache)),
            "hard_predicate_plan_cache_entries": int(
                len(self._hard_predicate_plan_cache)),
            "candidate_group_prefetch_entries": int(
                len(self._candidate_group_prefetch_cache)),
            "candidate_eligibility_posting_entries": int(
                len(self._candidate_eligibility_posting_cache)),
            "predicate_cache_entries": int(
                len(self._predicate_eligibility_cache)),
            "support_satisfaction_entries": int(
                len(self._support_satisfaction_cache)),
            "support_satisfaction_packed_entries": int(
                len(self._support_satisfaction_packed_cache)),
            "support_leaf_prefix_entries": int(
                len(self._support_leaf_prefix_cache)),
        }
        total_entries = int(sum(counts.values()))
        return {
            "immutable_snapshot_required": True,
            **counts,
            "result_cache_max_entries": int(
                self._serving_result_cache_max_entries),
            "candidate_group_result_cache_max_entries": int(
                self._candidate_group_result_cache_max_entries),
            "predicate_cache_max_entries": int(
                self._predicate_eligibility_cache_max_entries),
            "candidate_predicate_plan_cache_max_entries": int(
                self._candidate_predicate_plan_cache_max_entries),
            "hard_predicate_plan_cache_hits": int(
                self._hard_predicate_plan_cache_hits),
            "hard_predicate_plan_cache_misses": int(
                self._hard_predicate_plan_cache_misses),
            "candidate_group_prefetch_cache_hits": int(
                self._candidate_group_prefetch_cache_hits),
            "candidate_group_prefetch_cache_misses": int(
                self._candidate_group_prefetch_cache_misses),
            "candidate_eligibility_posting_cache_hits": int(
                self._candidate_eligibility_posting_cache_hits),
            "candidate_eligibility_posting_cache_misses": int(
                self._candidate_eligibility_posting_cache_misses),
            "candidate_eligibility_posting_cache_bytes": int(
                self._candidate_eligibility_posting_cache_bytes),
            "candidate_eligibility_posting_cache_max_entries": int(
                self._candidate_eligibility_posting_cache_max_entries),
            "candidate_eligibility_posting_cache_max_bytes": int(
                self._candidate_eligibility_posting_cache_max_bytes),
            "predicate_eligibility_cache_hits": int(
                self._predicate_eligibility_cache_hits),
            "predicate_eligibility_cache_misses": int(
                self._predicate_eligibility_cache_misses),
            "active_predicate_cache": bool(
                self._active_predicate_eligibility is not None),
            "total_entries": total_entries,
            "all_serving_caches_empty": bool(
                total_entries == 0 and
                self._active_predicate_eligibility is None),
            "eviction": "bounded full reset",
        }

    def configure_candidate_cohort_cache(
            self, max_entries, *, dense_cache_max_bytes=256 * 1024 * 1024):
        """Size exact serving caches for one declared query cohort.

        Compiled plans and compact group-prefetch decisions are bounded by
        cohort cardinality.  Dense per-predicate eligibility state receives a
        separate byte cap so a multi-million-row snapshot never allocates one
        dense vector per cohort query.
        """
        max_entries = int(max_entries)
        dense_cache_max_bytes = int(dense_cache_max_bytes)
        if max_entries < 1 or dense_cache_max_bytes < 1:
            raise ValueError("candidate cohort cache limits must be positive")
        dense_entry_bytes = max(1, len(self.target_table))
        dense_entries = max(
            1, min(max_entries, dense_cache_max_bytes // dense_entry_bytes))
        self._candidate_predicate_plan_cache_max_entries = max_entries
        self._candidate_eligibility_posting_cache_max_entries = max_entries
        self._candidate_eligibility_posting_cache_max_bytes = \
            dense_cache_max_bytes
        self._predicate_eligibility_cache_max_entries = dense_entries
        self._support_satisfaction_cache_max_entries = max(
            self._support_satisfaction_cache_max_entries, max_entries)
        return {
            "candidate_plan_and_prefetch_max_entries": max_entries,
            "dense_predicate_cache_max_entries": int(dense_entries),
            "dense_predicate_cache_max_bytes": dense_cache_max_bytes,
            "dense_predicate_entry_bytes": int(dense_entry_bytes),
        }

    def clear_all_serving_caches(self):
        """Reset every mutable online cache before a cold measurement."""
        self._support_satisfaction_cache.clear()
        self._support_satisfaction_packed_cache.clear()
        self._support_leaf_prefix_cache.clear()
        self._serving_result_cache.clear()
        self._candidate_group_result_cache.clear()
        self._candidate_group_result_cache_entries = 0
        self._candidate_predicate_plan_cache.clear()
        self._hard_predicate_plan_cache.clear()
        self._candidate_group_prefetch_cache.clear()
        self._candidate_eligibility_posting_cache.clear()
        self._candidate_eligibility_posting_cache_bytes = 0
        self._predicate_eligibility_cache.clear()
        self._active_predicate_eligibility = None
        self._hard_predicate_plan_cache_hits = 0
        self._hard_predicate_plan_cache_misses = 0
        self._candidate_group_prefetch_cache_hits = 0
        self._candidate_group_prefetch_cache_misses = 0
        self._candidate_eligibility_posting_cache_hits = 0
        self._candidate_eligibility_posting_cache_misses = 0
        self._predicate_eligibility_cache_hits = 0
        self._predicate_eligibility_cache_misses = 0
        manifest = self.serving_cache_manifest()
        if not manifest["all_serving_caches_empty"]:
            raise AssertionError("serving cache reset is incomplete")
        return manifest

    def record_executor_cache_hits(self, accepted, rejected, abstained=0):
        """Account for statistical decisions consumed inside a native kernel."""
        accepted = int(accepted)
        rejected = int(rejected)
        abstained = int(abstained)
        hits = accepted + rejected + abstained
        if not hits:
            return
        self.counters["unresolved"] += hits
        self.counters["statistical"] += accepted + rejected
        self.counters["abstain"] += abstained
        self.counters["inference_cache_hits"] += hits
        self.counters["statistical_accept"] += accepted
        self.counters["statistical_reject"] += rejected

    def _result_cache_key(self, residual, template, tuple_id):
        """Return the immutable serving signature for one statistical result."""
        tuple_id = int(tuple_id)
        location = (-1 if not template.location_aware else
                    int(self._target_ancestor_paths[tuple_id][0]))
        body = tuple(
            (item.attr, _freeze_key(self._target_columns[item.attr][tuple_id]))
            for item in template.body)
        return (self._predicate_key(residual), template.template_id,
                location, body)

    def _remember_result(self, key, result):
        if key is None:
            return result
        if len(self._serving_result_cache) >= \
                self._serving_result_cache_max_entries:
            # Workloads are evaluated in deterministic order.  A bounded full
            # reset avoids per-hit OrderedDict mutation on the critical path
            # while keeping memory stable for long-lived snapshots.
            self._serving_result_cache.clear()
        self._serving_result_cache[key] = result
        return result

    def _record_cached_result(self, result):
        self.counters["inference_cache_hits"] += 1
        if result.state == "abstain":
            self.counters["abstain"] += 1
        elif result.state == "statistical":
            self.counters["statistical"] += 1
            self.counters["backoff_steps_total"] += int(result.backoff_steps)
            self.counters["backoff_steps_max"] = max(
                self.counters["backoff_steps_max"], int(result.backoff_steps))
            self.counters["statistical_with_backoff"] += int(
                result.backoff_steps > 0)
            self.counters["statistical_with_ancestor_context"] += int(
                result.ancestor_backoff)
            self.counters["statistical_with_scalar_relaxation"] += int(
                result.scalar_backoff)
            self.counters[
                "statistical_accept" if result.eligible
                else "statistical_reject"] += 1

    def compile_hard_predicate(self, predicate):
        """Compile direct checks for the executor's semantics-equivalent kernel.

        Independently supplied deterministic constraints may inspect an entire
        pandas row, so they deliberately retain the scalar reference path.
        ``in`` predicates are likewise uncommon in the declared workloads and
        fall back rather than introducing a variable-width kernel encoding.
        """
        if self.deterministic_constraints:
            return None
        predicate_key = self._predicate_key(predicate.atoms)
        cached = self._hard_predicate_plan_cache.get(predicate_key)
        if cached is not None:
            self._hard_predicate_plan_cache_hits += 1
            return cached
        self._hard_predicate_plan_cache_misses += 1
        op_codes = {"le": 0, "ge": 1, "between": 2, "eq": 3}
        attrs = []
        ops = []
        lower = []
        upper = []
        for atom in predicate.atoms:
            attr_index = self._fast_attr_index.get(atom.attr)
            if attr_index is None or atom.op not in op_codes:
                return None
            categorical = self.support_index.dtypes.get(atom.attr) != "num"
            if categorical and atom.op != "eq":
                return None
            attrs.append(int(attr_index))
            ops.append(int(op_codes[atom.op]))
            if categorical:
                code = self._fast_category_codes.get(atom.attr, {}).get(
                    _freeze_key(atom.a), -2)
                lower.append(float(code))
                upper.append(float(code))
            else:
                lower.append(float(atom.a))
                upper.append(float(atom.b) if atom.b is not None else 0.0)
        plan = {
            "values": self._fast_values,
            "missing": self._fast_missing,
            "attrs": np.ascontiguousarray(attrs, dtype=np.int32),
            "ops": np.ascontiguousarray(ops, dtype=np.int8),
            "lower": np.ascontiguousarray(lower, dtype=np.float64),
            "upper": np.ascontiguousarray(upper, dtype=np.float64),
            "has_unresolved_attributes": bool(any(
                atom.attr in self._fast_attrs_with_missing
                for atom in predicate.atoms)),
        }
        if len(self._hard_predicate_plan_cache) >= \
                self._candidate_predicate_plan_cache_max_entries:
            self._hard_predicate_plan_cache.clear()
        self._hard_predicate_plan_cache[predicate_key] = plan
        return plan

    def _support_joint_satisfaction(self, residual, *, cache=True):
        key = self._predicate_key(residual)
        cached = (self._support_satisfaction_cache.get(key)
                  if cache else None)
        if cached is not None:
            return cached
        satisfied = self.support_index.predicate_satisfaction_mask(residual)
        satisfied.setflags(write=False)
        if cache:
            if len(self._support_satisfaction_cache) >= \
                    self._support_satisfaction_cache_max_entries:
                self._support_satisfaction_cache.clear()
                self._support_satisfaction_packed_cache.clear()
            self._support_satisfaction_cache[key] = satisfied
        return satisfied

    def result_equivalent_rows(
            self, tuple_id, template_id, residual_codes):
        """Rows sharing one exact serving-result signature and residual."""
        if template_id is None or template_id not in \
                self._template_equivalence_group_ids:
            return np.asarray([int(tuple_id)], dtype=np.int64)
        group_id = int(
            self._template_equivalence_group_ids[template_id][int(tuple_id)])
        if group_id < 0:
            return np.asarray([int(tuple_id)], dtype=np.int64)
        candidates = self._template_equivalence_rows[template_id][group_id]
        desired_code = int(residual_codes[int(tuple_id)])
        if desired_code <= 0:
            return np.asarray([int(tuple_id)], dtype=np.int64)
        return candidates[residual_codes[candidates] == desired_code]

    def compile_candidate_group_prefetch(self, predicate):
        """Compile exact one-unresolved-attribute decisions by result group."""
        if self.deterministic_constraints:
            return None
        unresolved_atoms = tuple(
            atom for atom in predicate.atoms
            if atom.attr in self._fast_attrs_with_missing)
        targets = tuple(sorted({atom.attr for atom in unresolved_atoms}))
        if len(targets) != 1 or not unresolved_atoms:
            return None
        target_attr = targets[0]
        predicate_key = self._predicate_key(unresolved_atoms)
        cache_key = (targets, predicate_key)
        cached = self._candidate_group_prefetch_cache.get(cache_key)
        if cached is not None:
            self._candidate_group_prefetch_cache_hits += 1
            return cached
        self._candidate_group_prefetch_cache_misses += 1

        templates = [
            template for template in self.policy.templates
            if template.targets == targets]
        if len(templates) == 1:
            decisions = self._compile_categorical_group_prefetch(
                templates[0], unresolved_atoms)
            if decisions is not None:
                cached = {
                    "group_ids": np.ascontiguousarray(
                        self._template_equivalence_group_ids[
                            templates[0].template_id], dtype=np.int32),
                    "decisions": decisions,
                    "predicate_key": predicate_key,
                    "source": "categorical_contingency",
                }
                if len(self._candidate_group_prefetch_cache) >= \
                        self._candidate_predicate_plan_cache_max_entries:
                    self._candidate_group_prefetch_cache.clear()
                self._candidate_group_prefetch_cache[cache_key] = cached
                return cached

        layout = self._candidate_group_prefetch_layouts.get(targets)
        if layout is None:
            if not templates:
                return None
            row_groups = np.full(
                len(self.target_table), -1, dtype=np.int32)
            group_rows = []
            for template in templates:
                template_groups = self._template_equivalence_group_ids[
                    template.template_id]
                offset = len(group_rows)
                selected = template_groups >= 0
                row_groups[selected] = (
                    template_groups[selected] + offset).astype(
                        np.int32, copy=False)
                group_rows.extend(
                    self._template_equivalence_rows[template.template_id])
            layout = (row_groups, tuple(group_rows))
            self._candidate_group_prefetch_layouts[targets] = layout
        row_groups, group_rows = layout
        decisions = np.full(len(group_rows), 3, dtype=np.int8)
        missing = self._fast_missing[self._fast_attr_index[target_attr]]
        for group_index, rows in enumerate(group_rows):
            unresolved_rows = rows[missing[rows]]
            if not len(unresolved_rows):
                continue
            result = self.evaluate(
                int(unresolved_rows[0]), predicate, record=False,
                assume_unresolved=True)
            decisions[group_index] = (
                1 if result.eligible else
                2 if result.state == "statistical" else 3)
        cached = {
            "group_ids": row_groups,
            "decisions": decisions,
            "predicate_key": predicate_key,
        }
        if len(self._candidate_group_prefetch_cache) >= \
                self._candidate_predicate_plan_cache_max_entries:
            self._candidate_group_prefetch_cache.clear()
        self._candidate_group_prefetch_cache[cache_key] = cached
        return cached

    def compile_candidate_eligibility_posting(
            self, predicate, *, materialize=False):
        """Return an immutable exact eligible-row posting for one predicate.

        The normal one-target group prefetch remains the cheaper first choice.
        This covers fixed-cohort predicates with multiple potentially
        unresolved attributes.  Materialization is opt-in so a normal online
        request never performs corpus-wide preparation; the measurement
        harness invokes it only during its explicitly excluded prewarm phase.
        """
        if self.deterministic_constraints:
            return None
        key = self._predicate_key(predicate.atoms)
        cached = self._candidate_eligibility_posting_cache.get(key)
        if cached is not None:
            self._candidate_eligibility_posting_cache_hits += 1
            return cached
        self._candidate_eligibility_posting_cache_misses += 1
        if not materialize:
            return None
        materialized = self.materialize_calibration_policy(predicate)
        if materialized is None:
            return None
        states, scores, _ = materialized
        eligible = (
            (states == 1) |
            ((states == 2) & (scores >= float(self.theta))))
        posting = np.ascontiguousarray(
            np.flatnonzero(eligible), dtype=np.uint32)
        if (posting.nbytes >
                self._candidate_eligibility_posting_cache_max_bytes):
            return posting
        while self._candidate_eligibility_posting_cache and (
                len(self._candidate_eligibility_posting_cache) >=
                self._candidate_eligibility_posting_cache_max_entries or
                self._candidate_eligibility_posting_cache_bytes +
                posting.nbytes >
                self._candidate_eligibility_posting_cache_max_bytes):
            oldest = next(iter(self._candidate_eligibility_posting_cache))
            removed = self._candidate_eligibility_posting_cache.pop(oldest)
            self._candidate_eligibility_posting_cache_bytes -= int(
                removed.nbytes)
        self._candidate_eligibility_posting_cache[key] = posting
        self._candidate_eligibility_posting_cache_bytes += int(
            posting.nbytes)
        return posting

    def _evaluate_single_residual_candidate_batch(
            self, tuple_ids, predicate, predicate_key, predicate_plan):
        """Fast exact path when one missing atom selects one template."""
        batch_started = perf_counter()
        support_inference_ms = 0.0
        selected = predicate_plan.get("single_selected_templates")
        if selected is None:
            targets = (predicate.atoms[0].attr,)
            selected = self._selected_template_ids.get(targets)
            if selected is None:
                return None
            predicate_plan["single_selected_templates"] = selected
        selected_ids = selected[tuple_ids]
        template_id = selected_ids[0]
        if template_id is None or not np.all(selected_ids == template_id):
            return None
        group_ids = self._template_equivalence_group_ids[
            template_id][tuple_ids]
        if np.any(group_ids < 0):
            raise RuntimeError(
                "selected template has no candidate equivalence group")
        cache_key = (predicate_key, str(template_id))
        cached_codes = self._candidate_group_result_cache.get(cache_key)
        if cached_codes is None:
            cached_codes = np.zeros(
                len(self._template_equivalence_rows[template_id]),
                dtype=np.int8)
        output = cached_codes[group_ids].copy()
        found = output != 0
        cache_accepts = int(np.count_nonzero(output[found] == 1))
        cache_rejects = int(np.count_nonzero(output[found] == 2))
        cache_abstains = int(np.count_nonzero(output[found] == 3))
        missing_positions = np.flatnonzero(~found)
        if not len(missing_positions):
            total_ms = (perf_counter() - batch_started) * 1000.0
            return output, {
                "representative_evaluations": 0,
                "group_cache_hits": int(len(output)),
                "cache_accepts": cache_accepts,
                "cache_rejects": cache_rejects,
                "cache_abstains": cache_abstains,
                "support_inference_ms": 0.0,
                "cache_lookup_ms": float(total_ms),
            }

        missing_groups = group_ids[missing_positions]
        unique_groups, first, inverse, counts = np.unique(
            missing_groups, return_index=True, return_inverse=True,
            return_counts=True)
        unique_codes = np.empty(len(unique_groups), dtype=np.int8)
        for group_slot, first_local in enumerate(first):
            representative = int(tuple_ids[
                missing_positions[int(first_local)]])
            inference_started = perf_counter()
            result = self.evaluate(
                representative, predicate, assume_unresolved=True)
            support_inference_ms += (
                perf_counter() - inference_started) * 1000.0
            unique_codes[group_slot] = (
                1 if result.eligible else
                2 if result.state == "statistical" else 3)
        missing_codes = unique_codes[inverse]
        output[missing_positions] = missing_codes
        duplicate_counts = counts - 1
        cache_accepts += int(duplicate_counts[
            unique_codes == 1].sum(initial=0))
        cache_rejects += int(duplicate_counts[
            unique_codes == 2].sum(initial=0))
        cache_abstains += int(duplicate_counts[
            unique_codes == 3].sum(initial=0))

        if (self._candidate_group_result_cache_entries +
                len(unique_groups) >
                self._candidate_group_result_cache_max_entries):
            self._candidate_group_result_cache.clear()
            self._candidate_group_result_cache_entries = 0
            cached_codes = np.zeros(
                len(self._template_equivalence_rows[template_id]),
                dtype=np.int8)
        if len(unique_groups) <= \
                self._candidate_group_result_cache_max_entries:
            cached_codes[unique_groups] = unique_codes
            self._candidate_group_result_cache[cache_key] = cached_codes
            self._candidate_group_result_cache_entries += int(
                len(unique_groups))
        total_ms = (perf_counter() - batch_started) * 1000.0
        return output, {
            "representative_evaluations": int(len(unique_groups)),
            "group_cache_hits": int(
                cache_accepts + cache_rejects + cache_abstains),
            "cache_accepts": cache_accepts,
            "cache_rejects": cache_rejects,
            "cache_abstains": cache_abstains,
            "support_inference_ms": float(support_inference_ms),
            "cache_lookup_ms": float(max(
                total_ms - support_inference_ms, 0.0)),
        }

    def evaluate_candidate_batch(self, tuple_ids, predicate):
        """Evaluate unresolved compact candidates by exact serving group.

        Rows in one precomputed template equivalence group have the same
        selected template, location context, and scalar body.  When their
        residual predicates are also identical, ``evaluate`` is immutable and
        returns the same result.  One representative evaluation is therefore
        sufficient; unlike reference propagation, no N-length residual-code
        vector is constructed.
        """
        batch_started = perf_counter()
        support_inference_ms = 0.0
        tuple_ids = np.ascontiguousarray(tuple_ids, dtype=np.int64)
        if tuple_ids.ndim != 1:
            raise ValueError("candidate tuple IDs must be one-dimensional")
        if not len(tuple_ids):
            return np.empty(0, dtype=np.int8), {
                "representative_evaluations": 0,
                "group_cache_hits": 0,
                "cache_accepts": 0,
                "cache_rejects": 0,
                "cache_abstains": 0,
                "support_inference_ms": 0.0,
                "cache_lookup_ms": 0.0,
            }
        if self.deterministic_constraints:
            raise RuntimeError(
                "candidate inference batching excludes deterministic constraints")
        atoms = tuple(predicate.atoms)
        if len(atoms) > 63:
            raise RuntimeError(
                "candidate inference batching supports at most 63 atoms")
        predicate_key = self._predicate_key(atoms)
        predicate_plan = self._candidate_predicate_plan_cache.get(
            predicate_key)
        if predicate_plan is None:
            predicate_plan = {
                "attr_indices": np.asarray([
                    self._fast_attr_index[atom.attr] for atom in atoms],
                    dtype=np.int64),
                "powers": np.left_shift(
                    np.uint64(1),
                    np.arange(len(atoms), dtype=np.uint64)),
                "residuals": {},
            }
            if len(self._candidate_predicate_plan_cache) >= \
                    self._candidate_predicate_plan_cache_max_entries:
                self._candidate_predicate_plan_cache.clear()
            self._candidate_predicate_plan_cache[
                predicate_key] = predicate_plan
        if len(atoms) == 1:
            fast_result = self._evaluate_single_residual_candidate_batch(
                tuple_ids, predicate, predicate_key, predicate_plan)
            if fast_result is not None:
                return fast_result
        attr_indices = predicate_plan["attr_indices"]
        missing = self._fast_missing[
            attr_indices[:, None], tuple_ids[None, :]]
        residual_codes = np.sum(
            missing.astype(np.uint64) *
            predicate_plan["powers"][:, None], axis=0,
            dtype=np.uint64)
        if np.any(residual_codes == 0):
            raise RuntimeError(
                "candidate inference batch contains a resolved tuple")

        output = np.empty(len(tuple_ids), dtype=np.int8)
        representative_evaluations = 0
        cache_accepts = 0
        cache_rejects = 0
        cache_abstains = 0
        for residual_code in np.unique(residual_codes):
            positions = np.flatnonzero(residual_codes == residual_code)
            residual_slot = int(residual_code)
            residual_plan = predicate_plan["residuals"].get(
                residual_slot)
            if residual_plan is None:
                residual = tuple(
                    atom for atom_index, atom in enumerate(atoms)
                    if residual_slot & (1 << atom_index))
                targets = tuple(sorted({atom.attr for atom in residual}))
                residual_plan = (
                    residual,
                    self._predicate_key(residual),
                    self._selected_template_ids.get(targets))
                predicate_plan["residuals"][
                    residual_slot] = residual_plan
            residual, residual_key, selected = residual_plan
            if selected is None:
                output[positions] = 3
                cache_abstains += int(len(positions))
                continue
            selected_ids = selected[tuple_ids[positions]]
            first_template = selected_ids[0]
            if np.all(selected_ids == first_template):
                positions_by_template = {
                    first_template: np.arange(
                        len(selected_ids), dtype=np.int64)}
            else:
                positions_by_template = {}
                for local_position, template_id in enumerate(selected_ids):
                    positions_by_template.setdefault(
                        template_id, []).append(int(local_position))
            for template_id, local_positions in positions_by_template.items():
                current_positions = positions[np.asarray(
                    local_positions, dtype=np.int64)]
                if template_id is None:
                    output[current_positions] = 3
                    cache_abstains += int(len(current_positions))
                    continue
                group_ids = self._template_equivalence_group_ids[
                    template_id][tuple_ids[current_positions]]
                if np.any(group_ids < 0):
                    raise RuntimeError(
                        "selected template has no candidate equivalence group")
                cache_key = (residual_key, str(template_id))
                cached_codes = self._candidate_group_result_cache.get(
                    cache_key)
                if cached_codes is None:
                    cached_codes = np.zeros(
                        len(self._template_equivalence_rows[template_id]),
                        dtype=np.int8)
                if (np.any(group_ids < 0) or
                        np.any(group_ids >= len(cached_codes))):
                    raise RuntimeError(
                        "candidate equivalence group is out of range")
                group_codes = cached_codes[group_ids]
                found = group_codes != 0
                found_positions = np.flatnonzero(found)
                if len(found_positions):
                    found_codes = group_codes[found_positions]
                    output[current_positions[found_positions]] = found_codes
                    cache_accepts += int(np.count_nonzero(found_codes == 1))
                    cache_rejects += int(np.count_nonzero(found_codes == 2))
                    cache_abstains += int(np.count_nonzero(found_codes == 3))

                missing_local = np.flatnonzero(~found)
                if not len(missing_local):
                    continue
                missing_groups = group_ids[missing_local]
                unique_groups, first, inverse, counts = np.unique(
                    missing_groups, return_index=True, return_inverse=True,
                    return_counts=True)
                unique_codes = np.empty(len(unique_groups), dtype=np.int8)
                for group_slot, first_local in enumerate(first):
                    representative_position = current_positions[
                        missing_local[int(first_local)]]
                    representative = int(
                        tuple_ids[representative_position])
                    inference_started = perf_counter()
                    result = self.evaluate(
                        representative, predicate,
                        assume_unresolved=True)
                    support_inference_ms += (
                        perf_counter() - inference_started) * 1000.0
                    representative_evaluations += 1
                    unique_codes[group_slot] = (
                        1 if result.eligible else
                        2 if result.state == "statistical" else 3)
                missing_codes = unique_codes[inverse]
                output[current_positions[missing_local]] = missing_codes
                duplicate_counts = counts - 1
                cache_accepts += int(duplicate_counts[
                    unique_codes == 1].sum(initial=0))
                cache_rejects += int(duplicate_counts[
                    unique_codes == 2].sum(initial=0))
                cache_abstains += int(duplicate_counts[
                    unique_codes == 3].sum(initial=0))

                if (self._candidate_group_result_cache_entries +
                        len(unique_groups) >
                        self._candidate_group_result_cache_max_entries):
                    self._candidate_group_result_cache.clear()
                    self._candidate_group_result_cache_entries = 0
                    cached_codes = np.zeros(
                        len(self._template_equivalence_rows[template_id]),
                        dtype=np.int8)
                if len(unique_groups) <= \
                        self._candidate_group_result_cache_max_entries:
                    cached_codes[unique_groups] = unique_codes
                    self._candidate_group_result_cache[cache_key] = \
                        cached_codes
                    self._candidate_group_result_cache_entries += int(
                        len(unique_groups))

        group_cache_hits = cache_accepts + cache_rejects + cache_abstains
        total_ms = (perf_counter() - batch_started) * 1000.0
        return output, {
            "representative_evaluations": int(
                representative_evaluations),
            "group_cache_hits": int(group_cache_hits),
            "cache_accepts": int(cache_accepts),
            "cache_rejects": int(cache_rejects),
            "cache_abstains": int(cache_abstains),
            "support_inference_ms": float(support_inference_ms),
            "cache_lookup_ms": float(max(
                total_ms - support_inference_ms, 0.0)),
        }

    def evaluate(self, tuple_id, predicate, *, record=True,
                 assume_unresolved=False, cache=True,
                 offline_cache=False):
        if record and self.deterministic_constraints:
            row = self.target_table.iloc[int(tuple_id)]
            for atom in predicate.atoms:
                if not _missing(row[atom.attr]):
                    continue
                domain = self._certified_domain(row, atom.attr)
                if domain is None:
                    continue
                self.counters["deterministic_atoms"] += 1
                certified = domain.classify(atom)
                if certified == TRUE:
                    self.counters["deterministic_pass"] += 1
                elif certified == FALSE:
                    self.counters["deterministic_fail"] += 1
        if assume_unresolved and not self.deterministic_constraints:
            residual = tuple(
                atom for atom in predicate.atoms
                if self._fast_missing[
                    self._fast_attr_index[atom.attr], int(tuple_id)])
            state = "unresolved"
            if not residual:
                # Defensive fallback for callers that did not establish the
                # fused direct-decision precondition.
                state, residual = self.hard_decision(tuple_id, predicate)
        else:
            state, residual = self.hard_decision(tuple_id, predicate)
        if state == "pass":
            if record:
                self.counters["hard_pass"] += 1
            return InferenceResult("hard_pass", True, score=1.0, raw_score=1.0)
        if state == "fail":
            if record:
                self.counters["hard_fail"] += 1
            return InferenceResult("hard_fail", False, score=0.0, raw_score=0.0)
        if record:
            self.counters["unresolved"] += 1
        if self.template_selection_mode == "indexed":
            template = self.policy.select_indexed(
                (atom.attr for atom in residual), self._target_columns, tuple_id)
        else:
            template = self.policy.select(
                (atom.attr for atom in residual),
                self.target_table.iloc[int(tuple_id)])
        if template is None:
            if record:
                self.counters["abstain"] += 1
            return InferenceResult("abstain", False)

        cache_key = None
        # Correctness oracles call ``record=False`` and must remain completely
        # side-effect free: in particular, an exact audit that runs after one
        # timed query must not prewarm the next online query.  Serving searches
        # use the default ``record=True`` path and retain the exact cache.
        # Offline calibration owns a short-lived inference snapshot and may
        # opt into the same immutable-result caches without incrementing any
        # online counters.  The caller clears those caches before the test
        # model is constructed; ordinary ``record=False`` audits remain fully
        # side-effect free.
        cache_active = bool((record or offline_cache) and cache and
                            not self.deterministic_constraints)
        if cache_active:
            cache_key = self._result_cache_key(residual, template, tuple_id)
            cached = self._serving_result_cache.get(cache_key)
            if cached is not None:
                if record:
                    self._record_cached_result(cached)
                return cached
            if record:
                self.counters["inference_cache_misses"] += 1

        estimates = []
        contexts = self._contexts(template, tuple_id)
        satisfaction_atoms = (
            (tuple(residual),) if self.support_mode == "joint"
            else tuple((atom,) for atom in residual))
        support_satisfied = [
            self._support_joint_satisfaction(atoms, cache=cache_active)
            for atoms in satisfaction_atoms]
        support_satisfied_packed = []
        for atoms, satisfied in zip(satisfaction_atoms, support_satisfied):
            packed_key = self._predicate_key(atoms)
            packed = (
                self._support_satisfaction_packed_cache.get(packed_key)
                if cache_active else None)
            if packed is None:
                packed = self.support_index.pack_mask(satisfied)
                if cache_active:
                    self._support_satisfaction_packed_cache[packed_key] = packed
            support_satisfied_packed.append(packed)
        support_mask_cache = {}
        prefix = None
        if not template.body and self.support_retrieval_mode == "indexed":
            prefix = []
            for atoms, satisfied in zip(satisfaction_atoms, support_satisfied):
                prefix_key = (
                    tuple(template.targets), self._predicate_key(atoms))
                cached_prefix = (
                    self._support_leaf_prefix_cache.get(prefix_key)
                    if cache_active else None)
                if cached_prefix is None:
                    cached_prefix = self.support_index.leaf_prefix_joint_counts(
                        template, satisfied)
                    if cache_active:
                        if len(self._support_leaf_prefix_cache) >= \
                                self._support_leaf_prefix_cache_max_entries:
                            self._support_leaf_prefix_cache.clear()
                        self._support_leaf_prefix_cache[prefix_key] = cached_prefix
                prefix.append(cached_prefix)
        leaf_node = int(contexts[0][0]) if template.location_aware else None
        for step, (node_id, levels) in enumerate(contexts):
            counts = []
            for atom_index, satisfied in enumerate(support_satisfied):
                if self.support_retrieval_mode == "scan":
                    count = self.support_index.count_satisfaction_scan(
                        template, self._target_columns, tuple_id, satisfied,
                        node_id=node_id, level_by_attr=levels,
                        record_stats=record)
                elif prefix is not None:
                    count = self.support_index.count_joint_from_leaf_prefix(
                        template, node_id, prefix[atom_index],
                        record_stats=record)
                else:
                    count = self.support_index.count_joint_indexed(
                        template, self._target_columns, tuple_id, satisfied,
                        node_id=node_id, level_by_attr=levels,
                        mask_cache=support_mask_cache,
                        cache_body=cache_active,
                        record_stats=record,
                        satisfied_packed=support_satisfied_packed[atom_index])
                counts.append(count)
            support_n = counts[0][0]
            if any(count[0] != support_n for count in counts[1:]):
                raise AssertionError("marginal support denominators drifted")
            atom_plus = [count[1] for count in counts]
            if record:
                self.counters["support_requests"] += 1
                if template.location_aware and int(node_id) != leaf_node:
                    self.counters["ancestor_context_requests"] += 1
                if any(level > 0 for level in levels.values()):
                    self.counters["scalar_relaxation_requests"] += 1
            if support_n:
                probabilities = np.asarray(atom_plus, dtype=np.float64) / support_n
                if self.support_mode == "joint":
                    raw = float(probabilities[0])
                elif self.support_mode == "product":
                    raw = float(np.prod(probabilities))
                else:
                    raw = float(np.min(probabilities))
                plus = int(round(raw * support_n))
            else:
                plus = 0
                raw = np.nan
            estimates.append({"node": node_id, "levels": levels, "n": support_n,
                              "plus": plus, "raw": raw, "step": step})

        selected = next((index for index, item in enumerate(estimates)
                         if item["n"] >= template.min_support and
                         np.isfinite(item["raw"])), None)
        if selected is None:
            if record:
                self.counters["abstain"] += 1
            return self._remember_result(cache_key, InferenceResult(
                "abstain", False, template_id=template.template_id,
                backoff_steps=len(estimates)))

        broad = estimates[-1]["raw"]
        if not np.isfinite(broad):
            broad = 0.0
        shrunken = float(broad)
        for item in reversed(estimates[selected:-1]):
            if np.isfinite(item["raw"]):
                shrunken = ((item["n"] * item["raw"] + template.kappa * shrunken) /
                            (item["n"] + template.kappa))
        raw_score = float(np.clip(shrunken, 0.0, 1.0))
        score = float(self.policy.score_maps[template.template_id].predict(raw_score))
        chosen = estimates[selected]
        ancestor_backoff = bool(
            template.location_aware and int(chosen["node"]) != leaf_node)
        scalar_backoff = bool(
            any(level > 0 for level in chosen["levels"].values()))
        if record:
            self.counters["statistical"] += 1
            self.counters["backoff_steps_total"] += int(chosen["step"])
            self.counters["backoff_steps_max"] = max(
                self.counters["backoff_steps_max"], int(chosen["step"]))
            self.counters["statistical_with_backoff"] += int(
                chosen["step"] > 0)
            self.counters["statistical_with_ancestor_context"] += int(
                ancestor_backoff)
            self.counters["statistical_with_scalar_relaxation"] += int(
                scalar_backoff)
            self.counters["score_map_evaluations"] += 1
            self.counters[
                "statistical_accept" if score >= self.theta
                else "statistical_reject"] += 1
        return self._remember_result(cache_key, InferenceResult(
            "statistical", score >= self.theta, score=score,
            raw_score=raw_score, template_id=template.template_id,
            context_node=int(chosen["node"]),
            context_depth=int(self.tree.nodes[chosen["node"]].depth),
            support_n=int(chosen["n"]), support_n_plus=int(chosen["plus"]),
            backoff_steps=int(chosen["step"]),
            ancestor_backoff=ancestor_backoff,
            scalar_backoff=scalar_backoff,
        ))

    def materialize_calibration_policy(self, predicate):
        """Materialize exact hard/statistical calibration arrays by group.

        This is an offline-only equivalent of calling :meth:`evaluate` for
        every target row.  Direct predicate states are vectorized using the
        same compiled scalar representation as the exact online executor.
        Unresolved rows are grouped only when they share the residual
        predicate, selected template, target leaf, and scalar body values --
        the exact immutable-result equivalence relation already used by
        ``evaluate_candidate_batch``.  One representative therefore supplies
        the identical state and score for every row in its group.

        ``None`` requests the scalar fallback for unsupported predicates or
        deterministic constraints.
        """
        if self.deterministic_constraints:
            return None
        atoms = tuple(predicate.atoms)
        if not atoms or len(atoms) > 63:
            return None
        plan = self.compile_hard_predicate(predicate)
        if plan is None:
            return None

        n_rows = len(self.target_table)
        states = np.zeros(n_rows, dtype=np.int8)
        scores = np.zeros(n_rows, dtype=np.float64)
        failed = np.zeros(n_rows, dtype=np.bool_)
        residual_codes = np.zeros(n_rows, dtype=np.uint64)
        for atom_index in range(len(atoms)):
            attr = int(plan["attrs"][atom_index])
            missing = plan["missing"][attr]
            values = plan["values"][attr]
            op = int(plan["ops"][atom_index])
            if op == 0:
                satisfied = values <= plan["lower"][atom_index]
            elif op == 1:
                satisfied = values >= plan["lower"][atom_index]
            elif op == 2:
                satisfied = (
                    (values >= plan["lower"][atom_index]) &
                    (values <= plan["upper"][atom_index]))
            else:
                satisfied = values == plan["lower"][atom_index]
            failed |= (~missing) & (~satisfied)
            residual_codes[missing] |= (
                np.uint64(1) << np.uint64(atom_index))

        unresolved = (~failed) & (residual_codes != 0)
        hard_pass = (~failed) & (~unresolved)
        states[hard_pass] = 1
        # 0 = hard_fail, 1 = hard_pass, 2 = statistical, 3 = abstain.
        states[unresolved] = 3
        representative_evaluations = 0
        equivalence_groups = 0

        for code in np.unique(residual_codes[unresolved]):
            code_int = int(code)
            code_rows = np.flatnonzero(unresolved & (residual_codes == code))
            residual = tuple(
                atom for atom_index, atom in enumerate(atoms)
                if code_int & (1 << atom_index))
            targets = tuple(sorted({atom.attr for atom in residual}))
            selected = self._selected_template_ids.get(targets)
            if selected is None:
                continue
            selected_ids = selected[code_rows]
            for template in self.policy.templates:
                template_id = template.template_id
                if template.targets != targets:
                    continue
                template_rows = code_rows[selected_ids == template_id]
                if not len(template_rows):
                    continue
                group_ids = self._template_equivalence_group_ids[
                    template_id][template_rows]
                if np.any(group_ids < 0):
                    raise RuntimeError(
                        "calibration row has no template equivalence group")
                unique_groups, first, inverse = np.unique(
                    group_ids, return_index=True, return_inverse=True)
                group_states = np.empty(len(unique_groups), dtype=np.int8)
                group_scores = np.zeros(len(unique_groups), dtype=np.float64)
                for group_slot, first_position in enumerate(first):
                    representative = int(template_rows[int(first_position)])
                    result = self.evaluate(
                        representative, predicate, record=False,
                        assume_unresolved=True, offline_cache=True)
                    representative_evaluations += 1
                    if result.state == "statistical":
                        group_states[group_slot] = 2
                        group_scores[group_slot] = float(result.score)
                    elif result.state == "abstain":
                        group_states[group_slot] = 3
                    elif result.state == "hard_pass":
                        group_states[group_slot] = 1
                        group_scores[group_slot] = 1.0
                    else:
                        group_states[group_slot] = 0
                states[template_rows] = group_states[inverse]
                scores[template_rows] = group_scores[inverse]
                equivalence_groups += int(len(unique_groups))

        audit = {
            "path": "vectorized_hard_grouped_statistical_v1",
            "rows": int(n_rows),
            "hard_pass": int(np.count_nonzero(states == 1)),
            "hard_fail": int(np.count_nonzero(states == 0)),
            "statistical": int(np.count_nonzero(states == 2)),
            "abstain": int(np.count_nonzero(states == 3)),
            "representative_evaluations": int(representative_evaluations),
            "equivalence_groups": int(equivalence_groups),
        }
        return states, scores, audit

    def materialize_eligibility(self, predicate):
        return np.asarray([
            self.evaluate(row, predicate, record=False).eligible
            for row in range(len(self.target_table))
        ], dtype=np.bool_)


@dataclass(frozen=True)
class ValidationInstance:
    tuple_id: int
    predicate: Predicate
    label: int
    fold: int


def fit_cross_fitted_score_maps(
        policy, inference, instances: Iterable[ValidationInstance], *,
        min_map_instances=2, min_template_coverage=0.8,
        min_shape_coverage=0.8):
    """Fit per-template maps without evaluating an instance on its own fold.

    Returns a deployment policy refitted on all resolved validation instances and
    an audit list of out-of-fold predictions.
    """
    instances = list(instances)
    min_map_instances = int(min_map_instances)
    if min_map_instances < 2:
        raise ValueError("min_map_instances must be at least two")
    if not 0 <= min_template_coverage <= 1:
        raise ValueError("min_template_coverage must be in [0, 1]")
    if not 0 <= min_shape_coverage <= 1:
        raise ValueError("min_shape_coverage must be in [0, 1]")

    evaluated = []
    raw = []
    for instance_index, instance in enumerate(instances):
        _, residual = inference.hard_decision(instance.tuple_id, instance.predicate)
        shape = repr((
            tuple(sorted((atom.attr, atom.op) for atom in instance.predicate.atoms)),
            tuple(sorted(atom.attr for atom in residual)),
        ))
        result = inference.evaluate(
            instance.tuple_id, instance.predicate, record=False)
        evaluated.append((instance_index, instance, result, shape))
        if result.state == "statistical":
            raw.append((instance_index, instance, result.template_id,
                        result.raw_score, shape))

    out_of_fold = []
    for fold in sorted({item.fold for _, item, _, _, _ in raw}):
        for template in policy.templates:
            train = [(index, item, value, shape)
                     for index, item, identifier, value, shape in raw
                     if identifier == template.template_id and item.fold != fold]
            heldout = [(index, item, value, shape)
                       for index, item, identifier, value, shape in raw
                       if identifier == template.template_id and item.fold == fold]
            if not train or not heldout:
                continue
            score_map = MonotoneScoreMap.fit(
                [value for _, _, value, _ in train],
                [item.label for _, item, _, _ in train])
            for index, item, value, shape in heldout:
                out_of_fold.append({
                    "instance_index": int(index),
                    "tuple_id": int(item.tuple_id), "fold": int(item.fold),
                    "template_id": template.template_id,
                    "shape": shape,
                    "raw_score": float(value),
                    "score": float(score_map.predict(value)),
                    "label": int(item.label),
                })
    final_maps = {}
    retained_templates = []
    template_audit = {}
    for template in policy.templates:
        values = [(item, value) for _, item, identifier, value, _ in raw
                  if identifier == template.template_id]
        assigned = sum(result.template_id == template.template_id
                       for _, _, result, _ in evaluated)
        oof_count = sum(item["template_id"] == template.template_id
                        for item in out_of_fold)
        folds = sorted({int(item.fold) for item, _ in values})
        coverage = oof_count / assigned if assigned else 0.0
        deployable = (len(values) >= min_map_instances and
                      len(folds) >= 2 and oof_count >= min_map_instances and
                      coverage >= min_template_coverage)
        template_audit[template.template_id] = {
            "assigned_instances": int(assigned),
            "resolved_instances": int(len(values)),
            "folds": folds,
            "out_of_fold_predictions": int(oof_count),
            "out_of_fold_coverage": float(coverage),
            "deployable": bool(deployable),
        }
        if not deployable:
            continue
        retained_templates.append(template)
        final_maps[template.template_id] = MonotoneScoreMap.fit(
            [value for _, value in values], [item.label for item, _ in values])
    retained_ids = {template.template_id for template in retained_templates}
    template_requirements_admissible = all(
        item["assigned_instances"] == 0 or item["deployable"]
        for item in template_audit.values())
    resolved_indices = {
        int(item["instance_index"]) for item in out_of_fold
        if item["template_id"] in retained_ids
    }
    shape_audit = {}
    for shape in sorted({shape for _, _, _, shape in evaluated}):
        indices = [index for index, _, _, item_shape in evaluated
                   if item_shape == shape]
        resolved = sum(index in resolved_indices for index in indices)
        coverage = resolved / len(indices)
        shape_audit[shape] = {
            "instances": int(len(indices)),
            "out_of_fold_predictions": int(resolved),
            "out_of_fold_coverage": float(coverage),
            "admissible": bool(coverage >= min_shape_coverage),
        }
    policy_admissible = (bool(retained_templates) and
                         template_requirements_admissible and
                         all(item["admissible"] for item in shape_audit.values()))
    manifest = {
        "cross_fitted": True,
        "folds": sorted({int(item.fold) for _, item, _, _, _ in raw}),
        "resolved_instances": int(len(raw)),
        "out_of_fold_predictions": int(len(out_of_fold)),
        "final_refit": True,
        "min_map_instances": min_map_instances,
        "min_template_coverage": float(min_template_coverage),
        "min_shape_coverage": float(min_shape_coverage),
        "retained_templates": int(len(retained_templates)),
        "discarded_templates": int(len(policy.templates) - len(retained_templates)),
        "template_audit": template_audit,
        "template_requirements_admissible": template_requirements_admissible,
        "shape_audit": shape_audit,
        "policy_admissible": policy_admissible,
    }
    return DeploymentPolicy(tuple(retained_templates), final_maps, manifest), out_of_fold
