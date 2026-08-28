"""Validation-only joint structure/policy selection and grouped calibration."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import time

import numpy as np

from mifa.index.projected_tree import ProjectedPartitionTree
from mifa.inference.paper_policy import (
    BodyAttribute,
    DeploymentPolicy,
    IndexedJointInference,
    MatchLevel,
    SupportTemplate,
    ValidationInstance,
    fit_cross_fitted_score_maps,
)
from mifa.inference.rules import discover_rules
from mifa.inference.support_index import SharedSupportIndex


@dataclass(frozen=True)
class StructuralConfig:
    projection_dim: int
    leaf_capacity: int

    @property
    def key(self):
        return f"p{self.projection_dim}_leaf{self.leaf_capacity}"


@dataclass(frozen=True)
class PolicyCandidate:
    candidate_id: str
    policy: DeploymentPolicy


@dataclass
class DesignSelectionResult:
    structural_config: StructuralConfig
    policy: DeploymentPolicy
    candidate_policy: DeploymentPolicy
    candidate_id: str
    rows: list[dict]


def default_body_attribute(attr, dtype, support_values,
                           numeric_fractions=(0.05, 0.15, 0.30)):
    """Create a finite, predeclared monotone scalar matching ladder."""
    if dtype != "num":
        return BodyAttribute(
            attr, (MatchLevel.equality(), MatchLevel.unrestricted()))
    values = np.asarray(support_values, dtype=float)
    values = values[np.isfinite(values)]
    span = 0.0 if not len(values) else float(
        np.quantile(values, 0.95) - np.quantile(values, 0.05))
    radii = sorted(set(float(fraction) * span for fraction in numeric_fractions))
    levels = tuple(MatchLevel.numeric_radius(radius) for radius in radii)
    return BodyAttribute(attr, levels + (MatchLevel.unrestricted(),))


def discover_scalar_body_candidates(spec, training_observed, residual_target_sets,
                                    *, max_bodies_per_target_set=4,
                                    dep_conf=0.20, min_support=30):
    """Discover a finite scalar-body family from training evidence only."""
    target_sets = sorted({tuple(sorted(set(targets)))
                          for targets in residual_target_sets if targets})
    all_targets = sorted({attr for targets in target_sets for attr in targets})
    rules = discover_rules(
        spec, training_observed, cell_of=None, targets=all_targets,
        dep_conf=dep_conf, min_support=min_support, coarse_to_fine=True)
    output = {}
    for targets in target_sets:
        candidates = {}
        best_by_target = []
        for target in targets:
            usable = []
            for rule in rules.get(target, ()):  # location is a separate template flag
                body = tuple(sorted(attr for attr in rule.X
                                    if attr != "cell" and attr not in targets))
                if not body:
                    continue
                score = float(rule.confidence * np.log1p(rule.support) / len(body))
                candidates[body] = max(candidates.get(body, -np.inf), score)
                usable.append((score, body))
            if usable:
                best_by_target.append(max(usable)[1])
        if len(best_by_target) == len(targets) and best_by_target:
            combined = tuple(sorted(set().union(*map(set, best_by_target)) - set(targets)))
            if combined:
                candidates[combined] = max(
                    candidates.get(combined, -np.inf),
                    float(np.mean([candidates[body] for body in best_by_target])))
        ranked = sorted(candidates, key=lambda body: (-candidates[body], len(body), body))
        output[targets] = ranked[:int(max_bodies_per_target_set)]
    return output


def build_policy_candidate_family(spec, training_observed, residual_target_sets,
                                  body_candidates, *, support_values=(10, 30, 50),
                                  shrinkage_values=(0.0, 10.0, 30.0)):
    """Build a finite family of validation-selectable ordered policies.

    The paper defines a candidate policy as an *ordered subset* of the
    training-discovered templates.  Keeping only the all-body policy makes the
    training ranking effectively final: whenever body 0 resolves an instance,
    lower-ranked bodies can never be selected or evaluated by validation.  In
    addition to the four mechanism families, expose one predeclared policy per
    body rank whenever multiple bodies exist.  Validation can then reject a
    misleading training-ranked body without consulting calibration or test.

    Rank policies share the same rank across target shapes.  This keeps the
    candidate set linear in the maximum number of discovered bodies rather
    than taking an exponential Cartesian product across predicate shapes.
    """
    target_sets = sorted({tuple(sorted(set(targets)))
                          for targets in residual_target_sets if targets})
    n_body_ranks = max(
        (len(body_candidates.get(targets, ())) for targets in target_sets),
        default=0)
    candidates = []
    for min_support in support_values:
        for kappa in shrinkage_values:
            global_templates = []
            location_templates = []
            scalar_templates = []
            full_templates = []
            scalar_rank_templates = {rank: [] for rank in range(n_body_ranks)}
            full_rank_templates = {rank: [] for rank in range(n_body_ranks)}
            priority = 0
            for targets in target_sets:
                suffix = "_".join(targets)
                bodies = []
                for body_names in body_candidates.get(targets, ()):
                    body = tuple(default_body_attribute(
                        attr, spec.dtypes[attr], training_observed[attr].to_numpy())
                        for attr in body_names)
                    bodies.append(body)
                for body_index, body in enumerate(bodies):
                    loc = SupportTemplate(
                        f"loc_body{body_index}_{suffix}_s{min_support}_k{kappa:g}",
                        targets, body, True, min_support, kappa, priority)
                    scalar = SupportTemplate(
                        f"scalar_body{body_index}_{suffix}_s{min_support}_k{kappa:g}",
                        targets, body, False, min_support, kappa, priority)
                    scalar_templates.append(scalar)
                    full_templates.append(loc)
                    scalar_rank_templates[body_index].append(scalar)
                    full_rank_templates[body_index].append(loc)
                    priority += 1
                loc_free = SupportTemplate(
                    f"loc_free_{suffix}_s{min_support}_k{kappa:g}",
                    targets, (), True, min_support, kappa, priority)
                global_free = SupportTemplate(
                    f"global_{suffix}_s{min_support}_k{kappa:g}",
                    targets, (), False, min_support, kappa, priority + 1)
                location_templates.extend((loc_free, global_free))
                scalar_templates.append(global_free)
                full_templates.extend((loc_free, global_free))
                global_templates.append(global_free)
                for body_rank in range(n_body_ranks):
                    # A target shape without this rank still retains the same
                    # location/global fallback ladder and therefore remains
                    # admissible under the declared coverage checks.
                    scalar_rank_templates[body_rank].append(global_free)
                    full_rank_templates[body_rank].extend(
                        (loc_free, global_free))
                priority += 2
            tag = f"s{min_support}_k{kappa:g}"
            candidates.extend((
                PolicyCandidate(f"global_{tag}", DeploymentPolicy(tuple(global_templates))),
                PolicyCandidate(f"location_{tag}", DeploymentPolicy(tuple(location_templates))),
                PolicyCandidate(f"scalar_{tag}", DeploymentPolicy(tuple(scalar_templates))),
                PolicyCandidate(f"full_{tag}", DeploymentPolicy(tuple(full_templates))),
            ))
            # With a single discovered body these candidates would duplicate
            # the existing scalar/full policies.  Add them only when validation
            # has a genuine ordered-subset choice to make.
            if n_body_ranks > 1:
                for body_rank in range(n_body_ranks):
                    candidates.extend((
                        PolicyCandidate(
                            f"scalar_body{body_rank}_{tag}",
                            DeploymentPolicy(tuple(
                                scalar_rank_templates[body_rank]))),
                        PolicyCandidate(
                            f"full_body{body_rank}_{tag}",
                            DeploymentPolicy(tuple(
                                full_rank_templates[body_rank]))),
                    ))
    return candidates


class OfflineDesignSelector:
    """Replay an explicit finite `(Gamma, policy)` family on validation only."""

    def __init__(self, training_embeddings, training_table, dtypes,
                 validation_embeddings, validation_table, validation_instances,
                 *, lambda_time=0.0, min_coverage=0.8,
                 min_template_coverage=0.8, min_shape_coverage=0.8,
                 min_map_instances=2, seed=0, progress_callback=None,
                 candidate_workers=1, objective_tolerance=0.0):
        self.training_embeddings = np.ascontiguousarray(
            training_embeddings, dtype=np.float32)
        self.training_table = training_table.reset_index(drop=True)
        self.dtypes = dict(dtypes)
        self.validation_embeddings = np.ascontiguousarray(
            validation_embeddings, dtype=np.float32)
        self.validation_table = validation_table.reset_index(drop=True)
        self.validation_instances = tuple(validation_instances)
        self.lambda_time = float(lambda_time)
        self.min_coverage = float(min_coverage)
        self.min_template_coverage = float(min_template_coverage)
        self.min_shape_coverage = float(min_shape_coverage)
        self.min_map_instances = int(min_map_instances)
        self.seed = int(seed)
        self.progress_callback = progress_callback
        self.candidate_workers = int(candidate_workers)
        self.objective_tolerance = float(objective_tolerance)
        if not self.validation_instances:
            raise ValueError("validation selection needs labeled instances")
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be in [0, 1]")
        if not 0 <= self.min_template_coverage <= 1:
            raise ValueError("min_template_coverage must be in [0, 1]")
        if not 0 <= self.min_shape_coverage <= 1:
            raise ValueError("min_shape_coverage must be in [0, 1]")
        if self.min_map_instances < 2:
            raise ValueError("min_map_instances must be at least two")
        if self.candidate_workers < 1:
            raise ValueError("candidate_workers must be positive")
        if self.objective_tolerance < 0:
            raise ValueError("objective_tolerance must be non-negative")

    def _emit_progress(self, stage, **details):
        if self.progress_callback is not None:
            self.progress_callback(stage, **details)

    def _build_temporary_structures(self, config):
        """Build the candidate-independent structures once per Gamma.

        The paper's validation protocol shares one temporary hierarchy and
        support index across every policy candidate evaluated under the same
        structural configuration.  Rebuilding them once per policy is both
        redundant and substantially slower on real datasets.
        """
        begin = time.perf_counter()
        tree = ProjectedPartitionTree(
            config.projection_dim, config.leaf_capacity, self.seed)
        # The split tests are learned from D_tr only.  Validation tuples are
        # routed into that frozen hierarchy and never influence its topology.
        tree.fit_structure(
            self.training_embeddings,
            topology_source="training_embeddings")
        tree.materialize(self.validation_embeddings)
        index = SharedSupportIndex(
            self.training_table, self.dtypes,
            tree.route(self.training_embeddings), tree)
        return tree, index, float(time.perf_counter() - begin)

    def _replay(self, config, candidate, *, tree=None, index=None,
                structure_build_seconds=0.0):
        if tree is None or index is None:
            tree, index, structure_build_seconds = \
                self._build_temporary_structures(config)
        stats_before = dict(index.stats)
        begin = time.perf_counter()
        inference = IndexedJointInference(
            self.validation_table, self.validation_embeddings, tree, index,
            candidate.policy, theta=0.5)
        fitted, out_of_fold = fit_cross_fitted_score_maps(
            candidate.policy, inference, self.validation_instances,
            min_map_instances=self.min_map_instances,
            min_template_coverage=self.min_template_coverage,
            min_shape_coverage=self.min_shape_coverage)
        retained_ids = {template.template_id for template in fitted.templates}
        resolved = sum(item["template_id"] in retained_ids
                       for item in out_of_fold)
        coverage = resolved / len(self.validation_instances)
        predictions = np.zeros(len(self.validation_instances), dtype=np.float64)
        for item in out_of_fold:
            if item["template_id"] in retained_ids:
                predictions[int(item["instance_index"])] = float(item["score"])
        labels = np.asarray(
            [item.label for item in self.validation_instances], dtype=np.float64)
        # The paper assigns inference-stage abstention score zero for InfLoss;
        # it does not condition Brier loss on successful resolution.
        brier = float(np.mean((predictions - labels) ** 2))
        inf_score = 1.0 - brier
        operation_cost = sum(
            index.stats[key] - stats_before[key]
            for key in ("candidate_rows", "postings", "intersections"))
        elapsed = time.perf_counter() - begin
        return {
            "structural_config": config,
            "candidate": candidate,
            "policy": fitted,
            "coverage": float(coverage),
            "brier": brier,
            "inf_score": inf_score,
            "operation_cost": int(operation_cost),
            "wall_seconds_diagnostic": float(elapsed),
            "shared_structure_build_seconds": float(structure_build_seconds),
            "out_of_fold_predictions": out_of_fold,
            "admissible": (
                coverage >= self.min_coverage and bool(out_of_fold) and
                bool(fitted.fit_manifest["policy_admissible"])),
            "tree_manifest": tree.manifest(),
            "support_index_manifest": index.manifest(),
        }

    def select(self, structural_configs, policy_candidates):
        configs = tuple(structural_configs)
        candidates = tuple(policy_candidates)
        if not configs or not candidates:
            raise ValueError("the candidate family cannot be empty")
        replays = []
        total = len(configs) * len(candidates)
        completed = 0
        for config in configs:
            tree, index, build_seconds = self._build_temporary_structures(config)
            self._emit_progress(
                "validation_structure_ready", structure=config.key,
                build_seconds=build_seconds, policies=len(candidates))
            def replay_candidate(candidate):
                # Postings and tree arrays are immutable after construction.
                # The lazily populated support masks are not: a shallow copy
                # used to let candidate workers clear and mutate the same
                # dictionaries, causing missing keys and iteration races.
                # Keep the large immutable arrays shared, but fork every
                # mutable runtime cache and the counters per worker.
                worker_index = copy.copy(index)
                for cache_name in (
                        "_target_availability_masks",
                        "_body_mask_cache",
                        "_body_mask_packed_cache",
                        "_body_mask_denominator_cache"):
                    setattr(
                        worker_index, cache_name,
                        dict(getattr(index, cache_name)))
                worker_index.stats = {key: 0 for key in index.stats}
                return self._replay(
                    config, candidate, tree=tree, index=worker_index,
                    structure_build_seconds=build_seconds)

            if self.candidate_workers == 1:
                config_rows = map(replay_candidate, candidates)
                executor = None
            else:
                executor = ThreadPoolExecutor(max_workers=self.candidate_workers)
                config_rows = executor.map(replay_candidate, candidates)
            try:
                for candidate, row in zip(candidates, config_rows):
                    replays.append(row)
                    completed += 1
                    self._emit_progress(
                        "validation_candidate_replayed", completed=completed,
                        total=total, structure=config.key,
                        policy=candidate.candidate_id,
                        seconds=row["wall_seconds_diagnostic"],
                        admissible=row["admissible"],
                        inf_score=row["inf_score"])
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
        finite_costs = [row["operation_cost"] for row in replays
                        if row["admissible"]]
        if not finite_costs:
            raise ValueError("no admissible structure-policy candidate")
        scale = max(max(finite_costs), 1)
        for row in replays:
            normalized_cost = row["operation_cost"] / scale
            row["normalized_time_cost"] = float(normalized_cost)
            row["objective"] = (
                row["inf_score"] - self.lambda_time * normalized_cost
                if row["admissible"] else -float("inf"))
        admissible_rows = [row for row in replays if row["admissible"]]
        best_objective = max(row["objective"] for row in admissible_rows)
        stable_rows = [
            row for row in admissible_rows
            if row["objective"] >=
            best_objective - self.objective_tolerance - 1e-12]
        if self.objective_tolerance == 0.0:
            selected = min(stable_rows, key=lambda row: (
                -row["objective"],
                row["structural_config"].projection_dim,
                row["structural_config"].leaf_capacity,
                row["candidate"].candidate_id))
        else:
            # Avoid a finely split hierarchy for a numerically negligible
            # validation improvement.  This is a validation-only stability
            # rule and also reduces the deployed structure footprint.
            selected = min(stable_rows, key=lambda row: (
                -row["structural_config"].leaf_capacity,
                row["structural_config"].projection_dim,
                -row["objective"], row["candidate"].candidate_id))
        audit_rows = []
        for row in replays:
            audit_rows.append({
                "structural_config": row["structural_config"].key,
                "candidate_id": row["candidate"].candidate_id,
                "coverage": row["coverage"], "brier": row["brier"],
                "inf_score": row["inf_score"],
                "operation_cost": row["operation_cost"],
                "normalized_time_cost": row["normalized_time_cost"],
                "objective": row["objective"],
                "admissible": row["admissible"],
                "wall_seconds_diagnostic": row["wall_seconds_diagnostic"],
                "shared_structure_build_seconds":
                    row["shared_structure_build_seconds"],
                "selected": row is selected,
            })
        selected["policy"].fit_manifest.update({
            "joint_design_selection": True,
            "selection_source": "validation_only",
            "selection_objective": (
                "one_minus_brier" if self.lambda_time == 0.0 else
                "one_minus_brier_minus_lambda_normalized_support_work_proxy"),
            "lambda_time": self.lambda_time,
            "objective_tolerance": self.objective_tolerance,
            "stable_candidate_count": len(stable_rows),
            "near_optimal_structure_tiebreak": (
                "coarser_leaf_then_smaller_projection"
                if self.objective_tolerance > 0 else "disabled"),
            "paper_exact_executor_replay_timecost": False,
            "selected_structure": selected["structural_config"].key,
            "selected_candidate": selected["candidate"].candidate_id,
        })
        return DesignSelectionResult(
            selected["structural_config"], selected["policy"],
            selected["candidate"].policy,
            selected["candidate"].candidate_id, audit_rows)


def refit_deployment_policy(
        training_embeddings, training_table, dtypes,
        validation_embeddings, validation_table, validation_instances,
        structural_config, candidate_policy, *, topology_embeddings=None,
        min_map_instances=2,
        min_template_coverage=0.8, min_shape_coverage=0.8, seed=0,
        support_mode="joint", context_mode="hierarchical"):
    """Refit selected ``c*`` after the D_tr-only hierarchy is fixed.

    The temporary validation hierarchy is selection-only.  Section 3.2
    requires learning every projection and split test from D_tr only, routing
    validation instances through that frozen topology, and fitting the
    deployed score maps before calibration.  ``topology_embeddings`` remains
    as a compatibility guard: when supplied, it must be exactly D_tr.
    """
    training_embeddings = np.ascontiguousarray(
        training_embeddings, dtype=np.float32)
    validation_embeddings = np.ascontiguousarray(
        validation_embeddings, dtype=np.float32)
    if topology_embeddings is not None:
        supplied_topology = np.ascontiguousarray(
            topology_embeddings, dtype=np.float32)
        if supplied_topology.shape != training_embeddings.shape or not \
                np.array_equal(supplied_topology, training_embeddings):
            raise ValueError(
                "topology_embeddings must equal training_embeddings; "
                "validation/searchable embeddings cannot fit topology")
    tree = ProjectedPartitionTree(
        structural_config.projection_dim,
        structural_config.leaf_capacity,
        seed)
    tree.fit_structure(
        training_embeddings,
        topology_source="training_embeddings")
    tree.materialize(validation_embeddings)
    index = SharedSupportIndex(
        training_table.reset_index(drop=True), dtypes,
        tree.route(training_embeddings), tree)
    inference = IndexedJointInference(
        validation_table.reset_index(drop=True), validation_embeddings,
        tree, index, candidate_policy, theta=0.5,
        support_mode=support_mode, context_mode=context_mode)
    deployed, _ = fit_cross_fitted_score_maps(
        candidate_policy, inference, validation_instances,
        min_map_instances=min_map_instances,
        min_template_coverage=min_template_coverage,
        min_shape_coverage=min_shape_coverage)
    if not deployed.fit_manifest["policy_admissible"]:
        raise ValueError(
            "selected candidate is not deployable under the final "
            "searchable hierarchy")
    deployed.fit_manifest.update({
        "joint_design_selection": True,
        "selection_source": "validation_only",
        "selected_structure": structural_config.key,
        "final_structure_refit": True,
        "final_structure_source": "training_embeddings",
        "final_refit_support_source": "training_only",
        "final_refit_instance_source": "validation_only",
        "support_mode": support_mode,
        "context_mode": context_mode,
    })
    return deployed, {
        "tree": tree.manifest(),
        "support_index": index.manifest(),
    }


def _select_calibration_row(curve, loss_tolerance=0.0):
    """Pick a precision-stable point inside a near-optimal top-k band."""
    tolerance = float(loss_tolerance)
    if tolerance < 0:
        raise ValueError("loss_tolerance must be non-negative")
    best_loss = min(row["group_macro_loss"] for row in curve)
    near_optimal = [
        row for row in curve
        if row["group_macro_loss"] <= best_loss + tolerance + 1e-12]
    return min(near_optimal, key=lambda row: (
        -float(row.get("eligibility_f1", -1.0)),
        row["group_macro_loss"], -row["theta"]))


def calibrate_threshold_grouped(inference, queries, truth_ids, group_ids,
                                theta_candidates=None, progress_callback=None,
                                eligibility_truth_masks=None, *,
                                loss_tolerance=0.0,
                                use_optimized_materialization=True,
                                return_timing=False):
    """Select only theta using query loss averaged over independent groups."""
    total_started = time.perf_counter()
    queries = list(queries)
    truth_ids = list(truth_ids)
    groups = np.asarray(group_ids)
    if len(queries) != len(truth_ids) or groups.shape != (len(queries),):
        raise ValueError("queries, truth IDs, and groups must align")
    if eligibility_truth_masks is not None:
        eligibility_truth_masks = [
            np.asarray(mask, dtype=np.bool_)
            for mask in eligibility_truth_masks]
        if len(eligibility_truth_masks) != len(queries) or any(
                mask.shape != (len(inference.target_table),)
                for mask in eligibility_truth_masks):
            raise ValueError(
                "eligibility truth masks must align with queries and rows")
    theta_candidates = (np.linspace(0.0, 1.0, 101) if theta_candidates is None
                        else np.asarray(theta_candidates, dtype=float))
    n_rows = len(inference.target_table)
    prepared = []
    materialization_started = time.perf_counter()
    inference_seconds = 0.0
    distance_seconds = 0.0
    distance_order_seconds = 0.0
    materialized_cache = {}
    materialization_audits = []
    aggregate_states = {
        "hard_pass": 0, "hard_fail": 0, "statistical": 0, "abstain": 0}
    fast_queries = 0
    fallback_queries = 0
    representative_evaluations = 0
    materialized_cache_resident_bytes = 0
    prepared_query_array_bytes = 0
    for query_index, (query, truth) in enumerate(zip(queries, truth_ids), start=1):
        predicate = query["phi"]
        predicate_key = (
            inference._predicate_key(predicate.atoms)
            if hasattr(inference, "_predicate_key") else None)
        cached = materialized_cache.get(predicate_key)
        if cached is None:
            inference_started = time.perf_counter()
            materialized = None
            if (use_optimized_materialization and
                    hasattr(inference, "materialize_calibration_policy")):
                materialized = inference.materialize_calibration_policy(
                    predicate)
            if materialized is None:
                states = np.zeros(n_rows, dtype=np.int8)
                scores = np.zeros(n_rows, dtype=np.float64)
                for row in range(n_rows):
                    result = inference.evaluate(
                        row, predicate, record=False)
                    if result.state == "hard_pass":
                        states[row] = 1
                    elif result.state == "statistical":
                        states[row] = 2
                        scores[row] = result.score
                    elif result.state == "abstain":
                        states[row] = 3
                audit = {
                    "path": "scalar_evaluate_fallback",
                    "rows": int(n_rows),
                    "hard_pass": int(np.count_nonzero(states == 1)),
                    "hard_fail": int(np.count_nonzero(states == 0)),
                    "statistical": int(np.count_nonzero(states == 2)),
                    "abstain": int(np.count_nonzero(states == 3)),
                    "representative_evaluations": int(n_rows),
                }
                fallback_queries += 1
            else:
                states, scores, audit = materialized
                fast_queries += 1
            inference_seconds += time.perf_counter() - inference_started
            states = np.ascontiguousarray(states, dtype=np.int8)
            scores = np.ascontiguousarray(scores, dtype=np.float64)
            states.setflags(write=False)
            scores.setflags(write=False)
            cached = (states, scores, audit)
            materialized_cache_resident_bytes += int(
                states.nbytes + scores.nbytes)
            if predicate_key is not None:
                materialized_cache[predicate_key] = cached
        else:
            states, scores, original_audit = cached
            audit = {**original_audit, "exact_predicate_cache_hit": True}
        states, scores, _ = cached
        materialization_audits.append(audit)
        representative_evaluations += int(
            audit.get("representative_evaluations", 0)
            if not audit.get("exact_predicate_cache_hit") else 0)
        aggregate_states["hard_pass"] += int(np.count_nonzero(states == 1))
        aggregate_states["hard_fail"] += int(np.count_nonzero(states == 0))
        aggregate_states["statistical"] += int(np.count_nonzero(states == 2))
        aggregate_states["abstain"] += int(np.count_nonzero(states == 3))
        hard = states == 1
        statistical = states == 2

        distance_started = time.perf_counter()
        vector = np.asarray(query["q_e"], dtype=np.float32)
        delta = inference.target_embeddings - vector
        d2 = np.einsum(
            "ij,ij->i", delta, delta)
        distance_seconds += time.perf_counter() - distance_started
        order_started = time.perf_counter()
        distance_order = np.lexsort((
            np.arange(n_rows, dtype=np.int64), d2))
        distance_order_seconds += time.perf_counter() - order_started
        prepared.append((hard, statistical, scores, distance_order,
                         np.asarray(truth, dtype=np.int64), int(query["k"])))
        prepared_query_array_bytes += int(
            hard.nbytes + statistical.nbytes + distance_order.nbytes)
        if progress_callback is not None and (
                query_index == len(queries) or query_index % 10 == 0):
            progress_callback(
                "calibration_policy_materialized", completed=query_index,
                total=len(queries))

    # The calibration model is distinct from the test model.  Still clear the
    # explicit offline-cache opt-in so callers that retain this object never
    # observe a prewarmed online query.
    if (use_optimized_materialization and
            hasattr(inference, "clear_all_serving_caches")):
        inference.clear_all_serving_caches()
    materialization_seconds = time.perf_counter() - materialization_started

    theta_sweep_started = time.perf_counter()
    curve = []
    for theta in theta_candidates:
        query_losses = np.empty(len(queries), dtype=np.float64)
        eligibility_tp = eligibility_fp = eligibility_fn = eligibility_tn = 0
        for index, (hard, statistical, scores, distance_order,
                    truth, k) in enumerate(prepared):
            eligible = hard | (statistical & (scores >= theta))
            if eligibility_truth_masks is not None:
                eligibility_truth = eligibility_truth_masks[index]
                eligibility_tp += int(np.count_nonzero(
                    eligible & eligibility_truth))
                eligibility_fp += int(np.count_nonzero(
                    eligible & ~eligibility_truth))
                eligibility_fn += int(np.count_nonzero(
                    ~eligible & eligibility_truth))
                eligibility_tn += int(np.count_nonzero(
                    ~eligible & ~eligibility_truth))
            returned = np.asarray(
                distance_order[eligible[distance_order]][:k],
                dtype=np.int64)
            expected = set(int(value) for value in truth if value >= 0)
            actual = set(int(value) for value in returned)
            overlap = len(expected & actual)
            precision = overlap / len(actual) if actual else (1.0 if not expected else 0.0)
            recall = overlap / len(expected) if expected else 1.0
            f1 = (0.0 if precision + recall == 0 else
                  2.0 * precision * recall / (precision + recall))
            query_losses[index] = 1.0 - f1
        group_losses = [float(query_losses[groups == group].mean())
                        for group in np.unique(groups)]
        row = {
            "theta": float(theta),
            "group_macro_loss": float(np.mean(group_losses)),
            "group_macro_f1": float(1.0 - np.mean(group_losses)),
            "independent_groups": int(len(group_losses)),
        }
        if eligibility_truth_masks is not None:
            precision_denom = eligibility_tp + eligibility_fp
            recall_denom = eligibility_tp + eligibility_fn
            far_denom = eligibility_fp + eligibility_tn
            eligibility_precision = (
                eligibility_tp / precision_denom if precision_denom else 1.0)
            eligibility_recall = (
                eligibility_tp / recall_denom if recall_denom else 1.0)
            row.update({
                "eligibility_precision": float(eligibility_precision),
                "eligibility_recall": float(eligibility_recall),
                "eligibility_f1": float(
                    0.0 if eligibility_precision + eligibility_recall == 0
                    else 2.0 * eligibility_precision * eligibility_recall /
                    (eligibility_precision + eligibility_recall)),
                "false_accept_rate": float(
                    eligibility_fp / far_denom if far_denom else 0.0),
                "false_reject_rate": float(
                    eligibility_fn / recall_denom if recall_denom else 0.0),
                "eligibility_counts": {
                    "tp": eligibility_tp, "fp": eligibility_fp,
                    "fn": eligibility_fn, "tn": eligibility_tn,
                },
            })
        curve.append(row)
    theta_sweep_seconds = time.perf_counter() - theta_sweep_started
    # Equal/near-equal top-k plateaus are common because nearby thresholds
    # induce the same result set.  Inside the declared calibration-only loss
    # band, prefer the point with the strongest tuple-eligibility F1.  This
    # reduces brittle false accepts/rejects without consulting test data.
    selected = _select_calibration_row(curve, loss_tolerance)
    total_seconds = time.perf_counter() - total_started
    state_total = max(1, int(sum(aggregate_states.values())))
    timing = {
        "offline_calibration_materialization_seconds": float(
            materialization_seconds),
        "offline_theta_sweep_seconds": float(theta_sweep_seconds),
        "offline_calibration_total_seconds": float(total_seconds),
        "calibration_loss_tolerance": float(loss_tolerance),
        "selection_rule": "near_optimal_topk_then_eligibility_f1",
        "excluded_from_online_e2e": True,
        "online_timer_fields_unchanged": True,
        "materialization_breakdown_seconds": {
            "inference_evaluate_or_grouped": float(inference_seconds),
            "distance_rows": float(distance_seconds),
            "distance_order_once_per_query": float(distance_order_seconds),
            "other": float(max(
                0.0, materialization_seconds - inference_seconds -
                distance_seconds - distance_order_seconds)),
        },
        "materialization_path": (
            "vectorized_grouped_with_scalar_fallback"
            if use_optimized_materialization else "scalar_reference"),
        "fast_queries": int(fast_queries),
        "fallback_queries": int(fallback_queries),
        "exact_predicate_cache_hits": int(sum(
            bool(item.get("exact_predicate_cache_hit"))
            for item in materialization_audits)),
        "representative_evaluations": int(representative_evaluations),
        "offline_materialized_cache_resident_bytes": int(
            materialized_cache_resident_bytes),
        "offline_prepared_query_array_bytes": int(
            prepared_query_array_bytes),
        "offline_owned_array_bytes": int(
            materialized_cache_resident_bytes + prepared_query_array_bytes),
        "state_counts": aggregate_states,
        "state_fractions": {
            key: float(value / state_total)
            for key, value in aggregate_states.items()},
        "query_audits": materialization_audits,
    }
    if return_timing:
        return float(selected["theta"]), curve, timing
    return float(selected["theta"]), curve
