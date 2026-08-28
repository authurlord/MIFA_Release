import numpy as np
import pandas as pd

from mifa.index.projected_tree import ProjectedPartitionTree
from mifa.inference.paper_policy import (
    BodyAttribute,
    CertifiedDomain,
    DeploymentPolicy,
    IndexedJointInference,
    MatchLevel,
    MonotoneScoreMap,
    SupportTemplate,
)
from mifa.inference.support_index import SharedSupportIndex
from mifa.predicates import Atom, Predicate


def _fixture():
    training_x = np.asarray([
        [-3.0, 0.0], [-2.8, 0.1], [-2.5, -0.1], [-2.2, 0.0],
        [2.2, 0.0], [2.5, 0.1], [2.8, -0.1], [3.0, 0.0],
    ], dtype=np.float32)
    support = pd.DataFrame({
        "body": ["a", "a", "a", "b", "a", "a", "b", "b"],
        "y": [1, 1, 0, 0, 1, 0, 0, 0],
    })
    target = support.copy()
    target.loc[0, "y"] = np.nan
    tree = ProjectedPartitionTree(
        projection_dim=2, leaf_capacity=2, seed=0).fit_structure(training_x)
    tree.materialize(training_x)
    support_leaves = tree.route(training_x)
    index = SharedSupportIndex(
        support, {"body": "cat", "y": "cat"}, support_leaves, tree)
    return training_x, support, target, tree, index


def test_shared_support_retrieval_is_nested_across_ancestor_and_scalar_backoff():
    x, _, target, tree, index = _fixture()
    body = BodyAttribute(
        "body", (MatchLevel.equality(), MatchLevel.unrestricted()))
    template = SupportTemplate(
        "body_loc_y", ("y",), (body,), location_aware=True,
        min_support=1, kappa=0.0)
    row = target.iloc[0]
    path = tree.ancestor_path(x[0], 0, fine_to_broad=True)
    support_sets = [set(index.retrieve(
        template, row, node_id=node, level_by_attr={"body": 0}))
        for node in path]
    for fine, broad in zip(support_sets, support_sets[1:]):
        assert fine <= broad
    all_body = set(index.retrieve(
        template, row, node_id=tree.root_id, level_by_attr={"body": 1}))
    assert support_sets[-1] <= all_body
    assert all_body == set(range(len(target)))


def test_isotonic_score_map_is_monotone_and_pools_violations():
    score_map = MonotoneScoreMap.fit(
        [0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1])
    grid = np.linspace(0, 1, 101)
    predicted = score_map.predict(grid)
    assert np.all(np.diff(predicted) >= -1e-12)
    assert predicted.min() >= 0 and predicted.max() <= 1
    assert score_map.predict(-1.0) == score_map.values[0]
    assert score_map.predict(2.0) == score_map.values[-1]


def test_indexed_inference_uses_an_ancestor_before_global():
    x, _, target, tree, index = _fixture()
    template = SupportTemplate(
        "loc_y", ("y",), (), location_aware=True,
        min_support=3, kappa=2.0)
    policy = DeploymentPolicy((template,), {
        template.template_id: MonotoneScoreMap.identity(),
    })
    inference = IndexedJointInference(
        target, x, tree, index, policy, theta=0.4)
    result = inference.evaluate(0, Predicate([Atom("y", "eq", 1)]))
    leaf_depth = tree.nodes[
        tree.leaf_node_ids[tree.route(x[[0]])[0]]].depth
    assert result.state == "statistical"
    assert result.support_n >= 3
    assert result.context_depth < leaf_depth
    assert result.context_node != tree.root_id
    assert result.template_id == "loc_y"
    assert inference.counters["score_map_evaluations"] == 1
    assert inference.counters["backoff_steps_total"] == result.backoff_steps
    assert inference.counters["backoff_steps_max"] == result.backoff_steps
    assert result.ancestor_backoff
    assert inference.counters["statistical_with_ancestor_context"] == 1


def test_leaf_only_context_abstains_instead_of_using_an_ancestor():
    x, _, target, tree, index = _fixture()
    template = SupportTemplate(
        "loc_y", ("y",), (), location_aware=True,
        min_support=3, kappa=2.0)
    policy = DeploymentPolicy((template,), {
        template.template_id: MonotoneScoreMap.identity(),
    })
    predicate = Predicate([Atom("y", "eq", 1)])
    hierarchical = IndexedJointInference(
        target, x, tree, index, policy, theta=0.4,
        context_mode="hierarchical").evaluate(0, predicate, record=False)
    leaf_only = IndexedJointInference(
        target, x, tree, index, policy, theta=0.4,
        context_mode="leaf_only").evaluate(0, predicate, record=False)
    assert hierarchical.state == "statistical"
    assert hierarchical.ancestor_backoff
    assert leaf_only.state == "abstain"


def test_context_mode_rejects_unknown_value():
    x, _, target, tree, index = _fixture()
    with np.testing.assert_raises_regex(ValueError, "unknown context mode"):
        IndexedJointInference(
            target, x, tree, index, DeploymentPolicy(()), theta=0.4,
            context_mode="not-a-mode")


def test_e3_semantic_fallbacks_match_indexed_joint_result():
    x, _, target, tree, index = _fixture()
    body = BodyAttribute(
        "body", (MatchLevel.equality(), MatchLevel.unrestricted()))
    template = SupportTemplate(
        "body_loc_y", ("y",), (body,), location_aware=True,
        min_support=1, kappa=0.0)
    policy = DeploymentPolicy((template,), {
        template.template_id: MonotoneScoreMap.identity(),
    })
    predicate = Predicate([Atom("y", "eq", 1)])
    indexed = IndexedJointInference(
        target, x, tree, index, policy, theta=0.4).evaluate(
            0, predicate, record=False)
    fallback = IndexedJointInference(
        target, x, tree, index, policy, theta=0.4,
        template_selection_mode="row_scan",
        support_retrieval_mode="scan").evaluate(
            0, predicate, record=False)
    assert fallback == indexed


def test_joint_product_and_min_use_common_support_but_distinct_conjunctions():
    x = np.asarray([
        [-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0],
    ], dtype=np.float32)
    support = pd.DataFrame({"y": [1, 1, 0, 0], "z": [1, 0, 1, 0]})
    target = support.copy()
    target.loc[0, ["y", "z"]] = np.nan
    tree = ProjectedPartitionTree(
        projection_dim=2, leaf_capacity=2, seed=0).fit_structure(x)
    tree.materialize(x)
    index = SharedSupportIndex(
        support, {"y": "cat", "z": "cat"}, tree.route(x), tree)
    template = SupportTemplate(
        "global_yz", ("y", "z"), (), location_aware=False,
        min_support=1, kappa=0.0)
    policy = DeploymentPolicy((template,), {
        template.template_id: MonotoneScoreMap.identity(),
    })
    predicate = Predicate([Atom("y", "eq", 1), Atom("z", "eq", 1)])
    results = {
        mode: IndexedJointInference(
            target, x, tree, index, policy, theta=0.4,
            support_mode=mode).evaluate(0, predicate, record=False)
        for mode in ("joint", "product", "min")
    }
    assert results["joint"].raw_score == 0.25
    assert results["product"].raw_score == 0.25
    assert results["min"].raw_score == 0.5
    assert not results["joint"].eligible
    assert not results["product"].eligible
    assert results["min"].eligible


def test_statistical_result_cache_survives_query_local_cache_reset():
    x, _, target, tree, index = _fixture()
    template = SupportTemplate(
        "loc_y", ("y",), (), location_aware=True,
        min_support=3, kappa=2.0)
    inference = IndexedJointInference(
        target, x, tree, index,
        DeploymentPolicy((template,), {
            template.template_id: MonotoneScoreMap.identity(),
        }), theta=0.4)
    predicate = Predicate([Atom("y", "eq", 1)])
    first = inference.evaluate(0, predicate)
    inference.begin_query()
    second = inference.evaluate(0, predicate)
    assert second == first
    assert inference.counters["inference_cache_misses"] == 1
    assert inference.counters["inference_cache_hits"] == 1
    # The cached decision is counted, but support and score-map work is not
    # repeated for an immutable deployment snapshot.
    assert inference.counters["statistical"] == 2
    assert inference.counters["score_map_evaluations"] == 1


def test_correctness_oracle_does_not_preheat_serving_result_cache():
    x, _, target, tree, index = _fixture()
    template = SupportTemplate(
        "loc_y", ("y",), (), location_aware=True,
        min_support=3, kappa=2.0)
    inference = IndexedJointInference(
        target, x, tree, index,
        DeploymentPolicy((template,), {
            template.template_id: MonotoneScoreMap.identity(),
        }), theta=0.4)
    predicate = Predicate([Atom("y", "eq", 1)])

    inference.materialize_eligibility(predicate)
    assert inference.serving_cache_manifest()["result_cache_entries"] == 0
    assert inference.counters["inference_cache_hits"] == 0
    assert inference.counters["inference_cache_misses"] == 0

    inference.evaluate(0, predicate)
    assert inference.counters["inference_cache_misses"] == 1
    assert inference.serving_cache_manifest()["result_cache_entries"] == 1


def test_clear_all_serving_caches_resets_every_mutable_online_cache():
    x, _, target, tree, index = _fixture()
    template = SupportTemplate(
        "loc_y", ("y",), (), location_aware=True,
        min_support=3, kappa=2.0)
    inference = IndexedJointInference(
        target, x, tree, index,
        DeploymentPolicy((template,), {
            template.template_id: MonotoneScoreMap.identity(),
        }), theta=0.4)
    predicate = Predicate([Atom("y", "eq", 1)])
    inference.begin_query(predicate)
    inference.evaluate(0, predicate)
    inference._candidate_group_result_cache[("test",)] = (None, None)
    inference._candidate_group_result_cache_entries = 1
    inference._candidate_predicate_plan_cache[("test",)] = object()
    inference._support_satisfaction_packed_cache[("test",)] = object()
    inference._support_leaf_prefix_cache[("test",)] = object()

    before = inference.serving_cache_manifest()
    assert before["total_entries"] > 0
    assert before["active_predicate_cache"]
    after = inference.clear_all_serving_caches()
    assert after["total_entries"] == 0
    assert not after["active_predicate_cache"]
    assert after["all_serving_caches_empty"]


def test_external_certified_domains_precede_statistical_inference():
    x, _, target, tree, index = _fixture()
    target.loc[[0, 1, 2], "y"] = np.nan

    def broad_constraint(row, attr):
        return CertifiedDomain.finite((0, 1)) if attr == "y" else None

    def narrow_constraint(row, attr):
        if attr != "y":
            return None
        return CertifiedDomain.finite((1,)) if row.name == 0 else \
            CertifiedDomain.finite((0,)) if row.name == 1 else None

    inference = IndexedJointInference(
        target, x, tree, index, DeploymentPolicy(()), theta=0.5,
        deterministic_constraints=(broad_constraint, narrow_constraint))
    predicate = Predicate([Atom("y", "eq", 1)])
    assert inference.evaluate(0, predicate).state == "hard_pass"
    assert inference.evaluate(1, predicate).state == "hard_fail"
    assert inference.evaluate(2, predicate).state == "abstain"
    assert inference.counters["deterministic_pass"] == 1
    assert inference.counters["deterministic_fail"] == 1
