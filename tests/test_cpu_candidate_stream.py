import numpy as np
import pandas as pd

from mifa.index.ann_backend import ExactBackend
from mifa.index.projected_tree import ProjectedPartitionTree
from mifa.inference.paper_policy import DeploymentPolicy, IndexedJointInference
from mifa.inference.support_index import SharedSupportIndex
from mifa.predicates import Atom, Predicate
from mifa.search.cpu_candidate_stream import (
    NumpyExactCandidateStreamProvider,
)
from mifa.search.gpu_candidate_stream import HARD_PASS, UNRESOLVED
from mifa.search.unified_executor import UnifiedBestFirstExecutor


def test_cpu_candidate_pages_preserve_exact_tie_order_and_hard_states():
    base = np.asarray([
        [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0],
        [2.0, 0.0], [-2.0, 0.0],
    ], dtype=np.float32)
    queries = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    values = np.asarray([[1, 1, 0, 0, 1, 1]], dtype=np.float64)
    missing = np.asarray(
        [[False, False, False, True, False, False]], dtype=np.bool_)
    plans = [{
        "values": values,
        "missing": missing,
        "attrs": np.asarray([0], dtype=np.int32),
        "ops": np.asarray([3], dtype=np.int8),
        "lower": np.asarray([1.0], dtype=np.float64),
        "upper": np.asarray([1.0], dtype=np.float64),
    } for _ in queries]
    provider = NumpyExactCandidateStreamProvider(base)
    handle = provider.begin_batch(queries, plans)
    first = provider.next_page(
        handle, np.asarray([True, True]), page_size=2)
    second = provider.next_page(
        handle, np.asarray([True, True]), page_size=3)

    np.testing.assert_array_equal(first.ids, [[0, 1], [0, 1]])
    np.testing.assert_array_equal(first.states, np.full(
        (2, 2), int(HARD_PASS), dtype=np.int8))
    np.testing.assert_array_equal(second.ids, [[3, 4, 5], [4, 3, 5]])
    np.testing.assert_array_equal(second.states, [
        [int(UNRESOLVED), int(HARD_PASS), int(HARD_PASS)],
        [int(HARD_PASS), int(UNRESOLVED), int(HARD_PASS)],
    ])
    assert handle.telemetry["compact_d2h_bytes_total"] == 0
    assert handle.telemetry["peak_gpu_allocated_bytes"] == 0
    provider.close(handle)


def test_cpu_candidate_executor_matches_exact_leaf_reference():
    rng = np.random.default_rng(919)
    embeddings = rng.normal(size=(317, 9)).astype(np.float32)
    table = pd.DataFrame({
        "category": np.asarray(["a", "b", "c"] * 106, dtype=object)[:317],
        "score": np.linspace(0.0, 1.0, 317),
    })
    table.loc[np.arange(7, 317, 19), "category"] = np.nan
    table.loc[np.arange(11, 317, 23), "score"] = np.nan
    tree = ProjectedPartitionTree(
        projection_dim=6, leaf_capacity=16, seed=31).fit_structure(embeddings)
    tree.materialize(embeddings)
    index = SharedSupportIndex(
        table, {"category": "cat", "score": "num"},
        tree.route(embeddings), tree)
    inference = IndexedJointInference(
        table, embeddings, tree, index, DeploymentPolicy(()), theta=0.5)
    executor = UnifiedBestFirstExecutor(
        embeddings, tree, inference,
        ann_backend=ExactBackend().build(embeddings), seed_size=16,
        fused_cpu=True, leaf_frontier=True)
    queries = rng.normal(size=(32, 9)).astype(np.float32)
    predicates = [Predicate([
        Atom("category", "eq", "a"), Atom("score", "ge", 0.2),
    ]) for _ in queries]
    ks = [10] * len(queries)
    reference = [
        executor.search(query, k, predicate, epsilon=0.0)
        for query, k, predicate in zip(queries, ks, predicates)]
    provider = NumpyExactCandidateStreamProvider(embeddings)
    actual = executor.search_candidate_stream_batch(
        queries, ks, predicates, provider,
        initial_page_size=16, max_page_size=64)

    for expected, current in zip(reference, actual):
        np.testing.assert_array_equal(current[0], expected[0])
        # The reference evaluates one tuple dot product at a time, whereas
        # the provider uses SGEMM.  Ordered IDs must be identical; the two
        # valid float32 accumulation orders may differ by a few ULPs.
        np.testing.assert_allclose(current[1], expected[1], rtol=2e-6, atol=2e-6)
        assert current[2]["candidate_stream_fallback"] == 0
        assert current[2]["candidate_stream_compact_d2h_bytes"] == 0


def test_cpu_batch_one_eligibility_first_path_is_exact_and_cached():
    rng = np.random.default_rng(23)
    base = rng.normal(size=(4096, 32)).astype(np.float32)
    values = np.asarray([
        np.linspace(0.0, 1.0, len(base)),
    ], dtype=np.float64)
    missing = np.zeros_like(values, dtype=np.bool_)
    missing[0, 7::19] = True
    group_ids = np.arange(len(base), dtype=np.int32) % 7
    decisions = np.asarray([1, 2, 1, 2, 1, 2, 1], dtype=np.int8)
    plan = {
        "values": values,
        "missing": missing,
        "attrs": np.asarray([0], dtype=np.int32),
        "ops": np.asarray([2], dtype=np.int8),
        "lower": np.asarray([0.2], dtype=np.float64),
        "upper": np.asarray([0.7], dtype=np.float64),
        "statistical_group_ids": group_ids,
        "statistical_group_decisions": decisions,
    }
    query = rng.normal(size=(1, base.shape[1])).astype(np.float32)
    provider = NumpyExactCandidateStreamProvider(base)

    reference_handle = provider.begin_batch(query, [plan])
    reference_page = provider.next_page(
        reference_handle, np.asarray([True]), page_size=10)
    provider.close(reference_handle)

    first = provider.begin_batch_topk(query, [plan], topk=10)
    second = provider.begin_batch_topk(query, [plan], topk=10)
    assert first is not None
    assert second is not None
    np.testing.assert_array_equal(
        first.prefetched_page.ids, reference_page.ids)
    np.testing.assert_array_equal(
        second.prefetched_page.ids, reference_page.ids)
    np.testing.assert_allclose(
        first.prefetched_page.distances, reference_page.distances,
        rtol=2e-6, atol=2e-6)
    assert second.telemetry["batch1_eligibility_cache_hits"] == 1
    assert second.telemetry["batch1_distance_rows_evaluated"] < len(base)
    assert second.telemetry["peak_gpu_allocated_bytes"] == 0


def test_cpu_batch_one_fused_exhaustive_screen_preserves_exact_topk():
    rng = np.random.default_rng(28)
    base = rng.normal(size=(2048, 48)).astype(np.float32)
    query = rng.normal(size=(1, base.shape[1])).astype(np.float32)
    eligible_ids = np.asarray(
        rng.choice(len(base), size=1379, replace=False), dtype=np.uint32)
    plan = {
        "values": np.empty((0, len(base)), dtype=np.float64),
        "missing": np.empty((0, len(base)), dtype=np.bool_),
        "attrs": np.empty(0, dtype=np.int32),
        "ops": np.empty(0, dtype=np.int8),
        "lower": np.empty(0, dtype=np.float64),
        "upper": np.empty(0, dtype=np.float64),
        "precomputed_eligible_ids": eligible_ids,
    }
    provider = NumpyExactCandidateStreamProvider(
        base, batch_one_fused_exact_min_work=1)
    provider.prewarm_eligibility([plan])

    handle = provider.begin_batch_topk(query, [plan], topk=10)
    assert handle is not None
    page = handle.prefetched_page
    exact_distances = np.sum(
        (base[eligible_ids] - query[0]) ** 2, axis=1, dtype=np.float32)
    keys = (
        exact_distances.view(np.uint32).astype(np.uint64) << np.uint64(32)
    ) | eligible_ids.astype(np.uint64)
    expected_order = np.argsort(keys)[:10]

    np.testing.assert_array_equal(
        page.ids[0], eligible_ids[expected_order].astype(np.int64))
    np.testing.assert_allclose(
        page.distances[0], exact_distances[expected_order],
        rtol=2e-6, atol=2e-6)
    assert handle.telemetry["batch1_exhaustive_full_dimension_screen"]
    assert handle.telemetry["batch1_distance_rows_evaluated"] == len(
        eligible_ids)
