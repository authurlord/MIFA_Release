"""Strict compatibility checks for reusing frozen baseline measurements."""
from __future__ import annotations


_PARTITION_FIELDS = (
    "base_n", "query_n", "mask_digest", "workload_digest",
    "query_source_corpus", "incomplete_tuple_rate", "masked_cell_rate",
)


def baseline_reuse_signature(report):
    """Return only fields that determine a baseline's experimental inputs.

    Method-specific source revisions and outputs are intentionally excluded:
    this signature answers whether a frozen baseline row was measured on the
    same data, embeddings, masks, and validation/calibration/test workload.
    """
    dataset = report.get("dataset", {})
    query_source = report.get("query_source", {})
    split = report.get("split", {})
    train_mask = report.get("train_mask", {})
    partitions = report.get("partitions", {})
    missingness = report.get("missingness", {})
    canonical_missingness = {"kind": missingness.get("kind")}
    if missingness.get("kind") in {
            "single_attribute_mcar_nested", "tuple_predicate_mcar",
            "tuple_query_answer_mnar",
            "joint_tuple_block"}:
        canonical_missingness["block_size"] = missingness.get("block_size")
        canonical_missingness["block_attrs"] = missingness.get("block_attrs", [])
    if missingness.get("training_kind") is not None:
        canonical_missingness["training_kind"] = missingness.get("training_kind")
    if missingness.get("kind") == "tuple_query_answer_mnar":
        canonical_missingness["answer_impact_fraction"] = missingness.get(
            "answer_impact_fraction", 1.0)
        canonical_missingness["answer_cells_per_tuple"] = missingness.get(
            "answer_cells_per_tuple", 1)
    if missingness.get("kind") in {"tuple_attr_skewed", "tuple_attribute_skewed"}:
        canonical_missingness["attribute_weights"] = missingness.get(
            "attribute_weights", {})
    return {
        "dataset": {key: dataset.get(key) for key in (
            "name", "n", "d", "hash", "attrs")},
        "seed": report.get("seed"),
        "rho_tuple": report.get("rho_tuple"),
        "missingness": canonical_missingness,
        "split": {key: split.get(key) for key in (
            "strategy", "counts", "digest")},
        "train_mask_digest": train_mask.get("digest"),
        "query_source": {key: query_source.get(key) for key in (
            "kind", "corpus", "source_file_hashes",
            "role_source_ids_digest", "role_embeddings_digest")},
        "partitions": {
            role: {key: partitions.get(role, {}).get(key)
                   for key in _PARTITION_FIELDS}
            for role in ("validation", "calibration", "test")
        },
    }


def compare_baseline_reuse(current, frozen):
    current_signature = baseline_reuse_signature(current)
    frozen_signature = baseline_reuse_signature(frozen)
    differences = []

    def walk(left, right, path=""):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), f"{path}.{key}" if path else key)
        elif left != right:
            differences.append({"field": path, "current": left, "frozen": right})

    walk(current_signature, frozen_signature)
    return {
        "reusable": not differences,
        "differences": differences,
        "current_signature": current_signature,
        "frozen_signature": frozen_signature,
    }
