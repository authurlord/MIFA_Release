#!/usr/bin/env python3
"""Paper-audit-v2 Phase A: leakage-free exact-search effectiveness.

This is the first formal protocol entry point, not the final systems table. It
uses train-only support/imputer fitting, validation-only policy selection,
calibration-only threshold selection, source-disjoint query tuples, and a hidden
test relation. ANN backends are deliberately absent so this phase isolates
missing-value eligibility error. Exact masked search uses the production
Numba-fused kernel; the reference per-query Python path is not used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mifa.datasets import DatasetSpec, load_dataset
from mifa.eval.metrics import precision_at_k, recall_at_k
from mifa.index.grid import CoarseCellIndex
from mifa.inference.fitted_imputers import MeanModeImputer, MixedIterativeImputer
from mifa.inference.joint_support import JointSupportEngine
from mifa.inference.query_cls import QueryConditionedClassifier
from mifa.inference.rules import discover_rules
from mifa.knn import active_backend, query_dists_batch
from mifa.knn_numba import topk_masked_batch_numba
from mifa.missingness import make_missing, training_missingness_kind
from mifa.predicates import TRUE
from mifa.protocol import allocate_query_roles, make_four_way_split, subset_spec
from mifa.workload_v2 import (
    explicit_native_filter_workload,
    native_filter_workload,
    native_range_workload,
    sample_selectivity_matched_arity_workload,
    sample_stratified_workload,
    workload_counts,
    workload_digest,
)


def _sha_json(value):
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _source_provenance():
    paths = [
        "scripts/experiment_v2.py",
        "src/mifa/protocol.py",
        "src/mifa/missingness.py",
        "src/mifa/workload_v2.py",
        "src/mifa/inference/joint_support.py",
        "src/mifa/inference/joint_support_numba.py",
        "src/mifa/inference/fitted_imputers.py",
        "src/mifa/inference/query_cls.py",
        "src/mifa/knn.py",
        "src/mifa/knn_numba.py",
    ]
    hashes = {}
    for relative in paths:
        content = (ROOT / relative).read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=ROOT, text=True)
    return {"files": hashes, "manifest_hash": _sha_json(hashes),
            "git_dirty_for_manifest": bool(status.strip())}


def _mask_digest(mask):
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def _progress(stage, **details):
    print(json.dumps({"stage": stage, **details}, default=str),
          file=sys.stderr, flush=True)


def _win_check(mifa_f1, baseline_f1, *, required_gain_pp=5.0):
    """Report a plain win separately from the predeclared absolute gain."""
    delta = float(mifa_f1 - baseline_f1)
    required_gain = float(required_gain_pp) / 100.0
    return {
        "best_baseline_f1": float(baseline_f1),
        "mifa_f1": float(mifa_f1),
        "delta": delta,
        "required_absolute_gain_pp": float(required_gain_pp),
        "won_this_seed": bool(delta > 0.0),
        "passed_this_seed": bool(delta + 1e-12 >= required_gain),
    }


def _attribute_weight_vector(value, attrs):
    """Resolve a predeclared workload-frequency map in dataset column order."""
    if not value:
        return None, {}
    if isinstance(value, str):
        mapping = {}
        for item in value.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(
                    "attribute weights must use attr=weight comma pairs")
            attr, weight = item.split("=", 1)
            attr = attr.strip()
            if attr in mapping:
                raise ValueError(f"duplicate attribute weight: {attr}")
            mapping[attr] = float(weight)
    elif isinstance(value, dict):
        mapping = {str(attr): float(weight)
                   for attr, weight in value.items()}
    else:
        raise ValueError("attribute weights must be a mapping or CLI string")
    unknown = sorted(set(mapping) - set(attrs))
    if unknown:
        raise ValueError(f"attribute weights contain unknown attrs: {unknown}")
    if any(weight < 0 for weight in mapping.values()):
        raise ValueError("attribute weights must be non-negative")
    vector = np.asarray([mapping.get(attr, 0.0) for attr in attrs], dtype=float)
    if vector.sum() <= 0:
        raise ValueError("at least one attribute weight must be positive")
    normalized = vector / vector.sum()
    manifest = {attr: float(weight) for attr, weight in zip(attrs, normalized)}
    return normalized, manifest


def _resolve_attribute_weights(kind, cli_value, source_value, attrs):
    """Activate configured weights only for attribute-skewed missingness."""
    if kind != "tuple_attr_skewed":
        if cli_value:
            raise ValueError(
                "--attribute-weights requires tuple_attr_skewed missingness")
        return None, {}
    weights, manifest = _attribute_weight_vector(
        cli_value or source_value, attrs)
    if weights is None:
        raise ValueError(
            "tuple_attr_skewed requires predeclared attribute weights")
    return weights, manifest


def _holdout_query_sources(spec, n_queries, seed, role):
    if n_queries >= spec.n:
        raise ValueError(f"{role}: n_queries must leave searchable tuples")
    rng = np.random.default_rng(seed)
    source_rows = np.sort(rng.choice(spec.n, size=n_queries, replace=False))
    search_rows = np.flatnonzero(~np.isin(np.arange(spec.n), source_rows))
    return (subset_spec(spec, source_rows, name=f"{spec.name}_{role}_query"),
            subset_spec(spec, search_rows, name=f"{spec.name}_{role}_search"),
            source_rows, search_rows)


def _query_spec(embeddings, rows, *, metric, corpus, name, allow_source_reuse=False):
    rows = np.asarray(rows, dtype=np.int64)
    return DatasetSpec(
        name=name,
        A_e=np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32)[rows]),
        table=pd.DataFrame(index=np.arange(len(rows))),
        dtypes={},
        metric=metric,
        query_relevant_attrs=[],
        meta={"source_ids": rows.tolist(), "source_corpus": corpus,
              "allow_source_reuse": bool(allow_source_reuse)},
    )


def _prepare_query_sources(spec, paper_config, *, role_counts, seed,
                           embedding_tag=None):
    """Freeze source-disjoint query roles before any database masking."""
    source_cfg = paper_config.get("query_sources", {}).get(spec.name)
    if source_cfg is None:
        return spec, {}, {}, {"kind": "partition_holdout_legacy"}
    kind = source_cfg["kind"]
    if kind == "heldout_content":
        role_rows, database_rows = allocate_query_roles(
            spec.n, role_counts, seed=seed)
        query_specs = {
            role: _query_spec(
                spec.A_e, rows, metric=spec.metric,
                corpus=f"{spec.name}:heldout_content", name=f"{spec.name}_{role}_query")
            for role, rows in role_rows.items()
        }
        database = subset_spec(spec, database_rows, name=spec.name)
        aux = {}
        corpus_size = spec.n
    elif kind in {"external_embeddings", "external_native_filters"}:
        embedding_choices = source_cfg.get("embeddings_by_tag", {})
        query_value = embedding_choices.get(embedding_tag, source_cfg.get("embeddings"))
        if query_value is None:
            raise ValueError(
                f"no external query embeddings declared for tag={embedding_tag!r}")
        query_path = ROOT / query_value
        embeddings = np.load(query_path, mmap_mode="r")
        if embeddings.ndim != 2 or embeddings.shape[1] != spec.d:
            raise ValueError(
                f"external query shape {embeddings.shape} does not match database d={spec.d}")
        candidate_rows = np.arange(len(embeddings), dtype=np.int64)
        metadata = None
        if source_cfg.get("metadata"):
            metadata = pd.read_parquet(ROOT / source_cfg["metadata"])
            if len(metadata) != len(embeddings):
                raise ValueError("external query metadata is not row-aligned")
            if source_cfg.get("category_column"):
                column = source_cfg["category_column"]
                candidate_rows = np.flatnonzero(
                    metadata[column].astype(str).to_numpy() ==
                    str(source_cfg["category_value"]))
        if kind == "external_native_filters" and metadata is None:
            raise ValueError("external native filters require query metadata")
        group_counts = {
            role: int(source_cfg.get("role_group_counts", {}).get(role, count))
            for role, count in role_counts.items()
        }
        role_local, _ = allocate_query_roles(
            len(candidate_rows), group_counts, seed=seed)
        role_rows = {role: candidate_rows[rows] for role, rows in role_local.items()}
        query_specs = {
            role: _query_spec(
                embeddings, rows, metric=spec.metric,
                corpus=source_cfg["corpus"], name=f"{source_cfg['corpus']}_{role}",
                # ``allocate_query_roles`` returns every protocol role,
                # including optional roles with a zero count.  Legacy replay
                # callers do not request the fresh-final reservation, so use
                # a zero default instead of indexing their four-role mapping.
                allow_source_reuse=int(role_counts.get(role, 0)) > len(rows))
            for role, rows in role_rows.items()
        }
        database = spec
        aux = {}
        if kind == "external_native_filters":
            atom_templates = source_cfg.get("filter_atoms", ())
            if not atom_templates:
                raise ValueError(
                    "external native filters require filter_atoms")
            for role, rows in role_rows.items():
                role_filters = []
                for row in rows:
                    atoms = []
                    record = metadata.iloc[int(row)]
                    for template in atom_templates:
                        attr = str(template["attr"])
                        op = str(template["op"])
                        a = record[template["a_column"]]
                        b_column = template.get("b_column")
                        b = record[b_column] if b_column else None
                        if isinstance(a, np.generic):
                            a = a.item()
                        if isinstance(b, np.generic):
                            b = b.item()
                        atoms.append({"attr": attr, "op": op, "a": a,
                                      "b": b})
                    role_filters.append({
                        "family": "explicit",
                        "workload_family": source_cfg.get(
                            "workload_family", "external_native_filter"),
                        "atoms": atoms,
                    })
                aux[role] = role_filters
        corpus_size = len(candidate_rows)
        source_file_hashes = {
            "embeddings": hashlib.sha256(query_path.read_bytes()).hexdigest(),
        }
        if source_cfg.get("metadata"):
            metadata_path = ROOT / source_cfg["metadata"]
            source_file_hashes["metadata"] = hashlib.sha256(
                metadata_path.read_bytes()).hexdigest()
    elif kind in {"arxiv_native_r", "arxiv_native_mixed"}:
        from mifa.datasets_arxiv import _hf, _read_jsonl, load_arxiv_queries
        repo = source_cfg.get("repo", "SPCL/arxiv-for-fanns-medium")
        embeddings, range_filters = load_arxiv_queries(
            repo, metric=spec.metric)
        if embeddings.shape[1] != spec.d:
            raise ValueError("native ArXiv query dimension does not match database")
        exact_filters = (_read_jsonl(_hf(repo, "em_query_attributes.jsonl"))
                         if kind == "arxiv_native_mixed" else None)
        if exact_filters is not None and len(exact_filters) != len(embeddings):
            raise ValueError("native ArXiv EM filters are not query-row aligned")
        role_rows, _ = allocate_query_roles(
            len(embeddings), role_counts, seed=seed)
        corpus = ("arxiv-for-fanns-native-r-em"
                  if kind == "arxiv_native_mixed" else "arxiv-for-fanns-native-r")
        query_specs = {
            role: _query_spec(
                embeddings, rows, metric=spec.metric,
                corpus=corpus, name=f"{kind}_{role}")
            for role, rows in role_rows.items()
        }
        aux = {}
        family_counts = {}
        for role, rows in role_rows.items():
            items = []
            counts = {"r": 0, "em": 0}
            for local, row in enumerate(rows):
                family = "r" if kind == "arxiv_native_r" or local % 2 == 0 else "em"
                source = range_filters if family == "r" else exact_filters
                items.append({"family": family, **source[int(row)]})
                counts[family] += 1
            aux[role] = items
            family_counts[role] = {key: value for key, value in counts.items() if value}
        database = spec
        corpus_size = len(embeddings)
        native_files = ["query_vectors.fvecs", "r_query_attributes.jsonl"]
        if exact_filters is not None:
            native_files.append("em_query_attributes.jsonl")
        source_file_hashes = {
            name: hashlib.sha256(Path(_hf(repo, name)).read_bytes()).hexdigest()
            for name in native_files
        }
    else:
        raise ValueError(f"unknown main query source kind: {kind}")
    joined = np.concatenate([
        np.asarray(query_specs[role].meta["source_ids"], dtype=np.int64)
        for role in role_counts if role in query_specs
    ])
    if kind == "heldout_content":
        source_file_hashes = {"parent_dataset_hash": spec.hash()}
    selected_vectors = np.concatenate([
        np.ascontiguousarray(query_specs[role].A_e, dtype=np.float32)
        for role in role_counts if role in query_specs
    ], axis=0)
    manifest = {
        "kind": kind,
        "corpus": source_cfg.get("corpus", kind),
        "corpus_size": int(corpus_size),
        "role_counts": {role: int(len(query_specs[role].meta["source_ids"]))
                        for role in query_specs},
        "role_query_instances": {
            role: int(role_counts.get(role, 0)) for role in query_specs},
        "role_source_ids_digest": hashlib.sha256(
            np.ascontiguousarray(joined, dtype=np.int64).tobytes()).hexdigest(),
        "role_embeddings_digest": hashlib.sha256(
            selected_vectors.tobytes()).hexdigest(),
        "source_file_hashes": source_file_hashes,
        "source_roles_disjoint": len(joined) == len(np.unique(joined)),
        "query_sources_disjoint_from_search_relation": True,
    }
    if kind in {"external_embeddings", "external_native_filters"}:
        manifest["embedding_tag"] = embedding_tag
    if kind in {"arxiv_native_r", "arxiv_native_mixed"}:
        manifest["native_family_counts"] = family_counts
    return database, query_specs, aux, manifest


def _query_cls_training_reservation(paper_config, n_validation_queries,
                                    *, enabled=True):
    """Return the query-source reservation used by the faithful baseline run.

    Paper-only runners do not train QueryCls, but they must still reserve its
    source-disjoint query rows.  Otherwise heldout-content sources change the
    searchable relation, split, masks, and workloads, making frozen baseline
    reuse invalid.
    """
    if not enabled:
        return 0
    return min(
        int(paper_config["workload"].get("query_training_queries", 200)),
        max(60, int(n_validation_queries)),
    )


def _make_workload(query_spec, base, n_queries, k, *, seed, attrs,
                   predicate_reference, native_filters=None,
                   signature_mix=None, canonical_bucket_weights=(0.5, 0.5, 0.0),
                   matched_arity_signatures=None):
    if native_filters is not None:
        if native_filters and native_filters[0].get("family") == "explicit":
            return explicit_native_filter_workload(
                query_spec, base, native_filters, k,
                predicate_reference_spec=predicate_reference)
        return native_filter_workload(
            query_spec, base, native_filters, k,
            predicate_reference_spec=predicate_reference)
    if matched_arity_signatures:
        return sample_selectivity_matched_arity_workload(
            query_spec, base, n_queries, k, seed=seed,
            signatures=matched_arity_signatures,
            predicate_reference_spec=predicate_reference,
            bucket_weights=canonical_bucket_weights)
    return sample_stratified_workload(
        query_spec, base, n_queries, k, seed=seed,
        predicate_attrs=attrs,
        predicate_reference_spec=predicate_reference,
        signature_mix=signature_mix,
        canonical_bucket_weights=canonical_bucket_weights,
    )


def _prepare_eval(partition, train_observed, cell_index, *, role, n_queries,
                  k, rho, mask_seed, workload_seed, attrs,
                  predicate_reference, missingness_kind, block_size,
                  block_attrs=(), attribute_weights=None,
                  answer_impact_fraction=1.0,
                  answer_cells_per_tuple=1,
                  query_spec=None, native_filters=None, signature_mix=None,
                  canonical_bucket_weights=(0.5, 0.5, 0.0)):
    if query_spec is None:
        query_spec, base, query_rows, search_rows = _holdout_query_sources(
            partition, n_queries, workload_seed + 1009, role)
    else:
        if query_spec.n < n_queries and not query_spec.meta.get("allow_source_reuse"):
            raise ValueError(f"{role}: frozen query role has {query_spec.n}, expected {n_queries}")
        base = partition
        query_rows = np.asarray(query_spec.meta["source_ids"], dtype=np.int64)
        search_rows = np.arange(base.n, dtype=np.int64)
    base.query_relevant_attrs = list(attrs)
    queries = _make_workload(
        query_spec, base, n_queries, k, seed=workload_seed, attrs=attrs,
        predicate_reference=predicate_reference, native_filters=native_filters,
        signature_mix=signature_mix,
        canonical_bucket_weights=canonical_bucket_weights)
    Q = np.asarray([query["q_e"] for query in queries], dtype=np.float32)
    X = base.normalized_embeddings()
    D2 = query_dists_batch(Q, X, np.einsum("ij,ij->i", X, X))
    truth_elig = np.asarray([
        query["phi"].eval_table(base.table) == TRUE for query in queries
    ])
    truth_ids = topk_masked_batch_numba(D2, truth_elig, k)[0]
    observed, mask, _ = make_missing(
        base, rho, missingness_kind, seed=mask_seed, block_size=block_size,
        block_attrs=block_attrs, usage_freq=attribute_weights,
        queries=queries, truth_ids=truth_ids,
        answer_impact_fraction=answer_impact_fraction,
        answer_cells_per_tuple=answer_cells_per_tuple)
    attr_position = {attr: index for index, attr in enumerate(attrs)}
    masked_answer_affected = []
    for query, answer_ids in zip(queries, truth_ids):
        rows = answer_ids[answer_ids >= 0]
        columns = sorted({attr_position[atom.attr] for atom in query["phi"].atoms})
        affected = bool(len(rows) and columns and mask[np.ix_(rows, columns)].any())
        masked_answer_affected.append(affected)
    cells = cell_index.route(X)
    return {
        "role": role,
        "base": base,
        "observed": observed,
        "mask": mask,
        "queries": queries,
        "D2": D2,
        "truth_elig": truth_elig,
        "truth_ids": truth_ids,
        "cells": cells,
        "query_rows": query_rows,
        "search_rows": search_rows,
        "query_source_corpus": query_spec.meta.get("source_corpus", query_spec.name),
        "mask_digest": _mask_digest(mask),
        "workload_digest": workload_digest(queries),
        "workload_counts": workload_counts(queries),
        "incomplete_tuple_rate": float(mask.any(axis=1).mean()),
        "masked_cell_rate": float(mask.mean()),
        "masked_answer_affected": np.asarray(masked_answer_affected, dtype=bool),
    }


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _score_ids(ids, truth_ids, queries, headline_admitted=None,
               masked_answer_affected=None):
    rows = []
    for i, (returned, truth) in enumerate(zip(ids, truth_ids)):
        returned = returned[returned >= 0]
        truth = truth[truth >= 0]
        precision = precision_at_k(returned, truth)
        recall = recall_at_k(returned, truth)
        rows.append({
            "query_index": i,
            "source_row": int(queries[i]["src"]),
            "signature": _query_signature(queries[i]),
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "arity": queries[i]["arity"],
            "predicate_type": queries[i]["predicate_type"],
            "selectivity_bucket": queries[i]["selectivity_bucket"],
        })

    def aggregate(selected):
        return {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in ("precision", "recall", "f1")
        } | {"n_queries": len(selected)}

    by = {}
    for key in ("arity", "predicate_type", "selectivity_bucket"):
        by[key] = {}
        for value in sorted({str(row[key]) for row in rows}):
            selected = [row for row in rows if str(row[key]) == value]
            by[key][value] = aggregate(selected)
    if headline_admitted is not None:
        for row, admitted in zip(rows, headline_admitted):
            row["headline_admitted"] = bool(admitted)
    if masked_answer_affected is not None:
        for row, affected in zip(rows, masked_answer_affected):
            row["masked_answer_affected"] = bool(affected)
    result = {"macro": aggregate(rows), "strata": by, "per_query": rows}
    if headline_admitted is not None:
        admitted_rows = [row for row, admitted in zip(rows, headline_admitted) if admitted]
        if not admitted_rows:
            raise ValueError("support-rich headline admission selected zero queries")
        result["headline"] = aggregate(admitted_rows)
    if masked_answer_affected is not None:
        affected_rows = [
            row for row, affected in zip(rows, masked_answer_affected) if affected]
        unaffected_rows = [
            row for row, affected in zip(rows, masked_answer_affected) if not affected]
        if affected_rows:
            result["masked_answer"] = aggregate(affected_rows)
        if unaffected_rows:
            result["unaffected_answer"] = aggregate(unaffected_rows)
    return result


def _score_eligibility(ctx, eligible):
    ids = topk_masked_batch_numba(
        ctx["D2"], np.asarray(eligible, dtype=bool),
        ctx["queries"][0]["k"])[0]
    return _score_ids(ids, ctx["truth_ids"], ctx["queries"],
                      ctx.get("headline_admitted"),
                      ctx.get("masked_answer_affected"))


def _attach_headline_admission(ctx, training_observed, *, min_joint_support):
    """Freeze support-rich query admission from masked training evidence only."""
    support = np.asarray([
        int(np.sum(query["phi"].eval_table(training_observed) == TRUE))
        for query in ctx["queries"]
    ], dtype=np.int64)
    admitted = support >= int(min_joint_support)
    if not admitted.any():
        raise ValueError("no query meets min_training_joint_support")
    ctx["training_joint_support"] = support
    ctx["headline_admitted"] = admitted
    ctx["headline_admission"] = {
        "rule": "training_joint_support_gte",
        "threshold": int(min_joint_support),
        "admitted_queries": int(admitted.sum()),
        "scope_check_queries": int((~admitted).sum()),
        "support_min": int(support.min()),
        "support_median": float(np.median(support)),
        "support_max": int(support.max()),
        "digest": hashlib.sha256(
            np.ascontiguousarray(support, dtype=np.int64).tobytes()).hexdigest(),
        "uses_test_hidden_values": False,
    }


def _table_policy(ctx, completed):
    return np.asarray([
        query["phi"].eval_table(completed) == TRUE for query in ctx["queries"]
    ])


def _joint_certificates(ctx, support, support_cells, body_by_target, dtypes,
                        policy, mode="joint"):
    engine = JointSupportEngine(
        support, ctx["observed"], mode=mode,
        support_cells=support_cells if policy["location"] == "cell" else None,
        target_cells=ctx["cells"] if policy["location"] == "cell" else None,
        body_by_target=body_by_target if policy.get("scalar_body") else None,
        dtypes=dtypes,
        min_support=policy["min_support"], kappa=policy["kappa"],
    )
    certs = engine.certificate_batch_numba(
        [query["phi"] for query in ctx["queries"]])
    if "enabled_targets" in policy:
        certs = _apply_target_gate(ctx, certs, set(policy["enabled_targets"]))
    if "enabled_signatures" in policy:
        certs = _apply_signature_gate(ctx, certs, set(policy["enabled_signatures"]))
    return engine, certs


def _query_signature(query):
    return "&".join(sorted(atom.attr for atom in query["phi"].atoms))


def _apply_signature_gate(ctx, certs, enabled_signatures):
    gated = []
    for query, cert in zip(ctx["queries"], certs):
        if _query_signature(query) in enabled_signatures:
            gated.append(cert)
            continue
        copy = {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in cert.items()}
        statistical = copy["uncertain_mask"] | copy["abstain_mask"]
        copy["uncertain_mask"][statistical] = False
        copy["abstain_mask"][statistical] = True
        copy["p_phi"][statistical] = 0.0
        copy["states"][statistical] = "false"
        copy["context_name"][statistical] = "policy_drop"
        gated.append(copy)
    return gated


def _per_query_f1(ctx, eligible):
    ids = topk_masked_batch_numba(
        ctx["D2"], np.asarray(eligible, dtype=bool),
        ctx["queries"][0]["k"])[0]
    values = []
    for returned, truth in zip(ids, ctx["truth_ids"]):
        returned = returned[returned >= 0]
        truth = truth[truth >= 0]
        precision = precision_at_k(returned, truth)
        recall = recall_at_k(returned, truth)
        values.append(_f1(precision, recall))
    return np.asarray(values)


def _learn_signature_gate(ctx, certs, min_queries=2, min_source_groups=1,
                          min_delta=0.0):
    drop = np.asarray([
        query["phi"].eval_table(ctx["observed"]) == TRUE for query in ctx["queries"]
    ])
    drop_f1 = _per_query_f1(ctx, drop)
    signatures = np.asarray([_query_signature(query) for query in ctx["queries"]])
    admitted = np.asarray(
        ctx.get("headline_admitted", np.ones(len(signatures), dtype=bool)),
        dtype=bool)
    best = None
    for theta in np.linspace(0.0, 1.0, 21):
        inferred = np.asarray([
            JointSupportEngine.eligible(cert, theta) for cert in certs
        ])
        inferred_f1 = _per_query_f1(ctx, inferred)
        enabled = []
        manifest = {}
        for signature in sorted(set(signatures)):
            rows = (signatures == signature) & admitted
            n = int(rows.sum())
            if not n:
                continue
            mifa = float(inferred_f1[rows].mean())
            baseline = float(drop_f1[rows].mean())
            selected_rows = np.flatnonzero(rows)
            group_deltas = {}
            for row in selected_rows:
                source = int(ctx["queries"][row]["src"])
                group_deltas.setdefault(source, []).append(
                    float(inferred_f1[row] - drop_f1[row]))
            group_values = np.asarray(
                [np.mean(values) for values in group_deltas.values()], dtype=float)
            group_delta = float(group_values.mean()) if len(group_values) else 0.0
            use = (n >= min_queries and len(group_values) >= min_source_groups and
                   group_delta > min_delta + 1e-12)
            if use:
                enabled.append(signature)
            manifest[signature] = {
                "n_queries": n, "inference_f1": mifa, "drop_f1": baseline,
                "delta": mifa - baseline,
                "source_groups": int(len(group_values)),
                "group_macro_delta": group_delta,
                "min_required_delta": float(min_delta),
                "enabled": use,
            }
        use_inferred = np.isin(signatures, enabled)
        gated = np.where(use_inferred[:, None], inferred, drop)
        macro = float(_per_query_f1(ctx, gated)[admitted].mean())
        row = {"theta": float(theta), "f1": macro,
               "enabled_signatures": enabled, "manifest": manifest}
        if best is None or (row["f1"], -abs(theta - 0.5)) > \
                (best["f1"], -abs(best["theta"] - 0.5)):
            best = row
    return best


def _apply_target_gate(ctx, certs, enabled_targets):
    gated = []
    for query, cert in zip(ctx["queries"], certs):
        copy = {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in cert.items()}
        statistical = np.flatnonzero(copy["uncertain_mask"] | copy["abstain_mask"])
        for row in statistical:
            residual = {atom.attr for atom in query["phi"].atoms
                        if ctx["observed"][atom.attr].isna().iat[row]}
            if not residual.issubset(enabled_targets):
                copy["uncertain_mask"][row] = False
                copy["abstain_mask"][row] = True
                copy["p_phi"][row] = 0.0
                copy["states"][row] = "false"
                copy["context_name"][row] = "policy_drop"
        gated.append(copy)
    return gated


def _learn_target_gate(ctx, certs, min_pairs=20):
    losses = {}
    for query, truth, cert in zip(ctx["queries"], ctx["truth_elig"], certs):
        statistical = np.flatnonzero(cert["uncertain_mask"] | cert["abstain_mask"])
        for row in statistical:
            residual = {atom.attr for atom in query["phi"].atoms
                        if ctx["observed"][atom.attr].isna().iat[row]}
            for attr in residual:
                item = losses.setdefault(attr, {"n": 0, "mifa_loss": 0.0,
                                                 "drop_loss": 0.0, "positives": 0})
                y = float(truth[row])
                item["n"] += 1
                item["positives"] += int(y)
                item["mifa_loss"] += float((cert["p_phi"][row] - y) ** 2)
                item["drop_loss"] += y  # (0-y)^2 for binary y
    enabled = []
    manifest = {}
    for attr, item in losses.items():
        n = item["n"]
        mifa = item["mifa_loss"] / n if n else 1.0
        drop = item["drop_loss"] / n if n else 0.0
        use = n >= min_pairs and mifa < drop
        if use:
            enabled.append(attr)
        manifest[attr] = {
            **item, "mifa_brier": mifa, "drop_brier": drop,
            "enabled": use, "validation_only": True,
        }
    return sorted(enabled), manifest


def _policy_diagnostics(ctx, certs):
    brier_values = []
    resolved = 0
    unresolved = 0
    local = 0
    body = 0
    support = []
    for truth, cert in zip(ctx["truth_elig"], certs):
        statistical = cert["uncertain_mask"] | cert["abstain_mask"]
        if statistical.any():
            brier_values.extend((cert["p_phi"][statistical] - truth[statistical]) ** 2)
        unresolved += int(statistical.sum())
        resolved += int(cert["uncertain_mask"].sum())
        local += int(cert["local_used"].sum())
        body += int(cert["body_used"].sum())
        support.extend(cert["support_n"][cert["uncertain_mask"]].tolist())
    coverage = resolved / unresolved if unresolved else 1.0
    return {
        "brier": float(np.mean(brier_values)) if brier_values else 0.0,
        "coverage": coverage,
        "local_fraction": local / resolved if resolved else 0.0,
        "body_fraction": body / resolved if resolved else 0.0,
        "mean_support": float(np.mean(support)) if support else 0.0,
        "unresolved_pairs": unresolved,
    }


def _select_policy(validation, support, support_cells, body_by_target, dtypes,
                   *, fast_grid=False, signature_gate_config=None):
    signature_gate_config = signature_gate_config or {}
    candidates = [
        {"location": "global", "scalar_body": False,
         "min_support": 30, "kappa": 0.0},
    ]
    if fast_grid:
        candidates.extend([
            {"location": "global", "scalar_body": True,
             "min_support": 10, "kappa": 0.0},
            {"location": "global", "scalar_body": True,
             "min_support": 30, "kappa": 10.0},
            {"location": "cell", "scalar_body": False,
             "min_support": 10, "kappa": 10.0},
            {"location": "cell", "scalar_body": True,
             "min_support": 10, "kappa": 10.0},
        ])
    else:
        candidates.extend(
            {"location": "global", "scalar_body": True,
             "min_support": support_min, "kappa": kappa}
            for support_min in (10, 30, 50) for kappa in (0.0, 10.0, 30.0)
        )
        candidates.extend(
            {"location": "cell", "scalar_body": scalar_body,
             "min_support": support_min, "kappa": kappa}
            for scalar_body in (False, True)
            for support_min in (10, 30, 50) for kappa in (0.0, 10.0, 30.0)
        )
    rows = []
    for policy in candidates:
        _, raw_certs = _joint_certificates(
            validation, support, support_cells, body_by_target, dtypes, policy)
        signature_gate = _learn_signature_gate(
            validation, raw_certs,
            min_queries=int(signature_gate_config.get("min_query_instances", 2)),
            min_source_groups=int(signature_gate_config.get("min_source_groups", 1)),
            min_delta=float(signature_gate_config.get("min_group_macro_delta", 0.0)))
        policy = {**policy,
                  "enabled_signatures": signature_gate["enabled_signatures"]}
        certs = _apply_signature_gate(
            validation, raw_certs, set(signature_gate["enabled_signatures"]))
        diagnostics = _policy_diagnostics(validation, certs)
        # Coverage is a declared admissibility constraint. A candidate below it
        # remains in the artifact but cannot win by abstaining on hard cases.
        raw_diagnostics = _policy_diagnostics(validation, raw_certs)
        admissible = (raw_diagnostics["coverage"] >= 0.80 and
                      bool(signature_gate["enabled_signatures"]))
        score = -signature_gate["f1"] if admissible else 10.0
        rows.append({"policy": policy, "diagnostics": diagnostics,
                     "raw_diagnostics": raw_diagnostics,
                     "signature_gate": signature_gate,
                     "admissible": admissible, "selection_score": score})
    selected = min(rows, key=lambda row: (row["selection_score"],
                                          -row["diagnostics"]["coverage"]))
    return selected["policy"], rows


def _calibrate_theta(calibration, support, support_cells, body_by_target, dtypes, policy):
    engine, certs = _joint_certificates(
        calibration, support, support_cells, body_by_target, dtypes, policy)
    curve = []
    for theta in np.linspace(0.0, 1.0, 101):
        eligible = [engine.eligible(cert, theta) for cert in certs]
        score = _score_eligibility(calibration, eligible)["macro"]
        curve.append({"theta": float(theta), **score})
    # Stable tie-break: higher precision, then threshold nearest 0.5.
    best = max(curve, key=lambda row: (row["f1"], row["precision"],
                                       -abs(row["theta"] - 0.5)))
    return best["theta"], curve


def _calibrate_query_cls(calibration, classifier):
    probabilities = classifier.probabilities(
        calibration["observed"], calibration["base"].normalized_embeddings(),
        calibration["queries"])
    curve = []
    for theta in np.linspace(0.0, 1.0, 101):
        score = _score_eligibility(
            calibration, classifier.eligible(probabilities, theta))["macro"]
        curve.append({"theta": float(theta), **score})
    best = max(curve, key=lambda row: (row["f1"], row["precision"],
                                       -abs(row["theta"] - 0.5)))
    return best["theta"], curve


def _fit_imputers(train_observed, dtypes, seed, include_slow):
    fitted = {"mean_mode": MeanModeImputer(dtypes).fit(train_observed)}
    timing = {}
    if include_slow:
        for name, kind in (("mice", "mice"), ("missforest", "missforest")):
            start = time.perf_counter()
            fitted[name] = MixedIterativeImputer(
                dtypes, kind=kind, seed=seed,
                max_iter=5, n_estimators=50, n_jobs=8,
            ).fit(train_observed)
            timing[name] = time.perf_counter() - start
    return fitted, timing


def _evaluate_test(ctx, support, support_cells, body_by_target, dtypes,
                   policy, theta, imputers, query_cls=None, query_cls_theta=None):
    results = {}
    n_queries = len(ctx["queries"])
    start = time.perf_counter()
    drop = np.asarray([
        query["phi"].eval_table(ctx["observed"]) == TRUE for query in ctx["queries"]
    ])
    results["drop"] = _score_eligibility(ctx, drop)
    elapsed = time.perf_counter() - start
    results["drop"]["online_batch_seconds"] = elapsed
    results["drop"]["online_ms_per_query"] = 1000.0 * elapsed / n_queries
    for name, imputer in imputers.items():
        start = time.perf_counter()
        completed = imputer.transform(ctx["observed"])
        materialization_seconds = time.perf_counter() - start
        start = time.perf_counter()
        results[name] = _score_eligibility(ctx, _table_policy(ctx, completed))
        elapsed = time.perf_counter() - start
        results[name]["materialization_seconds"] = materialization_seconds
        results[name]["online_batch_seconds"] = elapsed
        results[name]["online_ms_per_query"] = 1000.0 * elapsed / n_queries

    if query_cls is not None:
        start = time.perf_counter()
        probabilities = query_cls.probabilities(
            ctx["observed"], ctx["base"].normalized_embeddings(), ctx["queries"])
        eligible = query_cls.eligible(probabilities, query_cls_theta)
        results["query_cls"] = _score_eligibility(ctx, eligible)
        elapsed = time.perf_counter() - start
        results["query_cls"]["online_batch_seconds"] = elapsed
        results["query_cls"]["online_ms_per_query"] = 1000.0 * elapsed / n_queries
        results["query_cls"]["theta"] = float(query_cls_theta)

    start = time.perf_counter()
    engine, certs = _joint_certificates(
        ctx, support, support_cells, body_by_target, dtypes, policy, mode="joint")
    results["mifa_joint"] = _score_eligibility(
        ctx, [engine.eligible(cert, theta) for cert in certs])
    elapsed = time.perf_counter() - start
    results["mifa_joint"]["online_batch_seconds"] = elapsed
    results["mifa_joint"]["online_ms_per_query"] = 1000.0 * elapsed / n_queries
    results["mifa_joint"]["diagnostics"] = _policy_diagnostics(ctx, certs)

    global_policy = {"location": "global", "scalar_body": False,
                     "min_support": 30, "kappa": 0.0}
    for label, mode, use_policy in (
        ("global_joint", "joint", global_policy),
        ("marginal_product", "product", policy),
        ("min_marginal", "min", policy),
    ):
        start = time.perf_counter()
        variant, variant_certs = _joint_certificates(
            ctx, support, support_cells, body_by_target, dtypes,
            use_policy, mode=mode)
        results[label] = _score_eligibility(
            ctx, [variant.eligible(cert, theta) for cert in variant_certs])
        elapsed = time.perf_counter() - start
        results[label]["online_batch_seconds"] = elapsed
        results[label]["online_ms_per_query"] = 1000.0 * elapsed / n_queries
        results[label]["diagnostics"] = _policy_diagnostics(ctx, variant_certs)
    return results


def _discover_scalar_bodies(spec, train_observed, support_cells):
    """Freeze one statistical scalar body per target using training only."""
    rules = discover_rules(
        spec, train_observed, cell_of=support_cells,
        targets=spec.query_relevant_attrs, dep_conf=0.20,
        min_support=30, coarse_to_fine=True,
    )
    bodies = {}
    manifest = {}
    for target, candidates in rules.items():
        usable = []
        for rule in candidates:
            body = tuple(attr for attr in rule.X if attr != "cell")
            if not body:
                continue
            # Statistical use only: even a sample-perfect rule is not promoted
            # to a deterministic certificate by this audit harness.
            score = float(rule.confidence * np.log1p(rule.support) / len(body))
            usable.append((score, rule, body))
        if usable:
            _, rule, body = max(usable, key=lambda item: item[0])
            bodies[target] = body
            manifest[target] = {
                "body": list(body), "confidence": float(rule.confidence),
                "support": int(rule.support), "training_only": True,
                "deterministic": False,
            }
    return bodies, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--paper-config", default="configs/paper_v2.yaml")
    parser.add_argument("--emb", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument(
        "--missingness-kind", default="tuple_mcar",
        choices=("tuple_mcar", "tuple_mcar_nested",
                 "single_attribute_mcar_nested", "tuple_attr_skewed",
                 "tuple_predicate_mcar", "tuple_query_answer_mnar",
                 "joint_tuple_block"))
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument(
        "--block-attrs", default="",
        help=("comma-separated workload-correlated attributes masked together; "
              "empty keeps random tuple blocks"))
    parser.add_argument("--answer-impact-fraction", type=float, default=1.0)
    parser.add_argument("--answer-cells-per-tuple", type=int, default=1)
    parser.add_argument(
        "--attribute-weights", default="",
        help=("workload-frequency weights as attr=weight comma pairs; "
              "required for tuple_attr_skewed unless declared by the query source"))
    parser.add_argument("--n-test-queries", type=int, default=300)
    parser.add_argument("--n-validation-queries", type=int, default=200)
    parser.add_argument("--n-calibration-queries", type=int, default=100)
    parser.add_argument(
        "--workload-profile", choices=("stratified", "canonical"),
        default="stratified")
    parser.add_argument(
        "--workload-reference", choices=("masked_train", "complete_train"),
        default="masked_train",
        help=("evaluator-only table used to freeze predicate values and "
              "selectivity; complete_train is for paired rho sweeps and is "
              "never exposed to inference"))
    parser.add_argument("--skip-slow-imputers", action="store_true")
    parser.add_argument("--skip-query-cls", action="store_true")
    parser.add_argument("--fast-policy-grid", action="store_true")
    parser.add_argument("--tag", default="audit")
    args = parser.parse_args()
    block_attrs = tuple(
        item.strip() for item in args.block_attrs.split(",") if item.strip())
    if block_attrs and args.missingness_kind not in {
            "single_attribute_mcar_nested", "tuple_predicate_mcar",
            "tuple_query_answer_mnar",
            "joint_tuple_block"}:
        raise ValueError(
            "--block-attrs requires single_attribute_mcar_nested, "
            "tuple_predicate_mcar, "
            "tuple_query_answer_mnar, or "
            "joint_tuple_block missingness")
    if (block_attrs and args.missingness_kind == "joint_tuple_block" and
            len(block_attrs) != args.block_size):
        raise ValueError("--block-attrs count must equal --block-size")
    if args.missingness_kind == "tuple_predicate_mcar" and not block_attrs:
        raise ValueError("tuple_predicate_mcar requires --block-attrs")
    if (args.missingness_kind == "single_attribute_mcar_nested" and
            len(block_attrs) != 1):
        raise ValueError(
            "single_attribute_mcar_nested requires one --block-attrs value")
    if not 0.0 <= args.answer_impact_fraction <= 1.0:
        raise ValueError("--answer-impact-fraction must be in [0, 1]")

    paper_config = yaml.safe_load((ROOT / args.paper_config).read_text())
    dataset_config = yaml.safe_load((ROOT / args.dataset_config).read_text())
    if args.emb:
        dataset_config["dataset"]["emb_file"] = f"emb_{args.emb}.npy"
    dataset_config["seed"] = args.seed
    spec = load_dataset(dataset_config)
    source_cfg = paper_config.get("query_sources", {}).get(spec.name, {})
    attrs = list(source_cfg.get("predicate_attrs", spec.query_relevant_attrs))
    unknown_attrs = set(attrs) - set(spec.dtypes)
    if unknown_attrs:
        raise ValueError(f"query source declares unknown predicate attrs: {sorted(unknown_attrs)}")
    spec.query_relevant_attrs = attrs
    attribute_weights, attribute_weights_manifest = _resolve_attribute_weights(
        args.missingness_kind, args.attribute_weights,
        source_cfg.get("attribute_missingness_weights"), attrs)
    signature_mix = None
    if args.workload_profile == "canonical":
        signature_mix = source_cfg.get("canonical_signatures")
        if not signature_mix:
            raise ValueError(
                f"{spec.name}: canonical workload has no predeclared signatures")
    canonical_bucket_weights = tuple(source_cfg.get(
        "canonical_selectivity_weights",
        paper_config["workload"].get(
            "canonical_selectivity_weights", (0.5, 0.5, 0.0))))

    n_query_cls_queries = _query_cls_training_reservation(
        paper_config, args.n_validation_queries,
        enabled=not args.skip_query_cls)
    role_counts = {
        "query_cls_training": n_query_cls_queries,
        "validation": args.n_validation_queries,
        "calibration": args.n_calibration_queries,
        "test": args.n_test_queries,
    }
    spec, query_specs, native_filters, query_source_manifest = _prepare_query_sources(
        spec, paper_config, role_counts=role_counts,
        seed=int(paper_config["split"]["seed"]) + args.seed * 100003,
        embedding_tag=args.emb or None)
    spec.query_relevant_attrs = attrs
    _progress("dataset_ready", dataset=spec.name, n=spec.n, d=spec.d,
              query_source=query_source_manifest["kind"])

    split = make_four_way_split(spec.n, seed=paper_config["split"]["seed"])
    parts = {name: subset_spec(spec, rows, name=f"{spec.name}_{name}")
             for name, rows in split.partitions().items()}
    for part in parts.values():
        part.query_relevant_attrs = attrs

    train_missingness_kind = training_missingness_kind(
        args.missingness_kind, args.answer_cells_per_tuple)
    train_block_attrs = (
        block_attrs or tuple(attrs)
        if args.missingness_kind == "tuple_query_answer_mnar"
        else block_attrs)
    train_observed, train_mask, _ = make_missing(
        parts["train"], args.rho, train_missingness_kind,
        seed=args.seed * 100 + 1, block_size=args.block_size,
        block_attrs=train_block_attrs, usage_freq=attribute_weights,
        answer_cells_per_tuple=args.answer_cells_per_tuple)
    predicate_reference = subset_spec(
        parts["train"], np.arange(parts["train"].n),
        name=f"{spec.name}_training_predicate_reference")
    # Baseline fitting continues to use ``train_observed``; this copy is read
    # only by deterministic query/predicate construction.
    predicate_reference.table = (
        parts["train"].table.copy()
        if args.workload_reference == "complete_train"
        else train_observed.copy())
    X_train = parts["train"].normalized_embeddings()
    cell_index = CoarseCellIndex(pca_dim=16, n_cells=256, seed=args.seed).build(X_train)
    support_cells = cell_index.cell_of
    body_by_target, body_manifest = _discover_scalar_bodies(
        parts["train"], train_observed, support_cells)
    _progress("training_support_ready", scalar_bodies=len(body_manifest))

    validation = _prepare_eval(
        parts["validation"], train_observed, cell_index,
        role="validation", n_queries=args.n_validation_queries,
        k=paper_config["workload"]["k_default"], rho=args.rho,
        mask_seed=args.seed * 100 + 2, workload_seed=args.seed * 100 + 12, attrs=attrs,
        predicate_reference=predicate_reference, missingness_kind=args.missingness_kind,
        block_size=args.block_size, block_attrs=block_attrs,
        attribute_weights=attribute_weights,
        answer_impact_fraction=args.answer_impact_fraction,
        answer_cells_per_tuple=args.answer_cells_per_tuple,
        query_spec=query_specs.get("validation"),
        native_filters=native_filters.get("validation"),
        signature_mix=signature_mix,
        canonical_bucket_weights=canonical_bucket_weights)
    calibration = _prepare_eval(
        parts["calibration"], train_observed, cell_index,
        role="calibration", n_queries=args.n_calibration_queries,
        k=paper_config["workload"]["k_default"], rho=args.rho,
        mask_seed=args.seed * 100 + 3, workload_seed=args.seed * 100 + 13, attrs=attrs,
        predicate_reference=predicate_reference, missingness_kind=args.missingness_kind,
        block_size=args.block_size, block_attrs=block_attrs,
        attribute_weights=attribute_weights,
        answer_impact_fraction=args.answer_impact_fraction,
        answer_cells_per_tuple=args.answer_cells_per_tuple,
        query_spec=query_specs.get("calibration"),
        native_filters=native_filters.get("calibration"),
        signature_mix=signature_mix,
        canonical_bucket_weights=canonical_bucket_weights)
    test = _prepare_eval(
        parts["test"], train_observed, cell_index,
        role="test", n_queries=args.n_test_queries,
        k=paper_config["workload"]["k_default"], rho=args.rho,
        mask_seed=args.seed * 100 + 4, workload_seed=args.seed * 100 + 14, attrs=attrs,
        predicate_reference=predicate_reference, missingness_kind=args.missingness_kind,
        block_size=args.block_size, block_attrs=block_attrs,
        attribute_weights=attribute_weights,
        answer_impact_fraction=args.answer_impact_fraction,
        answer_cells_per_tuple=args.answer_cells_per_tuple,
        query_spec=query_specs.get("test"),
        native_filters=native_filters.get("test"),
        signature_mix=signature_mix,
        canonical_bucket_weights=canonical_bucket_weights)
    headline_cfg = paper_config["position"].get("headline_admission")
    if headline_cfg:
        min_joint_support = headline_cfg["min_training_joint_support"]
        for context in (validation, calibration, test):
            _attach_headline_admission(
                context, train_observed, min_joint_support=min_joint_support)
    _progress("workloads_ready", validation=len(validation["queries"]),
              calibration=len(calibration["queries"]), test=len(test["queries"]))

    policy, policy_rows = _select_policy(
        validation, train_observed, support_cells, body_by_target, spec.dtypes,
        fast_grid=args.fast_policy_grid,
        signature_gate_config=paper_config.get("inference", {}).get("validation_gate"))
    _progress("policy_selected", policy=policy)
    theta, calibration_curve = _calibrate_theta(
        calibration, train_observed, support_cells, body_by_target, spec.dtypes, policy)
    imputers, fit_seconds = _fit_imputers(
        train_observed, spec.dtypes, args.seed, not args.skip_slow_imputers)
    _progress("imputers_fitted", methods=sorted(imputers), seconds=fit_seconds)

    query_cls = None
    query_cls_theta = None
    query_cls_curve = []
    query_cls_manifest = {"skipped": True}
    if not args.skip_query_cls:
        query_cls_start = time.perf_counter()
        if "query_cls_training" in query_specs:
            query_cls_query = query_specs["query_cls_training"]
            query_cls_base = parts["train"]
            query_cls_observed = train_observed
            query_cls_mask = train_mask
        else:
            n_query_cls_queries = min(
                n_query_cls_queries, max(4, parts["train"].n // 5))
            query_cls_query, query_cls_base, _, _ = _holdout_query_sources(
                parts["train"], n_query_cls_queries, args.seed * 100 + 7109,
                "query_cls_training")
            query_cls_observed, query_cls_mask, _ = make_missing(
                query_cls_base, args.rho, args.missingness_kind,
                seed=args.seed * 100 + 71, block_size=args.block_size,
                block_attrs=block_attrs, usage_freq=attribute_weights)
        query_cls_base.query_relevant_attrs = attrs
        query_cls_queries = _make_workload(
            query_cls_query, query_cls_base, n_query_cls_queries,
            paper_config["workload"]["k_default"], seed=args.seed * 100 + 72,
            attrs=attrs, predicate_reference=predicate_reference,
            native_filters=native_filters.get("query_cls_training"),
            signature_mix=signature_mix,
            canonical_bucket_weights=canonical_bucket_weights)
        query_cls = QueryConditionedClassifier(
            spec.dtypes, cell_index.pca_, seed=args.seed).fit(
                query_cls_base.table, query_cls_observed,
                query_cls_base.normalized_embeddings(), query_cls_queries)
        fit_seconds["query_cls"] = time.perf_counter() - query_cls_start
        query_cls_theta, query_cls_curve = _calibrate_query_cls(
            calibration, query_cls)
        query_cls_manifest = {
            **query_cls.manifest,
            "mask_digest": _mask_digest(query_cls_mask),
            "workload_digest": workload_digest(query_cls_queries),
            "source_disjoint": True,
            "theta_source": "calibration_only",
        }
        _progress("query_cls_fitted", theta=query_cls_theta,
                  training_pairs=query_cls.manifest["training_pairs"],
                  seconds=fit_seconds["query_cls"])
    results = _evaluate_test(
        test, train_observed, support_cells, body_by_target, spec.dtypes,
        policy, theta, imputers, query_cls, query_cls_theta)
    _progress("test_evaluated", methods=sorted(results))

    primary_metric_block = paper_config["position"].get("primary_metric_block", "macro")
    best_baseline = max(
        (name for name in results if name not in {"mifa_joint", "global_joint",
                                                  "marginal_product", "min_marginal"}),
        key=lambda name: results[name][primary_metric_block]["f1"],
    )
    mifa_f1 = results["mifa_joint"][primary_metric_block]["f1"]
    broad_best_baseline = max(
        (name for name in results if name not in {"mifa_joint", "global_joint",
                                                  "marginal_product", "min_marginal"}),
        key=lambda name: results[name]["macro"]["f1"],
    )
    payload = {
        "status": paper_config["protocol"]["status"],
        # Keep the report self-describing.  Digest alignment proves that two
        # methods saw the same data, while the arguments below preserve the
        # intended protocol (including evaluator-only workload_reference) even
        # when the report is copied without its launcher log.
        "experiment_arguments": vars(args),
        "dataset": {"name": spec.name, "n": spec.n, "d": spec.d,
                    "hash": spec.hash(), "attrs": attrs},
        "protocol": paper_config["protocol"],
        "workload_profile": args.workload_profile,
        "query_source": query_source_manifest,
        "seed": args.seed,
        "rho_tuple": args.rho,
        "missingness": {
            "kind": args.missingness_kind,
            "training_kind": train_missingness_kind,
            "answer_impact_fraction": (
                float(args.answer_impact_fraction)
                if args.missingness_kind == "tuple_query_answer_mnar"
                else None),
            "answer_cells_per_tuple": (
                int(args.answer_cells_per_tuple)
                if args.missingness_kind == "tuple_query_answer_mnar"
                else None),
            "block_size": (args.block_size
                           if args.missingness_kind == "joint_tuple_block"
                           else int(args.answer_cells_per_tuple)
                           if args.missingness_kind == "tuple_query_answer_mnar"
                           else 1),
            "block_attrs": list(block_attrs),
            "attribute_weights": attribute_weights_manifest,
        },
        "split": {**split.as_dict(include_indices=False), "digest": split.digest()},
        "train_mask": {
            "digest": _mask_digest(train_mask),
            "incomplete_tuple_rate": float(train_mask.any(axis=1).mean()),
            "masked_cell_rate": float(train_mask.mean()),
        },
        "partitions": {
            ctx["role"]: {
                "base_n": ctx["base"].n,
                "query_n": len(ctx["queries"]),
                "mask_digest": ctx["mask_digest"],
                "workload_digest": ctx["workload_digest"],
                "workload_counts": ctx["workload_counts"],
                "incomplete_tuple_rate": ctx["incomplete_tuple_rate"],
                "masked_cell_rate": ctx["masked_cell_rate"],
                "query_source_corpus": ctx["query_source_corpus"],
                "headline_admission": ctx.get("headline_admission"),
                "masked_answer_affected_queries": int(
                    ctx["masked_answer_affected"].sum()),
                "masked_answer_affected_digest": hashlib.sha256(
                    np.packbits(ctx["masked_answer_affected"]).tobytes()).hexdigest(),
            } for ctx in (validation, calibration, test)
        },
        "selected_policy": policy,
        "training_scalar_bodies": body_manifest,
        "validation_candidates": policy_rows,
        "selected_theta": theta,
        "calibration_curve": calibration_curve,
        "query_cls_calibration_curve": query_cls_curve,
        "query_cls_manifest": query_cls_manifest,
        "fit_seconds": fit_seconds,
        "results": results,
        "win_check": {
            "regime": paper_config["position"]["primary_regime"],
            "metric_block": primary_metric_block,
            "best_missing_value_baseline": best_baseline,
            **_win_check(
                mifa_f1,
                results[best_baseline][primary_metric_block]["f1"],
                required_gain_pp=paper_config["position"].get(
                    "required_absolute_gain_pp", 5.0)),
        },
        "scope_check": {
            "regime": paper_config["position"].get("scope_check_regime", "macro"),
            "best_missing_value_baseline": broad_best_baseline,
            "best_baseline_f1": results[broad_best_baseline]["macro"]["f1"],
            "mifa_f1": results["mifa_joint"]["macro"]["f1"],
            "delta": (results["mifa_joint"]["macro"]["f1"] -
                      results[broad_best_baseline]["macro"]["f1"]),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "knn_backend": active_backend(),
            "masked_topk_backend": "numba_fused_d2",
            "joint_support_backend": "numba_fused_grouped_context",
            "jit_warmup_in_timing": False,
            "git_head": _git_head(),
            "source_provenance": _source_provenance(),
            "paper_config_hash": _sha_json(paper_config),
            "dataset_config_hash": _sha_json(dataset_config),
        },
    }
    report_stem = paper_config["artifacts"].get("report_stem", "paper_v2_phase_a")
    out = ROOT / "reports" / f"{report_stem}_{spec.name}_seed{args.seed}_{args.tag}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"report": str(out), "policy": policy, "theta": theta,
                      "win_check": payload["win_check"]}, indent=2))


if __name__ == "__main__":
    main()
