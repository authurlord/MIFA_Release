#!/usr/bin/env python3
"""Build unbiased, nested, unique-row arXiv scale contexts.

The official 2,735,264-row release is projected to the same normalized d=128
prefix used by the existing scale runner.  A single fixed permutation of the
entire release defines nested 20K/100K/1M (and optional larger) corpora.  Each
N receives its own HCE rho=0.20 mask and complete-data exact top-k truth; no
row repetition or vector jitter is used.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import arxiv_large_static_runner as static
import experiment_v2 as legacy
from mifa.datasets import DatasetSpec, load_dataset
from mifa.protocol import make_four_way_split, subset_spec
from mifa.scale_cohort import project_normalize


OFFICIAL_ROWS = 2_735_264
NATIVE_DIMENSION = 4096
PROJECTION_DIMENSION = 128
PERMUTATION_SEED = 0x78C38D41
PAPER_CONFIG = ROOT / "configs/paper_three_regime.yaml"
MEDIUM_CONFIG = ROOT / "configs/arxiv_medium.yaml"
ATTRS = ("update_date", "number_of_sub_categories")
DTYPES = {"update_date": "num", "number_of_sub_categories": "num"}


def _fvec_shape(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        dimension = int(np.frombuffer(stream.read(4), dtype="<i4")[0])
    row_bytes = (dimension + 1) * 4
    if size % row_bytes:
        raise RuntimeError(f"misaligned fvecs file: {path}")
    return size // row_bytes, dimension


def _project_fvecs(source: Path, target: Path, dimension: int,
                   chunk_rows: int) -> tuple[int, int]:
    rows, native_dimension = _fvec_shape(source)
    expected_shape = (rows, int(dimension))
    if target.is_file():
        existing = np.load(target, mmap_mode="r")
        if existing.shape != expected_shape or existing.dtype != np.float32:
            raise RuntimeError(f"existing projected array mismatch: {target}")
        return rows, native_dimension
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".building")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=expected_shape)
    raw = np.memmap(
        source, mode="r", dtype="<i4", shape=(rows, native_dimension + 1))
    started = time.monotonic()
    for start in range(0, rows, int(chunk_rows)):
        end = min(rows, start + int(chunk_rows))
        headers = raw[start:end, 0]
        if not np.all(headers == native_dimension):
            raise RuntimeError(f"fvec dimension header mismatch at row {start}")
        values = np.asarray(
            raw[start:end, 1:dimension + 1].view("<f4"), dtype=np.float32)
        values = np.ascontiguousarray(values)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        output[start:end] = values / norms
        if end == rows or end % (chunk_rows * 16) == 0:
            rate = end / max(time.monotonic() - started, 1e-9)
            print(
                f"project rows={end}/{rows} rows_per_second={rate:.1f}",
                flush=True,
            )
    output.flush()
    del output, raw
    temporary.replace(target)
    return rows, native_dimension


def _flatten_attributes(source: Path, target: Path, expected_rows: int) -> None:
    if target.is_file():
        existing = pd.read_parquet(target, columns=list(DTYPES))
        if len(existing) != expected_rows:
            raise RuntimeError(f"existing attribute table mismatch: {target}")
        return
    columns: dict[str, list] = {
        "main_category": [], "update_date": [], "number_of_authors": [],
        "number_of_versions": [], "number_of_main_categories": [],
        "number_of_sub_categories": [], "has_comments": [], "license": [],
    }
    with source.open() as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            main = record.get("main_categories") or [None]
            columns["main_category"].append(main[0] if main else None)
            columns["update_date"].append(record.get("update_date"))
            columns["number_of_authors"].append(record.get("number_of_authors"))
            columns["number_of_versions"].append(record.get("number_of_versions"))
            columns["number_of_main_categories"].append(
                record.get("number_of_main_categories"))
            columns["number_of_sub_categories"].append(
                record.get("number_of_sub_categories"))
            columns["has_comments"].append(bool(record.get("has_comments")))
            columns["license"].append(record.get("license"))
            if (index + 1) % 250_000 == 0:
                print(f"attributes rows={index + 1}/{expected_rows}", flush=True)
    table = pd.DataFrame(columns)
    if len(table) != expected_rows:
        raise RuntimeError(
            f"attribute/vector row mismatch: {len(table)} != {expected_rows}")
    temporary = target.with_suffix(target.suffix + ".building")
    table.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(target)


def build_corpus(raw_dir: Path, output_dir: Path, *, chunk_rows: int,
                 require_official_rows: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, native_dimension = _project_fvecs(
        raw_dir / "database_vectors.fvecs",
        output_dir / "database_embeddings_d128.npy",
        PROJECTION_DIMENSION, chunk_rows)
    if require_official_rows and rows != OFFICIAL_ROWS:
        raise RuntimeError(f"official row count mismatch: {rows} != {OFFICIAL_ROWS}")
    _flatten_attributes(
        raw_dir / "database_attributes.jsonl",
        output_dir / "complete_attributes.parquet", rows)
    query_rows, query_dimension = _project_fvecs(
        raw_dir / "query_vectors.fvecs",
        output_dir / "query_embeddings_d128.npy",
        PROJECTION_DIMENSION, chunk_rows)
    permutation_path = output_dir / "global_permutation.npy"
    if not permutation_path.is_file():
        permutation = np.random.default_rng(PERMUTATION_SEED).permutation(rows)
        np.save(permutation_path, permutation.astype(np.int64, copy=False))
    else:
        permutation = np.load(permutation_path, mmap_mode="r")
        if len(permutation) != rows:
            raise RuntimeError("existing global permutation length mismatch")
    manifest = {
        "source": str(raw_dir.resolve()),
        "rows": int(rows),
        "native_dimension": int(native_dimension),
        "query_rows": int(query_rows),
        "query_native_dimension": int(query_dimension),
        "projection": "normalized fixed 128-dimensional prefix",
        "dimension": PROJECTION_DIMENSION,
        "permutation_seed": PERMUTATION_SEED,
        "extension": "one global unique-row permutation; N are nested prefixes",
        "repetition": False,
        "jitter": False,
    }
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def _medium_workload() -> tuple[DatasetSpec, dict, dict, DatasetSpec, list, list]:
    paper = yaml.safe_load(PAPER_CONFIG.read_text())
    dataset = yaml.safe_load(MEDIUM_CONFIG.read_text())
    dataset["seed"] = 0
    spec = load_dataset(dataset)
    source_cfg = paper["query_sources"]["arxiv"]
    attrs = list(source_cfg["predicate_attrs"])
    spec.query_relevant_attrs = attrs
    role_counts = {
        "query_cls_training": 100,
        "validation": 200,
        "calibration": 100,
        "selection_reservation": 0,
        "test": 1000,
    }
    spec, query_specs, native_filters, _ = legacy._prepare_query_sources(
        spec, paper, role_counts=role_counts,
        seed=int(paper["split"]["seed"]), embedding_tag="native")
    fraction_cfg = paper["split"].get("fractions", {})
    fractions = tuple(float(fraction_cfg.get(name, default)) for name, default in (
        ("train", 0.50), ("validation", 0.20),
        ("calibration", 0.10), ("test", 0.20)))
    split = make_four_way_split(
        spec.n, seed=int(paper["split"]["seed"]), fractions=fractions)
    train = subset_spec(spec, split.train, name="arxiv_train_reference")
    train.query_relevant_attrs = attrs
    predicate_reference = subset_spec(
        train, np.arange(train.n), name="arxiv_complete_train_reference")
    predicate_reference.table = train.table.copy()
    signature_mix = source_cfg["canonical_signatures"]
    bucket_weights = tuple(source_cfg.get(
        "canonical_selectivity_weights",
        paper["workload"].get("canonical_selectivity_weights", (0.5, 0.5, 0.0))))
    return (spec, query_specs, native_filters, predicate_reference,
            signature_mix, list(bucket_weights))


def _project_query_spec(spec: DatasetSpec) -> DatasetSpec:
    return DatasetSpec(
        name=spec.name,
        A_e=project_normalize(spec.A_e, PROJECTION_DIMENSION),
        table=spec.table.copy(), dtypes=dict(spec.dtypes), metric=spec.metric,
        query_relevant_attrs=list(spec.query_relevant_attrs),
        meta=dict(spec.meta),
    )


def build_contexts(output_dir: Path, sizes: tuple[int, ...], *,
                   device: str, distance_batch_size: int) -> None:
    full_embeddings = np.load(
        output_dir / "database_embeddings_d128.npy", mmap_mode="r")
    complete_attributes = pd.read_parquet(
        output_dir / "complete_attributes.parquet")
    permutation = np.load(
        output_dir / "global_permutation.npy", mmap_mode="r")
    if len(full_embeddings) != len(complete_attributes) or len(permutation) != len(full_embeddings):
        raise RuntimeError("processed corpus assets are not row aligned")

    (_medium, query_specs, native_filters, predicate_reference,
     signature_mix, bucket_weights) = _medium_workload()
    historical_vectors, historical_descriptors = static._historical_query_bundle(
        query_specs["query_cls_training"],
        native_filters["query_cls_training"], list(ATTRS), signature_mix)
    historical_vectors = project_normalize(
        historical_vectors, PROJECTION_DIMENSION)
    projected_query_specs = {
        role: _project_query_spec(query_specs[role])
        for role in ("validation", "test")
    }

    for size in sizes:
        if size < 1 or size > len(permutation):
            raise ValueError(f"invalid unique scale size: {size}")
        context_dir = output_dir / f"context_n{size}"
        context_dir.mkdir(parents=True, exist_ok=True)
        row_ids = np.asarray(permutation[:size], dtype=np.int64)
        embeddings = np.ascontiguousarray(full_embeddings[row_ids], dtype=np.float32)
        table = complete_attributes.iloc[row_ids].reset_index(drop=True)
        np.save(context_dir / "embeddings_d128.npy", embeddings)
        np.save(context_dir / "source_row_ids.npy", row_ids)
        table.to_pickle(context_dir / "complete_table.pkl")
        for role, query_count, mask_seed, workload_seed in (
            ("validation", 200, 2, 12), ("test", 1000, 4, 14)):
            role_dir = context_dir / role
            role_dir.mkdir(parents=True, exist_ok=True)
            base = DatasetSpec(
                name=f"arxiv_large_unique_n{size}_{role}", A_e=embeddings,
                table=table, dtypes=dict(DTYPES), metric="cosine",
                query_relevant_attrs=list(ATTRS),
                meta={"source": "SPCL/arxiv-for-fanns-large",
                      "global_permutation_seed": PERMUTATION_SEED},
            )
            context = static._prepare_eval(
                base, role=role, n_queries=query_count, k=10, rho=0.20,
                mask_seed=mask_seed, workload_seed=workload_seed,
                attrs=list(ATTRS), predicate_reference=predicate_reference,
                missingness_kind="historical_candidate_exposure",
                block_size=2, block_attrs=ATTRS, attribute_weights=None,
                answer_impact_fraction=1.0, answer_cells_per_tuple=1,
                query_spec=projected_query_specs[role],
                native_filters=native_filters[role],
                signature_mix=signature_mix,
                canonical_bucket_weights=tuple(bucket_weights),
                historical_query_vectors=historical_vectors,
                historical_query_descriptors=historical_descriptors,
                historical_top_l=100, historical_beta_exp=0.50,
                historical_device=device,
                historical_batch_size=distance_batch_size,
            )
            np.save(role_dir / "mask.npy", context["mask"])
            np.save(role_dir / "truth_ids.npy", context["truth_ids"])
            np.save(
                role_dir / "masked_answer_affected.npy",
                context["masked_answer_affected"])
            context["observed"].to_pickle(role_dir / "observed_table.pkl")
            with (role_dir / "queries.pkl").open("wb") as stream:
                pickle.dump(
                    context["queries"], stream,
                    protocol=pickle.HIGHEST_PROTOCOL)
            manifest = {
                "schema_version": 1,
                "kind": "arxiv-large-global-permutation-unique-prefix",
                "role": role,
                "n": int(size), "dimension": PROJECTION_DIMENSION,
                "query_count": len(context["queries"]), "k": 10,
                "rho": 0.20,
                "missingness_kind": "historical_candidate_exposure",
                "historical_beta_exp": 0.50,
                "historical_top_l": 100,
                "block_attrs": list(ATTRS),
                "permutation_seed": PERMUTATION_SEED,
                "repetition": False, "jitter": False,
                "embeddings_file": "../embeddings_d128.npy",
                "source_row_ids_file": "../source_row_ids.npy",
                "complete_table_file": "../complete_table.pkl",
                "observed_table_file": "observed_table.pkl",
                "mask_file": "mask.npy",
                "queries_file": "queries.pkl",
                "truth_ids_file": "truth_ids.npy",
                "masked_answer_affected_file": (
                    "masked_answer_affected.npy"),
                "mask_digest": context["mask_digest"],
                "workload_digest": context["workload_digest"],
                "workload_counts": context["workload_counts"],
                "affected_query_count_direct": context[
                    "affected_query_count_direct"],
                "affected_query_rate_direct": context[
                    "affected_query_rate_direct"],
                "masked_cell_rate": context["masked_cell_rate"],
                "incomplete_tuple_rate": context["incomplete_tuple_rate"],
            }
            (role_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(manifest), flush=True)
            del context, base
        del embeddings, table, row_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path,
        default=ROOT / "data/raw/arxiv-for-fanns-large")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/processed/arxiv_large_unique_d128")
    parser.add_argument("--stage", choices=("corpus", "contexts", "all"), default="all")
    parser.add_argument("--sizes", default="20000,100000,1000000")
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distance-batch-size", type=int, default=16)
    parser.add_argument("--allow-nonofficial-row-count", action="store_true")
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value.strip())
    if args.stage in {"corpus", "all"}:
        build_corpus(
            args.raw_dir.resolve(), args.output_dir.resolve(),
            chunk_rows=args.chunk_rows,
            require_official_rows=not args.allow_nonofficial_row_count)
    if args.stage in {"contexts", "all"}:
        build_contexts(
            args.output_dir.resolve(), sizes,
            device=args.device,
            distance_batch_size=args.distance_batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
