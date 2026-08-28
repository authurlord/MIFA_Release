"""DatasetSpec + loaders + D* construction.

This module lives inside the ``mifa.datasets`` package so the formal dataset
API and the newer dataset adapters can coexist without Python import shadowing.

A DatasetSpec bundles everything a run needs:
  - A_e:  (n, d) float32 embeddings
  - table: pandas DataFrame of scalar attributes (mixed numerical/categorical)
  - dtypes: {col: 'num'|'cat'}
  - metric: 'l2' | 'cosine'
  - query_relevant_attrs: list[str]

D* construction: drop tuples whose query-relevant attrs are *originally* missing,
so the remaining relation is complete ground truth. Synthetic data has no original
missingness, so D* == full relation there.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetSpec:
    name: str
    A_e: np.ndarray                      # (n, d) float32
    table: pd.DataFrame                  # scalar attributes
    dtypes: dict                         # col -> 'num' | 'cat'
    metric: str                          # 'l2' | 'cosine'
    query_relevant_attrs: list
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.A_e.shape[0]

    @property
    def d(self) -> int:
        return self.A_e.shape[1]

    def normalized_embeddings(self) -> np.ndarray:
        """For cosine, L2-normalize so that L2 ranking == cosine ranking."""
        if self.metric == "cosine":
            norms = np.linalg.norm(self.A_e, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (self.A_e / norms).astype(np.float32)
        return self.A_e.astype(np.float32)

    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.A_e.tobytes())
        h.update(pd.util.hash_pandas_object(self.table, index=True).values.tobytes())
        return h.hexdigest()[:16]


def make_synthetic(n=20000, d=64, n_clusters=40, metric="l2", seed=0,
                   sep=3.0, noise=1.0, label_noise=0.0, score_signal=1.0) -> DatasetSpec:
    """Self-contained dataset with KNOWN imputability structure.

    Difficulty knobs (Codex review): lower `sep` / higher `noise` makes clusters
    overlap; `label_noise` flips a fraction of cluster_label; `score_signal` in
    [0,1] mixes embedding signal with independent noise for latent_score.

    Attributes:
    - cluster_label : categorical from the Gaussian blob (+ label_noise flips)
                      -> embedding-imputable unless made hard.
    - latent_score  : score_signal * f(cluster) + (1-score_signal) * noise
                      -> tunably imputable.
    - random_tag    : categorical drawn independently of the embedding
                      -> NOT imputable (negative control, should FAIL Gate 1).
    - random_cat40  : independent categorical with the SAME cardinality as
                      cluster_label -> matched-cardinality negative control.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, size=(n_clusters, d)).astype(np.float32) * sep
    true_labels = rng.integers(0, n_clusters, size=n)
    A_e = (centers[true_labels]
           + rng.normal(0, noise, size=(n, d)).astype(np.float32)).astype(np.float32)

    # observed cluster_label: optionally corrupted by label noise
    labels = true_labels.copy()
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        labels[flip] = rng.integers(0, n_clusters, size=int(flip.sum()))

    # latent_score: blend embedding-derived signal with independent noise
    cluster_score = np.linspace(0, 100, n_clusters)[true_labels]
    indep = rng.uniform(0, 100, size=n)
    latent_score = (score_signal * cluster_score
                    + (1.0 - score_signal) * indep
                    + rng.normal(0, 3.0, size=n))

    random_tag = rng.integers(0, 8, size=n)
    random_cat40 = rng.integers(0, n_clusters, size=n)  # matched cardinality control

    table = pd.DataFrame({
        "cluster_label": labels.astype("int64"),
        "latent_score": latent_score.astype("float64"),
        "random_tag": random_tag.astype("int64"),
        "random_cat40": random_cat40.astype("int64"),
    })
    dtypes = {"cluster_label": "cat", "latent_score": "num",
              "random_tag": "cat", "random_cat40": "cat"}
    qra = ["cluster_label", "latent_score", "random_tag", "random_cat40"]
    return DatasetSpec(
        name="synthetic", A_e=A_e, table=table, dtypes=dtypes, metric=metric,
        query_relevant_attrs=qra,
        meta={"n_clusters": n_clusters, "sep": sep, "noise": noise,
              "label_noise": label_noise, "score_signal": score_signal},
    )


def make_synthetic_deps(n=20000, d=64, n_clusters=20, metric="l2", seed=0):
    """Synthetic data with PLANTED dependencies, to test rule-based inference.

    Signal structure (deliberately separates scalar-evidence from embedding-evidence):
      EMBEDDING-imputable (the regime where embedding-kNN already wins):
        - cluster_label : the Gaussian blob id
        - category      : a function of the blob (so embedding predicts it)
      SCALAR-imputable, NOT embedding-imputable (where rule-MIFA must win):
        - zip   : categorical, INDEPENDENT of the embedding
        - city  : = f(zip)  (FUNCTIONAL dependency zip->city, 100% -> Sigma_ind)
        - region: = g(zip)  (also functional)
        - brand : depends on (category, price_bucket)  (Sigma_dep, high but <100%)
      price : numerical, drawn from a category-specific range (category->price interval)
      random_tag : independent noise (negative control)

    Because city/region are uncorrelated with the embedding, embedding-kNN imputes
    them poorly, while the rule zip->city recovers them exactly when zip is observed.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, size=(n_clusters, d)).astype(np.float32) * 3.0
    blob = rng.integers(0, n_clusters, size=n)
    A_e = (centers[blob] + rng.normal(0, 1.0, size=(n, d)).astype(np.float32)).astype(np.float32)

    # embedding-derived category (each blob -> a category distribution, mostly 1:1)
    n_cat = 8
    blob_to_cat = rng.integers(0, n_cat, size=n_clusters)
    category = blob_to_cat[blob]

    # zip independent of embedding; functional zip->city, zip->region
    n_zip = 200
    zip_code = rng.integers(0, n_zip, size=n)
    zip_to_city = rng.integers(0, 40, size=n_zip)      # functional map
    zip_to_region = rng.integers(0, 5, size=n_zip)
    zip_to_score = rng.uniform(0, 100, size=n_zip)
    city = zip_to_city[zip_code]
    region = zip_to_region[zip_code]
    zip_score = zip_to_score[zip_code]

    # price ~ category-specific range (category -> price interval, Sigma_dep numeric)
    cat_base = np.linspace(10, 200, n_cat)[category]
    price = cat_base + rng.normal(0, 8.0, size=n)

    # brand depends on (category, price_bucket): high-confidence but not functional
    price_bucket = np.clip((price // 50).astype(int), 0, 4)
    key = category * 5 + price_bucket
    key_to_brand = rng.integers(0, 12, size=n_cat * 5)
    brand = key_to_brand[key].copy()
    flip = rng.random(n) < 0.15                         # 15% noise -> ~85% confidence
    brand[flip] = rng.integers(0, 12, size=int(flip.sum()))

    random_tag = rng.integers(0, 8, size=n)

    table = pd.DataFrame({
        "cluster_label": blob.astype("int64"),
        "category": category.astype("int64"),
        "zip_code": zip_code.astype("int64"),
        "city": city.astype("int64"),
        "region": region.astype("int64"),
        "zip_score": zip_score.astype("float64"),
        "price": price.astype("float64"),
        "brand": brand.astype("int64"),
        "random_tag": random_tag.astype("int64"),
    })
    dtypes = {"cluster_label": "cat", "category": "cat", "zip_code": "cat",
              "city": "cat", "region": "cat", "zip_score": "num",
              "price": "num", "brand": "cat",
              "random_tag": "cat"}
    qra = ["category", "city", "region", "price", "brand", "random_tag"]
    return DatasetSpec(
        name="synthetic_deps", A_e=A_e, table=table, dtypes=dtypes, metric=metric,
        query_relevant_attrs=qra,
        meta={"planted_rules": ["zip_code->city (100%)", "zip_code->region (100%)",
                                "zip_code->zip_score (100%)",
                                "category->price (interval)", "category,price->brand (~85%)"],
              "n_clusters": n_clusters, "n_zip": n_zip})


def build_Dstar(spec: DatasetSpec) -> DatasetSpec:
    """Drop tuples whose query-relevant attrs are originally missing -> complete D*."""
    qra = spec.query_relevant_attrs
    keep = spec.table[qra].notna().all(axis=1).values
    if keep.all():
        return spec
    return DatasetSpec(
        name=spec.name,
        A_e=spec.A_e[keep],
        table=spec.table.loc[keep].reset_index(drop=True),
        dtypes=spec.dtypes,
        metric=spec.metric,
        query_relevant_attrs=spec.query_relevant_attrs,
        meta={**spec.meta, "dropped_original_missing": int((~keep).sum())},
    )


def load_processed(d: dict) -> DatasetSpec:
    """Load a cached processed dataset (embeddings.npy + table.parquet + meta.json),
    e.g. produced by scripts/prepare_amazon.py / prepare_movielens.py."""
    import json
    base = Path(d["path"])
    # emb_file selects which embedding to load (R3: sweep bge-m3 / qwen3 / minilm in one
    # dataset dir, sharing table.parquet). Defaults to the legacy embeddings.npy.
    A_e = np.load(base / d.get("emb_file", "embeddings.npy")).astype(np.float32)
    table = pd.read_parquet(base / "table.parquet").reset_index(drop=True)
    meta = json.loads((base / "meta.json").read_text()) if (base / "meta.json").exists() else {}
    dtypes = d["dtypes"]
    qra = d["query_relevant_attrs"]
    table = table[list(dtypes.keys())]
    return DatasetSpec(name=d["name"], A_e=A_e, table=table, dtypes=dtypes,
                       metric=d.get("metric", "cosine"), query_relevant_attrs=qra,
                       meta=meta)


def load_dataset(cfg: dict) -> DatasetSpec:
    d = cfg["dataset"]
    loader = d.get("loader", d["name"])
    if loader == "synthetic":
        spec = make_synthetic(
            n=d.get("n", 20000),
            d=d.get("d", 64),
            n_clusters=d.get("n_clusters", 40),
            metric=d.get("metric", "l2"),
            seed=cfg.get("seed", 0),
            sep=d.get("sep", 3.0),
            noise=d.get("noise", 1.0),
            label_noise=d.get("label_noise", 0.0),
            score_signal=d.get("score_signal", 1.0),
        )
        if d.get("query_relevant_attrs"):
            spec.query_relevant_attrs = d["query_relevant_attrs"]
    elif loader == "synthetic_deps":
        spec = make_synthetic_deps(
            n=d.get("n", 20000), d=d.get("d", 64),
            n_clusters=d.get("n_clusters", 20), metric=d.get("metric", "l2"),
            seed=cfg.get("seed", 0))
        if d.get("query_relevant_attrs"):
            spec.query_relevant_attrs = d["query_relevant_attrs"]
    elif loader == "processed":
        spec = load_processed(d)
    elif loader == "arxiv":
        from mifa.datasets_arxiv import load_arxiv  # lazy: needs `datasets`
        spec = load_arxiv(d, seed=cfg.get("seed", 0))
    else:
        raise ValueError(f"unknown loader: {loader}")
    return build_Dstar(spec)
