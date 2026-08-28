#!/usr/bin/env python3
"""Prepare the PAPER's Amazon-C4 product-search workload (faithful, no 1M join).

McAuley-Lab/Amazon-C4: real complex product-search QUERIES (query text -> target item_id) over
~1M items with {item_id, category (31 flat), metadata text}. We sample the corpus to all query
targets + a random fill, and build a filtered-ANN workload: query embedding = query text;
predicate = category == target's category (the paper instantiates predicates over product
metadata incl. category); ground truth = exact filtered top-k on complete data. Saves a processed
dataset (table.parquet=category, text.parquet=item metadata, queries.parquet=query text+target).
Embeddings are added separately by scripts/embed_dataset.py (bge-m3 / qwen3 / minilm).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-corpus", type=int, default=60000)
    ap.add_argument("--n-queries", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/amazon_c4")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    rng = np.random.default_rng(args.seed)

    qp = hf_hub_download("McAuley-Lab/Amazon-C4", "test.csv", repo_type="dataset")
    mp = hf_hub_download("McAuley-Lab/Amazon-C4", "sampled_item_metadata_1M.jsonl", repo_type="dataset")
    q = pd.read_csv(qp)
    items = {}
    with open(mp) as f:
        for line in f:
            d = json.loads(line)
            items[d["item_id"]] = (d.get("category"), d.get("metadata") or "")
    # sample queries, keep only those whose target is in the corpus + has a category
    q = q[q["item_id"].isin(items)].reset_index(drop=True)
    q = q[q["item_id"].map(lambda i: items[i][0] is not None)].reset_index(drop=True)
    if len(q) > args.n_queries:
        q = q.sample(n=args.n_queries, random_state=args.seed).reset_index(drop=True)
    targets = list(dict.fromkeys(q["item_id"].tolist()))   # unique, order-preserving

    # corpus = all targets + random fill to n_corpus
    all_ids = list(items.keys())
    fill_n = max(0, args.n_corpus - len(targets))
    others = [i for i in rng.permutation(all_ids)[: fill_n * 2] if i not in set(targets)][:fill_n]
    corpus_ids = targets + others
    cats = [items[i][0] for i in corpus_ids]
    texts = [items[i][1][:512] for i in corpus_ids]
    row_of = {iid: r for r, iid in enumerate(corpus_ids)}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"category": cats}).to_parquet(out / "table.parquet")
    pd.DataFrame({"text": texts}).to_parquet(out / "text.parquet")
    # queries: query text + target row + target category (the predicate)
    qdf = pd.DataFrame({"query": q["query"].tolist(),
                        "target_row": [row_of[i] for i in q["item_id"]],
                        "target_category": [items[i][0] for i in q["item_id"]]})
    qdf.to_parquet(out / "queries.parquet")
    (out / "meta.json").write_text(json.dumps(
        {"name": "amazon_c4", "n": len(corpus_ids), "n_queries": len(qdf),
         "categories": int(len(set(cats))), "source": "McAuley-Lab/Amazon-C4"}, indent=2))
    print(f"[c4] corpus={len(corpus_ids)} queries={len(qdf)} categories={len(set(cats))} -> {out}")


if __name__ == "__main__":
    main()
