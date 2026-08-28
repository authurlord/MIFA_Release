#!/usr/bin/env python3
"""Multi-category Amazon: pull items from several categories so `main_category` is a
real attribute with a strong store->main_category dependency (brands tend to be
category-exclusive). The target main_category is masked and recovered from store;
the embedding (title+description text) is the competing predictor. Tests whether a
strong real near-FD lets rule-MIFA beat embedding-kNN on real data.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd


def parse_float(x):
    try:
        return np.nan if x in (None, "", "None") else float(x)
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=[
        "All_Beauty", "Software", "Video_Games", "Musical_Instruments",
        "Office_Products", "Pet_Supplies"])
    ap.add_argument("--per-cat", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="data/processed/amazon_multi")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    rng = np.random.default_rng(args.seed)

    frames = []
    for cat in args.categories:
        fn = f"raw/meta_categories/meta_{cat}.jsonl"
        print(f"[amz-multi] {cat} ...", flush=True)
        p = hf_hub_download("McAuley-Lab/Amazon-Reviews-2023", fn, repo_type="dataset")
        rows = []
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                store = d.get("store"); title = (d.get("title") or "").strip()
                if not title or not store:
                    continue
                desc = d.get("description")
                if isinstance(desc, list):
                    desc = " ".join(desc)
                rows.append({"main_category": cat, "store": store,
                             "text": (title + ". " + (desc or ""))[:512],
                             "average_rating": parse_float(d.get("average_rating")),
                             "rating_number": parse_float(d.get("rating_number"))})
        df = pd.DataFrame(rows).dropna(subset=["average_rating", "rating_number"])
        if len(df) > args.per_cat:
            df = df.sample(n=args.per_cat, random_state=args.seed)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # keep stores with >=5 items (so store-based rules have support)
    vc = df["store"].value_counts()
    df = df[df["store"].isin(vc[vc >= 5].index)].reset_index(drop=True)
    # how functional is store->main_category?
    g = df.groupby("store")["main_category"].nunique()
    print(f"[amz-multi] {len(df)} items, {df.store.nunique()} stores, "
          f"{df.main_category.nunique()} categories; "
          f"store->category purity (1 cat): {float((g==1).mean()):.3f}", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device="cuda")
    emb = model.encode(df["text"].tolist(), batch_size=args.batch, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    keep = ["main_category", "store", "average_rating", "rating_number"]
    df[keep].to_parquet(out / "table.parquet")
    df[["text"]].to_parquet(out / "text.parquet")  # R3: re-embed with other models
    (out / "meta.json").write_text(json.dumps(
        {"n": len(df), "d": int(emb.shape[1]), "categories": args.categories,
         "n_stores": int(df.store.nunique())}, indent=2))
    print(f"[amz-multi] saved -> {out} (n={len(df)} d={emb.shape[1]})", flush=True)


if __name__ == "__main__":
    main()
