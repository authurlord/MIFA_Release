#!/usr/bin/env python3
"""Prepare the PAPER's Yelp workload from the Yelp Open Dataset business table
(yashraizad/yelp-open-dataset-business: business_id, name, city, state, postal_code, categories,
stars, review_count). Structured attrs: city, state, stars, review_count, primary_category.
Real dependency city->state (functional, MIFA Sigma_ind). Text = name + categories (embedded
separately by embed_dataset). Query embeddings = held-out businesses (paper generates queries
from business names/categories). Saves processed dataset for rule_compare.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/yelp")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("yashraizad/yelp-open-dataset-business", "business.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    df = df.dropna(subset=["city", "state", "stars", "review_count", "categories", "name"])
    # keep states with enough support so city->state FD has support; primary category = first
    df["primary_category"] = df["categories"].map(lambda s: str(s).split(",")[0].strip())
    # keep frequent primary categories (>=50) so the attr is meaningful
    vc = df["primary_category"].value_counts()
    df = df[df["primary_category"].isin(vc[vc >= 50].index)]
    if len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=args.seed)
    df = df.reset_index(drop=True)
    df["text"] = (df["name"].astype(str) + ". " + df["categories"].astype(str)).str.slice(0, 512)
    keep = ["city", "state", "stars", "review_count", "primary_category"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df[keep].to_parquet(out / "table.parquet")
    df[["text"]].to_parquet(out / "text.parquet")
    # city->state FD purity (sanity)
    g = df.groupby("city")["state"].nunique()
    (out / "meta.json").write_text(json.dumps(
        {"name": "yelp", "n": len(df), "states": int(df.state.nunique()),
         "cities": int(df.city.nunique()), "primary_categories": int(df.primary_category.nunique()),
         "city_to_state_FD_purity": float((g == 1).mean()),
         "source": "yashraizad/yelp-open-dataset-business"}, indent=2))
    print(f"[yelp] {len(df)} businesses, {df.state.nunique()} states, {df.city.nunique()} cities, "
          f"{df.primary_category.nunique()} primary-cats, city->state purity {float((g==1).mean()):.3f} -> {out}")


if __name__ == "__main__":
    main()
