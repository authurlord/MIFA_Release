#!/usr/bin/env python3
"""Prepare MovieLens 25M for MIFA (structured attrs with real dependencies).

Attrs: primary_genre, n_genres, decade (from title year), avg_rating, rating_count,
top_tag (argmax genome-tag relevance per movie). Strong dependency: top_tag -> genre
(e.g. a tag like 'space'/'superhero' implies Sci-Fi/Action), and genre <-> decade.
Embeds 'title [genres]' with a compact SBERT. Restricted to movies that have genome
tags (the well-annotated subset) and >= a few ratings. Resource-bounded.
"""
import argparse, sys, io, zipfile, urllib.request, os
from pathlib import Path
import numpy as np, pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--cache", default="data/raw/ml-25m")
    ap.add_argument("--out", default="data/processed/movielens")
    ap.add_argument("--text-mode", choices=["title_genres", "title"], default="title_genres",
                    help="title_genres leaks genre into the embedding; title is the fair test")
    args = ap.parse_args()
    cache = Path(args.cache)

    if not (cache / "movies.csv").exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"[ml] downloading {URL} ...", flush=True)
        zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
        for name in zf.namelist():
            if name.endswith((".csv",)):
                base = os.path.basename(name)
                if base in ("movies.csv", "genome-scores.csv", "genome-tags.csv", "ratings.csv"):
                    with zf.open(name) as src, open(cache / base, "wb") as dst:
                        dst.write(src.read())
        print("[ml] extracted", flush=True)

    movies = pd.read_csv(cache / "movies.csv")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    movies["decade"] = (movies["year"] // 10 * 10)
    movies["primary_genre"] = movies["genres"].str.split("|").str[0]
    movies["n_genres"] = movies["genres"].str.split("|").apply(len)
    movies = movies[movies["primary_genre"] != "(no genres listed)"]

    # top genome tag per movie (argmax relevance), chunked to bound RAM
    print("[ml] computing top genome tag per movie ...", flush=True)
    gtags = pd.read_csv(cache / "genome-tags.csv").set_index("tagId")["tag"]
    best = {}
    for chunk in pd.read_csv(cache / "genome-scores.csv", chunksize=2_000_000):
        idx = chunk.groupby("movieId")["relevance"].idxmax()
        top = chunk.loc[idx, ["movieId", "tagId", "relevance"]]
        for mid, tid, rel in top.itertuples(index=False):
            if mid not in best or rel > best[mid][1]:
                best[mid] = (tid, rel)
    movies["top_tag"] = movies["movieId"].map(lambda m: gtags.get(best[m][0]) if m in best else None)

    # rating aggregates (chunked)
    print("[ml] aggregating ratings ...", flush=True)
    s_sum = {}; s_cnt = {}
    for chunk in pd.read_csv(cache / "ratings.csv", chunksize=5_000_000):
        g = chunk.groupby("movieId")["rating"].agg(["sum", "count"])
        for mid, row in g.iterrows():
            s_sum[mid] = s_sum.get(mid, 0.0) + row["sum"]
            s_cnt[mid] = s_cnt.get(mid, 0) + row["count"]
    movies["rating_count"] = movies["movieId"].map(s_cnt)
    movies["avg_rating"] = movies["movieId"].map(lambda m: s_sum.get(m, 0) / s_cnt[m] if s_cnt.get(m) else np.nan)

    df = movies.dropna(subset=["top_tag", "primary_genre", "decade", "avg_rating"])
    df = df[df["rating_count"] >= 5]
    if len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=args.seed)
    df = df.reset_index(drop=True)
    print(f"[ml] {len(df)} movies, genres={df.primary_genre.nunique()} tags={df.top_tag.nunique()}", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device="cuda")
    if args.text_mode == "title":
        texts = df["title"].astype(str).tolist()
    else:
        texts = (df["title"].astype(str) + " [" + df["genres"].astype(str) + "]").tolist()
    emb = model.encode(texts, batch_size=args.batch, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    keep = ["primary_genre", "top_tag", "decade", "n_genres", "avg_rating", "rating_count"]
    df[keep].to_parquet(out / "table.parquet")
    import pandas as _pd  # R3: persist embedded text for re-embedding with other models
    _pd.DataFrame({"text": texts}).to_parquet(out / "text.parquet")
    import json
    (out / "meta.json").write_text(json.dumps(
        {"n": len(df), "d": int(emb.shape[1]), "model": args.model,
         "genres": int(df.primary_genre.nunique()), "tags": int(df.top_tag.nunique())}, indent=2))
    print(f"[ml] saved -> {out} (n={len(df)} d={emb.shape[1]})", flush=True)


if __name__ == "__main__":
    main()
