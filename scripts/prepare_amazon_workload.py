#!/usr/bin/env python3
"""Build a single Amazon Reviews 2023 product workload (Books / Electronics / Home)
for MIFA, faithful to the paper's per-category product workloads.

Scalar attributes (real inter-attribute dependencies):
  - subcategory     : categories[1] of the product taxonomy hierarchy
                      (high-cardinality categorical -> the FD TARGET; this is the
                      embedding-competing attribute that MIFA recovers from a rule).
  - store (brand)   : categorical -> the FD EVIDENCE (store -> subcategory near-FD,
                      brands tend to be subcategory-exclusive, like amazon_multi's
                      store -> main_category).
  - price           : numerical (sparse in raw Amazon; kept only where present).
  - average_rating  : numerical.
  - rating_number   : numerical.

Text (title + description) is persisted for embedding via scripts/embed_dataset.py
(bge-m3). Rows whose query-relevant attrs are originally missing are DROPPED (-> D*).

Resource-aware: bounded sample, single category file read once.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd


def _resolve_cdn(category):
    """Resolve the proxy-friendly classic-CDN URL for a category meta file.
    HF `resolve/main` 302-redirects to a (non-xet) LFS CDN URL that works through
    the local proxy; the xet URLs return 403 behind the proxy."""
    import requests
    base = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
            f"resolve/main/raw/meta_categories/meta_{category}.jsonl")
    r = requests.get(base, allow_redirects=False, timeout=30)
    loc = r.headers.get("location")
    if not loc:
        raise RuntimeError(f"no redirect for {category} (status {r.status_code})")
    return loc


def iter_stream_lines(category, max_bytes):
    """Stream the meta JSONL via HTTP Range up to ~max_bytes, yielding raw lines.
    These files are 5-15 GB; we only need a prefix to sample ~20k rows from, so we
    fetch a bounded prefix instead of the whole file."""
    import requests
    url = _resolve_cdn(category)
    headers = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {}
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        buf = b""
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            buf += chunk
            parts = buf.split(b"\n")
            buf = parts.pop()
            for ln in parts:
                if ln:
                    yield ln


def parse_float(x):
    try:
        if x in (None, "", "None"):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True,
                    help="Amazon Reviews 2023 category file stem, e.g. Books / Electronics / Home_and_Kitchen")
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-store-support", type=int, default=5,
                    help="keep stores with >= this many items (so store-rules have support)")
    ap.add_argument("--min-subcat-support", type=int, default=30,
                    help="drop subcategories rarer than this (stabilizes the workload)")
    ap.add_argument("--max-records", type=int, default=600000,
                    help="cap raw records scanned before filtering (resource bound)")
    ap.add_argument("--stream-bytes", type=int, default=0,
                    help="if >0, stream a Range prefix of this many bytes from the HF CDN "
                         "instead of downloading the whole multi-GB file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.stream_bytes > 0:
        print(f"[amz] streaming {args.category} prefix {args.stream_bytes/1e9:.2f}GB ...", flush=True)
        line_iter = iter_stream_lines(args.category, args.stream_bytes)
    else:
        from huggingface_hub import hf_hub_download
        fn = f"raw/meta_categories/meta_{args.category}.jsonl"
        print(f"[amz] resolving {fn} ...", flush=True)
        p = hf_hub_download("McAuley-Lab/Amazon-Reviews-2023", fn, repo_type="dataset")
        line_iter = open(p, "rb")

    rows = []
    seen = 0
    if True:
        for line in line_iter:
            seen += 1
            if seen > args.max_records:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            store = d.get("store")
            title = (d.get("title") or "").strip()
            if not title or not store:
                continue
            cats = d.get("categories") or []
            if not isinstance(cats, list) or len(cats) < 2:
                continue  # need a taxonomy with >=2 levels
            # LEAF subcategory (cats[-1]) -> genuinely HIGH-cardinality FD target.
            subcat = str(cats[-1]).strip()
            if not subcat:
                continue
            desc = d.get("description")
            if isinstance(desc, list):
                desc = " ".join(desc)
            rows.append({
                "title": title,
                "text": (title + ". " + (desc or ""))[:512],
                "subcategory": subcat,
                "store": str(store).strip(),
                "average_rating": parse_float(d.get("average_rating")),
                "rating_number": parse_float(d.get("rating_number")),
                "price": parse_float(d.get("price")),
            })
    df = pd.DataFrame(rows)
    print(f"[amz] scanned {seen} records, {len(df)} with title+store+subcategory", flush=True)

    # D*: drop rows missing query-relevant scalar attrs (price is sparse -> not query-relevant)
    df = df.dropna(subset=["average_rating", "rating_number", "subcategory", "store"])
    df = df[df["subcategory"].str.len() > 0]

    # CRITICAL for MIFA's FD showcase: keep only stores with enough items so the
    # store->subcategory near-FD has discoverable SUPPORT. Without this, every store
    # has ~1 item: the FD is trivially "pure" but unusable (no rule support). We keep
    # the store filter throughout (no degrading to >=2 after sampling).
    vc = df["store"].value_counts()
    df = df[df["store"].isin(vc[vc >= args.min_store_support].index)]
    # keep subcategories with enough support (stable, evaluable high-card categorical)
    sc = df["subcategory"].value_counts()
    df = df[df["subcategory"].isin(sc[sc >= args.min_subcat_support].index)]
    df = df.reset_index(drop=True)

    if len(df) > args.sample:
        # sample WHOLE STORES (not rows) so the kept stores retain their multi-item
        # support -> the store->subcategory FD stays evidence-backed after sampling.
        stores = df["store"].drop_duplicates().sample(frac=1.0, random_state=args.seed).tolist()
        chosen, n_acc = [], 0
        for s in stores:
            grp = df.index[df["store"] == s].tolist()
            chosen.extend(grp); n_acc += len(grp)
            if n_acc >= args.sample:
                break
        df = df.loc[chosen].reset_index(drop=True)
        # re-apply subcategory support after store-sampling
        sc = df["subcategory"].value_counts()
        df = df[df["subcategory"].isin(sc[sc >= args.min_subcat_support].index)].reset_index(drop=True)

    # store -> subcategory functionality (the near-FD that MIFA exploits)
    g = df.groupby("store")["subcategory"].nunique()
    purity = float((g == 1).mean())
    # confidence: fraction of rows whose store maps to its own modal subcategory
    modal = df.groupby("store")["subcategory"].transform(lambda s: s.value_counts().index[0])
    fd_conf = float((df["subcategory"] == modal).mean())
    print(f"[amz] {len(df)} items, {df.store.nunique()} stores, "
          f"{df.subcategory.nunique()} subcategories; "
          f"store->subcat purity(1) {purity:.3f}  fd_conf {fd_conf:.3f}  "
          f"price_cov {df.price.notna().mean():.2f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    keep = ["subcategory", "store", "average_rating", "rating_number", "price"]
    df[keep].to_parquet(out / "table.parquet")
    df[["text"]].to_parquet(out / "text.parquet")
    meta = {"category": args.category, "n": len(df),
            "n_stores": int(df.store.nunique()),
            "n_subcategories": int(df.subcategory.nunique()),
            "store_subcat_purity": purity, "store_subcat_fd_conf": fd_conf,
            "price_coverage": float(df.price.notna().mean()),
            "sample": args.sample, "seed": args.seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[amz] saved -> {out}  (n={len(df)})", flush=True)


if __name__ == "__main__":
    main()
