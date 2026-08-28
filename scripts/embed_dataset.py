#!/usr/bin/env python3
"""Re-embed a processed dataset's text with any model (R3 embedding axis).

Reads <dir>/text.parquet (row-aligned with table.parquet), embeds via a vLLM
OpenAI-compatible server (default) or a local SentenceTransformer, and writes
<dir>/emb_<tag>.npy plus an `embeddings` entry in meta.json. Select it at eval time
with `--emb <tag>` (sets cfg.dataset.emb_file = emb_<tag>.npy).

Examples:
  # Start an OpenAI-compatible embedding server for Qwen/Qwen3-Embedding-0.6B.
  conda run -n mifa python scripts/embed_dataset.py --dir data/processed/amazon_beauty --emb-tag qwen3 --port 8001
  conda run -n mifa python scripts/embed_dataset.py --dir data/processed/amazon_beauty --emb-tag bgem3 --port 8002
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="processed dataset dir (has text.parquet)")
    ap.add_argument("--emb-tag", required=True, help="output tag -> emb_<tag>.npy")
    ap.add_argument("--backend", choices=["vllm", "st"], default="vllm")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--batch", type=int, default=256)
    # vllm
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model-id", default=None, help="served id (auto if None)")
    ap.add_argument("--dim", type=int, default=None,
                    help="Matryoshka (MRL) output width, e.g. Qwen3 dimensions=512 (vllm path only)")
    ap.add_argument("--query-file", default=None,
                    help="optional parquet containing an external/native query corpus")
    ap.add_argument("--query-text-col", default="query")
    ap.add_argument("--query-output", default=None,
                    help="default: <dir>/query_emb_<tag>.npy")
    ap.add_argument("--query-only", action="store_true",
                    help="embed only --query-file; preserve existing base embeddings")
    # st
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    base = Path(args.dir)
    tp = base / "text.parquet"
    if not tp.exists():
        sys.exit(f"[embed] missing {tp} -- re-run the prepare script to persist text first")
    if args.query_only and not args.query_file:
        sys.exit("[embed] --query-only requires --query-file")
    texts = [] if args.query_only else \
        pd.read_parquet(tp)[args.text_col].astype(str).tolist()
    if texts:
        print(f"[embed] {base.name}: {len(texts)} texts -> emb_{args.emb_tag} ({args.backend})", flush=True)

    if args.backend == "vllm":
        from mifa.embed_client import embed_texts, wait_for_server
        url = f"http://{args.host}:{args.port}/v1"
        served = wait_for_server(url, args.api_key)
        print(f"[embed] server up: {served}", flush=True)
        model_name = args.model_id or served
        def encode(items):
            return embed_texts(
                items, base_url=url, api_key=args.api_key,
                model=model_name, batch_size=args.batch, normalize=True,
                dimensions=args.dim)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.st_model, device="cuda")
        model_name = args.st_model
        def encode(items):
            return model.encode(
                items, batch_size=args.batch, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    out = base / f"emb_{args.emb_tag}.npy"
    meta_p = base / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    emb = None
    if texts:
        emb = encode(texts)
        np.save(out, emb)
        embedding_meta = {
            "model": model_name,
            "dim": int(emb.shape[1]),
            "file": out.name,
            "backend": args.backend,
            "normalization": "l2",
        }
        if args.backend == "vllm":
            embedding_meta.update({
                "endpoint": url,
                "requested_dimensions": args.dim,
                "matryoshka_request": args.dim is not None,
            })
        meta.setdefault("embeddings", {})[args.emb_tag] = embedding_meta
    if args.query_file:
        query_path = Path(args.query_file)
        query_texts = pd.read_parquet(query_path)[args.query_text_col].astype(str).tolist()
        print(f"[embed] native queries: {len(query_texts)} texts", flush=True)
        query_embeddings = encode(query_texts)
        query_out = Path(args.query_output) if args.query_output else \
            base / f"query_emb_{args.emb_tag}.npy"
        np.save(query_out, query_embeddings)
        meta.setdefault("query_embeddings", {})[args.emb_tag] = {
            "model": model_name, "dim": int(query_embeddings.shape[1]),
            "file": query_out.name, "queries": len(query_texts),
            "source": str(query_path),
            "backend": args.backend,
            "normalization": "l2",
            **({
                "endpoint": url,
                "requested_dimensions": args.dim,
                "matryoshka_request": args.dim is not None,
            } if args.backend == "vllm" else {}),
        }
    meta_p.write_text(json.dumps(meta, indent=2))
    if emb is not None:
        print(f"[embed] saved {out}  shape={emb.shape}  model={model_name}", flush=True)
    elif args.query_file:
        print(f"[embed] saved query embeddings only; base file preserved: {out}", flush=True)


if __name__ == "__main__":
    main()
