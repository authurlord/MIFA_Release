#!/usr/bin/env python3
"""Generate the BGE-M3 or Jina-v5 dense vectors used by the release.

The output is float32 and row-wise L2 normalized.  BGE-M3 uses the final
hidden state's CLS token, matching the paper asset builder.  Jina-v5 uses its
retrieval adapter with document/query prompt names and a 1024-d output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "jina-v5": "jinaai/jina-embeddings-v5-text-small",
}


def load_texts(path: Path, column: str) -> list[str]:
    if path.suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        return [str(value or "") for value in values]
    frame = pd.read_parquet(path)
    return frame[column].fillna("").astype(str).tolist()


def normalize(array) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def bge_encode(model_id: str, texts: list[str], *, device: str,
               batch_size: int, max_length: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device).eval()
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            values = model(**batch).last_hidden_state[:, 0]
        rows.append(values.float().cpu().numpy())
    return normalize(np.concatenate(rows, axis=0))


def jina_encode(model_id: str, texts: list[str], *, role: str,
                device: str, batch_size: int, max_length: int) -> np.ndarray:
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    rows = []
    prompt = "document" if role == "document" else "query"
    for start in range(0, len(texts), batch_size):
        values = model.encode(
            texts=texts[start:start + batch_size], task="retrieval",
            prompt_name=prompt, max_length=max_length, truncate_dim=1024)
        if torch.is_tensor(values):
            values = values.detach().float().cpu().numpy()
        rows.append(np.asarray(values, dtype=np.float32))
    return normalize(np.concatenate(rows, axis=0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--role", choices=("document", "query"),
                        default="document")
    parser.add_argument("--target-row-column", default="",
                        help="derive query vectors by indexing --document-vectors")
    parser.add_argument("--document-vectors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if args.target_row_column:
        if not args.document_vectors:
            parser.error("--target-row-column requires --document-vectors")
        rows = pd.read_parquet(args.input)[args.target_row_column].to_numpy(
            dtype=np.int64)
        values = np.load(args.document_vectors, mmap_mode="r")[rows]
        embeddings = np.ascontiguousarray(values, dtype=np.float32)
    else:
        texts = load_texts(args.input, args.text_column)
        model_id = MODELS[args.model]
        if args.model == "bge-m3":
            embeddings = bge_encode(
                model_id, texts, device=args.device,
                batch_size=args.batch_size, max_length=args.max_length)
        else:
            embeddings = jina_encode(
                model_id, texts, role=args.role, device=args.device,
                batch_size=args.batch_size, max_length=args.max_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    print(json.dumps({
        "output": str(args.output),
        "model": MODELS[args.model],
        "role": args.role,
        "rows": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "normalization": "row_l2",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

