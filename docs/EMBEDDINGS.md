# Embedding models and generation

The release preserves the exact vectors used by the paper.  This section also
documents how those vectors were produced.

| Datasets | Model | HF link | Paper dimension | Method |
|---|---|---|---:|---|
| WANDS, Books, MovieLens, Yelp, H&M, Amazon-multi, Amazon-C4 | BGE-M3 | <https://huggingface.co/BAAI/bge-m3> | 1024 | tokenize with truncation, take `last_hidden_state[:,0]`, cast to float32, row-wise L2 normalize |
| AppReviews | Jina Embeddings v5 text small | <https://huggingface.co/jinaai/jina-embeddings-v5-text-small> | 1024 | `task=retrieval`, `prompt_name=document/query`, `truncate_dim=1024`, row-wise L2 normalize; paper asset revision `dd76d535f5447ca3897a9c893fb1e612ead98192` |
| ArXiv / ArXiv-large | Stella 400M vectors supplied by FANNS | <https://huggingface.co/NovaSearch/stella_en_400M_v5> | 4096 native | use the official `SPCL/arxiv-for-fanns-*` FVEC files directly and L2 normalize for cosine search |
| ArXiv-large-1M | same official Stella vectors | <https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large> | 128 serving | take the first 128 coordinates of each official 4096d vector, then row-wise L2 normalize; apply the frozen release permutation and use its 1M prefix |
| Optional MRL regeneration | Qwen3-Embedding-0.6B | <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B> | requested 128/256/512/1024 | serve with an OpenAI-compatible vLLM embedding endpoint and request the native Matryoshka dimension; paper helper revision `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418` |

The BGE/Jina generator is `scripts/embed_hf_dense.py`.  Examples:

```bash
# BGE-M3 document vectors
python scripts/embed_hf_dense.py \
  --model bge-m3 --input data/processed/wands/text.parquet \
  --text-column text --role document \
  --output data/processed/wands/emb_bgem3.npy

# BGE-M3 query vectors
python scripts/embed_hf_dense.py \
  --model bge-m3 --input data/processed/wands/queries.parquet \
  --text-column query --role query \
  --output data/processed/wands/query_emb_bgem3.npy

# AppReviews Jina-v5 document vectors
python scripts/embed_hf_dense.py \
  --model jina-v5 --input data/processed/appreviews/text.parquet \
  --text-column review --role document \
  --output data/processed/appreviews/emb_jina_v5_mrl_d1024.npy

# AppReviews query vectors are the frozen target rows of the document matrix
python scripts/embed_hf_dense.py \
  --model jina-v5 --input data/processed/appreviews/queries.parquet \
  --target-row-column target_row \
  --document-vectors data/processed/appreviews/emb_jina_v5_mrl_d1024.npy \
  --output data/processed/appreviews/query_emb_jina_v5_mrl_d1024.npy
```

For Qwen3 through vLLM, use `scripts/embed_dataset.py`, for example:

```bash
# Start the paper-version embedding server in its separate environment.
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-0.6B \
  --served-model-name qwen3-embed \
  --host 127.0.0.1 --port 8001 --dtype float16

# Run the release client from the MIFA environment.
python scripts/embed_dataset.py \
  --dir data/processed/wands --emb-tag qwen3_d128 \
  --backend vllm --host localhost --port 8001 \
  --model-id qwen3-embed --dim 128
```

The server environment is pinned in `requirements-vllm.txt` to vLLM 0.17.1;
it is intentionally separate from the MIFA runtime environment because vLLM
selects its own compatible PyTorch/CUDA packages.

ArXiv-large projection and unique-prefix construction are implemented in
`scripts/prepare_arxiv_large.py`; no new language-model
encoding is performed for that dataset.
