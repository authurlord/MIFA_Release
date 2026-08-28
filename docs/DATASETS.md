# Dataset release layout

The Git repository never contains dataset or experiment-result payloads.  The
separate data ZIP contains the paper-ready complete snapshots and the locally
retained upstream source files for every non-ArXiv-large workload.  Extract it
at the repository root so that its `data/` directory merges with this tree.

Release archives are hosted in the public
[MIFA Google Drive folder](https://drive.google.com/drive/folders/1Q1z0wcBLx5WnIft0C8lUNw96b9F3WZYl?usp=sharing):

- `MIFA_NON_ARXIV_LARGE_COMPLETE_AND_SOURCE_DATA.zip`
- `MIFA_INJECTED_MCAR_RHO020.zip`
- `MIFA_INJECTED_HCE_RHO020.zip`
- `MIFA_INJECTED_QAE_RHO020.zip`
- `DATA_PACKAGE_SIZES.csv` and `DATA_PACKAGE_SIZES.json`

| Release key | Paper dataset | Upstream source | Release representation |
|---|---|---|---|
| `wands` | WANDS | <https://github.com/wayfair/WANDS> | complete scalar table, BGE-M3 1024d corpus/query embeddings, retained upstream CSV files |
| `books_v2` | Books | <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023> | complete scalar table and BGE-M3 1024d embeddings |
| `movielens` | MovieLens | <https://grouplens.org/datasets/movielens/25m/> | complete scalar table and BGE-M3 1024d embeddings |
| `yelp` | Yelp | <https://huggingface.co/datasets/yashraizad/yelp-open-dataset-business> | complete scalar table and BGE-M3 1024d embeddings |
| `qdrant_hm` | H&M | <https://huggingface.co/datasets/Qdrant/hm_ecommerce_products> | complete scalar table, native filter queries and BGE-M3 1024d embeddings |
| `amazon_multi` | Amazon-multi | <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023> | complete scalar table and BGE-M3 1024d embeddings |
| `appreviews` | AppReviews | BigVectorBench AppReviews | complete scalar table, native queries and Jina-v5 1024d embeddings |
| `amazon_c4` | Amazon-C4 | <https://huggingface.co/datasets/McAuley-Lab/Amazon-C4> | complete scalar table, native queries and BGE-M3 1024d embeddings |
| `arxiv` | ArXiv | <https://huggingface.co/datasets/SPCL/arxiv-for-fanns-medium> | official medium FVEC/JSONL files |

ArXiv-large is intentionally not in any ZIP.  Download the official source at
<https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large/tree/79a4cff088965f7039b8766e6115a7769645adb2>
and run:

```bash
python scripts/prepare_arxiv_large.py \
  --raw-dir data/raw/arxiv-for-fanns-large \
  --output-dir data/processed/arxiv_large_unique_d128 \
  --sizes 20000,100000,1000000
```

The three injected-data ZIPs contain split-specific observed scalar tables and
masks.  They deliberately do not duplicate embeddings.  Extract one at the
repository root and pass its directory with `--injected-root`.
