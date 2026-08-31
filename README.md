# MIFA code release

This repository contains the MIFA implementation, the dataset-specific exact
CPU/GPU serving kernels used by the paper, the MCAR/HCE/QAE missingness
generators, and single-command paper-aligned launchers.  MIFA uses
[Numba](https://numba.readthedocs.io/en/stable/) to compile its CPU hot paths
and [Triton](https://triton-lang.org/main/) to compile fused GPU search
primitives.  Dataset and experiment-result payloads are distributed separately
and are ignored by Git.

## Install

Python 3.11 and Linux are recommended.

CPU-only reproducibility environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e . --no-deps
```

Paper GPU kernels (CUDA 12.4 PyTorch build):

```bash
pip install -r requirements-gpu.txt
pip install -e . --no-deps
```

The released Triton target is Linux on NVIDIA GPUs with compute capability
8.0 or newer, matching Triton's documented NVIDIA compatibility range.  The
paper measurements used RTX 4090 GPUs (Ada, compute capability 8.9).  CPU
execution is independent of the GPU architecture and is optimized for
multicore x86-64 systems.

To regenerate BGE-M3, Jina-v5, or other Hugging Face embeddings locally:

```bash
pip install -r requirements-embedding.txt
pip install -e . --no-deps
```

For the separate OpenAI-compatible BGE/Qwen embedding server used by
`scripts/embed_dataset.py`:

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -r requirements-vllm.txt
```

The equivalent CPU Conda setup is:

```bash
conda env create -f environment.yml
conda activate mifa-release
pip install -e . --no-deps
```

## Download the release data

The non-ArXiv-large paper snapshots and the three injected-data archives are
available in the public
[MIFA Google Drive folder](https://drive.google.com/drive/folders/1Q1z0wcBLx5WnIft0C8lUNw96b9F3WZYl?usp=sharing).
Download the complete/source archive and whichever MCAR, HCE, or QAE archive
you want to reproduce.  Archive names and sizes are documented in
[docs/DATASETS.md](docs/DATASETS.md).

Extract `MIFA_NON_ARXIV_LARGE_COMPLETE_AND_SOURCE_DATA.zip` at the repository
root.  Details and upstream links are in [docs/DATASETS.md](docs/DATASETS.md).
Embedding model links, text construction inputs, normalization, and generation
commands are in [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md).

## Embedding models

| Workloads | Hugging Face model | Released representation |
|---|---|---|
| WANDS, Books, MovieLens, Yelp, H&M, Amazon-multi, Amazon-C4 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 1024d CLS dense vector, FP32 row-wise L2 normalization |
| AppReviews | [jinaai/jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small) | retrieval document/query prompts, 1024d, FP32 row-wise L2 normalization |
| ArXiv and ArXiv-large | [NovaSearch/stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5) vectors supplied by FANNS | official 4096d vectors; ArXiv-large-1M uses the normalized 128d release projection |
| Optional MRL reproduction | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | native Matryoshka dimensions requested from the vLLM embedding endpoint |

See the [embedding generation guide](docs/EMBEDDINGS.md) for text fields,
query/document roles, model revisions, normalization, and runnable commands.

## Generate the three paper missingness regimes

Each command uses the frozen seed 0, `rho=0.20`, `k=10`, and one final run.
HCE uses `beta_exp=0.5`; QAE hides one predicate cell per selected tuple.

```bash
bash scripts/inject_mcar.sh all
bash scripts/inject_hce.sh all
bash scripts/inject_qae.sh all
```

The resulting files are written under `data/injected/{mcar,hce,qae}`.  They can
be packaged without embeddings using `scripts/package_release_data.py`.

For missingness-only repetitions, pass `--injection-seed N` to
`scripts/run_release.py`.  This varies the injected mask while retaining the
paper's seed-0 split, query cohort, deployment policy, threshold, and kernel.
The ordinary `--seed` inside the experiment runner controls all of those
frozen experiment components and should remain zero for paper-table replay.

## Run on generated/injected data

CPU, one dataset:

```bash
python scripts/run_release.py \
  --datasets wands --mechanism hce --platforms cpu \
  --injected-root data/injected/hce
```

One GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_release.py \
  --datasets wands --mechanism hce --platforms 1gpu \
  --injected-root data/injected/hce
```

Two GPUs using the paper's two-process replica deployment:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_multigpu_replicas.py \
  --datasets wands --mechanism hce \
  --injected-root data/injected/hce
```

All standard paper datasets on CPU or two GPUs:

```bash
bash scripts/run_cpu.sh all hce
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_multigpu.sh all hce
```

Outputs are written to `reports/` and remain outside Git. Each command executes
one paper-configured run with the selected dataset kernel.

## ArXiv-large

ArXiv-large is link-only and is never included in a release ZIP.  Its official
source, fixed revision, global-permutation seed, d=128 projection, and 1M build
command are documented in [docs/DATASETS.md](docs/DATASETS.md).

The separate archive names, byte counts, GB/GiB totals, and AList destination
are recorded in [docs/ALIST_DATA_RELEASE.md](docs/ALIST_DATA_RELEASE.md).

## Kernel dispatch

Every release experiment launched through `scripts/run_release.py`
automatically loads `configs/kernel_registry.json`.  The registry binds each
dataset to its selected Numba CPU path, Triton GPU path, page size, prewarming
strategy, and multi-GPU deployment.  This per-dataset dispatch lets a kernel be
tuned for a workload's vector dimension, predicate regime, selectivity, and
memory footprint without replacing the implementation selected for another
dataset.

The design principles, implementation map, GPU architecture scope, and a
step-by-step guide for adding a Numba/Triton kernel for a new dataset are in
the [Kernel design and implementation guide](docs/KERNELS.md).
