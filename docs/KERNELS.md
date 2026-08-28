# Kernel design and implementation

## Runtime dispatch

`configs/kernel_registry.json` is the runtime dispatch source of truth.
`scripts/run_release.py` loads it automatically, resolves the requested
dataset key, and passes that entry's execution parameters to the paper runner.
The two-GPU launcher creates two processes that independently load the same
dataset entry.

The registry keeps tuning local to a workload. A change for a high-dimensional
numerical workload therefore does not replace the kernel selected for a small
categorical or mixed-predicate workload.

## Design principles

1. **Preserve MIFA semantics.** Eligibility inference, candidate ordering,
   exact FP32 refinement, deterministic tuple-ID tie breaking, and top-k output
   use the same contract on CPU and GPU.
2. **Specialize by workload shape.** Dispatch depends on corpus size, vector
   dimension, predicate type, selectivity, repeated predicate signatures,
   query concurrency, and whether serving state fits device memory.
3. **Fuse memory-bound stages.** Predicate evaluation, candidate compaction,
   distance accumulation, and local top-k are fused when this avoids writing a
   full eligibility or distance matrix.
4. **Keep serving arrays contiguous and typed.** Categories are encoded as
   integer IDs, numerical columns use contiguous FP32 arrays, missingness uses
   compact masks, and postings use CSR-style offsets and tuple IDs.
5. **Separate latency and throughput deployment.** One request process reports
   single-GPU latency. Two independent GPU processes serve disjoint query
   streams and their throughputs are summed.

## Released workload families

| Workload family | CPU strategy | GPU strategy |
|---|---|---|
| WANDS, Books, MovieLens, Amazon-multi | Numba quantized projection with FP32 refinement | compact candidate pages with fused Triton scoring/top-k |
| Yelp | Numba eligible gather plus resident SGEMV | compact candidate pages with fused Triton scoring/top-k |
| H&M | posting-first Numba eligibility/distance/top-k | compact candidate pages with fused Triton scoring/top-k |
| AppReviews | fused Numba eligibility/distance/top-k | compact candidate pages with fused Triton scoring/top-k |
| Amazon-C4 | exact multi-attribute Numba postings | compact candidate pages with fused Triton scoring/top-k |
| ArXiv | posting-first fused Numba path | posting-first compact pages with fused Triton scoring/top-k |
| ArXiv-large-1M | frozen exact CPU configuration | two exact GPU serving-process replicas |

The main implementation files are:

- CPU: `src/mifa/search/cpu_candidate_stream.py`,
  `src/mifa/search/fused_exact.py`, and
  `src/mifa/inference/joint_support_numba.py`.
- GPU: `src/mifa/search/triton_candidate.py`,
  `src/mifa/search/triton_tree_probe.py`, and
  `src/mifa/search/gpu_candidate_stream.py`.
- Shared orchestration: `src/mifa/search/unified_executor.py` and
  `src/mifa/inference/paper_policy.py`.

## CPU kernels with Numba

[Numba](https://numba.readthedocs.io/en/stable/) compiles the array-oriented
CPU hot paths to native code. Kernel inputs should be fixed-dtype contiguous
NumPy arrays; dataframe and Python-object operations belong in preparation,
not in the query loop.

Choose the CPU layout from the workload:

- Highly selective categorical filters: intersect compact postings first,
  then fuse distance evaluation and top-k over the surviving tuple IDs.
- Medium selectivity: materialize an eligibility bitmap once, gather eligible
  vectors, and use the resident SGEMV/blocked-BLAS path.
- Broad numerical or mixed filters: stream vector blocks and fuse predicate,
  distance, and top-k so the full distance matrix is never materialized.
- Repeated predicate signatures: precompute typed routing metadata and reuse
  preallocated output buffers.

Parallelize over independent queries or sufficiently large vector blocks with
`prange`; avoid nested parallel regions. Compile and prewarm every signature
before latency measurement so JIT compilation is outside online latency.

## GPU kernels with Triton

[Triton](https://triton-lang.org/main/) implements the fused GPU primitives.
PyTorch owns CUDA tensors, streams, and process placement; Triton kernels read
those tensors directly and fuse eligibility, FP32 distance accumulation, and
local top-k where appropriate.

The released environment targets Linux and NVIDIA compute capability 8.0 or
newer; it was measured on RTX 4090 (compute capability 8.9) with CUDA 12.4,
PyTorch 2.6, and Triton 3.2. The upstream
[Triton compatibility section](https://github.com/triton-lang/triton#compatibility)
lists NVIDIA compute capability 8.0+ for the current NVIDIA backend.

For a new kernel:

- Keep corpus vectors and frequently reused predicate metadata resident when
  they fit VRAM; otherwise stream compact host pages through pinned memory.
- Map a program to a query/candidate tile, use coalesced vector loads, pad the
  vector dimension with masks, and accumulate distances in FP32.
- Use postings before vector loads for selective filters. Use tiled scans for
  broad filters where posting construction would cost more than it saves.
- Fuse compaction, scoring, and local top-k when intermediate tensors dominate
  memory traffic. Merge tile-local results deterministically by distance and
  tuple ID.
- Keep synchronization outside the inner loop and reuse device buffers across
  requests.

## Adding a new dataset

1. Add its schema/data adapter and paper configuration under `configs/`.
2. Add the dataset to `configs/release_experiments.json`, including query
   counts, selected MIFA structure/policy/theta, embedding tag, and predicate
   attributes.
3. Record corpus size `n`, dimension `d`, predicate regime, selectivity range,
   signature repetition, and serving-state bytes. These properties determine
   the kernel family; the dataset name itself should not contain performance
   logic.
4. Start from the exact generic Numba and Triton implementations. Compare
   posting-first, eligible-gather, and streaming-fused layouts using the same
   query cohort and output semantics.
5. Put reusable CPU code in `cpu_candidate_stream.py` or `fused_exact.py` and
   reusable GPU code in `triton_candidate.py` or `triton_tree_probe.py`.
   Dataset-specific constants belong in the registry, not in global switches.
6. Add one `datasets` entry to `configs/kernel_registry.json`:

   ```json
   {
     "display": "NewDataset",
     "dataset_key": "new_dataset",
     "cpu_kernel": "exact_posting_first_fused",
     "exact_prewarm_strategy": "posting_first",
     "cpu_fused_exact_min_work": 4000000,
     "cpu_page_initial": 64,
     "gpu_kernel": "triton_exact_compact_pages",
     "gpu_page_initial": 4096,
     "gpu_deployment": "one process per GPU; query-stream replicas"
   }
   ```

7. Run `scripts/run_release.py --datasets new_dataset ...`. The launcher
   prints the registry path and selected kernel names before starting the
   experiment. Confirm the returned IDs against the exact reference on a
   representative query batch, then retain the fastest layout for that
   dataset without changing other registry entries.

## Multi-GPU execution

`scripts/run_multigpu_replicas.py` launches one serving process per visible
GPU. Each process uses the same frozen dataset entry and serves a disjoint
query stream. Aggregate QPS is the sum of simultaneously measured replica
throughput; reported P50/P95 use the slower replica. Corpus sharding is a
separate design for datasets that cannot fit on one GPU and must not be mixed
with replica-throughput numbers.
