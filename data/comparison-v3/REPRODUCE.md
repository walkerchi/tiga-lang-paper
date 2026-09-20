# Runtime performance and memory efficiency — revised experiment

This revision uses only pure Torch graph construction/aggregation for the Torch
dynamic baseline and extends exact kNN to 131,072 points. It replaces the main
comparison curves, not the archived earlier results. All 36 configurations run
in separate, sequential processes with four warmups and ten checked samples.

## Implementation

- Ranked executables rebind coordinates across semantically equivalent Graph
  descriptors; changed coordinates never imply cached neighbor lists.
- The compiler emits an exact tile-key bound before sorting/merging: retain the
  old state only if no candidate can improve it. All candidate distances are
  still evaluated. Packed source IDs preserve stable ties.
- Pure Torch computes squared distances by direct coordinate differences, in
  blocks of at most 16,777,216 pairs (64 MiB per FP32 plane). Multiple planes can
  coexist. kNN materializes selected indices then gathers/sums; radius builds
  CSR with Euclidean edge weights and calls torch.sparse.mm. No Tiga/PyG builder
  is used by these baselines. Neither baseline is an optimized spatial index.

## Commands

Use the same Torch/PyG installation as the previous experiment (Torch
2.11.0+cu128, Triton 3.6.0, PyG 2.8.0.post1, pyg-lib 0.9.0+pt211cu128,
torch-cluster 1.6.3+pt211cu128). The local Python executable is
`/home/walkerchi/Code/ComfyUI/.venv/bin/python`; optional PyG wheels live separately
in `/tmp/tiga-paper-v2-deps.AW5KyW` without modifying service environments.

Build the archived compiler source with LLVM/MLIR 22.1.8 using the project's
CMake instructions. Both gf-opt and gf-translate must come from that build;
the prior translator does not include ranked tile pruning. From the source:

```bash
export PYTHONPATH=python:.:/path/to/isolated-pyg-dependencies
export TIGA_OPT=/path/to/rebuilt/gf-opt
export TIGA_TRANSLATE=/path/to/rebuilt/gf-translate
export OMP_NUM_THREADS=1
python -m benchmarks.graph_operations.run_paper_comparison --output /path/to/new-results
```

`runs.json` records actual commands, stdout and stderr. Every configuration must
succeed. Complete Torch/PyG edge sets are compared for both coordinate snapshots;
every timed provider output is checked. Radius uses the existing binary-grid
fixture and nontruncating neighbor cap; kNN uses uniform FP32 points. The Torch
baseline evaluates squared distances directly rather than using a GEMM identity.
Every actual dynamic call copies changed coordinates and rebuilds its relation.

The three `audit-*.json` files are separate two-sample largest-shape native-device
allocation checks, not formal timing trials. Their tracked native allocation is
zero; main figures report Torch allocated bytes, excluding reserve/context.
`source.tar.gz` includes implementation, compiler source, benchmarks and tests.
Each formal record includes Python, benchmark, pure-Torch helper and compiler
binary hashes; data/index hashes are checked by the paper renderer.

The 1B/paging archive and earlier distributed runs retain their original source
snapshot. Their performance is not silently attributed to this implementation.
No service was stopped and no package/repository/paper was published.
