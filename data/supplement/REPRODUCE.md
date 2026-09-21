# Topology-controlled execution, differentiated geometry and first-call cost

This archive contains 54 successful fresh-process runs from September 21, 2026.
All configurations and retained samples are included. Smoke checks are not
pooled into these records. No service was stopped for measurement.

## Source and environment

`source.tar.gz` is the compiler repository at commit
`a530580ea3d195f6e1fafa1050619f552b2d0f6e`. Records include SHA-256 hashes of
executed Python sources and compiler binaries, software versions, GPU state
and the paper benchmark script, `tools/run_supplement.py`.
Build the archived compiler with LLVM/MLIR 22.1.8 and install matching Torch
and Triton CUDA versions (the records are authoritative).

Download the [Stanford Bunny archive](https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz).
Its SHA-256 is `a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84`.
Only `bunny/reconstruction/bun_zipper.ply` is read, without filesystem extraction.
The Stanford University Computer Graphics Laboratory supplies the geometry;
see its [research-use and attribution terms](https://graphics.stanford.edu/data/3Dscanrep/).
The model is not redistributed in the source archive or arXiv package.

From the paper checkout, with paths set to the archived compiler checkout:

```bash
export PYTHONPATH=/path/to/compiler/python
export TIGA_OPT=/path/to/build/bin/gf-opt
export TIGA_TRANSLATE=/path/to/build/bin/gf-translate
export TIGA_RUNTIME_LIBRARY=/path/to/build/lib/Runtime/libtiga_runtime.so
python tools/run_supplement.py --project /path/to/compiler \
  --bunny /path/to/bunny.tar.gz --output /path/to/new-supplement
python tools/build_supplement.py --check
```

The first command collects new observations; the second validates the checked-in
archive, not the new directory. Omit `--check` to regenerate tables and figures
from checked-in observations. Runs refuse to overwrite configuration records.
Preserve failed runs and use a new directory when changing the protocol.

## Interpretation

- `fusion` compares compiled aggregation, explicit Torch gather/multiply/scatter
  and Torch sparse on identical stored radius CSR. Search is before timing.
  This controls topology, not a single compiler pass or generated-neighbor fusion.
  Topology hashes must match across providers within each seeded trial.
- `edgenn` uses 35,947 Bunny vertices, centered and divided by the largest
  bounding-box extent. Radius 0.015 produces 356,260 directed edges. Features
  are random (width 8 or 32); a 32-unit ReLU MLP has three outputs. MSE targets
  are normalized coordinates. No optimizer or convergence is measured; edge
  membership stays fixed during differentiation.
- Each process alternates two feature snapshots, performs four warmups and ten
  timed calls, and validates every output outside timing. EdgeNN also checks
  all six input/parameter gradients, including relative L2 error <0.002.
  Compiled paths are asserted. Error bars span three process medians, not
  within-process calls or confidence intervals.
- Forward includes dispatch, execution and synchronization. Backward includes
  MSE construction and `autograd.grad`, then synchronization. Input copies are
  before timing. Memory is peak allocated Torch bytes during the call, including
  live graph/input/cache storage, output, tape and workspaces, not allocator
  reserve, device context or unrelated processes.
- First call follows eager-oracle and CUDA initialization. Cold JIT pairs use
  isolated Triton, TorchInductor and CUDA cache directories; the second worker
  reuses those directories but not an in-memory executable. First-call time is
  not pure compiler time. No user cache is deleted or used for these cold pairs.
  Torch gather/scatter has already run as the correctness oracle: its first
  timed call is not cold library initialization. Amortization estimates exclude
  common setup and describe this execution-ready starting state, not Python
  process startup or an observed cumulative-time curve.
- Wider EdgeNN backward is slower than Torch despite lower memory, and Tiga is
  not uniformly faster than sparse multiplication. These results are retained.
