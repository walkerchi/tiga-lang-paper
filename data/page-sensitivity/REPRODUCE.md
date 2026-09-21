# Page-size sensitivity at 10M edges

Nine randomized fresh-process calls use 16,384, 65,536 or 262,144 rows per page,
three calls per setting. Each traverses 10,000,000 explicitly stored edges with
32 FP32 features and checks all 20,000,000 output scalars exactly afterward.
These new observations do not replace the original 1B capacity evidence.

The compiler source is shared with
[`../supplement/source.tar.gz`](../supplement/source.tar.gz).
`benchmarks/memory_hierarchy/billion_edges.py` is used unchanged; its hash and
runtime/compiler hashes are recorded. The controller and captured commands
are `tools/run_paging_sensitivity.py` and `runs.json`.

```bash
export TIGA_OPT=/path/to/build/bin/gf-opt
export TIGA_TRANSLATE=/path/to/build/bin/gf-translate
export TIGA_RUNTIME_LIBRARY=/path/to/build/lib/Runtime/libtiga_runtime.so
python tools/run_paging_sensitivity.py --project /path/to/compiler \
  --output /path/to/new-page-sweep
python tools/build_page_sensitivity.py --check
```

The controller creates a dedicated temporary fixture (~165 MB) and leaves it
for inspection. No user data or caches are deleted. Build and validation are
outside forward. Forward includes staging, JIT as encountered, execution and
output assembly. The device budget is 12 GiB; prefetch is disabled. Caches are
not flushed, so this is not a cold-disk benchmark.

The table reports median/range of three calls and the largest scoped native GPU
peak per page size, distinct from physical device usage, RSS and OS page cache.
The graph fits device memory in principle: this tests paging cost sensitivity,
not extra capacity or offload superiority. No timing is extrapolated to 1B.
