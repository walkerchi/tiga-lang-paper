# Billion-edge host and CUDA profile

This is one instrumented 1,000,000,000-edge forward call, not an extrapolation
from 10M and not a new three-trial latency sweep. The original uninstrumented
capacity medians remain in `data/billion/`. All 2,000,000,000 output scalars
are checked exactly after the profiling window.

Extract `source.tar.gz`, build with LLVM/MLIR 22.1.8, and use the Python, NumPy,
Triton, compiler hashes and CUDA environment recorded in `profile.json`.
The actual run used Python 3.12, Torch-environment libraries, a 16 GiB 5070 Ti,
Nsight Systems 2025.3.2, 65,536-row pages and a 12 GiB native allocation budget.
Existing GPU services were not stopped.

```bash
export PYTHONPATH=python:.
export TIGA_TENSOR_BACKEND=native
export TIGA_OPT=/path/to/build/bin/gf-opt
export TIGA_TRANSLATE=/path/to/build/bin/gf-translate
export OMP_NUM_THREADS=1
python -m benchmarks.memory_hierarchy.billion_edges --build \
  --edges 1000000000 --cache-dir /path/to/fixtures
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/path/to/new-run/trace \
  python -m benchmarks.memory_hierarchy.profile_paging --edges 1000000000 \
  --cache-dir /path/to/fixtures --cuda-capture --evict-fixture \
  --output /path/to/new-run/profile.json
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv --output /path/to/new-run/cuda /path/to/new-run/trace.nsys-rep
```

The exclusive host stages sum to `profile.forward_seconds` (414.810 s).
`forward_s` (417.052 s) additionally includes profiler start/stop overhead.
CUDA kernel/copy durations are contained within the host stages, not extra
additive stages. Host gather includes mmap page-fault IO. Fixture-only eviction
hints produced 16.500 GB of process physical reads, but do not isolate pure
NVMe service latency or prove a graph larger than host RAM. The raw Nsight
capture, exported CSVs and exact source snapshot are included.
