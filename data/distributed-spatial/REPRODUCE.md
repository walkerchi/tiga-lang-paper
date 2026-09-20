# Distributed execution of spatial mesh graphs

The main distributed experiment uses nonperiodic trilinear hexahedral mesh
connectivity, not an artificial graph with every fourth row redirected remotely.
Every pair of distinct nodes sharing an element is connected in both directions.
Interior nodes have 26 neighbors; exterior rows have fewer. The operation sums
16 FP32 features over neighbors. This is not a complete FEM stiffness solve.

## Geometry and communication

Side lengths are 32, 64 and 96. Node counts are 32,768, 262,144 and 884,736;
directed edge counts are 797,816, 6,596,856 and 22,508,920. Nodes are numbered
x-major, so the equal-row two-rank partition is a spatial plane. Per rank, the
boundary and ghost layers each contain L² nodes out of L³/2 owned nodes. Expected
boundary fractions are 6.25%, 3.125%, 2.0833%. Both-rank send totals are 0.125,
0.500 and 1.125 MiB per call. Raw traffic and runtime traces verify these counts.

## Execution protocol

One rank uses RTX 5070 Ti/Linux and the other RTX 4070 Ti SUPER/Ubuntu WSL.
Existing services remain running (about 3 GiB and 12 GiB occupied). NCCL 2.28.9
uses Socket over private WireGuard; not NVLink/RDMA. Both ranks have matching
Python sources, compiler binary hashes, benchmark and NCCL library hashes.

Each GPU runs the complete graph alone. Distributed modes are communication then
compute (`serialized`) and an interior/halo overlap schedule (`automatic`). The
latter enqueues halo exchange, computes independent interior rows, then waits
before boundary work and output assembly. It does not guarantee real GPU/network
concurrency, nor autotune the fastest plan. Interior/halo partition setup is cached.

Each configuration uses two warmups and five changed-input samples; all output
scalars are checked exactly against the independent integer-valued CPU oracle.
Timing is synchronized wall time through completed output, excluding setup,
input allocation, control barriers and checking. Topology is replicated at setup;
this does not measure aggregate two-device graph capacity. Per-sample rank times
are paired by step and reduced by max; only then is the median computed. Host
stage breakdown uses the critical rank of that median paired sample, not sums
of unrelated stage medians. Host stages contain waits/runtime work, not isolated
communication service times. Launch order is fixed, on shared rather than
clock-locked hosts; no cross-process confidence intervals are claimed.

## Reproduction

Build matching sources/compiler tools from `source.tar.gz` with LLVM/MLIR 22.1.8;
use the prior artifact's native-runtime build instructions. `commands.json`
contains the actual local/private-network launch; adjust machine paths and
trusted private addresses when repeating. No credentials are included. Generic:

```bash
export PYTHONPATH=python:.
export TIGA_TENSOR_BACKEND=native
export TIGA_OPT=/path/to/gf-opt
export TIGA_TRANSLATE=/path/to/gf-translate
export TIGA_NCCL_LIBRARY=/path/to/libnccl.so.2
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=PRIVATE_INTERFACE
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
python -m benchmarks.distributed.paper_scaling \
  --rank 0 --host PRIVATE_RANK0_IP --port 29745 --transport nccl \
  --topology mesh --mesh-sides 32,64,96 --features 16 \
  --warmup 2 --repeats 5 --output mesh-rank0.json
```

Run rank 1 concurrently with `--rank 1` and a different output path. Files retain
each rank's full samples, traces, traffic, environment, exact commands and hashes.
An initial diagnostic run preceded a metadata correction (unused legacy ring
parameters are now null for mesh); it remains in the local
`distributed-spatial-initial` directory and is not pooled with these final samples.
Older ring/large-halo tests remain in the historical artifact. No release/push.
