# NCCL probe — initial failure and resolved follow-up, 2026-09-19

## Initial probe (retained, not overwritten)

Outcome: both ranks exited with launcher timeout code 124 after 50 seconds.
The TCP control plane reached `initializing-nccl` on both hosts; no data exchange
completed, so there is no bandwidth or correctness pass to publish.

Both use the same NCCL 2.28.9 library. The original 4070 WSL environment has
2.21.5; the probe used an isolated copy through `TIGA_NCCL_LIBRARY` instead of
upgrading that environment. The two process command records were:

- Rank 0: `NCCL_SOCKET_IFNAME=tailscale0 NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 timeout 50s python -m benchmarks.distributed.multi_host_nccl_probe --rank 0 --host 100.64.0.7 --port 29650 --output output/paper-20260919/nccl-probe/rank0.json`
- Rank 1: `TIGA_NCCL_LIBRARY=/home/walker/tiga-paper-20260919.Axi0tf/libnccl.so.2 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 timeout 50s python -m benchmarks.distributed.multi_host_nccl_probe --rank 1 --host 100.64.0.7 --port 29650 --output output/nccl-probe/rank1.json`

Diagnostic excerpt from rank 0:

```text
NCCL INFO NCCL version 2.28.9+cuda12.9
NCCL INFO Bootstrap: Using tailscale0:100.64.0.7<0>
NCCL INFO socketPollConnect: connect to 172.17.194.153<43893> returned No route to host, retrying (1/34) after sleep for 100 msec
```

The remote Ubuntu WSL has only `lo` and `eth0` (172.17.194.153/20), behind the
Windows host's NAT. The observed failure is reverse reachability to that WSL
address, not evidence that the GPUs or the compiler cannot execute NCCL kernels.
No firewall, routing, WSL configuration, or existing service was changed.

## Authorized follow-up: passed

The subsequent user-approved network adjustment created only an isolated
WireGuard interface (`tiga-wg0`, 10.203.77.1/32 ↔ 10.203.77.2/32). Its remote
endpoint initiates traffic to the local private LAN endpoint, maintaining the
WSL NAT mapping. Neither default routing nor Windows/WSL firewall rules changed;
there was no service or machine restart. Private keys remain root-only outside
the project and are not part of this archive.

An initial tunnel MTU of 1380 passed small packets but stalled NCCL bootstrap.
WSL `eth0` actually has MTU 1280. A 1300-byte DF ping failed; a 1100-byte ping
passed. Socket diagnostics showed unacknowledged large payloads and repeated
retransmissions. Setting the tunnel MTU to 1200 on both hosts resolved the stall.
No speculative NCCL cuMem/NVLS workarounds are retained in the successful run.

Both ranks then passed a checked bidirectional 1 MiB exchange (`nccl-passed-rank*.json`).
The graph gate (`nccl-graph-rank*.json`) checks an eight-node ring on RTX 5070 Ti
and RTX 4070 Ti SUPER: serialized, automatic, and changed-input forward/VJP.
Both ranks report `correct=true`; automatic execution has two interior and two
boundary rows. The gradient is 3 at scale 1 and 6 at scale 2. Both files record
identical Python source, benchmark and NCCL library hashes. The library is NCCL
2.28.9 (SHA-256 `1792291e26d2b27fe51e59cf4dbd361dbfa84cbfe8ee8196c087c7ebf5146d48`).

Launch environment on both hosts:

```sh
NCCL_SOCKET_IFNAME=tiga-wg0 NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1
```

Rank 0 used the local `ComfyUI/.venv` Python, built `gf-opt`/`gf-translate`, and
its NCCL library. Rank 1 used the isolated `tiga-paper-20260919.Axi0tf` checkout,
the existing tilelang Python with the isolated Triton dependency directory,
and the matching copied NCCL library. Neither Python environment was upgraded.

```sh
# On each host, with its recorded PYTHONPATH, compiler and NCCL paths:
timeout 50s python -m benchmarks.distributed.multi_host_nccl_probe \
  --rank RANK --host 10.203.77.1 --port 29657 --output rank-probe.json
timeout 90s python -m benchmarks.distributed.multi_host_gpu_gate \
  --rank RANK --host 10.203.77.1 --port 29659 --transport nccl --output rank-graph.json
```

These are correctness checks. The first-exchange durations include connection
setup and are not a bandwidth curve. NCCL used its Socket network plugin over
WireGuard, not GPUDirect RDMA. The earlier TCP scaling measurements are unchanged.

Configuration references: [WireGuard quick start](https://www.wireguard.com/quickstart/)
and [NCCL interface selection](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname).
