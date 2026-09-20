"""Matched single-GPU and two-host CUDA forward timings (trusted TCP peers only).

Run one rank per host with matching arguments and source. The two-rank critical
path is the maximum of paired rank durations, not their mean. No GPU-direct or
GPU event overlap claim is made for this host-staged transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import time

import numpy as np
import tiga as tg
from tiga.distributed import DistributedRuntime, TCPTransport


class MeanNeighbors(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x

    def node(self, dst, aggregate):
        return aggregate * (1.0 / 16)


def native(array, device):
    array = np.ascontiguousarray(array)
    dtype = {np.dtype('float32'): tg.float32, np.dtype('int64'): tg.int64}[array.dtype]
    value = tg.empty(array.shape, dtype=dtype, device=device)
    value._buffer.write(array.tobytes())
    return value


def case(nodes, features):
    """25% boundary rows, 16 edges/row; IDs and feature values are deterministic."""
    ids = np.arange(nodes, dtype=np.int64)
    half = nodes // 2
    columns = (ids[:, None] % half + np.arange(1, 17)) % half + (ids[:, None] // half) * half
    columns[ids % 4 == 0] = (columns[ids % 4 == 0] + half) % nodes
    values = ((ids[:, None] % 97) + np.arange(features)[None, :] % 7).astype(np.float32)
    expected = np.zeros_like(values)
    for e in range(16):
        expected += values[columns[:, e]]
    return np.arange(nodes + 1, dtype=np.int64) * 16, columns.ravel(), values, expected / 16


class MeasuredTransport:
    def __init__(self, base):
        self.base = base
        self.rank, self.world_size = base.rank, base.world_size
        self.prefer_compute_overlap = True
        self.events = []

    def send(self, peer, data):
        return self.base.send(peer, data)

    def receive(self, peer):
        return self.base.receive(peer)

    def send_bytes(self, peer, data):
        start = time.perf_counter_ns()
        self.base.send_bytes(peer, data)
        self.events.append(('send', len(data), start, time.perf_counter_ns()))

    def receive_bytes(self, peer, size):
        start = time.perf_counter_ns()
        value = self.base.receive_bytes(peer, size)
        self.events.append(('receive', size, start, time.perf_counter_ns()))
        return value

    def barrier(self, label):
        if self.rank == 0:
            self.base.send(1, label)
            assert self.base.receive(1) == label
        else:
            assert self.base.receive(0) == label
            self.base.send(0, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=29641)
    parser.add_argument('--sizes', default='8192,32768,131072')
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(n) for n in args.sizes.split(',')]
    if any(n < 32 or n % 2 for n in sizes) or min(args.features, args.repeats) < 1:
        parser.error('even sizes >=32 and positive features/repeats required')
    device = 'cuda:0'
    connect = TCPTransport.host if args.rank == 0 else TCPTransport.join
    base = connect(args.rank, 2, host=args.host, port=args.port, timeout=180)
    transport = MeasuredTransport(base)
    root = Path(__file__).resolve().parents[2]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root/'python/tiga').rglob('*.py'))}
    result = dict(schema='tiga.two-host-scaling.v1', rank=args.rank,
                  hostname=socket.gethostname(), platform=platform.platform(),
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,memory.total',
                                               '--format=csv,noheader'], text=True).strip(),
                  transport='tcp-host-staged', features=args.features, degree=16,
                  boundary_fraction=.25, warmup=args.warmup, repeats=args.repeats,
                  versions={name: importlib.metadata.version(name) for name in ('numpy','triton')},
                  python=platform.python_version(),
                  compiler_hashes={name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
                                   for name in ('TIGA_OPT','TIGA_TRANSLATE') if os.environ.get(name)},
                  source_hashes=hashes, benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='host wall time through synchronized output; topology/input creation, barriers and oracle excluded; changed inputs; forward only', rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DistributedRuntime(transport) as runtime:
            for nodes in sizes:
                rowptr, columns, values, expected = case(nodes, args.features)
                full = tg.Graph.from_csr(native(rowptr, device), native(columns, device), num_src=nodes)
                sharded = full.halo(tg.DeviceMesh('cuda', 2), depth=1)
                begin, end = args.rank * (nodes//2), (args.rank+1) * (nodes//2)
                program = MeanNeighbors()
                for mode in ('single', 'serialized', 'automatic'):
                    samples, traces, traffic = [], [], []
                    runtime._force_serialized = True if mode == 'serialized' else None
                    for step in range(args.warmup + args.repeats):
                        scale = float(1 + step % 3)
                        x = native((values if mode == 'single' else values[begin:end]) * scale, device)
                        transport.barrier((nodes, mode, step))
                        transport.events.clear()
                        start = time.perf_counter_ns()
                        y = program(graph=full if mode == 'single' else sharded, src={'x': x}, dst={})
                        y.realize()
                        if y.ready_event is not None:
                            y.ready_event.wait()
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        np.testing.assert_array_equal(y.to_numpy(), (expected if mode == 'single' else expected[begin:end]) * scale)
                        if step >= args.warmup:
                            samples.append(elapsed)
                            traces.append(None if mode == 'single' else runtime.last_execution_trace)
                            events = transport.events[:]
                            traffic.append(dict(send_bytes=sum(e[1] for e in events if e[0]=='send'),
                                                receive_bytes=sum(e[1] for e in events if e[0]=='receive'),
                                                socket_span_ms=(max(e[3] for e in events)-min(e[2] for e in events))/1e6 if events else 0))
                        del y, x
                    result['rows'].append(dict(nodes=nodes, mode=mode, correct=True,
                                               median_ms=statistics.median(samples), samples_ms=samples,
                                               traces=traces, traffic=traffic))
                    args.output.write_text(json.dumps(result, indent=2)+'\n')
                    print(args.rank, nodes, mode, round(statistics.median(samples), 3), flush=True)
                del full, sharded
    finally:
        base.close()


if __name__ == '__main__':
    main()
