"""Recorded, bounded supplementary GPU experiments; no predicted measurements.

The controller runs every configuration in a fresh process. Numerical checking
is outside synchronized wall timers. This compares execution paths on identical
fixed topology, not different search algorithms or isolated compiler passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNNY_URL = 'https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bunny_points(archive):
    """Read just the ASCII vertices; never extract untrusted archive paths."""
    with tarfile.open(archive) as source:
        member = source.extractfile('bunny/reconstruction/bun_zipper.ply')
        if member is None:
            raise ValueError('Missing Stanford Bunny reconstruction')
        lines = member.read().decode('ascii').splitlines()
    end = lines.index('end_header')
    assert 'format ascii 1.0' in lines[:end]
    count = int(next(x.split()[-1] for x in lines[:end]
                     if x.startswith('element vertex ')))
    return [[float(v) for v in line.split()[:3]]
            for line in lines[end + 1:end + 1 + count]]


def worker(args):
    import math
    import torch
    import tiga as tg
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU/reference fallback accepted')
    rng = torch.Generator(device='cuda').manual_seed(args.seed)
    before = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.used,memory.total',
        '--format=csv,noheader'], text=True).strip()

    class Weighted(tg.MessagePassing):
        reducer = tg.sum()

        def edge(self, src, dst, edge):
            return src.x * edge.w

    class EdgeNN(tg.MessagePassing):
        reducer = tg.sum()

        def __init__(self, module):
            super().__init__()
            self.mlp = tg.nn.trace(module)

        def edge(self, src, dst, edge):
            return self.mlp(edge.displacement, src.x)

    if args.workload == 'edgenn':
        p = torch.tensor(bunny_points(args.bunny), device='cuda')
        p = (p - p.mean(0)) / (p.max(0).values - p.min(0).values).max()
        cutoff = 0.015
    else:
        p = (torch.rand(args.nodes, 3, device='cuda', generator=rng)*1024).floor()/1024
        cutoff = math.sqrt(math.floor((1024*(32/(args.nodes*4*math.pi/3))**(1/3))**2)+.5)/1024
    n = len(p)
    if args.workload == 'edgenn':
        p.requires_grad_()
    # One identical radius-search result is shared by all consumers. Search is
    # deliberately excluded from this fixed-topology measurement.
    started = time.perf_counter()
    relation = tg.Graph.radius(p, cutoff=cutoff)
    row, col = relation.resolve_csr()
    destinations = torch.arange(n, device='cuda').repeat_interleave(row[1:]-row[:-1])
    torch.cuda.synchronize()
    setup_ms = 1000*(time.perf_counter()-started)
    topology_sha = hashlib.sha256(row.cpu().numpy().tobytes()+col.cpu().numpy().tobytes()).hexdigest()
    edges = col.numel()
    degrees = row[1:]-row[:-1]
    degree_stats = dict(mean=float(degrees.float().mean()), maximum=int(degrees.max()))
    del degrees
    x = torch.randn(n, args.width, device='cuda', generator=rng)
    snapshots = [x.cpu(), (x*1.01+.001).cpu()]
    expected = []
    program = None
    variables = ()
    oracle_errors = []

    if args.workload == 'edgenn':
        # Real reconstructed geometry; synthetic features and regression target.
        # This is a differentiated operator workload, not trained accuracy.
        x.requires_grad_()
        mlp = nn.Sequential(nn.Linear(args.width+3, 32), nn.ReLU(), nn.Linear(32, 3)).cuda()
        target = p.detach().clone()
        variables = (x, p, *mlp.parameters())

        def eager():
            message = mlp(torch.cat((p[col]-p[destinations], x[col]), -1))
            return torch.zeros(n, 3, device='cuda').index_add(0, destinations, message)

        for values in snapshots:
            with torch.no_grad():
                x.copy_(values)
            out = eager()
            grads = torch.autograd.grad((out-target).square().mean(), variables)
            expected.append((out.detach().cpu(), [g.detach().cpu() for g in grads]))
            del out, grads
        if args.provider == 'tiga':
            program = EdgeNN(mlp)
            call = lambda: program(graph=relation, src={'x': x}, dst={})
        else:
            call = eager
    else:
        weight = (p[col]-p[destinations]).norm(dim=-1)

        def eager():
            return torch.zeros_like(x).index_add(0, destinations, x[col]*weight[:, None])

        for values in snapshots:
            x.copy_(values)
            expected.append((eager().cpu(), []))
        if args.provider == 'tiga':
            graph = tg.Graph.from_csr(row, col, num_src=n)
            program = Weighted()
            call = lambda: program(graph=graph, src={'x': x}, dst={}, edge={'w': weight})
        elif args.provider == 'sparse':
            sparse = torch.sparse_csr_tensor(row, col, weight, size=(n, n))
            call = lambda: torch.sparse.mm(sparse, x)
        else:
            call = eager
        # Same live topology and geometry metadata in each worker. No peer's
        # message arrays or oracle outputs are retained on the device.

    def evaluate(step, check=True):
        with torch.no_grad():
            x.copy_(snapshots[step % 2])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = call()
        torch.cuda.synchronize()
        middle = time.perf_counter()
        gradients = (torch.autograd.grad((out-target).square().mean(), variables)
                     if variables else ())
        if variables:
            torch.cuda.synchronize()
        finish = time.perf_counter() if variables else middle
        record = dict(forward_ms=1000*(middle-start),
                      backward_ms=1000*(finish-middle) if variables else 0.,
                      total_ms=1000*(finish-start),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if check:
            reference, reference_grads = expected[step % 2]
            observed = out.detach().cpu()
            torch.testing.assert_close(observed, reference, rtol=5e-4, atol=5e-5)
            err = dict(output_max_abs=float((observed-reference).abs().max()),
                       gradients_max_abs=[], gradients_relative_l2=[])
            for actual, wanted in zip(gradients, reference_grads):
                actual = actual.detach().cpu()
                torch.testing.assert_close(actual, wanted, rtol=8e-4, atol=8e-5)
                err['gradients_max_abs'].append(float((actual-wanted).abs().max()))
                relative = float((actual-wanted).norm()/wanted.norm().clamp_min(1e-12))
                assert relative < 2e-3, ('gradient relative L2', relative)
                err['gradients_relative_l2'].append(relative)
            oracle_errors.append(err)
        return record

    # Context, common graph construction and eager oracle are initialized before
    # the first Tiga call; first-call time is not pure compiler time.
    first = evaluate(0)
    for i in range(4):
        evaluate(i)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    samples = [evaluate(i) for i in range(args.repeats)]
    explanation = program.explain() if program is not None else None
    if program is not None:
        if args.workload == 'edgenn':
            assert 'gf-python-emit-edge-nn-tile-vjp' in explanation, explanation
        else:
            assert 'ttir' in explanation.lower() and 'torch.sparse.mm' not in explanation, explanation
    project = Path(args.project).resolve()
    record = dict(schema='tiga.paper.supplement.v1', workload=args.workload,
                  provider=args.provider, nodes=n, edges=edges, width=args.width,
                  seed=args.seed, trial=args.trial, phase=args.phase,
                  cutoff=cutoff, degree=degree_stats, topology_sha256=topology_sha,
                  first_call=first, warmup=4, samples=samples,
                  medians={k: statistics.median(s[k] for s in samples)
                           for k in ('forward_ms', 'backward_ms', 'total_ms')},
                  peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in samples),
                  common_topology_setup_ms=setup_ms, numerical_checks=oracle_errors,
                  correct=True, explanation=explanation,
                  timing='synchronized host wall; fixed topology; feature copy and validation excluded; backward includes MSE loss, no optimizer',
                  memory='Torch allocated: live inputs/topology/cache plus outputs/tape/workspace; excludes reserve, CUDA context and other processes',
                  cache={key: os.environ.get(key) for key in
                         ('TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH')},
                  versions=dict(torch=torch.__version__, triton=__import__('triton').__version__),
                  gpu_before=before, timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
                  source_sha256={str(p.relative_to(project)):digest(p)
                                 for p in sorted((project/'python/tiga').rglob('*.py'))},
                  compiler_sha256={key:digest(os.environ[key]) for key in ('TIGA_OPT','TIGA_TRANSLATE')},
                  benchmark_sha256=digest(__file__),
                  dataset=(dict(url=BUNNY_URL, archive_sha256=digest(args.bunny),
                                member='bunny/reconstruction/bun_zipper.ply')
                           if args.workload == 'edgenn' else None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(args.output.name, record['medians'], record['peak_allocated_bytes'], flush=True)


def controller(args):
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for trial in range(args.trials):
        seed = 20260921 + trial
        for n in (8192, 32768):
            for width in (32, 128):
                for provider in ('tiga','eager','sparse'):
                    jobs.append(('fusion',provider,n,width,trial,seed,'warm'))
        for width in (8,32):
            for provider in ('tiga','eager'):
                jobs.append(('edgenn',provider,35947,width,trial,seed,'warm'))
    random.Random(20260921).shuffle(jobs)
    if args.quick:
        jobs = [('fusion','tiga',8192,32,0,20260921,'warm'),
                ('edgenn','tiga',35947,8,0,20260921,'warm')]
    logs = []

    def run(job, cache=None):
        workload,provider,n,width,trial,seed,phase = job
        name = f'{workload}-{provider}-{n}-{width}-r{trial}-{phase}.json'
        path = args.output/name
        if path.exists():
            raise FileExistsError(f'Refusing to replace existing measurement: {path}')
        command = [sys.executable,__file__,'--worker','--workload',workload,
                   '--provider',provider,'--nodes',str(n),'--width',str(width),
                   '--trial',str(trial),'--seed',str(seed),'--phase',phase,
                   '--repeats',str(args.repeats),'--project',str(args.project),
                   '--bunny',str(args.bunny),'--output',str(path)]
        env = os.environ.copy()
        if cache is not None:
            for key, sub in [('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:
                env[key] = str(cache/sub)
            env.pop('TRITON_ALWAYS_COMPILE',None)
        result = subprocess.run(command,env=env,text=True,capture_output=True)
        logs.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        (args.output/'runs.json').write_text(json.dumps(logs,indent=2)+'\n')
        print(result.stdout, end='', flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr)

    for job in jobs:
        run(job)
    if not args.quick:
        for trial in range(args.trials):
            cache = Path(tempfile.mkdtemp(prefix='tiga-paper-jit-'))
            for phase in ('cold','disk-cache'):
                run(('fusion','tiga',32768,32,trial,20260921+trial,phase),cache)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--workload',choices=['fusion','edgenn'],default='fusion')
    parser.add_argument('--provider',choices=['tiga','eager','sparse'],default='tiga')
    parser.add_argument('--nodes',type=int,default=8192)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20260921)
    parser.add_argument('--trial',type=int,default=0)
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--phase',default='warm')
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--project',type=Path,default=ROOT.parent/'tiga-lang')
    parser.add_argument('--bunny',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if min(args.repeats,args.trials,args.nodes,args.width) < 1:
        parser.error('positive sizes and sample counts required')
    (worker if args.worker else controller)(args)
