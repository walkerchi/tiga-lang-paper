"""Import, validate and summarize the measured billion-edge forward profile."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/paging-1b'


def read(path):
    return json.loads(path.read_text())


def validate():
    for name, digest in read(DATA / 'index.json').items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
    r = read(DATA / 'profile.json')
    assert r['status'] == 'ok' and r['edges'] == 1_000_000_000
    assert r['checked_values'] == 2_000_000_000 and r['max_abs_error'] == 0
    assert r['execution']['pages'] == 954
    p = r['profile']
    assert p['cuda_profiler_capture'] and p['fixture_eviction_hint']
    assert all(v >= 0 for v in p['exclusive_host_seconds'].values())
    assert abs(sum(p['exclusive_host_seconds'].values()) - p['forward_seconds']) < 1e-7
    with tarfile.open(DATA / 'source.tar.gz') as archive:
        expected = dict(r['provenance']['source_hashes'])
        expected['benchmarks/memory_hierarchy/billion_edges.py'] = r['provenance']['benchmark_sha256']
        expected['benchmarks/memory_hierarchy/profile_paging.py'] = p['benchmark_sha256']
        for name, digest in expected.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
    return r


def summary():
    r = validate()
    with (DATA / 'cuda_cuda_gpu_kern_sum.csv').open() as file:
        kernels = list(csv.DictReader(file))
    with (DATA / 'cuda_cuda_gpu_mem_time_sum.csv').open() as file:
        transfers = list(csv.DictReader(file))
    device = {q['Operation']: float(q['Total Time (ns)']) / 1e9 for q in transfers}
    return dict(edges=r['edges'], host_seconds=r['profile']['forward_seconds'],
                kernel_seconds=sum(float(q['Total Time (ns)']) for q in kernels) / 1e9,
                h2d_seconds=device['[CUDA memcpy Host-to-Device]'],
                d2h_seconds=device['[CUDA memcpy Device-to-Host]'],
                exclusive_host_seconds=r['profile']['exclusive_host_seconds'],
                staged_bytes=r['profile']['bytes'],
                physical_read_bytes=r['forward_io_delta']['read_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-run', type=Path)
    parser.add_argument('--project', type=Path, default=ROOT.parent / 'graphforgev2')
    args = parser.parse_args()
    if args.import_run:
        DATA.mkdir(parents=True, exist_ok=True)
        for name in ('profile.json', 'trace.nsys-rep', 'cuda_cuda_gpu_kern_sum.csv',
                     'cuda_cuda_gpu_mem_time_sum.csv', 'cuda_cuda_gpu_mem_size_sum.csv'):
            shutil.copy2(args.import_run / name, DATA / name)
        with tarfile.open(DATA / 'source.tar.gz', 'w:gz') as archive:
            for directory in ('python', 'lib', 'include', 'tools', 'cmake',
                              'python_bindings', 'benchmarks/memory_hierarchy'):
                for path in sorted((args.project / directory).rglob('*')):
                    if path.is_file() and '__pycache__' not in path.parts:
                        archive.add(path, arcname=str(path.relative_to(args.project)))
            for name in ('CMakeLists.txt', 'pyproject.toml'):
                archive.add(args.project / name, arcname=name)
        index = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(DATA.iterdir()) if p.is_file() and p.name != 'index.json'}
        (DATA / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(summary(), indent=2))


if __name__ == '__main__':
    main()
