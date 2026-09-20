"""Import, validate and render the measured billion-edge capacity experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/billion'
SIZES = (10_000_000, 100_000_000, 1_000_000_000)
TRIALS = (1, 2, 3)


def validate(rows):
    assert len(rows) == 9
    assert {(r['edges'], r['trial']) for r in rows} == {(e, t) for e in SIZES for t in TRIALS}
    provenance = rows[0]['provenance']
    for row in rows:
        assert row['status'] == 'ok' and row['mode'] == 'paged'
        assert row['gpu_before']['uuid'] == rows[0]['gpu_before']['uuid']
        assert row['features'] == 32 and row['degree'] == 16
        assert row['nodes'] * 16 == row['edges']
        assert row['page_rows'] == 65536
        assert row['execution']['page_backends'] == ['cuda-ttir-triton']
        assert row['execution']['pages'] == (row['nodes']+65535)//65536
        assert row['max_abs_error'] == 0 and row['checked_values'] == row['nodes']*32
        assert row['forward_s'] > 0 and row['check_s'] > 0
        field = row['nodes']*32*4
        topology = (row['nodes']+1+row['edges'])*8
        assert row['resident_lower_bound_bytes'] == topology+2*field
        assert row['stored_bytes'] == topology+field
        assert row['output_bytes'] == field
        peak = row['forward_memory']['peak_bytes']['device']
        assert field <= peak <= row['budget_bytes']
        assert row['budget_bytes'] < row['gpu_before']['total_bytes']
        assert row['gpu_samples'] and not row['gpu_monitor_errors']
        assert row['memory']['live_bytes']['device'] == 0
        for name in ('source_hashes', 'compiler_hashes', 'benchmark_sha256'):
            assert row['provenance'][name] == provenance[name]
        peers = [r for r in rows if r['edges'] == row['edges']]
        assert all(r['ingest']['payload_sha256'] == row['ingest']['payload_sha256'] for r in peers)
        if row['edges'] == 1_000_000_000:
            assert row['resident_lower_bound_bytes'] > row['gpu_before']['total_bytes']
            # The capacity conclusion also holds with 32-bit indices.
            assert topology//2 + 2*field > row['gpu_before']['total_bytes']


def summarize(rows):
    summary = []
    for edges in SIZES:
        group = [r for r in rows if r['edges'] == edges]
        times = [r['forward_s'] for r in group]
        summary.append(dict(edges=edges, median_s=statistics.median(times),
                            min_s=min(times), max_s=max(times),
                            resident_gib=group[0]['resident_lower_bound_bytes']/2**30,
                            output_gib=group[0]['output_bytes']/2**30,
                            native_peak_gib=max(r['forward_memory']['peak_bytes']['device'] for r in group)/2**30,
                            rss_peak_gib=max(r['forward_peak_rss_bytes'] for r in group)/2**30,
                            sampled_device_gib=max(s['used_bytes'] for r in group for s in r['gpu_samples'])/2**30,
                            physical_gib=group[0]['gpu_before']['total_bytes']/2**30))
    return summary


def render(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points = summarize(rows)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1))
    x = [r['edges'] for r in points]
    axes[0].plot(x, [r['resident_gib'] for r in points], '--o', color='#64748b',
                 label='Full-residency lower bound')
    axes[0].plot(x, [r['native_peak_gib'] for r in points], '-o', color='#087f76',
                 label='Paged: measured native peak')
    axes[0].axhline(points[0]['physical_gib'], ls=':', color='#b45309', label='Physical device capacity')
    axes[0].set_ylabel('Memory (GiB)')
    axes[0].set_ylim(0, 26)
    axes[0].legend(fontsize=7, loc='upper left')
    for key, color, offset in [('resident_gib', '#64748b', 8),
                               ('native_peak_gib', '#087f76', -28)]:
        axes[0].annotate(f"{points[-1][key]:.1f} GiB", (x[-1], points[-1][key]),
                         xytext=(-4, offset), textcoords='offset points',
                         ha='right', fontsize=8, color=color)
    times = [r['median_s'] for r in points]
    axes[1].errorbar(x, times,
                     yerr=[[r['median_s']-r['min_s'] for r in points],
                           [r['max_s']-r['median_s'] for r in points]],
                     fmt='-o', capsize=4, color='#087f76')
    axes[1].set_yscale('log')
    axes[1].set_yticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    axes[1].set_ylabel('Complete forward time (s, log scale)')
    axes[1].set_title('Median and range of 3 fresh processes', fontsize=8)
    axes[1].annotate(f"{times[-1]:.0f} s", (x[-1], times[-1]),
                     xytext=(-4, 9), textcoords='offset points', ha='right',
                     color='#087f76', fontsize=8)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(x, ['10M', '100M', '1B'])
        ax.set_xlabel('Directed edges')
        ax.grid(alpha=.18)
    fig.tight_layout()
    for suffix in ('pdf', 'png'):
        fig.savefig(ROOT/f'figures/billion-capacity.{suffix}', bbox_inches='tight', dpi=180,
                    **({'metadata': {'CreationDate': None}} if suffix == 'pdf' else {}))
    plt.close(fig)
    largest = points[-1]
    text = r'''\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/billion-capacity.pdf}
\caption{One billion explicit edges on a single RTX 5070 Ti. Left: measured
native allocation peak versus the calculated full-residency lower bound, not
an executed resident baseline. Right: median and range of three complete
forward calls in fresh processes, including host staging and output assembly.}
\label{fig:billion}
\end{figure}
'''
    text += (f"All nine trials complete with zero elementwise error. At one billion edges, "
             f"the median forward takes {largest['median_s']:.2f} s "
             f"(range {largest['min_s']:.2f}--{largest['max_s']:.2f} s). "
             f"The maximum native GPU allocation peak across trials is {largest['native_peak_gib']:.2f} GiB, "
             f"including the {largest['output_gib']:.2f} GiB output; forward process peak RSS is "
             f"{largest['rss_peak_gib']:.2f} GiB. The maximum sampled whole-device usage is "
             f"{largest['sampled_device_gib']:.2f} GiB, including existing services and runtime pools; "
             "one-second sampling is not an exact physical peak. Each trial checks all two billion "
             "FP32 output values outside the forward timing interval.\n")
    (ROOT/'generated/billion-results.tex').write_text(text)
    (ROOT/'generated/billion-summary.json').write_text(json.dumps(points, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.run:
        if not args.project:
            parser.error('--project is required with --run')
        rows = []
        for edges in SIZES:
            for trial in TRIALS:
                row = json.loads((args.run/f'final-e{edges}-r{trial}.json').read_text())
                row['trial'] = trial
                rows.append(row)
        validate(rows)
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        for path in ('benchmarks/memory_hierarchy/billion_edges.py',
                     'python/tiga/message_passing/paged.py'):
            shutil.copyfile(args.project/path, DATA/Path(path).name)
        for name in ('source.tar.gz', 'ENVIRONMENT.md', 'resident-preflight.json'):
            shutil.copyfile(args.run/name, DATA/name)
        shutil.copyfile(args.project/'benchmarks/memory_hierarchy/README.md', DATA/'REPRODUCE.md')
        index = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in DATA.iterdir() if p.name != 'index.json'}
        (DATA/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    for name, digest in json.loads((DATA/'index.json').read_text()).items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest, name
    rows = json.loads((DATA/'results.json').read_text())
    validate(rows)
    assert hashlib.sha256((DATA/'billion_edges.py').read_bytes()).hexdigest() == rows[0]['provenance']['benchmark_sha256']
    assert hashlib.sha256((DATA/'paged.py').read_bytes()).hexdigest() == rows[0]['provenance']['source_hashes']['python/tiga/message_passing/paged.py']
    with tarfile.open(DATA/'source.tar.gz', 'r:gz') as archive:
        for name, digest in rows[0]['provenance']['source_hashes'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
        script = archive.extractfile('benchmarks/memory_hierarchy/billion_edges.py').read()
        assert hashlib.sha256(script).hexdigest() == rows[0]['provenance']['benchmark_sha256']
    if not args.check:
        render(rows)
    print('Nine fully checked billion-edge scaling trials validated.')


if __name__ == '__main__':
    main()
