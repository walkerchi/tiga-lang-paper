"""Validate recorded supplementary runs and render tables/figures."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/supplement'
BUNNY_SHA = 'a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84'


def validate(data=DATA):
    records = [json.loads(p.read_text()) for p in sorted(data.glob('*-r*.json'))]
    expected = {(w,p,n,f,r,phase)
                for r in range(3)
                for w,ps,ns,fs,phases in (
                    ('fusion',('tiga','eager','sparse'),(8192,32768),(32,128),('warm',)),
                    ('edgenn',('tiga','eager'),(35947,),(8,32),('warm',)),
                    ('fusion',('tiga',),(32768,),(32,),('cold','disk-cache')))
                for p in ps for n in ns for f in fs for phase in phases}
    key = lambda r: tuple(r[k] for k in ('workload','provider','nodes','width','trial','phase'))
    assert len(records) == len(expected) == 54
    assert {key(r) for r in records} == expected
    benchmark = hashlib.sha256((ROOT/'tools/run_supplement.py').read_bytes()).hexdigest()
    for r in records:
        assert r['schema'] == 'tiga.paper.supplement.v1' and r['correct']
        assert len(r['samples']) == 10 and r['warmup'] == 4
        assert r['benchmark_sha256'] == benchmark
        for field in ('source_commit','source_sha256','compiler_sha256','versions'):
            assert r[field] == records[0][field], field
        assert len(r['numerical_checks']) == 15
        for checks in r['numerical_checks']:
            if r['workload'] == 'edgenn':
                assert len(checks['gradients_relative_l2']) == 6
                assert all(0 <= x < 2e-3 for x in checks['gradients_relative_l2'])
        for field in ('forward_ms','backward_ms','total_ms'):
            assert r['medians'][field] == st.median(s[field] for s in r['samples'])
        assert r['peak_allocated_bytes'] == max(s['peak_allocated_bytes'] for s in r['samples'])
        for s in r['samples']:
            assert s['forward_ms'] > 0 and s['backward_ms'] >= 0
            assert abs(s['total_ms']-s['forward_ms']-s['backward_ms']) < 1e-8
        peers = [q for q in records if (q['workload'],q['nodes'],q['width'],q['trial']) ==
                 (r['workload'],r['nodes'],r['width'],r['trial'])]
        assert all(q['topology_sha256'] == r['topology_sha256'] for q in peers)
        if r['workload'] == 'edgenn':
            assert r['dataset']['archive_sha256'] == BUNNY_SHA
            if r['provider'] == 'tiga':
                assert 'gf-python-emit-edge-nn-tile-vjp' in r['explanation']
        if r['phase'] in ('cold','disk-cache'):
            assert all(r['cache'].values())
    logs = json.loads((data/'runs.json').read_text())
    assert len(logs) == 54 and all(r['returncode'] == 0 for r in logs)
    with tarfile.open(data/'source.tar.gz') as archive:
        for name, sha in records[0]['source_sha256'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == sha
    return records


def summarize(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['workload'],r['provider'],r['nodes'],r['width'],r['phase'])].append(r)
    result = {}
    for key, rows in grouped.items():
        assert len(rows) == 3
        summary = dict(edges=[r['edges'] for r in rows])
        for field in ('forward_ms','backward_ms','total_ms'):
            medians = [r['medians'][field] for r in rows]
            summary[field] = dict(median=st.median(medians),low=min(medians),high=max(medians))
        for label, values in (
            ('first_ms',[r['first_call']['total_ms'] for r in rows]),
            ('memory_mib',[r['peak_allocated_bytes']/2**20 for r in rows])):
            summary[label] = dict(median=st.median(values),low=min(values),high=max(values))
        result[key] = summary
    return result


def render(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    results = summarize(records)
    get = lambda w,p,n,f,phase='warm': results[(w,p,n,f,phase)]
    colors = {'tiga':'#087f76','eager':'#64748b','sparse':'#c27b24'}
    fig, axes = plt.subplots(1,2,figsize=(8,3.25),layout='constrained')
    combinations = [(8192,32),(8192,128),(32768,32),(32768,128)]
    for j,p in enumerate(('eager','sparse','tiga')):
        values = [get('fusion',p,n,f)['forward_ms'] for n,f in combinations]
        means = [v['median'] for v in values]
        error = [[v['median']-v['low'] for v in values], [v['high']-v['median'] for v in values]]
        axes[0].bar(np.arange(4)+(j-1)*.25,means,width=.23,color=colors[p],
                    label={'eager':'Torch gather/scatter','sparse':'Torch sparse','tiga':'Tiga'}[p],
                    yerr=error,capsize=2)
    axes[0].set_xticks(range(4),['8K\nF=32','8K\nF=128','32K\nF=32','32K\nF=128'])
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Nodes (K = 1,024) and feature width F', fontsize=8)
    axes[0].set_ylabel('Forward wall time (ms)')
    axes[0].set_title('(a) Same stored radius graph')
    axes[0].legend(fontsize=7,frameon=False)
    for j,p in enumerate(('eager','tiga')):
        values = [get('edgenn',p,35947,f) for f in (8,32)]
        means = [v['total_ms']['median'] for v in values]
        error = [[v['total_ms']['median']-v['total_ms']['low'] for v in values],
                 [v['total_ms']['high']-v['total_ms']['median'] for v in values]]
        axes[1].bar(np.arange(2)+(j-.5)*.32,means,width=.29,color=colors[p],
                    label='Torch eager' if p=='eager' else 'Tiga',yerr=error,capsize=2)
    axes[1].set_xticks(range(2),['F=8','F=32'])
    axes[1].set_ylabel('Forward + loss + backward (ms)')
    axes[1].set_title('(b) EdgeNN on Bunny geometry')
    axes[1].legend(fontsize=8,frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    for ext in ('pdf','png'):
        fig.savefig(ROOT/f'figures/supplement-execution.{ext}',dpi=180)
    plt.close(fig)
    lines = [r'\begin{tabular}{llrrrr}',r'\toprule',
             r'Input width & Path & Forward (ms) & Backward (ms) & Total (ms) & Peak (MiB) \\',r'\midrule']
    for f in (8,32):
        for p in ('eager','tiga'):
            r=get('edgenn',p,35947,f)
            values=[r[k]['median'] for k in ('forward_ms','backward_ms','total_ms','memory_mib')]
            lines.append(f"{f} & {'Torch eager' if p=='eager' else 'Tiga'} & "+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'generated/supplement-edgenn.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{tabular}{lrr}',r'\toprule',r'Path & First result (ms) & Warm call (ms) \\',r'\midrule']
    for p,phase,label in [('tiga','cold','Tiga, empty isolated caches'),
                          ('tiga','disk-cache','Tiga, reused disk caches'),
                          ('sparse','warm','Torch sparse, fresh process'),
                          ('eager','warm','Torch gather/scatter, fresh process')]:
        r=get('fusion',p,32768,32,phase)
        first=r['first_ms']; warm=r['forward_ms']
        lines.append(f"{label} & {first['median']:.1f} ({first['low']:.1f}--{first['high']:.1f}) & {warm['median']:.3f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'generated/supplement-jit.tex').write_text('\n'.join(lines)+'\n')
    cold=get('fusion','tiga',32768,32,'cold')
    eager=get('fusion','eager',32768,32)
    delta=eager['forward_ms']['median']-cold['forward_ms']['median']
    assert delta > 0
    crossover=1+math.ceil((cold['first_ms']['median']-eager['first_ms']['median'])/delta)
    (ROOT/'generated/supplement-amortization.tex').write_text(
        f'Using the reported medians, the model predicts a crossover against explicit '
        f'Torch gather/scatter after approximately {crossover:,} calls. '
        'This is a model-derived estimate, not an observed crossover.\n')
    large={p:get('fusion',p,32768,128) for p in ('tiga','eager','sparse')}
    summary={'records':[dict(workload=k[0],provider=k[1],nodes=k[2],width=k[3],phase=k[4],**v)
                        for k,v in sorted(results.items())]}
    (ROOT/'generated/supplement-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text = ('At 32,768 nodes and 128 features, measured forward times are '
            + ', '.join(f"{large[p]['forward_ms']['median']:.3f} ms" for p in ('tiga','eager','sparse'))
            + ' for Tiga, Torch gather/scatter and Torch sparse, respectively. '
            + 'Median process-peak allocations are '
            + ', '.join(f"{large[p]['memory_mib']['median']:.1f} MiB" for p in ('tiga','eager','sparse'))
            + '. These measurements hold search and topology fixed; execution strategy and launch overhead still differ.\n')
    (ROOT/'generated/supplement-findings.tex').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    records=validate()
    if not args.check:
        render(records)
    else:
        print('Validated 54 fresh-process supplement records and matched topology/provenance.')
