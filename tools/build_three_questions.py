"""Validate and render exactly three question-led experiment figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/three-questions'
Q1_DATA = ROOT/'data/comparison-v3'
COLORS = {'tiga': '#087f76', 'pyg': '#64748b', 'sparse': '#c27b24', 'materialized': '#c27b24'}


def read(path):
    return json.loads(path.read_text())


def validate_comparison(rows):
    assert rows
    revised = rows[0]['schema'] == 'tiga.three-questions.comparison.v2'
    knn_sizes = (1024,4096,16384,32768,65536,131072) if revised else (1024,4096,16384)
    expected = {(w, n, p) for w, sizes, peers in (
        ('csr', (8192,32768,131072), ('tiga','pyg','sparse')),
        ('radius', (8192,32768,131072), ('tiga','pyg','materialized')),
        ('knn', knn_sizes, ('tiga','pyg','materialized')))
        for n in sizes for p in peers}
    assert len(rows) == len(expected) == (36 if revised else 27)
    assert {(r['workload'],r['nodes'],r['provider']) for r in rows} == expected
    for r in rows:
        assert r['correct'] and len(r['samples_ms']) == r['repeats'] == 10
        assert all(v > 0 for v in r['samples_ms'])
        assert r['median_ms'] == st.median(r['samples_ms'])
        assert r['peak_allocated_bytes'] == max(r['peak_samples_bytes']) > 0
        assert r['features'] == (32 if r['workload'] == 'csr' else 1)
        for field in ('source_sha256','compiler_sha256','benchmark_sha256','versions'):
            assert r[field] == rows[0][field]
        assert r['schema'] == rows[0]['schema']
        if revised:
            assert r['torch_distance_pair_budget'] == 16*1024*1024
            assert r['torch_baseline_sha256'] == rows[0]['torch_baseline_sha256']
        group = [q for q in rows if (q['workload'],q['nodes']) == (r['workload'],r['nodes'])]
        assert all(q['edges'] == r['edges'] for q in group)
        if r['provider'] == 'tiga':
            assert 'gf-kernel-to-ttir-' in r['diagnostics']


def validate_distributed(ranks):
    assert {r['rank'] for r in ranks} == {0,1}
    for r in ranks:
        assert r['transport'] == 'nccl-socket'
        assert r['repeats'] == 5 and r['warmup'] == 2
        assert len(r['rows']) == 9
        assert {(q['nodes'],q['mode']) for q in r['rows']} == {(n,m) for n in (32768,131072,524288) for m in ('single','serialized','automatic')}
        for field in ('source_hashes','compiler_hashes','benchmark_sha256','boundary_every','features','degree','nccl_version','nccl_library_sha256'):
            assert r[field] == ranks[0][field]
        for q in r['rows']:
            assert q['correct'] and len(q['samples_ms']) == 5
            assert all(v > 0 for v in q['samples_ms'])
            assert len(q['traffic']) == len(q['traces']) == 5
            if q['mode'] == 'automatic':
                assert all(t['schedule'] == 'interior||device-halo->boundary' for t in q['traces'])


def paired_samples(ranks, n, mode):
    records = [next(q for q in r['rows'] if q['nodes'] == n and q['mode'] == mode) for r in ranks]
    return [max(pair) for pair in zip(*(q['samples_ms'] for q in records), strict=True)]


def host_phases(record, step):
    t=record['traces'][step]; traffic=record['traffic'][step]
    boundaries=[traffic['forward_started_ns'], traffic['exchange_started_ns'],
                t['interior_started_ns'], t['interior_finished_ns'], traffic['forward_finished_ns']]
    values=[(end-begin)/1e6 for begin,end in zip(boundaries,boundaries[1:])]
    assert all(v >= 0 for v in values)
    assert abs(sum(values)-record['samples_ms'][step]) < 1e-6
    return values


def validate():
    from build_paging_profile import validate as validate_profile
    validate_profile()
    index = read(DATA/'index.json')
    for name, digest in index.items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest, name
    rows = read(DATA/'comparison.json')
    validate_comparison(rows)
    with tarfile.open(DATA/'source.tar.gz') as archive:
        for name,digest in rows[0]['source_sha256'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
        name='benchmarks/graph_operations/paper_comparison.py'
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == rows[0]['benchmark_sha256']
    for name in ('local','remote'):
        ranks=[read(DATA/f'{name}-rank{rank}.json') for rank in (0,1)]
        validate_distributed(ranks)
        assert ranks[0]['source_hashes'] == rows[0]['source_sha256']
        assert ranks[0]['compiler_hashes'] == rows[0]['compiler_sha256']
        with tarfile.open(DATA/'source.tar.gz') as archive:
            assert hashlib.sha256(archive.extractfile('benchmarks/distributed/paper_scaling.py').read()).hexdigest() == ranks[0]['benchmark_sha256']
        for rank in ranks:
            for q in rank['rows']:
                if q['mode']=='automatic':
                    for step in range(5): host_phases(q,step)
    for name in ('paging-profile','paging-warm'):
        r = read(DATA/f'{name}.json')
        assert r['status'] == 'ok' and r['checked_values'] == 20_000_000 and r['max_abs_error'] == 0
        p = r['profile']
        assert all(t >= 0 for t in p['exclusive_host_seconds'].values())
        assert abs(sum(p['exclusive_host_seconds'].values())-p['forward_seconds']) < 1e-7
    for workload in ('csr','radius','knn'):
        r=read(DATA/f'audit-{workload}.json')
        assert r['correct'] and r['provider']=='tiga'
        assert r['native_allocation_audit']['peak_bytes']['device'] == 0
        assert r['source_sha256'] == rows[0]['source_sha256']
    return rows


def validate_revised_comparison():
    index = read(Q1_DATA/'index.json')
    for name,digest in index.items():
        assert hashlib.sha256((Q1_DATA/name).read_bytes()).hexdigest() == digest, name
    rows=read(Q1_DATA/'comparison.json')
    validate_comparison(rows)
    assert rows[0]['schema']=='tiga.three-questions.comparison.v2'
    with tarfile.open(Q1_DATA/'source.tar.gz') as archive:
        expected=dict(rows[0]['source_sha256'])
        expected['benchmarks/graph_operations/paper_comparison.py']=rows[0]['benchmark_sha256']
        expected['benchmarks/graph_operations/torch_graph_baseline.py']=rows[0]['torch_baseline_sha256']
        for name,digest in expected.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest, name
    for workload in ('csr','radius','knn'):
        r=read(Q1_DATA/f'audit-{workload}.json')
        assert r['correct'] and r['nodes']==131072
        assert r['source_sha256']==rows[0]['source_sha256']
        assert r['compiler_sha256']==rows[0]['compiler_sha256']
        assert r['native_allocation_audit']['peak_bytes']['device']==0
    return rows


def save(fig, name):
    for ext in ('pdf','png'):
        fig.savefig(ROOT/f'figures/{name}.{ext}', dpi=190, bbox_inches='tight',
                    **({'metadata': {'CreationDate': None}} if ext == 'pdf' else {}))


def render_comparison(rows, plt):
    fig, axes = plt.subplots(2,3,figsize=(7.6,5.4), sharex='col')
    for col, workload in enumerate(('csr','radius','knn')):
        selected = [r for r in rows if r['workload'] == workload]
        sizes = sorted({r['nodes'] for r in selected})
        peers = ('tiga','pyg','sparse' if workload == 'csr' else 'materialized')
        labels = {'tiga':'Tiga', 'pyg':'PyG', 'sparse':'Torch sparse',
                  'materialized': 'Torch (chunked)'}
        for provider in peers:
            points = sorted((r for r in selected if r['provider'] == provider), key=lambda r:r['nodes'])
            medians = [r['median_ms'] for r in points]
            axes[0,col].plot(sizes, medians, '-o', ms=4, color=COLORS[provider], label=labels[provider])
            axes[0,col].fill_between(sizes, [min(r['samples_ms']) for r in points], [max(r['samples_ms']) for r in points], color=COLORS[provider], alpha=.12)
            axes[1,col].plot(sizes, [r['peak_allocated_bytes']/2**20 for r in points], '-o', ms=4, color=COLORS[provider])
        axes[0,col].set_title({'csr':'Stored CSR · F=32','radius':'Dynamic radius · F=1','knn':'Dynamic exact kNN · F=1'}[workload], fontsize=9)
        axes[0,col].legend(fontsize=7, loc='best')
        for row in range(2):
            ax=axes[row,col]
            ax.set_xscale('log'); ax.set_yscale('log')
            ticks = sizes if len(sizes)<=3 else [1024,4096,16384,131072]
            ax.set_xticks(ticks, [f'{n//1024}k' for n in ticks])
            ax.grid(alpha=.16)
        axes[1,col].set_xlabel('Nodes (k = 1,024)')
    axes[0,0].set_ylabel('Forward latency (ms) ↓')
    axes[1,0].set_ylabel('Peak allocated GPU memory (MiB) ↓')
    fig.tight_layout(h_pad=1.5)
    save(fig,'q1-performance-memory'); plt.close(fig)


def render_capacity(plt):
    from build_billion_results import summarize
    points = summarize(read(ROOT/'data/billion/results.json'))
    fig = plt.figure(figsize=(7.6,5.0))
    grid = fig.add_gridspec(2,2,height_ratios=[1.15,.65],hspace=.8,wspace=.35)
    ax=fig.add_subplot(grid[0,0]); x=[r['edges'] for r in points]
    ax.errorbar(x,[r['median_s'] for r in points],
                yerr=[[r['median_s']-r['min_s'] for r in points],[r['max_s']-r['median_s'] for r in points]],fmt='-o',color='#087f76',capsize=4)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(x,['10M','100M','1B'])
    ax.set_xlabel('Explicit directed edges'); ax.set_ylabel('Complete forward (s)')
    ax.set_title('(a) Capacity through 1B edges',fontsize=9)
    ax.annotate(f"{points[-1]['median_s']:.0f} s",(1e9,points[-1]['median_s']),xytext=(-8,-20),textcoords='offset points',ha='right')
    ax.grid(alpha=.16)
    ax=fig.add_subplot(grid[0,1]); p=points[-1]
    raw=next(r for r in read(ROOT/'data/billion/results.json') if r['edges']==1_000_000_000)
    values=[raw['stored_bytes']/2**30,p['rss_peak_gib'],p['native_peak_gib'],p['resident_gib']]
    bars=ax.bar(range(4),values,color=['#64748b','#94a3b8','#087f76','#d1d5db'],width=.65)
    bars[-1].set_hatch('//')
    ax.set_xticks(range(4),['Disk\nfiles','Process\nRAM peak','Native\nGPU peak','Full GPU\nlower\nbound'],fontsize=7)
    ax.set_ylabel('GiB (distinct accounting scopes)'); ax.set_ylim(0,28)
    ax.set_title('(b) 1B-edge storage footprint',fontsize=9)
    for i,v in enumerate(values): ax.text(i,v+.5,f'{v:.2f}',ha='center',fontsize=8)
    ax.text(.98,.95,'16 GiB GPU; lower bound is not measured OOM',transform=ax.transAxes,ha='right',va='top',fontsize=6.6)
    ax=fig.add_subplot(grid[1,:])
    labels=['Topology read / decode','Field gather (incl. page faults)','Host buffer copies','Host pack + H2D / D2H','Realize / JIT + other host']
    colors=['#64748b','#94a3b8','#e3b66e','#3d9b92','#cbd5e1']
    from build_paging_profile import summary
    profile=summary(); s=profile['exclusive_host_seconds']
    values=[s['topology_read_decode_validate'],s['field_mmap_gather'],s['host_buffer_copy'],s['host_pack_and_H2D']+s['D2H_and_host_copy'],s['realize_compile_submit_wait']+s['other_host_setup_assembly']]
    left=0
    for label,color,v in zip(labels,colors,values):
        ax.barh(0,v,left=left,color=color,label=label,height=.45)
        ax.text(left+v/2,0,f'{v:.1f}',ha='center',va='center',fontsize=8)
        left+=v
    ax.set_yticks([0],['1B edges\n954 pages']); ax.set_ylim(-.5,.5)
    ax.set_xlabel('Exclusive host wall time (s); one fully checked 1B-edge profile')
    ax.set_title(f"(c) 1B-edge execution cost: {profile['host_seconds']:.2f} s",fontsize=9)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.32),ncol=3,fontsize=6.8,frameon=False)
    ax.set_xlim(0,430)
    fig.text(.5,-.08,f"1B Nsight activity: kernels {profile['kernel_seconds']:.2f} s; H2D {profile['h2d_seconds']:.2f} s; D2H {profile['d2h_seconds']:.2f} s.\nDevice activity is included in the host stages, not added to them.",ha='center',fontsize=7)
    save(fig,'q2-capacity-cost'); plt.close(fig)


def render_distributed(plt):
    from matplotlib.ticker import NullFormatter
    fig,axes=plt.subplots(2,2,figsize=(7.6,5.5))
    summary={}
    for col,name in enumerate(('local','remote')):
        ranks=[read(DATA/f'{name}-rank{rank}.json') for rank in (0,1)]
        nodes=(32768,131072,524288); x=[n*16 for n in nodes]
        for rank,color,label in ((0,'#64748b','5070 Ti alone'),(1,'#c27b24','4070 Ti SUPER alone')):
            medians=[st.median(next(q for q in ranks[rank]['rows'] if q['nodes']==n and q['mode']=='single')['samples_ms']) for n in nodes]
            axes[0,col].plot(x,medians,'--o',ms=4,color=color,label=label)
        for mode,color,label in (('serialized','#94a3b8','Communication then compute'),('automatic','#087f76','Overlap schedule')):
            samples=[paired_samples(ranks,n,mode) for n in nodes]
            axes[0,col].plot(x,[st.median(s) for s in samples],'-o',ms=4,color=color,label=label)
            axes[0,col].fill_between(x,[min(s) for s in samples],[max(s) for s in samples],color=color,alpha=.12)
        waits=[]; interiors=[]; traffic=[]; stages=[]
        for n in nodes:
            qs=[next(q for q in r['rows'] if q['nodes']==n and q['mode']=='automatic') for r in ranks]
            wait=[]; interior=[]
            for step in range(5):
                critical=max(qs,key=lambda q:q['samples_ms'][step]); t=critical['traces'][step]
                wait.append((t['communication_wait_finished_ns']-t['communication_wait_started_ns'])/1e6)
                interior.append((t['interior_finished_ns']-t['interior_started_ns'])/1e6)
            waits.append(st.median(wait)); interiors.append(st.median(interior))
            traffic.append(sum(q['traffic'][0]['send_bytes'] for q in qs))
            paired=[max(q['samples_ms'][i] for q in qs) for i in range(5)]
            middle=sorted(range(5),key=lambda i:paired[i])[2]
            critical=max(qs,key=lambda q:q['samples_ms'][middle])
            stages.append(host_phases(critical,middle))
        phase_labels=['Before exchange','Halo assembly / readiness','Interior realization','Boundary / output + remaining']
        phase_colors=['#94a3b8','#087f76','#c27b24','#cbd5e1']
        left=[0.,0.,0.]
        for i,(label,color) in enumerate(zip(phase_labels,phase_colors)):
            heights=[s[i] for s in stages]
            axes[1,col].bar(range(3),heights,bottom=left,label=label,color=color,width=.6)
            left=[a+b for a,b in zip(left,heights)]
        axes[0,col].set_title('Local ring · small halo' if name=='local' else '25% remote rows · large halo',fontsize=9)
        axes[0,col].legend(fontsize=6.8)
        axes[1,col].set_xlabel('Global directed edges')
        axes[0,col].set_xscale('log'); axes[0,col].set_xticks(x,['0.52M','2.10M','8.39M'])
        axes[0,col].xaxis.set_minor_formatter(NullFormatter())
        axes[1,col].set_xticks(range(3),['0.52M','2.10M','8.39M'])
        for ax in axes[:,col]: ax.grid(alpha=.16,axis='y')
        axes[0,col].set_yscale('log')
        ticks=[5,10,20,40] if col==0 else [5,20,100,400]
        axes[0,col].set_yticks(ticks,[str(t) for t in ticks])
        axes[0,col].yaxis.set_minor_formatter(NullFormatter())
        summary[name]=dict(nodes=list(nodes),sent_bytes_both_ranks=traffic,
                           exposed_wait_ms=waits,interior_host_ms=interiors,
                           median_sample_host_phases_ms=stages,
                           automatic_ms=[st.median(paired_samples(ranks,n,'automatic')) for n in nodes],
                           serialized_ms=[st.median(paired_samples(ranks,n,'serialized')) for n in nodes],
                           single_ms=[[st.median(next(q for q in r['rows'] if q['nodes']==n and q['mode']=='single')['samples_ms']) for n in nodes] for r in ranks])
    axes[0,0].set_ylabel('Complete forward (ms, log) ↓')
    axes[1,0].set_ylabel('Median critical sample: host stages (ms)')
    fig.tight_layout(h_pad=1.4)
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.10),ncol=2,fontsize=7,frameon=False)
    save(fig,'q3-distributed'); plt.close(fig)
    (ROOT/'generated/three-questions-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    a,b=summary['local'],summary['remote']
    paragraph=(f"At 8.39M edges, the local-ring single-GPU medians are {a['single_ms'][0][-1]:.2f} ms "
               f"(5070 Ti) and {a['single_ms'][1][-1]:.2f} ms (4070 Ti SUPER). Two-rank serialized "
               f"execution takes {a['serialized_ms'][-1]:.2f} ms and automatic scheduling "
               f"{a['automatic_ms'][-1]:.2f} ms. Neither beats the faster single GPU. "
               f"With the large halo, the 5070 Ti alone takes {b['single_ms'][0][-1]:.2f} ms, "
               f"versus {b['serialized_ms'][-1]:.2f} ms serialized and {b['automatic_ms'][-1]:.2f} ms automatic "
               f"across two hosts. The automatic slowdown is {b['automatic_ms'][-1]/b['single_ms'][0][-1]:.1f} times.\n\n"
               "The large-halo host stages include packing, allocation, ghost assembly, "
               "readiness and transport progress; they are not pure network wire time. "
               "A tiny final stream-wait interval does not mean communication was free: "
               "earlier runtime operations may already wait for progress. Equal-row partitioning "
               "also leaves the faster GPU waiting for the slower rank. These results support "
               "transport-aware scheduling and heterogeneous load balancing as engineering "
               "priorities, not a claim that automatic overlap or two GPUs always help. "
               "No capacity expansion beyond single-GPU storage is measured by this setup.\n")
    (ROOT/'generated/q3-findings.tex').write_text(paragraph)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-run',type=Path)
    parser.add_argument('--import-comparison',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.import_comparison:
        run=args.import_comparison
        rows=[read(p) for p in sorted(run.glob('*-*.json')) if p.name!='runs.json' and not p.name.startswith('audit-')]
        validate_comparison(rows)
        Q1_DATA.mkdir(parents=True,exist_ok=True)
        (Q1_DATA/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
        for name in ('source.tar.gz','REPRODUCE.md','runs.json','native_allocation_audit.py','audit-csr.json','audit-radius.json','audit-knn.json'):
            shutil.copy2(run/name,Q1_DATA/name)
        index={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Q1_DATA.iterdir()) if p.is_file() and p.name!='index.json'}
        (Q1_DATA/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    if args.import_run:
        run=args.import_run; DATA.mkdir(parents=True,exist_ok=True)
        rows=[read(p) for p in sorted((run/'comparison-v2').glob('*-*.json'))]
        validate_comparison(rows)
        (DATA/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
        for name in ('local','remote'):
            for rank in (0,1): shutil.copy2(run/f'distributed-final/{name}-rank{rank}.json',DATA)
        for p in run.glob('paging-*'):
            if p.suffix in ('.json','.csv','.nsys-rep'): shutil.copy2(p,DATA)
        for p in run.glob('audit-*.json'): shutil.copy2(p,DATA)
        shutil.copy2(run/'distributed-final/commands.json',DATA/'distributed-commands.json')
        shutil.copy2(run/'comparison-v2/runs.json',DATA/'comparison-commands.json')
        for name in ('source.tar.gz','REPRODUCE.md'):
            if (run/name).exists(): shutil.copy2(run/name,DATA/name)
        index={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.rglob('*')) if p.is_file() and p.name!='index.json'}
        (DATA/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    rows=validate()
    if Q1_DATA.exists():
        rows=validate_revised_comparison()
    spatial=(ROOT/'data/distributed-spatial/index.json').exists()
    if spatial:
        from build_spatial_results import validate as validate_spatial
        validate_spatial()
    if args.check:
        print(f'Experiment data: {len(rows)} current matched comparisons, archived comparisons/NCCL, spatial meshes, 1B profile and historical 10M diagnostics verified')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    render_comparison(rows,plt); render_capacity(plt)
    if spatial:
        from build_spatial_results import render as render_spatial
        render_spatial(plt)
    else:
        render_distributed(plt)


if __name__=='__main__':
    main()
