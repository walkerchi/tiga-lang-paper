"""Import measured artifacts and generate paper curves; never invent missing points."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate(gpu, memory):
    keys = [(r['nodes'], r['features'], r['provider']) for r in gpu['rows']]
    assert len(keys) == len(set(keys)) == 24
    sizes = (8192, 32768, 131072, 524288)
    assert set(keys) == {(n, f, p) for n in sizes for f in (16, 64)
                         for p in ('tiga.auto', 'torch.sparse.mm', 'torch.index_add')}
    assert all(r['correct'] and len(r['samples_ms']) == 20 for r in gpu['rows'])
    for row in gpu['rows']:
        assert statistics.median(row['samples_ms']) == row['median_ms']
        assert all(t > 0 for t in row['samples_ms'])
    assert len(memory['rows']) == 36
    memory_keys = [(r['nodes'], r['mode'], r['page_rows'], r['trial']) for r in memory['rows']]
    assert len(set(memory_keys)) == 36
    assert set(memory_keys) == {(n, mode, page, trial) for n in sizes
                               for mode, page in [('resident', n), ('paged', 1024), ('paged', 8192)]
                               for trial in range(3)}
    for row in memory['rows']:
        assert row['status'] in ('ok', 'budget_exceeded')
        assert row['budget_bytes'] == 8 * 2**20
        assert row['memory_report']['peak_bytes']['ram'] <= row['budget_bytes']
        if row['status'] == 'ok':
            assert row['max_abs_error'] == 0
            assert row['total_s'] > 0
            assert row['total_s'] == row['load_s'] + row['first_call_s']


def figures(gpu, memory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'tiga-paper'})
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name+'.pdf'), bbox_inches='tight', metadata={'CreationDate': None})
        fig.savefig(out / (name+'.png'), bbox_inches='tight', dpi=180)
        plt.close(fig)

    nodes = sorted({r['nodes'] for r in gpu['rows']})
    def axes(ax, ylabel):
        ax.set_xscale('log', base=2)
        ax.set_xticks(nodes, ['8k', '32k', '128k', '512k'])
        ax.set_xlabel('Nodes (powers of two)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)

    fig, axs = plt.subplots(1, 2, figsize=(7, 2.9))
    for ax, width in zip(axs, [16, 64]):
        for provider, name, color, style in [('tiga.auto','Tiga ordinary','#4f46e5','-'),
                ('torch.sparse.mm','Torch sparse','#475569','--'),
                ('torch.index_add','Torch gather/scatter','#94a3b8',':')]:
            rows = sorted([r for r in gpu['rows'] if r['features']==width and r['provider']==provider], key=lambda r:r['nodes'])
            ax.plot(nodes, [r['median_ms'] for r in rows], marker='o', color=color, ls=style, label=name)
        axes(ax, 'Warm forward time (ms, log scale)')
        ax.set_yscale('log')
        ax.set_title(f'{width} features')
    axs[0].legend(fontsize=7)
    save(fig,'gpu-latency')

    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    rng = np.random.default_rng(20260919)
    ratios = []
    for width, color, style in [(16,'#4f46e5','-'),(64,'#64748b','--')]:
        y, low, high = [], [], []
        for n in nodes:
            pair = {r['provider']:r for r in gpu['rows'] if r['nodes']==n and r['features']==width}
            a = np.array(pair['tiga.auto']['samples_ms'])
            b = np.array(pair['torch.sparse.mm']['samples_ms'])
            indices = rng.integers(0,len(a),(5000,len(a)))
            bootstrap = np.median(b[indices],axis=1)/np.median(a[indices],axis=1)
            ratio = float(np.median(b)/np.median(a))
            lo, hi = np.quantile(bootstrap,[.025,.975])
            y.append(ratio); low.append(lo); high.append(hi)
            ratios.append({'nodes':n,'features':width,'ratio':ratio,'ci_low':float(lo),'ci_high':float(hi)})
        ax.plot(nodes,y,marker='o',color=color,ls=style,label=f'{width} features')
        ax.fill_between(nodes,low,high,color=color,alpha=.13)
    ax.axhline(1,color='#94a3b8',lw=1,ls=':')
    axes(ax,'Speedup over Torch sparse (x)')
    ax.legend()
    save(fig,'gpu-speedup')
    (ROOT/'generated/gpu-ratios.json').write_text(json.dumps(ratios,indent=2)+'\n')

    grouped = []
    configs = [('resident',None,'Resident','#475569','--'),
               ('paged',1024,'Paged: 1k rows','#4f46e5','-'),
               ('paged',8192,'Paged: 8k rows','#818cf8',':')]
    for mode,page,name,color,style in configs:
        groups = [[r for r in memory['rows'] if r['nodes']==n and r['mode']==mode
                  and (page is None or r['page_rows']==page)] for n in nodes]
        assert all(len(g)==3 and len({r['status'] for r in g})==1 for g in groups)
        grouped.append((name,color,style,groups))
    fig, axs = plt.subplots(1,2,figsize=(7,2.9))
    for name,color,style,groups in grouped:
        good = [g[0]['status']=='ok' for g in groups]
        managed = [max(r['memory_report']['peak_bytes']['ram'] for r in g)/2**20 if ok else np.nan for g,ok in zip(groups,good)]
        rss = [max(r['peak_rss_bytes'] for r in g)/2**20 for g in groups]
        axs[0].plot(nodes,managed,marker='o',ls=style,color=color,label=name)
        axs[1].plot(nodes,rss,marker='o',ls=style,color=color,label=name)
        for n,ok,value in zip(nodes,good,rss):
            if not ok:
                axs[0].plot(n,8,'x',color=color,ms=8)
                axs[1].plot(n,value,'x',color=color,ms=8)
    axs[0].axhline(8,ls=':',color='#475569',label='8 MiB native budget')
    axes(axs[0],'Peak managed RAM (MiB)')
    axes(axs[1],'Peak process RSS (MiB, log scale)')
    axs[1].set_yscale('log')
    axs[0].set_ylim(0,10)
    axs[0].set_title('Crosses: resident allocation rejected')
    axs[1].set_title('RSS includes Python/compiler memory')
    axs[0].legend(fontsize=6.8,loc='upper left')
    save(fig,'memory-capacity')

    fig, ax = plt.subplots(figsize=(6.3,2.7))
    for name,color,style,groups in grouped:
        y = [statistics.median(r['total_s'] for r in g) if g[0]['status']=='ok' else np.nan for g in groups]
        low = [min(r['total_s'] for r in g) if g[0]['status']=='ok' else np.nan for g in groups]
        high = [max(r['total_s'] for r in g) if g[0]['status']=='ok' else np.nan for g in groups]
        ax.plot(nodes,y,marker='o',ls=style,color=color,label=name)
        ax.fill_between(nodes,low,high,color=color,alpha=.13)
    axes(ax,'Load + first forward + JIT (seconds)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    save(fig,'memory-time')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--gf-opt',type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check',action='store_true')
    action.add_argument('--replot',action='store_true',help='validate and plot archived data without the implementation or a GPU')
    args=parser.parse_args()
    data=ROOT/'data'
    if args.check or args.replot:
        index=json.loads((data/'index.json').read_text())
        for entry in index['files']:
            assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
        gpu=json.loads((data/'gpu-scaling.json').read_text())
        memory=json.loads((data/'memory-results.json').read_text())
        validate(gpu,memory)
        if args.replot:
            figures(gpu,memory)
        print('Fresh GPU/memory samples and archived IR/source checksums verified.')
        return
    if not (args.project and args.run and args.gf_opt):
        parser.error('--project, --run and --gf-opt are required for import')
    data.mkdir(exist_ok=True)
    mapping={'gpu-scaling-final.json':'gpu-scaling.json','memory-single/results.json':'memory-results.json',
             'gpu-run-final.json':'gpu-run.json','memory-fresh-process-run.json':'memory-run.json'}
    files=[]
    for source,name in mapping.items():
        target=data/name
        shutil.copyfile(args.run/source,target)
        files.append(target)
    for name in ('gpu-run.json','memory-run.json'):
        assert json.loads((data/name).read_text())['returncode']==0
    # Keep the exact benchmark sources next to raw data for later reproducibility.
    for name in ['benchmarks/sparse_compute/paper_scaling.py','benchmarks/memory_hierarchy/capacity_scaling.py']:
        target=data/Path(name).name
        shutil.copyfile(args.project/name,target)
        files.append(target)
    subprocess.run([sys.executable,str(args.project/'tools/render_ir_docs.py'),'--gf-opt',str(args.gf_opt),'--check'],check=True)
    ir=ROOT/'generated/ir'
    ir.mkdir(exist_ok=True)
    for source in [args.project/'tests/mlir/distributed-plan.mlir',*(args.project/'docs/includes/ir').glob('*.mlir')]:
        target=ir/source.name
        shutil.copyfile(source,target)
        files.append(target)
    index={'status':'local dirty-checkout experiments; not a tagged release',
           'files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
           'gf_opt_sha256':hashlib.sha256(args.gf_opt.read_bytes()).hexdigest()}
    (data/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    gpu=json.loads((data/'gpu-scaling.json').read_text())
    memory=json.loads((data/'memory-results.json').read_text())
    validate(gpu,memory)
    figures(gpu,memory)
    print('Four fresh-measurement figures generated, with raw samples and provenance.')


if __name__=='__main__':
    main()
