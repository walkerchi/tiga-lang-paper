"""Validate and plot real two-host / CUDA paging measurements, including failures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/runtime'


def read():
    ranks=[json.loads((DATA/f'rank{i}.json').read_text()) for i in (0,1)]
    gpu=json.loads((DATA/'gpu-paged.json').read_text())
    validate(ranks, gpu)
    validate_nccl([json.loads((DATA/f'nccl-graph-rank{i}.json').read_text()) for i in (0,1)])
    for i in (0,1):
        probe=json.loads((DATA/f'nccl-passed-rank{i}.json').read_text())
        assert probe['rank']==i and probe['correct'] and probe['status']=='passed'
    return ranks, gpu


def validate_nccl(ranks):
    a,b=ranks
    assert a['hostname']!=b['hostname'] and a['capability']!=b['capability']
    assert a['source_hashes']==b['source_hashes']
    assert a['benchmark_sha256']==b['benchmark_sha256']
    assert a['transport_metadata']['nccl_library_sha256']==b['transport_metadata']['nccl_library_sha256']
    for i,rank in enumerate(ranks):
        assert rank['rank']==i and rank['correct'] and rank['transport']=='nccl'
        assert len(rank['records'])==3
        assert [(r['serialized'],r['scale']) for r in rank['records']]==[(True,1.),(False,1.),(False,2.)]
        expected=([8.,3.,6.,9.],[12.,15.,18.,13.])[i]
        for row in rank['records']:
            assert row['correct'] and row['gradient']==[3.*row['scale']]*4
            assert row['output']==[v*row['scale'] for v in expected]
            assert row['forward_backend']==row['backward_backend']=='cuda-ttir-triton'
            if not row['serialized']:
                assert row['trace']['schedule']=='interior||device-halo->boundary'
                assert row['trace']['measured_overlap_ms'] is None


def validate(ranks, gpu):
    a,b=ranks
    assert a['rank']==0 and b['rank']==1 and a['hostname']!=b['hostname']
    assert a['source_hashes']==b['source_hashes']
    assert a['compiler_hashes']==b['compiler_hashes']
    assert a['benchmark_sha256']==b['benchmark_sha256']
    for rank in ranks:
        assert rank['features']==16 and rank['degree']==16 and rank['repeats']==10
        keys={(r['nodes'],r['mode']) for r in rank['rows']}
        assert len(rank['rows'])==len(keys)==9
        assert keys=={(n,m) for n in (8192,32768,131072) for m in ('single','serialized','automatic')}
        assert all(r['correct'] and len(r['samples_ms'])==10 and min(r['samples_ms'])>0 for r in rank['rows'])
    assert gpu['budget_mib']==24 and gpu['steps']==10 and gpu['trials']==3
    assert len(gpu['rows'])==27
    keys={(r['nodes'],r['mode'],r['page_rows'],r['trial']) for r in gpu['rows']}
    assert keys=={(n,m,p,t) for n in (32768,131072,524288)
                  for m,p in [('resident',n),('paged',1024),('paged',8192)] for t in range(3)}
    for row in gpu['rows']:
        expected_status='budget_exceeded' if row['mode']=='resident' and row['nodes']>32768 else 'ok'
        assert row['status']==expected_status
        assert row['features']==4 and row['edges']==row['nodes']*16
        assert row['memory']['budget_bytes']['device']==24*2**20
        assert row['memory']['peak_bytes']['device']<=24*2**20
        if row['status']=='ok':
            assert len(row['steps'])==10 and row['max_abs_error']<3e-4
            assert all(s['seconds']>0 for s in row['steps'])
            assert [s['step'] for s in row['steps']]==list(range(10))
            assert all(s['max_abs_error']<3e-4 for s in row['steps'])


def plot(ranks,gpu):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('pdf','png'):
            fig.savefig(ROOT/f'figures/{name}.{suffix}',bbox_inches='tight',dpi=180,
                        **({'metadata':{'CreationDate':None}} if suffix=='pdf' else {}))
        plt.close(fig)
    def style(ax,nodes,label):
        ax.set_xscale('log',base=2)
        ax.set_xticks(nodes,[f'{n//1024}k' for n in nodes])
        ax.set_xlabel('Nodes (powers of two)'); ax.set_ylabel(label); ax.grid(alpha=.2)
    def lookup(rank,n,mode):
        return next(r for r in ranks[rank]['rows'] if r['nodes']==n and r['mode']==mode)
    nodes=[8192,32768,131072]
    fig,ax=plt.subplots(figsize=(6.5,3.1))
    for label,color,ls,rank,mode in [('5070 Ti single','#475569','--',0,'single'),
                                   ('4070 Ti SUPER single','#94a3b8',':',1,'single'),
                                   ('Two-host serialized','#818cf8','--',None,'serialized'),
                                   ('Two-host automatic','#4f46e5','-',None,'automatic')]:
        samples=[lookup(rank,n,mode)['samples_ms'] if rank is not None else
                 np.maximum(lookup(0,n,mode)['samples_ms'],lookup(1,n,mode)['samples_ms']) for n in nodes]
        ax.plot(nodes,[np.median(s) for s in samples],marker='o',color=color,ls=ls,label=label)
        ax.fill_between(nodes,[min(s) for s in samples],[max(s) for s in samples],color=color,alpha=.07)
    style(ax,nodes,'Synchronized host-wall latency (ms)'); ax.set_yscale('log'); ax.legend(fontsize=8)
    save(fig,'two-host-latency')
    fig,ax=plt.subplots(figsize=(6.5,2.8))
    values={'End-to-end critical path':[], 'Socket send/receive span':[], 'Interior realization span':[]}
    for n in nodes:
        rows=[lookup(r,n,'automatic') for r in (0,1)]
        values['End-to-end critical path'].append(np.median(np.maximum(*[r['samples_ms'] for r in rows])))
        values['Socket send/receive span'].append(np.median(np.maximum(*[[t['socket_span_ms'] for t in r['traffic']] for r in rows])))
        values['Interior realization span'].append(np.median(np.maximum(*[[(t['interior_finished_ns']-t['interior_started_ns'])/1e6 for t in r['traces']] for r in rows])))
    for (label,ys),color,ls in zip(values.items(),['#4f46e5','#475569','#94a3b8'],['-','--',':']):
        ax.plot(nodes,ys,marker='o',color=color,ls=ls,label=label)
    style(ax,nodes,'Host-wall interval (ms)'); ax.legend(fontsize=8)
    save(fig,'two-host-components')
    nodes=[32768,131072,524288]
    configs=[('resident',None,'Resident','#475569','--'),('paged',1024,'Paged: 1k rows','#4f46e5','-'),('paged',8192,'Paged: 8k rows','#818cf8',':')]
    fig,axs=plt.subplots(1,2,figsize=(7,3))
    for mode,page,label,color,ls in configs:
        groups=[[r for r in gpu['rows'] if r['nodes']==n and r['mode']==mode and (page is None or r['page_rows']==page)] for n in nodes]
        good=[all(r['status']=='ok' for r in g) for g in groups]
        peaks=[max(r['memory']['peak_bytes']['device'] for r in g)/2**20 if ok else np.nan for g,ok in zip(groups,good)]
        medians=[[statistics.median(s['seconds'] for s in r['steps'][1:]) for r in g] if ok else [np.nan] for g,ok in zip(groups,good)]
        axs[0].plot(nodes,peaks,marker='o',ls=ls,color=color,label=label)
        axs[1].plot(nodes,[np.median(x) for x in medians],marker='o',ls=ls,color=color,label=label)
        for n,ok in zip(nodes,good):
            if not ok: axs[0].plot(n,24,'x',ms=8,color=color)
    axs[0].axhline(24,color='#475569',ls=':',label='24 MiB native budget'); axs[0].set_ylim(0,30)
    style(axs[0],nodes,'Peak native GPU buffers (MiB)')
    style(axs[1],nodes,'Forward seconds (GC/oracle excluded)'); axs[1].set_yscale('log')
    axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
    save(fig,'gpu-paged-capacity')
    fig,ax=plt.subplots(figsize=(6.5,2.8))
    for page,color,ls in [(1024,'#4f46e5','-'),(8192,'#64748b','--')]:
        rows=[r for r in gpu['rows'] if r['nodes']==524288 and r['mode']=='paged' and r['page_rows']==page]
        for kind,marker in [('live_bytes','o'),('peak_bytes','x')]:
            ys=[max(r['steps'][i]['memory'][kind]['device'] for r in rows)/2**20 for i in range(10)]
            ax.plot(range(1,11),ys,ls=ls,color=color,marker=marker,
                    label=f'{page//1024}k rows: '+('live after GC' if kind=='live_bytes' else 'cumulative peak'))
    ax.axhline(24,color='#94a3b8',ls=':',label='native budget')
    ax.set_xlabel('Recurrent forward step'); ax.set_ylabel('Native GPU buffers (MiB)'); ax.set_ylim(0,30)
    ax.grid(alpha=.2); ax.legend(fontsize=7,ncol=2)
    save(fig,'gpu-paged-iterations')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path); p.add_argument('--run',type=Path)
    p.add_argument('--check',action='store_true'); a=p.parse_args()
    if a.run:
        if not a.project: p.error('--project required for import')
        DATA.mkdir(parents=True,exist_ok=True)
        mapping={'distributed-final/rank0.json':'rank0.json','distributed-final/rank1.json':'rank1.json',
                 'gpu-paged-final/results.json':'gpu-paged.json',
                 'distributed-final-rank0-run.json':'distributed-run.json','gpu-paged-run.json':'gpu-paged-run.json',
                 'nccl-probe/rank0.json':'nccl-rank0.json','nccl-probe/rank1.json':'nccl-rank1.json',
                 'nccl-wireguard-mtu/rank0.json':'nccl-passed-rank0.json',
                 'nccl-wireguard-mtu/rank1.json':'nccl-passed-rank1.json',
                 'nccl-graph-final/rank0.json':'nccl-graph-rank0.json',
                 'nccl-graph-final/rank1.json':'nccl-graph-rank1.json'}
        for source,target in mapping.items(): shutil.copyfile(a.run/source,DATA/target)
        for source in ('benchmarks/distributed/paper_scaling.py','benchmarks/memory_hierarchy/gpu_paged_scaling.py',
                       'benchmarks/distributed/multi_host_nccl_probe.py',
                       'benchmarks/distributed/multi_host_gpu_gate.py'):
            shutil.copyfile(a.project/source,DATA/Path(source).name)
        index={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in DATA.iterdir() if path.name!='index.json'}
        (DATA/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    for name,digest in json.loads((DATA/'index.json').read_text()).items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    ranks,gpu=read()
    if not a.check: plot(ranks,gpu)
    print('Two-host and recurrent CUDA paging evidence validated.')


if __name__=='__main__':
    main()
