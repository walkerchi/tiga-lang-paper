"""Validate two-rank spatial-mesh data and render the main distributed figure."""
import hashlib
import json
from pathlib import Path
import statistics as st
import tarfile
import argparse
import shutil

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/distributed-spatial'


def read(path):
    return json.loads(path.read_text())


def validate():
    for name,digest in read(DATA/'index.json').items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    ranks=[read(DATA/f'mesh-rank{rank}.json') for rank in (0,1)]
    assert [r['rank'] for r in ranks]==[0,1]
    for r in ranks:
        assert r['topology']=='mesh' and r['transport']=='nccl-socket'
        assert r['mesh_sides']==[32,64,96] and r['features']==16
        assert r['warmup']==2 and r['repeats']==5
        for key in ('source_hashes','compiler_hashes','benchmark_sha256','nccl_library_sha256','nccl_version'):
            assert r[key]==ranks[0][key],key
        modes={q['mode'] for q in r['rows']}
        assert modes in ({'single','serialized'}, {'single','serialized','automatic'})
        assert modes=={q['mode'] for q in ranks[0]['rows']}
        assert len(r['rows'])==3*len(modes)
        assert {(q['mesh_side'],q['mode']) for q in r['rows']}=={(s,m) for s in (32,64,96) for m in modes}
        for q in r['rows']:
            side=q['mesh_side']; n=side**3
            assert q['nodes']==n and q['edges']==(3*side-2)**3-n
            assert q['boundary_rows']==q['ghost_nodes']==side**2
            assert q['boundary_fraction']==2/side
            assert q['correct'] and len(q['samples_ms'])==5
            assert q['median_ms']==st.median(q['samples_ms'])
            assert all(v>0 for v in q['samples_ms'])
            for t in q['traffic']:
                expected=0 if q['mode']=='single' else side**2*16*4
                assert t['send_bytes']==t['receive_bytes']==expected
            if q['mode']=='automatic':
                from build_three_questions import host_phases
                for i,t in enumerate(q['traces']):
                    assert t['boundary_rows']==side**2
                    assert t['interior_rows']==n//2-side**2
                    host_phases(q,i)
    with tarfile.open(DATA/'source.tar.gz') as archive:
        sources=dict(ranks[0]['source_hashes'])
        sources['benchmarks/distributed/paper_scaling.py']=ranks[0]['benchmark_sha256']
        for name,digest in sources.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest,name
    return ranks


def summarize(ranks):
    from build_three_questions import paired_samples,host_phases
    rows=[]
    for side in (32,64,96):
        n=side**3
        qs=[next(q for q in r['rows'] if q['nodes']==n and q['mode']=='serialized') for r in ranks]
        automatic_ms=phases=None
        if all(any(q['nodes']==n and q['mode']=='automatic' for q in r['rows']) for r in ranks):
            automatic=[next(q for q in r['rows'] if q['nodes']==n and q['mode']=='automatic') for r in ranks]
            paired=paired_samples(ranks,n,'automatic')
            middle=sorted(range(5),key=lambda i:paired[i])[2]
            critical=max(automatic,key=lambda q:q['samples_ms'][middle])
            automatic_ms=st.median(paired); phases=host_phases(critical,middle)
        rows.append(dict(side=side,nodes=n,edges=qs[0]['edges'],
            boundary_fraction=2/side,sent_bytes=2*side**2*16*4,
            single_ms=[next(q for q in r['rows'] if q['nodes']==n and q['mode']=='single')['median_ms'] for r in ranks],
            serialized_ms=st.median(paired_samples(ranks,n,'serialized')),
            automatic_ms=automatic_ms,host_phases_ms=phases))
    return rows


def render(plt):
    from build_three_questions import paired_samples,save
    ranks=validate(); rows=summarize(ranks)
    fig=plt.figure(figsize=(7.6,2.9))
    grid=fig.add_gridspec(1,2,wspace=.38)
    ax=fig.add_subplot(grid[0,0]); x=[r['edges']/1e6 for r in rows]
    labels=[f'{v:.2f}M' for v in x]
    for rank,color,label in ((0,'#364152','5070 Ti alone'),(1,'#c27b24','4070 Ti SUPER alone')):
        ax.plot(x,[r['single_ms'][rank] for r in rows],'--o',color=color,ms=4,label=label)
    for mode,color,label in (('serialized','#087f76','Distributed: communicate then compute'),):
        samples=[paired_samples(ranks,r['nodes'],mode) for r in rows]
        ax.plot(x,[st.median(s) for s in samples],'-o',color=color,ms=4,label=label)
        ax.fill_between(x,[min(s) for s in samples],[max(s) for s in samples],color=color,alpha=.12)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(x,labels); ax.set_xlabel('Directed mesh edges')
    ax.set_ylabel('Complete forward (ms)')
    ax.set_title('(a) Spatial-mesh distributed execution',fontsize=9)
    ax.legend(fontsize=6.5); ax.grid(alpha=.16)
    ax=fig.add_subplot(grid[0,1])
    amounts=[r['sent_bytes']/2**20 for r in rows]
    ax.bar(range(3),amounts,color='#087f76',width=.55)
    for i,(amount,r) in enumerate(zip(amounts,rows)):
        ax.text(i,amount+.03,f'{amount:.3f} MiB\n{100*r["boundary_fraction"]:.2f}% boundary',ha='center',fontsize=7)
    ax.set_xticks(range(3),[f'{r["side"]}³' for r in rows])
    ax.set_ylim(0,1.55); ax.set_xlim(-.8,2.8); ax.set_xlabel('Mesh nodes (L³)')
    ax.set_ylabel('Halo bytes sent, both ranks (MiB)')
    ax.set_title('(b) Halo grows with interface area',fontsize=9)
    ax.grid(alpha=.16,axis='y')
    save(fig,'q3-distributed'); plt.close(fig)
    (ROOT/'generated/spatial-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    r=rows[-1]
    text=(f"At {r['edges']/1e6:.2f}M directed edges, the 5070 Ti and 4070 Ti SUPER single-GPU medians "
          f"are {r['single_ms'][0]:.2f} and {r['single_ms'][1]:.2f} ms. The communication-then-compute "
          f"schedule takes {r['serialized_ms']:.2f} ms "
          f"on the paired-rank critical path, {r['serialized_ms']/r['single_ms'][0]:.2f} times the faster single-GPU latency. "
          f"Only {100*r['boundary_fraction']:.2f}\\% of each rank's destination rows touch the interface, "
          f"and total halo traffic is {r['sent_bytes']/2**20:.3f} MiB. "
          "This fixed partition assigns half the destination rows to each GPU. "
          "Compute-aware partitioning is a potential optimization for the measured heterogeneous pair.\n")
    (ROOT/'generated/q3-findings.tex').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-run',type=Path)
    args=parser.parse_args()
    if args.import_run:
        DATA.mkdir(parents=True,exist_ok=True)
        for name in ('mesh-rank0.json','mesh-rank1.json','commands.json','source.tar.gz',
                     'REPRODUCE.md','launch.py','rank0.log','rank1.log'):
            shutil.copy2(args.import_run/name,DATA/name)
        index={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.iterdir()) if p.is_file() and p.name!='index.json'}
        (DATA/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    rows=summarize(validate())
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
