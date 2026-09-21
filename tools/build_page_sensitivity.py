"""Validate and summarize recorded page-size measurements, without rerunning."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/page-sensitivity'


def validate():
    records=[json.loads(p.read_text()) for p in sorted(DATA.glob('page-*-r*.json'))]
    assert len(records)==9
    reference=records[0]
    for r in records:
        assert r['status']=='ok' and r['max_abs_error']==0
        assert r['checked_values']==20000000 and r['edges']==10000000
        assert r['features']==32 and r['degree']==16 and r['mode']=='paged'
        assert r['execution']['page_backends']==['cuda-ttir-triton']
        assert r['forward_s']>0
        for field in ('source_hashes','compiler_hashes','benchmark_sha256'):
            assert r['provenance'][field]==reference['provenance'][field]
        assert r['ingest']['payload_sha256']==reference['ingest']['payload_sha256']
    assert {r['page_rows'] for r in records}=={16384,65536,262144}
    assert all(sum(r['page_rows']==p for r in records)==3 for p in (16384,65536,262144))
    with tarfile.open(ROOT/'data/supplement/source.tar.gz') as archive:
        content=archive.extractfile('benchmarks/memory_hierarchy/billion_edges.py').read()
        assert hashlib.sha256(content).hexdigest()==reference['provenance']['benchmark_sha256']
    return records


def render(records):
    lines=[r'\begin{tabular}{rrrr}',r'\toprule',
           r'Rows per page & Pages & Forward (s) & Native peak (MiB) \\',r'\midrule']
    summaries=[]
    for page in (16384,65536,262144):
        rows=[r for r in records if r['page_rows']==page]
        assert len(rows)==3
        times=[r['forward_s'] for r in rows]
        peaks=[r['forward_memory']['peak_bytes']['device']/2**20 for r in rows]
        summary=dict(page_rows=page,forward_s=statistics.median(times),
                     range_s=[min(times),max(times)],peak_mib=max(peaks),
                     pages=(625000+page-1)//page)
        summaries.append(summary)
        lines.append(f"{page:,} & {summary['pages']} & {summary['forward_s']:.2f} ({min(times):.2f}--{max(times):.2f}) & {max(peaks):.1f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'generated/page-sensitivity.tex').write_text('\n'.join(lines)+'\n')
    (ROOT/'generated/page-sensitivity.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    records=validate()
    if args.check:
        print('Validated nine fully checked 10M-edge page-size trials.')
    else:
        render(records)
