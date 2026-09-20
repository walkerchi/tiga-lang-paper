"""Package current and historical evidence for the bilingual documentation."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
args=parser.parse_args()
args.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for directory in ('data/three-questions','data/comparison-v3',
                      'data/distributed-spatial','data/billion','data/paging-1b'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file(): archive.write(path,path.relative_to(ROOT))
    for name in ('tools/build_three_questions.py','tools/build_spatial_results.py',
                 'tools/build_billion_results.py','tools/build_paging_profile.py','generated/spatial-summary.json',
                 'generated/three-questions-summary.json'):
        archive.write(ROOT/name,name)
print(hashlib.sha256(args.output.read_bytes()).hexdigest(),args.output)
