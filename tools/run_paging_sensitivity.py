"""Small page-size sensitivity sweep using the existing checked capacity runner."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def run(project, output):
    output.mkdir(parents=True,exist_ok=True)
    fixture=Path(tempfile.mkdtemp(prefix='tiga-page-sensitivity-'))
    env=os.environ.copy()
    env['TIGA_TENSOR_BACKEND']='native'
    env['PYTHONPATH']=str(project/'python')+os.pathsep+str(project)
    base=[sys.executable,'-m','benchmarks.memory_hierarchy.billion_edges',
          '--edges','10000000','--cache-dir',str(fixture)]
    logs=[]

    def execute(command):
        result=subprocess.run(command,cwd=project,env=env,text=True,capture_output=True)
        logs.append(dict(command=command,returncode=result.returncode,
                         stdout=result.stdout,stderr=result.stderr))
        (output/'runs.json').write_text(json.dumps(logs,indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(result.stderr)

    execute(base+['--build'])
    configurations=[(page,trial) for page in (16384,65536,262144) for trial in range(3)]
    random.Random(20260921).shuffle(configurations)
    for page,trial in configurations:
        path=output/f'page-{page}-r{trial}.json'
        if path.exists():
            raise FileExistsError(path)
        execute(base+['--page-rows',str(page),'--output',str(path.resolve())])
        record=json.loads(path.read_text())
        assert record['status']=='ok' and record['checked_values']==20000000
        print(path.name,record['forward_s'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.project.resolve(),args.output.resolve())
