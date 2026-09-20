"""Local two-host spatial-graph experiment launcher; no service changes."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/home/walkerchi/Code/graphforgev2')
out=root/'output/paper-three-questions/distributed-spatial'
out.mkdir(parents=True,exist_ok=True)
remote='/home/walker/tiga-three-questions.G4uUUs'
python='/home/walkerchi/Code/ComfyUI/.venv/bin/python'
env=os.environ | dict(PYTHONPATH='python:.',TIGA_TENSOR_BACKEND='native',OMP_NUM_THREADS='1',
    TIGA_OPT='/tmp/tiga-docs-review/compiler/bin/gf-opt',
    TIGA_TRANSLATE='/tmp/tiga-docs-review/compiler/bin/gf-translate',
    TIGA_NCCL_LIBRARY='/home/walkerchi/Code/ComfyUI/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2',
    NCCL_SOCKET_IFNAME='tiga-wg0',NCCL_SOCKET_FAMILY='AF_INET',NCCL_IB_DISABLE='1')
common=['-m','benchmarks.distributed.paper_scaling','--host','10.203.77.1',
    '--port','29745','--transport','nccl','--topology','mesh','--mesh-sides','32,64,96',
    '--features','16','--warmup','2','--repeats','5']
local=[python,*common,'--rank','0','--output',str(out/'mesh-rank0.json')]
remote_args=' '.join(['/home/walker/venvs/tilelang/bin/python',*common,
                     '--rank','1','--output',f'{remote}/output/mesh-rank1.json'])
shell=(f'(ip link show tiga-wg0 >/dev/null 2>&1 || bash /var/lib/tiga-bench-network/tiga-wg-restore.sh remote) && cd {remote} && '
    'runuser -u walker -- env PYTHONPATH=python:/home/walker/tiga-sanity.NOxT0L/deps '
    f'TIGA_TENSOR_BACKEND=native OMP_NUM_THREADS=1 TIGA_OPT={remote}/gf-opt TIGA_TRANSLATE={remote}/gf-translate '
    'TIGA_NCCL_LIBRARY=/home/walker/tiga-paper-20260919.Axi0tf/libnccl.so.2 '
    'NCCL_SOCKET_IFNAME=tiga-wg0 NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 '
    f'timeout 1200 {remote_args}')
ssh=['ssh','-o','BatchMode=yes','4070ti',f'wsl -d Ubuntu -u root -e bash -lc "{shell}"']
(out/'commands.json').write_text(json.dumps(dict(local=local,remote=ssh,
    env={k:env[k] for k in env if k.startswith(('TIGA_','NCCL_','OMP_','PYTHONPATH'))}),indent=2)+'\n')
with (out/'rank0.log').open('w') as lf,(out/'rank1.log').open('w') as rf:
    rp=subprocess.Popen(ssh,stdout=rf,stderr=subprocess.STDOUT)
    lp=subprocess.Popen(local,cwd=root,env=env,stdout=lf,stderr=subprocess.STDOUT)
    try:
        lc=lp.wait(timeout=1200)
        rc=rp.wait(timeout=60)
    except subprocess.TimeoutExpired:
        lp.terminate();rp.terminate()
        raise
print('mesh',lc,rc,flush=True)
if lc or rc:
    print((out/'rank0.log').read_text(),(out/'rank1.log').read_text(),flush=True)
    raise SystemExit(1)
fetch=subprocess.run(['ssh','-o','BatchMode=yes','4070ti',
    f'wsl -d Ubuntu -u walker -e cat {remote}/output/mesh-rank1.json'],capture_output=True,check=True)
(out/'mesh-rank1.json').write_text(json.dumps(json.loads(fetch.stdout),indent=2)+'\n')
