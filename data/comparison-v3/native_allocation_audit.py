import json
from pathlib import Path
import runpy
import sys
import tiga as tg

output=Path(sys.argv[-1])
sys.argv[0]='paper_comparison.py'
with tg.execution() as scope:
    runpy.run_module('benchmarks.graph_operations.paper_comparison',run_name='__main__')
    audit=scope.memory_report()
record=json.loads(output.read_text())
record['native_allocation_audit']=audit
output.write_text(json.dumps(record,indent=2)+'\n')
print('native allocation audit',audit)
