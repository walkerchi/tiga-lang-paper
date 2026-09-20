"""Diagnostic only: profile warmed public calls, never replace formal samples."""
import cProfile
import io
import pstats
import runpy
import sys
from pathlib import Path

import torch

profiler = cProfile.Profile()
original_reset = torch.cuda.reset_peak_memory_stats
started = False


def start_profile(*args, **kwargs):
    global started
    original_reset(*args, **kwargs)
    if not started:
        started = True
        profiler.enable()


torch.cuda.reset_peak_memory_stats = start_profile
try:
    runpy.run_module('benchmarks.graph_operations.paper_comparison', run_name='__main__')
finally:
    profiler.disable()
    target = Path(sys.argv[sys.argv.index('--output') + 1]).with_suffix('.profile.txt')
    report = io.StringIO()
    pstats.Stats(profiler, stream=report).strip_dirs().sort_stats('cumulative').print_stats(55)
    target.write_text(report.getvalue())
    print(report.getvalue())
