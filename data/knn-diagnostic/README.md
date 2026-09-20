# Warm public-call kNN diagnostic — 2026-09-19

These two instrumented runs diagnose the Q1 implementation; they are **not**
additional formal timing samples and do not change the three experiment figures.
Both runs use the unchanged benchmark and implementation source recorded in
`../three-questions/`. Each checks both complete dynamic neighbor sets, performs
the benchmark's first call and four warmups, then profiles three checked calls.

Run from the implementation source with its existing native build:

```bash
export PYTHONPATH=python:.:/tmp/tiga-paper-v2-deps.AW5KyW
export TIGA_OPT=/tmp/tiga-docs-review/compiler/bin/gf-opt
export TIGA_TRANSLATE=/tmp/tiga-docs-review/compiler/bin/gf-translate
export OMP_NUM_THREADS=1
/home/walkerchi/Code/ComfyUI/.venv/bin/python /tmp/tiga-knn-profile.py \
  --workload knn --provider tiga --nodes 1024 --features 1 --degree 16 \
  --repeats 3 --output /tmp/tiga-knn-diagnostic-1024.json
/home/walkerchi/Code/ComfyUI/.venv/bin/python /tmp/tiga-knn-profile.py \
  --workload knn --provider tiga --nodes 16384 --features 1 --degree 16 \
  --repeats 3 --output /tmp/tiga-knn-diagnostic-16384.json
```

Adjust local executable/dependency paths when reproducing. The wrapper source
is included beside this note. Profiling begins at the first measured call's
memory-counter reset and also includes subsequent validation and report writing;
the total profile duration is therefore **not** forward latency. Inspect the
three `_execute_ranked` and `prepare_ttir_ranked` invocations for the warmed
execution path. Nested cumulative times must not be summed.

In each run, `prepare_ttir_ranked` takes approximately 0.047 seconds over three
calls. The public-call profiled medians are 17.225 ms (1,024 points) and 26.479 ms
(16,384 points). Profiling overhead and shared-host variability prevent using
these as replacements for the original 16.711 ms and 25.801 ms medians.

Source inspection explains the repeated preparation: `_RankedExecutable.try_run`
in `python/tiga/interop/torch/message_passing.py` rejects a different Graph object;
the benchmark intentionally constructs a new descriptor for changed coordinates.
The miss enters capture, lowering and provider preparation. The lower-level
compiler can reuse cached artifacts: repeated preparation does not establish
repeated machine-code generation. The compiler's candidate-tile sorting and
merging are separate GPU work, not isolated by this Python-only profile.

No runtime changes, service restarts, full benchmark reruns or releases were made.
