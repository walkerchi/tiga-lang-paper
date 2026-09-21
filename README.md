# Tiga technical report

![Tiga](figures/tiga-logo.svg)

A technical report on graph message-passing compilation: the programming model,
multi-level IR, automatic differentiation, memory hierarchy and distributed execution.

[Read the PDF](tiga-lang.pdf) ·
[Download PDF](https://github.com/walkerchi/tiga-lang-paper/raw/refs/heads/main/tiga-lang.pdf) ·
[Compiler project](https://github.com/walkerchi/TIGA-lang) ·
[Report source](https://github.com/walkerchi/tiga-lang-paper)

Author: **walkerchi**, Independent Developer.
Contact: [walker.chi.000@gmail.com](mailto:walker.chi.000@gmail.com).

## Read and build

The manuscript is `main.tex`; [tiga-lang.pdf](tiga-lang.pdf) is the compiled
report checked into the repository root alongside this README. Local builds
write `build/main.pdf`; `make pdf` refreshes the checked-in copy.
Background includes related work, and Appendix A contains a runnable EdgeNN
example with Torch forward and gradient checks.

```bash
make pdf TECTONIC=/path/to/tectonic
make check PYTHON=/path/to/python PROJECT=/path/to/tiga-lang
```

Alternatively, `make` uses latexmk. Tectonic may download its TeX bundle on the
first build. Stored figures compile without shell escape or external figure execution.
The full evidence check also reads the compiler project's archived benchmark
inputs; set `PROJECT` to a checkout of the linked compiler repository.
Building the PDF alone does not require that checkout or a GPU.

## Evaluation

- **Performance and memory:** stored CSR and changing radius/kNN relations,
  compared with matched Torch and PyG implementations.
- **Single-GPU capacity:** 10M–1B explicit edges, disk/RAM/GPU accounting,
  and an instrumented billion-edge forward execution.
- **Distributed execution:** spatial mesh partitions, interface-sized halo
  exchange and completed-call latency on heterogeneous GPUs.

The measurements cover FP32 forward workloads. Training throughput,
larger-than-host-RAM capacity and compute-aware repartitioning are outside
this evaluation. Full conditions and source snapshots accompany each dataset.

## Reproduce figures

```bash
python tools/build_three_questions.py --check
python tools/build_three_questions.py
```

These commands validate and plot stored observations without running GPU experiments.
The generated figures are `q1-performance-memory`, `q2-capacity-cost` and
`q3-distributed`, in PDF and PNG formats.

| Measurement | Data and reproduction |
|---|---|
| Torch/PyG comparison | [comparison-v3 protocol](data/comparison-v3/REPRODUCE.md), 36 configurations |
| Single-GPU capacity | `data/billion/`; validate with `python tools/build_billion_results.py --check` |
| Billion-edge profiling | [paging-1b protocol](data/paging-1b/REPRODUCE.md), Nsight trace and host intervals |
| Distributed spatial mesh | [distributed-spatial protocol](data/distributed-spatial/REPRODUCE.md), both rank records and interface counts |

The [reproduction overview](data/three-questions/REPRODUCE.md) describes accounting
definitions and environments. Run each dataset's commands to collect new hardware
measurements; replotting alone uses the stored samples.

## Source layout

- `sections/`: manuscript sections; `sections/archive/` is excluded from the build.
- `figures/`: diagrams and generated plots.
- `generated/ir/`: compiler-checked fixtures and outputs.
- `examples/edge_nn.py`: executable appendix.
- `references.bib`: bibliography.
- `tests/`: numerical, provenance and manuscript checks.

The report has not been submitted to arXiv.
