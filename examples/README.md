# GRiD examples

Three tracks, three audiences. Pick by how you want to *use* GRiD.

| track | dir | "I want to…" |
|-------|-----|--------------|
| **Python bindings** | [`notebooks/`](notebooks/) | call GRiD dynamics from Python (numpy / torch / JAX), batched on the GPU — **start here** |
| **Codegen** | [`codegen/`](codegen/) | generate a `grid.cuh` from a URDF and drop it into my own C++/CUDA project |
| **Hand-written CUDA** | [`cuda/`](cuda/) | write my own CUDA kernel that `#include`s the generated header |

## `notebooks/` — the Python-bindings tour (start here)

Tutorial-style Jupyter notebooks for the `grid_rbd` **register-then-run** UX:
register a robot once (it parses the URDF, generates `grid.cuh`, compiles a
per-robot `.so` into a cache), then call dynamics / kinematics / gradients
many times, batched over axis 0. Covers the numpy, **torch**, and **JAX**
backends, plus an in-notebook nvcc CUDA walkthrough. Every notebook ends in
`assert` cells so a green *Run All* validates the *numbers*.

See [`notebooks/README.md`](notebooks/README.md) for the index and setup.

## `codegen/` — generate `grid.cuh` for your own project

The codegen workflow: drive `GRiDCodeGenerator` directly to emit CUDA from a
URDF, and print the generated kernels / CPU reference values. This is what you
reach for when you want the *generated CUDA header itself* — to compile into
your own MPC/RL/controls binary — rather than calling GRiD through Python.

| script | what it does |
|--------|--------------|
| `generate_iiwa14.py` | generate a fixed-base iiwa14 `grid.cuh` (zero-config) |
| `generate_go2_floating.py` | generate a floating-base Go2 `grid.cuh` (with `--profile`) |
| `print_grid.py` | compile + run the built-in `printGRiD` kernel to dump generated outputs |
| `print_reference_values.py` | print the `RBDReference` CPU oracle values for a URDF (validate CUDA output) |

Run from the repo root so `URDFParser` / `GRiDCodeGenerator` import, e.g.
`python examples/codegen/generate_iiwa14.py --output /tmp/grid.cuh`. The
installed `grid-generate` CLI is the general (any-URDF) entry point.

## `cuda/` — write your own kernel against the header

Hand-written, compiled-and-validated CUDA that `#include`s a generated
`grid.cuh` and calls the `grid::` kernels (`_inner` / `_device` / `_host`
surfaces, dynamic-shared-memory registration, batched one-block-per-timestep).
This is the C++/CUDA counterpart to track 2: track 2 *produces* the header,
track 3 *consumes* it.

See [`cuda/README.md`](cuda/README.md). The notebook
[`notebooks/07_inline_cuda.ipynb`](notebooks/07_inline_cuda.ipynb) is the
tutorial version (compile + run a kernel inline with `!nvcc`).

> **Codegen vs bindings — not redundant.** `codegen/` gives you *CUDA source*
> for your own project; `notebooks/` calls the *compiled bindings* from Python.
> Use codegen when you'll write/own CUDA; use the bindings when you want
> dynamics callable from Python.
