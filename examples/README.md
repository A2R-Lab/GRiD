# GRiD examples

Four tracks, grouped under the two entry points the
[site](https://a2r-lab.org/GRiD/) presents: the Python wrappers (the easy
path: `pip install grid-rbd`, call the compiled bindings from Python) and
header-only CUDA (the power-user path: generate `grid.cuh`, `#include` it in
your own kernels). Pick a track by how you intend to use GRiD.

| entry point | track | dir | "I want to…" |
|-------------|-------|-----|--------------|
| Easy: Python wrappers | Python bindings | [`notebooks/`](notebooks/) | call GRiD dynamics from Python (numpy / torch / JAX), batched on the GPU. Start here. |
| Easy: Python wrappers | Runnable scripts | [`../bindings/examples/`](../bindings/examples/) | task-shaped, runnable binding scripts (GPU residency, tools/payloads, multi-contact, sysID, named EE targets, runtime params) |
| Power user: header-only CUDA | Codegen | [`codegen/`](codegen/) | generate a `grid.cuh` from a URDF and drop it into my own C++/CUDA project |
| Power user: header-only CUDA | Hand-written CUDA | [`cuda/`](cuda/) | write my own CUDA kernel that `#include`s the generated header |

## `notebooks/`: the Python-bindings tour

Tutorial-style Jupyter notebooks for the `grid_rbd` register-then-run model:
register a robot once (it parses the URDF, generates `grid.cuh`, compiles a
per-robot `.so` into a cache), then call dynamics / kinematics / gradients
many times, batched over axis 0. The notebooks cover the numpy, torch, and JAX
backends, plus an in-notebook nvcc CUDA walkthrough. Every notebook validates
its numbers with `assert` cells (notebook 07 ends with a scratch-dir cleanup
after its asserts), so a green *Run All* means the values match, not only that
no exception was raised. This is the recommended starting point.

See [`notebooks/README.md`](notebooks/README.md) for the index and setup.

## `../bindings/examples/`: runnable binding scripts

Task-shaped scripts on the same `grid_rbd` surface the notebooks teach. Each
one is a runnable `.py` with a workflow narrative: GPU-resident JAX/torch
pipelines with CUDA graphs, welded tools/payloads, multi-contact stance
forces, least-squares system identification, named end-effector targets,
runtime parameter mutation. They are indexed in
[`../bindings/examples/AGENT_INTEGRATION_GUIDE.md`](../bindings/examples/AGENT_INTEGRATION_GUIDE.md)
under "Runnable examples".

## `codegen/`: generate `grid.cuh` for your own project

The codegen workflow: drive `grid_codegen` (its `GRiDCodeGenerator` class) directly to emit CUDA from a
URDF, and print the generated kernels / CPU reference values. Use this track
when you need the generated CUDA header itself, to compile into your own
MPC/RL/controls binary, rather than calling GRiD through Python.

| script | what it does |
|--------|--------------|
| `generate_iiwa14.py` | generate a fixed-base iiwa14 `grid.cuh` from the vendored `grid_assets/robots/iiwa14.urdf` (no arguments needed) |
| `generate_go2_floating.py` | generate a floating-base Go2 `grid.cuh` (with `--profile`) |
| `generate_collision.py` | generate iiwa14 `grid.cuh` with the two-tier `config_free` collision routine (spherized broad/fine geometry) |
| `generate_multi_target.py` | generate iiwa14 `grid.cuh` with batched `multi_target_position{,_gradient}` kernels (one FK/Jacobian launch over many baked EE targets) |
| `generate_runtime_params.py` | generate iiwa14 `grid.cuh` with runtime-mutable inertia / fixed-transform tables (`set_inertia_params` / `set_transform_params`, no recompile) |
| `generate_regressor_gradient.py` | generate the regressor state-derivative `inverse_dynamics_regressor_gradient` (dY/dx; `dY_dx[c]·π == ∂τ/∂x[:,c]`) surface |
| `print_grid.py` | compile + run `printGRiD.cu` to dump every generated kernel's output for a `grid.cuh` |
| `printGRiD.cu` | the CUDA source `print_grid.py` compiles (it `#include`s `grid.cuh` and prints each kernel's result); not meant to be built by hand |
| `print_reference_values.py` | print the `RBDReference` CPU oracle values for a URDF (validate CUDA output) |

Run from the repo root so `URDFParser` / `grid_codegen` import, e.g.
`python examples/codegen/generate_iiwa14.py --output /tmp/grid.cuh`. The
installed `grid-generate` CLI is the general (any-URDF) entry point.

## `cuda/`: write your own kernel against the header

Hand-written, compiled-and-validated CUDA that `#include`s a generated
`grid.cuh` and calls the `grid::` kernels (`_inner` / `_device` / kernel
surfaces plus the generated host launchers, dynamic-shared-memory
registration, batched one-block-per-timestep). This is the CUDA counterpart
to the `codegen/` track: codegen produces the header and this track consumes
it. Together they are the "Power user: header-only CUDA" entry point.

See [`cuda/README.md`](cuda/README.md). The notebook
[`notebooks/07_inline_cuda.ipynb`](notebooks/07_inline_cuda.ipynb) is the
tutorial version (compile + run a kernel inline with `!nvcc`).

The codegen and bindings tracks are not redundant. `codegen/` gives you CUDA
source for your own project; `notebooks/` calls the compiled bindings from
Python. Use codegen when you will write and own CUDA (power user); use the
bindings when you want dynamics callable from Python (easy).
