# grid-rbd example notebooks

A guided tour of the `grid_rbd` Python bindings and their register-then-run
model: register a robot once (parses the URDF, generates `grid.cuh`, compiles a
per-robot `.so` into a content-addressed cache), then call algorithms many
times, all batched over axis 0.

Every notebook validates its numbers with `assert` cells that cross-check
GRiD against `RBDReference` or a finite-difference of its own forward map
(notebook 07 ends with a scratch-dir cleanup cell after its asserts), so a
green *Run All* validates the numbers, not only the absence of exceptions.
They can be run headless as smoke tests (`pytest --nbval-lax examples/notebooks/`,
needs a GPU + `grid-rbd` + `nbval`); they are not part of the default CI lanes
or the GPU receipt.

## Notebooks

| # | Notebook | Covers |
|---|----------|--------|
| 01 | [`01_quickstart_iiwa14`](01_quickstart_iiwa14.ipynb) | Register a robot; `inverse_dynamics` / `forward_dynamics` / `minv`, single + batched on one handle; `f_ext`; validate vs `RBDReference`. |
| 02 | [`02_autograd_torch`](02_autograd_torch.ipynb) | The torch backend: autograd through `forward_dynamics`, analytic-vs-finite-difference VJP, a tiny IK descent. |
| 03 | [`03_plant_control`](03_plant_control.ipynb) | The control surface: `plant_step`, quadratic state/input costs, end-effector cost, joint-limit log-barriers. |
| 04 | [`04_kinematics`](04_kinematics.ipynb) | FK (`end_effector_pose` / `fk_batched`), `end_effector_pose_gradient` / `_hessian`, geometric `frame_jacobian` / `_dot` with runtime frame + target selection, `com` / `ccrba` / `osc_inertia`. |
| 05 | [`05_gradients_secondorder`](05_gradients_secondorder.ipynb) | Analytic dynamics derivatives: `inverse_dynamics_gradient`, `forward_dynamics_gradient` (first order) and `idsva_so` / `fdsva_so` (second order), each vs finite-difference. |
| 06 | [`06_jax_backend`](06_jax_backend.ipynb) | The JAX backend (`grid_rbd.jax`): `jax.jit` over dynamics, the native batch axis (vmap-equivalent), GRiD's analytic gradient kernels jitted + FD-validated, `idsva_so` vs the numpy handle. Needs the `[jax]` extra. |
| 07 | [`07_inline_cuda`](07_inline_cuda.ipynb) | Inline CUDA: generate `grid.cuh`, write a kernel that calls `grid::inverse_dynamics_device`, compile with `nvcc` in-notebook, run, and validate vs `RBDReference`. The tutorial version of [`../cuda/`](../cuda/). |

These notebooks (with the runnable scripts in `bindings/examples/`) are the
"Easy: Python wrappers" entry point. The "Power user: header-only CUDA" entry
point lives under [`../codegen/`](../codegen/) and [`../cuda/`](../cuda/);
notebook 07 is the inline tutorial version of the latter. See the
[`examples/` overview](../README.md) for how the four tracks map onto the two
entry points.

## Setup

The notebooks require a CUDA GPU and `nvcc` on PATH. They run against the
released `grid-rbd` from PyPI; install editable from this repository only when
you are developing GRiD and want the bindings that match your checkout:

```bash
pip install grid-rbd                           # numpy backend
pip install "grid-rbd[jax]" "jax[cuda12]"      # + JAX FFI bridge — the extra pins CPU jax; install the CUDA wheel yourself (we run jax[cuda12]; cuda13 is untested)
pip install "grid-rbd[torch]"                  # + torch autograd bridge (notebook 02) — needs a CUDA (cu1xx) torch wheel; torch >= 2.14 compiles the .so with C++20

# developing GRiD instead? from the repo root:
pip install -e ".[all]"                        # same extras, editable
pip install -r install/requirements-dev.txt    # nbval, for running the notebooks as tests
```

The notebooks load the vendored URDFs under `grid_assets/robots/`, so run them
from a checkout (or point `URDF` at your own file).

If you keep multiple checkouts or worktrees, confirm that `import grid_rbd`
resolves to this one: an editable install made from a different worktree
silently shadows it. Check with:

```bash
python -c "import grid_rbd, inspect; print(inspect.getfile(grid_rbd))"
```

Re-run `pip install -e .` from the checkout you want if it points elsewhere.

## Running

Open any notebook and *Run All*, or run them headless as smoke tests:

```bash
pytest --nbval-lax examples/notebooks/
```

The first `register_robot` of a robot pays a one-time `nvcc` build. The
measured cold/warm numbers per robot are in the *What to expect: cold vs. warm*
table of the
[Fast Robot Setup](https://a2r-lab.org/GRiD/docs/user_guide/getting_started/fast_robot_setup.html)
page (iiwa14, the robot these notebooks use, is the cheapest). A kernel
restart + Run All is a cache hit (no recompile). Humanoids (g1/h1_2) take
minutes; pre-warm their cache out-of-band (`grid_rbd.precompile`). The cache
lives under `~/.cache/grid-rbd/` (override with `$GRID_RBD_CACHE_DIR`).
