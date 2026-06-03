# grid-rbd example notebooks

A guided tour of the `grid_rbd` Python bindings — the **register-then-run** UX:
register a robot once (parses the URDF, generates `grid.cuh`, compiles a
per-robot `.so` into a content-addressed cache), then call algorithms many
times, all batched over axis 0.

Every notebook ends in `assert` cells that cross-check GRiD against
`RBDReference` or a finite-difference of its own forward map — so a green
**Run All** validates the *numbers*, not just "no exception." They double as CI
smoke tests (`pytest --nbval-lax examples/notebooks/`).

## Notebooks

| # | Notebook | Covers |
|---|----------|--------|
| 01 | [`01_quickstart_iiwa14`](01_quickstart_iiwa14.ipynb) | Register a robot; `inverse_dynamics` / `forward_dynamics` / `minv`, single + batched on one handle; `f_ext`; validate vs `RBDReference`. |
| 02 | [`02_autograd_torch`](02_autograd_torch.ipynb) | The **torch** backend: autograd through `forward_dynamics`, analytic-vs-finite-difference VJP, a tiny IK descent. |
| 03 | [`03_plant_control`](03_plant_control.ipynb) | The control surface: `plant_step`, quadratic state/input costs, end-effector cost, joint-limit log-barriers. |
| 04 | [`04_kinematics`](04_kinematics.ipynb) | FK (`end_effector_pose` / `fk_batched`), `end_effector_pose_gradient` / `_hessian`, geometric `frame_jacobian` / `_dot` with **runtime** frame + target selection, `com` / `ccrba` / `osc_inertia`. |
| 05 | [`05_gradients_secondorder`](05_gradients_secondorder.ipynb) | Analytic dynamics derivatives: `inverse_dynamics_gradient`, `forward_dynamics_gradient` (first order) and `idsva_so` / `fdsva_so` (second order), each vs finite-difference. |
| 06 | [`06_jax_backend`](06_jax_backend.ipynb) | The **JAX** backend (`grid_rbd.jax`): `jax.jit` over dynamics, the native batch axis (vmap-equivalent), GRiD's analytic gradient kernels jitted + FD-validated, `idsva_so` vs the numpy handle. Needs `pip install -e "bindings/[jax]"`. |
| 07 | [`07_inline_cuda`](07_inline_cuda.ipynb) | **Inline CUDA**: generate `grid.cuh`, write a kernel that calls `grid::inverse_dynamics_device`, compile with `nvcc` in-notebook, run, and validate vs `RBDReference`. The tutorial version of [`../cuda/`](../cuda/). |

There are also CUDA-level examples (write-your-own-kernel walkthroughs) under
[`../cuda/`](../cuda/) — notebook 07 is the inline tutorial version. See the
[`examples/` overview](../README.md) for the three tracks (bindings vs codegen
vs hand-written CUDA).

## Setup

Requires a **CUDA GPU + `nvcc` on PATH**. Install `grid_rbd` **editable from
this repository** (not from PyPI — you want the bindings that match this
checkout):

```bash
# from the repo root
pip install -e bindings/                 # numpy backend
pip install -e "bindings/[jax]"          # + JAX FFI bridge
pip install -e "bindings/[torch]"        # + torch autograd bridge (notebook 02)
pip install -r requirements-dev.txt     # nbval, for running the notebooks as tests
```

> **Importing the right tree.** If you keep multiple checkouts/worktrees, make
> sure `import grid_rbd` resolves to *this* one — an editable install installed
> from a different worktree silently shadows it. Check with:
> ```bash
> python -c "import grid_rbd, inspect; print(inspect.getfile(grid_rbd))"
> ```
> Re-run `pip install -e bindings/` from the checkout you want if it points
> elsewhere.

## Running

Open any notebook and **Run All**, or run them headless as smoke tests:

```bash
pytest --nbval-lax examples/notebooks/
```

**Compile-time expectation.** iiwa14 (7-DOF fixed-base) is the cheapest robot —
the first `register_robot` is *seconds to tens of seconds* of `nvcc`; a kernel
restart + Run All is a **cache hit** (no recompile). Humanoids (g1/h1_2) take
*minutes* — pre-warm their cache out-of-band. The cache lives under
`~/.cache/grid-rbd/` (override with `$GRID_RBD_CACHE_DIR`).
