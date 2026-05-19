# grid-rbd

Python wrappers for GPU-accelerated rigid body dynamics on top of
[GRiD](https://github.com/A2R-Lab/GRiD). Two-tier UX:

```python
import grid_rbd

# One-time per (robot, options, GRiD version, CUDA arch): ~30s-15min.
# Generates grid.cuh, compiles to .so, caches under ~/.cache/grid-rbd/.
handle = grid_rbd.register_robot(
    name="iiwa14",
    urdf_path="iiwa.urdf",
    floating_base=False,
    max_batch_size=256,
)

# Many times, fast. All methods take (B, NJ) and return (B, NJ) or (B, NJ, NJ).
qdd = handle.forward_dynamics(q, qd, u)
M   = handle.crba(q)
```

## Status — v0.2

Methods bound and validated against [`RBDReference`](https://github.com/A2R-Lab/RBDReference)
at float32 precision:

| Method | Returns | max_err vs RBDReference |
|---|---|---|
| `rnea(q, qd, qdd=None, gravity=9.81)` | `(B, NJ)` | 4.9e-6 |
| `minv(q)` | `(B, NJ, NJ)` | 1.1e-4 |
| `forward_dynamics(q, qd, u, gravity=9.81)` | `(B, NJ)` | 5.7e-5 |
| `aba(q, qd, u, gravity=9.81)` | `(B, NJ)` | 5.7e-5 |
| `crba(q, gravity=9.81)` | `(B, NJ, NJ)` | 2.7e-7 |
| `end_effector_pose(q)` | `(B, 6*NUM_EES)` | 1.4e-7 |
| `end_effector_pose_gradient(q)` | `(B, 6*NUM_EES, NJ)` | 3.1e-7 |
| `end_effector_pose_hessian(q)` | `(B, 6*NUM_EES, NJ, NJ)` | 3.1e-7 |
| `rnea_grad(q, qd, qdd=None, gravity=9.81)` | `(B, NJ, 2*NJ)` | 1.6e-5 |
| `forward_dynamics_grad(q, qd, u, gravity=9.81)` | `(B, NJ, 2*NJ)` | 1.3e-4 |
| `idsva_so(q, qd, qdd, gravity=9.81)` | `(B, NJ, 3*NJ)` | 1e-4 |
| `fdsva_so(q, qd, u, gravity=9.81)` | `(B, NJ, 3*NJ)` | 1e-4 |

`register_robot` accepts `ee_joint_names=[...]` to pin specific
end-effector frames (default: all leaf links).

## JAX FFI (`grid_rbd[jax]`)

Install with `pip install grid-rbd[jax]` for the JAX-side bridge,
which shares the same per-robot `.so` cache. Methods exposed via
`jax.ffi.ffi_call` so they slot into `jax.jit` graphs and run on
JAX-supplied CUDA streams (device-resident — no host round-trip):

```python
import grid_rbd.jax as grid_jax, jax
handle = grid_jax.register_robot(name="iiwa14", urdf_path="iiwa.urdf")

@jax.jit
def step(q, qd, u):
    return handle.forward_dynamics(q, qd, u)
```

v0.2 JAX surface: `rnea`, `minv`, `forward_dynamics`, `aba`, `crba`.
The remaining methods (EE pose family, derivative kernels, SO) still
work through the plain `grid_rbd.RobotHandle`; extending them to JAX
FFI is mechanical follow-up.

## Architecture

* **Pure-Python orchestration** under `grid_rbd/` for cache management,
  codegen invocation, and nvcc shell-out.
* **Small pybind11 extension** at `grid_rbd/_core` — built once at
  `pip install` time — that dlopens the per-robot .so and dispatches
  numpy↔C-ABI calls.
* **Per-robot `.so`** built at `register_robot` time, cached by content
  hash. Each .so embeds `grid.cuh` (from GRiDCodeGenerator) plus the
  robot-agnostic `wrapper.cu` (in this package) that exposes the
  `extern "C"` symbols the runner looks up.

See [`docs/python_wrappers_plan.md`](../docs/python_wrappers_plan.md)
for the full design rationale.

## Gravity convention

This wrapper passes `gravity` as a **positive magnitude** (default 9.81)
matching GRiD's internal convention. If you cross-check against
`RBDReference.rnea(..., GRAVITY=-9.81)`, pass `gravity=9.81` here.

## Cache layout

```
~/.cache/grid-rbd/
├── manifest.json              # name -> cache_key
└── store/<cache_key>/
    ├── grid.cuh
    ├── wrapper.cu
    ├── robot.so
    ├── meta.json
    └── robot.build.log
```

Override the cache location with `$GRID_RBD_CACHE_DIR` or
`cache_dir=...` on `register_robot`.

## Requirements

* Python ≥ 3.10
* CUDA Toolkit (`nvcc` on PATH) at `register_robot` time. Not needed
  for `pip install grid-rbd` itself.
* numpy ≥ 1.23

## Install (editable, from a GRiD checkout)

```bash
cd path/to/GRiD
pip install -e python/
```

A PyPI release will follow once the surface is feature-complete.
