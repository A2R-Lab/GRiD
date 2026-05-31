# grid-rbd

Python wrappers for GPU-accelerated rigid body dynamics on top of
[GRiD](https://github.com/A2R-Lab/GRiD). Two-tier UX:

```python
import grid_rbd

# One-time per (robot, options, GRiD version, CUDA arch): ~30s-15min.
# Generates grid.cuh, compiles to .so, caches under ~/.cache/grid-rbd/.
handle = grid_rbd.register_robot(
    name="iiwa14",
    urdf_path="iiwa.urdf",      # or urdf_string="<inline URDF text>"
    floating_base=False,
    max_batch_size=256,
    backend="numpy",            # "numpy" (default) | "jax" | "torch"
)

# Many times, fast. All methods are 2D-batched on axis 0.
qdd = handle.forward_dynamics(q, qd, u)
M   = handle.crba(q)
```

Full reference (gravity convention, cache layout, EE-target selection,
JAX FFI, etc.) lives in the
[main docs](https://a2r-lab.github.io/GRiD/).

## Status — v0.4

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
| `idsva_so(q, qd, qdd, gravity=9.81)` | tuple of 4 × `(B, NV, NV, NV)` | 1e-4 |
| `fdsva_so(q, qd, u, gravity=9.81)` | tuple of 4 × `(B, NV, NV, NV)` | 1e-4 |

`register_robot` accepts `ee_joint_names=[...]` to pin specific
end-effector frames (default: all leaf links).

## JAX FFI (`grid_rbd[jax]`)

`pip install grid-rbd[jax]` enables the JAX-side bridge, which shares
the same per-robot `.so` cache. All methods are exposed via
`jax.ffi.ffi_call` and run device-resident on JAX-supplied CUDA streams
— no host round-trip — so they slot directly into `jax.jit` graphs:

```python
import grid_rbd.jax as grid_jax, jax
handle = grid_jax.register_robot(name="iiwa14", urdf_path="iiwa.urdf")

@jax.jit
def step(q, qd, u):
    return handle.forward_dynamics(q, qd, u)
```

Full parity with the plain wrapper as of v0.3.

## PyTorch backend (`backend="torch"`)

`register_robot(..., backend="torch")` returns a `TorchRobotHandle`
whose methods return `torch.Tensor`. The four differentiable algorithms
(`rnea` / `forward_dynamics` / `aba` / `integrator`) are autograd-aware,
with analytic backward passes that reuse the existing `*_gradient`
kernels; the remaining methods are forward-only ops. The `.so` is shared
with the numpy/JAX surfaces (same content-addressed cache):

```python
import grid_rbd, torch

h = grid_rbd.register_robot("iiwa14", urdf_path="iiwa.urdf", backend="torch")

q  = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)
qd = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)
u  = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)

qdd = h.forward_dynamics(q, qd, u)   # autograd-aware torch.Tensor
qdd.sum().backward()                 # gradients flow to q, qd, u

# CUDA-Graphs replay for fixed-batch MPC / training:
g = h.capture("forward_dynamics", q, qd, u)   # off-graph warmup + capture
qdd = g(q_new, qd_new, u_new)                 # copy_ + replay
```

> **GPU/torch compatibility:** the backward VJP contractions run torch's
> own CUDA kernels, so the installed torch build must support the GPU's
> compute capability. On an RTX 5090 (sm_120) you need a torch **cu128**
> (or newer) build — a cu124 wheel (max sm_90) cannot launch on sm_120.
> The `grid` / `grid_plant` kernels themselves are always nvcc-built for
> the detected arch and are unaffected.

## `grid_plant` cost / barrier / plant-step methods

The handle also exposes the generated `grid_plant` trajectory-optimization
surface (validated against `RBDReference._PlantMixin`). All take/return 2D
arrays with axis 0 = batch; cost methods return `(value, grad, hess)` and
barriers return `(value, grad, hess_diag)`:

| Method | Returns |
|---|---|
| `quadratic_state_cost(x, x_des, Q)` | `value (B,)`, `grad (B, NX)`, `hess (B, NX, NX)` |
| `quadratic_input_cost(u, u_des, R)` | `value (B,)`, `grad (B, NV)`, `hess (B, NV, NV)` |
| `ee_pos_cost(q, p_des, W)` | `value (B,)`, `grad (B, NX)`, GN `hess (B, NX, NX)` |
| `joint_position_barrier(var, lower, upper, mu)` | `value (B,)`, `grad (B, NP)`, `hess_diag (B, NP)` |
| `joint_velocity_barrier(var, lower, upper, mu)` | as above over `NV` |
| `joint_torque_barrier(var, lower, upper, mu)` | as above over `NV` |
| `plant_step(x, u, dt, integrator_type="euler")` | `(B, NX)` next state |

## External forces (`f_ext`)

Per-body external forces are an opt-in feature of the underlying CUDA
codegen and the `RBDReference` oracle (body-local frame, subtracted from
the per-body force; an empty/`None` value reproduces the no-force path).
The CUDA host wrappers carry the `d_f_ext` argument; an `f_ext=` kwarg on
the `RobotHandle` algorithm methods is on the roadmap.

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
