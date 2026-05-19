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

# Many times, fast. All methods are 2D-batched on axis 0.
qdd = handle.forward_dynamics(q, qd, u)
M   = handle.crba(q)
```

Full reference (gravity convention, cache layout, EE-target selection,
JAX FFI, etc.) lives in the
[main docs](https://a2r-lab.github.io/GRiD/).

## Status — v0.3

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
