# Integrating GRiD via `grid_rbd` — a guide for agents & users

How to call GRiD's GPU rigid-body dynamics from Python **as effectively as possible**. The
golden rule: **place your data on the GPU once and keep it there.** GRiD is a GPU library;
its speed comes from staying resident across an entire control / learning pipeline, not from
one-shot calls that round-trip to the host.

## The three surfaces — pick by where your data lives

| Handle | Get it with | In / out | Use when |
|--------|-------------|----------|----------|
| **numpy** (base) | `grid_rbd.get_robot(name)` / `register_robot(...)` | host `np.ndarray` | scripting, tests, "just give me the answer". Convenience, **not** the speed path — every call is H2D + D2H. |
| **jax** | `grid_rbd.jax.get_robot(name)` | device `jax.Array` | JAX pipelines: `jit` / `vmap` / `grad` / `lax.scan`. **The fast path.** |
| **torch** | `grid_rbd.torch.get_robot(name)` | CUDA `torch.Tensor` | PyTorch training / MPC; autograd-aware; `capture()` for CUDA-Graphs replay. **The fast path.** |

All three share **one cache** (the same compiled `.so`). Build once, use from any surface.

## Build once, start instantly

```python
import grid_rbd
grid_rbd.precompile("iiwa14", "/path/to/iiwa14.urdf",
                    floating_base=False,            # True for free-base humanoids/quadrupeds
                    ee_joint_names=["iiwa_joint_ee"],
                    max_batch_size=1024,             # cap the batch you'll run; bake it in
                    backends=("jax", "torch"))       # warm the surfaces you'll use
```

`precompile` is idempotent: a cached robot is an instant no-op (no `nvcc`). First build of a
small arm is ~30–60 s; large humanoids with second-order kernels take longer. Ship the cache
dir (`grid_rbd.default_cache_dir()`, default `~/.cache/grid-rbd/`) and every later run starts
in well under a second. `register_robot(..., backend="jax")` is the one-shot build+get.

## The fast path (JAX): stay resident, compose, differentiate

```python
import jax, jax.numpy as jnp, grid_rbd.jax as gj
h = gj.get_robot("iiwa14")                     # JaxRobotHandle; methods return jax.Array
q  = jax.device_put(jnp.asarray(q_np))         # H2D ONCE
qd = jax.device_put(jnp.asarray(qd_np))
u  = jax.device_put(jnp.asarray(u_np))

@jax.jit                                        # fuse GRiD calls + your math into one GPU program
def step(q, qd, u):
    qdd = h.forward_dynamics(q, qd, u)          # FFI call — output never leaves the GPU
    return jnp.mean(qdd**2) + 1e-3*jnp.mean(u**2)

c      = step(q, qd, u)                          # device-resident scalar
g_u    = jax.grad(step, argnums=2)(q, qd, u)     # ANALYTIC gradient via GRiD's custom_vjp
batched = jax.vmap(step)(qb, qdb, ub)            # batch with no Python loop
```

- Every method is a `jax.custom_vjp` over `jax.ffi.ffi_call`: it **composes** under
  `jit`/`vmap`/`grad` and **chains** into the next GRiD call with no host hop.
- Gradients are GRiD's **analytic** Jacobians (a matvec FFI call), not autodiff or
  finite-difference — correct and fast.
- **Resident rollout:** put a GRiD call inside `jax.lax.scan` so a whole K-step MPC/rollout
  horizon is one GPU program and the state is carried device-to-device. See
  [`jax_gpu_resident.py`](jax_gpu_resident.py) for a timed comparison vs the host-roundtrip
  anti-pattern (often 10×+).
- `donate_argnums=` lets XLA reuse an input buffer in place.

## The fast path (PyTorch): autograd + CUDA-Graphs

```python
import torch, grid_rbd.torch as gt
h = gt.get_robot("iiwa14")                      # TorchRobotHandle; methods return CUDA tensors
qdd = h.forward_dynamics(q, qd, u)              # q,qd,u are cuda tensors → qdd is a cuda tensor
loss = qdd.pow(2).mean(); loss.backward()        # analytic grads flow to q/qd/u

g = h.capture("forward_dynamics", q, qd, u)     # record a CUDA Graph (fixed batch)
out = g(q_new, qd_new, u_new)                    # memcpy-in + replay + memcpy-out (low overhead)
```

`capture()` collapses dozens of per-kernel launches into a single graph replay — a large win
in tight MPC / RL loops where launch overhead dominates. See
[`torch_cuda_graphs.py`](torch_cuda_graphs.py).

## Zero-copy interop (share the GPU pointer)

JAX ↔ PyTorch via dlpack, no copy — the literal device buffer is shared:
```python
t = torch.utils.dlpack.from_dlpack(jax_array)    # JAX → torch
j = jax.dlpack.from_dlpack(torch_tensor.contiguous())   # torch → JAX
```

## Methods (all batched on axis 0; jax/torch are fp32)

`inverse_dynamics(q,qd,qacc)` · `forward_dynamics(q,qd,u)` · `aba(q,qd,u)` · `crba(q)` ·
`minv(q)` · `inverse_dynamics_gradient(q,qd,qacc)` · `forward_dynamics_gradient(q,qd,u)` ·
`idsva_so(q,qd,qacc)` · `fdsva_so(q,qd,u)` (2nd-order) · `end_effector_pose(q)` /
`_gradient` / `_hessian` · `inverse_dynamics_regressor(q,qd,qacc)` · `integrator(q,qd,u,dt)` /
`_gradient` · `plant_step(x,u,dt)` / `_gradient` · cost terms (`quadratic_state_cost`,
`ee_pos_cost`, `com_cost`, `momentum_cost`). Properties: `num_joints`, `num_vel`, `num_ees`,
`floating_base`, `max_batch`.

## Output conventions

Default is **Pinocchio** convention. For MuJoCo/MJX-native I/O use the `.mujoco` view
(`h.mujoco.forward_dynamics(...)`) or `output_convention="mujoco"` — floating-base only
(fixed-base, the two coincide). The `.mujoco` view applies the convention per-call, so it's
safe to mix with pinocchio-convention calls on the same handle.

## Do / Don't

- **Do** `device_put` inputs once and keep outputs as device arrays/tensors across calls.
- **Do** wrap multi-call logic in `@jax.jit` (or `capture()` for torch) so GRiD calls fuse /
  replay instead of dispatching one at a time.
- **Do** set `max_batch_size` to the largest batch you'll run, at build time.
- **Don't** convert to `np.asarray` / `.cpu()` between GRiD calls in a hot loop — that's a
  D2H+H2D round-trip per step and erases the GPU advantage (see the timed anti-pattern in
  `jax_gpu_resident.py`).
- **Don't** rebuild per run — `precompile` once and reuse the cache.

## Runnable examples in this folder
- [`quickstart_iiwa14.py`](quickstart_iiwa14.py) — register + call every method (numpy).
- [`jax_gpu_resident.py`](jax_gpu_resident.py) — residency, jit/vmap/grad, `lax.scan` rollout,
  donate, dlpack. The reference for the JAX fast path.
- [`torch_cuda_graphs.py`](torch_cuda_graphs.py) — CUDA tensors, autograd, CUDA-Graphs replay.
