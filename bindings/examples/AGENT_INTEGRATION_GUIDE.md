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

All three share **one cache** (the same compiled `.so`, keyed by URDF bytes + codegen options +
GRiD version + CUDA arch). Build once, use from any surface. Install is opt-in per backend:
`pip install -e "bindings/[jax]"` / `[torch]` / `[all]` (base is numpy-only). `nvcc` must be on
`PATH` at build time (not at `pip install` time); the per-robot `.so` is built on first use.

## Lifecycle: register → precompile → get_robot

Three entry points over the same cache:

| Call | Does | Use when |
|------|------|----------|
| `register_robot(name, urdf_path, ...)` | build-if-missing **and** return a handle | one-shot build + get |
| `precompile(name, urdf_path, tiers=..., backends=...)` | build + cache only (no handle) | warm the cache offline (CI / Docker), maybe several tiers/backends |
| `get_robot(name)` | look up an already-registered robot by name | fast start once the cache is warm; raises `RobotNotRegisteredError` if absent |

```python
import grid_rbd
grid_rbd.precompile("iiwa14", "/path/to/iiwa14.urdf",
                    floating_base=False,            # True for free-base humanoids/quadrupeds
                    ee_joint_names=["iiwa_joint_ee"],  # default: all leaf links
                    max_batch_size=1024,            # cap the batch you'll run; bake it in
                    tiers=[{}, {"floating_base": True}],  # prebuild several .so variants
                    backends=("jax", "torch"))      # warm the surfaces you'll use
```

`precompile` is idempotent: a cached tier is an instant no-op (no `nvcc`). First build of a
small arm is ~30–60 s; large humanoids with second-order kernels take minutes (and lots of RAM).
Ship the cache dir (`grid_rbd.default_cache_dir()`, default `~/.cache/grid-rbd/` or
`$GRID_RBD_CACHE_DIR`) and every later run starts in well under a second.

- `max_batch_size` is a **compile-time** cap. Calls with `batch ≤ max_batch` run in one launch;
  larger batches must be chunked by the caller. Bake in the largest batch you'll run.
- `urdf_string="<inline URDF>"` works in place of `urdf_path` (cache keys on the bytes, so an
  inline string and the equivalent file dedupe to the same `.so`).
- `algorithm_list=["forward_dynamics", ...]` builds only a **subset** (plus auto-pulled
  dependencies) — big win in nvcc time / RAM / `.so` size for robots with heavy second-order
  kernels. An un-built method raises a clear "add to `algorithm_list` and rebuild" error, never a
  segfault.

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

`B` = batch, `NJ` = `num_joints` (= `nq`), `NV` = `num_vel` (tangent space). **Fixed base:
`NV == NJ`** and all shapes coincide; floating base: `NJ > NV` (the free-flyer adds a quat slot).

**Dynamics / kinematics** — `inverse_dynamics(q,qd,qdd=None)`→`(B,NJ)` (alias `rnea`; `qdd=None`
gives the bias `c=h−g`, nonzero adds `M·qdd`) · `forward_dynamics(q,qd,u)`→`(B,NJ)` (alias `fd`) ·
`aba(q,qd,u)`→`(B,NJ)` · `crba(q)`→`(B,NV,NV)` · `minv(q)`→`(B,NV,NV)` ·
`end_effector_pose(q)`/`_gradient`/`_hessian` → `(B,6·num_ees[,NV[,NV]])`.

**Derivatives** — `inverse_dynamics_gradient(q,qd,qdd=None)`→`(B,NV,2·NV)` (=`[∂/∂q | ∂/∂qd]`;
qdd-aware) · `forward_dynamics_gradient(q,qd,u)`→`(B,NV,2·NV)` · second-order
`idsva_so(q,qd,qdd)` / `fdsva_so(q,qd,u)` → a `SecondOrderID`/`SecondOrderFD` NamedTuple of
4 × `(B,NV,NV,NV)` tensors (named fields, unpack positionally).

**Trajectory-opt (`grid_plant`)** — `integrator(q,qd,u,dt)`/`_gradient` and
`plant_step(x,u,dt)`→`(B,NX)` / `_gradient`→`(B,2·NV,3·NV)` `[A|B]`, where state `x=[q;qd]`
(`NX=2·NV`). `integrator_type=` is `euler` (default) / `semi_implicit_euler` / `midpoint` /
`rk3` / `rk4`. Cost terms return `(value, grad, hess)`: `quadratic_state_cost(x,x_des,Q)`,
`quadratic_input_cost(u,u_des,R)`, `ee_pos_cost(q,p_des,W)`, `com_cost`, `momentum_cost`; barriers
(`joint_position_barrier`, `joint_velocity_barrier`, `joint_torque_barrier`) return
`(value, grad, hess_diag)`. These are the building blocks for an on-GPU MPC/DDP step.

**numpy-only extras** (not yet on jax/torch): `com`, `ccrba`, `dccrba`, `cmm_time_variation`,
`coriolis_matrix`, `energy`, `generalized_gravity`, `nonlinear_effects`,
`{kinetic,potential}_energy_regressor`, `frame_jacobian`/`_dot` (runtime `target_jid` +
`reference_frame` LOCAL/WORLD/LOCAL_WORLD_ALIGNED), `osc_inertia`, and the runtime-target EE pose
(`end_effector_pose_runtime`, `..._gradient_runtime` — pick any target frame + offset at call
time, one compiled robot serves all).

Properties: `num_joints`, `num_vel`, `num_ees`, `floating_base`, `max_batch`, `output_convention`.

> **fp32 caveat:** the jax/torch surfaces are **strictly fp32** (compute and I/O). The numpy
> handle defaults to fp32 too but supports a true fp64 tier via `dtype="float64"` (separate `.so`)
> or the legacy `allow_fp64=True` fp32-compute-with-fp64-cast convenience.

## Gradients are analytic

`jax.grad` / `loss.backward()` do **not** autodiff or finite-difference through GRiD — each
differentiable method carries GRiD's own **analytic** Jacobian (a matvec FFI call). On JAX:
`inverse_dynamics`/`forward_dynamics`/`aba`/`end_effector_pose`/`integrator` + the π-regressor VJP
+ `f_ext` parity. On torch: `inverse_dynamics`/`forward_dynamics`/`aba`/`integrator` (the rest are
forward-only). The `inverse_dynamics` gradient is qdd-aware (includes the `∂(M·q̈)/∂q` term).

## Output conventions

Default is **Pinocchio** convention. For MuJoCo/MJX-native I/O (wxyz quat, global-linear free-joint
velocity) use the `.mujoco` view (`h.mujoco.forward_dynamics(...)`), `output_convention="mujoco"`,
or set `handle.output_convention`. It is **floating-base only** (fixed-base, the two coincide) and
a **runtime** setting (not in the cache key — same `.so`). The `.mujoco` view applies the convention
per-call and is thread-safe, so it's safe to mix with pinocchio-convention calls on the same handle.
Today the **value** methods (id/fd/aba/crba/minv) honor mujoco mode; the derivative/second-order
surfaces raise a clear error in mujoco mode (use pinocchio and transform).

## Do / Don't

- **Do** `device_put` inputs once and keep outputs as device arrays/tensors across calls.
- **Do** wrap multi-call logic in `@jax.jit` (or `capture()` for torch) so GRiD calls fuse /
  replay instead of dispatching one at a time.
- **Do** set `max_batch_size` to the largest batch you'll run, at build time.
- **Do** match input dtype to the surface: fp32 for jax/torch (an fp64 array forces a copy).
- **Don't** convert to `np.asarray` / `.cpu()` between GRiD calls in a hot loop — that's a
  D2H+H2D round-trip per step and erases the GPU advantage (see the timed anti-pattern in
  `jax_gpu_resident.py`).
- **Don't** rebuild per run — `precompile` once and reuse the cache.
- **Don't** size a batch above `max_batch` (chunk instead), and don't expect derivative/SO
  methods in `mujoco` mode yet — both raise a clear error rather than returning wrong data.

## Common pitfalls

- **`RobotNotRegisteredError`** from `get_robot` → the cache isn't warm; run `register_robot` /
  `precompile` first (or ship the cache dir).
- **"symbol missing" / "not built into this `.so`"** → you used `algorithm_list=` and called a
  method outside the subset; add it and rebuild. Same on all three surfaces.
- **torch "no kernel image available for sm_120"** → the backward VJP runs torch's own CUDA
  kernels, so the installed torch wheel must support the GPU arch (e.g. RTX 5090 / sm_120 needs a
  **cu128+** build; cu124 maxes at sm_90). The GRiD kernels themselves are always nvcc-built for
  the detected arch and are fine.
- **torch `capture()` during stream capture** → `capture()` runs a mandatory off-graph warmup for
  one-time device setup (the >48 KB dynamic-smem opt-in) before recording; don't call methods
  cold inside your own `torch.cuda.graph` context.
- **Big-robot SO builds are heavy** (10s of GB RAM, minutes) — use `algorithm_list=` to build only
  what you need, and prebuild offline.

## Runnable examples in this folder
- [`quickstart_iiwa14.py`](quickstart_iiwa14.py) — register + call every method (numpy).
- [`jax_gpu_resident.py`](jax_gpu_resident.py) — residency, jit/vmap/grad, `lax.scan` rollout,
  donate, dlpack. The reference for the JAX fast path.
- [`torch_cuda_graphs.py`](torch_cuda_graphs.py) — CUDA tensors, autograd, CUDA-Graphs replay.
