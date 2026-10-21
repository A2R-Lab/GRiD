# Integrating GRiD via `grid_rbd`: a guide for agents and users

*Last reviewed 2026-10-14 against main `f4f72e7` (grid-rbd 0.5.1). The API surface below (entry points, `get_robot` compile-on-miss, cache layout and `gc_cache`, build knobs, method shapes) was re-checked against `bindings/grid_rbd` and `docs/source/user_guide/tutorials/python_wrappers.rst`. The measured numbers quoted inline (the 2026-10-01 host-copy timings, the iiwa14 FFI block-size figure, the CUDA-graph submit costs) were not re-measured. If the code and this guide disagree, trust the code and fix the guide.*

This guide covers how to call GRiD's GPU rigid-body dynamics from Python
effectively. The governing rule is to place your data on the GPU once and keep it
there. GRiD is a GPU library; its speed comes from staying resident across an
entire control / learning pipeline, not from one-shot calls that round-trip to
the host.

## The three surfaces

Pick the surface by where your data lives.

| Handle | Get it with | In / out | Use when |
|--------|-------------|----------|----------|
| numpy (base) | `grid_rbd.get_robot(name)` / `register_robot(...)` | host `np.ndarray` | scripting, tests, one-off answers. Convenience, not the speed path: every call is H2D + D2H. |
| jax | `grid_rbd.jax.get_robot(name)` | device `jax.Array` | JAX pipelines: `jit` / `vmap` / `grad` / `lax.scan`. The fast path. |
| torch | `grid_rbd.torch.get_robot(name)` | CUDA `torch.Tensor` | PyTorch training / MPC; autograd-aware; `capture()` for CUDA-Graphs replay. The fast path. |

All three share one cache: the same compiled `.so`, keyed by URDF bytes + codegen options +
GRiD version + CUDA arch + wrapper/codegen source hashes. Editing grid_codegen or the
wrapper rotates every key; rebuilds are automatic, and no force_rebuild is needed. Build once, use from any surface. Install is opt-in per backend:
`pip install "grid-rbd[jax]"` / `[torch]` / `[all]` from PyPI (base is numpy-only; `pip install -e ".[all]"`
from a checkout to develop GRiD). `nvcc` must be on
`PATH` at build time (not at `pip install` time); the per-robot `.so` is built on first use.

## Lifecycle: load_robot / register, precompile, get_robot

### Runtime contexts

A handle dispatches to the artifact's default context
(`handle.ctx_id == 0`, shared by every handle over the same `.so`). `handle.context()` (numpy,
jax and torch handles alike) gives an isolated context on the same artifact, with its own
arena, tables, streams and launch overrides, closed when that handle closes. `handle.device_profile`
reports what the context was fitted to (device/artifact arch, memory at init, arena bytes, workspace
slots, `max_batch`). Ids are salted per artifact: a foreign or closed id raises a clear error, never a
use-after-free. Details: the *Runtime contexts* concepts page.

### Model versions

Runtime-parameter mutations (`set_inertia_params`, `set_transform_params`,
`set_joint_dynamics`, `attach_tool`/`detach_tool`) take the context's admission lock exclusively (ordered
after every admitted call, before every later one) and bump `handle.model_version`. torch/JAX autograd
forwards stamp that version on device at execution time and their backwards refuse a stale stamp
(`model mutated between forward and backward … recompute the forward`). A jitted `jax.grad` keeps
working across mutations, because forward and backward stamp and check inside one execution; only a backward
deferred across a mutation is refused. Launch overrides serialize the same way but do not bump the version.

### Operand validation

Every input belongs to one operand class (configuration `(B, nq)`, force
`(B, 6*num_bodies)`, plant state `(B, nq+nv)`, plant operands, tool operands, runtime offset, scalars) and each
surface enforces the class rule natively: `1 <= B <= max_batch`, 2D with the exact last dim, every operand
carrying the leading operand's batch, `f_ext` never broadcast. numpy coerces layout/dtype (contiguous copy,
cast); torch refuses CPU / non-contiguous / wrong-dtype tensors; JAX checks in Python first and again in every
FFI handler. Details and the not-checked column: the *Operand validation* concepts page.

### Build calls

Three build calls plus one lookup, all over the same cache (the *Fast Robot
Setup* page is the canonical description):

| Call | Does | Use when |
|------|------|----------|
| `load_robot(urdf_path, backend=...)` | no ceremony: derives a stable content-addressed name, builds-if-missing, returns a handle | the frictionless default; no name to invent |
| `register_robot(name, urdf_path, ...)` | build-if-missing and return a handle under your name; the full option surface | you want a human-friendly handle name or non-default build options |
| `precompile(name, urdf_path, tiers=..., backends=...)` | build + cache only (no handle) | warm the cache offline (CI / Docker), maybe several tiers/backends |
| `get_robot(name[, urdf_path=..., **register_opts])` | look up a registered robot by name; since 0.5.1, with `urdf_path=` a *miss* compiles and registers that URDF just-in-time (a hit ignores it) | fast start once the cache is warm, or "get or build" in one call. Without `urdf_path` a miss raises `RobotNotRegisteredError`; `StaleRobotError` means the cached build is from another wrapper/ABI/arch (rebuild) |

```python
import grid_rbd
grid_rbd.precompile("iiwa14", "/path/to/iiwa14.urdf",
                    floating_base=False,            # True for free-base humanoids/quadrupeds
                    ee_joint_names=["iiwa_joint_ee"],  # default: all leaf links
                    max_batch_size=1024,            # cap the batch you'll run; bake it in
                    tiers=[{}, {"floating_base": True}],  # prebuild several .so variants
                    backends=("jax", "torch"))      # warm the surfaces you'll use
```

`precompile` is idempotent: a cached tier is an instant no-op (no `nvcc`). The `numpy` backend is
warmed without a handle or a CUDA context (a GPU-less build box works with `cuda_arch=`);
`grid_rbd.build_plan(name, urdf, cuda_arch=...)` shows what a build would key on and whether the
cache already holds it, without building. Build times are tabulated on the
*Fast Robot Setup* page (0.5.1 multi-unit build: iiwa14 ~100 s, go2-floating ~2 min, g1
pin-only ~6 min on an RTX 5090 / 24-core host; humanoid second-order units need
a few GB of RAM each). A first build announces itself with a one-line `WARNING`
on the `grid_rbd` logger (robot, unit count, rough duration, cache path);
`GRID_RBD_QUIET=1` silences it. `GRID_RBD_BUILD_JOBS=N` caps the parallel
units; `GRID_RBD_SPLIT_COMPILE=N` (opt-in) trades RAM for a faster heavy unit;
`GRID_RBD_CUDA_ARCHS=86,89,90,120` (opt-in) builds one fat `.so` for several GPUs,
keyed apart from the single-arch build and carrying the launch configs of its
`cuda_arch` (see the build knobs in `bindings/README.md`);
`python -m grid_rbd.gc_cache` lists (`--execute` deletes) store entries no name
binds any more.
Ship the cache dir (`grid_rbd.default_cache_dir()`, default `~/.cache/grid-rbd/` or
`$GRID_RBD_CACHE_DIR`) and every later run starts in well under a second.

- `max_batch_size` is a compile-time cap. Calls with `batch ≤ max_batch` run in one launch;
  larger batches must be chunked by the caller. Bake in the largest batch you will run.
- `urdf_string="<inline URDF>"` works in place of `urdf_path` (cache keys on the bytes, so an
  inline string and the equivalent file dedupe to the same `.so`).
- `algorithm_list=["forward_dynamics", ...]` builds only a subset (plus auto-pulled
  dependencies), which reduces nvcc time, RAM and `.so` size for robots with heavy second-order
  kernels. An un-built method raises a clear "add to `algorithm_list` and rebuild" error, never a
  segfault.

## Host arrays (numpy): allocate once, reuse

For the big outputs (`idsva_so` / `fdsva_so`), allocate a page-locked buffer once with
`h.pinned_empty((B, 4*h.num_vel**3))` and pass it as `out=` on every call: the device-to-host
copy lands in it directly (no extra host memcpy) and the returned tensors are views of it.
The result is ~40 ms vs 120 ms per call on g1 @1024 (702 MB), measured 2026-10-01.

`inverse_dynamics_gradient` / `forward_dynamics_gradient` take `out=` as well: allocate
`h.pinned_empty((B, 2*h.num_vel**2))` once; the returned `(B, NV, 2NV)` array is a view of
it (column-major per item, so not C-contiguous), with no per-call allocation and no host re-layout.

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

- Every method is a `jax.custom_vjp` over `jax.ffi.ffi_call`: it composes under
  `jit`/`vmap`/`grad` and chains into the next GRiD call with no host hop.
- Gradients are GRiD's analytic Jacobians (a matvec FFI call), not autodiff or
  finite-difference.
- For a resident rollout, put a GRiD call inside `jax.lax.scan` so a whole K-step MPC/rollout
  horizon is one GPU program and the state is carried device-to-device. See
  [`jax_gpu_resident.py`](jax_gpu_resident.py) for a timed comparison vs the host-roundtrip
  anti-pattern (often 10×+).
- `donate_argnums=` lets XLA reuse an input buffer in place.
- To get numpy at the end, `gj.to_host(outputs)` (array or pytree) moves through XLA's
  `pinned_host` memory kind and returns zero-copy numpy views: 41 ms vs 141 ms for
  `jax.device_get` on a 702 MB `idsva_so` output (g1 @1024, measured 2026-10-01).

## The fast path (PyTorch): autograd + CUDA-Graphs

```python
import torch, grid_rbd.torch as gt
h = gt.get_robot("iiwa14")                      # TorchRobotHandle; methods return CUDA tensors
qdd = h.forward_dynamics(q, qd, u)              # q,qd,u are cuda tensors → qdd is a cuda tensor
loss = qdd.pow(2).mean(); loss.backward()        # analytic grads flow to q/qd/u

g = h.capture("forward_dynamics", q, qd, u)     # record a CUDA Graph (fixed batch)
out = g(q_new, qd_new, u_new)                    # memcpy-in + replay + memcpy-out (low overhead)
```

`capture()` collapses dozens of per-kernel launches into a single graph replay, a large win
in tight MPC / RL loops where launch overhead dominates. See
[`torch_cuda_graphs.py`](torch_cuda_graphs.py).

Host round trips follow the same allocate-once pattern. `.cpu()` on a big output is a pageable staged copy.
Allocate page-locked mirrors once (`host = gt.pinned_host_like(g.static_out)`) and reuse them with
`g.replay_into(host)` (graph) or `gt.copy_to_host(host, out)` (eager). The result is 39.5 ms vs 238 ms for
`.cpu()` on g1 `idsva_so` @1024 (702 MB), measured 2026-10-01.

## Zero-copy interop (share the GPU pointer)

JAX and PyTorch share the device buffer via dlpack, with no copy:
```python
t = torch.utils.dlpack.from_dlpack(jax_array)    # JAX → torch
j = jax.dlpack.from_dlpack(torch_tensor.contiguous())   # torch → JAX
```

## Methods (all batched on axis 0; jax/torch are fp32)

`B` = batch, `NJ` = `num_joints` (= `nq`), `NV` = `num_vel` (tangent space). On a fixed base,
`NV == NJ` and all shapes coincide; on a floating base, `NJ > NV` (the free-flyer adds a quat slot).

Dynamics and kinematics:

| Method | Returns | Notes |
|--------|---------|-------|
| `inverse_dynamics(q,qd,qdd=None)` | `(B,NJ)` | alias `rnea`; `qdd=None` gives the bias `c=h−g`, nonzero adds `M·qdd` |
| `forward_dynamics(q,qd,u)` | `(B,NJ)` | alias `fd` |
| `aba(q,qd,u)` | `(B,NJ)` | |
| `crba(q)` | `(B,NV,NV)` | |
| `minv(q)` | `(B,NV,NV)` | |
| `end_effector_pose(q)`/`_gradient`/`_hessian` | `(B,6·num_ees[,NV[,NV]])` | |

Derivatives:

| Method | Returns | Notes |
|--------|---------|-------|
| `inverse_dynamics_gradient(q,qd,qdd=None)` | `(B,NV,2·NV)` | `[∂/∂q \| ∂/∂qd]`; qdd-aware |
| `forward_dynamics_gradient(q,qd,u)` | `(B,NV,2·NV)` | |
| `idsva_so(q,qd,qdd)` / `fdsva_so(q,qd,u)` | a `SecondOrderID`/`SecondOrderFD` NamedTuple of 4 × `(B,NV,NV,NV)` tensors | named fields, unpack positionally |

Trajectory optimization (`grid_plant`):

| Method | Returns | Notes |
|--------|---------|-------|
| `integrator(q,qd,u,dt)` / `integrator_gradient` | | `integrator_type=` is `euler` (default) / `semi_implicit_euler` / `midpoint` / `rk3` / `rk4` |
| `plant_step(x,u,dt)` | `(B,NX)` | state `x=[q;qd]` (`NX=2·NV`) |
| `plant_step_gradient(x,u,dt)` | `(B,2·NV,3·NV)` | `[A\|B]` |

Cost terms return `(value, grad, hess)`: `quadratic_state_cost(x,x_des,Q)`,
`quadratic_input_cost(u,u_des,R)`, `ee_pos_cost(q,p_des,W)`, `com_cost`, `momentum_cost`. Barriers
(`joint_position_barrier`, `joint_velocity_barrier`, `joint_torque_barrier`) return
`(value, grad, hess_diag)`. These are the building blocks for an on-GPU MPC/DDP step.

Value ops (all three backends; forward-only on jax/torch): `com`, `ccrba`, `dccrba`,
`cmm_time_variation`, `coriolis_matrix`, `energy`, `generalized_gravity`, `nonlinear_effects`,
`{kinetic,potential}_energy_regressor`, `frame_jacobian`/`_dot` (runtime `target_jid` +
`reference_frame` LOCAL/WORLD/LOCAL_WORLD_ALIGNED), `osc_inertia`, and the runtime-target EE pose
(`end_effector_pose_runtime`, `..._gradient_runtime`: pick any target frame + offset at call
time, one compiled robot serves all).

π-regressor, tool and contact ops:

| Method | Returns | Notes |
|--------|---------|-------|
| `inverse_dynamics_regressor(q,qd,qdd=None)` | `(B,NV,10·NB)` | the joint-torque regressor `Y`, `tau = Y·π`; all three backends |
| `inverse_dynamics_wrt_params(q,qd,params)` | | differentiable in π; jax/torch; autograd flows `∂c/∂π = Y` to `params` |
| `forward_dynamics_parameter_gradient(q,qd,u)` | `(B,NV,10·NB)` | jax/torch; `∂q̈/∂π = −M⁻¹Y` |
| `tool_fext(q, wrench)` | `(B,6·NB)` | maps a world-aligned tool-tip wrench to joint-local `f_ext`; needs `enable_tool=True` |
| `contact_fext(q, f_c)` | `(B,6·NB)` | maps per-contact-frame wrenches to joint-local `f_ext`; needs `register_robot(contact_frames=[...])` |

The canonical per-backend surface list is
`docs/source/user_guide/tutorials/python_wrappers.rst`.

Properties: `num_joints`, `num_vel`, `num_ees`, `num_bodies`, `floating_base`, `max_batch`,
`output_convention`, plus the aliases `nq` / `nv` / `nb` (configuration width /
tangent width / body count).

The jax/torch surfaces are strictly fp32 (compute and I/O). The numpy
handle defaults to fp32 too but supports a true fp64 tier via `dtype="float64"` (separate `.so`)
or the legacy `allow_fp64=True` fp32-compute-with-fp64-cast convenience.

## Gradients are analytic

`jax.grad` / `loss.backward()` do not autodiff or finite-difference through GRiD. Each
differentiable method carries GRiD's own analytic Jacobian (a matvec FFI call). On JAX:
`inverse_dynamics`/`forward_dynamics`/`aba`/`end_effector_pose`/`integrator` + the π-regressor VJP
+ `f_ext` parity. On torch: `inverse_dynamics`/`forward_dynamics`/`aba`/`end_effector_pose`/
`integrator` (the rest are forward-only). The `inverse_dynamics` gradient is qdd-aware (includes
the `∂(M·q̈)/∂q` term).

- On a floating base, the kernels' d/dq Jacobians live in the Pinocchio free-flyer tangent chart
  (`[v_lin local, ω local, joints]`, nv-wide). What `jax.grad`/`q.grad` return is the exact
  pullback to the public `q = [pos, quat_xyzw, joints]` layout of `f ∘ normalize` (the kernels
  evaluate `R(p/|p|)`); finite differences over every ambient `q` component agree, unit or not.
- `f_ext` is non-differentiated, but the `q`/`qd` gradients are taken at the
  force you pass.
- `f_ext` has shape `(B, 6*num_bodies)`, body-local; jax materializes `(6nb,)`/`(1, 6nb)`
  broadcasts, and anything else is rejected (natively too).

## Runtime-mutable model parameters (no recompile)

Two opt-in tables let you change the model after compile with no nvcc rebuild, on all
three backends: the jax/torch kernels read the same device-resident table the numpy mutator
writes, so one `set_*_params` call is seen by every surface over that `.so`:

| Build flag | Mutator | Table shape | Mutates |
|------------|---------|-------------|---------|
| `runtime_inertia=True` | `handle.set_inertia_params(t)` | `(num_bodies, 10)` rows `[m, hx,hy,hz, Ixx,Ixy,Ixz, Iyy,Iyz, Izz]` | spatial inertia in every dynamics call (id/fd/aba/crba/minv/gradients) |
| `runtime_transform=True` | `handle.set_transform_params(t)` | `(num_joints, 6)` rows `[x,y,z,roll,pitch,yaw]` (URDF `<origin>`) | each joint's `Xfixed` in the dynamics only; the baked end-effector kinematics keep the URDF geometry after a transform update (documented partial capability) |

Fetch the baked table from `handle.inertia_params` / `handle.transform_params`, mutate, set it
back. Passing the baked values back is byte-identical to a plain build. Each flag re-keys the
cache (the mutable `.so` coexists with the baked one). This is the entry point for
system identification, payload changes, domain randomization, and kinematic calibration; see
[`runtime_params.py`](runtime_params.py).

```python
h = grid_rbd.register_robot("arm", urdf, runtime_inertia=True)
I = h.inertia_params.copy()           # (num_bodies, 10)
I[-1, 0] += 0.5; I[-1, 1:4] *= ...    # +0.5 kg payload on the last link (scale h=m*c)
h.set_inertia_params(I)               # every later forward_dynamics uses it — no rebuild
```

## Named end-effector targets

The EE kernels target leaf links by default. To target a specific frame (tool flange, TCP,
sensor), select it by joint name. There are two routes; see [`ee_named_targets.py`](ee_named_targets.py).

- Baked (codegen, jittable, all surfaces): `register_robot(..., ee_joint_names=["tool_joint"])`.
  `ee_joint_names` is in the cache key, so distinct targets land in distinct entries. The named
  target flows through `end_effector_pose`, `_gradient` and `_hessian`.
- Runtime (all backends): one compiled robot, choose the frame and an offset point per call.
  `end_effector_pose_runtime(q, ee_joint_names=..., ee_offsets=...)` returns `(B, NUM_EE, 6)` and
  `end_effector_pose_gradient_runtime(...)` returns `(B, NUM_EE, 6, NV)`. `ee_joint_names` is `None`
  (all leaves) / a name / a list; `ee_offsets` is `None` (origin) / one `[x,y,z]` per frame. This
  suits OSC / task-space control where the target or offset changes online.

## Output conventions

Default is Pinocchio convention. For MuJoCo/MJX-native I/O (wxyz quat, global-linear free-joint
velocity) use the `.mujoco` view (`h.mujoco.forward_dynamics(...)`), `output_convention="mujoco"`,
or set `handle.output_convention`. It is floating-base only (fixed-base, the two coincide) and
a runtime setting (not in the cache key; same `.so`). The `.mujoco` view applies the convention
per-call and is thread-safe, so it can be mixed with pinocchio-convention calls on the same handle.
The value methods (id/fd/aba/crba/minv) and the derivative/second-order surfaces
(`inverse_dynamics_gradient` / `forward_dynamics_gradient` / `idsva_so` / `fdsva_so`) honor
mujoco mode; the mjx kernel twins serve them natively (floating, non-mimic robots built with
`enable_mujoco_kernels=True`, the default). A mimic/skew robot or an
`enable_mujoco_kernels=False` build has no twins and raises a clear error instead.

## Performance: block size depends on launch path and use case

GRiD picks a per-algorithm CUDA block size (threads-per-block) by autotuning. The optimal
block size is not a property of the kernel alone: it depends on the launch path and on how
the kernel is used. The C++/host autotune optimizes pipelined throughput; the
python bindings launch the same kernel through the jax/torch FFI path, and for the
batch-to-land use case (fire one batched launch of N, wait for all N to land, the control /
trajectory-opt use case) that path has a different optimum. On iiwa14 `fd` the host path is
fastest at 128 threads but the FFI path is fastest at ~768: the same kernel, ~1.6× apart.

- The bindings ship FFI-tuned defaults (the `ffi_bases` profile in
  `grid_assets/launch_configs/<robot>/<gpu>.json`, baked into the `.so`). The FFI-fast path applies
  on the robots/GPUs we tuned (per-algo fallback to the host pick for anything un-tuned).
- Run the autotune for your own robot / GPU / use case; there is no single "true" block size:
  ```bash
  python test/benchmarks/autotune_ffi.py --robot <robot> --base both   # writes ffi_bases, then rebuild the binding
  ```
- Override at runtime without rebuilding: `handle.set_threads_per_block(n)` forces `n` threads
  for every algo (e.g. to A/B a count, or tune for a non-default batch size). `-1` / default uses
  the baked per-algo config.
- A wrapper call is still a few µs of jax/XLA dispatch slower than the raw kernel even at the best
  block size. That dispatch cost is fixed per call, so stay GPU-resident and `jit`/`capture` to
  amortize it (see the recommended practice below). Threads fix the kernel regime; residency fixes the dispatch cost.

Per-algo introspection: `handle.kernel_max_threads(key)` returns the real compiled
`__launch_bounds__` ceiling for that algo's baked-tier kernel (keys = the autotune
short names: id, fd, minv, id_du, fd_du, ee_pose, idsva_so, ...; -1 = unknown/absent).
Batch-regime switching: `set_threads_for_n`/`apply_batch_overlay`/`get_batch_switch`
apply per-algo small-batch thread overrides from the config's `ffi_bases_by_n` block.


## Fusing GRiD into your own CUDA kernel

A consumer that calls the generated `_inner` functions from its own kernel
(a merit kernel running one forward dynamics and one cost per block) sizes its
block from the header, not from `MAX_PERF_LEVEL_THREADS`: every header ends
with `grid::<ALGO>_SUGGESTED_THREADS`, the smallest multiple of 32 covering the
algorithm's widest block-parallel compute loop, and the fused kernel takes the
max over the algorithms it calls (`TRACKING_COST_SUGGESTED_THREADS` for the
plant preset). The `s_topology_helpers` block the `_inner` functions take is
`grid::TOPOLOGY_HELPERS_COUNT` ints: zero only for a serial chain whose joints
all turn about +Z, `n` for any other serial chain and `5n + 1` more for a
branching tree (`n = max(NUM_POS, NUM_JOINTS)`); pass a filled block whenever
the count is nonzero, reserve `grid::TOPOLOGY_HELPERS_T_COUNT<T>()` T elements
for it in an arena sized in T, and generate with `DEBUG_MODE=True` to have
every helper-reading `_inner` assert the pointer is not null.
See `examples/cuda/wrapper_types.md`.

## Recommended practice

- `device_put` inputs once and keep outputs as device arrays/tensors across calls.
- Wrap multi-call logic in `@jax.jit` (or `capture()` for torch) so GRiD calls fuse /
  replay instead of dispatching one at a time.
- Set `max_batch_size` to the largest batch you will run, at build time.
- Match input dtype to the surface: fp32 for jax/torch (an fp64 array forces a copy).
- Do not convert to `np.asarray` / `.cpu()` between GRiD calls in a hot loop. That is a
  D2H+H2D round-trip per step and erases the GPU advantage (see the timed anti-pattern in
  `jax_gpu_resident.py`).
- Do not rebuild per run; `precompile` once and reuse the cache.
- When embedding the generated header directly in C++, use the nonterminating
  `init_robotModel_checked` / `init_joint_limits_checked` / `free_robotModel_checked`
  family (or `robotModel_owner<T>`): status-returning, rollback-safe, names the failed
  operation, never `exit()`s or resets the context. The un-suffixed `init_*()` keep the
  historical fail-fast policy. See docs: *Library-safe initialization and cleanup*.
- Do not size a batch above `max_batch` (chunk instead). The error names the compiled
  limit and the `max_batch_size=` knob that raises it.

## Common errors

| Symptom | Cause and remedy |
|---------|------------------|
| `RobotNotRegisteredError` from `get_robot` | the cache is not warm and no `urdf_path=` was given; pass `urdf_path=` (compile on miss), run `register_robot` / `precompile` first, or ship the cache dir |
| `StaleRobotError` from `get_robot` | the manifest names a build this `grid_rbd` cannot load (older wrapper / torch-jax ABI / GPU arch, or a pre-2026-09-22 entry with no build record); re-run `register_robot` / `precompile`; byte-identical sources re-hit the store without `nvcc` |
| "symbol missing" / "not built into this `.so`" | you used `algorithm_list=` and called a method outside the subset; add it and rebuild. Same on all three surfaces |
| torch "no kernel image available for sm_120" | the backward VJP runs torch's own CUDA kernels, so the installed torch wheel must support the GPU arch (e.g. RTX 5090 / sm_120 needs a cu128+ build; cu124 maxes at sm_90). The receipt ran a cu128 wheel on a CUDA 13.2 toolkit; other torch wheels are untested. The GRiD kernels themselves are always nvcc-built for the detected arch |
| torch `capture()` during stream capture | `capture()` runs a mandatory off-graph warmup for one-time device setup (the >48 KB dynamic-smem opt-in) before recording; do not call methods cold inside your own `torch.cuda.graph` context |
| big-robot SO builds are heavy (10s of GB RAM, minutes) | use `algorithm_list=` to build only what you need, and prebuild offline |

## Runnable examples in this folder
- [`quickstart_iiwa14.py`](quickstart_iiwa14.py): register + call every method (numpy).
- [`jax_gpu_resident.py`](jax_gpu_resident.py): residency, jit/vmap/grad, `lax.scan` rollout,
  donate, dlpack. The reference for the JAX fast path.
- [`jax_gpu_resident_go2.py`](jax_gpu_resident_go2.py): the floating-base twin (go2, nq=19/nv=18):
  same demos plus GRiD's own `integrator` kernel inside the scan for the on-manifold base retract.
- [`torch_cuda_graphs.py`](torch_cuda_graphs.py): CUDA tensors, autograd, CUDA-Graphs replay
  (the graph win is CPU-submit cost, ~13 us eager vs ~2 us replay; wall time is ~break-even).
- [`torch_cuda_graphs_go2.py`](torch_cuda_graphs_go2.py): the floating-base twin.
- [`derivatives.py`](derivatives.py): analytic first-order (`inverse/forward_dynamics_gradient`
  = id_du/fd_du, `end_effector_pose_gradient`) + second-order (`idsva_so`/`fdsva_so`) tensors,
  and the `jit`/`vmap`/`grad` autodiff idioms that pull the same Jacobians through a cost.
- [`runtime_params.py`](runtime_params.py): `set_inertia_params` (runtime_inertia) +
  `set_transform_params` (runtime_transform): sysID / payload / domain-rand / calibration,
  no recompile (numpy).
- [`ee_named_targets.py`](ee_named_targets.py): named EE frames: baked `ee_joint_names=`
  (value + gradient + hessian) and the runtime `end_effector_pose[_gradient]_runtime` frame/offset.
- [`tool_use.py`](tool_use.py): weld a rigid tool/payload at runtime (`attach_tool`/`detach_tool`),
  no recompile: payload inertia + a full SE(3) tool-tip frame, attach anywhere, gripping + closed-loop
  recipes, payload-hypothesis sweep.
- [`multi_contact_fext.py`](multi_contact_fext.py): multi-contact stance forces: register go2 with
  `contact_frames=[four feet]`, map per-foot world-aligned wrenches to joint-local `f_ext` in one
  `contact_fext` call, feed the result to `inverse_dynamics`/`forward_dynamics` as `f_ext=`.
- [`system_identification.py`](system_identification.py): least-squares inertial-parameter fit with
  the joint-torque regressor (`tau = Y·π`), verified in torque space, pushed back into the running
  model via `set_inertia_params`, plus the differentiable outer loop (`inverse_dynamics_wrt_params`
  under `jax.grad`).

## Welded tools / payloads (`attach_tool`, no recompile)

Register with `enable_tool=True` (turns on the runtime inertia table + runtime contact surface; the
runtime EE surfaces are already in the default build), then:

```python
h = grid_rbd.register_robot("arm", urdf_path=..., enable_tool=True)
# a rigid tool = payload inertia on a link + an SE(3) tip frame off that joint:
h.attach_tool("iiwa_joint_7", mass=2.0, com=[0,0,0.08],
              inertia=np.diag([0.02,0.02,0.008]),   # about the payload CoM (optional)
              tip_transform=X_tool)                  # 4x4 SE(3) in the joint frame (optional)
h.inverse_dynamics(q, qd)          # dynamics now carry the tool's weight
h.end_effector_pose_runtime(q)     # defaults to the SE(3) tool-tip frame
h.detach_tool()                    # restore the baked robot
```

- A tool attaches anywhere: the first arg is a joint name; the tool welds to that joint's child link and
  the tip hangs off that joint. Mid-chain payloads change upstream torques and leave the downstream
  tip untouched. Omit `tip_transform` for a pure carried payload (inertia only).
- To model gripping a tool, treat a firm grasp as a rigid attach on the palm/wrist; do not remove the
  finger DOFs (that would need a recompile). Hold the fingers with `qd=0` (optionally stiff via
  `runtime_joint_dynamics`). Only an articulated tool (one that adds a DOF) needs a rebuild.
- For tip forces (grinding, pushing, a second gripper finger), `h.tool_fext(q, wrench)` maps a
  world-aligned tool-tip wrench `(B, 6)` = `[n_w; f_w]` to a joint-local `f_ext` `(B, 6*num_bodies)`
  you pass straight to `inverse_dynamics(f_ext=...)` / `aba(f_ext=...)`. The `∂/∂f_c` and `∂/∂q`
  derivatives are validated at the device level (`grid::f_ext_body_jacobian_d{fc,q}_runtime_device`)
  for solvers that need the chain-rule term.
- A two-finger closed-loop grasp, a tool bridging two fingertips, is a closed kinematic loop (not
  representable in a tree). Attach to one fingertip and model the other finger's grip as a `tool_fext`
  wrench on the tool tip; the result is an open tree with no closed loop.
- Requires `enable_tool=True`; one tool at a time. Payload composition is additive and
  pinocchio-validated; a null tool is byte-identical to the baked robot.
