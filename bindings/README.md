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
| `inverse_dynamics(q, qd, qdd=None, gravity=-9.81)` | `(B, NJ)` | 4.9e-6 |
| `minv(q)` | `(B, NV, NV)` | 1.1e-4 |
| `forward_dynamics(q, qd, u, gravity=-9.81)` | `(B, NJ)` | 5.7e-5 |
| `aba(q, qd, u, gravity=-9.81)` | `(B, NJ)` | 5.7e-5 |
| `crba(q, gravity=-9.81)` | `(B, NV, NV)` | 2.7e-7 |
| `end_effector_pose(q)` | `(B, 6*NUM_EES)` | 1.4e-7 |
| `end_effector_pose_gradient(q)` | `(B, 6*NUM_EES, NV)` | 3.1e-7 |
| `end_effector_pose_hessian(q)` | `(B, 6*NUM_EES, NV, NV)` | 3.1e-7 |
| `inverse_dynamics_gradient(q, qd, qdd=None, gravity=-9.81)` | `(B, NV, 2*NV)` | 1.6e-5 |
| `forward_dynamics_gradient(q, qd, u, gravity=-9.81)` | `(B, NV, 2*NV)` | 1.3e-4 |
| `idsva_so(q, qd, qdd, gravity=-9.81)` | tuple of 4 × `(B, NV, NV, NV)` | 1e-4 |
| `fdsva_so(q, qd, u, gravity=-9.81)` | tuple of 4 × `(B, NV, NV, NV)` | 1e-4 |

Shape legend: `NJ = num_joints (== num_pos == nq)`, `NV = num_vel (tangent /
velocity space)`. Inputs `q`, `qd`, `qdd`, `u` and the value outputs (`c`, `qdd`)
are `NJ`-wide (GRiD's kernels consume velocity vectors at the nq stride; for a
floating base the base 6-dof velocity sits in the leading slots with a padded
quaternion-offset slot). Matrix/Jacobian outputs are tangent-space (pinocchio
convention) and `NV`-dimensioned. For a **FIXED base `NV == NJ`**, so every shape
above is identical to the pre-v0.4.1 behaviour.

> **Breaking change (v0.4.1) — floating-base only.** `crba`/`minv` now return
> `(B, NV, NV)` and `inverse_dynamics_gradient`/`forward_dynamics_gradient` return
> `(B, NV, 2*NV)` instead of the previous `NJ`-sized shapes. The CUDA kernels have
> always written these as `NV`-dimensioned (`NUM_VEL*NUM_VEL`); the old binding
> over-sized the copy-out to `NUM_JOINTS*NUM_JOINTS`, which (a) appended garbage
> padding rows/cols and (b) **corrupted every matrix after the first for
> `batch > 1`** via a per-timestep stride mismatch (324 written vs 361 read). The
> new shapes match `RBDReference` / pinocchio's `nv`-space mass matrix and
> Jacobians and fix the binding-side `batch > 1` corruption. Fixed-base robots
> are unaffected (`NV == NJ`). NOTE: the JAX-FFI and torch backends still expose
> the old `NJ`-sized floating shapes pending a coordinated autodiff-side
> migration; the numpy `register_robot(...)` handle is the corrected surface.
>
> Known separate issue (NOT a binding bug; not fixed here): the floating-base
> **CRBA kernel** itself returns inconsistent mass matrices for `batch > 1` (an
> identical-`q` batch yields differing `M` across batch slots). `minv` and the
> dynamics gradients are correct batched; standalone (`batch == 1`) `crba` matches
> the reference. This lives in the generated CUDA kernel (codegen), not the
> binding, and needs a kernel-side fix.

`register_robot` accepts `ee_joint_names=[...]` to pin specific
end-effector frames (default: all leaf links), and `allow_fp64=True` for an
fp64-in/fp64-out convenience cast (compute stays fp32).

`inverse_dynamics` (alias `rnea`) and `forward_dynamics` (alias `fd`) take an
optional `qdd=`: for `inverse_dynamics`, `qdd=None` ⇒ the bias `c = h − g` and a
nonzero `qdd` adds the `M·qdd` term. The torch/JAX gradient is qdd-aware (it
returns the correct ∂τ/∂(q,q̇) including the ∂(M·q̈)/∂q term, not the qdd=0
Jacobian). The second-order tuples are returned as `SecondOrderID` /
`SecondOrderFD` NamedTuples (plain positional tuples with named fields).

### Centroidal / energy / general-frame kinematics

Convenience compositions over the same surface (numpy handle only — not yet on
the JAX/torch backends), validated against the `RBDReference` centroidal /
energy / frame mixins:

| Method | Returns | max_err vs RBDReference |
|---|---|---|
| `com(q)` | `(p_com (B,3), J_com (B,3,NV))` | 1e-4 |
| `ccrba(q, qd)` | `(A (B,6,NV), h (B,6))` | 1e-4 |
| `energy(q, qd, gravity=-9.81)` | `(B, 3)` = `[KE, PE, KE+PE]` | 1e-4 |
| `generalized_gravity(q, gravity=-9.81)` | `(B, NV)` | 1e-5 |
| `nonlinear_effects(q, qd, gravity=-9.81)` | `(B, NV)` | 1e-5 |
| `coriolis_matrix(q, qd)` | `(B, NV, NV)` `C(q,q̇)` (`C·q̇ + g = nle`) | 1e-4 |
| `kinetic_energy_regressor(q, qd)` | `(B, 10*NB)` `y_KE` (`KE = y_KE·π`) | 1e-4 |
| `potential_energy_regressor(q, gravity=-9.81)` | `(B, 10*NB)` `y_PE` (`PE = y_PE·π`) | 1e-4 |
| `dccrba(q)` | `(B, 6, NV, NV)` ∂A/∂q tensor | 1e-4 |
| `cmm_time_variation(q, qd)` | `(B, 6, NV)` Ȧ | 1e-4 |
| `frame_jacobian(q, target_jid=None, reference_frame=None)` | `(B, 6, NV)` `[lin; ang]` | 1e-5 |
| `frame_jacobian_dot(q, qd, target_jid=None, reference_frame=None)` | `(B, 6, NV)` | 1e-3 |
| `osc_inertia(q)` | `(B, 6, 6)` task inertia Λ | 1e-3 |
| `end_effector_pose_runtime(q, ee_joint_names=None, ee_offsets=None)` | `(B, 6*NUM_EES)` | 1e-5 |
| `end_effector_pose_gradient_runtime(q, ee_joint_names=None, ee_offsets=None)` | `(B, 6*NUM_EES, NV)` | 1e-5 |

`dccrba` / `cmm_time_variation` run on mimic robots and on big floating-base
robots (sweep-pool spill); they raise a clear `RuntimeError` only on the rare
oversized-centroidal-pool case. `end_effector_pose_runtime` /
`..._gradient_runtime` take the target joint(s) + per-target offset at RUNTIME,
so one compiled robot serves any leaf/target frame.

`frame_jacobian` / `frame_jacobian_dot` take the target frame at RUNTIME:
`target_jid` selects the frame's joint id (default: the leaf end-effector joint)
and `reference_frame` is `LOCAL` (0) / `WORLD` (1) / `LOCAL_WORLD_ALIGNED` (2,
the default) — passed as the string or the int. `osc_inertia` still targets the
codegen-baked leaf-EE / LWA frame.

## JAX FFI (`grid_rbd[jax]`)

The `[jax]` extra (see the [install matrix](#install-editable-from-a-grid-checkout))
enables the JAX-side bridge, which shares
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

The core dynamics / kinematics / SO methods are bound via FFI, with
autograd-aware `inverse_dynamics` / `forward_dynamics` (qdd-aware),
`end_effector_pose`, `f_ext` parity, and the inertial-parameter (π) regressor
VJP path. The newer value ops (`coriolis_matrix`, the energy regressors,
`dccrba` / `cmm_time_variation`) are on the numpy handle only so far.

## PyTorch backend (`backend="torch"`)

The `[torch]` extra (see the [install matrix](#install-editable-from-a-grid-checkout))
enables the torch backend. `register_robot(..., backend="torch")` returns a `TorchRobotHandle`
whose methods return `torch.Tensor`. The four differentiable algorithms
(`inverse_dynamics` / `forward_dynamics` / `aba` / `integrator`) are autograd-aware,
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
| `com_cost(q, p_des, W)` | `value (B,)`, `grad (B, NX)`, GN `hess (B, NX, NX)` (CoM tracking) |
| `momentum_cost(q, qd, h_des, W)` | `value (B,)`, `grad (B, NX)`, GN `hess (B, NX, NX)` (centroidal-momentum tracking) |
| `joint_position_barrier(var, lower, upper, mu)` | `value (B,)`, `grad (B, NP)`, `hess_diag (B, NP)` |
| `joint_velocity_barrier(var, lower, upper, mu)` | as above over `NV` |
| `joint_torque_barrier(var, lower, upper, mu)` | as above over `NV` |
| `plant_step(x, u, dt, integrator_type="euler")` | `(B, NX)` next state |
| `plant_step_gradient(x, u, dt, integrator_type="euler")` | `(B, 2*NV, 3*NV)` `[A\|B]` = `d x_{k+1}/d(x,u)` |

## External forces (`f_ext`)

Per-body external forces are an opt-in feature of the underlying CUDA
codegen and the `RBDReference` oracle (body-local frame, subtracted from
the per-body force; an empty/`None` value reproduces the no-force path).
An `f_ext=` kwarg (shape `(B, 6*num_bodies)`, body-major) is exposed on the
`RobotHandle` algorithm methods that support it — `inverse_dynamics` /
`forward_dynamics` / `aba` and the inverse-/forward-dynamics gradients — on
both the numpy and JAX surfaces.

## Requirements

* Python ≥ 3.10
* CUDA Toolkit (`nvcc` on PATH) at `register_robot` time. Not needed
  for `pip install grid-rbd` itself.
* numpy ≥ 1.23

## Install (editable, from a GRiD checkout)

The base install is deliberately minimal (numpy + platformdirs); each backend
is an opt-in extra. Pick the row for the wrapper surface you want:

```bash
cd path/to/GRiD
pip install -e "bindings/"          # base: numpy handle only
pip install -e "bindings/[jax]"     # + JAX FFI surface (grid_rbd.jax)
pip install -e "bindings/[torch]"   # + torch backend (backend="torch")
pip install -e "bindings/[all]"     # jax + torch (both backends)
pip install -e "bindings/[dev]"     # + pytest (run the bindings' tests)
```

| Extra | Pulls in | Unlocks |
|---|---|---|
| *(base)* | numpy, platformdirs | numpy handle (`register_robot(..., backend="numpy")`) + `grid_plant` |
| `[jax]` | + jax | JAX FFI surface — `import grid_rbd.jax` (device-resident, `jax.jit`-able) |
| `[torch]` | + torch | torch backend — `register_robot(..., backend="torch")`, autograd + CUDA-Graphs |
| `[all]` | jax + torch | both backend surfaces (recursive self-extra; no dev/bench weight) |
| `[dev]` | + pytest | run the bindings' own test suite |

Notes:

* **`nvcc` is needed at `register_robot()` / `precompile()` time, not at
  `pip install` time** — pip only stages Python deps; the per-robot `.so` is
  built (and cached) on first use.
* **The GPU wheels are the user's choice**, mirroring `jax` vs `jax[cuda]`: we
  pin only the minimum API version, never a CUDA-variant. For `[torch]` you must
  install a torch **CUDA wheel matching the GPU arch** (e.g. a **cu128** build
  for sm_120 / RTX 5090 — a cu124 wheel maxes at sm_90 and cannot launch on
  sm_120). For `[jax]`, install `jax[cuda12]` for your platform.
* The heavy comparator/oracle stack (Pinocchio / mjx / frax / bard) is **not**
  in any extra here — that's a developer concern carried by the repo-root
  `requirements-dev.txt` / `developer_install.sh`.

A PyPI release will follow once the surface is feature-complete.

### Precompile (AOT cache-warming)

`register_robot()` compiles the per-robot `.so` on first use; subsequent runs
are instant cache hits. To do that build offline — e.g. in a Docker image build
or CI step, so production never pays the one-time nvcc cost — call
`grid_rbd.precompile()`:

```python
import grid_rbd

# Build + cache one or more tiers ahead of time. Each tier dict overrides the
# codegen-affecting options; warm whichever backend surfaces you ship.
grid_rbd.precompile(
    name="iiwa14",
    urdf_path="iiwa.urdf",
    tiers=[{}, {"floating_base": True}],   # fixed- and floating-base .so
    backends=("numpy", "jax", "torch"),
    max_batch_size=256,
)
```

A tier already in the cache is a no-op (no nvcc); a missing tier compiles once.
Build the cache offline, ship/keep the cache dir, and every later
`register_robot` / `get_robot` / `jax.jit` starts in well under a second. On a
successful build the `build.log` path is reported too (not just on failure), so
you can inspect nvcc/ptxas output for warnings.
