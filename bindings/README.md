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

## Status — v0.5

The full method surface — core dynamics/kinematics, analytical gradients, the
second-order derivatives, and the centroidal / energy / general-frame value ops
(`com`, `ccrba`, `dccrba`, `cmm_time_variation`, `coriolis_matrix`, `energy`,
`generalized_gravity`, `nonlinear_effects`, the KE/PE regressors,
`frame_jacobian`/`_dot`, `osc_inertia`, and the runtime-target EE pose /
gradient) — is bound and validated against
[`RBDReference`](https://github.com/A2R-Lab/RBDReference) at float32 precision,
and is available on **all three backends** (numpy / jax / torch; the value ops
are forward-only on jax/torch). The per-method table (shapes, arguments,
conventions) lives in the
[Python wrappers docs](https://a2r-lab.github.io/GRiD/user_guide/tutorials/python_wrappers.html)
(`docs/source/user_guide/tutorials/python_wrappers.rst` in-repo).

Shape legend: `NJ = num_joints (== num_pos == nq)`, `NV = num_vel (tangent /
velocity space)`. Inputs `q`, `qd`, `qdd`, `u` and the value outputs (`c`, `qdd`)
are `NJ`-wide; matrix/Jacobian outputs are tangent-space (pinocchio convention)
and `NV`-dimensioned. For a **FIXED base `NV == NJ`**.

> **Breaking change (v0.5) — floating-base only.** `crba`/`minv` now return
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
> Previously-noted floating-base CRBA batch issue — **RESOLVED**. The floating-base
> CRBA kernel used to return inconsistent mass matrices across `batch > 1` slots for
> an identical-`q` batch (a warp-scheduling-order shared-parent `atomicAdd` in the
> composite-inertia fold). Fixed in codegen by a deterministic parent-major
> fixed-order reduction (`grid_codegen/algorithms/_crba.py`, ~L571). Verified
> 2026-07-24 on go2-floating: a 64-wide identical-`q` batch returns bit-identical `M`
> across all slots (`max |M[b]−M[0]| = 0`) and is bit-identical run-to-run.

## JAX FFI (`grid_rbd[jax]`)

The `[jax]` extra (see the [install matrix](#install-editable-from-a-grid-checkout))
enables the JAX-side bridge, which shares the same per-robot `.so` cache.
Methods are exposed via `jax.ffi.ffi_call` and run device-resident on
JAX-supplied CUDA streams — no host round-trip — so they slot directly into
`jax.jit` graphs:

```python
import grid_rbd.jax as grid_jax, jax
handle = grid_jax.register_robot(name="iiwa14", urdf_path="iiwa.urdf")

@jax.jit
def step(q, qd, u):
    return handle.forward_dynamics(q, qd, u)
```

See the [Python wrappers docs](https://a2r-lab.github.io/GRiD/user_guide/tutorials/python_wrappers.html)
for the JAX surface details (autograd-aware methods, `f_ext` parity, the
π-regressor VJP path, and the forward-only value ops).

## PyTorch backend (`backend="torch"`)

The `[torch]` extra (see the [install matrix](#install-editable-from-a-grid-checkout))
enables the torch backend. `register_robot(..., backend="torch")` returns a
`TorchRobotHandle` whose methods return `torch.Tensor`; `inverse_dynamics` /
`forward_dynamics` / `aba` / `integrator` are autograd-aware (analytic backward
passes), and CUDA-Graphs capture is available via `h.capture(...)`. The `.so`
is shared with the numpy/JAX surfaces (same content-addressed cache):

```python
import grid_rbd, torch

h = grid_rbd.register_robot("iiwa14", urdf_path="iiwa.urdf", backend="torch")

q  = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)
qd = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)
u  = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)

qdd = h.forward_dynamics(q, qd, u)   # autograd-aware torch.Tensor
qdd.sum().backward()                 # gradients flow to q, qd, u
```

> **GPU/torch compatibility:** the backward VJP contractions run torch's
> own CUDA kernels, so the installed torch build must support the GPU's
> compute capability. On an RTX 5090 (sm_120) you need a torch **cu128**
> (or newer) build — a cu124 wheel (max sm_90) cannot launch on sm_120.
> The `grid` / `grid_plant` kernels themselves are always nvcc-built for
> the detected arch and are unaffected.

## Everything else — see the docs

The full reference for the rest of the surface lives in the
[Python wrappers docs](https://a2r-lab.github.io/GRiD/user_guide/tutorials/python_wrappers.html)
(`docs/source/user_guide/tutorials/python_wrappers.rst` in-repo):

* the per-method table (shapes, `qdd=`-aware gradients, `ee_joint_names`,
  `allow_fp64`),
* the `grid_plant` cost / barrier / plant-step methods,
* per-body external forces (`f_ext=`),
* build cost on big floating-base robots and the `enable_mujoco_kernels`
  flag (pin-only builds; the flag enters the `.so` cache key only when
  `False`, so existing caches stay valid),
* the cache layout and runtime-mutable model parameters.

## Requirements

* Python ≥ 3.10
* CUDA Toolkit (`nvcc` on PATH) at `register_robot` time. Not needed
  for `pip install grid-rbd` itself.
* numpy ≥ 1.23

## Install (editable, from a GRiD checkout)

`grid_rbd` ships as part of the single repo distribution, so `pip install -e .`
installs the codegen toolkit and the wrapper together; each GPU backend is an
opt-in extra on top of the numpy base. Pick the row for the wrapper surface you want:

```bash
cd path/to/GRiD
pip install -e "."          # base: numpy backend only
pip install -e ".[jax]"     # + JAX FFI surface (grid_rbd.jax)
pip install -e ".[torch]"   # + torch backend (backend="torch")
pip install -e ".[all]"     # jax + torch (both backends)
pip install -e ".[dev]"     # all backends + pytest (run the bindings' tests)
```

| Extra | Pulls in | Unlocks |
|---|---|---|
| *(base)* | numpy, platformdirs | numpy handle (`register_robot(..., backend="numpy")`) + `grid_plant` |
| `[jax]` | + jax | JAX FFI surface — `import grid_rbd.jax` (device-resident, `jax.jit`-able) |
| `[torch]` | + torch | torch backend — `register_robot(..., backend="torch")`, autograd + CUDA-Graphs |
| `[all]` | jax + torch | both backend surfaces (recursive self-extra; no dev/bench weight) |
| `[dev]` | jax + torch + pytest | run the bindings' own test suite (which exercises both backends; the real-MuJoCo cross-check is optional/skipped if `mujoco` is absent) |

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
  in any extra here — that's a developer concern carried by
  `install/requirements-dev.txt` / `install/developer_install.sh`.

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
