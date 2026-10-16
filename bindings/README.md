# grid-rbd

Python wrappers for GPU-accelerated rigid body dynamics on top of
[GRiD](https://github.com/A2R-Lab/GRiD). The wrappers have two tiers: a
one-time registration that builds the robot, and batched calls that run many
times.

```python
import grid_rbd

# One-time per (robot, options, GRiD version, CUDA arch) — build times are
# tabulated on the Fast Robot Setup page. Generates grid.cuh, compiles to
# .so, caches under ~/.cache/grid-rbd/.
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

The full reference (gravity convention, cache layout, EE-target selection,
JAX FFI) lives in the
[main docs](https://a2r-lab.org/GRiD/).

## Status: 0.5.1 (on PyPI as `grid-rbd`)

The full method surface is bound and validated against
[`RBDReference`](https://github.com/A2R-Lab/RBDReference) at float32 precision,
with operation-specific backend coverage. The surface comprises the core
dynamics/kinematics, analytical gradients, the second-order derivatives, the
centroidal / energy / general-frame value ops (`com`, `ccrba`, `dccrba`,
`cmm_time_variation`, `coriolis_matrix`, `energy`, `generalized_gravity`,
`nonlinear_effects`, the KE/PE regressors, `frame_jacobian`/`_dot`,
`osc_inertia`, and the runtime-target EE pose / gradient), the
inertial-parameter (π) regressor family (`inverse_dynamics_regressor`, the
differentiable `inverse_dynamics_wrt_params` / `forward_dynamics_wrt_params`,
and `forward_dynamics_parameter_gradient` = ∂q̈/∂π), runtime welded tools
(`attach_tool` / `tool_fext`), and multi-contact wrench mapping
(`contact_fext` / `register_robot(contact_frames=...)`). `*_wrt_params` and
`forward_dynamics_parameter_gradient` are JAX/PyTorch interfaces, not NumPy
handle methods. See the [backend inventory](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/backend_coverage.html).
The per-method table (shapes, arguments, conventions) lives in the
[Python wrappers docs](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/python_wrappers.html)
(`docs/source/user_guide/tutorials/python_wrappers.rst` in-repo).

Shape legend: `NJ = num_joints (== num_pos == nq)`, `NV = num_vel (tangent /
velocity space)`, `NB = num_bodies`. Every handle also exposes the read-only
aliases `h.nq` / `h.nv` / `h.nb`. `q` is `NJ`-wide; `qd`, `qdd`, `u`,
the value outputs (`c`, `qdd`) and every matrix/Jacobian axis are `NV`-wide
(tangent space, Pinocchio / MuJoCo convention) on every surface. For a
fixed base without spherical joints, `NV == NJ`; each independent spherical
joint adds one quaternion coordinate beyond its three tangent coordinates.
The per-method differentiability table (VJP inputs, fixed-base
restrictions, output representations) lives in the Python wrappers docs and is
cross-checked against the ABI specification by a CPU test.

### Floating-base shapes since v0.5

On floating-base robots, `crba`/`minv` return `(B, NV, NV)` and
`inverse_dynamics_gradient`/`forward_dynamics_gradient` return `(B, NV, 2*NV)`
instead of the `NJ`-sized shapes of earlier releases. The CUDA kernels have
always written these as `NV`-dimensioned (`NUM_VEL*NUM_VEL`); the old binding
over-sized the copy-out to `NUM_JOINTS*NUM_JOINTS`, which appended garbage
padding rows/cols and corrupted every matrix after the first for `batch > 1`
through a per-timestep stride mismatch (324 written vs 361 read). The new
shapes match `RBDReference` / pinocchio's `nv`-space mass matrix and Jacobians
and remove the binding-side `batch > 1` corruption. Fixed-base robots are
unaffected (`NV == NJ`). The JAX-FFI and torch backends expose the same
corrected `NV`-sized floating shapes.

The floating-base CRBA kernel folds composite inertias in a deterministic
parent-major fixed order (`grid_codegen/algorithms/_crba.py`, ~L571), so an
identical-`q` batch returns the same mass matrix in every slot. On go2-floating
a 64-wide identical-`q` batch returns bit-identical `M` across all slots
(`max |M[b]−M[0]| = 0`) and is bit-identical run-to-run.

## JAX FFI (`grid_rbd[jax]`)

The `[jax]` extra (see [Install](#install)) enables the JAX-side bridge, which
shares the same per-robot `.so` cache. Methods are exposed via
`jax.ffi.ffi_call` and run device-resident on JAX-supplied CUDA streams with no
host round-trip, so they slot directly into `jax.jit` graphs:

```python
import grid_rbd.jax as grid_jax, jax
handle = grid_jax.register_robot(name="iiwa14", urdf_path="iiwa.urdf")

@jax.jit
def step(q, qd, u):
    return handle.forward_dynamics(q, qd, u)
```

See the [Python wrappers docs](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/python_wrappers.html)
for the JAX surface details (autograd-aware methods, `f_ext` parity, the
π-regressor VJP path, and the forward-only value ops).

## PyTorch backend (`backend="torch"`)

The `[torch]` extra (see [Install](#install)) enables the torch backend.
`register_robot(..., backend="torch")` returns a `TorchRobotHandle` whose
methods return `torch.Tensor`; `inverse_dynamics` / `forward_dynamics` / `aba`
/ `integrator` are autograd-aware (analytic backward passes), and CUDA-Graphs
capture is available via `h.capture(...)`. The `.so` is shared with the
numpy/JAX surfaces (same content-addressed cache):

```python
import grid_rbd, torch

h = grid_rbd.register_robot("iiwa14", urdf_path="iiwa.urdf", backend="torch")

q  = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)
qd = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)
u  = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)

qdd = h.forward_dynamics(q, qd, u)   # autograd-aware torch.Tensor
qdd.sum().backward()                 # gradients flow to q, qd, u
```

The backward VJP contractions run torch's own CUDA kernels, so the installed
torch build must support the GPU's compute capability. On an RTX 5090 (sm_120)
a torch cu128 (or newer) build is required; a cu124 wheel (max sm_90) cannot
launch on sm_120. The `grid` / `grid_plant` kernels themselves are always
nvcc-built for the detected arch and are unaffected. The release receipt ran a
cu128 torch wheel on a CUDA 13.2 toolkit; other torch wheels are untested.

## Further reference

The full reference for the rest of the surface lives in the
[Python wrappers docs](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/python_wrappers.html)
(`docs/source/user_guide/tutorials/python_wrappers.rst` in-repo):

* the per-method table (shapes, `qdd=`-aware gradients, `ee_joint_names`,
  `allow_fp64`),
* the π-regressor family (`inverse_dynamics_regressor`,
  `inverse_dynamics_wrt_params` / `forward_dynamics_wrt_params`,
  `forward_dynamics_parameter_gradient`),
* welded tools (`attach_tool` / `tool_fext`, via `enable_tool=True`) and
  multi-contact (`contact_fext`, via `register_robot(contact_frames=...)`),
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

## Install

`grid_rbd` ships inside the `grid-rbd` distribution on PyPI (0.5.1: Linux
x86-64 wheels and an sdist, CPython 3.10–3.13) together with the code
generator and the `grid-generate` / `grid-spherize` CLIs. Each GPU backend is
an opt-in extra on top of the numpy base; pick the row for the wrapper surface
you need:

```bash
pip install grid-rbd            # base: numpy backend only
pip install "grid-rbd[jax]"     # + JAX FFI surface (grid_rbd.jax)
pip install "grid-rbd[torch]"   # + torch backend (backend="torch")
pip install "grid-rbd[all]"     # jax + torch (both backends)
```

To develop GRiD (or to use the checkout's vendored robots under
`grid_assets/robots/`), install editable from a checkout instead, with the same
extras plus `[dev]` for the bindings' own test suite:

```bash
cd path/to/GRiD
pip install -e ".[all]"     # or "." / ".[jax]" / ".[torch]"
pip install -e ".[dev]"     # all backends + pytest (run the bindings' tests)
```

| Extra | Pulls in | Unlocks |
|---|---|---|
| *(base)* | numpy, platformdirs | numpy handle (`register_robot(..., backend="numpy")`) + `grid_plant` |
| `[jax]` | + jax | JAX FFI surface: `import grid_rbd.jax` (device-resident, `jax.jit`-able) |
| `[torch]` | + torch | torch backend: `register_robot(..., backend="torch")`, autograd + CUDA-Graphs |
| `[all]` | jax + torch | both backend surfaces (recursive self-extra; no dev/bench weight) |
| `[dev]` | jax + torch + pytest | run the bindings' own test suite (which exercises both backends; the real-MuJoCo cross-check is optional/skipped if `mujoco` is absent) |

Three points on the extras:

* `nvcc` is needed at `register_robot()` / `precompile()` time, not at
  `pip install` time. pip only stages Python deps; the per-robot `.so` is
  built (and cached) on first use.
* The GPU wheels are the user's choice, mirroring `jax` vs `jax[cuda]`: we
  pin only the minimum API version, never a CUDA-variant. For `[jax]`, install
  `jax[cuda12]` for your platform (what the receipt ran on a CUDA 13.2 toolkit;
  `jax[cuda13]` is untested); for `[torch]`, a torch CUDA wheel matching the
  GPU arch (see the compatibility note in the PyTorch section above).
* `[dev]` is `[all]` + pytest, nothing more. The Pinocchio oracle (`pin<4`
  via cmeel) comes only from `install/requirements-dev.txt`, which
  `install/developer_install.sh` installs; that script also installs
  `jax[cuda12]`, a cu128 torch wheel and `frax`, and builds CppADCodeGen for
  the Pinocchio benchmark column. No script installs mjx or BARD; add them by
  hand for those benchmark columns (`test/benchmarks/README.md`).

### Precompile (AOT cache-warming)

`register_robot()` compiles the per-robot `.so` on first use; subsequent runs
are instant cache hits. To do that build offline, for example in a Docker
image build or CI step so that production never pays the one-time nvcc cost,
call `grid_rbd.precompile()`:

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
The `numpy` backend is warmed without constructing a handle (no dlopen, no CUDA
context), so a build box with `nvcc` but no usable GPU can populate the cache.
Pass `cuda_arch=` (e.g. `120`) on such a box, since there is no `nvidia-smi` to
ask; only the `jax` / `torch` backends need the device at warm time. To see
what a registration would key on and whether the cache already holds it,
without building anything, call `grid_rbd.build_plan(name, urdf_path,
cuda_arch=...)`. Build the cache offline, ship or keep the cache dir, and
every later `register_robot` / `get_robot` / `jax.jit` starts in well under a
second.

Registration failures (a failed device allocation, a bad toolkit) raise a Python
exception naming the failed operation; the generated initializers never
`exit()` or reset the CUDA context inside the interpreter (they use the
library-safe `*_checked` path; see the *Library-safe initialization* page). On a
successful build the `build.log` path is reported too (not only on failure), so
nvcc/ptxas output can be inspected for warnings.

A first use announces itself: a one-line `WARNING` on the `grid_rbd` logger
(printed to stderr with no logging setup) says which robot is being compiled,
how many units on how many cores, roughly how long (1–3 min for an arm or
quadruped, ~10 min for a humanoid) and where it is cached; a second line reports
the elapsed time. `GRID_RBD_QUIET=1` silences it. `get_robot(name, urdf_path=...)`
compiles and registers on a *miss* (a hit ignores the URDF), so a script can be
written as "get or build". `python -m grid_rbd.gc_cache` reports store entries
no name binds any more (plus dangling pointers and dead staging dirs); `--execute`
deletes them; per-unit objects are kept unless `--objects-days N`.

The build itself is multi-unit: the wrapper is split into a core unit plus one
unit per algorithm family, compiled with `nvcc -c` in parallel and host-linked
into `robot.so` (wall time ≈ the slowest family, not the sum; a humanoid unit
peaks at ~2.5 GB). `GRID_RBD_BUILD_JOBS=N` caps the parallel units (default
`min(8, cores, MemAvailable / 3 GB)`; `1` = sequential, still a smaller peak
than a single unit). Per-unit objects are cached under
`<cache_dir>/objects/` keyed by their exact compile inputs, so a later build
whose `grid.cuh` is unchanged recompiles only the units whose text changed;
`force_rebuild=True` recompiles every unit. Objects are kept forever, like the
`store/` entries; set `GRID_RBD_OBJECT_CACHE_DAYS=N` to prune objects not built
or linked from in N days after each successful build. The per-unit `rc`/seconds/cache hits are in `robot.build.log`.
Two more knobs: `GRID_RBD_CACHE_DIR` overrides the cache root (default: the
platformdirs user cache; `register_robot(cache_dir=...)` wins over the env), and
`GRID_RBD_SPLIT_COMPILE=N` (opt-in, default off) passes nvcc `--split-compile N`
on the heaviest unit(s) (`GRID_RBD_SPLIT_COMPILE_UNITS`, default
`integrator_gradient`) for ~3x faster compiles of that unit at ~3x its RAM; it
changes ptxas register allocation (outputs were bitwise identical in our A/B,
kernel time within noise), is folded into the cache key, and is never used for
the test runners.

`GRID_RBD_CUDA_ARCHS` (opt-in, default unset) builds a fat binary: a comma list
of compute capabilities as integers (`86,89,90,120`) adds one
`-gencode=arch=compute_X,code=sm_X` per listed arch, so one cache store, or the
GHCR Docker image, carries a `robot.so` that loads on several GPUs. The arch the
build was made for (`cuda_arch=`, or the detected GPU) is always embedded. The
sorted list is folded into both cache keys and recorded in the build identity
and in `meta.json` as `cuda_archs`, beside `cuda_arch`, so a fat build never
collides with a single-arch build of the same inputs, and `get_robot` accepts a
stored entry whose `cuda_archs` contains the GPU it finds. Launch configs are
selected per arch at codegen time: a fat binary carries the configs of its
`cuda_arch`, and the other arches run the same kernels with that tuning
(the kernels are thread-count invariant, so results are identical; throughput on
those arches may differ from an arch-native build). Precompiling on a machine
without a GPU works with an explicit `cuda_arch=` (`grid_rbd.precompile(name,
urdf, cuda_arch=120)`): nvcc is needed, `nvidia-smi` is not. Unset, the build
command, keys and meta are unchanged.
