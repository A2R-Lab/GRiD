# Python Wrappers Plan

> **Status (2026-05-18):** Design locked in. Implementation scheduled after
> the GLASS TRAILING_SYNC rollout lands. This doc captures the design so
> any agent picking up the implementation has the full context without
> re-debating the architecture.

## Goal

Replace the deprecated `bindings/` directory with two new Python-side
surfaces that share a single compile + cache pipeline:

1. **`grid-rbd`** — a standard PyPI package, nanobind-based, for direct
   Python use.
2. **`grid-rbd[jax]`** (optional extra) — a JAX FFI integration that
   reuses the same per-robot compiled `.so`.

Both ship in the same source tree to keep the codegen + compile + cache
logic in one place.

## User experience

Two phases:

**Registration** (slow, one-time per (robot, options, GRiD version, CUDA
arch)):

```python
import grid_rbd
handle = grid_rbd.register_robot(
    name="iiwa14",
    urdf_path="path/to/iiwa.urdf",
    floating_base=False,
)
# Returns a RobotHandle after generating grid.cuh, compiling to
# ~/.cache/grid-rbd/store/<key>/robot.so, and registering name → key in
# the manifest. Idempotent — cache hit re-uses prior .so.
```

**Call** (fast, every-frame). All algorithm methods take and return
**2D arrays** where axis 0 is the batch dimension. Single-call mode is
not exposed via the wrapper — the single-call path is bench-only and
batch=1 covers it functionally with negligible overhead.

```python
qdd = handle.forward_dynamics(q, qd, u)        # (B, NV) -> (B, NV)
M   = handle.crba(q)                           # (B, NV) -> (B, NV, NV)
```

If a robot isn't registered yet, `get_robot(name)` raises
`RobotNotRegisteredError` with a message pointing at `register_robot`.

## Architecture: one package, two backends

```
grid-rbd/                        # ONE PyPI package
├── grid_rbd/
│   ├── __init__.py             # public: register_robot, get_robot, list_registered
│   ├── _core/                  # shared C++ layer (nanobind)
│   │   ├── runner.cpp          # dlopen + dispatch to robot.so symbols
│   │   ├── codegen.py          # wraps GRiDCodeGenerator.gen_all_code()
│   │   └── compile.py          # invokes nvcc on the generated header
│   ├── _cache.py               # ~/.cache/grid-rbd manifest + key hashing
│   ├── _cli.py                 # `grid-rbd register urdf.xml --name iiwa` etc.
│   └── jax/                    # optional submodule
│       ├── __init__.py         # JaxRobotHandle, JAX-flavored API
│       └── _ffi.cpp            # XLA_FFI_DEFINE_HANDLER + dispatch into robot.so
└── pyproject.toml              # [jax] optional extra (jax>=0.4.40,<0.5)
```

Install:
```
pip install grid-rbd            # base (nanobind only)
pip install grid-rbd[jax]       # adds JAX FFI bridge; reuses same cache
```

## Tech choices (confirmed via 2026 web research)

- **nanobind**: 4× faster compile, 5× smaller binaries, 10× lower runtime
  overhead than pybind11. Built-in `ndarray` that bridges NumPy / PyTorch
  / JAX / TensorFlow through one C++ type. Targets Python Stable ABI on
  3.12+, so one wheel works across Python versions.
- **JAX FFI**: still officially "experimental" in 2026 but is the only
  sanctioned C++/CUDA bridge for JAX. Pin a narrow JAX version range
  (`jax>=0.4.40,<0.5`) and bump as needed. Pattern:
  `XLA_FFI_DEFINE_HANDLER` + `XLA_FFI_REGISTER_HANDLER` on platform
  `"CUDA"`, `cudaStream_t` via `ffi::PlatformStream<cudaStream_t>()`.

Alternative deferred: `jax-tvm-ffi` (NVIDIA's adapter that decouples from
JAX's FFI ABI). Adds a dep we don't need yet; reconsider only if JAX FFI
breaks on us.

## Build strategy: build at register_robot, not at pip-install

The PyPI wheel ships:
- The nanobind runner stub (no robot baked in).
- The Python orchestration code (codegen + compile invokers).

The wheel does **not** ship any CUDA code precompiled. Robot-specific
`.so` files are built at `register_robot()` time, which requires `nvcc`
on the user's machine (they need it anyway since GRiD targets CUDA).

This eliminates the CUDA-version-matrix problem in wheel building. The
wheel is pure-Python + a small nanobind runner; the heavy lifting
happens on the user's GPU at register time.

## Cache structure

```
~/.cache/grid-rbd/
├── manifest.json              # name -> cache_key registry
├── compile.log                # rolling build log (debug)
└── store/
    ├── 7d81d3ff.../           # cache_key prefix
    │   ├── grid.cuh           # generated header
    │   ├── robot.so           # compiled wrapper
    │   ├── meta.json          # NUM_JOINTS, NUM_EES, NV, options
    │   └── build.log
    └── ...
```

**Cache key** = `sha256(urdf_bytes + canonical_json(options) +
grid_rbd_version + cuda_arch)`. CUDA arch is part of the key so a user
with two GPUs (laptop sm_86 + desktop sm_120) keeps separate `.so`s.

**Manifest** = `name → cache_key` mapping. If a user re-registers the
same name with a different URDF, the old name binding is overwritten and
the old `.so` becomes eligible for GC (e.g. via `grid-rbd gc` CLI command).

## Public API surface

```python
def register_robot(
    name: str,
    urdf_path: str,
    *,
    floating_base: bool = False,
    floating_base_convention: str = "pinocchio",
    ee_joint_names: Iterable[str] | None = None,
    target_shared_mem_bytes: int = 98304,
    linalg_backend: str = "glass",
    cache_dir: str | None = None,
    force_rebuild: bool = False,
) -> RobotHandle: ...

def get_robot(name: str, cache_dir: str | None = None) -> RobotHandle: ...

def list_registered(cache_dir: str | None = None) -> list[dict]: ...


class RobotHandle:
    name: str
    num_joints: int
    num_ees: int
    num_vel: int                          # NV - 1 for floating
    floating_base: bool

    # All methods take/return 2D arrays where axis 0 is batch.
    def rnea(self, q, qd, qdd=None, *, gravity=-9.81): ...     # (B,NV)->(B,NV)
    def aba(self, q, qd, tau, *, gravity=-9.81): ...           # (B,NV)->(B,NV)
    def crba(self, q): ...                                     # (B,NV)->(B,NV,NV)
    def minv(self, q): ...                                     # (B,NV)->(B,NV,NV)
    def forward_dynamics(self, q, qd, u, *, gravity=-9.81): ...
    def rnea_grad(self, q, qd, qdd=None, *, gravity=-9.81): ...
    def forward_dynamics_grad(self, q, qd, u, *, gravity=-9.81): ...
    def end_effector_pose(self, q): ...
    def end_effector_pose_gradient(self, q): ...
    def end_effector_pose_hessian(self, q): ...
    def idsva_so(self, q, qd, qdd, *, gravity=-9.81): ...
    def fdsva_so(self, q, qd, u, *, gravity=-9.81): ...

    def close(self): ...
    def __enter__(self): return self
    def __exit__(self, *a): self.close()
```

## JAX side

`grid_rbd.jax.register_robot` calls the standard `register_robot`, then
additionally registers per-robot JAX FFI handlers typed to that robot's
`NUM_JOINTS` / `NUM_EES`. Result handles are `JaxRobotHandle`s with the
same algorithm surface but DeviceArray-flavored.

```python
import grid_rbd.jax as grid_jax
import jax

handle = grid_jax.register_robot("iiwa14", urdf_path="iiwa.urdf")
# Shares cache with standard API — if already compiled, no re-compile.

@jax.jit
def policy(q, qd, u):
    return handle.forward_dynamics(q, qd, u)

# Works under vmap; FFI handler runs on the JAX-supplied cudaStream_t.
batched = jax.vmap(policy)(q_batch, qd_batch, u_batch)
```

## Bench-the-bindings benchmark

Add `test/benchmarks/benchmark_wrappers.py` that times:
- Direct CUDA kernel launch (raw runner, no wrapper).
- nanobind path: `handle.rnea(q, qd)` round-trip.
- JAX path: `jax.jit(handle.rnea)(q_jax, qd_jax)` (after warmup).
- Pinocchio CPU reference.

Catches wrapper-level regressions and gives users a clear "how much
overhead does the wrapper add" answer.

## Open questions deferred to implementation

1. **JAX FFI version pinning policy** — start narrow (`>=0.4.40,<0.5`);
   bump as JAX releases land.
2. **Plural `register_robots(...)`** for parallel compile of multiple
   robots. Useful "model zoo" pattern; add in v1.
3. **Cross-arch caching** — handled automatically because CUDA arch is in
   the cache key.
4. **`RobotHandle` pickleability** — deferred to v2; pickle as `name +
   cache_dir` and reload on unpickle.
5. **License of generated `.so`** — user-owned (their URDF baked in);
   nothing to do, just document.

## Migration: delete `bindings/`

Once `grid-rbd` lands on PyPI:
- Delete `bindings/` from the main repo.
- Add a top-level `README.md` section pointing JAX users at
  `pip install grid-rbd[jax]` and standard users at `pip install grid-rbd`.

## Implementation phases

1. **Phase A**: scaffolding — `pyproject.toml`, `scikit-build-core`
   setup, nanobind runner stub, cache module, manifest schema.
2. **Phase B**: codegen + compile invokers (Python). Test
   `register_robot` works end-to-end for iiwa14_fixed.
3. **Phase C**: `RobotHandle` algorithm methods. Validate output vs
   `RBDReference` for one robot.
4. **Phase D**: JAX FFI submodule + `JaxRobotHandle`. Validate `jax.jit`
   compatibility.
5. **Phase E**: docs + examples (`examples/wrapper_quickstart.py`,
   `examples/jax_policy_loop.py`, etc.).
6. **Phase F**: bench-the-bindings + delete `bindings/`.
7. **Phase G**: publish to PyPI.

## Sources (web research, 2026-05-18)

- nanobind documentation — https://nanobind.readthedocs.io/
- nanobind benchmarks — https://nanobind.readthedocs.io/en/latest/benchmark.html
- nanobind vs pybind11 in 2026 — https://www.matecdev.com/posts/nanobind-vs-pybind11-cpp-python.html
- JAX FFI docs — https://docs.jax.dev/en/latest/ffi.html
- JAX FFI examples — https://github.com/jax-ml/jax/tree/main/examples/ffi
- Extending JAX with C++ / CUDA — https://dfm.io/posts/extending-jax/
- XLA Custom Calls (OpenXLA) — https://openxla.org/xla/custom_call
