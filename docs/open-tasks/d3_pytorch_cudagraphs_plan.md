# D.3 — PyTorch in-memory compile + CUDA-Graphs callable

**Status:** planning (read-only exploration done 2026-05-30). Backlogged in
`HANDOFF.md:1233` ("D.3 PyTorch in-memory compile + CUDA-Graphs callable"),
motivation at `HANDOFF.md:1112-1121`.

**Goal.** A PyTorch analog of the existing `grid_rbd` JAX/pybind11 bindings:
(a) compile a per-robot generated `grid.cuh` and expose the kernels as PyTorch
custom ops (autograd-aware where a `*_gradient` kernel exists), and (b) a
CUDA-Graphs-captured callable for fixed-batch, low-launch-overhead repeated
calls (MPC / training loops).

This plan deliberately **reuses the existing `grid_rbd` C++ launch layer**
rather than building a parallel one. The new code is a thin `grid_rbd.torch`
subpackage + a small additive block in `wrapper_template.cu`.

---

## 0. What already exists (prior art to reuse)

The `grid_rbd` package (`python/grid_rbd/`, v0.3) already solves codegen,
compile, cache, and device-resident kernel dispatch. Key files:

- **Codegen+compile pipeline** — `python/grid_rbd/_compile.py`.
  `generate_grid_cuh()` (`_compile.py:70`) runs `URDFParser` +
  `GRiDCodeGenerator.gen_all_code()`; `compile_so()` (`_compile.py:154`)
  invokes `nvcc` with `-gencode=arch=compute_<cc>,code=sm_<cc>` (`_compile.py:176`),
  `--shared --compiler-options=-fPIC` (`_compile.py:30-40`), and a
  conditional `-DGRID_RBD_WITH_JAX` block (`_compile.py:188-195`). The .so
  lands in a content-addressed cache.
- **Content-addressed cache** — `python/grid_rbd/_cache.py`. Cache key =
  `sha256(urdf + canonical_options + version + cuda_arch + wrapper_template_hash)`
  (`_cache.py:97-107`). `detect_cuda_arch()` (`_cache.py:51`) returns e.g.
  `120` for sm_120.
- **Robot-agnostic C ABI + JAX FFI** — `python/grid_rbd/wrapper_template.cu`
  (1251 lines). This is the **shared launch layer**. It holds device-side
  singleton state (`g_data`, `g_robot`, `g_streams`, `g_block_dimms`,
  `g_thread_dimms` — `wrapper_template.cu:39-46`), a lifecycle
  (`grid_rbd_init`/`grid_rbd_close` — `:50-65`), metadata getters (`:69-86`),
  a plain `extern "C"` host-path ABI per algorithm (`:113-480`), and — gated
  on `-DGRID_RBD_WITH_JAX` — fully device-resident XLA FFI handlers
  (`:505-1251`) that D→D-repack inputs into `g_data->d_q_qd_u` and launch the
  **kernel directly on a caller-supplied stream**.
- **pybind11 Runner** — `python/src/_core.cpp`. `Runner` dlopens the .so,
  `require_sym`s the `extern "C"` symbols, and marshals numpy↔C ABI.
- **Python handle surfaces** — `python/grid_rbd/_handle.py` (`RobotHandle`,
  numpy) and `python/grid_rbd/jax/__init__.py` (`JaxRobotHandle`, jittable
  FFI). 14 methods on each; the reshape/transpose conventions for every
  output live in these files and must be mirrored exactly.

**The single most important reuse decision:** the JAX FFI handlers in
`wrapper_template.cu` already do *exactly* what a PyTorch op needs — take a
CUDA stream + device pointers, D→D-pack into the singleton, launch the kernel
on that stream, D→D-copy the result out. PyTorch custom ops give us a
`cudaStream_t` (`at::cuda::getCurrentCUDAStream()`) and device pointers
(`tensor.data_ptr()`). So the torch path is **the same launch body with a
different ABI shim** — not a reimplementation.

---

## 1. In-memory compilation approach

### Options considered

| Option | Verdict |
|---|---|
| **A. `torch.utils.cpp_extension.load_inline` / `load`** | **Recommended.** |
| B. NVRTC (runtime PTX from source string) | Rejected for v1. |
| C. Reuse the existing cached `.so`, load via `torch.ops.load_library` | Recommended **as the fast path / cache hit**. |

### Recommendation: a two-tier scheme (C then A)

1. **Cache-hit fast path (C):** the `.so` produced by the existing
   `grid_rbd` pipeline already contains the kernels and the shared launch
   layer. For the torch surface we compile a *second, torch-aware* shim TU
   (the `TORCH` block of `wrapper_template.cu`, see §2) into the same .so by
   adding `-DGRID_RBD_WITH_TORCH=1` to the nvcc invocation in
   `compile_so()` (`_compile.py:154`), exactly mirroring the existing
   `-DGRID_RBD_WITH_JAX` gate (`_compile.py:188`). Then on the Python side
   load it with `torch.ops.load_library(so_path)` (or dlopen + a small
   pybind registrar). **No second compile** for users who already registered
   the robot — the cache key must gain a `with_torch` bit so a JAX-only build
   isn't silently reused.

2. **`load_inline` path (A):** only needed if we want torch to be able to
   build a robot *without* nvcc-on-PATH being separately orchestrated, i.e.
   to make `import grid_rbd.torch; register_robot(...)` self-contained.
   `torch.utils.cpp_extension.load[_inline]` already knows how to find the
   CUDA toolkit, set `-fPIC`, build a `.so`, and `torch.ops`-register it.
   We hand it the generated `grid.cuh` + the torch shim and the GLASS include
   dirs. This is the same nvcc call the existing pipeline makes, just driven
   by torch's build machinery instead of our `subprocess.run`.

   **Decision:** keep our own `subprocess`-driven `nvcc` (option C build) as
   the canonical compile — it already handles GLASS includes
   (`_compile.py:182-184`), the arch flag, and the cache. Make the torch
   shim a *compile flag* on that path, not a separate torch-driven build. Use
   `load_inline` only as a documented fallback if `torch.ops.load_library`
   on our hand-built .so proves brittle across torch ABI versions (the torch
   custom-op registration macros bake in the torch C++ ABI; if torch is
   present at compile time we should let torch's own headers in, which means
   adding `torch.utils.cpp_extension.include_paths()` and
   `library_paths()` to the nvcc command — mirror `_jax_ffi_include_dir()`
   at `_compile.py:145`).

### sm_120 / dynamic-shared-memory launch concerns

These are **already solved by the existing init path** and the torch shim
inherits the fix for free, but the implementer must not regress them:

- **`-arch`:** reuse `detect_cuda_arch()` (`_cache.py:51`) →
  `-gencode=arch=compute_120,code=sm_120` (`_compile.py:176`). No change.
- **>48 KB dynamic shared memory opt-in:** GRiD kernels request up to
  ~100 KB of dynamic smem (e.g. `IDSVA_SO_BODY_FRAME` PERF tier,
  `grid.cuh:371`). On every arch ≥ sm_70 a kernel that requests >48 KB must
  be opted in via `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
  *before launch*, or the launch fails with `cudaErrorInvalidValue`. This is
  done by `init_grid_kernel_attrs<T>()` (`grid.cuh:18713`), which is called
  by `init_grid<T>()` (`grid.cuh:18898-18899`), which the shared
  `grid_rbd_init()` already calls (`wrapper_template.cu:52`). **So the torch
  shim must call `grid_rbd_init()` once at handle construction (as the
  JaxRobotHandle path implicitly does via the lazy `if (!g_data)` guards in
  every handler).** This is also why the opt-in MUST happen before CUDA-graph
  capture (see §3) — it is a host-side device-global registration, idempotent,
  and gated on the device max (`grid.cuh:18723-18724`).
- **Launch shared-mem argument:** each kernel launch passes the matching
  `*_DYNAMIC_SHARED_MEM_BYTES<T>()` (e.g. `grid::ID_DYNAMIC_SHARED_MEM_BYTES<T>()`
  at `wrapper_template.cu:555`). The torch shim copies the JAX handler bodies
  verbatim, so it gets the right smem bytes per kernel automatically.

---

## 2. PyTorch op surface

### Where the ops live

Add a `TORCH` block to `wrapper_template.cu`, gated `#ifdef GRID_RBD_WITH_TORCH`,
structurally identical to the `#ifdef GRID_RBD_WITH_JAX` block
(`wrapper_template.cu:505-1251`). Each torch op:

1. Takes `torch::Tensor` args (asserted `.is_cuda()`, `.is_contiguous()`,
   `.dtype()==kFloat32`, shape `(B, NJ)`).
2. Grabs the stream: `cudaStream_t stream = at::cuda::getCurrentCUDAStream();`.
3. Lazily `grid_rbd_init()` (same guard as the JAX handlers).
4. D→D-repacks inputs into `g_data->d_q_qd_u` with the *same* `cudaMemcpy2DAsync`
   pitch trick the JAX handlers use (`wrapper_template.cu:541-546`).
5. Allocates the output with `torch::empty({...}, opts)` on the same device/stream.
6. Launches the same kernel on `stream` with the matching smem bytes.
7. D→D-copies the singleton's output buffer into the output tensor on `stream`.
8. Returns the tensor (no host sync — keep it async/stream-ordered so it can be
   graph-captured).

Register with the torch dispatcher: `TORCH_LIBRARY(grid_rbd_<key>, m){ ... }`
+ `TORCH_LIBRARY_IMPL(..., CUDA, m){ ... }`. The library name must be keyed by
the cache_key (like `_ffi_target_name` at `jax/__init__.py:51`) so two robots
in one process don't collide on the op namespace.

### Algorithm list + autograd wiring

Forward-only (no gradient kernel; expose as plain custom ops, mark
non-differentiable):
`minv`, `crba`, `end_effector_pose`, `end_effector_pose_gradient` (itself a
Jacobian, but we don't differentiate *through* it), `end_effector_pose_hessian`,
`idsva_so`, `fdsva_so`, `integrator_gradient`. These are first-class ops; if a
user puts them in an autograd graph we raise (or return a `.detach()`-style
result) per torch custom-op conventions.

**Autograd-aware pairs** (forward op + `torch.autograd.Function.backward`
using the existing analytic gradient kernel):

| Forward op | Output | Backward uses | Backward maps to |
|---|---|---|---|
| `rnea(q,qd)` → c | (B,NJ) | `rnea_grad` → (B,NJ,2·NJ) | VJP: `grad_c @ [dc_dq \| dc_dqd]` → grad wrt (q,qd) |
| `forward_dynamics(q,qd,u)` → qdd | (B,NJ) | `forward_dynamics_grad` → (B,NJ,2·NJ) for (q,qd); `minv` for ∂qdd/∂u (= M⁻¹) | VJP wrt (q,qd,u) |
| `aba(q,qd,u)` → qdd | (B,NJ) | same as forward_dynamics (qdd identical) | reuse fd_grad + minv |
| `integrator(q,qd,u,dt)` → x_{k+1} | (B,NP+NV) | `integrator_gradient` → (B,2·NV,3·NV) = [d/dq\|d/dqd\|d/du] | VJP: `grad_x @ dAB` |

**Backward = vector-Jacobian product, batched.** GRiD emits the *full*
analytic Jacobian per timestep; PyTorch autograd needs `vᵀJ`. So the backward
is: call the gradient kernel once (forward of the gradient), reshape to the
`_handle.py` row-major convention (e.g. `rnea_grad` reshape at
`_handle.py:229`), then `torch.einsum`/`bmm` the upstream `grad_output`
against it. This means the gradient kernel runs in the **backward pass on the
same stream** and is itself graph-capturable.

Implementation: a `torch.autograd.Function` subclass per differentiable
algorithm. `forward()` saves `(q,qd[,u],gravity)` for backward and calls the
forward op; `backward(grad_out)` calls the gradient op + does the batched VJP
contraction in torch (so the contraction is autograd-traceable and on-GPU).
Wrap each in a Python method on `TorchRobotHandle` mirroring `_handle.py`.

Note: GRiD's `rnea`/`rnea_grad`/`idsva_so` currently run with
`USE_QDD_FLAG=false` (`wrapper_template.cu:122-124, 285, 365`), so `qdd` is
accepted-but-ignored, exactly like the numpy/JAX surfaces. Keep that parity;
don't introduce a `qdd` gradient path the other surfaces lack.

### Batch / stride / dtype conventions (match `grid_rbd` exactly)

- **dtype:** float32 only (the C ABI is `using T = float`,
  `wrapper_template.cu:23`). Assert, don't silently cast (torch users expect
  explicit dtype).
- **Layout:** 2D `(B, NJ)` inputs, C-contiguous; `B ≤ kMaxBatch`
  (compile-time `GRID_RBD_MAX_BATCH`, `wrapper_template.cu:26-29`). Validate
  like `_prep_2d` (`jax/__init__.py:123`).
- **Output reshapes:** copy the per-method reshape/transpose from `_handle.py`
  verbatim — e.g. `minv` lower-triangle symmetrize (`_handle.py:163-167`),
  `end_effector_pose_gradient` `(B,NEE,NV,6)→(B,6·NEE,NV)`
  (`_handle.py:213`), `rnea_grad`/`fd_grad` `(B,2,NJ,NJ)` col-major→row-major
  concat (`_handle.py:229`, `:241`), `idsva_so`/`fdsva_so` 4×NV³ slicing
  (`_handle.py:268`, `:280`), `integrator_gradient` `(B,3·NV,2·NV)→transpose`
  (`_handle.py:308`). Do these reshapes **in torch** (they're cheap views/
  copies and stay on-GPU + graph-capturable).
- **gravity:** runtime float kwarg, default 9.81 (`_handle.py:139`); `dt` +
  integrator-type int for the integrator (`_handle.py:282`, `_INTEGRATOR_CODES`
  at `_handle.py:28`).

---

## 2b. Notebook / interactive register-then-run UX

**This is a first-class requirement of the torch backend, not a nice-to-have.**
The canonical interactive workflow is: a user (in a Jupyter / Colab cell) defines
or points at a URDF, calls `register_robot(...)` in one cell, and a few cells
later calls algorithm methods on the returned handle — and on a notebook *re-run*
(kernel restart, "Run All") the registration cell is a **cache hit**, not a
recompile. The torch handle must deliver exactly the same two-tier UX the JAX
path already gives (§0): register-once (slow, compile) → call-many (fast), backed
by the existing content-addressed cache (`_cache.py:97-107`).

### The handle is the notebook object

`grid_rbd.register_robot(name=..., urdf_path=..., backend="torch", ...)` (or the
mirror entry point `grid_rbd.torch.register_robot(...)`, parallel to
`grid_rbd.jax.register_robot` at `jax/__init__.py:380`) returns a
`TorchRobotHandle` whose methods (§2) are autograd-aware torch ops returning
`torch.Tensor`. Decision: expose **both** a `backend=` kwarg on the top-level
`register_robot` (`__init__.py:54`) *and* a `grid_rbd.torch` submodule, so a
notebook can do either:

```python
import grid_rbd
h = grid_rbd.register_robot("iiwa14", urdf_path="iiwa14.urdf", backend="torch")  # cell 1
# ... markdown, plots, other cells ...
qdd = h.forward_dynamics(q, qd, u)            # cell N — torch.Tensor, autograd-aware
qdd.sum().backward()                           # gradient flows (§2 autograd wiring)
```

The `backend=` kwarg just dispatches to the right handle constructor; the
underlying `register_robot` body (parse → `generate_and_compile` → cache →
manifest) is **unchanged** (`__init__.py:131-141`). The torch handle, like
`JaxRobotHandle`, wraps the base `RobotHandle` plus the `.so`/`cache_key` pulled
from the manifest (`jax/__init__.py:410-420`).

### Why register-then-run "just works" across cells and re-runs

- **First call in the session compiles** (or cache-hits). `register_robot` is
  idempotent and content-addressed: the `.so` lands under
  `store_dir(cache_dir, cache_key)` (`_cache.py:114`). A notebook kernel restart
  + "Run All" recomputes the same `cache_key` (sha256 over urdf+options+version+
  arch+wrapper hash, `_cache.py:97-107`) and the `if not so_path.exists()` guard
  (`__init__.py:131`) skips straight to loading the cached `.so` — **no nvcc**.
  This is the same mechanism that makes the JAX smoke test "start in <1s" on the
  second run (`test/python_wrappers/test_iiwa14_smoke.py:6-7`).
- **Process-global op registration is idempotent.** The torch
  `TORCH_LIBRARY(grid_rbd_<key>, ...)` registration (§2) is keyed by `cache_key`
  (like `_ffi_target_name` at `jax/__init__.py:51`) and guarded by a
  `_REGISTERED` set + lock exactly like the JAX FFI target registry
  (`jax/__init__.py:47-48, 67-86`). So calling a method in a *later* cell
  registers-on-first-use and is a no-op thereafter — re-running a downstream cell
  many times never re-registers or re-loads.
- **One `.so` per kernel process; survives `get_robot`.** A second handle to the
  same robot in a later cell (`grid_rbd.get_robot(name, ...)` →
  torch variant, mirror of `jax/__init__.py:423`) reuses the cached `.so` and the
  already-registered ops. (Note the single-robot-singleton device-state caveat,
  §6.5: two *different* robots are fine; concurrent capture of two robots sharing
  the singleton scratch is the v2 limitation.)

### `urdf_string=` support — inline URDFs with no file on disk (REQUIRED for fully inline notebooks)

A notebook cell should be able to define the URDF as a Python string literal and
register it without writing a file first. Today `register_robot` only accepts
`urdf_path` and immediately does `Path(urdf_path).read_bytes()`
(`__init__.py:108-111`). Add a mutually-exclusive `urdf_string: str | None = None`
parameter. **This is a small, surgical change with two exact touch points:**

1. **`__init__.py` `register_robot` (`__init__.py:105-127`)** — the only place
   bytes are sourced and the cache key is computed. Replace the
   `urdf_path`-resolve-and-read block with:
   - if `urdf_string` is given: `urdf_bytes = urdf_string.encode("utf-8")` (no
     filesystem touch); else the existing `urdf_p.read_bytes()` path.
   - **the cache key already hashes `urdf_bytes`** (`compute_cache_key`,
     `_cache.py:97-107` / `:102` `h.update(urdf_bytes)`), so an inline string and
     the equivalent file produce the **identical** `cache_key` — *no change to
     `compute_cache_key` is needed*. Two notebook cells with byte-identical URDF
     text are automatic cache hits, and an inline URDF that matches an on-disk one
     dedupes to the same `.so`.
   - Mirror the same `urdf_string` branch in `grid_rbd.jax.register_robot`
     (`jax/__init__.py:380-409`) and the torch `register_robot`, since both
     delegate to the base `register_robot` (`jax/__init__.py:400`) — so passing
     `urdf_string=` through the base is all that's required; the jax/torch
     wrappers just forward the new kwarg.

2. **`_compile.generate_and_compile` / `generate_grid_cuh`
   (`_compile.py:70-90, 212-226`)** — the URDF reaches the parser as a *path*
   (`URDFParser().parse(str(urdf_path), ...)`, `_compile.py:86-90`). Two options,
   pick the **temp-file** one for v1 (lowest risk, parser API unchanged):
   - **(chosen) temp-file:** in `register_robot`, when `urdf_string` is given,
     write it to a `NamedTemporaryFile(suffix=".urdf")` inside the cache entry dir
     (`entry_dir`, `__init__.py:128`) — i.e. persist it as `entry_dir/robot.urdf`
     so re-runs and debugging can see the exact source — and pass that path into
     `generate_and_compile`. The cache key is still computed from the *string
     bytes*, not the temp path, so the path being non-deterministic doesn't leak
     into the key.
   - (deferred) feed the parser directly: add a `parse_string` entry point to
     `URDFParser` and thread an in-memory branch through `generate_grid_cuh`
     (`_compile.py:86`). More invasive (parser change); defer to v2.

   Net: `generate_and_compile`'s signature is unchanged — it still receives a
   `Path`. Only `register_robot` learns to materialize the string to a temp path
   under `entry_dir` before calling it.

**Gap flag:** until `urdf_string=` lands, fully self-contained inline notebooks
must `urdf_path=` a file (e.g. one fetched via `robot_descriptions`, as the
existing examples do — `examples/quickstart_iiwa14.py:16-21`). `urdf_string=` is
therefore a **D.3 prerequisite** for the "define the URDF in a cell" notebooks
(see the notebook plan, `notebook_examples_plan.md`). It is independent of the
torch op work and could land first.

### First-compile latency guidance (set notebook expectations)

The first `register_robot` of a robot runs `nvcc` on the generated `grid.cuh`
(`_compile.py:154-209`), and **`grid.cuh` is large** (the repo's checked-in one
is ~1.1 MB; per-robot ones scale with DOF and with the SO kernels, which
`generate_grid_cuh` always enables — `_compile.py:118` `enable_floating_second_order=True`).
Practical guidance to put in every notebook's first cell:

- **iiwa14 (7-DOF fixed-base): seconds to tens of seconds** of nvcc — fine for a
  live demo.
- **g1 / h1_2 (floating-base humanoids, ~30-40 DOF): minutes** of nvcc (the SO
  kernels dominate). For these, either pre-warm the cache out-of-band (run
  `register_robot` once in a setup script / CI cache-priming step before the demo)
  or scope the live cells to the kinematics/first-order methods. **Recommend small
  robots (iiwa14, go2) for live/CI notebooks; gate the humanoid notebook behind a
  "this cell compiles for minutes / use a pre-warmed cache" markdown banner.**
- **Cache pre-warming pattern for CI/demos:** call `register_robot(...)` in a
  fixture or a `make warm-cache` step so the notebook's registration cell is a
  guaranteed cache hit (`so_path.exists()` true → no nvcc, `__init__.py:131`).
  The cache is content-addressed and host-local, so a warmed `~/.cache/grid-rbd/`
  is reusable across notebook runs and across the JAX/torch/numpy surfaces (they
  share the `.so`, `jax/__init__.py:18-21`).

### Runtime smem opt-in + L2 setup timing in a notebook context

The interactive path inherits the device-global-state setup already required for
correctness (§1, §3):

- The **>48 KB dynamic-shared-memory opt-in** (`cudaFuncSetAttribute` via
  `init_grid_kernel_attrs<T>()`, `grid.cuh:18713`, called from `grid_rbd_init()`,
  `wrapper_template.cu:52`) happens **once, lazily, on the first method call** in
  the notebook (the `if (!g_data)` guard, §1). So a user who registers in cell 1
  and first *calls* a method in cell 5 pays the one-time init at cell 5, not at
  registration. This is correct and invisible, but worth a one-line doc note so a
  user doesn't misread the first call's latency as per-call overhead.
- The **L2-persisting window setup** is *not* on the per-call torch path (the
  torch ops launch kernels kernel-direct, bypassing the host wrappers that call
  `grid_begin_l2_persisting`, §3 point 2). So an interactive (non-captured) call
  does no L2 stream-attribute setup; the L2 window only matters for the
  CUDA-Graphs capture path (§3, risk #1), which is an explicit `handle.capture(...)`
  the notebook opts into (see the CUDA-Graphs micro-demo in
  `notebook_examples_plan.md`).

**Consistency with §1:** all of the above reuses the `grid_rbd` build pipeline
and the existing cache — the notebook UX adds **no new compile path**. It is the
same `subprocess`-driven nvcc + `-DGRID_RBD_WITH_TORCH` flag (§1 option C), driven
the same way whether the caller is a script, a pytest fixture, or a notebook cell.
We do **not** introduce `torch.utils.cpp_extension.load_inline` for the notebook
path — `load_inline` remains only the documented fallback for torch-ABI brittleness
(§1 option A). A notebook is just another caller of the one canonical pipeline.

---

## 3. CUDA-Graphs capture design

### The callable

Expose `handle.capture(method, *example_inputs, **kwargs) -> GraphCallable`.
The returned object holds:
- fixed input tensors (`static_in`) of the captured batch size,
- the output tensor(s) (`static_out`),
- a `torch.cuda.CUDAGraph`.

Usage (MPC loop):
```python
g = handle.capture("forward_dynamics", q0, qd0, u0)   # fixed B
for k in range(horizon):
    g.static_in[0].copy_(q_k); g.static_in[1].copy_(qd_k); g.static_in[2].copy_(u_k)
    g.replay()
    qdd_k = g.static_out   # device tensor, no relaunch overhead
```

Capture pattern (standard `torch.cuda.graph`):
```python
# 1. WARMUP off-graph (forces lazy init + smem opt-in + any first-touch allocs)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): static_out = op(*static_in, **kw)
torch.cuda.current_stream().wait_stream(s)
# 2. CAPTURE
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = op(*static_in, **kw)
```

### Why warmup is load-bearing here (the device-global-state landmines)

1. **`grid_rbd_init()` / `init_grid_kernel_attrs`.** The dynamic-smem opt-in
   (`grid.cuh:18713-18899`) calls `cudaFuncSetAttribute` and a device query.
   These are **illegal during stream capture**. They MUST run in warmup
   (which they do, on first op call). The capture itself then only issues the
   `cudaMemcpy2DAsync` repack + kernel launch + `cudaMemcpyAsync` out — all
   capturable, stream-ordered operations.

2. **`grid_begin_l2_persisting` (L2-pinned scratch).** The generated *host
   wrappers* call `grid_begin_l2_persisting(0, d_workspace, ...)` per
   algorithm (e.g. `grid.cuh:10510` for id_du, `:11624` for fd_du,
   `:14205` for integrator, `:5077` for ee-grad). This sets a **stream
   attribute** (`cudaStreamSetAttribute` access-policy window) and is
   **NOT capturable**. **Critical design point:** the torch ops (like the JAX
   FFI handlers) must launch the **kernel directly**, NOT the host wrapper —
   the JAX handlers already do exactly this (`wrapper_template.cu:553` calls
   `grid::inverse_dynamics_kernel<T><<<...,stream>>>` directly, bypassing the
   `grid::inverse_dynamics<T>(...)` host wrapper used by the plain C ABI at
   `wrapper_template.cu:129`). The torch shim copies the JAX bodies, so it
   inherits the kernel-direct launch and **never touches L2-persisting or
   `cudaDeviceSynchronize` inside the capturable region.** If L2-persisting is
   wanted for the captured loop, set it **once in warmup** on the capture
   stream and leave it (the window persists on the stream).

3. **`d_workspace` / `d_idsva_so` scratch aliasing.** The singleton's
   `d_workspace` and SO scratch (`d_idsva_so`) are reused across calls
   (`fdsva_so` aliases `d_idsva_so` — `jax/__init__.py:322`,
   `wrapper_template.cu:1119`). Within one captured graph this is fine (the
   graph linearizes one op). But **two captured graphs that both use the
   singleton scratch cannot replay concurrently** on different streams — they
   stomp `g_data`. Document: one capture is live at a time per handle; for
   concurrent capture, the singleton must become per-handle (a v2 concern,
   same limitation the JAX path already has, see `_core.cpp:19-24`).

### Fixed-batch capture & recapture

- The captured batch size B is baked into the graph (output shapes + the
  `batch` arg passed to the kernel are constants at capture time). **Recapture
  when B changes** (different horizon / minibatch). Provide
  `handle.capture(..., batch=B)`; cache graphs by B in the `GraphCallable`
  factory.
- `gravity`, `dt`, integrator-type are captured as **constants** (they're
  scalar kernel args, not tensors). If a user varies `dt` per step they must
  recapture, OR (better) we promote `dt` to a 1-element device tensor input so
  it can be `.copy_()`'d between replays — a small codegen-free change since
  `dt` is already a runtime kernel arg (`wrapper_template.cu:1149`). Recommend
  the device-scalar-`dt` variant for the integrator capture path specifically,
  since varying `dt` is common in adaptive integrators.
- Interaction with `set_threads_per_block` (`_handle.py:119`,
  `wrapper_template.cu:75`): the block/thread dims are read from the
  `g_thread_dimms` global at launch (`wrapper_template.cu:554`). Changing it
  after capture has **no effect** on a captured graph (the launch config is
  baked). Document: set threads-per-block before capture; changing it
  requires recapture. The T5 autotune (PERF→whichever-tier-best) picks the
  block size at register time, so the captured config inherits the tuned one.

---

## 4. Reuse vs divergence from the JAX-FFI path

**Shared (the win):**
- The entire codegen+compile+cache pipeline (`_compile.py`, `_cache.py`) —
  add one `with_torch` cache-key bit + one nvcc flag (`-DGRID_RBD_WITH_TORCH`)
  + torch include/lib paths, mirroring the JAX gate (`_compile.py:188-195`).
- The singleton device state + the **kernel-direct, stream-ordered launch
  bodies**. The torch shim's per-algorithm body ≈ the JAX handler body with
  `ffi::Buffer<F32>`→`torch::Tensor` and `c->typed_data()`→`out.data_ptr<float>()`.
  Strongly consider factoring the common middle (the `cudaMemcpy2DAsync`
  repack + kernel launch + `cudaMemcpyAsync` out) into a `static inline`
  helper per algorithm shared by both `#ifdef` blocks, to avoid drift when the
  tier/spill rollout changes a kernel signature (the memory note flags this:
  "when a kernel gains a d_workspace param, its wrapper FFI handler + caller
  must be updated"). **Recommendation: extract the launch core now**, so T4's
  `d_f_ext` (see §6) is plumbed in one place for both surfaces.
- The output reshape conventions (`_handle.py`) — reimplemented in torch but
  semantically identical; share via a docstring/table, not code (numpy vs
  torch).

**Divergent:**
- ABI shim: XLA FFI (`XLA_FFI_DEFINE_HANDLER_SYMBOL` + capsule registration,
  `jax/__init__.py:83`) vs torch dispatcher (`TORCH_LIBRARY` + autograd
  `Function`). Torch needs the autograd backward wiring (JAX gets VJP for free
  via `jax.custom_vjp` if/when added; today the JAX surface is forward-only).
- torch needs an explicit CUDA-Graphs callable; JAX gets graph capture
  implicitly via `jax.jit` + XLA's own command-buffer/graph lowering.
- Build-time ABI: torch custom-op headers bake the torch C++ ABI into the .so,
  so a torch-built .so is tied to the torch version. Add torch version to the
  cache key (like `package_version()` at `_cache.py:67`).

---

## 5. Testing strategy

Mirror `test/python_wrappers/test_iiwa14_smoke.py` (parity vs `RBDReference`,
`_TOL=5e-3`, iiwa14 fixed-base, session-scoped register).

1. **Parity vs existing bindings** (`test/python_wrappers/test_iiwa14_torch_smoke.py`):
   for every method, assert `torch_handle.<m>(...).cpu().numpy()` matches
   `RobotHandle.<m>(...)` (numpy) AND `JaxRobotHandle.<m>(...)` within tol,
   on the same random `(q,qd,u)`. This pins the reshape conventions.
2. **Autograd gradcheck:** `torch.autograd.gradcheck` on the differentiable
   ops (`rnea`, `forward_dynamics`, `aba`, `integrator`) in float64 if a
   float64 build is available — but the kernels are float32-only, so instead
   do an **analytic-vs-finite-difference** check: compare the custom backward's
   VJP against a central-difference VJP of the forward op at float32 tol
   (~1e-2 relative; document the loose tol). Also cross-check the backward's
   Jacobian against `RBDReference`'s `rnea_grad`/`forward_dynamics_grad`
   directly (the gradient kernel output is already validated there).
3. **CUDA-Graphs equivalence:** capture each capturable op, replay with N
   different inputs `.copy_()`'d in, assert replay output == eager op output
   bit-for-bit (same kernel, same config → identical). Assert a second
   `.replay()` with new inputs gives new correct outputs (no stale capture).
4. **Recapture-on-batch-change:** capture at B=64, then B=128, assert both
   correct and that capturing at B>kMaxBatch raises.
5. **Negative tests:** non-CUDA tensor, non-contiguous, wrong dtype, B>max_batch,
   calling an L2-persisting/`cudaDeviceSynchronize` path inside capture (assert
   we DON'T — i.e. a capture of any op succeeds, proving no illegal op leaked
   into the captured region).
6. Gate the whole module on `pytest.importorskip("torch")` +
   `torch.cuda.is_available()` + `nvcc` on PATH, like the JAX smoke test.

---

## 6. Open questions / risks

1. **[BIGGEST] L2-persisting + capture interaction must be proven, not
   assumed.** The plan asserts the kernel-direct launch path (copied from the
   JAX handlers) never invokes `grid_begin_l2_persisting` or
   `cudaDeviceSynchronize`, so capture is legal. This is true for the *current*
   JAX handler bodies — but several handlers launch kernels that, in the host
   wrapper, are paired with an L2-persisting call (id_du `grid.cuh:10510`,
   fd_du `:11624`, integrator `:14205`). The torch op skips the host wrapper,
   so it ALSO skips the L2-persisting setup — meaning the captured kernel runs
   **without** the L2 access-policy window the perf numbers assume. **Open Q:
   do the gradient/integrator kernels under-perform when launched kernel-direct
   (no L2 window) vs via the host wrapper?** If yes, we need to set the L2
   window once in warmup on the capture stream (capturable-safe because it's
   outside the captured region) and confirm the window persists across
   `.replay()`. Must measure on sm_120 once a GPU is free. This is the one
   design point that could force a structural change.

2. **Torch ABI vs our hand-rolled nvcc build.** `torch.ops.load_library` on a
   .so we built with our own `nvcc` command (not torch's `cpp_extension`)
   requires the torch registration symbols to be ABI-compatible. If
   `TORCH_LIBRARY` macros need torch's exact compile flags
   (`-D_GLIBCXX_USE_CXX11_ABI=...`, etc.), we must query
   `torch.utils.cpp_extension` for them and add to the nvcc command. Risk:
   subtle ABI mismatch → load-time symbol errors. Fallback: drive the build
   through `cpp_extension.load` entirely (option A), losing some control over
   the GLASS include wiring. Prototype both on iiwa14 early.

3. **T4 `d_f_ext` signature change is landing into the shared launch layer.**
   T4 (`HANDOFF.md:1182`, ratified `:1226-1230`) adds a trailing `d_f_ext`
   to RNEA/FD/ABA/ID-grad/FD-grad **kernel signatures** (the f_ext is read
   once in the inner; `*_DYNAMIC_SHARED_MEM_BYTES` stay byte-identical, no tier
   perturbation). Every torch op that launches those kernels must pass the new
   arg — **another reason to extract one shared launch-core helper** (§4) so
   T4 is threaded once for JAX+torch+the plain C ABI. Sequence D.3 *after* T4
   merges, or build against the pre-T4 signature and rebase.

4. **T5 `TIER_PERF→TIER_SHARED` rename.** T5 (`HANDOFF.md:1183`) renames the
   tier value (keeps the `RESOURCE_TIER` template param) and ships a
   `TIER_PERF=TIER_SHARED` alias (`HANDOFF.md:1204-1205`). The torch shim
   references tiers only indirectly via `*_DYNAMIC_SHARED_MEM_BYTES<T>()`
   (which take `TIER = GRID_DEFAULT_RESOURCE_TIER`, `grid.cuh:350`), so the
   rename is transparent **as long as we don't hardcode `TIER_PERF`**. The
   captured launch config inherits T5's autotuned default tier — good, but
   means the capture must happen *after* register-time autotune picks the tier
   (it does; tier is baked at codegen/register).

5. **Single-robot singleton.** Same limitation as JAX/pybind
   (`_core.cpp:19-24`): one `.so` = one set of device buffers. Two captured
   graphs sharing the singleton scratch can't run concurrently. Acceptable for
   v1 (MPC loops are single-robot); flag per-handle device state as v2.

6. **`integrator` device-scalar `dt` for capture.** Recommended (§3) so MPC
   can vary `dt` without recapture. Low-risk (dt is already a runtime kernel
   arg, `wrapper_template.cu:1149`) but adds one input tensor + a small kernel
   shim. Decide whether to ship it in v1 or defer.

7. **`idsva_so` floating-base dispatch.** The JAX handler hardcodes the
   body-frame kernel (`wrapper_template.cu:1052-1062`, with a TODO for
   floating-base `#define` dispatch). The torch shim inherits this limitation;
   floating-base `idsva_so`/`fdsva_so` parity is blocked on the same codegen
   `#define` follow-up. Scope v1 torch to fixed-base SO (matching current JAX).

---

## 7. Suggested implementation order

1. Extract a shared per-algorithm launch-core helper in `wrapper_template.cu`
   (refactor JAX handlers to call it) — de-risks T4 and torch at once.
2. Add `#ifdef GRID_RBD_WITH_TORCH` block (forward ops only) + `-DGRID_RBD_WITH_TORCH`
   gate in `_compile.py` + `with_torch`/torch-version in the cache key.
3. `grid_rbd/torch/__init__.py`: `TorchRobotHandle` + `register_robot`/`get_robot`
   mirroring `jax/__init__.py`, forward-only methods first.
4. Parity smoke test (forward) vs numpy + JAX.
5. Autograd `Function`s for the 4 differentiable algos + FD-vs-analytic test.
6. `GraphCallable` capture + capture/replay tests + the L2-window measurement (risk #1).
7. Docs + the device-scalar-`dt` integrator capture variant (optional v1).
