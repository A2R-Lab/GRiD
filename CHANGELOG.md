# Changelog

GRiD has never been published; this changelog tracks notable in-tree
changes since the GLASS rollout for our own historical reference.

## Unreleased — v2.0 — cuBLASDx removal + resource-tier system

**Archive tag for pre-rip state:** `archive/last-cublasdx` (commit
`5177070`). Use `git show archive/last-cublasdx -- <path>` to see the
exact pre-rip content of any file.

**Design docs:**
- [docs/source/user_guide/concepts/cublasdx_removal_design.rst](docs/source/user_guide/concepts/cublasdx_removal_design.rst) — original cuBLASDx-removal rationale.
- [docs/source/user_guide/concepts/resource_tier_system.rst](docs/source/user_guide/concepts/resource_tier_system.rst) — tier system shipped on top, plus the deferred LITE-48KB follow-up.

### Why

The 2026-05-18 per-host-autotuned sweep showed cuBLASDx losing to SIMT
across every algorithm × robot × base in the bench (notably 2.6× behind
SIMT on the 4×4×4 GEMM inside `end_effector_pose_gradient_hessian`).
With cuBLASDx gone, **CUDA-inline users get tier-controlled resource
profiles via `grid::*_kernel<T, RESOURCE_TIER><<<…>>>` and the
matching tier-aware `_device` / `_inner` functions** — picking the
`(launch_bounds, smem footprint, register cap)` trade-off that fits
their outer kernel.

### Codegen / generated headers

- Removed the `glass_nvidia` backend, `GRID_CUDA_LINALG_BACKEND` macro,
  `GRID_CUBLASDX_HEADER_AVAILABLE`, `GRID_CUSOLVERDX_HEADER_AVAILABLE`,
  `GRID_CUDA_USE_GLASS_NVIDIA`, and all related dispatch logic.
- Generated headers now vendor only the SIMT GLASS subset
  (`L1/dot`, `L2/gemv`, `L3/gemm`). No external SDK dependencies.
- Added `RESOURCE_TIER` non-type template parameter (default `TIER_PERF`)
  to every emitted `__global__` kernel + the new inline-CUDA `_device`
  / `_inner` functions listed below. `__launch_bounds__` is now
  `tier_max_threads<RESOURCE_TIER>()` — at TIER_PERF this resolves to
  `SUGGESTED_THREADS` (= current behavior, byte-identical to pre-tier
  build); at TIER_LITE to `min(2*SUGGESTED, 768)`; at TIER_MINIMAL to
  `1024` (the sm_120 hardware thread cap).
- New inline-CUDA tier-aware surfaces (each takes
  `template <typename T, int RESOURCE_TIER = TIER_PERF>` + a caller-
  provided `T *s_workspace = nullptr` arg):
    - `fdsva_so_inner` — 4*nv³ inner scratch routes between s_temp
      (PERF) and s_workspace (LITE/MINIMAL).
    - `forward_dynamics_gradient_device` — whole s_temp arena routes
      per tier via `if constexpr` in `gen_declare_shared_arena`.
    - `inverse_dynamics_gradient_device` — same pattern.
    - `end_effector_pose_gradient_hessian_device` — d2eeTemp slot
      routes per tier; inner_no_d2 stays in smem.
    - `idsva_so_device` (new) — codegen-time frame dispatcher (body
      for fixed-base, world for floating-base) mirroring the host-
      level `idsva_so` dispatcher.
- New per-tier sizing constexprs (call from host to allocate the right
  buffers): `FDSVA_SO_INNER_*`, `FD_DU_DEVICE_INLINE_*`,
  `ID_DU_DEVICE_INLINE_*`, `D2EE_DEVICE_INLINE_*`,
  `IDSVA_SO_DEVICE_INLINE_*` — each as
  `*_SMEM_BYTES<T, TIER>()` and `*_WORKSPACE_BYTES<T, TIER>()`.
- Reusable arena helper:
  `gen_declare_shared_arena(tier_workspace_expr=...)` makes the s_temp
  slot conditionally route to a caller-provided pointer at TIER_LITE+
  via `if constexpr (RESOURCE_TIER == TIER_PERF)` branches.
- Inner functions already used block-stride loops via
  `gen_add_parallel_loop`, so any block size in `[1, tier_max_threads]`
  works correctly. Smaller block sizes are slower (block-cooperative
  work amortized over fewer threads); larger block sizes need
  TIER_LITE or TIER_MINIMAL.

### Humanoid-scale spill (`humanoid-tier-spill`, landed 2026-05-23)

- **TIER_LITE now picks a genuine intermediate smem rung** (~48 KB
  target via `select_shared_tier_3way`), no longer an alias of
  TIER_MINIMAL on the smem axis.
- **Every previously-overflowing kernel now fits the sm_120 ~100 KB
  cap at all tiers** via surgical per-tier spill: Minv/FD (Minv-F),
  ABA (whole inner), EE_POSE_GRAD/D2EE/id_du/fd_du/fdsva_so (3-6 level
  ladders), **idsva_so body+world** (output → BC → whole inner), and
  the **time-integrator value + gradient**. Value spills the FD
  inner's Minv-F; the gradient uses a 4-rung ladder (Dqdd → +dAB +
  id_du-selective → +whole inner). Validated: iiwa14 @ MINIMAL,
  g1_floating @ PERF (selective rung), tier_instantiation_smoke on
  iiwa14/go2/h1_2. See `resource_tier_system.rst` + `HANDOFF.md`.
- Still open: a finer hot/cold de-alias of the monolithic
  idsva_so/fdsva_so inners (so MINIMAL keeps more hot data in smem),
  the ABA surgical retrofit, and tuning the 48 KB LITE target via the
  deferred full sweep. See `HANDOFF.md` → Backlog.

### Warnings

- RBDReference `mxS` flattens its subspace column to 1-D so `mx1-mx6`
  receive a scalar `alpha` — fixes the NumPy `ndim>0 to scalar`
  DeprecationWarning at the source (was ~177k/run in the integrator
  equivalence suite).

### Python / JAX wrappers

- New API: `RobotHandle.suggested_threads` /
  `RobotHandle.threads_per_block` properties +
  `RobotHandle.set_threads_per_block(n)` method.
- C ABI gains `grid_rbd_suggested_threads`,
  `grid_rbd_threads_per_block`, `grid_rbd_set_threads_per_block`.
- JAX FFI handlers continue to use the wrapper's `g_thread_dimms`, so
  the same setter affects JAX launches transparently.
- No public-API breakage: existing call sites keep working at
  SUGGESTED_THREADS by default.

### Bench harness

- Dropped the `glass_nvidia` column from
  `test/benchmarks/run_multi_version.py` (`--columns` choices).
- Removed `--linalg-backend`, `--mathdx-root`, `--with-cusolverdx`,
  `--cicc-opt-level` flags from `test/benchmarks/baselines/grid/run.py`,
  `test/benchmarks/run_multi_version.py`,
  `test/benchmarks/run_benchmarks.py`,
  `test/benchmarks/run_overnight_sweep.sh`.
- `test/benchmarks/generate_report.py` no longer renders the `glass_nv`
  column or `glass_nv/glass` ratio.

### Tests

- New: `test/python_wrappers/test_any_thread_count.py` — sweeps every
  bound method on iiwa14_fixed at block sizes {64, 128, 256,
  SUGGESTED_THREADS, 512} and asserts numerical agreement vs the
  SUGGESTED_THREADS reference.
- New: `test/benchmarks/any_thread_count_microbench.py` — committed
  artifact that times RNEA at each block size for iiwa14, go2, g1,
  h1_2. Run with
  `.venv/bin/python test/benchmarks/any_thread_count_microbench.py`.
- Deleted six cuBLASDx-specific tests from
  `test/cuda_equivalents/test_cuda_codegen_layout.py`
  (`test_linalg_backend_glass_nvidia_requires_sm_macro`,
  `test_linalg_backend_required_cublasdx_requires_header`,
  `test_linalg_backend_auto_cublasdx_compiles_with_mathdx`,
  `test_linalg_backend_nvidia_row_strided_helpers_compile_with_mathdx`,
  `test_linalg_backend_gemv_and_transpose_b_compile_with_mathdx`,
  `test_fixed_kinematics_derivative_wrappers_compile_cublasdx`).

### Docs

- New page:
  `docs/source/user_guide/concepts/cublasdx_removal_design.rst` —
  rationale, scope, revert path, deferred research questions.
- Updated:
  `docs/source/user_guide/concepts/codegen_architecture.rst` —
  thread-count section now describes block-cooperative compute +
  grid-stride batching + the tens-to-hundreds batch-size sweet spot.
- Stripped libmathdx / cuBLASDx setup from
  `docs/source/user_guide/getting_started/installation.rst`,
  `docs/source/user_guide/getting_started/docker_setup.rst`,
  `docs/source/user_guide/tutorials/cuda_validation.rst`,
  `docs/source/user_guide/tutorials/benchmarks.rst`,
  `docs/source/user_guide/tutorials/python_wrappers.rst`,
  `README.md`, `python/README.md`, `test/benchmarks/README.md`.

### GLASS submodule

**Not touched.** GLASS as a standalone library keeps its cuBLASDx
support (`GLASS/glass-nvidia.cuh`, `GLASS/src/nvidia/`,
`GLASS/bench/autotune.py`, the tuning tables). External users of
GLASS for non-RBD GEMM workloads may legitimately want it. Removing
the integration at the GRiD side is the right separation.

### Deferred (v1.x research backlog)

The design doc captures three follow-on opportunities preserved for
future work — all SIMT-friendly, none require re-introducing cuBLASDx:

1. **`fdsva_so` `iL,Ljk` contraction layout refactor** — the one place
   cuBLASDx might pay off at humanoid DOF (h1_2: ~27M FMAs for the
   final contraction), if the shared-memory layout is restructured
   first.
2. **Limb-parallel output sparsity** in gradient / SO tensor writes
   for branched topologies (humanoids, quadrupeds).
3. **Limb-batched SIMT dispatch** for spatial-algebra ops.

## v0.3 — JAX FFI parity (2026-05-18, commit `bc5d092`)

- All 12 plain-wrapper methods now bound via JAX FFI, device-resident,
  `jax.jit`-compatible.
- Docs restructure: homepage cards by user-path (Python / JAX / raw CUDA);
  new `codegen_architecture.rst` page documenting the four emission
  layers; orphan pages cleaned up.
- 29 JAX smoke tests pass at float32 precision vs the plain wrapper.
