# Changelog

GRiD has never been published; this changelog tracks notable in-tree
changes since the GLASS rollout for our own historical reference.

## Unreleased — cuBLASDx removal + any-thread-count

**Archive tag for pre-rip state:** `archive/last-cublasdx` (commit
`5177070`). Use `git show archive/last-cublasdx -- <path>` to see the
exact pre-rip content of any file.

**Design doc:** [docs/source/user_guide/concepts/cublasdx_removal_design.rst](docs/source/user_guide/concepts/cublasdx_removal_design.rst).

### Why

The 2026-05-18 per-host-autotuned sweep showed cuBLASDx losing to SIMT
across every algorithm × robot × base in the bench (notably 2.6× behind
SIMT on the 4×4×4 GEMM inside `end_effector_pose_gradient_hessian`).
With cuBLASDx gone, every emitted kernel can drop
`__launch_bounds__(SUGGESTED_THREADS)`, which unblocks the larger goal:
**CUDA-inline users can now call `grid::*_kernel<T><<<…, anysize>>>`
without thread-count constraints**.

### Codegen / generated headers

- Removed the `glass_nvidia` backend, `GRID_CUDA_LINALG_BACKEND` macro,
  `GRID_CUBLASDX_HEADER_AVAILABLE`, `GRID_CUSOLVERDX_HEADER_AVAILABLE`,
  `GRID_CUDA_USE_GLASS_NVIDIA`, and all related dispatch logic.
- Generated headers now vendor only the SIMT GLASS subset
  (`L1/dot`, `L2/gemv`, `L3/gemm`). No external SDK dependencies.
- Stripped `__launch_bounds__(SUGGESTED_THREADS)` from every emitted
  kernel. `SUGGESTED_THREADS` is preserved as a *hint*, not an enforced
  floor.
- Inner functions already used block-stride loops via
  `gen_add_parallel_loop`, so any block size works correctly. Smaller
  block sizes are slower (block-cooperative work amortized over fewer
  threads); larger block sizes are limited only by the per-block max
  (1024 on current GPUs).

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
