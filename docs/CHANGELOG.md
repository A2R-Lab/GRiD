# Changelog

GRiD has never been published; this changelog tracks notable in-tree
changes since the GLASS rollout for our own historical reference.

## Unreleased — v2.0 — cuBLASDx removal + resource-tier system

### 2026-09-08 (catch-up: 08-28 → 09-08)
- Wrapper collapse (P1): C-ABI bodies, kernel_max_threads table, and all 30 mjx
  twins now EMITTED from grid_codegen/abi_specs.py into three checked-in
  generated regions of wrapper_template.cu (drift-gated in CI).
- B2: kernel-attr manifest registers divergent-BAKED-tier instantiations
  (baked_launch_cfg = single load path with the launch_cfg<> bake).
- B1/E5: per-tier autotune matrix blocks written for all 6 roster robots.
- H6 slice 1: bench ALGOS arity + SIG_MJX flag dict derive from ABI_SPECS.
- Bench: XLA prealloc disabled in standalone drivers (launch-time OOM class);
  autotune skips unlaunchable kernels loudly; fresh full receipt @ night-7.
- C0/C1/C2 cleanup: dead code deletes, archive moves, docs truth fixes
  (see docs/open-tasks/cleanup_wave_2026-09-07.md).

### 2026-07-21 — single-distribution packaging + project hygiene

- **One `pip install -e .`:** the codegen toolkit and the `grid_rbd` Python
  wrapper are now a single distribution. The nested `bindings/pyproject.toml` +
  `setup.py` folded into the root project (name `grid-rbd`, Python ≥ 3.10,
  MIT); the `grid_rbd._core` pybind11 extension builds from a minimal root
  `setup.py`. The old two-step (`pip install -e .` then `pip install -e
  bindings/`) is gone; install docs updated repo-wide. `grid_rbd.__version__`
  is now single-sourced from the installed metadata (was a hand-maintained
  0.4.0 that had drifted from the distribution). The unified distribution
  version is **0.5.0** (codegen and wrapper were previously at incoherent
  1.0.0 / 0.4.x lines).
- **pyproject metadata:** added authors / keywords / classifiers / URLs.
- **Codegen cleanup:** removed 17 dead arena-size locals from
  `GRiDCodeGenerator.py` (generated `grid.cuh` byte-identical on
  iiwa14/go2/fr3/h2_plus), a duplicated fd-gradient TODO, and renamed the
  misleadingly-named `grid_codegen/_test.py` → `_reference_impl.py` (it is a
  numpy reference-impl mixin, not a pytest module). Codegen-time diagnostic
  switched from `print` to `warnings.warn`.
- **Docs coherence:** fixed broken `external/` + `install/` paths, the stale
  manual clone/submodule install block, old `robot-acceleration` org links
  (→ `A2R-Lab`), unpublished `pip install grid-rbd` snippets, the Python
  version badge, and RST link syntax; removed a tracked `docs/.DS_Store`.

### 2026-07-07 — tooling + codegen maintainability

- **GPU-proof receipts now via PyPI:** the `pytest-gpu-proof` plugin moved
  from a vendored `test/pytest-gpu-proof` submodule to the published PyPI
  package (`requirements-dev.txt`); CI install + `test/conftest.py` +
  `run_gpu_proof.sh` updated, and the verify-receipt CI job no longer needs
  submodules (the fingerprint pins them by gitlink SHA). See
  `tutorials/cuda_validation.rst`.
- **Descriptor table (per-algo metadata unification), Steps 1+2:** one
  `AlgoDescriptor` row per algorithm in `algo_registry.py` is now the single
  source of truth driving the `GridAlgo` enum, the launch-config symbol map,
  and the `KERNEL_ATTR_MANIFEST` / mjx manifest heads — replacing several
  hand-maintained module dicts. Byte-identical generated output;
  `test/test_algo_descriptor_parity.py` locks the table. Step 3 (arena/spill) is
  now landed too: per-algo arena/spill `t_count` math lives in the table
  (`ArenaRegion`/`SpillRung`/`compose_arena_*`) and drives every
  `select_shared_tier_3way` site; `test/test_shared_arena_covers_carve.py` guards
  it independently (launch macro vs the kernel's actual carve). See
  `concepts/codegen_architecture.rst`.

### 2026-06-07 — new value ops, de-gating, runtime params, joint types

- **New algorithms (codegen + CUDA-validated):** Coriolis matrix
  `C(q,q̇)`; kinetic + potential energy regressors; `dccrba` (∂A/∂q
  tensor) + `cmm_time_variation` (Ȧ); runtime arbitrary multi-EE
  (`end_effector_pose_runtime` + `_gradient`, with runtime target joint
  id + per-target offset).
- **New analytic oracle:** `RBDReference.dccrba` is now the analytic
  ∂A/∂q tensor (replacing the prior finite-difference oracle).
- **De-gating (per-robot gating essentially eliminated):** the
  centroidal family (`com` / `ccrba` / `energy` / `dccrba` /
  `cmm_time_variation`) now runs on **mimic** robots, and
  `dccrba`/`cmm_time_variation` now run on **big floating-base** robots
  (`g1` / `h1_2`-floating) via sweep-pool spill.
- **Bindings:** `qdd` wired through `inverse_dynamics` (numpy/jax/torch)
  + qdd-aware autograd gradient (returns the correct ∂τ/∂(q,q̇)
  including ∂(M·q̈)/∂q); `rnea`/`fd` aliases; `SecondOrderID` /
  `SecondOrderFD` NamedTuples; JAX `f_ext` parity; fp64-in/out
  convenience cast (`allow_fp64=`, compute stays fp32); compile-progress
  logging + import-guards. The five new value ops
  (`coriolis_matrix`, `kinetic_energy_regressor`,
  `potential_energy_regressor`, `dccrba`, `cmm_time_variation`) are
  exposed on the numpy `RobotHandle`.
- **Runtime-mutable inertia (flag-gated):** an opt-in `d_inertia_params`
  table + `set_inertia_params` device entry for sysID /
  domain-randomization with no recompile; the baked default path is
  byte-identical.
- **Joint types (stage 1):** an arbitrary/skew `<axis>` (non-cardinal)
  is supported via a dense 6-vector motion subspace `S`, for
  `inverse_dynamics` and `crba` only so far (cardinal axes
  byte-identical; helical / planar / spherical and the other algorithms
  are later stages).
- **Perf:** the RNEA-backward (#2) and #6 experiments measured neutral
  (Amdahl) and were reverted/dropped — no user-facing change.

### 2026-05-31 — frame Jacobian / OSC, mimic gradients, SO parallelization

- General-frame geometric Jacobian (`LOCAL`/`WORLD`/`LOCAL_WORLD_ALIGNED`) added: numpy reference `frame_jacobian` + `frame_jacobian_dot` (J̇) + `osc_inertia` (Λ=(J·M⁻¹·Jᵀ)⁻¹), validated vs pinocchio `getFrameJacobian`/`getJointJacobian`/`computeJointJacobiansTimeVariation` across all three frames on iiwa14 + go2.
- CUDA codegen emits the frame Jacobian J (`frame_jacobian`), J̇ (`frame_jacobian_dot`), and OSC Λ (`osc_inertia`) as opt-in keys (require `ee_pose`, non-mimic robots), each validated on-device vs the numpy reference across all three frames on iiwa14 + go2 + g1. The Λ kernel takes a precomputed M⁻¹ (on-device `direct_minv_inner` compose is a follow-up).
- Floating-base mimic robots now support `ee_pose_gradient` / `ee_pose_hessian` codegen (the 6 independent root v-slots decompose into singleton columns).
- Fixed-base mimic robots now support second-order codegen (`idsva_so` / `fdsva_so`) via the body-frame internal-NUM_BODIES sweep with an alpha-fold to the reduced output.
- Fixed a shared `matmul` block-index wrap bug (`%NUM_JOINTS` → `%NUM_BODIES`) that corrupted the composite-inertia path on mimic robots.
- Parallelized `id_du` and the world-frame `idsva_so` inner; deduplicated the `fdsva_so` emitter.

### 2026-05-30 — external forces, plant layer, torch backend, tier rename

This subsection collects the features merged onto `modernizing-tests`
(parent `164b655`) after the initial tier rollout below.

**External forces (`f_ext`).** Opt-in per-body external forces are now
threaded through the CUDA codegen and the Python reference:

- A new trailing `T *d_f_ext` arg on RNEA / forward_dynamics / ABA /
  `inverse_dynamics_gradient` (id_du) / `forward_dynamics_gradient`
  (fd_du) and the integrator surfaces. It is **GLOBAL, body-major
  (`6*NUM_BODIES`), local-frame**, and **subtracted** from the per-body
  force at a single site. `nullptr` (the default) reproduces the prior
  no-fext path byte-for-byte.
- `RBDReference` gains `apply_external_forces(f_in, f_ext)` and an
  `f_ext=` kwarg on `rnea`/`rnea_fpass`/`aba` (same subtract convention
  as the CUDA path and GATO/pinocchio `fext`). An empty/`None` `f_ext`
  is a no-op.
- f_ext **gradients** are not yet wired (roadmap; see
  `docs/open-tasks/`).

**`grid_plant` cost / barrier / plant-step layer + Python surface.**

- `_plant.py` emits a sibling `namespace grid_plant { ... }` after the
  `grid` namespace: `plant_step` / `plant_step_gradient[_and_value]`
  (thin wrappers over `grid::integrator`), quadratic state/input costs,
  `ee_pos_cost` (Gauss-Newton hessian `J_pᵀ W J_p`), and joint
  position/velocity/torque log-barriers (value + grad + diag-hessian).
- These are now callable from the Python `RobotHandle`:
  `quadratic_state_cost`, `quadratic_input_cost`, `ee_pos_cost`,
  `joint_position_barrier`, `joint_velocity_barrier`,
  `joint_torque_barrier`, `plant_step`. Validated against
  `RBDReference._PlantMixin`.

**PyTorch backend (`grid_rbd.torch`).**

- `register_robot(..., backend="torch")` returns a `TorchRobotHandle`
  whose four differentiable algorithms (rnea / forward_dynamics / aba /
  integrator) are autograd-aware torch ops with analytic backward
  passes that reuse the existing `*_gradient` kernels; the rest are
  forward-only ops. `register_robot` now also accepts
  `backend="numpy"|"jax"|"torch"` and `urdf_string=` (inline URDF).
- CUDA-Graphs capture/replay via `handle.capture(method, *example_inputs)`
  → `GraphCallable` for fixed-batch low-launch-overhead replay (MPC /
  training), with a mandatory off-graph warmup for the >48 KB dynamic-
  smem opt-in.
- The torch op block is compiled into the shared `.so` under
  `-DGRID_RBD_WITH_TORCH` when torch is present at register time.
- **Env caveat:** torch's own backward kernels (bmm/eye) must support
  the GPU arch, so the installed torch build must match it — on sm_120
  (RTX 5090) use a torch cu128 (or newer) build; a cu124 wheel cannot
  launch on sm_120.

**Resource-tier rename + autotune.**

- `TIER_PERF` → `TIER_SHARED` everywhere (it is the default tier).
  A deprecated alias `constexpr int TIER_PERF = TIER_SHARED;` is still
  emitted so sibling branches keep compiling; prefer `TIER_SHARED` in
  new code.
- New post-codegen joint **(tier × threads) autotune picker** in
  `test/benchmarks/baselines/grid/run.py`: it argmin's per-tier timing
  data (with per-tier thread caps: SHARED = max-perf threads,
  LITE = `min(2×, 768)`, MINIMAL = 1024) and emits a schema-2
  `algo_picks` block. Covered by
  `test/benchmarks/test_autotune_picker.py`.

**Codegen consolidation (internal).**

- Shared emitters `gen_device_wrapper` (×7) and `gen_tier_dispatch`
  (×12) replace the per-algorithm copies; output is byte-identical
  across all robots.

**`pin_so_ext` self-building.**

- `RBDReference/equivalents/pin_so_ext/` (the second-order pinocchio
  oracle) now builds on demand, prepending its own `PKG_CONFIG_PATH`,
  so fresh clones need no manual build step before the SO equivalence
  tests run.

**RBDReference numpy reference oracles.**

- New mixins validated against pinocchio, available as a reference /
  oracle surface: `_energy.py` (generalized gravity, nonlinear
  effects, kinetic/potential/mechanical energy, Coriolis matrix),
  `_centroidal.py` (CoM, CoM Jacobian, CCRBA, centroidal momentum),
  `_regressor.py` (joint-torque regressor), `_plant.py` (the plant /
  cost / barrier reference for the CUDA `grid_plant` layer).

**G0 mimic-gradient guard.**

- Gradient codegen for robots with **mimic joints** now raises a clear
  `NotImplementedError` instead of silently emitting zeroed gradients.
  Non-gradient algorithms work for mimic robots; mimic gradients (and
  floating+mimic) are deferred (roadmap; `docs/open-tasks/`).
  *(Superseded by the 2026-05-31 entry above: `id_du`/`fd_du`,
  `ee_pose_gradient`/`ee_pose_hessian`, and fixed-base second-order now
  emit correct mimic-reduced gradients; only integrator/`f_ext`
  gradients and floating-base second-order still refuse.)*

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
