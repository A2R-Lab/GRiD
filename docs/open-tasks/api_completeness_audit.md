# GRiD API Completeness & Gap Audit (2026-06-01)

Synthesis of a 3-agent read-only audit (core-dynamics / gradients-SO / kinematics-centroidal-plant).
Goal: a UNIFIED, COMPLETE, CORRECTLY-NAMED API with FULL test coverage. Backwards-compat is NOT a
concern — we rename to the correct name and delete the old one. Single-block-per-function stays.
This doc supersedes the stale rows in `test_coverage_matrix.md` / `coverage_parity_matrix.md`.

## A. Surface completeness — DEVICE-ONLY functions (no host/kernel/timing → unbenchmarkable + unbindable)
- **`frame_jacobian`** (inner+device), **`frame_jacobian_dot`** (device only), **`osc_inertia`** (device
  only). All have numpy+pin refs and a CUDA smoke, but NO `_kernel`/`_host`/`_single_timing`/
  `_compute_only`/batch. → can't be in PER_ALGO_SPECS, can't be benchmarked, not in bindings.
  **`frame_jacobian_dot` is in ALGO_REGISTRY but not PER_ALGO_SPECS → emits a "missing spec" warning.**
- `grid_plant::*` are device+kernel only (binding kernels, benchmark-exempt by design — OK).
- CORRECTION: `com`/`ccrba`/`energy`/`generalized_gravity`/`nonlinear_effects` are **full-surface +
  benchmarkable** (NOT device-only — earlier matrices were wrong).
- **Action:** emit kernel+3-mode host+batch wrappers + a gridData output buffer for frame_jacobian/
  frame_jacobian_dot/osc_inertia. Prereq for benchmarking + the HJCD "parallelize frame_jacobian" ask.

## B. Naming / uniformity — the clean-break rename (pervasive; pick ONE canonical name per algo)
Registry-key ≠ emitted host-symbol almost everywhere:
- `id`→`inverse_dynamics` (+ stray `rnea*` aliases), `fd`→`forward_dynamics`, `minv`→`direct_minv`
  (bench calls `direct_minv_single_timing`, 3 spellings), `id_du`→`inverse_dynamics_gradient`,
  `fd_du`→`forward_dynamics_gradient`, `regressor`→`inverse_dynamics_regressor`,
  `ee_pose`→`end_effector_pose`, `ee_pose_gradient`→`end_effector_pose_gradient`,
  `ee_pose_hessian`→`end_effector_pose_gradient_hessian` (host carries a `_gradient_` the key lacks).
- **`integrator_with_gradient` 4-way clash**: key / host `integrator_gradient_with_x_kp1` / printf / bench all differ.
- algorithm_list token ≠ key: **`f_ext_grad`** vs `f_ext_gradient`.
- `idsva_so_world_frame` is NOT a first-class algorithm_list key (flag-gated via `enable_*` only) while
  `idsva_so_body_frame` is — asymmetric.
- `_host` suffix only on `idsva_so_body_frame_host`/`idsva_so_world_frame_host` (no sibling has it).
- Missing uniform `_inner`: `fd_du` (reuses id_du band), `f_ext_gradient_dq` (kernel-only inline FD),
  `integrator_gradient`/`integrator_with_gradient` (use `_multistage` / `_inner_python`).
- Signature non-uniformity:
  - `fdsva_so_kernel`: `d_workspace` is 4th param (after stride), not 2nd; threads `d_idsva_so` as a
    5th input/scratch param. → reorder to `(out, d_workspace, in, stride, …)`.
  - `regressor`/`fd_parameter_gradient` host+kernel take a **caller-owned output buffer** (`d_Y`,
    `d_dqdd_dpi`) NOT in gridData → bench must cudaMalloc TU-static buffers. Move outputs into gridData.
  - `com`/`ccrba` host take **no `gravity`** while energy/gg/nle do → family not uniformly callable.
  - integrator hosts carry `dt` (necessary) → dynamics-vs-integrator host signature blocks diverge.
- **Gravity-sign convention split**: GRiD energy/centroidal PE uses `+gravity` (+9.81) while
  RBDReference uses `GRAVITY=-9.81` — a real cross-surface convention wart tests have to flip for.
- `com`/`ccrba`/`energy`/`gg`/`nle` are emitted gated on a SIBLING key (`ee_pose`/`id`), not their own
  registry key — requesting `['id','ee_pose']` silently emits the whole centroidal family.
- aba `s_va` docstring says NUM_BODIES but allocates `12*NUM_JOINTS` (doc-only; validated correct).
- **Action:** one canonical name per algo = registry key == host symbol == bench symbol == algo-list
  token; uniform `(hd_data, d_robotModel, [gravity], [dt], …, d_workspace, …)` signature shape; uniform
  `_inner`→`device`→`kernel`→`host`→`single_timing`/`compute_only`→batch surface set; outputs in gridData;
  drop `_host` suffix; make idsva_so_world_frame + frame_jacobian* first-class keys; remove the
  SUGGESTED_THREADS back-compat alias.

## C. Test-coverage gaps (full-coverage mandate)
- **C1 (HIGH, correctness):** `ee_pose_hessian` orientation-hessian KNOWN BUG (CUDA + analytic share
  it); pin parity scoped to {iiwa14,fr3}. Fix the orientation block, then widen robots.
- **C2 (HIGH):** `fd_parameter_gradient` numpy ref is **never asserted vs any oracle** (regressor test
  checks only Y, never the −M⁻¹Y compose). Add a numpy equivalence test (oracle pieces exist).
- **C3 (HIGH):** `com_cost`/`momentum_cost` have **no CUDA equivalence test** (plant smoke runner makes
  0 calls). numpy + FD tests now exist; the emitted device kernels are CUDA-unverified.
- **C4 (MED):** centroidal CUDA breadth — only iiwa14-fixed + go2-floating; `gg`/`nle` support mimic but
  are never CUDA-checked on a mimic robot (= the backlog-D mimic-safe runner; the existing centroidal
  runner is non-mimic-only because it drives com_cost/momentum_cost).
- **C5 (MED):** integrator family CUDA tested only on small robots (iiwa14,go2,fr3) — no g1/h1_2 spill paths.
- **C6 (RESOLVED 2026-06-01):** floating-mimic SO desync investigated. Codegen emits floating-mimic
  `idsva_so` (WORLD-frame inner) + `fdsva_so`; the stale CUDA exe test gate (`MIMIC_FLOATING_UNSUPPORTED_GRADIENTS`)
  listed both as unsupported. Verified vs the pin_so_ext oracle on fr3-floating (+ go2-floating non-mimic control):
  - **idsva_so (world-frame): CORRECT** (rel ~5e-7). Removed from the gate; already exercised by
    `test_cuda_idsva_so_world_frame.py` (fr3 is its floating-mimic sentinel — confirmed passing).
  - **fdsva_so: REAL floating-base bug** the desync was hiding. Its `daba_dqdq` (q-q) block is wrong
    (rel ~6e-2 fr3-floating, ~4.4e-2 go2-floating NON-mimic → floating-base bug, NOT mimic-specific;
    fixed-base fr3 fdsva PASSES). Root-caused: idsva inputs (di2_dq, dM_dq) + s_df_dq are each correct
    and the contract formula is correct (numpy emulation of the exact kernel math → pin to ~6e-7), so the
    in-kernel scratch the contract consumes is corrupted on the fused floating path. The structural lead:
    `_fdsva_so.py:325-329` — fixed-base runs `gen_idsva_so_body_frame_public_dvdq_layout_repair()` after its
    idsva inner; floating-base (world inner) applies NO repair, and the inline FD-gradient runs before the
    world inner sharing `s_temp`. `fdsva_so` stays refused for floating-mimic (and is broken for ALL
    floating-base) until that fused-path scratch/repair is fixed. NEW open item: **C6b — floating-base
    fdsva_so daba_dqdq bug (all floating robots).**
- **C7 (LOW):** `integrator_with_gradient` no standalone numpy test (covered transitively). FK-batched
  (`ee_pose_fk_batched`) has no equivalence diff vs `end_effector_pose` (binding coverage ≠ correctness).

## D. Feature gaps (completeness)
- **`plant_step_hessian` MISSING** — the only entirely-absent plant 2nd-order primitive; gated behind a
  missing analytic 2nd-order integrator (`_plant.py:82-87 TODO(plant-2nd-order)`). Cost layer uses
  Gauss-Newton hessians (all present). This is the #1 customer ask (PDDP `compute_fxx` + GATO exact-Newton).
- **Bindings incomplete:** NOT exposed in pybind/JAX/C-ABI: frame_jacobian, frame_jacobian_dot,
  osc_inertia, com, ccrba, energy, generalized_gravity, nonlinear_effects, com_cost, momentum_cost,
  **plant_step_gradient** (its value counterpart `plant_step` IS exposed — asymmetric).
- **floating-mimic `integrator_gradient`/`integrator_with_gradient` CUDA-refused** (multi-stage RK
  stage-projection bug; numpy ref handles it, so CUDA-only hole) — the only remaining true mimic refusal.
- Big-robot perf: crba/aba/minv/fd lose to Pinocchio CPU on h1_2 (in-block parallelism only; `id` wins 5×).

## E. Proposed sequence (single-block kept throughout)
1. Sweep efficiency: tier-dedup + narrow thread grid + validated concurrent timing of non-contending cells.
2. Clean-break rename pass (B) — the connective tissue; do it once, update all call sites/docs/tests.
3. Surface completion (A): frame_jacobian/frame_jacobian_dot/osc_inertia host wrappers → benchmarkable + bindable.
4. Max-perf completion (big-robot in-block parallelism) — measured with the now-fast sweep.
5. Test completion (C): C2/C3 first (correctness holes), then C1 (d2ee bug), C4/C5/C6 breadth, C7.
6. Feature completion (D): plant_step_hessian, missing bindings, floating-mimic integrator-grad.
