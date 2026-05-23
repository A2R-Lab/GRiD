# Time-Integrator Codegen — Handoff

Status as of 2026-05-22 (branch `humanoid-tier-spill`; RBDReference on
`modernizing-tests`). Documents the time-integrator work (Euler /
Semi-Implicit Euler / Midpoint / RK3 / RK4): what is built, what is validated,
and what is still open. The floating Euler gradient bug that earlier versions of
this doc flagged as a "CRITICAL OPEN QUESTION" is **RESOLVED** — see §3.

---

## 1. What was built

GRiD emits time-integrator kernels alongside the existing dynamics algorithms,
in the usual inner / device / kernel / host layering, dispatched at compile time
on `template <typename T, IntegratorType IT>`.

Per integrator, three entry points:
- `integrator<T,IT>`                       — value only, `x_{k+1}`
- `integrator_gradient<T,IT>`              — gradient only, `dAB = [A|B]`
- `integrator_gradient_with_x_kp1<T,IT>`   — both at once

State / shape conventions:
- `q` size `nq` (= `nv` fixed-base; `= nv+1` floating-base, xyzw quaternion).
- `v` size `nv`, GRiD internal order `[ω(3), v_lin(3), joint_v…]` (angular first).
- `x_{k+1}` size `nq + nv` (NOT `2·nv`); host wrappers truncate the
  over-allocated `d_x_kp1` tail.
- `dAB` shape `(2·nv) × (3·nv)`, column-major, columns `[∂/∂q | ∂/∂v | ∂/∂u]`
  all in **tangent space** (so the shape is identical fixed vs floating).

Canonical Python lives in the `RBDReference` submodule (not the adapter):
- `RBDReference.integrator(q,qd,u,dt,integrator_type)` — [RBDReference/RBDReference.py:333](RBDReference/RBDReference.py#L333)
- `RBDReference.integrator_grad(...)`                  — [RBDReference/RBDReference.py:372](RBDReference/RBDReference.py#L372)
- SE(3)/SO(3) Lie-group helpers (`_quat_mul_xyzw`, `_quat_exp_from_half_omega`,
  `_so3_right_jacobian`, `_se3_Q_block`, `integrate`, `dIntegrate`) —
  [RBDReference/RBDReference.py:98-330](RBDReference/RBDReference.py#L98).

`ProjectModelAdapter.integrator / integrator_gradient` are thin pass-throughs
that wrap RBDReference and apply `normalize_vector` / `normalize_matrix`
([project_adapter.py:112-119](test/pinocchio_equivalents/utils/project_adapter.py#L112)).
`normalize_matrix` is the IDENTITY; RBDReference already emits in GRiD internal
order, so CUDA-vs-ProjectModelAdapter is an apples-to-apples internal-order
comparison.

CUDA codegen:
- `GRiDCodeGenerator/algorithms/_integrator.py` — value path incl. floating
  Lie-group retract helpers for the q-update.
- `GRiDCodeGenerator/algorithms/_integrator_gradient.py` — gradient assembly.
  Floating Euler reads precomputed `s_dInt_q_6x6` / `s_dInt_v_6x6` for the top
  `nv` rows; bottom rows are `dt·J_qq | I+dt·J_qv | dt·Minv`.
- `GRID_HAS_INTEGRATOR` / `GRID_HAS_INTEGRATOR_GRADIENT` macros gate consumer
  code so a value-only build still compiles.
- Kernel tier: `RESOURCE_TIER` defaults to `GRID_DEFAULT_RESOURCE_TIER` (so the
  bench `-DGRID_DEFAULT_RESOURCE_TIER=…` macro reaches integrators; TIER_PERF
  behavior unchanged). Python keeps its `TIER_PERF` default.

---

## 2. What is validated (green, trusted)

- **Fixed-base, all 5 integrators, value + gradient + both-at-once**: CUDA ↔
  ProjectModelAdapter(RBDReference) passes at rtol=atol=5e-4 on iiwa14-fixed and
  go2-fixed across dt ∈ {1e-3, 1e-2, 1e-1} and corner + random samples.
- **Floating-base, all 5 integrators, VALUE path**: passes (~5e-7) on
  go2-floating; q-update is the Lie-group retract, cross-checked vs
  `pin.integrate` (`test_integrator_pinocchio_equivalence.py`, 20/20).
- **Floating-base GRADIENT, all 5 integrators** (Euler / SI-Euler / Midpoint /
  RK3 / RK4): value + gradient + both-at-once pass on iiwa14-floating and
  go2-floating at rtol=atol=5e-4, at 32 and 448 threads. The floating multi-stage
  gradients project each stage's `J_qq` columns through the SE(3) `dIntegrate`
  6x6 blocks (`_integrator_gradient.py`); SI-Euler evaluates `dIntegrate` at
  `dt*v_new` and adds the `dInt_v @ dv/dX` top-row matmul.
- The CUDA equivalence suite now **sweeps block thread counts** (1 warp +
  multi-warp + a session-random non-multiple-of-32) so any future thread-count
  race fails the suite. See `test/TESTING_STRATEGY.md`. The integrator smoke
  runner clamps the requested count to `SUGGESTED_THREADS` because the kernels
  are `__launch_bounds__(tier_max_threads<TIER>())` and launching above that
  bound is a hard `cudaErrorInvalidValue` (hit on iiwa14-fixed, bound 352 < 448).
- All five `IntegratorType` instantiations are registered for
  `cudaFuncSetAttribute` (init_grid_kernel_attrs); previously only the default
  (Euler) was, so a non-Euler floating gradient whose arena exceeds the 48 KB
  device default failed to launch.

Test files:
- CUDA equivalence: `test/cuda_equivalents/test_cuda_integrator_equivalence.py`
  (smoke runner `cuda_integrator_smoke_runner.cu`).
- Pinocchio equivalence: `test/pinocchio_equivalents/tests/test_integrator_pinocchio_equivalence.py`.
- FD sanity (analytical ↔ finite diff): `test/pinocchio_equivalents/tests/test_integrator_gradient_fd_sanity.py`
  (fixed-base only).

---

## 3. The floating Euler gradient bug — RESOLVED 2026-05-22

It was a **CUDA thread-count race**, not a math/convention defect. The whole CUDA
equivalence suite previously launched every kernel at `<<<1,32>>>` (one warp),
while the integrator runner uses `SUGGESTED_THREADS` (448). RBDReference matches
Pinocchio; the CUDA side was correct *at 32 threads* and raced above one warp, so
the error grew with operand magnitude — which earlier debugging misread as a
"velocity-scaled dropped-coupling term."

Root causes, both CUDA-side, floating-base only, in
`_inverse_dynamics_gradient.py` (`gen_inverse_dynamics_gradient_inner`):
1. The `da/du` init zeroed `s_temp[Offset_da_dq]` in one parallel loop and then
   `+=`-accumulated into it in the next with **no `__syncthreads` between** —
   correct only within a single warp. Fixed by adding the barrier.
2. The `MxS(dv/du)·qd` accumulation for the **floating root** had all 6 root axes
   (`dof_id` 0–5 → `jid` 0) `mxX_peq_scaled` into the *same* `da/du` column; that
   helper assumes one writer per destination, so the 6-way `+=` raced. Fixed by
   serializing the root sum onto a single lane.

A **separate, real** bug fixed alongside it: the SE(3) right-Jacobian `Q`-block
had a sign-flipped `c3` coefficient in BOTH `RBDReference._se3_Q_block` and the
CUDA `grid_se3_Q_block` (verified vs `pin.dIntegrate(ARG1)` to ~1e-14). It
affected the SI-Euler floating gradient and any non-tiny `dIntegrate` increment;
it matched Python↔CUDA before only because both were wrong.

After both fixes, standalone `forward_dynamics_gradient_qd`,
`inverse_dynamics_gradient_qd`, and the integrator `dAB` all match the reference
to float32 noise at **32 and 448 threads**.

Fix commits: GRiDCodeGenerator `501501b`, RBDReference `df76001`.

---

## 4. Open / next (nothing blocking; pick up here)

In rough priority order:

1. **Floating FD-sanity test.** `test_integrator_gradient_fd_sanity.py` is
   fixed-base only; floating needs an SE(3) log for the tangent-space
   perturbation of the q columns.
2. **Integrator VALUE-path spill (optional).** Only the gradient kernel is
   tier-spillable so far (its `s_D_qdd_stage` dominates). The value kernel's
   scaffold (`s_stage_*`) is small and fits everywhere, so it stays full-smem;
   add the same `d_workspace` plumbing only if a big robot's value path ever
   overflows.
3. **Bench the integrator algos.** `integrator`, `integrator_gradient`, and
   `integrator_with_gradient` are in `ALGO_REGISTRY` but have no row in
   `PER_ALGO_SPECS` (`test/benchmarks/baselines/grid/run.py`), so the bench
   currently SKIPS them (with a warning — they no longer hard-fail the sweep as
   of 2026-05-23). Wiring them in needs custom handling: their host signatures
   take extra `dt` (and `IntegratorType`, a template arg) beyond the uniform
   `(d, m, GRAVITY, N, ...)` shape the spec rows assume, plus integrator-specific
   I/O buffer setup. Add the spec rows + buffer plumbing to include integrator
   timings in the tier sweep.

Out of scope here but on the longer roadmap: **JAX FFI bindings** to replace the
stale Pybind11 layer (generate-compile-run-fast fit; see project memory).

---

## 5. Integrator gradient tier spill (done 2026-05-22)

The integrator-gradient kernel is inner-controlled-placement-aware like
FD/Minv/ABA. The dominant cold buffer `s_D_qdd_stage` (`max_stages·nv·3nv`,
≈59 KB float on g1-floating) lives in smem at TIER_PERF and spills to the
L2-pinned `d_workspace` grad section at TIER_LITE/MINIMAL. Mechanism:
- `select_shared_tier_3way` picks per-tier placement; the kernel emits per-tier
  bodies (like `_emit_fd_du_kernel_body_for_flags`) gated on `RESOURCE_TIER`,
  with `s_D_qdd_stage` conditionally in `extra_t_buffers` (smem) vs pointed into
  `&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]`.
- `INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T,TIER>()` is tier-aware;
  `INTEGRATOR_DU_D_QDD_IN_SMEM<TIER>()` gives the placement; `d_workspace` is
  threaded through the kernel + host (L2-pinned when `GRID_INTEGRATOR_DU_USES_WORKSPACE`).
- All five `IntegratorType` instantiations × 3 tiers compile in
  `tier_instantiation_smoke.py` (iiwa14/go2/h1_2). Verified: go2-floating
  gradient matches the reference at TIER_PERF (smem) AND TIER_MINIMAL (spilled),
  32 + 448 threads. Exercise the spill via `GRID_CUDA_INTEGRATOR_TIER=TIER_MINIMAL`.

Note: the arena helper (`gen_declare_shared_arena`) only natively spills the
`s_temp` inner slot via `tier_workspace_expr`; `s_D_qdd_stage` is a cold
extra-buffer, hence the per-tier-body approach rather than a single placement bool.

## 6. Known limitations

- Floating FD-sanity deferred (§4 item 1).
- Integrator value-path is full-smem only (§4 item 2) — fits all current robots.

---

## History

Earlier revisions of this doc carried a long "CRITICAL OPEN QUESTION" thread that
attributed the floating Euler gradient mismatch to a structural dropped
linear↔angular velocity-coupling term in floating-base `forward_dynamics_gradient`,
then went back and forth on which side carried it. **That entire narrative was a
misdiagnosis** — the real cause was the 32-thread launch hiding a multi-warp race
(§3). The lesson is now encoded as a regression guard: the CUDA equivalence tests
sweep thread counts including a random non-multiple-of-32 (see
`test/TESTING_STRATEGY.md`, Principle 2). Do not reintroduce fixed-32-thread
launches.
