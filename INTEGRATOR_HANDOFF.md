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

1. **Integrator inner-controlled spill (workspace plumbing).** The integrator
   inners are NOT yet placement-aware: no `d_workspace` param, no
   `s_temp`+`s_workspace`+placement template, so LITE/MINIMAL smem doesn't shrink
   and the cold per-stage scaffold can't move to L2-pinned global. Not needed for
   fixed or go2-floating (they fit smem today); **needed for g1/h1_2 floating
   gradients** (nv=35 overflows float smem ~73 KB). When tackled, apply the
   top-down ordering (spill the cold outer stage scaffold first, keep the
   dynamics inner in smem longest) and reuse the established
   inner-controlled-placement pattern from FD/Minv/ABA. Also register integrator
   kernels in `test/diagnostics/tier_instantiation_smoke.py` (special-case the
   extra `IntegratorType` template arg).
2. **Floating FD-sanity test.** `test_integrator_gradient_fd_sanity.py` is
   fixed-base only; floating needs an SE(3) log for the tangent-space
   perturbation of the q columns.

Out of scope here but on the longer roadmap: **JAX FFI bindings** to replace the
stale Pybind11 layer (generate-compile-run-fast fit; see project memory).

---

## 5. Known limitations

- Floating gradient smem budget: go2 floating (nv=18) fits in float (~73 KB);
  g1 floating (nv=35) needs item 1's selective spill.
- Floating FD-sanity deferred (item 2 above).

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
