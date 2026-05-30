# GRiD `humanoid-tier-spill` — Handoff

Status as of 2026-05-23 (branch `humanoid-tier-spill`; RBDReference + URDFParser
submodules on `modernizing-tests`). This is the single handoff for the branch.
It covers three intertwined efforts:

1. **Time-integrator codegen** (Euler / Semi-Implicit Euler / Midpoint / RK3 /
   RK4) — value + gradient. Detailed below (§1-6).
2. **Resource-tier shared-memory spill rollout** — every overflowing kernel now
   fits the sm_120 ~100 KB cap at all tiers via surgical per-tier spill. The
   architecture + per-algo details live in the concepts doc
   `docs/source/user_guide/concepts/resource_tier_system.rst`.
3. **Warning cleanup** — the RBDReference `mxS` NumPy `ndim>0` deprecation is
   fixed at the source.

See **Backlog / open items** at the end for what's left. The floating Euler
gradient bug that earlier revisions flagged as a "CRITICAL OPEN QUESTION" is
**RESOLVED** — see §3.

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
([reference_backend.py:112-119](RBDReference/equivalents/reference_backend.py#L112)).
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
- Pinocchio equivalence: `RBDReference/tests/test_integrator_pinocchio_equivalence.py`.
- FD sanity (analytical ↔ finite diff): `RBDReference/tests/test_integrator_gradient_fd_sanity.py`
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

1. **Floating FD-sanity test — WON'T DO (covered elsewhere).** A floating
   version of `test_integrator_gradient_fd_sanity.py` would need SE(3)-log
   machinery for the tangent-space q perturbation and output difference. Not
   worth it: the floating integrator gradient is already validated by
   `test_integrator_pinocchio_equivalence.py` (vs `pin.dIntegrate`) and the CUDA
   equivalence suite (g1_floating passes). The FD-sanity test stays fixed-base.
2. **Integrator VALUE-path spill — DONE 2026-05-23.** The value kernel now
   threads the FD inner's `MINV_F_IN_SMEM` lever: at LITE/MINIMAL (and PERF on
   h1_2) the Minv F-region (`6·NV²`) spills to `d_workspace`, keeping the hot FD
   path in smem. h1_2 fixed/floating drop from 103/124 KB to ~41/46 KB.
   `integrator_kernel` gained `unsigned char *d_workspace` (2nd arg);
   `INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, TIER>` is tier-aware. Validated:
   iiwa14 equivalence passes at TIER_MINIMAL.
3. **Bench the integrator algos — DONE 2026-05-23.** `integrator`,
   `integrator_gradient`, and `integrator_with_gradient` are now timed by the
   bench: `PER_ALGO_SPECS` rows (per-algo path) + measures in the monolithic
   `timeGRiD_{single,batch}.cu` (the default path). Their host signatures take an
   extra `dt` (a fixed bench dt is used; IntegratorType defaults to EULER) and
   `gridData` already allocates the I/O (`d_x_kp1`, `d_dAB`). Verified on iiwa14:
   integrator 15.6 us, gradient 48 us @ N=256 (compute-only).

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
- **rdc + launch_bounds (FIXED 2026-05-23).** Under `-rdc=true` (the bench's
  single-call build), the integrator kernels failed ptxas at LITE/MINIMAL for
  medium+ robots: `__launch_bounds__(tier_max_threads<TIER>())` budgeted ~64-85
  regs but the (non-inlined) RBD callees need 86-99 (`load_update_XImats`,
  `direct_minv_inner`, `inverse_dynamics_gradient_inner`). The integrator is
  register-bound by those callees, so launch_bounds is now pinned to
  `SUGGESTED_THREADS` (PERF cap, ≤512 → ≥128 regs) at ALL tiers — tier behavior
  remains the s_D_qdd_stage smem spill. Verified: compiles under rdc at all 3
  tiers (go2-floating) + MINIMAL equivalence still passes (iiwa14). The earlier
  "verified at MINIMAL" only covered the inlined (no-rdc) equivalence build.
- **Integrator smem overflow on big floating robots — RESOLVED 2026-05-23.**
  Both kernels now have surgical per-tier spill ladders (see §4 items 2-3 and
  `docs/source/user_guide/concepts/resource_tier_system.rst` → "Integrator
  surgical spill"). Value: spills the Minv F-region. Gradient: 4-rung ladder
  (Dqdd → +dAB+id_du-selective → +whole FD-grad inner) composing the existing
  fd_du levers. All robots/tiers now fit the sm_120 ~100 KB cap — g1_floating
  PERF keeps the hot path in smem via the selective rung; h1_2 (both) use the
  whole-inner rung (the FD-grad inner alone is 160-441 KB, physically can't fit
  smem, so the spill is forced). Verified: all integrator kernels compile at all
  tiers (incl. g1_floating selective rung + h1_2 whole-inner); iiwa14 numerical
  equivalence passes at TIER_MINIMAL (value F-spill + gradient whole-inner) and
  g1_floating passes at PERF (the selective rung). tier_instantiation_smoke
  passes for iiwa14/go2/h1_2 at all 3 tiers.

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

---

## Backlog / open items — CURRENT TODO LIST

Last updated 2026-05-25. Grouped by theme; rough priority within each.

## CURRENT PLAN — re-prioritized 2026-05-26 (perf-cleanup)

### ee_pose_gradient GEOMETRIC-JACOBIAN REWRITE — 2026-05-27 (LATEST; pick up here)

**STEP A + B + C LANDED + COMPREHENSIVELY VALIDATED + PUSHED (2026-05-27).**
Branch HEAD: parent `5dde921`, GRiDCodeGenerator `df70675`, RBDReference `0e71d06`.

**Validation matrix (CUDA equivalence vs pinocchio oracle):**
- iiwa14 fixed: 10/10 samples GREEN
- iiwa14 floating: 1/1 zero GREEN + 6/6 non-degenerate (conservative, high_velocity,
  high_acceleration, energetic_random_0, floating_quat_positive, floating_quat_mixed)
- go2 fixed: GREEN (multi-EE branched, 4 leaves)
- go2 floating: 1/1 zero GREEN + 6/6 non-degenerate (branched + floating + non-zero
  pitch — the *key* validation that catches sign errors like the row5 `(sp/cp)*(cy*Jw0 +
  sy*Jw1) + Jw2` formula)
- g1 fixed: 10/10 (29 DOF, 4 EEs)
- g1 floating: 1/1 zero GREEN
- h1_2 fixed: 10/10 (51 DOF, 12 EEs — biggest serial-equivalent we can run)
- h1_2 floating: SKIPPED on pre-existing `forward_dynamics` smem-cap (120 KB > 101 KB,
  the deferred ancestor-scratch de-alias issue) — NOT a regression from this work.
  ee_pose_gradient kernel itself never crashes; FD does, before ee_grad runs.

Floating-zero is degenerate (pitch=0 zeros the `sp/cp` term in row5, identity rotation
hides axis-mapping bugs). The non-degenerate sample set on iiwa14+go2 floating is the
load-bearing validation — both passed. Algorithm is solid.

**Important validation-gap fix:** `FLOATING_CUDA_ALGORITHMS` (the default for floating
equivalence tests) excludes `end_effector_pose_gradient` / `end_effector_pose_hessian`
(they sit in `FLOATING_CUDA_CANDIDATE_ALGORITHMS` instead, opt-in via
`GRID_CUDA_FLOATING_ALGORITHMS=all`). Earlier "iiwa14/go2 floating GREEN" runs were
therefore not actually comparing ee_pose_gradient on CUDA — they tested every other
algo. Fixed with focused re-runs:
- iiwa14-floating `GRID_CUDA_FLOATING_ALGORITHMS=end_effector_pose,end_effector_pose_gradient`
  + 6 non-degenerate samples: GREEN
- go2-floating same args: GREEN (branched + multi-EE + floating + non-zero pitch ⇒
  exercises every codegen path that matters)
The CUDA d/dv ee_pose_gradient is now genuinely validated.

### PERF WIN — Step C closes the floating ee_pose_gradient gap (2026-05-28)

Sweep `test/benchmarks/results/ee_grad_step_c_perf_v2/` (parent `1200115` =
shared-chain geometric Jacobian + dxhom-skip). The dxhom-skip was the load-bearing
change: the new inner doesn't use `s_dXhom` (`(void)`-ed) but the device+kernel were
still computing the per-joint LOCAL d-transforms — for floating base that includes an
expensive quaternion-derivative of the base transform. Setting `include_gradients=False`
in the helpers + passing `s_dXhom=nullptr` to the inner killed that wasted work.

ee_pose_gradient batch (N=256 compute-only µs/prob), GRiD vs pinocchio:
| robot  | base     | OLD-GRiD | new-GRiD | pin     | OLD ratio (vs pin) | new ratio          |
|--------|----------|----------|----------|---------|--------------------|--------------------|
| iiwa14 | fixed    | 0.058    | 0.057    | 0.138   | GRiD 2.38x         | GRiD 2.43x         |
| iiwa14 | floating | 2.116    | **0.178**| 0.133   | pin 15.9x          | **pin 1.34x**      |
| go2    | fixed    | 0.064    | 0.058    | 0.133   | GRiD 2.08x         | GRiD 2.28x         |
| go2    | floating | (n/a)    | **0.179**| 0.151   | (pin 14.0x est.)   | **pin 1.18x**      |
| g1     | fixed    | 0.166    | 0.102    | 0.193   | GRiD 1.16x         | GRiD 1.89x         |
| g1     | floating | 2.200    | **0.219**| 0.288   | pin 7.52x          | **GRiD 1.31x WIN** |
| h1_2   | fixed    | 0.333    | 0.157    | 0.245   | pin 1.16x          | GRiD 1.56x WIN     |
| h1_2   | floating | 2.434    | **0.286**| 0.315   | pin 7.14x          | **GRiD 1.10x WIN** |

Single-call ee_pose_gradient (floating) went 260-312µs → **16-29µs** (10-16× faster);
pinocchio still wins single-call (sub-µs codegen-CPU) but the gap is reasonable now.

### Floating-base first-order vs pinocchio (snapshot from same sweep)

Same `ee_grad_step_c_perf_v2` data — N=256 compute-only µs/prob, GRiD vs pin:
- **Big robots win across the board (h1_2):** aba 1.31x, crba 1.22x, minv 3.83x, fd 2.43x, id 1.34x.
- **iiwa14 (small floating) GRiD loses:** aba pin 1.08x, minv pin 1.51x, fd pin 1.29x. (crba 4.48x WIN, id 1.44x WIN.)
- **go2 (small floating, branched) GRiD loses several:** aba pin 1.75x, crba pin 1.01x ~tie, minv pin 1.11x, fd pin 1.15x. (id 1.35x WIN.)
- **g1 mixed:** aba pin 1.40x, crba **pin 1.73x (biggest concrete loss)**, minv/fd/id WIN.

Top floating-base first-order targets if/when this becomes the next priority:
1. **g1-floating crba** (pin 1.73x) — biggest concrete miss.
2. **go2-floating aba** (pin 1.75x) — second biggest.
3. **iiwa14-floating minv/fd** (pin 1.51x/1.29x) — small robot launch overhead.

### ✅ GPU d2ee floating-base d/dv rewrite — DONE 2026-05-29 (audit-verified)

**Closed via analytic, not the FD-on-Jacobian recipe below.** The original section
described an FD-on-Jacobian approach. The actual implementation went analytic:
- **RBDReference `843a302` (A.1 Python):** closed-form analytic d²(pose)/dv² via
  direct 2nd-order Taylor expansion of the chain world transform. No FD.
- **GRiDCodeGenerator `2bf6d53` (A.1 GPU, parent `080debd`):** analytic chain-
  composition Hessian replaces the older FD-on-Jacobian; iiwa14 fixed+floating +
  go2 floating CUDA equivalence GREEN; matches `RBDReference.end_effector_pose_
  hessian_analytic` to ~1e-13 numpy sim.
- **CUDA emits 6×nv×nv** (the tangent-space output, pinocchio convention).
- **ee_pose_hessian d/dv:** Python `342465d`, GPU codegen `10282e9` (FD-on-
  Jacobian d/dv rewrite, 6*nv*nv output; iiwa14/go2/g1/h1_2 fixed + iiwa14/go2/g1
  floating CUDA equivalence GREEN). The d2ee-vs-pin oracle uses `pin_so_ext`
  (composed `getJointKinematicHessian(LOCAL_WORLD_ALIGNED)`).
- **C.7 sweep will measure analytic-vs-pin speedup at N=256** (deferred until
  the sweep runs).

The original FD-on-Jacobian pickup recipe is preserved below for historical
reference but is no longer the active plan.

---

**ORIGINAL (HISTORICAL) PICKUP RECIPE — superseded by the analytic path above:**

1. **Architectural change.** d2ee inner currently takes pre-computed `s_Xhom`, `s_dXhom`,
   `s_d2Xhom` and assumes one fixed q. FD-on-Jacobian needs to recompute Xhom for each
   perturbed q, which means the inner has to call `load_update_XmatsHom_helpers` + the
   new `end_effector_pose_gradient_inner` internally — requires `d_robotModel` and
   `s_topology_helpers` in the inner signature. Look at the canonical pattern in
   `_eepose_gradient_hessian.py::gen_end_effector_pose_gradient_hessian_inner`
   (line 1362) to see what's changing.

2. **Algorithm.** Mirror `RBDReference.end_effector_pose_hessian` (the FD-on-d/dv-J
   approach): for each `vi ∈ [0, nv)`:
     - SE(3) integrate `q_plus = integrate(q, +h*e_vi)`, `q_minus = integrate(q, -h*e_vi)`.
     - Recompute Xhom for each perturbed q via `load_update_XmatsHom_helpers`.
     - Compute J(q_plus) and J(q_minus) via `end_effector_pose_gradient_inner`.
     - `H[:, :, vi] = (J_plus - J_minus) / (2h)`.
   Then symmetrize: `H[a, i, j] = 0.5 * (H[a, i, j] + H[a, j, i])`.

3. **SE(3) integrate codegen** for the perturbation. Codegen-time specialized per vi:
     - **Fixed-base:** trivially `q_pert = q + h*e_vi` (just add h to q[vi]).
     - **Floating vi ∈ [0, 3):** linear base — `q_pert[0..3] = q[0..3] + h*R(q[3:7])[:, vi]`;
       quat and joints unchanged.
     - **Floating vi ∈ [3, 6):** angular base — quaternion multiplication by small
       axis quat `(h/2 * e_{vi-3}; sqrt(1 - h²/4))`; xyz and joints unchanged.
     - **Floating vi ≥ 6:** arm joint — `q_pert[vi+1] = q[vi+1] + h`; rest unchanged
       (assumes single-DOF arm joints; check `get_joint_index_q` if any multi-DOF).
   Reference implementation: `RBDReference.RBDReference.integrate(q, v_dt)`.

4. **Output shape & consumer ripple.** `d_d2eePos` allocator changes from
   `6*NUM_JOINTS*NUM_JOINTS*NUM_EES` → `6*NUM_VEL*NUM_VEL*NUM_EES`. Update:
     - `GRiDCodeGenerator.py:1157,1161` (gridData allocator).
     - `test/cuda_equivalents/cuda_equivalence_runner.cu` (h_d2ee/d_d2ee sizing,
       memcpy, print_vector). 5+ sites.
     - `python/grid_rbd/wrapper_template.cu` (C extern + JAX FFI memcpy + buffer
       validation; the d2ee handler around line 875+).
     - `python/src/_core.cpp` (`py::array_t<float> out({batch, 6*nees, nv, nv})`).
     - `python/grid_rbd/_handle.py` + `python/grid_rbd/jax/__init__.py`
       (d2ee reshape: use `num_vel` instead of `num_joints`).
     - `printGRiD.cu` (`printMat<T,NUM_VEL,NUM_VEL>` for d2eePos block, offset
       calculation uses NUM_VEL).
     - `test/cuda_equivalents/test_cuda_executable_equivalence.py:1041-1054`
       (the d2ee oracle path; already calls `reference_model.end_effector_pose_hessian`
       which now returns 6×nv×nv — so this might be OK already, just verify).

5. **Smem budget.** For h1_2 floating, s_d2eePos = 6*57*57*12 ≈ 900 KB — way over the
   ~101 KB cap. Will need workspace spilling like the existing d2ee (existing macros
   like `GRID_D2EE_USES_WORKSPACE_TEMP` may need re-keying for the new layout). Scratch
   for the per-iter J_plus + J_minus + q_pert + Xhom_pert + inner_temp is ~14K floats
   (~57 KB) for h1_2-floating — fits in smem at PERF tier; LITE/MINIMAL spill the
   inner workspace as the current d2ee already does.

6. **Validation.** After codegen + consumer ripple, run iiwa14-floating CUDA
   equivalence with `GRID_CUDA_FLOATING_ALGORITHMS=end_effector_pose,end_effector_pose_gradient,end_effector_pose_hessian`
   and a non-degenerate sample set (e.g. conservative, high_velocity,
   high_acceleration, floating_quat_positive, floating_quat_mixed). Once GREEN, expand
   to go2-floating then g1/h1_2-floating.

**Independent cleanup tasks the user flagged for later (2026-05-29 EVENING-2 audit
re-verified):**
- Remove orphaned helper `_emit_eepose_grad_compacted_nonserial` in
  `_eepose_gradient_hessian.py:667` (zero callers, ~18 lines). **NOTE: the sibling
  `_emit_eepose_grad_extraction` at line 632 is NOT orphaned — it still has 1 caller
  at line 853** (the eepose_grad_hessian path). Earlier HANDOFF note conflated the
  two; only `_compacted_nonserial` is safe to drop.
- Drop `s_dXhom` param from `end_effector_pose_gradient_inner` signature (still
  `(void)`-marked dead arg at `_eepose_gradient_hessian.py:467`) — propagate through
  device/kernel call sites.
- Per-tier smem allocator can also drop the dxhom-shared logic (the spilling-to-
  workspace path for dxhom is never taken on the gradient path).
- Documentation: add the d/dv convention note + the geometric-Jacobian explanation
  in `docs/source/user_guide/concepts/`, and a section in `docs/python_wrappers_plan.md`.

### ee_pose_hessian d/dv (RBDReference + pinocchio_backend ONLY) — 2026-05-28

Python d²(pose)/dv² landed (RBDReference `342465d`, parent `2748c4f`):
- `RBDReference.end_effector_pose_hessian` rewritten as central-difference FD of the
  (now-correct) d/dv Jacobian on `self.integrate(q, h*e_i)`, then symmetrized. Replaces
  ~285 lines of analytic d²/dq² (whose per-row atan2 second derivatives had known
  orientation-row errors at non-small angles).
- `pinocchio_backend.end_effector_pose_hessian` symmetrized FD of d/dv Jacobian on
  `pin.integrate` — matches the project adapter.
- iiwa14 fixed + floating Python equivalence GREEN.

**CUDA d2ee codegen STILL emits d²/dq² (6×nq×nq).** Default floating equivalence tests
do not include d2ee so they still pass. Opt-in `GRID_CUDA_FLOATING_ALGORITHMS=all` would
shape-mismatch floating because RBDReference now returns 6×nv×nv. The GPU d2ee rewrite
is a separate (large) ripple deferred for now — would need either FD-on-d/dv-gradient on
device or analytic d²/dv² (spatial second derivatives, pinocchio-frame-Hessian style).
Recommendation: defer until a real consumer needs d/dv on GPU.

- **Step C (`df70675` in GRiDCodeGenerator, `d8b1bcb` in parent):**
  - `_eepose_gradient_hessian.py::gen_end_effector_pose_gradient_inner` rewritten as
    shared-chain geometric Jacobian: one FK pass builds world transforms for every joint
    via BFS-level chain-up; per-(ee, chain joint, S-col) compile-time-unrolled column
    fills compute `J_v = aw x (p_ee - p_j)` (revolute) or `J_v = aw` (prismatic) with
    `aw = R_j_world * S_local`; per-ee `(cy,sy,cp,sp)` cached for `E(rpy)^{-1}`; closed-
    form rpy rows: row3 `(cy*Jw0 + sy*Jw1)/cp`, row4 `-sy*Jw0 + cy*Jw1`, row5
    `(sp/cp)*(cy*Jw0 + sy*Jw1) + Jw2`. (Sign on row5's sy term took one iteration to
    catch — adj(E)[2,1] = sy*sp, plus not minus.) The old per-(djid,ee) re-chain + the
    floating compacted nonserial path are dead code (helpers `_emit_eepose_grad_*`
    still in the file as harmless orphans; cleanup pass later).
  - Scratch layout (in s_temp): `Xworld[16*n_joints] | Jv[3*nv*ee] | Jw[3*nv*ee] | E_sc[4*ee]`.
    Much smaller than the old `2*2*16*nq*ee` arena.
  - Output buffer flips `6*nq*num_ees` -> `6*nv*num_ees`. For fixed-base nq==nv so no
    functional change; for floating the base block is now the spatial Jacobian (omega; v)
    matching pinocchio's tangent convention.
  - Ripple: `GRiDCodeGenerator.py` allocator (`d_deePos`/`h_deePos` use `NUM_VEL`);
    `cuda_equivalence_runner.cu` (h_dee/d_dee sized to NUM_VEL); `wrapper_template.cu`
    (FFI memcpy uses NUM_VEL for both C extern + JAX); `_core.cpp` pybind output array
    `(batch, 6*NUM_EES, NV)`; `_handle.py` + `jax/__init__.py` reshape to NV; `printGRiD.cu`
    uses `printMat<T,6,NUM_VEL>`.
  - Fixed-target gradient (`fixed_target_name` != "") raises NotImplementedError in the
    rewrite; the only caller is `examples/quickstart_iiwa14.py` (unused by bench /
    equivalence). Re-add when a real consumer surfaces.

- **Step A (`087d458` in RBDReference):** `RBDReference.end_effector_pose_gradient` rewritten as
- **Step A (`087d458` in RBDReference):** `RBDReference.end_effector_pose_gradient` rewritten as
  shared-chain geometric Jacobian producing d/dv (tangent), 6×nv per ee. Validated 96/96 ee×scale
  combos vs the proven prototype on 5 robots × 2 bases, worst max-abs-err 2.4e-15 (machine
  precision); FD-validated on non-zero offset (~1e-9, FD floor).
- **Step B (`0e71d06`):** `pinocchio_backend.end_effector_pose_gradient` rewritten to d/dv via
  central-difference FD on `pin.integrate(q, h*e_i)` (Lie-group tangent). All 9 robots × 2 bases
  GREEN in `test_kinematics_derivatives_equivalence.py` — RBDReference d/dv ≡ pinocchio d/dv. The
  "match pinocchio" property is proven.
- **Step C — GPU codegen rewrite — PENDING.** Multi-file ripple:
  (a) `_eepose_gradient_hessian.py::gen_end_effector_pose_gradient_inner` — replace per-(djid,ee)
  re-chain with shared-chain FK + per-chain-joint J_v/J_w via cross-products + E(rpy)⁻¹·J_w.
  (b) Output buffer size for floating: `6 * NUM_EES * NUM_JOINTS` → `6 * NUM_EES * NUM_VEL` in the
  `gridData<T>` allocator (`GRiDCodeGenerator.py:1156,1160`), all `6*n*num_ees` strides in the
  codegen, kernel save-result sizes, and the host-copy memcpy bytes.
  (c) Consumers: `wrapper_template.cu` (FFI), JAX bindings reshape, CUDA equivalence harness
  expected shape, `python_wrappers` smoke tests, `printGRiD.cu`.
  (d) For fixed-base nq==nv so no buffer-size change; only floating shape flips. C++ constant
  `NUM_VEL` is already emitted (`GRiDCodeGenerator.py:715`) so no new constant needed.
  Per-robot regen + CUDA equivalence (fixed + floating) is the validation gate.

**PROVEN in Python (`/tmp/geom_jac_proto.py` → `test/benchmarks/_scratch_geom_jac_proto.py`),
ready to implement.** The floating ee_pose_gradient
"outlier" is ALGORITHMIC, not a bug: GRiD re-chains a full 4×4 transform PER Jacobian column
(`O(nq·depth)`); the fix is the shared-chain geometric (spatial) Jacobian (`O(nq+depth)`) — FK once,
then `J_v=â×(p_ee−p_j)`, `J_w=â` (revolute) / `J_v=â` (prismatic), pose-grad `=[J_v; E(rpy)⁻¹ J_w]`.
(The earlier "100× vs fixed" was inflated — fixed ee_pose_gradient is partially LICM-elided in the
bench; real gap ≈16× vs pinocchio.) PROOF: matches the existing analytic gradient to MACHINE
PRECISION (fixed iiwa14/go2/baxter incl. branched; floating arm cols); **3–22.7× fewer 4×4-mults**
(grows with size: h1_2 22.7×). E(rpy) for GRiD's RPY convention (R=Rz(yaw)Ry(pitch)Rx(roll))
verified: E=[[cy·cp,−sy,0],[sy·cp,cy,0],[−sp,0,1]].

**DECISION (user 2026-05-27): output d/dv (tangent, 6×nv) to MATCH PINOCCHIO; document in
RBDReference + GPU code + docs. d/dv ONLY** (no d/dq — no current consumers; d/dq is non-standard
quaternion-component derivs; trivially derivable later via `d/dq = d/dv · base quaternion-rate map`).
Fixed base: nq==nv, unchanged. Floating: switches current d/dq(6×nq) → d/dv(6×nv tangent).

**IMPL PLAN (ordered, with checkpoints):**
1. **RBDReference.end_effector_pose_gradient** → geometric shared-chain Jacobian (d/dv). Preserve
   interface (q, ee_joint_names, ee_offsets → list of 6×nv per ee) + offset + fixed-joint handling
   (offset only shifts p_ee; fixed-joint EE: X_ee=X_world[parent]@fixed_T, chain=parent's). Use
   `get_S_by_id` + `get_joint_index_v` (both exist). Validate new==prototype (machine precision).
2. **pinocchio_backend.end_effector_pose_gradient** → d/dv (pin frame Jacobian, LOCAL_WORLD_ALIGNED,
   mapped to [xyz,rpy] via E⁻¹). Validate RBDReference d/dv == pinocchio d/dv ALL robots = the
   "match pinocchio" proof. CHECKPOINT before GPU.
3. **GPU codegen** `gen_end_effector_pose_gradient*` → shared-chain d/dv (kills the per-column
   re-chain + the floating non-serial compacted path). Re-validate equivalence + perf (expect the
   16× floating gap to close + the fixed path to drop too).
4. Docs + JAX bindings note the d/dv (tangent, pinocchio-matching) convention + the floating
   shape change (6×nv).
5. **RIPPLE (separate task):** ee_pose_hessian (d2ee) needs the same d/dv treatment for consistency.
Proof + scaling + E(rpy) in memory `project_grid_competitive_analysis.md`.

### COMPETITIVE ANALYSIS + crba FIX + SO DATA GAP — 2026-05-27

**crba regression FIXED (code; perf re-measure pending):** root cause = `6cdba85`'s depth-stepped
M-fill (3 `__syncthreads`/chain-depth → sync storm; regressed crba 2.5–6×, even unbranched iiwa14).
Reverted to the sync-free one-thread-per-jid ancestor walk, KEEPING the S-index correctness (M
entry now indexes by the PARENT's S index/sign via compile-time tables). Correctness validating
(b9g9c2kvg): go2 (branched) + iiwa14 PASS; baxter pending. NOT yet committed (gated on baxter).

**Limb-parallelism diagnostic (user asked if go2>iiwa means lost BFS parallelism):** NO — at
single-call go2/iiwa is 0.88–1.39× across all algos (limb parallelism works); go2>iiwa is purely a
BATCH effect (saturated GPU → work-bound, 12 vs 7 DOF). id_du/fd_du clean (1.07–1.08×), disconfirming
a systemic regression. Only crba regressed; minv/aba/crba share a pre-existing ~1.8× go2/iiwa
batch-occupancy gap (not a regression).

**Competitive vs Pinocchio (see memory `project_grid_competitive_analysis.md`):** throughput-vs-
latency split — pinocchio codegen-CPU wins single-call EVERYWHERE (sub-µs); GRiD wins batch N=256
(1.2–11×/prob), except crba (regression artifact → ~parity post-fix). GRiD numbers are compute-only
(GPU-resident assumption). pre_glass id/minv/id_du (iiwa14+go2 fixed, reused): glass HELD/beat it.

**SECOND-ORDER pinocchio data GAP (user-flagged, collect next sweep):** no pinocchio SO timing yet.
The pinocchio harness DOES support `idsva_so_body_frame` (`PINOCCHIO_ALGOS`); `fdsva_so` skipped (no
equivalent). Add the pinocchio column to the next sweep. Caveat: pinocchio nv³ SO codegen for g1/h1_2
may be very slow / hit the 1500s/algo timeout.

**crba FIX COMMITTED + COLLECTION DONE (`crba_so_collection`, EXIT 0):** crba recovered to baseline
(iiwa14 N=256 27.3→11.05µs, go2→13.0, g1→60.4, h1_2→103.7) and now BEATS pinocchio at batch
(2.5×/8.9×/1.2×/1.2×). Pinocchio second-order timing collected for ALL robots (no timeout).
**KEY FINDING — GRiD second-order LOSES to pinocchio on big robots:** idsva_so N=256/prob iiwa14
GRiD 6.8× / go2 4.5× WIN, but g1 pin 2.5× / h1_2 pin 2.7× LOSS. GRiD idsva_so scales badly with DOF
(nv³, compute+smem-bound) → **TOP optimization target** (see memory `project_grid_competitive_analysis.md`).

**NEXT: P4 — crba regression resolved, so the branch is mergeable.** Decide P4 merge
`perf-cleanup → modernizing-tests`. Open perf items (not merge-blockers): idsva_so big-robot SO
scaling (top target), minv/aba/crba batch-occupancy, single-call launch-overhead, idsva_so de-alias.

### P3 PERF SWEEP RESULTS — 2026-05-27
Ran the first trustworthy sweep on the P1-B-parallelized tooling: glass × {iiwa14,go2,g1,h1_2}
× {fixed,floating} × {perf,lite,minimal}, 24/24 cases produced (EXIT 0, ~4h21m). Output:
`test/benchmarks/results/perf_cleanup_overnight/` + report `test/benchmarks/perf_cleanup_overnight.md`.
Compared vs baseline `tier_sweep_20260525_002438` on N=256 compute-only (script in /tmp/compare_sweep.py).

**Headline: 101 wins, 71 regressions (>5%). Net = big-robot dynamics got ~2× faster, but crba
regressed hard.**
- **WINS (the GLASS/inner-owns payoff):** h1_2 (and g1) dynamics roughly HALVED vs baseline —
  h1_2-fixed minv 95 vs 178µs (0.54×), fd 147 vs 285 (0.51×), ee_pose_gradient 85 vs 167
  (0.51×), integrator 0.53×. ~100 cases >5% faster.
- **REGRESSION #1 — crba, systemic, MOST ACTIONABLE:** crba is 2.5–6× slower than baseline on
  EVERY robot, and the gap GROWS with batch (iiwa14 1.6×→2.5×, g1 1.6×→4.9× single→N=256;
  h1_2 ~6×, g1-fixed-minimal 8.5×). It hits even UNBRANCHED iiwa14, so it is NOT just the
  branched S-index correctness fix — it's a systemic crba codegen perf issue introduced
  relative to the 05-25 baseline (prime suspect: the crba P1 "depth-stepped fill" GLASS pass —
  verify whether the baseline pre/post-dates it via git archaeology on `_crba.py`). NEEDS
  root-cause; decide correctness-vs-perf tradeoff with user.
- **REGRESSION #2 — ee_pose_gradient on iiwa14 ~1.9× slower** (all tiers). Minor vs crba.
- **idsva_so whole-arena spill cost (the deferred-de-alias signal): CONFIRMED a real bottleneck
  on big robots** — g1-fixed idsva_so N=256: perf 3001µs → lite 6162 (2.05×) → minimal 5475
  (1.82×); h1_2-fixed: perf 21687 → lite 29455 (1.36×) → minimal 25887 (1.19×). Small robots
  noisy/flat. So the design-notes gate is MET: the deferred surgical ancestor-scratch de-alias
  IS warranted (would avoid the ~2× lite penalty on g1). See `docs/idsva_so_inner_refactor_notes.md`.

**Caveats:** glass-vs-glass only (pre_glass/pinocchio/mjx/frax columns not run); some small-robot
(<100µs) tier numbers are measurement noise (lite occasionally "faster" than perf).

**Suggested next (AM):** (1) root-cause the crba regression (git-bisect `_crba.py` vs baseline;
likely the depth-stepped-fill pass — may need to revert/retune it); (2) given the confirmed
lite/minimal spill cost, schedule the idsva_so surgical de-alias; (3) P4 merge once crba is
understood (don't merge a 6× crba regression unexamined).

### STATUS SNAPSHOT — 2026-05-26 (pick up at the P3 results above)
Branch `perf-cleanup`: **parent HEAD `d5be248`, RBDReference submodule `3e012b7`, codegen
`0b3cd29`, GLASS `3e910e1` — all pushed.** (Uncommitted working-tree noise NOT ours and
left alone: `URDFParser` t8.txt deletion, `test/dev_notes/` deletions [vestigial, to fold
into the Stage-5 sweep], `benchmark_multi_version.md`, an untracked tier matrix md.)

**d2ee BUG IS FIXED (2026-05-26, committed+pushed)** — see "KNOWN BUG — d2ee" below, now
resolved. Root cause: the pitch-row sqrt-term 2nd derivative used `s'_i = T_i/s` as the
quotient-rule numerator where it must use `T_i` (dropped a `1/s` factor), so the error
vanished near home (`s≈1`) and grew with joint angle. Fixed in BOTH the RBDReference
analytic hessian (both joint-loop copies, `3e012b7`) and the GRiD `ee_pose_hessian` codegen
(`_eepose_gradient_hessian.py`, `0b3cd29`). Validated: CPU finite-diff of the verified
gradient on iiwa14/go2/g1-fixed (~1e-11), and CUDA hard-pass vs the pinocchio oracle on
iiwa14 **fixed and floating** (KNOWN_FAILING now empty → d2ee is a hard requirement again).

**SO expansion RESULTS (2026-05-26):** g1 fallback (fixed+floating) GREEN (~12m, compile-
bound), g1 world-frame GREEN (35s), h1_2 world-frame GREEN (60s) — all via the pinocchio
oracle = the SO speed-win proof (was hours of pure-Python reference, now nvcc-compile-bound).
**h1_2 fallback (fixed + floating diag) FAILS**: `idsva_so` requests **168784 B** of shared
mem but the device caps at **101376 B/block** → `GPUassert: invalid configuration argument`.
This is the known device-path smem cap (P2 / perf-cleanup agent-11): `idsva_so` body+world
are the two kernels that still lack the per-tier surgical spill the other 9 kernels got, so
the largest robot overflows even on the forced-fallback path. NOT a regression, NOT from
d2ee. **Decision: SO test defaults stay `iiwa14`** (g1 fallback is a ~12m compile; the
`GRID_CUDA_*_ROBOTS` env vars already make g1/h1_2 zero-code for nightly/broad runs).
g1 is validated-green for permanent inclusion if a slower default is ever wanted; h1_2
fallback must wait for the `idsva_so` spill.

**DONE this session (committed + pushed):**
- **P1-A restructure**: the whole pinocchio-equivalence layer moved into the RBDReference
  submodule. `RBDReference/equivalents/` = reusable lib (conventions + `reference_backend`
  + `pinocchio_backend` + `pin_so_ext`); `RBDReference/tests/` = the suite + its infra
  (manifest/sampling/tolerances/comparators/model_sources/source_lock). Base vs dev
  requirements; README documents the swap. All GRiD importers + `developer_install.sh`
  rewired; 936 tests collect.
- **crba S-index fix CONFIRMED** on baxter+fetch fixed+floating.
- **Independent pinocchio oracle wired** as the default (`GRID_REFERENCE_BACKEND=pinocchio`):
  `build_adapter(backend=…)`; project_model (pure-Python URDFParser) is ONLY the codegen /
  model-under-test input, never a value oracle. Executable harness validated green on
  iiwa14 + go2 (fixed+floating). SO tests (idsva_so body+world, fdsva_so) wired to the EXACT
  `pin_so_ext` oracle, validated green on iiwa14.
- **Real d2ee bug found + filed** (see KNOWN BUG below); d2ee compared vs the independent
  oracle + listed in `KNOWN_FAILING_ALGORITHMS` (tracked, loud, non-fatal, NOT masked).

**DONE since the snapshot (committed+pushed):**
- **Vestigial sweep (Stage-5)** — `test/dev_notes/` removed (`c88153b`); restructure audited
  clean (no dead code/dups/orphans/shims); stale doc link fixed (`8c4e9d5`).
- **P1-B perf tooling FIXED** (`27ebfc7`): the sweep serialized CPU-bound nvcc compiles on the
  false premise "GPU work is serial" (iiwa14-fixed alone = 460s compile; big-robot floating =
  the 25-min cases). Split into a parallel **BUILD** phase + serial isolated **MEASURE** phase:
  `run.py` gained `--compile-only` (compile + populate the content-keyed binary cache, skip
  timing) and `--build-dir` (isolate per-case working grid.cuh/.o/.exe — the shared `results/`
  dir was the only real collision risk; caches are content-keyed). `run_multi_version.py`
  gained `--build-jobs` (auto from cores+free RAM, =8 on this box) that fans all GRiD compiles
  across cores, then the existing measure loop runs `--no-recompile` = pure cache-hit + GPU
  timing. `runner_key` hashes only compile inputs (not build_dir/compile_only), so the build
  phase produces the EXACT binary the measure phase loads; timing stays strictly serial →
  numbers unaffected. Validated: cold parallel build of iiwa14-fixed×{perf,lite,minimal} =
  ~499s wall (max, not the 919s sum), 3/3 no collision, measure phase hit every cache (0.0s).

**P2 RESOLVED/RE-SCOPED (2026-05-26):** the idsva_so surgical de-alias spill + inner-owns
placement + fdsva pool→global are ALREADY committed (`a0f3df0`, `86ad706`). The residual gap
— h1_2 body-frame IDSVA-SO requests 168784 B > the ~101 KB device cap *even at the forced
10 KB budget* (both `test_fixed_second_order_forced_fallback[h1_2-fixed]` and
`test_floating_second_order_diagnostic[h1_2-floating]`) — is the DEEP ancestor-pair-scratch
de-alias that `docs/idsva_so_inner_refactor_notes.md` explicitly DEFERS until a perf sweep
proves whole-arena spill is a real bottleneck. Per user (2026-05-26): **sweep-first, defer
the deep de-alias.** Interim: a self-healing device-cap SKIP landed in `_run_runner`
(`308f950`) — any kernel whose most-spilled smem request exceeds the per-block cap now skips
honestly (h1_2 body-frame SO; world-frame is h1_2's production path and passes). Skip
auto-disappears if the de-alias later makes it fit.

**P1-C DONE (2026-05-26) — broad correctness, all 9 robots × fixed+floating vs pinocchio:**
5 clean pass (iiwa14, go2, g1, gen3, fetch). The 4 "failures" were ALL first-time coverage
of robots beyond the old gate, and NONE were core GRiD dynamics-math bugs:
- **Comparator gaps (fixed, `f2c73e9`)** — not GRiD bugs: (A) added a magnitude-scaled atol
  floor `max(atol, rtol·max|expected|)` (mirrors `RBDReference/tests/comparators.py`) so
  large-`Minv` robots (h1_2 near-singular M, entries ~1e6–1e8, rel err ~1e-7) stop
  false-failing; does NOT re-mask d2ee (O(1)-scale). (B) EE-pose orientation rows fold ±π
  (atan2 branch) to exact agreement + a scoped rpy gimbal-lock/derivative-blowup skip;
  POSITION rows always stay strict. → baxter now green.
- **rizon4 = broken upstream asset (fixed, `0e58f74`)**: resolved URDF has 0 `<inertial>`
  blocks (flexiv xacro emits bare inertia tags) → zero-mass model → crba NaN. Honest
  `_model_inertia_is_degenerate` skip (self-heals if a fixed asset resolves). NOT a GRiD bug.
- **fr3 finger-2 position — OPEN (intentionally still failing, not masked)**: `fr3_finger_joint2`
  is a `<mimic>` joint; GRiD doesn't model mimic coupling so finger-2 position (+ derivs)
  diverge from pinocchio. Orientation matches. A model/convention LIMITATION. DECISION
  PENDING: add mimic-joint support vs skip mimic-affected leaves.
- **h1_2 `direct_minv` dynamic-smem crash — OPEN (GRiD robustness)**: at high (session-random)
  thread counts `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` hard-fails (`GPUassert:
  invalid argument`) instead of the graceful "shared-memory request … device supports" guard
  the other kernels use → FLAKY by thread count. Fix: make direct_minv's kernel-attr
  registration guard/skip over-cap like the rest. (Distinct from the idsva_so body cap.)
See memory `project_grid_broad_coverage_findings.md`.

**BACKLOG / FEATURE REQUEST (user-filed 2026-05-26):** GRiD doesn't support all URDF
features. Do a single AUDIT of URDF features (`URDFParser` + codegen) vs the spec and add the
missing ones in ONE clean pass later. First concrete gap = **mimic joints** (fr3); other
candidates: continuous/planar joints, `<dynamics>` damping/friction, `<limit>`, massless-link
robustness, `<transmission>`. See memory `project_grid_urdf_feature_support_backlog.md`. NOT
scheduled — backlog.

**h1_2-floating over-cap crash FIXED (`d4d601f`):** the runner's floating block re-registered
kernels with UNGUARDED `cudaFuncSetAttribute` (the generated `init_grid` path is guarded; this
block isn't, for its runner-local kernels). h1_2-floating `forward_dynamics` (120624 B) /
`direct_minv` exceed the ~101 KB cap at PERF tier → hard GPUassert. New
`grid_runner_set_smem_or_skip` helper emits the standard "shared-memory request … device
supports" message + exits cleanly → harness SKIPS (whole-case). Validated: iiwa14-floating
8 passed (no regression); h1_2-fixed 4 passed; h1_2-floating 4 skipped (clear reason). Whole-
case skip is coarse (loses h1_2-floating's fitting algos) — finer per-kernel skip / tier-aware
big-robot testing is future refinement.

**REMAINING TODOS (current):**
1. **OPEN from P1-C:** fr3 mimic-joint → folded into the URDF-feature audit backlog above.
   Also: auto-parallel test sizing (P1-C used a manual 6-way shell shard); finer per-kernel
   over-cap skip + tier-aware h1_2-floating testing.
2. **P3** full sweep vs `tier_sweep_20260525_002438` (now parallelized by P1-B; MUST run with
   the GPU idle — no concurrent compiles/tests — so timing isn't skewed). If it shows
   whole-arena idsva_so spill is a real LITE/MINIMAL bottleneck, THEN do the deferred deep
   de-alias (refactor idea 1 in the notes). Also naming/warnings/docs sweep.
3. **P4** merge `perf-cleanup → modernizing-tests` once validated.

---

Branch `perf-cleanup` (parent HEAD d414915, codegen 6cdba85, GLASS 3e910e1).
Key re-prioritization: **validation + measurement infrastructure (P1) is now ABOVE
further perf work**, driven by two hard learnings this effort — (1) we are optimizing
**unmeasured** (the sweep tooling stalls), and (2) the crba pass uncovered a **latent
branched-robot bug** that the 4-robot equivalence gate did not expose (coverage gaps
hide bugs). Both point to the same fix: a fast reference + broader, faster validation.

**DONE (committed/pushed on perf-cleanup):**
- Inner-owns + surgical-spill correctness wave, all 9 algorithms (validated 6/8 gate
  cases incl. g1 fixed+floating spill rungs; pure-relocation/byte-identical).
- GLASS primitives (segmented_gemv, indexed_batched_gemm, dot_strided_coalesced) wired
  into codegen. 5 perf passes: **id P1** (BFS-level GEMV fusion) + **minv P1**
  (forward-pass sync fusion) — green, racecheck-clean; **crba P1** (depth-stepped fill)
  — green + **fixed a latent parent-vs-jid S-index bug** on branched robots
  (matches `RBDReference H[ind,j]=S_j^T*fh`; numpy old=1.6err/fixed=0); **ee/d2ee P2**
  (floating dense->in-chain compaction, the ~535us outlier) — numpy byte-identical,
  CUDA re-validation DEFERRED to fast-reference; **fdsva_so P3** — documented NO-OP
  (coalesced dot doesn't fit the n^2-stride contraction; SO-spill cost is inherent
  global-mem latency, not coalescing).

**P0 — finish current wave:** DONE. The **crba S-index fix is CONFIRMED GREEN**
(2026-05-26) on the branched robots that exercise the path: baxter + fetch, fixed +
floating, all pass CUDA equivalence (crba never appears in any failure). One incidental
finding: baxter-**fixed** `q=0` is an rpy/atan2 **gimbal-lock singularity** — the EE
pose Jacobian/Hessian is non-finite in BOTH the reference and CUDA (not a codegen bug).
Handled by a harness guard that skips a comparison only when the *reference* itself is
non-finite (a finite reference is always asserted, so no real NaN is masked).

**KNOWN BUG — d2ee orientation rows (found 2026-05-26 — ✅ FIXED 2026-05-26, see STATUS
SNAPSHOT above for the fix + validation):** the
`end_effector_pose_hessian` (d2ee) **orientation (roll/pitch/yaw) rows are wrong at
non-small joint angles in BOTH the GRiD CUDA codegen AND the RBDReference analytic
hessian** — they match each other (so CUDA-vs-analytic equivalence passed and hid it),
but both diverge from the truth as joint angles grow (iiwa14-fixed: orientation-row
error 3e-6 near q=0 → 1.1e-3 at ~0.8 rad → **4.789 at large angle**, analytic max ~8.6).
Position (xyz) rows are correct everywhere. Found by pointing d2ee at the INDEPENDENT
finite-diff/pinocchio oracle (step-convergent, position rows exact, worst q not at
gimbal lock → the FD is the truth). Locus: the d²(roll/pitch/yaw)/dq² atan2 chain-rule
(`RBDReference.py:~1159-1165` + the matching GRiD `ee_pose_hessian` codegen). The CUDA
executable harness now compares d2ee against the independent oracle and lists it in
`KNOWN_FAILING_ALGORITHMS`, so mismatches are reported as a tracked known bug (loud, NOT
masked) but non-fatal. **TODO: fix the orientation-hessian formula in both, then drop
d2ee from `KNOWN_FAILING_ALGORITHMS`.** Note this also moots the "d2ee 72-min pole" plan:
the analytic d2ee was both slow AND wrong; the pinocchio finite-diff d2ee is fast
(~seconds) and correct, so it is the right oracle.

**P1 — Validation & measurement infra (TOP PRIORITY, unblocks everything):**
- **P1-A · Pinocchio + C++ fast reference** (keystone — see §7 item): sub-call
  composition + thin C++ glue, pinocchio kinematics/derivatives for EE, C++ finite-diff
  for the d2ee hessian, a `RBDReference` hooks file (one shared interface both sides),
  hybrid pure-Python fallback. Turns hours-long references into seconds. Then run the
  deferred P0 broad re-validation (now fast) to confirm crba + the whole wave.
  - **DONE 2026-05-26 — relocation + shared interface + one-line swap.** The whole
    pinocchio-equivalence layer moved OUT of `test/pinocchio_equivalents/` INTO the
    `RBDReference` submodule, so the reference is self-contained + self-testing:
    `RBDReference/equivalents/` is the reusable lib (`conventions.py` ← normalization;
    `reference_backend.py` ← the pure-Python adapter; `pinocchio_backend.py` ← the
    pinocchio+`pin_so_ext` adapter — both expose the IDENTICAL surface; plus
    `model_sources`/`source_lock`/`state_sampling`/`tolerances`/`comparators`, the two
    JSON manifests, and the `pin_so_ext` C++ ext). `RBDReference/tests/` holds the 17
    equivalence tests + conftest (817 cases collect green). The **one-line swap** is
    `RBDReference.equivalents.build_adapter(..., backend="reference"|"pinocchio")` /
    env `GRID_REFERENCE_BACKEND` — wired into the CUDA harness at the
    `build_adapter(...)` call. Requirements split: base `requirements.txt`
    (numpy+sympy) vs `requirements-dev.txt` (pin/robot_descriptions/bs4/pybind11/
    pytest). All 7 GRiD importers + `developer_install.sh` pin_so_ext path rewired;
    validated both backends build + agree on rnea (iiwa14, 7e-15) and iiwa14 rnea
    equivalence passes through the moved suite. **Committed + pushed** on `perf-cleanup`
    (submodule `abdbd95`, parent `3da85d2`, gitlink bumped).
  - **Pinocchio-as-oracle scope, settled 2026-05-26 (iiwa14-fixed end-to-end test):**
    the pinocchio backend is an **EXACT** CUDA oracle for ALL dynamics
    (rnea/aba/fd/minv/crba/rnea_grad/fd_grad) **and** `ee_pose` **and**
    `ee_pose_gradient`. It is **NOT** a valid oracle for **`d2ee`**: pinocchio's d2ee is
    finite-difference and blows up order-1 near the rpy/atan2 wraps (iiwa14 high_velocity:
    CUDA-analytic −2.08 vs finite-diff +0.11), so the CUDA *executable* harness — which
    compares analytic d2ee — stays a **single pure-Python oracle** (no half-pinocchio
    mix; `build_project_adapter`, documented inline). This revises the earlier
    "C++ finite-diff hessian" plan: keep analytic d2ee; a fast d2ee oracle, if ever
    needed, must be analytic, not finite-diff.
  - **Two distinct multi-hour reference poles (profiled h1_2-fixed, nv=51, 2026-05-26):**
    every dynamics algo + ee_pose/ee_pose_gradient is <1.1s; the poles are (1) the SO
    refs (`idsva_so`/`fdsva_so`) — solved cleanly by the exact `pin_so_ext` oracle; and
    (2) **`ee_pose_hessian` = ~4347s (≈72 min) for ONE evaluation** (12 leaves, analytic
    pure-Python). The executable harness runs d2ee per sample (~10), so h1_2-fixed
    executable equivalence is many hours, dominated entirely by analytic d2ee. Pinocchio
    CANNOT fix this (finite-diff invalid). Options for the d2ee pole (separate task):
    derive an analytic fast d2ee (pinocchio frame 2nd derivatives / C++), down-sample
    d2ee for the largest robots, or accept the cost. Distinct from the SO win.
  - **TODO (still P1-A) — the real pinocchio win:** wire the **EXACT** `pin_so_ext`
    oracle into the **second-order CUDA tests** (`idsva_so`/`fdsva_so`), the actual
    documented multi-hour pole, via the same project(codegen)/reference(oracle) split
    proven on the executable harness; expand those SO tests beyond iiwa14 to the big
    robots (g1/h1_2) where the pure-Python SO reference takes hours.
- **P1-B · Perf-measurement tooling:** `run_multi_version.py` stalls / 25-min compiles /
  can't parallelize (it's a bench). Fix (parallel compiles, isolate only the timing
  window) or build a light per-algorithm timing harness. *We have ZERO perf numbers.*
- **P1-C · Coverage + parallel testing:** once P1-A is fast, validate all 9 manifest
  robots x bases x tiers; add cores+RAM-sized parallel test launch (pytest-xdist `-n`
  or a (robot,base) sharding runner). See "Parallel equivalence testing" item below.

**P2 — Correctness/cleanup debts (from the wave):**
- **h1_2-floating device-path smem cap**: make inline `*_device` paths tier-aware/
  spillable (also completes the deferred **fd audit**) OR guard the runner setattr;
  then validate h1_2-floating kernels. (See §6 item.)
- **`gen_idsva_so_device` dispatcher reconcile**: drop the now-redundant XImats reload,
  thread `scratch_in_smem` by tier (body+world inners now own the load + take d_robotModel).
- **Stage 5**: naming/uniformity sweep; nvcc `-Werror` warnings sweep; code-bloat/
  streamlining audit (consolidate dup helpers across the 14 algos); design-doc
  reconciliation + encode friction corrections (crba NOT called by fd; idsva world cold
  buffer is `Xdown` not `f_w`; `integrator_with_gradient` lives in `_integrator_gradient.py`;
  add base `end_effector_pose_inner` conformance-table row; stale inner-signature docstrings).

**P3 — Data-driven perf (ONLY after P1-B tooling exists):**
- Full sweep (wave + GLASS passes) vs baseline `tier_sweep_20260525_002438`, incl.
  lite/minimal tiers. Then *measured* refinements: id backward distinct-parent grouping;
  aba floating interior-cold-hole; **idsva body-frame ~7x slower than world** (seen in the
  partial sweep block: body single ~3945us vs world ~547us) — investigate.

**P4 — Longer-term:** branch merge perf-cleanup -> modernizing-tests once validated;
remaining naming (§6) + pinocchio-alignment (§7) items.

### Resolved 2026-05-24/25 (no longer backlog)
- **LITE/MINIMAL spill crashes fixed**: (a) null-`s_temp` passed to the
  XImats/XmatsHom load helper in whole-arena spill rungs — `aba`,
  `ee_pose_gradient`; (b) a bug *class* — a single-valued PERF-pick macro
  (`GRID_*_USES_DA_DF_SPILL`) used as a **per-rung inner template flag** —
  `id_du`, `fd_du`, `integrator_gradient`. All committed; generated C++
  verified (LITE=`true`, PERF/MIN=`false`).
- **fdsva_so / idsva_so big-robot fit** via inner-owns-placement
  (`fdsva_so_full_inner` + world-inner `SCRATCH_IN_SMEM` + pool→global tier).
  h1_2 fdsva 198/142 KB → 53.8/45.8 KB; overnight gate + SO equivalence PASSED.
- **Full tier-validation sweep** ran (overnight + targeted re-sweep);
  iiwa/go2 12/12 all tiers; matrix in `test/benchmarks/results/tier_sweep_20260525_002438/`.
- **`ee_pose_hessian` (d2ee) promoted to a first-class algo**: registered in
  `ALGO_REGISTRY`; fixed the codegen single-timing printf label (was
  `EE_POSE_GRADIENT`); added to bench `PER_ALGO_SPECS`; equivalence runner
  standardized (real `hd_data->d_workspace`, removed the bespoke
  `GRID_CUDA_RUN_FLOATING_EEPOSE_HESSIAN` gate). FLOATING accuracy validated
  (iiwa14). Timing recapture + FIXED accuracy are backlogged (§4, §5).

### A. Pinocchio competitive gaps (perf work)
1. ✅ **d2ee analytic d/dv Hessian on GPU DONE (codegen `2bf6d53`, 2026-05-28
   shipped + 2026-05-29 PM verified).** Closed-form analytic via direct
   2nd-order Taylor expansion of the chain world transform; handles per-joint
   Δ + intra-joint multi-DOF (revolute / prismatic / SE(3) free-flyer)
   uniformly via chain composition `L_a · A^{local}_i · P_{a→b} · A^{local}_j ·
   R_b`. Python reference: `RBDReference.end_effector_pose_hessian_analytic`
   (RBDReference `843a302`). Validated to ~1e-9 on iiwa14/floating +
   go2/floating; FD-noise-floor (1.6e-7) on iiwa14/fixed near pitch=π/2. GPU
   path validated via the analytic-vs-pinocchio-analytic oracle (`pin_so_ext`):
   iiwa14 fixed + iiwa14 floating GREEN at machine precision. Full derivation
   in `docs/d2ee_analytic_derivation.md`. **Open speedup measurement
   (analytic-vs-pin at N=256) → C.7 sweep will capture.** Side benefit:
   surfaced an fr3 mimic-joint bug in URDFParser (D.2; partial Python fix
   landed but parked, see Done list).
2. **idsva_so big-robot scaling.** g1/h1_2 lose 2.5–2.7× at N=256 batch; nv³
   kernel is compute+smem-bound. Same family as A.1; ties to ancestor-scratch
   de-alias (B.1).
3. **Core-dynamics batch losses (floating-base only).**
   **2026-05-29 — ABA + Minv FIXED, CRBA per-jid Phase 2 + BFS-parallel
   Phase 1 BOTH LANDED.** ALL losses are/were floating-base. Root cause was
   the single-threaded 6×6 root invert. GLASS `invertMatrix_dense` swap
   landed (GLASS 773cff5, codegen 0e0a9e2, call-site cleanup 617a057, CRBA
   refactor `9eb51a6` per-jid Phase 2 + `809b145` BFS-parallel Phase 1).
   **Measured wins on iiwa14-floating** (pre-`809b145`): ABA 2.12×
   (109.7→51.9 µs, 1.89× pin loss → 1.09× win); Minv 1.77× (66.7→37.6 µs,
   1.11× pin loss → 1.55× win); CRBA 1.26× (35.0→27.8 µs, 1.42× pin
   loss → 1.14× pin loss). **Post-`809b145` + `b140319` surgical XImats
   (2026-05-29 PM):** iiwa14 stays byte-identical (1-wide BFS);
   go2/g1/h1_2 floating expected to pick up additional wins from sync
   reduction (go2 Phase 1 ~24→6 syncs ≈ ~1.8 µs/launch; g1 ~38 syncs ≈
   ~3.8 µs/launch). Surgical XImats `b140319` adds `SKIP_FLOATING_BASE_X`
   template to elide the per-call recomputation of the floating root
   `s_XImats[0..35]` (CRBA never reads X[0]) — closes more of the iiwa14
   residual gap. C.7 sweep will measure all of these. See
   `docs/a3_core_dynamics_floating_loss_audit.md` for the full audit +
   measured A/B table.
4. ✅ **`fdsva_so` pinocchio baseline DONE (parent `c6ab64c`, 2026-05-29).**
   Pinocchio has no direct fdsva_so; synthesized via chain rule:
   `ComputeRNEASecondOrderDerivatives` (RNEA SO tensors) +
   `computeABADerivatives` (Minv + fd_dq + fd_dqd) → apply chain rule. Wired
   into `timing_parser.py` + `run.py` + `generate_report.py` so C.7 sweep
   picks it up. iiwa14 fixed single 22.4 µs/iter, floating 67.4 µs/iter.
   **Open:** numerical cross-check vs GRiD output deferred (formula matches
   Python `pinocchio_backend.fdsva_so` exactly).

### B. Architecture cleanup (gated on perf data)
1. ✅ **De-alias `idsva_so` / `fdsva_so` inners — DONE in prior commits,
   verified 2026-05-29 (codegen `124be77` docs only).** The body inner now
   has `SCRATCH_IN_SMEM × BC_IN_SMEM` (whole-arena + surgical BC); world
   inner has `SCRATCH_IN_SMEM × COLD_IN_SMEM` (whole-arena + surgical cold
   trio Xdown/v_w/a_w); fdsva_so_device has the canonical 3-lever pattern.
   Conformance audit in `docs/idsva_so_inner_refactor_notes.md` shows ALL
   algorithms now conform to the inner-owns-placement design.
2. ~~**ABA whole-arena → surgical retrofit** (Minv-F sub-split pattern).~~
   **DONE (pre-2026-05-29 audit):** `gen_aba_inner_floating`
   (`_aba.py:7-60`) already exposes the `TEMP_IN_SMEM` × `COLD_IN_SMEM`
   selective-spill levers — the cold `vcross` slab `[36*NJ, 72*NJ)` and the
   floating-base root tail `[140*NJ, 140*NJ+138)` spill via `s_vcross_cold`
   / `s_fb_cold` to packed `d_cold` (`= d_workspace`); the hot recursion
   stays in `s_temp`. This backlog entry was stale.
3. ✅ **idsva body kernel `output_temp` → body-`_device` `SCRATCH_IN_SMEM` —
   ALREADY DONE in prior commits, verified 2026-05-29 (codegen `124be77`
   docs only).** The body kernel at `_idsva_so.py:2390` maps
   `s_temp_in_global=True` → `SCRATCH_IN_SMEM=false` and calls the body
   inner via `gen_idsva_so_body_frame_inner_function_call(... scratch_in_
   smem_expr = ...)`; the inner does the repoint at line 1372.
4. ✅ **h1_2-floating inline DEVICE path smem cap — DONE 2026-05-29 PM**
   (codegen `8f39604` + parent `27c2f0b`). `forward_dynamics_device` is
   now tier-aware: at `TIER_LITE`/`TIER_MINIMAL` the whole FD inner arena
   routes to L2-pinned `d_workspace` (freeing ~120 KB smem). Runner uses
   `TIER_MINIMAL`. Mirrors the established tier pattern from
   `idsva_so_device` / `d2ee_device` / `id_du_device`. **NOT end-to-end
   validated on h1_2-floating yet** (blocked by `idsva_so_body_frame_inner`
   `t_index_map` pre-existing NJ vs NV bug when mimic URDFParser is used;
   not blocking C.7 since parent's URDFParser is mimic-unaware).
5. *(deferred B.1 follow-ups)*
   ✅ **B.5.b cosmetic rename DONE 2026-05-29 (codegen f433d9c, parent
   124e690):** `fdsva_so_inner` → `fdsva_so_contract` across 20 refs in
   `_fdsva_so.py` + `GRiDCodeGenerator.py` + `tier_instantiation_smoke.py`;
   ``_temp_no_inner`` local var also renamed to `_temp_no_contract`.
   iiwa14 fixed + floating regen verified.
   - **REMAINING:** collapse the simple-algo auto-alloc `_device` wrappers
     (id / minv / fd / aba / crba / ee_pose* / integrator / idsva_so_*).
     Touches the equivalence-runner test kernels that still consume them
     (~2-4 hour refactor).
6. **Codegen interface cleanup.** Pays back on every future algorithm
   addition. Sub-items:
   (a) ✅ **Drop thread-group plumbing — FULLY DONE 2026-05-29 (codegen
   `b228756` partial, then `75089f8`/parent `1ff8d54` complete).**
   `b228756` stripped 191 lines of dead `if use_thread_group:` branches
   across 15 codegen files. `75089f8` dropped the `use_thread_group`
   parameter itself across 1079 scrubs in 16 files. Audit 2026-05-29
   EVENING-2: grep `use_thread_group` across `GRiDCodeGenerator/**.py`
   returns 0 hits. iiwa14 fixed+floating + go2 floating equivalence GREEN.
   (b) ✅ **Consolidate emitter helpers — MAIN PASS DONE 2026-05-29 (codegen
   `75089f8`):** consolidated `gen_kernel_load_inputs` / `gen_kernel_save_result`
   with their `_single_timing` variants (single fn each via optional stride
   kwarg, 74 call sites rewritten). 2026-05-29 EVENING-2 polish also dropped
   `select_shared_tier` (singular, shadowed by 3way), `gen_add_debug_print_code_line`
   (singular), and several dead `_temp_mem_size` shells (codegen `bd1bdd3`).
   Residual: one-off `gen_add_*` shims in `_code_generation_helpers.py` could
   still collapse into a smaller canonical set (parallel-loop helper, workspace
   pointer-carve helper) — fold this into the next emitter rewrite.
   (c) **Dedup repeated branches** — algorithm emitters fan out on
   `compute_c` / `use_qdd_input` / `use_qdd_Minv_input`; factor into
   table-driven helpers. *2026-05-29 scoping note:* surveyed
   `_inverse_dynamics.py`, `_forward_dynamics.py`,
   `_inverse_dynamics_gradient.py`, `_forward_dynamics_gradient.py` (~78
   flag refs). These flags gate conditionally-augmented emission inside
   one body (function-name string, extra param, extra code line) — NOT
   duplicate parallel blocks. Real dedup requires a new design layer
   (emit-spec dataclass per variant, then a single shared traversal) —
   non-mechanical, multi-hour. Deferred until paired with a wider
   emitter rewrite.
   (d) ✅ **Codegen author guide DONE 2026-05-29 (parent 4394343):**
   `docs/source/user_guide/tutorials/adding_an_algorithm.rst` — step-by-
   step worked example using fdsva_so, common pitfalls (caller-side
   s_temp repoint, single-thread invert, PERF-pick macro misuse), helper
   cheat sheet. Linked from the tutorials index.
   Large surgery but high leverage — directly attacks the code-bloat the user
   has flagged repeatedly.

### C. Cleanup + comprehensive perf re-sweep (do as one phase)
1. **Validation matrix completion** — PERF green for all 4×2 (pre-existing);
   **LITE on big robots NOW GREEN 2026-05-29** (g1-fixed + g1-floating +
   h1_2-fixed all PASSED at `GRID_CUDA_TARGET_SHARED_MEM_BYTES=49152`).
   h1_2-floating SKIPPED on the pre-existing `forward_dynamics` smem cap
   (120 KB > 101 KB) → tracks as B.4. MINIMAL on big robots (target=16384
   or similar) still uncollected; ~1 hr run if wanted before C.7.
2. ✅ **Auto-parallel equivalence harness DONE 2026-05-29 (parent
   084af68):** `test/cuda_equivalents/run_parallel.sh` sizes
   `pytest-xdist -n` from `free -g / GB_PER_JOB` (default 5 GB), clamps
   to `nproc`, uses `--dist loadgroup` to keep header cache hot per
   (robot, base). `pytest-xdist` added to `requirements-dev.txt`.
3. ✅ **nvcc/ptxas `-Werror`-style warnings sweep DONE 2026-05-29:** iiwa14
   floating compile clean under `-Wall -Wextra` (no #177-D). The
   `(void)dof_id;` casts at all 5 emit sites in
   `_inverse_dynamics_gradient.py:365` already suppress the
   `inverse_dynamics_gradient_device_qdd` warning class; the earlier
   HANDOFF entry was stale. **Bigger robots / non-PERF tiers still TBD**;
   would re-probe after the overnight sweep.
4. ✅ **Autotune `performance_threads` DONE 2026-05-29 (parent `32ac963`).**
   New `--autotune-threads` flag in `test/benchmarks/baselines/grid/run.py` +
   `run_multi_version.py`. Coarse sweep + midpoint refinement, picks land
   in `algo_picks` of benchmark JSON. ~14 s/(robot, base) wall. Backward-
   compatible. Follow-ups: codegen-cap-aware grid clipping, floating +
   branched untested, single-call path not yet tuned.
5. ~~**`fdsva_so` single_us recapture** — was dropped on -rdc regcount
   error (now fixed).~~ **DONE — verified 2026-05-29:** iiwa14-fixed
   `fdsva_so` reports `single_us` = 49.5 μs in the latest sweep
   (`ee_grad_step_c_perf_v2`); idsva_so + aba also present. The single_us
   pipeline is healthy on all collected algos.
6. ~~**Vendor URDFs instead of pulling `robot_descriptions`.**~~
   **DONE 2026-05-29 (parent 34492f7, RBDReference 990786a):** vendored
   the 9 smoke-tier URDFs (~255 KB total) at `robot_assets/` with SHA
   provenance in `URDF_SOURCES.md`. Added `"vendored"` source_kind to
   `model_sources.py`; manifest now lists it FIRST with the
   `robot_descriptions` candidate as fallback. Equivalence resolution
   verified vendored-first; the multi-GB `robot_descriptions` cache is
   now only needed for new robots not yet vendored.
7. **→ Full perf re-sweep** with everything cleaned up. Captures fdsva_so
   single_us + d2ee timing under the standard pipeline (today both required
   manual binary runs).

### D. Reference + runtime modernization (separate effort, post-perf)
1. **URDFParser/RBDReference pinocchio-alignment** — strict-parse API, structured
   diagnostics, `nq`/`nv`/quat-order helpers, mark fixed-base-only methods. See
   `RBDReference/tests/PINOCCHIO_ALIGNMENT_BACKLOG.md`.
2. **URDF feature support — PARTIAL LANDED on submodule branches, NOT yet
   in parent.** Mimic joints (the urgent missing feature) parsed +
   kinematic-Jacobian / analytic-Hessian aware + CRBA/MINV/RNEA-grad/
   FD-grad/ABA mimic-aware (Python) as of 2026-05-29. Pin backend mimic-
   bypass via implicit-function decomposition for FD-grad. Plan for
   CUDA codegen propagation in `docs/d2_codegen_mimic_plan.md`. URDFParser
   `511e398` (on `modernizing-tests` branch) + RBDReference `0b1a89d` (on
   `perf-cleanup` branch). fr3 analytic d²(pose)/dv² now matches FD oracle
   to ~2e-11 (was the A.1 false-positive) and matches pinocchio's
   `getJointKinematicHessian(LOCAL_WORLD_ALIGNED)`. h1_2 (12 mimics,
   multipliers 1.0/1.6/2.4) kinematics + RNEA tests GREEN (173/173 across
   kinematics+RNEA+metadata).
   **Why parked, not bumped into parent:**
   (a) **Dynamics ripple OPEN.** `aba`, `crba`, `minv`, RNEA-grad, fd-grad
   still ASSIGN to mimicked column instead of `+= mimic_scale * …`. The
   Robot.py nq/nv reduction surfaces this; tests
   `test_aba_equivalence[fr3]`, `test_crba_equivalence[fr3,h1_2]`,
   `test_minv_equivalence[fr3]`, `test_integrator_pinocchio_equivalence[fr3]`
   now fail (were passing by coincidence with naive-on-naive). Fix mirrors
   the RNEA pattern.
   (b) **CUDA codegen has NO mimic awareness.** Kernel signatures use
   `NUM_POS`/`NUM_VEL` macros that don't drop mimic; S/Xmat unpack treats
   each joint independently. Bumping submodules into parent without
   codegen propagation makes CUDA-equiv vs RBDReference mismatch on
   fr3/h1_2 across multiple algos.
   (c) **Pre-existing FK orientation bug surfaces here.**
   `URDFParser/Joint.set_type` composes `Xmat_sp_hom` by directly summing
   `t_free + t_origin` without rotating `t_free` through the origin's
   rotation. Invisible on every robot whose origin rpy is zero (iiwa, go2,
   g1, fr3_finger_joint1 — most cases). Mis-places fr3_finger_joint2
   (origin rpy = π around Z). Independent of mimic but blocks fr3
   selection of finger_joint2 as an EE target.
   **Pickup recipe:** finish dynamics ripple in RBDReference; propagate
   mimic awareness into CUDA codegen (NUM_VEL macros, S/Xmat for mimic);
   fix the FK orientation bug; then bump submodules into parent and
   un-skip fr3/h1_2 in CUDA equivalence. Estimated: 1 focused day for the
   dynamics ripple, 1+ day for codegen propagation, ~hour for the FK bug.
3. **PyTorch in-memory compile + re-link → CUDA-Graphs callable.** Use
   `torch.utils.cpp_extension.load_inline` (or equivalent JIT path) to compile
   a per-robot generated header *in memory* and expose the resulting kernels
   as PyTorch ops. The CUDA-Graphs angle is the load-bearing motivation: ops
   that survive `torch.cuda.graph(...)` stream-capture let downstream MPC /
   training loops capture a whole step into a graph, avoiding per-launch
   overhead. Today GRiD generates a `.cuh` that has to be compiled offline
   into a `.so` then loaded — slow iteration loop for users specializing per
   robot. Pair with the grid-rbd binding so PyTorch users can `import` a
   newly-codegen'd robot without touching nvcc.
4. **Runtime mass / inertia parameters (two-variant emit).** Today XImats and
   robot constants are baked into the codegen (consts folded into PTX → fast).
   Add a runtime-parameter variant: kernels read inertias from a per-robot
   parameter struct passed at launch, enabling system-ID, adaptive control,
   parameter sweeps, and online adaptation without re-codegen. Keep the
   baked-constant fast path as the default; the runtime path is opt-in and
   expected to cost ~10–30% (more on small robots where constant folding
   matters most, less on big robots where memory bandwidth dominates). Two
   templates per algorithm: emit both, measure the gap on the standard sweep.

### E. Branch merge
- **`perf-cleanup` → `modernizing-tests`** — held pending validation; most is
  green. Gate on C.1 (full validation matrix) + any d2ee/fdsva_so spill-path
  bugs surfaced. (The earlier `humanoid-tier-spill` merge happened pre-branch.)

### Done (since this backlog was last refactored 2026-05-28)
- **2026-05-29 EVENING-2 batch (h1_2 MINIMAL bug verification + rpy-snap fix + polish/cleanup):**
  - **Bug 1 (h1_2 MINIMAL CRBA `M[0,13]≈0`) — STALE, RETRACTED.** The earlier
    HANDOFF entry (lines 1195+ in the previous revision) flagged this as a
    pre-existing tier-emit bug. An empirical verification run (compiled MINIMAL
    runner direct-probe on h1_2-floating-zero) shows the current codebase
    returns `M[0,13] = -3.856656` (matches expected ~-3.86) **deterministically
    across threads ∈ {32, 64, 128, 256, 384, 512}**, with M symmetric to ~1e-7.
    No memory-ordering hazard. No codegen edit needed. The BFS-parallel CRBA
    refactor (`809b145`) + per-jid chain-walk refactor (`9eb51a6`) +
    URDFParser FK + rpy-snap fixes closed the gap silently. **The h1_2-fixed
    `M[0,13]=0` value is analytically correct** (jid 0 = `left_hip_yaw`,
    jid 13 = `left_shoulder_pitch`, disjoint subtrees → zero cross-inertia);
    the old HANDOFF leaked the floating-base "expected -3.86" into the
    fixed-base narrative incorrectly.
  - **Bug 2 (h1_2 ee_pose ±π) — FIXED in URDFParser `ce01c53`** (perf-cleanup).
    Root cause: `sp.nsimplify(tolerance=1e-6)` in `Joint.py` leaves a 3.67e-6
    residual on URDF rpy=`"-1.5708"` (which means -π/2 truncated). The residual
    cascades through float32 kinematic chains and the GPU `atan2` in ee_pose
    rpy extraction flips a yaw component by exactly π (e.g. h1_2 L_thumb_distal
    yaw -4.71 vs CPU-ref -1.57). Fix: rpy-grid snap before nsimplify — snap to
    exact `N*π/2` for `|N| ≤ 4` when input is within 1e-5 of the grid point.
    Bounded N keeps the snap targeted. **Byte-identical for iiwa14/go2/g1**
    (their URDF rpy values are either exact 0 or full-precision π/2, already
    captured by existing nsimplify). **Snaps 6 joints on h1_2** (thumb proximal
    yaw/pitch L+R, R_middle_proximal, etc.). Pure URDFParser-side fix; no codegen
    edit. Equivalence validation in flight at HANDOFF time.
  - **Polish/cleanup A-batch — codegen `bd1bdd3`.** Dropped ~110 lines of pure
    dead Python (zero callers verified across `.py`/`.cu`/`.cuh`): the singular
    `select_shared_tier` (shadowed by the 3-way variant), `gen_add_debug_print_code_line`
    (singular; only the `_lines` plural is used), `_any_algo_uses_workspace_spill`
    (planned L2-persistence hook that landed elsewhere), `_gravity_shim_full_spill_count`
    (replaced by `_gravity_shim_use_full_spill`), `gen_idsva_so_body_frame_device`
    + `gen_idsva_so_body_frame_device_temp_mem_size` (no callers — body_frame is
    dispatched through `gen_idsva_so_device`), and the two
    `gen_end_effector_pose_gradient{,_hessian}_device_temp_mem_size` shells.
    Also removed the stale `# self.gen_idsva_so_body_frame_device(False) TODO`
    commented call. **Codegen output unchanged** (Python-only dead-code).
  - **REAL h1_2-floating MINIMAL equivalence failure (the one the morning
    C.1 MINIMAL run actually surfaced):** a **shape mismatch** — CUDA M is
    (57, 57) while the mimic-aware RBDReference/`pinocchio_backend` reference
    is (45, 45). h1_2 has 12 mimic joints (51 raw → 39 reduced) + 6 base =
    45. This is the D.2 CUDA codegen-mimic gap, planned at
    `docs/d2_codegen_mimic_plan.md` (4-phase, ~3 focused days). Affects every
    algo whose output dimension scales with NV (crba, minv, rnea, fd,
    ee_pose_gradient, ee_pose_hessian). Does NOT affect `ee_pose` (the rpy
    extraction output is `6 * N_ee`, independent of NV).
    **For the C.7 perf sweep:** timing is unaffected by the shape mismatch
    (the sweep measures kernel runtime, not equivalence). h1_2 floating timing
    will be reported on size-57 not size-45 matrices, same as the prior
    `tier_sweep_20260525_002438` baseline (apples-to-apples comparison).
  - **Sweep scoping decision** (resolved this session): **launch the full
    sweep, all 4 robots × {fixed, floating} × {PERF, LITE, MINIMAL}**. No
    h1_2 MINIMAL skip needed — the equivalence shape-mismatch is irrelevant
    for timing. Plan: `python test/benchmarks/run_multi_version.py
    --robots iiwa14 go2 g1 h1_2 --bases fixed floating --columns glass
    --tiers perf lite minimal --autotune-threads --output-dir
    test/benchmarks/results/perf_cleanup_<ts>`. Gated on (a) the rpy-snap
    equivalence validation clearing, (b) the user's personal review of D.2
    RBDReference (`0b1a89d` → `d0e552a`).
- **2026-05-29 EVENING evening batch (8-agent + 2-direct landings):**
  - **D.2 FK orientation fix** — URDFParser `acbdab8` (perf-cleanup branch) +
    `8182770` (modernizing-tests branch). `Joint.set_type` now rotates `t_free`
    through the origin's rotation before adding `t_origin`. Invisible when
    origin rpy=0 (every iiwa/go2/g1 revolute, fr3_finger_joint1, etc.); fixes
    fr3_finger_joint2 (origin rpy=π around Z). fr3 world_T_finger_joint2 chain
    composition now matches pinocchio `oMi` to ~9e-13 (was 1.25e-01); iiwa14
    + go2 + g1 + h1_2 + baxter unchanged at machine precision. Parent
    `675249d` bumps URDFParser to perf-cleanup tip. **CORRECTION:** earlier
    HANDOFF claimed gen3 + fetch had PRE-EXISTING URDFParser composition bugs
    based on chain diffs ~7.9e-1 and ~1.0e0 vs pinocchio. The follow-up
    rotation-block audit (2026-05-29 PM) found this was a **validation
    harness artifact**, not a URDFParser bug: gen3 + fetch have URDF
    `continuous` joints which pinocchio's `buildModelFromUrdf` encodes as
    `JointModelRUBZ` (2-D `(cos, sin)` q encoding, so `pin.nq > grid.nq`).
    Filling `pin_q[idx_q] = theta` puts the raw angle in the cos slot and
    0 in the sin slot. The `pinocchio_backend.py` adapter via
    `expand_continuous_joint_positions_for_pin` (`conventions.py:51-73`)
    already handles RUBZ correctly. With proper pin_q construction, gen3
    max diff is 5.9e-11 and fetch is 2.1e-12 — both machine precision. **No
    URDFParser rotation block bug exists.** Original "side finding" left
    in the historical record for cross-reference; superseded by this audit.
  - **A.3 surgical floating XImats lever** — codegen `b140319`. Added
    `SKIP_FLOATING_BASE_X` template flag to `gen_load_update_XImats_helpers`;
    CRBA caller passes `skip=True` to elide the per-call recomputation of
    `s_XImats[0..35]` (the heavy floating-root quat→rotation expansion). CRBA
    never reads `s_XImats[0..35]` — Phase 1 BFS starts at level 1 and
    Phase 2's chain walk only dereferences `X[X_id != 0]`. iiwa14 + go2 + g1
    floating equivalence GREEN.
  - **B.4 forward_dynamics_device tier-aware** — codegen `8f39604` + parent
    `27c2f0b` runner threading. `forward_dynamics_device<T, RESOURCE_TIER>`
    now accepts `d_workspace`; at `TIER_LITE`/`TIER_MINIMAL` the FD inner
    arena (incl. the Minv-F band) routes to L2-pinned `d_workspace`, freeing
    ~120 KB smem on humanoid-scale robots. Mirrors `idsva_so_device` /
    `d2ee_device` / `id_du_device` tier pattern. Equivalence runner now
    calls `forward_dynamics_device<T, TIER_MINIMAL>` with the existing
    `hd_data->d_workspace` so the inline device path can fit h1_2-floating.
    Added `FD_DEVICE_INLINE_{SMEM,WORKSPACE}_BYTES<T,TIER>` macros.
    iiwa14 fixed+floating + g1 fixed equivalence GREEN. **Not end-to-end
    validated on h1_2-floating yet** (blocked by `idsva_so_body_frame_inner`
    pre-existing bug — see below).
  - **D.2 dynamics ripple (CRBA/MINV/RNEA-grad/FD-grad)** — RBDReference
    `8c351ad` (perf-cleanup). All four functions now mimic-aware with the
    canonical `+= mimic_scale *` accumulation pattern; `_qd[idx]` /
    `gravity-transport` reads scale by `α`; indexing uses
    `get_joint_index_v(ind)` throughout. `minv()` detects mimic robots and
    falls back to `np.linalg.inv(crba(q))` (pinocchio's reduced-model
    `M^{-1} = (G^T M_full G)^{-1}` is not equal to `G^T M_full^{-1} G` so
    the ABA-style recursion can't be `+=`-patched). Side fix in
    `pinocchio_backend.py`: reordered `rnea_grad` / `forward_dynamics_grad`
    so mimic-axis reduction runs BEFORE the q-Jacobian chain reducer.
    **Validated:** 5 target tests GREEN (`crba[fr3-fixed]`,
    `crba[h1_2-fixed]`, `minv[fr3-fixed]`, `rnea_grad[fr3-fixed]`,
    `rnea_grad[h1_2-fixed]`). Full RBDReference suite:
    707 PASSED / 50 failed / 68 skipped; all 50 failures are
    ABA-bound on fr3/h1_2 (next task — see D.2 OPEN ISSUE below). Zero
    non-mimic regressions.
  - **d2Xhom_owners defensive fix** — codegen `9c61e37`.
    `_global_hom_second_derivative_matrices` fixed-base branch now builds
    NJ-length owners list (was n_pos-length, IndexError when NJ != n_pos).
    Surfaces only with mimic-aware URDFParser on h1_2-fixed (51 joints / 39
    DoFs). Defensive: doesn't change current sweep behavior (parent's
    URDFParser pointer is mimic-unaware).
  - **D.2 ABA mimic** — RBDReference `bea0ac1` (perf-cleanup, parked).
    Algebraic-decomposition strategy: `qdd = M_reduced^{-1} * (tau -
    rnea(q, qd, 0))`. Sidesteps the per-body `(S, U, d)` recursion (whose
    α/α² scaling diverges from slot-accumulated `+=`) by reusing
    already-mimic-aware CRBA + RNEA. Bonus fixes: `has_invertible_mass_
    matrix` checks reduced M (pin's unreduced M is structurally singular
    for mimic models); `_pin_dIntegrate` size handling for integrator;
    h1_2 aba/rnea/minv tolerance overrides (cond(M_reduced) ~7e5 fixed /
    ~5e6 floating amplifies cross-library float64-eps to ~1e-3). Target
    tests GREEN: `aba[fr3-fixed]`, `aba[fr3-floating]`, `aba[h1_2-fixed]`,
    `aba[h1_2-floating]`, `forward_dynamics[*-fr3-*]`/`[*-h1_2-*]`,
    integrator state for fr3/h1_2.
  - **D.2 FD-grad mimic** — RBDReference `aa3eaa1` (perf-cleanup, parked).
    Pin backend bypass: pin's `computeABADerivatives` per-body recursion
    on unreduced model doesn't commute with `+= alpha *` fold (same root
    cause as ABA). Use implicit-function identity: `dqdd/dq = -M^{-1} ·
    drnea/dq`, all three components already mimic-aware. fr3 + h1_2
    fixed/floating GREEN; non-mimic 16/18 unchanged (2 rizon4 pre-existing
    skip).
  - **idsva_so_body_frame_inner t_index_map fix** — codegen `b573849`.
    Pre-existing bug: `t_index_map` sized `NV × NV` but indexed by `jid`
    (assumes `NJ == NV`). Now sized `NJ × NJ`; S/psid downstream stays jid-
    indexed. iiwa14 + g1 floating equivalence GREEN (regression check).
    h1_2-fixed regen smoke (with mimic-aware URDFParser, 51 joints / 39
    DoFs) now succeeds where it previously raised IndexError.
  - **d2Xhom_owners defensive fix** — codegen `9c61e37`.
    `_global_hom_second_derivative_matrices` fixed-base branch now builds
    NJ-length owners list (was n_pos-length, IndexError when NJ != n_pos).
    Defensive: doesn't change current sweep behavior.
  - **D.2 CUDA codegen mimic SCOPING DOC** — `docs/d2_codegen_mimic_plan.md`
    (parent, uncommitted at HANDOFF time). 4-phase plan ~3 focused days
    total: (P1, 1d) foundation helpers + ID + CRBA `+= α_i α_j` fold;
    (P2, 0.5d) `direct_minv` for mimic via CRBA-then-invert, `aba_kernel`
    for mimic via Minv·(u−c) algebraic decomposition; (P3, 0.75d)
    gradients (ID-du + FD-du + integrator); (P4, 1d) ee_pose + idsva_so
    + fdsva_so. Critical insight: per-body ABA / direct-Minv recursion
    fundamentally builds `M_full^{-1}`, not `M_red^{-1}`, so those algos
    can't be `+= α`-patched in codegen either; must fall back to CRBA +
    invert. Also flagged: `NUM_JOINTS` C++ macro is actually `nq`
    (reduced), not raw NJ — naming mismatch dangerous, P1 renames to
    `NUM_POS` + adds `NUM_LINKS` for un-reduced count.
  - **C.4 autotune `performance_threads`** — parent `32ac963`. New
    `--autotune-threads` flag (+ `--autotune-thread-grid`,
    `--autotune-N`). Coarse sweep + midpoint refinement, persists picks
    under `algo_picks` in benchmark JSON. ~14 sec wall per (robot, base).
    iiwa14-fixed example picks: aba 128 thr→14.26 µs, crba 224 thr→10.89
    µs, fdsva_so 192 thr→81.22 µs. Backward-compatible (no flag = old
    schema). Follow-ups: codegen-cap-aware grid clipping (currently wastes
    384/512 probes on small robots), floating + branched untested,
    single-call path not yet tuned.
  - **RUBZ audit on pinocchio_backend continuous joints** — CLOSED, NO
    BUG. `pinocchio_backend.py:_to_pin_q → _expand_project_q_to_pin_full
    → normalize_project_q_for_pin → expand_continuous_joint_positions_for
    _pin` (in `conventions.py:51-73`) already handles continuous-joint
    expansion to `(cos, sin)` correctly. gen3 (4 continuous joints) and
    fetch (5 cont) are the only RUBZ robots in the manifest; both match
    pinocchio at ~5e-11 / ~2e-12 via the adapter. The FK rotation agent's
    earlier failure mode was test-harness-specific.
  - **Open issues after this evening evening batch:**
    - **D.2 CUDA codegen mimic-awareness NOT done.** Bumping
      URDFParser+RBDReference into parent would now break fr3/h1_2 CUDA
      equivalence (Python correct, CUDA still naive). URDFParser
      perf-cleanup (`acbdab8`) is FK-only; mimic commits live on
      `modernizing-tests` (`511e398`, `8182770`). RBDReference
      perf-cleanup at `aa3eaa1` has D.2 morning + dynamics + ABA + FD-grad;
      parent still pins `990786a`. Net: D.2 parked. Plan in
      `docs/d2_codegen_mimic_plan.md` ready to pick up.
    - ✅ **D.2 idsva_so / fdsva_so mimic DONE** (Python side) —
      RBDReference `d0e552a`. Per-body unique internal slot indexing +
      R-matrix axis fold to project layout (elegant solve for "last-
      write-wins" issue with mimic-sharing v-slots). Pin backend bypass:
      3-axis reduce of unreduced pin SO tensor for `idsva_so`;
      `pin.aba`/`pin.computeMinverse` swap for mimic-aware
      `self.aba`/`self.minv` in fdsva_so. **All 52 SO equivalence tests
      GREEN** (idsva_so body+world + fdsva_so on fr3 + h1_2 fixed/floating).
      Full RBDReference suite: **765 passed / 26 failed / 34 skipped**
      (was 707/50/68 yesterday — net +58 passes). All 26 remaining
      failures verified pre-existing, none from D.2 work. Tolerance entry
      added for `(h1_2, second_order_fdsva)` (rtol=1e-4, atol=1e-1) —
      cond(M_reduced)~4.4e6 amplifies cross-impl 1e-7 to 1e-3 absolute
      through `Minv @ ... @ Minv` composition; relative residual stays
      ~1e-7. Same precedent as existing g1/h1_2 entries.
    - **D.2 ABA external-forces** — `f_ext` not threaded through mimic
      fast path. Niche feature, no failing tests. Defer.
    - **NEW finding: h1_2 MINIMAL-tier CRBA + ee_pose bugs surfaced by
      C.1 MINIMAL run** — **RETRACTED / SUPERSEDED** by the EVENING-2
      batch above. CRBA `M[0,13]≈0.001` claim was stale (current codebase
      returns -3.856656 deterministically; HANDOFF entry pre-dated the
      BFS-parallel CRBA refactor `809b145` + chain-walk refactor `9eb51a6`
      + URDFParser FK fix `acbdab8` + rpy-snap `ce01c53`). For h1_2-fixed
      `M[0,13]=0` is analytically correct (disjoint subtrees).
      `end_effector_pose ±π` was real but is fixed by the rpy-grid snap
      in URDFParser `ce01c53`. The actual remaining h1_2-floating MINIMAL
      equivalence failure is the D.2 codegen-mimic shape mismatch (57 vs
      45), tracked in `docs/d2_codegen_mimic_plan.md`.
  - **User-flagged review obligation**: user wants to personally review
    all 2026-05-29 D.2 RBDReference changes (`0b1a89d`, `8c351ad`,
    `bea0ac1`, `aa3eaa1`, + the idsva_so/fdsva_so commit when it lands)
    before E merge. Memory note set; do NOT auto-bump RBDReference into
    parent without flagging this.
- **2026-05-29 PM batch (parallel sub-agent sweep + investigation):**
  - **A.1 GPU d2ee analytic port — VERIFIED ALREADY DONE in codegen `2bf6d53`**
    (yesterday's analytic landing was sibling Python + GPU, not Python-then-GPU
    as HANDOFF previously suggested). Re-validated iiwa14 fixed + iiwa14
    floating (with `GRID_CUDA_FLOATING_ALGORITHMS=…hessian`) GREEN.
  - **A.3 BFS-parallel CRBA floating body** — codegen `809b145`: fused per-
    level forward+backward over `36*k` siblings at each BFS level; iiwa14
    stays byte-identical (1-wide BFS); go2 Phase 1 collapses ~24 syncs → ~6;
    g1 has partial-shared-parent siblings handled via atomic accumulation.
    iiwa14 floating + go2 floating + g1 floating equivalence GREEN. Estimated
    saves go2 ~1.8 µs/launch, g1 ~3.8 µs/launch (C.7 sweep will measure).
  - **A.4 pinocchio fdsva_so synthesis baseline** — parent `c6ab64c`: chain
    `ComputeRNEASecondOrderDerivatives` + `computeABADerivatives` + `Minv`
    (mirrors `pinocchio_backend.fdsva_so`). Wired into `timing_parser.py` +
    `run.py` + `generate_report.py`. iiwa14 fixed single 22.4 µs/iter,
    floating 67.4 µs/iter. Numerical cross-check vs GRiD output deferred
    (formula matches the Python backend so high confidence).
  - **B.3 idsva body `output_temp` → SCRATCH_IN_SMEM + B.1 idsva/fdsva
    cold-only de-alias** — VERIFIED ALREADY IMPLEMENTED in prior commits.
    Body kernel at `_idsva_so.py:2390` maps `s_temp_in_global=True` →
    `SCRATCH_IN_SMEM=false`; inner does the repoint at line 1372; the
    world inner already has `SCRATCH_IN_SMEM × COLD_IN_SMEM` (surgical cold
    trio Xdown/v_w/a_w). Docs landed in codegen `124be77` make the contract
    explicit; conformance audit in `docs/idsva_so_inner_refactor_notes.md`
    refreshed.
  - **C.1 LITE on g1+h1_2** — 3/4 GREEN (g1-fixed + g1-floating + h1_2-fixed).
    h1_2-floating SKIPPED on the pre-existing `forward_dynamics` smem cap
    (120 KB > 101 KB; matches B.4 backlog item, ancestor-scratch de-alias).
    NOT a regression.
  - **D.2 mimic-joint support — PARTIAL LANDING (parked on submodule branches,
    NOT bumped into parent yet).**
    - URDFParser `511e398` (on `modernizing-tests`): parse `<mimic>`,
      `Joint.get_num_dof()` returns 0 for mimic joints, `Robot._refresh_
      mimic_index_maps` builds dense q/v slot maps, `Robot.q_for_joint`
      exposes `mult * q[target] + offset`.
    - RBDReference `0b1a89d` (on `perf-cleanup`): kinematic Jacobian +
      analytic Hessian accumulate per-chain contributions and scale by
      mimic multiplier; pinocchio backend expands project→pin and reduces
      pin→project with mimic folding; new `test_pose_hessian_analytic_
      matches_fd[fr3-fixed]` smoke. Test suites GREEN for kinematics +
      RNEA + parse/metadata (173/173).
    - **Why parked (not yet bumped into parent):**
      1. Dynamics ripple is OPEN: `aba`, `crba`, `minv`, RNEA-grad, fd-grad
         still ASSIGN to the mimicked column instead of `+= mimic_scale * …`;
         the Robot.py nq/nv reduction now exposes this. Tests
         `test_aba_equivalence[fr3]`, `test_crba_equivalence[fr3,h1_2]`,
         `test_minv_equivalence[fr3]`, `test_integrator_pinocchio_
         equivalence[fr3]` now fail (they were passing by coincidence with
         naive sides on both).
      2. **CUDA codegen has no mimic awareness.** Bumping submodules into
         parent would make CUDA-equivalence-vs-RBDReference mismatch on
         fr3/h1_2 across multiple algos.
      3. Pre-existing FK orientation bug in `URDFParser/Joint.set_type`
         (sums `t_free + t_origin` without rotating `t_free` through origin
         rpy) surfaces only on `fr3_finger_joint2` (origin rpy = π around Z).
         Independent of mimic but in the way.
    - **fr3 analytic d²(pose)/dv² now matches FD oracle ~2e-11 and matches
      pinocchio's `getJointKinematicHessian(LOCAL_WORLD_ALIGNED)`**. h1_2
      (12 mimics, multipliers 1.0/1.6/2.4) kinematics + RNEA GREEN.
    - **Pickup recipe:** finish dynamics ripple in RBDReference (per the
      RNEA pattern, every `… = …` becomes `+= mimic_scale * …`); propagate
      mimic awareness into CUDA codegen; then bump submodules into parent.
      Or: gate fr3/h1_2 mimic-touching tests as xfail with link to this entry.
- **B.6.a + B.6.b (2026-05-29):** dropped the dead `use_thread_group`
  parameter from helper signatures + 1079 call sites across 16 codegen files
  (production always passes False; the conditional branches had been stripped
  earlier in b228756, leaving only signatures + plumbing); consolidated
  `gen_kernel_load_inputs` with `_single_timing` (same for save) into one fn
  each via optional `stride` kwarg (74 call sites rewritten). iiwa14
  fixed+floating + go2 floating CUDA equivalence GREEN. codegen `75089f8`,
  parent `1ff8d54`.
- **A.1 + A.3 scoping (2026-05-28 PM):** d2ee analytic derivation captured in
  `docs/d2ee_analytic_derivation.md` (fixed-base machine-precision validated in
  Python; floating intra-joint open gap); core-dynamics floating loss audit in
  `docs/a3_core_dynamics_floating_loss_audit.md` (top targets: ABA + CRBA
  floating root). C.3 partial: `-Wall` warnings probe on iiwa14-fixed runner
  found one class (42× `d_temp_spill` #177-D), silenced via `(void)` cast.
- **Cleanup batch** (2026-05-28): collapsed orchestrators to 3 layers
  (`_host` / `_kernel` / `_device`); per-tier L2 gates for fdsva_so;
  LITE/MINIMAL tier columns in `generate_report.py`; analytic d2ee in
  pinocchio_backend (joint targets + default offset); design docs + emitter
  README updated to the 3-layer convention; macro/name audit pass.
  iiwa14 fixed + floating CUDA equivalence GREEN.
- Floating `ee_pose_gradient` gap **CLOSED** (Step C, 2026-05-28: pin 7-16× → GRiD
  wins/parity on g1/h1_2 floating, pin only 1.18-1.34× on iiwa14/go2 floating).
- `ee_pose_hessian` (d2ee) Python d/dv rewrite (RBDReference + pinocchio_backend).
- d2ee GPU codegen rewrite (FD-on-Jacobian, d/dv tangent, pinocchio convention);
  iiwa14/go2/g1/h1_2 fixed + iiwa14/go2/g1 floating CUDA equivalence GREEN.
- d2ee timing wired into the bench (`timeGRiD_batch.cu` +
  `pinocchio/timePinocchio.cpp` via `computeJointKinematicHessians`) and measured
  end-to-end vs pinocchio.
- FIXED-path kinematics wired in CUDA equivalence runner (iiwa14 validated).
- `crba` regression fix (2026-05-27, 34edcfc) — recovered to baseline + beats
  pinocchio at batch on all robots.
- Pinocchio-as-fast-reference (equivalence-test oracle now defaults to pinocchio
  via `RBDReference/equivalents/pinocchio_backend.py`; first + second-order +
  d2ee all covered; pure-Python `RBDReference` is the fallback).

### Reference docs (kept separate, linked from here)
- `docs/source/user_guide/concepts/resource_tier_system.rst` — tier/spill architecture.
- `docs/idsva_so_inner_refactor_notes.md` — deferred inner de-alias design.
- `docs/python_wrappers_plan.md` — grid-rbd bindings (historical plan; v0.3 shipped).
- `test/benchmarks/overnight_tier_sweep.md` — partial sweep results (pre-fix run).
- `RBDReference/tests/PINOCCHIO_ALIGNMENT_BACKLOG.md` — Pinocchio alignment.
