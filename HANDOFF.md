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

**STEP A + B + C LANDED + PUSHED (2026-05-27).** Branch HEAD: parent `d8b1bcb`,
GRiDCodeGenerator `df70675`, RBDReference `0e71d06`. **Step C (GPU codegen + multi-file
ripple) is in: iiwa14 fixed (10/10) + floating (1/1) CUDA equivalence GREEN.**
Validation matrix for go2/g1/h1_2 fixed+floating running (background) — pick up that
result before declaring full success.

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

### 1. Inner-owns-placement UNITY (new 2026-05-25; current code works via kernel-repoint — purity, not correctness)
- **Full orchestrator → `*_full_inner` migration** for `id_du`, `fd_du`,
  `integrator_gradient` (mirror `fdsva_so_full_inner`) so the kernel never
  repoints `s_temp`. The crash bug is fixed; this is the remaining design-purity
  step. Standard is documented in `docs/idsva_so_inner_refactor_notes.md`.
- **`gen_fdsva_so_device`** (inline API) still duplicates the orchestration →
  rewire to call `fdsva_so_full_inner`.
- **Standalone idsva body kernel `output_temp` rung** → fold its kernel-level
  `s_temp` repoint into a body-inner `SCRATCH_IN_SMEM`.
- **L2-persisting gates** in 5 host wrappers (`id_du`/`fd_du`/`fdsva`/`d2ee`/
  `ee_grad`) key off the PERF-pick macro → LITE/MINIMAL spills aren't L2-*pinned*.
  Perf nit, not correctness (`d_workspace` is allocated unconditionally for
  `GRID_DATA_ALL`). Make the gate per-tier.

### 2. Spill perf refinements (surgical; gated on sweep data)
- **De-alias the idsva_so / fdsva_so inners** for surgical cold-only spill
  instead of whole-arena — `docs/idsva_so_inner_refactor_notes.md`. Same idea
  helps the integrator-gradient rung-3 on h1_2.
- **ABA whole-arena spill → surgical retrofit** (sub-split like Minv-F).
- **Tune the LITE smem target** (48 KB vs 32/64) — falls out of the sweep.

### 3. Perf investigations (from the tier sweeps)
- **`ee_pose_gradient` FLOATING ~535µs batch-N=256 outlier** (vs ~8µs fixed,
  ~1µs Pinocchio) — slow floating world-frame EE-Jacobian path.
- **GRiD loses to Pinocchio at batch N=256 on some core-dynamics cells**
  (FD/ABA/CRBA/Minv) — launch overhead / occupancy / per-block work.
- **GRiD second-order ~3× slower than Pinocchio on g1/h1_2 at N=256** —
  big-robot SO regression.
- **Document SO speedups vs Pinocchio**: finish wiring + rendering the Pinocchio
  IDSVA_SO/FDSVA_SO baseline column (Pinocchio CPU SO is ms-scale).

### 4. Reporting / tooling
- **Tier (LITE/MINIMAL) columns in `generate_report.py`** — data is captured in
  the per-cell JSONs, not yet rendered.
- **Autotune `performance_threads`**: binary-search the batch-throughput-
  maximizing launch thread count (≤ `MAX_PERF_LEVEL_THREADS`); expose it.
- **Spilled-tier re-sweep — `fdsva_so` single_us + `ee_pose_hessian` (d2ee)
  timing** (bundle into ONE re-run). `ee_pose_hessian` is now in the bench
  (`PER_ALGO_SPECS`) but has never been timed; `fdsva_so` single_us was dropped
  when its TU hit the -rdc regcount error (now fixed) so single-call numbers need
  recapture. ⚠️ **TIER-DEBUGGING RISK**: the d2ee and fdsva_so *spill* paths at
  LITE/MINIMAL on big robots (g1/h1_2) are **unexercised by the bench so far** —
  this re-sweep is the first time they run under the timing harness, so expect
  possible per-cell crashes/OOB to debug (same bug classes as §Resolved). Budget
  for it. Do NOT run heavy CPU/GPU work concurrently (skews timing).

### 5. Correctness / coverage
- **FIXED-path kinematics now wired + validated on iiwa14 (2026-05-25)**: the
  equivalence runner's FIXED branch (`#else`) now calls the `end_effector_pose` /
  `_gradient` / `_gradient_hessian` host wrappers and prints `h_eePos`/`h_deePos`/
  `h_d2eePos`; all three added to `FIXED_CUDA_ALGORITHMS`. iiwa14-fixed PASSED
  (threads32 + threadssuggested) against the Python reference, alongside the
  earlier iiwa14-floating pass. ⚠️ **BROADER VALIDATION STILL NEEDED**: FIXED-base
  EE accuracy is confirmed for iiwa14 ONLY — go2 / g1 / h1_2 (fixed + floating),
  and the **spilled tiers** (LITE/MINIMAL, where d2ee uses `d_workspace`), are not
  yet run for these kinematics. Run the full robot × base × tier matrix before
  considering EE-pose/gradient/hessian accuracy fully validated.
- **nvcc/ptxas compile-warnings sweep** (`-Werror`-style build pass). Python
  reference path is already DeprecationWarning-clean.
- **Parallel equivalence testing** (proven 2026-05-26): the CUDA equivalence tests are
  correctness-only and independent per `(robot, base)`, so they run concurrently — ran
  5 cases in parallel with no issue. Each full `grid.cuh` `nvcc -O0` compile peaks
  ~5 GB RSS; size parallelism to `min(nproc, free_RAM/~5 GB)` (this box 24c/62 GB →
  ~8–10 wide). Clear `.pytest_cache/grid_cuda` after codegen changes (cache key is
  model+config, NOT source). Use `-s` for live progress (`-q` buffers to the end).
  TODO: make the harness auto-parallel — add `pytest-xdist` `-n` sized by cores+RAM, or
  a custom `(robot,base)` sharding runner. The PERF sweep must stay isolated.
- **h1_2-floating inline DEVICE path exceeds the sm_120 ~99 KB smem cap** (found
  2026-05-26): the always-smem inline `*_device` paths (forward_dynamics, etc.) request
  >99 KB dynamic smem on the biggest floating robot, so the equivalence runner's
  `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` fails ("invalid argument")
  *before any kernel runs* → h1_2-floating equivalence aborts early. **PRE-EXISTING**
  (the device-path code + `FD_DEVICE_DYNAMIC_SHARED_MEM_BYTES` are byte-identical to
  pre-wave base); the production KERNEL path spills and fits, and g1-floating already
  validates the floating spill rungs. FIX: make the inline device paths tier-aware/
  spillable (completes the deferred fd audit) OR guard the runner to skip device-path
  setattr above the device cap; then h1_2-floating kernel paths can validate too.

### 6. Naming / API clarity
- **Broader name audit**: `*_DYNAMIC_SHARED_MEM_BYTES`, the tier names, and the
  `s_*`-named-but-device pointers (e.g. `s_temp_spill` → `d_`).

### 7. Pinocchio-alignment backlog (lower urgency)
- URDFParser/RBDReference additive improvements: strict-parse API, structured
  parse diagnostics, Pinocchio-shaped metadata helpers (`nq`/`nv`/quaternion
  order), mark fixed-base-only methods. See
  `RBDReference/tests/PINOCCHIO_ALIGNMENT_BACKLOG.md`.
- **Pinocchio as a FAST reference — CORE, run as a dedicated project AFTER this
  perf-cleanup session** (user 2026-05-26): the pure-Python `RBDReference` is the
  equivalence-test bottleneck (the second-order `idsva_so`/`fdsva_so` refs on h1_2 take
  *hours* of single-threaded Python). Replace with 1:1 pinocchio (C++) references
  wherever they match, to make the reference fast and never the bottleneck:
  - Compose pinocchio sub-calls + thin C++ glue for GRiD algos lacking a direct entry
    (`fdsva_so` ← `idsva_so` + `minv`; `fd` ← `rnea`/`minv`; etc.).
  - Use pinocchio kinematics + kinematic derivatives for `ee_pose`/`_gradient`.
  - Use a **C++ finite-difference** reference for the `d2ee` hessian (vs pure Python).
  - Cleanest architecture: a hooks file in `RBDReference` exposing pinocchio-equivalent
    inputs/outputs for every function; simplify the tests to call one shared interface
    for both the CUDA and reference sides (true 1:1, fairest, most testable).
  - Keep pure-Python `RBDReference` as the fallback/cross-check where pinocchio lacks an
    identical quantity/layout. Builds on `RBDReference/equivalents/`. Mind joint-order/
    frame conventions (adapter already resolves pinocchio order) + float32 tolerance.

### 8. Branch merge (the big one)
- **FF-merge `humanoid-tier-spill` → `modernizing-tests`** — held pending final
  validation; most is now green. Gate on the targeted re-sweep + equivalence
  confirming the complete dataset.

### Reference docs (kept separate, linked from here)
- `docs/source/user_guide/concepts/resource_tier_system.rst` — tier/spill architecture.
- `docs/idsva_so_inner_refactor_notes.md` — deferred inner de-alias design.
- `docs/python_wrappers_plan.md` — grid-rbd bindings (historical plan; v0.3 shipped).
- `test/benchmarks/overnight_tier_sweep.md` — partial sweep results (pre-fix run).
- `RBDReference/tests/PINOCCHIO_ALIGNMENT_BACKLOG.md` — Pinocchio alignment.
