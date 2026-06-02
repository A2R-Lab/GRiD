# GRiD — Backlog to Full Correctness & Completeness (2026-06-01)

The shared work-list to get every supported feature/algorithm to {correct, complete, uniformly named,
fully tested}, single-block kept, no back-compat. Status: ✅ done · 🔵 in progress · ⬜ open.
Detail/evidence: `api_completeness_audit.md`, `rename_mapping.md`.

## 1. CORRECTNESS — known wrong results (must-fix bugs)
- ✅ **B1 / C1 — `ee_pose_hessian` orientation-hessian** — STALE backlog item: already fixed by the prior
  chain-composition `d²/dv²` rewrite on BOTH surfaces. Verified empirically (2026-06-01): analytic-vs-FD
  ~1e-11; the residual vs pin (~1e-5) is pinocchio's own `getJointKinematicHessian` FD floor (FD-vs-pin ==
  analytic-vs-pin), NOT a GRiD error. Numpy test already at full-fleet scope (63 pass incl. 18 hessian-vs-pin
  fixed+floating); CUDA confirmed fresh (non-cached) on iiwa14-fixed + go2-floating. Only stale docstrings
  corrected.
- ✅ **B2 / C6b — `fdsva_so` floating-base `daba_dqdq` block** FIXED 2026-06-01. The scratch/repair lead was a
  RED HERRING; the real bug was a **jk-transpose in `gen_fdsva_so_contract`'s daba_dqdq assembly** — the final
  `-Minv` reduction reads `inner_dq` jk-transposed, harmless for the symmetric `dM_dq*da_dq` term but DROPPING the
  jk-asymmetry of `d2tau_dqdq` on the 6-DoF floating root q-q columns (1-DoF fixed joints are jk-symmetric, so
  fixed-base was always correct). Fix: floating branch only adds `d2tau_dqdq` jk-transposed; fixed-base
  BYTE-IDENTICAL (iiwa14+fr3 verified). go2-floating (non-mimic control) now passes the strict floating SO
  diagnostic (norm_rel ~1e-6, was 4.4e-2); double-precision emulation matches pin to ~3e-16. Gate cleared
  (`MIMIC_FLOATING_UNSUPPORTED_GRADIENTS` now empty). NOTE: fr3-floating mimic fdsva residual ~6e-3 is float32
  amplification through the ill-conditioned reduced Minv (NOT this bug) → needs a per-robot floating conditioning
  bucket in `_fdsva_so_tolerance` (deferred, tolerance-policy owner); fr3 diagnostic also blocked upstream by a
  separate pre-existing `idsva_so_body_frame` mimic-column failure. See C6b in `api_completeness_audit.md`.
- ⬜ **B3 — floating-mimic `integrator_gradient`/`integrator_with_gradient` multi-stage RK bug** (stage
  projection at floating∩multistage∩mimic). numpy ref is correct; CUDA refused. The only remaining true
  mimic refusal.
- ⬜ **B5 — grid_rbd FLOATING `end_effector_pose_gradient` binding broken** (surfaced by V6; BINDING bug, NOT
  a kernel bug — the kernel passes vs pin at threads {1,32,128} via the .cu path). Two parts in `python/grid_rbd`:
  (a) `_handle.py:end_effector_pose_gradient` reshapes raw to `(B,NEE,NV,6)` but the kernel emits NUM_POS
  columns (go2 19≠18) → `ValueError` crash; (b) the RAW floating `ee_pose_gradient` device buffer comes back
  uninitialized/garbage (~1e31, non-finite, launch-dependent) through the binding output path. Also a NUM_POS-vs-NV
  (position vs tangent) convention question to resolve for the public surface. Customer-facing (grid_rbd v0.3).
  Fixed-base is fine + thread-invariant. Lower urgency than kernel bugs; does NOT block the CUDA sweep but should
  land before shipping. *(V6 test skips these cells with documented reason.)*
- ⬜ **B4 — suspected `idsva_so_body_frame` fr3 mimic-column bug** (surfaced by B2, INDEPENDENT of it):
  fr3 mimic column 13 shows `last_two_axis_transpose_rel_norm=1.16` (huge) — fails identically with B2's fix
  stashed, so pre-existing and not fdsva-related. Investigate the body-frame inner's last-two-axis transpose
  on mimic columns. Lower priority than B3 (body-frame SO on a mimic robot is a narrow path), but a real
  suspected wrong-result. Also blocks the fr3-floating SO diagnostic upstream. *(also noted in C6b audit.)*

## 2. CORRECTNESS — verification gaps (untested code that could hide bugs)
- ✅ **C2 — `fd_parameter_gradient` numpy** now tested vs −M⁻¹Y pin oracle (ref was correct). *(committed)*
- ✅ **idsva_so floating-mimic** verified correct vs pin (world-frame), gate enabled. *(C6, committing)*
- ✅ **V1 / C3 — `com_cost`/`momentum_cost` CUDA equivalence test added** (iiwa14-fixed + go2-floating);
  EXPOSED + FIXED a real emitter bug: `com_cost_gradient` zeroed `[nq, nq+nv)` but the meaningful gradient
  lives in `[0, nv)`, leaving `[nv, nq)` UNINITIALIZED on floating-base (nq>nv) → stale shared mem (go2
  `s_grad[18]`). Fixed in `_plant.py` to zero the full `[nv, nx)` tail (byte-identical fixed-base; matches the
  GN-hessian `[0,nv)` convention). `momentum_cost_gradient` was already correct. Both cells green. *(committed)*
- ✅ **V2 / C4 + D — mimic-safe centroidal CUDA runner added** (`cuda_centroidal_mimic_smoke_runner.cu` +
  test, codegens `algorithm_list=["id"]` so only generalized_gravity/nonlinear_effects emit; non-mimic runner
  untouched). fr3-fixed (mimic, NB=9>NV=8) + iiwa14-fixed control PASS ~5e-7 — runtime confirms the device
  `s_vaf=18*NB` fix. EXPOSED + FIXED a second mimic-overflow bug: the HOST arena macro `id_bias_t_count`
  (`GRiDCodeGenerator.py:662`) still sized `s_vaf` by `18*n` (NV) while the device wrapper uses `18*NB` →
  h1_2-fixed (NB=51>NV=39) crashed with an illegal shared write (under-budget by ~99 floats). Fixed to
  `18*nb_vaf` (NB for mimic; byte-identical non-mimic). Pending h1_2 validation → then commit. *(committing)*
- 🔵 **V3 / C5 — integrator CUDA breadth to g1/h1_2 + PERF/LITE tier sweep** (committed `6aaad70`). Test infra
  done: 5 robots × 2 base × 2 tiers = 20 cells; floating-mimic runs value-only (B3-refused gradient); mimic
  `s_vaf=18*NB` sizing confirmed. VALIDATED subset green (g1-fixed SHARED+LITE, h1_2-fixed-LITE mimic spill —
  the hardest cell). REMAINING (reasoned, not run, due to ~8-way contention): h1_2-fixed-PERF, g1-floating×2,
  h1_2-floating value-only → **must get a full-matrix green run in the pre-sweep gate** (see I1 area).
- ⬜ **V4 / C7 — `integrator_with_gradient` no standalone numpy test**; **FK-batched** has no equivalence
  diff vs `end_effector_pose` (binding coverage ≠ correctness).
- ⬜ **V5 — widen d2ee + SO tests to big/floating robots** once B1/B2 land.
- ⬜ **V6 — kinematics thread-count-invariance matrix** (user-requested, before the sweep): the single-block
  kinematics kernels (`end_effector_pose`, `ee_pose_gradient`, `frame_jacobian`) must give identical results
  at ANY thread count — test single-thread (1) and warp-sized (32) explicitly, plus a sweep of thread counts
  {1,2,16,32,64,128,256} × batch sizes {1,16,256}, asserting equivalence vs the numpy/pin reference at every
  cell. Catches reduction/sync bugs that only surface at extreme thread counts. Any divergence = a new
  backlog bug to fix BEFORE the sweep.

## 3. COMPLETENESS — missing surfaces
- ⬜ **S1 — `frame_jacobian` / `frame_jacobian_dot` / `osc_inertia` are DEVICE-ONLY.** Add
  kernel + 3-mode host + batch wrappers + gridData output buffers → benchmarkable + bindable. Prereq for
  the HJCD "parallelize frame_jacobian" perf ask.
- ⬜ **S2 — uniform `_inner` missing** for `fd_du` (reuses id_du band), `f_ext_gradient_dq` (kernel-only),
  `integrator_gradient` (uses `_multistage`).

## 4. COMPLETENESS — missing features
- ⬜ **F1 — `plant_step_hessian`** (true analytic 2nd-order plant/integrator Hessian) is absent; needs an
  analytic 2nd-order integrator (`_plant.py:82-87 TODO`). **#1 customer ask** (PDDP `compute_fxx`, GATO).
- ⬜ **F2 — bindings incomplete:** expose `frame_jacobian`, `frame_jacobian_dot`, `osc_inertia`, `com`,
  `ccrba`, `energy`, `generalized_gravity`, `nonlinear_effects`, `com_cost`, `momentum_cost`,
  `plant_step_gradient` (its value form `plant_step` is already exposed — asymmetric).

## 5. READABILITY — clean-break rename + uniformity (scheme LOCKED: rename_mapping.md)
- ⬜ **R1 — verbose rename** (id→inverse_dynamics, etc.; keep aba/crba/ccrba/minv/idsva_so/fdsva_so;
  rnea→docstring not alias). One name as key==symbol==bench==token==label.
- ⬜ **R2 — outputs into `gridData`** for `regressor`/`fd_parameter_gradient` (drop caller-owned buffers).
- ⬜ **R3 — uniform signatures + surface sets;** reorder `fdsva_so_kernel` args (`d_workspace`→2nd).
- ⬜ **R4 — drop** `_host`/`_with_x_kp1`/`direct_` + `rnea`/`eepos`/`deepos` labels + `SUGGESTED_THREADS` alias.
- ⬜ **R5 — unify gravity-sign convention** (GRiD `+9.81` vs RBDReference `-9.81`).
- ⬜ **R6 — emit centroidal family on OWN keys** (com/ccrba/energy/gg/nle currently gated on a sibling key).

## 6. PERFORMANCE — max-perf completion (single-block ONLY)
- ⬜ **P1 — big-robot `crba`/`aba`/`minv`/`fd` in-block parallelism.** They lose to Pinocchio CPU on h1_2;
  `id` wins 5× there → the in-block-parallelism pattern to follow. The only place GRiD loses.
- ⬜ **P2 — parallelize `frame_jacobian_inner`** (HJCD; after S1 wrappers exist to measure it).

## 7. INFRA (mostly done)
- ✅ Sweep efficiency: collapse + narrow + aliasing-fix + SASS tier-dedup (~6× faster, correct per-tier).
- ⬜ **PRE-SWEEP VALIDATION GATE** (run once the backlog is otherwise green, in a GPU-idle/overnight window,
  BEFORE I1): a single clean full-matrix equivalence run with cleared header cache covering every robot × base
  × representative tiers — INCLUDING the big-robot integrator cells C5 only reasoned (h1_2-fixed-PERF,
  g1-floating×2, h1_2-floating value-only) and V6's kinematics thread×batch matrix. No known-open bug at sweep
  time (user directive). Any red cell = a bug to fix before the sweep, not after.
- ⬜ **I1 — re-sweep on the fixed harness** to set real autotune defaults + true best-tier numbers (later).

## Proposed sequence (for discussion)
1. **Correctness bugs first** — B1, B2, B3. (Don't ship known-wrong results.)
2. **Verification** — V1–V5 lock the rest with tests (cheap, high-confidence).
3. **Clean-break rename** — R1–R6 as ONE coordinated pass over the now-correct code (so new work lands on final names).
4. **Surface + feature completion** — S1/S2, then F1/F2.
5. **Perf** — P1, then P2.
6. **I1 re-sweep** whenever a GPU-idle window opens (sets defaults; non-blocking).
