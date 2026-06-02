# GRiD — Backlog to Full Correctness & Completeness (2026-06-01)

The shared work-list to get every supported feature/algorithm to {correct, complete, uniformly named,
fully tested}, single-block kept, no back-compat. Status: ✅ done · 🔵 in progress · ⬜ open.
Detail/evidence: `api_completeness_audit.md`, `rename_mapping.md`.

## 1. CORRECTNESS — known wrong results (must-fix bugs)
- ⬜ **B1 / C1 — `ee_pose_hessian` orientation-hessian is WRONG** (both CUDA codegen AND numpy analytic
  share the derivation). The test dodges it by scoping pin-parity to {iiwa14, fr3}. Fix the orientation
  block in both surfaces, then widen the test. *(d2ee — agent ran out, redo solo.)*
- ⬜ **B2 / C6b — `fdsva_so` floating-base `daba_dqdq` block is WRONG on ALL floating robots** (not
  mimic-specific; go2-floating non-mimic shows it; fixed-base passes). Root-caused to the fused floating
  scratch path: `_fdsva_so.py:325-329` — floating (world-inner) branch applies NO `dvdq_layout_repair`,
  and the inline FD-gradient runs before the world inner sharing `s_temp`. `fdsva_so` stays refused for
  floating-mimic until fixed.
- ⬜ **B3 — floating-mimic `integrator_gradient`/`integrator_with_gradient` multi-stage RK bug** (stage
  projection at floating∩multistage∩mimic). numpy ref is correct; CUDA refused. The only remaining true
  mimic refusal.

## 2. CORRECTNESS — verification gaps (untested code that could hide bugs)
- ✅ **C2 — `fd_parameter_gradient` numpy** now tested vs −M⁻¹Y pin oracle (ref was correct). *(committed)*
- ✅ **idsva_so floating-mimic** verified correct vs pin (world-frame), gate enabled. *(C6, committing)*
- ⬜ **V1 / C3 — `com_cost`/`momentum_cost` have NO CUDA equivalence test** (plant smoke runner never
  calls them). numpy+FD tests exist; the device kernels are CUDA-unverified.
- ⬜ **V2 / C4 — centroidal CUDA breadth + mimic.** `generalized_gravity`/`nonlinear_effects` support
  mimic but are never CUDA-checked on a mimic robot → need the **mimic-safe centroidal runner** (also
  the runtime check the `s_vaf` NB-fix still lacks). `com`/`ccrba`/`energy` are non-mimic-only (structural).
- ⬜ **V3 / C5 — integrator family CUDA only on small robots** (iiwa14,go2,fr3) — no g1/h1_2 spill paths.
- ⬜ **V4 / C7 — `integrator_with_gradient` no standalone numpy test**; **FK-batched** has no equivalence
  diff vs `end_effector_pose` (binding coverage ≠ correctness).
- ⬜ **V5 — widen d2ee + SO tests to big/floating robots** once B1/B2 land.

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
- ⬜ **I1 — re-sweep on the fixed harness** to set real autotune defaults + true best-tier numbers (later).

## Proposed sequence (for discussion)
1. **Correctness bugs first** — B1, B2, B3. (Don't ship known-wrong results.)
2. **Verification** — V1–V5 lock the rest with tests (cheap, high-confidence).
3. **Clean-break rename** — R1–R6 as ONE coordinated pass over the now-correct code (so new work lands on final names).
4. **Surface + feature completion** — S1/S2, then F1/F2.
5. **Perf** — P1, then P2.
6. **I1 re-sweep** whenever a GPU-idle window opens (sets defaults; non-blocking).
